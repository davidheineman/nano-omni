"""
train_audio.py -- one file to prepare data for, and train, a discrete-audio model
by *speech pretraining* (Marin-Audio-style), reporting validation performance.
Trains from scratch OR continues from a pretrained nanoGPT-speedrun base.
Deliberately minimal: one stage, one corpus, a plain GPT.

Speech pretraining, not SFT. The point of Marin Audio is learning to model speech:
audio is quantized by the Mimi codec into discrete tokens, and the model is trained
as a plain LM on interleaved speech-text documents (YODAS2). This is next-token
prediction on a token stream -- a normal GPT, no connector, no response masking.

Two tokenizers, one vocabulary. To let a GPT-2 speedrun model's text embeddings
transfer, we reconcile the two tokenizers into one vocab (the "GPT2x" layout,
analogous to Marin's Qwen3x):

    [ 0 .. 50256 ]     GPT-2 BPE text tokens   (identical ids to the speedrun model)
    [ 50257 .. 50262 ] 6 specials: bot, eot, text_start, text_end, audio_start, audio_end
    [ 50263 .. 66646 ] 16384 Mimi audio tokens (8 codebooks x 2048)

Data prep lives in data/audio/marin_audio.py (the GPT2x speech-text mixture); training reads the
resulting uint32 token shards from data/audio/marin_mix_gpt2x/ and does plain LM (loss on every
token). On warm-start the first 50257 (text) rows of embed/lm_head are copied from the base.

Run from the repo root:
    python data/audio/marin_audio.py                               # build data (needs internet)
    torchrun --standalone --nproc_per_node=8 audio/train_audio.py    # pretrain from scratch
    torchrun --standalone --nproc_per_node=8 audio/train_audio.py \
        --init_from results/pretrain/.../checkpoint.pt               # continue from a speedrun base
    python audio/train_audio.py --model_dim 256 --num_layers 4 --num_heads 4 \
        --seq_len 1024 --micro_seqs 4 --num_iterations 40            # 1-GPU smoke test

Match --model_dim/--num_layers/--num_heads to the base checkpoint's body when
warm-starting.
"""

import argparse
import json
import math
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# combined "GPT2x" vocabulary: GPT-2 text ids + specials + Mimi audio tokens
# ---------------------------------------------------------------------------

MAGIC = 20240930  # shard magic (nanoGPT fineweb uses 20240520)
HEADER_INTS = 256


# ---------------------------------------------------------------------------
# model: clean modern-arch GPT (rotary + QK-norm + ReLU^2 MLP, untied head)
# ---------------------------------------------------------------------------


def rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = (1.0 / base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = torch.arange(x.size(1), device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
        sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Half-split convention matching modded-nanogpt's apply_rotary_emb exactly, so
    # a speedrun checkpoint's attention weights transfer faithfully.
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([x2, -x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=2)
        q, k = rmsnorm(q), rmsnorm(k)  # QK-norm
        cos, sin = self.rotary(q)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = Attention(dim, num_heads)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


def next_multiple_of_n(v: int, n: int = 128) -> int:
    return ((v + n - 1) // n) * n


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int):
        super().__init__()
        self.padded_vocab = next_multiple_of_n(vocab_size, 128)
        self.embed = nn.Embedding(self.padded_vocab, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.lm_head = nn.Linear(model_dim, self.padded_vocab, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.lm_head.weight, std=0.5 * model_dim**-0.5)

    def _init(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        x = rmsnorm(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(rmsnorm(x)).float()


# ---------------------------------------------------------------------------
# warm-start from a nanoGPT-speedrun checkpoint (body + shared GPT-2 text rows)
# ---------------------------------------------------------------------------


def _remap_speedrun_ckpt(src: dict) -> dict:
    """Map a modded-nanogpt record state_dict into this repo's GPT naming:
    `transformer.h.N.attn.c_{q,k,v}` (split) -> fused `blocks.N.attn.qkv`, etc.
    Vocab-coupled `wte`/`lm_head` are dropped. If already in our naming (e.g. a
    checkpoint saved by this script), returned unchanged."""
    if not any("transformer.h." in k for k in src):
        return src
    layers = {int(re.search(r"transformer\.h\.(\d+)\.", k).group(1))
              for k in src if re.search(r"transformer\.h\.(\d+)\.", k)}
    out = {}
    for i in sorted(layers):
        p = f"transformer.h.{i}."
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat(
            [src[p + "attn.c_q.weight"], src[p + "attn.c_k.weight"], src[p + "attn.c_v.weight"]], dim=0)
        out[f"blocks.{i}.attn.proj.weight"] = src[p + "attn.c_proj.weight"]
        out[f"blocks.{i}.mlp.fc.weight"] = src[p + "mlp.c_fc.weight"]
        out[f"blocks.{i}.mlp.proj.weight"] = src[p + "mlp.c_proj.weight"]
    return out


def warm_start(model: GPT, path: str, log, text_vocab: int = 0) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    src = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in src.items()}

    n_txt = 0
    if text_vocab > 0:  # GPT2x: our first text_vocab ids ARE the speedrun model's GPT-2 ids
        wte = src.get("transformer.wte.weight", src.get("embed.weight"))
        head = src.get("lm_head.weight", wte)  # speedrun models tie wte/lm_head
        if wte is not None and wte.shape[1] == model.embed.weight.shape[1]:
            n_txt = min(text_vocab, wte.shape[0], model.embed.weight.shape[0])
            with torch.no_grad():
                model.embed.weight[:n_txt].copy_(wte[:n_txt])
                if head is not None and head.shape[1] == model.lm_head.weight.shape[1]:
                    model.lm_head.weight[:n_txt].copy_(head[:n_txt])

    src = _remap_speedrun_ckpt(src)
    tgt = model.state_dict()  # captures the embed/lm_head rows copied above
    loaded = []
    for k, v in tgt.items():
        if k.startswith("embed") or k.startswith("lm_head"):
            continue  # keep (partially GPT-2-copied) rows
        if k in src and tuple(src[k].shape) == tuple(v.shape):
            tgt[k] = src[k]
            loaded.append(k)
    model.load_state_dict(tgt, strict=True)
    tail = (f"; copied {n_txt} GPT-2 text embed/lm_head rows (text not relearned)"
            if n_txt else "; embed/lm_head re-initialized for audio vocab")
    log(f"warm-start from {path}: loaded {len(loaded)}/{len(tgt)} body tensors{tail}")
    if not loaded:
        log("WARNING: no body tensors matched -- arch differs from the checkpoint; "
            "effectively from scratch. Check --model_dim/--num_layers/--num_heads.")


# ---------------------------------------------------------------------------
# data loader: packed uint32 token stream (plain LM)
# ---------------------------------------------------------------------------


def _open_shard(bin_path: str):
    header = np.fromfile(bin_path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC, f"bad magic in {bin_path}"
    n = int(header[2])
    return np.memmap(bin_path, dtype=np.uint32, mode="r", offset=HEADER_INTS * 4, shape=(n,))


class PackedLoader:
    """Contiguous (x, y) windows of length seq_len, sharded across ranks. Plain
    LM: every token is a target (no masking). Simplification vs the speedrun
    loader: no per-document attention mask across packed boundaries."""

    def __init__(self, bin_path: str, seq_len: int, rank: int, world: int):
        self.tokens = _open_shard(bin_path)
        self.T, self.rank, self.world = seq_len, rank, world
        self.stride = seq_len
        self.pos = rank * self.stride

    def next(self, micro_seqs: int):
        T, N = self.T, len(self.tokens)
        xs, ys = [], []
        for _ in range(micro_seqs):
            if self.pos + T + 1 > N:
                self.pos = self.rank * self.stride  # wrap
            a = self.pos
            chunk = torch.from_numpy(self.tokens[a : a + T + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.pos += self.world * self.stride
        return torch.stack(xs).cuda(non_blocking=True), torch.stack(ys).cuda(non_blocking=True)


@torch.no_grad()
def evaluate(model, loader: PackedLoader, cfg, world: int) -> dict:
    model.eval()
    tot_loss = torch.zeros((), device="cuda")
    tot_correct = torch.zeros((), device="cuda")
    tot_count = torch.zeros((), device="cuda")
    seqs = max(1, cfg.val_tokens // (cfg.seq_len * world * cfg.micro_seqs))
    loader.pos = loader.rank * loader.stride
    for _ in range(seqs):
        x, y = loader.next(cfg.micro_seqs)
        logits = model(x)
        tot_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        tot_correct += (logits.argmax(-1).view(-1) == y.view(-1)).sum()
        tot_count += y.numel()
    if world > 1:
        for t in (tot_loss, tot_correct, tot_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    avg = (tot_loss / tot_count).item()
    return {"loss": avg, "ppl": math.exp(min(avg, 20)), "acc": (tot_correct / tot_count).item()}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # data (repo-root-relative; run from the repo root)
    data_dir: str = "data/audio/marin_mix_gpt2x"
    seq_len: int = 2048
    # model (match the base checkpoint's body when warm-starting)
    model_dim: int = 768
    num_layers: int = 12
    num_heads: int = 6
    # optimization
    batch_tokens: int = 524288  # global tokens per optimizer step
    micro_seqs: int = 8         # sequences per forward on each rank; grad-accum fills batch_tokens
    num_iterations: int = 2000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.02
    cooldown_frac: float = 0.4
    grad_clip: float = 1.0
    # init / eval / io
    init_from: str = ""         # path to a speedrun checkpoint .pt (expects ["model"] state_dict)
    val_every: int = 200
    val_tokens: int = 4_194_304
    save_to: str = ""
    seed: int = 1337


def lr_at(step: int, cfg: Config) -> float:
    warm = max(1, int(cfg.warmup_frac * cfg.num_iterations))
    cool_start = int((1 - cfg.cooldown_frac) * cfg.num_iterations)
    if step < warm:
        return cfg.learning_rate * (step + 1) / warm
    if step < cool_start:
        return cfg.learning_rate
    return cfg.learning_rate * max(0.0, (cfg.num_iterations - step) / max(1, cfg.num_iterations - cool_start))


def _shard_prefix(cfg: Config) -> str:
    return os.path.basename(cfg.data_dir.rstrip("/"))


def train(cfg: Config):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    master = rank == 0
    torch.manual_seed(cfg.seed + rank)

    def log(msg):
        if master:
            print(msg, flush=True)

    meta = json.load(open(os.path.join(cfg.data_dir, "meta.json")))
    log(f"data={cfg.data_dir} vocab={meta['vocab_size']} text_vocab={meta.get('text_vocab', 0)}")

    model = GPT(meta["vocab_size"], cfg.num_layers, cfg.num_heads, cfg.model_dim).cuda()
    if cfg.init_from:
        warm_start(model, cfg.init_from, log, text_vocab=meta.get("text_vocab", 0))
    else:
        log("no --init_from: training from scratch")
    log(f"model: dim={cfg.model_dim} layers={cfg.num_layers} heads={cfg.num_heads} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    raw_model = model
    ddp_handle = None
    if ddp:
        ddp_handle = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
        model = torch.compile(ddp_handle)
    else:
        model = torch.compile(raw_model)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay, fused=True)

    pfx = _shard_prefix(cfg)
    train_ldr = PackedLoader(os.path.join(cfg.data_dir, f"{pfx}_train.bin"), cfg.seq_len, rank, world)
    val_ldr = PackedLoader(os.path.join(cfg.data_dir, f"{pfx}_val.bin"), cfg.seq_len, rank, world)

    accum = max(1, cfg.batch_tokens // (cfg.seq_len * cfg.micro_seqs * world))
    log(f"batch: {cfg.batch_tokens} tok/step => grad_accum={accum} x micro({cfg.micro_seqs}) x {world} ranks")

    for step in range(cfg.num_iterations + 1):
        last = step == cfg.num_iterations
        if step % cfg.val_every == 0 or last:
            m = evaluate(raw_model, val_ldr, cfg, world)
            log(f"step {step:5d} | val loss {m['loss']:.4f} | ppl {m['ppl']:.2f} | acc {m['acc']:.4f}")
            if last:
                break
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            x, y = train_ldr.next(cfg.micro_seqs)
            sync = (not ddp) or (micro == accum - 1)
            with (nullcontext() if sync else ddp_handle.no_sync()):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / accum
                loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 50 == 0:
            log(f"step {step:5d} | train loss {(loss.item()*accum):.4f} | lr {lr:.2e}")

    if cfg.save_to and master:
        os.makedirs(os.path.dirname(cfg.save_to) or ".", exist_ok=True)
        torch.save({"model": raw_model.state_dict(), "cfg": vars(cfg), "meta": meta}, cfg.save_to)
        log(f"saved audio model -> {cfg.save_to}")
    if ddp:
        dist.destroy_process_group()


def _parse_train_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser()
    for f, v in vars(cfg).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{f}", type=type(v), default=v)
    return Config(**vars(ap.parse_args()))


if __name__ == "__main__":
    train(_parse_train_args())
