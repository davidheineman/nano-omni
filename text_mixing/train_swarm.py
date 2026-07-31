import argparse
import glob
import json
import math
import os
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evals"))
import ppl as eval_ppl  # evals/ppl.py  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "/datasets/pretraining_data/dhei/speedrun/text_mixing/tok")
CKPT_DIR = os.environ.get("CKPT_DIR", "/checkpoint/transformer2/dhei/speedrun/text_mixing")
# results go to SHARED storage (collect.py reads them; /checkpoint may be node-local)
RESULT_DIR = os.environ.get("RESULT_DIR", "/datasets/pretraining_data/dhei/speedrun/text_mixing/results")
MIXTURES_JSON = os.environ.get("MIXTURES_JSON", os.path.join(HERE, "mixtures.json"))


# ----------------------------------------------------------------------- model
class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layer) ** 0.5

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class GPTConfig:
    def __init__(self, vocab_size=50304, n_layer=8, n_head=8, n_embd=512):
        self.vocab_size, self.n_layer, self.n_head, self.n_embd = vocab_size, n_layer, n_head, n_embd

    def to_dict(self):
        return dict(vocab_size=self.vocab_size, n_layer=self.n_layer,
                    n_head=self.n_head, n_embd=self.n_embd)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def _trunk(self, idx):
        x = self.wte(idx)
        for b in self.h:
            x = b(x)
        return rmsnorm(x)

    def forward(self, idx, targets):
        logits = self.lm_head(self._trunk(idx)).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    def forward_logits(self, idx):
        return self.lm_head(self._trunk(idx))


# ----------------------------------------------------------------------- Muon
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      backend_steps=backend_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # fused QKV
                    g = torch.cat([zeropower_via_newtonschulz5(gi, group["backend_steps"])
                                   for gi in g.split(g.size(1))])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_via_newtonschulz5(g, group["backend_steps"])
                    scale = max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr * scale)


# ----------------------------------------------------------------- data loader
def _load_shard(fn):
    with open(fn, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1, f"bad .bin header: {fn}"
        toks = np.frombuffer(f.read(), dtype=np.uint16)
    return toks


class MixtureLoader:
    """Weighted multi-source loader over pools' .bin shards.

    At each step a pool is drawn ~ weights and a contiguous B*T+1 block is read from that
    pool's cursor (cycling its shards). Pools absent on disk (not yet downloaded) or too
    small are dropped and the weights renormalized; `coverage` records the retained mass.
    """
    def __init__(self, weights, B, T, data_root=DATA_ROOT, seed=0):
        self.B, self.T = B, T
        need = B * T + 1
        avail, w = [], []
        for name, wt in weights.items():
            if wt <= 0:
                continue
            shards = sorted(glob.glob(os.path.join(data_root, name, "shard_*.bin")))
            if not shards:
                continue
            avail.append((name, shards))
            w.append(float(wt))
        if not avail:
            raise RuntimeError("no pools from the mixture are present on disk")
        w = np.array(w, dtype=np.float64)
        self.coverage = float(w.sum() / sum(x for x in weights.values() if x > 0))
        self.names = [a[0] for a in avail]
        self.shards = [a[1] for a in avail]
        self.probs = w / w.sum()
        self.n_pools = len(avail)
        self.rng = np.random.default_rng(seed)
        self.need = need
        # per-pool state: current shard index, loaded tokens, position
        self.cur_shard = [0] * self.n_pools
        self.cur_tok = [None] * self.n_pools
        self.cur_pos = [0] * self.n_pools

    def _ensure(self, i):
        if self.cur_tok[i] is None:
            self.cur_tok[i] = _load_shard(self.shards[i][self.cur_shard[i]])
            self.cur_pos[i] = 0

    def _advance(self, i):
        self.cur_shard[i] = (self.cur_shard[i] + 1) % len(self.shards[i])
        self.cur_tok[i] = _load_shard(self.shards[i][self.cur_shard[i]])
        self.cur_pos[i] = 0

    def next_batch(self, device):
        i = int(self.rng.choice(self.n_pools, p=self.probs))
        self._ensure(i)
        # guarantee a contiguous need-token block, cycling shards if short
        tries = 0
        while self.cur_pos[i] + self.need > len(self.cur_tok[i]):
            self._advance(i)
            tries += 1
            if tries > len(self.shards[i]) + 1:  # shard smaller than a batch (shouldn't happen)
                # pad by tiling
                self.cur_tok[i] = np.resize(self.cur_tok[i], self.need + 1)
                self.cur_pos[i] = 0
        p = self.cur_pos[i]
        buf = self.cur_tok[i][p:p + self.need].astype(np.int64)
        self.cur_pos[i] = p + self.B * self.T
        buf = torch.from_numpy(buf)
        x = buf[:-1].view(self.B, self.T).to(device, non_blocking=True)
        y = buf[1:].view(self.B, self.T).to(device, non_blocking=True)
        return x, y


# ----------------------------------------------------------------- checkpoint
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model


# ----------------------------------------------------------------------- train
def get_mixture(args):
    if args.weights_json:
        return args.run_id or f"adhoc-{uuid.uuid4().hex[:8]}", json.loads(args.weights_json)
    mixes = json.load(open(MIXTURES_JSON))
    if args.index is not None:            # job-array entry point
        m = mixes[args.index]
    elif args.mixture:
        m = {x["run_id"]: x for x in mixes}[args.mixture]
    else:
        raise SystemExit("pass --index N, --mixture <run_id>, or --weights-json <json>")
    return m["run_id"], m["weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, help="row index into MIXTURES_JSON (array launch)")
    ap.add_argument("--mixture", help="run_id in mixtures.json")
    ap.add_argument("--weights-json", help="inline {pool: weight} JSON")
    ap.add_argument("--run-id", help="override run id (with --weights-json)")
    args = ap.parse_args()

    run_id, weights = get_mixture(args)
    if os.path.exists(os.path.join(RESULT_DIR, f"{run_id}.json")):
        print(f"{run_id}: result exists, skipping", flush=True)
        return
    device = "cuda"
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig(
        vocab_size=50304,
        n_layer=int(os.environ.get("N_LAYER", 8)),
        n_head=int(os.environ.get("N_HEAD", 8)),
        n_embd=int(os.environ.get("N_EMBD", 512)),
    )
    T = int(os.environ.get("SEQ_LEN", 1024))
    B = int(os.environ.get("BATCH_SEQS", 32))
    tokens_budget = int(float(os.environ.get("TOKENS", 3e8)))
    steps = max(1, tokens_budget // (B * T))
    warmup = min(256, steps // 20 + 1)
    lr_adam = float(os.environ.get("LR_ADAM", 6e-4))
    lr_muon = float(os.environ.get("LR_MUON", 0.02))

    import hashlib
    seed = int(hashlib.md5(run_id.encode()).hexdigest(), 16) % (2**31)  # stable per run_id
    loader = MixtureLoader(weights, B, T, seed=seed)

    # Coverage guard: if too many of this mixture's pools aren't tokenized yet, skip
    # cleanly (no result written) so launch_swarm can resubmit once the build catches up.
    min_cov = float(os.environ.get("MIN_COVERAGE", "0.0"))
    if loader.coverage < min_cov:
        print(f"run={run_id} coverage={loader.coverage:.3f} < MIN_COVERAGE={min_cov}; "
              f"skipping (will be retried later)", flush=True)
        return

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Muon on 2D hidden matrices; Adam on the (tied) embedding/head.
    muon_params, adam_params = [], []
    for n, p in model.named_parameters():
        if p is model.wte.weight or p is model.lm_head.weight:
            continue  # handled once below
        (muon_params if p.ndim == 2 else adam_params).append(p)
    adam_params.append(model.lm_head.weight)  # tied embed+head
    opt_adam = torch.optim.AdamW(adam_params, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(math.pi * t))  # cosine to 0.1x

    print(f"run={run_id} params={n_params/1e6:.1f}M pools={loader.n_pools} "
          f"coverage={loader.coverage:.3f} steps={steps} B={B} T={T} "
          f"budget={tokens_budget:,}", flush=True)

    model.train()
    t0 = time.time()
    last_loss = float("nan")
    for step in range(steps):
        x, y = loader.next_batch(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        s = lr_scale(step)
        for g in opt_adam.param_groups:
            g["lr"] = lr_adam * s
        for g in opt_muon.param_groups:
            g["lr"] = lr_muon * s
        opt_adam.step()
        opt_muon.step()
        opt_adam.zero_grad(set_to_none=True)
        opt_muon.zero_grad(set_to_none=True)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            last_loss = loss.item()
            print(f"  step {step:>5}/{steps}  loss {last_loss:.4f}  "
                  f"lr {lr_adam*s:.2e}  {time.time()-t0:.0f}s", flush=True)

    train_secs = time.time() - t0

    # ---- inline Minerva masked PPL (full multi-set scoring is done later via swarm.py reeval) ----
    md = os.environ.get("MINERVA_MAX_DOCS")
    m = eval_ppl.score(model, device, eval_ppl.EVAL_SETS["minerva"],
                       max_docs=int(md) if md else None, max_len=T)
    metrics = dict(minerva_bpb=m["bpb"], minerva_ppl=m["ppl"], n_docs=m["n_docs"])
    print(f"  minerva_bpb={m['bpb']:.4f} ppl={m['ppl']:.3f} (docs={m['n_docs']})", flush=True)

    # ---- save checkpoint (to /checkpoint) + result (to shared /datasets) ----
    ck_dir = os.path.join(CKPT_DIR, run_id)
    os.makedirs(ck_dir, exist_ok=True)
    if int(os.environ.get("SAVE_CKPT", "1")):
        torch.save(dict(model=model.state_dict(), config=cfg.to_dict(),
                        run_id=run_id), os.path.join(ck_dir, "model.pt"))

    result = dict(run_id=run_id, n_params=int(n_params), pools_used=loader.n_pools,
                  coverage=loader.coverage, steps=steps, tokens=int(steps * B * T),
                  final_train_loss=last_loss, train_secs=round(train_secs, 1),
                  weights=weights, **metrics)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{run_id}.json"), "w") as f:
        json.dump(result, f)
    print(f"wrote {RESULT_DIR}/{run_id}.json", flush=True)


if __name__ == "__main__":
    main()
