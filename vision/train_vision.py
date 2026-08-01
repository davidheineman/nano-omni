"""Molmo-style vision SFT on top of a modded-nanogpt GPT backbone.

A second file that leaves ``train_gpt.py`` unchanged (aside from the ``__main__`` guard that makes
its ``GPT`` importable + an env-gated checkpoint flag). It stitches a from-scratch SigLIP ViT +
attention-pool connector onto a GPT pretrained by the speedrun and runs single-image supervised
fine-tuning, reporting masked validation loss. See VISION_SFT.md for setup + how this differs from
the real Molmo 2.

Data path -> model:
  image -> MulticropPreprocessor -> crops[n,729,588] + <im_patch> token layout
        -> SiglipViT -> 1152-d patch features
        -> Connector (2x2 attention-meanq pool -> SwiGLU projector) -> model_dim features
  text+<im_patch> tokens -> GPT.embed -> features ADDED at <im_patch> positions -> GPT blocks -> lm_head
  loss = masked cross-entropy over assistant-response tokens only.

Run from the repo root (needs CUDA + FlashAttention-3, i.e. Hopper; this file lives in vision/,
train_gpt.py stays at the repo root). The defaults reproduce the verified run — newest
logs/*/state_step*.pt backbone + results/vision/siglip_so400m_378.pt + data/vision/molmo2_sft_simple:
  torchrun --standalone --nproc_per_node=1 vision/train_vision.py
  # smoke (no data/weights): ... --synthetic --backbone "" --siglip ""
  # override:                ... --backbone <ckpt> --hf_dataset <name> --max_steps 1000

The vision modules + preprocessor below are pure torch/numpy and import fine on CPU (the heavyweight
``import train_gpt`` is deferred to :func:`build_backbone`), so the preprocessor alignment invariant
can be unit-tested without a GPU.
"""
import argparse
import contextlib
import dataclasses
import json
import math
import os
import queue
import statistics
import threading
import time
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperparameters (nanogpt style: everything at the top)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class VisionConfig:
    # SigLIP-so400m/14 @ 378px -- must match the converted checkpoint (olmo SIGLIP_VISION_BACKBONE)
    image_size: int = 378
    patch_size: int = 14
    vit_dim: int = 1152
    vit_layers: int = 27
    vit_heads: int = 16
    vit_head_dim: int = 72
    vit_mlp_dim: int = 4304
    vit_eps: float = 1e-6
    # crop / pooling (olmo MultiCropConfig defaults for molmo2, crop_mode="overlap-and-resize-c2")
    max_crops: int = 6              # olmo MultiCropConfig max_crops (SFT)
    high_res_max_crops: int = 24    # olmo high_res_max_crops (SFT); informational for our single-scale port
    max_multi_image_crops: int = 8  # crops per image when an example has multiple images (sft.py:275)
    max_images: int = 5             # cap images per example (sft.py:276)
    overlap_margins: Tuple[int, int] = (4, 4)
    pooling_h: int = 2
    pooling_w: int = 2
    use_col_tokens: bool = True


@dataclasses.dataclass
class TrainConfig:
    backbone: str = "auto"      # "auto" -> newest logs/*/state_step*.pt; "" -> random GPT; or an explicit path
    siglip: str = "results/vision/siglip_so400m_378.pt"   # "" -> random ViT (from vision/convert_siglip.py)
    out_dir: str = "results/vision/checkpoints"
    # data source (exactly one): --data_dir (local), --hf_dataset <name>, or --synthetic
    data_dir: str = "data/vision/molmo2_sft_simple"   # {train,val}.jsonl + images/ (from data/vision/molmo2_sft_simple.py)
    mix_dir: str = ""           # Molmo2-recipe mix from data/vision/molmo2_sft_full.py (mixture.json + <src>__<split>.jsonl); takes precedence
    synthetic: bool = False     # random images + trivial Q/A; no data/weights needed (smoke test)
    hf_dataset: str = ""        # a single HF dataset with bundled images, e.g. HuggingFaceM4/ChartQA
    hf_train_split: str = "train"
    hf_val_split: str = "val"
    # optimization (defaults reproduce the verified 300-step ChartQA/molmo2_sft_simple run on 1xH200)
    seq_len: int = 2048         # max packed tokens per micro-sequence
    device_batch_size: int = 2  # examples per micro-batch (packed into one 1-D sequence)
    max_steps: int = 300
    warmup_steps: int = 200
    connector_lr: float = 2e-4  # connector trains from scratch -> highest LR
    vit_lr: float = 6e-6        # pretrained ViT -> tiny LR
    llm_lr: float = 2e-5        # pretrained LLM -> small LR
    freeze_vit: bool = True     # DEFAULT ON: freeze the SigLIP ViT (no bwd through it) -> ~1.5x throughput,
                                # no quality change at this scale. --no-freeze_vit to also fine-tune the ViT.
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_every: int = 50
    val_batches: int = 8          # (synthetic/hf paths) number of val micro-batches
    val_examples_per_source: int = 128  # (--data_dir path) fixed held-out val examples scored per source
    text_val_blocks: int = 16   # held-out FineWeb LM-loss blocks (of seq_len) scored each val step (0=off);
                                # monitors whether the LLM's text ability degrades during vision SFT
    seed: int = 90218
    max_minutes: float = 0.0      # wallclock training budget (0 = disabled; stop at max_steps instead)
    metrics_out: str = ""         # JSONL metrics log (train_loss / per-source val / MFU / tokens_per_s); rank-0 only
    val_hf: str = ""              # HF repo built by data/vision/molmo2_sft_build_validation.py; pull VAL from
                                  # there (per-source, so the curve uses the shared portable set). Overrides
                                  # local val. Reconstructed to a mix-dir layout -> same per-source val path.
    val_hf_split: str = "full"    # which split of --val_hf to validate on ("full" or "simple")


# ---------------------------------------------------------------------------
# Image special tokens -- reuse unused GPT-2 vocab slots (real tokens are 0..50256;
# the model pads vocab to 50304, so 50257..50303 are free rows in embed / value_embeds).
# ---------------------------------------------------------------------------
IM_PATCH_ID = 50257           # <im_patch>   high-res + (default) global patch placeholder
IM_LOW_ID = 50258             # <im_low>     (unused unless use_low_res_token_global_crops)
IM_START_ID = 50259           # <im_start>
IM_END_ID = 50260             # <im_end>
IM_COL_ID = 50261             # <im_col>     end-of-row marker
LOW_RES_IM_START_ID = 50262   # <low_res_im_start>
EOT_ID = 50256                # <|endoftext|> (GPT-2), used as turn/sequence delimiter
IMAGE_TOKENS = (IM_PATCH_ID, IM_LOW_ID, IM_START_ID, IM_END_ID, IM_COL_ID, LOW_RES_IM_START_ID)

# SigLIP attention precision: fp32 by default (upstream fidelity); VIT_ATTN_BF16=1 -> bf16 (faster).
_VIT_FP32_ATTN = not bool(os.environ.get("VIT_ATTN_BF16"))


# ===========================================================================
# 1. SigLIP ViT  (port of olmo/nn/image_vit.py :: SiglipVisionTransformer)
#    Param names mirror the olmo module so a converted checkpoint loads directly.
# ===========================================================================
class ViTAttention(nn.Module):
    def __init__(self, cfg: VisionConfig, input_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.vit_dim
        in_dim = input_dim or d
        self.wq = nn.Linear(in_dim, cfg.vit_heads * cfg.vit_head_dim, bias=True)
        self.wk = nn.Linear(in_dim, cfg.vit_heads * cfg.vit_head_dim, bias=True)
        self.wv = nn.Linear(in_dim, cfg.vit_heads * cfg.vit_head_dim, bias=True)
        self.wo = nn.Linear(cfg.vit_heads * cfg.vit_head_dim, d, bias=True)

    def forward(self, q_in: torch.Tensor, kv_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv_in = q_in if kv_in is None else kv_in
        B, Nq = q_in.shape[:2]
        Nk = kv_in.shape[1]
        h, hd = self.cfg.vit_heads, self.cfg.vit_head_dim
        q = self.wq(q_in).view(B, Nq, h, hd)
        k = self.wk(kv_in).view(B, Nk, h, hd)
        v = self.wv(kv_in).view(B, Nk, h, hd)
        # SigLIP upstream uses float32 attention for numerical fidelity, but fp32 SDPA runs far below
        # the bf16 tensor-core rate on H200. VIT_ATTN_BF16=1 keeps it in bf16 (big ViT speedup; the ViT
        # is being fine-tuned anyway so it absorbs the small precision change).
        if _VIT_FP32_ATTN:
            q, k, v = q.float(), k.float(), v.float()
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)
        out = out.reshape(B, Nq, h * hd).to(q_in.dtype)
        return self.wo(out)


class ViTMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.vit_dim, cfg.vit_mlp_dim, bias=True)
        self.w2 = nn.Linear(cfg.vit_mlp_dim, cfg.vit_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")  # gelu_pytorch_tanh

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.attention = ViTAttention(cfg)
        self.feed_forward = ViTMLP(cfg)
        self.attention_norm = nn.LayerNorm(cfg.vit_dim, eps=cfg.vit_eps)
        self.ffn_norm = nn.LayerNorm(cfg.vit_dim, eps=cfg.vit_eps)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class BlockCollection(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(cfg) for _ in range(cfg.vit_layers)])

    def forward(self, x):
        for r in self.resblocks:
            x = r(x)
        return x  # we only need the last layer (olmo default vit_layers=(-1,))


class SiglipViT(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        n_pos = (cfg.image_size // cfg.patch_size) ** 2  # 27*27 = 729
        self.positional_embedding = nn.Parameter(torch.zeros(n_pos, cfg.vit_dim))
        self.patch_embedding = nn.Linear(cfg.patch_size * cfg.patch_size * 3, cfg.vit_dim, bias=True)
        self.transformer = BlockCollection(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_crops, n_patches=729, 588) -> (n_crops, 729, 1152)."""
        x = self.patch_embedding(x)
        x = x + self.positional_embedding[None].to(x.dtype)  # fixed 27x27 grid, no interpolation
        return self.transformer(x)


# ===========================================================================
# 2. Connector  (olmo/nn/vision_backbone.py :: attention_meanq pool + ImageProjectorMLP)
# ===========================================================================
class Connector(nn.Module):
    def __init__(self, cfg: VisionConfig, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.vit_dim
        # attention pool: 1 query (the mean of the 2x2 group) attends over the group members
        self.pool = ViTAttention(cfg, input_dim=d)
        # SwiGLU projector 1152 -> out_dim (Molmo sizes hidden from the LLM's mlp_hidden_size; we
        # use 4*out_dim, a self-contained choice since the nanogpt GPT has no such config).
        hidden = hidden or (4 * out_dim)
        self.w1 = nn.Linear(d, hidden // 2, bias=False)
        self.w3 = nn.Linear(d, hidden // 2, bias=False)
        self.w2 = nn.Linear(hidden // 2, out_dim, bias=False)

    def forward(self, feats: torch.Tensor, pooled_idx: torch.Tensor) -> torch.Tensor:
        """feats: (n_crops, 729, 1152) ; pooled_idx: (n_pool, 4) into flattened crop patches.

        Returns (n_valid_pool, out_dim) in the order the <im_patch> tokens appear.
        """
        dim = feats.shape[-1]
        flat = feats.reshape(-1, dim)                          # (n_crops*729, 1152)
        valid = pooled_idx >= 0                                # (n_pool, 4)
        gather = flat[pooled_idx.clamp(min=0)]                 # (n_pool, 4, 1152)
        gather = gather * valid.float()[:, :, None]
        query = gather.mean(dim=1, keepdim=True)               # (n_pool, 1, 1152)
        pooled = self.pool(query, gather)[:, 0]                # (n_pool, 1152)
        x = self.w2(F.silu(self.w1(pooled)) * self.w3(pooled))  # (n_pool, out_dim)
        # Drop rows whose whole 2x2 group was padding (none in practice, but keep the invariant).
        keep = valid.any(dim=1)
        return x[keep]


# ===========================================================================
# 3. Multicrop preprocessor  (port of olmo/preprocessing/{image,multicrop}_preprocessor.py,
#    crop_mode="overlap-and-resize-c2"). Pure numpy/torch-cpu, unit-testable.
# ===========================================================================
def _select_tiling(h, w, patch_size, max_num_crops):
    tilings = [(i, j) for i in range(1, max_num_crops + 1) for j in range(1, max_num_crops + 1)
               if i * j <= max_num_crops]
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate = np.array(tilings, dtype=np.int32)
    resolutions = candidate * patch_size
    original = np.stack([h, w]).astype(np.float32)
    with np.errstate(divide="ignore"):
        scale_d = resolutions.astype(np.float32) / original
    required = np.min(scale_d, axis=-1, keepdims=True)
    if np.all(required < 1):
        ix = int(np.argmax(required))
    else:
        required = np.where(required < 1.0, 1e9, required)
        ix = int(np.argmin(required))
    return candidate[ix]


def _siglip_resize(image_u8: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize (antialias off, matching olmo) + siglip normalize to [-1, 1]."""
    t = torch.from_numpy(image_u8).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False, antialias=False)
    t = t.clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy()
    return (t / 255.0) * 2.0 - 1.0  # siglip: x*2-1 on [0,1]


def _pixels_to_patches(arr: np.ndarray, patch: int) -> np.ndarray:
    n, h, w, c = arr.shape
    hp, wp = h // patch, w // patch
    arr = arr.reshape(n, hp, patch, wp, patch, c).transpose(0, 1, 3, 2, 4, 5)
    return arr.reshape(n, hp * wp, patch * patch * c)


def _arange_for_pooling(idx_arr: np.ndarray, ph: int, pw: int) -> np.ndarray:
    h_pad = ph * ((idx_arr.shape[0] + ph - 1) // ph) - idx_arr.shape[0]
    w_pad = pw * ((idx_arr.shape[1] + pw - 1) // pw) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
                     mode="constant", constant_values=-1)
    H, W = idx_arr.shape
    out = idx_arr.reshape(H // ph, ph, W // pw, pw).transpose(0, 2, 1, 3)
    return out.reshape(H // ph, W // pw, ph * pw)


class MulticropPreprocessor:
    """Turn a HxWx3 uint8 image into (tokens, crop_patches, pooling_idx)."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.crop = cfg.image_size
        self.patch = cfg.patch_size
        self.cpp = self.crop // self.patch  # crop patches per dim (27)

    def __call__(self, image: np.ndarray, max_crops: Optional[int] = None):
        cfg = self.cfg
        mc = max_crops or cfg.max_crops   # multi-image examples cap crops lower (olmo max_multi_image_crops)
        lm, rm = cfg.overlap_margins
        total_margin = self.patch * (lm + rm)
        window_patches = self.cpp - (lm + rm)
        window_size = window_patches * self.patch
        H, W = image.shape[:2]

        tiling = _select_tiling(max(H - total_margin, 1), max(W - total_margin, 1),
                                window_size, mc)
        src = _siglip_resize(
            image, (tiling[0] * window_size + total_margin, tiling[1] * window_size + total_margin))

        # Slide a window_size step over src, extracting overlapping crop-sized tiles.
        n_crops = int(tiling[0] * tiling[1])
        crop_arr = np.zeros([n_crops, self.crop, self.crop, 3], dtype=np.float32)
        patch_idx_arr = np.zeros([n_crops, self.cpp, self.cpp], dtype=np.int32)
        on_crop = 0
        for i in range(tiling[0]):
            y0 = i * window_size
            for j in range(tiling[1]):
                x0 = j * window_size
                crop_arr[on_crop] = src[y0:y0 + self.crop, x0:x0 + self.crop]
                pidx = np.arange(self.cpp * self.cpp).reshape(self.cpp, self.cpp)
                pidx += on_crop * self.cpp * self.cpp
                if i != 0:
                    pidx[:lm, :] = -1
                if j != 0:
                    pidx[:, :lm] = -1
                if i != tiling[0] - 1:
                    pidx[-rm:, :] = -1
                if j != tiling[1] - 1:
                    pidx[:, -rm:] = -1
                patch_idx_arr[on_crop] = pidx
                on_crop += 1
        # Reorder crop-by-crop -> left-to-right reading order, drop overlap (-1), stitch.
        pia = patch_idx_arr.reshape(tiling[0], tiling[1], self.cpp, self.cpp)
        pia = pia.transpose(0, 2, 1, 3).reshape(-1)
        pia = pia[pia >= 0].reshape(src.shape[0] // self.patch, src.shape[1] // self.patch)

        pooling_idx = _arange_for_pooling(pia, cfg.pooling_h, cfg.pooling_w)
        h, w = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape(-1, cfg.pooling_h * cfg.pooling_w)

        # High-res token layout: [im_start] (im_patch*w [+im_col]) x h [im_end]
        per_row = np.full(w, IM_PATCH_ID, dtype=np.int32)
        if cfg.use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_ID]])
        hires_tokens = [np.array([IM_START_ID], np.int32), np.tile(per_row, h),
                        np.array([IM_END_ID], np.int32)]

        # Global low-res crop (whole image resized to one 378 crop).
        resized = _siglip_resize(image, (self.crop, self.crop))[None]
        crop_arr = np.concatenate([resized, crop_arr], 0)             # global crop is index 0
        resize_idx = np.arange(self.cpp * self.cpp).reshape(self.cpp, self.cpp)
        resize_pool = _arange_for_pooling(resize_idx, cfg.pooling_h, cfg.pooling_w)
        h2, w2 = resize_pool.shape[:2]
        resize_pool = resize_pool.reshape(-1, cfg.pooling_h * cfg.pooling_w)

        # Global crop goes first -> shift high-res indices by one crop worth of patches.
        shift = self.cpp * self.cpp
        pooling_idx = np.where(pooling_idx >= 0, pooling_idx + shift, -1)
        pooling_idx = np.concatenate([resize_pool, pooling_idx], 0)

        gper = np.full(w2, IM_PATCH_ID, dtype=np.int32)
        if cfg.use_col_tokens:
            gper = np.concatenate([gper, [IM_COL_ID]])
        global_tokens = [np.array([IM_START_ID], np.int32), np.tile(gper, h2),
                         np.array([IM_END_ID], np.int32)]

        tokens = np.concatenate(global_tokens + hires_tokens, 0)
        images = _pixels_to_patches(crop_arr, self.patch).astype(np.float32)  # (n_crops+1, 729, 588)
        return tokens, images, pooling_idx.astype(np.int64)


# ===========================================================================
# 4. VisionGPT: wrap the speedrun GPT, inject image features at <im_patch>.
# ===========================================================================
class _ImageInjectingEmbed(nn.Module):
    """Drop-in replacement for GPT.embed that adds pending image features at <im_patch> positions.

    Wrapping .embed (rather than editing GPT.forward) is what lets us inject vision features without
    touching the speedrun's entangled forward (it reads input_ids directly for value/bigram embeds).
    """

    def __init__(self, base: nn.Embedding):
        super().__init__()
        self.base = base
        self._pending: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (features, mask)

    def set_pending(self, features: torch.Tensor, mask: torch.Tensor):
        self._pending = (features, mask)

    def clear_pending(self):
        self._pending = None

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.base(input_seq)
        if self._pending is not None:
            feats, mask = self._pending
            x = x.clone()
            x[mask] = x[mask] + feats.to(x.dtype)
            self._pending = None
        return x


class VisionGPT(nn.Module):
    def __init__(self, gpt: nn.Module, vcfg: VisionConfig, model_dim: int):
        super().__init__()
        self.gpt = gpt
        self.vit = SiglipViT(vcfg)
        self.connector = Connector(vcfg, out_dim=model_dim)
        self.freeze_vit = False   # set by main(); when True the ViT runs under no_grad (no backward)
        # Wrap embed AFTER the backbone checkpoint has been loaded (see build_backbone).
        self.gpt.embed = _ImageInjectingEmbed(self.gpt.embed)

    def forward(self, batch, schedule_cfg):
        """`batch` is the dict from collate_packed (all sequence tensors 1-D, varlen-packed).

        The <im_patch> positions in batch["input_seq"] line up 1:1 with connector output rows.
        """
        input_seq, images, pooled_idx = batch["input_seq"], batch["images"], batch["pooled_idx"]
        im_mask = (input_seq == IM_PATCH_ID)
        if images.numel() > 0 and bool(im_mask.any()):
            with torch.no_grad() if self.freeze_vit else contextlib.nullcontext():
                feats = self.vit(images.to(self.vit.patch_embedding.weight.dtype))
            img_features = self.connector(feats, pooled_idx)      # (n_valid_pool, model_dim)
            assert int(im_mask.sum()) == img_features.shape[0], \
                f"<im_patch> count {int(im_mask.sum())} != connector rows {img_features.shape[0]}"
            self.gpt.embed.set_pending(img_features, im_mask)
        else:
            self.gpt.embed.clear_pending()  # text-only batch: nothing to splice

        # Force the GPT's clean eval-branch loss (per-token CE, no FP8/MTP) while keeping grads.
        was_training = self.gpt.training
        self.gpt.eval()
        loss_per_token = self.gpt(input_seq, batch["target_seq"], batch["seqlens"],
                                  batch["bigram_input_seq"], schedule_cfg)
        if was_training:
            self.gpt.train()

        lm = batch["loss_mask"].reshape(-1).float()
        return (loss_per_token * lm).sum() / lm.sum().clamp(min=1.0)


# ===========================================================================
# 5. Backbone loading (deferred import of train_gpt so this file imports on CPU)
# ===========================================================================
def build_backbone(ckpt_path: str, vcfg: VisionConfig):
    """Import the (guarded) speedrun module, instantiate GPT, load the checkpoint, wrap vision."""
    # SFT uses the GPT eval-branch (no FP8), and some compute nodes carry a stray/broken HTTP proxy
    # that breaks train_gpt's FA3 get_kernel fetch -- set both before importing train_gpt.
    os.environ.setdefault("DISABLE_FP8", "1")
    for _p in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(_p, None)
    # This file lives in vision/; train_gpt.py + triton_kernels.py are at the repo root. Put the
    # repo root on sys.path and point sys.argv[0] at train_gpt.py so its top-level
    # `open(dirname(sys.argv[0])/triton_kernels.py)` (it reads its own source for logging) resolves.
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    sys.argv[0] = os.path.join(repo_root, "train_gpt.py")
    import train_gpt  # runs train_gpt top-level setup (dist/device/FA3), NOT its training driver
    from train_gpt import GPT, device, world_size

    model = GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
                max_seq_len=8192).cuda()
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    for name in ["attn_gate_bank", "ve_gate_bank", "qk_bank", "vo_bank", "mlp_bank",
                 "mudd_w1", "mudd_w2", "mudd_b2"]:
        p = getattr(model, name, None)
        if p is not None:
            p.data = p.data.bfloat16()

    if ckpt_path:
        # The speedrun checkpoint was pickled from train_gpt.py as __main__ and its "optimizer"
        # entry holds custom classes (ParamConfig, ...). Inject train_gpt's classes into __main__
        # so the full dict unpickles; we only keep ["model"].
        import __main__ as _main
        for _n in dir(train_gpt):
            _o = getattr(train_gpt, _n)
            if isinstance(_o, type) and not hasattr(_main, _n):
                setattr(_main, _n, _o)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}  # strip torch.compile prefix
        # qk_bank / vo_bank are padded to a multiple of world_size. The checkpoint was trained on
        # 8 GPUs (extra rows are zeros); this run may use fewer. Slice dim-0 to the model's shape
        # (the forward only uses the first num_qk_groups / num_attn_layers*2 rows anyway).
        msd = model.state_dict()
        to_load = {}
        for k, v in sd.items():
            if k not in msd:
                continue
            mv = msd[k]
            if v.shape == mv.shape:
                to_load[k] = v
            elif v.dim() == mv.dim() and v.shape[1:] == mv.shape[1:] and v.shape[0] >= mv.shape[0]:
                to_load[k] = v[: mv.shape[0]]
                print(f"[backbone] sliced {k}: {tuple(v.shape)} -> {tuple(mv.shape)}")
            else:
                print(f"[backbone] skip shape-mismatch {k}: ckpt {tuple(v.shape)} vs model {tuple(mv.shape)}")
        missing, unexpected = model.load_state_dict(to_load, strict=False)
        print(f"[backbone] loaded {len(to_load)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[backbone] no --backbone given: using randomly-initialized GPT (smoke test only)")

    vgpt = VisionGPT(model, vcfg, model_dim=768).cuda()
    return vgpt, device, world_size


# ===========================================================================
# 6. Data: chat template + tokenize + collate to a 1-D packed batch
# ===========================================================================
@lru_cache(maxsize=1)
def _get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def encode_example(enc, tokens_image, question: str, answer: str, seq_len: int):
    r"""Build (input_ids, target_ids, loss_mask) for one <image?, Q, A> example.

    Template: [image tokens] "\nUser: <q>\nAssistant: <a><eot>". Loss on the answer + eot only.
    Non-image (text-only) examples pass tokens_image=None.
    """
    q_ids = enc.encode(f"\nUser: {question}\nAssistant:")
    a_ids = enc.encode(f" {answer}") + [EOT_ID]
    parts = []
    if tokens_image is not None:
        parts.append(np.asarray(tokens_image, dtype=np.int64))
    parts.append(np.asarray(q_ids, dtype=np.int64))
    prompt_len = sum(len(p) for p in parts)
    parts.append(np.asarray(a_ids, dtype=np.int64))
    ids = np.concatenate(parts)[:seq_len]

    # next-token targets; loss only on answer tokens (positions >= prompt_len-1 predict the answer)
    input_ids = ids[:-1]
    target_ids = ids[1:]
    loss_mask = np.zeros_like(target_ids)
    loss_mask[max(prompt_len - 1, 0):] = 1
    loss_mask[np.isin(target_ids, IMAGE_TOKENS)] = 0  # never supervise image-placeholder positions
    return input_ids, target_ids, loss_mask


def collate_packed(examples, seq_len, pre: MulticropPreprocessor):
    """Pack a list of dicts {image(uint8 or None), question, answer} into one 1-D varlen batch."""
    enc = _get_tokenizer()
    all_in, all_tgt, all_mask, seqlens = [], [], [], []
    all_images, all_pool = [], []
    pool_offset = 0  # crops are concatenated across examples; shift each example's pooled idx
    for ex in examples:
        toks_img = None
        if ex.get("image") is not None:
            toks_img, images, pooled = pre(ex["image"])
            pooled = np.where(pooled >= 0, pooled + pool_offset, -1)
            pool_offset += images.shape[0] * (pre.cpp * pre.cpp)
            all_images.append(images)
            all_pool.append(pooled)
        inp, tgt, msk = encode_example(enc, toks_img, ex["question"], ex["answer"], seq_len)
        all_in.append(inp); all_tgt.append(tgt); all_mask.append(msk)
        seqlens.append(len(inp))

    input_ids = np.concatenate(all_in)
    target_ids = np.concatenate(all_tgt)
    mask_ids = np.concatenate(all_mask)
    # FlashAttention-3 varlen requires the total packed length to be a multiple of 16
    # (train_gpt CausalSelfAttention asserts T % 16 == 0). Pad with a filler doc.
    pad = (-input_ids.shape[0]) % 16
    if pad:
        input_ids = np.concatenate([input_ids, np.full(pad, EOT_ID, np.int64)])
        target_ids = np.concatenate([target_ids, np.zeros(pad, np.int64)])
        mask_ids = np.concatenate([mask_ids, np.zeros(pad, np.int64)])
        seqlens.append(pad)
    input_seq = torch.from_numpy(input_ids).to(torch.int32)
    target_seq = torch.from_numpy(target_ids).long()
    loss_mask = torch.from_numpy(mask_ids).long()
    cu = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
    images = torch.from_numpy(np.concatenate(all_images, 0)) if all_images else torch.zeros(0)
    pooled = torch.from_numpy(np.concatenate(all_pool, 0)) if all_pool else torch.zeros(0, 4).long()
    return dict(input_seq=input_seq, target_seq=target_seq, loss_mask=loss_mask,
                seqlens=cu, images=images, pooled_idx=pooled)


def iter_synthetic_examples(seed: int, image_prob: float = 0.8):
    """Smoke-test data: random images + short random-token answers. No data download needed.

    The answer is a short run of real GPT-2 tokens so the loss is non-trivial (not a constant
    string the model can memorize in one step), which lets us see val loss actually move.
    """
    enc = _get_tokenizer()
    rng = np.random.RandomState(seed)
    vocab = np.arange(1000, 6000)  # a slice of ordinary GPT-2 token ids for answers
    while True:
        has_img = rng.random() < image_prob
        image = None
        if has_img:
            H = int(rng.randint(200, 900)); W = int(rng.randint(200, 900))
            image = rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        q = "Describe the image." if has_img else "Continue the sequence."
        a = enc.decode([int(t) for t in rng.choice(vocab, size=int(rng.randint(4, 24)))])
        yield dict(image=image, question=q, answer=a)


def iter_hf_examples(name: str, split: str, seed: int):
    """Real image SFT data straight from a HuggingFace dataset with bundled images (no olmo needed).

    Handles common VQA/chart field names (image/query/question/label/answer). This is the simplest
    real-data path; the full Molmo 2 mixture (video/pointing/tracking) is out of scope here.
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    rng = np.random.RandomState(seed)
    n = len(ds)
    order = rng.permutation(n)
    i = 0
    while True:
        ex = ds[int(order[i % n])]; i += 1
        img = ex.get("image")
        if img is not None:
            img = np.asarray(img.convert("RGB")) if hasattr(img, "convert") else np.asarray(img)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
        q = ex.get("question") or ex.get("query") or ex.get("instruction") or ""
        a = ex.get("answer") or ex.get("label") or ex.get("answers") or ""
        if isinstance(a, list):
            a = a[0] if a else ""
        yield dict(image=img, question=str(q), answer=str(a))


def iter_local_examples(data_dir: str, split: str, seed: int):
    """Read a local dataset built by data/vision/molmo2_sft_simple.py ({split}.jsonl + images/)."""
    import json
    from PIL import Image
    rows = [json.loads(l) for l in open(os.path.join(data_dir, f"{split}.jsonl"))]
    if not rows:
        raise SystemExit(f"{data_dir}/{split}.jsonl is empty")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(rows))
    i = 0
    while True:
        r = rows[int(order[i % len(rows)])]; i += 1
        img = np.asarray(Image.open(os.path.join(data_dir, r["image"])).convert("RGB"))
        yield dict(image=img, question=r["question"], answer=r["answer"])


def load_val_by_source(data_dir: str, per_source: int, seed: int):
    """A FIXED held-out val set grouped by source (for a comparable, low-variance curve).

    Unlike the rolling val iterator, this returns the *same* deterministically-chosen examples every
    time validate() runs, so val points at different steps measure the identical set. Rows keep their
    "source" so we can report per-source loss (much more informative than one blended number).
    """
    import json
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "val.jsonl"))]
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    rng = np.random.RandomState(seed)
    fixed = {}
    for src in sorted(by_src):
        rs = by_src[src]
        idx = rng.permutation(len(rs))[:per_source]
        fixed[src] = [rs[int(i)] for i in idx]
    return fixed


def load_example_row(data_dir: str, r: dict):
    from PIL import Image
    img = np.asarray(Image.open(os.path.join(data_dir, r["image"])).convert("RGB"))
    return dict(image=img, question=r["question"], answer=r["answer"])


# ===========================================================================
# 6b. Molmo2-recipe mixture path (consumes data/vision/molmo2_sft_full.py output)
#     mixture.json + <source>__<split>.jsonl rows {image|[image], convos, n_subsegments, message_weight}.
#     Reproduces Molmo2's sqrt-size weighted sampler + root_subsegments_root_tokens loss weighting +
#     multi-turn + multi-image ("Image N" prefixes). Loss weights ride in `loss_mask` (a float weight
#     vector), which VisionGPT.forward already reduces as a weighted mean -- no forward change needed.
# ===========================================================================
def load_mixture(mix_dir: str):
    """-> (source->sampling_prob, source->message_weight). Sampling = sqrt(size) normed within group
    x group weight, then globally normed (port of olmo DataLoaderConfig._build_mixture)."""
    import json
    m = json.load(open(os.path.join(mix_dir, "mixture.json")))
    rate, mw = {}, {}
    for g in m["groups"].values():
        sizes = {d["source"]: math.sqrt(max(d["size"], 1.0)) for d in g["datasets"]}
        tot = sum(sizes.values()) or 1.0
        for d in g["datasets"]:
            rate[d["source"]] = sizes[d["source"]] / tot * g["weight"]
            mw[d["source"]] = d.get("message_weight", 1.0)
    z = sum(rate.values()) or 1.0
    return {k: v / z for k, v in rate.items()}, mw


def _load_mix_rows(mix_dir: str, split: str):
    import glob, json
    out = {}
    for p in glob.glob(os.path.join(mix_dir, f"*__{split}.jsonl")):
        src = os.path.basename(p).split("__")[0]
        rows = [json.loads(l) for l in open(p)]
        if rows:
            out[src] = rows
    return out


def hf_val_to_mixdir(repo: str, hf_split: str, out_dir: str):
    """Download the portable val set (data/vision/molmo2_sft_build_validation.py) and reconstruct it into
    the local mix-dir layout (<source>__validation.jsonl + images/), so the existing per-source val path
    consumes it unchanged. Returns out_dir. Each HF config = one source; rows carry images + convos_json."""
    import json, datasets
    from huggingface_hub import get_dataset_config_names
    datasets.disable_progress_bars()
    os.makedirs(out_dir, exist_ok=True)
    for source in get_dataset_config_names(repo):
        try:
            ds = datasets.load_dataset(repo, source, split=hf_split)
        except Exception:
            continue  # this source has no rows for the requested split
        img_dir = os.path.join(out_dir, "images", source); os.makedirs(img_dir, exist_ok=True)
        rows, img_i = [], 0
        for ex in ds:
            rels = []
            for im in ex["images"]:
                rel = os.path.join("images", source, f"{img_i}.jpg")
                im.convert("RGB").save(os.path.join(out_dir, rel), quality=90); rels.append(rel); img_i += 1
            rows.append({"source": source,
                         "image": (rels if len(rels) != 1 else rels[0]) if rels else None,
                         "convos": json.loads(ex["convos_json"]),
                         "n_subsegments": ex["n_subsegments"], "message_weight": ex["message_weight"]})
        with open(os.path.join(out_dir, f"{source}__validation.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  [val_hf] {source} [{hf_split}]: {len(rows)} rows", flush=True)
    return out_dir


def _flatten_convos(row):
    """message_list subsegments -> one multi-turn stream sharing the image; n_subseg for root weighting."""
    convos = row.get("convos") or []
    turns = [t for c in convos for t in c]
    return turns, max(len(convos), 1)


def iter_mix_examples(mix_dir: str, split: str, seed: int):
    rate, mw = load_mixture(mix_dir)
    rows = _load_mix_rows(mix_dir, split)
    srcs = [s for s in rate if s in rows]
    if not srcs:
        raise SystemExit(f"{mix_dir}: no {split} rows for any source in mixture.json")
    probs = np.array([rate[s] for s in srcs], dtype=np.float64); probs /= probs.sum()
    rng = np.random.RandomState(seed)
    order = {s: rng.permutation(len(rows[s])) for s in srcs}
    ptr = {s: 0 for s in srcs}
    while True:
        s = srcs[int(rng.choice(len(srcs), p=probs))]
        if ptr[s] >= len(order[s]):
            order[s] = rng.permutation(len(rows[s])); ptr[s] = 0
        r = rows[s][int(order[s][ptr[s]])]; ptr[s] += 1
        turns, n_sub = _flatten_convos(r)
        if len(turns) < 2:
            continue
        img = r.get("image")
        imgs = img if isinstance(img, list) else ([img] if img else [])
        yield dict(images=imgs, turns=turns, n_subsegments=n_sub,
                   message_weight=mw.get(s, 1.0), source=s, mix_dir=mix_dir)


def mix_row_to_example(row, mix_dir):
    """A materialized mix row -> the example dict collate_mix/iter_mix_examples consume."""
    turns, n_sub = _flatten_convos(row)
    img = row.get("image")
    imgs = img if isinstance(img, list) else ([img] if img else [])
    return dict(images=imgs, turns=turns, n_subsegments=n_sub,
                message_weight=row.get("message_weight", 1.0), source=row.get("source"), mix_dir=mix_dir)


def _image_segment(pre: "MulticropPreprocessor", vcfg: VisionConfig, enc, imgs, mix_dir):
    """Preprocess 0+ images -> (token_ids, crops[N,729,588], pooled[n,4] offset within this example).
    Multi-image: each image after the first is prefixed with literal text 'Image {i+1}: '."""
    from PIL import Image
    if not imgs:
        return np.zeros(0, np.int64), np.zeros((0, pre.cpp * pre.cpp * 3), np.float32), np.zeros((0, 4), np.int64)
    mc = None if len(imgs) <= 1 else vcfg.max_multi_image_crops
    seg, crops, pooled, crop_off = [], [], [], 0
    for i, rel in enumerate(imgs[:vcfg.max_images]):
        arr = np.asarray(Image.open(os.path.join(mix_dir, rel)).convert("RGB"))
        toks, cr, pl = pre(arr, max_crops=mc)
        if i > 0:
            seg.append(np.asarray(enc.encode(f" Image {i + 1}: "), dtype=np.int64))
        seg.append(np.asarray(toks, dtype=np.int64))
        pl = np.where(pl >= 0, pl + crop_off, -1)
        crop_off += cr.shape[0] * (pre.cpp * pre.cpp)
        crops.append(cr); pooled.append(pl)
    return (np.concatenate(seg), np.concatenate(crops, 0), np.concatenate(pooled, 0))


def encode_mix_turns(enc, seg_ids, turns, msg_weight, n_subseg, seq_len):
    """Build (input_ids, target_ids, loss_weight) for [image seg][user0][asst0][user1][asst1]...
    Per-token loss weight on assistant tokens = message_weight / sqrt(#asst tokens) / sqrt(#subsegments)
    (Molmo2 root_subsegments_root_tokens); 0 elsewhere and on image-placeholder targets."""
    full, wfull = [], []
    if seg_ids.size:
        full.append(seg_ids); wfull.append(np.zeros(seg_ids.size))
    for k in range(0, len(turns) - 1, 2):
        u_ids = np.asarray(enc.encode(f"\nUser: {turns[k]}\nAssistant:"), np.int64)
        a_ids = np.asarray(enc.encode(f" {turns[k + 1]}") + [EOT_ID], np.int64)
        full.append(u_ids); wfull.append(np.zeros(u_ids.size))
        w = msg_weight / math.sqrt(max(a_ids.size, 1)) / math.sqrt(max(n_subseg, 1))
        full.append(a_ids); wfull.append(np.full(a_ids.size, w, dtype=np.float64))
    ids = np.concatenate(full)[:seq_len]
    wt = np.concatenate(wfull)[:seq_len]
    input_ids, target_ids = ids[:-1], ids[1:]
    loss_w = wt[1:].astype(np.float64).copy()
    loss_w[np.isin(target_ids, IMAGE_TOKENS)] = 0.0   # never supervise image placeholders
    return input_ids, target_ids, loss_w


def collate_mix(examples, seq_len, pre: "MulticropPreprocessor", vcfg: VisionConfig):
    """Pack mixture examples (multi-image, multi-turn, weighted) into one 1-D varlen batch."""
    enc = _get_tokenizer()
    all_in, all_tgt, all_w, seqlens, all_crops, all_pool = [], [], [], [], [], []
    pool_offset = 0
    for ex in examples:
        seg, crops, pooled = _image_segment(pre, vcfg, enc, ex["images"], ex["mix_dir"])
        inp, tgt, w = encode_mix_turns(enc, seg, ex["turns"], ex["message_weight"],
                                       ex["n_subsegments"], seq_len)
        # encode_mix_turns truncates the token stream to seq_len, which can cut trailing <im_patch>
        # tokens off a large (esp. multi-image) example while crops/pooled still carry every row.
        # The forward requires (# <im_patch>) == (# connector rows == pooled rows). <im_patch> tokens
        # all live in the front image segment and map 1:1, in order, to pooled rows -> the survivors
        # are a prefix, so keep the first k pooled rows (and only the crops those rows reference).
        k = int((inp == IM_PATCH_ID).sum())
        if k < pooled.shape[0]:
            pooled = pooled[:k]
            valid = pooled[pooled >= 0]
            n_cr = (int(valid.max()) // (pre.cpp * pre.cpp) + 1) if valid.size else 0
            crops = crops[:n_cr]
        if crops.shape[0]:
            pooled = np.where(pooled >= 0, pooled + pool_offset, -1)
            pool_offset += crops.shape[0] * (pre.cpp * pre.cpp)
            all_crops.append(crops); all_pool.append(pooled)
        all_in.append(inp); all_tgt.append(tgt); all_w.append(w); seqlens.append(len(inp))
    input_ids = np.concatenate(all_in); target_ids = np.concatenate(all_tgt); w_ids = np.concatenate(all_w)
    pad = (-input_ids.shape[0]) % 16
    if pad:
        input_ids = np.concatenate([input_ids, np.full(pad, EOT_ID, np.int64)])
        target_ids = np.concatenate([target_ids, np.zeros(pad, np.int64)])
        w_ids = np.concatenate([w_ids, np.zeros(pad, np.float64)])
        seqlens.append(pad)
    cu = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
    images = torch.from_numpy(np.concatenate(all_crops, 0)) if all_crops else torch.zeros(0)
    pooled = torch.from_numpy(np.concatenate(all_pool, 0)) if all_pool else torch.zeros(0, 4).long()
    return dict(input_seq=torch.from_numpy(input_ids).to(torch.int32),
                target_seq=torch.from_numpy(target_ids).long(),
                loss_mask=torch.from_numpy(w_ids).float(),   # float weights; forward does weighted mean
                seqlens=cu, images=images, pooled_idx=pooled)


def _train_iter(cfg: "TrainConfig", seed: int):
    """A fresh train-example iterator for the configured source, seeded by `seed` (each prefetch worker
    gets a distinct seed -> a distinct shuffle -> different data, so N workers don't duplicate work)."""
    if cfg.mix_dir:
        return iter_mix_examples(cfg.mix_dir, "train", seed)
    if cfg.synthetic:
        return iter_synthetic_examples(seed)
    if cfg.data_dir:
        return iter_local_examples(cfg.data_dir, "train", seed)
    if cfg.hf_dataset:
        return iter_hf_examples(cfg.hf_dataset, cfg.hf_train_split, seed)
    raise SystemExit("No data source: pass --data_dir, --hf_dataset <name>, or --synthetic.")


def _val_iter(cfg: "TrainConfig"):
    """Val-example iterator (rank-independent seed so every rank scores the identical held-out set).
    Only the rolling hf/synthetic path uses this; --data_dir/--mix_dir build a fixed per-source set."""
    s = cfg.seed + 7
    if cfg.mix_dir:
        return iter_mix_examples(cfg.mix_dir, "validation", s)
    if cfg.synthetic:
        return iter_synthetic_examples(s)
    if cfg.data_dir:
        return iter_local_examples(cfg.data_dir, "val", s)
    if cfg.hf_dataset:
        return iter_hf_examples(cfg.hf_dataset, cfg.hf_val_split, s)
    raise SystemExit("No data source: pass --data_dir, --hf_dataset <name>, or --synthetic.")


# ===========================================================================
# 7. Optimizer + LR schedule
# ===========================================================================
def build_optimizer(vgpt: VisionGPT, cfg: TrainConfig):
    # base_lr is stashed per group so the schedule just scales it (no hardcoded LR list in the loop).
    groups = [
        {"params": list(vgpt.connector.parameters()), "lr": cfg.connector_lr, "base_lr": cfg.connector_lr},
        {"params": list(vgpt.vit.parameters()), "lr": cfg.vit_lr, "base_lr": cfg.vit_lr},
        {"params": list(vgpt.gpt.parameters()), "lr": cfg.llm_lr, "base_lr": cfg.llm_lr},
    ]
    # Drop params/groups that don't train (e.g. a frozen ViT) so AdamW allocates no optimizer state for them.
    groups = [{**g, "params": [p for p in g["params"] if p.requires_grad]} for g in groups]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)


def lr_scale(step: int, elapsed_min: float, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    # Anneal over WALLCLOCK for a --max_minutes run (so the cosine actually reaches its floor even though
    # max_steps is just a high cap); fall back to step-fraction when there's no time budget.
    if cfg.max_minutes > 0:
        frac = elapsed_min / cfg.max_minutes
    else:
        frac = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))  # cosine to alpha_f=0.1


# ===========================================================================
# 8. Training loop
# ===========================================================================
class _Prefetcher:
    """Overlap the CPU collate of upcoming train batches with GPU compute, across N worker threads.

    The collate (PIL decode + multicrop preprocess + tiktoken) is dominated by C-level numpy/PIL/Rust ops
    that release the GIL, so multiple threads genuinely parallelize it. Each worker owns its own data
    iterator (distinct seed via iter_factory(w)) so there's no shared-iterator race, and all feed one
    queue (order-agnostic for SGD). Workers never touch CUDA (H2D copy stays on the main thread).
    DATA_WORKERS sets N; NO_PREFETCH=1 disables entirely.
    """

    def __init__(self, iter_factory, dbs, collate_cpu, n_workers=1):
        self._dbs, self._collate = dbs, collate_cpu
        self._q = queue.Queue(maxsize=2 * n_workers)
        for w in range(n_workers):
            it = iter_factory(w)
            threading.Thread(target=self._worker, args=(it,), daemon=True).start()

    def _worker(self, it):
        try:
            while True:
                self._q.put(self._collate([next(it) for _ in range(self._dbs)]))
        except Exception as e:  # forward to the consumer instead of dying silently
            self._q.put(e)

    def next(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        if isinstance(f.default, bool):
            # BooleanOptionalAction -> both --flag and --no-flag, so default-True bools can be disabled.
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    cfg = TrainConfig(**vars(ap.parse_args()))
    vcfg = VisionConfig()

    torch.manual_seed(cfg.seed)

    backbone = cfg.backbone
    if backbone == "auto":  # newest speedrun checkpoint (the pretrain job writes to logs/<run_id>/)
        import glob
        cands = sorted(glob.glob("logs/*/state_step*.pt"), key=os.path.getmtime)
        backbone = cands[-1] if cands else ""
        print(f"[backbone] auto -> {backbone or '(none found; random GPT)'}")
    vgpt, _, world_size = build_backbone(backbone, vcfg)
    # build_backbone imported train_gpt, which set up the process group (under torchrun) and these:
    from train_gpt import rank, master_process
    is_dp = world_size > 1
    if master_process:
        print(f"[dp] world_size={world_size} rank={rank} effective_batch={cfg.device_batch_size * world_size}", flush=True)

    # Load pretrained SigLIP into the ViT (keys mirror the converted checkpoint from convert_siglip.py).
    if cfg.siglip and os.path.exists(cfg.siglip):
        vit_sd = torch.load(cfg.siglip, map_location="cpu", weights_only=False)
        miss, unexp = vgpt.vit.load_state_dict(vit_sd, strict=False)
        print(f"[siglip] loaded {cfg.siglip}: missing={len(miss)} unexpected={len(unexp)}")
    else:
        print(f"[siglip] {cfg.siglip or '(none)'} not found: using randomly-initialized ViT (smoke only)")
    vgpt.vit.to(torch.bfloat16)

    if cfg.freeze_vit:  # frozen SigLIP feature extractor: no grad/optimizer state, no backward through it
        for p in vgpt.vit.parameters():
            p.requires_grad_(False)
        vgpt.freeze_vit = True
        if master_process:
            print("[freeze] ViT frozen (no backward); training connector + LLM only", flush=True)

    # The ViT is ~75% of train FLOPs and was running eager (27 blocks dispatched in Python per step).
    # Compile it in place (params/state_dict keys unchanged); dynamic=True since the crop count varies
    # per batch. Set NO_COMPILE=1 to disable. Biggest single MFU lever measured for this trainer.
    if not os.environ.get("NO_COMPILE"):
        vgpt.vit.compile(dynamic=True)
        if master_process:
            print("[compile] vit compiled (dynamic)", flush=True)

    from train_gpt import ForwardScheduleConfig, get_bigram_hash
    # Full attention within each doc (windows >= seq_len); SFT sequences are short.
    sched = ForwardScheduleConfig(mtp_weights=None, ws_short=cfg.seq_len,
                                  ws_long=cfg.seq_len, train_max_seq_len=cfg.seq_len)

    pre = MulticropPreprocessor(vcfg)
    opt = build_optimizer(vgpt, cfg)
    val_it = _val_iter(cfg)

    # ---- MFU bookkeeping: per-stack param counts (used in the analytic 6*N*D estimate) ----
    N_gpt = sum(p.numel() for p in vgpt.gpt.parameters())
    N_vit = sum(p.numel() for p in vgpt.vit.parameters())
    POOL = vcfg.pooling_h * vcfg.pooling_w        # ViT sees ~POOL x the pooled (<im_patch>) tokens
    VIT_COEF = 2.0 if cfg.freeze_vit else 6.0     # frozen ViT is forward-only (2ND); else fwd+bwd (6ND)
    PEAK_FLOPS = 989e12                            # H200 SXM bf16 tensor-core peak (no sparsity), per GPU
    if master_process:
        print(f"[mfu] params: gpt={N_gpt/1e6:.1f}M vit={N_vit/1e6:.1f}M ; peak={PEAK_FLOPS/1e12:.0f} TFLOP/s/gpu", flush=True)

    # Split CPU collate (the ~40%-of-step image decode + multicrop, GIL-releasing) from the H2D copy so
    # a background thread can overlap the next batch's collate with this batch's GPU compute (see _Prefetcher).
    _CUDA_KEYS = ("input_seq", "target_seq", "seqlens", "bigram_input_seq", "images", "pooled_idx", "loss_mask")

    def collate_cpu(examples):
        b = (collate_mix(examples, cfg.seq_len, pre, vcfg) if cfg.mix_dir
             else collate_packed(examples, cfg.seq_len, pre))
        b["bigram_input_seq"] = get_bigram_hash(b["input_seq"])  # pins memory -> needs a CPU tensor
        return b

    def to_cuda(b):
        return {k: b[k].cuda(non_blocking=True) for k in _CUDA_KEYS}

    def to_batch(examples):                       # synchronous (val path)
        return to_cuda(collate_cpu(examples))

    def next_batch(it):
        return to_batch([next(it) for _ in range(cfg.device_batch_size)])

    # Fixed, per-source val set of ready-to-batch examples (comparable + low-variance); else rolling.
    def _build_val_fixed():
        val_mix_dir = cfg.mix_dir
        if cfg.val_hf:   # pull VAL from the shared HF set (reconstruct to a mix-dir layout, then reuse it)
            if os.path.isdir(cfg.val_hf):     # already-reconstructed local dir (hf_val_to_mixdir output)
                val_mix_dir = cfg.val_hf
            else:                              # a HF repo id -> download + reconstruct (needs HF online)
                val_mix_dir = os.path.join(cfg.out_dir, "_val_hf")
                if master_process:
                    hf_val_to_mixdir(cfg.val_hf, cfg.val_hf_split, val_mix_dir)
                if is_dp:
                    dist.barrier()
        if val_mix_dir:
            rows = _load_mix_rows(val_mix_dir, "validation")
            rng = np.random.RandomState(cfg.seed + 7)
            return {s: [mix_row_to_example(rows[s][int(i)], val_mix_dir)
                        for i in rng.permutation(len(rows[s]))[:cfg.val_examples_per_source]]
                    for s in sorted(rows)}
        if cfg.data_dir and not cfg.synthetic:
            by = load_val_by_source(cfg.data_dir, cfg.val_examples_per_source, cfg.seed + 7)
            return {s: [load_example_row(cfg.data_dir, r) for r in rs] for s, rs in by.items()}
        return None
    val_fixed = _build_val_fixed()

    # Fixed held-out FineWeb text blocks -> LM loss, to watch the LLM's text ability during vision SFT.
    def _build_text_val():
        if cfg.text_val_blocks <= 0 or not master_process:   # only rank 0 ever scores text-val
            return []
        import glob
        from pathlib import Path
        from train_gpt import _load_data_shard
        files = sorted(glob.glob(os.path.join(os.environ.get("DATA_PATH", "."), "data/text/fineweb10B/fineweb_val_*.bin")))
        if not files:
            return []
        toks = _load_data_shard(Path(files[0])).to(torch.int64)   # uint16 -> int64 token ids
        L = cfg.seq_len
        blocks = [toks[i * L: i * L + L + 1] for i in range(cfg.text_val_blocks)]  # +1 for the target shift
        return [b for b in blocks if b.numel() == L + 1]
    text_blocks = _build_text_val()

    def text_val():
        """Mean next-token CE (nats/tok) of the GPT backbone on the fixed FineWeb blocks (text-only)."""
        if not text_blocks:
            return None
        vgpt.gpt.embed.clear_pending()
        losses = []
        with torch.no_grad():
            for blk in text_blocks:
                inp, tgt = blk[:-1], blk[1:]
                n = inp.numel()
                pad = (-n) % 16
                if pad:  # FA3 varlen needs total length % 16; pad tokens form their own doc, excluded below
                    inp = torch.cat([inp, torch.full((pad,), EOT_ID, dtype=inp.dtype)])
                    tgt = torch.cat([tgt, torch.zeros(pad, dtype=tgt.dtype)])
                seqlens = [n] + ([pad] if pad else [])
                cu = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32).cuda()
                inp32 = inp.to(torch.int32)
                lpt = vgpt.gpt(inp32.cuda(), tgt.cuda(), cu, get_bigram_hash(inp32).cuda(), sched)
                losses.append(lpt[:n].mean().item())
        return sum(losses) / len(losses)

    def validate():
        """Returns (macro_avg_loss, {source: loss}). per_src is {} on the rolling (hf/synthetic) path."""
        vgpt.eval()
        per_src = {}
        with torch.no_grad():
            if val_fixed is not None:
                for src, exs_all in val_fixed.items():
                    losses, dbs = [], cfg.device_batch_size
                    for i in range(0, len(exs_all), dbs):
                        losses.append(vgpt(to_batch(exs_all[i:i + dbs]), sched).item())
                    per_src[src] = sum(losses) / max(len(losses), 1)
                macro = sum(per_src.values()) / max(len(per_src), 1)
            else:
                total = sum(vgpt(next_batch(val_it), sched).item() for _ in range(cfg.val_batches))
                macro = total / max(cfg.val_batches, 1)
        vgpt.train()
        return macro, per_src

    def report_val(step):
        macro, per_src = validate()
        tv = text_val()
        line = f"step:{step}/{cfg.max_steps} val_loss:{macro:.4f}"
        if tv is not None:
            line += f" text_val:{tv:.4f}"
        if per_src:
            line += "  [" + " ".join(f"{s}:{l:.3f}" for s, l in per_src.items()) + "]"
        print(line, flush=True)
        return macro, per_src, tv

    mf = open(cfg.metrics_out, "w") if (cfg.metrics_out and master_process) else None
    def log_metric(rec):
        if mf is not None:
            mf.write(json.dumps(rec) + "\n"); mf.flush()

    diag = bool(os.environ.get("VISION_DIAG"))   # per-phase timing (data vs fwd+bwd vs opt) for MFU tuning
    # Per-rank base seed so each DP rank draws a different shuffle (real data parallelism, not redundant work).
    _base_seed = cfg.seed + 1 + rank * 1000
    n_data_workers = 0 if (os.environ.get("NO_PREFETCH") or cfg.synthetic) else int(os.environ.get("DATA_WORKERS", "4"))
    if n_data_workers > 0:
        prefetch, train_it = _Prefetcher(lambda w: _train_iter(cfg, _base_seed + w * 97),
                                         cfg.device_batch_size, collate_cpu, n_data_workers), None
        if master_process:
            print(f"[data] prefetch with {n_data_workers} worker threads", flush=True)
    else:
        prefetch, train_it = None, _train_iter(cfg, _base_seed)
    dp_params = [p for p in vgpt.parameters() if p.requires_grad]  # fixed order for the DP grad all-reduce
    vgpt.train()
    t0 = time.perf_counter()
    tokens_cum = 0
    mfu_hist, tps_hist = [], []
    step = 0
    while True:
        elapsed = time.perf_counter() - t0
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * lr_scale(step, elapsed / 60.0, cfg)

        time_up = cfg.max_minutes > 0 and (elapsed / 60.0) >= cfg.max_minutes
        if is_dp and cfg.max_minutes > 0:  # agree on the stop so no rank skips a collective the others hit
            flag = torch.tensor([1 if time_up else 0], device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            time_up = bool(flag.item())
        at_max = step >= cfg.max_steps

        if (step % cfg.val_every == 0) or at_max or time_up:
            if master_process:
                macro, per_src, tv = report_val(step)
                log_metric({"event": "val", "step": step, "t_elapsed": elapsed,
                            "val_macro": macro, "val_per_source": per_src, "text_val": tv})
            if is_dp:
                dist.barrier()
        if at_max or time_up:
            break

        # ---- one train step (timed for throughput / MFU) ----
        tstep = time.perf_counter()
        batch = to_cuda(prefetch.next()) if prefetch is not None else next_batch(train_it)
        T_gpt = int(batch["input_seq"].numel())                 # packed tokens through the GPT (this rank)
        n_pool = int((batch["input_seq"] == IM_PATCH_ID).sum()) # pooled <im_patch> tokens
        T_vit = POOL * n_pool                                    # ViT sees ~POOLx that many patch tokens
        if diag:
            torch.cuda.synchronize(); t_data = time.perf_counter()
        loss = vgpt(batch, sched)
        loss.backward()
        if is_dp:  # average grads across ranks. Iterate a FIXED trainable-param list and fill in zeros for
                   # any None grad, so every rank issues the SAME all_reduce sequence: a mix micro-batch can
                   # be all-text on one rank (connector unused -> grad None) while another has images, which
                   # would otherwise desync the collectives and hang (NCCL timeout).
            for p in dp_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if diag:
            torch.cuda.synchronize(); t_bwd = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(vgpt.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if diag and master_process and step % 10 == 0:
            print(f"[diag] step:{step} data={ (t_data-tstep)*1e3:.0f}ms fwd+bwd={(t_bwd-t_data)*1e3:.0f}ms "
                  f"opt={(time.perf_counter()-t_bwd)*1e3:.0f}ms total={(time.perf_counter()-tstep)*1e3:.0f}ms "
                  f"T_gpt={T_gpt} T_vit={T_vit}", flush=True)
        step_time = time.perf_counter() - tstep

        tokens_cum += T_gpt * world_size
        mfu = (6.0 * N_gpt * T_gpt + VIT_COEF * N_vit * T_vit) / step_time / PEAK_FLOPS  # per GPU
        tps = T_gpt * world_size / step_time
        if step >= 5:                                           # drop compile/warmup steps from the summary
            mfu_hist.append(mfu); tps_hist.append(tps)
        if step % 20 == 0:                                      # log cadence: all ranks reduce, rank 0 writes
            gloss = loss.detach()
            if is_dp:
                dist.all_reduce(gloss, op=dist.ReduceOp.AVG)
            if master_process:
                gl = gloss.item()
                print(f"step:{step}/{cfg.max_steps} train_loss:{gl:.4f} "
                      f"mfu:{mfu*100:.1f}% tok/s:{tps:,.0f} step:{step_time*1e3:.0f}ms", flush=True)
                log_metric({"event": "train", "step": step, "t_elapsed": elapsed,
                            "lr": opt.param_groups[0]["lr"], "train_loss": gl,
                            "step_time_s": step_time, "tokens_per_s": tps,
                            "tokens_cum": tokens_cum, "mfu": mfu})
        step += 1

    if master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save({"step": step, "model": vgpt.state_dict()},
                   os.path.join(cfg.out_dir, "vision_sft_final.pt"))
        if mfu_hist:
            print(f"[mfu] trained_steps={step} median_mfu={statistics.median(mfu_hist)*100:.1f}% "
                  f"mean_mfu={statistics.mean(mfu_hist)*100:.1f}% median_tok/s={statistics.median(tps_hist):,.0f} "
                  f"tokens_total={tokens_cum:,} elapsed_min={(time.perf_counter()-t0)/60.0:.1f}", flush=True)
            log_metric({"event": "summary", "trained_steps": step,
                        "median_mfu": statistics.median(mfu_hist), "mean_mfu": statistics.mean(mfu_hist),
                        "median_tokens_per_s": statistics.median(tps_hist), "tokens_total": tokens_cum,
                        "elapsed_min": (time.perf_counter() - t0) / 60.0})
        if mf is not None:
            mf.close()
    if is_dp:
        dist.barrier()


if __name__ == "__main__":
    main()
