#!/usr/bin/env python3
"""Portable per-source held-out loss eval for models trained by vision/train_vision.py.

Loads a VisionGPT checkpoint (the `{"step","model"}` dict saved by train_vision.py) and scores it against
the shared vision validation set `davidheineman/vision-ppl` (built by data/vision/molmo2_sft_build_validation.py;
HF config = source, HF split = "full" | "simple"), reporting the SAME per-source + macro cross-entropy the
trainer prints in-loop. This is the standalone counterpart to train_vision's `--val_hf` validation: it reuses
train_vision's own model + data code so the numbers are directly comparable.

Everything heavy is imported from vision/train_vision.py (build_backbone, VisionGPT, the multicrop
preprocessor, the mix-dir/val_hf loaders, collate_mix). Because build_backbone imports train_gpt, whose FA3
attention kernel only initializes when RANK is in the env, this MUST run under torchrun on a CUDA + Hopper
node with FA3 (same requirement as training). One GPU is plenty:

    .venv/bin/torchrun --standalone --nproc_per_node=1 evals/vision.py \
        results/vision/checkpoints_valhf/vision_sft_final.pt \
        --val_hf davidheineman/vision-ppl --val_hf_split full --metrics_out results/vision/eval_valhf.jsonl

See vision/run_holdout_loss.sbatch for the launcher. Render the metrics JSONL with vision/plot_training_curves.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# This file lives at modded-nanogpt/evals/. Put the repo root (train_gpt.py) and vision/ (train_vision.py)
# on sys.path so we can import train_vision's reusable model+data helpers. build_backbone itself re-derives
# the repo root from train_vision.py's own location, so importing it from here is safe.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VISION = os.path.join(_ROOT, "vision")
for _p in (_ROOT, _VISION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import train_vision as tv   # noqa: E402  (module-level helpers; CPU-safe to import, no train_gpt yet)

# The 1-D packed tensors VisionGPT.forward consumes (mirrors train_vision's _CUDA_KEYS, line 1029).
_CUDA_KEYS = ("input_seq", "target_seq", "seqlens", "bigram_input_seq", "images", "pooled_idx", "loss_mask")


def _fit_state_dict(model, sd):
    """Dim-0 slice the checkpoint tensors to the model's shapes (mirrors build_backbone, train_vision.py:497-509).

    qk_bank/vo_bank are padded to a multiple of world_size at save time; an 8-GPU checkpoint (64/24) loaded
    into a 1-GPU eval model (true 60/20) needs the leading dim sliced. The forward only reads the first
    num_qk_groups / num_attn_layers*2 rows, so the slice is exact.
    """
    msd = model.state_dict()
    out = {}
    for k, v in sd.items():
        mv = msd.get(k)
        if mv is None or v.shape == mv.shape:
            out[k] = v
        elif v.dim() == mv.dim() and v.shape[1:] == mv.shape[1:] and v.shape[0] >= mv.shape[0]:
            out[k] = v[: mv.shape[0]]
            print(f"[ckpt] sliced {k}: {tuple(v.shape)} -> {tuple(mv.shape)}", flush=True)
        else:
            print(f"[ckpt] skip shape-mismatch {k}: ckpt {tuple(v.shape)} vs model {tuple(mv.shape)}", flush=True)
    return out


def build_val_fixed(val_hf, val_hf_split, out_dir, per_source, seed, master_process, is_dp):
    """Reconstruct the shared val set into a mix-dir layout and return {source: [ready-to-collate example]}.

    `val_hf` is either a local mix-dir (used as-is) or an HF repo id (downloaded + reconstructed via
    train_vision.hf_val_to_mixdir -- the exact code the trainer's --val_hf path uses). Sampling is the fixed,
    deterministic per-source subset train_vision.py:1058-1063 builds, so points are comparable run-to-run.
    """
    if os.path.isdir(val_hf):
        val_mix_dir = val_hf
    else:
        val_mix_dir = os.path.join(out_dir, "_val_hf")
        if master_process:
            tv.hf_val_to_mixdir(val_hf, val_hf_split, val_mix_dir)
        if is_dp:
            import torch.distributed as dist
            dist.barrier()
    rows = tv._load_mix_rows(val_mix_dir, "validation")
    rng = np.random.RandomState(seed + 7)
    return {s: [tv.mix_row_to_example(rows[s][int(i)], val_mix_dir)
                for i in rng.permutation(len(rows[s]))[:per_source]]
            for s in sorted(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="VisionGPT checkpoint from train_vision.py (the {'step','model'} .pt)")
    ap.add_argument("--val_hf", default="davidheineman/vision-ppl",
                    help="HF repo id OR a local reconstructed mix-dir to validate against")
    ap.add_argument("--val_hf_split", default="full", choices=["full", "simple"])
    ap.add_argument("--val_examples_per_source", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device_batch_size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default="results/vision/_eval", help="scratch dir for the reconstructed val set")
    ap.add_argument("--metrics_out", default="", help="optional JSONL: one {'event':'val',...} record")
    args = ap.parse_args()

    vcfg = tv.VisionConfig()
    torch.manual_seed(args.seed)

    # Build VisionGPT with an EMPTY backbone (random GPT) -- the SFT checkpoint's "model" dict already carries
    # trained gpt.*/vit.*/connector.*. build_backbone imports train_gpt (FA3 init under torchrun) + wraps the
    # image-injecting embed, so the state_dict keys line up for a strict load.
    vgpt, _, world_size = tv.build_backbone("", vcfg)
    from train_gpt import rank, master_process, ForwardScheduleConfig, get_bigram_hash
    is_dp = world_size > 1

    # Mirror train_vision main()'s dtype setup so the loaded params land in the exact precision training used:
    # cast the ViT to bf16 BEFORE loading (else bf16 weights up-cast into fp32 params), then load strict.
    vgpt.vit.to(torch.bfloat16)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    sd = _fit_state_dict(vgpt, sd)   # slice 8-GPU-padded banks down to this run's world_size
    missing, unexpected = vgpt.load_state_dict(sd, strict=False)
    if master_process:
        print(f"[ckpt] {args.checkpoint}: loaded {len(sd)} tensors; "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    vgpt.freeze_vit = True   # eval: never backprop through the ViT
    vgpt.eval()

    sched = ForwardScheduleConfig(mtp_weights=None, ws_short=args.seq_len,
                                  ws_long=args.seq_len, train_max_seq_len=args.seq_len)
    pre = tv.MulticropPreprocessor(vcfg)

    def to_batch(examples):
        """CPU collate (train_vision.collate_mix) + bigram hash, then H2D copy -- mirrors collate_cpu/to_cuda."""
        b = tv.collate_mix(examples, args.seq_len, pre, vcfg)
        b["bigram_input_seq"] = get_bigram_hash(b["input_seq"])   # CPU tensor (pins memory)
        return {k: b[k].cuda(non_blocking=True) for k in _CUDA_KEYS}

    val_fixed = build_val_fixed(args.val_hf, args.val_hf_split, args.out_dir,
                                args.val_examples_per_source, args.seed, master_process, is_dp)
    if master_process:
        print(f"[val] {args.val_hf} [{args.val_hf_split}]: {len(val_fixed)} sources, "
              f"{sum(len(v) for v in val_fixed.values())} examples", flush=True)

    # Per-source weighted-mean CE (VisionGPT.forward reduces the float loss_mask as a weighted mean -> exactly
    # the per-source loss the trainer reports). Macro = unweighted mean across sources.
    t0 = time.perf_counter()
    per_src, dbs = {}, args.device_batch_size
    with torch.no_grad():
        for src, exs in val_fixed.items():
            losses = [vgpt(to_batch(exs[i:i + dbs]), sched).item() for i in range(0, len(exs), dbs)]
            per_src[src] = sum(losses) / max(len(losses), 1)
    macro = sum(per_src.values()) / max(len(per_src), 1)

    if master_process:
        print("=" * 64, flush=True)
        for s in sorted(per_src):
            print(f"  {s:48s} {per_src[s]:.4f}", flush=True)
        print("-" * 64, flush=True)
        print(f"  {'MACRO (per-source mean)':48s} {macro:.4f}   "
              f"({len(per_src)} sources, {time.perf_counter()-t0:.1f}s)", flush=True)
        print("=" * 64, flush=True)
        if args.metrics_out:
            os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
            with open(args.metrics_out, "w") as f:
                f.write(json.dumps({"event": "val", "step": 0, "val_macro": macro,
                                    "val_per_source": per_src, "text_val": None}) + "\n")
            print(f"wrote {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # skip any lingering dataloader/FA3 finalizer hang
