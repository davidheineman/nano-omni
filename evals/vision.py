import argparse
import hashlib
import logging
import os
import sys

import numpy as np

# This script lives at speedrun/modded-nanogpt/vision/; the molmo2 reference repo is
# a sibling of modded-nanogpt at speedrun/molmo2 (two levels up).
_MOLMO2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "molmo2"))
if _MOLMO2 not in sys.path:
    sys.path.insert(0, _MOLMO2)

log = logging.getLogger("vision_eval")

HOLDOUT_SEED = 71237
DEFAULT_VAL_FRAC = 0.02


# ── per-dataset holdout ─────────────────────────────────────────────────────
def _stable_seed(name):
    h = hashlib.sha256(name.encode()).hexdigest()
    return (HOLDOUT_SEED + int(h[:8], 16)) % (2**31)


class HoldoutView:
    """Wrap a molmo2 Dataset, exposing only a deterministic train/val index shard.

    Satisfies the Dataset interface used by the dataloader: __len__ + get(item, rng).
    The val shard is a fixed seeded fraction; training would use which="train" with
    the SAME seed/frac to keep the split disjoint. (Against an already-fully-trained
    checkpoint the shard is in-distribution, not uncontaminated — see README note.)
    """

    def __init__(self, base, name, which, val_frac):
        self.base = base
        n = len(base)
        perm = np.random.RandomState(_stable_seed(name)).permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val = set(perm[:n_val].tolist())
        want_val = which == "val"
        self.idxs = [i for i in range(n) if (i in val) == want_val]
        if not self.idxs:  # tiny dataset: don't return an empty shard
            self.idxs = list(range(n))

    def __len__(self):
        return len(self.idxs)

    def get(self, item, rng):
        return self.base.get(self.idxs[item], rng)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


_SKIPS = [0]


def install_skip_missing(max_retries=64):
    """Make the loader resilient to examples whose media isn't on disk (partial
    coverage). Monkeypatches DeterministicDataset.get to resample a different index
    on any per-example failure. On the real cluster (full corpus) this never fires;
    the skip count is logged so partial-coverage runs stay honest."""
    import olmo.data.dataset as dsmod
    orig = dsmod.DeterministicDataset.get

    def safe_get(self, idx, epoch=0):
        n = len(self)
        for k in range(min(n, max_retries)):
            try:
                return orig(self, (idx + k * 7919) % n, epoch)
            except Exception:
                _SKIPS[0] += 1
        return orig(self, idx, epoch)  # give up -> let it raise

    dsmod.DeterministicDataset.get = safe_get


def install_holdout(which="val", val_frac=DEFAULT_VAL_FRAC):
    """Monkeypatch get_dataset_by_name so every mixture dataset is a holdout shard.

    Patches the name bound in olmo.data.data_loader (imported by-value at data_loader.py:17,
    which is the binding used at dataloader-build time) and the source module.
    """
    import olmo.data.data_loader as dl
    import olmo.data.get_dataset as gd
    orig = gd.get_dataset_by_name

    def patched(dataset_name, split):
        return HoldoutView(orig(dataset_name, split), dataset_name, which, val_frac)

    dl.get_dataset_by_name = patched
    gd.get_dataset_by_name = patched
    log.info(f"holdout installed: which={which} val_frac={val_frac}")
    return orig


# ── model config (training preprocessing) ───────────────────────────────────
def training_model_cfg(checkpoint):
    """Build the SFT model config. With a checkpoint, reuse sft.py get_model()
    verbatim (exact training preprocessor). Without one (--dry-run), build a
    standalone config carrying the same preprocessing knobs so the data pipeline
    matches, minus weights."""
    if checkpoint:
        from launch_scripts.sft import get_model
        from olmo.util import select_checkpoint
        return get_model(select_checkpoint(checkpoint), "video")

    # standalone: mirror get_model()'s preprocessing settings without loading weights
    from olmo.model_configs import SIGLIP2_VISION_BACKBONE
    from olmo.models.molmo2.molmo2 import Molmo2Config
    from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
    from olmo.nn.llm import LlmConfig
    from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
    from olmo.preprocessing.data_formatter import DataFormatter
    from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
    from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
    from olmo.tokenizer import TokenizerConfig

    formatter = DataFormatter(
        prompt_templates="uber_model_v2", message_format="qwen3",
        system_prompt="demo_or_style_v2", pointing_format="html-v2",
    )
    formatter.p_multi_point_all_image = 0.5
    formatter.p_choice_content_in_mc = 1.0
    cfg = Molmo2Config(
        llm=LlmConfig(tokenizer=TokenizerConfig("Qwen/Qwen2-7B"), vocab_size=152064),
        vision_backbone=MolmoVisionBackboneConfig(vit=SIGLIP2_VISION_BACKBONE),
        data_formatter=formatter,
        mm_preprocessor=Molmo2PreprocessorConfig(
            video=VideoPreprocessorConfig(
                pooling_h=3, pooling_w=3, time_mode="per-frame-compact", max_frames=128,
                loading_method="torchcodec_exact", time_sampling=True,
                frame_sample_mode="uniform_last_frame", max_fps=[2], max_subtitle_tokens=None,
            ),
            image=MultiCropConfig(max_crops=12, max_images=5, max_multi_image_crops=8),
        ),
    )
    # match the SFT loss-token weighting (this is what makes loss_masks non-binary)
    cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"
    cfg.llm.max_sequence_length = 16384
    return cfg


# ── mixture: real weights + availability filtering ──────────────────────────
def debug_mixture():
    """Tiny CPU-validatable mixture over cached datasets, with a real message weight
    on the caption-like split so loss_masks come out non-binary."""
    from olmo.data.data_loader import KwargsMixture, WeightedDataset
    from olmo.preprocessing.text_preprocessor import MessageWeight
    cap_w = MessageWeight(weight=0.1, root_length=False, root_subsegments=False)
    return [
        KwargsMixture(0.5, [WeightedDataset("chart_qa")], "image_academic"),
        KwargsMixture(0.5, [WeightedDataset("cosyn_chart_exp", message_weight=cap_w)], "demo"),
    ]


def filtered_mixture(name, split, model_cfg):
    """Return (mixture, report). Drops datasets whose media isn't on disk (probed by
    actually preprocessing a sample example, so datasets that construct but have no
    readable media are dropped too) and renormalizes group rates, tracking dropped
    weight for honest coverage."""
    from launch_scripts.sft import get_training_mixture
    from olmo.data.dataset import DeterministicDataset
    import olmo.data.get_dataset as gd

    preproc = model_cfg.build_preprocessor(is_training=False, for_inference=False, include_image=True)

    def _usable(dataset_name):
        ds = gd.get_dataset_by_name(dataset_name, split)  # holdout-wrapped
        if len(ds) == 0:
            raise ValueError("empty shard")
        DeterministicDataset(ds, preproc, seed=0).get(0)   # must preprocess a real example
        return True

    groups = get_training_mixture(name)
    total_rate = sum(g.rate for g in groups)
    kept_groups, report = [], []
    for g in groups:
        kept_ds, dropped_ds = [], []
        for wd in g.datasets:
            try:
                _usable(wd.dataset_name)
                kept_ds.append(wd)
            except Exception as e:
                dropped_ds.append((wd.dataset_name, repr(e)[:60]))
        frac = g.rate / total_rate
        report.append(dict(group=g.name, rate=frac, kept=[w.dataset_name for w in kept_ds],
                           dropped=dropped_ds))
        if kept_ds:
            g.datasets = kept_ds
            kept_groups.append(g)

    kept_total = sum(g.rate for g in kept_groups)
    for g in kept_groups:  # renormalize surviving group rates to sum to 1
        g.rate = g.rate / kept_total
    covered = kept_total / total_rate
    return kept_groups, report, covered


def print_coverage(report, covered):
    log.info("=" * 64)
    log.info(f"MIXTURE COVERAGE: {covered*100:.1f}% of SFT objective by weight")
    for r in report:
        status = "KEPT" if r["kept"] else "DROPPED"
        log.info(f"  [{status}] {r['group']:16s} weight={r['rate']*100:5.1f}%  "
                 f"kept={len(r['kept'])} dropped={len(r['dropped'])}")
        for dn, err in r["dropped"][:4]:
            log.info(f"       - drop {dn}: {err}")
    log.info("=" * 64)


# ── build the mixture dataloader ────────────────────────────────────────────
def build_mesh(device_type):
    """World mesh with the dim names molmo2's packer/loader expect (incl. cp)."""
    from olmo.dist_util import build_world_mesh
    from olmo.train.trainer_config import ParallelismConfig
    p = ParallelismConfig()
    return build_world_mesh(
        dp=p.data_parallel_config,
        cp=p.context_parallel_config,
        tp=p.tensor_parallel_config,
        device_type=device_type,
    )


def build_loader(mixture, model_cfg, args, mesh, global_batch_size):
    from olmo.data.data_loader import DataLoaderConfig
    from olmo.data.dynamic_packer import PackingConfig
    data = DataLoaderConfig(
        kwargs_mixture=mixture,
        split="train",              # holdout monkeypatch carves the val shard out of it
        shuffle=True, drop_last=True,
        sequence_length=args.seq_len, max_text_seq_len=None,
        num_workers=args.num_workers, pad="to_max", pin_memory=False,
        prefetch_factor=(args.prefetch_factor if args.num_workers else None),
        seed=50189,
        packing=PackingConfig(buffer_size=48, image_weight=30, shortcut_max_len_images=False),
    )
    return data.build_train_dataloader(model_config=model_cfg, mesh=mesh,
                                       global_batch_size=global_batch_size)


# ── dry-run: validate weights / packing / holdout on CPU (no model) ─────────
def dry_run(mixture, model_cfg, args):
    import torch.distributed as tdist

    if not tdist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        tdist.init_process_group(backend="gloo", world_size=1, rank=0)
    mesh = build_mesh("cpu")

    loader = build_loader(mixture, model_cfg, args, mesh, global_batch_size=1)
    log.info("iterating loader (CPU, no model)…")

    saw_nonbinary = False
    max_subsegs = 0
    n_batches = 0
    for batch in loader:
        n_batches += 1
        lm = batch["loss_masks"]
        lm = lm.numpy() if hasattr(lm, "numpy") else np.asarray(lm)
        pos = lm[lm > 0]
        if pos.size and np.any(np.abs(pos - 1.0) > 1e-6):
            saw_nonbinary = True
        # packing groups multiple examples per sequence -> multiple subsegment ids
        if "subsegment_ids" in batch:
            ss = batch["subsegment_ids"]
            ss = ss.numpy() if hasattr(ss, "numpy") else np.asarray(ss)
            uniq = np.unique(ss[ss >= 0])
            max_subsegs = max(max_subsegs, len(uniq))
        if n_batches <= 3:
            log.info(f"  batch {n_batches}: loss_masks>0={int((lm>0).sum())} "
                     f"min_pos={pos.min():.4f} max_pos={pos.max():.4f} "
                     f"(non-binary weights => message-weighting active)")
        if n_batches >= args.max_batches:
            break

    log.info("-" * 64)
    log.info(f"VALIDATION: batches={n_batches}")
    log.info(f"  #3 message-weighted loss_masks are non-binary: {saw_nonbinary}")
    log.info(f"  #3 packing groups up to {max_subsegs} examples/sequence: {max_subsegs > 1}")
    log.info("  #1 mixture rates + #4 training preprocessor: applied via build_train_dataloader")
    log.info(f"  skipped {_SKIPS[0]} examples with missing media (partial-coverage runs only)")
    log.info("-" * 64)
    if tdist.is_initialized():
        tdist.destroy_process_group()
    return saw_nonbinary


# ── full run: load checkpoint, run LossDatasetEvaluator over the mixture ────
def full_run(checkpoint, mixture, model_cfg, args):
    import torch
    from olmo.checkpoint import load_model_state
    from olmo.eval.loss_evaluator import LossDatasetEvaluator, LossMetrics
    from olmo.util import select_checkpoint, resource_path
    from olmo.torch_util import get_world_size

    device = torch.device("cuda")
    ckpt_dir = select_checkpoint(checkpoint)
    from olmo.models.molmo.molmo import MolmoConfig
    loaded_cfg = MolmoConfig.load(resource_path(ckpt_dir, "config.yaml"), key="model",
                                  validate_paths=False)
    # keep the checkpoint's architecture, but use our training preprocessing knobs
    loaded_cfg.mm_preprocessor = model_cfg.mm_preprocessor
    loaded_cfg.data_formatter = model_cfg.data_formatter
    loaded_cfg.llm.max_sequence_length = args.seq_len

    with torch.device("meta"):
        model = loaded_cfg.build_model()
    model.to_empty(device=device)
    load_model_state(ckpt_dir, model)
    model.eval()

    mesh = build_mesh("cuda")
    loader = build_loader(mixture, loaded_cfg, args, mesh,
                          global_batch_size=args.device_batch_size * get_world_size())
    num_batches = max(1, args.max_examples // (args.device_batch_size * get_world_size()))
    evaluator = LossDatasetEvaluator(
        label="holdout", eval_loader=loader, evaluator=LossMetrics(device),
        num_batches=num_batches, response_logits_only=True,
    )
    metrics = evaluator.run(model, device, autocast_precision=torch.bfloat16, pbar=True)
    log.info("=" * 64)
    log.info(f"HELD-OUT SFT LOSS  CrossEntropyLoss={metrics.get('CrossEntropyLoss'):.4f}  "
             f"Accuracy={metrics.get('Accuracy'):.4f}")
    log.info("=" * 64)
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None, help="Molmo2 checkpoint dir (omit for --dry-run)")
    ap.add_argument("--mixture", default="molmo2", help='"molmo2" (real) or "debug" (cached CPU test)')
    ap.add_argument("--dry-run", action="store_true", help="validate the data pipeline, no model/GPU")
    ap.add_argument("--which", default="val", choices=["val", "train"], help="holdout shard to eval")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--device-batch-size", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=20000, help="approx examples for the full run")
    ap.add_argument("--max-batches", type=int, default=6, help="batches for --dry-run")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    args = ap.parse_args()

    install_holdout(which=args.which, val_frac=args.val_frac)
    install_skip_missing()
    model_cfg = training_model_cfg(args.checkpoint)

    if args.mixture == "debug":
        mixture = debug_mixture()
        log.info("using debug mixture (chart_qa + cosyn_chart_exp, cached)")
    else:
        mixture, report, covered = filtered_mixture(args.mixture, "train", model_cfg)
        print_coverage(report, covered)
        if not mixture:
            raise SystemExit("no mixture datasets available on disk — nothing to evaluate")

    if args.dry_run or not args.checkpoint:
        ok = dry_run(mixture, model_cfg, args)
        if not ok:
            log.warning("loss_masks looked binary — message-weighting may not be active")
    else:
        full_run(args.checkpoint, mixture, model_cfg, args)


if __name__ == "__main__":
    main()
