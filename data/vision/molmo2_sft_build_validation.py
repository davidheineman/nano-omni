#!/usr/bin/env python3
"""Build a portable HF validation set from the two local SFT ingestions and push it to the Hub.

Reads the held-out **val** data already materialized by:
  * data/vision/molmo2_sft_full/   (<source>__validation.jsonl + images/)   -- the recipe-parity mix
  * data/vision/molmo2_sft_simple/ (val.jsonl + images/)                     -- the lightweight subset

and pushes one HF dataset where, per the two axes the user asked for:
  * HF **subset (config)** = each data source (chart_qa_weighted, cosyn_chart_exp, info_qa, ... for full;
    chartqa, textvqa, cosyn_chart, ... for simple)  -> `load_dataset(REPO, "<source>", split=...)`
  * HF **split**          = "full" or "simple" (which ingestion the subset came from)

Every row is normalized to ONE schema so both pipelines coexist:
  { source, split, images: Sequence(Image), convos_json, n_subsegments, message_weight, style }
where convos_json = json.dumps([[user, asst, user, asst, ...], ...])  (simple -> [[question, answer]]).
This is a superset of what vision/train_vision.py's mixture val path consumes, so the trainer can pull it
back via `--val_hf` (see hf_val_to_mixdir there).

Run on the host (HF reachable + token; see README.md), e.g.:
    python data/vision/molmo2_sft_build_validation.py                    # push to davidheineman/vision-ppl
    python data/vision/molmo2_sft_build_validation.py --rebuild          # delete the repo first, then rebuild clean
    python data/vision/molmo2_sft_build_validation.py --dry_run          # build + summarize, no push
"""
import argparse
import glob
import json
import os

REPO_DEFAULT = "davidheineman/vision-ppl"


def _image_paths(base_dir, image_field):
    """image_field is a rel path or list of rel paths under base_dir -> list of ABSOLUTE paths that exist.
    We pass PATHS (not decoded PIL) so HF's Image() feature encodes each lazily at push time -- keeping
    memory bounded to one source's shard instead of holding every val image in RAM (that OOM-killed v1)."""
    rels = image_field if isinstance(image_field, list) else ([image_field] if image_field else [])
    paths = [os.path.join(base_dir, r) for r in rels]
    return [p for p in paths if os.path.exists(p)]


def rows_from_full(full_dir):
    """Yield (source, unified_row) for every full val example. convos/images kept as-is (multi-image ok)."""
    for path in sorted(glob.glob(os.path.join(full_dir, "*__validation.jsonl"))):
        source = os.path.basename(path).split("__")[0]
        for line in open(path):
            r = json.loads(line)
            yield source, {
                "source": source, "split": "full", "images": _image_paths(full_dir, r.get("image")),
                "convos_json": json.dumps(r.get("convos") or []),
                "n_subsegments": int(r.get("n_subsegments", 1)),
                "message_weight": float(r.get("message_weight", 1.0)),
                "style": r.get("style") or source,
            }


def rows_from_simple(simple_dir):
    """Yield (source, unified_row) for every simple val example. {question,answer} -> one-turn convo."""
    val_path = os.path.join(simple_dir, "val.jsonl")
    if not os.path.exists(val_path):
        return
    for line in open(val_path):
        r = json.loads(line)
        source = r.get("source", "unknown")
        yield source, {
            "source": source, "split": "simple", "images": _image_paths(simple_dir, r.get("image")),
            "convos_json": json.dumps([[r["question"], r["answer"]]]),
            "n_subsegments": 1, "message_weight": 1.0, "style": source,
        }


def _features():
    import datasets
    return datasets.Features({
        "source": datasets.Value("string"),
        "split": datasets.Value("string"),
        "images": datasets.Sequence(datasets.Image()),
        "convos_json": datasets.Value("string"),
        "n_subsegments": datasets.Value("int32"),
        "message_weight": datasets.Value("float32"),
        "style": datasets.Value("string"),
    })


def collect(pairs, only, limit):
    """(source, row) stream -> {source: [rows]} with optional --only filter and per-source --limit."""
    by_src = {}
    for source, row in pairs:
        if only and source not in only:
            continue
        rows = by_src.setdefault(source, [])
        if limit and len(rows) >= limit:
            continue
        rows.append(row)
    return by_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--full_dir", default="data/vision/molmo2_sft_full")
    ap.add_argument("--simple_dir", default="data/vision/molmo2_sft_simple")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these source names")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per (source, split); 0 = all")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the HF repo and recreate it before pushing (clean full replace; the old "
                         "task-keyed config layout partially overlaps ours, so an in-place push leaves stale splits)")
    ap.add_argument("--dry_run", action="store_true", help="build + summarize; do not push")
    args = ap.parse_args()

    import datasets
    datasets.disable_progress_bars()
    feats = _features()

    # split name -> {source: [rows]}
    built = {
        "full": collect(rows_from_full(args.full_dir), args.only, args.limit),
        "simple": collect(rows_from_simple(args.simple_dir), args.only, args.limit),
    }

    # Summary table (source x split counts)
    all_sources = sorted(set(built["full"]) | set(built["simple"]))
    print(f"{'source':44s} {'full':>7s} {'simple':>7s}")
    for s in all_sources:
        print(f"{s:44s} {len(built['full'].get(s, [])):>7d} {len(built['simple'].get(s, [])):>7d}")
    n_configs = len(all_sources)
    n_rows = sum(len(v) for d in built.values() for v in d.values())
    print(f"\n{n_configs} subsets (configs), {n_rows} rows total across splits {list(built)}")
    if args.dry_run:
        print("dry-run: not pushing"); return

    if args.rebuild:
        # Wipe the repo so the new (config=source, split=full/simple) layout fully REPLACES whatever was
        # there (e.g. the old build_molmo2_valid.py task-keyed BPB configs, some of which share names).
        from huggingface_hub import HfApi
        api = HfApi()
        api.delete_repo(args.repo, repo_type="dataset", missing_ok=True)
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"rebuild: deleted + recreated {args.repo}", flush=True)

    # Push one (config=source, split) shard at a time. First push to a repo creates it; later configs/
    # splits append. Ordering full-before-simple per source keeps commits grouped.
    pushed = 0
    for source in all_sources:
        for split in ("full", "simple"):
            rows = built[split].get(source)
            if not rows:
                continue
            ds = datasets.Dataset.from_list(rows, features=feats)   # images are PATHS; Image() encodes at push
            ds.push_to_hub(args.repo, config_name=source, split=split, private=args.private)
            del ds  # free the shard before the next source (bounded memory)
            pushed += 1
            print(f"  pushed {source} [{split}] ({len(rows)} rows)  [{pushed}]", flush=True)
    print(f"\nDONE -> https://huggingface.co/datasets/{args.repo}  ({pushed} config/split shards)")


if __name__ == "__main__":
    main()
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)   # skip datasets/pyarrow finalizer hang after push
