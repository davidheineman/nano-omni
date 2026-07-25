#!/usr/bin/env python3
"""Ingest the (HuggingFace-available) Molmo 2 image-SFT datasets into a local directory.

Molmo 2's full SFT mixture is defined in the `olmo` package and much of it needs COCO/VG joins,
web-scraped PixMo image URLs, or source videos. This script pulls the practical subset that ships
as self-contained HF datasets with bundled images (chart / document / infographic / diagram /
science / text VQA), normalizes every example to {image, question, answer}, and writes:

    data/vision/molmo2_sft/
      images/<source>/<i>.jpg
      train.jsonl   # {"image": "images/<source>/<i>.jpg", "question": ..., "answer": ..., "source": ...}
      val.jsonl

Point the SFT trainer at it:  torchrun ... vision/train_vision.py --data_dir data/vision/molmo2_sft

Run where HF is reachable (a compute node / login node), e.g.:
    .venv/bin/python data/prepare_molmo2_sft.py --max_train 8000 --max_val 200
"""
import argparse
import json
import os

# (repo, config) of Molmo-2-relevant image-SFT datasets that are self-contained on HF (images bundled).
DATASETS = [
    ("HuggingFaceM4/ChartQA", None),      # charts
    ("lmms-lab/DocVQA", "DocVQA"),        # documents
    ("lmms-lab/DocVQA", "InfographicVQA"),# infographics
    ("lmms-lab/textvqa", None),           # scene text
    ("lmms-lab/ai2d", None),              # diagrams (multiple-choice)
    ("derek-thomas/ScienceQA", None),     # science (multiple-choice, image subset)
]

QUESTION_FIELDS = ["question", "query", "instruction", "prompt"]
ANSWER_FIELDS = ["answer", "answers", "label", "labels", "response", "conversations"]
CHOICE_FIELDS = ["choices", "options"]


def _find_image_col(ds):
    from datasets import Image as HFImage
    for name, feat in ds.features.items():
        if isinstance(feat, HFImage):
            return name
    return "image" if "image" in ds.features else None


def _first_field(row, names):
    for n in names:
        if n in row and row[n] not in (None, "", []):
            return n
    return None


def _to_answer(row):
    """Normalize the answer to a plain string, resolving multiple-choice where needed."""
    af = _first_field(row, ANSWER_FIELDS)
    if af is None:
        return None
    a = row[af]
    cf = _first_field(row, CHOICE_FIELDS)
    choices = row[cf] if cf else None
    if isinstance(a, list):
        a = a[0] if a else None
    if isinstance(a, int) and choices and 0 <= a < len(choices):   # index into choices
        a = choices[a]
    elif isinstance(a, str) and len(a) == 1 and a.upper().isalpha() and choices:  # "A"/"B"...
        idx = ord(a.upper()) - ord("A")
        if 0 <= idx < len(choices):
            a = choices[idx]
    return None if a is None else str(a).strip()


def _question(row):
    qf = _first_field(row, QUESTION_FIELDS)
    q = row[qf] if qf else ""
    cf = _first_field(row, CHOICE_FIELDS)
    if cf:  # append options so a multiple-choice answer is well-posed
        opts = row[cf]
        if isinstance(opts, list) and opts:
            q = f"{q}\nOptions: " + "; ".join(str(o) for o in opts)
    return str(q).strip()


def ingest_one(repo, config, splits_wanted, max_per, out_dir, writers):
    from datasets import load_dataset, get_dataset_split_names
    src = (repo.split("/")[-1] + ("_" + config if config else "")).lower()
    avail = get_dataset_split_names(repo, config)
    split_map = {}
    if "train" in avail:
        split_map["train"] = "train"
    for v in ("validation", "val", "test"):
        if v in avail:
            split_map["val"] = v
            break
    n_written = 0
    for split_key, hf_split in split_map.items():
        cap = max_per[split_key]
        ds = load_dataset(repo, config, split=hf_split, streaming=True)
        img_col = None
        img_dir = os.path.join(out_dir, "images", src)
        os.makedirs(img_dir, exist_ok=True)
        kept = 0
        for i, row in enumerate(ds):
            if kept >= cap:
                break
            if img_col is None:
                from datasets import Image as HFImage
                img_col = next((k for k, v in ds.features.items() if isinstance(v, HFImage)), None) or "image"
            img = row.get(img_col)
            q, a = _question(row), _to_answer(row)
            if img is None or not q or not a:
                continue
            try:
                rel = os.path.join("images", src, f"{kept}.jpg")
                img.convert("RGB").save(os.path.join(out_dir, rel), quality=90)
            except Exception:
                continue
            writers[split_key].write(json.dumps({"image": rel, "question": q, "answer": a, "source": src}) + "\n")
            kept += 1
        print(f"  {src} [{split_key}<-{hf_split}]: wrote {kept}", flush=True)
        n_written += kept
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vision/molmo2_sft")
    ap.add_argument("--max_train", type=int, default=8000, help="cap examples per dataset (train)")
    ap.add_argument("--max_val", type=int, default=200, help="cap examples per dataset (val)")
    ap.add_argument("--only", nargs="+", default=None, help="repo substrings to include (default: all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    max_per = {"train": args.max_train, "val": args.max_val}
    writers = {k: open(os.path.join(args.out, f"{k}.jsonl"), "w") for k in ("train", "val")}
    total, failed = 0, []
    for repo, config in DATASETS:
        if args.only and not any(o.lower() in repo.lower() for o in args.only):
            continue
        tag = f"{repo}" + (f":{config}" if config else "")
        print(f"[{tag}] ingesting...", flush=True)
        try:
            total += ingest_one(repo, config, ("train", "val"), max_per, args.out, writers)
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed {tag}: {e}", flush=True)
            failed.append((tag, str(e)))
    for w in writers.values():
        w.close()
    print(f"\nDONE: {total} examples -> {args.out}")
    for k in ("train", "val"):
        p = os.path.join(args.out, f"{k}.jsonl")
        print(f"  {k}.jsonl: {sum(1 for _ in open(p))} rows")
    if failed:
        print("Failed datasets:")
        for tag, err in failed:
            print(f"  - {tag}: {err}")


if __name__ == "__main__":
    main()
