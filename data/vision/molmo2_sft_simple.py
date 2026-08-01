#!/usr/bin/env python3
"""Ingest the self-contained, single-image subset of the Molmo 2 SFT mixture into a local dir.

Molmo 2's full SFT mixture (see molmo2 `launch_scripts/sft.py :: get_training_mixture("molmo2")`)
spans image / video / pointing / tracking groups. Most of it needs COCO/VG image joins, web-scraped
PixMo URLs, source videos, or manual RRC registration (docvqa/infovqa/st-vqa) -- none of which this
lightweight single-image trainer can consume. This script pulls the members of Molmo 2's
`image_academic` group that ARE self-contained on HuggingFace with **bundled images** and that expose
Molmo 2's **native train + validation splits**, so train and val come from the same distribution
(the earlier port mixed in val-only datasets -> a meaningless val curve). Sources here match Molmo 2's
own (`HuggingFaceM4/ChartQA`, `derek-thomas/ScienceQA`, TextVQA, and `allenai/CoSyn-400K`, the exact
synthetic-document source Molmo 2 uses -- the 7 `cosyn_*_exp` doc types, dropping circuit/graphic/
nutrition just as Molmo 2 does).

Normalizes every example to {image, question, answer, source} and writes:

    data/vision/molmo2_sft_simple/
      images/<source>/<i>.jpg
      train.jsonl   # {"image": "images/<source>/<i>.jpg", "question": ..., "answer": ..., "source": ...}
      val.jsonl

Point the SFT trainer at it:  torchrun ... vision/train_vision.py --data_dir data/vision/molmo2_sft_simple

Run where HF is reachable (host bridge / login node), in the modded-nanogpt venv:
    .venv/bin/python data/vision/molmo2_sft_simple.py --max_train 6000 --max_val 500
"""
import argparse
import json
import os

# Self-contained, single-image members of Molmo 2's `image_academic` group.
#   (repo, config, train_split, val_split, source_tag)
# All expose bundled images + a native train/validation(or val) split (verified against HF).
DATASETS = [
    ("HuggingFaceM4/ChartQA",  None,        "train", "val",        "chartqa"),      # Molmo2: chart_qa
    ("lmms-lab/textvqa",       None,        "train", "validation", "textvqa"),      # Molmo2: text_vqa
    ("derek-thomas/ScienceQA", None,        "train", "validation", "scienceqa"),    # Molmo2: science_qa_img
    ("allenai/CoSyn-400K",     "chart",     "train", "validation", "cosyn_chart"),    # Molmo2: cosyn_chart_exp
    ("allenai/CoSyn-400K",     "chemical",  "train", "validation", "cosyn_chemical"), # Molmo2: cosyn_chemical_exp
    ("allenai/CoSyn-400K",     "diagram",   "train", "validation", "cosyn_diagram"),  # Molmo2: cosyn_diagram_exp
    ("allenai/CoSyn-400K",     "document",  "train", "validation", "cosyn_document"), # Molmo2: cosyn_document
    ("allenai/CoSyn-400K",     "math",      "train", "validation", "cosyn_math"),     # Molmo2: cosyn_math_exp
    ("allenai/CoSyn-400K",     "music",     "train", "validation", "cosyn_music"),    # Molmo2: cosyn_music_exp
    ("allenai/CoSyn-400K",     "table",     "train", "validation", "cosyn_table"),    # Molmo2: cosyn_table_exp
]

# CoSyn packs several Q/A per image; cap how many we emit per image (each reuses the one saved image).
MAX_QA_PER_IMAGE = 4

QUESTION_FIELDS = ["question", "query", "instruction", "prompt"]
ANSWER_FIELDS = ["answer", "answers", "label", "labels", "response"]
CHOICE_FIELDS = ["choices", "options"]


def _first_field(row, names):
    for n in names:
        if n in row and row[n] not in (None, "", []):
            return n
    return None


def _resolve_answer(a, choices):
    """Normalize an answer value to a plain string, resolving multiple-choice where needed."""
    if isinstance(a, list):
        a = a[0] if a else None
    if isinstance(a, int) and choices and 0 <= a < len(choices):          # index into choices
        a = choices[a]
    elif isinstance(a, str) and len(a) == 1 and a.upper().isalpha() and choices:  # "A"/"B"...
        idx = ord(a.upper()) - ord("A")
        if 0 <= idx < len(choices):
            a = choices[idx]
    return None if a is None else str(a).strip()


def _qa_from_generic(row):
    """One (question, answer) for the standard VQA/chart/science schema. Yields 0 or 1 pair."""
    qf = _first_field(row, QUESTION_FIELDS)
    q = str(row[qf]).strip() if qf else ""
    cf = _first_field(row, CHOICE_FIELDS)
    choices = row[cf] if cf else None
    if cf and isinstance(choices, list) and choices:  # make MC well-posed
        q = f"{q}\nOptions: " + "; ".join(str(o) for o in choices)
    af = _first_field(row, ANSWER_FIELDS)
    a = _resolve_answer(row[af], choices) if af else None
    if q and a:
        yield q, a


def _qa_from_cosyn(row):
    """CoSyn stores `qa_pairs` = {'question': [...], 'answer'/'answers': [...]} (or a list of dicts)."""
    qa = row.get("qa_pairs")
    pairs = []
    if isinstance(qa, dict):
        qs = qa.get("question") or qa.get("questions") or []
        ans = qa.get("answer") or qa.get("answers") or []
        pairs = list(zip(qs, ans))
    elif isinstance(qa, list):
        for d in qa:
            if isinstance(d, dict):
                pairs.append((d.get("question"), d.get("answer") or d.get("answers")))
    n = 0
    for q, a in pairs:
        if n >= MAX_QA_PER_IMAGE:
            break
        q = str(q).strip() if q else ""
        a = _resolve_answer(a, None)
        if q and a:
            n += 1
            yield q, a


def iter_qa(row, source):
    """Dispatch to the right (question, answer) extractor for this source; yields >=0 pairs."""
    if source.startswith("cosyn"):
        yield from _qa_from_cosyn(row)
    else:
        yield from _qa_from_generic(row)


def ingest_one(repo, config, split, source, cap, out_dir, writer):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    img_dir = os.path.join(out_dir, "images", source)
    os.makedirs(img_dir, exist_ok=True)
    kept_examples = kept_images = 0
    for row in ds:
        if kept_examples >= cap:
            break
        img = row.get("image")
        if img is None:            # e.g. text-only ScienceQA rows -> the image-only subset
            continue
        pairs = list(iter_qa(row, source))
        if not pairs:
            continue
        rel = os.path.join("images", source, f"{kept_images}.jpg")
        try:
            img.convert("RGB").save(os.path.join(out_dir, rel), quality=90)
        except Exception:
            continue
        kept_images += 1
        for q, a in pairs:
            if kept_examples >= cap:
                break
            writer.write(json.dumps({"image": rel, "question": q, "answer": a, "source": source}) + "\n")
            kept_examples += 1
    print(f"  {source} [{split}]: {kept_examples} examples from {kept_images} images", flush=True)
    return kept_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vision/molmo2_sft_simple")
    ap.add_argument("--max_train", type=int, default=6000, help="cap examples per dataset (train)")
    ap.add_argument("--max_val", type=int, default=500, help="cap examples per dataset (val)")
    ap.add_argument("--only", nargs="+", default=None, help="source tags to include (default: all)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
                    help="which splits to (re)build; default both. Use --splits val to refresh val.jsonl "
                         "without truncating an existing train.jsonl.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    writers = {k: open(os.path.join(args.out, f"{k}.jsonl"), "w") for k in args.splits}
    totals = {k: 0 for k in args.splits}
    failed = []
    for repo, config, train_split, val_split, source in DATASETS:
        if args.only and source not in args.only:
            continue
        tag = f"{repo}" + (f":{config}" if config else "")
        print(f"[{tag}] ({source}) ingesting...", flush=True)
        splits = [("train", train_split, args.max_train), ("val", val_split, args.max_val)]
        for split_key, hf_split, cap in [s for s in splits if s[0] in args.splits]:
            try:
                totals[split_key] += ingest_one(repo, config, hf_split, source, cap, args.out,
                                                 writers[split_key])
            except Exception as e:  # noqa: BLE001
                print(f"  ! failed {tag} [{split_key}<-{hf_split}]: {e}", flush=True)
                failed.append((f"{tag}:{split_key}", str(e)))
    for w in writers.values():
        w.close()

    print(f"\nDONE -> {args.out}")
    for k in args.splits:
        p = os.path.join(args.out, f"{k}.jsonl")
        print(f"  {k}.jsonl: {sum(1 for _ in open(p))} rows (running total {totals[k]})")
    if failed:
        print("Failed:")
        for tag, err in failed:
            print(f"  - {tag}: {err}")


if __name__ == "__main__":
    main()
    # pyarrow/numpy can hang or raise a fatal error during interpreter finalization once all rows +
    # images are already written; skip finalizers entirely (everything is flushed to disk by here).
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)
