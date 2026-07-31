#!/usr/bin/env python3
"""Materialize the (feasible) Molmo 2 SFT mixture into a trainer-consumable local dir.

This runs in the **molmo2 / olmo env** (host conda where `import olmo` works), NOT the modded-nanogpt
.venv. It reuses Molmo 2's OWN pipeline for fidelity:
  * `get_dataset_by_name(name, split)`      (olmo/data/get_dataset.py)  -> the raw example dicts
  * `DataFormatter(...)`                     (olmo/preprocessing/data_formatter.py) with the SFT settings
    (`prompt_templates="uber_model_v2"`, `pointing_format="html-v2"`, `system_prompt="demo_or_style_v2"`)
    -> finalized conversation **text** (points already rendered as html-v2 `<points ...>` XML).
We keep `message_format="none"` so the turns are clean content (no Qwen3 `<|im_start|>` markup) -- the
modded-nanogpt trainer re-tokenizes with GPT-2 and applies its own chat template + multicrop.

Output (mirrors what vision/train_vision.py consumes):
  <out>/
    images/<source>/<i>.jpg           # every image saved locally (single or multi-image lists)
    <source>__{train,validation}.jsonl  # one row per example:
        {"source","style","image": rel | [rel,...],
         "convos": [[u,a,u,a,...], ...],   # >1 conversation == message_list subsegments (shared image)
         "n_subsegments": int, "message_weight": float}
    mixture.json                      # {loss_token_weighting, groups:{g:{weight, datasets:[{name,source,size,message_weight}]}}}

The trainer reproduces Molmo 2's sampler from mixture.json (per-dataset size=sqrt(len), normalize within
group x group weight, global normalize) and the `root_subsegments_root_tokens` loss weighting.

Run (host conda, HF reachable) -- use the launcher, which sets env + skips PixMo + is resumable:
    bash data/vision/run_full_materialize.sh          # re-run to resume; --resume skips finished datasets
Direct:  MOLMO_DATA_DIR=/storage/home/dhei/molmo_data python data/vision/molmo2_sft_full.py --resume
Smoke:  ... --only chart_qa_weighted cosyn_point --max_train 20 --max_val 8 --smoke
--rebuild_mixture_only re(builds) mixture.json from whatever <source>__train.jsonl are already on disk.
"""
import argparse
import json
import os
import sys

import numpy as np

# This script lives at speedrun/modded-nanogpt/data/vision/; put the molmo2 reference repo
# (a sibling of modded-nanogpt at speedrun/molmo2) on the path so `import olmo` works without
# needing PYTHONPATH set. Still requires an env that carries olmo's deps (host conda).
_MOLMO2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "molmo2"))
if _MOLMO2 not in sys.path:
    sys.path.insert(0, _MOLMO2)

# The feasible image+text+pointing slice of get_training_mixture("molmo2") (launch_scripts/sft.py),
# grouped with the same group weights. (name, cap_or_None, message_weight). Manual-RRC (doc/info/st_qa)
# and video/tracking groups are omitted; see data/vision/DATA_SOURCES.md.
MIXTURE = {
    "image_academic": {"weight": 0.25, "datasets": [
        ("coco_2014_vqa_multi", None, 1.0), ("text_vqa", None, 1.0), ("okvqa", None, 1.0),
        ("chart_qa_weighted", None, 1.0),
        # ai2_diagram_v2_mix_transparent DROPPED: upstream ai2-website.s3.amazonaws.com/data/ai2d-all.zip
        # is dead (404). No auto mirror; would need a manual AI2D pull (see DATA_SOURCES.md).
        ("a_okvqa_mc", None, 1.0), ("a_okvqa_da", None, 1.0), ("science_qa_img", None, 1.0),
        ("tabwmp_da", None, 1.0), ("tally_qa", None, 1.0),
        ("mantis_instruct_llava_665k_multi_multi_only", None, 1.0),
        ("mantis_instruct_nlvr2_multi_only", None, 1.0),
        ("mantis_instruct_spot-the-diff_multi_only", None, 1.0),
        # dv_qa / figure_qa / plot_qa DROPPED: their olmo builders fetch from dead/quota'd external
        # hosts (Google Drive `drive.usercontent.google.com` for dv/plot, `download.microsoft.com`
        # for figure) -> FileNotFoundError. Synthetic-chart QA is already covered by CoSyn + ChartQA.
        ("cosyn_chart_exp", None, 1.0), ("cosyn_chemical_exp", None, 1.0), ("cosyn_diagram_exp", None, 1.0),
        ("cosyn_document", None, 1.0), ("cosyn_math_exp", None, 1.0), ("cosyn_music_exp", None, 1.0),
        ("cosyn_table_exp", None, 1.0),
        ("cosyn_multidoc_chart_exp", None, 1.0), ("cosyn_multidoc_chemical_exp", None, 1.0),
        ("cosyn_multidoc_diagram_exp", None, 1.0), ("cosyn_multidoc_doc_exp", None, 1.0),
        ("cosyn_multidoc_music_exp", None, 1.0), ("cosyn_multidoc_table_exp", None, 1.0),
    ]},
    "demo": {"weight": 0.15, "datasets": [          # PixMo: web-scraped image URLs, best-effort (rot)
        ("pixmo_ask_model_anything", None, 1.0), ("pixmo_cap", 100000, 0.1),
        ("pixmo_cap_qa_as_user_qa", None, 1.0), ("pixmo_multi_image_qa_multi_only_max5", None, 1.0),
    ]},
    "image_pointing": {"weight": 0.1, "datasets": [  # point_weight=0.2 (sft.py)
        ("pixmo_multi_points", None, 0.2), ("pixmo_points_train", None, 0.2),
        ("pixmo_count_train", None, 0.2), ("pixmo_points_high_freq_train", None, 0.2),
        ("cosyn_point", None, 0.2),
    ]},
    "nlp": {"weight": 0.1, "datasets": [("tulu4", None, 1.0)]},   # text-only
    "hardcodes": {"weight": 0.0005, "datasets": [("molmo2_hardcodes", None, 1.0)]},
}


def build_formatter():
    from olmo.preprocessing.data_formatter import DataFormatter
    # Mirror launch_scripts/sft.py get_model(), minus the tokenizer-specific chat markup.
    return DataFormatter(
        prompt_templates="uber_model_v2",
        message_format="none",            # clean turns; trainer applies its own GPT-2 template
        system_prompt="demo_or_style_v2",
        pointing_format="html-v2",
        p_multi_point_all_image=0.5,
        p_choice_content_in_mc=1.0,
    )


def _convos_from_messages(messages):
    """DataFormatter output -> list of conversations, each a list of alternating turn strings."""
    if not messages:
        return []
    if isinstance(messages[0], list):     # message_list: several convos share the image (subsegments)
        return [list(m) for m in messages]
    return [list(messages)]               # single conversation


_DOWNLOADED = set()


def _ensure_downloaded(name):
    """Some datasets (science_qa_img/ai2d/tabwmp/dv/figure/plot_qa, ...) read pre-prepared local files
    and need their .download() run once first; HF-native ones (chart_qa/cosyn/mantis/tulu4) no-op or cache."""
    if name in _DOWNLOADED:
        return
    _DOWNLOADED.add(name)
    from olmo.data.get_dataset import download_dataset_by_name
    try:
        download_dataset_by_name(name, n_procs=4)
    except Exception as e:  # noqa: BLE001
        print(f"  (download step {name}: {type(e).__name__}: {str(e)[:140]})", flush=True)


def ingest_dataset(name, split, cap, msg_weight, out_dir, fmt, seed, out_split=None):
    from olmo.data.get_dataset import get_dataset_by_name
    from olmo.preprocessing.image_preprocessor import load_image
    from PIL import Image
    out_split = out_split or split      # olmo load split may differ from the file's canonical split name
    _ensure_downloaded(name)
    rng = np.random.RandomState(seed)
    ds = get_dataset_by_name(name, split)
    n = len(ds)
    order = rng.permutation(n)
    img_dir = os.path.join(out_dir, "images", name)
    os.makedirs(img_dir, exist_ok=True)
    rows, kept, img_i = [], 0, 0
    for oi in order:
        if cap and kept >= cap:
            break
        try:
            ex = dict(ds.get(int(oi), rng))
        except Exception:
            continue
        # Resolve image(s): olmo gives a path (or list) under DATA_HOME, or an ndarray for HF-embedded.
        raw = ex.get("image", None)
        is_multi = isinstance(raw, (list, tuple))
        try:
            loaded = [load_image(x) for x in raw] if is_multi else ([load_image(raw)] if raw is not None else [])
        except Exception:
            continue
        # The formatter needs the loaded image (for point h/w scaling).
        if raw is not None:
            ex["image"] = (loaded if is_multi else loaded[0])
        try:
            messages, _ = fmt(ex, True, False, rng)
        except Exception:
            continue
        convos = _convos_from_messages(messages)
        if not convos or not any(len(c) >= 2 for c in convos):
            continue
        # Save image(s) locally as jpg.
        rels = []
        try:
            for arr in loaded:
                rel = os.path.join("images", name, f"{img_i}.jpg")
                Image.fromarray(np.asarray(arr)).convert("RGB").save(os.path.join(out_dir, rel), quality=90)
                rels.append(rel); img_i += 1
        except Exception:
            continue
        image_field = (rels if is_multi else (rels[0] if rels else None))
        rows.append({"source": name, "style": ex.get("style"), "image": image_field,
                     "convos": convos, "n_subsegments": len(convos), "message_weight": msg_weight})
        kept += 1
    path = os.path.join(out_dir, f"{name}__{out_split}.jsonl")
    # Atomic write: a killed run leaves no half-written .jsonl that --resume would trust.
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print(f"  {name} [{out_split}]: {kept} examples, {img_i} images -> {os.path.basename(path)}", flush=True)
    return kept, n


def _jsonl_count(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for _ in f:
            n += 1
    return n


def _load_sizes(out_dir):
    p = os.path.join(out_dir, "_sizes.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _save_sizes(out_dir, sizes):
    p = os.path.join(out_dir, "_sizes.json")
    with open(p + ".tmp", "w") as f:
        json.dump(sizes, f)
    os.replace(p + ".tmp", p)


def build_mixture_json(out_dir, sizes):
    """(Re)build mixture.json from every non-empty <source>__train.jsonl present on disk, using the
    MIXTURE spec for group/weight/message_weight and _sizes.json (falls back to the on-disk row count)
    for the sqrt-size sampling weight. Runs at the end of every materialize, so piecewise/resumed runs
    always yield a mixture.json covering ALL sources currently materialized -- not just this run's."""
    mixture = {"loss_token_weighting": "root_subsegments_root_tokens", "groups": {}}
    for gname, g in MIXTURE.items():
        entries = []
        for name, cap, mw in g["datasets"]:
            src = name
            train_path = os.path.join(out_dir, f"{src}__train.jsonl")
            n_train = _jsonl_count(train_path)
            if n_train == 0:
                continue
            size = float(sizes.get(src, n_train))
            entries.append({"name": name, "source": src, "size": size, "message_weight": mw})
        if entries:
            mixture["groups"][gname] = {"weight": g["weight"], "datasets": entries}
    with open(os.path.join(out_dir, "mixture.json"), "w") as f:
        json.dump(mixture, f, indent=2)
    return mixture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vision/molmo2_sft_full")
    ap.add_argument("--max_train", type=int, default=6000)
    ap.add_argument("--max_val", type=int, default=500)
    ap.add_argument("--only", nargs="+", default=None, help="dataset NAMES to include (default: all feasible)")
    ap.add_argument("--skip", nargs="+", default=None, help="dataset NAMES to exclude (e.g. pixmo_*)")
    ap.add_argument("--resume", action="store_true", help="skip (source,split) whose non-empty .jsonl already exists")
    ap.add_argument("--rebuild_mixture_only", action="store_true", help="just (re)build mixture.json from disk and exit")
    ap.add_argument("--smoke", action="store_true", help="print a couple formatted rows per dataset")
    ap.add_argument("--seed", type=int, default=90218)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sizes = _load_sizes(args.out)
    if args.rebuild_mixture_only:
        m = build_mixture_json(args.out, sizes)
        print(f"Rebuilt mixture.json -> groups={ {g: len(v['datasets']) for g, v in m['groups'].items()} }")
        return

    # (canonical output split, olmo load-split candidates, cap, seed offset). Val naming is not uniform:
    # most datasets take "validation", but OkVqa/AOkVqa reject it (config is "val") and a few only ship
    # "test" -- try candidates in order, keep the first with >0 rows (val is eval-only, so test is fine).
    SPLITS = [("train", ["train"], args.max_train, 0),
              ("validation", ["validation", "val", "test"], args.max_val, 7)]
    fmt = build_formatter()
    failed = []
    for g in MIXTURE.values():
        for name, cap, mw in g["datasets"]:
            if (args.only and name not in args.only) or (args.skip and name in args.skip):
                continue
            for canonical, candidates, gcap, seed_off in SPLITS:
                out_path = os.path.join(args.out, f"{name}__{canonical}.jsonl")
                n_have = _jsonl_count(out_path)
                if args.resume and n_have > 0:
                    print(f"  = {name} [{canonical}]: resume-skip ({n_have} rows on disk)", flush=True)
                    continue
                use_cap = min(cap, gcap) if cap else gcap
                for hf_split in candidates:
                    try:
                        kept, total = ingest_dataset(name, hf_split, use_cap, mw, args.out, fmt,
                                                     args.seed + seed_off, out_split=canonical)
                        if canonical == "train":
                            sizes[name] = float(total)
                            _save_sizes(args.out, sizes)
                        if kept > 0:
                            break  # got rows; no need to try the next candidate split
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! {name} [{canonical}<-{hf_split}]: {type(e).__name__}: {str(e)[:140]}", flush=True)
                        failed.append((f"{name}:{canonical}:{hf_split}", str(e)[:140]))
    mixture = build_mixture_json(args.out, sizes)
    print(f"\nDONE -> {args.out} ; mixture.json groups={list(mixture['groups'])}")
    if failed:
        print("Failed:")
        for tag, err in failed:
            print(f"  - {tag}: {err}")

    if args.smoke:
        import glob
        for p in sorted(glob.glob(os.path.join(args.out, "*__train.jsonl")))[:6]:
            rows = [json.loads(l) for l in open(p)]
            if not rows:
                continue
            r = rows[0]
            print(f"\n### {os.path.basename(p)}  (n={len(rows)}) style={r['style']} n_subseg={r['n_subsegments']} img={r['image']}")
            for turn_i, turn in enumerate(r["convos"][0][:4]):
                print(f"    [{'user' if turn_i%2==0 else 'asst'}] {turn[:200]!r}")


if __name__ == "__main__":
    main()
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)   # skip datasets/pyarrow finalizer hang after all rows are written
