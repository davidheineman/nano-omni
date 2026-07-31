#!/usr/bin/env python
"""Canonical sub-pool taxonomy for the Dolma data-mixing swarm.

A "sub-pool" is one `{source, quality-bucket}` slice that we tokenize to ~1B GPT-2
tokens and expose as a single *mixing dimension* to the swarm. This module is the one
place that enumerates every pool and records how to read its raw text.

Granularity (per the study design):
  - common_crawl          : one pool per (topic, vigintile)   [dolma3_pool]   ~387
  - olmocr_science_pdfs   : one pool per topic                 [dolma3_pool]   ~25
  - dolmino               : one pool per distinctive named set [dolma3_dolmino_pool] ~24
  - stack-edu             : one pool per language              [HuggingFaceTB/stack-edu] 15
  - finemath              : one pool per config                [HuggingFaceTB/finemath]  4
  - slimpajama            : one pool per RedPajama source       [DKYoon/SlimPajama-6B]   7

NOTE: the task's external sources RedPajama-Data-1T (data.together.xyz) and dolma v1_7
(olmo-data.org) publish their raw shards on off-HF hosts that are unreachable from this
cluster (curl -> 000). We substitute the HF-native, deduplicated **SlimPajama-6B** (the
same 7 RedPajama sources: CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange),
split by `meta.redpajama_set_name`. The `url_jsonl` reader is retained in build_pools.py
in case those hosts return, but no pool uses it by default.

Every pool dict has:
  name        unique id (dir name + mixture key)
  group       source group (cc|olmocr|dolmino|stack_edu|finemath|redpajama|dolma)
  reader      how build_pools.py reads it: hf_zst | hf_parquet | url_jsonl
  repo        HF dataset repo (for hf_* readers)
  patterns    list of path globs within `repo` (hf_* readers)
  url_repo    HF repo holding the URL list (url_jsonl)
  url_file    repo-relative .txt listing shard URLs (url_jsonl)
  url_filter  substring URLs must contain to belong to this pool (url_jsonl; else None)
  text_field  json/parquet column with the document text (always "text" here)
  topic       CC/olmocr topic (metadata) or None
  vigintile   CC quality bucket int (2..19, higher=better) or None

`all_pools()` returns the ordered list; `python pools.py` writes `pools.json` next to
this file and prints a summary. The dolma3_pool / dolmino_pool dirs are enumerated live
from the HF API (needs internet -> run on the host, not a SLURM node); the small
external taxonomies are hardcoded. Results are cached to pools.json so downstream code
(build_pools.py, swarm_config.py) never needs the network to know the pool set.
"""
import argparse
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_JSON = os.path.join(HERE, "pools.json")

DOLMA3_POOL = "allenai/dolma3_pool"
DOLMINO_POOL = "allenai/dolma3_dolmino_pool"

# --- external taxonomies (hardcoded; small & stable) -------------------------------
STACK_EDU_LANGS = [
    "C", "CSharp", "Cpp", "Go", "Java", "JavaScript", "Markdown", "PHP",
    "Python", "Ruby", "Rust", "SQL", "Shell", "Swift", "TypeScript",
]
FINEMATH_CONFIGS = ["finemath-3plus", "finemath-4plus", "infiwebmath-3plus", "infiwebmath-4plus"]
# SlimPajama-6B (HF-native) stands in for RedPajama-1T + dolma (off-HF hosts unreachable).
# name suffix -> the meta.redpajama_set_name value to filter on.
SLIMPAJAMA_SOURCES = {
    "commoncrawl": "RedPajamaCommonCrawl", "c4": "RedPajamaC4", "github": "RedPajamaGithub",
    "book": "RedPajamaBook", "arxiv": "RedPajamaArXiv", "wikipedia": "RedPajamaWikipedia",
    "stackexchange": "RedPajamaStackExchange",
}

# dolmino named sets to KEEP (drop the ones that duplicate other pools:
# common_crawl-high-quality_*, olmocr_science_pdfs-high_quality-*, stack_edu_fim-*).
DOLMINO_KEEP = [
    "code-meta-reasoning", "cranecode", "cranemath", "dolmino-math", "dolmino_1-flan",
    "gemini-reasoning-traces", "general_reasoning_mix", "llama_nemotron-reasoning-traces",
    "math-meta-reasoning", "megamatt", "nemotron-synth-qa", "omr-rewrite-fullthoughts",
    "openthoughts2-reasoning-traces", "program_verifiable", "qwq-reasoning-traces",
    "r1-reasoning-traces", "reddit_to_flashcards", "stem-heavy-crawl",
    "tinymath-mind", "tinymath-pot", "tulu-3-sft", "verifiable-gpt41",
    "verifiable-o4mini", "wiki_to_rcqa",  # wiki_to_rcqa-part{1,2,3} collapsed via glob
]


def _list_dolma3_dirs(repo, prefix):
    """Return the set of top-level `data/<dir>` names in `repo` starting with `prefix`."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    dirs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "data" and parts[1].startswith(prefix):
            dirs.add(parts[1])
    return dirs


def _cc_pools(cc_dirs):
    pools = []
    pat = re.compile(r"^common_crawl-(?P<topic>.+)-(?P<vig>\d{4})$")
    for d in sorted(cc_dirs):
        m = pat.match(d)
        if not m:
            continue
        topic, vig = m.group("topic"), int(m.group("vig"))
        pools.append(dict(
            name=f"cc-{topic}-v{vig:02d}", group="cc", reader="hf_zst",
            repo=DOLMA3_POOL, patterns=[f"data/{d}/*.jsonl.zst"],
            url_repo=None, url_file=None, url_filter=None,
            text_field="text", topic=topic, vigintile=vig,
        ))
    return pools


def _olmocr_pools(cc_dirs):
    # collapse the `-part1/-part2` variants of a topic into one pool via glob.
    topics = set()
    for d in cc_dirs:
        rest = d[len("olmocr_science_pdfs-"):]
        rest = re.sub(r"-part\d+$", "", rest)
        topics.add(rest)
    pools = []
    for t in sorted(topics):
        pools.append(dict(
            name=f"olmocr-{t}", group="olmocr", reader="hf_zst", repo=DOLMA3_POOL,
            patterns=[f"data/olmocr_science_pdfs-{t}/*.jsonl.zst",
                      f"data/olmocr_science_pdfs-{t}-part*/*.jsonl.zst"],
            url_repo=None, url_file=None, url_filter=None,
            text_field="text", topic=t, vigintile=None,
        ))
    return pools


def _dolmino_pools():
    pools = []
    for cfg in DOLMINO_KEEP:
        # collapse -partN with a glob prefix.
        pools.append(dict(
            # fnmatch '*' spans '/', so a single '*' covers files directly under the dir
            # AND any nested subdirs (fnmatch has no real recursive '**').
            name=f"dolmino-{cfg}", group="dolmino", reader="hf_zst", repo=DOLMINO_POOL,
            patterns=[f"data/{cfg}/*.jsonl.zst", f"data/{cfg}-part*/*.jsonl.zst"],
            url_repo=None, url_file=None, url_filter=None,
            text_field="text", topic=None, vigintile=None,
        ))
    return pools


def _external_pools():
    pools = []
    for lang in STACK_EDU_LANGS:
        pools.append(dict(
            name=f"stackedu-{lang}", group="stack_edu", reader="hf_parquet",
            repo="HuggingFaceTB/stack-edu", patterns=[f"{lang}/*.parquet"],
            url_repo=None, url_file=None, url_filter=None,
            text_field="text", topic=None, vigintile=None,
        ))
    for cfg in FINEMATH_CONFIGS:
        pools.append(dict(
            name=f"finemath-{cfg}", group="finemath", reader="hf_parquet",
            repo="HuggingFaceTB/finemath", patterns=[f"{cfg}/*.parquet"],
            url_repo=None, url_file=None, url_filter=None,
            text_field="text", topic=None, vigintile=None,
        ))
    for suffix, setname in SLIMPAJAMA_SOURCES.items():
        pools.append(dict(
            name=f"slimpajama-{suffix}", group="slimpajama", reader="hf_parquet",
            repo="DKYoon/SlimPajama-6B", patterns=["data/train-*.parquet"],
            url_repo=None, url_file=None, url_filter=None,
            filter_field="meta.redpajama_set_name", filter_value=setname,
            text_field="text", topic=None, vigintile=None,
        ))
    return pools


def build_pools():
    """Enumerate every pool (hits the HF API for the dolma3 repos)."""
    cc_dirs = _list_dolma3_dirs(DOLMA3_POOL, "common_crawl-")
    ol_dirs = _list_dolma3_dirs(DOLMA3_POOL, "olmocr_science_pdfs-")
    pools = []
    pools += _cc_pools(cc_dirs)
    pools += _olmocr_pools(ol_dirs)
    pools += _dolmino_pools()
    pools += _external_pools()
    # stable order & unique names
    seen = set()
    for p in pools:
        assert p["name"] not in seen, f"duplicate pool {p['name']}"
        seen.add(p["name"])
    return pools


def all_pools(refresh=False):
    """Load pools.json (cached); rebuild from HF if missing or refresh=True."""
    if not refresh and os.path.exists(POOLS_JSON):
        with open(POOLS_JSON) as f:
            return json.load(f)
    pools = build_pools()
    with open(POOLS_JSON, "w") as f:
        json.dump(pools, f, indent=1)
    return pools


def summarize(pools):
    by = defaultdict(int)
    for p in pools:
        by[p["group"]] += 1
    print(f"{len(pools)} pools:")
    for g, n in sorted(by.items()):
        print(f"  {g:12} {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="rebuild from HF (needs internet)")
    args = ap.parse_args()
    pools = all_pools(refresh=args.refresh)
    summarize(pools)
    print(f"\nwrote {POOLS_JSON}")
