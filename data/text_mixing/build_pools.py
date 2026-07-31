import argparse
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pools as pools_mod  # noqa: E402

DEFAULT_OUT = "/datasets/pretraining_data/dhei/speedrun/text_mixing/tok"
DEFAULT_CACHE = "/datasets/pretraining_data/dhei/speedrun/text_mixing/hf_cache"
SHARD_TOKENS = 100_000_000
BATCH_DOCS = 1024
EOT = 50256
MAGIC = 20240520
# per-group raw-byte fetch budget (compressed/on-disk); generous so tokenize can hit target
RAW_BYTES = {"cc": 1.8e9, "olmocr": 1.8e9, "dolmino": 2.0e9,
             "stack_edu": 2.5e9, "finemath": 2.5e9, "slimpajama": float("inf")}

# configured per-process from main()
CACHE_DIR = DEFAULT_CACHE
OFFLINE = False

_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


# ------------------------------------------------------------------ .bin writer
def write_datafile(filename, toks_np):
    assert toks_np.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(toks_np)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


# ------------------------------------------------------------------ repo file cache
def _repo_files(repo, out):
    """Cached list of all files in a HF dataset repo (glob source for hf_* readers)."""
    cdir = os.path.join(out, ".repo_files")
    os.makedirs(cdir, exist_ok=True)
    cache = os.path.join(cdir, repo.replace("/", "__") + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    tmp = cache + ".tmp"
    json.dump(files, open(tmp, "w"))
    os.replace(tmp, cache)
    return files


def _match(files, patterns):
    import fnmatch
    out = []
    for pat in patterns:
        out += [f for f in files if fnmatch.fnmatch(f, pat)]
    return sorted(set(out))


def _shard_paths(pool, out):
    return _match(_repo_files(pool["repo"], out), pool["patterns"])


def _local(repo, path):
    """Resolve a repo file to a local path via the shared HF cache.
    OFFLINE=True (tokenize on compute nodes) only reads what fetch already downloaded."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", cache_dir=CACHE_DIR,
                           local_files_only=OFFLINE)


# ------------------------------------------------------------------ raw readers
def _iter_zst_lines(fileobj):
    import zstandard as zstd
    reader = zstd.ZstdDecompressor().stream_reader(fileobj)
    yield from io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def _texts_hf_zst(pool, out):
    tf = pool["text_field"]
    for path in _shard_paths(pool, out):
        try:
            lp = _local(pool["repo"], path)
        except Exception:
            break  # past the fetched prefix (offline) — stop
        try:
            with open(lp, "rb") as fo:
                for line in _iter_zst_lines(fo):
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = d.get(tf) or d.get("text") or ""
                    if t:
                        yield t
        except Exception as e:
            print(f"  [{pool['name']}] shard err {path}: {e}", flush=True)


def _texts_hf_parquet(pool, out):
    import pyarrow.parquet as pq
    tf = pool["text_field"]
    ffield, fval = pool.get("filter_field"), pool.get("filter_value")
    ftop, fsub = (ffield.split(".", 1) if ffield and "." in ffield else (ffield, None))
    for path in _shard_paths(pool, out):
        try:
            lp = _local(pool["repo"], path)
        except Exception:
            break
        try:
            pf = pq.ParquetFile(lp)
            names = pf.schema_arrow.names
            col = tf if tf in names else ("text" if "text" in names else None)
            if col is None:
                strs = [n for n, t in zip(names, pf.schema_arrow.types) if "string" in str(t)]
                col = strs[0] if strs else None
            if col is None:
                continue
            cols = [col] + ([ftop] if (ftop and ftop in names) else [])
            for batch in pf.iter_batches(batch_size=2048, columns=cols):
                texts = batch.column(0).to_pylist()
                if ftop and ftop in names:
                    metas = batch.column(1).to_pylist()
                    for t, m in zip(texts, metas):
                        if t and (m.get(fsub) if isinstance(m, dict) and fsub else m) == fval:
                            yield t
                else:
                    for t in texts:
                        if t:
                            yield t
        except Exception as e:
            print(f"  [{pool['name']}] parquet err {path}: {e}", flush=True)


READERS = {"hf_zst": _texts_hf_zst, "hf_parquet": _texts_hf_parquet}


# ------------------------------------------------------------------ fetch phase
def _is_done(pool, out, target):
    m = os.path.join(out, pool["name"], "manifest.json")
    if os.path.exists(m):
        try:
            d = json.load(open(m))
            return d.get("done") and d.get("tokens", 0) >= min(target, d.get("target", target))
        except Exception:
            return False
    return False


def fetch_pool(pool, out, target):
    """Download raw shards for one pool into the shared cache, up to its byte budget."""
    if _is_done(pool, out, target):
        return dict(name=pool["name"], fetched_bytes=0, files=0, skipped=True)
    # enough raw for `target` tokens (~8 raw bytes/token, generous), capped by group budget
    budget = min(RAW_BYTES.get(pool["group"], 2.0e9), target * 8)
    t0 = time.time()
    total = 0
    nfiles = 0
    err = None
    try:
        for path in _shard_paths(pool, out):
            lp = _local(pool["repo"], path)  # OFFLINE must be False here
            total += os.path.getsize(lp)
            nfiles += 1
            # stop on byte budget OR file-count cap (some pools have 1000s of tiny
            # shards; capping avoids stalling on per-file API overhead — the cached
            # prefix already far exceeds the token target for such large sets).
            if total >= budget or nfiles >= 400:
                break
    except Exception:
        err = traceback.format_exc()
    return dict(name=pool["name"], group=pool["group"], fetched_bytes=int(total),
                files=nfiles, elapsed=round(time.time() - t0, 1), error=err)


# ------------------------------------------------------------------ tokenize phase
def build_one(pool, out, target):
    """Tokenize a pool's (already-fetched) raw shards into .bin, capped at target tokens."""
    name = pool["name"]
    pdir = os.path.join(out, name)
    os.makedirs(pdir, exist_ok=True)
    if _is_done(pool, out, target):
        return {**json.load(open(os.path.join(pdir, "manifest.json"))), "skipped": True}
    for f in os.listdir(pdir):
        if f.endswith(".bin"):
            os.remove(os.path.join(pdir, f))

    reader = READERS[pool["reader"]]
    e = enc()
    buf = np.empty(SHARD_TOKENS + 2 * BATCH_DOCS * 4096, dtype=np.uint16)
    fill = shard = total = 0
    err = None

    def flush(n):
        nonlocal shard
        write_datafile(os.path.join(pdir, f"shard_{shard:06d}.bin"), buf[:n])
        shard += 1

    def emit(toks_list):
        nonlocal fill, total
        stop = False
        for toks in toks_list:
            n = len(toks) + 1
            buf[fill] = EOT
            buf[fill + 1:fill + n] = np.asarray(toks, dtype=np.uint16)
            fill += n
            total += n
            if fill >= SHARD_TOKENS:
                flush(SHARD_TOKENS)
                rem = fill - SHARD_TOKENS
                buf[:rem] = buf[SHARD_TOKENS:fill]
                fill = rem
            if total >= target:
                stop = True
                break
        return stop

    t0 = time.time()
    try:
        batch = []
        for text in reader(pool, out):
            batch.append(text)
            if len(batch) >= BATCH_DOCS:
                if emit(e.encode_ordinary_batch(batch)):
                    batch = []
                    break
                batch = []
        else:
            if batch:
                emit(e.encode_ordinary_batch(batch))
    except Exception:
        err = traceback.format_exc()

    if fill > 0:
        flush(fill)
    m = dict(name=name, group=pool["group"], tokens=int(total), shards=shard,
             target=int(target), done=(err is None and total > 0),
             elapsed=round(time.time() - t0, 1), error=err)
    json.dump(m, open(os.path.join(pdir, "manifest.json"), "w"), indent=1)
    return m


def _fetch_worker(a):
    pool, out, target = a
    try:
        return fetch_pool(pool, out, target)
    except Exception:
        return dict(name=pool["name"], files=0, error=traceback.format_exc())


def _tok_worker(a):
    pool, out, target = a
    try:
        return build_one(pool, out, target)
    except Exception:
        return dict(name=pool["name"], group=pool["group"], tokens=0, shards=0,
                    done=False, error=traceback.format_exc())


# ------------------------------------------------------------------ driver
def main():
    global CACHE_DIR, OFFLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "fetch", "tokenize"], default="full")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--target", type=float, default=1e9, help="tokens per pool")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--probe", nargs="*", help="only these pool names")
    ap.add_argument("--groups", nargs="*", help="only these groups")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    CACHE_DIR = args.cache_dir
    OFFLINE = (args.mode == "tokenize")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    pools = pools_mod.all_pools()
    if args.probe:
        keep = set(args.probe)
        pools = [p for p in pools if p["name"] in keep]
    if args.groups:
        gk = set(args.groups)
        pools = [p for p in pools if p["group"] in gk]
    if args.limit:
        pools = pools[:args.limit]
    pools = pools[args.shard_index::args.num_shards]  # disjoint slice for array tasks

    target = int(args.target)
    tag = f"[{args.mode} shard {args.shard_index}/{args.num_shards}]"
    print(f"{tag} {len(pools)} pools -> {args.out} (target {target:,}, {args.workers} workers)",
          flush=True)

    # Pre-cache repo file lists once (avoids each worker re-listing the 273k-file dolma3_pool).
    if args.mode in ("fetch", "full"):
        for repo in sorted({p["repo"] for p in pools if p["reader"].startswith("hf_") and p["repo"]}):
            print(f"  caching file list: {repo}", flush=True)
            _repo_files(repo, args.out)

    worker = _fetch_worker if args.mode == "fetch" else _tok_worker
    verb = "fetched" if args.mode == "fetch" else "tokenized"
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, (p, args.out, target)): p["name"] for p in pools}
        for i, fut in enumerate(as_completed(futs), 1):
            m = fut.result()
            results.append(m)
            if args.mode == "fetch":
                s = "skip" if m.get("skipped") else ("OK" if not m.get("error") else "FAIL")
                print(f"{tag} [{i}/{len(pools)}] {s:4} {m['name']:40} "
                      f"{m.get('files',0)} files {m.get('fetched_bytes',0)/1e9:.2f}GB", flush=True)
            else:
                s = "skip" if m.get("skipped") else ("OK" if m.get("done") else "FAIL")
                print(f"{tag} [{i}/{len(pools)}] {s:4} {m['name']:40} "
                      f"{m.get('tokens',0):>12,} tok  {m.get('elapsed','?')}s", flush=True)
            if m.get("error"):
                print("   " + m["error"].splitlines()[-1], flush=True)

    # aggregate manifest only for single-process runs (array tasks would race on it;
    # per-pool manifest.json is authoritative — regenerate the aggregate separately).
    if args.mode != "fetch" and args.num_shards == 1:
        agg = os.path.join(args.out, "ingest_manifest.json")
        prior = {}
        if os.path.exists(agg):
            try:
                prior = {r["name"]: r for r in json.load(open(agg))}
            except Exception:
                prior = {}
        for m in results:
            prior[m["name"]] = m
        json.dump(list(prior.values()), open(agg, "w"), indent=1)

    ok = sum(1 for m in results if (m.get("done") or (args.mode == "fetch" and not m.get("error"))))
    print(f"\n{tag} {verb} {ok}/{len(results)} pools, {round(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
