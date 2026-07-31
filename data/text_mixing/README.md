## data mixing ingestion

builds `{source, quality-bucket}` corpora using Olmo 3 data. only pulls ~1B toks and pre-tokenizes to `.bin` using GPT-2 tokenizer (docs delimited by `<|endoftext|>`=50256).

domains:

- 387 splits `common_crawl-<topic>-<vigintile>` (24 topics × 20 quality vigintiles; higher=better)
- 24 splits `olmocr`
- 24 splits `dolmino`
- 15 splits `stack_edu`
- 4 splits `finemath`
- 7 splits `slimpajama`

### setup

```bash
# build pools.json
python data/text_mixing/pools.py --refresh

# fetch (host) -> tokenize array (cpu_lowest)
bash data/text_mixing/fast_build.sh

# probe a few pools first:
python data/text_mixing/build_pools.py \
  --probe cc-literature-v19 stackedu-Python \
  --target 5e6 \
  --out /datasets/pretraining_data/dhei/speedrun/text_mixing/tok_probe
```

### fast build

pull and tokenize subsets in parallel:

```bash
#!/bin/bash
# Fast build: FETCH raw on the host (internet, I/O-bound), then fan out BPE TOKENIZE
# across a cpu_lowest sbatch array. Cuts the ~6h single-host tokenize wall to ~30-40 min.
# The already-running watch_and_launch.sh submits the swarm once tok/ is ~ready.
#
#   WORKERS=32 nohup bash fast_build.sh > fast_build.log 2>&1 &
set -uo pipefail
cd /storage/home/dhei/speedrun/modded-nanogpt
export HF_HUB_OFFLINE=0
WORKERS="${WORKERS:-32}"; TARGET="${TARGET:-1e9}"; NUM_SHARDS="${NUM_SHARDS:-40}"
OUT=/datasets/pretraining_data/dhei/speedrun/text_mixing/tok
CACHE=/datasets/pretraining_data/dhei/speedrun/text_mixing/hf_cache

echo "[fast_build] FETCH phase (host, $WORKERS workers, target $TARGET) ..."
.venv/bin/python data/text_mixing/build_pools.py --mode fetch \
    --workers "$WORKERS" --target "$TARGET" --out "$OUT" --cache-dir "$CACHE"

echo "[fast_build] fetch done -> submitting tokenize array ($NUM_SHARDS tasks, cpu_lowest)"
jid=$(sbatch --parsable --export=ALL,NUM_SHARDS=$NUM_SHARDS,TARGET=$TARGET \
    data/text_mixing/tokenize_array.sbatch | tail -1)
echo "[fast_build] tokenize array job=$jid ; watch_and_launch.sh will fire the swarm when ready."
```
