# modded-nanogpt setup from scratch

One-time env setup for reproducing `track_1_short` on 8×H200 (CUDA 12.8). The
`pretrain/` harness extracts + runs the records; this covers what it needs.

**Two envs** (FA3 differs by era, don't mix — the 2025 wheel is ABI-pinned to torch 2.9):
- `.venv` — torch 2.10+cu128, 2026 records (FA3 via `kernels.get_kernel(...)`).
- `.venv_fa3` — torch 2.9+cu128, 2025 `*_FA3` records (FA3 via `flash_attn_interface`).

`.venv` is uv-managed (no `pip` — use `uv pip`, or plain pip hits `/opt/conda`).

## 1. Data
```bash
.venv/bin/python data/cached_fineweb10B.py 12   # ~12 train + 1 val shard, from HF kjj0/fineweb10B-gpt2
```

## 2. `.venv` (2026)
Present with torch 2.10; only pin needed:
```bash
VIRTUAL_ENV=$PWD/.venv uv pip install "kernels==0.13.0"
```
0.13.0 is the one version accepting all three `get_kernel(...)` styles the records use.

## 3. `.venv_fa3` (2025)
Install the prebuilt wheel:
```bash
uv venv --python 3.12 .venv_fa3
export VIRTUAL_ENV=$PWD/.venv_fa3
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps wheels/flash_attn_3-built-torch290cu128.whl
.venv_fa3/bin/python -c "from flash_attn_interface import flash_attn_varlen_func; print('FA3 OK')"
```
Rebuild from source if needed: `guilhermeleobas/flash-attention @ fa3-compile`, `hopper/`, SM90-only, `python setup.py bdist_wheel` (~10 min).

## 4. Run
```bash
python pretrain/prepare.py --submit   # build + launch all records on Slurm
python pretrain/report.py             # status + results table
```
See `pretrain/README.md`; results land in `results/pretrain/`.
