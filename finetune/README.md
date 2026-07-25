# Stage-2 transfer: are the newest speedrun records better after finetuning?

Every `track_1_short` record is tuned to hit ~3.28 FineWeb val fastest. We take each
record's `checkpoint.pt`, finetune on **FineWebEdu** for **120 s** with one **fixed
cosine LR** (5% warmup, peak 0.2× base), and compare the final loss.

## Run
```bash
cd modded-nanogpt/finetune        # code; outputs -> ../results/finetune/
python3 build_finetune.py         # splice shim -> per-record main_finetune.py + run_finetune.sbatch
python3 run_finetune.py submit    # launch all on Slurm
python3 run_finetune.py submit failed   # resubmit failures
python3 run_finetune.py report    # -> RESULTS.md + finetune_loss_vs_recency.png
```
`report` needs system `python3` (matplotlib); the rest use `../.venv`.

## How
`build_finetune.py` splices a tiny era-aware shim into each record's `main.py`
(math untouched): swap data to FineWebEdu, load `checkpoint.pt`, run a 120 s budget at
the fixed cosine LR (LR hook auto-detected per era). `run_finetune.py report` reads each
`slurm.out` + the stage-1 pretrain log.

## Result
Median final FineWebEdu — 2024: 3.18, 2025: 3.12, 2026: 3.08. Newer records transfer
better across eras (Spearman ≈ −0.9) but are **flat within 2026** (≈ +0.5): the newest
are not the best. Recent speedups cut FineWeb time ~13× yet don't lower FineWebEdu after
equal finetuning. Runs ending worse than their ~3.28 FineWeb loss (2024 value-embed and
FP8 families) are excluded as diverged.
