# `pretrain/` — track_1_short reproduction harness

Re-runs every reproducible record in `records/track_1_short/` on 8×H200 via Slurm.
Results (logs + weights) land in `../results/pretrain/`; see
[`../results/pretrain/README.md`](../results/pretrain/README.md) for the write-up and results table.

## Pipeline

```bash
cd modded-nanogpt
python pretrain/prepare.py --submit    # 1. records/ -> per-record run dirs, then launch all on Slurm at once
python pretrain/report.py              # 2. status table + refresh results/pretrain/README.md
```

`prepare.py` without `--submit` just builds the run dirs; re-run one record by hand
with `sbatch results/pretrain/track_1_short/<nn>_<record>/run.sbatch`.

## Files

| file | role |
|------|------|
| `config.py` | all knobs: paths, Slurm partition/qos, env choice, per-record overrides, infeasible list |
| `prepare.py` | scan records, extract each source, inject shims, write per-record `run.sbatch` + `manifest.json`; `--submit` launches them all |
| `report.py` | print a status table and regenerate `results/pretrain/README.md` |
| `runstate.py` | shared helpers: query squeue, parse a run's val_loss/rc, classify a run's state |

State file (regenerated): `manifest.json` (the plan).

All record-specific handling (env split, source/walltime overrides, the source
fixes, and which records are infeasible + why) is data in `config.py` / the
`SRC_FIXES` list in `prepare.py` — no logic is hidden per-record.
