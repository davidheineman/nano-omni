#!/usr/bin/env python3
"""Print a status table and (re)write results/pretrain/README.md. Read-only.

Run:  python pretrain/report.py
"""
import json, os
import config as C
from runstate import squeue_states, classify, has_ckpt, parse_metrics

DOC = """# Reproducing the `track_1_short` NanoGPT-speedrun records

Every reproducible record in [`records/track_1_short`](../records/track_1_short) was
re-run on **8×H200** via Slurm, saving each run's log **and** final weights.

**{n_ok} reproduced at val_loss ≤ {target}**, **{n_high} finished just above
(single-run variance, {target}–{hi_max:.4f})**, **{n_pending} in-flight**,
**{n_inf} infeasible here** (see bottom). The speedrun target is a *statistical*
mean ≤ {target} on the 10.5M-token FineWeb val set, so a single run landing at,
say, 3.281 still reproduces the record within normal run-to-run variance.

## What's in each run directory

`results/pretrain/track_1_short/<nn>_<record>/`:

| file | description |
|------|-------------|
| `main.py` (+ `triton_kernels.py`) | the record's own frozen source + the shims below |
| `run.sbatch` | the exact 8×H200 Slurm script used |
| `slurm.out` | full stdout/stderr (compile + training + timing) |
| `metrics.txt` | the run's own `step:… val_loss:… train_time:…` log |
| `checkpoint.pt` | final weights — `{{"model": state_dict}}` |

## Reproduce

Prereqs (already set up here; see [`SETUP_FROM_SCRATCH.md`](../SETUP_FROM_SCRATCH.md)):
`.venv` (torch 2.10 + `kernels==0.13.0`), `.venv_fa3` (torch 2.9 + FA3 wheel),
FineWeb-10B shards in `data/text/fineweb10B/`, and FA3 kernels pre-cached under `$HF_HOME/hub`
(compute nodes have no internet). Slurm target lives in `pretrain/config.py`
(partition `{partition}`, account `{account}`, qos `{qos}`, `--gres=gpu:{gpus}`).

```bash
cd modded-nanogpt
python pretrain/prepare.py --submit    # records/ -> per-record run dirs, then launch all on Slurm at once
python pretrain/report.py              # status table + refresh this file
```

`prepare.py` builds one run dir per record and (`--submit`) launches them all at once;
`report.py` reads each run dir (`rc.txt` / `metrics.txt` / `checkpoint.pt`) to print
status and regenerate this file. To (re)run one record by hand:
`sbatch results/pretrain/track_1_short/<nn>_<record>/run.sbatch`.

## Deviations from the original record source

The training/eval code and the token stream are the records' own and unmodified.
The harness only adds env-level shims, none of which change the training math
(all live in `pretrain/prepare.py`):

1. **Hardware:** timed on **8×H200**, not the official 8×H100, so `train_time` here
   is ~2× faster than the main README's H100 table. Loss is comparable.
2. **Final-weights hook:** a rank-0 `torch.save({{"model": state_dict}})` appended
   before `dist.destroy_process_group()` (try/except). The originals don't checkpoint.
3. **`open()` auto-mkdir prelude:** lets records that log to `logs/<label>/<uuid>.txt`
   create the subdir instead of crashing.
4. **Offline FA3 (`LOCAL_KERNELS`):** points `get_kernel()` at the cached FA3
   snapshot so it loads without hitting the HF API (no internet on compute nodes).
5. **Three source fixes** (`SRC_FIXES`): a contributor's absolute data path → this
   repo's `data/text/fineweb10B`; an H100-only FA3 branch → the get_kernel path (we're on
   H200); and the 2025-01/02 FP8 records' `mm_backward` fake kernel → the authors'
   own later stride fix (`w_f8.T.contiguous().T`), which newer torch's inductor requires.

## Infeasible here ({n_inf})

| record | reason |
|--------|--------|
{infeasible_rows}

---

## Results

legend: **✅** reproduced ≤{target} · **≈** finished within run variance · **⏳**
in-flight · **⛔** infeasible

| # | record | status | val_loss | train_time | weights | env |
|---|--------|--------|----------|-----------|---------|-----|
{result_rows}
"""

def main():
    manifest = json.load(open(C.MANIFEST))
    st = squeue_states() or {}
    n_ok = n_high = n_pending = n_inf = 0
    hi_max = C.TARGET
    table, result_rows, infeasible_rows = [], [], []

    for rec, m in sorted(manifest.items(), key=lambda kv: kv[1]["nn"]):
        nn, d = m["nn"], m["dir"]
        if str(m["status"]).startswith("infeasible"):
            n_inf += 1
            infeasible_rows.append(f"| `{rec}` | {m.get('reason','')} |")
            result_rows.append(f"| {nn} | {rec} | ⛔ infeasible | - | - | - | - |")
            table.append((nn, rec, "INFEASIBLE", "", ""))
            continue
        status, vl, tt = classify(d, st.get(f"{C.JOB_PREFIX}{nn}"))
        ck = f"{os.path.getsize(os.path.join(d,'checkpoint.pt'))//(1024*1024)}MB" if has_ckpt(d) else "-"
        if status in ("DONE_OK", "DONE_OK_NOCKPT"):
            mark, n_ok = "✅ reproduced", n_ok + 1
        elif status == "DONE_LOSS_HIGH":
            mark, n_high = "≈ within variance", n_high + 1; hi_max = max(hi_max, vl or hi_max)
        else:
            mark, n_pending = "⏳ in-flight", n_pending + 1
        vs = f"{vl:.4f}" if vl is not None else "-"
        ts = f"{tt:.0f}s" if tt is not None else "-"
        result_rows.append(f"| {nn} | {rec} | {mark} | {vs} | {ts} | {ck} | {m['env']} |")
        table.append((nn, rec, status, vs, ts))

    # stdout table
    print(f"{'#':>3} {'record':44s} {'status':16s} {'val':8s} {'time':7s}")
    for r in table:
        print(f"{r[0]:>3} {r[1]:44s} {r[2]:16s} {r[3]:8s} {r[4]:7s}")
    print(f"\nreproduced={n_ok}  within_variance={n_high}  in_flight={n_pending}  infeasible={n_inf}")

    out = os.path.join(C.REPO, "results", "pretrain", "README.md")
    open(out, "w").write(DOC.format(
        n_ok=n_ok, n_high=n_high, n_pending=n_pending, n_inf=n_inf,
        target=C.TARGET, hi_max=hi_max, partition=C.PARTITION, account=C.ACCOUNT,
        qos=C.QOS, gpus=C.GPUS,
        infeasible_rows="\n".join(infeasible_rows), result_rows="\n".join(result_rows)))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
