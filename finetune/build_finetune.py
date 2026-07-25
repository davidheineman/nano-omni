#!/usr/bin/env python3
"""Build the stage-2 finetune jobs, one per pretrained record.

Each `track_1_short` record was pretrained (stage 1) on FineWeb to the ~3.28 val target;
its weights live at `results/pretrain/track_1_short/<rec>/checkpoint.pt`. To finetune it
on FineWebEdu we take the record's own `main.py` and splice in three small snippets — the
training math itself is never touched:

    1. USE FINEWEBEDU  – point `args` at the finewebedu .bin files (before loaders build)
    2. LOAD CHECKPOINT – load `checkpoint.pt` just before the training loop
    3. FIXED SCHEDULE  – train for FINETUNE_BUDGET seconds at one fixed cosine LR
                         (5% warmup -> decay to 0, peak 0.2x base), identical for every run

Snippets 2–3 attach at anchors that differ across ~1.5 years of records, so we detect an
"era" per record (see `detect_era`). Everything else is uniform.

Output per record:  results/finetune/track_1_short/<rec>/{main_finetune.py, run_finetune.sbatch}
Run `run_finetune.py submit` next to launch them.
"""
import os, re, glob, stat

HERE = os.path.dirname(os.path.abspath(__file__))                  # modded-nanogpt/finetune
REPO = os.path.dirname(HERE)                                       # modded-nanogpt
PRETRAIN = os.path.join(REPO, "results/pretrain/track_1_short")    # stage-1 checkpoints (read)
OUTPUT = os.path.join(REPO, "results/finetune/track_1_short")      # finetune run dirs (write)

# Finetune settings.
BUDGET_S = 120        # seconds of *train* time per run (val excluded)
PEAK = 0.2            # cosine LR peak, as a multiple of each record's base LR
WARMUP = 0.05         # linear warmup over the first 5% of the budget
# FineWebEdu data as a RELATIVE glob (the run dir has a `data` symlink). Relative works for
# glob.glob, for records that prepend a data path, and for the few that use Path.cwd().glob.
EDU_TRAIN = "data/text/finewebedu10B/finewebedu_train_*.bin"
EDU_VAL = "data/text/finewebedu10B/finewebedu_val_*.bin"

# Slurm (hardcoded so finetune has no dependency on the separately-maintained repro harness).
QOS = "h200_comm_shared"                  # high per-user GPU cap -> all jobs queue at once
ACCOUNT, GPUS, CPUS = "transformer2", 8, 64
HF_HOME = "/storage/home/dhei/.cache/huggingface"
TIKTOKEN = "/storage/home/dhei/.cache/tiktoken"

# ---------------------------------------------------------------------------------------
# The three snippets we splice in. They read settings from env vars (set by the sbatch) and
# talk to the record's own variables (`args`, `model`, `training_time_ms`, `t0`, ...).
# ---------------------------------------------------------------------------------------

# (1+3a) Goes right after `args = Hyperparameters()`: read settings, repoint data at
# FineWebEdu, and define the shared cosine-LR multiplier (paced by `_ft_frac`, which the
# training loop updates each step).
SETUP = '''
# ===== finetune setup (injected) =====
import os as _ft_os, math as _ft_math
_ft_budget_s = float(_ft_os.environ.get("FINETUNE_BUDGET", "0"))
_ft_ckpt     = _ft_os.environ.get("FINETUNE_CKPT", "")
_ft_peak     = float(_ft_os.environ.get("FINETUNE_PEAK", "0.2"))
_ft_warmup   = float(_ft_os.environ.get("FINETUNE_WARMUP", "0.05"))
_ft_frac     = 0.0   # fraction of the time budget elapsed (updated in the loop)
for _attr, _env in (("train_files", "FINETUNE_TRAIN"), ("val_files", "FINETUNE_VAL"),
                    ("input_bin", "FINETUNE_TRAIN"), ("input_val_bin", "FINETUNE_VAL"),
                    ("train_bin", "FINETUNE_TRAIN"), ("val_bin", "FINETUNE_VAL")):
    if _ft_os.environ.get(_env) and hasattr(args, _attr):
        setattr(args, _attr, _ft_os.environ[_env])
def _ft_lr(*_a, **_k):                     # cosine LR multiplier at the current budget frac
    if _ft_frac < _ft_warmup:
        return _ft_peak * (_ft_frac / max(1e-9, _ft_warmup))
    t = (_ft_frac - _ft_warmup) / max(1e-9, 1.0 - _ft_warmup)
    return _ft_peak * 0.5 * (1.0 + _ft_math.cos(_ft_math.pi * min(1.0, t)))
# =====================================
'''

# (2) Goes just before the training loop: load the stage-1 weights, then redirect the
# record's LR schedule to `_ft_lr`. The LR line is filled per era (see LR_HOOK).
LOAD_AND_LR = '''# ===== finetune: load checkpoint + fixed LR (injected) =====
if _ft_ckpt:
    _ft_model = model
    while hasattr(_ft_model, "module"): _ft_model = _ft_model.module   # unwrap DDP
    _ft_state = __import__("torch").load(_ft_ckpt, map_location="cpu", weights_only=True)
    _ft_model.load_state_dict(_ft_state["model"]); del _ft_state
    __import__("torch").cuda.synchronize()
    if int(_ft_os.environ.get("RANK", "0")) == 0:
        print("[finetune] loaded", _ft_ckpt, flush=True)
{lr_hook}
# ===========================================================
'''

# How each era exposes its LR schedule, and how we point it at `_ft_lr`.
LR_HOOK = {
    "schedule_obj": "training_schedule.get_lr = _ft_lr",           # ~2026 records
    "global_fn":    "get_lr = _ft_lr",                             # ~2025 records
    "lambda_lr":    "for _ft_s in schedulers:\n    _ft_s.lr_lambdas = [_ft_lr] * len(_ft_s.lr_lambdas)",  # ~2024
}

# (3b) Replaces the loop header. Runs until the budget elapses (updating `_ft_frac`), then
# stops after one final val. Past the record's own schedule end, the step fed to the body is
# pinned at N-1 so per-step schedules (attention window, sparse-grad, MTP tables) never see an
# out-of-range step -> no recompiles, no index overruns, no collective deadlocks.
def loop_header(nsteps, clock):
    return (LOAD_AND_LR + "\n"
        "_ft_budget_ms = _ft_budget_s * 1000.0\n"
        f"_ft_steps = range({nsteps} + 1) if _ft_budget_ms <= 0 else range(10**9)\n"
        "for _ft_raw in _ft_steps:\n"
        "    if _ft_budget_ms <= 0:\n"
        "        step = _ft_raw\n"
        f"        last_step = (step == {nsteps})\n"
        "    else:\n"
        f"        _ft_elapsed = training_time_ms + 1000.0 * ({clock}() - t0)\n"
        "        _ft_frac = min(1.0, _ft_elapsed / _ft_budget_ms)\n"
        f"        step = min(_ft_raw, {nsteps} - 1)\n"
        "        last_step = (_ft_elapsed >= _ft_budget_ms)")

# ---------------------------------------------------------------------------------------

def splice(src):
    """Splice the three snippets into a record's main.py. Returns (new_src, era) or
    (None, reason) if the record's structure isn't recognized."""
    # Which LR mechanism does this era use?
    if "training_schedule.get_lr" in src:
        era = "schedule_obj"
    elif "LambdaLR(" in src:
        era = "lambda_lr"
        if "schedulers" not in src:
            return None, "LambdaLR era but no `schedulers` variable"
    elif re.search(r"^def get_lr\(", src, re.M) or "get_lr(step)" in src:
        era = "global_fn"
    else:
        return None, "no LR hook found"

    # The training loop header — two shapes across the record history. Capture the exact text
    # (with its `last_step` line) so we can replace it, and note the step-count var + clock.
    hdr = re.search(r"^for step in range\(train_steps \+ 1\):\n[ \t]*last_step = \(?step == train_steps\)?", src, re.M)
    nsteps = "train_steps"
    if not hdr:
        hdr = re.search(r"^for step in range\(args\.num_iterations \+ 1\):\n[ \t]*last_step = \(?step == args\.num_iterations\)?", src, re.M)
        nsteps = "args.num_iterations"
    if not hdr:
        return None, "loop header not matched"
    clock = "time.perf_counter" if "t0 = time.perf_counter()" in src else \
            ("time.time" if "t0 = time.time()" in src else None)
    if clock is None:
        return None, "loop timer `t0` not found"

    if src.count("args = Hyperparameters()") != 1:
        return None, "expected exactly one `args = Hyperparameters()`"
    src = src.replace("args = Hyperparameters()", "args = Hyperparameters()\n" + SETUP, 1)
    new_header = loop_header(nsteps, clock).replace("{lr_hook}", LR_HOOK[era])
    return src.replace(hdr.group(0), new_header, 1), era


def venv_of(src_dir):
    """The venv the stage-1 record used, read from its run.sbatch (default .venv)."""
    sb = os.path.join(src_dir, "run.sbatch")
    if os.path.exists(sb):
        m = re.search(r"/(\.venv[\w]*)/bin/activate", open(sb).read())
        if m:
            return m.group(1)
    return ".venv"


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    built, skipped = {}, []
    for src_dir in sorted(glob.glob(os.path.join(PRETRAIN, "*"))):
        rec = os.path.basename(src_dir)
        ckpt = os.path.join(src_dir, "checkpoint.pt")
        main_py = os.path.join(src_dir, "main.py")
        if not (os.path.exists(ckpt) and os.path.exists(main_py)):
            skipped.append((rec, "no checkpoint.pt")); continue
        # Fix a latent FP8 bug: the mm_t_backward fake declares grad_w column-major, but the
        # real _scaled_mm returns contiguous -> inductor's stride guard trips. Metadata only.
        src = open(main_py).read().replace(
            "return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)",
            "return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32).contiguous()")
        new_src, era = splice(src)
        if new_src is None:
            skipped.append((rec, era)); continue

        run_dir = os.path.join(OUTPUT, rec)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        open(os.path.join(run_dir, "main_finetune.py"), "w").write(new_src)
        tk = os.path.join(src_dir, "triton_kernels.py")          # copy if the record is multifile
        if os.path.exists(tk):
            open(os.path.join(run_dir, "triton_kernels.py"), "w").write(open(tk).read())
        sbatch_path = os.path.join(run_dir, "run_finetune.sbatch")
        open(sbatch_path, "w").write(sbatch(rec, run_dir, venv_of(src_dir), ckpt))
        os.chmod(sbatch_path, os.stat(sbatch_path).st_mode | stat.S_IEXEC)
        built.setdefault(era, []).append(rec)

    print(f"built {sum(len(v) for v in built.values())} finetune runs")
    for era, recs in sorted(built.items()):
        print(f"  {era:12s} {len(recs)}")
    print(f"skipped {len(skipped)}: " + ", ".join(f"{r} ({why})" for r, why in skipped))


SBATCH = """#!/bin/bash
#SBATCH -J finetune-{nn}
#SBATCH -A {account}
#SBATCH -q {qos}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=00:20:00
#SBATCH -o {run_dir}/slurm.out
#SBATCH -e {run_dir}/slurm.out
set -x
REPO={repo}
cd "{run_dir}"
ln -sfn "$REPO/data" data
source "$REPO/{venv}/bin/activate"
export OMP_NUM_THREADS=1 DATA_PATH="$REPO" HF_HOME={hf_home} HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR={tiktoken}
HUB={hf_home}/hub
export LOCAL_KERNELS="varunneal/flash-attention-3=$HUB/models--varunneal--flash-attention-3/snapshots/$(cat $HUB/models--varunneal--flash-attention-3/refs/main):kernels-community/flash-attn3=$HUB/models--kernels-community--flash-attn3/snapshots/$(cat $HUB/models--kernels-community--flash-attn3/refs/main)"
export FINETUNE_BUDGET={budget} FINETUNE_CKPT="{ckpt}" FINETUNE_PEAK={peak} FINETUNE_WARMUP={warmup}
export FINETUNE_TRAIN="{edu_train}" FINETUNE_VAL="{edu_val}"
unset SAVE_CKPT   # never overwrite the stage-1 checkpoint
echo "FINETUNE_START $(date) host=$(hostname) record={rec}"
torchrun --standalone --nproc_per_node={gpus} main_finetune.py
echo "FINETUNE_DONE rc=$? $(date)"
"""

def sbatch(rec, run_dir, venv, ckpt):
    return SBATCH.format(nn=rec.split("_")[0], account=ACCOUNT, qos=QOS, gpus=GPUS, cpus=CPUS,
                         run_dir=run_dir, repo=REPO, venv=venv, hf_home=HF_HOME, tiktoken=TIKTOKEN,
                         budget=BUDGET_S, ckpt=ckpt, peak=PEAK, warmup=WARMUP,
                         edu_train=EDU_TRAIN, edu_val=EDU_VAL, rec=rec)


if __name__ == "__main__":
    main()
