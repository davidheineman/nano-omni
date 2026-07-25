#!/usr/bin/env python3
"""Scan records/track_1_short/ and build a runnable copy of each record.

Each record is a self-logging file: source at the top, a line of '=' separators,
then the run log. For every reproducible record this writes
  results/pretrain/track_1_short/<nn>_<record>/
    main.py (+ triton_kernels.py)   the record's own frozen source + small shims
    run.sbatch                      an 8xH200 Slurm script
and records the plan (including why any record is skipped) in manifest.json.

Run:  python pretrain/prepare.py            # build run dirs only
      python pretrain/prepare.py --submit   # build, then launch every record on Slurm at once
"""
import os, re, json, stat, sys, subprocess
import config as C

SEP = re.compile(r"^={20,}\s*$")
DASH = re.compile(r"^-{10,}\s*$")
CODE_MARKERS = ("import torch", "def main", "class GPT", "with open(sys.argv[0])")

# --- shims injected into every main.py (see results/pretrain/README.md "Deviations") ---

# make open() auto-create parent dirs (some records log to logs/<label>/<uuid>.txt)
PRELUDE = (
    "# --- repro prelude: auto-mkdir parent dirs on file writes ---\n"
    "import builtins as _b, os as _os\n"
    "_repro_open = _b.open\n"
    "def _repro_mkopen(file, mode='r', *a, **k):\n"
    "    try:\n"
    "        if isinstance(file,(str,_os.PathLike)) and any(m in str(mode) for m in ('w','a','x')):\n"
    "            _d = _os.path.dirname(_os.fspath(file))\n"
    "            if _d: _os.makedirs(_d, exist_ok=True)\n"
    "    except Exception: pass\n"
    "    return _repro_open(file, mode, *a, **k)\n"
    "_b.open = _repro_mkopen\n"
    "# --- end repro prelude ---\n"
)

# save final weights on rank 0 (guarded; runs after training, cannot affect it)
CKPT_HOOK = (
    "\n# --- repro: save final weights (best-effort, rank 0) ---\n"
    "if int(__import__('os').environ.get('RANK','0'))==0 and __import__('os').environ.get('SAVE_CKPT'):\n"
    "    try:\n"
    "        _m = model\n"
    "        while hasattr(_m,'module'): _m = _m.module\n"
    "        __import__('torch').save({'model': _m.state_dict()}, __import__('os').environ['SAVE_CKPT'])\n"
    "        print('[repro] saved checkpoint to', __import__('os').environ['SAVE_CKPT'], flush=True)\n"
    "    except Exception as _e:\n"
    "        print('[repro] checkpoint save FAILED:', repr(_e), flush=True)\n"
    "# --- end repro hook ---\n"
)

# textual fixes for contributor-machine paths / hardware assumptions / a known bug
SRC_FIXES = [
    # absolute data path baked into one record -> relative (resolves via run-dir symlink)
    ("/data/250010180/bjx/data/fineweb10B", "data/text/fineweb10B"),
    # data was reorganized under data/text/ -> rewrite the frozen records' relative path
    ("data/fineweb10B", "data/text/fineweb10B"),
    # one record gates FA3 on an H100-only branch, else-importing a system flash_attn;
    # on H200 force the get_kernel path (works via LOCAL_KERNELS).
    ('if "H100" in gpu_name:', 'if True:  # repro: H200 uses the get_kernel FA3 path'),
    # 2025-01/02 FP8 records: mm_backward's fake kernel declared grad_w contiguous,
    # but the real op returns a transposed view -> newer inductor's assert_size_stride
    # fails. This is the exact fix the authors shipped later (see 2025-05-30_noallreduce);
    # math-neutral (fake kernels only affect shape/meta inference).
    ("return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)",
     "return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)"),
]

SBATCH = """#!/bin/bash
#SBATCH -J {prefix}{nn}
#SBATCH -p {partition}
#SBATCH -A {account}
#SBATCH -q {qos}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={walltime}
#SBATCH -o {d}/slurm.out
#SBATCH -e {d}/slurm.out
set -x
REPO={repo}
RUNDIR={d}
cd "$RUNDIR"
ln -sfn "$REPO/data" data
source "$REPO/{env}/bin/activate"
export OMP_NUM_THREADS=1
export DATA_PATH="$REPO"
export HF_HOME={hf_home}
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR={tiktoken}
# Load FA3 from the local cache (compute nodes have no internet); this makes
# get_kernel() skip its HF build-tree listing call.
HUB={hf_home}/hub
export LOCAL_KERNELS="varunneal/flash-attention-3=$HUB/models--varunneal--flash-attention-3/snapshots/$(cat $HUB/models--varunneal--flash-attention-3/refs/main):kernels-community/flash-attn3=$HUB/models--kernels-community--flash-attn3/snapshots/$(cat $HUB/models--kernels-community--flash-attn3/refs/main)"
export SAVE_CKPT="$RUNDIR/checkpoint.pt"
echo "REPRO_START $(date) host=$(hostname) record={rec} env={env}"
nvidia-smi -L
torchrun --standalone --nproc_per_node={gpus} main.py
rc=$?
echo "REPRO_DONE rc=$rc $(date)"
newest=$(ls -t logs/*.txt logs/*/*.txt 2>/dev/null | head -1)   # stash the run's metrics log
[ -n "$newest" ] && cp "$newest" metrics.txt
echo "$rc" > rc.txt
"""

# --- find each record's source file + which venv it needs ---

def _looks_like_code(text):
    return any(m in text for m in CODE_MARKERS)

def find_source(record_dir):
    """Return (kind, source_relpath_or_None, env). kind in combined|standalone|logonly."""
    files = [os.path.join(r, f)
             for r, _, fs in os.walk(record_dir) for f in fs
             if f.endswith((".txt", ".log", ".py")) and f != "README.md"]
    # prefer the largest file that is a combined source+log
    combined = []
    for p in files:
        if os.path.getsize(p) < 2000:
            continue
        code = code_section(p)
        if _looks_like_code("\n".join(code)):
            combined.append((p, code))
    combined.sort(key=lambda c: len(c[1]), reverse=True)
    if combined:
        p, code = combined[0]
        return "combined", os.path.relpath(p, C.REPO), pick_env("\n".join(code))
    # else a standalone .py source with separate logs?
    pys = [p for p in files if p.endswith(".py")
           and _looks_like_code(open(p, errors="replace").read())]
    if pys:
        pys.sort(key=os.path.getsize, reverse=True)
        return "standalone", os.path.relpath(pys[0], C.REPO), pick_env(open(pys[0], errors="replace").read())
    return "logonly", None, None

def pick_env(code):
    if "get_kernel(" in code:
        return ".venv"                                   # 2026 kernels-based FA3
    if "flash_attn_interface" in code or "from flash_attn" in code:
        return ".venv_fa3"                               # 2025 FA3 wheel
    return ".venv"                                        # flex_attention / older

# --- extract + transform the source ---

def code_section(path):
    """The source portion of a self-logging record file (drops the run log)."""
    lines = open(path, errors="replace").read().split("\n")
    seps = [i for i, l in enumerate(lines) if SEP.match(l)]
    if seps and seps[0] <= 1 and len(seps) >= 2:         # leading-rule format
        return lines[seps[0] + 1: seps[1]]
    return lines[:seps[0]] if seps else lines            # code-before-rule format

def split_multifile(code):
    """Split a 'main + # triton_kernels.py' concatenation back into two files."""
    h = next(i for i, l in enumerate(code) if l.strip() == "# triton_kernels.py")
    assert DASH.match(code[h - 1]) and DASH.match(code[h + 1]), "bad multifile separator"
    main, tk = code[:h - 1], code[h + 2:]
    while main and not main[-1].strip(): main.pop()
    while tk and not tk[-1].strip(): tk.pop()
    return main, tk

def transform_main(main_lines):
    txt = "\n".join(main_lines)
    for bad, good in SRC_FIXES:
        txt = txt.replace(bad, good)
    marker = "dist.destroy_process_group()"
    idx = txt.rfind(marker)
    txt = (txt[:idx] + CKPT_HOOK + "\n" + txt[idx:]) if idx != -1 else (txt + "\n" + CKPT_HOOK)
    return PRELUDE + txt

def sbatch_for(nn, d, env, rec):
    return SBATCH.format(prefix=C.JOB_PREFIX, nn=nn, partition=C.PARTITION,
                         account=C.ACCOUNT, qos=C.QOS, gpus=C.GPUS, cpus=C.CPUS,
                         walltime=C.WALLTIME_OVERRIDE.get(rec, C.WALLTIME),
                         d=d, repo=C.REPO, env=env,
                         hf_home=C.HF_HOME, tiktoken=C.TIKTOKEN_CACHE, rec=rec)

def main():
    os.makedirs(C.RESULTS, exist_ok=True)
    records = sorted(d for d in os.listdir(C.RECORDS)
                     if os.path.isdir(os.path.join(C.RECORDS, d)))
    manifest = {}
    for i, rec in enumerate(records, 1):
        nn = f"{i:02d}"
        d = os.path.join(C.RESULTS, f"{nn}_{rec}")
        kind, source, env = find_source(os.path.join(C.RECORDS, rec))
        env = C.ENV_OVERRIDE.get(rec, env)
        entry = {"nn": nn, "dir": d, "record": rec, "env": env, "kind": kind}
        if kind == "logonly":
            entry.update(status="infeasible_logonly", reason=C.REASON_LOGONLY)
        elif rec in C.INFEASIBLE:
            entry["status"], entry["reason"] = C.INFEASIBLE[rec]
        else:
            os.makedirs(os.path.join(d, "logs"), exist_ok=True)
            code = code_section(os.path.join(C.REPO, C.SOURCE_OVERRIDE.get(rec, source)))
            multifile = any(l.strip() == "# triton_kernels.py" for l in code)
            entry["multifile"] = multifile
            if multifile:
                main_lines, tk = split_multifile(code)
                open(os.path.join(d, "triton_kernels.py"), "w").write("\n".join(tk) + "\n")
            else:
                main_lines = code
            open(os.path.join(d, "main.py"), "w").write(transform_main(main_lines) + "\n")
            sbp = os.path.join(d, "run.sbatch")
            open(sbp, "w").write(sbatch_for(nn, d, env, rec))
            os.chmod(sbp, os.stat(sbp).st_mode | stat.S_IEXEC)
            entry["status"] = "prepared"
        manifest[rec] = entry

    json.dump(manifest, open(C.MANIFEST, "w"), indent=1)
    prepared = [m for m in manifest.values() if m["status"] == "prepared"]
    print(f"prepared {len(prepared)} records; "
          f"skipped {len(manifest) - len(prepared)} "
          f"({sorted(m['record'] for m in manifest.values() if m['status'] != 'prepared')})")
    print("envs:", {e: sum(1 for m in prepared if m["env"] == e) for e in (".venv", ".venv_fa3")})
    return manifest

def submit_all(manifest):
    """Submit every prepared record's run.sbatch to Slurm at once."""
    n = 0
    for rec, m in sorted(manifest.items(), key=lambda kv: kv[1]["nn"]):
        if m["status"] != "prepared":
            continue
        out = subprocess.run(["sbatch", "--parsable", os.path.join(m["dir"], "run.sbatch")],
                             capture_output=True, text=True)
        jid = next((t for t in out.stdout.split() if t.isdigit()), None)
        print(f"  {m['nn']} {rec} -> {jid or out.stderr.strip()[:80]}")
        n += jid is not None
    print(f"submitted {n} jobs")

if __name__ == "__main__":
    manifest = main()
    if "--submit" in sys.argv:
        submit_all(manifest)
