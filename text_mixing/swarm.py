import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_JSON = os.path.join(HERE, "..", "data", "text_mixing", "pools.json")
RESULTS_ROOT = os.path.join(os.path.dirname(HERE), "results", "text_mixing")  # repo-side artifacts
DATA_ROOT = "/datasets/pretraining_data/dhei/speedrun/text_mixing/tok"
DS_BASE = "/datasets/pretraining_data/dhei/speedrun/text_mixing/results"      # per-model jsons
CKPT_BASE = "/checkpoint/transformer2/dhei/speedrun/text_mixing"
TIKTOKEN = os.path.join(os.path.dirname(HERE), "evals", ".tiktoken_cache")  # repo-local BPE cache


def paths(exp):
    d = os.path.join(RESULTS_ROOT, exp)
    return dict(exp_dir=d, mixtures=f"{d}/mixtures.json", logs=f"{d}/logs",
                results=f"{DS_BASE}/{exp}/results", reeval=f"{DS_BASE}/{exp}/reeval",
                ckpt=f"{CKPT_BASE}/{exp}")


def load_pools():
    return json.load(open(POOLS_JSON))


def _njobs(name):
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", "dhei"), "-h", "-o", "%j"],
                         capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l.strip() == name)


def _wait(name, poll=60):
    print(f"waiting for job '{name}' to finish ...", flush=True)
    while True:
        n = _njobs(name)
        if n == 0:
            print(f"  '{name}' done", flush=True)
            return
        time.sleep(poll)


def pool_subset(pools, mode):
    if mode == "all":
        return pools
    if mode == "cc_top_vig":  # only the highest-quality CC bucket + all non-CC sources
        vmax = max(p["vigintile"] for p in pools if p["group"] == "cc")
        return [p for p in pools if p["group"] != "cc" or p["vigintile"] == vmax]
    raise SystemExit(f"unknown --pools {mode}")


# ---------------------------------------------------------------- config
def gen_mixtures(names, groups, n, seed):
    """Diverse mixtures: pinned baselines + sparse Dirichlet over random k-subsets."""
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for nm, g in zip(names, groups):
        by_group[g].append(nm)
    mixes = []

    def add(kind, w):
        s = sum(v for v in w.values() if v > 0)
        w = {k: v / s for k, v in w.items() if v > 0}
        mixes.append(dict(run_id=f"m{len(mixes):04d}", kind=kind, weights=w))

    add("uniform_all", {nm: 1.0 for nm in names})
    for g, gp in by_group.items():
        add(f"only_{g}", {nm: 1.0 for nm in gp})
    ks = [k for k in (2, 4, 8, 16, 32, 64, len(names)) if k <= len(names)]
    while len(mixes) < n:
        k = int(rng.choice(ks))
        idx = rng.choice(len(names), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        add(f"sparse_k{k}", {names[i]: float(x) for i, x in zip(idx, w)})
    return mixes[:n]


def cmd_config(args):
    p = paths(args.exp)
    os.makedirs(p["exp_dir"], exist_ok=True)
    pools = pool_subset(load_pools(), args.pools)
    names = [x["name"] for x in pools]
    groups = [x["group"] for x in pools]
    mixes = gen_mixtures(names, groups, args.n, args.seed)
    json.dump(mixes, open(p["mixtures"], "w"))
    supp = [len(m["weights"]) for m in mixes]
    print(f"exp={args.exp}  pools={len(names)} {dict(Counter(groups))}")
    print(f"wrote {len(mixes)} mixtures -> {p['mixtures']}  (support "
          f"min={min(supp)} med={int(np.median(supp))} max={max(supp)})")


# ---------------------------------------------------------------- launch (train array)
TRAIN_SBATCH = """#!/bin/bash
#SBATCH --job-name=sw_{exp}
#SBATCH --partition=h200
#SBATCH --qos=h200_transformer2_high
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time={time}
#SBATCH --array=0-{last}%{throttle}
#SBATCH --output={logs}/train_%A_%a.out
set -euo pipefail
cd {here}
export DATA_ROOT={data_root} MIXTURES_JSON={mixtures}
export RESULT_DIR={results} CKPT_DIR={ckpt} TIKTOKEN_CACHE_DIR={tiktoken}
export N_LAYER={n_layer} N_EMBD={n_embd} N_HEAD={n_head}
export SEQ_LEN={seq_len} BATCH_SEQS={batch_seqs} TOKENS={tokens}
export SAVE_CKPT={save_ckpt} MIN_COVERAGE={min_cov} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../.venv/bin/python train_swarm.py --index "${{SLURM_ARRAY_TASK_ID}}"
"""


def cmd_launch(args):
    p = paths(args.exp)
    os.makedirs(p["logs"], exist_ok=True)
    n = len(json.load(open(p["mixtures"])))
    sb = os.path.join(p["exp_dir"], "train_array.sbatch")
    open(sb, "w").write(TRAIN_SBATCH.format(
        exp=args.exp, time=args.time, last=n - 1, throttle=args.throttle, here=HERE,
        logs=p["logs"], data_root=DATA_ROOT, mixtures=p["mixtures"], results=p["results"], ckpt=p["ckpt"],
        tiktoken=TIKTOKEN, n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        seq_len=args.seq_len, batch_seqs=args.batch_seqs, tokens=args.tokens,
        save_ckpt=args.save_ckpt, min_cov=args.min_coverage))
    jid = subprocess.run(["sbatch", "--parsable", sb], capture_output=True, text=True
                         ).stdout.strip().splitlines()[-1]
    print(f"exp={args.exp}: submitted training array {jid} ({n} models, 0-{n-1}%{args.throttle})")
    if args.wait:
        _wait(f"sw_{args.exp}")


# ---------------------------------------------------------------- reeval (eval array)
REEVAL_SBATCH = """#!/bin/bash
#SBATCH --job-name=re_{exp}
#SBATCH --partition=h200
#SBATCH --qos=h200_transformer2_high
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-{last}%{throttle}
#SBATCH --output={logs}/reeval_%A_%a.out
set -euo pipefail
cd {here}
export TIKTOKEN_CACHE_DIR={tiktoken} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../.venv/bin/python swarm.py reeval-worker --exp {exp} \
    --shard "${{SLURM_ARRAY_TASK_ID}}" --num-shards {shards} --max-docs {max_docs}
"""


def cmd_reeval(args):
    p = paths(args.exp)
    os.makedirs(p["logs"], exist_ok=True)
    os.makedirs(p["reeval"], exist_ok=True)
    sb = os.path.join(p["exp_dir"], "reeval_array.sbatch")
    open(sb, "w").write(REEVAL_SBATCH.format(
        exp=args.exp, last=args.shards - 1, throttle=args.throttle, here=HERE,
        tiktoken=TIKTOKEN, shards=args.shards, max_docs=args.max_docs, logs=p["logs"]))
    jid = subprocess.run(["sbatch", "--parsable", sb], capture_output=True, text=True
                         ).stdout.strip().splitlines()[-1]
    print(f"exp={args.exp}: submitted reeval array {jid} ({args.shards} shards)")
    if args.wait:
        _wait(f"re_{args.exp}")


def cmd_reeval_worker(args):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evals"))
    import train_swarm
    import ppl as eval_ppl  # evals/ppl.py
    p = paths(args.exp)
    run_ids = [m["run_id"] for m in json.load(open(p["mixtures"]))][args.shard::args.num_shards]
    weights = {m["run_id"]: m["weights"] for m in json.load(open(p["mixtures"]))}
    device = "cuda"
    for rid in run_ids:
        ck = os.path.join(p["ckpt"], rid, "model.pt")
        outp = os.path.join(p["reeval"], f"{rid}.json")
        if os.path.exists(outp) or not os.path.exists(ck):
            continue
        try:
            model = train_swarm.load_model(ck, device)
        except Exception as e:
            print(f"{rid}: load fail {e}", flush=True)
            continue
        rec = dict(run_id=rid, weights=weights.get(rid, {}))
        rec.update(eval_ppl.score_sets(model, device, max_docs=args.max_docs))
        json.dump(rec, open(outp, "w"))
        print(f"{rid}: " + " ".join(f"{s}={rec[s+'_bpb']:.3f}" for s in eval_ppl.DEFAULT_SETS), flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- collect
def cmd_collect(args):
    import glob
    p = paths(args.exp)
    g_of = {x["name"]: x["group"] for x in load_pools()}
    groups = sorted(set(g_of.values()))
    src = p["reeval"] if glob.glob(p["reeval"] + "/*.json") else p["results"]
    if "reeval" in src:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evals"))
        sets = __import__("ppl").DEFAULT_SETS
    else:
        sets = ["minerva"]
    rows = []
    for fn in glob.glob(src + "/*.json"):
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        gw = defaultdict(float)
        for pool, w in (r.get("weights") or {}).items():
            gw[g_of.get(pool, "?")] += w
        row = {"run_id": r.get("run_id")}
        for s in sets:
            row[f"{s}_bpb"] = r.get(f"{s}_bpb")
        for g in groups:
            row[f"w_{g}"] = round(gw.get(g, 0.0), 5)
        rows.append(row)
    if not rows:
        print(f"no results in {src}")
        return
    cols = ["run_id"] + [f"{s}_bpb" for s in sets] + [f"w_{g}" for g in groups]
    out = os.path.join(p["exp_dir"], "results.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"exp={args.exp}: {len(rows)} models ({src.split('/')[-1]}) -> {out}")
    key = f"{sets[0]}_bpb"
    valid = [r for r in rows if isinstance(r[key], (int, float)) and np.isfinite(r[key])]
    valid.sort(key=lambda r: r[key])
    print(f"\nbest 8 by {sets[0]}:")
    for r in valid[:8]:
        dom = sorted(((r[f"w_{g}"], g) for g in groups), reverse=True)[:3]
        print(f"  {r['run_id']} {key}={r[key]:.3f}  " +
              " ".join(f"{g}:{w:.2f}" for w, g in dom if w > 0))


def cmd_status(args):
    p = paths(args.exp)
    n = len(json.load(open(p["mixtures"]))) if os.path.exists(p["mixtures"]) else 0
    res = len(glob.glob(p["results"] + "/*.json"))
    rev = len(glob.glob(p["reeval"] + "/*.json"))
    tr = _njobs(f"sw_{args.exp}")
    re = _njobs(f"re_{args.exp}")
    print(f"exp={args.exp}: {n} mixtures | trained {res}/{n} (jobs queued/running: {tr}) | "
          f"reeval'd {rev}/{n} (jobs: {re})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config"); c.add_argument("--exp", required=True)
    c.add_argument("--pools", default="cc_top_vig", choices=["all", "cc_top_vig"])
    c.add_argument("--n", type=int, default=1000); c.add_argument("--seed", type=int, default=20260731)
    c.set_defaults(fn=cmd_config)
    l = sub.add_parser("launch"); l.add_argument("--exp", required=True)
    l.add_argument("--wait", action="store_true", help="block until the training array finishes")
    l.add_argument("--time", default="00:40:00"); l.add_argument("--throttle", type=int, default=128)
    l.add_argument("--n-layer", type=int, default=8); l.add_argument("--n-embd", type=int, default=512)
    l.add_argument("--n-head", type=int, default=8); l.add_argument("--seq-len", type=int, default=1024)
    l.add_argument("--batch-seqs", type=int, default=32); l.add_argument("--tokens", default="3e8")
    l.add_argument("--save-ckpt", type=int, default=1); l.add_argument("--min-coverage", type=float, default=0.5)
    l.set_defaults(fn=cmd_launch)
    r = sub.add_parser("reeval"); r.add_argument("--exp", required=True)
    r.add_argument("--wait", action="store_true", help="block until the reeval array finishes")
    r.add_argument("--shards", type=int, default=32); r.add_argument("--throttle", type=int, default=32)
    r.add_argument("--max-docs", type=int, default=2000); r.set_defaults(fn=cmd_reeval)
    rw = sub.add_parser("reeval-worker"); rw.add_argument("--exp", required=True)
    rw.add_argument("--shard", type=int, required=True); rw.add_argument("--num-shards", type=int, required=True)
    rw.add_argument("--max-docs", type=int, default=2000); rw.set_defaults(fn=cmd_reeval_worker)
    co = sub.add_parser("collect"); co.add_argument("--exp", required=True); co.set_defaults(fn=cmd_collect)
    st = sub.add_parser("status"); st.add_argument("--exp", required=True); st.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    args.fn(args)


if __name__ == "__main__":
    main()
