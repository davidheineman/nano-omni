#!/usr/bin/env python3
"""Shared run-state inspection used by report.py.

A run is only "done" if its sbatch reached the end (rc.txt written); an
intermediate val_loss from a preempted/killed job does NOT count as terminal.
"""
import os, re, subprocess
import config as C

ACTIVE = {"R"}
PENDING = {"PD", "CF", "CG", "RQ", "RD", "RH", "S"}

def squeue_states():
    """Map our job names (rp<nn>) to their Slurm state; None if squeue fails."""
    try:
        out = subprocess.check_output(["squeue", "-u", "dhei", "-h", "-o", "%j %t"], text=True)
    except Exception:
        return None
    st = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].startswith(C.JOB_PREFIX):
            st[p[0]] = p[1]
    return st

def parse_metrics(d):
    """Final (val_loss, train_time_s) from the run's log, else (None, None)."""
    for fn in ("metrics.txt", "slurm.out"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            ms = re.findall(r"val_loss:([0-9.]+)\s+train_time:([0-9]+)ms",
                            open(p, errors="replace").read())
            if ms:
                return float(ms[-1][0]), int(ms[-1][1]) / 1000.0
    return None, None

def read_rc(d):
    p = os.path.join(d, "rc.txt")
    if os.path.exists(p):
        try:
            return int(open(p).read().strip())
        except Exception:
            return None
    return None

def has_ckpt(d):
    p = os.path.join(d, "checkpoint.pt")
    return os.path.exists(p) and os.path.getsize(p) > 1_000_000

def classify(d, jstate):
    """Return (status, val_loss, train_time_s) for a prepared record's run dir."""
    rc = read_rc(d)
    vl, tt = parse_metrics(d)
    if jstate in ACTIVE:
        return "RUNNING", vl, tt
    if jstate in PENDING:
        return "QUEUED", vl, tt
    if rc == 0 and vl is not None:
        if vl <= C.TARGET:
            return ("DONE_OK" if has_ckpt(d) else "DONE_OK_NOCKPT"), vl, tt
        return "DONE_LOSS_HIGH", vl, tt          # finished above target (run variance)
    if rc is not None:
        return "FAILED", vl, tt                  # nonzero exit
    return "TODO", vl, tt                         # never ran / interrupted
