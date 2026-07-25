#!/usr/bin/env python3
"""Shared configuration for the track_1_short reproduction harness.

Pipeline:  prepare.py [--submit]  ->  report.py

All paths derive from this file's location, so the harness can live anywhere
inside the repo.  Per-record outputs land in results/pretrain/track_1_short/.
"""
import os

HARNESS = os.path.dirname(os.path.abspath(__file__))     # .../modded-nanogpt/pretrain
REPO = os.path.dirname(HARNESS)                            # .../modded-nanogpt
RECORDS = os.path.join(REPO, "records/track_1_short")     # source records
RESULTS = os.path.join(REPO, "results/pretrain/track_1_short")  # per-record run dirs
MANIFEST = os.path.join(HARNESS, "manifest.json")         # the plan (regenerated)

TARGET = 3.28            # val_loss target on the 10.5M-token FineWeb val set

# --- Slurm submission (8xH200) ---
PARTITION = "h200"
ACCOUNT = "transformer2"
QOS = "h200_transformer2_high"
GPUS = 8
CPUS = 64
WALLTIME = "00:45:00"
JOB_PREFIX = "rp"       # job name = rp<nn>; used to find our jobs in squeue

# --- host caches (shared NFS; compute nodes have no internet) ---
HF_HOME = "/storage/home/dhei/.cache/huggingface"
TIKTOKEN_CACHE = "/storage/home/dhei/.cache/tiktoken"

# The two Python envs (see results/pretrain/README.md):
#   .venv     torch 2.10 + kernels 0.13 (get_kernel FA3)  -> 2026 + pre-FA3 records
#   .venv_fa3 torch 2.9  + FA3 wheel (flash_attn_interface) -> 2025-05..09 records
# classify.py picks one per record; ENV_OVERRIDE can force a choice.
ENV_OVERRIDE = {}

# Force a specific source .txt for a record (classify picks the largest combined
# file; some records ship several run variants with different hardcoded data paths).
SOURCE_OVERRIDE = {
    # 50Bruns ships variants for both fineweb100B (unavailable) and fineweb10B;
    # use the fineweb10B variant so it runs on the data we have.
    "2024-11-04_50Bruns":
        "records/track_1_short/2024-11-04_50Bruns/3d715d41-453a-40d6-9506-421ba69766b2.txt",
}

# Per-record walltime overrides (HH:MM:SS). ScaleUp1B is a 1.5B/~20k-step run.
WALLTIME_OVERRIDE = {
    "2024-10-20_ScaleUp1B": "16:00:00",
}

# Records that cannot be reproduced on this box, with the reason shown in the docs.
REASON_LOGONLY = "pure-log baseline (llm.c C code / raw log); no runnable source in repo"
INFEASIBLE = {
    # aux/example scripts (not numbered records) needing 3rd-party optimizers;
    # recovered once muon / distributed_shampoo are installed into .venv.
    "2024-10-29_Optimizers":
        ("infeasible_deps", "aux optimizer-comparison script; imports distributed_shampoo (install pending)"),
    "2025-05-25_MuonWithAuxAdamExample":
        ("infeasible_deps", "example script; imports the external `muon` package (install pending)"),
}
