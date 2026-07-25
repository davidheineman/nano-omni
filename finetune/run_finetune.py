#!/usr/bin/env python3
"""Launch the finetune jobs and report results. No state files — everything is read back
from the per-record run dirs and the stage-1 pretrain logs.

    python3 run_finetune.py submit          # sbatch every record
    python3 run_finetune.py submit failed   # only records without a finished result
    python3 run_finetune.py report          # -> RESULTS.md + finetune_loss_vs_recency.png

`report` needs the system python3 (has matplotlib); the rest use ../.venv.
"""
import os, re, sys, glob, subprocess, datetime as dt, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNS = os.path.join(REPO, "results/finetune/track_1_short")       # finetune run dirs
PRETRAIN = os.path.join(REPO, "results/pretrain/track_1_short")   # stage-1 logs (FineWeb loss/time)
OUT_MD = os.path.join(REPO, "results/finetune/RESULTS.md")
OUT_PNG = os.path.join(REPO, "results/finetune/finetune_loss_vs_recency.png")


def final_loss_and_time(log):
    """(last val_loss, last train_time_ms) from a speedrun log, else (None, None)."""
    if not os.path.exists(log):
        return None, None
    hits = re.findall(r"val_loss:([0-9.]+) train_time:([0-9]+)ms", open(log, errors="replace").read())
    return (float(hits[-1][0]), int(hits[-1][1])) if hits else (None, None)


def finished(run_dir):
    log = os.path.join(run_dir, "slurm.out")
    return os.path.exists(log) and "FINETUNE_DONE rc=0" in open(log, errors="replace").read()


def submit(only_failed):
    n = 0
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if only_failed and finished(run_dir):
            continue
        jid = subprocess.run(f"sbatch --parsable {run_dir}/run_finetune.sbatch", shell=True,
                             capture_output=True, text=True).stdout.strip().split("\n")[-1]
        print(f"{os.path.basename(run_dir)} -> {jid}"); n += 1
    print(f"submitted {n} jobs")


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    rank = lambda v: {k: i for i, k in enumerate(sorted(range(n), key=lambda j: v[j]))}
    rx, ry = rank(xs), rank(ys)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def report():
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        rec = os.path.basename(run_dir)                          # <nn>_<YYYY-MM-DD>_<title>
        final, ft_ms = final_loss_and_time(os.path.join(run_dir, "slurm.out"))
        if final is None:
            continue
        fineweb, s1_ms = None, None                              # stage-1 FineWeb loss + runtime
        for f in ("metrics.txt", "slurm.out"):
            fineweb, s1_ms = final_loss_and_time(os.path.join(PRETRAIN, rec, f))
            if fineweb is not None:
                break
        rows.append(dict(nn=rec[:2], title=rec[14:], date=dt.date.fromisoformat(rec[3:13]),
                         fineweb=fineweb, s1_min=s1_ms / 60000 if s1_ms else None,
                         final=final, ft_s=ft_ms / 1000 if ft_ms else None))
    rows.sort(key=lambda r: r["date"])
    if not rows:
        print("no finished runs yet"); return

    # A healthy run can't end up worse than the model's own ~3.28 FineWeb loss; those are broken.
    good = [r for r in rows if not (r["fineweb"] and r["final"] > r["fineweb"])]
    bad = [r for r in rows if r not in good]
    rho = spearman([r["date"].toordinal() for r in good], [r["final"] for r in good])
    g26 = [r for r in good if r["date"].year == 2026]
    rho26 = spearman([r["date"].toordinal() for r in g26], [r["final"] for r in g26])
    median = lambda y: statistics.median([r["final"] for r in good if r["date"].year == y] or [float("nan")])

    lines = ["| nn | record | date | FineWeb val | stage-1 runtime | finetuned FineWebEdu | ft time |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        fineweb = f"{r['fineweb']:.4f}" if r["fineweb"] else "-"
        s1 = f"{r['s1_min']:.1f} min" if r["s1_min"] else "-"
        ft = f"{r['ft_s']:.0f} s" if r["ft_s"] else "-"
        flag = " ⚠︎diverged" if r in bad else ""
        lines.append(f"| {r['nn']} | {r['title']} | {r['date']} | {fineweb} | {s1} | {r['final']:.4f}{flag} | {ft} |")
    summary = (f"\n**{len(good)} clean, {len(bad)} diverged.** Median finetuned FineWebEdu: "
               f"2024={median(2024):.4f}, 2025={median(2025):.4f}, 2026={median(2026):.4f}. "
               f"Spearman(recency, loss) = {rho:+.2f} all, {rho26:+.2f} within 2026 "
               f"(≥0 ⇒ newest not better).\n")
    open(OUT_MD, "w").write("# Stage-2 finetune (FineWebEdu, 120s, fixed cosine) — results\n\n"
                            + "\n".join(lines) + "\n" + summary)
    print("\n".join(lines)); print(summary); print("wrote", OUT_MD)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        color = {2024: "#4C78A8", 2025: "#F58518", 2026: "#54A24B"}
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for year, c in color.items():
            pts = [r for r in good if r["date"].year == year]
            if pts:
                ax.scatter([r["date"] for r in pts], [r["final"] for r in pts], c=c, s=46,
                           edgecolor="white", lw=0.6, zorder=3, label=str(year))
                ax.plot([min(r["date"] for r in pts), max(r["date"] for r in pts)],
                        [median(year)] * 2, c=c, lw=2, alpha=0.5, zorder=2)
        ax.set_xlabel("record date"); ax.set_ylabel("finetuned FineWebEdu val loss (120s)")
        ax.set_title("Stage-2 transfer: post-finetune FineWebEdu loss vs record recency\n"
                     f"all reach ~3.28 FineWeb; identical fixed cosine LR   |   "
                     f"Spearman(all)={rho:+.2f}, (2026)={rho26:+.2f}   |   {len(bad)} diverged excluded")
        ax.grid(alpha=0.25, zorder=0); ax.legend(frameon=False, title="record year")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
        print("wrote", OUT_PNG)
    except Exception as e:
        print("plot skipped:", repr(e))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "submit":
        submit(only_failed=len(sys.argv) > 2 and sys.argv[2] == "failed")
    else:
        report()
