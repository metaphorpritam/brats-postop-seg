#!/usr/bin/env python3
"""06_report_figures.py — regenerate the report analysis figures (3-seed sweep).

Every plotted number is recomputed from committed sources of truth:
  - Training curves (mean ± SD band) : ~/brats/runs/{a,b}_s{0,1,2}/metrics.csv
  - Per-class / mean TEST Dice + SD  : reports/eval_{a,b}_s{0,1,2}.json
  - GT-present counts (D1)           : reports/eval_{a,b}_s0.json (fixed test set → seed-invariant)
  - D9 data-load speed               : reports/results.md (the only place it is logged)

Nothing is hard-coded from the task brief; values are read from the files below and
labels are formatted from those parsed values. Error bars are the sample SD (n-1) over
the 3 training seeds {0,1,2} on the FIXED 490/105/105 split (seed 42). Colours are the
Wong colourblind-safe palette. Track A = vermillion, Track B = blue.

Outputs -> reports/figures/report/*.png  (dpi=150)
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- paths
HOME = Path.home()
REPO = Path("/home/pritam/code/brats-postop-seg")
REPORTS = REPO / "reports"
RUNS = HOME / "brats" / "runs"
OUTDIR = REPORTS / "figures" / "report"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]

# ------------------------------------------------------------------ colourblind-safe
A_COLOR = "#D55E00"  # vermillion  -> Track A (faithful defects)
B_COLOR = "#0072B2"  # blue        -> Track B (corrected)
GRID = "#cccccc"
INK = "#222222"

CLASSES = ["NETC", "SNFH", "ET", "RC"]  # class order 1..4

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# ----------------------------------------------------------------------- data loaders
def load_metrics(track: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    with (RUNS / f"{track}_s{seed}" / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            parsed: dict = {}
            for k, v in r.items():
                if v is None or v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def load_eval(track: str, seed: int) -> dict:
    return json.loads((REPORTS / f"eval_{track}_s{seed}.json").read_text())


def ms(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def dice_band(track: str):
    """Return (epochs, mean, sd) of mean_fg_dice across seeds, over epochs all seeds logged."""
    per_seed = {}
    for s in SEEDS:
        d = {}
        for r in load_metrics(track, s):
            if r.get("mean_fg_dice") is not None:
                d[int(r["epoch"])] = r["mean_fg_dice"]
        per_seed[s] = d
    common = sorted(set.intersection(*[set(d) for d in per_seed.values()]))
    mean, sd = [], []
    for e in common:
        m, s_ = ms([per_seed[s][e] for s in SEEDS])
        mean.append(m); sd.append(s_)
    return np.array(common), np.array(mean), np.array(sd)


def best_epochs(track: str, key: str) -> list[int]:
    out = []
    for s in SEEDS:
        be, bv = None, float("-inf")
        for r in load_metrics(track, s):
            v = r.get(key)
            if v is not None and v > bv:
                bv, be = v, int(r["epoch"])
        out.append(be)
    return out


def test_stats(track: str):
    E = [load_eval(track, s) for s in SEEDS]
    pc = {c: ms([float(e["per_class"][c]) for e in E]) for c in CLASSES}
    mfg = ms([float(e["mean_fg"]) for e in E])
    return pc, mfg, E[0]["counts"], E[0]["n_test"]


def parse_d9() -> tuple[float, float, float]:
    text = (REPORTS / "results.md").read_text()
    secA = secB = factor = None
    for line in text.splitlines():
        if "data load / epoch" in line:
            secs = re.findall(r"~?\s*([0-9]+(?:\.[0-9]+)?)\s*s", line)
            if len(secs) >= 2:
                secA, secB = float(secs[0]), float(secs[1])
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[×x]\s*data-loading speedup", text)
    if m:
        factor = float(m.group(1))
    if secA is None or secB is None or factor is None:
        raise ValueError("could not parse D9 speeds/factor from results.md")
    return secA, secB, factor


# ----------------------------------------------------------------------------- helpers
def finish(fig, ax, path: Path):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================================ load
pcA, mfgA, countsA, N = test_stats("a")
pcB, mfgB, countsB, _ = test_stats("b")
paths: list[Path] = []

# ------------------------------------------------------------------ 1) training_curves
eA, yA, sA = dice_band("a")
eB, yB, sB = dice_band("b")
beA, beB = best_epochs("a", "val_accuracy"), best_epochs("b", "mean_fg_dice")

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(eB, yB, color=B_COLOR, lw=2, marker="o", ms=3.5,
        label=f"Track B (corrected) → {mfgB[0]:.3f} ± {mfgB[1]:.3f}")
ax.fill_between(eB, yB - sB, yB + sB, color=B_COLOR, alpha=0.18, lw=0)
ax.plot(eA, yA, color=A_COLOR, lw=2, marker="s", ms=3.5,
        label=f"Track A (faithful defects) → {mfgA[0]:.3f} ± {mfgA[1]:.3f}")
ax.fill_between(eA, yA - sA, yA + sA, color=A_COLOR, alpha=0.18, lw=0)
ax.annotate(f"B selects on mean_fg_dice\n(best @ ep {min(beB)}–{max(beB)})",
            (eB[-1], yB[-1]), xytext=(-8, -30), textcoords="offset points",
            ha="right", fontsize=8.5, color=B_COLOR)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation mean foreground Dice")
ax.set_title("Training progress: mean foreground Dice (mean ± SD, 3 seeds)")
ax.set_ylim(0, 0.75)
ax.set_xlim(left=1)
ax.legend(loc="center right")
paths.append(finish(fig, ax, OUTDIR / "training_curves.png"))

# ---------------------------------------------------------------- 2) perclass_test_dice
mA = [pcA[c][0] for c in CLASSES]; eA_ = [pcA[c][1] for c in CLASSES]
mB = [pcB[c][0] for c in CLASSES]; eB_ = [pcB[c][1] for c in CLASSES]
x = np.arange(len(CLASSES)); w = 0.38
fig, ax = plt.subplots(figsize=(7.8, 4.8))
bA = ax.bar(x - w / 2, mA, w, yerr=eA_, capsize=4, ecolor=INK,
            error_kw={"elinewidth": 1, "capthick": 1},
            color=A_COLOR, edgecolor="white", linewidth=0.6, label="Track A (faithful defects)")
bB = ax.bar(x + w / 2, mB, w, yerr=eB_, capsize=4, ecolor=INK,
            error_kw={"elinewidth": 1, "capthick": 1},
            color=B_COLOR, edgecolor="white", linewidth=0.6, label="Track B (corrected)")
for xi, m, e in list(zip(x - w / 2, mA, eA_)) + list(zip(x + w / 2, mB, eB_)):
    ax.annotate(f"{m:.3f}", (xi, m + e), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x, CLASSES)
ax.set_xlabel("Class"); ax.set_ylabel("Voxel-wise Dice")
ax.set_title(f"Held-out TEST per-class Dice, mean ± SD over 3 seeds (n={N})")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
paths.append(finish(fig, ax, OUTDIR / "perclass_test_dice.png"))

# --------------------------------------------------------------------------- 3) delta_test
dstats = []
for c in CLASSES:
    E_A = [load_eval("a", s) for s in SEEDS]; E_B = [load_eval("b", s) for s in SEEDS]
    dm, dsd = ms([float(E_B[i]["per_class"][c]) - float(E_A[i]["per_class"][c]) for i in range(len(SEEDS))])
    dstats.append((c, dm, dsd))
dstats.sort(key=lambda t: t[1])
labels = [c for c, _, _ in dstats]
dvals = [d for _, d, _ in dstats]
dsds = [s for _, _, s in dstats]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.barh(range(len(labels)), dvals, xerr=dsds, capsize=4, ecolor=INK,
        error_kw={"elinewidth": 1, "capthick": 1},
        color=B_COLOR, edgecolor="white", linewidth=0.6)
for i, (d, s_) in enumerate(zip(dvals, dsds)):
    ax.annotate(f"+{d:.3f} ± {s_:.3f}", (d + s_, i), xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9.5, fontweight="bold")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Dice improvement  (Track B − Track A)")
ax.set_title("Per-class Dice recovery, A → B (TEST, mean ± SD)")
ax.set_xlim(0, max(d + s for d, s in zip(dvals, dsds)) * 1.28)
ax.grid(axis="y", visible=False)
paths.append(finish(fig, ax, OUTDIR / "delta_test.png"))

# ----------------------------------------------------------------------------- 4) d9_speed
secA, secB, factor = parse_d9()
fig, ax = plt.subplots(figsize=(6.2, 4.6))
bars = ax.bar(["Track A\n(naive, no cache)", "Track B\n(PersistentDataset)"],
              [secA, secB], color=[A_COLOR, B_COLOR], edgecolor="white",
              linewidth=0.6, width=0.6)
for b in bars:
    h = b.get_height()
    ax.annotate(f"~{h:.0f} s", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Data load / epoch (s)")
ax.set_title("D9: I/O-bound loader vs cached")
ax.set_ylim(0, secA * 1.25)
ax.annotate("", xy=(1, secB), xytext=(0, secA),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(0.5, secA * 1.10, f"~{factor:g}× faster", ha="center", va="bottom",
        fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
paths.append(finish(fig, ax, OUTDIR / "d9_speed.png"))

# ----------------------------------------------------------------------- 5) count_inflation
cA = [countsA[c] for c in CLASSES]
cB = [countsB[c] for c in CLASSES]
x = np.arange(len(CLASSES))
fig, ax = plt.subplots(figsize=(7.8, 4.8))
bA = ax.bar(x - w / 2, cA, w, color=A_COLOR, edgecolor="white", linewidth=0.6,
            label="Track A (bilinear label resize)")
bB = ax.bar(x + w / 2, cB, w, color=B_COLOR, edgecolor="white", linewidth=0.6,
            label="Track B (nearest / faithful)")
for xi, h in list(zip(x - w / 2, cA)) + list(zip(x + w / 2, cB)):
    ax.annotate(f"{int(round(h))}", (xi, h), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, CLASSES)
ax.set_xlabel("Class"); ax.set_ylabel(f"TEST cases with class present (of {N})")
ax.set_title("D1: bilinear label resize invents minority-class voxels")
ax.set_ylim(0, N * 1.18)
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
paths.append(finish(fig, ax, OUTDIR / "count_inflation.png"))

# ------------------------------------------------------------------------------- verify
print("Wrote figures to", OUTDIR)
for p in paths:
    with Image.open(p) as im:
        im.verify()
    with Image.open(p) as im:
        w_px, h_px = im.size
        fmt = im.format
    kb = p.stat().st_size / 1024
    print(f"  OK  {p.name:24s} {fmt} {w_px}x{h_px}px  {kb:6.1f} KB")

print("\nRecomputed values (from source files):")
print("  TEST A mean_fg   :", f"{mfgA[0]:.3f} ± {mfgA[1]:.3f}")
print("  TEST B mean_fg   :", f"{mfgB[0]:.3f} ± {mfgB[1]:.3f}")
print("  per-class A      :", {c: f"{pcA[c][0]:.3f}±{pcA[c][1]:.3f}" for c in CLASSES})
print("  per-class B      :", {c: f"{pcB[c][0]:.3f}±{pcB[c][1]:.3f}" for c in CLASSES})
print("  delta (sorted)   :", [(c, f"{d:.3f}±{s:.3f}") for c, d, s in dstats])
print("  counts A / B     :", {c: (countsA[c], countsB[c]) for c in CLASSES})
print("  best epochs A/B  :", beA, "/", beB)
print(f"  D9 A/B/factor    : ~{secA:.0f}s / ~{secB:.0f}s / ~{factor:g}x (from results.md)")
