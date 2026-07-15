#!/usr/bin/env python3
"""06_report_figures.py — regenerate the report analysis figures.

Every plotted number is recomputed from committed sources of truth:
  - Training curves & per-class counts : ~/brats/runs/{a,b}/metrics.csv + eval_{a,b}.json
  - Per-class / mean TEST Dice + counts: reports/eval_{a,b}.json
  - D9 data-load speed                 : reports/results.md (the only place it is logged)

Nothing is hard-coded from the task brief; values are read from the files below and
labels are formatted from those parsed values. Colours are the Wong colourblind-safe
palette (validated: worst adjacent CVD ΔE 21.9). Track A = vermillion, Track B = blue.

Outputs -> reports/figures/report/*.png  (dpi=150)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------- paths
HOME = Path.home()
REPO = Path("/home/pritam/code/brats-postop-seg")
REPORTS = REPO / "reports"
RUNS = HOME / "brats" / "runs"
OUTDIR = REPORTS / "figures" / "report"
OUTDIR.mkdir(parents=True, exist_ok=True)

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
def load_metrics(path: Path) -> list[dict]:
    """Read a metrics.csv, returning rows with numeric fields parsed (blank -> None)."""
    rows: list[dict] = []
    with path.open() as fh:
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


def load_eval(track: str) -> dict:
    with (REPORTS / f"eval_{track}.json").open() as fh:
        return json.load(fh)


def dice_curve(rows: list[dict]) -> tuple[list[float], list[float]]:
    """(epochs, mean_fg_dice) for rows that logged a dice value."""
    xs, ys = [], []
    for r in rows:
        if r.get("mean_fg_dice") is not None:
            xs.append(r["epoch"])
            ys.append(r["mean_fg_dice"])
    return xs, ys


def best_epoch(rows: list[dict], key: str) -> tuple[float, float]:
    """Epoch and value of the max of `key` (the metric each track selected on)."""
    best_e, best_v = None, float("-inf")
    for r in rows:
        v = r.get(key)
        if v is not None and v > best_v:
            best_v, best_e = v, r["epoch"]
    return best_e, best_v


def parse_d9() -> tuple[float, float, float]:
    """Extract Track A / Track B 'data load / epoch' seconds and the documented
    speedup factor from results.md. The bar values (~23 s / ~8 s) and the headline
    factor (~2.8x) are all approximate in the source, so the factor is read as
    logged rather than recomputed from the rounded seconds (which would over-state
    precision as 2.9x)."""
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
def barlabels(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def finish(fig, ax, path: Path):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================================ load
mA = load_metrics(RUNS / "a" / "metrics.csv")
mB = load_metrics(RUNS / "b" / "metrics.csv")
eA = load_eval("a")
eB = load_eval("b")

paths: list[Path] = []

# ------------------------------------------------------------------ 1) training_curves
xA, yA = dice_curve(mA)
xB, yB = dice_curve(mB)
# Track A selected on val_accuracy (D4 defect); Track B on mean_fg_dice (D4 fix).
beA, _ = best_epoch(mA, "val_accuracy")
beB, bvB = best_epoch(mB, "mean_fg_dice")

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(xB, yB, color=B_COLOR, lw=2, marker="o", ms=4,
        label=f"Track B (corrected) → {yB[-1]:.3f}")
ax.plot(xA, yA, color=A_COLOR, lw=2, marker="s", ms=4,
        label=f"Track A (faithful defects) → {yA[-1]:.3f}")
# mark each track's selected best epoch
ax.scatter([beB], [bvB], s=90, facecolors="none", edgecolors=B_COLOR, lw=1.6, zorder=5)
ax.annotate(f"B best @ ep{int(beB)}\n(mean_fg_dice)", (beB, bvB),
            xytext=(-6, -34), textcoords="offset points", ha="right", fontsize=8.5,
            color=B_COLOR)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean foreground Dice")
ax.set_title("Training progress: mean foreground Dice per epoch")
ax.set_ylim(0, 0.6)
ax.set_xlim(left=1)
ax.legend(loc="center right")
paths.append(finish(fig, ax, OUTDIR / "training_curves.png"))

# ---------------------------------------------------------------- 2) perclass_test_dice
valsA = [eA["per_class"][c] for c in CLASSES]
valsB = [eB["per_class"][c] for c in CLASSES]
import numpy as np

x = np.arange(len(CLASSES))
w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 4.6))
bA = ax.bar(x - w / 2, valsA, w, color=A_COLOR, edgecolor="white", linewidth=0.6,
            label="Track A (faithful defects)")
bB = ax.bar(x + w / 2, valsB, w, color=B_COLOR, edgecolor="white", linewidth=0.6,
            label="Track B (corrected)")
barlabels(ax, bA, lambda h: f"{h:.3f}")
barlabels(ax, bB, lambda h: f"{h:.3f}")
ax.set_xticks(x, CLASSES)
ax.set_xlabel("Class")
ax.set_ylabel("Voxel-wise Dice")
ax.set_title(f"Held-out TEST per-class Dice (n={eA['n_test']})")
ax.set_ylim(0, 0.95)
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
paths.append(finish(fig, ax, OUTDIR / "perclass_test_dice.png"))

# --------------------------------------------------------------------------- 3) delta_test
deltas = [(c, eB["per_class"][c] - eA["per_class"][c]) for c in CLASSES]
deltas.sort(key=lambda t: t[1])  # ascending -> largest ends at top of barh
labels = [c for c, _ in deltas]
dvals = [d for _, d in deltas]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.barh(range(len(labels)), dvals, color=B_COLOR, edgecolor="white", linewidth=0.6)
for i, d in enumerate(dvals):
    ax.annotate(f"+{d:.3f}", (d, i), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Dice improvement  (Track B − Track A)")
ax.set_title("Per-class Dice recovery, A → B (TEST set)")
ax.set_xlim(0, max(dvals) * 1.18)
ax.grid(axis="y", visible=False)
paths.append(finish(fig, ax, OUTDIR / "delta_test.png"))

# ----------------------------------------------------------------------------- 4) d9_speed
secA, secB, factor = parse_d9()
fig, ax = plt.subplots(figsize=(6.2, 4.6))
bars = ax.bar(["Track A\n(naive, no cache)", "Track B\n(PersistentDataset)"],
              [secA, secB], color=[A_COLOR, B_COLOR], edgecolor="white",
              linewidth=0.6, width=0.6)
barlabels(ax, bars, lambda h: f"~{h:.0f} s")
ax.set_ylabel("Data load / epoch (s)")
ax.set_title("D9: I/O-bound loader vs cached")
ax.set_ylim(0, secA * 1.25)
# annotate the speedup with a bracket between the two bars
ytop = secA * 1.10
ax.annotate("", xy=(1, secB), xytext=(0, secA),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(0.5, ytop, f"~{factor:g}× faster", ha="center", va="bottom",
        fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
paths.append(finish(fig, ax, OUTDIR / "d9_speed.png"))

# ----------------------------------------------------------------------- 5) count_inflation
cA = [eA["counts"][c] for c in CLASSES]
cB = [eB["counts"][c] for c in CLASSES]
x = np.arange(len(CLASSES))
fig, ax = plt.subplots(figsize=(7.6, 4.6))
bA = ax.bar(x - w / 2, cA, w, color=A_COLOR, edgecolor="white", linewidth=0.6,
            label="Track A (bilinear label resize)")
bB = ax.bar(x + w / 2, cB, w, color=B_COLOR, edgecolor="white", linewidth=0.6,
            label="Track B (nearest / faithful)")
barlabels(ax, bA, lambda h: f"{int(round(h))}")
barlabels(ax, bB, lambda h: f"{int(round(h))}")
ax.set_xticks(x, CLASSES)
ax.set_xlabel("Class")
ax.set_ylabel(f"TEST cases with class present (of {eA['n_test']})")
ax.set_title("D1: bilinear label resize invents minority-class voxels")
ax.set_ylim(0, eA["n_test"] * 1.18)
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

# echo the recomputed numbers so the log is self-auditing
print("\nRecomputed values (from source files):")
print("  Track A dice curve  :", [f"{v:.3f}" for v in yA])
print("  Track B dice curve  :", [f"{v:.3f}" for v in yB], f"(best ep{int(beB)}={bvB:.3f})")
print("  TEST per-class A    :", {c: round(eA['per_class'][c], 3) for c in CLASSES},
      "mean_fg", round(eA["mean_fg"], 3))
print("  TEST per-class B    :", {c: round(eB['per_class'][c], 3) for c in CLASSES},
      "mean_fg", round(eB["mean_fg"], 3))
print("  Delta (B-A) sorted  :", [(c, round(d, 3)) for c, d in deltas])
print("  Counts A / B        :", {c: (eA['counts'][c], eB['counts'][c]) for c in CLASSES})
print(f"  D9 speed A/B/factor : ~{secA:.0f}s / ~{secB:.0f}s / ~{factor:g}x (from results.md)")
