#!/usr/bin/env python3
"""Figures for the CV-defense dossier. Every value read from reports/dossier_facts.json.

Outputs -> reports/figures/dossier/*.png (dpi 150). Wong colourblind-safe palette;
Track A = vermillion, Track B = blue, to match the report figures.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path("/home/pritam/code/brats-postop-seg")
F = json.loads((REPO / "reports/dossier_facts.json").read_text())
OUT = REPO / "reports/figures/dossier"
OUT.mkdir(parents=True, exist_ok=True)

A_COL, B_COL, INK, GRID, ZERO = "#D55E00", "#0072B2", "#222222", "#cccccc", "#c1121f"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def finish(fig, ax, name):
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = OUT / name
    fig.savefig(p); plt.close(fig)
    with Image.open(p) as im:
        im.verify()
    print(f"  OK {name}  ({p.stat().st_size/1024:.0f} KB)")


# ---- 1. the delta with its 95% CI, vs the null of zero -------------------------
s = F["stats"]
d, md, ci = s["per_seed_delta"], s["delta_mean"], s["ci95"]
fig, ax = plt.subplots(figsize=(7.6, 2.9))
ax.axvline(0, color=ZERO, ls="--", lw=1.4)
ax.text(0.004, 1.35, "null: no effect", color=ZERO, fontsize=9, va="center")
ax.errorbar([md], [1], xerr=[[md - ci[0]], [ci[1] - md]], fmt="D", color=B_COL,
            ms=11, capsize=6, elinewidth=2, capthick=2, zorder=4,
            label="mean Δ · 95% CI")
ax.scatter(d, [1, 1, 1], color=A_COL, s=42, zorder=5, alpha=.85, label="per-seed Δ")
ax.annotate(f"Δ = {md:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
            (md, 1), xytext=(0, 26), textcoords="offset points", ha="center",
            fontsize=10.5, fontweight="bold")
ax.annotate(f"paired t(2) = {s['paired_t']:.1f}\np = {s['p_two_sided']:.4f}\nCohen's d = {s['cohens_d']:.0f}",
            (0.02, 0.10), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f8", ec="#ccc", lw=.7))
ax.set_xlim(-0.03, 0.42); ax.set_ylim(0.5, 1.7)
ax.set_yticks([]); ax.set_xlabel("Δ mean foreground Dice (Track B − Track A)")
ax.set_title("The delta excludes zero by a wide margin (n = 3 seeds)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", visible=False)
finish(fig, ax, "delta_ci.png")

# ---- 2. stratification: RC prevalence held constant across splits --------------
st = F["stratification"]["splits"]
order = ["train", "val", "test"]
pcts = [st[k]["rc_pct"] for k in order]
ns = [st[k]["n"] for k in order]
overall = 100 * sum(st[k]["rc"] for k in order) / sum(st[k]["n"] for k in order)
fig, ax = plt.subplots(figsize=(7.2, 4.3))
bars = ax.bar([f"{k}\n(n={n})" for k, n in zip(order, ns)], pcts,
              color=[B_COL, "#4a9fd4", "#7bb8e0"], edgecolor="white", width=.62)
ax.axhline(overall, color=A_COL, ls="--", lw=1.6, label=f"overall {overall:.1f}%")
for b, p in zip(bars, pcts):
    ax.annotate(f"{p:.1f}%", (b.get_x() + b.get_width()/2, p), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 100); ax.set_ylabel("cases containing the resection cavity (RC)")
ax.set_title("Stratification holds RC prevalence constant across splits")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
finish(fig, ax, "stratification.png")

# ---- 3. parameter budget by channel width (sums to the exact total) ------------
by = F["params"]["by_channel"]
widths = ["16", "32", "64", "128", "256"]
vals = [by[w] for w in widths]
total = F["params"]["total"]
fig, ax = plt.subplots(figsize=(7.4, 4.3))
bars = ax.bar([f"{w} ch" for w in widths], vals, color=B_COL, edgecolor="white", width=.66)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:,}", (b.get_x()+b.get_width()/2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_ylabel("trainable parameters")
ax.set_title(f"Where the {total:,} parameters live (by feature-map width)")
ax.grid(axis="x", visible=False)
ax.annotate(f"Σ = {total:,} trainable params\n(held byte-identical across both tracks)",
            (0.97, 0.9), xycoords="axes fraction", ha="right", va="top",
            fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fb", ec=B_COL, lw=.8))
finish(fig, ax, "param_breakdown.png")

# ---- 4. why accuracy lies: Track A high accuracy, near-zero rare-class Dice -----
fid = F["fidelity"]["a"]
netc_a = F["per_class"]["NETC"]["A_mean"]
labels = ["voxel\naccuracy", "mean foreground\nDice", "NETC (rarest)\nDice"]
vals = [fid["val_acc_mean"], fid["val_mean_fg_mean"], netc_a]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
bars = ax.bar(labels, vals, color=[A_COL, "#e08a5a", "#efbfa6"], edgecolor="white", width=.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.08); ax.set_ylabel("score")
ax.set_title("Track A: 99% accuracy, near-zero rare-class Dice")
ax.annotate("selecting the checkpoint on accuracy (defect D4)\nrewards exactly this degenerate model",
            (0.5, 0.55), xycoords="axes fraction", ha="center", fontsize=9.5, color="#444")
ax.grid(axis="x", visible=False)
finish(fig, ax, "track_a_fidelity.png")

print("wrote dossier figures ->", OUT)
