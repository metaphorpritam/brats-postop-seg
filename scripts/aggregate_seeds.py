#!/usr/bin/env python
"""Aggregate the 3-seed sweep -> mean ± SD per class per track, and the A->B delta.

Reads reports/eval_{a,b}_s{0,1,2}.json (written by the sweep) and emits
reports/results_700_seeds.json plus a printed table. Sample SD (n-1) over seeds.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLS = ["NETC", "SNFH", "ET", "RC"]
SEEDS = [0, 1, 2]


def _load(track: str, seed: int) -> dict:
    return json.loads((REPO / f"reports/eval_{track}_s{seed}.json").read_text())


def _agg(track: str) -> dict:
    runs = [_load(track, s) for s in SEEDS]
    out: dict = {"per_class": {}, "counts": runs[0]["counts"], "n_test": runs[0]["n_test"], "mean_fg": {}}
    for c in CLS:
        vals = [float(r["per_class"][c]) for r in runs]
        out["per_class"][c] = {"mean": statistics.mean(vals), "sd": statistics.stdev(vals), "vals": vals}
    mfg = [float(r["mean_fg"]) for r in runs]
    out["mean_fg"] = {"mean": statistics.mean(mfg), "sd": statistics.stdev(mfg), "vals": mfg}
    return out


def main() -> None:
    A, B = _agg("a"), _agg("b")
    # paired per-seed delta on mean foreground Dice
    dmfg = [B["mean_fg"]["vals"][i] - A["mean_fg"]["vals"][i] for i in range(len(SEEDS))]
    dmean, dsd = statistics.mean(dmfg), statistics.stdev(dmfg)

    print(f"3-seed sweep (n_test={A['n_test']}, seeds={SEEDS})\n")
    print(f"{'class':8}{'Track A (mean±sd)':>22}{'Track B (mean±sd)':>22}{'Δ mean':>9}")
    for c in CLS:
        a, b = A["per_class"][c], B["per_class"][c]
        print(f"{c:8}{a['mean']:>13.3f} ±{a['sd']:.3f}{b['mean']:>13.3f} ±{b['sd']:.3f}{b['mean'] - a['mean']:>+9.3f}")
    print(f"{'MEAN_FG':8}{A['mean_fg']['mean']:>13.3f} ±{A['mean_fg']['sd']:.3f}"
          f"{B['mean_fg']['mean']:>13.3f} ±{B['mean_fg']['sd']:.3f}{dmean:>+9.3f}")
    print(f"\nA->B mean-foreground delta: {dmean:+.3f} ± {dsd:.3f} (mean ± SD over {len(SEEDS)} seeds)")

    out = {"n_seeds": len(SEEDS), "seeds": SEEDS, "n_test": A["n_test"],
           "track_a": A, "track_b": B,
           "delta_mean_fg": {"mean": dmean, "sd": dsd, "vals": dmfg}}
    (REPO / "reports/results_700_seeds.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/results_700_seeds.json")


if __name__ == "__main__":
    main()
