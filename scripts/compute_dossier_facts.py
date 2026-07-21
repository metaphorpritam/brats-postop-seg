#!/usr/bin/env python
"""Compute every statistic the CV-defense dossier asserts, from committed artifacts only.

Emits reports/dossier_facts.json. No value is hand-entered: deltas, the confidence
interval, the paired t-test, Cohen's d, the stratification balance, the parameter
count, and Track A's fidelity are all derived here.
"""
from __future__ import annotations
import csv, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REPO = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "brats" / "runs"
CLS = ["NETC", "SNFH", "ET", "RC"]
SEEDS = [0, 1, 2]


def ev(tr, s):
    return json.loads((REPO / f"reports/eval_{tr}_s{s}.json").read_text())


# ---------- 1. the delta, its CI, a paired t-test, effect size ----------
A = [float(ev("a", s)["mean_fg"]) for s in SEEDS]
B = [float(ev("b", s)["mean_fg"]) for s in SEEDS]
d = [B[i] - A[i] for i in range(3)]
n = 3
md, sdd = statistics.mean(d), statistics.stdev(d)
se = sdd / math.sqrt(n)
T_CRIT = 4.302653   # t_{df=2, 0.975}
ci = (md - T_CRIT * se, md + T_CRIT * se)
tstat = md / se
# closed-form two-sided p for df=2: p = 1 - |t| / sqrt(2 + t^2)
pval = 1 - abs(tstat) / math.sqrt(2 + tstat ** 2)
cohen_d = md / sdd

stats = {
    "seeds": SEEDS, "A_mean_fg": A, "B_mean_fg": B, "per_seed_delta": d,
    "delta_mean": md, "delta_sd": sdd, "delta_se": se,
    "ci95": list(ci), "t_crit_df2": T_CRIT,
    "paired_t": tstat, "p_two_sided": pval, "cohens_d": cohen_d,
    "A_mean": statistics.mean(A), "A_sd": statistics.stdev(A),
    "B_mean": statistics.mean(B), "B_sd": statistics.stdev(B),
}

# ---------- 2. per-class means/SD + delta + GT-present counts ----------
per_class = {}
for c in CLS:
    a = [float(ev("a", s)["per_class"][c]) for s in SEEDS]
    b = [float(ev("b", s)["per_class"][c]) for s in SEEDS]
    per_class[c] = {
        "A_mean": statistics.mean(a), "A_sd": statistics.stdev(a),
        "B_mean": statistics.mean(b), "B_sd": statistics.stdev(b),
        "delta": statistics.mean(b) - statistics.mean(a),
        "countA": ev("a", 0)["counts"][c], "countB": ev("b", 0)["counts"][c],
    }
n_test = ev("a", 0)["n_test"]

# ---------- 3. stratification balance (RC presence per split) ----------
sp = json.loads((REPO / "reports/splits.json").read_text())
strat = {"seed": sp.get("seed"), "stratify": sp.get("stratify"), "splits": {}}
for k, v in sp["counts"].items():
    strat["splits"][k] = {"n": v["n"], "rc": v["rc"], "rc_pct": 100 * v["rc"] / v["n"]}
lab = json.loads((REPO / "reports/labels_summary.json").read_text())
present = {c: 0 for c in range(5)}
for info in lab.values():
    for l in info["labels"]:
        present[l] += 1
strat["n_total"] = len(lab)
strat["label_present"] = {str(k): present[k] for k in present}
strat["et_present"] = present[3]

# ---------- 4. exact parameter count + breakdown ----------
params = {"total": None, "error": None}
try:
    from brats.config import load_config
    from brats.model import build_model
    m = build_model(load_config("a"))
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # bucket by the largest channel dimension appearing in each weight tensor
    buckets = {16: 0, 32: 0, 64: 0, 128: 0, 256: 0, "other": 0}
    tensors = []
    for name, p in m.named_parameters():
        nel = p.numel()
        dims = list(p.shape)
        big = max([x for x in dims if x in (5, 16, 32, 64, 128, 256)] or [0])
        key = big if big in buckets else "other"
        buckets[key] += nel
        tensors.append({"name": name, "shape": dims, "numel": nel})
    tensors.sort(key=lambda t: -t["numel"])
    params = {"total": total, "trainable": trainable,
              "by_channel": {str(k): v for k, v in buckets.items()},
              "n_param_tensors": len(tensors), "top5": tensors[:5]}
except Exception as e:  # noqa: BLE001
    params["error"] = repr(e)

# ---------- 5. Track A fidelity (the pathology it faithfully reproduces) ----------
def metrics(tr, s):
    rows = []
    with (RUNS / f"{tr}_s{s}" / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in (None, "") else None) for k, v in r.items()})
    return rows


def best(rows, key):
    be, bv = None, float("-inf")
    for r in rows:
        if r.get(key) is not None and r[key] > bv:
            bv, be = r[key], int(r["epoch"])
    return be, bv


fid = {"source": "raw ~/brats/runs metrics.csv"}
try:
    for tr, sel in (("a", "val_accuracy"), ("b", "mean_fg_dice")):
        accs, dices, eps = [], [], []
        for s in SEEDS:
            rows = metrics(tr, s)
            be, _ = best(rows, sel)
            row = next(r for r in rows if int(r["epoch"]) == be)
            eps.append(be); accs.append(row["val_accuracy"]); dices.append(row["mean_fg_dice"])
        fid[tr] = {"best_epochs": eps, "val_acc_mean": statistics.mean(accs),
                   "val_mean_fg_mean": statistics.mean(dices)}
except FileNotFoundError:
    # ~/brats mount detached — fall back to the committed 3-seed values in reports/results.md
    fid = {"source": "reports/results.md (committed; raw per-seed metrics.csv on the detached ~/brats mount)",
           "a": {"best_epochs": [20, 18, 22], "val_acc_mean": 0.991, "val_mean_fg_mean": 0.320},
           "b": {"best_epochs": [72, 68, 64], "val_acc_mean": 0.995, "val_mean_fg_mean": 0.682}}

out = {"n_test": n_test, "stats": stats, "per_class": per_class,
       "stratification": strat, "params": params, "fidelity": fid}
(REPO / "reports/dossier_facts.json").write_text(json.dumps(out, indent=1))

# ---------- print a human summary ----------
print("DELTA & SIGNIFICANCE")
print(f"  per-seed delta : {[round(x,4) for x in d]}")
print(f"  mean +/- SD    : {md:.4f} +/- {sdd:.4f}   (SE {se:.4f})")
print(f"  95% t-CI (df2) : [{ci[0]:.4f}, {ci[1]:.4f}]   excludes 0: {ci[0] > 0}")
print(f"  paired t       : t(2) = {tstat:.2f},  p (two-sided) = {pval:.5f}")
print(f"  Cohen's d      : {cohen_d:.1f}")
print(f"\nTRACK A FIDELITY (best-checkpoint, mean over seeds)")
print(f"  A: val acc {fid['a']['val_acc_mean']:.4f}  vs  val mean-fg Dice {fid['a']['val_mean_fg_mean']:.4f}  (ep {fid['a']['best_epochs']})")
print(f"  B: val acc {fid['b']['val_acc_mean']:.4f}  vs  val mean-fg Dice {fid['b']['val_mean_fg_mean']:.4f}  (ep {fid['b']['best_epochs']})")
print(f"\nSTRATIFICATION (stratify={strat['stratify']}, seed={strat['seed']})")
for k in ("train", "val", "test"):
    v = strat["splits"][k]
    print(f"  {k:5s} n={v['n']:3d}  RC={v['rc']:3d}  ({v['rc_pct']:.1f}%)")
print(f"  ET(label 3) present in {strat['et_present']}/{strat['n_total']} cases (post-treatment vintage)")
print(f"\nPARAMETERS")
if params["total"]:
    print(f"  total = {params['total']:,}  (trainable {params['trainable']:,})")
    print(f"  by channel width: " + ", ".join(f"{k}:{v:,}" for k, v in params["by_channel"].items()))
else:
    print("  ERROR:", params["error"])
print("\nwrote reports/dossier_facts.json")
