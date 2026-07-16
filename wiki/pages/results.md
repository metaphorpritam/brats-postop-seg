---
title: Results
type: concept
status: active
tags: [dice, evaluation, track-a, track-b, metrics, seed-sweep]
sources:
  - ../reports/results.md
  - ../reports/results_700_seeds.json
code: []
links:
  relates: [Experimental Design, Defect Inventory, Data Provenance, Environment, Build Journal]
  derived-from: [Data Provenance]
---

Held-out results for the faithful-defect pipeline (Track A) vs the corrected pipeline (Track B). Metric is **voxel-wise Dice**, NOT BraTS lesion-wise — not comparable to challenge leaderboards. Task is flat 5-class post-treatment BraTS-GLI. Both tracks run one byte-identical shared MONAI `UNet` (**1,983,069 params**). **Design: 700 cases, split fixed at seed 42 (490/105/105, stratified by RC presence); each track trained on 3 seeds {0,1,2}, so all numbers are mean ± SD over 3 runs on one shared test set.** See [[Experimental Design]] for the A/B design, [[Defect Inventory]] for the defect IDs, and [[Build Journal]] for the scale-up decision.

## TEST set (n=105) — per-class voxel Dice (mean ± SD, 3 seeds)

| Class | Track A | Track B | Δ (B−A) | n GT-present (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.009 ± 0.008 | 0.466 ± 0.004 | +0.457 | 105 / 48 |
| SNFH (2) | 0.683 ± 0.017 | 0.870 ± 0.003 | +0.187 | 105 / 105 |
| ET (3) | 0.299 ± 0.033 | 0.650 ± 0.007 | +0.351 | 100 / 85 |
| RC (4) | 0.403 ± 0.026 | 0.694 ± 0.002 | +0.291 | 87 / 87 |
| **Mean foreground** | **0.349 ± 0.013** | **0.670 ± 0.002** | **+0.321 ± 0.013** | |

Raw values (`reports/results_700_seeds.json`): mean_fg A `0.349 ± 0.013` → B `0.670 ± 0.002`. **Headline: mean foreground Dice 0.349 → 0.670, Δ = +0.321 ± 0.013** on unseen data (≈1.9×), reproducing the 200-case pilot's +0.333 **with error bars** and tiny seed variance.

**Interpretation.** NETC recovers most (+0.457) — the genuinely rare class Track A collapses on (Dice 0.009). ET (+0.351) and RC (+0.291) recover strongly; SNFH was already learnable in A (0.683) and still gains to 0.870. The gain is attributable to the pipeline, not the network, since the architecture is held constant (§4.2).

**Scale effect (vs pilot).** At 140 train cases Track A collapsed on everything (mean_fg 0.180); at 490 cases it partly learns the *prevalent* classes (ET/RC GT-present ~83%) and rises to 0.349, while Track B rose further (0.513 → 0.670) — so the absolute delta held even as the ratio fell from ~2.8× to ~1.9×. Only NETC still collapses in A. The sharpened story: the defects' damage is absorbed by data except on the rare class, where the corrections still decide it. See [[Build Journal]].

## Defect D1 exposed by the GT-present counts

Track A reports NETC present in **105/105** test cases and ET in **100/105**, but the faithful labels (Track B) show only **48** and **85**. Bilinear `cv2.resize` of the label volume (D1) interpolates across class boundaries and, after int truncation, **invents** minority-class voxels — the "can invent/destroy labels" harm measured directly, and more starkly than the pilot's 30/30-vs-17. This is why per-class case counts are reported alongside Dice, never Dice alone. See [[Data Provenance]] and [[Defect Inventory]].

## Validation set — per-class Dice (mean ± SD, each track at its own best epoch)

| Class | Track A | Track B | Δ (B−A) |
|---|---|---|---|
| NETC | 0.007 ± 0.004 | 0.545 ± 0.009 | +0.538 |
| SNFH | 0.674 ± 0.014 | 0.856 ± 0.003 | +0.182 |
| ET | 0.261 ± 0.022 | 0.653 ± 0.010 | +0.392 |
| RC | 0.337 ± 0.017 | 0.674 ± 0.003 | +0.337 |
| **Mean foreground** | **0.320 ± 0.008** | **0.682 ± 0.003** | **+0.362** |

Validation voxel accuracy: **A 0.991 / B 0.995** — degenerate (~98%+ background), which is exactly the D4 motivation, so it is *not* a test-set headline. The test evaluator reports Dice only.

## Model selection (D4)

Track A checkpointed on `val_accuracy` (the degenerate metric) — best @ **epochs 18–22** across seeds. Track B checkpointed on **mean foreground Dice** (the fix) — best @ **epochs 64–72**. Selecting on the degenerate accuracy in A locks in a checkpoint that never learns the minority classes.

## Speed (D9)

Naive per-sample loader (Track A, no cache) is I/O-bound at **~23 s/epoch** data load vs **~8 s** with MONAI `PersistentDataset` (Track B, warm cache) — a **~2.8×** data-loading speedup (measured at Gate 2.10). Less dramatic than the original's datacenter 5–10× because the local data is small and gzip-cached by the OS. See [[Environment]].

## Attribution

Track A → Track B differs only in: label interpolation (D1), normalization (D5), loss + class-balanced sampling (D3), model-selection metric (D4), slice sampling / patching (D6), and caching (D9). Architecture held constant. All numbers come from committed run logs under `~/brats/runs/{a,b}_s{0,1,2}/`, aggregated in `reports/results_700_seeds.json`; the 200-case pilot remains in git history.
