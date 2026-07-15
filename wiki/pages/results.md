---
title: Results
type: concept
status: active
tags: [dice, evaluation, track-a, track-b, metrics]
sources:
  - ../reports/results.md
  - ../reports/eval_a.json
  - ../reports/eval_b.json
code: []
links:
  relates: [Experimental Design, Defect Inventory, Data Provenance, Environment]
  derived-from: [Data Provenance]
---

Held-out results for the faithful-defect pipeline (Track A) vs the corrected pipeline (Track B). Metric is **voxel-wise Dice**, NOT BraTS lesion-wise — not comparable to challenge leaderboards. Task is flat 5-class post-treatment BraTS-GLI. Both tracks run one byte-identical shared MONAI `UNet` (**1,983,069 params**), seed 42, split 140/30/30 stratified by RC presence. See [[Experimental Design]] for the A/B design and [[Defect Inventory]] for the defect IDs (D1, D3–D6, D9) referenced below.

## TEST set (n=30) — per-class voxel Dice

| Class | Track A | Track B | Δ (B−A) | n GT-present (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.000 | 0.140 | +0.140 | 30 / 17 |
| SNFH (2) | 0.580 | 0.807 | +0.227 | 30 / 30 |
| ET (3) | 0.054 | 0.567 | +0.513 | 30 / 25 |
| RC (4) | 0.087 | 0.538 | +0.451 | 25 / 25 |
| **Mean foreground** | **0.180** | **0.513** | **+0.333** | |

Raw values (`eval_a.json` / `eval_b.json`, verified): mean_fg A `0.18037693668` → B `0.51312113181`. Headline: mean foreground Dice **0.180 → 0.513**, a ~2.8× improvement on unseen data.

**Interpretation.** ET (+0.513) and RC (+0.451) recover the most — Track A had all but collapsed on both. NETC stays hardest (B 0.140), the smallest/rarest class. SNFH was already learnable in A (0.580) and improves to 0.807. The gain is attributable to the pipeline, not the network, since the architecture is held constant (§4.2).

## Defect D1 exposed by the GT-present counts

Track A reports NETC present in **30/30** test cases and ET in **30/30**, but the faithful labels (Track B) show only **17** and **25**. Bilinear `cv2.resize` of the label volume (D1) interpolates across class boundaries and, after int truncation, **invents** minority-class voxels — the "can invent/destroy labels" harm measured directly. This is why per-class case counts are reported alongside Dice, never Dice alone. See [[Data Provenance]] and [[Defect Inventory]].

## Validation set — per-class Dice (each track at its own best epoch)

| Class | Track A | Track B | Δ (B−A) |
|---|---|---|---|
| NETC | 0.000 | 0.162 | +0.162 |
| SNFH | 0.615 | 0.794 | +0.179 |
| ET | 0.036 | 0.572 | +0.536 |
| RC | 0.140 | 0.624 | +0.484 |
| **Mean foreground** | **0.198** | **0.538** | **+0.340** |

Validation voxel accuracy: **A 0.9896 / B 0.9924** — degenerate (~98% background), which is exactly the D4 motivation, so it is *not* a test-set headline. The test evaluator reports Dice only.

## Model selection (D4)

Track A checkpointed on `val_accuracy` (the degenerate metric) — best @ **epoch 10**. Track B checkpointed on **mean foreground Dice** (the fix) — best @ **epoch 38**. Selecting on the degenerate accuracy in A locks in an early checkpoint that never learns the minority classes.

## Speed (D9)

Naive per-sample loader (Track A, no cache) is I/O-bound at **~23 s/epoch** data load vs **~8 s** with MONAI `PersistentDataset` (Track B, warm cache) — a **~2.8×** data-loading speedup. Less dramatic than the original's datacenter 5–10× because the local data is small and gzip-cached by the OS. See [[Environment]].

## Attribution

Track A → Track B differs only in: label interpolation (D1), normalization (D5), loss + class-balanced sampling (D3), model-selection metric (D4), slice sampling / patching (D6), and caching (D9). Architecture held constant. All numbers come from committed run logs under `~/brats/runs/{a,b}/`.
