---
title: Experimental Design
type: concept
status: active
tags: [ab-experiment, methodology, tracks, phase-gates]
sources:
  - ../CLAUDE.md
  - ../configs/track_a.yaml
  - ../configs/track_b.yaml
code: [../src/brats/model.py]
links:
  relates: [Defect Inventory, Results, Data Provenance, Environment]
---

# Experimental Design

A controlled **A/B experiment** (CLAUDE.md §4). The deliverable is not a leaderboard score — it is the **A→B delta**, evidence that specific named defects (see [[Defect Inventory]]) suppress minority-class segmentation and that fixing them recovers it. "I re-ran someone's code" is a weak CV pointer; "I diagnosed why it underperformed and proved the fix" is a strong one.

## The two tracks

- **Track A — faithful reproduction** (`configs/track_a.yaml`). Reproduces defects D1–D7 and D9. Its job is to **fail in a documented way** (guardrail #5): ~98% voxel accuracy, respectable SNFH Dice, near-zero NETC/ET/RC Dice. A Track A that performs well means the reproduction is wrong. Config: bilinear label resize (D1), global-max normalization (D5), `int(j*2.5)` slice stride (D6), whole-volume 128×128×48, no augmentation, plain `CrossEntropyLoss`, selection on `val_accuracy` (D4), no cache (D9), 10 epochs, batch 1.
- **Track B — corrected pipeline** (`configs/track_b.yaml`). Fixes the same defects. Nearest-neighbour labels (D1), z-score per-modality on non-zero voxels (D5), foreground crop + `RandCropByLabelClassesd` patch sampling with ratios `[1,2,2,2,2]` (D3/D6), `DiceCELoss(include_background=False)`, selection on **mean foreground Dice** (D4), `PersistentDataset` cache (D9), 40 epochs, batch 2 patches of 96³.

Only the **data pipeline, loss, and model-selection criterion** differ between tracks.

## Hold-architecture-constant rule (§4.2)

> The network must be **byte-identical** between Track A and Track B.

Both tracks call the same `build_model` factory (`src/brats/model.py`) against the shared `model:` block in `configs/base.yaml` — a single MONAI `UNet` (spatial_dims=3, in_channels=3, out_channels=5, channels 16→32→64→128→256, strides 2/2/2/2, num_res_units=0, dropout=0.1), **1,983,069 trainable params**. This is what makes the comparison credible: if Track B also "improved" the architecture (InstanceNorm, residual units, deeper encoder), the delta would be confounded and the claim collapses under one interview question ("how do you know it was the loss and not the norm layers?"). D8 (no norm layers) is therefore **noted but not fixed** — or fixed on *both* tracks if at all.

## Why the delta is the deliverable

Post-treatment glioma segmentation is genuinely hard, and this reproduction runs under a heavy self-imposed handicap: ~200 cases (vs ~1350), a 1.98M-param plain U-Net (not an nnU-Net ensemble), ~40 epochs (not 1000), 3 channels (not 5), no TTA, no ensembling, 8 GB VRAM. Absolute numbers will therefore land far below published SOTA. **The A→B delta is immune to all of this, because both tracks carry the identical handicap** — the design is A/B precisely so the result does not depend on beating a leaderboard. See [[Data Provenance]] for the calibration-vs-SOTA argument and guardrail #14 (never chase pre-op numbers). Result: mean foreground Dice **0.180 → 0.513** (~2.8×) on the held-out test set — full numbers on [[Results]].

## Ablation ladder (§4.3 — CUT for time)

Rather than one A→B jump, the fixes were to be applied cumulatively so each gain could be attributed:

| Run | Config | Attributes |
|---|---|---|
| A0 | Baseline (all defects) | — |
| A1 | + NN labels (D1) | data integrity |
| A2 | + per-modality z-score (D5) | normalization |
| B0 | + DiceCE loss & fg-Dice selection (D3, D4) | expected dominant effect |
| B1 | + class-balanced patch sampling (D6) | sampling |

This is the **designated sacrifice** in the 4-day schedule (§8.3, WBS 3.3 — zero float). The pointer becomes "A vs B", not "attributed across five runs". Even a partial ladder (A0 → B0 → B1) would have been more defensible than a single before/after, but it was cut without agonising.

## Phase gates (§7)

Do not proceed past a gate until it passes; report status explicitly.

| Phase | Gate |
|---|---|
| 0 Env | `torch.cuda.is_available()`, GPU printed, `/mnt/` path guard fires |
| 1 Data | Label-contract test passes; RC present in ≥15% of each split |
| 2 Cache | Batch loads with correct shapes/dtypes; labels use NN interp; 2nd-epoch cache hit faster |
| 3 Track A | Reproduces pathology: accuracy ≳0.98 **and minority Dice < 0.05** — else Track A is not faithful |
| 4 Track B | Mean foreground Dice materially exceeds Track A; recorded with per-class case counts |
| 5 Eval | Test-set eval + overlay figures + reproducible README |
| 6 (stretch) | Ablation ladder / second architecture — only if 0–5 done |

Both tracks selected their best checkpoint by their own criterion: Track A best @ **epoch 10** (`val_accuracy`, the degenerate D4 metric), Track B best @ **epoch 38** (`mean_fg_dice`, the fix). The gate-3 pathology held (Track A minority Dice: NETC 0.000, ET 0.054) and gate 4 passed. See [[Results]] and [[Defect Inventory]].
