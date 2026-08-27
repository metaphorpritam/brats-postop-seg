---
title: Defect Inventory
type: concept
status: active
tags: [defects, reproduction, track-a, track-b, dice]
sources:
  - ../CLAUDE.md
  - ../reference/Model/3D-U-Net/Categorical-CrossEntropy/unet_cc.py
  - ../reports/results.md
code: [../src/brats/transforms.py, ../src/brats/metrics.py]
links:
  relates: [Experimental Design, Results, Data Provenance, Environment]
---

The **spine of the project.** D1–D9 are real defects found by reading the original TF/Keras 3D U-Net (`reference/Model/3D-U-Net/Categorical-CrossEntropy/unet_cc.py`, immutable). [[Experimental Design|Track A]] reproduces D1–D7 faithfully; Track B fixes them. The architecture is held byte-identical across tracks (§4.2), so the A→B delta is attributable to the pipeline, not the network. Headline: mean foreground voxel-Dice **0.180 → 0.513** (~2.8×) on the held-out test set — see [[Results]].

Each entry: **what** (offending line/behaviour) · **why it matters** · **Track B fix** · **empirical confirmation** (where this project measured it).

## D1 — Bilinear label resampling (data-integrity bug)
- **What:** `unet_cc.py:141` resizes the *segmentation* volume with `cv2.resize(seg[...], (128,128))`, whose default is `INTER_LINEAR`. Fractional labels are produced along boundaries, then truncated to int downstream.
- **Why:** Interpolating a categorical label across boundaries **invents and destroys** minority-class voxels — a genuine data-integrity corruption, not just blur.
- **Fix (B):** nearest-neighbour geometry only; MONAI `Orientationd`/crop chain never linearly interpolates labels (`src/brats/transforms.py`, `_track_b_deterministic_head`). Track A reproduces the bug verbatim (`track_a_load_case`).
- **Confirmed:** Track A's GT-present case counts are inflated — NETC **30/30** cases vs the faithful **17**, ET **30/30** vs **25** ([[Results]]; `eval_a.json` vs `eval_b.json`). Bilinear resize literally invented minority labels in cases that have none. This is why Dice is never reported without per-class case counts (§6.5).

## D2 — Per-class Dice shape bug (metric is untrustworthy)
- **What:** `dice_coef_necrotic/edema/enhancing` (`unet_cc.py:174-184`) index `y_true[:,:,:,:,N]` (5 dims) against `y_pred[:,:,:,N]` (4 dims) — a silent broadcasting mismatch. The RC metric is commented out entirely (`:186-188`).
- **Why:** Every minority-class Dice in the original training logs is computed on mis-shaped tensors and is **not trustworthy**. Any claim built on those numbers is unsound — this is the defect this project is careful not to reproduce in its own reporting.
- **Fix (B):** delegate arithmetic to `monai.metrics.DiceMetric(include_background=False, get_not_nans=True)`; one-hot both sides with correct shapes; NaN-aware aggregation over GT-present cases only (`src/brats/metrics.py`, `PerClassDice`).
- **Confirmed:** ours verified correct on synthetic data (metric unit test); reported Dice carries per-class case counts throughout.

## D3 — Class imbalance entirely unhandled (the headline finding)
- **What:** background ≈98% of voxels. Model compiled with plain `loss="categorical_crossentropy"` (`unet_cc.py:277`). An `alpha = [0.05, 0.25, 0.25, 0.25, 0.20]` weight vector is **defined (`:275`) and never used**.
- **Why:** plain CCE lets the model coast on background + SNFH; minority classes are ignored while accuracy reads ~98%.
- **Fix (B):** `DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)` + class-balanced patch sampling (`RandCropByLabelClassesd`, ratios `[1,2,2,2,2]`) — fixing imbalance at both loss and sampling level.
- **Confirmed:** Track A minority Dice is near-zero — NETC **0.000**, ET **0.054**, RC **0.087** — while SNFH holds at **0.580** (test set). Track B recovers to 0.140 / 0.567 / 0.538. The single largest gains: ET **+0.513**, RC **+0.451** ([[Results]]).

## D4 — Model selection on `val_accuracy` (degenerate)
- **What:** `ModelCheckpoint(..., monitor='val_accuracy', mode='max')` (`unet_cc.py:281`).
- **Why:** with ~98% background, accuracy is degenerate; the "best" saved checkpoint is not the best segmenter.
- **Fix (B):** checkpoint on **mean foreground Dice** (§6.6).
- **Confirmed:** validation voxel accuracy is A **0.9896** / B **0.9924** — a 0.003 spread that is meaningless against a 0.34 Dice gap. Track A's best checkpoint lands at **epoch 10** (on val_acc); Track B's at **epoch 38** (on mean_fg_dice) (`~/brats/runs/{a,b}/metrics.csv`). The test evaluator reports Dice only.

## D5 — Crude global-max normalization
- **What:** `return X/np.max(X)` (`unet_cc.py:147`) — one scalar max over all three modalities together.
- **Why:** MRI has no standard intensity units; a single global max ignores per-modality distributions and hurts convergence.
- **Fix (B):** `NormalizeIntensityd(nonzero=True, channel_wise=True)` — per-modality z-score over non-zero voxels (`transforms.py`).
- **Confirmed:** folded into the A→B delta (not isolated — the ablation ladder was the designated schedule sacrifice, §8.3).

## D6 — Non-uniform slice sampling
- **What:** `VOLUME_START_AT + int(j*2.5)` (`unet_cc.py:136-141`) yields strides 0,2,5,7,10,12… — irregular, anisotropic, with no resampling rationale.
- **Why:** introduces uncontrolled geometric distortion.
- **Fix (B):** foreground crop + principled patch-based sampling (`RandCropByLabelClassesd`, 96³ patches), replacing the ad-hoc slice grab.
- **Confirmed:** folded into the A→B delta.

## D7 — Wrong label contract / dead provenance (the narrative entry point)
- **What (CLAUDE.md narrative):** §3/§6.1.2 describe an orphaned `Y[Y==5]=4` line and a `# original 4 -> converted into 3` comment, vestigial from a BraTS-2020 tutorial where labels were `{0,1,2,4}`.
- **What (the committed file — a STRONGER defect):** `unet_cc.py` does not merely carry dead code. Line **144 `Y[Y==4] = 3`** is **active**, and line **145 `tf.one_hot(Y, 4)`** one-hots to **4** classes; `build_unet` uses `num_classes = 4` (`:212`) with a 4-way softmax head (`:265`). So the reference pipeline **collapses RC (label 4) into ET (label 3) and trains a 4-class model** — RC is destroyed before training, and the RC Dice metric is commented out (`:186-188`) to match. The `SEGMENT_CLASSES` dict still lists 5 classes (`:67-73`), so the code silently contradicts its own label map. This is a live label-contract corruption on 2024 post-treatment data (native `{0,1,2,3,4}`), not just a no-op remap.
- **Why:** signals the pipeline was adapted from a pre-op tutorial without re-validating the label contract — pulling this thread reveals it was never audited end-to-end. See [[Data Provenance]] for the vintage table.
- **Fix (B):** `AssertLabelSetd` fails loudly if `set(labels) ⊄ {0,1,2,3,4}` at load time — asserts, never coerces (`transforms.py`). Track B keeps all 5 classes, RC included.

## D8 — Reference network has no normalization layers
- **What:** `build_unet` is Conv3D + Dropout only — no BatchNorm/InstanceNorm anywhere (`unet_cc.py:211-270`).
- **Why:** slower/less stable convergence.
- **Fix (both tracks):** the MONAI `UNet` uses **instance normalization** (its default) on both tracks — the reference's no-norm was *not* reproduced. Applied identically to A and B, so it is corrected on both arms and never confounds the A/B (§4.2). The header describes the *reference*, not this reimplementation.

## D9 — Pipeline is I/O-bound; the GPU starves
- **What:** `DataGenerator.__getitem__` (`unet_cc.py:120-141`) re-loads four gzipped NIfTIs and runs 192 `cv2.resize` calls **per step, every epoch**. Nothing is cached.
- **Why:** the original's ~1000 s/epoch is a gzip-decompression benchmark, not a GPU benchmark (~2% effective utilisation; the log's own `seff` shows 2 cores, 38.5% CPU efficiency).
- **Fix (B):** one-time preprocessing + MONAI `PersistentDataset` caching the deterministic transform head. Track A keeps the naive re-reading loader (`TrackANaiveDataset`) for fidelity.
- **Confirmed:** Track A data-load ~**23 s/epoch** (I/O-bound, no cache) vs Track B ~**8 s** (cache warm) = **~2.8× speedup** ([[Results]]). Less dramatic than the datacenter 5–10× because the laptop dataset is small and OS-cached — stated honestly.

---
*All empirical numbers recomputed from `reports/{eval_a,eval_b}.json` and `~/brats/runs/{a,b}/metrics.csv` (guardrail #9). See [[Results]] for the full table and [[Experimental Design]] for the A/B rationale.*
