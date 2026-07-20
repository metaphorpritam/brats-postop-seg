# BraTS Post-Treatment Glioma Segmentation — PyTorch + MONAI

A controlled **A/B experiment** that reimplements a 3D U-Net pipeline for **flat
5-class post-treatment glioma segmentation** (BraTS-GLI 2024) in modern PyTorch +
MONAI, faithfully reproduces a set of named defects (**D1–D9**) from the original
TensorFlow/Keras research code in **Track A**, then fixes them in **Track B** while
holding the network architecture byte-identical between the two. The deliverable is
**not a leaderboard score** — it is the **A→B delta in mean foreground Dice**
(**0.349 → 0.670, +0.321 ± 0.013** over 3 seeds on a held-out test set), which is immune
to the project's deliberate handicaps because both tracks carry them equally.

📖 **[Read the explainer note online →](https://metaphorpritam.github.io/brats-postop-seg/)** — a
ground-up walkthrough (brain tumours and MRI → CNNs, U-Nets and loss functions derived step by step
→ this experiment and what its numbers mean), alongside the
[technical report](https://metaphorpritam.github.io/brats-postop-seg/report.html).

> **[CLAUDE.md](CLAUDE.md) is the authoritative build spec** — mission, hard
> constraints, the full defect inventory (D1–D9), experimental design, phase gates,
> and guardrails. Read it for anything this README summarizes. Section numbers below
> (§n) refer to it.

---

## Results — held-out TEST set, per-class voxel-wise Dice (n_test = 105, mean ± SD over 3 seeds)

One shared MONAI `UNet` (**1,983,069 params**), byte-identical across tracks (§4.2).
**700 cases**; split `reports/splits.json` (**490/105/105** train/val/test, stratified by RC,
fixed at seed 42); each track trained on **3 seeds {0,1,2}**. Numbers from `reports/results.md` /
`reports/results_700_seeds.json` (guardrail #9 — every number comes from a committed run log).

| Class | Track A | Track B | Δ (B−A) | n GT-present (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.009 ± 0.008 | 0.466 ± 0.004 | **+0.457** | 105 / 48 |
| SNFH (2) | 0.683 ± 0.017 | 0.870 ± 0.003 | **+0.187** | 105 / 105 |
| ET (3)   | 0.299 ± 0.033 | 0.650 ± 0.007 | **+0.351** | 100 / 85 |
| RC (4)   | 0.403 ± 0.026 | 0.694 ± 0.002 | **+0.291** | 87 / 87 |
| **Mean foreground** | **0.349 ± 0.013** | **0.670 ± 0.002** | **+0.321 ± 0.013** | |

**Headline:** mean foreground Dice **0.349 → 0.670**, **Δ = +0.321 ± 0.013** (≈1.9×) on unseen
data; largest recovery on the **rarest class, NETC (+0.457)**, then ET (+0.351) and RC (+0.291).
This reproduces the 200-case pilot's +0.333 **with error bars** (tiny per-seed SD). Track A
reproduces the intended pathology — ~99.1% *validation* voxel accuracy while the rare class sits
near zero (NETC 0.009) — the whole point of a faithful baseline (a Track A that scored well would
mean the reproduction was wrong, guardrail #5). NETC stays the lowest absolute Dice even after the fix.

<sub>Validation set (each track at its own best epoch): mean fg Dice **A 0.320 → B 0.682**;
validation voxel accuracy **A 0.991 / B 0.995** (degenerate — ~98%+ background, which is the whole
D4 point). **D4 model selection:** Track A checkpoints on `val_accuracy` (the degenerate metric —
best @ epochs 18–22); Track B on **mean foreground Dice** (the fix — best @ epochs 64–72). See
`reports/results.md` for the full validation table.</sub>

### Key findings behind the delta

- **D1 — bilinear label resize invents minority labels (measured directly).** Track A
  resizes the *label* volume with `cv2.resize` (default `INTER_LINEAR`), then truncates
  to int. The per-class case counts expose the damage: Track A reports **NETC present in
  105/105** test cases and **ET in 100/105**, but the faithful (nearest-neighbour) labels
  show only **48** and **85**. Interpolating across class boundaries **fabricates**
  minority-class voxels — exactly the "can invent/destroy labels" harm in §3/D1, which
  is why Dice is never reported without its per-class case count (§6.5, guardrail #4).
- **D4 — model selection on the wrong metric.** With ~98% background, `val_accuracy` is
  degenerate; Track A's "best" checkpoint is chosen on it (peaks @ epochs 18–22). Track B
  selects on mean foreground Dice (peaks @ epochs 64–72) — a genuinely better segmenter.
- **D9 — I/O-bound naive loader vs. cached (≈2.8× data-loading speedup).** Track A keeps
  the original's uncached loader; Track B uses MONAI `PersistentDataset`.

  | | Track A (naive, no cache) | Track B (PersistentDataset) |
  |---|---|---|
  | data load / epoch | ~23 s | ~8 s (cache warm) |
  | compute / epoch | ~5 s | ~4 s |

  Track A is I/O-bound (load ≫ compute), reproducing the original's pathology on a
  laptop. The speedup is less dramatic than the original's datacenter 5–10× because our
  data is small and gets cached by the OS page cache — stated honestly.

**Attribution.** Track A → Track B differs *only* in the data pipeline, loss, and
model-selection criterion: label interpolation (D1), per-modality z-score normalization
(D5), DiceCE loss + class-balanced patch sampling (D3), foreground-Dice selection (D4),
slice sampling / patching (D6), and caching (D9). The architecture is held constant
(§4.2), so the delta is attributable to the pipeline, not the network.

---

## Read these caveats before interpreting any number

1. **Voxel-wise Dice, NOT BraTS lesion-wise (§14.3).** The BraTS 2024 challenge scores
   *lesion-wise* Dice + Hausdorff-95 per connected lesion. Ours is plain voxel-wise Dice —
   correct and internally consistent for an A/B comparison, but a **different number on
   identical predictions**. **Never compare these values to challenge leaderboards.**
2. **Harder task than most tutorials.** This is *flat 5-class post-treatment*
   segmentation, not the merged hierarchical ET/TC/WT regions of pre-op BraTS. Treatment
   effect mimics tumour, RC and NETC are genuinely confusable, and RC is a new class. A
   pre-op paper's "WT Dice 0.90" and our "SNFH Dice 0.807" are not on speaking terms
   (§2.3, §14.3).
3. **Partial community mirror — cite the source, not the mirror.** Data is a subsample of
   the Kaggle mirror `i212385nomanarif/2024-brats-glioma` (~700 of the ~1,350 official
   cases). **Cite the BraTS 2024 challenge (de Verdier et al., [arXiv:2405.18368]), not
   the Kaggle re-upload.** The mirror is **resampled to 182×218×182 (MNI-like) space —
   NOT the native BraTS 240×240×155.** We used 200 cases; exact IDs and the dataset hash
   are in `reports/splits.json` and `reports/data_provenance.json`.
4. **D1 inflates Track A's case counts (see above).** Track A's NETC 30/30 and ET 30/30
   are an *artifact of the bilinear label bug*, not real prevalence (true: 17, 25). This
   is reported as a finding, not hidden.
5. **D7 — the committed reference is 4-class, a stronger defect than the spec's
   narrative.** CLAUDE.md §3/D7 describes a vestigial `Y[Y==5] = 4` remap. The actually
   committed reference,
   `reference/Model/3D-U-Net/Categorical-CrossEntropy/unet_cc.py` (lines 144–145), is
   worse: it runs `Y[Y==4] = 3` and `tf.one_hot(Y, 4)` — i.e. it **merges RC (4) into ET
   (3) and one-hot-encodes only 4 classes**, silently destroying the resection-cavity
   class on 2024 post-treatment data. **Our pipeline is deliberately 5-class** — the
   architecture is held constant (§4.2) and we keep RC as its own class specifically to
   *measure* it. So Track A here is the 5-class faithful baseline; it does not replicate
   the reference's RC-merging, which would erase the very class the project exists to
   study. This discrepancy is documented rather than papered over.
6. **SOTA handicap — do not read low absolute numbers as failure (§2.3).** Published
   BraTS 2024 GLI solutions use ~1,350 cases, nnU-Net ensembles, ~1,000 epochs, 4–5
   channels, test-time augmentation, and STAPLE/weighted ensembling. We use **~200 cases,
   a ~2M-param plain U-Net, ~10/40 epochs, 3 channels, no TTA, no ensemble, on an 8 GB
   laptop GPU.** Materially lower numbers are expected and by design; the A→B delta —
   which carries the identical handicap on both sides — is the result.

---

## Reproduce

### Environment (§1.1–1.4)
- `uv` on Python 3.13, PyTorch (bf16 autocast, **no GradScaler** — Ada needs none),
  MONAI 1.6.x, all inside WSL2 on an 8 GB RTX 4060 Mobile.
- **Data, cache, and runs live on native ext4 (`~/brats/...`), OUTSIDE the repo and
  NEVER on a Windows `/mnt/c` drive (§1.3)** — WSL2 reaches Windows drives over the 9p
  protocol, which turns thousands of small NIfTI reads into a 5–10× throughput loss. `00_verify_env.py` asserts no configured path
  starts with `/mnt/`.

```bash
# One-time: sync the uv environment (torch + MONAI etc. from the committed uv.lock)
uv sync

# Fragmentation-reducing allocator (§1.4). 03_train.py also sets this before torch
# imports, but export it for the other scripts too:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Pipeline (run from the repo root; `src/` package is `brats`)

```bash
# Phase 0 — env + GPU + /mnt guard
uv run python scripts/00_verify_env.py

# Phase 1 — fetch from Kaggle, all 700 cases, seeded stratified 490/105/105 split.
# Writes reports/splits.json, reports/data_provenance.json, labels_summary.json.
# (Needs a Kaggle API token at ~/.config/kaggle/kaggle.json; data lands on ~/brats/raw.)
uv run python scripts/01_fetch_subsample.py --n 700 --seed 42 --stage all
# Or run the whole 700-case, 3-seed A/B end-to-end (fetch → sweep → aggregate):
#   bash scripts/run_full_experiment.sh

# Phase 2 — synthetic end-to-end smoke test (no download; guardrail #11)
uv run python scripts/02_smoke_test.py

# Phases 3–4 — train each track. Track A keeps the naive loader (D9); Track B uses the
# PersistentDataset cache (cache_dir on ext4) — there is no separate cache-build step.
uv run python scripts/03_train.py --track a     # faithful baseline, ~10 epochs
uv run python scripts/03_train.py --track b     # corrected pipeline, ~40 epochs

# Phase 5 — evaluate each checkpoint on the held-out TEST split (sliding-window
# inference). Writes reports/eval_<track>.json with per-class Dice AND case counts (§6.5).
uv run python scripts/04_evaluate.py --track a --ckpt ~/brats/runs/a/best.pt
uv run python scripts/04_evaluate.py --track b --ckpt ~/brats/runs/b/best.pt
```

Each track's behaviour is driven entirely by its config
(`configs/base.yaml` + `configs/track_{a,b}.yaml`) — the single source of truth for what
differs between A and B (§4.2). Trained checkpoints, `metrics.csv`, TensorBoard event
files, and `meta.json` land in `~/brats/runs/{a,b}/` (outside the repo; `.gitignore`
excludes `*.pt` and `runs/` — copy logs into `reports/` to keep them as artifacts).

### Where things live
| What | Path |
|---|---|
| Code (this repo) | `~/code/brats-postop-seg` (ext4) |
| Raw data, cache | `~/brats/raw`, `~/brats/cache` (ext4, gitignored, not in repo) |
| Runs / checkpoints / logs | `~/brats/runs/{a,b}/` (ext4, not in repo) |
| Committed results | `reports/results.md`, `reports/eval_{a,b}.json`, `reports/splits.json`, `reports/data_provenance.json` |
| Original TF/Keras source | `reference/Model/` (read-only, immutable — guardrail #13) |

### Figures
![Track A vs Track B overlays](reports/figures/combined_overlay.png)

Three held-out test cases (axial slices) — columns **ground truth / Track A / Track B**,
5-class colormap (NETC blue, SNFH green, ET red, RC amber). Track A predicts essentially
only SNFH with **zero RC and zero NETC** — the minority-class collapse — while Track B
recovers ET and RC in spatial agreement with GT. Per-case panels: `reports/figures/overlay_*.png`;
regenerate with `scripts/05_figures.py`. *Display note:* the tracks infer in different
geometries (A: warped 128³×48; B: foreground-cropped RAS via sliding window) and are
resampled to the native array space for a common view — inference is faithful, only display resampled.

---

## Configuration at a glance

| | Track A (faithful, D1–D7 + D9) | Track B (corrected) |
|---|---|---|
| Label resize | bilinear `cv2.resize` (**D1**) | nearest-neighbour |
| Normalization | global `X / max(X)` (**D5**) | per-modality z-score, non-zero voxels |
| Geometry / sampling | whole 128×128×48, `int(j*2.5)` stride (**D6**) | foreground crop + `RandCropByLabelClasses` 96³, ratios `[1,2,2,2,2]` |
| Loss | plain `CrossEntropyLoss` (**D3**) | `DiceCELoss(include_background=False)` |
| Model selection | `val_accuracy` (**D4**) | mean foreground Dice |
| Data loading | uncached, reload every epoch (**D9**) | `PersistentDataset` cache |
| Epochs / batch | 10 / 1 whole volume | 40 / 2 patches |
| **Architecture** | **MONAI `UNet`, 16→32→64→128→256, 1,983,069 params — identical (§4.2)** | **identical** |

Input: 3 modalities (`t2f`, `t1c`, `t2w`) stacked as channels, matching the original
(which drops `t1n`). Optimizer AdamW, lr 1e-3, cosine schedule, bf16 autocast, seed 42.
**D8** (no norm layers in the net) is noted but deliberately *not* fixed — changing it
would confound the A/B (§4.2).
