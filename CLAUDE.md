# Master Plan — BraTS Post-Treatment Glioma Segmentation (PyTorch + MONAI)

> **Purpose of this file.** This is the authoritative build spec for Claude Code. Read it fully before writing any code. It defines the mission, the hard constraints, the experimental design, the phase gates, and the guardrails. When something here conflicts with a default instinct (e.g. "just use a bigger batch"), **this file wins**. If you believe a section is wrong, say so and stop — do not silently deviate.
>
> **Owner:** Pritam Sarkar (PGDBA Batch 11, IIM Calcutta / IIT Kharagpur / ISI Kolkata)
> **Intended use of output:** CV pointer for the PGDBA placement "Additional Projects" section (CDPO-governed) + interview talking points.
> **Status:** Not started. Created 2026-07-15. **Rev 6 — 4-DAY SCHEDULE IS AUTHORITATIVE (§8); knowledge layer in §13.** Rev 4 — Rev 2: Kaggle mirror replaces gated Synapse route (§6.1). Rev 3: BraTS vintage label-contract table (§6.1.2), guardrails 11–12. Rev 4: difficulty calibration vs published SOTA (§2.3), WBS + critical path (§12). Rev 5: defect D9 added (pipeline is I/O-bound, not compute-bound); training-time estimate corrected downward ~10x; timeline compressed to 4 working days (§8). Rev 6: pageindex-plus + llm-wiki integrated for context, memory, audit, and report generation (§13); guardrails 12–13 added.

---

## 0. Mission

Reimplement, in modern PyTorch + MONAI, a 3D U-Net pipeline for **5-class semantic segmentation of post-treatment glioma MRI** (BraTS-GLI). The source material is an existing TensorFlow/Keras research repo written for an HPC cluster.

**This is not a copy exercise.** The reproduction is a means to an end. The deliverable is:

1. A working, reproducible pipeline that trains on a single 8 GB laptop GPU.
2. A **controlled A/B experiment** showing that specific, named defects in the original pipeline suppress minority-class performance, and that fixing them recovers it.
3. Quantified before/after numbers + figures that survive an interview viva.

The A/B delta **is** the project. A faithful reproduction alone is a weak pointer ("I re-ran someone's code"). A diagnosed-and-fixed reproduction is a strong one ("I found why it underperformed and proved the fix").

---

## 1. Hard constraints

### 1.1 Hardware
| Resource | Spec | Implication |
|---|---|---|
| GPU | RTX 4060 **Mobile**, 8 GB VRAM, Ada (sm_89) | **The binding constraint.** bf16 native. No room for whole-volume batch >1. |
| CPU | i7-12650H (6P + 4E, 16 threads) | DataLoader `num_workers` ≈ 4–6. Do not oversubscribe. |
| OS | Windows + **WSL2** | All training runs inside WSL2. |

### 1.2 Software
- **Python 3.13**, managed by **`uv`**. Do not use conda, pip-in-venv, or poetry.
- **PyTorch** — on Linux, PyPI hosts CUDA-enabled wheels directly (since torch 2.11), so plain `uv add torch` works in WSL2. Verify `torch.cuda.is_available()` before proceeding. Only fall back to the `download.pytorch.org` index if the PyPI wheel resolves CPU-only.
- **MONAI 1.6.x** (requires Python ≥3.10; 3.13 is fine).
- Precision: **bf16 autocast**. Ada supports bf16 natively, so **no `GradScaler` is needed** — do not add one.

### 1.3 The filesystem rule — NON-NEGOTIABLE

> **Never place the dataset, the transform cache, or the run outputs under `/mnt/c/...` (or any Windows drive).**

WSL2 reaches Windows drives over the 9p protocol. This pipeline performs thousands of small NIfTI/cache reads per epoch, and 9p turns that into a 5–10× throughput loss. The GPU will sit idle waiting on I/O and you will misdiagnose it as a model problem.

- Dataset, cache, checkpoints, logs → **native ext4** (`~/brats/...`).
- Code → also ext4 (`~/code/brats-postop-seg`). If Pritam wants to browse it from Windows Explorer, use `\\wsl$\...` or a symlink — do **not** relocate the repo.
- Add a runtime assertion in `00_verify_env.py` that fails loudly if any configured path starts with `/mnt/`.

### 1.4 Environment variables
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces fragmentation OOMs
```

---

## 2. Source material

The original is a multi-architecture BraTS research repo (from IIT-KGP's Param Shakti cluster). It contains **5 architectures × 2 losses**: 2D U-Net, 3D U-Net, Res-U-Net, Attention-U-Net, MPU-Net; each in Categorical-Cross-Entropy and Dice-Loss variants, plus SLURM job scripts and training logs.

**We reimplement the 3D U-Net (CCE) only.** Everything else is out of scope. Do not attempt Attention/Res/MPU variants unless Phase 5 is explicitly reached with time to spare.

### 2.1 What the original does
- **Input:** 3 modalities — `t2f` (FLAIR), `t1c`, `t2w` — stacked as channels. (It **drops** `t1n`, the 4th available modality.)
- **Geometry:** native BraTS volumes are 240×240×155 @ 1 mm isotropic, skull-stripped, co-registered. The script slices to `128×128×48×3`.
- **Model:** plain 3D U-Net, encoder channels 16→32→64→128→256, ~5.6 M params, softmax over 5 classes.
- **Scale:** 1,350 cases (918 train / 270 val / 162 test), ~1,000 s/epoch on a datacenter GPU.

### 2.2 Label map (BraTS-GLI post-treatment)
| ID | Class | Notes |
|---|---|---|
| 0 | Background | ~98% of voxels |
| 1 | NETC — non-enhancing tumor core | minority |
| 2 | SNFH — surrounding non-enhancing FLAIR hyperintensity (the "edema" analog) | largest foreground class |
| 3 | ET — enhancing tissue | minority |
| 4 | RC — resection cavity | **often absent entirely in a given case** |

Class 4's frequent absence is important — see §6.5 on NaN handling in Dice.

### 2.3 Task difficulty — calibrated expectations

**Read this before interpreting any result.** Post-treatment glioma segmentation is substantially harder than the pre-op BraTS task most tutorials target.

**Published state of the art (lesion-wise Dice, BraTS 2024 GLI):**
- Winning MICCAI 2024 solution: **0.873 ± 0.17** overall (trained on n=1350, validated on n=188).
- Kim et al. (arXiv:2409.08143), nnU-Net ensembles, held-out validation N=188:

| Class | Best ensemble | Single nnU-Net (internal val, N=280) |
|---|---|---|
| SNFH | 0.868 | 0.837 |
| NETC | 0.786 | 0.839 |
| ET | 0.733 | 0.772 |
| **RC** | **0.695** | 0.770 |

**RC — the new class — is the hardest.** That is with 1000 epochs of SGD-Nesterov, patches up to 160×192×160 at batch 5, 4–5 input channels, test-time augmentation, and STAPLE/weighted ensembling. Note also the internal→held-out drop: the generalization gap is real even for strong teams.

**Why post-treatment is hard:**
1. **Treatment effect mimics tumour.** Radiation necrosis and pseudoprogression enhance on T1Gd exactly like active tumour. This is an unsolved *clinical* problem, not merely an ML one.
2. **RC vs NETC are genuinely confusable.** Both non-enhancing; chronic cavities track CSF signal on T1 and T2/FLAIR.
3. **Protocol subtleties.** Thin linear enhancement along cavity walls and dura is *excluded* from ET by annotation protocol — but looks like ET. The model must learn a rule, not a texture.
4. **Multi-focal, small, low-contrast lesions.** Post-treatment disease is scattered, not one blob — which is exactly why the challenge scores lesion-wise.
5. **RC absent entirely in many cases** (§6.5).
6. **Flat 5-class** is harder than the merged hierarchical ET/TC/WT regions pre-op work reports.

**Our handicap vs. the above:** ~200 cases (15% of theirs), a 5.6M-param plain U-Net (not an nnU-Net ensemble), ~50 epochs (not 1000), 3 channels (not 5), no TTA, no ensembling, 8 GB VRAM.

> **Expect materially lower numbers, and do not treat that as failure.** Track B landing near SNFH ~0.6–0.75 and minority classes ~0.3–0.6 voxel-wise would be a respectable outcome. **These are guesses to be measured, not targets** — do not tune toward them.
>
> **The deliverable is the A→B delta, which is immune to all of this**, because both tracks carry the identical handicap. This is the whole reason the experimental design is A/B rather than "beat the leaderboard." See guardrail #12.

---

## 3. Defect inventory — the reproduction target

These are real defects identified by reading the original source. **Track A must faithfully reproduce them; Track B fixes them.** Each is a CV/interview talking point, so each must be individually verifiable.

| # | Defect | Why it matters | Fix (Track B) |
|---|---|---|---|
| **D1** | **Segmentation mask resized with bilinear interpolation.** `cv2.resize` defaults to `INTER_LINEAR`; applied to the label volume it produces fractional labels, later truncated to int. | Corrupts class boundaries and can invent/destroy labels. A genuine data-integrity bug. | Nearest-neighbour resampling for labels, always. |
| **D2** | **Per-class Dice metrics have a shape bug.** All four `dice_coef_*` functions index `y_pred[:,:,:,N]` (4 indices) against `y_true[:,:,:,:,N]` (5 indices). | The minority-class Dice numbers in the original logs are **not trustworthy**. Any claim built on them is unsound. | Correct indexing; use MONAI `DiceMetric`. |
| **D3** | **Class imbalance is entirely unhandled.** Background ≈98% of voxels. Plain CCE lets the model coast on background + SNFH. An `alpha = [0.05, 0.25, 0.25, 0.25, 0.20]` weight vector **is defined in the script and then never used**. | This is the headline finding. Original logs show necrotic/enhancing Dice pinned near ~0.001–0.02 while accuracy reads 98.5%. | `DiceCELoss(include_background=False)` + class-balanced patch sampling. |
| **D4** | **Checkpoints selected on `val_accuracy`.** With 98% background, accuracy is a degenerate metric. | The "best" saved model is not the best segmenter. Model selection is broken. | Select on **mean foreground Dice**. |
| **D5** | **Crude normalization.** `X / np.max(X)` — a single global max over the batch, across all three modalities. | Ignores per-modality intensity distributions; MRI has no standard units. Hurts convergence. | Per-modality z-score over non-zero voxels. |
| **D6** | **Non-uniform slice sampling.** `VOLUME_START_AT + int(j*2.5)` yields strides of 0,2,5,7,10,12… — irregular and anisotropic, with no resampling rationale. | Introduces uncontrolled geometric distortion. | Foreground crop + principled resampling, or patch-based training. |
| **D7** | **Dead code / wrong provenance.** `Y[Y==5] = 4` and the comment `# original 4 -> converted into 3` are vestigial from a BraTS-2020 tutorial where labels were {0,1,2,4}. Post-treatment BraTS labels are natively {0,1,2,3,4}. | Signals the pipeline was adapted without re-validating the label contract. | Assert the label set explicitly at load time. |
| **D8** | **Reference network has no normalization layers.** | Slower/unstable convergence. | MONAI `UNet` uses instance norm (default) on **both** tracks — reference's no-norm not reproduced; identical across tracks, not an A/B knob (§4.2). |
| **D9** | **Data pipeline is I/O-bound; the GPU starves.** `DataGenerator` re-loads four gzipped NIfTI volumes and runs 192 `cv2.resize` calls **per step, every epoch**. Nothing is cached. | The original's ~1000 s/epoch is **not a GPU benchmark — it's a gzip-decompression benchmark**. 918 steps ÷ 1000 s = 1.09 s/step at ~305 GFLOP/step ≈ **0.28 TFLOPS effective on a datacenter GPU (~2% utilisation)**. Their own `seff` confirms it: `SLURM_CPUS_ON_NODE = 2`, `CPU Efficiency: 38.51%`, `Memory Utilized: 2.94 GB`. | One-time preprocessing + `PersistentDataset` cache. **Measure both** — Track A keeps the naive loader, Track B is cached, same GPU. The speedup is a free, independently quotable result. |

---

## 4. Experimental design

### 4.1 Two tracks
- **Track A — "baseline reproduction."** Reproduces D1–D7. Expected to reproduce the pathology: high accuracy (~98%), respectable SNFH Dice, near-zero NETC/ET/RC Dice.
- **Track B — "corrected pipeline."** Fixes D1–D7. Expected to substantially recover minority-class Dice.

### 4.2 Methodological rule — hold the architecture constant

> **The network architecture must be byte-identical between Track A and Track B.** Only the data pipeline, loss, and model-selection criterion may vary.

This is the whole reason the comparison is credible. If you also "improve" the architecture (add InstanceNorm, residual units, deeper encoder) in Track B, the delta becomes confounded and the claim collapses under a single interview question ("how do you know it was the loss and not the norm layers?").

Concretely: pick **one** MONAI `UNet` configuration and use it for both tracks. D8 is therefore **noted in the write-up but not fixed** — or, if you do add normalization, add it to *both* tracks. State whichever choice you make in the README.

### 4.3 Ablation (do this if time allows — it makes the pointer much stronger)
Rather than one A→B jump, run the fixes cumulatively so you can attribute the gain:

| Run | Config | Attributes |
|---|---|---|
| A0 | Baseline (all defects) | — |
| A1 | + nearest-neighbour labels (D1) | data integrity |
| A2 | + per-modality z-score (D5) | normalization |
| B0 | + DiceCE loss & foreground-Dice selection (D3, D4) | **expected dominant effect** |
| B1 | + class-balanced patch sampling (D6) | sampling |

Even a partial ladder (A0 → B0 → B1) is far more defensible than a single before/after.

---

## 5. Repo layout

```
~/code/brats-postop-seg/
├── README.md                  # results, figures, how to reproduce
├── CLAUDE.md                  # this file (agent instructions)
├── reference/Model/           # the ORIGINAL repo — READ-ONLY, immutable source
├── kb/                        # pageindex-plus (§13)
│   ├── sources/               #   papers + cleaned copy of output.txt
│   ├── corpus/                #   ingest output (.md + img/)   [gitignored, re-derivable]
│   └── index/                 #   pageindex.json + pages.jsonl [gitignored, re-derivable]
├── wiki/                      # llm-wiki (§13) — COMMITTED, not re-derivable
│   ├── index.md
│   ├── pages/
│   ├── memory/{log,decisions,questions}.md
│   └── .wiki/{config,graph,source_manifest,audit}.json
├── pyproject.toml             # uv-managed
├── uv.lock                    # committed
├── .gitignore                 # excludes data/, cache/, runs/
├── configs/
│   ├── base.yaml
│   ├── track_a.yaml
│   └── track_b.yaml
├── src/brats/
│   ├── __init__.py
│   ├── paths.py               # path resolution + the /mnt/ guard
│   ├── data.py                # dataset construction, splits
│   ├── transforms.py          # Track A vs Track B transform chains
│   ├── model.py               # single UNet factory, shared by both tracks
│   ├── losses.py
│   ├── metrics.py             # per-class Dice w/ NaN handling
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   ├── 00_verify_env.py
│   ├── 01_fetch_subsample.py
│   ├── 02_build_cache.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_figures.py
├── reports/
│   ├── results.md             # the numbers table
│   └── figures/
└── tests/
    ├── test_label_contract.py
    ├── test_transforms.py      # asserts NN interp for labels
    └── test_metrics.py         # asserts Dice correctness on synthetic data
```

**Data lives outside the repo:** `~/brats/{raw,cache,runs}`. Never commit data or checkpoints.

**Commit rule:** commit what cannot be regenerated. `wiki/` is hand-compiled knowledge → commit. `kb/corpus/` and `kb/index/` are derived from immutable sources by an idempotent ingester → gitignore.

---

## 6. Technical specification

### 6.1 Data acquisition — **Kaggle is the primary route (ungated)**

> **Revised 2026-07-15.** An earlier version of this plan treated Synapse registration as the only route and put it on the critical path. That was wrong. A Kaggle mirror of the post-treatment cohort exists and removes the gating risk entirely.

**Primary source:**
`kaggle.com/datasets/i212385nomanarif/2024-brats-glioma`
- Dataset metadata reads `Post_Treatment_Adult_Glioma` — **the correct cohort, with the RC class.**
- ~700 3D scans, 4 modalities + mask, NIfTI, `BraTS-GLI-XXXXX-XXX` folder convention — exactly what the source script expects.
- Requires only a free Kaggle account + API token. **No approval wait.**

**Backup mirrors** — emergency use only; see §6.1.2 before touching any of them:
- `kaggle.com/datasets/dschettler8845/brats-2021-task1` — BraTS 2021, 1,251 cases
- `kaggle.com/datasets/shakilrana/brats-2023-adult-glioma` — BraTS 2023, pre-op
- `kaggle.com/datasets/luumsk/asnr-miccai-brats-2023-gli-challenge-training-data` — BraTS 2023 GLI
- `kaggle.com/datasets/awsaf49/brats20-dataset-training-validation` — BraTS 2020, 369 train (+125 val **with no ground truth** — do not mistake these for usable data)
- Medical Segmentation Decathlon `Task01_BrainTumour` — ungated, 484 cases

**Every one of these lacks the RC class.** Falling back to any of them forfeits the post-treatment angle — the single thing that distinguishes this project from the many hundreds of pre-op BraTS repos on GitHub. The pipeline and defects D1–D5 survive; the novelty does not. Treat as a genuine last resort.

#### 6.1.1 Provenance discipline — MANDATORY

This is a **community re-upload**, not the official distribution. For a placement project subject to CDPO scrutiny, provenance must be handled honestly:

1. **Cite the source, not the mirror.** Credit the BraTS 2024 challenge paper (arXiv:2405.18368, de Verdier et al.) and the data contributors. "I got it from Kaggle" is not a data provenance statement.
2. **Do not claim to have used "the full BraTS 2024 dataset."** The mirror holds ~700 scans against an official training set of ~1,350. It is a **partial** mirror. State the actual case count you used, from your own `splits.json`.
3. **Verify the label contract on arrival** — do not trust the re-upload. Assert `set(labels) ⊆ {0,1,2,3,4}` across every case and print the observed label histogram. If you see label 5, or a max of 4 with 3 absent, you have a *different* BraTS vintage than you think and §3/D7 applies.
4. **Verify modality completeness** — confirm all of `t1n/t1c/t2w/t2f/seg` exist per case; re-uploads commonly drop files. Log and exclude any incomplete case.
5. **Record the dataset version/hash** in `reports/` so the run is reproducible.

**Subsample to ~200 cases** from the ~700. The full set is pointless on a laptop; 200 with a fixed seeded split demonstrates the method and keeps epochs tractable. Record exact case IDs in `reports/splits.json`.

**Split:** 70/15/15 train/val/test, seeded, **stratified by presence of class 4 (RC)** — otherwise a random split may leave the test set with almost no resection cavities and class-4 Dice becomes noise.

#### 6.1.2 BraTS vintages — the label contract is NOT stable across years

This matters more than it looks, because **it is the origin of defect D7.**

| Vintage | Label set | Foreground classes | RC? |
|---|---|---|---|
| BraTS 2018 / 2020 / 2021 | `{0, 1, 2, 4}` — **no label 3** | NCR/NET, ED, ET | No |
| BraTS 2023 GLI | *verify on arrival — do not assume* | — | No |
| **BraTS 2024 post-treatment (ours)** | `{0, 1, 2, 3, 4}` — contiguous | NETC, SNFH, ET, RC | **Yes** |

Because 2020/2021 volumes skip label 3, every tutorial for those years does `seg[seg == 4] = 3` to make the classes contiguous before one-hot encoding. **That is precisely where the source script's `# original 4 -> converted into 3` comment and its orphaned `Y[Y==5] = 4` line come from.** The author adapted a BraTS-2020-era notebook onto 2024 post-treatment data without re-validating the label contract. On 2024 data the remap is a no-op and the comment is simply wrong — the classes were already contiguous, and 3=ET / 4=RC natively.

The `SEGMENT_CLASSES` dict in the original *happens* to be correct for 2024. The remap code around it is dead. This is diagnostic, not fatal — but it's the thread that, when pulled, reveals the pipeline was never audited end-to-end. **Use this as the narrative entry point in the write-up.**

**Rules that follow:**
1. The label-contract assertion (§6.1.1 item 3) must be **vintage-aware** and must fail loudly, not coerce. If it sees `{0,1,2,4}`, you are holding a pre-op dataset and must stop.
2. **Never mix vintages in one experiment.** Ever.
3. **Do not smoke-test the pipeline on an older vintage.** See guardrail #11.
4. **Do not benchmark against pre-op published numbers.** See §13.3.

### 6.2 Caching
Use MONAI `PersistentDataset` with `cache_dir=~/brats/cache` (ext4). It caches the deterministic head of the transform chain to disk, so random augmentation still varies per epoch. This is the single biggest speedup vs. the original, which re-loaded and re-resized every NIfTI **every epoch**.

Do **not** use `CacheDataset` (in-RAM) for whole volumes — 200 cases will not fit comfortably.

### 6.3 Transforms

**Track B (corrected):**
```
LoadImaged(keys=["image", "label"])          # image = 3 stacked modalities
EnsureChannelFirstd
Orientationd(axcodes="RAS")
# --- assert label set ⊆ {0,1,2,3,4} here (D7) ---
NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)   # D5 fix
CropForegroundd(keys=["image", "label"], source_key="image")
# --- deterministic head ends; PersistentDataset caches to here ---
RandCropByLabelClassesd(
    keys=["image", "label"], label_key="label",
    spatial_size=(96, 96, 96), num_classes=5,
    ratios=[1, 2, 2, 2, 2],    # D3/D6 fix: oversample minority classes
    num_samples=2,
)
RandFlipd (axes 0,1,2) / RandRotate90d / RandScaleIntensityd / RandShiftIntensityd
```

**Track A (faithful):** replicate the original — bilinear label resize, global-max normalization, the `int(j*2.5)` slice stride, whole-volume `128×128×48`, no augmentation.

`RandCropByLabelClassesd` is the key move: it fixes the imbalance **at the sampling level**, which is more effective than loss weighting alone, and it conveniently makes the 8 GB VRAM budget comfortable.

### 6.4 Model — shared by both tracks
```python
from monai.networks.nets import UNet

def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=3,          # t2f, t1c, t2w — matches original's modality choice
        out_channels=5,
        channels=(16, 32, 64, 128, 256),   # matches original
        strides=(2, 2, 2, 2),
        num_res_units=0,        # plain U-Net, as in the original
        dropout=0.1,
    )
```
Log the parameter count and sanity-check it against the original's **5,645,845**. It will not match exactly (MONAI's block structure differs), but it should land in the same order of magnitude. If it's wildly off, stop and investigate.

### 6.5 Loss & metrics
- **Track A loss:** plain `torch.nn.CrossEntropyLoss()`.
- **Track B loss:** `monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)`.
- **Metric (both):** `monai.metrics.DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)`.

> **NaN trap — read this carefully.** Dice is undefined when a class is absent from *both* ground truth and prediction. Class 4 (RC) is absent in many cases, so naive averaging will silently produce NaN or, worse, a misleadingly high score if empty-empty is scored as 1.0. Use `get_not_nans=True` and aggregate per-class over only the cases where the class is actually present in the ground truth. **Report the per-class case counts alongside the Dice values.** An unqualified "RC Dice = 0.71" over 6 cases is not a result.

### 6.6 Training loop
- Batch: 2 patches of 96³ (Track B) / 1 whole volume (Track A).
- `torch.autocast("cuda", dtype=torch.bfloat16)`. **No GradScaler.**
- Optimizer: `AdamW`, lr 1e-3, cosine schedule.
- Epochs: 50 for Track B; Track A only needs enough to demonstrate the pathology (~20 is plenty — it plateaus early).
- Validate every N epochs with sliding-window inference; checkpoint on **mean foreground Dice** (D4 fix).
- Log to CSV **and** TensorBoard. Keep every log — they're artifacts (§9).
- Set all seeds; record them.

### 6.7 Inference
```python
from monai.inferers import sliding_window_inference
sliding_window_inference(inputs, roi_size=(96,96,96), sw_batch_size=1,
                         predictor=model, overlap=0.5)
```
Patch-train / sliding-window-infer is the standard MONAI idiom and resolves the train-time VRAM ceiling without compromising test-time whole-volume evaluation.

---

## 7. Phases & acceptance gates

**Do not proceed past a gate until it passes.** Report gate status explicitly.

| Phase | Work | Gate |
|---|---|---|
| **0** | `uv` env on Py3.13 in WSL2; install torch + MONAI + nibabel/matplotlib/tqdm; write `00_verify_env.py`. | `torch.cuda.is_available() == True`, GPU name printed, `nvidia-smi` works in WSL, `/mnt/` path guard fires correctly on a bad path. |
| **1** | Fetch + subsample ~200 cases to `~/brats/raw`. Seeded stratified split written to `reports/splits.json`. | Label-contract test passes on all cases; split counts printed; class-4 present in ≥15% of each split. |
| **2** | `PersistentDataset` cache built. Transform unit tests. | One batch loads; shapes/dtypes correct; `test_transforms.py` proves labels use NN interp; cache hit on 2nd epoch measurably faster. |
| **3** | **Track A** trains (~20 epochs). | Reproduces the pathology: accuracy ≳0.98, **minority-class Dice < 0.05**. If minority Dice is respectable, Track A is not faithful — go back. |
| **4** | **Track B** trains (~50 epochs). | Mean foreground Dice materially exceeds Track A. Numbers recorded in `reports/results.md` with per-class case counts. |
| **5** | Evaluate both on held-out test set; generate overlay figures; write README. | Figures render; README lets a stranger reproduce the runs from scratch. |
| **6** | *(stretch)* Ablation ladder (§4.3) and/or a second architecture. | Only if Phases 0–5 are done and the CV freeze is met. |

**Expected epoch time:** see §8.1 — analytically ~24–96 s/epoch of pure compute, so **~1–3 h for a 50-epoch Track B run once padded for I/O and validation.** Not an overnight job. **Measure it at gate 2.10 and revise if it's off** — do not just let a run go and hope.

---

## 8. Timeline — 4 working days (REVISED, authoritative)

**Rev 5 correction.** Earlier revisions said Track B was an overnight job and budgeted 2–4 days for it. **That was wrong.** It anchored on the original's ~1000 s/epoch, which is an I/O artefact (D9), not a compute cost.

### 8.1 Measured compute budget

Track B is ~96 TFLOP of math per epoch (140 cases × 2 patches of 96³, ~343 GFLOP/patch incl. backward):

| Effective throughput | s/epoch | 50 epochs |
|---|---|---|
| 1 TFLOPS (pessimistic floor) | 96 s | **80 min** |
| 2 TFLOPS | 48 s | 40 min |
| 4 TFLOPS (realistic) | 24 s | 20 min |

Pad 2–4× for data loading, sliding-window validation, and laptop thermal throttling → **~1–3 h for a 50-epoch Track B run.** Track A (~10 epochs, plateaus early) is well under an hour of compute — though if it keeps the naive loader for D9 fidelity, wall-clock will be dominated by gzip, which is the point.

**Training is not the constraint. Your attention is.** This is confirmed at gate 2.10 or the plan changes.

### 8.2 The 4-day plan

| Day | Work | Effort |
|---|---|---|
| **1** | **Start the Kaggle download first — before anything else.** Then: env (1.1), repo scaffold (1.2), **KB ingest + wiki init (§13) while the download runs**, pipeline build (2.1–2.7). | ~8h |
| **2** | Verify + split (1.4–1.5), cache + **measure epoch time (2.10 — GATE)**, smoke test (2.9), Track A run (3.1), launch Track B (3.2). | ~8h |
| **3** | Track B completes. Test-set eval (4.1), figures (4.2), `results.md` (4.3). | ~6h |
| **4** | README (4.4), CV pointers from real numbers (5.1–5.2), freeze. | ~4h |

≈26h effort. Tight but real, *because training runs while you do other things.*

### 8.3 What gets cut — decide now, not on Day 3
- **3.3 ablation ladder — GONE.** This is the designated sacrifice. The pointer becomes "A vs B", not "attributed across five runs". Accept it.
- Track A: 10 epochs, not 20.
- Track B: 30–40 epochs, not 50.
- One overlay figure, not a gallery.
- Drop cosmetic unit tests.

### 8.4 What must NOT be cut, at any time pressure
- **1.4 label-contract verification.** If the data is wrong, every number downstream is wrong.
- **2.6 metrics + NaN handling.** A broken metric invalidates the entire claim — that is literally defect D2 in the original.
- **2.9 synthetic smoke test.** Saves more time than it costs, every time.
- **Per-class case counts in `results.md`.** Non-negotiable (§6.5).

> **The download is dead time — spend it on §13.** The ingest and wiki init are off the critical path precisely because they run while bytes arrive. This is the only reason the knowledge layer fits in 4 days at all.

### 8.5 Biggest risk in a 4-day window
**The download.** ~700 cases × 5 volumes ≈ 3,500 gzipped NIfTIs ≈ **10–25 GB**. On decent broadband that's ~15–90 min; on a bad night it's the whole evening. It is the only task you cannot compress by working harder, and it has no float in a 4-day plan. **Start it before you install a single package.**

Secondary risk: no buffer. If a run dies at 2am on Day 2, Day 3 absorbs it and figures get thin. Accept degraded scope over a missed freeze.

### 8.6 Freeze-day minimum viable artifact
Track A + Track B each trained once, per-class Dice table with case counts, one overlay figure, public repo with logs. Sufficient for a defensible 3-pointer entry. Everything else is upside.

---

## 9. Artifacts checklist (CDPO compliance)

CDPO rules for the Additional Projects section are strict: **personal projects require full artifacts**, and pointers must be defensible. Preserve, from day one:

- [ ] Git repo with **honest incremental history** (not one squashed "initial commit").
- [ ] `uv.lock` committed — proves environment reproducibility.
- [ ] All training logs (CSV + TensorBoard event files) for **every** run, including failed ones.
- [ ] `reports/results.md` — per-class Dice for Track A and Track B, with **case counts per class** and seeds.
- [ ] `reports/splits.json` — exact case IDs per split.
- [ ] Overlay figures: prediction vs. ground truth, axial slices, both tracks side by side.
- [ ] README that a stranger can follow to reproduce.
- [ ] `wiki/` committed — pages, graph, and memory log (§13).
- [ ] **Green `wiki_audit.py` run** — no stale pages, no broken links, no uncovered sources. It exits 1 on error, so this is a checkable claim, not a vibe.
- [ ] HTML report generated *from the index*, every number recomputed in Python, every external claim cited to a page anchor (§13.6).

**Do not draft CV pointers until Phase 5 numbers exist.** Placeholder claims have a way of surviving into a frozen CV, and an unverifiable number in a placement viva is a catastrophic failure mode. When the numbers are real, the pointers follow the established house rules: 3 pointers, 110–115 character band, character counts verified in Python.

Provisional narrative shape (numbers TBD):
> *Reproduced a 3D U-Net BraTS post-treatment segmentation pipeline; diagnosed label-resampling, metric-indexing and class-imbalance defects; lifted mean foreground Dice from **X** to **Y** on a held-out test set.*

---

## 10. Guardrails — things the agent must NOT do

1. **Do not put data, cache, or runs on `/mnt/c`.** (§1.3)
2. **Do not change the architecture between Track A and Track B.** (§4.2)
3. **Do not add a `GradScaler`.** bf16 on Ada doesn't need it. (§1.2)
4. **Do not report Dice without per-class case counts.** (§6.5)
5. **Do not "fix" Track A.** Its job is to fail in a specific, documented way. A Track A that performs well means the reproduction is wrong.
6. **Do not attempt the Attention/Res/MPU variants** before Phase 5 is complete.
7. **Do not reuse the original repo's `.keras`/`.weights.h5` checkpoints.** They were saved under a TF 2.9-era / Python 3.7 stack and are (a) unlikely to load and (b) irrelevant — we're reimplementing in PyTorch. Retraining clean is what makes the pointer defensible.
8. **Do not silently increase batch size or resolution to "use the GPU better."** The 8 GB budget is intentional and tight.
9. **Do not fabricate or estimate results.** Every number in `reports/` must come from a run whose log is committed.
10. **Do not squash the git history** to make it look tidy.
11. **Do not smoke-test on a different BraTS vintage.** Tempting — BraTS 2020 is small and downloads fast. But it forks the label handling (`num_classes` 4 vs 5, different remap), and a label-contract fork is *exactly the defect class this project claims to have found* (D7). Introducing one to test the code that detects it is self-defeating. Use `monai.data.synthetic.create_test_image_3d` for pipeline smoke tests — synthetic volumes, no download, no label fork, runs in seconds.
12. **Never re-read raw sources to "get oriented" after a compaction.** Run the §13.3 ritual instead: `wiki/index.md` → last 3 log entries → `audit.json` counts. Re-reading sources is the failure mode the wiki exists to prevent, and it silently burns the budget.
13. **Never modify anything under `reference/Model/`.** Raw sources are immutable; the wiki is derived, the sources are truth. Pre-clean a *copy* of `output.txt` for ingest (§13.1).
14. **Do not chase pre-op leaderboard numbers.** BraTS 2020/2021 papers report ET/TC/WT Dice ≈0.7–0.9. That is a *different task* (pre-operative), on *different merged regions*, with *different metrics*. Post-treatment is harder and RC is new. If a Kaggle notebook shows 0.85 and your run gives 0.55, that is **not evidence of a bug**. Chasing that phantom will lead you to "fix" a working pipeline into a broken one. Your only valid comparison is Track A vs Track B.

---

## 11. OOM playbook

If CUDA OOM occurs, apply **in this order** and record which step was needed:

1. Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set.
2. Confirm bf16 autocast is actually active (not silently fp32).
3. Reduce `num_samples` in `RandCropByLabelClassesd` (2 → 1).
4. Reduce patch size 96³ → 80³ → 64³.
5. Reduce `sw_batch_size` to 1 during validation (validation OOM is common and often mistaken for training OOM).
6. Gradient checkpointing.
7. **Only as a last resort:** shrink encoder channels — and if you do, you must retrain Track A with the same change (§4.2).

Close Chrome/Slack first — on a mobile GPU, the Windows desktop compositor is already holding VRAM.

---

## 12. Work breakdown structure

Effort = hands-on hours. Duration = elapsed wall-clock. **These diverge sharply here and that divergence is the schedule's key feature** (see §13.2).

| WBS | Package | Effort | Duration | Pred | Gate |
|---|---|---|---|---|---|
| **1.0** | **Setup & data** | | | | |
| 1.1 | uv env + torch/MONAI, GPU verify | 2h | 0.5d | — | `cuda.is_available()` |
| 1.2 | Repo scaffold, git init, configs | 2h | 0.5d | 1.1 | — |
| 1.3 | Kaggle auth + download ~700 cases (~10–25 GB) | 0.2h | 0.25–1d | — | lands on ext4 |
| 1.4 | Integrity + label-contract verify | 2h | 0.5d | 1.3 | labels ⊆ {0…4} |
| 1.5 | Subsample 200 + stratified split | 2h | 0.5d | 1.4 | RC ≥15% per split |
| 1.6 | KB ingest: papers + cleaned `output.txt` + `scan_code.py` | 1h | 0.3d | 1.2 | index builds |
| 1.7 | `wiki_init.py` + 6 pages + memory files | 1.5h | 0.3d | 1.6 | audit green |
| **2.0** | **Pipeline** | | | | |
| 2.1 | Dataset + loaders | 3h | 0.5d | 1.2 | — |
| 2.2 | Track A transforms (faithful) | 2h | 0.5d | 2.1 | — |
| 2.3 | Track B transforms (corrected) | 3h | 0.5d | 2.1 | NN-interp test |
| 2.4 | Model factory (shared UNet) | 1h | 0.2d | 1.2 | param count sane |
| 2.5 | Losses (CE / DiceCE) | 1h | 0.2d | 2.4 | — |
| 2.6 | Metrics + NaN handling | 3h | 0.5d | 2.4 | synthetic unit test |
| 2.7 | Train harness (bf16, ckpt, logging) | 4h | 1d | 2.5, 2.6 | — |
| 2.8 | Sliding-window inference | 2h | 0.3d | 2.7 | — |
| 2.9 | Synthetic smoke test | 2h | 0.3d | 2.7 | end-to-end runs |
| 2.10 | Build cache + **measure epoch time** | 1h | 0.5d | 1.5, 2.9 | **schedule viable** |
| **3.0** | **Experiments** | | | | |
| 3.1 | Track A run (~10–20 ep) | 0.5h | <1h compute | 2.10 | minority Dice <0.05 |
| 3.2 | Track B run (~30–50 ep) | 0.5h | 1–3h | 3.1 | Dice ≫ Track A |
| 3.3 | Ablation A1/A2/B1 | 1h | 2d | 3.2 | *optional* |
| **4.0** | **Eval & artifacts** | | | | |
| 4.1 | Test-set eval, both tracks | 2h | 0.5d | 3.2 | — |
| 4.2 | Overlay figures | 3h | 0.5d | 4.1 | — |
| 4.3 | `results.md` + case counts | 2h | 0.3d | 4.1 | — |
| 4.4 | README | 3h | 0.5d | 4.3 | stranger can repro |
| 4.5 | HTML report from index (§13.6) | 2h | 0.3d | 4.3 | numbers recomputed |
| 4.6 | Wiki semantic lint + final audit | 0.5h | 0.1d | 4.5 | `wiki_audit.py` exits 0 |
| **5.0** | **CV integration** | | | | |
| 5.1 | Draft 3 pointers from real numbers | 2h | 0.3d | 4.3 | — |
| 5.2 | Char-count verify (110–115 band) | 0.5h | 0.1d | 5.1 | Python-verified |
| 5.3 | Freeze | — | — | 5.2 | **Jul 27** |

### 12.1 Critical path
`1.1 → 1.2 → 2.1 → 2.3 → 2.7 → 2.10 → 3.1 → 3.2 → 4.1 → 4.3 → 5.1 → 5.2`

**1.6/1.7 (knowledge layer) are NOT on the critical path** — they run in the download's dead time (§8.2). If they drift onto it, apply the §13.7 tripwire.

**1.3 (download) is NOT on the critical path** — it runs unattended while the pipeline is built. This is why the Synapse gate mattered and the Kaggle route doesn't.

### 12.2 Effort vs. duration — the scheduling insight
**Total effort ≈ 48h at full scope; ≈29h at the 4-day cut scope (§8.2–8.3), including ~3h for the knowledge layer (§13).** Machine time is unattended: 1.3 is a background download, and 3.1/3.2 consume ~1h of attention across ~1–3h of wall-clock each (§8.1).

Treating GPU hours as "work" makes this project look impossible when it isn't. Launch a run and go do Trilytics; it costs wall-clock, not attention. This is what makes 4 days viable at all.

### 12.3 Float and the sacrifice
- **3.3 (ablation) has zero float** against the freeze and is the designated sacrifice. Cut it first, without agonising.
- Everything else is near-critical.
- **2.10 is the real decision point.** If measured epoch time overshoots estimate, cut scope *there* — cases 200→120, or patch 96³→64³. Not on Day 9. A gate that doesn't change behaviour isn't a gate.

---

## 13. Knowledge management — pageindex-plus + llm-wiki

**Why this is here and not decoration.** Over ~26h of Claude Code work, context **will** compact, repeatedly. Without a memory layer, every compaction costs 10–20 min of re-orientation and risks silent drift from this spec. Six compactions ≈ 1–2h lost — roughly what the wiki costs to set up. It pays for itself on memory alone; grounding and audit are upside.

The two skills compose by design: **pageindex-plus is the ingestion layer, llm-wiki is the knowledge layer.**

```
sources/ ──ingest_notes.py──▶ kb/corpus/ ──build_pageindex.py──▶ kb/index/
   │                                                                  │
   └── scan_code.py ─────────▶ code_map.md                    tree-search / citations
                                    │                                 │
                                    └──────────▶ wiki/ (compiled pages + graph + memory + audit)
                                                          │
                                                  HTML report (html_notes_build.md)
```

### 13.1 What goes in the corpus (pageindex-plus)

| Source | Why | Ingest path |
|---|---|---|
| `reference/Model/output.txt` (14 MB training log) | **The evidence for D2, D3, D9.** Contains the near-zero minority Dice, the `seff` output (`CPU Efficiency: 38.51%`, 2 cores), per-epoch timings. `.txt` is a supported type. | `ingest_notes.py` |
| `reference/Model/**/*.py` (5 architectures) | Call graph + structure across U-Net / 3D-U-Net / Res-U-Net / Attention-U-Net / MPU-Net | `scan_code.py --root reference/Model` |
| BraTS 2024 challenge paper (arXiv:2405.18368) | Label semantics, annotation protocol, metric definitions. **Figures matter** (Fig 4 = common segmentation errors). | `ingest_notes.py` (PDF) |
| Kim et al. (arXiv:2409.08143) | The per-class SOTA numbers in §2.3 — cite by page, don't retype | `ingest_notes.py` (PDF) |

> **Practical trap:** `output.txt` is a raw terminal capture with `\r` line endings, not `\n`. Pre-clean it (`tr '\r' '\n'`) into a copy before ingesting, or it lands as one giant line and the page anchors become useless. **Clean the copy, never the original** — raw sources are immutable.

Keep the corpus tight. Do **not** ingest MONAI docs — that's what the web is for.

### 13.2 What goes in the wiki (llm-wiki)

Minimal page set for a 4-day sprint. Resist writing more.

| Page | Content |
|---|---|
| `index.md` | Map of content. **Every page needs a one-line entry** — the audit flags pages missing from it. |
| `pages/defect-inventory.md` | D1–D9, each with `sources:` anchors into the corpus. This is the spine of the whole project. |
| `pages/experimental-design.md` | Track A/B, the hold-architecture-constant rule (§4.2) |
| `pages/data-provenance.md` | Kaggle mirror, vintage table, label contract (§6.1) |
| `pages/environment.md` | Hardware, WSL/ext4 rule, versions |
| `pages/results.md` | The numbers — updated in place as runs complete |
| `memory/log.md` | Session journal (append via `wiki_log.py`) |
| `memory/decisions.md` | Every design decision + rationale |
| `memory/questions.md` | Open questions (e.g. "is the ~700-case mirror a random subset?") |

```bash
python SKILL_DIR/scripts/wiki_init.py wiki --name "BraTS Post-Treatment Reproduction" \
    --source-root ../src --source-root ../reference/Model --source-root ../reports \
    --code-glob "**/*.py"
```

**Read `references/schema.md` before writing the first page** — the frontmatter parser reports what it can't understand rather than guessing.

### 13.3 The compaction-recovery ritual — MANDATORY

On every new session or after any compaction, read **in this order**:
1. `wiki/index.md`
2. last ~3 entries of `wiki/memory/log.md`
3. the counts block of `wiki/.wiki/audit.json`

**Do NOT re-read raw sources to "get oriented."** That is precisely what the wiki exists to prevent. If the ritual doesn't restore enough state, the fix is a better log entry, not more source reading.

### 13.4 End every session
```bash
python SKILL_DIR/scripts/wiki_log.py wiki log "what changed; what's next"
python SKILL_DIR/scripts/wiki_graph.py wiki --print        # fix unresolved links NOW
python SKILL_DIR/scripts/wiki_audit.py wiki                # must be error-free
python SKILL_DIR/scripts/wiki_audit.py wiki --update-manifest   # only if you compiled
```
An unresolved link is either a typo or a page that wants to exist. An answer that isn't filed back is knowledge lost.

### 13.5 Audit as the freeze gate

`wiki_audit.py` **exits 1 on errors**. Wire it into the Day-4 checklist:
- **Stale pages** — source hashes vs manifest. Over 4 days of code churn this is the one that earns its keep: change `train.py`, and the page describing it is flagged. This is how the README avoids describing a pipeline that no longer exists.
- **Uncovered sources** — a source root nothing compiled from.
- **Broken links / orphans / duplicates / TODOs.**

A green audit is a checkable claim that the artifact is internally consistent — exactly what §9 needs.

### 13.6 Report generation

Use the `references/html_notes_build.md` recipe. Render **from the index**, not by reloading sources. Three rules from that recipe map directly onto this project's defensibility problem:

1. **Recompute every numerical example in Python before writing it. Never hand-type a derived number.** This is the same discipline as §6.5's per-class case counts and §9's "every number comes from a committed log."
2. **Figures: regenerate cleanly in matplotlib where the data is recoverable; embed the extracted PNG otherwise.** Your Dice curves and overlays are generated from your own logs → regenerate. The BraTS paper's Fig 4 → embed.
3. **Cite the source page/slide for every claim.** "The reference pipeline ran at 38.5% CPU efficiency on 2 cores" becomes citable to an anchor, not to memory.

Pairs with the `html-notes-academic` skill for typography and MathJax.

### 13.7 Cost control — the tripwire

Added cost ≈ **2–3h on a 26h budget (~10%)**. Acceptable, but adding tooling during a crunch is a classic way to lose a day to yak-shaving.

**Cut list, in order, if it isn't paying:**
1. `caption_figures.py` (LLM-vision hook) — **skip by default in 4 days.** Optional even at full scope.
2. Extra wiki pages beyond the 6 above.
3. The semantic lint pass — do it once on Day 4, not "every few sessions."
4. pageindex-plus entirely → fall back to grep over `reference/Model/`.

> **What you never cut: `wiki_log.py`.** The memory log is the handoff mechanism between Claude Code sessions. If everything else goes, keep the log. It is the piece that actually pays.

**Tripwire:** if the knowledge layer hasn't paid for itself by end of Day 1, drop to memory-only (log + decisions + index) and move on. Do not debug a wiki while a CV freeze is running.

---

## 14. Appendix

### 14.1 Quick commands
```bash
# Phase 0
cd ~/code && uv init brats-postop-seg && cd brats-postop-seg
uv add torch monai nibabel numpy matplotlib tqdm pyyaml tensorboard einops
uv add --dev kagglehub          # or the `kaggle` CLI
uv run python scripts/00_verify_env.py

# --- DATA: start this FIRST, it runs for hours ---
# Kaggle API token: kaggle.com -> Settings -> API -> Create New Token
#   -> place kaggle.json at ~/.config/kaggle/kaggle.json (chmod 600)
# NOTE: target must be on ext4, NEVER /mnt/c  (§1.3)
mkdir -p ~/brats/raw
uv run kaggle datasets download -d i212385nomanarif/2024-brats-glioma \
        -p ~/brats/raw --unzip
# Sanity-check what actually landed before trusting it (§6.1.1):
ls ~/brats/raw | head; ls ~/brats/raw | wc -l

# --- Knowledge layer (§13) — run WHILE the download is going ---
mkdir -p kb/sources
tr '\r' '\n' < reference/Model/output.txt > kb/sources/original_training_log.txt   # \r trap
# drop the two arXiv PDFs into kb/sources/ as well
uv run PI_SKILL/scripts/ingest_notes.py kb/sources kb/corpus
uv run PI_SKILL/scripts/scan_code.py --root reference/Model --out kb/corpus/code_map.md
uv run PI_SKILL/scripts/build_pageindex.py kb/corpus kb/index --name "BraTS reproduction KB"
uv run PI_SKILL/scripts/pageindex_query.py kb/index --search "seff CPU efficiency"

python WIKI_SKILL/scripts/wiki_init.py wiki --name "BraTS Post-Treatment Reproduction" \
    --source-root ../src --source-root ../reference/Model --source-root ../reports \
    --code-glob "**/*.py"
# ... compile the 6 pages ...
python WIKI_SKILL/scripts/wiki_graph.py wiki --print
python WIKI_SKILL/scripts/wiki_audit.py wiki --update-manifest

# Phases
uv run python scripts/01_fetch_subsample.py --n 200 --seed 42
uv run python scripts/02_build_cache.py
uv run python scripts/03_train.py --config configs/track_a.yaml
uv run python scripts/03_train.py --config configs/track_b.yaml
uv run python scripts/04_evaluate.py --run runs/track_b
uv run python scripts/05_figures.py
```

### 14.2 Case file naming (BraTS-GLI)
```
BraTS-GLI-XXXXX-XXX/
├── BraTS-GLI-XXXXX-XXX-t1n.nii.gz    # native T1        (unused — original drops it)
├── BraTS-GLI-XXXXX-XXX-t1c.nii.gz    # post-contrast T1 (channel 1)
├── BraTS-GLI-XXXXX-XXX-t2w.nii.gz    # T2              (channel 2)
├── BraTS-GLI-XXXXX-XXX-t2f.nii.gz    # T2-FLAIR        (channel 0)
└── BraTS-GLI-XXXXX-XXX-seg.nii.gz    # labels {0,1,2,3,4}
```

### 14.3 Official challenge metrics (optional upgrade)

The BraTS 2024 challenge does **not** score on plain voxel-wise Dice. It uses **lesion-wise Dice** and **lesion-wise Hausdorff-95**, computed per connected lesion. Reference implementation: `github.com/rachitsaluja/BraTS-2024-Metrics`.

Plain voxel Dice is **fine and correct for our A/B** — the comparison is internally consistent and that's what matters for the claim. But be precise in the write-up: say "voxel-wise Dice," not "BraTS Dice," and never compare your numbers to published challenge leaderboards, because they are not the same metric. Conflating them is exactly the kind of error an interviewer with domain knowledge will catch.

**Two separate incomparabilities — keep them straight:**
1. **Metric:** ours is voxel-wise; the challenge is lesion-wise. Different numbers on identical predictions.
2. **Task:** pre-op BraTS (2018–2023) reports Dice over *merged hierarchical regions* — ET / TC (tumor core) / WT (whole tumor) — not the flat per-class scheme we use. Post-treatment 2024 is a harder task with an additional class.

So a pre-op paper's "WT Dice 0.90" and our "SNFH Dice 0.62" are not on speaking terms. Say so explicitly in the README rather than leaving a reader to assume underperformance.

If Phase 6 is reached, adding lesion-wise Dice is a strong, cheap credibility upgrade.

### 14.4 Open questions to resolve during the build
- Does adding `t1n` as a 4th channel help? (Plausible for RC delineation — chronic resection cavities track CSF signal on T1, so native T1 carries real information the 3-channel setup discards.) **Park this** — it's a Phase 6 experiment, and adding it early would confound the A/B.
- Is 200 cases enough for stable class-4 Dice? Check per-class case counts after the split; if RC appears in <20 test cases, say so explicitly rather than reporting a fragile number.
- Does the Kaggle mirror's ~700 cases match the official cohort, or is it a specific subset? Worth one paragraph in the README. If it's a non-random subset, that's a sampling caveat worth naming honestly.

---

*End of spec. If any instruction here proves impossible or wrong on contact with reality, stop and report rather than improvising around it.*
