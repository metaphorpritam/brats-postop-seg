---
title: Data Provenance
type: entity
status: active
tags: [data, brats, provenance, kaggle, labels]
sources:
  - ../reports/data_provenance.json
  - ../reports/splits.json
  - ../CLAUDE.md
code: [../src/brats/data.py]
links:
  relates: [Experimental Design, Environment, Results]
  derived-from: [Defect Inventory]
---

# Data Provenance

The dataset used for both tracks of this project. It is a **community re-upload**, not the official BraTS distribution, so provenance is handled explicitly to survive CDPO scrutiny (CLAUDE.md §6.1.1).

## Source and citation

- **Acquisition mirror:** `kaggle:i212385nomanarif/2024-brats-glioma` (CC0), metadata `Post_Treatment_Adult_Glioma` — the correct cohort, carrying the RC class.
- **What to cite:** the **BraTS 2024 challenge paper** — de Verdier et al., *arXiv:2405.18368* — and the data contributors. **Cite the source, NOT the Kaggle mirror.** "I got it from Kaggle" is not a provenance statement.
- **PARTIAL mirror:** the re-upload holds **~700 scans against the official ~1,350** training set. Never claim "the full BraTS 2024 dataset." The actual case count used is drawn from `reports/splits.json`, not the mirror's headline.
- **Recorded version:** `content_sha256_16 = dce57ab5972de05e`; ~4.92 GB on disk (`total_bytes_on_disk = 4919565071`), logged in `reports/data_provenance.json` for reproducibility.

## Label contract — vintage-aware verification

The BraTS label set is **not stable across years**, and this is the origin of defect D7 (see [[Defect Inventory]]). Verification on arrival asserts `set(labels) ⊆ {0,1,2,3,4}` per case and fails loudly rather than coercing.

| Vintage | Label set | RC? |
|---|---|---|
| BraTS 2018 / 2020 / 2021 | `{0,1,2,4}` — no label 3 | No |
| BraTS 2023 GLI | verify on arrival | No |
| **BraTS 2024 post-treatment (ours)** | `{0,1,2,3,4}` contiguous | **Yes** |

Class map: 0 background, 1 NETC, 2 SNFH, 3 **ET**, 4 **RC**. **The presence of label 3 (ET) with contiguous `{0,1,2,3,4}` confirms the 2024 post-treatment vintage** — if the observed set were `{0,1,2,4}` the data would be a pre-op vintage and the run must stop. `reports/labels_summary.json` records the observed per-case label histogram and RC presence; the diagnostic never mixes vintages.

## Geometry caveat

- **On disk (mirror):** `182x218x182` — the re-upload is resampled to an **MNI-like space**.
- **Native BraTS:** `240x240x155` @ 1 mm isotropic.

This resampling is a provenance caveat, not the native geometry, and is recorded as such in `data_provenance.json`. Modalities kept: `t2f, t1c, t2w` (the `t1n` 4th modality is dropped, matching the original — see [[Experimental Design]]); label key `seg`.

## Subsample and split

- **~200 cases** subsampled from the ~700 (seed 42). The full set is pointless on an 8 GB laptop GPU (see [[Environment]]); 200 with a fixed seeded split demonstrates the method and keeps epochs tractable.
- **Split:** **140 / 30 / 30** train/val/test (a 70/15/15 ratio), **stratified by RC (class 4) presence** so the test set is not left with too few resection cavities to score.
- **RC ~83% present, in every fold:** train 116/140, val 25/30, test 25/30. Exact case IDs are frozen in `reports/splits.json` and are what downstream numbers ([[Results]]) are computed over.

Both tracks read the **identical raw NIfTIs** from this dataset (`src/brats/data.py`); only the pipeline differs, which is what makes the A→B delta attributable (see [[Experimental Design]]).
