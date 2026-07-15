# BraTS Post-Treatment Glioma Segmentation — PyTorch + MONAI

Reimplementation of a 3D U-Net pipeline for **5-class post-treatment glioma
segmentation** (BraTS-GLI 2024) as a controlled **A/B experiment**: a faithful
reproduction of a TensorFlow/Keras research pipeline (**Track A**, defects D1–D9
preserved) versus a corrected pipeline (**Track B**). The deliverable is the
**A→B delta** in minority-class Dice, not a leaderboard score.

See **[CLAUDE.md](CLAUDE.md)** for the authoritative build spec (mission, constraints,
defect inventory, experimental design, phase gates, guardrails).

## Status
Phase 0 (environment) complete. Pipeline, data, and experiments TBD.
**Do not draft result claims until Phase 5** — every number must come from a committed log (§9).

## Environment
- `uv` on Python 3.13, PyTorch (bf16, no GradScaler), MONAI 1.6.x, WSL2.
- Data/cache/runs on native ext4 (`~/brats` → `/mnt/wsl/brats`), never on a Windows drive (§1.3).
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (§1.4).

```bash
uv run python scripts/00_verify_env.py     # Phase 0 gate
```

## Data
Subsample of `kaggle.com/datasets/i212385nomanarif/2024-brats-glioma` — a community
mirror of the BraTS 2024 post-treatment cohort. **Cite the source, not the mirror:**
BraTS 2024 challenge (de Verdier et al., arXiv:2405.18368). Partial mirror (~700 of ~1350);
actual case count recorded in `reports/splits.json`.

## Results
> Per-class voxel-wise Dice (Track A vs Track B), **with per-class case counts** (§6.5). TBD.
