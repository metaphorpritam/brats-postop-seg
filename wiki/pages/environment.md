---
title: Environment
type: concept
status: active
tags: [hardware, wsl2, filesystem, ext4, bf16, uv]
sources:
  - ../CLAUDE.md
  - ../src/brats/paths.py
code: [../src/brats/paths.py]
links:
  relates: [Defect Inventory, Data Provenance, Results]
  part-of: [Experimental Design]
---

# Environment

The full pipeline runs on **one 8 GB laptop GPU inside WSL2**. Every constraint below is binding, not a preference; the tight VRAM budget and the filesystem rule are the two that shape the whole design.

## Hardware

| Resource | Spec | Implication |
|---|---|---|
| GPU | RTX 4060 **Laptop** (Mobile), 8 GB VRAM, Ada `sm_89` | The binding constraint. bf16 native. No whole-volume batch > 1. |
| CPU | i7-12650H (6P + 4E, 16 threads) | DataLoader `num_workers` ≈ 4–6; do not oversubscribe. |
| OS | Windows + **WSL2** | All training runs inside WSL2. |

## Precision — bf16, no GradScaler (§1.2)

Ada (`sm_89`) supports **bf16 autocast natively**, so training uses `torch.autocast("cuda", dtype=torch.bfloat16)` with **no `GradScaler`**. bf16 has the same exponent range as fp32 and does not underflow the way fp16 does, so loss scaling is unnecessary. Adding a GradScaler is an explicit guardrail violation (guardrail #3). Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOMs.

## The filesystem rule (§1.3) — NON-NEGOTIABLE

> Never place the dataset, transform cache, or run outputs on a Windows drive reached over 9p.

WSL2 reaches Windows drives (`/mnt/c`, `/mnt/d`, …) over the **9p protocol**. This pipeline does thousands of small NIfTI/cache reads per epoch, and 9p imposes a 5–10× throughput tax — the GPU starves waiting on I/O and the symptom is easily misdiagnosed as a model problem. This is exactly defect **D9** (see [[Defect Inventory]]). Data, cache, checkpoints, and logs must sit on **native ext4**; code lives on ext4 too.

**How it is realised on this machine:** `~/brats` is a symlink to `/mnt/wsl/brats`, a **D:-backed ext4 vhdx mounted as a native block device**. Note the subtlety: this path lives *under* `/mnt/`, yet it is genuinely native ext4, not a 9p share — a naive `/mnt/` prefix check would wrongly reject it. Data-loader throughput confirms the payoff: Track A's naive per-epoch reloader runs ~23 s/epoch (I/O-bound, no cache) vs Track B's `PersistentDataset` at ~8 s (~2.8×) — see [[Results]].

Path layout (from `paths.py`): `DATA_ROOT = ~/brats` (override `BRATS_DATA_ROOT`) with `raw/`, `cache/`, `runs/` beneath it; in-repo committed outputs go under `reports/`.

## The fs guard — keys on filesystem TYPE, not path prefix

`src/brats/paths.py` enforces §1.3 at runtime rather than by convention. `mount_fstype()` reads `/proc/self/mountinfo` and returns the fstype of the **longest matching mount** for a resolved path; `is_windows_backed()` flags a path when that fstype is in `{9p, v9fs, drvfs, cifs, smbfs}`. This keys on the actual **filesystem type**, so `/mnt/wsl/brats` (ext4) passes while `/mnt/c/...` (9p) fails — the opposite of what a prefix check would do. A defense-in-depth check additionally rejects `/mnt/<single-drive-letter>/...` (WSL's Windows-drive convention). `assert_native_storage()` raises `RuntimeError` on violation and is invoked by `ensure_data_dirs()` before any raw/cache/runs directory is created.

## Software stack

- **Python 3.13**, managed by **`uv`** (no conda, pip-venv, or poetry).
- **torch 2.13 + cu130** — on Linux, PyPI hosts CUDA-enabled wheels directly (since torch 2.11), so plain `uv add torch` resolves a GPU build in WSL2; verify `torch.cuda.is_available()` before proceeding.
- **MONAI 1.6** (requires Python ≥ 3.10; 3.13 is fine).
- Shared model across both tracks: MONAI `UNet`, **1,983,069 params** (see [[Experimental Design]]).

Phase-0 gate (`00_verify_env.py`): `torch.cuda.is_available() == True`, GPU name printed, `nvidia-smi` works inside WSL, and the `/mnt/` path guard fires correctly on a bad path.
