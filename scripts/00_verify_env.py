#!/usr/bin/env python
"""Phase 0 gate (CLAUDE.md §7).

Verifies the environment before any training and **exits non-zero** if a hard
requirement fails:
  - torch sees CUDA and the right GPU (bf16-capable Ada);
  - nvidia-smi is reachable in WSL;
  - PYTORCH_CUDA_ALLOC_CONF is set (§1.4);
  - MONAI imports;
  - the §1.3 filesystem guard passes on the data dirs AND correctly *fires* on /mnt/c, /mnt/d.

Run:  uv run python scripts/00_verify_env.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Make the src package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brats import paths  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, good: str, bad: str) -> None:
    print(("  ✓ " if ok else "  ✗ ") + (good if ok else bad))
    if not ok:
        FAILURES.append(bad)


print("== GPU / CUDA ==")
import torch  # noqa: E402

print(f"  torch {torch.__version__}")
cuda_ok = torch.cuda.is_available()
check(cuda_ok, "CUDA available", "CUDA NOT available — torch is likely CPU-only; fix before proceeding")
if cuda_ok:
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  device: {props.name}  (sm_{cap[0]}{cap[1]}, {props.total_memory / 1024**3:.1f} GiB)")
    check(
        torch.cuda.is_bf16_supported(),
        "bf16 supported (Ada) — no GradScaler needed (§1.2)",
        "bf16 NOT supported — unexpected on Ada sm_89",
    )

print("== nvidia-smi ==")
check(shutil.which("nvidia-smi") is not None, "nvidia-smi on PATH", "nvidia-smi missing in WSL")

print("== env vars (§1.4) ==")
alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
check(
    "expandable_segments:True" in alloc,
    f"PYTORCH_CUDA_ALLOC_CONF={alloc}",
    f"PYTORCH_CUDA_ALLOC_CONF should contain expandable_segments:True (got '{alloc}')",
)

print("== MONAI ==")
try:
    import monai  # noqa: E402

    print(f"  MONAI {monai.__version__}")
except Exception as exc:  # pragma: no cover
    FAILURES.append(f"MONAI import failed: {exc}")
    print(f"  ✗ MONAI import failed: {exc}")

print("== filesystem rule §1.3 (native ext4, no 9p) ==")
for d in (paths.DATA_ROOT, paths.RAW_DIR, paths.CACHE_DIR, paths.RUNS_DIR):
    try:
        paths.assert_native_storage(d)
        print(f"  ✓ {d} -> {Path(d).resolve()}  ({paths.mount_fstype(d)})")
    except Exception as exc:
        check(False, "", f"data path on Windows/9p fs: {exc}")

# Self-test: the guard MUST reject Windows drives.
for bad in ("/mnt/c/tmp", "/mnt/d/whatever"):
    fired = False
    try:
        paths.assert_native_storage(Path(bad))
    except Exception:
        fired = True
    check(fired, f"guard correctly rejects {bad}", f"guard FAILED to reject {bad} — §1.3 unenforced")

print()
if FAILURES:
    print(f"GATE 0 FAILED — {len(FAILURES)} issue(s):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("GATE 0 PASSED — environment ready.")
