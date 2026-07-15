"""Path resolution + the §1.3 filesystem guard.

CLAUDE.md §1.3 (NON-NEGOTIABLE): the dataset, transform cache, and run outputs must
never live on a Windows drive reached over 9p/DrvFs (``/mnt/c``, ``/mnt/d``, …).
9p turns the pipeline's thousands of small NIfTI reads per epoch into a 5–10× I/O
tax — which is literally defect **D9**. Data must sit on native ext4.

On this machine ``~/brats`` is a symlink to ``/mnt/wsl/brats`` — a D:-backed ext4
virtual disk mounted as a native block device. Note that path lives *under* ``/mnt``
yet is genuinely native ext4, so the guard keys on the **filesystem type**, not the
``/mnt/`` prefix (with a defense-in-depth check for ``/mnt/<drive-letter>``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Repo root: this file is src/brats/paths.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data lives OUTSIDE the repo, on the fast native-ext4 disk. Overridable for tests/CI.
DATA_ROOT = Path(os.environ.get("BRATS_DATA_ROOT", str(Path.home() / "brats"))).expanduser()
RAW_DIR = DATA_ROOT / "raw"
CACHE_DIR = DATA_ROOT / "cache"
RUNS_DIR = DATA_ROOT / "runs"

# In-repo, committed outputs
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SPLITS_JSON = REPORTS_DIR / "splits.json"

# Filesystem types that mean "Windows drive over 9p/DrvFs" — exactly what §1.3 forbids.
_WINDOWS_FS_TYPES = {"9p", "v9fs", "drvfs", "cifs", "smbfs"}


def _unescape_mountinfo(field: str) -> str:
    r"""Decode mountinfo octal escapes (\040 space, \011 tab, \012 newline, \134 backslash)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def mount_fstype(path: os.PathLike | str) -> str:
    """Filesystem type backing ``path``, via the longest matching mount in /proc/self/mountinfo."""
    target = str(Path(path).resolve())
    best_len, best_type = -1, "unknown"
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                try:
                    sep = parts.index("-")
                except ValueError:
                    continue
                # 0:mountid 1:parentid 2:maj:min 3:root 4:mountpoint … - fstype source superopts
                if len(parts) < 5 or sep + 1 >= len(parts):
                    continue
                mountpoint = _unescape_mountinfo(parts[4])
                fstype = parts[sep + 1]
                if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
                    if len(mountpoint) > best_len:
                        best_len, best_type = len(mountpoint), fstype
    except FileNotFoundError:
        pass
    return best_type


def is_windows_backed(path: os.PathLike | str) -> bool:
    """True if ``path`` resolves onto a Windows/9p filesystem (a §1.3 violation)."""
    resolved = Path(path).resolve()
    parts = resolved.parts
    # Defense in depth: /mnt/<single-letter> is WSL's Windows-drive convention.
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
        return True
    return mount_fstype(resolved) in _WINDOWS_FS_TYPES


def assert_native_storage(path: os.PathLike | str) -> None:
    """Raise if ``path`` is on a Windows/9p filesystem (CLAUDE.md §1.3, guardrail #1)."""
    p = Path(path)
    if is_windows_backed(p):
        raise RuntimeError(
            f"§1.3 VIOLATION: {p} -> {p.resolve()} is on a Windows/9p filesystem "
            f"(fstype={mount_fstype(p.resolve())}). Dataset/cache/runs must be on native ext4 "
            f"(e.g. ~/brats -> /mnt/wsl/brats). 9p makes small NIfTI reads 5–10× slower (defect D9)."
        )


def ensure_data_dirs() -> None:
    """Create raw/cache/runs after asserting they are on native storage."""
    assert_native_storage(DATA_ROOT)
    for d in (RAW_DIR, CACHE_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
