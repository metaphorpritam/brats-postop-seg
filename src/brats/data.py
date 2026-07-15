"""Datalist / dataset / dataloader construction for both tracks (CLAUDE.md §6.1–6.3).

``build_datalist`` turns the seeded case-ID lists in ``reports/splits.json`` into the
MONAI-style ``[{'image': [t2f, t1c, t2w], 'label': seg}, ...]`` datalist that *both*
tracks consume. ``build_dataset`` then branches on the track:

  * **Track A** -> :class:`brats.transforms.TrackANaiveDataset`, which re-reads the
    NIfTIs on every access — **no cache** (defect D9, guardrail #5).
  * **Track B** -> ``monai.data.PersistentDataset`` caching the deterministic
    transform head under ``~/brats/cache/<track>_<split>`` on native ext4 (the D9
    fix, §6.2). ``CacheDataset`` is deliberately *not* used — 200 whole volumes will
    not fit in RAM (§6.2).

``build_loaders`` wires up train/val ``DataLoader``\\ s with ``list_data_collate`` so
Track B's ``RandCropByLabelClassesd(num_samples=...)`` list-of-crops is flattened into
the batch (§6.3).
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from monai.data import DataLoader, PersistentDataset, list_data_collate
from torch.utils.data import Dataset

from .paths import CACHE_DIR, RAW_DIR, SPLITS_JSON, assert_native_storage
from .transforms import (
    TrackANaiveDataset,
    track_b_train_transforms,
    track_b_val_transforms,
)

# Channel order 0,1,2 — matches the original (t1n dropped), §2.1 / §14.2.
MODALITIES: tuple[str, ...] = ("t2f", "t1c", "t2w")
# DataLoader workers (§1.1: num_workers ≈ 4–6; do not oversubscribe the 16 threads).
DEFAULT_NUM_WORKERS = 4


def build_datalist(track: str, split: str) -> list[dict[str, Any]]:
    """Build the ``{'image': [t2f, t1c, t2w], 'label': seg}`` datalist for a split.

    Reads the case-ID lists from ``reports/splits.json`` (``splits.<split>``) and
    resolves every path under ``~/brats/raw/<case>/`` via :mod:`brats.paths`. The
    ``track`` argument is part of the shared interface but does not change the paths —
    both tracks read the identical raw NIfTIs (§4.2).

    Args:
        track: ``'a'`` or ``'b'`` (accepted for interface symmetry; paths are shared).
        split: one of ``'train'``, ``'val'``, ``'test'``.

    Returns:
        A list of dicts, one per case, in the split's recorded order.
    """
    if track not in ("a", "b"):
        raise ValueError(f"track must be 'a' or 'b', got {track!r}")

    with open(SPLITS_JSON, encoding="utf-8") as fh:
        splits = json.load(fh)["splits"]
    if split not in splits:
        raise ValueError(f"split must be one of {sorted(splits)}, got {split!r}")

    datalist: list[dict[str, Any]] = []
    for case_id in splits[split]:
        case_dir = RAW_DIR / case_id
        datalist.append(
            {
                "image": [str(case_dir / f"{case_id}-{m}.nii.gz") for m in MODALITIES],
                "label": str(case_dir / f"{case_id}-seg.nii.gz"),
            }
        )
    return datalist


def build_dataset(cfg: Mapping[str, Any], track: str, split: str) -> Dataset:
    """Build the dataset for ``(track, split)`` (CLAUDE.md §6.1–6.3).

    Track A returns the naive, uncached :class:`TrackANaiveDataset` (D9). Track B
    returns a ``PersistentDataset`` whose transform is the train chain for the
    ``train`` split and the deterministic val chain otherwise; its ``cache_dir`` is
    asserted to be on native ext4 (§1.3, guardrail #1) before use.
    """
    datalist = build_datalist(track, split)

    if track == "a":
        # D9 faithful: no cache, reload per __getitem__.
        return TrackANaiveDataset(datalist)

    if track == "b":
        transforms = (
            track_b_train_transforms(cfg) if split == "train" else track_b_val_transforms(cfg)
        )
        cache_dir = CACHE_DIR / f"{track}_{split}"
        assert_native_storage(CACHE_DIR)  # §1.3 — cache must not live on a Windows/9p drive
        cache_dir.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)

    raise ValueError(f"track must be 'a' or 'b', got {track!r}")


def build_loaders(cfg: Mapping[str, Any], track: str) -> tuple[DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader)`` for a track (CLAUDE.md §6.3, §6.6).

    Batch size comes from ``cfg['batch_size']`` (Track A = 1 whole volume, Track B =
    2 cases). ``list_data_collate`` flattens Track B's ``RandCropByLabelClassesd``
    list-of-``num_samples`` crops, so the effective Track B batch is
    ``batch_size * num_samples`` patches of 96³. Validation always uses batch size 1:
    Track A whole volumes and Track B foreground-cropped volumes are variable-sized
    and evaluated with sliding-window inference (§6.7).
    """
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", DEFAULT_NUM_WORKERS))
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": list_data_collate,
        "pin_memory": _cuda_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        build_dataset(cfg, track, "train"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        build_dataset(cfg, track, "val"),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def _cuda_available() -> bool:
    """Whether CUDA is available (gates ``pin_memory``); tolerant of import order."""
    import torch

    return torch.cuda.is_available()
