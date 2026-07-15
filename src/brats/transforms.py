"""Track A (faithful) and Track B (corrected) transform chains (CLAUDE.md §6.3).

The A/B split lives here. **Track A** reproduces the original ``DataGenerator``
(``reference/.../unet_cc.py``) byte-for-behaviour so it fails in the documented way
(guardrail #5): a numpy/cv2 loader that reproduces

  * **D1** — ``cv2.resize`` with its default ``INTER_LINEAR`` applied to the *label*
    volume, producing fractional labels that are then truncated to int;
  * **D5** — global-max normalisation ``X / np.max(X)`` (one scalar over all three
    modalities), ignoring per-modality intensity distributions;
  * **D6** — the non-uniform axial slice stride ``z = 22 + int(j * 2.5)``;
  * **D9** — no caching: :class:`TrackANaiveDataset` re-reads four gzipped NIfTIs
    from disk on every ``__getitem__`` (faithful I/O-bound behaviour).

**Track B** is the corrected MONAI ``Compose`` head/tail: per-modality z-score over
non-zero voxels (D5 fix), nearest-neighbour geometry via ``Orientationd`` (no
bilinear label bug — D1 fix), an explicit label-contract assertion (D7), and
class-balanced patch sampling with ``RandCropByLabelClassesd`` (D3/D6 fix). The
deterministic head ends at ``CropForegroundd`` so ``PersistentDataset`` can cache to
there (D9 fix); the random tail re-runs every epoch.
"""
from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

import cv2
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from torch.utils.data import Dataset

# --- Track A geometry constants (verbatim from the original unet_cc.py) --------
VOLUME_SLICES = 48          # number of axial slices sampled per case
VOLUME_START_AT = 22        # first slice index included
SLICE_STRIDE = 2.5          # D6: irregular stride 0,2,5,7,10,... via int(j*2.5)
IMG_SIZE = 128              # in-plane resize target (square)

KEYS = ("image", "label")


def _load_fdata(path: str) -> np.ndarray:
    """Load a NIfTI's scaled voxel array as float64 (nibabel ``get_fdata``)."""
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]


def track_a_slice_index(j: int) -> int:
    """Axial slice index for the ``j``-th sampled slice (defect D6).

    ``z = VOLUME_START_AT + int(j * 2.5)`` yields strides 0,2,5,7,10,12,... — the
    original's non-uniform, anisotropic sampling with no resampling rationale (§3/D6).
    """
    return VOLUME_START_AT + int(j * SLICE_STRIDE)


def track_a_load_case(
    image_paths: Sequence[str], label_path: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one case exactly as the original ``DataGenerator`` did (D1/D5/D6).

    Reads the three modality NIfTIs (order = ``[t2f, t1c, t2w]``) plus the seg NIfTI
    with nibabel, takes 48 axial slices at the D6 stride, and ``cv2.resize``\\ s every
    slice to ``128×128`` with the **default ``INTER_LINEAR``** interpolation — applied
    to the label too, which is the D1 bilinear-label bug. The stacked image is then
    scaled by its single global maximum (D5), and the fractionally-resized label is
    truncated to int (the downstream corruption D1 describes).

    Args:
        image_paths: three modality paths ``[t2f, t1c, t2w]`` (channel order 0,1,2).
        label_path: the ``-seg.nii.gz`` path.

    Returns:
        ``(image, label)`` where ``image`` is a float tensor ``[3, 128, 128, 48]`` and
        ``label`` is a long tensor ``[128, 128, 48]``.
    """
    if len(image_paths) != 3:
        raise ValueError(f"expected 3 modality paths [t2f, t1c, t2w], got {len(image_paths)}")

    vols = [_load_fdata(p) for p in image_paths]  # each (182, 218, 182) float64
    seg = _load_fdata(label_path)

    image = np.zeros((3, IMG_SIZE, IMG_SIZE, VOLUME_SLICES), dtype=np.float64)
    label = np.zeros((IMG_SIZE, IMG_SIZE, VOLUME_SLICES), dtype=np.float64)

    for j in range(VOLUME_SLICES):
        z = track_a_slice_index(j)
        for ch, vol in enumerate(vols):
            # D1: cv2.resize defaults to INTER_LINEAR (bilinear) — fine for images.
            image[ch, :, :, j] = cv2.resize(vol[:, :, z], (IMG_SIZE, IMG_SIZE))
        # D1 BUG: the *same* bilinear resize is applied to the label, inventing
        # fractional class values along boundaries.
        label[:, :, j] = cv2.resize(seg[:, :, z], (IMG_SIZE, IMG_SIZE))

    # D5: single global max across all three modalities (crude normalisation).
    image = image / np.max(image)

    # D1 (cont.): fractional labels truncated to int, corrupting class boundaries.
    label_int = label.astype(np.int64)

    return (
        torch.from_numpy(image).float(),       # [3, 128, 128, 48]
        torch.from_numpy(label_int).long(),    # [128, 128, 48]
    )


class TrackANaiveDataset(Dataset):
    """Faithful Track A dataset — re-reads from disk on every access (defect D9).

    Nothing is cached (guardrail #5, §3/D9): each ``__getitem__`` re-loads the four
    gzipped NIfTIs and re-runs the cv2 pipeline, reproducing the original's
    I/O-bound, gzip-decompression-dominated per-epoch cost. Items are the same
    ``{'image': [...], 'label': ...}`` dicts that :func:`brats.data.build_datalist`
    yields, so both tracks share one datalist format.
    """

    def __init__(self, datalist: Sequence[Mapping[str, Any]]) -> None:
        self.datalist = list(datalist)

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.datalist[index]
        image, label = track_a_load_case(item["image"], item["label"])
        return {"image": image, "label": label}


class AssertLabelSetd(MapTransform):
    """Fail loudly if a label volume carries values outside the allowed set (D7).

    CLAUDE.md §6.1.2/D7: the label contract is *not* stable across BraTS vintages,
    so we assert ``set(labels) ⊆ {0,1,2,3,4}`` on load rather than silently coercing
    (as the original's dead ``Y[Y==5]=4`` / ``Y[Y==4]=3`` remap would). A stray label
    5, or a ``{0,1,2,4}`` set, means a different vintage and must stop the run. The
    transform is a deterministic pass-through, so it stays in the cached head.
    """

    def __init__(
        self,
        keys: Sequence[str],
        allowed: Sequence[int] = (0, 1, 2, 3, 4),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.allowed = set(int(v) for v in allowed)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d: dict[Hashable, Any] = dict(data)
        for key in self.key_iterator(d):
            arr = np.asarray(d[key])
            observed = {int(round(float(v))) for v in np.unique(arr)}
            illegal = observed - self.allowed
            if illegal:
                src = ""
                meta = getattr(d[key], "meta", None)
                if isinstance(meta, Mapping):
                    src = f" ({meta.get('filename_or_obj', '')})"
                raise ValueError(
                    f"Label-contract violation (D7){src}: observed labels {sorted(observed)} "
                    f"contain {sorted(illegal)} outside allowed {sorted(self.allowed)}. "
                    f"This is a different BraTS vintage than expected — stop, do not coerce."
                )
        return d


def _track_b_deterministic_head(cfg: Mapping[str, Any]) -> list:
    """Deterministic (cacheable) head shared by Track B train and val (§6.3).

    Load → channel-first → RAS → label-contract assertion (D7) → per-modality z-score
    over non-zero voxels (D5 fix) → foreground crop. ``PersistentDataset`` caches the
    output of the final (deterministic) transform, so this is exactly the D9 fix
    boundary.
    """
    tcfg = cfg.get("transforms", {})
    axcodes = tcfg.get("orientation", "RAS")
    label_set = cfg.get("data", {}).get("label_set", (0, 1, 2, 3, 4))
    keys = list(KEYS)
    return [
        LoadImaged(keys=keys),                       # image = 3 stacked modalities
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=axcodes),
        AssertLabelSetd(keys=["label"], allowed=label_set),          # D7
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # D5 fix
        CropForegroundd(keys=keys, source_key="image"),  # end of cached head
    ]


def track_b_train_transforms(cfg: Mapping[str, Any]) -> Compose:
    """Track B training chain: cached deterministic head + random tail (§6.3).

    The tail is class-balanced patch sampling (``RandCropByLabelClassesd`` — the
    D3/D6 fix, oversampling minority classes via ``ratios``) followed by flips,
    90° rotations, and intensity scale/shift augmentation. All of it is random, so
    ``PersistentDataset`` re-applies it each epoch on top of the cached head.
    ``num_samples`` returns a *list* of crops per case — callers must flatten it with
    ``monai.data.list_data_collate``.
    """
    ps = cfg.get("transforms", {}).get("patch_sampling", {})
    spatial_size = tuple(ps.get("spatial_size", (96, 96, 96)))
    num_classes = int(ps.get("num_classes", 5))
    ratios = list(ps.get("ratios", [1, 2, 2, 2, 2]))
    num_samples = int(ps.get("num_samples", 2))
    keys = list(KEYS)

    tail = [
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=spatial_size,
            num_classes=num_classes,
            ratios=ratios,
            num_samples=num_samples,
            warn=False,  # absent minority classes are expected; don't spam the log
        ),
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]
    return Compose(_track_b_deterministic_head(cfg) + tail)


def track_b_val_transforms(cfg: Mapping[str, Any]) -> Compose:
    """Track B validation chain: deterministic head only, no crop/aug (§6.3, §6.7).

    Validation runs whole (cropped) volumes through sliding-window inference, so no
    random patch cropping or augmentation is applied — only the cacheable head.
    """
    return Compose(_track_b_deterministic_head(cfg))
