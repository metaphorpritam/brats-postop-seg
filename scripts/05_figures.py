#!/usr/bin/env python
"""Phase 5 (CLAUDE.md §7 gate 5, §8.6, §9): overlay figures for the A/B write-up.

For a handful of held-out TEST cases that actually contain tumour (ground truth with
ET and/or RC), this renders axial-slice PNGs that place, side by side:

    input (t2f)  |  input + GROUND TRUTH  |  input + Track A pred  |  input + Track B pred

using the 5-class BraTS colormap (0 bg transparent, 1 NETC, 2 SNFH, 3 ET, 4 RC).

**Faithful inference, two geometries (§6.3, §6.7).**
    * Track A runs its *faithful* pipeline: :func:`brats.transforms.track_a_load_case`
      builds the warped ``128×128×48`` whole volume (bilinear label resize D1, global-max
      norm D5, the ``int(j*2.5)`` slice stride D6), and the network is applied to the
      whole volume in one shot (batch 1) — exactly how Track A was trained/evaluated.
    * Track B runs ``sliding_window_inference(roi=(96,96,96), overlap=0.5)`` over the
      foreground-cropped, z-scored whole volume from
      :func:`brats.transforms.track_b_val_transforms` — exactly the §6.7 idiom used in
      :mod:`brats.evaluate`.

The two tracks live in different geometries (Track A: warped 128×128×48 in the raw
LAS array; Track B: RAS-oriented, foreground-cropped native resolution). To draw them
on ONE common axial view (§8.6 "one clean combined figure"), each prediction is mapped
back into the **native NIfTI array space** (182×218×182, the space of
``nib.load(...).get_fdata()`` shared by the ground truth):

    * Track A: the chosen axial slice is one the D6 stride actually sampled, and the
      128×128 prediction for it is nearest-neighbour-resized back to the native 182×218.
    * Track B: MONAI ``Invertd`` reverses ``Orientationd`` + ``CropForegroundd`` on the
      argmaxed prediction (``nearest_interp=True``), landing it in the native array.

Inference is faithful; only the *display* is resampled — stated honestly in the caption.

Outputs (under ``reports/figures/``):
    * ``overlay_<case>.png`` — per-case 4-panel detail (input / GT / A / B).
    * ``combined_overlay.png`` — the flagship montage: one row per case, columns
      GT / Track A / Track B, shared legend and the A→B mean-Dice delta (§8.6).

Every PNG is asserted non-empty and re-openable with PIL before the script exits.

Usage:
    PYTHONPATH=src uv run python scripts/05_figures.py
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

# Make the src/ package importable when run as a bare script (mirrors 04_evaluate.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless WSL2 — no display (§1.1)
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.data import MetaTensor  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import Invertd  # noqa: E402
from PIL import Image  # noqa: E402

from brats.config import load_config  # noqa: E402
from brats.data import build_datalist  # noqa: E402
from brats.evaluate import _load_checkpoint  # noqa: E402
from brats.model import build_model  # noqa: E402
from brats.paths import FIGURES_DIR, REPORTS_DIR, RUNS_DIR  # noqa: E402
from brats.transforms import (  # noqa: E402
    VOLUME_SLICES,
    track_a_load_case,
    track_a_slice_index,
    track_b_val_transforms,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TEST cases chosen for rich foreground (all four classes present; substantial ET+RC),
# so the 5-class colormap and the A→B minority-class recovery are both visible.
# All are drawn from reports/splits.json -> splits.test (verified present at runtime).
CASES: tuple[str, ...] = (
    "BraTS-GLI-02403-100",  # NETC/SNFH/ET/RC all present; large ET (~32k) + RC (~14k)
    "BraTS-GLI-02375-101",  # all four classes present; balanced
    "BraTS-GLI-00080-101",  # all four present; strong ET (~22k)
)

# Background modality for the overlays: channel 0 = t2f (FLAIR) — the primary modality
# (§2.1) on which SNFH is most conspicuous. Channel order is [t2f, t1c, t2w] (§14.2).
BG_MODALITY_IDX = 0
BG_MODALITY_NAME = "t2f"

# 5-class colormap (§2.2): 0 background is transparent; 1..4 are the foreground classes.
CLASS_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
CLASS_COLORS = {
    1: (0.231, 0.510, 0.965),  # NETC  — blue    (#3b82f6)
    2: (0.133, 0.773, 0.369),  # SNFH  — green   (#22c55e)
    3: (0.937, 0.267, 0.267),  # ET    — red     (#ef4444)
    4: (0.961, 0.620, 0.043),  # RC    — amber   (#f59e0b)
}
OVERLAY_ALPHA = 0.50

CKPT_A = RUNS_DIR / "a" / "best.pt"
CKPT_B = RUNS_DIR / "b" / "best.pt"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device):
    """bf16 autocast on CUDA (§1.2, guardrail #3 — no GradScaler); no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _axial(arr: np.ndarray) -> np.ndarray:
    """Orient a native (L, A[, C]) axial slice for display: anterior up, radiological aspect.

    Native array axes are (L, A, S); a fixed-``z`` slice is ``(L=182, A=218)`` (grayscale)
    or ``(182, 218, 4)`` (RGBA overlay). Swap the two spatial axes so A is vertical, then
    flip so anterior renders at the top — leaving any trailing channel axis intact.
    """
    if arr.ndim == 2:
        return np.flipud(arr.T)
    return np.flipud(np.transpose(arr, (1, 0, 2)))


def _window(gray2d: np.ndarray) -> np.ndarray:
    """Robust 1–99 percentile window of a grayscale slice to [0,1] for display contrast."""
    finite = gray2d[np.isfinite(gray2d)]
    if finite.size == 0:
        return np.zeros_like(gray2d, dtype=np.float32)
    lo, hi = np.percentile(finite, 1.0), np.percentile(finite, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((gray2d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _label_rgba(label2d: np.ndarray) -> np.ndarray:
    """Build an RGBA overlay from an integer label slice (0 => fully transparent)."""
    h, w = label2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        m = label2d == cls
        if m.any():
            rgba[m, 0], rgba[m, 1], rgba[m, 2] = color
            rgba[m, 3] = OVERLAY_ALPHA
    return rgba


def _draw_panel(ax, gray2d: np.ndarray, label2d: np.ndarray | None, title: str) -> None:
    """Grayscale background + optional colour label overlay on one axis."""
    ax.imshow(_axial(gray2d), cmap="gray", interpolation="nearest")
    if label2d is not None:
        ax.imshow(_axial(_label_rgba(label2d)), interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_handles():
    return [
        mpatches.Patch(color=CLASS_COLORS[c], label=f"{c} · {CLASS_NAMES[c]}")
        for c in (1, 2, 3, 4)
    ]


# ---------------------------------------------------------------------------
# Faithful inference for each track, mapped back into native array space
# ---------------------------------------------------------------------------

def _native_arrays(item: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (t2f background volume, integer GT seg) in native array space (182×218×182)."""
    bg = np.asarray(nib.load(item["image"][BG_MODALITY_IDX]).get_fdata(), dtype=np.float32)
    seg = np.asarray(nib.load(item["label"]).get_fdata()).round().astype(np.int64)
    return bg, seg


def _best_slice(seg: np.ndarray) -> tuple[int, int]:
    """Pick the Track-A-sampled axial slice richest in ET+RC (fallback: any foreground).

    The slice ``z`` must be one the D6 stride actually sampled, so Track A has a genuine
    whole-volume prediction there (not an interpolated gap). Returns ``(j0, z0)`` where
    ``z0 = track_a_slice_index(j0)``.
    """
    sampled = [(j, track_a_slice_index(j)) for j in range(VOLUME_SLICES)]

    def etrc(z: int) -> int:
        s = seg[:, :, z]
        return int(((s == 3) | (s == 4)).sum())

    def fg(z: int) -> int:
        return int((seg[:, :, z] > 0).sum())

    j0, z0 = max(sampled, key=lambda jz: (etrc(jz[1]), fg(jz[1])))
    return j0, z0


def _predict_track_a(model: torch.nn.Module, item: dict, device: torch.device,
                     j0: int, native_hw: tuple[int, int]) -> np.ndarray:
    """Faithful Track A whole-volume inference; return the native-space label slice for ``j0``.

    Runs the ``128×128×48`` warped volume through the net in one shot (§6.3 Track A), then
    nearest-neighbour-resizes the argmaxed 128×128 prediction at column ``j0`` back to the
    native ``(182, 218)`` axial grid.
    """
    image, _ = track_a_load_case(item["image"], item["label"])  # [3,128,128,48]
    with torch.no_grad(), _autocast(device):
        logits = model(image.unsqueeze(0).to(device))  # [1,5,128,128,48]
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # [128,128,48]
    slice128 = pred[:, :, j0]  # (128,128)
    h, w = native_hw  # (182, 218)
    return cv2.resize(slice128, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)


def _predict_track_b(model: torch.nn.Module, item: dict, device: torch.device,
                     val_transform) -> np.ndarray:
    """Faithful Track B sliding-window inference; return the FULL native-space label volume.

    Runs ``sliding_window_inference`` over the foreground-cropped, z-scored whole volume
    (§6.7), then ``Invertd`` reverses Orientationd + CropForegroundd (nearest interp) to
    place the prediction back into the native ``182×218×182`` array (§6.3 Track B).
    """
    data = val_transform({"image": item["image"], "label": item["label"]})
    img = data["image"].unsqueeze(0).to(device)
    with torch.no_grad(), _autocast(device):
        logits = sliding_window_inference(
            inputs=img, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model, overlap=0.5
        )
    pred_crop = logits.argmax(1, keepdim=True).float().cpu()[0]  # [1,D,H,W] in cropped space
    pred_mt = MetaTensor(
        pred_crop,
        meta=data["image"].meta,
        applied_operations=data["image"].applied_operations,
    )
    inverter = Invertd(
        keys="pred", transform=val_transform, orig_keys="image",
        nearest_interp=True, to_tensor=True,
    )
    inverted = inverter({"pred": pred_mt, "image": data["image"]})
    return np.asarray(inverted["pred"][0]).round().astype(np.int64)  # (182,218,182)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _verify_png(path: Path) -> tuple[int, int]:
    """Assert the PNG exists, is non-empty, and re-opens with PIL; return (width, height)."""
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"figure not written or empty: {path}")
    with Image.open(path) as im:
        im.load()  # force decode — proves it is a real, re-openable image
        w, h = im.size
    return w, h


def main() -> None:
    for ck in (CKPT_A, CKPT_B):
        if not ck.exists():
            raise SystemExit(f"checkpoint not found: {ck}")

    test_items = {
        d["label"].split("/")[-1].replace("-seg.nii.gz", ""): d
        for d in build_datalist("b", "test")
    }
    missing = [c for c in CASES if c not in test_items]
    if missing:
        raise SystemExit(f"cases not in TEST split: {missing}")

    device = _device()
    print(f"device: {device}")

    cfg_a, cfg_b = load_config("a"), load_config("b")
    model_a = build_model(cfg_a).to(device)
    _load_checkpoint(model_a, CKPT_A, device)
    model_a.eval()
    model_b = build_model(cfg_b).to(device)
    _load_checkpoint(model_b, CKPT_B, device)
    model_b.eval()
    val_transform_b = track_b_val_transforms(cfg_b)

    # A→B delta headline for the montage caption (from the committed eval JSONs).
    def _mean_fg(track: str) -> float:
        p = REPORTS_DIR / f"eval_{track}.json"
        return float(json.loads(p.read_text())["mean_fg"]) if p.exists() else float("nan")

    mean_a, mean_b = _mean_fg("a"), _mean_fg("b")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int, int]] = []

    # Cache per-case rendered slices so the montage reuses the per-case inference.
    rendered: list[dict] = []

    for cid in CASES:
        item = test_items[cid]
        bg_vol, seg_vol = _native_arrays(item)
        j0, z0 = _best_slice(seg_vol)
        native_hw = (bg_vol.shape[0], bg_vol.shape[1])  # (182, 218)

        pred_a_slice = _predict_track_a(model_a, item, device, j0, native_hw)
        pred_b_vol = _predict_track_b(model_b, item, device, val_transform_b)

        bg_slice = _window(bg_vol[:, :, z0])
        gt_slice = seg_vol[:, :, z0]
        pred_b_slice = pred_b_vol[:, :, z0]

        rendered.append(
            dict(cid=cid, z0=z0, bg=bg_slice, gt=gt_slice, a=pred_a_slice, b=pred_b_slice)
        )
        print(f"  {cid}: axial z={z0} (Track-A sampled slice j={j0}); "
              f"GT fg={int((gt_slice>0).sum())} vox")

        # ---- per-case 4-panel detail figure ----
        fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2))
        _draw_panel(axes[0], bg_slice, None, f"input · {BG_MODALITY_NAME}")
        _draw_panel(axes[1], bg_slice, gt_slice, "ground truth")
        _draw_panel(axes[2], bg_slice, pred_a_slice, "Track A (baseline)")
        _draw_panel(axes[3], bg_slice, pred_b_slice, "Track B (corrected)")
        fig.suptitle(
            f"{cid} — axial z={z0} (native space) · voxel-wise 5-class overlay",
            fontsize=12, y=0.99,
        )
        fig.legend(
            handles=_legend_handles(), loc="lower center", ncol=4,
            frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = FIGURES_DIR / f"overlay_{cid}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        w, h = _verify_png(out)
        written.append((out.name, w, h))

    # ---- flagship combined montage: rows = cases, cols = GT / A / B ----
    n = len(rendered)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    col_titles = ("ground truth", "Track A (baseline)", "Track B (corrected)")
    for r, rc in enumerate(rendered):
        panels = (rc["gt"], rc["a"], rc["b"])
        for c in range(3):
            _draw_panel(axes[r, c], rc["bg"], panels[c], col_titles[c] if r == 0 else "")
        axes[r, 0].set_ylabel(f"{rc['cid']}\nz={rc['z0']}", fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=38)
    delta = mean_b - mean_a
    fig.suptitle(
        "BraTS post-treatment — Track A vs Track B (held-out TEST, native-space axial)\n"
        f"mean foreground voxel Dice:  A={mean_a:.3f} → B={mean_b:.3f}  "
        f"(Δ +{delta:.3f})",
        fontsize=12.5, y=0.995,
    )
    fig.legend(
        handles=_legend_handles(), loc="lower center", ncol=4,
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.94))
    combined = FIGURES_DIR / "combined_overlay.png"
    fig.savefig(combined, dpi=140, bbox_inches="tight")
    plt.close(fig)
    w, h = _verify_png(combined)
    written.append((combined.name, w, h))

    print("\nwrote figures to", FIGURES_DIR)
    for name, w, h in written:
        print(f"  {name}: {w}x{h}px")


if __name__ == "__main__":
    main()
