"""Held-out TEST-set evaluation via whole-volume sliding-window inference.

CLAUDE.md §6.7: patch-train / sliding-window-infer is the standard MONAI idiom —
it resolves the 8 GB train-time VRAM ceiling without compromising test-time
whole-volume evaluation. Every case is run through
``sliding_window_inference(roi_size=(96,96,96), sw_batch_size=1, overlap=0.5)``.

CLAUDE.md §6.5 (NON-NEGOTIABLE): Dice is scored per class over **only** the cases
where that class is present in the ground truth, and the per-class case COUNTS are
reported alongside every Dice value (guardrail #4). All of that honesty lives in
:class:`brats.metrics.PerClassDice`; this module just drives it over the TEST split
and serialises the result to ``reports/eval_<track>.json``.

Both tracks share this evaluator. The track-specific preprocessing is supplied by
:func:`brats.data.build_dataset` — Track A yields its faithful ``128×128×48`` whole
volumes (defects D1/D5/D6), Track B yields foreground-cropped, z-scored volumes —
so each model is evaluated on inputs matching how it was trained (§4.2, §6.3).
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from monai.data import DataLoader, list_data_collate
from monai.inferers import sliding_window_inference

from .data import build_dataset
from .metrics import PerClassDice
from .model import build_model
from .paths import REPORTS_DIR

# Default sliding-window parameters (CLAUDE.md §6.7); overridable via cfg['inference'].
_DEFAULT_ROI = (96, 96, 96)
_DEFAULT_SW_BATCH = 1
_DEFAULT_OVERLAP = 0.5


def _select_device() -> torch.device:
    """Return CUDA if available, else CPU (§1.1 — training/eval run on the RTX 4060)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load model weights from ``ckpt_path`` into ``model`` (guardrail #7 — retrain clean).

    Robust to the common torch checkpoint layouts, since the exact format written by
    ``train.py`` is a plain state_dict or a wrapper dict: accepts a raw ``state_dict``
    or a dict carrying one under ``model_state_dict`` / ``state_dict`` / ``model``, and
    strips any ``module.`` prefix left by ``DataParallel``. Loads strictly — a mismatch
    means the checkpoint does not match the shared UNet architecture (§4.2) and must
    surface, not be silently ignored.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 — older/rich checkpoints need weights_only=False
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state: Mapping[str, Any] = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            inner = ckpt.get(key)
            if isinstance(inner, dict):
                state = inner
                break

    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)


def evaluate(
    cfg: Mapping[str, Any], track: str, ckpt_path: str | Path
) -> dict[str, Any]:
    """Evaluate a trained checkpoint on the held-out TEST split (§6.5, §6.7).

    Builds the shared UNet, loads ``ckpt_path``, runs whole-volume sliding-window
    inference over every TEST case, and accumulates per-class voxel-wise Dice with
    honest per-class case counts (:class:`PerClassDice`). Writes and returns
    ``{per_class, counts, mean_fg, n_test, ...}`` and saves it to
    ``reports/eval_<track>.json``.

    Args:
        cfg: Merged config dict from :func:`brats.config.load_config` (holds the
            shared ``model`` section and the ``inference`` sliding-window params).
        track: ``'a'`` or ``'b'`` — selects the track-faithful test preprocessing.
        ckpt_path: Path to the trained checkpoint (best mean-foreground-Dice model, D4).

    Returns:
        The results dict, also serialised to ``reports/eval_<track>.json``.
    """
    ckpt_path = Path(ckpt_path)
    device = _select_device()

    model = build_model(cfg).to(device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    # Track-faithful TEST preprocessing (Track A naive whole-volume; Track B cached head).
    dataset = build_dataset(cfg, track, "test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
    )

    inf = cfg.get("inference", {})
    roi_size = tuple(inf.get("roi_size", _DEFAULT_ROI))
    sw_batch_size = int(inf.get("sw_batch_size", _DEFAULT_SW_BATCH))
    overlap = float(inf.get("overlap", _DEFAULT_OVERLAP))
    num_classes = int(cfg.get("data", {}).get("num_classes", 5))

    metric = PerClassDice(num_classes=num_classes)
    metric.reset()

    # bf16 autocast on Ada, NO GradScaler (§1.2, guardrail #3); irrelevant at inference,
    # but keeps eval numerically consistent with training.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    n_test = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"]
            with autocast:
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                )
            # Score on CPU/float32 to keep the 8 GB budget free and the metric exact.
            metric.update(logits.float().cpu(), label.cpu())
            n_test += int(image.shape[0])

    agg = metric.aggregate()
    result: dict[str, Any] = {
        "track": track,
        "checkpoint": str(ckpt_path),
        "n_test": n_test,
        "per_class": agg["per_class"],
        "counts": agg["counts"],
        "mean_fg": agg["mean_fg"],
    }

    out_path = REPORTS_DIR / f"eval_{track}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    _print_summary(track, result)
    print(f"  wrote {out_path}")
    return result


def _print_summary(track: str, result: Mapping[str, Any]) -> None:
    """Print the per-class Dice table with case counts (guardrail #4, §6.5)."""
    per_class = result["per_class"]
    counts = result["counts"]
    print(f"== TEST-set voxel-wise Dice — track {track} (n_test={result['n_test']}) ==")
    print(f"  {'class':6s}  {'dice':>8s}  {'n (GT-present)':>15s}")
    for name, dice in per_class.items():
        print(f"  {name:6s}  {dice:8.4f}  {counts[name]:>15d}")
    print(f"  {'mean_fg':6s}  {result['mean_fg']:8.4f}")


if __name__ == "__main__":
    from .config import load_config

    import argparse

    _ap = argparse.ArgumentParser(description="Evaluate a track on the TEST split (§6.7).")
    _ap.add_argument("--track", required=True, choices=["a", "b"])
    _ap.add_argument("--ckpt", required=True)
    _args = _ap.parse_args()
    evaluate(load_config(_args.track), _args.track, _args.ckpt)
