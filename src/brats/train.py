"""Training loop for the BraTS 3D U-Net A/B experiment (CLAUDE.md §6.6).

Both tracks share this loop; the config decides everything that differs (§4.2).
The two invariants that make the A/B credible and 8 GB-safe live here:

  * **bf16 autocast, NO ``GradScaler``** — Ada supports bf16 natively, so a scaler
    is unnecessary and forbidden (§1.2, guardrail #3). Forward + loss run under
    ``torch.autocast('cuda', dtype=torch.bfloat16)``; backward runs outside it.
  * **Checkpoint selection is the D4 axis** (§3, §6.6). Track A selects ``best.pt``
    on ``val_accuracy`` — the degenerate metric under ~98 %% background that *is*
    defect D4. Track B selects on **mean foreground Dice** — the D4 fix. Both are
    computed every validation regardless of track; only which one drives selection
    changes, keyed off ``cfg['metric_selection']``.

Validation uses ``sliding_window_inference(roi_size=(96,96,96), sw_batch_size=1,
overlap=0.5)`` (§6.7) so patch-trained Track B is evaluated on whole (foreground-
cropped) volumes, and Track A's whole 128×128×48 volumes go through the same path.
Per-epoch metrics are written to a CSV and to TensorBoard under ``~/brats/runs/<track>``
(§6.6, §9 — every log is a committed artifact).
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .data import build_loaders
from .losses import build_loss
from .metrics import FG_CLASS_NAMES, PerClassDice
from .model import build_model, count_params
from .paths import RUNS_DIR, assert_native_storage

# Selection metric -> "higher is better" is true for both, so ``>`` always wins.
_SELECT_ACCURACY = "val_accuracy"      # Track A, defect D4 (§3, §6.6)
_SELECT_MEAN_FG_DICE = "mean_fg_dice"  # Track B, the D4 fix (§6.6)


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch + MONAI determinism (§6.6: 'set all seeds').

    ``set_determinism`` also seeds the transform RNGs so the cached deterministic
    head and the random augmentation tail are reproducible run-to-run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def _select_metric_name(cfg: Mapping[str, Any], track: str) -> str:
    """Resolve which validation metric drives ``best.pt`` selection (D4 axis, §6.6).

    Honours ``cfg['metric_selection']`` (``'val_accuracy'`` = Track A defect,
    ``'mean_fg_dice'`` = Track B fix); falls back to the track default so a stripped
    config still selects correctly.
    """
    raw = str(cfg.get("metric_selection", "")).strip().lower()
    if raw in (_SELECT_ACCURACY, "accuracy", "val_acc"):
        return _SELECT_ACCURACY
    if raw in (_SELECT_MEAN_FG_DICE, "mean_fg", "mean_fg_dice", "dice"):
        return _SELECT_MEAN_FG_DICE
    # Fallback: Track A -> accuracy (D4 defect), Track B -> mean fg Dice (D4 fix).
    return _SELECT_ACCURACY if track == "a" else _SELECT_MEAN_FG_DICE


def _autocast(enabled: bool):
    """bf16 CUDA autocast context (§1.2). Disabled -> no-op, so CPU smoke tests run.

    NOTE: no ``GradScaler`` anywhere — bf16 on Ada does not need one (guardrail #3).
    """
    if enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _labels_for_ce(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Squeeze a channel-first singleton label to match ``argmax`` output shape.

    Track A labels arrive as ``[B, D, H, W]`` (matches ``pred``); Track B as
    ``[B, 1, D, H, W]``. Used for pixel-accuracy, which compares against ``pred``.
    """
    return label if label.ndim == pred.ndim else label.squeeze(1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    amp: bool,
) -> dict[str, Any]:
    """Sliding-window validation: per-class Dice (+counts) and pixel accuracy (§6.6/§6.7).

    Returns a dict with ``per_class`` ({name: dice}), ``counts`` ({name: n_present}),
    ``mean_fg`` (mean foreground Dice over present classes), and ``accuracy`` (all-class
    voxel accuracy, including background — the degenerate D4 metric). ``sw_batch_size=1``
    keeps validation inside the 8 GB budget (§6.7, OOM playbook §11 step 5).
    """
    model.eval()
    dice = PerClassDice(num_classes=num_classes)
    correct = 0
    total = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        with _autocast(amp):
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        logits = logits.float()
        dice.update(logits, label)
        pred = logits.argmax(dim=1)
        lab = _labels_for_ce(pred, label)
        correct += int((pred == lab).sum().item())
        total += int(lab.numel())

    result = dice.aggregate()
    result["accuracy"] = (correct / total) if total else float("nan")
    return result


def _write_meta(run_dir: Path, cfg: Mapping[str, Any], track: str, seed: int,
                selection: str, n_params: int) -> None:
    """Record seeds + the config actually used, so the run is reproducible (§9)."""
    meta = {
        "track": track,
        "seed": seed,
        "metric_selection": selection,
        "n_params": n_params,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cfg": _jsonable(cfg),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a config mapping to JSON-serialisable primitives."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


_CSV_FIELDS = (
    ["epoch", "lr", "train_loss", "val_accuracy", "mean_fg_dice"]
    + [f"dice_{n}" for n in FG_CLASS_NAMES]
    + [f"count_{n}" for n in FG_CLASS_NAMES]
)


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler,
                     epoch: int, selection: str, value: float,
                     cfg: Mapping[str, Any], track: str) -> None:
    """Serialise model + optim/sched state + selection provenance for evaluate.py."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metric_selection": selection,
            "best_value": value,
            "track": track,
            "cfg": _jsonable(cfg),
        },
        path,
    )


def train(cfg: dict[str, Any], track: str) -> dict[str, Any]:
    """Train one track end-to-end (CLAUDE.md §6.6). Returns a run summary.

    Seeds everything; builds the shared UNet, the track's loss, and the train/val
    loaders; optimises with ``AdamW(lr=cfg['training']['lr'])`` under a
    ``CosineAnnealingLR(T_max=epochs)`` schedule; trains with bf16 autocast and no
    ``GradScaler`` (guardrail #3). Validates every ``cfg['val_interval']`` epochs
    (and on the last epoch) with sliding-window inference, logging per-epoch rows to
    ``metrics.csv`` and TensorBoard under ``~/brats/runs/<track>``. Saves ``last.pt``
    every epoch and ``best.pt`` whenever the selection metric improves — accuracy for
    Track A (D4 defect), mean foreground Dice for Track B (D4 fix).
    """
    if track not in ("a", "b"):
        raise ValueError(f"track must be 'a' or 'b', got {track!r}")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    loss_fn = build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    train_loader, val_loader = build_loaders(cfg, track)

    epochs = int(cfg["epochs"])
    lr = float(cfg["training"]["lr"])
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    num_classes = int(cfg.get("data", {}).get("num_classes", 5))
    inf = cfg.get("inference", {})
    roi_size = tuple(inf.get("roi_size", (96, 96, 96)))
    sw_batch_size = int(inf.get("sw_batch_size", 1))
    overlap = float(inf.get("overlap", 0.5))
    val_interval = int(cfg.get("val_interval", 2))

    selection = _select_metric_name(cfg, track)

    # Run outputs live on native ext4 (§1.3, guardrail #1), never on a 9p drive.
    run_dir = RUNS_DIR / track
    assert_native_storage(RUNS_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_meta(run_dir, cfg, track, seed, selection, n_params)

    writer = SummaryWriter(log_dir=str(run_dir))
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(_CSV_FIELDS)

    best_value = -math.inf
    best_epoch = -1
    best_val: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=use_cuda)
            label = batch["label"].to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_cuda):
                logits = model(image)
                loss = loss_fn(logits, label)
            loss.backward()   # backward outside autocast; no GradScaler (guardrail #3)
            optimizer.step()

            running += float(loss.detach().item())
            steps += 1

        scheduler.step()
        train_loss = running / steps if steps else float("nan")
        cur_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", cur_lr, epoch)

        do_val = (epoch % val_interval == 0) or (epoch == epochs)
        val: dict[str, Any] = {}
        if do_val and len(val_loader):
            if use_cuda:
                torch.cuda.empty_cache()
            val = validate(
                model, val_loader, device,
                num_classes=num_classes, roi_size=roi_size,
                sw_batch_size=sw_batch_size, overlap=overlap, amp=use_cuda,
            )
            _log_validation(writer, val, epoch)

        _append_csv(csv_path, epoch, cur_lr, train_loss, val)

        print(
            f"[{track}] epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  lr={cur_lr:.2e}"
            + (
                f"  acc={val['accuracy']:.4f}  mean_fg_dice={val['mean_fg']:.4f}"
                if val else ""
            ),
            flush=True,
        )

        # --- checkpointing: last.pt always; best.pt on the D4 selection metric. ---
        _save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler,
                         epoch, selection, best_value, cfg, track)
        if val:
            value = val["accuracy"] if selection == _SELECT_ACCURACY else val["mean_fg"]
            if value is not None and not math.isnan(value) and value > best_value:
                best_value = value
                best_epoch = epoch
                best_val = val
                _save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler,
                                 epoch, selection, best_value, cfg, track)

    writer.close()

    summary = {
        "track": track,
        "epochs": epochs,
        "n_params": n_params,
        "metric_selection": selection,
        "best_epoch": best_epoch,
        "best_value": best_value if best_epoch > 0 else float("nan"),
        "best_val": best_val,
        "run_dir": str(run_dir),
    }
    print(
        f"[{track}] done. best {selection}={summary['best_value']} @ epoch {best_epoch}. "
        f"artifacts -> {run_dir}",
        flush=True,
    )
    return summary


def _log_validation(writer: SummaryWriter, val: Mapping[str, Any], epoch: int) -> None:
    """Push validation scalars to TensorBoard, skipping NaN (absent-class) Dice."""
    writer.add_scalar("val/accuracy", val["accuracy"], epoch)
    if not math.isnan(val["mean_fg"]):
        writer.add_scalar("val/mean_fg_dice", val["mean_fg"], epoch)
    for name, dv in val["per_class"].items():
        if dv is not None and not math.isnan(dv):
            writer.add_scalar(f"val/dice_{name}", dv, epoch)


def _append_csv(csv_path: Path, epoch: int, lr: float, train_loss: float,
                val: Mapping[str, Any]) -> None:
    """Append one per-epoch row; validation columns are blank on non-val epochs (§9)."""
    row: dict[str, Any] = {
        "epoch": epoch,
        "lr": f"{lr:.6e}",
        "train_loss": f"{train_loss:.6f}",
        "val_accuracy": "",
        "mean_fg_dice": "",
    }
    for name in FG_CLASS_NAMES:
        row[f"dice_{name}"] = ""
        row[f"count_{name}"] = ""
    if val:
        row["val_accuracy"] = f"{val['accuracy']:.6f}"
        row["mean_fg_dice"] = f"{val['mean_fg']:.6f}"
        for name in FG_CLASS_NAMES:
            row[f"dice_{name}"] = f"{val['per_class'][name]:.6f}"
            row[f"count_{name}"] = val["counts"][name]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([row[k] for k in _CSV_FIELDS])
