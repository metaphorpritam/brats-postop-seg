#!/usr/bin/env python
"""Synthetic end-to-end smoke test (CLAUDE.md guardrail #11, WBS 2.9).

Exercises the whole training/inference contract for BOTH tracks on *synthetic*
volumes only — NO real data, NO Kaggle download, NO alternate BraTS vintage.
Guardrail #11 is explicit: smoke-testing on a different vintage forks the label
handling (`num_classes` 4 vs 5, a different remap) — exactly the defect class this
project claims to detect (D7). So we use ``monai.data.synthetic.create_test_image_3d``
to fabricate a few 3-channel volumes with 5-class integer labels and drive the real
public APIs end-to-end:

  * :func:`brats.config.load_config` -> :func:`brats.model.build_model` / count_params;
  * one forward pass under bf16 autocast on CUDA (no ``GradScaler``, §1.2 / guardrail #3);
  * the track loss — plain CE for Track 'a' with integer labels ``[B,D,H,W]`` (defect D3),
    ``DiceCELoss`` for Track 'b' with one-hot-ready ``[B,1,D,H,W]`` labels (D3 fix, §6.5);
  * a single ``loss.backward()`` (asserts gradients flow);
  * :class:`brats.metrics.PerClassDice` ``update``/``aggregate`` — asserting it returns
    per-class Dice + case COUNTS and does NOT crash when a foreground class (RC, label 4)
    is absent from every ground truth (the §6.5 NaN trap, guardrail #4);
  * one ``monai.inferers.sliding_window_inference`` call (§6.7).

Prints ``SMOKE PASS`` on success; exits non-zero on any failure (WBS 2.9 gate).

Run:  cd ~/code/brats-postop-seg && PYTHONPATH=src uv run python scripts/02_smoke_test.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# §1.4 — must be set before torch initialises its CUDA allocator (mirrors 03_train.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Make the src package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from contextlib import nullcontext  # noqa: E402

from monai.data.synthetic import create_test_image_3d  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402

from brats.config import load_config  # noqa: E402
from brats.losses import build_loss  # noqa: E402
from brats.metrics import FG_CLASS_NAMES, PerClassDice  # noqa: E402
from brats.model import build_model, count_params  # noqa: E402

# Small enough to run in seconds on CPU / fit the 8 GB budget; divisible by 16 so
# the 4-stride UNet (strides (2,2,2,2)) downsamples cleanly, and == roi_size so the
# sliding window covers a synthetic volume in a single window.
SPATIAL: int = 96
BATCH: int = 2
NUM_CLASSES: int = 5
RC_LABEL: int = 4  # class 4 == RC — deliberately kept ABSENT to exercise the §6.5 path.


def _autocast(enabled: bool):
    """bf16 CUDA autocast (§1.2). Disabled -> no-op so the CPU fallback still runs.

    NO ``GradScaler`` anywhere — bf16 on Ada does not need one (guardrail #3).
    """
    if enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_synthetic_batch(
    batch: int, size: int, rng: np.random.RandomState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fabricate a synthetic batch: 3-channel images + 5-class integer labels.

    Uses ``create_test_image_3d`` (guardrail #11 — synthetic, not another vintage) to
    build each modality channel and a base segmentation, then shapes the label into the
    project's 5-class contract {0 bg, 1 NETC, 2 SNFH, 3 ET, 4 RC}. Class 4 (RC) is
    forced ABSENT and classes 1/2/3 are planted as guaranteed foreground blocks, so the
    batch deterministically exercises BOTH the present-class and absent-class branches
    of :meth:`PerClassDice.aggregate` (§6.5) regardless of the random object placement.

    Returns:
        image ``[B, 3, size, size, size]`` float32, label ``[B, size, size, size]`` int64.
    """
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _ in range(batch):
        modalities: list[np.ndarray] = []
        seg: np.ndarray | None = None
        for _c in range(3):  # 3 modalities: t2f, t1c, t2w (t1n dropped) — §6.4
            img, s = create_test_image_3d(
                size, size, size,
                num_objs=12,
                rad_max=size // 4,
                rad_min=3,
                noise_max=0.4,
                num_seg_classes=NUM_CLASSES - 1,  # -> labels {0,1,2,3,4}
                channel_dim=None,
                random_state=rng,
            )
            modalities.append(img.astype(np.float32))
            if seg is None:
                seg = s
        assert seg is not None
        label = seg.astype(np.int64)

        # Enforce the 5-class distribution deterministically:
        #   * RC (4) absent from every case -> aggregate() must report count 0 / NaN Dice
        #     without crashing (§6.5 NaN trap, guardrail #4);
        #   * classes 1/2/3 present -> mean_fg is a real number and the present-branch runs.
        label[label == RC_LABEL] = 0
        label[2:8, 2:8, 2:8] = 1      # NETC
        label[2:8, 12:18, 2:8] = 2    # SNFH
        label[12:18, 2:8, 2:8] = 3    # ET

        images.append(np.stack(modalities, axis=0))  # [3, D, H, W]
        labels.append(label)

    image_t = torch.from_numpy(np.stack(images, axis=0)).float()   # [B, 3, D, H, W]
    label_t = torch.from_numpy(np.stack(labels, axis=0)).long()    # [B, D, H, W]
    return image_t, label_t


def smoke_track(track: str, device: torch.device, use_cuda: bool) -> None:
    """Drive one track (``'a'`` or ``'b'``) through the full forward/loss/backward/metric/infer path."""
    print(f"\n== Track {track} ==")
    cfg = load_config(track)

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"  model built: {n_params:,} trainable params (orig Keras ~5,645,845; §6.4)")

    loss_fn = build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    rng = np.random.RandomState(int(cfg.get("seed", 42)))
    image, label = make_synthetic_batch(BATCH, SPATIAL, rng)
    image = image.to(device)
    label = label.to(device)
    print(f"  synthetic batch: image {tuple(image.shape)}  label {tuple(label.shape)}")

    # --- forward + loss under bf16 autocast; backward OUTSIDE it (§1.2, guardrail #3) ---
    model.train()
    with _autocast(use_cuda):
        logits = model(image)
        # Track A CE wants integer labels [B,D,H,W]; Track B DiceCE (to_onehot_y) wants
        # a channel-first singleton [B,1,D,H,W] (§6.5).
        target = label if cfg["loss"] == "cross_entropy" else label.unsqueeze(1)
        loss = loss_fn(logits, target)

    expected = (BATCH, NUM_CLASSES, SPATIAL, SPATIAL, SPATIAL)
    assert tuple(logits.shape) == expected, f"logits {tuple(logits.shape)} != {expected}"
    loss_val = float(loss.detach().item())
    assert math.isfinite(loss_val), f"non-finite loss: {loss_val}"
    print(f"  forward OK: logits {tuple(logits.shape)}  loss={loss_val:.4f}")

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced by loss.backward()"
    total_grad = sum(float(g.abs().sum().item()) for g in grads)
    assert math.isfinite(total_grad) and total_grad > 0.0, "gradients are zero/non-finite"
    print(f"  backward OK: {len(grads)} param tensors have grad (sum|grad|={total_grad:.3e})")

    # --- PerClassDice: must return per-class Dice + counts, no crash on the absent class ---
    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()
    metric.update(logits.detach().float(), label)
    agg = metric.aggregate()

    assert set(agg) >= {"per_class", "counts", "mean_fg"}, f"aggregate keys: {set(agg)}"
    per_class, counts = agg["per_class"], agg["counts"]
    for name in FG_CLASS_NAMES:
        assert name in per_class and name in counts, f"missing class {name} in aggregate"
    # RC (class 4) is absent from every GT -> count 0, Dice NaN, excluded from mean_fg.
    assert counts["RC"] == 0, f"RC should be absent, got count {counts['RC']}"
    assert math.isnan(per_class["RC"]), f"absent RC Dice should be NaN, got {per_class['RC']}"
    # At least one planted class present -> mean_fg is a real number.
    assert any(counts[n] > 0 for n in FG_CLASS_NAMES), "no foreground class present in GT"
    assert not math.isnan(agg["mean_fg"]), "mean_fg is NaN despite present classes"
    dice_str = ", ".join(
        f"{n}={per_class[n]:.3f}(n={counts[n]})" if not math.isnan(per_class[n])
        else f"{n}=nan(n={counts[n]})"
        for n in FG_CLASS_NAMES
    )
    print(f"  PerClassDice OK: {dice_str}  mean_fg={agg['mean_fg']:.3f}")

    # --- sliding-window inference (§6.7) ---
    inf = cfg.get("inference", {})
    roi_size = tuple(inf.get("roi_size", (96, 96, 96)))
    sw_batch_size = int(inf.get("sw_batch_size", 1))
    overlap = float(inf.get("overlap", 0.5))
    model.eval()
    with torch.no_grad(), _autocast(use_cuda):
        sw_logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    assert tuple(sw_logits.shape) == expected, f"sw output {tuple(sw_logits.shape)} != {expected}"
    print(f"  sliding_window_inference OK: roi={roi_size} -> {tuple(sw_logits.shape)}")


def main() -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}  (bf16 autocast {'ON' if use_cuda else 'OFF — CPU fallback'})")

    for track in ("a", "b"):
        smoke_track(track, device, use_cuda)

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
