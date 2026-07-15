"""Tests for brats.metrics.PerClassDice — proves the D2 fix is correct (§3/§6.5).

These are the tests §8.4 flags as non-cuttable: "A broken metric invalidates the
entire claim — that is literally defect D2 in the original." They assert:
  (a) perfect prediction -> Dice 1.0 for every present class;
  (b) a class absent from all GT has count 0 and does NOT inflate mean_fg
      (the empty-empty / RC-absent trap of §6.5);
  (c) per-class GT-present case counts are correct on a mixed batch.
"""
from __future__ import annotations

import math

import torch

from brats.metrics import FG_CLASS_NAMES, PerClassDice

NUM_CLASSES = 5


def _logits_from_labels(label: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Build logits [B, C, D, H, W] whose argmax reproduces ``label`` exactly.

    ``label`` is [B, D, H, W] integers in ``0..num_classes-1``.
    """
    onehot = torch.nn.functional.one_hot(label.long(), num_classes=num_classes)  # [B,D,H,W,C]
    onehot = onehot.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]
    # Scale so argmax is unambiguous; exact values are irrelevant to argmax.
    return onehot * 10.0 - 5.0


def _blank(shape=(4, 4, 4)) -> torch.Tensor:
    return torch.zeros((1, *shape), dtype=torch.long)


def test_perfect_prediction_gives_dice_one_for_present_classes() -> None:
    """(a) Perfect prediction -> Dice == 1.0 for each class present in GT."""
    # One case containing every foreground class (and background).
    label = _blank()
    # Assign a distinct voxel to each of classes 1..4; rest stays background 0.
    label[0, 0, 0, 0] = 1  # NETC
    label[0, 0, 0, 1] = 2  # SNFH
    label[0, 0, 0, 2] = 3  # ET
    label[0, 0, 0, 3] = 4  # RC

    logits = _logits_from_labels(label)

    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()
    metric.update(logits, label)
    out = metric.aggregate()

    for name in FG_CLASS_NAMES:
        assert out["counts"][name] == 1, f"{name} should be present once"
        assert math.isclose(out["per_class"][name], 1.0, rel_tol=0, abs_tol=1e-6), (
            f"{name} Dice should be 1.0 on a perfect prediction, got {out['per_class'][name]}"
        )
    assert math.isclose(out["mean_fg"], 1.0, abs_tol=1e-6)


def test_perfect_prediction_accepts_channel_dim_labels() -> None:
    """update must accept [B, 1, D, H, W] labels as well as [B, D, H, W]."""
    label = _blank()
    label[0, 0, 0, 0] = 2  # SNFH only
    logits = _logits_from_labels(label)

    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()
    metric.update(logits, label.unsqueeze(1))  # [B, 1, D, H, W]
    out = metric.aggregate()

    assert out["counts"]["SNFH"] == 1
    assert math.isclose(out["per_class"]["SNFH"], 1.0, abs_tol=1e-6)


def test_absent_class_has_zero_count_and_does_not_inflate_mean() -> None:
    """(b) A class absent from all GT -> count 0, Dice NaN, excluded from mean_fg.

    Also exercises the empty-empty trap: RC (label 4) never appears in GT here,
    and even a false-positive RC prediction must not make it count.
    """
    # GT contains only SNFH (label 2). No NETC/ET/RC anywhere.
    label = _blank()
    label[0, 0, 0, 0] = 2

    logits = _logits_from_labels(label)
    # Inject a spurious RC prediction where GT is background (empty-vs-nonempty case).
    logits[0, 4, 3, 3, 3] = 50.0  # force argmax -> class 4 at that voxel

    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()
    metric.update(logits, label)
    out = metric.aggregate()

    # Present class scores perfectly.
    assert out["counts"]["SNFH"] == 1
    assert math.isclose(out["per_class"]["SNFH"], 1.0, abs_tol=1e-6)

    # Absent classes: count 0, Dice NaN, never scored as 1.0 (empty-empty trap).
    for name in ("NETC", "ET", "RC"):
        assert out["counts"][name] == 0, f"{name} must have count 0"
        assert math.isnan(out["per_class"][name]), f"{name} Dice must be NaN, not a number"

    # mean_fg is over PRESENT classes only -> equals SNFH's Dice (1.0),
    # NOT diluted (or inflated) by the absent classes.
    assert math.isclose(out["mean_fg"], 1.0, abs_tol=1e-6)


def test_counts_correct_on_mixed_batch() -> None:
    """(c) Per-class GT-present counts are correct across a multi-case batch."""
    shape = (4, 4, 4)

    # Case 0: NETC + SNFH present.
    c0 = torch.zeros((1, *shape), dtype=torch.long)
    c0[0, 0, 0, 0] = 1  # NETC
    c0[0, 0, 0, 1] = 2  # SNFH

    # Case 1: SNFH + ET present.
    c1 = torch.zeros((1, *shape), dtype=torch.long)
    c1[0, 0, 0, 0] = 2  # SNFH
    c1[0, 0, 0, 1] = 3  # ET

    # Case 2: SNFH + RC present.
    c2 = torch.zeros((1, *shape), dtype=torch.long)
    c2[0, 0, 0, 0] = 2  # SNFH
    c2[0, 0, 0, 1] = 4  # RC

    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()

    # Feed as two updates (batch of 2, then batch of 1) to also prove that
    # accumulation across update() calls aligns buffer rows with GT presence.
    batch01 = torch.cat([c0, c1], dim=0)  # [2, ...]
    metric.update(_logits_from_labels(batch01), batch01)
    metric.update(_logits_from_labels(c2), c2)

    out = metric.aggregate()

    # NETC in case0 only; SNFH in all three; ET in case1 only; RC in case2 only.
    assert out["counts"] == {"NETC": 1, "SNFH": 3, "ET": 1, "RC": 1}

    # Predictions are perfect, so every present class scores 1.0.
    for name in FG_CLASS_NAMES:
        assert math.isclose(out["per_class"][name], 1.0, abs_tol=1e-6)
    assert math.isclose(out["mean_fg"], 1.0, abs_tol=1e-6)


def test_imperfect_prediction_scores_between_zero_and_one() -> None:
    """Sanity: a partial overlap yields a Dice strictly in (0, 1) with the right count."""
    # GT: 4 SNFH voxels in a row.
    label = torch.zeros((1, 4, 4, 4), dtype=torch.long)
    label[0, 0, 0, :] = 2  # 4 voxels of SNFH

    # Prediction: reproduce GT, then wipe half the SNFH voxels back to background.
    logits = _logits_from_labels(label)
    logits[0, 2, 0, 0, 2:] = -20.0  # those two voxels no longer argmax to SNFH
    logits[0, 0, 0, 0, 2:] = 20.0   # ...they become background instead

    metric = PerClassDice(num_classes=NUM_CLASSES)
    metric.reset()
    metric.update(logits, label)
    out = metric.aggregate()

    assert out["counts"]["SNFH"] == 1
    d = out["per_class"]["SNFH"]
    assert 0.0 < d < 1.0, f"expected partial Dice, got {d}"
    # 2 of 4 voxels correct: Dice = 2*2 / (4 + 2) = 2/3.
    assert math.isclose(d, 2.0 / 3.0, abs_tol=1e-6)
