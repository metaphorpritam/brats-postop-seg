"""Per-class Dice with correct shapes + honest NaN handling (fixes defect D2).

CLAUDE.md §3/D2: the original's four ``dice_coef_*`` functions indexed
``y_pred[:, :, :, N]`` (4 indices) against ``y_true[:, :, :, :, N]`` (5 indices),
a silent shape bug that made every minority-class Dice in the original logs
untrustworthy. Ours delegates the arithmetic to ``monai.metrics.DiceMetric`` and
handles the empty-class trap explicitly.

CLAUDE.md §6.5 (the NaN trap): Dice is undefined when a class is absent from
*both* ground truth and prediction; class 4 (RC) is absent in many cases.
This module therefore:

  * aggregates each foreground class over **only** the cases where that class is
    present in the ground truth,
  * NEVER scores an empty-empty case as 1.0 (which would silently inflate the mean),
  * reports the per-class case count alongside every Dice value (guardrail #4).
"""
from __future__ import annotations

from typing import Dict, List

import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

# Foreground labels 1..4 in order. Index i here == metric-buffer column i
# (DiceMetric with include_background=False drops channel 0/background).
FG_CLASS_NAMES: tuple[str, ...] = ("NETC", "SNFH", "ET", "RC")


class PerClassDice:
    """Voxel-wise per-class Dice for BraTS 5-class segmentation (§6.5).

    Wraps ``DiceMetric(include_background=False, reduction='mean_batch',
    get_not_nans=True)``. ``update`` takes raw logits + integer labels, converts
    both to one-hot, and accumulates into the metric buffer while separately
    tracking which foreground classes are present in the ground truth of each
    case. ``aggregate`` averages each class over its GT-present cases only.
    """

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = num_classes
        self.metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=True,
        )
        self._gt_present: List[torch.Tensor] = []

    def reset(self) -> None:
        """Clear the metric buffer and the per-case GT-presence record."""
        self.metric.reset()
        self._gt_present = []

    def update(self, pred_logits: torch.Tensor, label: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            pred_logits: ``[B, C, D, H, W]`` raw network outputs (C == num_classes).
            label: integer ground truth, ``[B, 1, D, H, W]`` or ``[B, D, H, W]``.
        """
        # Normalise label to a channel-first singleton: [B, 1, D, H, W].
        if label.ndim == pred_logits.ndim - 1:
            label = label.unsqueeze(1)
        label = label.long()

        pred = pred_logits.argmax(dim=1, keepdim=True)  # [B, 1, D, H, W]
        pred_oh = one_hot(pred, num_classes=self.num_classes, dim=1)
        label_oh = one_hot(label, num_classes=self.num_classes, dim=1)

        # Buffer stores per-case, per-foreground-class Dice (NaN where undefined).
        self.metric(pred_oh, label_oh)

        # GT presence per foreground class (channels 1..num_classes-1). This is the
        # authoritative definition of "class present in GT" (§6.5) and is robust
        # regardless of MONAI's ignore_empty default.
        fg = label_oh[:, 1:]  # [B, C-1, D, H, W]
        present = fg.reshape(fg.shape[0], fg.shape[1], -1).sum(dim=2) > 0  # [B, C-1]
        self._gt_present.append(present.detach().cpu())

    def aggregate(self) -> Dict[str, object]:
        """Reduce to per-class Dice, GT-present case counts, and mean foreground Dice.

        Returns a dict with keys ``per_class`` ({name: dice}), ``counts``
        ({name: n_cases_with_class_present_in_GT}), and ``mean_fg`` (mean of the
        per-class Dice over classes present in at least one case). A class absent
        from every ground truth reports Dice ``nan`` and count ``0`` and does not
        contribute to ``mean_fg``.
        """
        n_fg = self.num_classes - 1

        if self._gt_present:
            present = torch.cat(self._gt_present, dim=0)  # [N, C-1] bool
        else:
            present = torch.zeros((0, n_fg), dtype=torch.bool)

        buf = self.metric.get_buffer()  # [N, C-1], NaN where class undefined
        if buf is None or buf.numel() == 0:
            buf = torch.full((present.shape[0], n_fg), float("nan"))
        buf = buf.detach().cpu().float()

        per_class: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        present_dice: List[float] = []

        for idx, name in enumerate(FG_CLASS_NAMES):
            mask = present[:, idx]
            n = int(mask.sum().item())
            counts[name] = n
            if n == 0:
                # No ground-truth support -> Dice undefined; excluded from mean_fg.
                per_class[name] = float("nan")
                continue
            # GT-present => Dice is defined; nanmean is belt-and-braces.
            dice = float(torch.nanmean(buf[mask, idx]).item())
            per_class[name] = dice
            present_dice.append(dice)

        mean_fg = float(sum(present_dice) / len(present_dice)) if present_dice else float("nan")
        return {"per_class": per_class, "counts": counts, "mean_fg": mean_fg}
