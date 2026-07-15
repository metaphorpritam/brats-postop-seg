"""Loss factory for the BraTS 3D U-Net A/B experiment (CLAUDE.md §6.5).

Track A uses plain, unweighted categorical cross-entropy — this *is* defect D3
(§3): the original defines ``alpha = [0.05, 0.25, 0.25, 0.25, 0.20]`` and never
uses it, so the model coasts on the ~98%-background majority. Track B fixes D3
with a foreground-only Dice + CE loss.
"""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss


def build_loss(cfg: dict) -> torch.nn.Module:
    """Build the training loss selected by ``cfg['loss']`` (CLAUDE.md §6.5).

    ``'cross_entropy'`` (Track A) -> :class:`torch.nn.CrossEntropyLoss` with **no
    class weights** — the faithful reproduction of defect D3 (§3, §6.5). Expects
    raw logits ``[B, 5, D, H, W]`` and an integer target ``[B, D, H, W]`` (long).

    ``'dice_ce'`` (Track B) -> :class:`monai.losses.DiceCELoss` with
    ``include_background=False, to_onehot_y=True, softmax=True`` — the D3 fix
    (§6.5). Expects raw logits ``[B, 5, D, H, W]`` and a label ``[B, 1, D, H, W]``.

    Args:
        cfg: Merged config dict; ``cfg['loss']`` selects the loss.

    Returns:
        A callable ``nn.Module`` loss.

    Raises:
        KeyError: If ``cfg`` has no ``'loss'`` key.
        ValueError: If ``cfg['loss']`` names an unrecognised loss.
    """
    name = cfg["loss"]
    if name == "cross_entropy":
        # Track A — plain CCE, NO class weights (defect D3, §3 / §6.5).
        return torch.nn.CrossEntropyLoss()
    if name == "dice_ce":
        # Track B — foreground-only Dice + CE, one-hot label + softmax (D3 fix, §6.5).
        return DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
    raise ValueError(
        f"Unknown loss {name!r}; expected 'cross_entropy' or 'dice_ce' (§6.5)."
    )
