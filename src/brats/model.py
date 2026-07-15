"""Single shared UNet factory (CLAUDE.md §6.4).

The network architecture is **byte-identical across Track A and Track B** — this is
the whole reason the A/B comparison is credible (§4.2, guardrail #2). Only the data
pipeline, loss, and model-selection criterion may vary. Both tracks therefore call
``build_model`` on a ``cfg`` whose ``model`` section comes from the shared
``configs/base.yaml`` (see §4.2, D8 held constant).

Config keys consumed from ``cfg['model']`` (per §6.4):
    spatial_dims=3, in_channels=3 (t2f, t1c, t2w — t1n dropped), out_channels=5
    (0 bg, 1 NETC, 2 SNFH, 3 ET, 4 RC), channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2), num_res_units=0 (plain U-Net), dropout=0.1.

The original Keras net had 5,645,845 params; MONAI's block structure differs, so the
count will not match exactly but must land in the same order of magnitude (§6.4).
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from monai.networks.nets import UNet

# Keys of cfg['model'] that are metadata, not UNet constructor arguments.
_NON_KWARG_KEYS = frozenset({"name"})

# Sequence-valued UNet arguments — normalized from YAML lists to tuples.
_SEQUENCE_KEYS = frozenset({"channels", "strides", "kernel_size", "up_kernel_size"})


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build the shared MONAI ``UNet`` from ``cfg['model']`` (CLAUDE.md §6.4).

    ``cfg`` is the deep-merged config dict from ``brats.config.load_config(track)``.
    The ``model`` section is identical for both tracks (§4.2); ``name`` is treated as
    metadata and dropped, every other key is forwarded to ``monai.networks.nets.UNet``.
    """
    model_cfg = cfg["model"]
    kwargs: dict[str, Any] = {}
    for key, value in model_cfg.items():
        if key in _NON_KWARG_KEYS:
            continue
        # YAML gives lists; MONAI expects sequences — tuples are the idiomatic form.
        kwargs[key] = tuple(value) if key in _SEQUENCE_KEYS else value
    return UNet(**kwargs)


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters (§6.4 sanity check vs 5,645,845)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from brats.config import load_config

    for track in ("a", "b"):
        model = build_model(load_config(track))
        print(f"track {track}: {count_params(model):,} trainable params")
