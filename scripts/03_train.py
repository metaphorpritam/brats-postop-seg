#!/usr/bin/env python
"""Phase 3/4 entry point: train Track A or Track B (CLAUDE.md §6.6, §7).

    uv run python scripts/03_train.py --track a     # faithful baseline (~10 ep, D1-D7,D9)
    uv run python scripts/03_train.py --track b     # corrected pipeline (~40 ep)

Loads the deep-merged config for the chosen track (base.yaml + track_<track>.yaml),
then hands off to ``brats.train.train``. The fragmentation-reducing allocator env is
set BEFORE torch is imported (§1.4); nothing else here touches the experiment logic —
the track's config is the single source of truth for what differs (§4.2).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# §1.4 — must be set before torch initialises its CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brats.config import load_config  # noqa: E402
from brats.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a BraTS 3D U-Net track (§6.6).")
    ap.add_argument(
        "--track",
        choices=["a", "b"],
        required=True,
        help="'a' = faithful baseline (D1-D7,D9); 'b' = corrected pipeline.",
    )
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the training seed (init/aug/sampling). Data split stays fixed.")
    ap.add_argument("--epochs", type=int, default=None, help="Override cfg epochs.")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the run dir, e.g. --tag s0 -> runs/<track>_s0 (seed sweep).")
    args = ap.parse_args()

    cfg = load_config(args.track)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    summary = train(cfg, args.track, tag=args.tag)

    print("=" * 60)
    print(f"Track {args.track}: best {summary['metric_selection']} = "
          f"{summary['best_value']} @ epoch {summary['best_epoch']}")
    print(f"Artifacts (best.pt / last.pt / metrics.csv / TB): {summary['run_dir']}")


if __name__ == "__main__":
    main()
