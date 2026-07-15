#!/usr/bin/env python
"""Phase 5 (CLAUDE.md §6.5, §6.7): evaluate a trained checkpoint on the held-out TEST split.

Runs whole-volume sliding-window inference over the TEST cases in
``reports/splits.json`` and writes per-class voxel-wise Dice — **with per-class case
counts** (§6.5, guardrail #4) — plus mean foreground Dice to ``reports/eval_<track>.json``.

Usage:
    uv run python scripts/04_evaluate.py --track b --ckpt ~/brats/runs/track_b/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when run as a bare script (mirrors 01_fetch_subsample.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brats.config import load_config  # noqa: E402
from brats.evaluate import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=["a", "b"],
                    help="which track's shared config / test preprocessing to use")
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="path to the trained checkpoint (best mean-foreground-Dice model, D4)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    cfg = load_config(args.track)
    evaluate(cfg, args.track, args.ckpt)


if __name__ == "__main__":
    main()
