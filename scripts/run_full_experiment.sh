#!/usr/bin/env bash
# Full 700-case, 3-seed A/B experiment: split -> modalities -> train+eval sweep -> aggregate.
# Prereq: seg masks for all 700 already downloaded + contract-verified (reports/labels_summary.json).
# The DATA SPLIT is fixed (seed 42); only the TRAINING seed (init/aug/sampling) varies across runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "STEP 1/5 — download + contract-verify seg masks for all 700 (idempotent, resumes)"
uv run python scripts/01_fetch_subsample.py --n 700 --seed 42 --workers 6 --stage masks

log "STEP 2/5 — stratified 70/15/15 split over 700"
uv run python scripts/01_fetch_subsample.py --n 700 --seed 42 --stage split

log "STEP 3/5 — modality download for all 700 (idempotent; skips the 200 already present)"
uv run python scripts/01_fetch_subsample.py --n 700 --seed 42 --workers 6 --stage modalities

log "STEP 4/5 — seed sweep: {a,b} x seeds {0,1,2}"
declare -A EP=( [a]=25 [b]=80 )
for t in a b; do
  for s in 0 1 2; do
    log "  TRAIN track=$t seed=$s epochs=${EP[$t]}"
    uv run python scripts/03_train.py --track "$t" --seed "$s" --tag "s$s" --epochs "${EP[$t]}"
    log "  EVAL  track=$t seed=$s"
    uv run python scripts/04_evaluate.py --track "$t" --ckpt "/home/pritam/brats/runs/${t}_s${s}/best.pt"
    cp "reports/eval_${t}.json" "reports/eval_${t}_s${s}.json"
  done
done

log "STEP 5/5 — aggregate mean±SD over seeds"
uv run python scripts/aggregate_seeds.py

log "FULL_EXPERIMENT_DONE"
