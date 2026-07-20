#!/usr/bin/env bash
# Rebuild the pageindex-plus KB (kb/corpus + kb/index) from immutable sources. Idempotent.
set -e; cd "$(dirname "$0")/.."
mkdir -p kb/sources
tr '\r' '\n' < "reference/Model/3D-U-Net/output.txt" > kb/sources/3dunet_cce_training_log.txt
tr '\r' '\n' < "reference/Model/3D-U-Net/output updated.txt" > kb/sources/3dunet_cce_training_log_updated.txt
[ -f kb/sources/brats2024_deVerdier_2405.18368.pdf ] || curl -sL -o kb/sources/brats2024_deVerdier_2405.18368.pdf https://arxiv.org/pdf/2405.18368
[ -f kb/sources/kim_nnunet_2409.08143.pdf ]        || curl -sL -o kb/sources/kim_nnunet_2409.08143.pdf https://arxiv.org/pdf/2409.08143
# build_journal.md + build_transcript.md are distilled from the session build transcript via
# scratchpad/clean_transcript.py + the transcript-distill workflow. A durable, tracked copy lives
# in wiki/pages/build-journal.md; here they are static sources ingest picks up automatically.
# project_figures.md is an authored figure catalogue (kept in place, not regenerated).

# Project source-of-truth docs + result figures for the RAG (theory / rationale / defects / results).
cp CLAUDE.md kb/sources/claude_master_plan.md
cp report.md kb/sources/achievement_report.md
cp reports/results.md kb/sources/results_700_seeds.md
mkdir -p kb/sources/figures
cp reports/figures/*.png reports/figures/report/*.png kb/sources/figures/ 2>/dev/null || true
# The generated explainer note sections (for coverage tracking + adversarial review of the note).
mkdir -p kb/sources/note_sections
cp note/sections/*.md kb/sources/note_sections/ 2>/dev/null || true

uv run skills/pageindex-plus/scripts/ingest_notes.py kb/sources kb/corpus
uv run skills/pageindex-plus/scripts/scan_code.py --root . --dirs src reference/Model --out kb/corpus/code_map.md
uv run skills/pageindex-plus/scripts/build_pageindex.py kb/corpus kb/index --name "BraTS reproduction KB"
echo "KB rebuilt -> kb/index/ (query: skills/pageindex-plus/scripts/pageindex_query.py kb/index --search '...')"
