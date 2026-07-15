#!/usr/bin/env bash
# Rebuild the pageindex-plus KB (kb/corpus + kb/index) from immutable sources. Idempotent.
set -e; cd "$(dirname "$0")/.."
mkdir -p kb/sources
tr '\r' '\n' < "reference/Model/3D-U-Net/output.txt" > kb/sources/3dunet_cce_training_log.txt
tr '\r' '\n' < "reference/Model/3D-U-Net/output updated.txt" > kb/sources/3dunet_cce_training_log_updated.txt
[ -f kb/sources/brats2024_deVerdier_2405.18368.pdf ] || curl -sL -o kb/sources/brats2024_deVerdier_2405.18368.pdf https://arxiv.org/pdf/2405.18368
[ -f kb/sources/kim_nnunet_2409.08143.pdf ]        || curl -sL -o kb/sources/kim_nnunet_2409.08143.pdf https://arxiv.org/pdf/2409.08143
uv run skills/pageindex-plus/scripts/ingest_notes.py kb/sources kb/corpus
uv run skills/pageindex-plus/scripts/scan_code.py --root . --dirs src reference/Model --out kb/corpus/code_map.md
uv run skills/pageindex-plus/scripts/build_pageindex.py kb/corpus kb/index --name "BraTS reproduction KB"
echo "KB rebuilt -> kb/index/ (query: skills/pageindex-plus/scripts/pageindex_query.py kb/index --search '...')"
