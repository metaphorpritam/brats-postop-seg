---
title: BraTS Post-Treatment Reproduction Index
type: concept
status: active
tags: [index]
---

# BraTS Post-Treatment Reproduction — wiki index

Compile-don't-retrieve knowledge base for the BraTS-GLI post-treatment **A/B reproduction**.
Read this first, drill into the pages, answer from the wiki. Headline: held-out test mean
foreground Dice **0.180 → 0.513 (+0.333, ~2.8×)** — Track A faithfully reproduces defects
D1–D9, Track B fixes them, architecture held constant. Raw evidence (original training logs,
BraTS-2024 + Kim arXiv papers, code map) is indexed separately in the **pageindex KB** at
`../kb/index/` (query with `skills/pageindex-plus/scripts/pageindex_query.py`).

## Core pages

- [[Defect Inventory]] — the spine: D1–D9, each with the offending original line, why it
  matters, the Track B fix, and this project's empirical confirmation (incl. the D7 4-class finding).
- [[Experimental Design]] — the two tracks, the hold-architecture-constant rule (§4.2), the
  phase gates, and why the A→B delta is the deliverable.
- [[Data Provenance]] — Kaggle partial mirror + correct citation, label-contract/vintage,
  the MNI-space (182³) caveat, and the RC-stratified 140/30/30 split.
- [[Environment]] — RTX 4060 / WSL2, the ext4 filesystem rule (defect D9), bf16 no-GradScaler,
  the tool/version stack.
- [[Results]] — per-class test & validation Dice (Track A vs B) with case counts, the delta,
  and the ~2.8× D9 loader speedup.

## Decisions & open questions

Tracked in `memory/decisions.md` and `memory/questions.md` (append via `wiki_log.py`). See also
`../report.md` (rich analysis + plots + diagrams) and `../README.md` (how to reproduce).
