# Dossier style contract — the CV-defense note

This is a **defense dossier**: it takes one CV entry (four bullets) and unpacks *every phrase*
into a defensible claim. The reader is an interviewer or a technically literate recruiter who may
challenge any word. Tone: precise, confident, and scrupulously honest — never overstate.

## The five-part pattern (use it for every phrase you unpack)

Each phrase of the bullet gets a `###` subsection built from these five moves, in order:

1. **Claim** — quote the exact phrase in bold.
2. **What it means** — define the terms from first principles; the theory behind the choice
   (link to the companion explainer with `§N` where deep derivations already live).
3. **Evidence** — the concrete numbers/plots/tables that back it. Use the EXACT figures from the
   fact sheet below. Embed the relevant figure with `![caption](path)`.
4. **Verify** — where an interviewer can check it: the specific committed file/artifact.
5. **Interview question** — the sharpest question this phrase invites, with a crisp, honest answer.
   Put it in an `!!! example "Interview question"` box.

Not every phrase needs all five as separate labels — weave them into flowing prose — but every
phrase must end with its **Interview question** box, and every quantitative claim must show its
**Evidence** and name where to **Verify** it.

## Formatting (same engine as the explainer)
- File starts with exactly `## N. Title {#id}` (given to you). `###` for each phrase.
- Math: `$ ... $` inline, `$$ ... $$` display (pymdownx.arithmatex + MathJax).
- Callouts (4-space indented body, blank line before):
  - `!!! example "Interview question"` — the likely question + a 2-4 sentence honest answer.
  - `!!! gotcha "Watch out"` — a caveat or a place the claim is easy to overstate.
  - `!!! intuition "The one-liner"` — the memorable defense of this phrase.
- Tables: markdown. Figures: `![caption](path)` using the EXACT paths in the fact sheet.
- No `#` H1, no `[TOC]`. Cross-reference the explainer as `§N` (its section numbers are stable).

## THE FACT SHEET — use these numbers verbatim; invent nothing

**Result (held-out TEST, n_test = 105; mean ± SD over 3 training seeds):**
- Mean foreground Dice: **Track A 0.349 ± 0.013 → Track B 0.670 ± 0.002**; **Δ = +0.321 ± 0.013**.
- Per-seed Track A mean-fg: [0.340, 0.342, 0.364]; Track B: [0.672, 0.668, 0.671].
- Per-seed **delta**: [0.332, 0.325, 0.307].
- Per-class (A → B, Δ): NETC 0.009 → 0.466 (**+0.457**); SNFH 0.683 → 0.870 (+0.187);
  ET 0.299 → 0.650 (+0.351); RC 0.403 → 0.694 (+0.291).

**Statistics of the delta (all computed in scripts/compute_dossier_facts.py):**
- SE = 0.0074; **95% t-CI (df = 2) = [0.290, 0.353]** — excludes 0.
- Paired **t(2) = 43.3, p = 0.0005** (two-sided); **Cohen's d = 25**.
- Honest caveat: n = 3 *seeds* on ONE fixed test split — this measures reproducibility across
  seeds, NOT population generalization. The real uncertainty is the single 105-case test set.

**Track A fidelity (the pathology it faithfully reproduces):** validation voxel accuracy **0.991**
vs validation mean-fg Dice **0.320** vs NETC Dice **0.009**; best checkpoint at epochs 18–22
(selected on accuracy = defect D4). Track B: val acc 0.995, val mean-fg 0.682, epochs 64–72.

**D1 count inflation (GT-present TEST cases, of 105):** Track A's bilinear-resized labels report
NETC in **105/105** and ET in **100/105**; the faithful nearest-neighbour labels show only **48**
and **85**. (SNFH 105/105, RC 87/87 on both.)

**Data:** 700 cases from a Kaggle mirror (`i212385nomanarif/2024-brats-glioma`) of BraTS-2024
post-treatment glioma (BraTS-GLI). Three input modalities used: **t2f** (T2-FLAIR), **t1c**
(post-contrast T1), **t2w** (T2) — the native T1 (t1n) was dropped, matching the reference.
Geometry **182×218×182** (a resampled, MNI-like grid), not native 240×240×155. Split **490/105/105**
(70/15/15), **stratified by resection-cavity (RC) presence**, fixed at **seed 42**. Stratification
result: **RC present in 82.9% of cases in train, val, AND test** (identical). Label contract
verified: all masks integer-valued ⊆ {0,1,2,3,4}; ET (label 3) present in **579/700** cases →
confirms post-treatment 2024 vintage (rules out a pre-op {0,1,2,4} set).

**Network (identical on both tracks):** MONAI `UNet`, spatial_dims = 3, in_channels = 3,
out_channels = 5 (four tumour classes + background), channels (16, 32, 64, 128, 256), strides
(2, 2, 2, 2), num_res_units = 0, dropout = 0.1. **Total = 1,983,069 trainable parameters.**
Params by feature width: 16 ch → 1,328; 32 → 18,208; 64 → 746,624; 128 → 331,904; 256 → 884,992.

**The five labels:** 0 background, 1 NETC (non-enhancing tumour core), 2 SNFH (surrounding
non-enhancing FLAIR hyperintensity / oedema), 3 ET (enhancing tumour), 4 RC (resection cavity).

**The nine defects (Track A reproduces D1,D3,D4,D5,D6,D9; D2 fixed on both; D7 enforced; D8 held):**
D1 bilinear `cv2.resize` on the label volume → invents/destroys labels (fix: nearest-neighbour);
D2 per-class Dice shape/index bug → original logs untrustworthy (fix: correct MONAI DiceMetric);
D3 class imbalance unhandled, plain cross-entropy (fix: DiceCELoss + class-balanced patch sampling);
D4 checkpoint selected on validation voxel accuracy (fix: select on mean foreground Dice);
D5 global-max normalisation X/max(X) (fix: per-modality z-score over brain voxels);
D6 non-uniform slice stride int(j·2.5) (fix: foreground crop + 96³ class-balanced patches);
D7 a dead label remap that merges RC into ET, i.e. a 4-class contract (fix: assert 5-class at load);
D8 no normalisation layers in the network (held constant on BOTH tracks — not part of the A/B);
D9 uncached I/O-bound loader (fix: MONAI PersistentDataset cache; ~23 s → ~8 s/epoch, ~2.8×).

**Provenance (defensible + honest):** the reference was a Colab export — `unet_cc.py`'s header reads
*"Copy of Untitled7.ipynb … Automatically generated by Colab"* — i.e. itself copied from an
unnamed shared notebook, carrying a BraTS-2020-era label remap onto 2024 data (that is defect D7).
It has **no author, no README, no LICENSE**; it was NOT redistributed (only its behaviour was
reproduced in PyTorch). No original weights were reused.

**Scale honesty:** at the 200-case pilot Track A collapsed on everything (mean-fg 0.180) and B hit
0.513 — a 2.8× ratio. At 490 training cases Track A recovers the prevalent classes (0.349) and B
rises to 0.670 — a 1.9× ratio. The **absolute** delta held (+0.333 → +0.321); the ratio fell only
because Track A's denominator grew. Quote the absolute delta, never the ratio.

**Metric caveat (state it whenever a Dice number appears):** these are **voxel-wise** Dice — NOT the
BraTS lesion-wise score — so they are internally consistent for an A/B but **not comparable to
challenge leaderboards**.

## Figure paths (use exactly; do not invent filenames)
- Delta with 95% CI + paired t: `reports/figures/dossier/delta_ci.png`
- Stratification (RC % per split): `reports/figures/dossier/stratification.png`
- Parameter budget: `reports/figures/dossier/param_breakdown.png`
- Track A accuracy-vs-Dice pathology: `reports/figures/dossier/track_a_fidelity.png`
- Per-class TEST Dice ±SD: `reports/figures/report/perclass_test_dice.png`
- Per-class A→B delta: `reports/figures/report/delta_test.png`
- D1 count inflation: `reports/figures/report/count_inflation.png`
- Training curves (mean±SD): `reports/figures/report/training_curves.png`
- Qualitative overlays (set 1): `reports/figures/combined_overlay.png`
- Qualitative overlays (set 2): `reports/figures/combined_overlay_2.png`
- Defect ladder diagram: `figures/diagrams/s20_defect_ladder.png`
