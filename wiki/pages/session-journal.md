---
title: Session Journal
type: session
status: active
tags: [journal, publication, cv, dossier, backup, sanitized]
sources:
  - ../CLAUDE.md
links:
  relates: [Build Journal, Results, Defect Inventory, Experimental Design, Data Provenance, Environment]
---

# Session Journal (2026-07-15 → 2026-09-30)

> A **sanitized** distillation of the interactive working session that took the project from a
> 200-case pilot to the published 700-case, 3-seed result, the public site, and the CV entry.
> Credentials, personal identifiers, machine names, disk identifiers, and third-party source code
> were deliberately left out (see *What was removed* at the end). Companion to [[Build Journal]],
> which covers the initial build (storage saga, WSL mounts, D7 discovery, collate crash).

## 1. Timeline

| Date (2026) | What happened |
|---|---|
| 07-15 | Master plan reviewed; data disk created and mounted; Phases 0–2 built; 200-case pilot trained (test mean-fg Dice 0.180 → 0.513); knowledge layer (pageindex KB + llm-wiki) and `report.md` produced; first copy of artefacts placed on the Windows drive. |
| 07-15 → 07-16 | **Scale-up**: 700 cases, 3 training seeds per track, split fixed at seed 42 (490/105/105). A Kaggle download hang at 698/700 masks was traced to a missing socket timeout and fixed. Final result: mean-fg Dice **0.349 ± 0.013 → 0.670 ± 0.002** (Δ +0.321 ± 0.013), NETC 0.009 → 0.466. All figures regenerated with SD whiskers; results/report/wiki refreshed; committed. |
| 07-16 | Prior build transcript distilled into [[Build Journal]] and indexed. **Explainer note** built: 24 sections (medical + DL theory from basics, derivations, 36+ mermaid diagrams, ~74 cited references), built by `scripts/build_note.py` with MathJax; adversarially reviewed (9 issues fixed); images reviewed by headless screenshot. Fixed scrollable left TOC with scroll-spy added; a second overlay montage added. |
| 07-20 | **Published**: repo public on GitHub (MIT), GitHub Pages from `main:/docs` with landing page, explainer, report. Third-party reference code purged from all git history before publishing (backup bundle kept locally). Repo created with a stray leading hyphen and renamed. A `.gitignore` pattern silently swallowed `docs/report.html` (404) — anchored to `/report.html`. Repo link moved into the landing hero. First 4-bullet CV draft (99–101 chars) and title. Reference-code provenance established: a Colab export with no author, README or license. |
| 07-21 | **CV Defense Dossier** built (phrase → claim / theory / evidence / verify / interview-question), statistics computed by `scripts/compute_dossier_facts.py` (paired t(2)=43.3, p=0.0005, 95% CI [0.290, 0.353], Cohen's d≈25), four dossier figures, adversarial review (5 minor fixes), published as a 4th site page. |
| 07-23 → 07-24 | Long Q&A: modalities, tissue classes, voxels, Dice, D1/D2/D5/D6, slice-start offset. Honest defect accounting settled: **9 diagnosed, 6 form the A/B contrast** (D1, D3, D4, D5, D6, D9); D2/D7/D8 corrected identically on both tracks. |
| 07-24 → 07-26 | **CV reframe** after senior feedback: drop A/B framing, plain language, "6 flaws" not 9, lead with Dice 0.67, show the 490/105/105 split. Site + dossier aligned ("align the surface, keep the depth"): landing hero and tables de-jargoned, dossier §0 quotes the new bullets, README reconciled. Explainer and report keep the A/B depth. |
| 08-20 → 08-27 | Recap of the 6 defects and the classes. **Architecture audit**: instantiated MONAI UNet inspected — the docs had wrongly called the network "norm-free"; it has 8 InstanceNorm3d layers (MONAI default) on both tracks. Corrected in 13 places + README + wiki; full architecture table added to dossier §3. |
| 09-30 | Backup audit before shrinking the WSL disk; this journal written. Open: README caveats still quote pilot-era numbers (200 cases, NETC 30/30, true 17/25) in three lines. |

## 2. Decisions (with the why)

1. **Scale to 700 cases × 3 seeds, split fixed at seed 42.** The pilot had one run per track and no error bars; fixing the split makes all six runs share one test set so only training-seed variance remains.
2. **Purge the reference code from git history before publishing.** It is unlicensed third-party code (a Colab export); it stays local for the reproduction study and is never redistributed.
3. **MIT license for the repo.**
4. **Anchor the `.gitignore` entry for the root `report.html`** so the published `docs/report.html` is tracked.
5. **Honest defect count**: 9 diagnosed; 6 in the A/B contrast; D2 (metric bug), D7 (5-class contract), D8 (normalization) are applied identically to both tracks and therefore cannot explain the delta.
6. **CV framing**: simplified per senior feedback — no A/B jargon, no p-values on the CV, "6 data-pipeline flaws", best Dice 0.67, split shown. Depth lives on the site and in the dossier, not on the CV.
7. **Voxel-wise Dice, stated everywhere.** Not the BraTS lesion-wise challenge score; not leaderboard-comparable. The honest headline is rare-class recovery.
8. **Never fabricate a "Live app" link** for this project; the link table lists only the site, repo, and the three HTML notes.
9. **D8 semantics corrected**: the *reference* had no normalization; the reimplementation uses instance norm on both tracks. "Handicap cancels in the delta" rests on the small model size (1.98 M params), not on the absence of norm.

## 3. Incidents and fixes

| Incident | Root cause | Fix |
|---|---|---|
| Fetch hung at 698/700 masks (0 % CPU, threads sleeping) | Kaggle client had no socket timeout; a stalled `recv()` wedged the pool | `socket.setdefaulttimeout(90)` in `scripts/01_fetch_subsample.py`; killed the tree, cleaned partial `.nii`, restarted → 0 failures |
| `compute_dossier_facts.py` crashed with FileNotFoundError | The data disk detaches after a WSL restart; per-seed `metrics.csv` live there | Fallback to the committed values in `reports/results.md` |
| My mid-run prediction that the delta would shrink to ~0.20 | Track B rose more than Track A at scale | Corrected explicitly; final Δ +0.321 |
| Repo created as `-brats-postop-seg` | Typo in the URL | `gh repo rename` |
| `docs/report.html` 404 on Pages | Unanchored ignore pattern | `/report.html` |
| `rm -rf` of a copied reference dir silently failed | Read-only permissions | `chmod -R u+rwX` then remove |
| Explainer claimed 4 classes in some places, 5 in others; a wrong ResNet link; SE-block arithmetic | Drafting inconsistency | Convention fixed: **four tumour tissue labels, five classes counting background**; 9 review issues fixed |
| A style-contract file sat inside the dossier source dir and would have rendered | Wrong location | Moved to `note/dossier_STYLE_CONTRACT.md` |
| "Fixed 9 pipeline defects" on the CV | Overclaim | 9 diagnosed / 6 in the contrast, propagated everywhere |
| Docs said the network is "norm-free" | Assumed from the reference, not from the instantiated model | Inspected the model: 8 × InstanceNorm3d; corrected in all documents |

## 4. Facts settled in Q&A (crib sheet)

- **Modalities**: t2f (FLAIR), t1c (post-contrast T1), t2w (T2). t1n dropped for memory.
- **Labels**: 0 background, 1 NETC (non-enhancing tumour core), 2 SNFH (surrounding non-enhancing FLAIR hyperintensity), 3 ET (enhancing tissue), 4 RC (resection cavity).
- **Data**: Kaggle mirror of BraTS 2024 post-treatment glioma, 700 cases, 182 × 218 × 182 (MNI-like, resampled), 3-D voxels. RC present in ~83 % of cases; ET present in 579/700.
- **Dice** = 2|A∩B| / (|A|+|B|), computed voxel-wise per class per case, averaged over cases where the class is present in the ground truth; mean-fg = mean over the four tissue classes.
- **D1**: bilinear resize of the *label* volume then int truncation → invents boundary classes (e.g. NETC/ET boundary → SNFH). Evidence: Track A's "ground truth" has NETC in 105/105 test cases vs 48 truly.
- **D2**: the original per-class Dice indexed a 5-D one-hot tensor with 4 indices in the denominator → its own logs were untrustworthy. Fixed identically on both tracks.
- **D3**: plain cross-entropy, no balanced sampling → coasts on background. **D4**: checkpoint chosen on voxel accuracy (≈98 % for "all background") instead of mean-fg Dice.
- **D5**: `X / max(X)` global normalization → replaced by per-modality z-score over brain voxels (in the deterministic transform head, which is also the cache boundary).
- **D6**: slice indexing `22 + int(j·2.5)` → alternating 2/3 stride over source slices 22–139; the 22 offset simply skips the empty inferior slices. Replaced by foreground crop + class-balanced patches.
- **D9**: uncached loader re-reads and re-resizes every epoch; ~23 s → ~8 s per epoch (~2.8×) with a persistent cache.
- **Network** (identical on both tracks): MONAI 3-D UNet, channels (16, 32, 64, 128, 256), strides (2, 2, 2, 2), `num_res_units=0`, dropout 0.1 → 9 conv layers (5 Conv3d + 4 ConvTranspose3d), 5 levels, 4 skip connections, **no residual units**, 8 × InstanceNorm3d, 8 × PReLU, 3 × 3 × 3 kernels, **1,983,069** parameters.
- **Track A vs B selection**: A val acc 0.991 / mean-fg 0.320, best epoch 18–22; B 0.995 / 0.682, best epoch 64–72.

## 5. The CV entry (final)

**BraTS 2024: post-treatment glioma segmentation (3D U-Net)**

- Built a 3D U-Net to segment brain-tumour regions from post-treatment MRI scans (BraTS 2024).
- Curated 700 post-treatment brain-tumour MRI scans into a 490/105/105 train/validation/test split.
- Fixed 6 data-pipeline flaws causing the model to miss rare, heavily under-represented tumour classes.
- Improved segmentation accuracy (Dice) to 0.67 on unseen scans; rare classes recovered from near zero.

| Surface | URL |
|---|---|
| Project website | https://metaphorpritam.github.io/brats-postop-seg/ |
| Explainer (24 sections) | https://metaphorpritam.github.io/brats-postop-seg/explainer.html |
| CV Defense Dossier | https://metaphorpritam.github.io/brats-postop-seg/dossier.html |
| Technical report | https://metaphorpritam.github.io/brats-postop-seg/report.html |
| Source (MIT) | https://github.com/metaphorpritam/brats-postop-seg |

## 6. Where everything lives (backup map, as of 2026-09-30)

| Item | Location | Backed up? |
|---|---|---|
| Code, configs, tests, wiki, note sources, dossier sources, reports (`results_700_seeds.json`, per-seed `eval_*.json`, splits, figures), `docs/` site | GitHub `main` (clean, in sync) | **Yes** |
| Built HTML notes (`explainer`, `dossier`, `report`) | `docs/` in git + Windows `important_files/` | **Yes** (regenerable via `scripts/publish_docs.sh`) |
| Raw data (700 cases), preprocessing cache | Dedicated ext4 virtual disk file on the Windows D: drive | Not in git; **re-downloadable** from Kaggle via `scripts/01_fetch_subsample.py` |
| Six trained checkpoints (`best.pt`, ~8 MB each) + per-seed training `metrics.csv` | Same virtual disk, under `runs/{a,b}_s{0,1,2}/` | **Not in git** (weights are gitignored). Re-trainable via `scripts/run_full_experiment.sh` (hours). Copy them out before deleting the disk. |
| Pageindex KB (`kb/`) | Local, gitignored | Re-derivable via `scripts/build_kb.sh` (except the transcript-derived source, whose durable form is [[Build Journal]] + this page) |
| Third-party reference code | Local only + a local backup bundle | Deliberately **never** in the public repo |
| Session transcript + assistant memory files | WSL home, under the assistant's project directory | Not in git; this page is the sanitized durable copy |

**Remount recipe for the data disk** (it detaches after every WSL restart): from an elevated PowerShell, `wsl --mount --vhd "<path to the brats vhdx on D:>" --bare`, then from WSL `wsl -u root mount -L brats /mnt/wsl/brats`. Mount under `/mnt/wsl` (shared namespace); fstab/systemd entries do not work here.

**Recreate the environment**: `uv sync` (Python 3.13, torch 2.13 + cu130, MONAI 1.6.0), then `PYTHONPATH=src uv run python scripts/00_verify_env.py`.

## 7. What was removed from this journal (and why)

- The Kaggle API token that was pasted into chat during setup — a credential; it should be **rotated** on Kaggle.
- Personal e-mail addresses, the Windows machine name and user profile path, and the virtual disk's filesystem UUID — identifiers.
- The third-party reference code and its archive — unlicensed; only its provenance (a Colab export with no author, README or license) is recorded.
- Assistant session identifiers and internal scratch paths.
