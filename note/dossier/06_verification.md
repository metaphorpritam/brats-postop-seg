## 6. Verification map — where every claim can be checked {#verification}

Nothing here is asserted on trust. Every number in the CV entry traces to a committed artifact you
can open, and the whole result regenerates from one script. This section is the index.

### Claim → artifact → how to check

| Claim (CV entry) | Value | Committed artifact | How to verify |
|---|---|---|---|
| Mean-fg Dice, A → B | 0.349 → 0.670 | `reports/results_700_seeds.json` | fields `track_a.mean_fg.mean`, `track_b.mean_fg.mean` |
| The delta | +0.321 ± 0.013 | `reports/results_700_seeds.json` | field `delta_mean_fg` (mean, sd, per-seed vals) |
| 95% CI / t / p / d | [0.290, 0.353], t=43.3, p=0.0005, d=25 | `reports/dossier_facts.json` | regenerate: `compute_dossier_facts.py` |
| Per-class Dice ± SD | see §4 table | `reports/eval_{a,b}_s{0,1,2}.json` | per-seed `per_class`; aggregated in `results_700_seeds.json` |
| Rarest class | NETC +0.457 | `reports/results_700_seeds.json` | `track_b.per_class.NETC − track_a.per_class.NETC` |
| D1 count inflation | A 105/100 vs B 48/85 | `reports/eval_{a,b}_s0.json` | field `counts` (identical across seeds) |
| Track A fidelity | acc 0.991 / Dice 0.320 | `reports/results.md` (val table) | validation accuracy vs mean-fg Dice |
| 700 cases, 490/105/105 | 70/15/15 | `reports/splits.json` | field `counts` (n per split) |
| Stratified | RC 82.9% every split | `reports/splits.json` | `rc / n` for train, val, test |
| Post-treatment vintage | ET in 579/700 | `reports/labels_summary.json` | count cases with label 3 present |
| 3 modalities | t2f, t1c, t2w | `scripts/01_fetch_subsample.py` | constant `INPUT_MODS` |
| 1,983,069 params | exact | `src/brats/model.py` | `build_model(...)`; count in `compute_dossier_facts.py` |
| Byte-identical net | shared config | `configs/base.yaml` + `configs/track_{a,b}.yaml` | the `model:` block lives in `base.yaml`; tracks override only pipeline/loss/selection |
| Nine defects (D1–D9) | — | `CLAUDE.md` §3; `wiki/pages/defect-inventory.md` | each defect with its offending original line and its fix |
| Reference provenance | Colab export, no license | *(local only — not in the public repo)* | `unet_cc.py` header: *"Copy of Untitled7.ipynb"* |

!!! gotcha "Watch out — two things are deliberately NOT in the public repo"
    The **data** (`~/brats`, outside the repo) and the **reference code** (`reference/`, purged from
    git history for licensing) are not published. The reference's provenance is quotable from its file
    header but the file itself was not redistributed. Everything in the table above *is* committed.

### Reproduce the whole result

```bash
# 1. the full experiment (fetch → stratified split → 6-run sweep → aggregate)
bash scripts/run_full_experiment.sh          # writes reports/results_700_seeds.json

# 2. recompute every statistic in this dossier (delta, CI, paired t, Cohen's d, params)
PYTHONPATH=src python scripts/compute_dossier_facts.py   # -> reports/dossier_facts.json

# 3. regenerate the figures
python scripts/06_report_figures.py          # per-class, delta, counts, curves (± SD)
python scripts/dossier_figures.py            # delta-CI, stratification, params, fidelity
PYTHONPATH=src python scripts/05_figures.py   # qualitative overlays

# 4. rebuild this dossier and the explainer
python scripts/build_note.py --src dossier --out cv_dossier.html --title "CV Defense Dossier"
python scripts/build_note.py                 # the explainer
```

### The public artifacts

- **Website:** <https://metaphorpritam.github.io/brats-postop-seg/>
- **Repository (MIT):** <https://github.com/metaphorpritam/brats-postop-seg>
- **Companion explainer** (`explainer.html`): the from-scratch theory — brain tumours, MRI, CNNs,
  U-Nets, and the Dice loss derived step by step — that this dossier cross-references as §N.

!!! intuition "The one-liner"
    Everything the CV claims is either in a committed JSON you can open, or falls out of one script
    on one command. A reproducible claim is a defensible claim.
