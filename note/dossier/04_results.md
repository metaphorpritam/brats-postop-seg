## 4. The results — Dice 0.67 and the recovered classes {#results}

This is the payoff line of the entry: a single number that says the nine fixes worked. Every token in it is load-bearing — *held-out* (not train), *foreground* (not the easy background), *mean over four classes* (not cherry-picked), *±0.013 over 3 seeds* (reproducible, not a lucky run), and *rarest class +0.457* (the hardest class moved the most). This section defends each in turn, and — because honesty is part of the defense — states plainly what the number is **not**: it is a voxel-wise, single-split, small-model result, and the *delta* is the claim, never the *level*.

!!! gotcha "Watch out — the caveat that governs the whole number"
    Every Dice figure below is **voxel-wise** Dice, not the BraTS **lesion-wise** challenge score. The numbers are internally consistent for an A/B comparison but are **not comparable to leaderboards**. Quote the *delta* (+0.321), not the *level* (0.670). See §5 for the full limitations ledger.

### "held-out"

**Claim: "held-out" mean foreground Dice.** The result is measured on the **test split of 105 cases** that was carved out at the very start and never touched again — not for training, not for early stopping, not for checkpoint selection, not for hyper-parameter choices.

**What it means.** The 700 curated cases were partitioned once, deterministically at **seed 42**, into three disjoint sets: **490 train / 105 validation / 105 test** (70/15/15), stratified by resection-cavity presence so that RC appears in **82.9%** of cases in *all three* partitions identically (see §2 for the stratification defense). The three partitions play three non-overlapping roles:

| Partition | Cases | Role | Touched during model choice? |
|---|---|---|---|
| Train | 490 | Fit network weights | Yes — gradients |
| Validation | 105 | Select the best epoch / checkpoint | Yes — model selection |
| **Test** | **105** | **Report the final number** | **No — opened once, at the end** |

Because the test set influenced *nothing* — no weight, no epoch pick, no threshold — its Dice is an **honest generalization estimate**: it answers "how does this model do on cases it has never had any causal contact with?" A number computed on train would measure memorization; a number computed on validation would be optimistically biased because validation *chose* the checkpoint. Only the held-out test set is free of both leaks. Critically, this discipline is applied **identically to both tracks** — Track A and Track B are scored on the same 105 cases — so the comparison is fair by construction.

**Verify.** The frozen split is committed at `reports/splits.json` (seed 42, the 105 test IDs enumerated); per-seed test scores are in `reports/eval_a_s{0,1,2}.json` and `reports/eval_b_s{0,1,2}.json`, aggregated in `reports/dossier_facts.json`.

!!! example "Interview question"
    **"How do I know you didn't tune on the test set?"** The split is fixed once at seed 42 and written to `reports/splits.json` before any training; checkpoint selection reads *only* validation Dice (`04_evaluate.py` selects on val mean-fg — that is the D4 fix), and the 105 test IDs are loaded only by the final evaluation pass. There is no code path in which a test case affects a weight, an epoch, or a threshold. You can diff the test IDs against the train/val IDs in `splits.json` and confirm the three sets are disjoint.

### "mean foreground Dice"

**Claim: "mean foreground Dice".** The headline metric is the **Dice coefficient**, averaged over the **four tumour (foreground) classes**, with the background class excluded.

**What it means.** Dice measures spatial overlap between the predicted mask $P$ and the ground-truth mask $G$ for a class:

$$\text{Dice} = \frac{2\,|P \cap G|}{|P| + |G|} \in [0, 1],$$

where 1 is perfect overlap and 0 is none (the full derivation and its relation to the F1 score live in the explainer §13). Two qualifiers make the metric honest:

- **"foreground"** = exclude class 0 (background). Background is ~99% of every volume, so including it would let a model that predicts "all background" score enormously well — exactly the pathology behind Track A's **0.991 voxel accuracy** (§3, D4). Foreground Dice refuses that free lunch.
- **"mean"** = the unweighted average over the four tumour classes {NETC, SNFH, ET, RC}. Unweighted means a rare class counts as much as a common one — so a model cannot hide a failed rare class behind a strong common one.

The worked class-averages that produce the two headline numbers:

$$\text{Track A} = \frac{0.009 + 0.683 + 0.299 + 0.403}{4} = \frac{1.394}{4} = 0.349,$$

$$\text{Track B} = \frac{0.466 + 0.870 + 0.650 + 0.694}{4} = \frac{2.680}{4} = 0.670.$$

**Verify.** Metric is MONAI's `DiceMetric` with background excluded (the D2 fix — the original per-class Dice had a shape/index bug); the four per-class means feeding each average are in `reports/dossier_facts.json` under `per_class`, and the aggregate `A_mean = 0.349`, `B_mean = 0.670` under `stats`.

!!! intuition "The one-liner"
    Foreground-mean Dice is the metric that *can't be gamed by the background* and *can't hide a dead rare class* — which is precisely why Track A scores 0.349 on it despite 0.991 voxel accuracy.

### "from 0.349 to 0.670"

**Claim: "from 0.349 to 0.670".** Averaged over 3 seeds on the 105-case test set, mean foreground Dice rose from **Track A 0.349 ± 0.013** to **Track B 0.670 ± 0.002** — an absolute lift of **+0.321**.

**What it means.** The lift is not uniform across classes; the per-class breakdown shows *where* the fixes bought their gains. Mean ± SD over the 3 seeds:

| Class | Track A (mean ± SD) | Track B (mean ± SD) | Δ |
|---|---|---|---|
| NETC (necrotic core) | 0.009 ± 0.008 | 0.466 ± 0.004 | **+0.457** |
| SNFH (oedema / FLAIR) | 0.683 ± 0.017 | 0.870 ± 0.003 | +0.187 |
| ET (enhancing tumour) | 0.299 ± 0.033 | 0.650 ± 0.007 | +0.351 |
| RC (resection cavity) | 0.403 ± 0.026 | 0.694 ± 0.002 | +0.291 |
| **Mean foreground** | **0.349 ± 0.013** | **0.670 ± 0.002** | **+0.321** |

Two things stand out. First, **every** class improves — this is not a trade where one class is sacrificed for another. Second, Track B's per-seed SDs collapse (0.002–0.007) relative to Track A's (0.008–0.033): the fixed pipeline is not just better on average, it is far more **stable** across seeds.

![Per-class TEST Dice with ±SD error bars, Track A vs Track B](reports/figures/report/perclass_test_dice.png)

![Per-class A→B delta on the held-out test set](reports/figures/report/delta_test.png)

Numbers convince the head; overlays convince the eye. The qualitative panels below show, per case, the ground-truth mask against Track A's and Track B's predictions on held-out slices — the collapse of Track A's rare-class structure and its recovery under Track B are visible directly.

![Qualitative overlays, set 1 — GT vs Track A vs Track B on held-out cases](reports/figures/combined_overlay.png)

![Qualitative overlays, set 2 — GT vs Track A vs Track B on held-out cases](reports/figures/combined_overlay_2.png)

**Verify.** Per-class and aggregate means in `reports/dossier_facts.json`; per-seed source scores in `reports/eval_{a,b}_s{0,1,2}.json`; the figures are regenerated deterministically by `scripts/dossier_figures.py`; overlays are rendered from the committed test predictions.

!!! example "Interview question"
    **"0.67 is far below the 0.9 you see on leaderboards — isn't this weak?"** Three answers. (1) These are **voxel-wise** Dice, not the BraTS **lesion-wise** score the leaderboards report — different denominators, not comparable. (2) The model is a **deliberate handicap**: a 1.98M-param U-Net, three modalities, on a resampled 182×218×182 mirror, held byte-identical across both arms so that the *only* thing that changes is the pipeline. (3) The result is **the delta, not the level** — +0.321 with the network frozen is causal evidence about the pipeline; pushing the absolute number to 0.9 would mean a bigger model and lesion-wise scoring, which would *destroy* the controlled comparison. The level is intentionally low so the delta can be clean.

### "(+0.321 ± 0.013, 3 seeds)"

**Claim: "(+0.321 ± 0.013, 3 seeds)".** The lift is **+0.321**, and the **±0.013** is the standard deviation of the delta computed across **3 independent training seeds**.

**What it means.** For each seed $i \in \{0,1,2\}$ both tracks were trained from scratch and the paired difference $\Delta_i = B_i - A_i$ computed on the same test set. The three per-seed deltas are:

$$\Delta = [0.332,\; 0.325,\; 0.307], \qquad \bar{\Delta} = 0.321, \qquad \mathrm{SD} = 0.013.$$

The full inferential statistics on the paired deltas (standard error $\text{SE} = \text{SD}/\sqrt{3} = 0.01286/\sqrt{3} = 0.0074$, using the unrounded SD):

- **95% t-CI** (df = 2, $t_{\text{crit}} = 4.303$): $0.321 \pm 4.303 \times 0.0074 = [0.290,\; 0.353]$ — **excludes 0**.
- **Paired t-test:** $t(2) = 43.3$, $p = 0.0005$ (two-sided).
- **Effect size:** Cohen's $d = 25$.

![Per-seed delta with 95% CI and paired-t annotation](reports/figures/dossier/delta_ci.png)

An effect size of 25 and a CI that clears zero by a wide margin say the improvement is not seed noise. **But** — and this is the honest boundary — these statistics describe **reproducibility across seeds on one fixed test split**, not generalization to a new population. The *t*-test's "sample" is three random seeds, so it answers "would I get this lift again if I re-ran training?" (yes, overwhelmingly). It does **not** answer "would this lift hold on a different 105 patients?" — for that the relevant, un-quantified uncertainty is the **single fixed test set** itself. With one split, that population variance is unestimated; the tight CI is a statement about training determinism, not about patients.

**Verify.** Every statistic is computed in `scripts/compute_dossier_facts.py` and dumped to `reports/dossier_facts.json` (`per_seed_delta`, `delta_mean`, `delta_sd`, `ci95`, `paired_t`, `p_two_sided`, `cohens_d`); the per-seed inputs are the six `reports/eval_{a,b}_s{0,1,2}.json` files.

!!! gotcha "Watch out — what n = 3 does and does not buy"
    The n in "n = 3" is **seeds, not patients**. It licenses "the lift reproduces across training runs," not "the lift generalizes to new populations." Never let the small, tidy CI be read as clinical-grade uncertainty — the real uncertainty lives in the single 105-case split, and it is not quantified here.

!!! example "Interview question"
    **"n = 3 is tiny — is +0.321 real?"** As a claim about *reproducibility*, yes and emphatically: three from-scratch reruns give deltas of 0.332, 0.325, 0.307 — a 95% CI of [0.290, 0.353] that clears zero, and Cohen's d = 25. But I'd correct the framing: n = 3 is small *for a population claim*, and I don't make one. The honest statement is "the lift is robust to seed"; the un-estimated risk is the fixed test split, which I flag rather than paper over (§5). If I wanted a population claim I'd need k-fold or multiple splits, which I state as future work, not as a result.

### "rarest class +0.457"

**Claim: "rarest class +0.457".** The **NETC** class (non-enhancing tumour core, the necrotic core) — the rarest of the four — improved the most, from **0.009 to 0.466**, a gain of **+0.457**.

**What it means.** NETC is rare in two senses. Physically, on the faithfully-labelled test set it is **GT-present in only 48 of 105 cases**, and where present it is often a handful of voxels. That rarity is exactly why Track A scores essentially **0.009** — a near-total failure. Track A's loss is plain cross-entropy (defect D3): a per-voxel loss has **no incentive** to find a few dozen NETC voxels when predicting "background/oedema" everywhere already drives the loss down. The gradient signal from a rare class is swamped. The rarest class is therefore the one a badly-configured pipeline abandons first.

The fixes target precisely that failure mode: **DiceCELoss** (which is overlap-based and does not vanish for a small structure) plus **class-balanced 96³ patch sampling** (D3/D6), which oversamples patches containing rare classes so NETC actually appears in the gradient. Because Track A started near zero, it also has the **most room** to recover — hence the rarest class posts the *largest* delta. The mechanism and the magnitude line up: the class the plain pipeline neglected is the class the fixes were built to rescue.

There is a second, sharper twist that ties back to D1. Track A's labels are resized with **bilinear `cv2.resize`** on the integer label volume, which interpolates class indices and *invents* voxels. The consequence for counting:

| Class | Track A (bilinear) reports GT-present | Faithful (nearest-neighbour) GT-present |
|---|---|---|
| NETC | **105 / 105** | **48 / 105** |
| ET | 100 / 105 | 85 / 105 |
| SNFH | 105 / 105 | 105 / 105 |
| RC | 87 / 87 | 87 / 87 |

![D1 count inflation — bilinear resize invents NETC and ET labels](reports/figures/report/count_inflation.png)

So Track A doesn't merely *segment* NETC badly — its very inputs claim NETC is present in **all 105** cases when the truth is **48**. The pipeline is being scored against a label field that has been *inflated* by an interpolation bug, which is part of why its NETC Dice is a meaningless 0.009 rather than an honestly-failed one. The D1 fix (nearest-neighbour resampling) is what makes the 48 the real denominator, and the class-balanced loss is what lets Track B reach 0.466 against it.

**Verify.** NETC A→B means and the `countA = 105` vs `countB = 48` inflation are in `reports/dossier_facts.json` under `per_class.NETC`; the label-count comparison is in `reports/labels_summary.json`; the D1 mechanism sits at the bottom of the defect ladder (§ explainer, `figures/diagrams/s20_defect_ladder.png`).

!!! example "Interview question"
    **"Why does the rarest class gain the most?"** Two compounding reasons. Mechanistically, plain cross-entropy (Track A's loss) has no incentive to find a handful of NETC voxels — it minimizes just as well by predicting background everywhere — so NETC collapses to 0.009; the fixes (DiceCELoss + class-balanced patch sampling) are aimed at exactly that failure, and a class starting near zero has the most headroom, so it posts the biggest delta (+0.457). Additionally, Track A's bilinear label resize (D1) *inflates* NETC to appear in 105/105 cases when it truly appears in 48 — so the fix isn't only better segmentation, it's scoring against honest labels. The rarest class is the one a broken pipeline neglects first and a fixed pipeline recovers most.
