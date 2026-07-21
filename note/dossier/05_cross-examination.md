## 5. Cross-examination: the hard questions, answered {#cross-examination}

The four preceding sections defended the CV entry phrase by phrase. This one does the opposite job: it *invites* the attack. An interviewer who knows the field will not ask you to explain your method — they will try to break it. Below are the seven sharpest objections this project draws, each answered honestly, with the exact number, the figure, and the committed artifact where it can be checked. Where the honest answer is a limitation, it is stated as a limitation and not smoothed over. The governing principle throughout: **the deliverable is a controlled +0.321 delta with the architecture held identical, not an absolute Dice level** — so most attacks on the absolute level miss the claim, and the two attacks that land (single test split; a resampled mirror) are conceded openly.

### "0.67 Dice is low — a competition model gets 0.9"

**The claim under attack** is the headline number: Track B reaches **mean foreground Dice 0.670 ± 0.002** on the held-out test set (n_test = 105), lifted from Track A's **0.349 ± 0.013**, a delta of **+0.321 ± 0.013**. Against a challenge leaderboard reporting ~0.9, 0.67 looks weak.

**What the objection misunderstands** is which quantity is the product. Three facts make the absolute level non-comparable, and none of them touch the delta. First, **the metric is voxel-wise Dice, not the BraTS lesion-wise score** — a different denominator (per-lesion connected-component matching versus per-voxel overlap), so it is internally consistent for an A/B but cannot be laid next to a leaderboard number. Second, the setup is deliberately austere: a single MONAI 3D U-Net of **1,983,069 trainable parameters**, **no test-time augmentation, no ensembling**, fits inside an **8 GB RTX 4060 laptop GPU**, trained on a **resampled 700-case community mirror** at 182×218×182 rather than a curated native-resolution challenge corpus. Competition scores of 0.9 come from large ensembles with TTA on native-geometry data — a different object entirely. Third, and decisively, **the architecture is held byte-identical across both tracks**, so the +0.321 is attributable to the pipeline fixes alone; the model is a *constant*, not a variable, and its absolute ceiling is irrelevant to a causal claim about the pipeline.

The parameter budget makes the "small model" point concrete rather than rhetorical:

![Parameter budget of the shared 1.98M-parameter U-Net, broken down by feature-width block](reports/figures/dossier/param_breakdown.png)

And the deliverable — the delta with its interval — is the thing to look at, not the height of either bar:

![Track A → Track B delta with 95% t-CI (df=2) and paired-t annotation](reports/figures/dossier/delta_ci.png)

**Verify:** the absolute levels and the delta are recomputed in `scripts/compute_dossier_facts.py`; the parameter count is asserted at model-build time (`out_channels = 5`, `channels (16,32,64,128,256)`, `num_res_units = 0`) and reproduced in `reports/figures/dossier/param_breakdown.png`.

!!! example "Interview question"
    *"Your best model is 0.67 and the leaderboard is 0.9 — isn't this just a weak result?"*
    The 0.67 is voxel-wise Dice from a 1.98M-parameter single model with no TTA or ensembling on an 8 GB RTX 4060 laptop GPU over a resampled mirror — it is not, and never claims to be, comparable to a lesion-wise leaderboard score. The product here is the **+0.321 delta measured with the architecture held identical across both arms**, which isolates the pipeline as the cause. If I wanted a higher absolute number I would ensemble and use native geometry; that would raise both tracks and leave the causal claim unchanged.

### "Track A is a strawman built to lose"

**The claim under attack** is that Track A is a fair baseline rather than a punching bag assembled to make the delta look big.

**What the objection gets right in spirit** is the burden it places on me: a controlled A/B is only worth anything if the "before" arm faithfully reproduces a *real* pathology rather than a caricature. So the defense is traceability. Track A reproduces **six of nine defects that were traced to specific lines of the reference code** — D1 (bilinear `cv2.resize` on the integer label volume), D3 (plain cross-entropy, class imbalance unhandled), D4 (checkpoint selected on voxel accuracy), D5 (global-max normalisation), D6 (non-uniform `int(j·2.5)` slice stride), and D9 (uncached I/O-bound loader). These are not inventions; each maps to a line in `unet_cc.py`.

**Track A is then validated as faithfully reproducing the pathology**, and the signature is unmistakable: **validation voxel accuracy 0.991 against validation mean-foreground Dice 0.320, and NETC Dice of just 0.009** — a model that looks excellent on the metric the reference selected on (accuracy, defect D4) while being nearly blind to the rarest class. That divergence *is* the pathology, reproduced:

![Track A's accuracy-vs-Dice pathology: 0.991 val accuracy masks 0.320 mean-fg Dice and 0.009 NETC](reports/figures/dossier/track_a_fidelity.png)

The bilinear-resize defect (D1) leaves an equally concrete fingerprint — it *invents* labels that are not in the ground truth. On the GT-present test cases, Track A's bilinear-resized labels report NETC in **105/105** and ET in **100/105**, whereas the faithful nearest-neighbour labels show only **48** and **85**:

![D1 count inflation: bilinear resize invents NETC (105 vs 48) and ET (100 vs 85) presence](reports/figures/report/count_inflation.png)

**The strongest rebuttal is that I was, if anything, *more generous* to the reference than a strawman-builder would be.** The reference's single worst defect is D7: it carries a BraTS-2020-era label remap that **merges the resection cavity (RC) into ET, making it a 4-class model**. Reproducing that would have inflated the delta the most — but it would also **change the network's output channels (5 → 4) and break the hold-architecture-constant rule**, so I *enforced* the 5-class contract on both tracks (`assert` at data load) and did **not** reproduce D7's collapse. The consequence is that Track A keeps a fighting chance on RC that the true reference would not have had. **So the +0.321 delta is, if anything, understated** relative to a faithful full reproduction of the reference.

**Verify:** the nine defects and their fix-status (`D1,D3,D4,D5,D6,D9` reproduced; `D2` fixed on both; `D7` enforced 5-class; `D8` held) are enumerated in the defect ladder and the per-defect notes; the fidelity numbers live in the Track A validation logs and `reports/figures/dossier/track_a_fidelity.png`; the count-inflation audit is `reports/figures/report/count_inflation.png`.

!!! gotcha "Watch out — the generosity cuts one way"
    Not reproducing D7 makes Track A *stronger*, not weaker. If challenged that this "hides" a defect, the honest reply is the reverse: it shrinks my own headline delta, and I accept that cost to keep the architecture — and therefore the causal claim — clean.

!!! example "Interview question"
    *"Did you just build a baseline designed to fail?"*
    Every defect Track A carries is traced to a line in the reference `unet_cc.py`, and Track A reproduces the exact pathological signature — 0.991 accuracy masking 0.320 Dice and 0.009 on NETC. The one place I deviated made the baseline *stronger*, not weaker: I refused to reproduce the reference's 4-class RC→ET collapse because it would break the identical-architecture constraint, which means my +0.321 delta is understated relative to the real reference, not inflated.

### "n = 3 seeds is statistically weak"

**The claim under attack** is the ± term: **±0.013** on the delta comes from only **three training seeds**.

**What the ±0.013 actually measures** must be stated precisely, because conceding too much here is as dishonest as conceding too little. The SD captures **seed-to-seed variance** — the run-to-run instability of stochastic training (weight init, patch sampling, augmentation order). The per-seed numbers are tight: Track A mean-fg **[0.340, 0.342, 0.364]**, Track B **[0.672, 0.668, 0.671]**, and the per-seed **delta [0.332, 0.325, 0.307]** — three positive, well-separated deltas with no overlap between the arms on any seed. What the SD does **not** measure is **population generalization**: all three seeds are evaluated on **one fixed 105-case test split**. Three seeds tell you the result is reproducible under retraining; they do not turn one test set into many.

![Per-seed deltas cluster tightly (0.307–0.332); the interval is over seeds, not over test sets](reports/figures/dossier/delta_ci.png)

**The integrity safeguard** is that the seeds were **fixed in advance and applied identically to both arms** — the same three seeds train Track A and Track B, so there is no room for seed-selection (running many seeds and reporting the flattering ones). Both arms drew from the same seed set; the comparison is paired at the seed level.

**Verify:** the per-seed vectors and the seed list are emitted by `scripts/compute_dossier_facts.py`; the interval and its derivation are in `reports/figures/dossier/delta_ci.png`.

!!! gotcha "Watch out — do not oversell the interval"
    ±0.013 is a *seed* interval on *one* test split. It is honest to say "reproducible across seeds"; it is dishonest to say "generalizes to the population." The real uncertainty is the single 105-case test set — carried into the next answer.

!!! example "Interview question"
    *"Three seeds — how can you claim anything statistically?"*
    Three seeds measure reproducibility across retraining, and there the signal is clean: per-seed deltas of 0.307, 0.325, 0.332, with no arm overlap on any seed. They do **not** measure population generalization — that would need many test sets, and I have one fixed 105-case split. The seeds were fixed before the runs and applied identically to both arms, so there is no seed-selection; the honest bound is one dataset, which I state plainly rather than dress up as external validity.

### "It's a difference of means, not a hypothesis test"

**The claim under attack** is that "+0.321" is just a gap between two averages with no inferential backing.

**What the objection deserves is a concession followed by the support.** I concede the framing: the headline is a difference of means. But it is not *only* that. Treating the three seeds as paired observations of the delta gives a **paired t(2) = 43.3, p = 0.0005 (two-sided), Cohen's d = 25**, with **SE = 0.0074** and a **95% t-CI (df = 2) of [0.290, 0.353] that excludes zero**. Those are supportive statistics, and they are enormous — but their size is exactly why they must be read carefully.

![Delta with 95% t-CI [0.290, 0.353] and the paired t(2)=43.3, p=0.0005 annotation](reports/figures/dossier/delta_ci.png)

**The honest reading** is that the tiny p-value and d = 25 reflect how *little seed noise* there is, not how broadly the result generalizes. A t-test on three seeds answers "is the delta reliably non-zero across retraining?" — yes, overwhelmingly. It does **not** answer "does the delta hold on a new cohort?" **The real limitation is the single 105-case test set, not the seed count.** I would rather state that a d of 25 is an artifact of low within-arm variance on one split than let an interviewer assume it means population-level certainty.

**Verify:** t, p, d, SE, and the CI are all computed in `scripts/compute_dossier_facts.py` and rendered in `reports/figures/dossier/delta_ci.png`.

!!! example "Interview question"
    *"That's a difference of means, not a test — where's your inference?"*
    Fair framing, and I add the inference: paired t(2) = 43.3, p = 0.0005, Cohen's d = 25, 95% CI [0.290, 0.353] excluding zero. But I read those honestly — a d of 25 reflects tiny seed-to-seed noise, not broad generalization. The binding limitation is the single fixed test split, not the number of seeds, and I would not let the p-value be mistaken for external validity.

### "A community mirror in MNI-like space isn't real BraTS data"

**The claim under attack** is the data provenance: the 700 cases come from a **Kaggle mirror (`i212385nomanarif/2024-brats-glioma`)** at geometry **182×218×182 — a resampled, MNI-like grid, not the native 240×240×155** of the official release.

**What defuses the objection for the A/B specifically** is that **both tracks consume the identical data** — same mirror, same split, same geometry. Any distributional quirk of the mirror is a *shared constant* across the two arms, so it cannot manufacture a delta; it cancels. The controlled comparison is therefore unaffected by the choice of mirror.

**What I concede, and cite properly, is the absolute framing.** For anything stated in absolute terms I point to the official source rather than the mirror: **BraTS 2024 post-treatment glioma (BraTS-GLI), de Verdier et al., arXiv:2405.18368**. And I flag the geometry as a provenance caveat: the 182×218×182 grid is resampled, so absolute numbers are not on native-space footing. There is a positive check that the mirror is genuinely the 2024 post-treatment vintage and not a mislabeled older set: **ET (label 3) is present in 579/700 cases**, which rules out a pre-operative {0,1,2,4} labelling and confirms the post-treatment 2024 contract. The stratification that the split rests on is also visible and identical across arms:

![Resection-cavity presence stratified to 82.9% in train, val, and test alike](reports/figures/dossier/stratification.png)

**Verify:** the mirror ID, geometry, split, and the ET-presence label-contract check (579/700) are in the data-provenance notes and `scripts/compute_dossier_facts.py`; the official corpus is de Verdier et al., arXiv:2405.18368.

!!! gotcha "Watch out — mirror vs official"
    Never cite the Kaggle mirror for an *absolute* claim about BraTS. The mirror licenses the *A/B* (both arms share it); the official de Verdier arXiv:2405.18368 corpus is the citation for anything absolute, with the 182×218×182 resampling flagged.

!!! example "Interview question"
    *"That's a resampled community mirror, not real BraTS — doesn't that invalidate your numbers?"*
    For the A/B it changes nothing: both tracks eat the identical mirror at the identical 182×218×182 geometry, so any mirror quirk is a shared constant that cancels out of the delta. For absolute framing I concede the point and cite the official BraTS 2024 corpus (de Verdier, arXiv:2405.18368), flagging the resampled geometry. I also verified the mirror is the genuine 2024 post-treatment vintage — ET present in 579/700 cases rules out an older pre-op label set.

### "Your D7 says the reference is 4-class — did you reproduce the wrong thing?"

**The claim under attack** targets defect D7 directly: if the reference is a **4-class** model and I built a **5-class** one, did I reproduce something the reference never was?

**What the objection inverts is the direction of the deviation.** The reference's D7 is a **dead label remap that merges RC into ET**, i.e. it trains on a **4-class contract** and never predicts the resection cavity as its own class. I deliberately did the opposite: I **kept all five classes** (0 background, 1 NETC, 2 SNFH, 3 ET, **4 RC**) and enforced a 5-class assertion at data load, precisely **so that RC can be measured** rather than silently folded away. This is not an accidental mismatch — it is a documented, intentional choice, because collapsing RC into ET would (a) hide the single class the post-treatment task most cares about and (b) change `out_channels` from 5 to 4 and break the identical-architecture constraint that licenses the whole causal claim.

The payoff is that RC becomes a *measurable* result instead of an invisible one: **RC Dice 0.403 → 0.694, a +0.291 improvement**, alongside the rarest class NETC at **+0.457** (0.009 → 0.466):

![Per-class test Dice ±SD across both tracks, including RC which the 4-class reference could not report](reports/figures/report/perclass_test_dice.png)

![Per-class A→B deltas: NETC +0.457, ET +0.351, RC +0.291, SNFH +0.187](reports/figures/report/delta_test.png)

**So the answer is the reverse of the accusation:** I did not reproduce the *wrong* thing — I refused to reproduce the reference's worst behavior, documented that refusal as D7, and thereby gained the ability to score RC at all. The 4-class nature of the reference is written down as a known discrepancy, not papered over.

**Verify:** the 5-class enforcement (`out_channels = 5`, load-time assertion) and the RC/NETC per-class deltas are in `scripts/compute_dossier_facts.py` and `reports/figures/report/perclass_test_dice.png` / `delta_test.png`; D7's status ("enforced") is recorded in the defect ladder.

!!! example "Interview question"
    *"If the reference is 4-class and you're 5-class, did you reproduce the wrong pipeline?"*
    I reproduced the reference's *defects* but deliberately declined its worst one: D7 merges RC into ET as a 4-class model, and I kept all five classes so RC is a measurable class rather than folded into ET. That choice is documented, not smoothed over — and it is what lets me report RC at +0.291 and keeps the network's output channels identical across both arms, which the causal claim requires.

### "Did you reuse the original's trained weights?"

**The claim under attack** is the most damaging one if true: that the "reproduction" leaned on the reference's own trained parameters.

**The answer is a flat no, and the reasons make reuse indefensible even if it were possible.** The network was **retrained from scratch in PyTorch/MONAI** for both tracks. The reference was a **Colab export** — `unet_cc.py`'s header reads *"Copy of Untitled7.ipynb … Automatically generated by Colab"* — built on a **different TensorFlow stack**; its weights are not architecturally transferable to the MONAI 3D U-Net used here, and even a nominal port would be scientifically indefensible because it would confound "reproduced pipeline" with "inherited a trained model." The provenance record is explicit that **no original weights were reused**; only the *behaviour* was reproduced, and the reference — which has **no author, no README, no LICENSE** — was **not redistributed**.

**Verify:** the from-scratch training entry points and per-seed checkpoints are the run artifacts behind `reports/figures/report/training_curves.png` (Track B reaching val mean-fg 0.682 at epochs 64–72; Track A's accuracy-selected checkpoint at epochs 18–22); the provenance statement — Colab export, no author/README/LICENSE, weights never reused, code never redistributed — is in the provenance note.

![Training curves (mean ± SD over seeds) for both tracks — both trained from scratch, no inherited weights](reports/figures/report/training_curves.png)

!!! intuition "The one-liner"
    Nothing was inherited: a different (TensorFlow, authorless, unlicensed) reference was re-implemented from scratch in PyTorch, so the +0.321 measures a rebuilt pipeline, never a borrowed model.

!!! example "Interview question"
    *"Did you start from the original's trained weights?"*
    No — both tracks were trained from scratch in PyTorch/MONAI. The reference was an authorless, unlicensed Colab export on a different TensorFlow stack; its weights are not transferable and reusing them would confound "reproduced the pipeline" with "inherited a model," which would invalidate the entire A/B. Only the behaviour was reproduced, and the reference code was never redistributed.
