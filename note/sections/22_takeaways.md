## 22. Takeaways, limitations, and where this sits {#takeaways}

### What the experiment proves

Held to its own terms, the result is clean and strong: **with the network frozen byte-for-byte, a
corrected data pipeline nearly doubles mean foreground Dice (Δ = +0.321 ± 0.013), and the gain
concentrates on the rare tumour classes the defects specifically suppress.** Because the only things
that changed were the label resampling, normalization, sampling, loss, and selection metric (§20),
the improvement is attributable to *data handling*, not to model capacity. That is the entire point:
the leverage was in not corrupting the data, and it is invisible to anyone who only tunes
architectures.

### What it does *not* prove — read these before quoting a number

!!! gotcha "Six honest limitations"
    - **Voxel-wise, not lesion-wise Dice.** BraTS scores each *connected lesion* separately and adds
      a boundary (Hausdorff) term (§16). Ours is plain voxel-wise Dice — correct and internally
      consistent for an A/B, but a *different number on identical predictions*. Never compared to a
      leaderboard here.
    - **A deliberately tiny model.** ~1.98M parameters, no test-time augmentation, no ensembling,
      25/80 epochs, 3 channels, an 8 GB laptop GPU. Absolute Dice is **low by design** — the A→B
      delta carries the same handicap on both sides, so the delta, not the level, is the result.
    - **A resampled community mirror.** 700 of ~1,350 official cases at 182×218×182 (MNI-like), not
      native 240×240×155. Fine for A/B; cite the official BraTS 2024 challenge for anything absolute.
    - **Flat 5-class, not merged clinical regions.** A pre-operative paper's "whole-tumour Dice 0.90"
      and our "SNFH Dice 0.870" are not on speaking terms — different targets on different data.
    - **The reference-code contradiction (D7).** The committed reference is genuinely *4-class* (it
      merges the resection cavity into enhancing tumour); this project deliberately keeps all five
      classes so RC can be measured at all. The discrepancy is documented, not smoothed over.
    - **The delta is a difference of means, not a formal test.** Three seeds give a tight SD (±0.013),
      but this is a stability estimate, not a paired significance test. The honest claim is "large,
      reproducible, and concentrated on the rare classes," not "$p < 0.05$".

### Where this sits in neuro-oncology

This is a **methodology demonstration**, not a clinical tool. Real deployment-grade brain-tumour
segmentation (§8) uses large self-configuring ensembles (nnU-Net-style) trained on the full dataset,
scored lesion-wise, and validated far more heavily. What this project contributes is upstream of any
of that: a controlled, reproducible demonstration that *how you feed the data* can matter as much as
*what model you feed it to* — the failure mode that silently caps countless well-intentioned
pipelines before the model architecture is ever the bottleneck.

![Where this project sits: it is a controlled methodology study on data-pipeline quality, one input to — not a replacement for — clinical-grade segmentation systems.](figures/diagrams/s22_where_it_sits.png)

!!! intuition "The transferable lesson"
    Most of the accuracy a beginner "leaves on the table" in a segmentation project is lost *before*
    the model runs — in a careless resize, a global normalization, an accuracy-based checkpoint. Fix
    the pipeline first; reach for a bigger network last. The nine defects here are a checklist as much
    as an experiment.

!!! example "Example question"
    An interviewer says: "You only reached 0.67 Dice — a competition model gets 0.9. Isn't your work
    weak?" Give the two-sentence defensible answer.

    **Solution.** "The 0.9 figures are lesion-wise Dice from large ensembles on the full native
    dataset; my 0.67 is voxel-wise Dice from a 2M-parameter model on a resampled 700-case mirror, so
    the numbers aren't comparable by construction. My result isn't the 0.67 — it's the *controlled
    +0.321 delta from fixing the data pipeline with the architecture held identical*, which isolates
    a cause the absolute score can't." That reframes the question from *level* to *causal claim*,
    which is what the experiment was built to support.

### Closing

You now have the whole chain: a tumour is a mass in a closed box (§2); gliomas are graded by biology
(§3–§4); treatment reshapes the scan (§5); MRI sees it through four sequences (§6) mapped to five
labels (§7) that clinicians need measured (§8). A U-Net (§9–§11) turns those voxels into class
probabilities via softmax and cross-entropy (§12), is judged and trained through Dice (§13) under
brutal class imbalance (§14), with careful normalization (§15), honest metrics (§16), and real 3D
training machinery (§17). Wrap that in a controlled A/B (§18) on this exact dataset and network
(§19), enumerate the nine defects (§20), and the numbers (§21) tell a single story: **fix the
pipeline, free the model.**
