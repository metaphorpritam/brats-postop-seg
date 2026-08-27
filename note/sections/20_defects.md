## 20. The A/B defect experiment: nine named flaws (D1–D9) {#defects}

Track A is not a strawman built to lose. It is a **faithful reproduction** of a real research
pipeline — the kind of TensorFlow/Keras code that circulates in tutorials and student projects —
and it fails in the specific, documented ways that code fails. Each failure is a named defect;
Track B fixes it; the architecture (§19) stays frozen so the fix is the only explanation for the
gain.

### The catalogue

| # | Defect (what Track A faithfully keeps) | Fix in Track B | Where it bites |
|---|---|---|---|
| **D1** | Bilinear `cv2.resize` applied to the *label* volume, then truncated to int | Nearest-neighbour label resampling | Invents/destroys minority voxels (evidence below) |
| **D2** | Per-class Dice computed with a shape/index bug — the original's own logs are untrustworthy | Correct MONAI `DiceMetric` with `get_not_nans` + per-class counts | Measurement (applied to both tracks) |
| **D3** | Class imbalance unhandled: plain cross-entropy, no balanced sampling | `DiceCELoss(include_background=False)` + class-balanced patch sampling | Model coasts on background (§14) |
| **D4** | Checkpoint chosen on **validation voxel accuracy** | Choose on **mean foreground Dice** | Picks a model that never learned the rare classes (§16) |
| **D5** | Global-max normalization $X/\max(X)$ | Per-modality z-score over brain voxels | Intensity scale is fragile & not standardized (§15) |
| **D6** | Non-uniform slice stride `int(j·2.5)` | Foreground crop + class-balanced 96³ patches | Which voxels the model ever sees (§17) |
| **D7** | A dead label remap that merges RC into ET (a 4-class contract) | Explicit 5-class contract asserted at load | The resection cavity silently disappears |
| **D8** | *Reference* had no normalization layers | Instance norm (MONAI default), on **both** tracks | Architecture (frozen, §19) |
| **D9** | Uncached loader re-reads + re-resizes every epoch | MONAI `PersistentDataset` cache | Speed: I/O-bound GPU starvation |

Track A reproduces the *data-pipeline* defects (D1, D3, D4, D5, D6, D9); D2 is a measurement fix
applied to **both** tracks so the numbers can be trusted at all; D7 is enforced as a contract; and
the network's normalization (D8) is **identical** on both sides — the MONAI U-Net uses instance norm,
the same for both tracks, so it can never be the confound §19's rule forbids.

![The defect ladder: each Track-A flaw (left) maps to a Track-B fix (right); the network in the middle is identical for both, so the measured gap is attributable to the pipeline.](figures/diagrams/s20_defect_ladder.png)

### Three defects worth seeing up close

**D1 — a resize that fabricates tumour.** An image can be smoothly interpolated; a *label map*
cannot. Suppose two adjacent voxels carry labels 1 (NETC) and 3 (ET). Bilinear interpolation
produces an in-between value on the way to a new grid:

$$v = \tfrac{1}{2}(1) + \tfrac{1}{2}(3) = 2$$

and truncation-to-int stores that as **label 2 (SNFH)** — a class that was in *neither* source
voxel. Across millions of boundary voxels, D1 both **invents** classes where they do not exist and
**erases** thin structures. It is not a rounding nuisance; it changes the ground truth the model is
trained and scored against. §21 measures it directly.

!!! gotcha "Gotcha — the sign of D1 is a Dice number that looks *good* for the wrong reason"
    Because D1 sprinkles minority labels along every boundary, Track A's "ground truth" reports the
    rare classes as present in almost every case. A naive reader sees "NETC present in 105/105 test
    cases" and assumes rich data; in fact the faithful labels show NETC in only 48. Always report a
    class's Dice **beside its case count** (§16), or D1-style fabrication hides in plain sight.

**D4 — grading the exam on the wrong question.** With ~98% of voxels being background, a model that
predicts "background everywhere" already scores ~98% voxel accuracy (§14). Selecting the checkpoint
that maximizes validation accuracy therefore rewards exactly the degenerate model that ignores
tumour. Track A checkpoints on accuracy and locks in an early, tumour-blind network (best epoch
18–22); Track B checkpoints on mean foreground Dice and keeps improving to epoch 64–72.

**D9 — benchmarking the disk, not the GPU.** Track A's loader re-reads each compressed volume and
re-runs the resize stack *every step of every epoch*, so it spends ~23 s/epoch moving data against
~5 s of actual compute — the GPU sits idle. Caching the preprocessed tensors (Track B) cuts data
loading to ~8 s, a ~2.8× speed-up on identical hardware. D9 costs no accuracy directly, but it is
the difference between an experiment that finishes and one that does not.

!!! intuition "Why bundle nine small flaws into one A/B?"
    Individually each defect looks minor — "it's just a resize", "just the selection metric". The
    experiment's lesson is that these mundane data-handling choices, none of which touches the
    network, together account for a doubling of foreground Dice. The interesting engineering is not
    in a fancier model; it is in *not corrupting the data before the model ever sees it*.

!!! example "Example question"
    A boundary voxel lies between label 3 (ET) and label 4 (RC). Under D1's bilinear resize +
    integer truncation, what label can it become, and why is that especially damaging here?

    **Solution.** The interpolated value is $\tfrac{1}{2}(3)+\tfrac{1}{2}(4)=3.5$, truncated to
    **3 (ET)**. So a resection-cavity boundary is relabelled as enhancing tumour. That is the worst
    possible confusion clinically (§7): a *treatment-created cavity* is being called *active
    tumour*. The fix — nearest-neighbour — instead copies whichever true label is closest, never
    manufacturing a third class.
