## 1. Orientation: what this note is and how to read it {#orientation}

This note answers one question in full: **why does fixing a data pipeline — while changing
nothing about the neural network — more than double a brain-tumour segmentation model's
accuracy on the classes that matter?** Answering it properly means crossing two fields, so
the note is built in four parts, each assuming only what came before.

![The project in one picture: the same 3D U-Net architecture is trained through two different pipelines; the difference in their test-set Dice is the deliverable.](figures/diagrams/00_overview.png)

- **Part I — Medical foundations** (§2–§8): what a brain tumour *is*, what a glioma is, what
  "post-treatment" changes on a scan, how MRI sees tissue, and what the five labels the model
  predicts actually mean anatomically. This is the *why* — the clinical problem the machine
  learning is trying to solve.
- **Part II — Deep-learning foundations** (§9–§18): segmentation as voxel classification, the
  convolution, the U-Net, the loss functions (with full derivations), class imbalance,
  normalization, evaluation metrics, training machinery, and how to design a trustworthy
  experiment. This is the *how*.
- **Part III — This project** (§19–§22): the dataset, the exact network, the A/B design and
  its nine deliberate defects (D1–D9), the results with error bars, and an honest reading of
  what they do and do not prove.

### How the boxes work

Three kinds of callout recur throughout. They are worth reading, not skipping:

!!! intuition "Intuition"
    The plain-English picture behind a formula or design choice — the thing you should
    remember even if the algebra fades.

!!! gotcha "Gotcha"
    A trap: a place where the obvious reading is wrong, a number that is easy to
    misinterpret, or an assumption that quietly fails. Several of this project's nine defects
    live in exactly these traps.

!!! example "Example question"
    A short worked problem to check understanding, with the full solution. Numbers are
    substituted step by step — nothing is left as "it can be shown that."

### A note on notation

Mathematics is rendered with real typesetting: an inline symbol like \(x\) or a displayed
equation like

\[ \mathrm{Dice}(P, G) = \frac{2\,|P \cap G|}{|P| + |G|} \]

appears the way it would in a textbook. Every variable is defined at its first use — here
\(P\) is the set of voxels the model predicts for a class, \(G\) is the set of ground-truth
voxels for that class, \(|\cdot|\) counts voxels, and \(P \cap G\) is their overlap. Do not
worry if that formula is opaque now; it is derived from scratch in §13.

!!! example "Example question"
    Suppose a model predicts a class in exactly the voxels where it truly occurs — a perfect
    prediction. What is the Dice score? **Solution:** if \(P = G\) then \(P \cap G = G\), so
    \(|P \cap G| = |G|\) and \(|P| = |G|\). Substituting,
    \(\mathrm{Dice} = \dfrac{2|G|}{|G| + |G|} = \dfrac{2|G|}{2|G|} = 1.\) A perfect overlap
    scores 1; we will see in §13 that no overlap scores 0.
