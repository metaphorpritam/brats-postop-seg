# CV Defense Dossier

## BraTS 2024: post-treatment glioma segmentation (3D U-Net)

*This document defends the project's CV entry in depth. The CV states the work simply — a
segmentation model, six data-pipeline fixes, a Dice score. This dossier shows the **rigorous method
behind each claim**: every number traced to a committed file, and the experimental design — a
controlled comparison with the network held identical — that makes "fixing 6 flaws recovered the rare
classes" a **causal** claim rather than a coincidence.*

### The CV entry being defended

> **BraTS 2024: post-treatment glioma segmentation (3D U-Net)**
>
> - Built a 3D U-Net to automatically segment brain-tumour regions from post-treatment MRI scans (BraTS).
> - Curated 700 post-treatment brain-tumour MRI scans (BraTS 2024) into train, validation and test sets.
> - Fixed 6 data-pipeline flaws causing the model to miss rare, heavily under-represented tumour classes.
> - Improved segmentation accuracy (Dice) to 0.67 on unseen scans; rare classes recovered from near zero.

!!! note "The CV is the summary; this is the proof"
    The résumé line is deliberately plain, for a fast read. Underneath, the fixes were validated as a
    **controlled experiment**: the *same* 1,983,069-parameter network, held **byte-identical**, was
    trained once with the flawed pipeline and once with the six fixes. Because nothing but the pipeline
    changed, the improvement — mean foreground Dice **0.349 → 0.670**, largest on the rarest class
    (NETC, near-zero → 0.466) — is attributable to the fixes, not to luck or a bigger model. That is
    what turns "I fixed some flaws" into "the flaws *caused* the failure, and here is the evidence."

### How the CV maps to this dossier

| CV claim | Defended in |
|---|---|
| Built a 3D U-Net to segment brain-tumour regions | §1 (the method), §3 (the exact network) |
| Curated 700 scans into train / validation / test | §2 (data & the split) |
| Fixed 6 data-pipeline flaws → rare classes recovered | §3 (the flaws & the frozen network), §1 (fidelity) |
| Dice 0.67; rare classes recovered from near zero | §4 (results & statistics) |
| the hardest interview questions, and where to check every number | §5 (cross-examination) · §6 (verification map) |

### How to read this

Three kinds of box recur — worth reading, not skipping:

!!! intuition "The one-liner"
    The single sentence that defends a claim if you remember nothing else.

!!! gotcha "Watch out"
    Where the claim is easy to *overstate* — the boundary you must not cross when talking about it.

!!! example "Interview question"
    The sharpest question the claim invites, with a crisp, honest answer.

A companion document — *the Explainer* (`explainer.html`) — carries the from-scratch theory (what a
glioma is, how MRI works, how a U-Net and the Dice loss are derived). This dossier cross-references it
as §N and stays focused on **defending the claims**.

!!! gotcha "Watch out — the metric caveat that governs the whole entry"
    Every Dice number here is **voxel-wise** Dice, *not* the BraTS lesion-wise challenge score. The
    numbers are internally consistent but are **not comparable to leaderboards**. The honest headline
    is the *recovery of the rare classes*, not the absolute 0.67 (see §4 and §5).

[TOC]
