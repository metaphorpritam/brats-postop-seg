# CV Defense Dossier

## BraTS: Controlled A/B in post-treatment glioma segmentation

*This document defends one CV entry, phrase by phrase. Every noun, number, and qualifier in the
four bullets below is unpacked into: the **claim**, **what it means**, the **evidence** (a plot,
table, or statistic from the committed run logs), **where to verify it**, and the **interview
question** it invites — answered honestly, including where the work is deliberately limited.*

### The CV entry being defended

> **BraTS: Controlled A/B in post-treatment glioma segmentation**
>
> - Reproduced a defective glioma-segmentation pipeline as a controlled A/B to isolate cause from model.
> - Curated 700 BraTS-2024 post-treatment glioma MRIs (3 modalities) into a stratified 490/105/105 split.
> - Fixed 9 pipeline defects with a 1.98M-param MONAI 3D U-Net held byte-identical across the two tracks.
> - Lifted held-out mean foreground Dice from 0.349 to 0.670 (+0.321 ± 0.013, 3 seeds); rarest class +0.457.

!!! note "The verdict in one line"
    Every quantitative claim in the entry is reproducible from committed artifacts, and the central
    causal claim — *the pipeline, not the model, caused the gap* — is licensed by holding a
    1,983,069-parameter network **byte-identical** across both arms. The honest limitations
    (voxel-wise not lesion-wise Dice; a small model on a resampled mirror; n = 3 seeds on one test
    split) are stated in §5 rather than hidden, which is itself part of the defense.

### How to read this

Each of the four sections below takes one bullet and walks its phrases in order. Three kinds of box
recur:

!!! intuition "The one-liner"
    The single sentence that defends a phrase if you remember nothing else.

!!! gotcha "Watch out"
    Where the claim is easy to *overstate* — the boundary you must not cross when talking about it.

!!! example "Interview question"
    The sharpest question the phrase invites, with a crisp, honest answer.

A companion document — *the Explainer* (`explainer.html`) — carries the from-scratch theory
(what a glioma is, how MRI works, how a U-Net and the Dice loss are derived). This dossier
cross-references it as `§N` rather than repeating those derivations; it stays focused on **defending
the claims**.

!!! gotcha "Watch out — the metric caveat that governs the whole entry"
    Every Dice number here is **voxel-wise** Dice, *not* the BraTS lesion-wise challenge score. The
    numbers are internally consistent for an A/B comparison but are **not comparable to
    leaderboards**. Quote the *delta*, not the absolute level (see §4 and §5).

[TOC]
