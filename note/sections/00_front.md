# Post-Treatment Glioma Segmentation — A Ground-Up Explainer

*A self-contained companion to the BraTS post-treatment A/B reproduction project. It builds
from the biology of brain tumors and the physics-lite of MRI, through the deep-learning
machinery of 3D segmentation, to the specific controlled experiment run here and what its
numbers mean. It assumes no medical training and no deep-learning background — every term,
equation, and derivation is explained from first principles.*

!!! note "The result, in one sentence"
    A single 3D U-Net, held **byte-identical** between two pipelines, scores a mean
    foreground Dice of **0.349 ± 0.013** when fed a deliberately defective data pipeline
    (Track A) and **0.670 ± 0.002** when the defects are fixed (Track B) — an improvement of
    **Δ = +0.321 ± 0.013** on a held-out test set, largest on the rarest tumour class. The
    experiment's whole point is that this gap is caused by the *pipeline*, not the network.

[TOC]
