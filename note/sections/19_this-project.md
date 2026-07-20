## 19. This project: the dataset, the network, and the fixed design {#this-project}

Everything up to here was general. Now we pin down the exact experiment: what data went in,
what network processed it, and the one design rule that makes the result mean something.

### The data

The cases come from a community **Kaggle mirror** of the BraTS 2024 *post-treatment* glioma
dataset (BraTS-GLI). Each case is a set of co-registered 3D MRI volumes plus an expert
segmentation. This project uses **700 cases** and the **three** input modalities the original
reference code kept — **t2f** (T2-FLAIR), **t1c** (post-contrast T1), and **t2w** (T2) — dropping
the native T1 (see §6 for what each sequence shows). The segmentation carries the **five labels**
of §7: 0 background, 1 NETC, 2 SNFH, 3 ET, 4 RC.

Two facts about the mirror matter and are stated honestly rather than hidden. First, the volumes
are **182×218×182** voxels — a resampled, MNI-like grid, *not* the native BraTS 240×240×155.
Second, it is a partial community re-upload (~700 of the ~1,350 official cases). Neither harms an
A/B comparison — both tracks see the identical data — but for absolute claims one should cite the
official BraTS 2024 challenge, not the mirror.

A cheap but decisive check confirms the data is the right *vintage*. Every mask is verified to be
integer-valued with labels drawn only from $\{0,1,2,3,4\}$, and the enhancing-tumour class (ET,
label 3) is required to appear somewhere. It appears in **579 of 700** cases — ruling out a
pre-operative $\{0,1,2,4\}$ dataset and confirming these are genuine post-treatment scans (§5).

### The split

The 700 cases are divided **once**, with a fixed random seed, into **490 / 105 / 105**
(train / validation / test) — a 70/15/15 split. Because the resection-cavity class (RC) is the
scarce structure to distribute fairly, the split is **stratified by RC presence**: cases with and
without RC are shuffled and divided separately, so each of the three sets gets a representative
share of RC. The test set of 105 cases is touched only for final scoring.

![Data flow: 700 mirror cases are contract-checked, RC-stratified into a fixed 490/105/105 split, then each track trains three seeds and is scored once on the shared 105-case test set.](figures/diagrams/s19_data_pipeline.png)

### The network — one architecture, frozen

Every result in this note comes from a single, deliberately small network: a **MONAI 3D U-Net**
(§11) with these exact settings.

| Setting | Value |
|---|---|
| spatial dimensions | 3 |
| input channels | 3 (t2f, t1c, t2w) |
| output channels | 5 (the five classes) |
| channel widths (depth) | 16 → 32 → 64 → 128 → 256 |
| downsampling strides | 2, 2, 2, 2 |
| residual units | 0 |
| dropout | 0.1 |
| **trainable parameters** | **1,983,069** |

That is a *tiny* network by segmentation standards (a competition-grade nnU-Net ensemble has
orders of magnitude more capacity). The smallness is intentional — see §22 — but the number that
matters most here is a rule, not a value:

!!! intuition "The one rule that makes the experiment honest"
    The network above is held **byte-identical** across the two tracks. Both call the *same*
    factory with the *same* configuration; not one channel width, stride, dropout value, or
    normalization layer differs. This is what lets us attribute any performance difference to the
    **data pipeline**, and not to the model. The moment Track B were allowed a "better" network,
    the comparison would be confounded (§18) and the whole claim would collapse under one
    interview question: *"how do you know it was the loss and not the architecture?"*

### What actually differs between the tracks

Only three things are allowed to change between Track A and Track B: the **data pipeline** (how
voxels are resized, normalized, and sampled), the **loss function**, and the **checkpoint-selection
metric**. Each difference is a named defect and its fix, catalogued in §20. Track A is the
faithful reproduction of the original research code's pipeline; Track B is the corrected one.

To sharpen the numbers into a *distribution* rather than a single lucky run, each track is trained
**three times** with different random seeds (weight initialization, augmentation, and sampling
order all vary), while the data split stays fixed. Reporting the mean and standard deviation over
those three runs (§18) is what turns "0.67" into "0.670 ± 0.002".

!!! gotcha "Gotcha"
    The *split* seed and the *training* seeds are different knobs. The split is frozen at one seed
    so all six runs (2 tracks × 3 seeds) are scored on the **same** 105 test cases — otherwise a
    track could look better simply by getting an easier test set. Only the training seed varies
    across the three runs of a track.

!!! example "Example question"
    The split is 70/15/15 of 700 cases. Verify the counts, and state how many cases the model
    never sees during training or model selection.

    **Solution.** Train $= 0.70 \times 700 = 490$; validation $= 0.15 \times 700 = 105$; test
    $= 700 - 490 - 105 = 105$. The model's weights are updated only on the 490 training cases; the
    105 validation cases are used to *choose* the checkpoint (§16, the D4 story in §20); and the
    105 **test** cases are never involved in training or selection — they are the honest estimate
    of performance on unseen data.
