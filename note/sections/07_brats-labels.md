## 7. The BraTS challenge and its post-treatment labels {#brats-labels}

We have now met the disease (gliomas, §3), its most aggressive form and how it is treated (§4), the confusing brain that treatment leaves behind (§5), and the four MRI "sequences" that let us see inside it (§6). This section pulls those threads together into a concrete, machine-readable task: given four MRI scans of a treated brain, label every tiny 3D image element as one of five classes — the four tumour tissues plus background. That task, and the way it is scored, is defined by the **BraTS challenge**.

### What BraTS is

**BraTS** stands for the **Brain Tumor Segmentation** challenge: a long-running international competition, held every year at the MICCAI medical-imaging conference since 2012, in which research teams build algorithms that automatically outline brain tumors on MRI and are ranked on how well their outlines match expert human ones [Sensors/MDPI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945730/). To **segment** an image means to assign every picture element a category label — here, which tissue type it is. The 3D equivalent of a pixel is a **voxel** (a "volume element", a tiny cube of tissue); a full brain MRI is a stack of slices making a 3D grid of voxels, and the model must decide a label for each one. We treat segmentation as a formal learning task in §9.

Historically BraTS scanned brand-new, **pre-operative** (untreated) gliomas. The 2024 edition (organized by Correia de Verdier and colleagues) was the first to tackle the harder, more common clinical reality of imaging gliomas *after* treatment — after surgery, radiation, and chemotherapy — using what the organizers call the largest expert-annotated post-treatment glioma dataset assembled, roughly 2,200 cases from seven institutions [BraTS 2024](https://arxiv.org/abs/2405.18368). Each case comes with the four sequences from §6: pre-contrast T1, contrast-enhanced T1-Gd (also written T1c), T2, and T2-FLAIR, because different tumor parts show up on different sequences [BraTS 2024](https://arxiv.org/html/2405.18368v1).

Why bother with a separate post-treatment task? Because treated tumors produce **pseudoprogression** — new or growing enhancement that later fades on its own without any change in therapy — and radiation change, both of which mimic real tumor regrowth [RANO 2.0](https://academic.oup.com/neuro-oncology/article/26/1/2/7286265). Pseudoprogression has been reported in roughly 13–24% of glioblastoma patients after chemoradiotherapy in one cohort [Radbruch et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3755387/), which is exactly why telling true tumor from treatment effect is so hard and so consequential (§5, §8).

### The four raw labels

BraTS 2024 defines four mutually exclusive tissue labels — every non-background voxel gets exactly one — encoded as integers in the released masks: **1 = NETC**, **2 = SNFH**, **3 = ET**, **4 = RC**, with **0 = background** [BraTS solutions doc](https://github.com/andre-fs-ferreira/BraTS_2023_2024_solutions/blob/main/BraTS2024_Task1.md).

| Label | Integer | What it is | How it looks on MRI |
|---|---|---|---|
| **ET** — enhancing tissue/tumor | 3 | Active tumor with a broken blood–brain barrier | Bright (hyperintense) on T1-Gd, excluding vessels [BraTS 2024](https://arxiv.org/html/2405.18368v1) |
| **NETC** — non-enhancing tumor core | 1 | Dead/necrotic or cystic core that takes up no contrast | Dark on both T1 and T1-Gd [BraTS 2024](https://arxiv.org/html/2405.18368v1) |
| **SNFH** — surrounding non-enhancing FLAIR hyperintensity | 2 | Edema + infiltrating tumor + radiation change + scar, lumped together | Bright on T2/FLAIR [BraTS 2024](https://arxiv.org/html/2405.18368v1) |
| **RC** — resection cavity | 4 | The fluid/air/blood-filled hole left by surgery | Isointense to spinal fluid on T1 and T2/FLAIR [BraTS 2024](https://arxiv.org/html/2405.18368v1) |

Two subtleties are worth flagging now. First, **ET** counts thick or nodular enhancement but deliberately *excludes* thin, linear enhancement running along the cavity wall or the dura, because that pattern is treatment-related rather than tumor [BraTS 2024](https://arxiv.org/html/2405.18368v1). Second, **RC** is a brand-new label unique to the post-treatment challenge — pre-operative BraTS tasks had no resection cavity because nothing had been cut out yet [BraTS 2024](https://arxiv.org/html/2405.18368v1).

!!! intuition "Intuition"
    Think of the four labels as the raw paint colors a model must dab onto every voxel. Radiologists, though, do not act on "label 3." They ask three questions: *how much active tumor is there?*, *how big is the solid tumor core?*, and *how far does the abnormality reach?* The evaluation regions below are the meaningful shapes you get by mixing the raw colors to answer those questions.

### From labels to nested evaluation regions

Models are **not** scored on the four raw labels directly. Instead the labels are combined by set **union** (the "∪" symbol: $A\cup B$ is the set of voxels in $A$ or $B$ or both) into three clinically meaningful **evaluation regions**:

$$\mathrm{ET}=\{3\},\quad \mathrm{TC}=\{1\}\cup\{3\},\quad \mathrm{WT}=\{1\}\cup\{2\}\cup\{3\},\quad \mathrm{ET}\subseteq\mathrm{TC}\subseteq\mathrm{WT}$$

Here **ET** (Enhancing Tumor) is label 3 alone; **TC** (Tumor Core) is NETC plus ET; **WT** (Whole Tumor) is NETC plus SNFH plus ET. A voxel belongs to a region if its label is in that region's set. The symbol "$\subseteq$" means "is a subset of": every ET voxel is also a TC voxel, and every TC voxel is also a WT voxel, giving the nesting $\mathrm{ET}\subseteq\mathrm{TC}\subseteq\mathrm{WT}$ [BraTS 2024](https://arxiv.org/html/2405.18368v1). In 2024 the resection cavity (label 4) is additionally scored as its own fourth region, so teams are ranked on four regions in total [BraTS solutions doc](https://github.com/andre-fs-ferreira/BraTS_2023_2024_solutions/blob/main/BraTS2024_Task1.md).

![How the four BraTS labels combine into the nested evaluation regions ET, TC, WT with RC scored separately](figures/diagrams/s7_labels_regions.png)

!!! gotcha "Gotcha"
    "Whole Tumor" does **not** mean "everything abnormal." $\mathrm{WT}=\text{NETC}\cup\text{SNFH}\cup\text{ET}$ (labels 1, 2, 3) — the resection cavity (label 4) is **excluded** from WT and scored on its own [BraTS 2024](https://arxiv.org/html/2405.18368v1). Also do not confuse the *label* NETC (non-enhancing tumor **core**, label 1) with the *region* TC (tumor core = labels 1 and 3). NETC is only a part of TC.

Let us build the region voxel counts for one worked case. Suppose the expert mask has label counts $n_{NETC}=100$, $n_{SNFH}=500$, $n_{ET}=200$, $n_{RC}=150$. Because the labels are mutually exclusive, the size of a union is just the sum of its parts:

$$\begin{aligned}
\mathrm{ET}_{\text{count}} &= n_{ET} = 200 \\
\mathrm{TC}_{\text{count}} &= n_{NETC} + n_{ET} = 100 + 200 = 300 \\
\mathrm{WT}_{\text{count}} &= n_{NETC} + n_{SNFH} + n_{ET} = 100 + 500 + 200 = 800 \\
\mathrm{RC}_{\text{count}} &= n_{RC} = 150
\end{aligned}$$

The counts rise as we go outward, $200 \le 300 \le 800$, exactly mirroring the containment $\mathrm{ET}\subseteq\mathrm{TC}\subseteq\mathrm{WT}$ — a useful sanity check. Note the cavity's 150 voxels are never folded into WT.

### How a region is scored: the Dice coefficient

To score a predicted region against the expert region we use the **Dice Similarity Coefficient (DSC)**, a measure of overlap between two sets. Its set form is

$$\mathrm{DSC}(P,G) = \frac{2\,|P \cap G|}{|P| + |G|}$$

where $P$ is the set of voxels the model *predicts* for a region, $G$ is the expert **ground-truth** set (the reference "correct answer"), $|\cdot|$ counts voxels, and $P\cap G$ is the overlap (voxels in both). Dice runs from 0 (no overlap) to 1 (perfect). We treat Dice, and the loss function built from it, in full in §13; here we just need to be able to compute it.

It is often more convenient to write Dice using the three **confusion-matrix** counts. Let $\mathrm{TP}$ (true positives) be voxels correctly labeled as the region, $\mathrm{FP}$ (false positives) be voxels wrongly labeled as the region, and $\mathrm{FN}$ (false negatives) be region voxels the model missed. Then, step by step:

$$\begin{aligned}
\mathrm{DSC} &= \frac{2\,|P \cap G|}{|P| + |G|} && \text{(definition)}\\
|P \cap G| &= \mathrm{TP} && \text{overlap = correctly predicted voxels}\\
|P| &= \mathrm{TP} + \mathrm{FP} && \text{a predicted voxel is right (TP) or wrong (FP)}\\
|G| &= \mathrm{TP} + \mathrm{FN} && \text{a truth voxel is found (TP) or missed (FN)}\\
\mathrm{DSC} &= \frac{2\,\mathrm{TP}}{(\mathrm{TP}+\mathrm{FP})+(\mathrm{TP}+\mathrm{FN})} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} && \text{substitute and collect the two TP terms}
\end{aligned}$$

The second and third lines hold because, for any voxel, "predicted as the region" splits cleanly into right-or-wrong, and "truly in the region" splits into found-or-missed; the two pieces of each split are disjoint and together account for everything. Plugging in concrete counts $\mathrm{TP}=80$, $\mathrm{FP}=10$, $\mathrm{FN}=20$:

$$\mathrm{DSC} = \frac{2(80)}{2(80)+10+20} = \frac{160}{190} \approx 0.842$$

So this region scores about 0.84 out of a perfect 1.0.

### Lesion-wise scoring: no hiding small misses

A brain can hold several separate tumor foci. If we pooled every voxel in the whole image into one Dice, a model could nail one big lesion and quietly miss several small ones, yet still score high. BraTS 2024 blocks this with **lesion-wise** metrics: each connected lesion is scored on its own, then averaged. Concretely, the organizers dilate (slightly expand) the ground-truth labels and run 3D connected-component analysis with 26-connectivity — meaning two voxels count as part of the same lesion if they touch on any of their 26 neighboring cubes — to split the mask into individual lesions [BraTS metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics). Every missed lesion (false negative) and every spurious lesion (false positive) is handed a Dice of 0 and a large boundary penalty [BraTS metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics). The per-case lesion-wise Dice is

$$\overline{\mathrm{DSC}}_{\text{lesion}} = \frac{1}{N_{TP}+N_{FP}+N_{FN}}\left(\sum_{i=1}^{N_{TP}} \mathrm{DSC}_i + \sum_{j=1}^{N_{FP}} 0 + \sum_{k=1}^{N_{FN}} 0\right)$$

where $N_{TP}$ is the number of correctly detected lesions (each contributing its own $\mathrm{DSC}_i$), $N_{FP}$ the number of spurious lesions, and $N_{FN}$ the number of missed lesions; dividing by the total lesion count spreads every detection error across the score. BraTS also reports the **lesion-wise 95% Hausdorff Distance (HD95)**, a boundary-error measure in millimetres where lower is better; we develop both metrics in §16.

![Lesion-wise Dice scores each lesion separately so a single miss and a false alarm pull the case score far below the average of the detected lesions](figures/diagrams/s7_lesion_scoring.png)

!!! gotcha "Gotcha"
    Lesion-wise Dice is **not** the same number as the old "global" whole-image Dice reported in pre-2023 BraTS papers. Because one missed 4 mm nodule scores a full 0 and is averaged in, lesion-wise values run lower and are not directly comparable to older leaderboards [BraTS metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics). Always check which metric a result uses before comparing.

!!! example "Example question"
    A model's prediction for one case has label voxel counts NETC = 50, SNFH = 300, ET = 120, RC = 90; the expert truth has NETC = 40, SNFH = 320, ET = 150, RC = 100. Assuming predicted and true voxels overlap as much as possible for each label, compute the Tumor Core (TC) counts and the best-case TC Dice.

    **Solution:**
    **Step 1 — build the region.** $\mathrm{TC}=\text{NETC}\cup\text{ET}$. Prediction $\mathrm{TC}=50+120=170$; ground truth $\mathrm{TC}=40+150=190$.
    **Step 2 — best-case overlap.** "Overlap as much as possible" means the smaller set sits entirely inside the larger, so $\mathrm{TP}=\min(170,190)=170$, $\mathrm{FP}=170-170=0$, $\mathrm{FN}=190-170=20$.
    **Step 3 — Dice.** $\mathrm{DSC}=\dfrac{2(170)}{2(170)+0+20}=\dfrac{340}{360}\approx 0.944$. The best achievable TC Dice here is about **0.94**, limited only by the 20 core voxels the model under-predicts.

!!! example "Example question"
    In one patient the ground truth has 4 separate enhancing-tumor lesions. A model finds 3 (per-lesion Dice 0.95, 0.70, 0.60), misses the 4th, and produces 2 false-positive enhancing blobs. Compute the lesion-wise ET Dice.

    **Solution:**
    **Step 1 — count lesions.** $N_{TP}=3$, $N_{FN}=1$, $N_{FP}=2$; denominator $=3+1+2=6$.
    **Step 2 — numerator.** Detected lesions contribute their Dice; misses and false alarms contribute 0: $0.95+0.70+0.60+0+0+0=2.25$.
    **Step 3 — average.** $\overline{\mathrm{DSC}}_{\text{lesion}}=2.25/6=0.375$. Although the three detected lesions averaged $(0.95+0.70+0.60)/3=0.75$ on their own, one miss and two false alarms drag the case score down to **0.375** — the metric punishes detection failures, not just sloppy edges.

!!! intuition "Intuition"
    The regions are Russian nesting dolls: ET (active tumor) sits inside TC (core = enhancing plus dead/cystic tissue), which sits inside WT (core plus surrounding FLAIR abnormality). Scoring these unions rather than the raw labels is also *forgiving* of the fuzzy, expert-disputed borders between adjacent tissue types: a voxel a model wrongly flips between NETC and ET still lands correctly inside TC and WT, so that internal disagreement never hurts the TC or WT score. This is the methodological reason BraTS scores nested regions — the clinical reason is that "how much enhancing tumor / core / abnormality" are the questions doctors actually ask (§8).

With the task, the four tumour labels (plus background), the nested regions, and the scoring rules now fixed, we are ready to look at what these segmentations mean for patient care (§8) before turning segmentation into a formal machine-learning problem (§9).
