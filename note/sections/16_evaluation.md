## 16. Evaluating a segmentation: metrics beyond accuracy {#evaluation}

In §13 we met the Dice coefficient as a *loss* the network minimises during training. Here we meet the same quantity — and several companions — in their second job: as *scores* that tell us, after training, how good a segmentation actually is. A segmentation model (§9) labels every **voxel** (a voxel is a 3D pixel, one small cube of the scanned volume) as either a tumour sub-region or healthy tissue. We need a single honest number that says how well those labels match a human expert's ground truth. The obvious candidate — the fraction of voxels labelled correctly — turns out to be actively misleading for brain tumours. This section builds the metrics that replace it, derives each from scratch, and shows the traps that make careful reporting a matter of scientific honesty rather than pedantry.

### The four counts everything is built from

Fix one class (say, enhancing tumour) and one patient. Every voxel falls into exactly one of four buckets by comparing what the model said to what is true:

- **TP** (true positive): predicted this class, and it really is — a correct hit.
- **FP** (false positive): predicted this class, but it is not — a false alarm.
- **FN** (false negative): did not predict this class, but it really is — a miss.
- **TN** (true negative): did not predict this class, and it is not — a correct pass.

Let $A$ be the set of voxels the model labels as the class, and $B$ the set of voxels that truly are the class (the ground truth). Then $|A\cap B| = \mathrm{TP}$ (the overlap), $|A| = \mathrm{TP}+\mathrm{FP}$ (everything predicted), and $|B| = \mathrm{TP}+\mathrm{FN}$ (everything true), where $|\cdot|$ denotes the number of voxels in a set. These identities drive every derivation below.

![A brain slice with predicted mask A and true mask B overlapping; the overlap is TP, A-only is FP, B-only is FN, and the whole surrounding region is TN. Dice/precision/recall use only TP, FP, FN; accuracy and specificity are dominated by the huge TN.](figures/diagrams/s16_confusion_regions.png)

### Why voxel accuracy fails

**Voxel accuracy** is the fraction of all voxels labelled correctly:

$$\text{Accuracy} = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.$$

Here the denominator is just $N$, the total number of voxels. The problem is **class imbalance** (§14): a tumour sub-region may occupy a tiny fraction of the brain, so TN is enormous and dominates the sum. Let the **prevalence** $p$ be the fraction of voxels that truly belong to the class, $p = (\mathrm{TP}+\mathrm{FN})/N$. Consider the laziest possible model: it labels *every* voxel "background."

$$\begin{aligned}
&\text{Predict background everywhere} \Rightarrow \mathrm{TP}=0,\ \mathrm{FP}=0, \\
&\mathrm{FN}=pN \quad(\text{every true voxel is now missed}), \\
&\mathrm{TN}=(1-p)N \quad(\text{every background voxel is "correctly" passed}).
\end{aligned}$$

Substitute into the accuracy formula:

$$\text{Accuracy}_{\text{all-neg}} = \frac{0+(1-p)N}{0+(1-p)N+0+pN} = \frac{(1-p)N}{N} = 1-p.$$

The $N$ cancels, leaving $1-p$. If a tumour fills $0.1\%$ of the scan, $p=0.001$ and accuracy $=0.999$ — a triumphant-looking 99.9% for a model that found *nothing*. Its Dice, as we will see, is exactly 0. Accuracy rewarded the model for the vast background it cannot help getting right. **Never report accuracy alone for segmentation.**

!!! intuition "Intuition"
    Accuracy counts the true negatives, and the brain scan is mostly "not this tumour class" — an easy region every model gets for free. Counting it drowns out the hard part. The metrics below all share one trick: they throw the true negatives away on purpose and grade only the region where the model and the truth actually disagree.

### Dice: overlap that ignores the background

The **Dice similarity coefficient (DSC)** is twice the overlap divided by the total size of both masks. We now derive its confusion-matrix form with no skipped algebra.

$$\begin{aligned}
\mathrm{DSC} &= \frac{2\,|A\cap B|}{|A|+|B|} &&\text{(1) definition: twice the overlap over summed sizes}\\
&= \frac{2\,\mathrm{TP}}{(\mathrm{TP}+\mathrm{FP})+(\mathrm{TP}+\mathrm{FN})} &&\text{(2) substitute } |A\cap B|=\mathrm{TP},\ |A|=\mathrm{TP}+\mathrm{FP},\ |B|=\mathrm{TP}+\mathrm{FN}\\
&= \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} &&\text{(3) collect the two TP terms: } \mathrm{TP}+\mathrm{TP}=2\,\mathrm{TP}.
\end{aligned}$$

Look hard at step (3): there is **no TN term**. The healthy background is invisible to Dice, which is exactly why Dice cannot be gamed by predicting all-background — such a model has $\mathrm{TP}=0$, giving $\mathrm{DSC}=0$. Dice ranges from 0 (no overlap) to 1 (perfect), and every voxel of disagreement (an FP or an FN) drags it down [Dice similarity coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

### Precision, recall, and specificity

Three more ratios, each reading off a different row or column of the four counts:

$$\text{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad \text{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad \text{Specificity}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}.$$

**Recall** (also called sensitivity or the true-positive rate) asks: of all the tumour that was truly there, what fraction did we find? **Precision** (also called positive predictive value) asks: of everything we *called* tumour, what fraction really was? **Specificity** asks: of all the healthy tissue, what fraction did we correctly leave alone? Because TN dwarfs FP in a brain scan, specificity sits pinned near 1.0 for essentially any model, good or bad — it is nearly useless here. Precision and recall are the informative pair, because FP lives in precision's small denominator where it actually moves the number.

!!! intuition "Intuition"
    Precision and recall pull in opposite directions. Predict "tumour" everywhere and recall shoots to 1 (you missed nothing) while precision collapses (almost all of it is wrong). Predict only the single voxel you are surest of and precision hits 1 while recall collapses. A good score requires getting *both* the extent and the placement right — which is exactly what Dice, sitting between them, demands.

### Dice is the F1 score

That "sitting between them" is precise: Dice equals the **F1 score**, the harmonic mean of precision $P$ and recall $R$. Writing $P=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ and $R=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$:

$$\begin{aligned}
PR &= \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}\cdot\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} = \frac{\mathrm{TP}^2}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})} &&\text{(1) multiply the fractions}\\
P+R &= \frac{\mathrm{TP}(\mathrm{TP}+\mathrm{FN})+\mathrm{TP}(\mathrm{TP}+\mathrm{FP})}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})} &&\text{(2) add over the common denominator}\\
&= \frac{\mathrm{TP}\,(2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN})}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})} &&\text{(3) factor out TP, combine the two TP terms}\\
F_1 = \frac{2PR}{P+R} &= \frac{2\,\dfrac{\mathrm{TP}^2}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})}}{\dfrac{\mathrm{TP}(2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN})}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})}} &&\text{(4) substitute (1) and (3) into the definition of }F_1\\
&= \frac{2\,\mathrm{TP}^2}{\mathrm{TP}\,(2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN})} &&\text{(5) the shared denominator cancels top and bottom}\\
&= \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} = \mathrm{DSC} &&\text{(6) cancel one factor of TP — identical to Dice.}
\end{aligned}$$

The equality is exact, not approximate. The harmonic mean punishes imbalance between $P$ and $R$: a model with $P=1,\ R=0.01$ scores about $0.02$, not the arithmetic $0.5$. That is why a single Dice number is a fair summary of both over-segmenting and missing tumour.

### A worked example, end to end

!!! example "Example question"
    A model segments enhancing tumour in one patient. The ground truth has 100 tumour voxels; the model labels 120 voxels as tumour, of which 90 are truly tumour. Compute TP, FP, FN, then Dice, precision, and recall, and check the F1 identity.

    **Solution.**
    - $\mathrm{TP}$ = voxels both predicted and true $= 90$.
    - $\mathrm{FP}$ = predicted but not true $= 120 - 90 = 30$.
    - $\mathrm{FN}$ = true but not predicted $= 100 - 90 = 10$.
    - $\mathrm{DSC} = \dfrac{2\cdot 90}{2\cdot 90 + 30 + 10} = \dfrac{180}{220} = 0.818.$
    - $\text{Precision} = \dfrac{90}{90+30} = \dfrac{90}{120} = 0.750.$
    - $\text{Recall} = \dfrac{90}{90+10} = \dfrac{90}{100} = 0.900.$

    The model finds 90% of the true tumour (good recall) but a quarter of what it calls tumour is wrong (precision 0.75) — it over-segments. Dice 0.818 sits between them. Check: $F_1 = \dfrac{2\cdot 0.75\cdot 0.90}{0.75+0.90} = \dfrac{1.35}{1.65} = 0.818$, matching Dice exactly.

### Jaccard: the same ranking, different scale

You will also see the **Jaccard index** (or intersection-over-union, IoU), $J = |A\cap B|/|A\cup B| = \mathrm{TP}/(\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$, where $A\cup B$ is the union of the two masks. It is not new information. Starting from $J$:

$$\begin{aligned}
1+J &= 1+\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} = \frac{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} &&\text{put 1 over the same denominator and add}\\
\frac{2J}{1+J} &= \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} = \mathrm{DSC} &&\text{the shared denominator cancels.}
\end{aligned}$$

Because $J\mapsto 2J/(1+J)$ is strictly increasing on $[0,1]$, Dice and Jaccard rank any set of models identically (they agree only at 0 and 1, and $J\le\mathrm{DSC}$ everywhere else) [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index). Reporting both is redundant.

### Boundary error: Hausdorff and HD95

Overlap metrics grade the *bulk* of the mask but say nothing about *where* the errors sit. Two predictions can share a near-identical Dice while one has a clean surface and the other a 2 cm spurious spike. To grade the edge we use a **boundary metric**, the **Hausdorff distance**. Let $\partial A$ and $\partial B$ be the surface voxels of the two masks. The directed distance $h(A,B)$ walks every point of $A$'s surface, measures its distance to the nearest point of $B$'s surface, and keeps the worst:

$$h(A,B)=\max_{a\in\partial A}\ \min_{b\in\partial B}\ \lVert a-b\rVert,\qquad H(A,B)=\max\{h(A,B),\,h(B,A)\},$$

where $\lVert a-b\rVert$ is Euclidean distance in millimetres and $H$ symmetrises by taking the larger of the two directions. The trouble: one stray misclassified voxel far from the tumour sets the entire score. BraTS therefore uses **HD95**, the 95th percentile of the surface distances instead of the maximum, discarding the worst 5% of outliers:

$$\mathrm{HD95}=\max\{\,P_{95}\,d(\partial A,\partial B),\ P_{95}\,d(\partial B,\partial A)\,\},$$

where $d(\partial A,\partial B)$ is the collection of nearest-surface distances and $P_{95}$ takes their 95th percentile [BraTS metrics; HDilemma MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/2469_paper.pdf). Units are millimetres; 0 mm is perfect. Report HD95 alongside Dice because the two fail in different ways — Dice, a volume ratio, barely feels a thin spike that HD95 flags loudly.

### Lesion-wise Dice: counting lesions, not voxels

The post-treatment brain (§5) often holds several disconnected lesions. Ordinary voxel-wise Dice pools all voxels together, which lets one large well-segmented tumour hide a completely missed small one. BraTS 2023 introduced **lesion-wise Dice**, which changes the unit of accounting from "voxel" to "lesion." It isolates each disconnected ground-truth lesion (dilating masks by roughly 3 voxels to define connected components), computes an ordinary Dice for each matched lesion, and scores every missed and every spurious lesion as a full zero:

$$\mathrm{DSC}_{\text{lesion}} = \frac{\displaystyle\sum_{i\in\text{matched}} \mathrm{DSC}_i}{\mathrm{TP}_{\text{les}} + \mathrm{FN}_{\text{les}} + \mathrm{FP}_{\text{les}}},$$

where the sum runs over matched (detected) lesions, $\mathrm{TP}_{\text{les}}$ is the number of correctly detected lesions, $\mathrm{FN}_{\text{les}}$ is ground-truth lesions with no prediction, and $\mathrm{FP}_{\text{les}}$ is predicted lesions with no matching truth. Dividing by the total lesion count spreads the penalty so a pea-sized metastasis counts as much as a golf-ball tumour [BraTS-2023-Metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics).

![Two disconnected lesions: a large one segmented perfectly and a small one missed. Voxel-wise Dice pools voxels and scores about 0.995; lesion-wise Dice scores each lesion as a unit, giving 1.0 and 0, averaged to 0.50.](figures/diagrams/s16_lesion_wise.png)

!!! example "Example question"
    A patient has two disconnected enhancing lesions: a large one (1000 voxels) and a small one (10 voxels). The model segments the large lesion perfectly and misses the small one entirely. Compute voxel-wise and lesion-wise Dice, and explain the gap.

    **Solution.**
    *Voxel-wise* (pool all voxels): $\mathrm{TP}=1000$, $\mathrm{FN}=10$, $\mathrm{FP}=0$, so
    $$\mathrm{DSC}=\frac{2\cdot 1000}{2\cdot 1000+0+10}=\frac{2000}{2010}=0.995.$$
    *Lesion-wise* (each lesion is a unit): the large lesion is a matched true positive with its own $\mathrm{DSC}_i=1.0$; the small lesion is a false-negative lesion scoring 0. The denominator is $\mathrm{TP}_{\text{les}}+\mathrm{FN}_{\text{les}}+\mathrm{FP}_{\text{les}}=1+1+0=2$, so
    $$\mathrm{DSC}_{\text{lesion}}=\frac{1.0+0}{2}=0.500.$$
    The gap (0.995 vs 0.500) is the point. Voxel-wise Dice barely notices the miss because the small lesion was only 10 of ~2010 voxels; lesion-wise Dice halves the score, matching how a radiologist thinks — "you missed a lesion" — rather than how a volume ratio thinks.

### Empty classes and the GT-present count

The BraTS 2024 post-treatment task scores four sub-regions — enhancing tumour (ET), non-enhancing tumour core (NETC), surrounding non-enhancing FLAIR hyperintensity (SNFH), and resection cavity (RC) — several of which are simply *absent* in many patients [BraTS 2024 post-treatment challenge](https://arxiv.org/abs/2405.18368). The scoring convention: if ground truth and prediction are both empty for a class, Dice $=1.0$; if ground truth is empty but the prediction is not, Dice $=0.0$ [BraTS 2024 post-treatment challenge](https://arxiv.org/abs/2405.18368). Those free 1.0s quietly poison a naive class average.

!!! gotcha "Gotcha"
    Empty ground truth is a scoring landmine. If a class is absent in many post-treatment patients, a per-class mean over *all* cases blends real segmentation skill with piles of free empty-vs-empty 1.0s — inflating the number and making it swing wildly if the fraction of empty cases changes. This is why per-class Dice must always be published with its **GT-present case count**, e.g. "ET Dice 0.75 over 3/5 GT-present cases." Two related traps: specificity is near-useless here (TN dwarfs FP, so it pins near 1.0 for any model — quote precision instead), and raw max-Hausdorff is dangerously outlier-sensitive, so always use HD95. HD95's handling of empty masks itself varies between implementations, another place empty cases silently distort averages [HDilemma MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/2469_paper.pdf).

!!! example "Example question"
    Class "enhancing tumour" is evaluated over 5 patients. Three have ET present, with per-case Dice 0.80, 0.70, 0.75. Two have no ET at all, and the model correctly predicts empty (Dice $=1.0$ each by convention). Report the mean two ways and explain why the GT-present count matters.

    **Solution.**
    - Mean over all 5 cases: $(0.80+0.70+0.75+1.0+1.0)/5 = 4.25/5 = 0.850.$
    - Mean over the 3 GT-present cases: $(0.80+0.70+0.75)/3 = 2.25/3 = 0.750.$

    The headline 0.85 is inflated by two free 1.0s that reflect no segmentation skill — the model merely drew nothing where there was nothing. The honest measure of how well it delineates real enhancing tumour is 0.75 over 3 GT-present cases. Reporting "Dice 0.85" without the "3/5" would overstate performance and would swing if the mix of empty cases changed. (For this project's own class scheme and its resection-cavity handling, see §19–§21.)

### What to report

| Metric | Answers | Watch out for |
|---|---|---|
| Voxel accuracy | Fraction of voxels correct | Useless under imbalance ($1-p$ for all-background) |
| Dice / F1 | Overlap = balance of precision & recall | A single per-case mean can be swayed by empty cases |
| Precision / recall | Over-segmenting vs missing | Report the pair, not one alone |
| Specificity | Healthy tissue spared | Pinned near 1.0; near-meaningless here |
| HD95 (mm) | Worst realistic boundary error | Undefined/penalised for empty masks |
| Lesion-wise Dice | Per-lesion detection | Small lesions weigh as much as large |

One last caution for multi-class scoring. Dice equals F1 only in the two-class (one-vs-rest) case per label. Multi-class segmentation is scored as several one-vs-rest Dice values — one per sub-region — then averaged; there is no single "multiclass Dice" equal to accuracy, and mixing the hierarchical BraTS regions with the disjoint label classes is a common reporting error. The reference architecture behind these systems is the U-Net (§11) and its self-configuring 3D successor nnU-Net [U-Net, MICCAI 2015](https://arxiv.org/abs/1505.04597) — but an architecture is only ever as trustworthy as the metric used to judge it, which is why this section matters as much as the ones on the network itself.
