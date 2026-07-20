## 8. Why voxel-accurate segmentation matters clinically {#clinical-impact}

Before we build the machinery that draws tumor outlines (the convolutional networks of §10 and the U-Net of §11), it is worth asking a blunt question: why does it matter whether a computer traces a tumor's border one voxel too wide or too narrow? A **voxel** is the 3D pixel of an MRI scan — a small box of tissue, typically a millimetre or so on each side. **Segmentation** means deciding, for every voxel in the scan, which class it belongs to: healthy tissue, or one of the tumor sub-regions of §7. This section explains why getting those per-voxel decisions right is not a cosmetic detail but the foundation of nearly every quantitative decision in neuro-oncology.

### From an outline to a number a doctor can act on

The moment you have an outline, you can count the voxels inside it and turn that count into a physical **volume** — the size of the tumor in millilitres. That number is what lets a clinician say the tumor grew or shrank between two scans. The conversion is simple arithmetic:

$$V = N \cdot (s_x\, s_y\, s_z)$$

Here $N$ is the number of voxels the segmentation marks as the region of interest; $s_x, s_y, s_z$ are the **voxel spacings** — the physical width of one voxel along each of the three axes, in millimetres, read straight from the MRI file's header. Their product $s_x\, s_y\, s_z$ is the volume of a single voxel in cubic millimetres, so multiplying by the count $N$ gives the total volume $V$ in $\text{mm}^3$. Dividing by 1000 converts to millilitres, because $1000\ \text{mm}^3 = 1\ \text{cm}^3 = 1\ \text{mL}$.

Let us walk one all the way through. Suppose the enhancing-tissue mask contains $N = 15000$ voxels and the scan is **isotropic** (equal spacing on every axis) at $s_x = s_y = s_z = 1\ \text{mm}$:

$$\begin{aligned}
\text{voxel volume} &= s_x\, s_y\, s_z = 1 \times 1 \times 1 = 1\ \text{mm}^3 \\
V &= N \cdot (s_x\, s_y\, s_z) = 15000 \times 1 = 15000\ \text{mm}^3 \\
V &= \frac{15000}{1000} = 15\ \text{mL}.
\end{aligned}$$

The first line is the size of one voxel; the second multiplies that by how many voxels there are; the third just changes units. So this tumor is 15 mL of enhancing tissue. This "objective measure for assessing residual tumor volume, which can guide treatment" is precisely the clinical value the BraTS organisers attribute to automated segmentation [BraTS 2024, arXiv:2405.18368](https://arxiv.org/html/2405.18368v1).

!!! gotcha "Gotcha"
    Never compute volume from the voxel count alone. Repeat the example with an **anisotropic** scan — thicker slices, $s_x = s_y = 1\ \text{mm}$ but $s_z = 3\ \text{mm}$ — keeping the same $N = 15000$. Now the voxel volume is $1 \times 1 \times 3 = 3\ \text{mm}^3$, so $V = 15000 \times 3 = 45000\ \text{mm}^3 = 45\ \text{mL}$. The identical outline reports **three times** the volume. Voxel spacing must always be applied, never assumed to be 1 mm, or serial and cross-hospital comparisons silently break.

### Scoring an outline: the Dice coefficient

To judge a predicted outline we compare it against a reference (ground-truth) outline drawn by experts. The standard score is the **Dice similarity coefficient (DSC)**, treated in depth in §13 and used as the primary BraTS metric:

$$\mathrm{DSC}(A,B) = \frac{2\,|A \cap B|}{|A| + |B|}$$

$A$ is the set of voxels the model labels as a given class; $B$ is the set the reference labels as that class; $|A|$ and $|B|$ are their voxel counts; and $|A \cap B|$ is the count of voxels **both** agree on. The score runs from 0 (no overlap at all) to 1 (perfect overlap). An equivalent form counts errors directly:

$$\mathrm{DSC} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}$$

where **TP** (true positives) are voxels correctly labelled tumor, **FP** (false positives) are voxels the model calls tumor but the reference does not, and **FN** (false negatives) are voxels the reference calls tumor but the model misses. This matches the first form because $|A \cap B| = \mathrm{TP}$, $|A| = \mathrm{TP} + \mathrm{FP}$, and $|B| = \mathrm{TP} + \mathrm{FN}$.

Worked example. Say the model predicts $|A| = 12000$ voxels, the reference has $|B| = 10000$, and they overlap on $|A \cap B| = 9000$:

$$\begin{aligned}
\mathrm{DSC} &= \frac{2 \cdot 9000}{12000 + 10000} && \text{substitute into the definition} \\
&= \frac{18000}{22000} && 2 \times 9000 = 18000;\ 12000 + 10000 = 22000 \\
&= 0.818\ldots \approx 0.82. && \text{divide}
\end{aligned}$$

Cross-check with the error form: $\mathrm{TP} = 9000$, $\mathrm{FP} = 12000 - 9000 = 3000$, $\mathrm{FN} = 10000 - 9000 = 1000$, so $\mathrm{DSC} = \frac{2 \cdot 9000}{2 \cdot 9000 + 3000 + 1000} = \frac{18000}{22000} = 0.82$ — the two forms agree exactly. A Dice of 0.82 sits inside the human inter-rater band, discussed below.

![Predicted mask A and reference mask B overlapping on one MRI slice, with the shared region marked TP and the two crescents marked FP and FN, annotated with the Dice formula](figures/diagrams/s8_dice_overlap.png)

!!! intuition "Intuition"
    Dice is a "how much do two blobs overlap" score, not an accuracy percentage. Tumor voxels are a tiny fraction of the whole brain, so a lazy model that predicts "no tumor anywhere" would score about 99% on plain pixel accuracy while being clinically useless. Dice ignores the vast correctly-labelled background and rewards only overlap on the tumor itself — which is why it, paired with a boundary metric, is the metric of record in BraTS. The class-imbalance problem this reflects is the subject of §14.

That boundary metric is the **95th-percentile Hausdorff distance (HD95)**:

$$\mathrm{HD95} = \max\Big\{ P_{95}\big(\{d(a,B): a \in \partial A\}\big),\; P_{95}\big(\{d(b,A): b \in \partial B\}\big) \Big\}$$

where $\partial A, \partial B$ are the boundary surfaces of the predicted and reference regions, $d(a,B)$ is the shortest distance from a boundary point $a$ to the other surface, and $P_{95}$ takes the 95th percentile of those distances (rather than the true maximum, so one stray voxel does not dominate). HD95, measured in millimetres, captures worst-case boundary error; smaller is better. See §16 for how Dice and HD95 are combined in the overall evaluation.

!!! gotcha "Gotcha"
    Dice is denominator-sensitive to region size: a fixed boundary error costs far more Dice on a small structure than a large one. A few misplaced voxels around a tiny residual enhancing focus can drop Dice from 0.9 to 0.5 even when the volume error is negligible — one reason BraTS pairs Dice with HD95, and why absent classes are scored by convention (a correct empty prediction scores Dice 1, a false positive scores 0).

### Why volumes are more sensitive than 2D measurements

Clinical response is still assessed largely in 2D. The **RANO 2.0** criteria keep the **bidimensional product** — the two longest perpendicular enhancing diameters multiplied together — as the primary measurement, allowing volumetry only "if available" and noting that trials have "not demonstrated a conclusive benefit of volumetric analysis over two-dimensional measurement" [RANO 2.0, PMC10860967](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/). For $n$ target lesions the summed product is $\mathrm{SPD} = \sum_{i=1}^{n} D_i \cdot d_i$, where $D_i$ is the longest in-plane enhancing diameter of lesion $i$ and $d_i$ the longest diameter perpendicular to it on the same slice.

But dimension matters enormously. Model a lesion as a sphere of diameter $D$ that shrinks so every diameter is scaled by $k = 0.8$ (a 20% linear shrink). Then:

$$\begin{aligned}
\text{1D (single diameter)} \propto D: &\quad \frac{0.8D - D}{D} = -0.20 = -20\% \\
\text{2D (RANO product)} \propto D^2: &\quad 0.8^2 - 1 = 0.64 - 1 = -36\% \\
\text{3D (volume)} \propto D^3: &\quad 0.8^3 - 1 = 0.512 - 1 = -48.8\% \approx -49\%.
\end{aligned}$$

Each line takes the same shrink factor $0.8$ and raises it to the power that matches the dimension — 1 for a length, 2 for an area-like product, 3 for a volume — then subtracts 1 to get the percent change. One physical event reads as 20%, 36%, or 49% "response" depending purely on the metric. RANO calls **partial response** at a 50% drop in the 2D product; here the product fell only 36%, so 2D says "not yet responding" while the volume nearly halved. Because real post-surgical cavities are irregular and high-grade gliomas grow eccentrically and nodularly, 2D measurement is unreliable exactly where it is needed most, and reproducible volumetry offers a more accurate representation of tumor-size change [AJNR, PMC7964732](https://pmc.ncbi.nlm.nih.gov/articles/PMC7964732/). The pipeline's direct output is the **volumetric response fraction** $\Delta V = \frac{V_t - V_0}{V_0} \times 100\%$, comparing a later volume $V_t$ to a baseline $V_0$.

!!! intuition "Intuition"
    Dimensional leverage cuts both ways. Because volume scales as the cube of size, volumetry is far more sensitive to genuine response than a single diameter — but for the same reason, a small segmentation error on a large tumor can move the reported volume by several millilitres. Accurate boundaries are what make the volume trustworthy.

### The model measures; the doctor decides

None of this replaces clinical judgement. After surgery, radiation, and the chemotherapy of §4, the post-treatment brain of §5 contains a resection cavity, blood products, scar tissue, and radiation-induced changes that all mimic tumor, and **pseudoprogression** — treatment effect that looks like growth — is common in the first ~12 weeks after radiotherapy, which is why RANO 2.0 uses the post-radiotherapy scan as baseline and requires confirmation of apparent early progression [RANO 2.0, PMC10860967](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/). Telling residual tumor from treatment effect is the clinician's call. The network's job is narrower and still valuable: it removes the slow, noisy, manual tracing step so the volume feeding that decision is fast, reproducible, and comparable across scans and hospitals. The BraTS organisers frame automated post-treatment segmentation as having "significant potential to increase workflow efficiency with more rapid and accurate volumetric assessments for treatment planning" [BraTS 2024, arXiv:2405.18368](https://arxiv.org/html/2405.18368v1). This matters because total tumor burden is a critical outcome indicator that often goes unmeasured in routine care for lack of practical volumetric tools [BraTS-METS 2023, arXiv:2306.00838](https://arxiv.org/pdf/2306.00838), and because manual contouring of many lesions is an unscalable workload in fast-turnaround settings like radiosurgery [Neuro-Oncology, doi:10.1093/neuonc/noab071](https://dx.doi.org/10.1093/neuonc/noab071).

![Two paths from an MRI scan to a treatment decision: a slow manual-tracing path versus a fast automated-segmentation path, both feeding volumes into the clinician's judgement](figures/diagrams/s8_measure_vs_decide.png)

### Inter-rater variability: the problem and the ruler

Manual segmentation is not just slow; it is inconsistent. Even expert raters draw meaningfully different outlines, and this variability is both what motivates automation and the yardstick that judges it. In the foundational multimodal BraTS benchmark, human raters segmenting tumor sub-regions agreed only to Dice scores of roughly **0.74–0.85** [Menze et al. 2015, PMID 25494501](https://pubmed.ncbi.nlm.nih.gov/25494501/); manual tracing is broadly recognised as time-consuming and prone to large intra- and inter-rater variability [DL survey, PMC8321266](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321266/). Because no single expert outline is definitive, BraTS builds its reference by fusing several expert labels — the 2024 post-treatment set (~2,200 cases from 7 institutions, the largest of its kind) used AI pre-segmentation refined manually and approved by board-certified neuroradiologists, with the test set annotated by two independent teams to quantify variability [BraTS 2024, arXiv:2405.18368](https://arxiv.org/html/2405.18368v1).

A tiny worked version shows why fusing helps. Two raters label a 10-voxel strip: rater 1 marks voxels $\{1..7\}$, rater 2 marks $\{4..10\}$, overlapping on $\{4,5,6,7\}$ (4 voxels). Their pairwise Dice is $\frac{2 \cdot 4}{7 + 7} = \frac{8}{14} = 0.571$ — substantial disagreement. Take the unanimous (2-of-2) consensus $C = \{4,5,6,7\}$; rater 1 now scores $\frac{2 \cdot 4}{7 + 4} = \frac{8}{11} = 0.727$ against it. A fused reference averages out each rater's idiosyncrasies and is more stable than any individual label.

!!! example "Example question"
    A model segments the enhancing tumor as 8,000 voxels; the reference is 6,000 voxels; they overlap on 5,000 voxels. (a) Compute the Dice score. (b) Is it within the human inter-rater range? (c) At $1 \times 1 \times 1\ \text{mm}$ spacing, what is the reference tumor volume in mL?

    **Solution:** (a) $\mathrm{DSC} = \frac{2 \cdot 5000}{8000 + 6000} = \frac{10000}{14000} = 0.714$. (b) The reference band is ~0.74–0.85; 0.714 sits just below it, so the model is slightly less consistent than expert raters — acceptable but improvable. (c) Voxel volume $= 1 \times 1 \times 1 = 1\ \text{mm}^3$, so $V = 6000 \times 1 = 6000\ \text{mm}^3 = 6.0\ \text{mL}$ of enhancing tumor.

!!! example "Example question"
    A spherical enhancing lesion shrinks from 40 mm to 32 mm in diameter. Compare the reported "response" under (a) a single diameter, (b) the RANO 2D product, and (c) volume. Which crosses RANO's 50%-decrease partial-response threshold?

    **Solution:** shrink factor $k = 32/40 = 0.8$. (a) Single diameter: $(32 - 40)/40 = -20\%$. (b) 2D product $\propto D^2$: $0.8^2 - 1 = 0.64 - 1 = -36\%$; since $-36\%$ is a smaller drop than the $-50\%$ needed, RANO does **not** call partial response. (c) Volume $\propto D^3$: $0.8^3 - 1 = 0.512 - 1 = -48.8\%$ — nearly halved. The same physical shrinkage reads as "not yet responding" by 2D RANO but "almost a 50% volume drop" by volumetry, illustrating why 2D can under-call genuine response and why reproducible automated volumes are clinically attractive.

!!! gotcha "Gotcha"
    Guidelines still lead with 2D. Because RANO 2.0 keeps the bidimensional product as primary and permits volumetry only "if available" [RANO 2.0, PMC10860967](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/), an automated volumetric pipeline is decision-support, not an approved trial endpoint. Present its volumes as a consistent, observer-independent measurement — not as a replacement for RANO's official response categories.
