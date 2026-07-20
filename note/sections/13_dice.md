## 13. Measuring overlap: the Dice coefficient and Dice loss {#dice}

In §9 we framed segmentation as labelling every voxel (a 3-D pixel) in the MRI scan, and in §12 we met **cross-entropy**, a loss that scores each voxel on its own. But when a clinician looks at a result they do not ask "what fraction of voxels are right?" — they ask "does the outlined tumour region match the real one?" We need a score for *region overlap*. That score is the **Dice coefficient**, and this section builds it from scratch, turns it into a trainable loss, and combines it with cross-entropy.

### 13.1 The Dice similarity coefficient

Call $P$ the set of voxels the model **predicts** as foreground (say, tumour), and $G$ the **ground-truth** set of tumour voxels — the region a radiologist outlined by hand. Write $|P|$ for the number of voxels in $P$, and $|P \cap G|$ for the number of voxels in *both* sets (their overlap). The **Dice similarity coefficient (DSC)**, also called the Sørensen–Dice index, is

$$\mathrm{DSC} = \frac{2\,|P \cap G|}{|P| + |G|}.$$

Here $|G|$ is the true region's size and $|P \cap G|$ is the shared area counted once. DSC runs from $0$ (the masks touch nowhere) to $1$ (they coincide exactly). It is the single most-reported overlap metric in medical-image segmentation and is used throughout BraTS ([Radiopaedia](https://radiopaedia.org/articles/dice-similarity-coefficient)).

We can rewrite the three set-sizes using the **confusion matrix** (the tally of true/false positives and negatives, defined here). A **true positive (TP)** is a voxel correctly called foreground; a **false positive (FP)** is background wrongly called foreground; a **false negative (FN)** is foreground the model missed. Then the overlap is $|P\cap G| = \mathrm{TP}$, the predicted region is $|P| = \mathrm{TP}+\mathrm{FP}$, and the true region is $|G| = \mathrm{TP}+\mathrm{FN}$. Substituting:

$$\mathrm{DSC} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}.$$

Notice what is missing: **true negatives (TN)** — the vast ocean of correctly-ignored background — never appear. That single absence is why Dice resists **class imbalance** (the topic of §14): a scan may be less than one percent tumour, yet a lazy "everything is background" prediction scores $\mathrm{DSC}=0$, not 99%, because it has zero true positives.

![Which voxels Dice counts: overlap TP, prediction-only FP, truth-only FN, and the ignored background TN](figures/diagrams/s13_dice_regions.png)

!!! intuition "Intuition"
    Draw two overlapping blobs, the prediction and the truth. Dice counts the shared area **twice** (that is the $2$ in the numerator) and divides by the two blob areas added together. Perfect stacking gives $2A/(A+A)=1$; no touching gives $0$. Without the factor of two, a perfect match would embarrassingly score only $\tfrac{1}{2}$.

### 13.2 Dice is exactly the F1 score

Precision and recall (§16) let us name the two ways a mask can be wrong. **Precision** $=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ asks "of the voxels I called tumour, how many really were?"; **recall** (sensitivity) $=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ asks "of the true tumour voxels, how many did I find?". Their **harmonic mean** is the **F1 score**. Remarkably, F1 *is* Dice — the two words name one quantity ([Chen Riang](https://chenriang.me/f1-equal-dice-coefficient.html)). Here is the full proof.

$$\begin{aligned}
F_1 &= \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}
     = \frac{2\,\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}\,\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}}{\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}+\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}} \\[2mm]
    &= \frac{\dfrac{2\,\mathrm{TP}^2}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})}}{\dfrac{\mathrm{TP}(\mathrm{TP}+\mathrm{FN})+\mathrm{TP}(\mathrm{TP}+\mathrm{FP})}{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})}} \\[2mm]
    &= \frac{2\,\mathrm{TP}^2}{\mathrm{TP}(\mathrm{TP}+\mathrm{FN})+\mathrm{TP}(\mathrm{TP}+\mathrm{FP})}
     = \frac{2\,\mathrm{TP}^2}{\mathrm{TP}\big[2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}\big]}
     = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
\end{aligned}$$

Step by step: line 1 writes the harmonic-mean definition and plugs in precision and recall. Line 2 puts the numerator over the common product $(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})$ and combines the two denominator fractions over that *same* product. Line 3 cancels the common product between the big numerator and denominator, factors $\mathrm{TP}$ out of the denominator, and cancels one $\mathrm{TP}$ against the numerator's $\mathrm{TP}^2$. The result is identical to the Dice TP/FP/FN form above — so $\mathrm{DSC}=F_1$. $\blacksquare$

!!! gotcha "Gotcha"
    Dice (F1) is **not** IoU / Jaccard, even though people swap the names. They are monotonically linked — $\mathrm{DSC}=2\,\mathrm{IoU}/(1+\mathrm{IoU})$ — but Dice is always $\ge$ IoU (except at $0$ and $1$) because Dice counts the overlap twice. Reporting one and calling it the other silently inflates or deflates your headline number.

### 13.3 From metric to loss: soft Dice

Dice as defined uses hard $0/1$ masks. That makes it a *step* function of the network's output: nudging a probability from $0.49$ to $0.51$ flips a voxel and jerks the score, so there is no usable **gradient** (the slope §17's gradient descent needs). Milletari et al. fixed this in the **V-Net paper** (2016) by feeding the network's *continuous* probabilities straight into the formula instead of thresholded labels ([V-Net](https://arxiv.org/abs/1606.04797)). Let $p_i\in[0,1]$ be the predicted probability at voxel $i$ (after the sigmoid/softmax of §12) and $g_i\in\{0,1\}$ the true label, over $N$ voxels. V-Net's **soft Dice** uses a squared denominator:

$$L_{\text{Dice}} = 1 - D,\qquad D = \frac{2\sum_{i=1}^{N} p_i\,g_i}{\sum_{i=1}^{N} p_i^{2} + \sum_{i=1}^{N} g_i^{2}}.$$

Because $D$ is now a smooth function of the $p_i$, we can minimise the loss $1-D$ by gradient descent. Many libraries instead use a **linear** denominator with a smoothing constant $\varepsilon>0$,

$$D = \frac{2\sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon},$$

which is the direct continuous analogue of the set-based DSC; the $\varepsilon$ prevents division by zero and makes $D\approx 1$ when both masks are empty. Milletari et al.'s stated motivation was exactly the medical case where foreground is tiny: soft Dice is a ratio of overlap to region size, so it up-weights the rare class automatically, with no hand-tuned class weights ([V-Net](https://arxiv.org/abs/1606.04797)).

!!! intuition "Intuition"
    Soft Dice lets the probabilities **vote**. A voxel that is 90% sure of tumour contributes $0.9$ to the overlap instead of a flat yes/no. Now the score glides smoothly as the network adjusts its confidence, so there is always a slope to follow downhill.

### 13.4 The soft-Dice gradient

To train, we need $\partial D/\partial p_j$, the sensitivity of the overlap score to one output $p_j$. Name the numerator $U=2\sum_i p_i g_i$ and denominator $V=\sum_i p_i^2+\sum_i g_i^2$. Only the $i=j$ term of each sum contains $p_j$, so $\partial U/\partial p_j = 2g_j$ and $\partial V/\partial p_j = 2p_j$ (the $\sum g_i^2$ part is constant). The **quotient rule** $\partial(U/V)/\partial p_j=(U'V-UV')/V^2$ then gives

$$\frac{\partial D}{\partial p_j} = \frac{(2g_j)V - \big(2\sum_i p_i g_i\big)(2p_j)}{V^2}
= 2\,\frac{ g_j\big(\sum_i p_i^2 + \sum_i g_i^2\big) - 2p_j\big(\sum_i p_i g_i\big) }{\big(\sum_i p_i^2 + \sum_i g_i^2\big)^2},$$

matching the V-Net paper exactly ([V-Net](https://ar5iv.labs.arxiv.org/abs/1606.04797)). Since $L_{\text{Dice}}=1-D$, we have $\partial L/\partial p_j = -\partial D/\partial p_j$. A quick **sign check** confirms the directions are right. At a true foreground voxel ($g_j=1$) the numerator is $2(V-2p_jA)$ with $A\equiv\sum_i p_ig_i\ge 0$; for $p_j$ not already saturated this is positive, so $\partial D/\partial p_j>0$ and $\partial L/\partial p_j<0$ — descent **raises** $p_j$ toward 1. At a background voxel ($g_j=0$) the first term vanishes, leaving $\partial D/\partial p_j = -4p_jA/V^2\le 0$, so $\partial L/\partial p_j\ge 0$ — descent **lowers** $p_j$ toward 0, suppressing false positives. Both are exactly what we want.

!!! example "Example question"
    A predicted mask has 5 foreground voxels; the ground truth has 4; they share 3. Compute Dice, precision, recall and F1, and confirm Dice = F1.

    **Solution:** The shared voxels are true positives, so $\mathrm{TP}=3$; the predicted-but-not-true are $\mathrm{FP}=5-3=2$; the true-but-missed are $\mathrm{FN}=4-3=1$. Then $\mathrm{DSC}=\dfrac{2\cdot 3}{2\cdot 3+2+1}=\dfrac{6}{9}=0.667$, matching the set form $\dfrac{2\cdot 3}{5+4}=\dfrac{6}{9}=0.667$. Precision $=3/5=0.60$, recall $=3/4=0.75$, and $F_1=\dfrac{2(0.60)(0.75)}{0.60+0.75}=\dfrac{0.90}{1.35}=0.667$. So $F_1=\mathrm{DSC}=0.667$. $\checkmark$

!!! example "Example question"
    For $p=[0.9,\,0.6,\,0.2,\,0.1]$ and $g=[1,1,0,0]$, compute the V-Net soft-Dice loss and the loss gradient $\partial L/\partial p$ at voxel 1 ($p_1=0.9$, foreground) and voxel 3 ($p_3=0.2$, background). Interpret the signs.

    **Solution:** Overlap $A=\sum p_ig_i = 0.9+0.6+0+0 = 1.5$, so numerator $2A=3.0$. Denominator $V=\sum p_i^2+\sum g_i^2=(0.81+0.36+0.04+0.01)+(1+1)=1.22+2=3.22$. Thus $D=3.0/3.22=0.9317$ and $L=1-D=0.0683$. With $V^2=3.22^2=10.3684$ and $\partial D/\partial p_j=2[g_jV-2p_jA]/V^2$:

    - Voxel 1: $2[1\cdot 3.22 - 2(0.9)(1.5)]/10.3684 = 2[3.22-2.70]/10.3684 = 0.1003$, so $\partial L/\partial p_1 = -0.1003$. Negative — descent **raises** $p_1$ toward 1. Good, it is real foreground.
    - Voxel 3: $2[0 - 2(0.2)(1.5)]/10.3684 = 2(-0.60)/10.3684 = -0.1157$, so $\partial L/\partial p_3 = +0.1157$. Positive — descent **lowers** $p_3$ toward 0. Good, it is a false positive being suppressed.

### 13.5 Why blend with cross-entropy: DiceCE

Pure soft Dice has a weak spot. Early in training, predictions barely overlap the truth, so the numerator $\sum p_ig_i\approx 0$, gradients are small and noisy, and one hard example can dominate a whole batch. Cross-entropy (§12), by contrast, hands every voxel its own clear, always-defined push. So the standard recipe simply **adds** them into the **DiceCE loss**:

$$L_{\text{DiceCE}} = \lambda_{\text{Dice}}\,L_{\text{Dice}} + \lambda_{\text{CE}}\,L_{\text{CE}},\qquad
L_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\big[g_i\log p_i + (1-g_i)\log(1-p_i)\big],$$

with nonnegative weights $\lambda_{\text{Dice}},\lambda_{\text{CE}}$ (often both $1$). Cross-entropy supplies a dense, stable per-voxel gradient everywhere; Dice steers the whole region toward the overlap you are actually scored on. This combination — as used in nnU-Net and MONAI's `DiceCELoss` — is the de-facto default for BraTS-style 3-D segmentation ([MONAI](https://docs.monai.io/en/stable/losses.html)). (The original U-Net paper used *neither* Dice loss; it used a distance-weighted cross-entropy to separate touching cells — Dice loss is a later, widely adopted convention, [U-Net](https://arxiv.org/abs/1505.04597).)

![DiceCE: logits become probabilities, then feed both a Dice loss and a cross-entropy loss whose weighted sum is the total](figures/diagrams/s13_dicece_flow.png)

!!! intuition "Intuition"
    DiceCE is **compass plus map**. Cross-entropy is the compass — a clear local push for every voxel. Dice is the map — it keeps the whole outlined region aiming at maximum overlap. Early on the map is blurry (tiny Dice gradient), so the compass keeps you walking until the masks overlap enough for Dice to take charge.

!!! example "Example question"
    Using the same $p=[0.9,0.6,0.2,0.1]$, $g=[1,1,0,0]$, compute binary cross-entropy and the DiceCE loss with $\lambda_{\text{Dice}}=\lambda_{\text{CE}}=1$. Which term dominates, and why does combining help?

    **Solution:** Per-voxel BCE $=-[g_i\log p_i+(1-g_i)\log(1-p_i)]$ (natural log). Voxel 1 ($g=1$): $-\ln 0.9 = 0.1054$. Voxel 2 ($g=1$): $-\ln 0.6 = 0.5108$. Voxel 3 ($g=0$): $-\ln 0.8 = 0.2231$. Voxel 4 ($g=0$): $-\ln 0.9 = 0.1054$. Mean $L_{\text{CE}} = (0.1054+0.5108+0.2231+0.1054)/4 = 0.9447/4 = 0.2362$. From the earlier box $L_{\text{Dice}}=0.0683$. So $L_{\text{DiceCE}} = 1(0.0683)+1(0.2362)=0.3045$. Here CE ($0.236$) dominates Dice ($0.068$): the masks already overlap well, so Dice is nearly satisfied, but CE still penalises each imperfect probability — the under-confident $0.6$ and the leftover $0.2$ false positive — supplying the stronger push to sharpen them while Dice guards the global overlap.

!!! gotcha "Gotcha"
    **The empty-mask trap.** If a slice truly has no tumour ($g$ all zeros) and the model correctly predicts none ($p$ all zeros), naive Dice is $0/0$ — undefined. The $\varepsilon$ smoothing removes the NaN, but the resulting value ($\approx 1$ or $\approx 0$ depending on where $\varepsilon$ sits) is a bookkeeping artefact, not real skill, and can noticeably shift a dataset-averaged Dice. Also mind **aggregation**: per-image Dice then averaged (macro) differs from pooling all voxels into one Dice; BraTS scores per case then averages (§16), so match your training reduction to how you will be graded. Finally, for BraTS the scored targets are overlapping *regions* (whole tumour, tumour core, enhancing tumour of §7), not the raw label channels — multi-class Dice is a design choice, not automatic.
