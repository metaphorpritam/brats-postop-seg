## 14. Class imbalance: when the target is a tiny fraction of voxels {#imbalance}

In §9 we framed segmentation as labelling every **voxel** (a 3D pixel — one cell of the scan volume) with a class, and in §7 we met the four tumour labels (plus background). Here we confront an uncomfortable fact about that task: the things we care about are vanishingly rare. A single BraTS volume is roughly $240 \times 240 \times 155 \approx 8.9$ million voxels, yet the tumour sub-regions — enhancing tumour, resection cavity, surrounding edema — can occupy well under 1% of them; reported foreground-to-background ratios run from about 1:100 to 1:5000 or worse [Sudre et al. 2017](https://arxiv.org/abs/1707.03237). Everything else is **background**: healthy brain, the zeros left where the skull was stripped away, and air. This lopsidedness is called **class imbalance**, and it quietly breaks the two tools a newcomer reaches for first — **accuracy** as a score and **plain cross-entropy** as a loss (§12).

!!! intuition "Intuition"
    Picture a 1,000-question exam where 999 questions are trivially easy and 1 is genuinely hard. If your grade is the plain average, you can score 99.9% while getting the single hard question wrong every time. Accuracy and plain cross-entropy grade a segmentation exactly this way: the 999 easy background voxels set your score, and the one tumour voxel that actually matters is a rounding error. The remedies in this section all change the grading so that finding the hard thing is what counts.

### Why accuracy degenerates

**Accuracy** is the fraction of voxels whose predicted label matches the truth. Write $\hat{y}_i$ for the predicted label of voxel $i$, $y_i$ for its true label, and $\mathbb{1}[\cdot]$ for the **indicator** (1 when the bracketed statement is true, 0 otherwise). Then

$$\mathrm{Acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\hat{y}_i = y_i\right], \qquad \mathrm{Acc}_{\text{all-bg}} = \frac{N_{bg}}{N}.$$

Here $N$ is the total number of voxels and $N_{bg}$ the number of true background voxels. The second expression is the accuracy of a lazy model that predicts "background" for *every* voxel: it gets every background voxel right and every foreground voxel wrong, so its score is just the background fraction $N_{bg}/N$ — about 0.999 under heavy imbalance, despite detecting zero tumour. Accuracy therefore cannot tell a useless model from a good one, which is why BraTS scores overlap metrics on the foreground instead (Dice/IoU, §13, §16) [motivation stated in V-Net, Milletari et al. 2016](https://arxiv.org/abs/1606.04797).

### Why plain cross-entropy makes the same mistake

Recall the multi-class pixel-wise cross-entropy loss from §12:

$$L_{\mathrm{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} g_{i,c}\,\log p_{i,c}.$$

$N$ is the voxel count, $C$ the number of classes (background included), $g_{i,c}$ the **one-hot** ground truth (1 if voxel $i$ truly belongs to class $c$, else 0), and $p_{i,c}$ the softmax probability the model assigns to that class. Because each voxel contributes exactly one live term $-\log p_{i,\text{true}}$ and we average over all $N$ voxels *equally*, classes with more voxels contribute proportionally more terms. The millions of easy background voxels therefore dominate both the loss and the gradient. Let us prove that this pushes the model to the empty mask.

#### Derivation 1 — plain CE's optimum is the background prevalence (the "predict background" trap)

Consider a binary problem (foreground vs. background) with a single output probability $p$ of foreground, and force the network to output the *same* constant $p$ at every voxel. This "featureless" worst case isolates the loss's built-in bias. There are $N_{fg}$ foreground voxels and $N_{bg}$ background voxels, with $N = N_{fg} + N_{bg}$. The mean cross-entropy — foreground voxels pay $-\log p$ for wanting $p$ high, background voxels pay $-\log(1-p)$ for wanting it low — is

$$L(p) = \frac{1}{N}\Big[\sum_{i\in fg}\!-\log p \; + \sum_{j\in bg}\!-\log(1-p)\Big].$$

Every foreground term is identical (constant $p$), so the sum over foreground is just $N_{fg}$ copies of $-\log p$; likewise for background. That collapses the objective to a function of the single number $p$:

$$L(p) = -\frac{N_{fg}}{N}\log p \; -\; \frac{N_{bg}}{N}\log(1-p).$$

Differentiate term by term. Using $\tfrac{d}{dp}[-\log p] = -\tfrac{1}{p}$ and $\tfrac{d}{dp}[-\log(1-p)] = +\tfrac{1}{1-p}$ (the chain rule's inner derivative of $(1-p)$ is $-1$, which cancels the leading minus):

$$\frac{dL}{dp} = -\frac{N_{fg}}{N}\cdot\frac{1}{p} \; +\; \frac{N_{bg}}{N}\cdot\frac{1}{1-p}.$$

The function is convex in $p$, so its global minimum is where the derivative is zero:

$$-\frac{N_{fg}}{N}\cdot\frac{1}{p} + \frac{N_{bg}}{N}\cdot\frac{1}{1-p} = 0.$$

Multiply through by $N$ (the common $1/N$ cancels) and move the background term to the other side:

$$\frac{N_{fg}}{p} = \frac{N_{bg}}{1-p}.$$

Cross-multiply the two fractions:

$$N_{fg}(1-p) = N_{bg}\,p.$$

Expand the left side and collect the $p$ terms, using $N_{fg} + N_{bg} = N$:

$$N_{fg} - N_{fg}\,p = N_{bg}\,p \;\Rightarrow\; N_{fg} = p\,(N_{fg} + N_{bg}) = p\,N.$$

Solve for $p$:

$$\boxed{\,p^{*} = \frac{N_{fg}}{N} = \text{prevalence}\,}.$$

The loss-minimizing constant output is exactly the foreground **prevalence** (the fraction of voxels that are foreground). With $N_{fg} = 1{,}000$ and $N = 1{,}000{,}000$ this is $p^{*} = 0.001$. Since $0.001 < 0.5$, thresholding at one half labels every voxel background. Plain cross-entropy literally rewards the empty mask — that is the degeneracy.

![Two forces pulling a prediction: many background voxels versus few foreground voxels](figures/diagrams/s14_tug_of_war.png)

!!! intuition "Intuition"
    A loss is a tug-of-war of gradients. Every background voxel pulls the prediction toward "background"; every foreground voxel pulls the other way. Plain cross-entropy hands one rope to each voxel, so 999,000 background ropes trivially overpower 1,000 foreground ropes and drag the model to a blank answer. Every remedy below is a way to even out the rope count so the foreground can actually pull.

### Remedy 1 — reweight the classes

If unequal counts are the problem, attach a per-class weight $w_c$ to undo them. **Class-weighted cross-entropy** is

$$L_{\mathrm{WCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} w_c\, g_{i,c}\,\log p_{i,c}, \qquad w_c = \frac{1}{N_c},$$

where $N_c$ is the number of voxels of class $c$; the **inverse-frequency** choice $w_c = 1/N_c$ gives rare classes large weights. This idea is old: the original U-Net used a per-pixel weighted cross-entropy with a weight map $w(x)$ to compensate for class frequency and emphasize the hard borders between touching cells [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597).

#### Derivation 2 — inverse-frequency weighting moves the optimum to 0.5

Reuse the constant-prediction setup, now with weights $w_{fg}, w_{bg}$ on the two terms:

$$L(p) = -\frac{1}{N}\big[w_{fg} N_{fg}\log p + w_{bg} N_{bg}\log(1-p)\big].$$

Differentiate exactly as before (the weights ride along as constant multipliers) and set to zero:

$$\frac{dL}{dp} = -\frac{w_{fg}N_{fg}}{N}\cdot\frac{1}{p} + \frac{w_{bg}N_{bg}}{N}\cdot\frac{1}{1-p} = 0.$$

Cancel $1/N$ and cross-multiply:

$$w_{fg}N_{fg}(1-p) = w_{bg}N_{bg}\,p.$$

Collect the $p$ terms and solve — the weighted analogue of the prevalence result:

$$p^{*} = \frac{w_{fg}N_{fg}}{w_{fg}N_{fg} + w_{bg}N_{bg}}.$$

Now choose inverse-frequency weights $w_c = 1/N_c$, so $w_{fg}N_{fg} = 1$ and $w_{bg}N_{bg} = 1$ — each class carries equal total mass in the loss. Then

$$\boxed{\,p^{*} = \frac{1}{1+1} = 0.5\,}.$$

The degenerate optimum now sits exactly on the decision boundary. The lazy constant output no longer favours background, so the network must actually use image features to separate the classes. This is precisely why weighted cross-entropy counteracts imbalance.

A more sophisticated relative, the **Generalized Dice Loss** (GDL), weights each class by the inverse *square* of its volume and is designed for "highly unbalanced segmentations," outperforming weighted CE and plain Dice at extreme imbalance [Sudre et al. 2017](https://arxiv.org/abs/1707.03237):

$$L_{\mathrm{GDL}} = 1 - 2\,\frac{\sum_{l=1}^{C} w_l \sum_{i} g_{i,l}\,p_{i,l}}{\sum_{l=1}^{C} w_l \sum_{i} \left(g_{i,l} + p_{i,l}\right)}, \qquad w_l = \frac{1}{\left(\sum_{i} g_{i,l}\right)^{2}} = \frac{1}{V_l^{2}}.$$

Here $l$ indexes classes, $V_l = \sum_i g_{i,l}$ is class $l$'s ground-truth volume, and $g_{i,l}, p_{i,l}$ are its one-hot label and predicted probability at voxel $i$. Why the *square*? Consider a near-perfect prediction, $p_{i,l} \approx g_{i,l}$, so the overlap sum $\sum_i g_{i,l} p_{i,l} \approx \sum_i g_{i,l} = V_l$. Unweighted, each class's pull on the loss scales with its size $V_l$, so background ($V_{bg} = 999{,}000$) outweighs foreground ($V_{fg} = 1{,}000$) by about 999:1. Multiplying by $w_l = 1/V_l^2$ turns the size-$V_l$ contribution into $V_l / V_l^2 = 1/V_l$, and the ratio of contributions inverts to $V_{bg}/V_{fg} \approx 999$ *in favour of* foreground. The $1/V^2$ choice deliberately trades background dominance for foreground emphasis — the right trade when missing a small lesion is the costly error. (Real implementations clamp $w_l$ so a single-voxel class cannot produce an astronomically large weight.)

### Remedy 2 — use an overlap loss (Dice)

The deepest fix is to change the objective so background count cannot be gamed at all. **Dice loss** (from §13) is built on region overlap:

$$L_{\mathrm{Dice}} = 1 - D = 1 - \frac{2\sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon},$$

with $p_i$ the predicted foreground probability, $g_i \in \{0,1\}$ the label, and $\varepsilon$ a small smoothing constant (e.g. $10^{-5}$) that keeps the ratio defined when an image has no foreground. V-Net introduced this precisely so that training need not weight classes by hand [Milletari et al. 2016](https://arxiv.org/abs/1606.04797).

#### Derivation 3 — Dice punishes the empty mask that CE rewards

Evaluate the all-background predictor, $p_i = 0$ for every voxel. The overlap term $\sum_i p_i g_i = 0$ and the predicted-mass term $\sum_i p_i = 0$, so

$$L_{\mathrm{Dice}} = 1 - \frac{2\cdot 0 + \varepsilon}{0 + \sum_i g_i + \varepsilon} = 1 - \frac{\varepsilon}{\sum_i g_i + \varepsilon}.$$

With any real foreground present, $\sum_i g_i = N_{fg} \gg \varepsilon$, so the fraction $\varepsilon / (N_{fg} + \varepsilon) \approx 0$ and

$$L_{\mathrm{Dice}} \approx 1.$$

The *same* prediction that gave plain CE its **minimum** gives Dice its **maximum** (worst) value. Dice measures overlap with the small foreground and never sees the background count in a way that predicting nothing can exploit — the structural reason it is imbalance-robust.

!!! intuition "Intuition"
    Dice loss never counts background directly. It only asks: "of the tumour I predicted and the tumour that exists, how much overlaps?" Because the denominator is the size of the *small* foreground, not the *huge* image, blowing up on the background costs the model nothing and helps it nothing. That built-in normalization is why one Dice loss works across wildly different tumour sizes with no hand-tuned weights.

A close relative attacks imbalance at the voxel level: **Focal loss** multiplies cross-entropy by a modulating factor $(1 - p_{t})^{\gamma}$,

$$L_{\mathrm{FL}} = -\frac{1}{N}\sum_{i=1}^{N} \alpha_{t}\,(1 - p_{t,i})^{\gamma}\,\log p_{t,i}, \qquad p_{t,i} = p_{i,\,y_i},$$

where $p_{t,i}$ is the probability the model assigns to voxel $i$'s *true* class, $\gamma \ge 0$ is the focusing parameter ($\gamma = 2$ is typical), and $\alpha_t$ is an optional per-class balance weight. The factor is near 0 for easy, well-classified voxels ($p_t \to 1$) and near 1 for hard ones, so the abundant easy background terms are suppressed. With $\gamma = 2$, an easy voxel at $p_t = 0.99$ is down-weighted by $(1-0.99)^2 = 10^{-4}$ — 10,000 times — relative to its raw CE term [Lin et al. 2017](https://arxiv.org/abs/1708.02002).

### Remedy 3 — drop the background channel

MONAI's `DiceLoss(include_background=False)` simply removes channel 0 (background, by convention) from the Dice average. The docs explain: "if the non-background segmentations are small compared to the total image size they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence" [MONAI 1.4 loss docs](https://monai.readthedocs.io/en/1.4.0/losses.html).

### Remedy 4 — fix the data, not the loss

**Class-balanced patch sampling** attacks imbalance before the loss ever sees a voxel. Instead of cropping training patches at random — where most windows land on empty brain — a sampler like MONAI's `RandCropByPosNegLabel` forces a chosen fraction of patches to be centred on foreground voxels, so the network sees tumour on most steps [MONAI transforms](https://docs.monai.io/en/stable/transforms.html).

![Random crops mostly miss the tumour; foreground-centred crops guarantee tumour in each patch](figures/diagrams/s14_sampling.png)

!!! intuition "Intuition"
    Balanced sampling and loss weighting attack the same imbalance from opposite ends. Weighting keeps the data biased and fixes the objective; sampling keeps the objective simple and fixes the data the model sees. They compose — nnU-Net-style BraTS pipelines typically do both (force roughly a third to a half of patches to contain tumour *and* use Dice+CE) because each alone leaves residual imbalance.

!!! gotcha "Gotcha"
    None of these fixes is free. (1) Dice loss has noisy, sometimes unstable gradients when the foreground is tiny or absent, which is why it is almost always paired with a CE or focal term (DiceCE, DiceFocal) for smoother per-voxel gradients — the standard BraTS choice. (2) `include_background=False` only excludes background from the *Dice average*; a softmax model still models the background probability. Reported Dice numbers with vs. without background are not comparable, since an easy background channel (Dice $\approx 0.999$) inflates the mean — always state which convention you used. (3) Over-cranked rare-class weights push the model to over-predict foreground, giving false positives and a fragmented mask; weights are usually clamped or normalized. (4) Balanced sampling biases the network's output priors — a model trained on 50%-tumour patches has seen far more tumour than exists at inference, so probabilities may be miscalibrated and need whole-volume sliding-window inference or threshold tuning. (5) A high average Dice can still hide clinically important misses of small satellite lesions; complement it with boundary metrics like Hausdorff distance and per-region ET/TC/WT reporting (§16).

!!! example "Example question"
    A skull-stripped BraTS volume has $N = 1{,}000{,}000$ voxels, of which $N_{fg} = 1{,}000$ are enhancing tumour and the rest background. (a) What accuracy does a model that predicts "background" everywhere achieve? (b) What is its foreground Dice? (c) If it instead outputs a single constant foreground probability $p$ everywhere, what $p$ minimizes plain cross-entropy, and what mask does thresholding at 0.5 produce?

    **Solution.**
    (a) $\mathrm{Acc} = N_{bg}/N = 999{,}000 / 1{,}000{,}000 = 0.999 = 99.9\%$. The all-background model looks almost perfect.
    (b) It predicts no foreground, so true positives $\mathrm{TP} = 0$. Then $\mathrm{Dice} = \tfrac{2\cdot\mathrm{TP}}{2\cdot\mathrm{TP} + \mathrm{FP} + \mathrm{FN}} = \tfrac{2\cdot 0}{0 + 0 + 1000} = 0$. Dice, unlike accuracy, exposes that the model found nothing.
    (c) From Derivation 1, the CE-minimizing constant is the prevalence $p^{*} = N_{fg}/N = 1000/1{,}000{,}000 = 0.001$. Since $0.001 < 0.5$, thresholding labels every voxel background — plain CE's optimum *is* the useless all-background mask. That is the whole reason we abandon accuracy and unweighted CE under imbalance.

!!! example "Example question"
    A prediction for the enhancing-tumour class has $\mathrm{TP} = 800$, $\mathrm{FP} = 100$, $\mathrm{FN} = 200$. (a) Compute the Dice coefficient and Dice loss. (b) In a 4-class softmax model the per-class Dice scores are background $= 0.999$, class1 $= 0.60$, class2 $= 0.70$, class3 $= 0.75$. Compute the mean Dice with `include_background=True` vs. `False`, and explain the gap.

    **Solution.**
    (a) $\mathrm{Dice} = \tfrac{2\cdot 800}{2\cdot 800 + 100 + 200} = \tfrac{1600}{1900} = 0.8421$; Dice loss $= 1 - 0.8421 = 0.1579$.
    (b) With background: $\tfrac{0.999 + 0.60 + 0.70 + 0.75}{4} = \tfrac{3.049}{4} = 0.7623$. Without: $\tfrac{0.60 + 0.70 + 0.75}{3} = \tfrac{2.05}{3} = 0.6833$. The trivially easy background channel (0.999) pulls the reported mean up by about 0.079 and dilutes each foreground class's share of the gradient from $1/3$ to $1/4$. Excluding it makes both the metric and the training signal reflect only the structures we care about — hence MONAI's guidance.

!!! example "Example question"
    For the same volume ($N_{fg} = 1{,}000$, $N_{bg} = 999{,}000$), switch to class-weighted CE with $w_c = 1/N_c$. Show the constant-prediction optimum becomes $p^{*} = 0.5$, and compute the ratio by which GDL's $1/V^2$ weights favour foreground over background.

    **Solution.**
    Weighted-CE optimum: $p^{*} = \tfrac{w_{fg}N_{fg}}{w_{fg}N_{fg} + w_{bg}N_{bg}}$. With $w_{fg} = 1/1000$ and $w_{bg} = 1/999000$, we get $w_{fg}N_{fg} = 1$ and $w_{bg}N_{bg} = 1$, so $p^{*} = \tfrac{1}{1+1} = 0.5$ — the lazy constant now sits on the decision boundary and can no longer default to background.
    GDL weights: $w_{fg} = 1/1000^2 = 10^{-6}$; $w_{bg} = 1/999000^2 \approx 1.002\times 10^{-12}$. The raw weight ratio is $w_{fg}/w_{bg} = 999000^2/1000^2 = 999^2 \approx 998{,}001$ — foreground weighted about $10^6\times$ more per voxel. After multiplying by each class's volume, the net contribution ratio is $V_{bg}/V_{fg} = 999000/1000 \approx 999$ in favour of foreground: GDL inverts the original 1:999 background dominance into roughly 999:1 foreground emphasis, deliberately protecting the small structure.

Putting it together: a common BraTS recipe combines Dice with a (weighted or focal) cross-entropy term, sets `include_background=False`, and trains on positively-biased patches — attacking imbalance at the loss *and* the data level at once. The next section (§15) turns to how the intensities themselves are normalized before any of this training begins.
