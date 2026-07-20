## 15. Two kinds of normalization (do not conflate them) {#normalization}

The word **normalization** does two completely different jobs in an MRI segmentation pipeline, and mixing them up is a classic source of confusion. Both jobs share the same tiny piece of arithmetic — *subtract a mean, divide by a standard deviation* — but they act at different places in the stack, for different reasons.

1. **Input intensity normalization** rescales the *raw voxel numbers of a scan* before the network ever sees them.
2. **Network normalization layers** (batch norm, instance norm) rescale the *hidden activations inside the network* during training, to keep the learning well-behaved.

Same verb, different object. This section defines both, proves the two properties that make the good recipes good, and works small numbers end to end. (A **voxel** is a 3D pixel: one cell of the volume. A **modality** is one MRI sequence — see §6.)

![Where the two normalizations sit in the pipeline: input z-score on the raw MRI, instance-norm layers inside the U-Net.](figures/diagrams/s15_pipeline.png)

### Why input intensity normalization is mandatory

Unlike a CT scan, whose numbers (Hounsfield units) have a fixed physical meaning, **MRI signal has no absolute scale**: the same tissue can read 140 on one scanner and 330 on another, depending on the coil, sequence settings, and calibration [nnU-Net (Isensee et al., Nature Methods 2021)](https://arxiv.org/pdf/1809.10486). So before training we must put every volume onto a comparable numeric footing. Two candidate recipes:

**Naive global-max scaling** divides every voxel by the single brightest voxel:

$$\tilde{x}_i = \frac{x_i}{\max_{j} x_j}.$$

Here $x_i$ is the raw intensity at voxel $i$ and $\max_j x_j$ is the brightest voxel in the whole image; the output $\tilde{x}_i$ lands in $[0,1]$ for nonnegative inputs. Note two weaknesses baked into the formula: the scale is fixed by *one* extreme voxel, and there is *no subtraction*, so it cannot remove a brightness offset.

**Per-modality z-score** subtracts the mean and divides by the standard deviation, computed over the **brain mask** $\mathcal{B}$ (the nonzero foreground voxels) only:

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sigma_{\mathcal{B}} + \epsilon}, \qquad \mu_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{j\in\mathcal{B}} x_j, \qquad \sigma_{\mathcal{B}} = \sqrt{\frac{1}{|\mathcal{B}|}\sum_{j\in\mathcal{B}} (x_j-\mu_{\mathcal{B}})^2}.$$

$\mathcal{B}$ is the set of brain voxels, $|\mathcal{B}|$ how many there are; $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}$ are the mean and standard deviation over that mask; $\epsilon$ is a tiny constant (e.g. $10^{-8}$) that stops division by zero; $\hat{x}_i$ is the normalized value. It is computed **separately for each modality** (T1, T1c, T2, FLAIR) because their intensity distributions differ. This is exactly what the self-configuring nnU-Net framework — the dominant BraTS baseline — does: per-patient, per-modality z-score for MRI, but clipping-plus-dataset-normalization for CT, because CT values *are* quantitative [nnU-Net (Isensee et al.)](https://arxiv.org/pdf/1809.10486). When its `use_mask_for_norm` flag is set, nnU-Net computes the statistics over nonzero voxels only, then resets background to 0 so the huge black surround cannot dominate [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet/discussions/2363).

### Derivation 1 — z-score outputs mean 0, std 1

Given $N$ values $x_1,\dots,x_N$ with mean $\mu=\frac1N\sum_i x_i$ and standard deviation $\sigma=\sqrt{\frac1N\sum_i (x_i-\mu)^2}$, define $z_i=\frac{x_i-\mu}{\sigma}$ (dropping $\epsilon$, which only guards the denominator). First the mean of the $z_i$:

$$\begin{aligned}
\bar{z} &= \frac{1}{N}\sum_{i=1}^N z_i = \frac{1}{N}\sum_{i=1}^N \frac{x_i-\mu}{\sigma} && \text{substitute the definition of } z_i,\\
&= \frac{1}{\sigma}\cdot\frac{1}{N}\sum_{i=1}^N (x_i-\mu) && \text{pull the constant } \tfrac{1}{\sigma}\text{ out of the sum},\\
&= \frac{1}{\sigma}\Big(\underbrace{\tfrac{1}{N}\sum_i x_i}_{=\,\mu} - \underbrace{\tfrac{1}{N}\cdot N\mu}_{=\,\mu}\Big) && \text{split the sum; } \mu \text{ is added } N \text{ times},\\
&= \frac{1}{\sigma}(\mu-\mu) = 0. &&
\end{aligned}$$

Centering removed the absolute-brightness offset. Now the variance. Because $\bar{z}=0$:

$$\begin{aligned}
\operatorname{Var}(z) &= \frac{1}{N}\sum_{i=1}^N (z_i-\bar{z})^2 = \frac{1}{N}\sum_{i=1}^N z_i^2 && \text{use } \bar{z}=0,\\
&= \frac{1}{N}\sum_{i=1}^N \frac{(x_i-\mu)^2}{\sigma^2} = \frac{1}{\sigma^2}\cdot\underbrace{\frac{1}{N}\sum_i (x_i-\mu)^2}_{=\,\sigma^2} && \text{substitute } z_i,\ \text{pull out } \tfrac{1}{\sigma^2},\\
&= \frac{\sigma^2}{\sigma^2} = 1. &&
\end{aligned}$$

So z-score *always* produces mean 0, standard deviation 1, whatever the original units.

### Derivation 2 — z-score is invariant to a scanner shift $y=a\,x+b$

Model a second scanner that records $y_i = a\,x_i + b$ for every voxel, with gain $a>0$ (contrast) and offset $b$ (brightness). We show the z-scores are unchanged.

$$\begin{aligned}
\mu_y &= \frac{1}{N}\sum_i (a x_i + b) = a\mu_x + b && \text{linearity; } b \text{ averages to } b,\\
y_i - \mu_y &= (a x_i + b) - (a\mu_x + b) = a(x_i-\mu_x) && \text{the } +b,\,-b \text{ cancel — offset gone},\\
\sigma_y^2 &= \frac{1}{N}\sum_i a^2 (x_i-\mu_x)^2 = a^2\sigma_x^2 \;\Rightarrow\; \sigma_y = a\,\sigma_x && \text{since } a>0,\ |a|=a,\\
\hat{y}_i &= \frac{y_i-\mu_y}{\sigma_y} = \frac{a(x_i-\mu_x)}{a\,\sigma_x} = \frac{x_i-\mu_x}{\sigma_x} = \hat{x}_i && \text{the gain } a \text{ cancels.}
\end{aligned}$$

Both scanner parameters vanish: any two acquisitions related by a linear intensity change produce *identical* z-scored inputs. Global-max fails this — it has no subtraction, so the offset $b$ survives.

### Derivation 3 — worked numbers: z-score vs global-max

Take five brain voxels $x=\{100,120,140,160,180\}$, $N=5$.

- Mean: $\mu=\frac{700}{5}=140$. Deviations $\{-40,-20,0,20,40\}$; squares $\{1600,400,0,400,1600\}$ sum to $4000$.
- Variance $\sigma^2=\frac{4000}{5}=800$, so $\sigma=\sqrt{800}\approx 28.284$.
- Z-scores $\hat{x}_i=(x_i-140)/28.284=\{-1.414,-0.707,0,0.707,1.414\}$. Check: $\frac{1}{5}(2+0.5+0+0.5+2)=1$ ✓ (matches Derivation 1).

Same tissue on a second scanner, $y=2x+50=\{250,290,330,370,410\}$: $\mu_y=2\cdot140+50=330$, $\sigma_y=2\cdot28.284=56.568$. Its z-scores $(250-330)/56.568=-1.414,\dots$ are **identical** to scanner 1. Z-score aligned them.

Now **global-max**. Scanner 1 divides by 180: $\{0.556,0.667,0.778,0.889,1.000\}$. Scanner 2 divides by 410: $\{0.610,0.707,0.805,0.902,1.000\}$ — different numbers for the same tissue. Worse, add one bright artifact voxel of 1000 to scanner 1: global-max now divides by 1000, crushing the real brain to $\{0.10,0.12,0.14,0.16,0.18\}$ — the tissue occupies just 8% of $[0,1]$, wasted by one voxel. Z-score over the mask barely moves.

!!! intuition "Intuition"
    Z-score is a **"shape" description**; global-max is a **"brightest-point" description**. Z-score asks *"how many standard deviations from typical brain is this voxel?"* — a relative answer that survives changing the units. Global-max asks *"what fraction of the single brightest voxel is this?"* — an answer hostage to one pixel. MRI has no absolute units (§6), so the shape description is the meaningful one.

!!! gotcha "Gotcha"
    **Bounded is not the same as standardized.** Global-max's tidy $[0,1]$ output looks safe, but it is a false comfort: one hyperintense voxel (fat, a metal/motion artifact, a bias-field hotspot) sets the denominator and compresses all real tissue into a sliver, and — because there is no subtraction — an additive brightness offset $b$ passes straight through. Separately, note z-score's invariance proof assumes a *linear* map $y=a\,x+b$; it does **not** remove nonlinear MRI bias fields, which need a dedicated step like N4 correction.

### Normalization layers inside the network

The second job lives inside the U-Net (§11). As data flows through, each layer produces **activations** — the intermediate numbers a layer outputs. Normalization layers re-center and re-scale those to keep gradients well-behaved. **Batch normalization** (introduced to reduce "internal covariate shift" and speed training) pools statistics over the whole mini-batch *and* the spatial locations, per channel $c$ [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167):

$$\hat{x}_{n,c,p} = \frac{x_{n,c,p}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\quad \mu_c=\frac{1}{N|P|}\sum_{n=1}^{N}\sum_{p\in P} x_{n,c,p},\quad y_{n,c,p}=\gamma_c\hat{x}_{n,c,p}+\beta_c,$$

where $x_{n,c,p}$ is the activation for sample $n$, channel $c$, spatial location $p$; $P$ is the set of spatial locations, $|P|$ its size; $N$ the batch size; $\gamma_c,\beta_c$ are learned per-channel scale and shift. Crucially $\mu_c$ pools over the batch, so images *share* statistics. **Instance normalization** instead carries both indices $n$ and $c$, pooling over space only [Ulyanov et al. 2016](https://arxiv.org/abs/1607.08022):

$$\hat{x}_{n,c,p} = \frac{x_{n,c,p}-\mu_{n,c}}{\sqrt{\sigma_{n,c}^2+\epsilon}},\quad \mu_{n,c}=\frac{1}{|P|}\sum_{p\in P} x_{n,c,p}.$$

Each sample is normalized on its own — completely independent of batch size $N$ and of how bright the other volumes are.

![Batch norm pools over the batch and space per channel; instance norm pools over space per (sample, channel).](figures/diagrams/s15_bn_vs_in.png)

### Derivation 4 — why small batches hurt batch norm

Batch of $N=2$ images, one channel, spatial $2\times2$. Image A activations $\{1,2,3,4\}$; Image B (a brighter volume) $\{10,20,30,40\}$. Set $\gamma=1,\beta=0$.

*Instance norm, A:* $\mu_A=2.5$, deviations $\{-1.5,-0.5,0.5,1.5\}$, $\sigma_A^2=\frac{5}{4}=1.25$, $\sigma_A\approx1.118$, giving $\{-1.342,-0.447,0.447,1.342\}$. *Image B:* $\mu_B=25$, $\sigma_B^2=125$, $\sigma_B\approx11.18$, giving the **same** $\{-1.342,-0.447,0.447,1.342\}$. Each volume's own brightness was removed; batch size never entered.

*Batch norm* pools all eight values: $\mu=\frac{110}{8}=13.75$; the squared deviations sum to $1517.5$, so $\sigma^2=\frac{1517.5}{8}=189.6875$, $\sigma\approx13.773$. Then A's value $1\to\frac{1-13.75}{13.773}=-0.926$, A's $4\to-0.708$, B's $10\to-0.272$, B's $40\to+1.906$. Image A sits almost entirely below 0 and B above — the shared mean (13.75) is dragged by whichever image is brighter, and it swings from batch to batch. That is the concrete reason nnU-Net **replaces batch norm with instance norm** throughout its U-Nets (paired with leaky ReLU, slope 0.01), since 3D MRI often forces batch sizes as low as 2 [nnU-Net for BraTS (arXiv:2011.00848)](https://arxiv.org/pdf/2011.00848). The U-Net itself is the base encoder-decoder for BraTS-style segmentation [Ronneberger et al. 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28).

!!! example "Example question"
    A FLAIR volume has brain-voxel mean 480 and standard deviation 90. A tumor-edema voxel reads 750; a normal white-matter voxel reads 435. Give the z-scored values and interpret. What would global-max give if the brightest voxel is 1200?

    **Solution:** Using $\hat{x}=(x-\mu)/\sigma$ with $\mu=480,\sigma=90$: edema $=\frac{750-480}{90}=\frac{270}{90}=+3.0$ — three standard deviations above typical brain, i.e. strongly hyperintense (edema signal on FLAIR). White matter $=\frac{435-480}{90}=\frac{-45}{90}=-0.5$ — half a deviation below the mean, near-typical tissue. These are unitless and scanner-comparable. Global-max (divide by 1200): edema $\to0.625$, white matter $\to0.3625$ — squeezed into a narrow band, carrying no "how unusual is this?" meaning, and they would change entirely if the brightest voxel were instead an artifact at 2000 (edema then $0.375$). The z-score answer ($+3.0,-0.5$) is the interpretable, robust one.

!!! example "Example question"
    You train a 3D U-Net on $128^3$ patches and can only fit batch size 2. A labmate insists on batch norm "because it's standard." Explain quantitatively why that is risky and what to use instead.

    **Solution:** Batch norm estimates $\mu_c=\frac{1}{N|P|}\sum_{n,p}x_{n,c,p}$; with $N=2$, only two volumes feed the batch-level average, and they may sit at very different levels (different patients/scanners). From Derivation 4, values $\{1,2,3,4\}$ and $\{10,20,30,40\}$ give a pooled mean 13.75 dominated by the brighter image, pushing the dimmer image's activations all negative — and that estimate swings with every re-pairing, producing noisy gradients plus a train/eval mismatch (inference uses stored running averages). Fix: **instance normalization**, which computes $\mu_{n,c}$ per sample over space only; in Derivation 4 both images mapped to the identical $\{-1.342,-0.447,0.447,1.342\}$ regardless of batch size. This is precisely nnU-Net's choice [nnU-Net for BraTS](https://arxiv.org/pdf/2011.00848).

**Bottom line.** Both normalizations subtract a mean and divide by a standard deviation, so learning the z-score algebra once pays off twice. Input z-score (per modality, over the brain mask) neutralizes scanner-to-scanner intensity shifts *before* training; instance-norm layers keep the *internal* activations stable *during* training without depending on tiny batch sizes. They are different tools at different layers of the stack — do not conflate them. Both feed the class-imbalance and training machinery discussed in §14 and §17.
