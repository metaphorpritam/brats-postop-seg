## 17. Training machinery: making 3D learning actually run {#training}

By now we have the model (the U-Net of §11), the loss (softmax cross-entropy of §12, Dice of §13), and the class-imbalance problem (§14). This section covers the plumbing that turns those ingredients into a training run that is *fast*, *stable*, and *fits in GPU memory*. None of it is specific to brain tumours — it is the standard recipe of modern medical-image segmentation, popularised by the **nnU-Net** framework built on the original U-Net ([Isensee et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)). But it explains almost every training-time decision in this project.

The core difficulty is size. A single BraTS case is about $240\times240\times155$ **voxels** (a voxel is a 3D pixel — one cube of tissue) across four co-registered MRI channels (§6). That is roughly 36 million numbers per channel; feeding the whole thing to a 3D network at once will not fit on a normal GPU. So we chop each volume into small cubes to train, and stitch predictions back together to test. Five pieces of machinery make this work: the **optimizer** (AdamW), the **learning-rate schedule** (cosine annealing), **class-balanced patch sampling**, **sliding-window inference**, and **bf16 mixed precision**.

![Training machinery: sample a class-balanced patch, forward pass in bf16, compute loss, AdamW step with cosine learning rate](figures/diagrams/s17_training_loop.png)

### AdamW: adaptive steps plus honest weight decay

The optimizer decides how to change each weight $\theta$ given its gradient $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ — the slope of the loss with respect to that weight at step $t$. **AdamW** is the near-universal default ([Loshchilov & Hutter, ICLR 2019](https://arxiv.org/abs/1711.05101)). It builds on **Adam** ([Kingma & Ba, ICLR 2015](https://arxiv.org/abs/1412.6980)), which keeps running averages of the gradient and its square:

$$\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \eta_t\left( \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
\end{aligned}$$

Here $m_t$ is the **first moment** (a smoothed average of the gradient) and $v_t$ the **second moment** (a smoothed average of the squared gradient); $\beta_1,\beta_2\in[0,1)$ are their decay rates (defaults $0.9,\,0.999$); $\hat m_t,\hat v_t$ are *bias-corrected* versions (derived below); $\eta_t$ is the learning rate at step $t$; $\epsilon=10^{-8}$ prevents division by zero; and $\lambda$ is the **weight-decay** coefficient, a gentle pull of every weight toward zero that discourages overfitting. The term $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ gives each weight its *own* effective step size: a weight with consistently large gradients (large $\hat v_t$) takes smaller, steadier steps.

The "W" is the last term. Classic Adam + L2 injects the penalty into the gradient itself, $g_t \leftarrow g_t + \lambda\theta_{t-1}$, so it then gets divided by $\sqrt{\hat v_t}$ — meaning busy weights secretly get *less* decay. AdamW instead subtracts $\eta_t\lambda\theta_{t-1}$ **outside** the adaptive scaling, so decay is honest and equal for every weight.

!!! intuition "Intuition"
    AdamW = "per-parameter adaptive step size" + "honest weight decay". The adaptive term hands each weight its own learning rate; decoupling the decay makes the shrink-toward-zero pressure identical for every weight instead of being quietly scaled down for the busy ones.

#### Deriving bias correction: $\hat m_t = m_t/(1-\beta_1^{\,t})$

Why divide by $1-\beta_1^t$ at all? Because $m_0$ starts at $0$, so early averages are dragged toward zero. Unroll the recursion:

$$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i}\, g_i$$

each past gradient $g_i$ weighted by $(1-\beta_1)$ times $\beta_1$ raised to how many steps ago it was. Take expectations (which is linear):

$$\mathbb{E}[m_t] = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i}\,\mathbb{E}[g_i]$$

Assume the gradient distribution is roughly stationary, so $\mathbb{E}[g_i]\approx\mathbb{E}[g]$ pulls out of the sum:

$$\mathbb{E}[m_t] \approx \mathbb{E}[g]\,(1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i}$$

Substitute $j=t-i$ to get a standard finite geometric series and sum it (valid since $0\le\beta_1<1$):

$$\sum_{i=1}^{t}\beta_1^{\,t-i} = \sum_{j=0}^{t-1}\beta_1^{\,j} = \frac{1-\beta_1^{\,t}}{1-\beta_1}$$

Substitute back; the $(1-\beta_1)$ factors cancel:

$$\mathbb{E}[m_t] \approx \mathbb{E}[g]\,(1-\beta_1)\cdot\frac{1-\beta_1^{\,t}}{1-\beta_1} = \mathbb{E}[g]\,(1-\beta_1^{\,t})$$

So the raw $m_t$ is biased toward zero by exactly $(1-\beta_1^{\,t})$. Dividing it out gives $\mathbb{E}[\hat m_t]\approx\mathbb{E}[g]$ — an (approximately) unbiased estimate. The same argument gives $\hat v_t = v_t/(1-\beta_2^t)$. As $t$ grows, $\beta_1^t\to 0$ and the correction fades to 1.

!!! example "Example question"
    With $\beta_1=0.9$ and $m_0=0$, how much does bias correction change the effective first-moment estimate at step $t=1$, and why does it matter?

    **Solution:** $m_1 = \beta_1 m_0 + (1-\beta_1)g_1 = 0.1\,g_1$. The correction divides by $1-\beta_1^1 = 1-0.9 = 0.1$, giving $\hat m_1 = 0.1\,g_1 / 0.1 = g_1$ — exactly the current gradient. *Without* correction the optimizer would step using only $0.1\,g_1$, i.e. 10% of the true magnitude, so the first steps would be $\sim$10x too small. At $t=2$, $1-0.9^2 = 0.19$ (factor $\approx 5.26$x); by $t=50$, $0.9^{50}\approx 5.2\times10^{-3}$ so $1-0.9^{50}\approx 0.995$ and the correction is negligible. Bias correction matters almost entirely in the opening dozens of steps, preventing an artificially timid start.

### Cosine annealing: explore hot, cool slowly

The learning rate $\eta_t$ should not be constant. **Cosine annealing** decays it along a half-cosine from a peak $\eta_{\max}$ to a floor $\eta_{\min}$ ([Loshchilov & Hutter, ICLR 2017](https://arxiv.org/abs/1608.03983)):

$$\eta_t = \eta_{\min} + \tfrac{1}{2}\,(\eta_{\max}-\eta_{\min})\left(1+\cos\!\left(\pi\,\frac{T_{\mathrm{cur}}}{T_{\max}}\right)\right)$$

where $T_{\mathrm{cur}}$ is elapsed epochs and $T_{\max}$ the total cycle length. Work it out for $\eta_{\max}=3\times10^{-4}$, $\eta_{\min}=0$, $T_{\max}=100$:

- $T_{\mathrm{cur}}=0$: $\cos 0 = 1 \Rightarrow \eta = \tfrac12(3\times10^{-4})(2) = 3\times10^{-4}$ — starts at the peak.
- $T_{\mathrm{cur}}=25$: $\cos(45^\circ)=0.7071 \Rightarrow \eta = \tfrac12(3\times10^{-4})(1.7071) = 2.56\times10^{-4}$ — only $\sim$15% down after a quarter; it lingers near the top.
- $T_{\mathrm{cur}}=50$: $\cos(90^\circ)=0 \Rightarrow \eta = \tfrac12(3\times10^{-4})(1) = 1.5\times10^{-4}$ — the exact midpoint.
- $T_{\mathrm{cur}}=75$: $\cos(135^\circ)=-0.7071 \Rightarrow \eta = \tfrac12(3\times10^{-4})(0.2929) = 4.39\times10^{-5}$ — steep decay now.
- $T_{\mathrm{cur}}=100$: $\cos\pi=-1 \Rightarrow \eta = 0 = \eta_{\min}$ — ends exactly at the floor.

The shape is "slow at the ends, fast in the middle": a high rate early lets the optimizer roam and find a good basin; the long cool-down lets it settle into a flat minimum. The original SGDR paper adds *warm restarts* — after each cycle the rate jumps back to $\eta_{\max}$ and the cycle length grows by $T_{\mathrm{mult}}$ — but many BraTS pipelines use a single cycle (no restarts).

!!! gotcha "Gotcha"
    The cosine argument $T_{\mathrm{cur}}/T_{\max}$ must map $[0,1]\to[0,\pi]$ in consistent units. A common bug is feeding *iteration* counts against an *epoch*-based $T_{\max}$ (or vice versa): the schedule then either ends far from $\eta_{\min}$ or anneals far too fast. And if you did not intend restarts, use a single-cycle scheduler — with warm restarts the rate suddenly *jumps back up*, which will look like a training bug if unexpected.

### Class-balanced patch sampling

Tumour tissue occupies only a tiny fraction of the brain, so a purely random crop is almost always healthy tissue or empty background — train on those and the network learns to output "no tumour", scores high on voxel accuracy, and is clinically useless. The nnU-Net fix is **foreground oversampling** ([Isensee et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)):

$$P(\text{patch centered on foreground}) = p_{fg},\qquad P(\text{class}=c \mid \text{foreground}) = \tfrac{1}{C_{fg}}$$

With probability $p_{fg}$ (nnU-Net uses $1/3$) the patch is forced to contain foreground: a foreground class $c$ is picked uniformly among the $C_{fg}$ tumour classes (the 5 BraTS labels of §7, minus background), then a random voxel of that class becomes the patch centre. With probability $1-p_{fg}$ the centre is any random voxel. This **decouples the training class distribution from the raw voxel frequencies** — the network sees tumour in most steps and can actually learn boundaries. The U-Net's skip connections, which carry high-resolution detail across the bottleneck, are what let it draw those boundaries precisely ([Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)).

!!! gotcha "Gotcha"
    Oversampling changes the class balance the network *trains* on, so its output is biased toward tumour unless you also use an imbalance-aware loss (Dice or Dice+CE, §13–§14). And plain voxel accuracy will still look deceptively high — evaluate with Dice per tumour sub-region as BraTS does (§16).

### Sliding-window inference

At test time we must predict the *whole* brain, but the network was only ever trained on small cubes. **Sliding-window inference** tiles the volume with overlapping patches, runs the network on each, and blends the overlaps ([MONAI docs](https://docs.monai.io/en/stable/inferers.html)):

$$N_d = \left\lceil \frac{D_d - P_d}{S_d} \right\rceil + 1,\qquad S_d = P_d\,(1-r),\qquad \hat y(x) = \frac{\sum_k w_k(x)\,\hat y_k(x)}{\sum_k w_k(x)}$$

For axis $d$: $D_d$ is the image size, $P_d$ the patch size, $r$ the overlap fraction, $S_d$ the **stride**, and $N_d$ the number of windows. For blending, $\hat y_k$ is patch $k$'s prediction and $w_k(x)$ its weight at voxel $x$ — usually a Gaussian centred on the patch so edge voxels (predicted with the least surrounding context) count least.

![Sliding-window inference: tile the whole volume into overlapping patches, run the network on each, blend overlaps with Gaussian weights into a full prediction](figures/diagrams/s17_sliding_window.png)

Worked example on $D=(240,240,155)$ with $P=(128,128,128)$, $r=0.5$: stride $S = 128(1-0.5)=64$. Then $N_x = \lceil (240-128)/64\rceil + 1 = \lceil 1.75\rceil+1 = 3$, likewise $N_y=3$, and $N_z = \lceil(155-128)/64\rceil+1 = \lceil 0.422\rceil+1 = 2$. Total $= 3\times3\times2 = 18$ forward passes for one case.

!!! example "Example question"
    You trained with patch size $96^3$ and want sliding-window inference at overlap $0.5$ on a padded $192\times192\times144$ volume. How many forward passes per case, and what does raising overlap to $0.75$ do?

    **Solution:** At $r=0.5$, $S=96(1-0.5)=48$. $N_x=\lceil(192-96)/48\rceil+1=\lceil2\rceil+1=3$; $N_y=3$; $N_z=\lceil(144-96)/48\rceil+1=\lceil1\rceil+1=2$. Total $=3\times3\times2=18$. At $r=0.75$, $S=96(0.25)=24$: $N_x=\lceil(192-96)/24\rceil+1=\lceil4\rceil+1=5$; $N_y=5$; $N_z=\lceil(144-96)/24\rceil+1=\lceil2\rceil+1=3$. Total $=5\times5\times3=75$ — about **4.2x** more compute for a modest boundary-smoothness gain. Overlap is a cost/quality knob, and Gaussian weighting keeps the extra overlaps from giving equal vote to unreliable patch edges.

### bf16 mixed precision — and why it needs no GradScaler

The final piece is the numerical format. **bf16** (bfloat16) uses 1 sign + 8 exponent + 7 mantissa bits, giving it the *same exponent range as FP32* ($\sim$1.2e-38 to $\sim$3.4e38) but coarser precision; **fp16** uses 1 + 5 + 10 bits — fine precision but a narrow range (smallest normal $\sim$6.1e-5) ([PyTorch AMP blog](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)). This matters because of tiny gradients:

$$\begin{aligned}
\text{fp16 smallest normal} &= 2^{-14} \approx 6.10\times10^{-5} \\
\text{bf16 smallest normal} &= 2^{-126} \approx 1.18\times10^{-38}
\end{aligned}$$

Take a realistic small gradient $g = 1\times10^{-7}$. In fp16, $10^{-7} < 6.10\times10^{-5}$, so it underflows into subnormals or rounds to zero — that weight stops learning. fp16's fix is **loss scaling**: multiply the loss by a large $S$ (e.g. $2^{16}$) before backprop, then divide it back out:

$$g_t^{scaled} = \frac{1}{S}\,\nabla_\theta\big(S\cdot \mathcal{L}\big) = \nabla_\theta \mathcal{L}$$

Numerically, $S\cdot g = 2^{16}\times10^{-7} = 65536\times10^{-7}\approx 6.55\times10^{-3}$, comfortably inside fp16's range; PyTorch's `GradScaler` does this and also backs off dynamically on inf/NaN. In bf16, however, $10^{-7} > 1.18\times10^{-38}$ is far above the underflow limit, so **$S=1$ is safe and GradScaler is unnecessary** ([PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)). The only cost is bf16's coarser 7-bit mantissa, which fp32 master weights and optimizer moments compensate for.

!!! intuition "Intuition"
    bf16 vs fp16 is "range vs precision". bf16 spends its bits on exponent (range) rather than mantissa (precision). For gradients, range is what matters — you must not lose tiny values to underflow — so bf16 is *both* faster *and* simpler, dropping the entire loss-scaling apparatus.

!!! gotcha "Gotcha"
    bf16 is *lower* precision than fp16, not higher — 7 mantissa bits vs 10 — so "newer = more accurate" is wrong; it wins purely on range. Do not add a GradScaler "just in case" under bf16: it solves a non-problem, adds overhead, and its inf/NaN backoff can mask real bugs. Conversely, forgetting GradScaler under fp16 silently kills learning. Match the tool to the dtype.

!!! example "Example question"
    A bf16 gradient element has true value $3\times10^{-8}$. A colleague wants to add GradScaler with $S=2^{15}$, fearing it will "vanish like in fp16". Necessary?

    **Solution:** No. bf16's smallest normal is $2^{-126}\approx1.18\times10^{-38}$, and $3\times10^{-8}$ is $\sim$30 orders of magnitude larger — represented as a normal bf16 value, only its mantissa rounded. In fp16, $3\times10^{-8}\ll 6.10\times10^{-5}$ *would* underflow, and there GradScaler helps ($S\cdot g = 32768\times3\times10^{-8}\approx 9.8\times10^{-4}$, back in range). Under bf16 it solves nothing. The correct bf16 recipe: autocast the forward/backward to bf16, keep fp32 master weights and optimizer state, and use *no* scaler.

Together these five choices explain the shape of the training run: the loss descends the way AdamW and the cosine schedule dictate, the network sees enough tumour thanks to class-balanced cropping despite only ever viewing small cubes, sliding-window inference reassembles whole-brain predictions, and bf16 makes the arithmetic fast without the fp16 loss-scaling boilerplate. The controlled experiments comparing these settings are laid out in §18, and the project's concrete configuration and its defects in §19–§20.
