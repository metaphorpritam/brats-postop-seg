## 18. Designing a trustworthy experiment: the controlled A/B {#ab-design}

Suppose someone tells you: "I changed the loss function and the segmentation got better." Should you believe them? Only if the experiment was built so that the changed loss was the **one and only thing** that could have moved the score. Everything in this section is in service of that single guarantee. We are no longer asking *how* a network learns (§10–§17) but *how to prove* that a design choice actually helped.

### A/B and ablation: two directions of one idea

An **A/B experiment** runs two complete pipelines — call them **arm A** (the *control*, the current setup) and **arm B** (the *treatment*, the setup with one deliberate change) — and compares them. An **ablation study** does the mirror image: it starts from a finished system and *removes* one component at a time to measure how much that piece was contributing. The word "ablation" is borrowed from surgery and neuroscience, where you remove a piece of tissue and watch how behaviour changes; in AI you remove a feature, layer, or module and measure the drop in performance, which is how a component earns its place in the model [Wikipedia: Ablation](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)) [Baeldung](https://www.baeldung.com/cs/ml-ablation-study). Both obey the same rule: **change one thing, hold the rest fixed.**

### The enemy: confounds

A **confound** is any variable that changes *alongside* your treatment and could explain the result instead of it. If arm B swaps in a new loss *and* also makes the network deeper *and* also trains twice as long, a higher Dice score (our overlap metric from §13) is unattributable — you cannot say which change earned it. Confound control means **freezing** everything you are not testing: the data splits (§19), the optimizer and learning-rate schedule (§17), the augmentations, the number of training steps, and above all the **architecture**. Holding the architecture constant is the highest-leverage freeze in a segmentation study, because architecture is such a large lever that a small unintended change to it can swamp the effect you are chasing. The U-Net (§11) is an ideal thing to freeze: its encoder–decoder-with-skip-connections design is stable and well understood, so you can hold it fixed while varying exactly one factor [Ronneberger, Fischer & Brox, MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28).

![Wrong vs right: many factors change and the effect is unattributable, versus one factor changes and delta is isolated](figures/diagrams/s18_confound_control.png)

!!! intuition "Intuition"
    Treat an A/B experiment as a courtroom. The treatment is the accused; every uncontrolled variable is "reasonable doubt." Freezing the architecture, data, and optimizer removes the alternative suspects, so when the score changes you can convict the one factor you varied. An ablation is the same trial run backwards: remove a component and see whether the system misses it.

### The wrinkle: training is random

Deep-learning training is **stochastic** — it involves genuine randomness. Weights start at random values, the data is shuffled in a random order, dropout drops random units, and some GPU operations are nondeterministic. Run the *same code twice* with two different **random seeds** (a seed is the integer that fixes the random-number generator) and you get two different scores. So a single before/after number can be pure luck: Henderson et al. showed that changing *only* the seed produced non-overlapping learning curves — different seeds looked like different algorithms [Henderson et al., arXiv:1709.06560](https://arxiv.org/abs/1709.06560). The fix is to run each arm over several seeds and report a **distribution** — a mean with error bars — and to state which sources of randomness those bars capture [Reproducibility in ML for Health, arXiv:1907.01463](https://arxiv.org/pdf/1907.01463).

### A model of a single run

To make "confounds cancel" precise, write any run's score as a sum of named parts:

$$Y_{c}(s) = \mu + \delta\,\mathbb{1}[c = B] + \beta_{s} + \varepsilon_{c,s}$$

Here $Y_c(s)$ is the measured metric (say Dice) for condition $c \in \{A, B\}$ on seed $s$; $\mu$ is arm A's baseline mean; $\delta$ is the **true treatment effect** (the causal quantity we want); $\mathbb{1}[c=B]$ is an indicator that is $1$ for the treatment arm and $0$ for the control; $\beta_s$ is the **seed effect** — the shared "luck of the draw" a given seed imposes on *both* arms; and $\varepsilon_{c,s}$ is leftover noise specific to condition $c$ on seed $s$, with mean zero. This additive model is a teaching idealization (it assumes no seed-treatment interaction), not a claim from a specific paper.

### Derivation: why the same-seed delta cancels confounds

$$\begin{aligned}
Y_{A}(s) &= \mu + \beta_{s} + \varepsilon_{A,s} \\
Y_{B}(s) &= \mu + \delta + \beta_{s} + \varepsilon_{B,s} \\
D(s) &= Y_{B}(s) - Y_{A}(s) = (\mu + \delta + \beta_{s} + \varepsilon_{B,s}) - (\mu + \beta_{s} + \varepsilon_{A,s}) \\
D(s) &= \delta + (\varepsilon_{B,s} - \varepsilon_{A,s})
\end{aligned}$$

Line 1 is the control on seed $s$: baseline $\mu$, the seed's shared luck $\beta_s$, and arm-A noise — no $\delta$ because the indicator is $0$. Line 2 is the treatment on the **same** seed: the same $\mu$ and the same $\beta_s$ (identical because we froze the architecture, data, and optimizer *and* reused the seed), plus $\delta$. Line 3 subtracts them seed-by-seed. Line 4 cancels: $\mu-\mu=0$ and $\beta_s-\beta_s=0$. Every shared nuisance factor has vanished, leaving $\delta$ plus a tiny noise difference. Taking expectations over seeds, $\mathbb{E}[D(s)] = \delta + \mathbb{E}[\varepsilon_{B,s}] - \mathbb{E}[\varepsilon_{A,s}] = \delta$, since the residuals have mean zero — the paired delta is an **unbiased** estimator of the treatment effect.

The failure mode makes the point sharp. If a confound $\gamma$ (say, a deeper network) rode along with the treatment, then $Y_B(s)=\mu+\delta+\gamma+\beta_s+\varepsilon_{B,s}$ and the subtraction gives $D(s)=\delta+\gamma+\dots$ — the confound does *not* cancel and gets absorbed into the delta. You would measure $\delta+\gamma$ and wrongly credit it all to the treatment.

![Same seed feeds both arms; their per-seed difference cancels the seed effect; the mean difference with its CI is the deliverable](figures/diagrams/s18_delta_pipeline.png)

### From one run to a trustworthy number: the standard error

Because one run is a random draw, we average over $n$ seeds and ask how much the *average itself* wobbles. Let $\bar{x}=\frac{1}{n}\sum_{i=1}^{n}X_i$ be the mean of $n$ run scores, each with true variance $\sigma^2$. Then

$$\begin{aligned}
\operatorname{Var}(\bar{x}) &= \operatorname{Var}\!\left(\tfrac{1}{n}\textstyle\sum_i X_i\right) = \tfrac{1}{n^{2}}\operatorname{Var}\!\left(\textstyle\sum_i X_i\right) \\
&= \tfrac{1}{n^{2}}\cdot n\sigma^{2} = \tfrac{\sigma^{2}}{n}
\end{aligned}$$

The first step pulls the constant $1/n$ out using $\operatorname{Var}(aZ)=a^2\operatorname{Var}(Z)$. The second uses that for **independent** runs the variance of a sum is the sum of variances, so $\sum_i \operatorname{Var}(X_i)=n\sigma^2$. Taking the square root gives the **standard error** $\mathrm{SE}(\bar{x})=\sigma/\sqrt{n}$: the typical wobble of the mean, which shrinks like $1/\sqrt{n}$. In practice $\sigma$ is unknown, so we use the **sample standard deviation** $s$, where $s^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$ (the $n-1$ is **Bessel's correction**), giving $\widehat{\mathrm{SE}}=s/\sqrt{n}$.

A **confidence interval (CI)** — the range of mean values statistically compatible with the data — is then

$$\bar{x} \pm t^{*}_{n-1}\,\frac{s}{\sqrt{n}}$$

where $t^{*}_{n-1}$ is the critical value of the Student-t distribution with $n-1$ **degrees of freedom**. We use $t$, not the normal $z=1.96$, because $s$ was estimated from the same small sample; the wider $t$ honestly pays for that extra uncertainty.

### Worked example: a single arm's 95% CI

Five seed runs give Dice $\{0.80, 0.82, 0.79, 0.83, 0.81\}$, so $n=5$.

$$\begin{aligned}
\bar{x} &= \tfrac{0.80+0.82+0.79+0.83+0.81}{5} = \tfrac{4.05}{5} = 0.81 \\
\textstyle\sum(x_i-\bar x)^2 &= (-0.01)^2+(0.01)^2+(-0.02)^2+(0.02)^2+0^2 = 0.0010 \\
s^{2} &= \tfrac{0.0010}{n-1} = \tfrac{0.0010}{4} = 0.00025, \quad s = 0.01581 \\
\mathrm{SE} &= \tfrac{0.01581}{\sqrt{5}} = \tfrac{0.01581}{2.2361} = 0.00707
\end{aligned}$$

With $t^{*}_{4,\,0.95}=2.776$, the interval is $0.81 \pm 2.776\times0.00707 = 0.81 \pm 0.0196 = [0.790,\ 0.830]$. Reporting "Dice $0.81$, 95% CI $[0.790, 0.830]$" tells a reader how much of the number is signal versus seed luck (§13, where this overlap metric is defined).

### The delta is the deliverable — and pairing makes it precise

The scientifically meaningful output of an A/B run is not arm B's raw score (contaminated by the dataset, the metric definition, the epoch count, the hardware — all frozen choices that inflate or deflate *both* arms together) but the **difference** between the arms and its uncertainty. That is what we mean by **"the delta is the deliverable."** And pairing on shared seeds tightens the delta dramatically. The variance of a difference of two means is

$$\operatorname{Var}(\bar{A}-\bar{B}) = \operatorname{Var}(\bar{A}) + \operatorname{Var}(\bar{B}) - 2\,\operatorname{Cov}(\bar{A},\bar{B})$$

When both arms run on the *same* seeds they rise and fall together, so $\operatorname{Cov}(\bar A,\bar B)>0$ and the $-2\operatorname{Cov}$ term *shrinks* the delta's variance. Independent (unpaired) arms have $\operatorname{Cov}=0$ and forfeit that benefit.

!!! intuition "Intuition"
    Subtracting two arms on the same seed is like weighing two people while both stand on the same bobbing boat: the boat's motion hits both equally and cancels, so you measure the true weight *difference*, not the waves. The seed's luck $\beta_s$ is the boat.

**Numeric demonstration.** Five shared seeds give arm A $\{0.80,0.82,0.79,0.83,0.81\}$ (so $\bar A=0.81$, $s_A^2=0.00025$ from above) and arm B $\{0.82,0.85,0.80,0.86,0.83\}$, giving $\bar B=0.832$ and $s_B^2=0.00057$. If we wrongly treated the arms as independent,

$$\mathrm{SE}_{\text{unpaired}} = \sqrt{\tfrac{s_A^{2}}{n}+\tfrac{s_B^{2}}{n}} = \sqrt{\tfrac{0.00025}{5}+\tfrac{0.00057}{5}} = \sqrt{0.000164} \approx 0.0128.$$

Now go paired. The per-seed deltas $D(s)=Y_B-Y_A$ are $\{0.02,0.03,0.01,0.03,0.02\}$, with $\bar D = 0.022$. Their deviations from $0.022$ are $\{-0.002, 0.008, -0.012, 0.008, -0.002\}$; the squares sum to $0.00028$, so

$$s_D^{2} = \tfrac{0.00028}{4} = 0.00007,\quad s_D \approx 0.00837,\quad \mathrm{SE}_{\text{paired}} = \tfrac{0.00837}{\sqrt{5}} \approx 0.00374.$$

The paired SE ($0.00374$) is about **3.4× smaller** than the unpaired one ($0.0128$) — from the *same numbers*, purely by exploiting shared seeds. The covariance identity confirms why: $\operatorname{Cov}(A,B)=\frac{0.0015}{4}=0.000375$ (correlation $\rho\approx0.99$), and $s_A^2+s_B^2-2\operatorname{Cov}=0.00025+0.00057-2(0.000375)=0.00007$, matching the paired variance exactly. Finally, the deliverable:

$$95\%\ \text{CI for }\delta:\quad 0.022 \pm 2.776\times0.00374 = 0.022 \pm 0.0104 = [0.0116,\ 0.0324].$$

This excludes $0$, so arm B is significantly better. The *unpaired* interval, $0.022 \pm 2.776\times0.0128 = [-0.0135,\ 0.0575]$, **includes** $0$ and would have called the very same result inconclusive. Same experiment, opposite verdict — because the delta, computed with pairing, is the deliverable.

!!! gotcha "Gotcha"
    Several traps quietly destroy an A/B: (1) **Changing two things at once** — a new loss *and* a deeper net makes the effect $\delta+\gamma$, unattributable. (2) **Confusing SD and SE** — $s$ (spread of runs) is not $s/\sqrt{n}$ (wobble of the mean); a caption must say which, and what randomness it captures. (3) **Using $z=1.96$ for small $n$** — with 5 seeds the right multiplier is $2.776$, so a $z$-interval is about 29% narrower than the correct $t$-interval (equivalently, $t$ is ~42% wider) — manufacturing false significance. (4) **Unpaired analysis of a paired design** — comparing same-seed arms with an independent-samples test throws away the $-2\operatorname{Cov}$ term, turning a real win into a null. (5) **Seed hacking** — reporting only the best seed for B fabricates a delta; fix seeds in advance and apply them identically to both arms.

!!! example "Example question"
    A team compares a Dice loss (arm A) against a Dice+boundary loss (arm B) on the **same** 4 seeds, U-Net and splits frozen. Per-seed deltas $B-A$ are $0.015, 0.020, 0.010, 0.019$. Find the mean delta, its SE, a 95% CI, and the conclusion.

    **Solution.** Mean: $\bar D = (0.015+0.020+0.010+0.019)/4 = 0.064/4 = 0.016$. Deviations from $0.016$: $-0.001, 0.004, -0.006, 0.003$; squares $0.000001, 0.000016, 0.000036, 0.000009$ sum to $0.000062$. Sample variance $s_D^2 = 0.000062/(4-1) = 0.0000207$, so $s_D = 0.004549$. $\mathrm{SE} = s_D/\sqrt{4} = 0.004549/2 = 0.002274$. With $t^{*}_{3,\,0.95}=3.182$, margin $= 3.182\times0.002274 = 0.00724$. CI $= 0.016 \pm 0.0072 = [0.0088,\ 0.0232]$. The interval excludes $0$, so the boundary term gives a statistically significant improvement of about $+0.016$ Dice. Because architecture and splits were frozen and seeds shared, this delta is a clean estimate of the loss-function effect alone.

!!! example "Example question"
    Two seeds give arm A $\{0.78, 0.86\}$ (mean $0.82$) and arm B $\{0.80, 0.88\}$ (mean $0.84$). A colleague reports "B is better by $0.02$" with no uncertainty. Why is this not yet credible, and what is the minimal fix?

    **Solution.** Within arm A the two runs differ by $0.08$ — four times the claimed $0.02$ gain — so seed-to-seed noise dwarfs the effect. With $n=2$ the SE is huge and the multiplier enormous ($t^{*}_{1,\,0.95}=12.71$), so any honest CI on $0.02$ comfortably includes $0$: the difference is indistinguishable from luck. Minimal fixes: (1) run **both arms on the same seeds** and analyze the paired deltas — here they are $0.80-0.78=0.02$ and $0.88-0.86=0.02$, a rock-steady $+0.02$ on every seed, because the shared swing ($0.78$-vs-$0.86$) is the seed effect $\beta_s$ and it cancels; (2) add more seeds so both $\mathrm{SE}\propto 1/\sqrt{n}$ and the $t$-multiplier shrink, then report $\bar D \pm$ CI. The claim earns credibility only once the *delta's* uncertainty, not the raw score, is shown to exclude $0$.

This machinery is exactly what §19–§21 rely on: the project's design freezes the U-Net and data so that each defect fix (§20) can be read as a clean, seed-paired delta with error bars, and interpreted honestly in §21.
