## 12. From logits to labels: softmax and cross-entropy {#softmax-ce}

The U-Net of §11 ends with a stack of numbers at every voxel — one number per class. For the five BraTS labels of §7 (background, enhancing tumour, non-enhancing tumour core, surrounding tissue, resection cavity) that is five numbers per voxel. These raw output numbers are called **logits**: real-valued scores, unbounded, possibly negative, that do *not* form a probability distribution. A logit of $4.0$ does not mean "40% sure"; it means nothing on its own. This section shows the two operations that turn logits into a trainable label decision — **softmax** and **cross-entropy** — and derives, with no algebra skipped, the small miracle that makes the pair practical: the gradient collapses to "prediction minus target."

![Three-panel flow: raw logits go through softmax to probabilities, then cross-entropy against the one-hot label gives one loss number; the backward arrow is labelled gradient equals p minus y](figures/diagrams/s12_pipeline.png)

### Softmax: logits to a probability distribution

**Softmax** squashes a vector of $K$ logits into $K$ positive numbers that sum to exactly $1$ — a probability distribution over the classes [Goodfellow et al.](https://www.deeplearningbook.org/contents/mlp.html):

$$p_i = \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}$$

Here $z = (z_1,\dots,z_K)$ is the vector of $K$ raw logits for one voxel, and $z_i$ is the logit for class $i$. The letter $e$ is Euler's number ($\approx 2.718$); raising it to any power gives a positive result, which is why every $p_i$ comes out positive. The denominator $S = \sum_{k=1}^{K} e^{z_k}$ is the sum of all the exponentiated logits, and dividing by it forces the outputs to sum to $1$. So $p_i$ is the model's predicted probability of class $i$, and $K$ is the number of classes.

!!! intuition "Intuition"
    Softmax is a "soft argmax." The plain `argmax` would just pick the single largest logit and throw the rest away — a hard, non-differentiable jump. Softmax instead puts the *most* probability on the largest logit but leaves smooth, non-zero mass on the others, so a gradient (§17) can still flow back and nudge every logit. The exponential exaggerates gaps: a logit that is a little larger becomes a lot more probable.

### Cross-entropy: measuring the error

We know the correct class for each training voxel. Encode it as a **one-hot** label $y = (y_1,\dots,y_K)$: $y_c = 1$ for the true class $c$ and $y_i = 0$ everywhere else. **Categorical cross-entropy** measures how far the predicted distribution $p$ is from this target [Bishop, PRML §4.3.4]:

$$L = -\sum_{i=1}^{K} y_i \log(p_i)$$

$\log$ is the natural logarithm. Because every $y_i$ is zero except the true class, only one term survives, and the loss reduces to

$$L = -\log(p_c),$$

the negative log-probability the model assigned to the correct class $c$. This is exactly the negative log-likelihood of the right answer. It is always $\ge 0$, it falls to $0$ as $p_c \to 1$ (perfect confidence in the truth), and it rises toward $+\infty$ as $p_c \to 0$ (confidently wrong). This softmax-plus-cross-entropy pairing is precisely the training objective of the original U-Net [Ronneberger et al.](https://arxiv.org/abs/1505.04597), whose "energy function" is a pixel-wise softmax combined with cross-entropy,

$$E = \sum_{x \in \Omega} w(x)\, \log\big(p_{\ell(x)}(x)\big),$$

where $x$ indexes voxels over the image domain $\Omega$, $\ell(x)$ is the true label at $x$, $p_{\ell(x)}(x)$ is the softmax probability given to that correct class, and $w(x)$ is an optional per-voxel weight (used to upweight boundaries or rare classes). Minimising cross-entropy is the same as maximising this weighted log-likelihood.

### The headline derivation: $\partial L/\partial z_j = p_j - y_j$

Training needs the derivative of the loss with respect to the raw logits $z$, because that is the signal backpropagated into the network (§17). We build it in four steps.

**Step 1 — softmax derivative, diagonal case ($i=j$).** Write $p_i = e^{z_i}/S$ with $S = \sum_k e^{z_k}$. Since $S$ contains $z_j$ in exactly one term, $\partial S/\partial z_j = e^{z_j}$. For $i=j$, apply the quotient rule $(f/g)' = (f'g - fg')/g^2$ with $f=e^{z_i}$, $g=S$:

$$\begin{aligned}
\frac{\partial p_i}{\partial z_i} &= \frac{(\partial_{z_i} e^{z_i})\,S - e^{z_i}\,(\partial_{z_i} S)}{S^2} \\
&= \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} \\
&= \frac{e^{z_i}}{S} - \frac{e^{z_i}}{S}\cdot\frac{e^{z_i}}{S} \\
&= p_i - p_i^2 = p_i(1 - p_i).
\end{aligned}$$

The first line is the quotient rule; the second substitutes $\partial_{z_i} e^{z_i} = e^{z_i}$ and $\partial_{z_i} S = e^{z_i}$; the third splits the fraction and recognises $e^{z_i}/S = p_i$ in each piece; the last factors out $p_i$.

**Step 2 — softmax derivative, off-diagonal case ($i \ne j$).** Now the numerator $e^{z_i}$ does not contain $z_j$, so $\partial_{z_j} e^{z_i} = 0$:

$$\frac{\partial p_i}{\partial z_j} = \frac{0\cdot S - e^{z_i} e^{z_j}}{S^2} = -\frac{e^{z_i}}{S}\cdot\frac{e^{z_j}}{S} = -p_i p_j.$$

Both cases fold into one line using the **Kronecker delta** $\delta_{ij}$ (equal to $1$ when $i=j$, else $0$) [Kurbiel](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1):

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j).$$

**Step 3 — cross-entropy derivative w.r.t. the probabilities.** From $L = -\sum_i y_i \log(p_i)$ and $\tfrac{d}{dp_i}\log p_i = 1/p_i$:

$$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}.$$

**Step 4 — chain rule and collapse.** The logit $z_j$ influences $L$ through *every* probability (softmax couples all outputs), so we sum over $i$:

$$\begin{aligned}
\frac{\partial L}{\partial z_j} &= \sum_{i=1}^{K} \frac{\partial L}{\partial p_i}\,\frac{\partial p_i}{\partial z_j} \\
&= \sum_{i=1}^{K} \left(-\frac{y_i}{p_i}\right)\big(p_i(\delta_{ij} - p_j)\big) \\
&= \sum_{i=1}^{K} -y_i(\delta_{ij} - p_j) \\
&= -\sum_{i=1}^{K} y_i\,\delta_{ij} + p_j\sum_{i=1}^{K} y_i \\
&= -y_j + p_j\cdot 1 = p_j - y_j.
\end{aligned}$$

Line by line: the chain rule sums each output's contribution; we substitute Steps 2 and 3; the $p_i$ from the Jacobian *cancels* the $1/p_i$ from the loss derivative (this cancellation is the whole point); we distribute and split into two sums; the first sum $\sum_i y_i\delta_{ij}$ picks out only the $i=j$ term ($y_j$) while $p_j$ factors out of the second; and finally $\sum_i y_i = 1$ because $y$ is one-hot. The result, in vector form $\nabla_z L = p - y$, is the whole reason frameworks fuse the two operations [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/).

!!! intuition "Intuition"
    The gradient $p - y$ is a "prediction error" in probability space. If the model already nails the answer with certainty ($p = y$) the gradient is zero and nothing changes. The more confidently wrong it is, the bigger the push — yet each component stays strictly between $-1$ and $+1$, which keeps training stable. And the components always sum to zero ($\sum_j (p_j - y_j) = 1 - 1 = 0$): softmax ignores the overall level of the logits and cares only about their differences, so there is nothing to learn in the "raise everything equally" direction.

### Worked numeric example

Take $K=3$, logits $z = (2.0,\ 1.0,\ 0.1)$, and one-hot label $y = (1,\ 0,\ 0)$ (true class $0$). Exponentiate: $e^{2.0}=7.389$, $e^{1.0}=2.718$, $e^{0.1}=1.105$, so $S = 7.389+2.718+1.105 = 11.212$. Divide each by $S$:

$$p = \left(\tfrac{7.389}{11.212},\ \tfrac{2.718}{11.212},\ \tfrac{1.105}{11.212}\right) = (0.659,\ 0.242,\ 0.099),$$

which sums to $1.000$. The loss is $L = -\log(p_0) = -\log(0.659) = 0.417$. And the gradient, straight from the headline formula:

$$\nabla_z L = p - y = (0.659-1,\ 0.242-0,\ 0.099-0) = (-0.341,\ 0.242,\ 0.099).$$

The true class gets a *negative* gradient — descent pushes its logit up — while the two wrong classes get positive gradients that push their logits down. The three components sum to $0.000$, exactly as required.

!!! gotcha "Gotcha"
    Do **not** apply softmax yourself and then feed the probabilities into a cross-entropy loss. PyTorch's `nn.CrossEntropyLoss` expects **raw logits** and applies log-softmax internally [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html); passing already-softmaxed values double-applies softmax and quietly degrades training. There is also a numerical reason the fused op is safer: computing $\log(\operatorname{softmax}(z))$ naively can overflow ($e^{z}$ for large $z$) or hit $\log(0) = -\infty$. The **log-sum-exp trick** — subtract $\max(z)$ before exponentiating (softmax is unchanged by a constant shift) and combine the two steps as $z_i - \log\sum_k e^{z_k}$ — is what the fused loss does for you. If you truly need the two-step path, apply `log_softmax` yourself and use `nn.NLLLoss`.

!!! example "Example question"
    For logits $z = (1, 3, 0)$ with true class index $1$, compute the softmax probabilities, the cross-entropy loss, and the gradient $\nabla_z L$. Verify the gradient sums to zero.

    **Solution:** Exponentiate: $e^1 = 2.718$, $e^3 = 20.086$, $e^0 = 1.000$, giving $S = 23.804$. Softmax $p = (2.718/23.804,\ 20.086/23.804,\ 1.000/23.804) = (0.1142,\ 0.8438,\ 0.0420)$, which sums to $1.000$. With $y = (0,1,0)$, the loss is $L = -\log(p_1) = -\log(0.8438) = 0.1699$. Gradient $\nabla_z L = p - y = (0.1142,\ 0.8438-1,\ 0.0420) = (0.1142,\ -0.1562,\ 0.0420)$. Check: $0.1142 - 0.1562 + 0.0420 = 0.0000$. The true class (index $1$) has a negative gradient (its logit is raised); the two wrong classes are pushed down; the sum is zero as required.

!!! example "Example question"
    A binary case: logits $z = (z_0, z_1)$, true class $1$. Show that softmax + categorical cross-entropy reduces to the sigmoid + binary cross-entropy form.

    **Solution:** Softmax for class $1$ is $p_1 = e^{z_1}/(e^{z_0}+e^{z_1})$. Divide top and bottom by $e^{z_1}$: $p_1 = 1/(1 + e^{z_0 - z_1}) = 1/(1 + e^{-(z_1 - z_0)}) = \sigma(\Delta)$, the logistic sigmoid of the logit *difference* $\Delta = z_1 - z_0$. So a two-class softmax depends only on one effective logit, exactly like a single sigmoid output. The loss $L = -\log(p_1) = -\log(\sigma(\Delta))$ is precisely binary cross-entropy for a positive example, and its gradient $dL/d\Delta = \sigma(\Delta) - 1 = p_1 - y_1$ matches the general $p - y$ result. This is why a binary classifier is usually built with one sigmoid rather than a two-way softmax — they are mathematically identical but the sigmoid uses half the final-layer parameters.

### Why this matters for BraTS

In segmentation the per-voxel cross-entropy is averaged (or class-weighted) over every voxel in the scan. But plain cross-entropy is dominated by the huge background class and can all but ignore the tiny tumour regions — the class-imbalance problem of §14. Practitioners answer this with the weight map $w(x)$ above, or by combining cross-entropy with the **Dice loss** of §13, as in the widely used nnU-Net recipe [Isensee et al.](https://www.nature.com/articles/s41592-020-01008-z). The clean $p - y$ gradient still governs the cross-entropy part; the overall training signal is just the weighted sum of the two.

!!! gotcha "Gotcha"
    The tidy $\nabla_z L = p - y$ holds only when $y$ is a valid distribution summing to $1$ (a one-hot label, or a soft label from label smoothing). If $y$ does not sum to $1$, the step $p_j\sum_i y_i \to p_j$ fails and the formula changes. Separately, "categorical" cross-entropy (softmax over $K$ mutually exclusive classes) differs from "binary" cross-entropy (a per-class sigmoid). BraTS is sometimes framed as overlapping regions — whole tumour, tumour core, enhancing tumour — where one voxel belongs to several nested regions at once; that multi-*label* setting wants per-class sigmoid + binary cross-entropy, not a single softmax.
