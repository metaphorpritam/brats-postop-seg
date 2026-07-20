## 10. CNNs and convolutions: the core operation {#cnns}

In §9 we framed segmentation as labelling every voxel of a brain MRI. The workhorse that actually does that labelling in this project is a **convolutional neural network (CNN)** — a neural network whose central operation is the **convolution**. This section builds the convolution from scratch: what it computes, the knobs that control it, how large a region each output "sees", and how to count the weights a layer must learn. Everything here feeds directly into the U-Net of §11.

### What a convolution does

A convolution slides a small window of learnable numbers — a **kernel** (also called a **filter**) — across the image and, at every position, computes a **weighted sum** of the pixels the window covers, then adds a single **bias** (a constant offset). That one number becomes one entry of the **output feature map** (the grid of numbers the layer produces). Because the *same* kernel is dragged across every location, a CNN has two defining properties: **local connectivity** (each output looks at only a small patch) and **weight sharing** (the same weights are reused everywhere) [CS231n](https://cs231n.github.io/convolutional-networks/).

Formally, for one output channel:

$$
Y(i,j) \;=\; b \;+\; \sum_{c=1}^{C_{in}} \sum_{u=0}^{F_h-1} \sum_{v=0}^{F_w-1} W(c,u,v)\, X\big(c,\; iS+u-P,\; jS+v-P\big)
$$

Here $Y(i,j)$ is the output value at row $i$, column $j$; $b$ is the scalar bias for this filter; $C_{in}$ is the number of **input channels** (explained below); $F_h$ and $F_w$ are the kernel's height and width; $W(c,u,v)$ is the learnable weight for channel $c$ at offset $(u,v)$ inside the window; $X$ is the (zero-padded) input; $S$ is the **stride** (how far the window jumps each step); and $P$ is the **padding** (a border of zeros added around the input). The triple sum is just "multiply every weight by the pixel underneath it and add them all up", across all input channels at once.

**Channels** are feature slots, not spatial directions. A colour photo has 3 input channels (red, green, blue); a filter spans all of them together. In BraTS the four MRI sequences — T1, T1ce (contrast-enhanced), T2, and FLAIR (§6) — enter as four input channels, exactly as R, G, B are channels of a photo [BraTS/Menze et al.](https://pubmed.ncbi.nlm.nih.gov/25494501/). So a first filter can learn a cross-modal pattern like "bright on FLAIR but dark on T1" in a single weighted sum.

![A 3x3 kernel slides over a padded input grid; each placement multiplies weights by the covered pixels and sums to one output cell, then the window strides on.](figures/diagrams/s10_sliding_kernel.png)

!!! intuition "Intuition"
    A kernel is a little pattern detector that reports how strongly its pattern appears at each location. Because the exact same detector is dragged across the whole image (weight sharing), a feature learned in one corner is automatically recognised everywhere. That is why CNNs need far fewer parameters than a fully-connected network — and, as we'll prove below, why the parameter count is completely blind to image size.

!!! gotcha "Gotcha"
    What every deep-learning library (PyTorch, TensorFlow) calls "convolution" is technically **cross-correlation**: the kernel is applied without the flip that strict mathematical convolution requires [Goodfellow et al.](https://www.deeplearningbook.org/contents/convnets.html). Because the weights are *learned*, the flip changes nothing about accuracy — but it matters if you hand-check the math against a signal-processing textbook.

### The output-size formula, derived

How big is the output? Along one axis the answer is

$$
O \;=\; \left\lfloor \dfrac{W - F + 2P}{S} \right\rfloor + 1
$$

where $O$ is the output size, $W$ the input size, $F$ the kernel size, $P$ the padding per side, and $S$ the stride. Here is every step [CS231n](https://cs231n.github.io/convolutional-networks/):

$$
\begin{aligned}
W' &= W + 2P &&\text{pad } P \text{ zeros on each side; } W' \text{ is the padded length}\\
p + F - 1 &\le W' - 1 &&\text{the kernel starting at index } p \text{ must fit inside}\\
p_{\max} &= W' - F &&\text{so the largest valid start is } W' - F\\
p &= 0,\,S,\,2S,\dots \le W'-F &&\text{with stride } S \text{ we only start at multiples of } S\\
k_{\max} &= \left\lfloor \dfrac{W'-F}{S}\right\rfloor &&\text{largest integer } k \text{ with } kS \le W'-F\\
O &= k_{\max} + 1 = \left\lfloor \dfrac{W - F + 2P}{S}\right\rfloor + 1 &&\text{count positions } k=0,\dots,k_{\max}
\end{aligned}
$$

The **floor** $\lfloor\cdot\rfloor$ (round down to the nearest integer) appears because $S$ may not divide $W'-F$ evenly; positions that would overshoot the edge are simply not taken. Sanity check: $W=5,\,F=3,\,P=0,\,S=1$ gives $\lfloor(5-3)/1\rfloor+1 = 3$, the familiar "a length-3 filter on a length-5 signal yields 3 outputs".

**"Same" padding.** We often want the output to keep the input's size. Setting $O=W$ with $S=1$:

$$
\begin{aligned}
W &= (W - F + 2P) + 1 &&\text{plug } O=W,\ S=1 \text{ (no floor since } S=1)\\
0 &= -F + 2P + 1 &&\text{subtract } W \text{ from both sides}\\
P &= \dfrac{F-1}{2} &&\text{solve for } P
\end{aligned}
$$

So $P=(F-1)/2$ preserves size — this is **"same" padding**. It needs $F$ odd (so $P$ is a whole number), which is why kernels are almost always $3,5,7$. Check $F=3$: $P=1$, and $O=\lfloor(W-3+2)/1\rfloor+1 = W$. **"Valid" padding** means $P=0$, which shrinks the output.

!!! gotcha "Gotcha"
    Padding choice silently changes output size. The original U-Net (§11) used *valid* (unpadded) $3\times3$ convolutions, so a $572\times572$ input became a $388\times388$ output [Ronneberger et al.](https://arxiv.org/abs/1505.04597). That mismatch is why its encoder feature maps must be cropped before being concatenated onto the decoder. Most modern reimplementations switch to *same* padding to sidestep the bookkeeping.

### Counting parameters

A single filter needs one weight per element of the patch it covers. The patch spans $F_h$ rows, $F_w$ columns, and *all* $C_{in}$ input channels, so it holds $F_h\,F_w\,C_{in}$ weights, plus one bias. A layer has $C_{out}$ such filters (one per output channel), giving

$$
N_{2D} \;=\; \big(F_h \cdot F_w \cdot C_{in} + 1\big)\cdot C_{out}.
$$

Nothing here mentions the image height or width — a direct consequence of weight sharing. For **3D convolution** the kernel gains a depth extent $F_d$, so

$$
N_{3D} \;=\; \big(F_d \cdot F_h \cdot F_w \cdot C_{in} + 1\big)\cdot C_{out}.
$$

A worked contrast (a BraTS-style first layer, $C_{in}=4$, $C_{out}=32$, kernel $3\times3\times3$) [Çiçek et al.](https://arxiv.org/abs/1606.06650):

- Kernel volume $=3\cdot3\cdot3 = 27$ weights per input channel.
- Per filter $= 27\cdot 4 = 108$ weights, plus 1 bias $= 109$.
- Total $= 109 \times 32 = 3488$ parameters.

The 2D analogue (same channels, $3\times3$ kernel): per filter $=3\cdot3\cdot4+1 = 37$, total $37\times32 = 1184$. The 3D layer has $3488/1184 \approx 2.95\times$ more weights — essentially the kernel-volume ratio $27/9 = 3$. This is the central engineering tension of the project: 3D kernels capture how tissue changes *between* MRI slices, but at roughly triple the weights and far more memory.

!!! gotcha "Gotcha"
    The parameter count omits image size — but **compute and activation memory very much do not**. Multiply-accumulate cost is $\text{MACs}=H_{out}\,W_{out}\,C_{out}\,(F_h F_w C_{in})$, scaling with the output grid. A 3D conv can have a modest parameter count yet blow up GPU memory because it stores a whole $H\times W\times D\times C_{out}$ activation volume — the reason BraTS training uses patches rather than whole brains (§17). Also: don't forget the $+1$ bias, and don't forget to multiply the kernel volume by $C_{in}$. (If a BatchNorm follows, the bias is usually disabled as redundant.)

![Two-D versus three-D convolution: a 3x3 kernel sliding inside one MRI slice on the left; a 3x3x3 kernel spanning three adjacent slices on the right, fed by four stacked MRI-sequence channels.](figures/diagrams/s10_2d_vs_3d.png)

### Receptive field: depth buys size

The **receptive field** is the region of the *original* input that influences one activation. It grows with depth via

$$
R_{\ell} \;=\; R_{\ell-1} + (F_{\ell}-1)\prod_{i=1}^{\ell-1} S_i,
\qquad R_0 = 1,
$$

where $R_\ell$ is the receptive field (in input pixels) at layer $\ell$, $F_\ell$ its kernel size, and the product is the cumulative stride of all earlier layers. For plain stride-1 stacking the product is 1, so $R_\ell = R_{\ell-1} + (F_\ell-1)$:

$$
R_0 = 1,\quad R_1 = 1+2 = 3,\quad R_2 = 3+2 = 5,\quad R_3 = 5+2 = 7.
$$

Two stacked $3\times3$ convolutions see a $5\times5$ region; three see $7\times7$ — matching a single $7\times7$ kernel's reach. And stacking is cheaper: per output channel over $C$ channels (ignoring bias), two $3\times3$ layers cost $2\cdot(3\cdot3\cdot C)C = 18C^2$ weights versus $(5\cdot5\cdot C)C = 25C^2$ for one $5\times5$ — plus the stack inserts an extra non-linearity, making the function more expressive. This is the VGG argument [Simonyan & Zisserman](https://arxiv.org/abs/1409.1556) behind most modern CNNs and the 3D U-Net (§11).

![Receptive-field staircase: one input pixel expands to a 3x3 region after one conv, 5x5 after two, and 7x7 after three stacked 3x3 convolutions.](figures/diagrams/s10_receptive_field.png)

!!! intuition "Intuition"
    Depth buys size for free. A single number deep in the network summarizes a large patch of the original image, even though every individual layer peers through only a tiny $3\times3$ window — like grasping a paragraph by reading overlapping few-word windows. The network sees big structures (a whole tumour) not by using big kernels but by stacking many small ones.

!!! example "Example question"
    A 3D conv layer takes a $128\times128\times128$ volume with 4 channels (the four MRI sequences), using 32 filters of size $3\times3\times3$, stride 1, "same" padding. (a) How many learnable parameters? (b) What is the output shape?

    **Solution.** (a) Kernel volume $=3\cdot3\cdot3=27$; across $C_{in}=4$ channels $=27\cdot4=108$ weights; plus 1 bias $=109$ per filter; times $C_{out}=32$ filters $=109\times32 = \mathbf{3488}$. (b) "Same" padding uses $P=(3-1)/2 = 1$, so each axis is preserved: $O=\lfloor(128-3+2)/1\rfloor+1 = 128$. Output channels $=$ number of filters $=32$. Output shape $=\mathbf{128\times128\times128\times32}$.

!!! example "Example question"
    A 2D conv layer receives a $224\times224\times3$ image and applies 64 filters of size $7\times7$, stride 2, padding 3 (ResNet's first layer). Give the output spatial size and parameter count.

    **Solution.** $O=\lfloor(224-7+2\cdot3)/2\rfloor+1 = \lfloor223/2\rfloor+1 = 111+1 = \mathbf{112}$, so the output is $112\times112\times64$. Parameters: per filter $=7\cdot7\cdot3+1 = 148$; times 64 $= \mathbf{9472}$. Note the large $112\times112\times64$ activation despite a small parameter count — compute and memory scale with spatial size, parameters do not.
