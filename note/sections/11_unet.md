## 11. The U-Net: encoder, decoder, and skip connections {#unet}

In §10 we built the **convolution** — a small sliding filter that detects a local pattern — and stacked convolutions into a network. That machinery is enough to *classify* a whole image (one label for the picture). But §9 set us a harder task: **semantic segmentation**, labelling *every* voxel of a brain MRI as healthy tissue, edema, enhancing tumour, and so on. The output must be as large and detailed as the input, yet the network must still understand the whole scene. The **U-Net** — introduced by Ronneberger, Fischer and Brox at MICCAI 2015 for biomedical image segmentation [Ronneberger, Fischer & Brox, MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) — is the design that made this practical, and it remains the default backbone for medical segmentation. This section builds it piece by piece.

A quick vocabulary note. A **pixel** is one dot of a 2D image; a **voxel** ("volume element") is its 3D counterpart, one tiny cube of a scanned volume. A **feature map** is the grid of numbers a layer produces — think of it as a stack of filtered images, one per **channel** (one per filter). "Downsampling" means shrinking that grid (fewer pixels, coarser); "upsampling" means growing it back.

### The U-shape: encoder, then decoder

The U-Net has two halves that give it its shape. The **contracting path**, or **encoder**, repeatedly downsamples the image, capturing *context* — what is present in a large region. The mirror-image **expanding path**, or **decoder**, upsamples back to full resolution, recovering *precise localization* — exactly where each boundary sits. The paper's central claim is that these two goals, usually in tension, can be met at once: the encoder "captures context" and the symmetric decoder "enables precise localization" [Ronneberger, Fischer & Brox, MICCAI 2015](https://arxiv.org/abs/1505.04597).

![U-shaped encoder–decoder with skip connections](figures/diagrams/s11_unet_ushape.png)

Concretely, the original U-Net is a CNN of about 23 convolutional layers. Each encoder **block** applies two $3\times3$ **unpadded** ("valid") convolutions, each followed by a **ReLU** (the "rectified linear" nonlinearity from §10, which zeroes negatives), then a $2\times2$ **max-pooling** with stride 2 that keeps the largest value in each $2\times2$ window and so halves each spatial dimension. Every downsampling step *doubles* the number of channels: $64 \to 128 \to 256 \to 512 \to 1024$ [Ronneberger, Fischer & Brox, MICCAI 2015](https://arxiv.org/abs/1505.04597). The intuition is a fair trade: as we throw away spatial resolution we buy the room to describe *more kinds* of pattern.

!!! intuition "Intuition"
    Encoder = "zoom out to understand"; decoder = "zoom back in to point precisely." Pooling trades away spatial detail for a wider view, so the network can tell *what* it is looking at (tumour vs healthy tissue). Upsampling rebuilds full resolution so it can say exactly *where* the boundary is. The U-shape drawn on paper is literally this out-then-in journey.

### The size formula, and why the map shrinks

The one arithmetic tool we need governs how a layer changes a spatial dimension. Applied independently to each axis:

$$o = \left\lfloor \frac{i + 2p - k}{s} \right\rfloor + 1$$

Here $o$ is the output length along one axis, $i$ the input length, $p$ the zero-**padding** added to each side, $k$ the **kernel** (filter) size, and $s$ the **stride** (how far the filter hops each step). The $\lfloor \cdot \rfloor$ is the floor (round down). Two special cases recur: a $3\times3$ *valid* convolution ($p=0,\,k=3,\,s=1$) gives $o = i-2$ — it shaves a one-pixel rim off each side; a $2\times2$ max-pool ($k=s=2,\,p=0$) gives $o = \lfloor i/2\rfloor$ — it halves the axis.

Let us trace a real input side length through the encoder, $i = 572$, exactly as in the paper.

$$\begin{aligned}
\text{conv 1:}\quad & o = \tfrac{572 - 3}{1} + 1 = 569 + 1 = 570 \\
\text{conv 2:}\quad & o = \tfrac{570 - 3}{1} + 1 = 567 + 1 = 568 \\
\text{pool:}\quad & o = \left\lfloor \tfrac{568 - 2}{2}\right\rfloor + 1 = 283 + 1 = 284
\end{aligned}$$

So the two valid convolutions together subtract 4 (each subtracts 2), taking $572 \to 568$, and the pool halves it to $284$. Note the pool only divides cleanly because $568$ is even. Repeating the block:

$$284 \to 280 \to 140 \to 136 \to 68 \to 64 \to 32,$$

where at each level the "$-4$" is the two convs and the halving is the pool. After the bottleneck's two convs the coarsest map is $32 \to 30 \to 28$: a $28\times28$ grid with 1024 channels, the point of widest context. Threaded back through the symmetric decoder, the $572\times572$ input yields a smaller $388\times388$ output [Ronneberger, Fischer & Brox, MICCAI 2015](https://arxiv.org/abs/1505.04597). That shrinkage is the price of *unpadded* convolutions, which avoid fabricating values at the border.

!!! gotcha "Gotcha"
    The original U-Net uses **unpadded** convolutions, so the output ($388\times388$) is smaller than the input ($572\times572$), and encoder feature maps must be **center-cropped** before they are joined to the decoder (e.g. a $64\times64$ encoder map cropped to $56\times56$ to meet an up-convolved $56\times56$ decoder map). Most modern re-implementations instead use "same" (padded) convolutions so sizes match and no cropping is needed — at the cost of slightly worse border predictions. If you assume padding when the design uses valid convolutions, your size and cropping logic will be wrong.

### Upsampling in the decoder

To grow a map back, the decoder uses an **up-convolution** (also called a transposed or "de-" convolution): a learnable upsampling whose output length is

$$o = s\,(i-1) + k - 2p.$$

U-Net uses $k=2,\,s=2,\,p=0$, giving $o = 2(i-1)+2 = 2i$ — it exactly doubles the axis, undoing one $2\times2$ pool. After each up-convolution the decoder concatenates the matching encoder map (next), then applies two $3\times3$ convolutions that also *halve* the channel count, mirroring the encoder's doubling.

### Skip connections: the decisive trick

Downsampling discards fine spatial detail; a decoder built from the coarse bottleneck alone would draw blurry, mislocalized boundaries. The **skip connection** fixes this: at each level the high-resolution encoder feature map is copied across and joined onto the matching decoder map. Crucially, in U-Net this join is a **concatenation** — the encoder and decoder maps are stacked along the channel axis — not the additive shortcut used in [ResNet](https://arxiv.org/abs/1512.03385) (the U-Net design itself is from [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)).

![Concatenating a 64-channel encoder map with a 64-channel decoder map yields 128 channels](figures/diagrams/s11_skip_concat.png)

!!! intuition "Intuition"
    A skip connection is a memory hand-off, not just a gradient trick. Pooling deletes fine detail; the encoder keeps a high-resolution "photocopy" at each level and staples it onto the decoder at the matching level. The decoder then reconstructs sharp edges from preserved detail instead of hallucinating them from a blurry map — which is why U-Net boundaries are crisp. As a bonus, the short encoder-to-decoder path lets gradients (§17) reach early layers easily, so training is faster and more stable.

Because concatenation *stacks* the two maps, it doubles the channel count: joining a 64-channel decoder map with a 64-channel encoder map gives 128 channels, so the next convolution has 128 input channels. ResNet-style addition would instead require identical channel counts and would blend the maps rather than preserve the encoder's detail.

### Why context is cheap: the receptive field

Pooling earns its keep through the **receptive field** $r_\ell$ — the size of the input region that influences one unit after layer $\ell$. It grows by the recurrence

$$r_{\ell} = r_{\ell-1} + (k_{\ell}-1)\prod_{m=1}^{\ell-1} s_m, \qquad j_{\ell} = j_{\ell-1}\,s_{\ell},$$

where $k_\ell$ is layer $\ell$'s kernel size, $s_m$ layer $m$'s stride, and $j_\ell$ the "jump" (input-pixel spacing between adjacent output units), starting from $r_0 = 1,\, j_0 = 1$. The product is the cumulative stride before layer $\ell$. Trace a mini-encoder — conv$(3,s1)$, conv$(3,s1)$, pool$(2,s2)$, conv$(3,s1)$:

$$\begin{aligned}
\text{conv 1:}\quad & r_1 = 1 + (3-1)\cdot 1 = 3, & j_1 &= 1 \\
\text{conv 2:}\quad & r_2 = 3 + (3-1)\cdot 1 = 5, & j_2 &= 1 \\
\text{pool:}\quad & r_3 = 5 + (2-1)\cdot 1 = 6, & j_3 &= 1\cdot 2 = 2 \\
\text{conv 3:}\quad & r_4 = 6 + (3-1)\cdot 2 = 10, & j_4 &= 2
\end{aligned}$$

The pool barely adds to $r$ but *doubles the jump* to 2 — so the very next identical $3\times3$ conv now reaches 4 input pixels instead of 2. Each pooling stage multiplies the reach of every later convolution; that is how the encoder acquires whole-tumour context after only a handful of layers. And exactly the detail pooling discards is what the skip connections restore.

### From 2D to 3D — and why it is expensive

The original U-Net is 2D, built for microscopy, and it won the ISBI 2015 cell-tracking challenge from very few images by leaning on data augmentation [Ronneberger, Fischer & Brox, MICCAI 2015](https://arxiv.org/abs/1505.04597). But a brain MRI is **volumetric** — a 3D stack of slices — and a tumour's shape only makes sense in 3D. Çiçek et al. (MICCAI 2016) introduced the **3D U-Net**, replacing every 2D operation with its 3D counterpart ($3\times3\times3$ convolutions, $2\times2\times2$ pooling, $2\times2\times2$ up-convolutions) so the network reasons across slices [Çiçek et al., MICCAI 2016](https://arxiv.org/abs/1606.06650). It also adds **batch normalization** (a per-layer rescaling, §15) before each ReLU for faster, more stable convergence, and starts from 32 channels rather than 64 [Çiçek et al., MICCAI 2016](https://arxiv.org/abs/1606.06650).

The catch is cost. The number of learnable weights in a convolution is

$$P_{2D} = (k_h k_w\,C_{in} + 1)\,C_{out}, \qquad P_{3D} = (k_d k_h k_w\,C_{in} + 1)\,C_{out},$$

with $k$ the kernel sizes, $C_{in}$/$C_{out}$ the input/output channels, and the $+1$ one bias per output channel. For $C_{in}=C_{out}=64$:

$$\begin{aligned}
P_{2D} &= (3\cdot3\cdot64 + 1)\cdot 64 = 577 \cdot 64 = 36{,}928 \\
P_{3D} &= (3\cdot3\cdot3\cdot64 + 1)\cdot 64 = 1729 \cdot 64 = 110{,}656
\end{aligned}$$

a ratio $110{,}656 / 36{,}928 \approx 3.0$, because the kernel volume grew from 9 to 27. Compute is far worse. Counting multiply-accumulate operations,

$$\text{MACs} = o_1 o_2 \cdots o_n \cdot k_1 k_2 \cdots k_n \cdot C_{in} \cdot C_{out},$$

with $o_1..o_n$ the output size along each of the $n$ axes. Compare a $128\times128$ map against a $128\times128\times128$ volume:

$$\begin{aligned}
\text{MACs}_{2D} &= (128\cdot128)\cdot 9 \cdot 64\cdot64 = 16{,}384 \cdot 36{,}864 \approx 6.04\times10^{8} \\
\text{MACs}_{3D} &= (128^3)\cdot 27 \cdot 64\cdot64 = 2{,}097{,}152 \cdot 110{,}592 \approx 2.32\times10^{11}
\end{aligned}$$

The ratio is $\approx 384$, and it factorises cleanly: $128\times$ more output positions (the extra depth axis) times $3\times$ more kernel weights $= 384$. Activation memory blows up the same way.

!!! gotcha "Gotcha"
    Naively feeding a whole brain MRI (say $240\times240\times155$) into a 3D U-Net at 64 starting channels will exhaust GPU memory — activation memory scales as (voxels $\times$ channels) and voxels grow cubically. That single $\sim$380$\times$ blow-up is why 3D U-Nets train on small cropped **patches** (e.g. $128\times128\times128$), use fewer starting channels (32), add batch normalization, and reassemble the full volume by **sliding-window inference**.

!!! example "Example question"
    A 3D U-Net encoder block applies two $3\times3\times3$ "same"-padded convolutions (stride 1) then a $2\times2\times2$ max-pool (stride 2). Starting from a $128\times128\times128$ patch, what is the spatial size after 4 blocks, and why must the patch side be a multiple of 16?

    **Solution.** With "same" padding each conv preserves size ($o=i$), so within a block only the pool changes it: $o=\lfloor i/2\rfloor$. Four halvings: $128 \to 64 \to 32 \to 16 \to 8$. After 4 blocks the map is $8\times8\times8$. Surviving 4 clean halvings requires divisibility by $2^4 = 16$; indeed $128 = 8\times16$. A size like 120 gives $120\to60\to30\to15\to7$ — the last $\lfloor 15/2\rfloor = 7$ drops a voxel, so the decoder's up-convolutions ($7\to14\to28\to56\to112$) no longer match the 120 input and the skip concatenation breaks. Hence patch dimensions are multiples of 16 for a 4-level network.

!!! example "Example question"
    Two networks process a $256\times256$ image: (a) one 2D $5\times5$ conv layer, (b) two stacked 2D $3\times3$ conv layers. Both have $C_{in}=C_{out}=32$. Compare parameter counts and receptive fields, and say which U-Net follows.

    **Solution.** Using $P=(k_h k_w C_{in}+1)C_{out}$: (a) $5\times5$: $P=(25\cdot32+1)\cdot32 = 801\cdot32 = 25{,}632$. (b) each $3\times3$: $(9\cdot32+1)\cdot32 = 289\cdot32 = 9{,}248$, so two = $18{,}496$. Receptive fields: a $5\times5$ covers $5\times5$; two stacked $3\times3$ also cover $5\times5$ (by the recurrence, $r = 1+2+2 = 5$). Same view, but the stacked design uses fewer weights ($18{,}496$ vs $25{,}632$) and inserts an extra ReLU between the convs, making it more expressive. U-Net follows design (b) — pairs of $3\times3$ convolutions at every level.

### Output and losses, in brief

At the very end, U-Net applies a **pixel-wise soft-max** over the final feature map (turning per-voxel class scores into probabilities) combined with a cross-entropy loss, plus a precomputed per-pixel **weight map** that up-weights the thin gaps between touching cells to force the network to learn separating borders [Ronneberger, Fischer & Brox, MICCAI 2015](https://arxiv.org/abs/1505.04597). That soft-max and cross-entropy, and the class-imbalance problem the weight map addresses, are the subjects of §12 and §14; the Dice-based losses that dominate modern tumour segmentation follow in §13. Here the takeaway is architectural: an encoder for context, a decoder for localization, and skip connections to marry the two — the foundation the rest of the pipeline (§15 normalization, §17 training, §19 this project's design) is built on.
