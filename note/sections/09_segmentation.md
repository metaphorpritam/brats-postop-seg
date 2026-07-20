## 9. From classification to semantic segmentation {#segmentation}

Before a network can be trained on brain scans, we have to be precise about *what question it answers*. Three tasks in computer vision sound similar but differ entirely in the **shape of their output**, and that difference dictates the whole architecture.

- **Image classification** answers "what is in this picture?" with a single label for the entire image (e.g. "tumor present").
- **Object detection** answers "what is where?" by drawing a few **bounding boxes** — rectangles — around objects, each with a class label.
- **Semantic segmentation** answers the same "what is where?" but *for every location*: it outputs a **label map** the same spatial size as the input, assigning each pixel a class [FCN](https://arxiv.org/abs/1411.4038). In 3D medical imaging the unit is not a pixel but a **voxel** — a "volume pixel", one cell in the 3D grid of the scan (see §6 for how MRI produces that grid). So semantic segmentation gives *one class label per voxel*.

![Three tasks on one brain slice: classification emits one label, detection a box, segmentation a label per voxel](figures/diagrams/s9_output_shapes.png)

The key mental model: **classification = one label per image; detection = a few boxes per image; semantic segmentation = one label per voxel, all decided together.** Classification and detection produce *sparse* output (one vector, or a handful of boxes); segmentation produces *dense* output — as detailed as the input itself [Minaee et al.](https://arxiv.org/pdf/2001.05566). (Note: semantic segmentation has no notion of *separate instances* — two touching tumors both just become "tumor" voxels; telling instances apart is a further task called instance segmentation, which we do not need for the BraTS regions of §7.)

!!! intuition "Intuition"
    Segmentation is "classification with the sliding window folded into the network". You *could* crop a little patch around every voxel and classify each patch separately — but that is enormously slow and redundant, since neighbouring patches overlap almost completely. Instead, an encoder-decoder computes shared features once and emits *all* voxel decisions in a single pass. The very last layer is literally a classifier applied in parallel at every location, with the same weights everywhere.

### Why a classification network cannot be used as-is

A standard classification CNN (convolutional neural network, §10) is *built to destroy* spatial resolution. It repeatedly **pools** and **strides** — steps that shrink the feature grid — collapsing a large image toward a tiny grid, then attaches a **fully-connected head** (a layer wired to every position at once) that emits one label vector for the whole image. That deliberately discards the per-location information segmentation needs. Two changes convert it into a segmentation network [FCN](https://arxiv.org/abs/1411.4038): **(1)** make it *fully convolutional* — replace the fully-connected head with convolutions, so the output is a spatial grid of logit vectors, one per location, at any input size; and **(2)** add a **decoder** that expands that coarse grid back to full resolution, with **skip connections** re-injecting fine detail. This is the **encoder-decoder** design, formalized by the **U-Net** [U-Net](https://arxiv.org/abs/1505.04597) and generalized to volumes by the **3D U-Net** [3D U-Net](https://arxiv.org/abs/1606.06650), which we build in full in §11.

### Derivation: tracing spatial size through the encoder and decoder

To see *quantitatively* why we must compress then re-expand, we track the size along one axis. The output size $O$ of a convolution or pooling operation, per axis, is

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

where $W$ is the input size along that axis, $K$ the **kernel** size (the width of the little window that slides across the image), $P$ the **padding** (zeros added to each side), $S$ the **stride** (how many positions the window jumps each step), and $\lfloor\cdot\rfloor$ the floor (round down). Applied per axis, this governs the whole network.

**Step 1 — a "same" convolution keeps the size.** Take a $3\times3$ convolution with $S=1$, $P=1$ on input $W=128$:

$$O = \left\lfloor \frac{128 - 3 + 2\cdot 1}{1} \right\rfloor + 1 = \left\lfloor 127 \right\rfloor + 1 = 128.$$

Size is preserved. Stacking such convolutions changes the *features* (what each location represents) without changing *resolution*. This padding choice is called **"same"** convolution.

**Step 2 — pooling halves the size (encoder downsampling).** Apply $2\times2$ max-pooling ($K=2$, $S=2$, $P=0$) on $W=128$:

$$O = \left\lfloor \frac{128 - 2 + 0}{2} \right\rfloor + 1 = \lfloor 63 \rfloor + 1 = 64.$$

Each pooling stage halves each dimension. Four stages give $128 \to 64 \to 32 \to 16 \to 8$.

**Step 3 — why shrink at all: the receptive field.** The **receptive field** is how many input voxels influence one output value. Downsampling makes each deep neuron "see" a large region of the original volume, giving the semantic context needed to distinguish tumor from healthy tissue. But an $8\times8$ grid is far too coarse to be one label per original voxel.

**Step 4 — a classifier would stop here** by flattening the $8\times8$ grid into a single label — exactly what we must *not* do.

**Step 5 — the decoder upsamples.** To undo Step 2 we double the size at each decoder stage, commonly with a **transposed convolution** (an "upsampling convolution"), whose per-axis output size is $O = S(W-1) + K - 2P$. With $W=8$, $K=2$, $S=2$, $P=0$:

$$O = 2\,(8-1) + 2 - 0 = 16.$$

Repeating: $8 \to 16 \to 32 \to 64 \to 128$. We are back to full resolution.

**Step 6 — skip connections restore lost detail.** Upsampling from an $8\times8$ grid cannot reinvent the sharp boundaries that pooling destroyed. So the encoder feature map of matching size (the $64\times64$ map) is copied straight across and concatenated into the $64\times64$ decoder stage. The decoder thus fuses coarse "what" with fine "where".

**Step 7 — the final $1\times1$ convolution is the per-voxel classifier.** At the restored resolution, a $1\times1$ convolution with $C$ output channels maps each voxel's feature vector to $C$ scores. This convolution *is* a classifier applied identically and in parallel at every location — "classification at every voxel", weights shared everywhere.

![Encoder halves size to a bottleneck of max context, decoder doubles it back, skip connections copy detail across](figures/diagrams/s9_encoder_decoder.png)

### From per-voxel scores to a label

At each voxel $i$ the network emits a vector of raw scores $z_{i,1},\dots,z_{i,C}$ called **logits**. A **softmax** turns each vector into a probability distribution over the $C$ mutually-exclusive classes:

$$p_{i,c} = \frac{\exp(z_{i,c})}{\sum_{k=1}^{C} \exp(z_{i,k})}$$

where $p_{i,c}$ is the predicted probability that voxel $i$ is class $c$; the denominator sums the exponentials over all classes so the $p_{i,c}$ are non-negative and sum to 1. The predicted label is the class with the highest probability (the **argmax**). We minimize a per-voxel **cross-entropy** loss, averaged over all $N$ voxels,

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\,\log p_{i,c},$$

where $y_{i,c}$ is the **one-hot** ground truth ($=1$ if voxel $i$ truly is class $c$, else $0$). The gradient that training follows takes a famously clean form — derived fully in §12 — namely predicted-probability minus target:

$$\frac{\partial \mathcal{L}_{CE}}{\partial z_{i,c}} = \frac{1}{N}\left(p_{i,c} - y_{i,c}\right).$$

![One voxel: feature vector to C logits via 1x1 conv, to probabilities via softmax, to a class via argmax](figures/diagrams/s9_voxel_pipeline.png)

!!! gotcha "Gotcha: softmax vs sigmoid is a modeling choice, not a detail"
    Softmax forces classes to be **mutually exclusive** — the probabilities sum to 1, so a voxel gets exactly one label. But BraTS evaluates **nested regions**: enhancing tumor sits inside tumor core, which sits inside whole tumor, so a voxel legitimately belongs to several regions at once [BraTS](https://www.synapse.org/brats). Overlapping targets like these need *independent per-region* sigmoid outputs, not a single softmax. Using softmax on nested targets silently mis-models the problem. (Class imbalance — tumor is often a tiny fraction of the voxels, quantified with sources in §14 — further breaks plain cross-entropy, which is why we add a Dice loss in §13 and revisit imbalance in §14.)

!!! example "Example question: softmax, loss, and gradient at one voxel"
    A voxel has logits $z = [2.0,\ 1.0,\ 0.1]$ for three classes $(0,1,2)$; its true class is $0$. Compute the softmax probabilities, the cross-entropy loss, and $\partial \mathcal{L}/\partial z$.

    **Solution.**
    Step 1 — exponentiate: $\exp(2.0)=7.3891$, $\exp(1.0)=2.7183$, $\exp(0.1)=1.1052$.
    Step 2 — denominator: $S = 7.3891 + 2.7183 + 1.1052 = 11.2126$.
    Step 3 — probabilities: $p_0 = 7.3891/11.2126 = 0.6590$, $p_1 = 2.7183/11.2126 = 0.2424$, $p_2 = 1.1052/11.2126 = 0.0986$ (sum $=1.000$).
    Step 4 — loss with one-hot $y=[1,0,0]$: $\mathcal{L} = -\log p_0 = -\ln(0.6590) = 0.4170$.
    Step 5 — gradient via $p - y$: $\partial\mathcal{L}/\partial z = [0.6590-1,\ 0.2424-0,\ 0.0986-0] = [-0.3410,\ 0.2424,\ 0.0986]$.
    The negative first component pushes logit 0 *up* (more confidence in the correct class); the positives push logits 1 and 2 *down*. The three components sum to $0$, as they must for softmax.

!!! example "Example question: Dice vs IoU on a tiny slice"
    On a $4\times4$ slice the true tumor mask has 6 tumor pixels; the model predicts 8, of which 5 coincide with true tumor. Compute the Dice coefficient and the IoU, and say which is stricter.

    **Solution.** Let $A$ = predicted set ($|A|=8$), $B$ = true set ($|B|=6$), overlap $|A\cap B|=5$.
    Dice: $\mathrm{DSC} = \dfrac{2|A\cap B|}{|A|+|B|} = \dfrac{2\cdot 5}{8+6} = \dfrac{10}{14} = 0.714$.
    IoU: union $= |A|+|B|-|A\cap B| = 8+6-5 = 9$, so $\mathrm{IoU} = \dfrac{5}{9} = 0.556$.
    IoU ($0.556$) is lower than Dice ($0.714$) on the same prediction — IoU is **stricter** because it keeps the whole union in the denominator. Cross-check with the identity $\mathrm{DSC} = \dfrac{2\,\mathrm{IoU}}{1+\mathrm{IoU}} = \dfrac{2\cdot 0.556}{1.556} = 0.714$. Consistent. (Both metrics return in §13 and §16.)

In the BraTS post-treatment setting, this exact machinery labels every voxel into tumor sub-regions — enhancing tumor, non-enhancing tumor core, edema, resection cavity (§7) — which are then grouped into the clinically evaluated nested regions. The rest of the note fills in each moving part: convolutions (§10), the U-Net (§11), softmax and cross-entropy (§12), and Dice (§13).
