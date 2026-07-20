## 6. How MRI sees tissue: the modalities {#mri-modalities}

In §2–§5 we met the tumour and the post-treatment brain we want to map. But before a computer can label a tumour (§7, §9), something has to *photograph* it. That something is **magnetic resonance imaging (MRI)** — and the surprising fact is that MRI does not take one photograph. From the same anatomy, by changing the timing of its radio pulses, the scanner produces several differently **weighted** images, called **sequences** or **modalities**. Each makes different tissues bright (**hyperintense**) or dark (**hypointense**). BraTS gives every patient exactly four co-registered structural sequences: native **T1-weighted (T1)**, **post-contrast T1-weighted (T1Gd**, also written T1c or T1CE), **T2-weighted (T2)**, and **T2 Fluid-Attenuated Inversion Recovery (T2-FLAIR)** [BraTS 2020 / UPenn CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html). No single one shows the whole tumour; each reveals a different facet.

### Two clocks: T1 and T2

An MRI works by tipping the tiny magnetic "spins" of hydrogen protons (mostly in water and fat) out of alignment with the scanner's strong field, then listening as they recover. Two independent recovery processes — two clocks — govern how bright a tissue looks.

The first is **longitudinal relaxation**: how fast the spins realign *with* the main field after a 90-degree tipping pulse. Its signal follows

$$M_z(t) = M_0\left(1 - e^{-t/T_1}\right)$$

Here $M_z(t)$ is the **longitudinal magnetization** — the component pointing along the field, which is what can be tipped over to give signal — at time $t$ after the pulse; $M_0$ is the **equilibrium magnetization**, the maximum possible, fixed by proton density and field strength; and $T_1$ is the **longitudinal relaxation time constant** of the tissue. A tissue with short $T_1$ recovers fast; a tissue with long $T_1$ recovers slowly.

The second clock is **transverse relaxation**: how fast the tipped-over spins lose step with each other, causing the detectable signal to fade:

$$M_{xy}(t) = M_{xy}(0)\, e^{-t/T_2}$$

$M_{xy}(t)$ is the **transverse magnetization** — the rotating component the receiver coil actually detects — at time $t$; $M_{xy}(0)$ is its value right after excitation; and $T_2$ is the **transverse relaxation time**. Long $T_2$ means the signal survives longer.

The scanner controls two knobs that select which clock dominates: **TR** (repetition time, the wait between pulses) and **TE** (echo time, the delay before reading out). A **T1-weighted** image uses short TR (~400–700 ms) and short TE; a **T2-weighted** image uses long TR (>2000 ms) and long TE (~60–120 ms) [MRImaster](https://mrimaster.com/t1-vs-t2-vs-pd-vs-flair-mri/).

!!! intuition "Intuition"
    $T_1$ and $T_2$ are two independent stopwatches. $T_1$ is "how fast a knocked-over spin stands back up" (realigns with the magnet). $T_2$ is "how fast a marching band of spins falls out of step." Water is *slow on both*: slow to stand up (long $T_1$) and slow to fall out of step (long $T_2$). That single fact explains almost everything below — including why cerebrospinal fluid is dark on T1 but bright on T2.

The practical consequences: on a **T1-weighted** image fat is bright and watery fluid — including **cerebrospinal fluid (CSF)**, the clear liquid bathing the brain — is dark, so T1 is nicknamed "fat-weighted." On a **T2-weighted** image water and CSF are bright, so T2 is "water-weighted." The fastest way to tell the two apart is CSF: dark on T1, bright on T2 [MRImaster](https://mrimaster.com/t1-vs-t2-vs-pd-vs-flair-mri/). This is why **edema** — tissue swelling from excess water, the "finger-like" spread around a tumour — lights up on T2 [MRImaster](https://mrimaster.com/t1-vs-t2-vs-pd-vs-flair-mri/).

![Four MRI sequences shown as four flashlights answering different questions about the same tumour](figures/diagrams/s6_four_sequences.png)

### FLAIR: T2 with the CSF switched off

Bright CSF on T2 is a problem: a watery lesion sitting next to the fluid-filled ventricles is camouflaged. **FLAIR** solves this. It is a T2-type image with an extra preparation step: a 180-degree **inversion** pulse flips the magnetization to $-M_0$ first, and the readout is timed to catch CSF exactly when its signal is zero, erasing it [Wikipedia: FLAIR](https://en.wikipedia.org/wiki/Fluid-attenuated_inversion_recovery). After a 180-degree inversion the longitudinal magnetization follows

$$M_z(t) = M_0\left(1 - 2\,e^{-t/T_1}\right)$$

Same symbols as before, but the recovery now starts at $-M_0$ instead of 0, so it must climb the full range from $-M_0$ up through 0 to $+M_0$ — that is where the factor 2 comes from. Here $t$ is measured from the inversion pulse and is called the **inversion time (TI)**. We want the TI at which CSF passes through zero.

**Derivation of the null point (no steps skipped):**

$$\begin{aligned}
0 &= M_0\left(1 - 2\,e^{-TI/T_1}\right) && \text{set } M_z(TI)=0 \text{ (no signal to tip over)}\\
0 &= 1 - 2\,e^{-TI/T_1} && \text{divide by } M_0 \neq 0\\
2\,e^{-TI/T_1} &= 1 && \text{move the exponential to one side}\\
e^{-TI/T_1} &= \tfrac{1}{2} && \text{divide by 2}\\
-\frac{TI}{T_1} &= \ln\!\tfrac{1}{2} && \text{take natural log of both sides}\\
-\frac{TI}{T_1} &= -\ln 2 && \text{use } \ln\tfrac1x = -\ln x\\
\frac{TI}{T_1} &= \ln 2 && \text{multiply by } -1\\
TI &= T_1 \ln 2 \approx 0.693\,T_1 && \text{multiply by } T_1
\end{aligned}$$

So the inversion time that erases a tissue is about 69% of its $T_1$ [Wikipedia: FLAIR](https://en.wikipedia.org/wiki/Fluid-attenuated_inversion_recovery). Suppressing bright CSF this way makes periventricular and perilesional edema and white-matter lesions — otherwise hidden by adjacent CSF — clearly visible [AQMDI](https://aqmdi.com/t2-flair-on-an-mri/).

!!! example "Example question"
    CSF at 1.5 T has $T_1 \approx 4000$ ms. Estimate the FLAIR TI needed to null it.

    **Solution:** Substitute into $TI = T_1 \ln 2$: $TI = 4000 \times 0.693 = 2772$ ms, about 2.77 s. Real FLAIR protocols use a somewhat *shorter* TI (~2000–2500 ms) because the repetition time TR is finite, so magnetization does not fully recover between shots and the true zero-crossing arrives earlier. The simple formula still gives the right ballpark and the key intuition: CSF's very long $T_1$ forces a very long TI.

### T1Gd: a dye that only leaks through broken tumour vessels

The fourth sequence is a plain T1 image taken *after* injecting a **gadolinium** contrast agent. Gadolinium is a paramagnetic metal that strongly shortens the $T_1$ of nearby water; shorter $T_1$ means faster recovery, hence higher signal (**enhancement**) on T1-weighted images [Wikipedia: MRI contrast agent](https://en.wikipedia.org/wiki/MRI_contrast_agent). Crucially, gadolinium chelates are water-loving and cannot cross an intact **blood–brain barrier (BBB)** — the tight seal that normally keeps blood out of brain tissue. They enhance only where the BBB is broken, as in aggressive tumour, so enhancement marks leaky, biologically active tumour and new vessel growth [Wikipedia: MRI contrast agent](https://en.wikipedia.org/wiki/MRI_contrast_agent). The concentration effect follows the relaxivity relation

$$\frac{1}{T_1} = \frac{1}{T_{1,0}} + r_1\,[\mathrm{Gd}]$$

where $T_1$ is the observed relaxation time after contrast, $T_{1,0}$ is the native (pre-contrast) value, $r_1$ is the agent's **longitudinal relaxivity** (units $\mathrm{mM^{-1}\,s^{-1}}$), and $[\mathrm{Gd}]$ is the local gadolinium concentration in millimolar (mM). Note the equation adds *rates* $1/T_1$ (per second), not times.

!!! example "Example question"
    Native tissue has $T_{1,0} = 1000$ ms. A typical agent has $r_1 = 4\ \mathrm{mM^{-1}\,s^{-1}}$; contrast leaks to $[\mathrm{Gd}] = 0.5$ mM. What is the new $T_1$, and why does the tissue brighten?

    **Solution, step by step:**
    (1) Native rate: $\dfrac{1}{T_{1,0}} = \dfrac{1}{1000\text{ ms}} = \dfrac{1}{1\text{ s}} = 1\ \mathrm{s^{-1}}$.
    (2) Gadolinium contribution: $r_1[\mathrm{Gd}] = 4 \times 0.5 = 2\ \mathrm{s^{-1}}$.
    (3) Add the rates: $\dfrac{1}{T_1} = 1 + 2 = 3\ \mathrm{s^{-1}}$.
    (4) Invert: $T_1 = \tfrac13\text{ s} \approx 333$ ms — down from 1000 ms.
    (5) From $M_z = M_0(1 - e^{-TR/T_1})$, a shorter $T_1$ means faster recovery, so at a fixed short TR the enhanced tissue has more magnetization and reads brighter. Because Gd only pools where the BBB is broken, only tumour lights up — that bright rim is the BraTS **enhancing tumour**.

!!! intuition "Intuition"
    Gadolinium does not itself glow — it is invisible. It works indirectly, by making the water around it relax faster. So "enhancement" is really a proxy map: *where did contrast leak out of the blood* → *where is the BBB broken* → *where is aggressive tumour*.

### Putting the four together: the nested BraTS regions

Each sequence contributes one channel of complementary evidence, and the tumour compartments are literally defined by *comparing* sequences. BraTS annotates three compartments — GD-enhancing tumour (**ET**), peritumoral edema (**ED**), and necrotic/non-enhancing core (**NCR/NET**) — then scores three **nested** regions [BraTS 2020 / UPenn CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html). The **enhancing tumour (ET)** is tissue brighter on T1Gd than on native T1; the **necrotic/non-enhancing core** is dark on T1Gd; and **peritumoral edema** is bright on FLAIR [TCIA BraTS-PEDs](https://www.cancerimagingarchive.net/collection/brats-peds/). Adding these up gives the scored regions: **tumour core (TC) = ET + necrotic core**, and **whole tumour (WT) = TC + edema**, so $\text{ET} \subseteq \text{TC} \subseteq \text{WT}$ (we return to these labels in detail in §7).

![Nested BraTS regions ET inside TC inside WT, each ring defined by which sequences it uses](figures/diagrams/s6_nested_regions.png)

| Tissue | T1 | T1Gd | T2 | FLAIR |
|---|---|---|---|---|
| CSF | dark | dark | bright | dark (nulled) |
| Fat | bright | bright | intermediate | bright |
| White matter | bright | bright | intermediate | intermediate |
| Edema | dark | dark | bright | bright |
| Enhancing tumour | dark | **bright** | bright | bright |
| Necrotic core | dark | dark | bright | intermediate |

!!! gotcha "Gotcha"
    The enhancing-tumour label needs *both* T1Gd **and** native T1: ET is signal that is brighter on T1Gd *relative to* T1. Something can look bright on T1Gd simply because it was already bright on T1 (fat, subacute blood, protein-rich fluid) — you must compare the pair, never read T1Gd alone [TCIA BraTS-PEDs](https://www.cancerimagingarchive.net/collection/brats-peds/). And enhancement is not the same as tumour: post-surgical change, inflammation, and normal structures also enhance, while low-grade and infiltrating tumour edges often do not. In post-treatment brains, radiation can cause treatment-related enhancement (**pseudoprogression**) that mimics tumour — a key reason the post-treatment segmentation of §5 is hard.

!!! example "Example question"
    Why provide four sequences and score three nested regions instead of segmenting on one "best" sequence?

    **Solution:** Because no single sequence contains all sub-regions, and the sub-regions are defined by *comparing* sequences. (1) ET is only definable by comparing T1Gd against native T1 — the tissue that brightens after gadolinium, marking a broken BBB and active tumour; neither image alone suffices [TCIA BraTS-PEDs](https://www.cancerimagingarchive.net/collection/brats-peds/). (2) The necrotic core stays dark on T1Gd; adding it to ET gives TC. (3) Edema is the bright FLAIR region; adding it to TC gives WT. So the scored regions nest ($\text{ET}\subseteq\text{TC}\subseteq\text{WT}$), each folding in one more sequence's evidence. Multi-parametric MRI is therefore not redundancy — each channel supplies information the others physically cannot [BraTS 2020 / UPenn CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html). This is exactly the four-channel input a segmentation network (§9–§11) receives.

!!! gotcha "Gotcha"
    The $TI = 0.693\,T_1$ formula assumes full recovery between shots (long TR); real protocols use a shorter TI, so do not expect it to match a scanner printout. FLAIR also nulls CSF only when the fluid truly has CSF-like $T_1$ — blood, high-protein fluid, or supplemental oxygen can leave "bright CSF" artefacts. And note the label *integers* are version-dependent (ET=4 in BraTS 2020, renumbered in later editions); the concept, not the code, is what matters (§7).
