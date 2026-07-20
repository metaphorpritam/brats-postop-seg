## 5. The post-treatment brain: why the scan changes {#post-treatment}

In §4 we followed a glioblastoma from diagnosis into treatment. This section is about what the brain — and its MRI scan — looks like *afterwards*, and why that "after" picture is so much harder to read than the "before". This matters directly to our project: the BraTS challenge we are solving is a **post-treatment** segmentation task, so almost every voxel we must label sits in tissue that treatment has rewritten.

### What treatment does to the brain

Standard care for glioblastoma is the **Stupp protocol**: maximal safe surgical removal of the tumour (**resection**), followed by **radiotherapy** (targeted radiation) given together with **temozolomide** (an oral chemotherapy drug), and then more temozolomide. Each of these three steps leaves a permanent fingerprint on later scans — a **resection cavity**, blood products, and radiation-induced inflammation and scarring (**gliosis**) — and it is exactly these fingerprints that make interpretation hard [BraTS 2024](https://arxiv.org/html/2405.18368v1).

- **Resection cavity (RC).** Where the surgeon removed tissue, a hole is left that fills with fluid, blood, air, and protein-rich material [BraTS 2024](https://arxiv.org/html/2405.18368v1). It sits precisely where the tumour was — and where any regrowth will appear.
- **Treatment effect.** Radiation and chemotherapy injure normal tissue and the **blood–brain barrier** (the tight lining that normally keeps blood and brain separate), producing inflammation, swelling (**edema**), and gliosis.

![Three treatments each overwrite the brain, and the post-treatment scan must be read through all the layers at once.](figures/diagrams/s5_treatment_layers.png)

!!! intuition "Intuition"
    Contrast enhancement is a *leaky-plumbing* signal, not a *cancer* signal. Radiologists inject **gadolinium**, a contrast agent that stays inside intact blood vessels but leaks out wherever the blood–brain barrier is broken. That leak happens with viable tumour — but *also* with fresh surgery, radiation injury, and inflammation [AJNR RANO](https://pmc.ncbi.nlm.nih.gov/articles/PMC6975322/). One fact, three consequences: treatment can make the scan light up *without* tumour, drugs can switch the light *off* without curing anything, and no honest reader ever trusts enhancement on its own.

### Two mimics: pseudoprogression vs. radiation necrosis

The central problem of post-treatment imaging is that treatment changes can look almost identical to the tumour coming back. Two mimics dominate.

**Pseudoprogression** is a transient, treatment-related increase in enhancement and edema that *mimics* tumour growth but then stabilises or resolves on its own, with no change in therapy — it is not real growth. It typically appears within the first 3 months (up to 6 months) after chemoradiation [J Neurosurg 2023](https://pubmed.ncbi.nlm.nih.gov/36790010/). It is common: roughly 30% of patients after combined chemoradiation versus about 15% after radiation alone, and in nearly 60% of cases it shows up within the first 3 months [Zikou 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6305027/). It is more frequent in tumours with **MGMT-promoter methylation** and **IDH mutation** (genetic features from §3), and — reassuringly — it is generally linked to *better* survival [Zikou 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6305027/).

**Radiation necrosis** is a later (months to years), more destructive form of treatment-related tissue death. On conventional MRI both radiation necrosis and true progression enlarge and enhance inside the radiation field, so they are frequently indistinguishable without advanced imaging such as perfusion, spectroscopy, or PET [Verma 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4139817/).

!!! gotcha "Gotcha"
    Pseudoprogression and radiation necrosis are *both* "treatment effect", but they are not the same thing. Pseudoprogression is **early** (typically under 3–6 months), often self-resolving, and prognostically **favourable** [Zikou 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6305027/); radiation necrosis is **late** and destructive [Verma 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4139817/). Lumping either one in with "progression" leads to wrong treatment decisions and wrong clinical-trial endpoints.

### Making the call reproducible: RANO

To judge whether a tumour is responding or progressing in the same way across hospitals and trials, neuro-oncologists use the **RANO criteria** (Response Assessment in Neuro-Oncology), updated to **RANO 2.0** in 2023. RANO measures each enhancing target lesion **bidimensionally** — two perpendicular diameters — and requires each to be at least 10 mm on at least two slices to count as measurable [AJNR RANO](https://pmc.ncbi.nlm.nih.gov/articles/PMC6975322/).

For one lesion $i$, the **product of perpendicular diameters** is

$$P_i = a_i \times b_i,$$

where $a_i$ is the longest enhancing diameter (mm) and $b_i$ is the diameter perpendicular to it on the same slice (mm). $P_i$ is a stand-in for the lesion's area (in mm²). Summing across all $n$ target lesions gives the **total tumour burden**

$$S = \sum_{i=1}^{n} P_i = \sum_{i=1}^{n} a_i b_i.$$

Response is judged from how $S$ changes. The **fractional change** at a follow-up timepoint $t$ is

$$\Delta = \frac{S_t - S_{\text{ref}}}{S_{\text{ref}}},$$

where $S_t$ is the current burden and $S_{\text{ref}}$ is the reference — the baseline $S_0$ when checking for shrinkage, or the smallest value seen so far (the **nadir**) when checking for growth. A negative $\Delta$ means shrinkage, positive means growth. The category follows fixed thresholds [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/):

$$\text{Category} = \begin{cases} \text{CR} & S_t = 0 \\ \text{PR} & \Delta \le -0.50 \\ \text{PD} & \Delta \ge +0.25 \ \text{or new lesion} \\ \text{SD} & \text{otherwise} \end{cases}$$

Here **CR** (complete response) means all enhancing disease has vanished, **PR** (partial response) a decrease of at least 50%, **PD** (progressive disease) an increase of at least 25% (or a new lesion), and **SD** (stable disease) anything in between — all subject to extra conditions on steroid dose and clinical status.

#### Worked example: from rulers to a category

Take a post-radiotherapy baseline scan with two target lesions: lesion 1 is $30 \times 20$ mm, lesion 2 is $18 \times 12$ mm (all four diameters $\ge 10$ mm, so both qualify).

$$\begin{aligned} P_1 &= 30 \times 20 = 600\ \text{mm}^2, \\ P_2 &= 18 \times 12 = 216\ \text{mm}^2, \\ S_0 &= P_1 + P_2 = 600 + 216 = 816\ \text{mm}^2. \end{aligned}$$

Three months later the *same* lesions measure $24 \times 15$ mm and $14 \times 10$ mm:

$$\begin{aligned} P_1' &= 24 \times 15 = 360\ \text{mm}^2, \\ P_2' &= 14 \times 10 = 140\ \text{mm}^2, \\ S_t &= 360 + 140 = 500\ \text{mm}^2. \end{aligned}$$

The fractional change against baseline is

$$\Delta = \frac{S_t - S_0}{S_0} = \frac{500 - 816}{816} = \frac{-316}{816} = -0.387,$$

a 38.7% decrease. Partial Response needs $\Delta \le -0.50$; this is a real shrinkage but not a big enough one, and it is nowhere near the $+0.25$ progression line — so the imaging category is **Stable Disease**. (Even this is provisional: RANO also demands stable-or-improved edema on the same-or-lower steroid dose and no clinical decline [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/).)

#### Why 2-D thresholds map onto 3-D volumes

RANO 2.0 also allows **volumetric** measurement — exactly what an automated segmenter produces — with cutoffs of about a 65% volume decrease for PR and a 40% volume increase for PD [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/). Where do those numbers come from? Model the lesion as roughly spherical with characteristic diameter $d$. Its bidimensional product behaves like an area and its volume like, well, a volume:

$$P \propto d^2, \qquad V \propto d^3.$$

Solve the first for $d$: $d \propto P^{1/2}$. Substitute into the second:

$$V \propto d^3 \propto \left(P^{1/2}\right)^3 = P^{3/2}.$$

So between two timepoints,

$$\frac{V_t}{V_0} = \left(\frac{P_t}{P_0}\right)^{3/2}.$$

For **PR**, a 50% area decrease means $P_t/P_0 = 0.50$:

$$\frac{V_t}{V_0} = 0.50^{3/2} = 0.50 \times 0.50^{1/2} = 0.50 \times 0.7071 = 0.3536,$$

i.e. a $1 - 0.3536 = 0.646$, a ~65% volume decrease — matching the RANO 2.0 cutoff. For **PD**, a 25% area increase means $P_t/P_0 = 1.25$:

$$\frac{V_t}{V_0} = 1.25^{3/2} = 1.25 \times 1.1180 = 1.3975,$$

a ~40% volume increase. The 2-D and 3-D thresholds are the *same* biological change in different geometry — which is precisely why a good segmentation's volume can be fed straight into RANO.

![RANO turns a post-radiotherapy baseline into a response category, with a confirmation branch for the 12-week pseudoprogression window.](figures/diagrams/s5_rano_flow.png)

### Why post-treatment segmentation is hard

RANO 2.0 deliberately guards against the pseudoprogression trap: it uses the **post-radiotherapy** scan (about 21–35 days after radiation), not the messy immediate post-surgical scan, as the baseline, and it requires apparent early growth to be *confirmed* on a repeat scan within the 12-week window before progression is declared [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/). The reason our segmentation task is difficult is the same one that forces these safeguards: "the combination of treatment-related changes, including resection cavities, blood products, post-radiation inflammation, and gliosis, combined with the already naturally ill-defined tumour borders seen in infiltrative diffuse gliomas makes segmentation a challenging task" [BraTS 2024](https://arxiv.org/html/2405.18368v1).

Concretely, BraTS 2024 asks a model to label four post-treatment tissue classes — enhancing tissue (ET), non-enhancing tumour core (NETC), surrounding non-enhancing FLAIR hyperintensity (SNFH), and the newly required **resection cavity** (RC) [BraTS 2024](https://arxiv.org/html/2405.18368v1). We define these four labels carefully in §7 and the MRI sequences that reveal them in §6. The hardest voxels — the thin rim of enhancement around the cavity, or blood inside it that must *not* be called tumour — are exactly the clinically decisive ones for radiation planning and RANO. That is why BraTS scores with lesion-wise Dice and boundary error (HD95) rather than one global number; we build those metrics up in §13 and §16.

!!! example "Example question"
    A glioblastoma patient's post-radiotherapy baseline MRI shows a single enhancing lesion of $40 \times 30$ mm. Eight weeks later it measures $50 \times 36$ mm; the patient is on an unchanged steroid dose, is clinically stable, and the new enhancement lies entirely within the prior high-dose radiation field. Using RANO, what is the arithmetic category, and what is the correct clinical action?

    **Solution.** Baseline $S_0 = 40 \times 30 = 1200\ \text{mm}^2$. Follow-up $S_t = 50 \times 36 = 1800\ \text{mm}^2$. Then $\Delta = (1800 - 1200)/1200 = 600/1200 = +0.50$ — a 50% area increase, which is $\ge +0.25$ and so *arithmetically* meets Progressive Disease. **But** this scan is only 8 weeks after radiotherapy (inside the 12-week window), the growth is entirely inside the high-dose field, and the patient is stable — the classic pseudoprogression picture, expected in 30–40% of IDH-wildtype glioblastomas [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/). The correct action is *not* to declare progression: continue therapy and obtain a confirmatory scan. Only a sustained increase on repeat imaging, or new enhancement *outside* the field, would confirm true progression [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/).

!!! example "Example question"
    A segmentation model outputs an enhancing-tumour volume of 12.0 cm³ at baseline and 4.0 cm³ at follow-up. Convert this to a RANO response category using the volumetric thresholds, and cross-check with the equivalent 2-D threshold.

    **Solution.** Volume ratio $= 4.0 / 12.0 = 0.333$, a 66.7% volume decrease. RANO 2.0's volumetric PR cutoff is ~65% [RANO 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10860967/), so this qualifies as at least **Partial Response**. Cross-check with geometry: since $V \propto d^3$ and $P \propto d^2$, we have $P_t/P_0 = (V_t/V_0)^{2/3} = 0.333^{2/3}$. Compute $0.333^{1/3} = 0.693$, then square: $0.693^2 = 0.480$. So the equivalent 2-D product ratio is 0.48 — a 52% area decrease, which is $\ge 50\%$, confirming PR on the classic 2-D criterion. The two frameworks agree. (Caveat: still contingent on stable/lower steroids and no pseudoresponse from antiangiogenic drugs [AJNR RANO](https://pmc.ncbi.nlm.nih.gov/articles/PMC6975322/).)
