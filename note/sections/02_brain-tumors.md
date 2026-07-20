## 2. Brain tumours: cells in a closed box {#brain-tumors}

Before we can teach a computer to outline a tumour on a brain scan (§9) we have to know what the thing being outlined actually *is*. A **brain tumour** is an abnormal mass of cells growing inside the skull or spinal canal — the **central nervous system (CNS)**, meaning the brain and spinal cord. The single most important fact about it is geometric: the skull is a rigid, closed box.

### The closed box

The skull does not stretch. Inside it three things — brain tissue, blood, and **cerebrospinal fluid** (CSF, the clear fluid that cushions the brain) — together fill a fixed volume. Add a tumour, plus the **edema** (swelling of surrounding tissue) it provokes, and something has to give: pressure rises and healthy brain gets squeezed. This one fact explains why symptoms appear. As the tumour and its swelling press on or damage healthy tissue, patients develop morning headaches, seizures, nausea and vomiting, vision or speech problems, personality or mood changes, weakness, and loss of balance [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq). In glioma case series, headache is reported in roughly 56% of patients, seizures in about 32%, cognitive problems in about 34%, and nausea or vomiting in about 13% [MSKCC](https://www.mskcc.org/cancer-care/types/glioma/glioma-signs-and-symptoms).

![Closed-skull schematic: a tumour plus surrounding edema displaces brain tissue, raising intracranial pressure, which produces headache, nausea, focal deficits and seizures](figures/diagrams/s2_closed_box.png)

!!! intuition "Intuition"
    The skull is a sealed jar. Whatever you add inside pushes out something else. A tumour that would be trivial in soft, expandable tissue becomes dangerous in the brain purely because there is **nowhere to go**. Headaches are classically worse in the morning because pressure builds while lying flat overnight — a direct read-out of the closed-box physics.

### Two independent axes

Tumours are classified along **two separate questions at once** — do not collapse them into one label.

The first axis is **origin**. A **primary** tumour starts in brain or CNS tissue itself (for example a **glioma**, from the glial support cells covered in §3, or a **meningioma**, from the membranes wrapping the brain). It may spread within the brain or to the spine but rarely spreads to the rest of the body. A **metastatic** (secondary) tumour is a cancer that began elsewhere — most often lung, breast, or melanoma — and travelled to the brain through the bloodstream [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq).

The second axis is **behaviour**. **Benign** tumours grow slowly, rarely invade neighbouring tissue, and may recur; **malignant** tumours grow quickly and infiltrate healthy brain [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq). To grade this behaviour finely, pathologists assign a **WHO grade from 1 to 4**: grade 1 cells look near-normal and grow slowly, while grade 4 cells look nothing like normal cells and grow and spread very quickly [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq). Since 2021, the WHO CNS5 classification also uses **molecular markers** (such as *IDH* mutation status) alongside microscope appearance — which is why "glioblastoma" now specifically means an IDH-wildtype, grade-4 diffuse glioma [RSNA](https://pubs.rsna.org/doi/full/10.1148/rg.210236). We unpack grading and this vocabulary in §3 and §4.

![Two independent classification axes: origin (primary vs metastatic) on one line, behaviour (benign/low-grade vs malignant/high-grade) on another](figures/diagrams/s2_two_axes.png)

!!! gotcha "Gotcha"
    "Benign" does **not** mean "harmless" in the brain. A slow-growing, non-malignant tumour that blocks CSF drainage or presses on the brainstem can be immediately life-threatening — location and mass effect matter as much as the cell type [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq). Grade measures cellular aggressiveness, not clinical danger.

### How common, and how deadly

Primary CNS tumours are relatively uncommon. In the US (CBTRUS 2018–2022) the average annual **age-adjusted incidence rate** is about **26.05 per 100,000** people for all primary CNS tumours combined — split into 6.86 malignant and 19.19 non-malignant per 100,000 [CBTRUS](https://cbtrus.org/cbtrus-fact-sheet/). Five-year relative survival is 91.7% for non-malignant tumours but only 34.8% for malignant ones, and glioblastoma — the most common malignant primary tumour — has a five-year relative survival of only about 7% [NBTS](https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-facts/). Metastatic tumours are actually more common than primaries — roughly 5× more frequent — simply because so many bodily cancers can seed the brain [NBTS](https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-facts/).

What is an "**age-adjusted incidence rate**"? Because brain-tumour risk rises steeply with age, we standardise to a fixed age structure so different populations can be compared fairly:

$$\text{ASR} = \frac{\sum_{i} r_i \, w_i}{\sum_{i} w_i}, \qquad r_i = \frac{c_i}{n_i}$$

Here $i$ indexes age groups (0–4, 5–9, …); $r_i$ is the age-specific rate in group $i$, equal to $c_i$ (new cases in that group) divided by $n_i$ (people at risk in that group); and $w_i$ is the weight for that age group taken from a fixed "standard" population. The numerator sums the weighted rates and the denominator sums the weights, so the ASR is a weighted average of age-specific rates.

### Derivation: from a rate to a headcount

A rate per 100,000 becomes a real number of people once we know the population $N$. Start from the definition of an incidence rate:

$$\text{rate} = \frac{C}{N} \times 100{,}000$$

where $C$ is the unknown number of new cases per year and $N$ is the population at risk; the factor 100,000 is just the reporting unit. Substitute the malignant rate 6.86 per 100,000:

$$6.86 = \frac{C}{N} \times 100{,}000$$

Divide both sides by 100,000 to isolate the plain per-person fraction:

$$\frac{6.86}{100{,}000} = \frac{C}{N}$$

Multiply both sides by $N$ to get $C$ alone:

$$C = \frac{6.86}{100{,}000} \times N = 6.86 \times \frac{N}{100{,}000}$$

The term $N/100{,}000$ counts how many 100,000-person blocks the population contains. Insert an approximate US population $N = 335{,}000{,}000$:

$$C = 6.86 \times \frac{335{,}000{,}000}{100{,}000} = 6.86 \times 3{,}350 = 22{,}981$$

So the rate implies roughly **23,000** new malignant primary CNS tumours per year — the right order of magnitude for the US, with the small gap coming from age-standardisation and the exact population used.

!!! example "Example question"
    A country of 50 million people reports an age-adjusted malignant brain-tumour incidence of 6.9 per 100,000 per year. About how many new cases occur per year, and how many of one year's patients would you expect alive at 5 years if relative survival is 34.8% and a matched cancer-free group has 95% five-year survival?

    **Solution.**
    Step 1 — annual cases: $C = \text{rate} \times \dfrac{N}{100{,}000} = 6.9 \times \dfrac{50{,}000{,}000}{100{,}000} = 6.9 \times 500 = 3{,}450$ new cases per year.
    Step 2 — turn *relative* survival into *observed* survival. Relative survival divides out background deaths, $\text{RSR} = S_{\text{obs}}/S_{\text{exp}}$, so $S_{\text{obs}} = \text{RSR} \times S_{\text{exp}} = 0.348 \times 0.95 = 0.3306 \approx 33.1\%$, where $S_{\text{obs}}$ is the fraction of patients actually alive at 5 years and $S_{\text{exp}}$ is the fraction a matched cancer-free group would have alive.
    Step 3 — survivors: $0.331 \times 3{,}450 \approx 1{,}142$ patients. So about 3,450 new cases per year, of whom roughly 1,140 (about 33%) are alive at 5 years; the remaining ~2,300 show the heavy toll.

!!! example "Example question"
    In a clinic, for every primary brain tumour there are about 5 metastatic ones. If 600 brain tumours are seen in a year, how many are metastatic, and what is the metastatic share?

    **Solution.** Let $P$ be primaries and $M$ metastases with $M = 5P$. Total $= M + P = 6P = 600$, so $P = 100$ and $M = 5 \times 100 = 500$. The metastatic share is $f = \dfrac{M}{M+P} = \dfrac{500}{600} = \dfrac{5}{6} \approx 0.833 = 83.3\%$. Notice the population size cancels: $\dfrac{5P}{5P+P} = \dfrac{5}{6}$ regardless of $P$. So metastases dominate adult intracranial tumours even though primaries get more research attention [NBTS](https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-facts/).

### Why brain tumours are so serious

Three forces compound. The affected organ tolerates little pressure or damage; malignant gliomas diffusely infiltrate healthy brain and cannot be fully removed by surgery; and the **blood–brain barrier** (the tight lining of brain blood vessels) limits which drugs reach the tumour [NCI](https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq). This is exactly why aggressive gliomas remain among the hardest cancers to treat — and why accurately measuring the tumour on MRI (§6, §7) matters clinically (§8). With the disease in view, §3 zooms into the specific family this project targets: gliomas and their WHO grading.
