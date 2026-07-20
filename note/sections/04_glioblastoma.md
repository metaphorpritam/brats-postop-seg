## 4. Glioblastoma and its standard treatment {#glioblastoma}

In §3 we placed **gliomas** — tumours arising from the brain's glial (support) cells — on the WHO grading scale from 1 (slow) to 4 (aggressive). This section zooms in on the grade-4 endpoint of that scale, **glioblastoma**, and on the standardised way it is treated. Understanding this treatment is not a detour: the whole reason the BraTS challenge segments a *post-treatment* brain (§5) is that surgery, radiation, and chemotherapy each leave distinctive marks on the MRI (§6) that our model must learn to tell apart from live tumour.

### What "glioblastoma" means today

**Glioblastoma (GBM)** is the most common and most aggressive primary malignant brain tumour in adults. It accounts for roughly 13.7% of all brain tumours and about 52% of the malignant ones, with an incidence near 3.2 per 100,000 people per year — around 14,000 new US cases annually — and it is more common in men and rises steeply with age, with a median age at diagnosis near 65 [CBTRUS Statistical Report (2018–2022), Neuro-Oncology](https://academic.oup.com/neuro-oncology/article/27/Supplement_4/iv1/8285946). "Primary" means it starts in the brain rather than spreading there from elsewhere; "malignant" means it grows invasively.

The precise definition matters for reading the literature. Under the **WHO 2021 CNS classification** (nicknamed **CNS5**), glioblastoma is defined as an *IDH-wildtype*, *H3-wildtype* diffuse astrocytic glioma of CNS grade 4, identified by hallmark aggressive features — **microvascular proliferation** (chaotic new blood-vessel growth) or **necrosis** (dead tissue at the tumour's core) — or by molecular markers such as a TERT promoter mutation, EGFR amplification, or the +7/−10 chromosome copy-number change [WHO CNS5 2021 classification (Frontiers review)](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1200815/full). ("IDH" and "H3" name genes; "wildtype" means the normal, unmutated version.)

!!! gotcha "Gotcha"
    The WHO 2021 rules changed *what the word "glioblastoma" refers to*. Tumours once called "IDH-mutant glioblastoma" are now a separate, generally better-prognosis entity — **astrocytoma, IDH-mutant, grade 4** [WHO CNS5 2021 classification (Frontiers review)](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1200815/full). Many pre-2021 papers pool the two together, so old survival numbers are **not** directly comparable to today's IDH-wildtype-only cohorts. When you read a survival figure, always ask which definition it used.

### The Stupp protocol: three sequential pillars

The first-line treatment has been essentially unchanged since 2005 and is called the **Stupp protocol**, after Roger Stupp, who led the landmark EORTC-NCIC trial. Its standard-of-care sequence is: **maximal safe surgical resection**, then **fractionated radiotherapy with concurrent daily temozolomide**, then **six or more monthly adjuvant temozolomide cycles** [StatPearls — Current Standards of Care in Glioblastoma Therapy](https://www.ncbi.nlm.nih.gov/books/NBK469987/).

![Three-stage Stupp protocol timeline: surgery, then concurrent chemoradiation, then adjuvant temozolomide](figures/diagrams/s4_stupp_timeline.png)

**Pillar 1 — maximal safe resection.** A neurosurgeon removes as much tumour as can be taken without causing unacceptable neurological harm. The word "safe" is the binding constraint: GBM diffusely infiltrates functioning brain and can never be fully excised, so the surgeon must stop short of damaging speech, movement, or vision. Even so, removing more helps — complete resection of the contrast-enhancing tumour gave a median survival near 15.2 months versus about 9.8 months for incomplete resection, and gross total resection can reach roughly 20–25 months in favourable patients [NCI/PMC multidisciplinary management review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11719842/). The cavity left behind is exactly the **resection cavity (RC)** label that the BraTS 2024 post-treatment challenge introduced (see §7).

!!! intuition "Intuition"
    "Maximal *safe* resection" is a tug-of-war, not a goal of total removal. GBM cells scatter microscopically far beyond what the MRI shows, so the surgeon can never cut it all out. They take the bulk while protecting the patient's abilities — more removal buys survival, but only up to the point where taking more would disable the person.

**Pillar 2 — radiotherapy.** External-beam radiation is delivered as **60 Gy** (gray, the unit of absorbed radiation dose) split into **30 daily fractions of 2 Gy** over about six weeks, for patients under roughly 70 with good performance status [StatPearls — Current Standards of Care in Glioblastoma Therapy](https://www.ncbi.nlm.nih.gov/books/NBK469987/). Why split one dose into thirty small ones? Because of a biological asymmetry we can quantify (below).

**Pillar 3 — temozolomide (TMZ).** TMZ is an oral **alkylating** chemotherapy: it attaches methyl groups to tumour DNA, damaging it. It is given in two phases — a **concurrent** phase of 75 mg/m² per day throughout the six weeks of radiotherapy, then, after a break, an **adjuvant** (maintenance) phase of 150–200 mg/m² per day on days 1–5 of each 28-day cycle for six cycles [StatPearls — Current Standards of Care in Glioblastoma Therapy](https://www.ncbi.nlm.nih.gov/books/NBK469987/). The "mg/m²" unit means the dose is scaled to the patient's body size, which we compute in the third worked example.

### Why fractionate? The biologically effective dose

To compare radiation schedules fairly we use the **linear-quadratic biologically effective dose (BED)**:

$$\text{BED} = n\,d\left(1 + \frac{d}{\alpha/\beta}\right)$$

Here $n$ is the number of fractions, $d$ is the dose per fraction in Gy, so $n\,d$ is the total physical dose; and $\alpha/\beta$ is a tissue-specific ratio (in Gy) describing radiosensitivity — about 10 Gy for rapidly proliferating tumour and about 3 Gy for late-responding normal brain. BED restates a schedule's biological potency in a common currency. Let us derive the BED of the Stupp schedule for both tissues, step by step.

$$\text{BED} = n\,d\left(1 + \frac{d}{\alpha/\beta}\right)$$

We start from the formula. The physical dose $n\,d$ is scaled up by a factor that accounts for the extra damage done by each fraction's size $d$.

$$n = 30,\quad d = 2\ \text{Gy}\ \Rightarrow\ n\,d = 30 \times 2 = 60\ \text{Gy}$$

Substituting the Stupp schedule, thirty daily 2 Gy fractions give the familiar 60 Gy total.

$$\text{Tumour: } \alpha/\beta = 10\ \text{Gy}\ \Rightarrow\ \frac{d}{\alpha/\beta} = \frac{2}{10} = 0.2$$

For rapidly dividing tumour cells $\alpha/\beta \approx 10$ Gy, so the per-fraction correction term is 0.2.

$$\text{BED}_{\text{tumour}} = 60\,(1 + 0.2) = 60 \times 1.2 = 72\ \text{Gy}_{10}$$

Multiplying, the schedule delivers a tumour BED of 72 Gy (the subscript 10 records which $\alpha/\beta$ was used).

$$\text{Late normal brain: } \alpha/\beta = 3\ \text{Gy}\ \Rightarrow\ \frac{d}{\alpha/\beta} = \frac{2}{3} \approx 0.667$$

Late-responding normal tissue has a lower $\alpha/\beta$ (~3 Gy), which makes it relatively *more* sensitive to the size of each fraction.

$$\text{BED}_{\text{brain}} = 60\,(1 + 0.667) = 60 \times 1.667 = 100\ \text{Gy}_{3}$$

Multiplying, normal brain sees a BED of about 100 Gy₃. Because normal tissue accumulates more biological dose *per fraction* than tumour does, keeping each fraction small (2 Gy) is what spares the brain — this is the entire rationale for fractionating rather than delivering 60 Gy in one blast.

!!! intuition "Intuition"
    Splitting 60 Gy into thirty small daily doses exploits the asymmetry the BED just made concrete: healthy brain is punished more by large fractions, so small fractions let it repair overnight while the tumour still accumulates a lethal dose (72 Gy₁₀). The fraction size, not just the total, decides who gets hurt.

### How well does it work? From hazard ratio to survival

Adding temozolomide to radiotherapy produced a real, reproducible gain. In the pivotal EORTC-NCIC trial, median overall survival rose from **12.1 to 14.6 months** with a **hazard ratio (HR) of 0.6** ($p < 0.001$) [StatPearls; EORTC-NCIC 5-year analysis (Stupp et al., Lancet Oncol 2009)](https://pubmed.ncbi.nlm.nih.gov/19269895/). A **hazard ratio** compares the instantaneous risk of death between two arms; HR = 0.6 means the treatment arm has a 40% lower moment-to-moment risk. To connect that ratio to a median survival, we use a simple constant-hazard model.

The **exponential survival function** is $S(t) = e^{-\lambda t}$, where $S(t)$ is the probability a patient is still alive at time $t$ and $\lambda$ is the (assumed constant) **hazard rate**, the risk of death per unit time. Its median is derived as follows.

$$S(t) = e^{-\lambda t}$$

We assume each arm's survival decays exponentially at its own death rate $\lambda$.

$$0.5 = e^{-\lambda\, t_{\text{med}}}$$

The **median survival** $t_{\text{med}}$ is by definition the time when half the patients remain, i.e. $S = 0.5$.

$$\ln(0.5) = -\lambda\, t_{\text{med}} \ \Rightarrow\ -\ln 2 = -\lambda\, t_{\text{med}}$$

Take the natural log of both sides; $\ln(0.5) = -\ln 2$.

$$t_{\text{med}} = \frac{\ln 2}{\lambda}$$

Solving for the median: median survival is inversely proportional to the hazard rate — double the hazard, halve the median. Now form the ratio of the two arms' medians.

$$\frac{t_{\text{med,treat}}}{t_{\text{med,control}}} = \frac{\ln 2/\lambda_{\text{treat}}}{\ln 2/\lambda_{\text{control}}} = \frac{\lambda_{\text{control}}}{\lambda_{\text{treat}}} = \frac{1}{\text{HR}}$$

The $\ln 2$ cancels, leaving the inverse of the hazard ratio (since $\text{HR} = \lambda_{\text{treat}}/\lambda_{\text{control}}$).

$$\frac{1}{\text{HR}} = \frac{1}{0.6} \approx 1.667$$

With HR = 0.6 the model predicts the treatment median should be about 1.67× the control median.

$$t_{\text{med,treat}} \approx 1.667 \times 12.1 = 20.2\ \text{months}$$

Applying this to the control median of 12.1 months gives an idealised prediction of about 20 months — yet the trial actually reported **14.6 months**, not 20.2. The gap is instructive: the constant-hazard assumption is only an approximation. The survival benefit is concentrated in a favourable minority (largely MGMT-methylated patients, below) who form a long tail of the survival curve, so a single median understates where the gain really lives.

!!! gotcha "Gotcha"
    Converting a hazard ratio into a median-survival ratio assumes **proportional hazards** — a constant HR over time. Glioblastoma violates this: the benefit is front-loaded and concentrated in favourable subgroups, so the naive prediction (~20 months) overshoots the observed median (14.6 months). Never quote a single derived number as if the model were exact.

The tails tell a more dramatic story than the median. Five-year overall survival was **9.8%** with radiotherapy + temozolomide versus **1.9%** with radiotherapy alone; at two years it was 27.2% vs 10.9%, and at three years 16.0% vs 4.4% [EORTC-NCIC 5-year analysis (Stupp et al., Lancet Oncol 2009)](https://pubmed.ncbi.nlm.nih.gov/19269895/).

| Time point | RT + temozolomide | RT alone |
|---|---|---|
| Median OS | 14.6 mo | 12.1 mo |
| 2-year survival | 27.2% | 10.9% |
| 3-year survival | 16.0% | 4.4% |
| 5-year survival | 9.8% | 1.9% |

### The MGMT switch: who benefits most

The single strongest predictor of temozolomide benefit is **MGMT promoter methylation**. MGMT is a DNA-repair enzyme whose job is to erase exactly the methyl damage temozolomide inflicts. When the MGMT gene's promoter is silenced by methylation, the tumour makes little MGMT, cannot undo the damage, and responds far better — roughly **46% two-year survival for methylated tumours versus about 14% for unmethylated** tumours treated with radiation + temozolomide [EORTC 22981 / Hegi MGMT analysis; EORTC-NCIC 5-year analysis](https://pubmed.ncbi.nlm.nih.gov/19269895/).

![Temozolomide methylates tumour DNA; MGMT erases the damage unless its gene is silenced](figures/diagrams/s4_mgmt_switch.png)

!!! intuition "Intuition"
    Temozolomide graffiti-tags tumour DNA with methyl groups; MGMT is the cleaning crew that scrubs them off. Silence the MGMT gene by methylation and the tumour loses its eraser, so the drug's damage sticks. It is a rare case where one molecular switch cleanly predicts who benefits from a drug.

Beyond the core Stupp backbone, **Tumor Treating Fields** (Optune) — wearable scalp electrodes delivering alternating electric fields — added about 4.8 months of median overall survival when combined with maintenance temozolomide in the randomized EF-14 trial [Novocure EF-14 final analysis (JAMA 2017 summary)](https://www.novocure.com/jama-publishes-final-analysis-of-ef-14-phase-3-pivotal-trial-of-optune-together-with-temozolomide-demonstrating-unprecedented-survival-results-for-newly-diagnosed-glioblastoma/). Even so, recurrence is nearly universal and GBM remains incurable — which is precisely why reproducible measurement of tumour and resection-cavity volumes on serial MRI, the goal of the BraTS post-treatment challenge (§8, §19), matters clinically.

### Dosing temozolomide to the patient

Chemotherapy doses in mg/m² are scaled to **body surface area (BSA)**, estimated with the **Mosteller formula**:

$$\text{BSA}\ (\text{m}^2) = \sqrt{\frac{H \cdot W}{3600}}$$

where $H$ is height in cm, $W$ is weight in kg, and 3600 is a normalizing constant. Multiplying BSA by the mg/m² dose gives the dose in milligrams. Worked step by step for a sample patient:

$$\text{BSA} = \sqrt{\frac{H \cdot W}{3600}}$$

We need BSA before converting a mg/m² dose into milligrams.

$$H = 170\ \text{cm},\quad W = 70\ \text{kg}\ \Rightarrow\ H\cdot W = 170 \times 70 = 11900$$

Substitute the patient's height and weight and multiply.

$$\frac{11900}{3600} = 3.306$$

Divide by the constant 3600.

$$\text{BSA} = \sqrt{3.306} = 1.818\ \text{m}^2$$

Take the square root to get the body surface area.

$$\text{Concurrent dose} = 75\ \tfrac{\text{mg}}{\text{m}^2} \times 1.818\ \text{m}^2 = 136.4\ \text{mg/day}$$

During radiotherapy TMZ is 75 mg/m² per day; multiplying by BSA gives about 136 mg/day, rounded to a practical capsule combination taken every day of the radiotherapy course.

$$\text{Adjuvant dose (150 mg/m}^2) = 150 \times 1.818 = 272.7\ \text{mg/day on days 1–5}$$

In the maintenance phase the dose per m² rises to 150 mg/m² (often escalated to 200 mg/m² if tolerated), so the same patient takes about 273 mg/day for days 1–5 of each 28-day cycle, then rests.

!!! gotcha "Gotcha"
    "Six adjuvant cycles" is the trial-defined minimum, not a biological cutoff, and the 150→200 mg/m² escalation is contingent on blood counts. Temozolomide causes **myelosuppression** (notably thrombocytopenia, low platelets), so real-world dosing is frequently delayed or reduced — the textbook mg/m² numbers are targets, not guarantees.

!!! example "Example question"
    A newly diagnosed glioblastoma patient completes gross total resection and starts the Stupp protocol. State the radiotherapy schedule, compute its biologically effective dose to the tumour ($\alpha/\beta = 10$ Gy), and give the concurrent temozolomide dose for a patient with BSA = 1.9 m².

    **Solution:** Radiotherapy schedule: 60 Gy total, given as 30 daily fractions of 2 Gy over ~6 weeks. BED to tumour: $\text{BED} = n\,d\,(1 + d/(\alpha/\beta)) = 30 \times 2 \times (1 + 2/10) = 60 \times 1.2 = 72\ \text{Gy}_{10}$. Concurrent temozolomide: $75\ \text{mg/m}^2 \times 1.9\ \text{m}^2 = 142.5\ \text{mg/day}$, taken every day throughout radiotherapy. After a ~4-week break the patient begins six adjuvant cycles at 150–200 mg/m² on days 1–5 of each 28-day cycle (i.e. ~285–380 mg/day for this patient) [StatPearls — Current Standards of Care in Glioblastoma Therapy](https://www.ncbi.nlm.nih.gov/books/NBK469987/).

!!! example "Example question"
    In the EORTC-NCIC trial, 5-year overall survival was 9.8% with radiotherapy + temozolomide vs 1.9% with radiotherapy alone. Express the added long-term benefit as an absolute difference and as a relative (fold) increase, and explain why this looks larger than the median-survival gain of only ~2.5 months.

    **Solution:** Absolute difference at 5 years $= 9.8\% - 1.9\% = 7.9$ percentage points. Relative (fold) increase $= 9.8 / 1.9 \approx 5.2\times$ — temozolomide roughly quintupled the chance of being alive at five years. This dwarfs the median gain (14.6 vs 12.1 months, ~2.5 months) because the benefit is not spread evenly: it concentrates in a favourable long-surviving subgroup (largely MGMT-methylated patients). The median measures the "typical" patient near the middle of the curve, where the two arms are close; the tail measures the minority who benefit most, where the arms diverge sharply. Both statistics are correct — they describe different parts of the same survival curve [EORTC-NCIC 5-year analysis (Stupp et al., Lancet Oncol 2009)](https://pubmed.ncbi.nlm.nih.gov/19269895/).

Each pillar of this pathway leaves a fingerprint on the post-treatment brain — a fluid-filled cavity from surgery, radiation-related tissue changes, and residual or recurrent enhancing tumour. Untangling those fingerprints is the subject of §5, and mapping them onto the four BraTS tumour labels (NETC, SNFH, ET, RC) is the subject of §7.
