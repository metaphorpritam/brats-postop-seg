## 3. Gliomas and the WHO grading system {#gliomas-grading}

In §2 we met brain tumours in general. This section zooms in on the family that dominates this project. A **glioma** is a tumour that arises from **glial cells** — the brain's support cells that surround and service the neurons. Gliomas are the most common primary malignant brain tumours in adults, and for over a century they were named and ranked almost entirely by how their cells *looked* under a microscope [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/). In 2021 that changed. Because the label attached to a brain scan is ultimately a *diagnosis*, and the scans in a segmentation dataset (§7, §9) inherit that diagnosis, we need to understand how modern glioma classification works.

### 3.1 The 2021 rewrite: from ~15 names to 3 entities

The World Health Organization publishes the official rulebook for naming brain tumours. Its **5th edition** (2021), known as **WHO CNS5** (CNS = *central nervous system*), collapsed the roughly 15 adult tumour names of the 2016 edition into just **three adult-type diffuse glioma entities**, each pinned to a molecular marker [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/):

1. **Astrocytoma, IDH-mutant**
2. **Oligodendroglioma, IDH-mutant and 1p/19q-codeleted**
3. **Glioblastoma, IDH-wildtype**

Two words in that list carry all the weight, so let us define them.

- **Histology** is the microscope picture: how densely packed the cells are, whether they are dividing, whether blood vessels are proliferating, whether tissue is dying.
- A **molecular marker** is a specific change in the tumour's DNA — a mutated gene or a missing chunk of chromosome — read by a laboratory test, not the eye.

WHO CNS5 requires *both*. Diagnosis is now a **layered/integrated report** that states the integrated diagnosis, the histologic diagnosis, the grade, and the molecular findings together; molecular profiling is required, not optional, for adult diffuse gliomas [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).

!!! intuition "Intuition"
    Think of a CNS5 diagnosis as a **two-key lock**. Key one is the microscope (how the cells look); key two is the molecular test (which genes and chromosomes are altered). You need both keys, and — as we will see — the molecular key can *override* the visual one.

### 3.2 The master switch: IDH

The first fork in the road is the **IDH gene** (*isocitrate dehydrogenase*, versions IDH1 or IDH2). Whether IDH is **mutant** (altered) or **wildtype** (normal) is the top-level branch point [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/).

Why does one gene matter so much? A mutant IDH enzyme overproduces an **oncometabolite** — a cancer-driving small molecule — called **2-hydroxyglutarate**. This molecule reprograms how the cell chemically marks its own DNA and histones (the "packaging" proteins), producing a distinctive **glioma CpG-island methylator phenotype (G-CIMP)**. The upshot: IDH-mutant tumours behave as a slower-growing, better-prognosis family, while IDH-wildtype adult diffuse astrocytic tumours point toward glioblastoma, the most aggressive form [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/).

The second fork applies only if IDH is mutant. An **oligodendroglioma** requires an IDH mutation *and* a whole-arm **codeletion of chromosome arms 1p and 19q** (both missing together). That codeletion is the defining marker separating oligodendroglioma from IDH-mutant astrocytoma [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/).

![Decision tree for classifying an adult-type diffuse glioma, starting from IDH status](figures/diagrams/s3_decision_tree.png)

!!! intuition "Intuition"
    IDH is the **master switch**; 1p/19q is the tie-breaker inside the mutant branch. IDH-mutant + intact 1p/19q = **astrocytoma**. IDH-mutant + 1p/19q codeleted = **oligodendroglioma** (generally the most favourable diffuse glioma). IDH-wildtype (with the right features) = **glioblastoma**.

### 3.3 What "grade" means

Type answers *what* the tumour is; **grade** answers *how aggressive* it is expected to be. Grade runs from **CNS WHO grade 1** (least malignant — often circumscribed and potentially cured by surgical removal, e.g. pilocytic astrocytoma) to **grade 4** (most malignant) [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full). Grade is a *forecast of behaviour*, not a measure of size or location.

Two bookkeeping changes in CNS5 matter for anyone merging datasets (§19):

- Grades are now written with **Arabic numerals (1–4)**, replacing the old **Roman numerals (I–IV)**. The 2016 "WHO grade IV" is today's "CNS WHO grade 4" [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).
- Grading is now done **within a tumour type**, so an astrocytoma, IDH-mutant is assigned grade 2, 3, or 4, and the grade is always read *with* the type name [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).

The decisive novelty is that **molecular features can override the microscope**. For an IDH-mutant astrocytoma, a **homozygous deletion of CDKN2A and/or CDKN2B** (both copies of these tumour-suppressor genes lost) forces **CNS WHO grade 4**, irrespective of how benign the histology looks [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236). Symmetrically, an IDH-wildtype adult diffuse astrocytic tumour can be called **glioblastoma, grade 4** even without the classic microscopic hallmarks of **necrosis** (dead tissue) or **microvascular proliferation** (runaway blood-vessel growth) — provided it shows at least one of a **TERT promoter mutation**, **EGFR amplification**, or **combined chromosome-7 gain with chromosome-10 loss (+7/−10)** [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).

!!! gotcha "Gotcha"
    A tumour can be **grade 4 without the textbook microscopic signs**. Because CDKN2A/B (for IDH-mutant astrocytoma) or TERT / EGFR / +7−10 (for IDH-wildtype glioblastoma) can *force* grade 4, you cannot judge grade from histology or imaging alone [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236). And note there is **no grade 1** among adult-type diffuse gliomas — grade 1 tumours like pilocytic astrocytoma are separate, circumscribed, mostly paediatric types [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full). Grade does *not* run 1–4 within every glioma family.

### 3.4 Why one gene reshapes prognosis — a worked survival model

Grade and IDH status together drive **prognosis** (expected outcome). In one CNS5-era cohort, approximate **median overall survival** — the time by which half the patients have died — was about **12.6 months** for IDH-wildtype glioblastoma (grade 4), **26.4 months** for IDH-mutant astrocytoma grade 4, **53.6** and **55.4 months** for grades 3 and 2, and **56.5** (grade 2) and **45.8 months** (grade 3) for oligodendroglioma [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full).

To see *why* a hazard difference produces such different survival, we build a simple model from scratch. (This is a deliberate simplification for teaching; real survival is analysed with Kaplan–Meier/Cox methods where risk need not be constant.)

**Step 1 — the survival function.** Assume every patient faces a constant risk of death per unit time, called the **hazard rate** $\lambda$. Then the probability of still being alive at time $t$ is the **exponential survival function**

$$S(t) = e^{-\lambda t}.$$

Here $S(t)$ is the fraction surviving to time $t$; $\lambda$ (units: per month) is the constant hazard; $t$ is months since diagnosis. A larger $\lambda$ means the curve drops faster.

**Step 2 — define median survival.** The **median survival** $t_{1/2}$ is the time when exactly half remain, i.e. $S(t_{1/2}) = 0.5$. Substituting the model:

$$e^{-\lambda t_{1/2}} = 0.5.$$

**Step 3 — take the natural log of both sides.** The logarithm undoes the exponential, turning the equation linear:

$$-\lambda\, t_{1/2} = \ln(0.5).$$

**Step 4 — simplify the log.** Since $\ln(0.5) = -\ln 2$, the two minus signs cancel:

$$-\lambda\, t_{1/2} = -\ln 2 \quad\Longrightarrow\quad t_{1/2} = \frac{\ln 2}{\lambda}.$$

This is the median-survival formula: median time is $\ln 2 \approx 0.693$ divided by the hazard.

**Step 5 — apply to two groups.** For group $A$ (say IDH-wildtype glioblastoma) and group $B$ (IDH-mutant astrocytoma), the same formula gives $t_{1/2,A} = \frac{\ln 2}{\lambda_A}$ and $t_{1/2,B} = \frac{\ln 2}{\lambda_B}$, where $\lambda_A$ and $\lambda_B$ are their hazard rates.

**Step 6 — take the ratio.** Dividing one by the other, the common factor $\ln 2$ cancels:

$$\frac{t_{1/2,A}}{t_{1/2,B}} = \frac{\ln 2 / \lambda_A}{\ln 2 / \lambda_B} = \frac{\lambda_B}{\lambda_A}.$$

**Step 7 — introduce the hazard ratio.** The **hazard ratio** $\mathrm{HR} = \lambda_A / \lambda_B$ compares the two risks. Substituting,

$$\frac{t_{1/2,A}}{t_{1/2,B}} = \frac{1}{\mathrm{HR}}.$$

In words: **twice the hazard means half the median survival.**

**Step 8 — numeric check.** Take $B$ = IDH-mutant astrocytoma with $t_{1/2,B} = 52$ months, and suppose glioblastoma has $\mathrm{HR} = 4$ relative to it. Then

$$t_{1/2,A} = t_{1/2,B}\cdot \frac{1}{\mathrm{HR}} = 52 \times \frac{1}{4} = 13 \text{ months},$$

close to the observed glioblastoma median of about 12.6 months [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full). A single molecular fact — IDH status — reshuffles the diagnosis, the name, and, through the hazard, the survival curve.

!!! example "Example question"
    Using the constant-hazard model, if IDH-mutant astrocytoma has a median survival of 52 months and IDH-wildtype glioblastoma has a hazard ratio of 4 relative to it, what median does the model predict, and how does it compare with data?

    **Solution:** Use $\dfrac{t_{1/2,A}}{t_{1/2,B}} = \dfrac{1}{\mathrm{HR}}$ with $B$ = astrocytoma ($t_{1/2,B}=52$ months) and $A$ = glioblastoma ($\mathrm{HR}=4$).
    Step 1: rearrange to $t_{1/2,A} = t_{1/2,B}/\mathrm{HR}$.
    Step 2: substitute $t_{1/2,A} = 52/4 = 13$ months.
    Step 3: compare — 13 months closely matches the reported ~12.6-month glioblastoma median [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full). A 4-fold higher hazard gives roughly one-quarter the median survival, showing why the IDH split is prognostically decisive.

### 3.5 How much one test moves the diagnosis — a Bayesian example

Because IDH status is the pivotal first test, it is worth quantifying *how much* a single wildtype result shifts belief toward glioblastoma. We use **Bayes' theorem**, the rule for updating a probability after seeing evidence.

Let $G$ be the event "the glioma is glioblastoma" and $\neg G$ its opposite ("an IDH-mutant type"). Let $W$ be the evidence "the IDH test returns *wildtype*." Bayes' theorem reads

$$P(G \mid W) = \frac{P(W \mid G)\,P(G)}{P(W \mid G)\,P(G) + P(W \mid \neg G)\,P(\neg G)}.$$

Here $P(G)$ is the **prior** (belief before the test); $P(G \mid W)$ is the **posterior** (belief after seeing a wildtype result); $P(W \mid G)$ is the **likelihood** — how often a glioblastoma tests wildtype; $P(W \mid \neg G)$ is how often a non-glioblastoma tests wildtype. The denominator is the total probability of a wildtype result across both classes.

**Step 1 — priors.** Suppose before testing the odds are even: $P(G) = 0.5$ and $P(\neg G) = 1 - 0.5 = 0.5$ (base rates must sum to 1). *(These are illustrative teaching numbers.)*

**Step 2 — likelihoods from biology.** Nearly all glioblastomas are IDH-wildtype, so $P(W \mid G) = 0.98$. An IDH-mutant tumour rarely returns a wildtype call, so $P(W \mid \neg G) = 0.05$.

**Step 3 — numerator (glioblastoma-and-wildtype path).** Multiply likelihood by prior: $0.98 \times 0.5 = 0.49$.

**Step 4 — second denominator term (not-glioblastoma path).** Same operation: $0.05 \times 0.5 = 0.025$.

**Step 5 — full denominator.** Add the two paths: $0.49 + 0.025 = 0.515$. This is the total probability of seeing a wildtype result.

**Step 6 — divide.**

$$P(G \mid W) = \frac{0.49}{0.515} \approx 0.951.$$

The single wildtype result lifted the probability of glioblastoma from **0.50 to about 0.95** — which is exactly why IDH is the first test ordered. Note, though, that a lower-grade-*looking* IDH-wildtype tumour still needs the TERT / EGFR / +7−10 checks before the grade-4 glioblastoma label is confirmed [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).

!!! gotcha "Gotcha"
    Under CNS5, **"glioblastoma" is reserved for IDH-*wildtype* tumours only**. The old term "glioblastoma, IDH-mutant" no longer exists — that tumour is now **astrocytoma, IDH-mutant, CNS WHO grade 4**, a biologically and clinically distinct diagnosis [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full). Legacy datasets using the old name must be re-mapped before any comparison (see the old-vs-new mapping below and §19–§20). Beware too that "Grade III" (Roman) and "CNS WHO grade 3" (Arabic) are the same rank from different editions [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).

![How 2016 histology-only names collapse into the three CNS5 molecular entities](figures/diagrams/s3_old_vs_new.png)

### 3.6 Putting it together

!!! example "Example question"
    A 45-year-old has an adult-type diffuse glioma. Histology looks like a grade 2 astrocytoma, but molecular testing reports an **IDH1 mutation**, **intact 1p/19q**, and **homozygous CDKN2A/B deletion**. What is the full WHO CNS5 diagnosis, and why?

    **Solution — apply the two keys in order:**
    Step 1 (IDH): IDH1 is mutated, so this is *not* glioblastoma (that label is IDH-wildtype only); it belongs to the IDH-mutant family [[Frontiers]](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1131642/full).
    Step 2 (1p/19q): 1p/19q is intact, so it is an **astrocytoma**, not an oligodendroglioma [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216527/).
    Step 3 (grade): although the histology looks grade 2, **homozygous CDKN2A/B deletion defines CNS WHO grade 4 irrespective of histology** [[RadioGraphics]](https://pubs.rsna.org/doi/full/10.1148/rg.210236).
    **Answer:** *Astrocytoma, IDH-mutant, CNS WHO grade 4* — written as type + molecular status + grade. The molecular marker overrode the microscope.

The takeaway for the rest of this note: every scan in a glioma dataset carries a label that is itself an integrated molecular-plus-histologic verdict. Glioblastoma — the IDH-wildtype, grade-4 entity that dominates the BraTS cohort — is the subject of §4, and the regions its treatment leaves behind become the segmentation labels of §7.
