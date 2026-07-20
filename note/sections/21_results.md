## 21. Results and interpretation {#results}

Six training runs (two tracks × three seeds), one shared 105-case test set, every number a mean ±
standard deviation over the three seeds. All metrics are **voxel-wise Dice** (§13, §16) — not the
BraTS lesion-wise score, so these numbers are internally comparable but **not** comparable to
challenge leaderboards (§16, §22).

### The headline table

| Class | Track A | Track B | Δ (B − A) | GT-present cases (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.009 ± 0.008 | 0.466 ± 0.004 | **+0.457** | 105 / 48 |
| SNFH (2) | 0.683 ± 0.017 | 0.870 ± 0.003 | **+0.187** | 105 / 105 |
| ET (3) | 0.299 ± 0.033 | 0.650 ± 0.007 | **+0.351** | 100 / 85 |
| RC (4) | 0.403 ± 0.026 | 0.694 ± 0.002 | **+0.291** | 87 / 87 |
| **Mean foreground** | **0.349 ± 0.013** | **0.670 ± 0.002** | **+0.321 ± 0.013** | |

The deliverable is the bottom-right cell: **Δ = +0.321 ± 0.013**. Fixing the data pipeline — with
the network frozen — nearly doubles mean foreground Dice, and the ±0.013 spread across seeds means
this is a stable effect, not a fluke.

![Per-class test Dice, Track A vs Track B, with per-seed standard-deviation error bars. Every class rises; the rarest, NETC, rises most.](reports/figures/report/perclass_test_dice.png)

![Per-class A to B improvement (Track B minus Track A). NETC +0.457 leads, then ET, RC, and SNFH.](reports/figures/report/delta_test.png)

### Reading the per-class story

The gains are **largest exactly where Track A collapsed**. NETC — the small, rare necrotic core —
goes from a Dice of 0.009 (essentially "never found") to 0.466: a **+0.457** recovery. This is the
cleanest possible evidence for the whole thesis. NETC is the class the defects suppress hardest
(plain cross-entropy has no incentive to find a handful of voxels, §14), and it is the class the
fixes rescue most. SNFH, the large and easy edema/FLAIR-hyperintensity region, was already learnable
by the coasting Track A (0.683) and merely refines to 0.870 — a real but smaller gain, because there
was no collapse to reverse.

!!! intuition "One sentence to remember"
    The corrections matter *most for the classes that matter most and are hardest* — the rare tumour
    sub-regions — and *least for the easy majority structure*. That is the signature of a genuine
    data-pipeline fix rather than a lucky global tweak.

### D1, caught red-handed

The right-hand column of the table is not a footnote — it is a measurement of defect D1. Track A
reports **NETC present in 105/105** test cases and **ET in 100/105**; the faithful nearest-neighbour
labels (Track B) show only **48** and **85**. Those extra "present" cases are voxels D1 *fabricated*
by interpolating across class boundaries (§20). Plotted side by side, the inflation is stark:

![Ground-truth-present case counts per class. Track A's bilinear-resized labels report minority classes in far more cases than the faithful labels — invented voxels, measured directly.](reports/figures/report/count_inflation.png)

### The two selection metrics diverge

Track A checkpoints on validation accuracy (D4) and plateaus low; Track B checkpoints on mean
foreground Dice and climbs. The training curves — each a 3-seed mean with an SD band — show the two
tracks separating and never rejoining:

![Validation mean-foreground Dice per epoch, mean ± SD over three seeds. Track B climbs to about 0.68; Track A plateaus near 0.32.](reports/figures/report/training_curves.png)

### Seeing it on a slice

Numbers abstract away the anatomy. On three held-out cases (ground truth | Track A | Track B), Track
A captures the easy SNFH and much of the ET but **misses the rare NETC and RC**; Track B recovers
them in spatial agreement with the ground truth:

![Axial overlays on three test cases: ground truth, Track A, and Track B, in the five-class colour map. Track B's predictions match the ground truth far more closely, especially on the rare classes.](reports/figures/combined_overlay.png)

Three further held-out cases tell the same story. Watch the **amber resection cavity (RC)** and
the **blue necrotic core (NETC)**: Track A drops them almost entirely — in case `00469` it predicts
only the green SNFH and misses the RC completely — while Track B restores both in the right place:

![Three additional held-out TEST cases (ground truth, Track A, Track B). Track A again captures SNFH and much of ET but omits the rare RC and NETC; Track B recovers them in spatial agreement with the ground truth.](reports/figures/combined_overlay_2.png)

### The honest caveat about "≈1.9×"

At the 200-case pilot scale, Track A was so data-starved it collapsed on *everything* (mean
foreground Dice 0.180), and Track B scored 0.513 — a **2.8×** ratio. At the full 490-case training
scale, Track A *recovers the prevalent classes* (ET and RC are present in ~81–83% of cases) and rises to
0.349, while Track B rises to 0.670. Both went up; Track B went up more.

!!! gotcha "Gotcha — quote the absolute delta, not the ratio"
    The **ratio** fell from ~2.8× to ~1.9× purely because the *denominator* (Track A) got stronger
    with more data — not because the fix got weaker. The **absolute** delta barely moved
    (+0.333 → +0.321). A ratio with a moving, small denominator is a treacherous headline; the
    additive gap on a shared test set is the trustworthy one, and it is why we quote +0.321 ± 0.013.

!!! example "Example question"
    Compute the mean-foreground Δ from the table's four per-class Track-A and Track-B numbers, and
    check it against the reported +0.321.

    **Solution.** Mean foreground Dice averages the four foreground classes.
    Track A: $\tfrac{1}{4}(0.009 + 0.683 + 0.299 + 0.403) = \tfrac{1}{4}(1.394) = 0.3485 \approx
    0.349.$ Track B: $\tfrac{1}{4}(0.466 + 0.870 + 0.650 + 0.694) = \tfrac{1}{4}(2.680) = 0.670.$
    Δ $= 0.670 - 0.349 = 0.321$ — matching the reported value. (The evaluator computes the per-class
    means over cases first, so tiny rounding aside, the class-average of the means equals the mean
    foreground number.)
