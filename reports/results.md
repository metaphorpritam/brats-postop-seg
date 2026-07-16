# Results — Track A (faithful defects) vs Track B (corrected)

**Metric:** voxel-wise Dice (NOT BraTS lesion-wise — see CLAUDE.md §14.3; do not compare to
challenge leaderboards). **Task:** flat 5-class post-treatment BraTS-GLI. **Model:** one shared
MONAI `UNet` (~1,983,069 params), byte-identical across tracks (§4.2). **Design:** 700 cases,
split `reports/splits.json` (**490/105/105** train/val/test, stratified by RC presence, **fixed at
seed 42**); each track trained on **3 seeds {0,1,2}** varying only init/augmentation/sampling, so
every number below is **mean ± SD over 3 runs on one shared test set**. Recomputed from
`reports/eval_{a,b}_s{0,1,2}.json` and `~/brats/runs/{a,b}_s{0,1,2}/metrics.csv`.

## Held-out TEST set — per-class Dice (n_test = 105, mean ± SD over 3 seeds)

| Class | Track A | Track B | Δ (B−A) | n GT-present (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.009 ± 0.008 | 0.466 ± 0.004 | **+0.457 ± 0.004** | 105 / 48 |
| SNFH (2) | 0.683 ± 0.017 | 0.870 ± 0.003 | **+0.187 ± 0.017** | 105 / 105 |
| ET (3)   | 0.299 ± 0.033 | 0.650 ± 0.007 | **+0.351 ± 0.028** | 100 / 85 |
| RC (4)   | 0.403 ± 0.026 | 0.694 ± 0.002 | **+0.291 ± 0.029** | 87 / 87 |
| **Mean foreground** | **0.349 ± 0.013** | **0.670 ± 0.002** | **+0.321 ± 0.013** | |

**Headline:** mean foreground Dice **0.349 → 0.670**, **Δ = +0.321 ± 0.013** on unseen data (≈1.9×).
The absolute delta reproduces the 200-case pilot's +0.333 **with error bars**, and the per-seed SD is
tiny (Track B ±0.002–0.007) — this is a stable effect, not a lucky run. Largest recovery on the
**rarest class, NETC (+0.457)**, then ET (+0.351) and RC (+0.291); SNFH was already learnable in A and
still gains +0.187. (Ratio note: the pilot's ~2.8× shrank to ~1.9× because more data lets even the
**defective** Track A partly learn the *prevalent* classes ET/RC — see the scale effect below. The
absolute A→B delta is what held.)

### Scale effect — Track A rose, the delta held
At 140 training cases (pilot) Track A was data-starved and collapsed on everything (mean_fg 0.180); at
490 cases it recovers the **prevalent** classes (ET GT-present in ~83%, RC ~83%) to Dice 0.30 / 0.40,
lifting its mean to 0.349. Track B rose further (0.513 → 0.670), so the delta survived. The one class
that **still collapses in Track A is NETC** (0.009) — genuinely rare (GT-present in 48/105 faithful
cases) — and it is exactly where Track B's fixes pay off most. The story sharpened: *the defects'
damage is largely absorbed by data except on the rare class, where the corrections still decide it.*

### The per-class case counts expose defect D1
Track A reports NETC present in **105/105** test cases and ET in **100/105**, but the faithful labels
(Track B) show only **48** and **85**. Bilinear `cv2.resize` of the label volume (D1) interpolates
across class boundaries and, after truncation to int, **invents** minority-class voxels — exactly the
"can invent/destroy labels" harm in §3/D1, measured directly and *more starkly than in the pilot*
(there 30/30 vs 17). This is why per-class case counts are reported alongside Dice (§6.5), never Dice
alone.

## Validation set — per-class Dice (mean ± SD, each track at its own best epoch)

| Class | Track A | Track B | Δ (B−A) |
|---|---|---|---|
| NETC | 0.007 ± 0.004 | 0.545 ± 0.009 | +0.538 |
| SNFH | 0.674 ± 0.014 | 0.856 ± 0.003 | +0.182 |
| ET   | 0.261 ± 0.022 | 0.653 ± 0.010 | +0.392 |
| RC   | 0.337 ± 0.017 | 0.674 ± 0.003 | +0.337 |
| **Mean foreground** | **0.320 ± 0.008** | **0.682 ± 0.003** | **+0.362** |

Validation voxel accuracy: **A 0.991 / B 0.995** (degenerate — ~98%+ background; this is exactly the
D4 motivation, so it is *not* a test-set headline). The test evaluator reports Dice only.

Model selection (**D4**): Track A checkpointed on `val_accuracy` (the degenerate metric — best @
**epochs 18–22** across seeds); Track B on **mean foreground Dice** (the fix — best @ **epochs 64–72**).

## Speed — defect D9 (I/O-bound naive loader vs cached)

| | Track A (naive, no cache) | Track B (PersistentDataset) |
|---|---|---|
| data load / epoch | ~23 s | ~8 s (cache warm) |
| compute / epoch | ~5 s | ~4 s |

Track A is **I/O-bound** (data load ≫ compute), reproducing the original's pathology on a laptop;
caching gives a **~2.8× data-loading speedup** (cold cache build ~10 s, once; measured at Gate 2.10).
Less dramatic than the original's datacenter 5–10× because our data is small and gzip-cached by the OS
— stated honestly.

## Attribution
Track A → Track B differs only in: label interpolation (D1), normalization (D5), loss + class-balanced
sampling (D3), model-selection metric (D4), slice sampling / patching (D6), and caching (D9). The
architecture is held constant (§4.2), so the delta is attributable to the pipeline, not the network.

*All numbers come from committed run logs (guardrail #9). Runs/logs under `~/brats/runs/{a,b}_s{0,1,2}/`;
aggregated in `reports/results_700_seeds.json`. The 200-case pilot remains in git history.*
