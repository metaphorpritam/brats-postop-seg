# Results — Track A (faithful defects) vs Track B (corrected)

**Metric:** voxel-wise Dice (NOT BraTS lesion-wise — see CLAUDE.md §14.3; do not compare to
challenge leaderboards). **Task:** flat 5-class post-treatment BraTS-GLI. **Model:** one shared
MONAI `UNet` (~1,983,069 params), byte-identical across tracks (§4.2). Seed 42; split
`reports/splits.json` (140/30/30 train/val/test, stratified by RC presence). Numbers below are
recomputed from `reports/eval_{a,b}.json` and `~/brats/runs/{a,b}/metrics.csv`.

## Held-out TEST set — per-class Dice (n_test = 30)

| Class | Track A | Track B | Δ (B−A) | n GT-present (A / B) |
|---|---|---|---|---|
| NETC (1) | 0.000 | 0.140 | **+0.140** | 30 / 17 |
| SNFH (2) | 0.580 | 0.807 | **+0.227** | 30 / 30 |
| ET (3)   | 0.054 | 0.567 | **+0.513** | 30 / 25 |
| RC (4)   | 0.087 | 0.538 | **+0.451** | 25 / 25 |
| **Mean foreground** | **0.180** | **0.513** | **+0.333** | |

**Headline:** mean foreground Dice **0.180 → 0.513** (~2.8×) on unseen data; largest recovery on
ET (+0.513) and RC (+0.451). Lands in the plan's "respectable" band (§2.3); NETC stays hardest.

### The per-class case counts expose defect D1
Track A reports NETC present in **30/30** test cases and ET in **30/30**, but the faithful labels
(Track B) show only **17** and **25**. Bilinear `cv2.resize` of the label volume (D1) interpolates
across class boundaries and, after truncation to int, **invents** minority-class voxels — exactly
the "can invent/destroy labels" harm in §3/D1, measured directly. This is why per-class case counts
are reported alongside Dice (§6.5), never Dice alone.

## Validation set — per-class Dice (each track at its own best epoch)

| Class | Track A | Track B | Δ (B−A) |
|---|---|---|---|
| NETC | 0.000 | 0.162 | +0.162 |
| SNFH | 0.615 | 0.794 | +0.179 |
| ET   | 0.036 | 0.572 | +0.536 |
| RC   | 0.140 | 0.624 | +0.484 |
| **Mean foreground** | **0.198** | **0.538** | **+0.340** |

Validation voxel accuracy: **A 0.9896 / B 0.9924** (degenerate — ~98% background; this is exactly the
D4 motivation, so it is *not* a test-set headline). The test evaluator reports Dice only.

Model selection (**D4**): Track A checkpointed on `val_accuracy` (the degenerate metric — best @ epoch 10);
Track B on **mean foreground Dice** (the fix — best @ epoch 38).

## Speed — defect D9 (I/O-bound naive loader vs cached)

| | Track A (naive, no cache) | Track B (PersistentDataset) |
|---|---|---|
| data load / epoch | ~23 s | ~8 s (cache warm) |
| compute / epoch | ~5 s | ~4 s |

Track A is **I/O-bound** (data load ≫ compute), reproducing the original's pathology on a laptop;
caching gives a **~2.8× data-loading speedup** (cold cache build ~10 s, once). Less dramatic than the
original's datacenter 5–10× because our data is small and gzip-cached by the OS — stated honestly.

## Attribution
Track A → Track B differs only in: label interpolation (D1), normalization (D5), loss + class-balanced
sampling (D3), model-selection metric (D4), slice sampling / patching (D6), and caching (D9). The
architecture is held constant (§4.2), so the delta is attributable to the pipeline, not the network.

*All numbers come from committed run logs (guardrail #9). Runs/logs under `~/brats/runs/{a,b}/`.*
