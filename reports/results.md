# Results — Track A vs Track B

> **Placeholder.** Filled at Phase 4/5 from committed run logs (CLAUDE.md §9). Every number
> must come from a run whose CSV/TensorBoard log is committed — never fabricated (guardrail #9).
> Report **per-class case counts** alongside Dice (§6.5): an unqualified "RC Dice" over a few
> cases is not a result. Use **voxel-wise** Dice language, never "BraTS Dice" (§14.3).

## Per-class voxel-wise Dice — held-out test set

| Class | Track A | Track B | n cases (GT-present) |
|---|---|---|---|
| NETC (1) | — | — | — |
| SNFH (2) | — | — | — |
| ET (3)   | — | — | — |
| RC (4)   | — | — | — |
| **Mean foreground** | — | — | — |

Seed: 42. Test-set size / split: see `reports/splits.json`.

## Speed (defect D9)

| | Track A (naive per-epoch loader) | Track B (PersistentDataset) |
|---|---|---|
| s/epoch | — | — |
