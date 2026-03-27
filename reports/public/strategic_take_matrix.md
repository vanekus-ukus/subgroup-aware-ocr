# Strategic Take Matrix

## Strongest Honest Central Thesis

Subgroup-aware evaluation and model interpretation are necessary for OCR on heterogeneous sequence crops, and predicted shape labels are good enough to make that operational.

## Strongest Supported Contributions

1. A fixed-split subgroup-aware evaluation protocol materially changes how OCR quality should be interpreted.
2. Predicted shape labels are accurate enough to replace oracle subgroup labels operationally on this benchmark.
3. Aggregate CER masks a stable hard-bucket cluster: `square + imported + inverse/high-contrast`.
4. Synthetic mixing improves the completed main study relative to real-only `shape_weighted`.
5. The evidence does not support schedule novelty as the central claim.

## Safe Claims

- Predicted shape labels are practically usable on this benchmark.
- Aggregate-only evaluation is insufficient on this benchmark.
- Synthetic mixing improves completed main results relative to real-only `shape_weighted`.

## Unsafe Claims

- `synthetic_curriculum` is the winner.
- Temporary synthetic replaces permanent synthetic support without loss.
- The main novelty is a new training schedule.
