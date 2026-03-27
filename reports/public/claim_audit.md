# Claim Audit

## Already Well Supported

- OCR on this benchmark behaves as a heterogeneous-subgroup problem rather than a single uniform task.
- Predicted shape labels are accurate enough for operational subgroup-aware handling.
- Aggregate-only evaluation is insufficient for model interpretation on this benchmark.
- The completed main synthetic family improves over the completed main real-only `shape_weighted` baseline.
- The strongest completed main synthetic variant is `synthetic_static`, not `synthetic_curriculum`.

## Moderately Supported

- The most defensible framing is evaluation/selection-centric rather than schedule-centric.
- Synthetic mixing helps broadly across tracked hard buckets, not only on one bucket.
- Temporary synthetic usage can unlock training without immediate collapse after removal, but that did not become the main result.

## Weak Or Exploratory

- Temporary synthetic can replace permanent synthetic support without loss.
- Real-only consolidation after synthetic warm-up can recover the hardest buckets well enough to become the central method result.
- Synthetic decay-to-zero is a stronger mechanism than completed static mixing.

## Not Supported

- `synthetic_curriculum` as the main winner.
- Temporary synthetic as the central claim.
- A new training schedule as the strongest contribution.
