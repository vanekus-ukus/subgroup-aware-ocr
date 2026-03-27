# subgroup-aware-ocr: results overview

This file summarizes the public-safe evidence retained from the finished private study.

## Completed Main Study

Weighted-shape CER mean across 3 seeds:

| Config | Weighted-Shape CER Mean | CER Mean | Exact Mean |
| --- | ---: | ---: | ---: |
| `synthetic_static` | `0.3904` | `0.2946` | `0.3289` |
| `synthetic_curriculum` | `0.3967` | `0.2967` | `0.3359` |
| `shape_weighted` | `0.4689` | `0.3434` | `0.2647` |

Source: `reports/public/main_experiment_summary.csv`

## Key Pairwise Deltas

- `shape_weighted - synthetic_static = +0.0785`
- `shape_weighted - synthetic_curriculum = +0.0722`
- `synthetic_curriculum - synthetic_static = +0.0063`

Source: `reports/public/main_config_pairwise_deltas.csv`

## Hardest Stable Buckets

The stable hard-bucket story is not just an artifact of tiny groups:
- `shape:square` is large enough to trust materially;
- `source:imported_sg` is also large enough to treat as a real subgroup signal;
- style buckets are smaller and should be interpreted more cautiously.

In the best completed main family, the hardest residual gaps remain:
- `shape:square`
- `source:imported_sg`
- `source:imported_usa` (small and noisy)
- inverse/high-contrast style pockets

Source: `reports/public/main_gap_summary.csv`, `reports/public/decisive_hard_bucket_audit.md`

## Operational Shape Labels

Predicted shape labels were accurate enough for operational subgroup-aware handling:
- shape transfer accuracy: `0.9813`
- balanced accuracy: `0.9720`

Source: `reports/public/key_numbers.csv`

## What Did Not Become The Main Result

The decisive pass did not support replacing global static synthetic mixing with a targeted static synthetic branch.

Source: `reports/public/decisive_targeted_synth_audit.md`
