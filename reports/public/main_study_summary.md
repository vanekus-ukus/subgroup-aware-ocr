# Main Study Summary

## Completed Main Families

| Config | Weighted-Shape CER Mean | CER Mean | Exact Mean |
| --- | ---: | ---: | ---: |
| `synthetic_static` | `0.3904` | `0.2946` | `0.3289` |
| `synthetic_curriculum` | `0.3967` | `0.2967` | `0.3359` |
| `shape_weighted` | `0.4689` | `0.3434` | `0.2647` |

## Interpretation

- Synthetic mixing improves the completed main study relative to real-only `shape_weighted`.
- `synthetic_static` is the best completed family under the subgroup-aware main metric.
- The strongest residual weakness remains concentrated in the hard-bucket cluster rather than disappearing in the aggregate.
