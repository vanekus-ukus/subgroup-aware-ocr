# Decisive Selection Audit

## What Changed Under Selection Rules

Within one targeted branch, aggregate-best and subgroup-aware-best checkpoint happened to coincide.

At the completed comparable model level, the justified best model differs by selection rule:
- aggregate CER favors `synthetic_curriculum / seed_62`
- subgroup-aware weighted-shape CER favors `synthetic_static / seed_62`

## Verdict

The strongest selection effect in this project is not checkpoint-level inside a single run, but model-selection-level between completed comparable models.

That is enough to support the claim that aggregate-only selection can favor a different model than subgroup-aware selection.
