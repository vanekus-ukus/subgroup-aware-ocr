# Reproducibility

## What Is Fully Reproducible From This Public Repository

- package installation;
- toy dataset generation;
- toy OCR training/evaluation smoke tests;
- report generation and aggregation logic;
- ablation runner logic on user-provided data;
- subgroup-aware metrics and selection logic.

## What Is Only Partially Reproducible

The full research protocol is public, but the exact main-study numbers require non-public data.

Publicly available here:
- configs;
- pipeline code;
- toy dataset;
- report schemas;
- curated public-safe summaries from the completed study.

Not publicly included:
- the private benchmark crops;
- heavy model checkpoints;
- raw intermediate output trees;
- qualitative figures built from private images.

## How To Recreate The Study Structure On Your Own Data

1. Prepare a flat OCR dataset with filenames like `000001_AB12CD.png`.
2. Create `shape_manifest.csv` and optional `style_manifest.csv` and `split_manifest.csv`.
3. Build an alphabet.
4. Train OCR with or without synthetic mixing.
5. Evaluate on the fixed split.
6. Build experiment summaries and subgroup reports.

See:
- `DATA_ACCESS.md`
- `docs/experiments.md`
- `docs/evaluation.md`

## Honest Boundary

This repository is not a claim of full open-data reproducibility. It is a strong public research package that preserves:
- the method implementation;
- the experiment structure;
- the evidence summaries;
- a working toy pipeline.
