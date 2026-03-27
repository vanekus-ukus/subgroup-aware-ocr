# subgroup-aware-ocr: experiments

## Publicly Represented Experiment Families

- `shape_weighted`: real-only comparator with subgroup-aware checkpointing
- `synthetic_static`: completed main synthetic baseline
- `synthetic_curriculum`: completed main synthetic decay/curriculum variant
- decisive targeted-synthetic pass: a focused negative-result check

## Main Public Summary Files

- `reports/public/main_experiment_summary.csv`
- `reports/public/main_subgroup_summary.csv`
- `reports/public/main_config_pairwise_deltas.csv`
- `reports/public/main_gap_summary.csv`
- `reports/public/key_numbers.csv`

## What Is Not Publicly Included

- full raw experiment directories;
- private benchmark manifests with sensitive provenance information;
- qualitative error contact sheets built from private images.

## Toy Experiment Flow

Use the toy benchmark to validate the package wiring:
1. generate toy dataset;
2. build alphabet;
3. run 1-epoch smoke train;
4. evaluate;
5. inspect report outputs.
