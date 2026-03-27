# subgroup-aware-ocr

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Research](https://img.shields.io/badge/focus-heterogeneous%20OCR-informational.svg)

`subgroup-aware-ocr` is a public-facing sanitized repository for research on OCR over heterogeneous sequence crops.

Canonical repository URL: `https://github.com/vanekus-ukus/subgroup-aware-ocr`

## Overview

This repository studies OCR on heterogeneous sequence crops, where average metrics can hide failures on critical subgroups. The project compares real-only and synthetic-enhanced training regimes, uses predicted shape labels for operational subgroup handling, and argues for subgroup-aware evaluation and model selection.

```mermaid
flowchart TD
    A["Heterogeneous OCR<br/>shape, source, style"]
    B["Predicted subgroup labels"]
    C["Training comparison<br/>real-only vs synthetic-enhanced"]
    D["Subgroup-aware evaluation<br/>aggregate + hard-bucket metrics"]
    E["Main conclusion<br/>synthetic helps hard buckets,<br/>selection must be subgroup-aware"]

    A --> B --> C --> D --> E
```

The repository packages three things:
- a reusable OCR training and evaluation package;
- a toy dataset and smoke-test pipeline that run end to end without private data;
- curated public-safe result summaries from the completed private study.

Public-facing branding is `subgroup-aware-ocr`. The internal Python package remains `shape_aware_ocr` for implementation stability.

## What Was Actually Done

Completed evidence in the underlying study supports the following points:
- `predicted shape labels` are accurate enough for operational subgroup-aware handling;
- `subgroup-aware evaluation` is necessary because aggregate CER hides the hardest buckets;
- `synthetic mixing` improves the completed main study over the real-only `shape_weighted` baseline;
- the strongest completed synthetic variant is currently `synthetic_static`, not `synthetic_curriculum`.

Key public numbers are listed in `RESULTS_OVERVIEW.md` and `reports/public/key_numbers.csv`.

## Key Findings

- Heterogeneous OCR should not be judged by aggregate CER alone.
- Predicted shape labels are accurate enough for operational subgroup handling.
- Synthetic-enhanced training improves robustness on the hardest buckets.
- `square` remains the hardest stable subgroup in the completed study.
- Subgroup-aware model selection can differ from aggregate-best selection.

## Pipeline Overview

```text
real crops + manifests
        |
        v
build alphabet
        |
        v
train OCR (optional synthetic mixing, optional subgroup-aware objective)
        |
        v
evaluate on fixed split
        |
        v
aggregate metrics + subgroup metrics + pairwise deltas + report package
```

## Repository Map

- `src/shape_aware_ocr/`: installable Python package
- `configs/`: ablation and decisive experiment configs
- `data/toy_research/`: small safe dataset for examples and smoke tests
- `docs/`: protocol, evaluation design, repo structure, release notes
- `reports/public/`: curated public-safe summaries from finished experiments
- `artifacts/public/`: public-safe lightweight visual assets
- `tests/`: unit and smoke tests
- `scripts/`: convenience wrappers for toy/demo runs

## Quickstart

Install in a fresh environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the toy pipeline:

```bash
python -m shape_aware_ocr.cli.generate_toy_dataset --out data/toy_research --clean
python -m shape_aware_ocr.cli.build_alphabet --data-root data/toy_research/real --out data/toy_research/artifacts
python -m shape_aware_ocr.cli.train \
  --data-root data/toy_research/real \
  --out outputs/toy_train \
  --alphabet data/toy_research/artifacts/alphabet.json \
  --shape-manifest data/toy_research/manifests/shape_manifest.csv \
  --sample-class-manifest data/toy_research/manifests/style_manifest.csv \
  --synth-root data/toy_research/synth_square \
  --synth-ratio 0.5 \
  --best-by weighted_shape \
  --epochs 1 \
  --batch-size 4 \
  --train-workers 0 \
  --val-workers 0 \
  --amp false
python -m shape_aware_ocr.cli.evaluate \
  --data-root data/toy_research/real \
  --checkpoint outputs/toy_train/checkpoints/best.pt \
  --alphabet data/toy_research/artifacts/alphabet.json \
  --shape-manifest data/toy_research/manifests/shape_manifest.csv \
  --sample-class-manifest data/toy_research/manifests/style_manifest.csv \
  --out outputs/toy_eval \
  --workers 0
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility Note

This repository is intentionally sanitized.

Fully reproducible from the public repo:
- package installation;
- toy dataset generation;
- smoke OCR training/evaluation;
- report-generation logic on toy or user-provided data.

Not included publicly:
- the private benchmark images;
- heavy checkpoints;
- raw qualitative error sheets built from private images;
- full experiment output trees.

See `REPRODUCIBILITY.md` and `DATA_ACCESS.md`.

## Limitations

- The full benchmark used for the main study is private.
- Public summaries are sufficient to inspect the evidence, but not to recreate the exact main-study numbers end to end.
- Style-bucket conclusions in the completed study are weaker than shape/source conclusions because some style buckets are small.

## Citation

If you use this repository, cite the code package via `CITATION.cff` and describe the data restrictions clearly.
