# subgroup-aware-ocr: evaluation design

## Why Aggregate CER Is Not Enough

This project studies OCR under subgroup heterogeneity. Aggregate CER can hide failures concentrated in underrepresented or structurally different buckets.

## Core Metrics

- CER
- exact match accuracy
- `cer_square`
- `cer_rect`
- `weighted_shape_cer`
- `macro_shape_cer`

## Subgroup Axes

- `shape`: `rect`, `square`
- `style`: appearance buckets such as inverse/high-contrast styles
- `source`: native vs imported provenance buckets

## Selection Logic

The repository supports subgroup-aware checkpoint selection and report aggregation.

Current public evidence supports the following practical rule:
- aggregate CER is useful but insufficient;
- subgroup-aware metrics provide a more justified model choice on heterogeneous OCR benchmarks.

## Bucket Caveats

- `square` and `imported_sg` are large enough to support stable conclusions in the private study.
- small style buckets and `imported_usa` should be treated as supporting evidence, not the main claim.
