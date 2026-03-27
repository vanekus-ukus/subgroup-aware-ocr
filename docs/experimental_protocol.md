# Experimental Protocol

## Research question

How should sequence OCR be trained when the data distribution is heterogeneous across geometry and appearance, and the most important subgroup is underrepresented?

## Hypotheses

1. `H1`: subgroup-aware model selection improves `square` performance over a plain CER objective while preserving acceptable `rect` quality.
2. `H2`: subgroup-matched synthetic square crops improve `square` CER more effectively than oversampling alone.
3. `H3`: explicit style buckets expose robustness failures on inverse-color or colored backgrounds that are hidden by aggregate CER.
4. `H4`: bootstrapped subgroup labels from a lightweight classifier are accurate enough to retain most of the shape-aware OCR gains versus oracle labels.
5. `H5`: provenance-aware reporting can separate gains from native data and imported low-resource subgroup data.

## Dataset design

Required inputs:
- real crops named as `000001_LABEL.png`
- `shape_manifest.csv` with `rect/square`
- optional `style_manifest.csv` for appearance buckets
- optional synthetic square set for the low-resource subgroup

For portability, synthetic samples should be aligned to real samples by canonical sequence label rather than raw filename stem. Raw instance IDs are an implementation detail and should not define semantic pairing.

Recommended split strategy:
- fixed validation split with a saved seed
- identical split across all ablations
- report counts for each subgroup in train and validation
- keep provenance manifests so imported or synthetic subsets can be measured separately at test time

## Baseline and ablations

1. Baseline OCR: no shape weighting, no synthetic data, no hard mining.
2. `+ shape-aware objective`: select best checkpoint by weighted `square/rect` CER.
3. `+ square oversampling`: increase contribution of the underrepresented subgroup.
4. `+ synthetic curriculum`: mix square synthetic data and decay it near convergence.
5. `+ static synthetic mix`: keep the same synthetic ratio without decay as a control against curriculum.
6. `+ hard mining`: oversample difficult real square samples identified by the previous model.
7. `+ style-aware monitoring`: keep per-style metrics for robustness analysis.
8. `oracle vs predicted subgroup labels`: replace the oracle `shape_manifest.csv` with classifier-generated labels and measure the delta.
9. `provenance-aware evaluation`: evaluate the best checkpoints with an additional sample-class manifest describing `native` vs imported data sources.

## Metrics

Primary metrics:
- CER
- exact match accuracy
- `square` CER
- `rect` CER
- weighted shape CER
- macro shape CER

Secondary metrics:
- per-style CER/exact
- per-provenance CER/exact
- bootstrap 95% confidence intervals for CER and exact match
- subgroup-label transfer accuracy, balanced accuracy, square recall, rect recall

## Statistical reporting

For any serious claim, do not report one run.

Minimum recommendation:
- 3 random seeds for the main ablations
- fixed train/val split seed if data is small
- bootstrap confidence intervals on validation metrics
- paired per-seed deltas for any comparison between two methods on the same benchmark split
- if using bootstrapped shape labels, report the classifier confusion matrix against an oracle subset

## Threats to validity

- synthetic data may leak superficial cues if the rendering pipeline is too narrow
- subgroup labels predicted by a classifier can inject annotation noise
- weighted subgroup metrics can overfit small validation subsets if counts are not reported
- provenance groups can be confounded with geometry if imported data is concentrated in one subgroup

## What to show in the paper

- one table with all ablations and subgroup metrics
- one figure with curriculum schedule and validation curves
- one robustness table with per-style metrics
- one transfer table: oracle vs predicted subgroup labels
- one provenance table: native vs imported subgroup performance
- one qualitative figure with hardest failures before and after synthetic curriculum
