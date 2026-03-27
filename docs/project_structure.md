# Subgroup-Aware OCR Project Structure

```text
src/shape_aware_ocr/         core package
src/shape_aware_ocr/cli/     package CLI entry points
configs/                     ablation and decisive configs
data/toy_research/           safe toy benchmark
data/manifests/              manifest templates
docs/                        protocol and public-release docs
reports/public/              curated public-safe summaries
artifacts/public/            safe lightweight assets
tests/                       unit and smoke tests
scripts/                     convenience wrappers for toy/demo flows
```

Design principles:
- code lives under `src/`;
- no `sys.path` bootstrap hacks;
- private data is not part of the public tree;
- reports are curated rather than raw-dump exports.
