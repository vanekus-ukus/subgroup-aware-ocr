#!/usr/bin/env bash
set -euo pipefail
python -m shape_aware_ocr.cli.generate_toy_dataset --out data/toy_research --clean
python -m shape_aware_ocr.cli.build_alphabet --data-root data/toy_research/real --out data/toy_research/artifacts
python -m shape_aware_ocr.cli.run_ablation \
  --data-root data/toy_research/real \
  --alphabet data/toy_research/artifacts/alphabet.json \
  --shape-manifest data/toy_research/manifests/shape_manifest.csv \
  --sample-class-manifest data/toy_research/manifests/style_manifest.csv \
  --synth-root data/toy_research/synth_square \
  --out-root outputs/toy_ablations \
  --config-dir configs/ablation \
  --config baseline \
  --seeds 42 \
  --batch-size 4 \
  --train-workers 0 \
  --val-workers 0 \
  --eval-workers 0 \
  --amp false
