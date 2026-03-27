from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.evaluation import EvaluationConfig, evaluate_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shape-aware sequence OCR")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--alphabet", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preprocess-mode", choices=["stand", "legacy"], default="stand")
    parser.add_argument("--shape-manifest", type=str, default="")
    parser.add_argument("--sample-class-manifest", action="append", default=[])
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--split-name", type=str, default="")
    parser.add_argument("--best-shape-square-weight", type=float, default=0.65)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EvaluationConfig(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        alphabet=args.alphabet,
        out=args.out,
        batch_size=args.batch_size,
        workers=args.workers,
        preprocess_mode=args.preprocess_mode,
        shape_manifest=args.shape_manifest,
        sample_class_manifests=tuple(args.sample_class_manifest),
        split_manifest=args.split_manifest,
        split_name=args.split_name,
        best_shape_square_weight=args.best_shape_square_weight,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_errors=args.max_errors,
    )
    summary = evaluate_ocr(config)
    print(f"[INFO] Report: {summary.report_path}")
    print(f"[INFO] Errors CSV: {summary.errors_csv}")
    print(f"[INFO] Subgroup CSV: {summary.subgroup_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
