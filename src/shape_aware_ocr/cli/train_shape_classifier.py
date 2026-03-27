from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.shape_classifier import ShapeClassifierConfig, train_shape_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rect-vs-square classifier for sequence crops")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--shape-manifest", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--val-workers", type=int, default=0)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ShapeClassifierConfig(
        data_root=args.data_root,
        shape_manifest=args.shape_manifest,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        img_size=args.img_size,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        balanced_sampler=args.balanced_sampler,
        amp=args.amp,
    )
    summary = train_shape_classifier(config)
    print(f"[INFO] Best checkpoint: {summary.best_checkpoint}")
    print(f"[INFO] Report: {summary.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
