from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.training import TrainingConfig, train_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shape-aware sequence OCR")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--alphabet", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--val-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--best-by", choices=["cer", "loss", "weighted_shape"], default="weighted_shape")
    parser.add_argument("--best-shape-square-weight", type=float, default=0.65)
    parser.add_argument("--preprocess-mode", choices=["stand", "legacy"], default="stand")
    parser.add_argument("--shape-manifest", type=str, default="")
    parser.add_argument("--sample-class-manifest", action="append", default=[])
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--square-oversample", type=float, default=1.0)
    parser.add_argument("--synth-root", type=str, default="")
    parser.add_argument("--synth-ratio", type=float, default=0.0)
    parser.add_argument("--synth-decay-last", type=int, default=0)
    parser.add_argument("--synth-schedule-mode", choices=["constant", "late_decay", "warmup_stop", "early_decay", "warmup_decay"], default="late_decay")
    parser.add_argument("--synth-warmup-epochs", type=int, default=0)
    parser.add_argument("--synth-stop-epoch", type=int, default=0)
    parser.add_argument("--synth-max-per-epoch", type=int, default=0)
    parser.add_argument("--synth-use-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--synth-anchor-mode", choices=["square", "hard_square", "all"], default="square")
    parser.add_argument("--hard-manifest", type=str, default="")
    parser.add_argument("--hard-topk", type=int, default=0)
    parser.add_argument("--hard-oversample", type=float, default=1.0)
    parser.add_argument("--finetune-square-oversample", type=float, default=1.0)
    parser.add_argument("--finetune-hard-oversample", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-file", type=str, default="train_curve.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        data_root=args.data_root,
        out=args.out,
        alphabet=args.alphabet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        val_split=args.val_split,
        seed=args.seed,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        best_by=args.best_by,
        best_shape_square_weight=args.best_shape_square_weight,
        preprocess_mode=args.preprocess_mode,
        shape_manifest=args.shape_manifest,
        sample_class_manifests=tuple(args.sample_class_manifest),
        split_manifest=args.split_manifest,
        square_oversample=args.square_oversample,
        synth_root=args.synth_root,
        synth_ratio=args.synth_ratio,
        synth_decay_last=args.synth_decay_last,
        synth_schedule_mode=args.synth_schedule_mode,
        synth_warmup_epochs=args.synth_warmup_epochs,
        synth_stop_epoch=args.synth_stop_epoch,
        synth_max_per_epoch=args.synth_max_per_epoch,
        synth_use_all=args.synth_use_all,
        synth_anchor_mode=args.synth_anchor_mode,
        hard_manifest=args.hard_manifest,
        hard_topk=args.hard_topk,
        hard_oversample=args.hard_oversample,
        finetune_square_oversample=args.finetune_square_oversample,
        finetune_hard_oversample=args.finetune_hard_oversample,
        amp=args.amp,
        plot_file=args.plot_file,
    )
    summary = train_ocr(config)
    print(f"[INFO] Best checkpoint: {summary.best_checkpoint}")
    print(f"[INFO] Last checkpoint: {summary.last_checkpoint}")
    print(f"[INFO] Train log: {summary.train_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
