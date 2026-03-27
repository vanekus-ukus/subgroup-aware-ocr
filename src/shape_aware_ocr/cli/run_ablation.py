from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.ablation import run_ablations, write_ablation_reports

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style OCR ablations across configs and seeds")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--alphabet", required=True, type=str)
    parser.add_argument("--out-root", required=True, type=str)
    parser.add_argument("--config-dir", type=str, default="configs/ablation")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--shape-manifest", type=str, default="")
    parser.add_argument("--sample-class-manifest", action="append", default=[])
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--test-split-name", type=str, default="test")
    parser.add_argument("--synth-root", type=str, default="")
    parser.add_argument("--hard-manifest", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--val-workers", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    if args.config:
        config_names = list(args.config)
    else:
        config_names = sorted(path.stem for path in config_dir.glob("*.json"))
    seeds = [int(chunk.strip()) for chunk in str(args.seeds).split(",") if chunk.strip()]
    base_train_kwargs = {
        "data_root": args.data_root,
        "out": "",
        "alphabet": args.alphabet,
        "batch_size": args.batch_size,
        "train_workers": args.train_workers,
        "val_workers": args.val_workers,
        "shape_manifest": args.shape_manifest,
        "sample_class_manifests": tuple(args.sample_class_manifest),
        "split_manifest": args.split_manifest,
        "synth_root": args.synth_root,
        "hard_manifest": args.hard_manifest,
        "amp": args.amp,
    }
    base_eval_kwargs = {
        "data_root": args.data_root,
        "checkpoint": "",
        "alphabet": args.alphabet,
        "out": "",
        "workers": args.eval_workers,
        "shape_manifest": args.shape_manifest,
        "sample_class_manifests": tuple(args.sample_class_manifest),
        "split_manifest": args.split_manifest,
        "split_name": args.test_split_name if args.split_manifest else "",
    }
    out_root = Path(args.out_root)
    run_results, aggregate_rows = run_ablations(
        config_dir=config_dir,
        config_names=config_names,
        seeds=seeds,
        base_train_kwargs=base_train_kwargs,
        base_eval_kwargs=base_eval_kwargs,
        out_root=out_root,
    )
    runs_csv, summary_csv, summary_json = write_ablation_reports(out_root, run_results, aggregate_rows)
    print(f"[INFO] Run table: {runs_csv}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
