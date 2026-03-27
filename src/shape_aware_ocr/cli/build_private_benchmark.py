from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.benchmark import build_private_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-split private benchmark from local sequence crops")
    parser.add_argument("--src-root", required=True, type=str)
    parser.add_argument("--shape-manifest", required=True, type=str)
    parser.add_argument("--sample-class-manifest", type=str, default="")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--rect-count", type=int, default=6000)
    parser.add_argument("--square-count", type=int, default=3000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--min-label-len", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_private_benchmark(
        src_root=Path(args.src_root),
        shape_manifest=Path(args.shape_manifest),
        sample_class_manifest=Path(args.sample_class_manifest) if args.sample_class_manifest else None,
        out_root=Path(args.out),
        rect_count=args.rect_count,
        square_count=args.square_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_mode=args.copy_mode,
        min_label_len=args.min_label_len,
        clean=args.clean,
    )
    print(f"[INFO] Benchmark root: {summary.out_root}")
    print(f"[INFO] Selected: total={summary.total_selected}, rect={summary.rect_count}, square={summary.square_count}")
    print(f"[INFO] Split counts: train={summary.train_count}, val={summary.val_count}, test={summary.test_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
