from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.reporting import build_source_class_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample-class manifest that captures data provenance on a private benchmark")
    parser.add_argument("--benchmark-source-manifest", required=True, type=str)
    parser.add_argument("--imported-manifest", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--native-class-name", type=str, default="native_real")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = build_source_class_manifest(
        benchmark_source_manifest=Path(args.benchmark_source_manifest),
        imported_manifest=Path(args.imported_manifest),
        out_path=Path(args.out),
        native_class_name=args.native_class_name,
    )
    print(f"[INFO] Source class manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
