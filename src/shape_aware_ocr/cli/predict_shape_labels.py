from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.shape_classifier import predict_shape_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rect/square labels for sequence crops")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_csv = predict_shape_labels(Path(args.data_root), Path(args.checkpoint), Path(args.out))
    print(f"[INFO] Saved predictions to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
