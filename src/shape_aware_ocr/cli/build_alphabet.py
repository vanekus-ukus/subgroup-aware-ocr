from __future__ import annotations

import argparse
from pathlib import Path

from shape_aware_ocr.alphabet import build_alphabet_from_root, save_alphabet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alphabet from filename-labeled sequence crops")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_alphabet_from_root(Path(args.data_root))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_alphabet(payload, out_dir / "alphabet.json")
    print(f"[INFO] Saved alphabet to {out_dir / 'alphabet.json'}")
    print(f"[INFO] Tokens: {''.join(payload['tokens'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
