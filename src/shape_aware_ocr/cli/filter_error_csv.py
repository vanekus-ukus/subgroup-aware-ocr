from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter OCR error rows by shape and/or sample-class substring")
    parser.add_argument("--errors-csv", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--shape", choices=["square", "rect"], default="")
    parser.add_argument("--sample-class-contains", action="append", default=[])
    parser.add_argument("--top-n", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_rows(Path(args.errors_csv))
    if args.shape:
        wanted_shape = str(args.shape).strip().lower()
        rows = [row for row in rows if str(row.get("shape", "")).strip().lower() == wanted_shape]
    for token in args.sample_class_contains:
        token = str(token).strip()
        if not token:
            continue
        rows = [row for row in rows if token in str(row.get("sample_class", ""))]
    rows.sort(key=lambda row: float(row.get("norm_ed", "0") or 0.0), reverse=True)
    if int(args.top_n) > 0:
        rows = rows[: int(args.top_n)]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["index", "gt", "pred", "norm_ed", "shape", "sample_class", "file"]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    print(f"[INFO] Filtered rows: {len(rows)}")
    print(f"[INFO] Output CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
