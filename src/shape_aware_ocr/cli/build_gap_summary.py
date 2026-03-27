from __future__ import annotations

import argparse
import csv
from pathlib import Path

from shape_aware_ocr.reporting import write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggregate-vs-subgroup gap summaries from report CSVs")
    parser.add_argument("--experiment-summary", required=True, type=str)
    parser.add_argument("--subgroup-summary", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    experiment_rows = _read_csv(Path(args.experiment_summary))
    subgroup_rows = _read_csv(Path(args.subgroup_summary))

    overall_by_key: dict[tuple[str, str], float] = {}
    shape_rows: list[dict[str, object]] = []
    for row in experiment_rows:
        key = (row["experiment_name"], row["config_name"])
        overall_cer = float(row["cer_mean"])
        overall_by_key[key] = overall_cer
        square_cer = float(row["cer_square_mean"])
        rect_cer = float(row["cer_rect_mean"])
        shape_rows.append(
            {
                "experiment_name": row["experiment_name"],
                "config_name": row["config_name"],
                "group": "shape:square",
                "cer_mean": square_cer,
                "overall_cer_mean": overall_cer,
                "gap_to_overall": square_cer - overall_cer,
                "runs": int(row["runs"]),
            }
        )
        shape_rows.append(
            {
                "experiment_name": row["experiment_name"],
                "config_name": row["config_name"],
                "group": "shape:rect",
                "cer_mean": rect_cer,
                "overall_cer_mean": overall_cer,
                "gap_to_overall": rect_cer - overall_cer,
                "runs": int(row["runs"]),
            }
        )

    rows: list[dict[str, object]] = list(shape_rows)
    for row in subgroup_rows:
        key = (row["experiment_name"], row["config_name"])
        if key not in overall_by_key:
            continue
        if row["group"] in {"overall", "square", "rect"}:
            continue
        overall_cer = overall_by_key[key]
        cer_mean = float(row["cer_mean"])
        rows.append(
            {
                "experiment_name": row["experiment_name"],
                "config_name": row["config_name"],
                "group": row["group"],
                "cer_mean": cer_mean,
                "overall_cer_mean": overall_cer,
                "gap_to_overall": cer_mean - overall_cer,
                "runs": int(row["runs"]),
            }
        )

    rows.sort(key=lambda row: (str(row["experiment_name"]), str(row["config_name"]), -abs(float(row["gap_to_overall"])), str(row["group"])))
    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"[INFO] Gap summary CSV: {out_path}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
