from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shape_aware_ocr.reporting import write_csv


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_train_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_best(rows: list[dict[str, str]], metric_name: str) -> tuple[int | None, float | None]:
    best_epoch = None
    best_value = None
    for row in rows:
        try:
            epoch = int(row["epoch"])
            value = float(row[metric_name])
        except Exception:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_epoch = epoch
    return best_epoch, best_value


def _summarize_run(run_dir: Path, fair_eval_dir_name: str) -> dict[str, object]:
    config_name = run_dir.parent.name
    seed_text = run_dir.name.split("_", 1)[1] if "_" in run_dir.name else run_dir.name
    row: dict[str, object] = {
        "config_name": config_name,
        "seed": int(seed_text),
        "run_dir": str(run_dir),
        "target_epochs": None,
        "logged_epochs": 0,
        "last_epoch": None,
        "train_complete": False,
        "raw_eval_complete": False,
        "fair_eval_complete": False,
        "status": "missing",
    }

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = _load_json(run_config_path)
        row["target_epochs"] = int(run_config.get("epochs", 0) or 0)
        row["best_by"] = str(run_config.get("best_by", ""))
        row["train_shape_manifest"] = str(run_config.get("shape_manifest", ""))

    train_log_path = run_dir / "train_log.csv"
    if train_log_path.exists():
        train_rows = _load_train_rows(train_log_path)
        row["logged_epochs"] = len(train_rows)
        if train_rows:
            row["last_epoch"] = int(train_rows[-1]["epoch"])
            row["last_val_cer"] = float(train_rows[-1]["val_cer"])
            row["last_val_weighted_shape"] = float(train_rows[-1]["val_cer_weighted_shape"])
            row["last_val_exact"] = float(train_rows[-1]["val_exact"])
            best_cer_epoch, best_cer = _find_best(train_rows, "val_cer")
            best_weighted_epoch, best_weighted = _find_best(train_rows, "val_cer_weighted_shape")
            row["best_val_cer_epoch"] = best_cer_epoch
            row["best_val_cer"] = best_cer
            row["best_val_weighted_epoch"] = best_weighted_epoch
            row["best_val_weighted_shape"] = best_weighted
        target_epochs = int(row.get("target_epochs") or 0)
        if target_epochs > 0 and int(row["logged_epochs"]) >= target_epochs:
            row["train_complete"] = True

    raw_eval_path = run_dir / "eval" / "eval_report.json"
    if raw_eval_path.exists():
        raw_eval = _load_json(raw_eval_path)
        row["raw_eval_complete"] = True
        row["raw_eval_cer"] = float(raw_eval["cer"])
        row["raw_eval_exact"] = float(raw_eval["exact_match"])
        row["raw_eval_weighted_shape"] = float(raw_eval["weighted_shape_cer"])
        row["raw_eval_macro_shape"] = float(raw_eval["macro_shape_cer"])

    fair_eval_path = run_dir / fair_eval_dir_name / "eval_report.json"
    if fair_eval_path.exists():
        fair_eval = _load_json(fair_eval_path)
        row["fair_eval_complete"] = True
        row["fair_eval_dir"] = fair_eval_dir_name
        row["fair_eval_cer"] = float(fair_eval["cer"])
        row["fair_eval_exact"] = float(fair_eval["exact_match"])
        row["fair_eval_weighted_shape"] = float(fair_eval["weighted_shape_cer"])
        row["fair_eval_macro_shape"] = float(fair_eval["macro_shape_cer"])

    if row["fair_eval_complete"]:
        row["status"] = "completed_fair_eval"
    elif row["raw_eval_complete"]:
        row["status"] = "completed_raw_eval"
    elif row["train_complete"]:
        row["status"] = "completed_train"
    elif train_log_path.exists():
        row["status"] = "in_progress"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit run status and metrics under an ablation experiment root")
    parser.add_argument("--experiment-root", required=True, type=str)
    parser.add_argument("--fair-eval-dir-name", type=str, default="eval_multiaxis")
    parser.add_argument("--out", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    run_dirs = sorted(path for path in experiment_root.glob("*/seed_*") if path.is_dir())
    rows = [_summarize_run(run_dir, args.fair_eval_dir_name) for run_dir in run_dirs]
    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"[INFO] Audit CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
