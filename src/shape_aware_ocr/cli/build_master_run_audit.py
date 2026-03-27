from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from shape_aware_ocr.reporting import write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


RUN_FIELDNAMES = [
    "study",
    "family_id",
    "comparison_family",
    "experiment_name",
    "config_name",
    "seed",
    "run_dir",
    "started",
    "finished",
    "raw_eval_complete",
    "fair_eval_complete",
    "status_label",
    "summary_package",
    "included_in_summary",
    "comparable_to_family",
    "article_grade",
    "best_by",
    "preprocess_mode",
    "split_manifest",
    "shape_manifest",
    "fair_eval_dir",
    "notes",
]


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary_index(path: Path) -> set[tuple[str, str, str]]:
    rows = _read_csv(path)
    return {
        (
            str(row["experiment_name"]).strip(),
            str(row["config_name"]).strip(),
            str(row["seed"]).strip(),
        )
        for row in rows
    }


def _load_train_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _status_label(run_config_exists: bool, train_rows: list[dict[str, str]], raw_eval_exists: bool, fair_eval_exists: bool) -> str:
    if fair_eval_exists:
        return "completed_fair_eval"
    if raw_eval_exists:
        return "completed_raw_eval_only"
    if train_rows:
        return "in_progress"
    if run_config_exists:
        return "configured_not_started"
    return "missing"


def _finished(status_label: str) -> bool:
    return status_label in {"completed_fair_eval", "completed_raw_eval_only"}


def _comparison_family(study: str, family_id: str) -> str:
    if study == "main":
        return "main_headline"
    if family_id == "pilot_predicted":
        return "pilot_predicted"
    if family_id == "pilot_oracle":
        return "pilot_oracle"
    return "pilot_control"


def _comparable_to_family(study: str, config_name: str, run_config: dict, fair_eval_exists: bool) -> tuple[bool, str]:
    preprocess_mode = str(run_config.get("preprocess_mode", "stand"))
    split_manifest = str(run_config.get("split_manifest", ""))
    best_by = str(run_config.get("best_by", ""))
    shape_manifest = str(run_config.get("shape_manifest", ""))

    if not fair_eval_exists:
        return False, "fair eval missing"
    if not split_manifest:
        return False, "split manifest missing"
    if study == "main":
        if config_name not in {"shape_weighted", "synthetic_static", "synthetic_curriculum"}:
            return False, "not in main headline config family"
        if preprocess_mode != "stand":
            return False, "preprocess mismatch"
        if best_by != "weighted_shape":
            return False, "best_by mismatch"
        if "predicted_shape_manifest_transfer.csv" not in shape_manifest:
            return False, "training shape manifest mismatch"
        return True, "comparable inside main headline family"
    if preprocess_mode == "legacy":
        return True, "legacy control on same benchmark but different preprocess"
    return True, "comparable inside pilot family"


def _run_rows_for_family(
    study: str,
    family_id: str,
    experiment_name: str,
    experiment_root: Path,
    summary_csv: Path,
    fair_eval_dir_name: str,
) -> list[dict[str, object]]:
    summary_index = _load_summary_index(summary_csv) if summary_csv.exists() else set()
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in experiment_root.glob("*/seed_*") if path.is_dir()):
        config_name = run_dir.parent.name
        seed = int(run_dir.name.split("_", 1)[1])
        run_config_path = run_dir / "run_config.json"
        run_config = _read_json(run_config_path) if run_config_path.exists() else {}
        train_rows = _load_train_rows(run_dir / "train_log.csv")
        raw_eval_exists = (run_dir / "eval" / "eval_report.json").exists()
        fair_eval_exists = (run_dir / fair_eval_dir_name / "eval_report.json").exists()
        status = _status_label(run_config_path.exists(), train_rows, raw_eval_exists, fair_eval_exists)
        started = bool(train_rows or raw_eval_exists or fair_eval_exists)
        finished = _finished(status)
        included = (str(experiment_name).strip(), str(config_name).strip(), str(seed).strip()) in summary_index
        comparable, comparable_reason = _comparable_to_family(study, config_name, run_config, fair_eval_exists)
        article_grade = fair_eval_exists and comparable
        notes = comparable_reason
        if status == "configured_not_started":
            notes = "run_config materialized; training not started"
        elif status == "completed_raw_eval_only":
            notes = "raw eval exists; fair oracle reevaluation still missing"
        elif status == "in_progress":
            notes = "training active or incomplete"
        rows.append(
            {
                "study": study,
                "family_id": family_id,
                "comparison_family": _comparison_family(study, family_id),
                "experiment_name": experiment_name,
                "config_name": config_name,
                "seed": seed,
                "run_dir": str(run_dir),
                "started": _bool_text(started),
                "finished": _bool_text(finished),
                "raw_eval_complete": _bool_text(raw_eval_exists),
                "fair_eval_complete": _bool_text(fair_eval_exists),
                "status_label": status,
                "summary_package": str(summary_csv),
                "included_in_summary": _bool_text(included),
                "comparable_to_family": _bool_text(comparable),
                "article_grade": _bool_text(article_grade),
                "best_by": str(run_config.get("best_by", "")),
                "preprocess_mode": str(run_config.get("preprocess_mode", "stand")),
                "split_manifest": str(run_config.get("split_manifest", "")),
                "shape_manifest": str(run_config.get("shape_manifest", "")),
                "fair_eval_dir": fair_eval_dir_name if fair_eval_exists else "",
                "notes": notes,
            }
        )
    return rows


def _render_markdown(rows: list[dict[str, object]]) -> str:
    lines: list[str] = ["# Completed Vs Incomplete Runs", ""]
    status_counts = Counter(str(row["status_label"]) for row in rows)
    lines.extend(["## Status Counts", ""])
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.append("")

    lines.extend(["## By Family", ""])
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    for family_id, subset in sorted(by_family.items()):
        lines.append(f"### {family_id}")
        lines.append("")
        lines.append("| Config | Seed | Status | Started | Raw Eval | Fair Eval | Included In Summary | Article Grade | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in sorted(subset, key=lambda item: (str(item["config_name"]), int(item["seed"]))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["config_name"]),
                        str(row["seed"]),
                        str(row["status_label"]),
                        str(row["started"]),
                        str(row["raw_eval_complete"]),
                        str(row["fair_eval_complete"]),
                        str(row["included_in_summary"]),
                        str(row["article_grade"]),
                        str(row["notes"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a master completed-vs-incomplete audit across pilot and main OCR runs")
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--out-md", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = [
        {
            "study": "pilot",
            "family_id": "pilot_predicted",
            "experiment_name": "predicted_shape",
            "root": PROJECT_ROOT / "outputs" / "ablations_eu_pilot_predicted",
            "summary_csv": PROJECT_ROOT / "reports" / "eu_article_pilot" / "experiment_runs.csv",
            "fair_eval_dir_name": "eval_multiaxis",
        },
        {
            "study": "pilot",
            "family_id": "pilot_oracle",
            "experiment_name": "oracle_shape",
            "root": PROJECT_ROOT / "outputs" / "ablations_eu_pilot_oracle",
            "summary_csv": PROJECT_ROOT / "reports" / "eu_article_pilot" / "experiment_runs.csv",
            "fair_eval_dir_name": "eval_multiaxis",
        },
        {
            "study": "pilot",
            "family_id": "pilot_legacy",
            "experiment_name": "legacy_preprocess",
            "root": PROJECT_ROOT / "outputs" / "ablations_eu_pilot_legacy",
            "summary_csv": PROJECT_ROOT / "reports" / "eu_article_pilot" / "experiment_runs.csv",
            "fair_eval_dir_name": "eval_multiaxis",
        },
        {
            "study": "main",
            "family_id": "main_predicted",
            "experiment_name": "predicted_shape_main",
            "root": PROJECT_ROOT / "outputs" / "ablations_eu_main_predicted",
            "summary_csv": PROJECT_ROOT / "reports" / "eu_article_main" / "experiment_runs.csv",
            "fair_eval_dir_name": "eval_multiaxis",
        },
    ]

    rows: list[dict[str, object]] = []
    for family in families:
        rows.extend(
            _run_rows_for_family(
                study=str(family["study"]),
                family_id=str(family["family_id"]),
                experiment_name=str(family["experiment_name"]),
                experiment_root=Path(family["root"]),
                summary_csv=Path(family["summary_csv"]),
                fair_eval_dir_name=str(family["fair_eval_dir_name"]),
            )
        )

    rows.sort(key=lambda row: (str(row["study"]), str(row["family_id"]), str(row["config_name"]), int(row["seed"])))
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    write_csv(out_csv, rows, fieldnames=RUN_FIELDNAMES)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"[INFO] Master run audit CSV: {out_csv}")
    print(f"[INFO] Master run audit MD: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
