from __future__ import annotations

import argparse
import json
from pathlib import Path

from shape_aware_ocr.reporting import (
    aggregate_run_rows,
    aggregate_subgroup_rows,
    pairwise_config_deltas,
    pairwise_experiment_deltas,
    render_markdown_report,
    scan_experiment_root,
    write_csv,
)

RUN_FIELDNAMES = [
    "experiment_name",
    "config_name",
    "seed",
    "report_path",
    "cer",
    "exact_match",
    "cer_square",
    "cer_rect",
    "weighted_shape_cer",
    "macro_shape_cer",
    "cer_bootstrap_mean",
    "cer_bootstrap_lower",
    "cer_bootstrap_upper",
    "exact_bootstrap_mean",
    "exact_bootstrap_lower",
    "exact_bootstrap_upper",
    "samples",
    "chars",
]

SUMMARY_FIELDNAMES = [
    "experiment_name",
    "config_name",
    "runs",
    "cer_mean",
    "cer_std",
    "exact_mean",
    "exact_std",
    "cer_square_mean",
    "cer_square_std",
    "cer_rect_mean",
    "cer_rect_std",
    "weighted_shape_cer_mean",
    "weighted_shape_cer_std",
    "macro_shape_cer_mean",
    "macro_shape_cer_std",
    "samples_mean",
    "chars_mean",
]

SUBGROUP_FIELDNAMES = [
    "experiment_name",
    "config_name",
    "group",
    "runs",
    "cer_mean",
    "cer_std",
    "exact_mean",
    "exact_std",
    "samples_mean",
    "chars_mean",
]

CONFIG_DELTA_FIELDNAMES = [
    "experiment_name",
    "left_config",
    "right_config",
    "runs",
    "seeds",
    "cer_delta_mean",
    "cer_delta_std",
    "exact_match_delta_mean",
    "exact_match_delta_std",
    "cer_square_delta_mean",
    "cer_square_delta_std",
    "cer_rect_delta_mean",
    "cer_rect_delta_std",
    "weighted_shape_cer_delta_mean",
    "weighted_shape_cer_delta_std",
    "macro_shape_cer_delta_mean",
    "macro_shape_cer_delta_std",
]

EXPERIMENT_DELTA_FIELDNAMES = [
    "config_name",
    "left_experiment",
    "right_experiment",
    "runs",
    "seeds",
    "cer_delta_mean",
    "cer_delta_std",
    "exact_match_delta_mean",
    "exact_match_delta_std",
    "cer_square_delta_mean",
    "cer_square_delta_std",
    "cer_rect_delta_mean",
    "cer_rect_delta_std",
    "weighted_shape_cer_delta_mean",
    "weighted_shape_cer_delta_std",
    "macro_shape_cer_delta_mean",
    "macro_shape_cer_delta_std",
]


def _parse_named_paths(values: list[str], option_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option_name} must use name=path syntax: {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f"{option_name} must use non-empty name=path syntax: {value!r}")
        result[name] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate OCR research runs into article-ready CSV and Markdown reports")
    parser.add_argument("--title", type=str, default="Shape-Aware OCR Research Report")
    parser.add_argument("--benchmark-summary", required=True, type=str)
    parser.add_argument("--experiment-root", action="append", required=True, default=[])
    parser.add_argument("--shape-transfer-report", action="append", default=[])
    parser.add_argument("--figure", action="append", default=[])
    parser.add_argument("--eval-dir-name", type=str, default="eval")
    parser.add_argument("--out-root", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_roots = _parse_named_paths(args.experiment_root, "--experiment-root")
    shape_reports_paths = _parse_named_paths(args.shape_transfer_report, "--shape-transfer-report")

    run_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for name, root in sorted(experiment_roots.items()):
        rows, subgroups = scan_experiment_root(name, Path(root), eval_dir_name=args.eval_dir_name)
        run_rows.extend(rows)
        subgroup_rows.extend(subgroups)

    summary_rows = aggregate_run_rows(run_rows)
    subgroup_summary_rows = aggregate_subgroup_rows(subgroup_rows)
    config_delta_rows = pairwise_config_deltas(run_rows)
    experiment_delta_rows = pairwise_experiment_deltas(run_rows)

    benchmark_summary = json.loads(Path(args.benchmark_summary).read_text(encoding="utf-8"))
    shape_reports = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in sorted(shape_reports_paths.items())}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "experiment_runs.csv", run_rows, fieldnames=RUN_FIELDNAMES)
    write_csv(out_root / "experiment_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    write_csv(out_root / "subgroup_summary.csv", subgroup_summary_rows, fieldnames=SUBGROUP_FIELDNAMES)
    write_csv(out_root / "config_pairwise_deltas.csv", config_delta_rows, fieldnames=CONFIG_DELTA_FIELDNAMES)
    write_csv(out_root / "experiment_pairwise_deltas.csv", experiment_delta_rows, fieldnames=EXPERIMENT_DELTA_FIELDNAMES)

    report_text = render_markdown_report(
        title=args.title,
        benchmark_summary=benchmark_summary,
        shape_reports=shape_reports,
        summary_rows=summary_rows,
        subgroup_summary_rows=subgroup_summary_rows,
        config_delta_rows=config_delta_rows,
        experiment_delta_rows=experiment_delta_rows,
        experiment_roots=experiment_roots,
        figures=list(args.figure),
    )
    report_path = out_root / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[INFO] Research report: {report_path}")
    print(f"[INFO] Experiment summary: {out_root / 'experiment_summary.csv'}")
    print(f"[INFO] Subgroup summary: {out_root / 'subgroup_summary.csv'}")
    print(f"[INFO] Config deltas: {out_root / 'config_pairwise_deltas.csv'}")
    print(f"[INFO] Experiment deltas: {out_root / 'experiment_pairwise_deltas.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
