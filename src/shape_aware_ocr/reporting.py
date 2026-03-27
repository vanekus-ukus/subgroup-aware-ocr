from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _metric_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _fmt_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def scan_experiment_root(experiment_name: str, root: Path, eval_dir_name: str = "eval") -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    eval_dir_name = str(eval_dir_name or "eval").strip()
    for report_path in sorted(root.glob(f"*/seed_*/{eval_dir_name}/eval_report.json")):
        seed_dir = report_path.parent.parent
        config_dir = seed_dir.parent
        config_name = config_dir.name
        seed = int(seed_dir.name.split("_", 1)[1])
        report = _load_json(report_path)
        run_rows.append(
            {
                "experiment_name": experiment_name,
                "config_name": config_name,
                "seed": seed,
                "report_path": str(report_path),
                "cer": float(report["cer"]),
                "exact_match": float(report["exact_match"]),
                "cer_square": float(report["cer_square"]),
                "cer_rect": float(report["cer_rect"]),
                "weighted_shape_cer": float(report["weighted_shape_cer"]),
                "macro_shape_cer": float(report["macro_shape_cer"]),
                "cer_bootstrap_mean": float(report.get("bootstrap", {}).get("cer", {}).get("mean", float("nan"))),
                "cer_bootstrap_lower": float(report.get("bootstrap", {}).get("cer", {}).get("lower", float("nan"))),
                "cer_bootstrap_upper": float(report.get("bootstrap", {}).get("cer", {}).get("upper", float("nan"))),
                "exact_bootstrap_mean": float(report.get("bootstrap", {}).get("exact", {}).get("mean", float("nan"))),
                "exact_bootstrap_lower": float(report.get("bootstrap", {}).get("exact", {}).get("lower", float("nan"))),
                "exact_bootstrap_upper": float(report.get("bootstrap", {}).get("exact", {}).get("upper", float("nan"))),
                "samples": int(report.get("samples", 0)),
                "chars": int(report.get("chars", 0)),
            }
        )

        subgroup_path = report_path.parent / "subgroup_metrics.csv"
        if subgroup_path.exists():
            for row in _read_csv(subgroup_path):
                subgroup_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "config_name": config_name,
                        "seed": seed,
                        "group": row["group"],
                        "cer": float(row["cer"]),
                        "exact": float(row["exact"]),
                        "samples": int(row["samples"]),
                        "chars": int(row["chars"]),
                    }
                )
    return run_rows, subgroup_rows


def aggregate_run_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["experiment_name"]), str(row["config_name"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (experiment_name, config_name), subset in sorted(grouped.items()):
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "config_name": config_name,
                "runs": len(subset),
                "cer_mean": _metric_mean([float(row["cer"]) for row in subset]),
                "cer_std": _metric_std([float(row["cer"]) for row in subset]),
                "exact_mean": _metric_mean([float(row["exact_match"]) for row in subset]),
                "exact_std": _metric_std([float(row["exact_match"]) for row in subset]),
                "cer_square_mean": _metric_mean([float(row["cer_square"]) for row in subset]),
                "cer_square_std": _metric_std([float(row["cer_square"]) for row in subset]),
                "cer_rect_mean": _metric_mean([float(row["cer_rect"]) for row in subset]),
                "cer_rect_std": _metric_std([float(row["cer_rect"]) for row in subset]),
                "weighted_shape_cer_mean": _metric_mean([float(row["weighted_shape_cer"]) for row in subset]),
                "weighted_shape_cer_std": _metric_std([float(row["weighted_shape_cer"]) for row in subset]),
                "macro_shape_cer_mean": _metric_mean([float(row["macro_shape_cer"]) for row in subset]),
                "macro_shape_cer_std": _metric_std([float(row["macro_shape_cer"]) for row in subset]),
                "samples_mean": _metric_mean([float(row["samples"]) for row in subset]),
                "chars_mean": _metric_mean([float(row["chars"]) for row in subset]),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["experiment_name"]), float(row["weighted_shape_cer_mean"])))
    return summary_rows


def aggregate_subgroup_rows(subgroup_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in subgroup_rows:
        key = (str(row["experiment_name"]), str(row["config_name"]), str(row["group"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (experiment_name, config_name, group_name), subset in sorted(grouped.items()):
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "config_name": config_name,
                "group": group_name,
                "runs": len(subset),
                "cer_mean": _metric_mean([float(row["cer"]) for row in subset]),
                "cer_std": _metric_std([float(row["cer"]) for row in subset]),
                "exact_mean": _metric_mean([float(row["exact"]) for row in subset]),
                "exact_std": _metric_std([float(row["exact"]) for row in subset]),
                "samples_mean": _metric_mean([float(row["samples"]) for row in subset]),
                "chars_mean": _metric_mean([float(row["chars"]) for row in subset]),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["experiment_name"]), str(row["config_name"]), str(row["group"])))
    return summary_rows


def pairwise_config_deltas(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["experiment_name"]), str(row["config_name"]))
        grouped.setdefault(key, {})[int(row["seed"])] = row

    metrics = ("cer", "exact_match", "cer_square", "cer_rect", "weighted_shape_cer", "macro_shape_cer")
    delta_rows: list[dict[str, object]] = []
    experiment_names = sorted({str(row["experiment_name"]) for row in run_rows})
    for experiment_name in experiment_names:
        config_names = sorted({str(row["config_name"]) for row in run_rows if row["experiment_name"] == experiment_name})
        for idx, left_config in enumerate(config_names):
            left_seed_rows = grouped.get((experiment_name, left_config), {})
            for right_config in config_names[idx + 1 :]:
                right_seed_rows = grouped.get((experiment_name, right_config), {})
                common_seeds = sorted(set(left_seed_rows) & set(right_seed_rows))
                if not common_seeds:
                    continue
                row: dict[str, object] = {
                    "experiment_name": experiment_name,
                    "left_config": left_config,
                    "right_config": right_config,
                    "runs": len(common_seeds),
                    "seeds": ",".join(str(seed) for seed in common_seeds),
                }
                for metric in metrics:
                    deltas = [float(left_seed_rows[seed][metric]) - float(right_seed_rows[seed][metric]) for seed in common_seeds]
                    row[f"{metric}_delta_mean"] = _metric_mean(deltas)
                    row[f"{metric}_delta_std"] = _metric_std(deltas)
                delta_rows.append(row)
    delta_rows.sort(key=lambda row: (str(row["experiment_name"]), str(row["left_config"]), str(row["right_config"])))
    return delta_rows


def pairwise_experiment_deltas(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["experiment_name"]), str(row["config_name"]))
        grouped.setdefault(key, {})[int(row["seed"])] = row

    metrics = ("cer", "exact_match", "cer_square", "cer_rect", "weighted_shape_cer", "macro_shape_cer")
    delta_rows: list[dict[str, object]] = []
    config_names = sorted({str(row["config_name"]) for row in run_rows})
    experiment_names = sorted({str(row["experiment_name"]) for row in run_rows})
    for config_name in config_names:
        available_experiments = [name for name in experiment_names if (name, config_name) in grouped]
        for idx, left_experiment in enumerate(available_experiments):
            left_seed_rows = grouped.get((left_experiment, config_name), {})
            for right_experiment in available_experiments[idx + 1 :]:
                right_seed_rows = grouped.get((right_experiment, config_name), {})
                common_seeds = sorted(set(left_seed_rows) & set(right_seed_rows))
                if not common_seeds:
                    continue
                row: dict[str, object] = {
                    "config_name": config_name,
                    "left_experiment": left_experiment,
                    "right_experiment": right_experiment,
                    "runs": len(common_seeds),
                    "seeds": ",".join(str(seed) for seed in common_seeds),
                }
                for metric in metrics:
                    deltas = [float(left_seed_rows[seed][metric]) - float(right_seed_rows[seed][metric]) for seed in common_seeds]
                    row[f"{metric}_delta_mean"] = _metric_mean(deltas)
                    row[f"{metric}_delta_std"] = _metric_std(deltas)
                delta_rows.append(row)
    delta_rows.sort(key=lambda row: (str(row["config_name"]), str(row["left_experiment"]), str(row["right_experiment"])))
    return delta_rows


def build_source_class_manifest(
    benchmark_source_manifest: Path,
    imported_manifest: Path,
    out_path: Path,
    native_class_name: str = "native_real",
) -> Path:
    imported_rows = _read_csv(imported_manifest)
    imported_by_output: dict[str, str] = {}
    imported_by_file: dict[str, str] = {}
    for row in imported_rows:
        origin = str(row.get("origin_dataset", "")).strip().lower()
        if not origin:
            continue
        class_name = f"imported_{origin}"
        output_name = Path(row.get("output_path", row.get("file", ""))).name
        file_name = Path(row.get("file", output_name)).name
        if output_name:
            imported_by_output[output_name] = class_name
        if file_name:
            imported_by_file[file_name] = class_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "match_key", "class_name"])
        writer.writeheader()
        for row in _read_csv(benchmark_source_manifest):
            source_name = Path(row["source_path"]).name
            file_name = row["file"]
            class_name = imported_by_output.get(source_name) or imported_by_file.get(source_name) or imported_by_file.get(file_name) or native_class_name
            writer.writerow({"file": file_name, "match_key": row["match_key"], "class_name": class_name})
    return out_path


def _find_summary_row(summary_rows: list[dict[str, object]], experiment_name: str, config_name: str) -> dict[str, object] | None:
    for row in summary_rows:
        if row["experiment_name"] == experiment_name and row["config_name"] == config_name:
            return row
    return None


def _find_first_summary_row(summary_rows: list[dict[str, object]], experiment_name: str, config_names: tuple[str, ...]) -> dict[str, object] | None:
    for config_name in config_names:
        row = _find_summary_row(summary_rows, experiment_name, config_name)
        if row is not None:
            return row
    return None


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown_report(
    title: str,
    benchmark_summary: dict[str, object],
    shape_reports: dict[str, dict[str, object]],
    summary_rows: list[dict[str, object]],
    subgroup_summary_rows: list[dict[str, object]],
    config_delta_rows: list[dict[str, object]],
    experiment_delta_rows: list[dict[str, object]],
    experiment_roots: dict[str, str],
    figures: list[str] | None = None,
) -> str:
    lines: list[str] = [f"# {title}", ""]

    lines.extend(
        [
            "## Privacy And Data Handling",
            "",
            "- Full article-grade experiments use a non-public benchmark that is intentionally excluded from version control.",
            "- Reports expose aggregate metrics, manifests, and optional sanitized example crops; no raw training corpus is published.",
            "- The benchmark is constructed as a fixed `train/val/test` split so every ablation is directly comparable.",
            "",
            "## Benchmark",
            "",
            f"- Total samples: `{benchmark_summary.get('total_selected', benchmark_summary.get('total_samples', 'n/a'))}`",
            f"- Shape balance: `rect={benchmark_summary.get('rect_selected', benchmark_summary.get('rect_count', 'n/a'))}`, `square={benchmark_summary.get('square_selected', benchmark_summary.get('square_count', 'n/a'))}`",
            f"- Split sizes: `train={benchmark_summary.get('train_count', 'n/a')}`, `val={benchmark_summary.get('val_count', 'n/a')}`, `test={benchmark_summary.get('test_count', 'n/a')}`",
            f"- Style-labeled subset: `{benchmark_summary.get('style_labeled_count', 'n/a')}`",
            f"- Copy mode: `{benchmark_summary.get('copy_mode', 'n/a')}`",
            "",
        ]
    )

    if shape_reports:
        lines.extend(["## Shape Label Transfer", ""])
        table_rows: list[list[str]] = []
        for name, report in sorted(shape_reports.items()):
            table_rows.append(
                [
                    name,
                    str(report.get("samples", "n/a")),
                    _fmt_metric(float(report.get("acc", float("nan")))),
                    _fmt_metric(float(report.get("balanced_acc", float("nan")))),
                    _fmt_metric(float(report.get("square_recall", float("nan")))),
                    _fmt_metric(float(report.get("rect_recall", float("nan")))),
                ]
            )
        lines.append(
            _markdown_table(
                ["Label Source", "Samples", "Accuracy", "Balanced Accuracy", "Square Recall", "Rect Recall"],
                table_rows,
            )
        )
        lines.append("")

    if summary_rows:
        lines.extend(["## Main Ablations", ""])
        main_rows = [
            [
                str(row["experiment_name"]),
                str(row["config_name"]),
                str(row["runs"]),
                _fmt_metric(float(row["cer_mean"])),
                _fmt_metric(float(row["cer_std"])),
                _fmt_metric(float(row["cer_square_mean"])),
                _fmt_metric(float(row["cer_rect_mean"])),
                _fmt_metric(float(row["weighted_shape_cer_mean"])),
                _fmt_metric(float(row["macro_shape_cer_mean"])),
                _fmt_metric(float(row["exact_mean"])),
            ]
            for row in summary_rows
        ]
        lines.append(
            _markdown_table(
                [
                    "Experiment",
                    "Config",
                    "Runs",
                    "CER Mean",
                    "CER Std",
                    "Square CER",
                    "Rect CER",
                    "Weighted CER",
                    "Macro CER",
                    "Exact",
                ],
                main_rows,
            )
        )
        lines.append("")

        lines.extend(["## Findings", ""])
        for experiment_name in sorted({str(row["experiment_name"]) for row in summary_rows}):
            subset = [row for row in summary_rows if row["experiment_name"] == experiment_name]
            if not subset:
                continue
            best = min(subset, key=lambda row: float(row["weighted_shape_cer_mean"]))
            lines.append(
                f"- `{experiment_name}` best weighted-shape result: `{best['config_name']}` with `weighted_shape_cer={_fmt_metric(float(best['weighted_shape_cer_mean']))}`."
            )
            baseline = _find_first_summary_row(summary_rows, experiment_name, ("pilot_baseline", "baseline"))
            shape_weighted = _find_first_summary_row(summary_rows, experiment_name, ("pilot_shape_weighted", "shape_weighted"))
            synth = _find_first_summary_row(summary_rows, experiment_name, ("pilot_synth_curriculum", "synthetic_curriculum"))
            legacy = _find_first_summary_row(summary_rows, experiment_name, ("pilot_legacy_baseline", "legacy_baseline"))
            if baseline and shape_weighted:
                delta = float(shape_weighted["weighted_shape_cer_mean"]) - float(baseline["weighted_shape_cer_mean"])
                lines.append(
                    f"- `{experiment_name}` shape-aware selection vs CER baseline: `delta_weighted_shape_cer={delta:+.4f}`."
                )
            if shape_weighted and synth:
                delta = float(synth["weighted_shape_cer_mean"]) - float(shape_weighted["weighted_shape_cer_mean"])
                lines.append(
                    f"- `{experiment_name}` synthetic curriculum vs shape-aware real-only: `delta_weighted_shape_cer={delta:+.4f}`."
                )
            if baseline and legacy:
                delta = float(legacy["cer_mean"]) - float(baseline["cer_mean"])
                lines.append(f"- `{experiment_name}` legacy preprocess vs stand preprocess: `delta_cer={delta:+.4f}`.")
        if len({str(row["experiment_name"]) for row in summary_rows}) >= 2:
            oracle_shape = _find_first_summary_row(summary_rows, "oracle_shape", ("pilot_shape_weighted", "shape_weighted"))
            predicted_shape = _find_first_summary_row(summary_rows, "predicted_shape", ("pilot_shape_weighted", "shape_weighted"))
            if oracle_shape and predicted_shape:
                delta = float(predicted_shape["weighted_shape_cer_mean"]) - float(oracle_shape["weighted_shape_cer_mean"])
                lines.append(f"- Bootstrapped shape labels vs oracle on `shape_weighted`: `delta_weighted_shape_cer={delta:+.4f}`.")
            oracle_synth = _find_first_summary_row(summary_rows, "oracle_shape", ("pilot_synth_curriculum", "synthetic_curriculum"))
            predicted_synth = _find_first_summary_row(summary_rows, "predicted_shape", ("pilot_synth_curriculum", "synthetic_curriculum"))
            if oracle_synth and predicted_synth:
                delta = float(predicted_synth["weighted_shape_cer_mean"]) - float(oracle_synth["weighted_shape_cer_mean"])
                lines.append(f"- Bootstrapped shape labels vs oracle on `synthetic_curriculum`: `delta_weighted_shape_cer={delta:+.4f}`.")
        lines.append("")

    if config_delta_rows:
        lines.extend(["## Paired Config Deltas", ""])
        rows = [
            [
                str(row["experiment_name"]),
                str(row["left_config"]),
                str(row["right_config"]),
                str(row["runs"]),
                _fmt_metric(float(row["weighted_shape_cer_delta_mean"])),
                _fmt_metric(float(row["weighted_shape_cer_delta_std"])),
                _fmt_metric(float(row["cer_delta_mean"])),
                _fmt_metric(float(row["exact_match_delta_mean"])),
            ]
            for row in config_delta_rows
        ]
        lines.append(
            _markdown_table(
                [
                    "Experiment",
                    "Left Config",
                    "Right Config",
                    "Runs",
                    "Weighted CER Delta",
                    "Weighted CER Delta Std",
                    "CER Delta",
                    "Exact Delta",
                ],
                rows,
            )
        )
        lines.append("")

    if experiment_delta_rows:
        lines.extend(["## Paired Experiment Deltas", ""])
        rows = [
            [
                str(row["config_name"]),
                str(row["left_experiment"]),
                str(row["right_experiment"]),
                str(row["runs"]),
                _fmt_metric(float(row["weighted_shape_cer_delta_mean"])),
                _fmt_metric(float(row["weighted_shape_cer_delta_std"])),
                _fmt_metric(float(row["cer_delta_mean"])),
                _fmt_metric(float(row["exact_match_delta_mean"])),
            ]
            for row in experiment_delta_rows
        ]
        lines.append(
            _markdown_table(
                [
                    "Config",
                    "Left Experiment",
                    "Right Experiment",
                    "Runs",
                    "Weighted CER Delta",
                    "Weighted CER Delta Std",
                    "CER Delta",
                    "Exact Delta",
                ],
                rows,
            )
        )
        lines.append("")

    sample_class_rows = [row for row in subgroup_summary_rows if str(row["group"]).startswith("sample_class:")]
    if sample_class_rows:
        lines.extend(["## Sample-Class Breakdown", ""])
        rows = [
            [
                str(row["experiment_name"]),
                str(row["config_name"]),
                str(row["group"]).split(":", 1)[1],
                str(row["runs"]),
                _fmt_metric(float(row["cer_mean"])),
                _fmt_metric(float(row["cer_std"])),
                _fmt_metric(float(row["exact_mean"])),
                f"{float(row['samples_mean']):.1f}",
            ]
            for row in sample_class_rows
        ]
        lines.append(
            _markdown_table(
                ["Experiment", "Config", "Class", "Runs", "CER Mean", "CER Std", "Exact Mean", "Samples Mean"],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Artifacts", ""])
    for name, root in sorted(experiment_roots.items()):
        lines.append(f"- `{name}` experiment root: `{root}`")
    if figures:
        for figure in figures:
            lines.append(f"- Figure: `{figure}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = list(fieldnames or (list(rows[0].keys()) if rows else []))
    if not rows and not resolved_fieldnames:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return path
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return path
