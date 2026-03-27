from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import EvaluationConfig, evaluate_ocr
from .training import TrainingConfig, train_ocr


@dataclass
class AblationRunResult:
    config_name: str
    seed: int
    train_out: str
    eval_out: str
    best_checkpoint: str
    cer: float
    exact_match: float
    cer_square: float
    cer_rect: float
    weighted_shape_cer: float
    macro_shape_cer: float


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _metric_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def run_ablations(
    config_dir: Path,
    config_names: list[str],
    seeds: list[int],
    base_train_kwargs: dict[str, Any],
    base_eval_kwargs: dict[str, Any],
    out_root: Path,
) -> tuple[list[AblationRunResult], list[dict[str, Any]]]:
    out_root.mkdir(parents=True, exist_ok=True)
    run_results: list[AblationRunResult] = []

    for config_name in config_names:
        config_path = config_dir / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        overrides = _load_json(config_path)
        for seed in seeds:
            train_out = out_root / config_name / f"seed_{seed}"
            train_kwargs = dict(base_train_kwargs)
            train_kwargs.update(overrides)
            train_kwargs["seed"] = int(seed)
            train_kwargs["out"] = str(train_out)
            print(f"[INFO] Ablation run: config={config_name}, seed={seed}")
            train_summary = train_ocr(TrainingConfig(**train_kwargs))

            eval_out = train_out / "eval"
            eval_kwargs = dict(base_eval_kwargs)
            eval_kwargs["checkpoint"] = str(train_summary.best_checkpoint)
            eval_kwargs["out"] = str(eval_out)
            eval_summary = evaluate_ocr(EvaluationConfig(**eval_kwargs))
            report = _load_json(Path(eval_summary.report_path))
            run_results.append(
                AblationRunResult(
                    config_name=config_name,
                    seed=int(seed),
                    train_out=str(train_out),
                    eval_out=str(eval_out),
                    best_checkpoint=str(train_summary.best_checkpoint),
                    cer=float(report["cer"]),
                    exact_match=float(report["exact_match"]),
                    cer_square=float(report["cer_square"]),
                    cer_rect=float(report["cer_rect"]),
                    weighted_shape_cer=float(report["weighted_shape_cer"]),
                    macro_shape_cer=float(report["macro_shape_cer"]),
                )
            )

    aggregate_rows: list[dict[str, Any]] = []
    for config_name in sorted({result.config_name for result in run_results}):
        subset = [result for result in run_results if result.config_name == config_name]
        aggregate_rows.append(
            {
                "config_name": config_name,
                "runs": len(subset),
                "cer_mean": _metric_mean([row.cer for row in subset]),
                "cer_std": _metric_std([row.cer for row in subset]),
                "exact_mean": _metric_mean([row.exact_match for row in subset]),
                "exact_std": _metric_std([row.exact_match for row in subset]),
                "cer_square_mean": _metric_mean([row.cer_square for row in subset]),
                "cer_square_std": _metric_std([row.cer_square for row in subset]),
                "cer_rect_mean": _metric_mean([row.cer_rect for row in subset]),
                "cer_rect_std": _metric_std([row.cer_rect for row in subset]),
                "weighted_shape_cer_mean": _metric_mean([row.weighted_shape_cer for row in subset]),
                "weighted_shape_cer_std": _metric_std([row.weighted_shape_cer for row in subset]),
                "macro_shape_cer_mean": _metric_mean([row.macro_shape_cer for row in subset]),
                "macro_shape_cer_std": _metric_std([row.macro_shape_cer for row in subset]),
            }
        )
    aggregate_rows.sort(key=lambda row: float(row["weighted_shape_cer_mean"]))
    return run_results, aggregate_rows


def write_ablation_reports(out_root: Path, run_results: list[AblationRunResult], aggregate_rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    runs_csv = out_root / "runs.csv"
    summary_csv = out_root / "summary_by_config.csv"
    summary_json = out_root / "summary_by_config.json"

    with open(runs_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_name",
                "seed",
                "train_out",
                "eval_out",
                "best_checkpoint",
                "cer",
                "exact_match",
                "cer_square",
                "cer_rect",
                "weighted_shape_cer",
                "macro_shape_cer",
            ],
        )
        writer.writeheader()
        for result in run_results:
            writer.writerow(result.__dict__)

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(aggregate_rows, handle, ensure_ascii=False, indent=2)

    return runs_csv, summary_csv, summary_json
