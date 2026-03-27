from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from shape_aware_ocr.reporting import (
    aggregate_run_rows,
    aggregate_subgroup_rows,
    build_source_class_manifest,
    pairwise_config_deltas,
    pairwise_experiment_deltas,
    render_markdown_report,
    scan_experiment_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    def test_scan_and_render_report(self):
        temp_root = Path(tempfile.mkdtemp(prefix="shape_aware_reporting_"))
        try:
            oracle_root = temp_root / "oracle_shape"
            eval_root = oracle_root / "pilot_baseline" / "seed_42" / "eval_multiaxis"
            eval_root.mkdir(parents=True, exist_ok=True)
            (eval_root / "eval_report.json").write_text(
                json.dumps(
                    {
                        "samples": 10,
                        "chars": 60,
                        "cer": 0.25,
                        "exact_match": 0.3,
                        "cer_square": 0.20,
                        "cer_rect": 0.28,
                        "weighted_shape_cer": 0.23,
                        "macro_shape_cer": 0.24,
                        "bootstrap": {
                            "cer": {"mean": 0.24, "lower": 0.20, "upper": 0.30},
                            "exact": {"mean": 0.31, "lower": 0.20, "upper": 0.40},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with open(eval_root / "subgroup_metrics.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["group", "cer", "exact", "samples", "chars"])
                writer.writeheader()
                writer.writerow({"group": "overall", "cer": 0.25, "exact": 0.3, "samples": 10, "chars": 60})
                writer.writerow({"group": "sample_class:white_on_blue", "cer": 0.15, "exact": 0.4, "samples": 3, "chars": 18})

            run_rows, subgroup_rows = scan_experiment_root("oracle_shape", oracle_root, eval_dir_name="eval_multiaxis")
            self.assertEqual(len(run_rows), 1)
            self.assertEqual(len(subgroup_rows), 2)

            summary_rows = aggregate_run_rows(run_rows)
            subgroup_summary_rows = aggregate_subgroup_rows(subgroup_rows)
            config_delta_rows = pairwise_config_deltas(run_rows)
            experiment_delta_rows = pairwise_experiment_deltas(run_rows)
            report = render_markdown_report(
                title="Pilot Report",
                benchmark_summary={
                    "total_selected": 10,
                    "rect_selected": 6,
                    "square_selected": 4,
                    "train_count": 8,
                    "val_count": 1,
                    "test_count": 1,
                    "style_labeled_count": 3,
                    "copy_mode": "hardlink",
                },
                shape_reports={
                    "predicted_shape": {
                        "samples": 10,
                        "acc": 0.9,
                        "balanced_acc": 0.85,
                        "square_recall": 0.8,
                        "rect_recall": 0.9,
                    }
                },
                summary_rows=summary_rows,
                subgroup_summary_rows=subgroup_summary_rows,
                config_delta_rows=config_delta_rows,
                experiment_delta_rows=experiment_delta_rows,
                experiment_roots={"oracle_shape": str(oracle_root)},
                figures=[str(PROJECT_ROOT / "artifacts" / "public" / "toy_contact_sheet.png")],
            )
            self.assertIn("Pilot Report", report)
            self.assertIn("pilot_baseline", report)
            self.assertIn("white_on_blue", report)
            self.assertIn("predicted_shape", report)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_pairwise_deltas(self):
        run_rows = [
            {
                "experiment_name": "predicted_shape",
                "config_name": "shape_weighted",
                "seed": 42,
                "cer": 0.50,
                "exact_match": 0.30,
                "cer_square": 0.55,
                "cer_rect": 0.47,
                "weighted_shape_cer": 0.52,
                "macro_shape_cer": 0.51,
            },
            {
                "experiment_name": "predicted_shape",
                "config_name": "synthetic_curriculum",
                "seed": 42,
                "cer": 0.40,
                "exact_match": 0.35,
                "cer_square": 0.43,
                "cer_rect": 0.38,
                "weighted_shape_cer": 0.41,
                "macro_shape_cer": 0.405,
            },
            {
                "experiment_name": "oracle_shape",
                "config_name": "synthetic_curriculum",
                "seed": 42,
                "cer": 0.45,
                "exact_match": 0.33,
                "cer_square": 0.50,
                "cer_rect": 0.42,
                "weighted_shape_cer": 0.46,
                "macro_shape_cer": 0.46,
            },
        ]

        config_rows = pairwise_config_deltas(run_rows)
        self.assertEqual(len(config_rows), 1)
        self.assertEqual(config_rows[0]["left_config"], "shape_weighted")
        self.assertAlmostEqual(float(config_rows[0]["weighted_shape_cer_delta_mean"]), 0.11, places=6)

        experiment_rows = pairwise_experiment_deltas(run_rows)
        self.assertEqual(len(experiment_rows), 1)
        self.assertEqual(experiment_rows[0]["left_experiment"], "oracle_shape")
        self.assertEqual(experiment_rows[0]["right_experiment"], "predicted_shape")
        self.assertAlmostEqual(float(experiment_rows[0]["weighted_shape_cer_delta_mean"]), 0.05, places=6)

    def test_build_source_manifest(self):
        temp_root = Path(tempfile.mkdtemp(prefix="shape_aware_source_manifest_"))
        try:
            source_manifest = temp_root / "source_manifest.csv"
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "match_key", "source_path"])
                writer.writeheader()
                writer.writerow({"file": "000001_NATIVE.png", "match_key": "000001_NATIVE", "source_path": "data/crops_eu/123_NATIVE.png"})
                writer.writerow({"file": "000002_IMPORTED.png", "match_key": "000002_IMPORTED", "source_path": "data/crops_eu/215001_IMPORTED.png"})

            imported_manifest = temp_root / "imported_square_crops_manifest.csv"
            with open(imported_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["origin_dataset", "output_path", "file"])
                writer.writeheader()
                writer.writerow(
                    {
                        "origin_dataset": "sg",
                        "output_path": "data/crops_eu/215001_IMPORTED.png",
                        "file": "215001_IMPORTED.png",
                    }
                )

            out_path = temp_root / "source_class_manifest.csv"
            build_source_class_manifest(source_manifest, imported_manifest, out_path, native_class_name="native_real")
            with open(out_path, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["class_name"], "native_real")
            self.assertEqual(rows[1]["class_name"], "imported_sg")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
