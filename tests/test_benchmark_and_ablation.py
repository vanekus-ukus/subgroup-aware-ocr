from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shape_aware_ocr.ablation import run_ablations, write_ablation_reports
from shape_aware_ocr.benchmark import build_private_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkAndAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_root = PROJECT_ROOT / "data" / "toy_research"
        subprocess.run(
            [sys.executable, "-m", "shape_aware_ocr.cli.generate_toy_dataset", "--out", str(cls.data_root), "--clean"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "shape_aware_ocr.cli.build_alphabet",
                "--data-root",
                str(cls.data_root / "real"),
                "--out",
                str(cls.data_root / "artifacts"),
            ],
            check=True,
        )
        cls.alphabet_path = cls.data_root / "artifacts" / "alphabet.json"

    def test_build_private_benchmark(self):
        temp_root = Path(tempfile.mkdtemp(prefix="shape_aware_benchmark_"))
        try:
            summary = build_private_benchmark(
                src_root=self.data_root / "real",
                shape_manifest=self.data_root / "manifests" / "shape_manifest.csv",
                sample_class_manifest=self.data_root / "manifests" / "style_manifest.csv",
                out_root=temp_root / "benchmark",
                rect_count=6,
                square_count=4,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=7,
                clean=True,
            )
            self.assertEqual(summary.total_selected, 10)
            real_files = list((temp_root / "benchmark" / "real").glob("*.png"))
            self.assertEqual(len(real_files), 10)
            split_manifest = temp_root / "benchmark" / "manifests" / "split_manifest.csv"
            self.assertTrue(split_manifest.exists())
            with open(split_manifest, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            self.assertGreaterEqual(len({row["split"] for row in rows}), 2)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_ablation_runner_with_fixed_split(self):
        temp_root = Path(tempfile.mkdtemp(prefix="shape_aware_ablation_"))
        try:
            benchmark_root = temp_root / "benchmark"
            build_private_benchmark(
                src_root=self.data_root / "real",
                shape_manifest=self.data_root / "manifests" / "shape_manifest.csv",
                sample_class_manifest=self.data_root / "manifests" / "style_manifest.csv",
                out_root=benchmark_root,
                rect_count=6,
                square_count=4,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=9,
                clean=True,
            )
            config_dir = temp_root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_dir / "baseline_smoke.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epochs": 1,
                        "patience": 2,
                        "batch_size": 4,
                        "best_by": "weighted_shape",
                        "square_oversample": 1.5,
                        "synth_ratio": 0.5,
                        "synth_decay_last": 0,
                        "amp": False,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            run_results, aggregate_rows = run_ablations(
                config_dir=config_dir,
                config_names=["baseline_smoke"],
                seeds=[13],
                base_train_kwargs={
                    "data_root": str(benchmark_root / "real"),
                    "out": "",
                    "alphabet": str(self.alphabet_path),
                    "batch_size": 4,
                    "train_workers": 0,
                    "val_workers": 0,
                    "shape_manifest": str(benchmark_root / "manifests" / "shape_manifest.csv"),
                    "sample_class_manifests": (str(benchmark_root / "manifests" / "style_manifest.csv"),),
                    "split_manifest": str(benchmark_root / "manifests" / "split_manifest.csv"),
                    "synth_root": str(self.data_root / "synth_square"),
                    "hard_manifest": "",
                    "amp": False,
                },
                base_eval_kwargs={
                    "data_root": str(benchmark_root / "real"),
                    "checkpoint": "",
                    "alphabet": str(self.alphabet_path),
                    "out": "",
                    "workers": 0,
                    "shape_manifest": str(benchmark_root / "manifests" / "shape_manifest.csv"),
                    "sample_class_manifests": (str(benchmark_root / "manifests" / "style_manifest.csv"),),
                    "split_manifest": str(benchmark_root / "manifests" / "split_manifest.csv"),
                    "split_name": "test",
                },
                out_root=temp_root / "runs",
            )
            self.assertEqual(len(run_results), 1)
            self.assertEqual(len(aggregate_rows), 1)
            runs_csv, summary_csv, summary_json = write_ablation_reports(temp_root / "reports", run_results, aggregate_rows)
            self.assertTrue(runs_csv.exists())
            self.assertTrue(summary_csv.exists())
            self.assertTrue(summary_json.exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
