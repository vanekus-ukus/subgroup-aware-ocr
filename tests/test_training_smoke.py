from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shape_aware_ocr.evaluation import EvaluationConfig, evaluate_ocr
from shape_aware_ocr.shape_classifier import ShapeClassifierConfig, train_shape_classifier
from shape_aware_ocr.training import TrainingConfig, train_ocr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTests(unittest.TestCase):
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

    def test_train_and_eval_smoke(self):
        workdir = Path(tempfile.mkdtemp(prefix="shape_aware_ocr_smoke_"))
        try:
            train_out = workdir / "ocr"
            summary = train_ocr(
                TrainingConfig(
                    data_root=str(self.data_root / "real"),
                    out=str(train_out),
                    alphabet=str(self.alphabet_path),
                    epochs=1,
                    batch_size=4,
                    train_workers=0,
                    val_workers=0,
                    val_split=0.25,
                    patience=2,
                    amp=False,
                    shape_manifest=str(self.data_root / "manifests" / "shape_manifest.csv"),
                    sample_class_manifests=(str(self.data_root / "manifests" / "style_manifest.csv"),),
                    square_oversample=1.5,
                    synth_root=str(self.data_root / "synth_square"),
                    synth_ratio=0.5,
                    synth_decay_last=0,
                )
            )
            self.assertTrue(Path(summary.best_checkpoint).exists())
            with open(Path(summary.train_log), "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(int(rows[-1]["train_synth_count"]), 0)
            eval_summary = evaluate_ocr(
                EvaluationConfig(
                    data_root=str(self.data_root / "real"),
                    checkpoint=str(Path(summary.best_checkpoint)),
                    alphabet=str(self.alphabet_path),
                    out=str(workdir / "eval"),
                    batch_size=4,
                    workers=0,
                    shape_manifest=str(self.data_root / "manifests" / "shape_manifest.csv"),
                    sample_class_manifests=(str(self.data_root / "manifests" / "style_manifest.csv"),),
                    bootstrap_samples=32,
                    max_errors=10,
                )
            )
            self.assertTrue(Path(eval_summary.report_path).exists())
            self.assertTrue(Path(eval_summary.errors_csv).exists())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_eval_supports_multiple_sample_class_manifests(self):
        workdir = Path(tempfile.mkdtemp(prefix="shape_aware_eval_multiaxis_"))
        try:
            source_manifest = workdir / "source_class_manifest.csv"
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "match_key", "class_name"])
                writer.writeheader()
                for image_path in (self.data_root / "real").glob("*.png"):
                    writer.writerow(
                        {
                            "file": image_path.name,
                            "match_key": image_path.stem,
                            "class_name": "native_demo",
                        }
                    )

            summary = train_ocr(
                TrainingConfig(
                    data_root=str(self.data_root / "real"),
                    out=str(workdir / "ocr"),
                    alphabet=str(self.alphabet_path),
                    epochs=1,
                    batch_size=4,
                    train_workers=0,
                    val_workers=0,
                    val_split=0.25,
                    patience=2,
                    amp=False,
                    shape_manifest=str(self.data_root / "manifests" / "shape_manifest.csv"),
                    sample_class_manifests=(str(self.data_root / "manifests" / "style_manifest.csv"),),
                )
            )
            eval_summary = evaluate_ocr(
                EvaluationConfig(
                    data_root=str(self.data_root / "real"),
                    checkpoint=str(Path(summary.best_checkpoint)),
                    alphabet=str(self.alphabet_path),
                    out=str(workdir / "eval"),
                    batch_size=4,
                    workers=0,
                    shape_manifest=str(self.data_root / "manifests" / "shape_manifest.csv"),
                    sample_class_manifests=(
                        str(self.data_root / "manifests" / "style_manifest.csv"),
                        str(source_manifest),
                    ),
                    bootstrap_samples=32,
                    max_errors=10,
                )
            )
            with open(Path(eval_summary.subgroup_csv), "r", newline="", encoding="utf-8") as handle:
                groups = [row["group"] for row in csv.DictReader(handle)]
            self.assertTrue(any(group.startswith("sample_class:style:") for group in groups))
            self.assertTrue(any(group.startswith("sample_class:source:") for group in groups))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_shape_classifier_smoke(self):
        workdir = Path(tempfile.mkdtemp(prefix="shape_group_smoke_"))
        try:
            summary = train_shape_classifier(
                ShapeClassifierConfig(
                    data_root=str(self.data_root / "real"),
                    shape_manifest=str(self.data_root / "manifests" / "shape_manifest.csv"),
                    out=str(workdir / "shape_clf"),
                    epochs=1,
                    batch_size=4,
                    train_workers=0,
                    val_workers=0,
                    balanced_sampler=True,
                    amp=False,
                )
            )
            self.assertTrue(Path(summary.best_checkpoint).exists())
            self.assertTrue(Path(summary.report).exists())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
