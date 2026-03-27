from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shape_aware_ocr.alphabet import build_alphabet_from_root
from shape_aware_ocr.dataset import iter_image_files, load_sample_class_maps_from_manifests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToyDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_root = PROJECT_ROOT / "data" / "toy_research"
        subprocess.run(
            [sys.executable, "-m", "shape_aware_ocr.cli.generate_toy_dataset", "--out", str(cls.data_root), "--clean"],
            check=True,
        )

    def test_dataset_exists(self):
        real_images = list(iter_image_files(self.data_root / "real"))
        synth_images = list(iter_image_files(self.data_root / "synth_square"))
        self.assertEqual(len(real_images), 18)
        self.assertEqual(len(synth_images), 10)

    def test_manifests_exist(self):
        shape_manifest = self.data_root / "manifests" / "shape_manifest.csv"
        style_manifest = self.data_root / "manifests" / "style_manifest.csv"
        with open(shape_manifest, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertIn("shape", rows[0])
        with open(style_manifest, "r", newline="", encoding="utf-8") as handle:
            style_rows = list(csv.DictReader(handle))
        self.assertEqual(len(style_rows), 18)
        self.assertIn("class_name", style_rows[0])

    def test_alphabet_builder(self):
        alphabet = build_alphabet_from_root(self.data_root / "real")
        self.assertGreaterEqual(alphabet["num_classes"], 10)
        self.assertIn("A", alphabet["tokens"])

    def test_multiple_sample_class_manifests_are_namespaced(self):
        temp_root = Path(tempfile.mkdtemp(prefix="shape_aware_toy_source_"))
        try:
            temp_source_manifest = temp_root / "source_class_manifest.csv"
            with open(temp_source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "match_key", "class_name"])
                writer.writeheader()
                for image_path in iter_image_files(self.data_root / "real"):
                    writer.writerow(
                        {
                            "file": image_path.name,
                            "match_key": image_path.stem,
                            "class_name": "native_demo",
                        }
                    )
            merged_map, conflicts = load_sample_class_maps_from_manifests(
                (self.data_root / "manifests" / "style_manifest.csv", temp_source_manifest)
            )
            self.assertEqual(conflicts, 0)
            first_value = next(iter(merged_map.values()))
            self.assertIn("style:", first_value)
            self.assertIn("source:", first_value)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
