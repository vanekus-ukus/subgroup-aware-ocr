from __future__ import annotations

from collections import defaultdict
import unittest

from shape_aware_ocr.metrics import accumulate_stats, bootstrap_confidence_interval, edit_distance, macro_group_cer, weighted_shape_cer
import torch


class MetricTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(edit_distance([1, 2, 3], [1, 4, 3]), 1)

    def test_weighted_shape_cer(self):
        value = weighted_shape_cer(0.2, 0.1, 0.75, fallback=0.3)
        self.assertAlmostEqual(value, 0.175)
        self.assertAlmostEqual(macro_group_cer(0.2, 0.1, fallback=0.3), 0.15)

    def test_bootstrap_ci(self):
        ci = bootstrap_confidence_interval([0.1, 0.2, 0.3, 0.4], n_bootstrap=100, seed=1)
        self.assertEqual(ci["samples"], 4)
        self.assertLessEqual(ci["lower"], ci["mean"])
        self.assertGreaterEqual(ci["upper"], ci["mean"])

    def test_accumulate_stats_with_multiple_sample_classes(self):
        stats = {
            "overall": {"edits": 0.0, "chars": 0.0, "exact": 0.0, "samples": 0.0},
            "square": {"edits": 0.0, "chars": 0.0, "exact": 0.0, "samples": 0.0},
            "rect": {"edits": 0.0, "chars": 0.0, "exact": 0.0, "samples": 0.0},
            "sample_classes": defaultdict(lambda: {"edits": 0.0, "chars": 0.0, "exact": 0.0, "samples": 0.0}),
        }
        accumulate_stats(
            stats,
            pred_sequences=[[1, 2, 3]],
            target_concat=torch.tensor([1, 2, 3], dtype=torch.long),
            target_lengths=torch.tensor([3], dtype=torch.long),
            shape_flags=torch.tensor([1], dtype=torch.long),
            sample_classes=["style:white_on_blue|source:native_eu"],
        )
        self.assertIn("style:white_on_blue", stats["sample_classes"])
        self.assertIn("source:native_eu", stats["sample_classes"])
        self.assertEqual(stats["sample_classes"]["style:white_on_blue"]["samples"], 1.0)
        self.assertEqual(stats["sample_classes"]["source:native_eu"]["samples"], 1.0)
