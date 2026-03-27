from __future__ import annotations

import unittest

from shape_aware_ocr.labels import label_from_filename_stem, normalized_match_stem, normalize_label


class LabelTests(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label("ab-12 cd"), "AB12CD")

    def test_duplicate_tail_is_collapsed(self):
        stem = "000123_AB12CD_AB12CD"
        self.assertEqual(normalized_match_stem(stem), "000123_AB12CD")
        self.assertEqual(label_from_filename_stem(stem), "AB12CD")

    def test_simple_filename_label(self):
        self.assertEqual(label_from_filename_stem("000321_QX456B"), "QX456B")
