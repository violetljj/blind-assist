from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_development.py")
SPEC = importlib.util.spec_from_file_location("ulr_run_development", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DevelopmentMechanicsTest(unittest.TestCase):
    def test_text_normalization_and_overlap_do_not_use_labels(self):
        self.assertEqual("星巴克starbucks", MODULE.normalize_text(" 星巴克 / Starbucks "))
        left = MODULE.text_grams(("STARBUCKS",))
        right = MODULE.text_grams(("Starbucks Coffee",))
        self.assertGreater(len(left & right), 0)

    def test_internal_split_is_deterministic(self):
        examples = [
            MODULE.Example(str(index), None, None, None, 0, "unknown", False, 1.0, 1.0)
            for index in range(100)
        ]
        first = MODULE.internal_train_split(examples)
        second = MODULE.internal_train_split(list(reversed(examples)))
        self.assertEqual({item.image_id for item in first[0]}, {item.image_id for item in second[0]})
        self.assertEqual({item.image_id for item in first[1]}, {item.image_id for item in second[1]})


if __name__ == "__main__":
    unittest.main()
