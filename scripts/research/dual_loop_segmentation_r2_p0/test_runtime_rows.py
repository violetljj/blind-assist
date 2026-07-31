from __future__ import annotations

import json
import unittest
from pathlib import Path

from .benchmark_runtime_rows import _summary
from .validate_runtime_rows import RuntimeValidationError, _summary as independent_summary


class RuntimeRowsTests(unittest.TestCase):
    def test_summary_contract_matches_independent_recompute(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_summary(values), independent_summary(values))

    def test_empty_runtime_rows_fail_closed(self) -> None:
        with self.assertRaises(RuntimeValidationError):
            independent_summary([])

    def test_runtime_schema_defines_nonnegative_numeric_stages(self) -> None:
        root = Path(__file__).resolve().parents[3]
        schema = json.loads(
            (
                root
                / "configs/dual_loop_segmentation_r2_p0/runtime_rows.schema.json"
            ).read_text(encoding="utf-8")
        )
        stages = schema["properties"]["stages_ms"]
        self.assertFalse(stages["additionalProperties"])
        self.assertEqual(set(stages["required"]), set(stages["properties"]))
        for definition in stages["properties"].values():
            self.assertEqual(definition, {"type": "number", "minimum": 0})


if __name__ == "__main__":
    unittest.main()
