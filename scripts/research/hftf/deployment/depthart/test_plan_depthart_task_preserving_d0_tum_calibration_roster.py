from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d0_tum_calibration_roster import (
    parse_rgb_index,
    select_rows,
)


class PlanDepthArtTaskPreservingD0TumCalibrationRosterTest(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "sequence"
            (sequence / "rgb").mkdir(parents=True)
            lines = ["# test"]
            for index in range(5):
                relative = f"rgb/{index}.png"
                (sequence / relative).write_bytes(bytes((index,)))
                lines.append(f"{index}.0 {relative}")
            (sequence / "rgb.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            first = select_rows(root, ["sequence"], {("sequence", "rgb/1.png")}, 3)
            second = select_rows(root, ["sequence"], {("sequence", "rgb/1.png")}, 3)
        self.assertEqual(first, second)
        self.assertNotIn("rgb/1.png", {row["rgb_path"] for row in first})

    def test_invalid_index_fails(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as handle:
            handle.write("invalid\n")
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "invalid rgb index"):
                parse_rgb_index(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
