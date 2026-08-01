from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_sanpo_proxy_replication import aggregate


def _report(session_id: str, *, terminal_ok: bool = True) -> dict[str, object]:
    return {
        "schema": "blindassist_hftf_sanpo_pose_geometry_authority_r0",
        "terminal": (
            "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED"
            if terminal_ok
            else "HFTF_H0_2_CANONICAL_PROXY_NOT_REPLICATED"
        ),
        "evaluation_mode": "frozen_canonical_replication",
        "source_session_ids": [session_id],
        "manifest_frame_count": 25,
        "transform_direction_canary": {
            "frozen_canonical_rank": 1,
            "frozen_canonical_replication_admitted": True,
            "admitted_semantics": (
                "p_world = R_xyzw @ p_opencv_camera + "
                "camera_translation_m"
            ),
            "frozen_canonical_hypothesis": {
                "median_relative_depth_error": 0.001,
                "p75_relative_depth_error": 0.002,
                "coverage": 0.9,
            },
        },
        "ground_and_body_proxy_canary": {
            "vertical_axis": "+Z",
            "standard_body_proxy_frame_admitted_for_h1": True,
            "physical_camera_to_body_calibration_admitted": False,
            "chosen_axis": {
                "median_ground_mad_m": 0.01,
                "median_camera_clearance_m": 1.4,
            },
        },
        "capability_decisions": {
            "physical_camera_to_person_calibration": "NOT_EVALUABLE"
        },
    }


class SanpoProxyReplicationAggregationTest(unittest.TestCase):
    def _write(
        self, root: Path, name: str, value: dict[str, object]
    ) -> Path:
        path = root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_three_independent_passes_admit_h1_canary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                self._write(root, name, _report(name))
                for name in ("session-a", "session-b", "session-c")
            ]
            result = aggregate(paths)

        self.assertEqual(
            result["terminal"],
            "HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED",
        )
        self.assertEqual(result["allowed_next_step"], "H1_GEOMETRY_TEACHER_CANARY")
        self.assertEqual(
            result["physical_camera_to_person_calibration"],
            "NOT_EVALUABLE",
        )

    def test_duplicate_parent_session_fails_independence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                self._write(root, f"copy-{index}", _report("same-session"))
                for index in range(3)
            ]
            result = aggregate(paths)

        self.assertFalse(result["independent_session_ids"])
        self.assertEqual(
            result["terminal"],
            "HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_NOT_EVALUABLE",
        )

    def test_one_failed_report_blocks_cohort_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                self._write(root, "a", _report("session-a")),
                self._write(root, "b", _report("session-b")),
                self._write(
                    root,
                    "c",
                    _report("session-c", terminal_ok=False),
                ),
            ]
            result = aggregate(paths)

        self.assertFalse(result["all_reports_pass"])
        self.assertEqual(
            result["allowed_next_step"], "REPAIR_OR_EXPAND_H0_2_BEFORE_H1"
        )


if __name__ == "__main__":
    unittest.main()
