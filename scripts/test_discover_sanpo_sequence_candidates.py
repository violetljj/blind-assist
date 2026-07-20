from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import discover_sanpo_sequence_candidates as discover


class DiscoverSanpoSequenceCandidatesCliTest(unittest.TestCase):
    @staticmethod
    def prefilter_frame(*, lateral: bool = True, path: bool = True, center_hazard: bool = False, center_lateral: bool = False) -> dict:
        return {"source_frame": 0, "profiles": {
            "lateral_pedestrian_or_ebike": lateral,
            "path_geometry_usable": path,
            "has_center_hazard": center_hazard,
            "has_center_lateral_target": center_lateral,
        }}

    def test_negative_start_index_fails_before_network(self) -> None:
        script = Path(__file__).with_name("discover_sanpo_sequence_candidates.py")
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                "unused.json",
                "--start-session-index",
                "-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("start-session-index non-negative", result.stderr)

    def test_unknown_profile_fails_before_network(self) -> None:
        script = Path(__file__).with_name("discover_sanpo_sequence_candidates.py")
        result = subprocess.run(
            [sys.executable, str(script), "--output", "unused.json", "--profiles", "not_a_profile"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_auto_camera_prefers_chest_then_falls_back_to_head(self) -> None:
        original = discover.first_mask_page
        try:
            calls: list[str] = []
            def fake_first_mask_page(prefix: str) -> list[dict]:
                calls.append(prefix)
                return ([] if "camera_chest" in prefix else [{"name": "000001.png"}] * 6)
            discover.first_mask_page = fake_first_mask_page
            selected = discover.select_mask_view("session", "auto", 6)
            self.assertEqual("camera_head", selected[0])
            self.assertEqual(2, len(calls))
            calls.clear()
            def chest_first_mask_page(prefix: str) -> list[dict]:
                calls.append(prefix)
                return [{"name": "000001.png"}] * 6
            discover.first_mask_page = chest_first_mask_page
            selected = discover.select_mask_view("session", "auto", 6)
            self.assertEqual("camera_chest", selected[0])
            self.assertEqual(1, len(calls))
        finally:
            discover.first_mask_page = original

    def test_camera_selection_never_substitutes_right_or_short_inventory(self) -> None:
        original = discover.first_mask_page
        try:
            discover.first_mask_page = lambda prefix: [{"name": "000001.png"}] * 5
            self.assertIsNone(discover.select_mask_view("session", "auto", 6))
            self.assertIsNone(discover.select_mask_view("session", "camera_chest", 6))
        finally:
            discover.first_mask_page = original

    def test_local_prefilter_pass_is_only_an_exact_gate_authorization(self) -> None:
        result = discover.summarize_local_lateral_prefilter(
            [self.prefilter_frame() for _ in range(16)], 16, 8, 8, 13,
        )
        self.assertEqual(result["decision"], "pass_for_exact_50_frame_gate")
        self.assertIn("not an acceptance", result["important_limit"])

    def test_local_prefilter_rejects_center_contamination(self) -> None:
        frames = [self.prefilter_frame() for _ in range(16)]
        frames[7] = self.prefilter_frame(center_lateral=True)
        result = discover.summarize_local_lateral_prefilter(frames, 16, 8, 8, 13)
        self.assertEqual(result["decision"], "reject")
        self.assertIn("local_center_lateral_target_contamination", result["rejection_reasons"])

    def test_local_prefilter_rejects_short_target_run(self) -> None:
        frames = [self.prefilter_frame(lateral=index % 2 == 0) for index in range(16)]
        result = discover.summarize_local_lateral_prefilter(frames, 16, 8, 8, 13)
        self.assertEqual(result["decision"], "reject")
        self.assertIn("local_lateral_target_run_below_minimum", result["rejection_reasons"])

    def test_sparse_step_curb_candidate_requires_lower_field_and_walkable_path(self) -> None:
        components = {
            2: [{"bottom_ratio": 0.72, "corridor_target_ratio": 0.02, "center_x_ratio": 0.50}],
            15: [],
        }
        accepted = discover.sparse_profile_evidence(components, {"walkable_corridor_ratio": 0.24})
        self.assertTrue(accepted["step_curb"])
        self.assertEqual(0.72, accepted["best_boundary_target"]["bottom_ratio"])
        rejected = discover.sparse_profile_evidence(components, {"walkable_corridor_ratio": 0.17})
        self.assertFalse(rejected["step_curb"])


if __name__ == "__main__":
    unittest.main()
