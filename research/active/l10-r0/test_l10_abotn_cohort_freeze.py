from __future__ import annotations

import unittest

from l10_abotn_cohort_freeze import freeze_cohort


def episode(scene: str, index: int, x: float, goal: str) -> dict:
    path = f"annotations/{scene}/traj_{index}.json"
    return {
        "path": path,
        "sha256": f"sha-{scene}-{index}",
        "scene_id": scene,
        "trajectory_points": 10 + index,
        "goal_label": goal,
        "endpoint_id": index,
        "end_point": [x, 0],
    }


class CohortFreezeTest(unittest.TestCase):
    def test_freezes_density_extremes_and_closest_pair_without_claiming_sibling_truth(self) -> None:
        rows = [
            episode("low", 0, 0, "low-a"),
            episode("low", 1, 1, "low-b"),
            episode("mid", 0, 0, "mid-a"),
            episode("mid", 1, 9, "mid-b"),
            episode("mid", 2, 10, "mid-c"),
            episode("high", 0, 0, "high-a"),
            episode("high", 1, 7, "high-b"),
            episode("high", 2, 8, "high-c"),
            episode("high", 3, 20, "high-d"),
        ]
        result = freeze_cohort(rows, "manifest")
        self.assertEqual([row["scene_id"] for row in result["cohort"]], ["low", "mid", "high"])
        self.assertEqual(
            [row["goal_label"] for row in result["cohort"][1]["frozen_pair"]],
            ["mid-b", "mid-c"],
        )
        self.assertIn("REQUIRES", result["cohort"][1]["pair_status"])
        control = result["cohort"][1]["target_absent_control"]
        self.assertFalse(control["target_present_in_local_annotation_roster"])
        self.assertIn("REQUIRES", control["control_status"])

    def test_does_not_allow_post_pixel_pair_substitution(self) -> None:
        rows = []
        for scene in ("a", "b", "c"):
            rows.extend([episode(scene, 0, 0, f"{scene}-a"), episode(scene, 1, 1, f"{scene}-b")])
        result = freeze_cohort(rows, "manifest")
        self.assertEqual(result["selection"]["substitution_rule"], "none after any selected-scene pixels are opened")
        self.assertTrue(all("DO_NOT_SUBSTITUTE" in row["pair_failure_rule"] for row in result["cohort"]))


if __name__ == "__main__":
    unittest.main()
