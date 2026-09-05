from __future__ import annotations

import copy
import unittest

import evaluate_dtr_final_visual_shell_probe as gate


def rows(episode_id: str, counts: list[int]) -> list[dict]:
    actor = {"transform": {"x": 1.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
             "bounding_box": {"extent": {"x": 0.5}},
             "local_position": {"forward_m": 1.0}, "command_velocity": {"x": 0.0}}
    return [{"episode_id": episode_id, "sample_index": index, "time_s": index / 10,
             "wearer_transform": {"x": index / 10, "yaw": 0.0},
             "camera_transform": {"x": index / 10, "yaw": 0.0},
             "actors": {"target": copy.deepcopy(actor)},
             "instance_visibility": {"target": {"pixels": count}},
             "truth": {"current_contact": index == 10,
                       "responsible_assets": ["target"] if index == 10 else [],
                       "expected_responsible_assets": ["target"], "expected_outcome": "CONTACT"}}
            for index, count in enumerate(counts)]


class VisualShellGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = {"episode_id": "ep_09", "reference_episode_id": "ep_11",
                     "target_asset": "target", "role": "PARTIAL", "window_s": [1.3, 2.4]}
        self.primary = rows("ep_09", [20] * 50)
        self.reference = rows("ep_11", [100] * 50)

    def test_partial_reference_fraction_passes(self) -> None:
        result = gate.evaluate_pair(self.pair, self.primary, self.reference)
        self.assertEqual(12, result["maximum_qualifying_run"])
        self.assertEqual(0.2, result["minimum_ratio"])

    def test_reference_invisible_fails_instead_of_zero_division_or_fake_ratio(self) -> None:
        self.reference[16]["instance_visibility"]["target"]["pixels"] = 0
        with self.assertRaisesRegex(ValueError, "reference_invisible"):
            gate.evaluate_pair(self.pair, self.primary, self.reference)

    def test_partial_zero_outside_qualifying_run_still_fails(self) -> None:
        self.primary[24]["instance_visibility"]["target"]["pixels"] = 0
        with self.assertRaisesRegex(ValueError, "partial_zero_pixels"):
            gate.evaluate_pair(self.pair, self.primary, self.reference)

    def test_partial_requires_contiguous_run(self) -> None:
        for index in (15, 20):
            self.primary[index]["instance_visibility"]["target"]["pixels"] = 80
        with self.assertRaisesRegex(ValueError, "partial_ratio_run"):
            gate.evaluate_pair(self.pair, self.primary, self.reference)

    def test_target_geometry_and_camera_mismatch_fail(self) -> None:
        for key in ("target", "camera", "wearer"):
            reference = copy.deepcopy(self.reference)
            if key == "target":
                reference[4]["actors"]["target"]["transform"]["x"] += 0.01
            else:
                reference[4][key + "_transform"]["x"] += 0.01
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "twin_"):
                gate.evaluate_pair(self.pair, self.primary, reference)

    def full(self, counts: list[int]) -> tuple[dict, list, list]:
        pair = {**self.pair, "role": "FULL", "window_s": [2.0, 3.2]}
        return pair, rows("ep_09", counts), rows("ep_11", [100] * len(counts))

    def test_full_with_pre_and_post_context_passes(self) -> None:
        pair, primary, reference = self.full([100] * 20 + [0] * 13 + [100] * 10)
        self.assertEqual(13, gate.evaluate_pair(pair, primary, reference)["maximum_qualifying_run"])

    def test_full_requires_eight_post_visible_frames(self) -> None:
        pair, primary, reference = self.full([100] * 20 + [0] * 13 + [100] * 7)
        with self.assertRaisesRegex(ValueError, "full_disappearance_context"):
            gate.evaluate_pair(pair, primary, reference)

    def test_full_window_inside_longer_disappearance_uses_true_start(self) -> None:
        pair, primary, reference = self.full([100] * 16 + [0] * 19 + [100] * 10)
        self.assertEqual(13, gate.evaluate_pair(pair, primary, reference)["maximum_qualifying_run"])

    def test_full_disappearance_outside_window_does_not_count(self) -> None:
        pair, primary, reference = self.full([100] * 8 + [0] * 8 + [100] * 30)
        with self.assertRaisesRegex(ValueError, "full_disappearance_context"):
            gate.evaluate_pair(pair, primary, reference)

    def test_full_requires_immediate_pretrack(self) -> None:
        counts = [100] * 20 + [0] * 13 + [100] * 10
        counts[15] = 0
        with self.assertRaisesRegex(ValueError, "full_disappearance_context"):
            gate.evaluate_pair(*self.full(counts))

    def test_outcome_and_exact_responsible_set(self) -> None:
        scenario = {"episode_id": "ep_09", "expected_outcome": "CONTACT",
                    "expected_responsible_assets": ["target"]}
        gate.check_outcome(self.primary, scenario)
        self.primary[10]["truth"]["responsible_assets"] = ["shell"]
        with self.assertRaisesRegex(ValueError, "responsible"):
            gate.check_outcome(self.primary, scenario)

    def test_outcome_mismatch(self) -> None:
        scenario = {"episode_id": "ep_09", "expected_outcome": "SAFE", "expected_responsible_assets": []}
        with self.assertRaisesRegex(ValueError, "outcome"):
            gate.check_outcome(self.primary, scenario)

    def test_alignment_tolerance_preserves_truth_and_wraps_angles(self) -> None:
        self.assertTrue(gate.close({"x": 1.0, "yaw": 180.0}, {"x": 1.00001, "yaw": -180.0}))
        self.assertFalse(gate.close({"current_contact": True}, {"current_contact": 1}))
        self.assertFalse(gate.close({"current_contact": True}, {"current_contact": False}))

    def test_witness_ignores_numeric_actor_id_but_detects_authority_and_truth_mismatch(self) -> None:
        source = copy.deepcopy(self.primary[0])
        source.update(layout_id="layout", layout_receipt_sha256="layout-hash", plan_receipt_sha256="plan-hash")
        source["actors"]["target"].update(actor_id=123, track_id="track", asset_key="target", role="target",
                                              kind="prop", actual_blueprint="mesh")
        source["truth"].update(scenario_role="CONTACT", twin_role="probe", future_contact_within_horizon=True,
                                    realized_time_to_contact_seconds=1.0, minimum_distance_m=1.0,
                                    collision_polygons_xy={"target": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]})
        witness = copy.deepcopy(source)
        witness["actors"]["target"]["actor_id"] = 987
        kwargs = {"mode": gate.join.ACTUAL_ALL_REPLAY, "authority": frozenset({"target"})}
        gate.check_witness([source], [witness], **kwargs)
        for field in ("target", "wearer", "truth"):
            changed = copy.deepcopy(witness)
            if field == "target":
                changed["actors"]["target"]["transform"]["x"] += 0.01
            elif field == "wearer":
                changed["wearer_transform"]["x"] += 0.01
            else:
                changed["truth"]["responsible_assets"] = ["shell"]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "witness_replay"):
                gate.check_witness([source], [changed], **kwargs)


if __name__ == "__main__":
    unittest.main()
