from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dtr_carla_c1_complex import (  # noqa: E402
    build_plan_receipt,
    contact_union,
    forbidden_model_paths,
    plan_authority,
    scenario_by_id,
    trajectory_position,
    trajectory_prefix_equal,
    validate_protocol,
)
from join_dtr_carla_c1_complex import (  # noqa: E402
    compare_replays,
    physical_occlusion_indices,
)


class DtrCarlaC1ComplexProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (HERE / "dtr_carla_c1_complex_protocol.json").read_text(encoding="utf-8")
        )
        validate_protocol(cls.protocol)

    def scenario(self, episode_id: str) -> dict:
        return scenario_by_id(self.protocol, episode_id)

    def trajectory(self, name: str) -> dict:
        return self.protocol["trajectory_library"][name]

    def test_scene_has_required_programmatic_asset_roles(self) -> None:
        counts = Counter(asset["role"] for asset in self.protocol["asset_cluster"])
        self.assertEqual(counts, Counter(self.protocol["admission"]["required_role_counts"]))
        self.assertEqual(len(self.protocol["asset_cluster"]), 15)

    def test_same_plan_twins_have_byte_identical_receipts(self) -> None:
        for first_id, second_id in (("ep_01", "ep_02"), ("ep_03", "ep_04"), ("ep_07", "ep_08")):
            first = build_plan_receipt(self.scenario(first_id)["issued_plan"])
            second = build_plan_receipt(self.scenario(second_id)["issued_plan"])
            self.assertEqual(first, second)
            self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])

    def test_realized_execution_cannot_change_plan_receipt(self) -> None:
        scenario = copy.deepcopy(self.scenario("ep_03"))
        before = build_plan_receipt(scenario["issued_plan"])
        scenario["wearer_trajectory"] = "wearer_straight"
        scenario["expected_contact"] = True
        after = build_plan_receipt(scenario["issued_plan"])
        self.assertEqual(before, after)

    def test_twin_motion_prefixes_are_equal_until_declared_intervention(self) -> None:
        sample_s = float(self.protocol["environment"]["sample_seconds"])
        pair_1_a = self.scenario("ep_01")
        pair_1_b = self.scenario("ep_02")
        self.assertTrue(
            trajectory_prefix_equal(
                self.trajectory(pair_1_a["asset_trajectories"]["target_primary"]),
                self.trajectory(pair_1_b["asset_trajectories"]["target_primary"]),
                end_s=4.0,
                sample_s=sample_s,
            )
        )
        pair_2_a = self.scenario("ep_03")
        pair_2_b = self.scenario("ep_04")
        self.assertTrue(
            trajectory_prefix_equal(
                self.trajectory(pair_2_a["wearer_trajectory"]),
                self.trajectory(pair_2_b["wearer_trajectory"]),
                end_s=2.5,
                sample_s=sample_s,
            )
        )
        self.assertNotEqual(
            trajectory_position(self.trajectory(pair_2_a["wearer_trajectory"]), 5.0),
            trajectory_position(self.trajectory(pair_2_b["wearer_trajectory"]), 5.0),
        )

    def test_invalid_plan_authority_is_exact_and_causal(self) -> None:
        stale = build_plan_receipt(self.scenario("ep_05")["issued_plan"])
        missing = build_plan_receipt(self.scenario("ep_06")["issued_plan"])
        self.assertEqual(
            plan_authority(stale, session_id="session_pair_03", time_s=0.0),
            "EXPIRED",
        )
        self.assertEqual(
            plan_authority(missing, session_id="session_pair_03", time_s=0.0),
            "NO_PLAN",
        )

    def test_model_boundary_allows_issued_plan_but_rejects_future_truth(self) -> None:
        allowed = {
            "time_s": 2.0,
            "wearer": {"x": 1.0, "y": 2.0},
            "issued_plan_receipt": build_plan_receipt(self.scenario("ep_01")["issued_plan"]),
            "current_actors": [{"track_id": "o_01", "x": 4.0, "y": 5.0}],
        }
        self.assertEqual(forbidden_model_paths(allowed), [])
        contaminated = copy.deepcopy(allowed)
        contaminated["truth"] = {"future_contact_within_horizon": True}
        self.assertEqual(
            forbidden_model_paths(contaminated),
            ["$.truth", "$.truth.future_contact_within_horizon"],
        )

    def test_contact_union_scores_all_collision_relevant_assets(self) -> None:
        polygons = {
            "o_01": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
            "o_02": [[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]],
        }
        contact, minimum, assets = contact_union(
            (0.6, 0.0), polygons, wearer_radius_m=0.45
        )
        self.assertTrue(contact)
        self.assertAlmostEqual(minimum, 0.4)
        self.assertEqual(assets, ["o_01"])

    def test_scripted_center_geometry_matches_frozen_contact_roster(self) -> None:
        sample_s = float(self.protocol["environment"]["sample_seconds"])
        duration_s = float(self.protocol["environment"]["duration_seconds"])
        steps = int(round(duration_s / sample_s))
        for scenario in self.protocol["scenarios"]:
            wearer = self.trajectory(scenario["wearer_trajectory"])
            primary = self.trajectory(
                scenario["asset_trajectories"]["target_primary"]
            )
            minimum = min(
                math.dist(
                    trajectory_position(wearer, index * sample_s),
                    trajectory_position(primary, index * sample_s),
                )
                for index in range(steps + 1)
            )
            self.assertEqual(
                minimum <= 0.6,
                scenario["expected_contact"],
                msg=f"center geometry mismatch for {scenario['episode_id']}: {minimum}",
            )

            for key in ("cyclist", "parallel", "child"):
                distractor = self.trajectory(scenario["asset_trajectories"][key])
                distractor_minimum = min(
                    math.dist(
                        trajectory_position(wearer, index * sample_s),
                        trajectory_position(distractor, index * sample_s),
                    )
                    for index in range(steps + 1)
                )
                self.assertGreater(
                    distractor_minimum,
                    1.5,
                    msg=f"{scenario['episode_id']} unintended center approach by {key}",
                )

    def test_cross_sensor_replay_ignores_ephemeral_ids_but_detects_state_drift(self) -> None:
        actor = {
            "track_id": "o_01",
            "asset_key": "target_primary",
            "scenario_role": "primary_crossing",
            "kind": "walker",
            "actual_blueprint": "walker.pedestrian.0002",
            "carla_actor_id": 17,
            "transform": {"x": 1.0, "y": 2.0, "z": 0.9, "pitch": 0.0, "yaw": 90.0, "roll": 0.0},
            "local_position": {"forward_m": 0.0, "right_m": 2.0},
            "command_velocity": {"x": 0.0, "y": -1.0, "z": 0.0},
            "bounding_box": {
                "location": {"x": 0.0, "y": 0.0, "z": 0.0},
                "extent": {"x": 0.2, "y": 0.2, "z": 0.9},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "world_vertices": [],
            },
        }
        row = {
            "episode_id": "ep_01",
            "sample_index": 1,
            "time_s": 0.05,
            "plan_receipt_sha256": "A",
            "plan_authority": "VALID",
            "actors": {"target_primary": actor},
            "truth": {
                "scenario_role": "valid_straight_follow_contact",
                "twin_role": "a",
                "expected_contact": True,
                "current_contact": False,
                "minimum_distance_m": 1.0,
                "responsible_asset": [],
                "collision_polygons_xy": {"target_primary": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                "future_contact_within_horizon": True,
                "realized_time_to_contact_seconds": 2.0,
            },
        }
        replay = copy.deepcopy(row)
        replay["actors"]["target_primary"]["carla_actor_id"] = 999
        self.assertEqual(compare_replays([row], [replay], candidate_sensor="depth"), [])
        replay["actors"]["target_primary"]["transform"]["x"] += 0.1
        self.assertEqual(
            compare_replays([row], [replay], candidate_sensor="depth")[0]["reason"],
            "actual_replay_state_mismatch",
        )

    def test_physical_occlusion_requires_fov_hidden_mask_and_van_intersection(self) -> None:
        row = {
            "sample_index": 20,
            "camera_transform": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "actors": {"target_primary": {"transform": {"x": 10.0, "y": 0.0}}},
            "instance_visibility": {"target_primary": {"visible": False}},
            "truth": {
                "collision_polygons_xy": {
                    "occluder_van": [[4.0, -1.0], [6.0, -1.0], [6.0, 1.0], [4.0, 1.0]]
                }
            },
        }
        self.assertEqual(physical_occlusion_indices([row], 90.0), [20])
        row["instance_visibility"]["target_primary"]["visible"] = True
        self.assertEqual(physical_occlusion_indices([row], 90.0), [])


if __name__ == "__main__":
    unittest.main()
