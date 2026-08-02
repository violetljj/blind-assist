from __future__ import annotations

import unittest

from bridge_thor_magni_route_evidence import _sanitized_route_evidence, _select_round_robin


class BridgeThorMagniRouteEvidenceTest(unittest.TestCase):
    def _sample(self) -> dict[str, object]:
        return {
            "sample_id": "route-1",
            "anchor_scene_frame": 390,
            "qtm_frame": 1250,
            "qtm_time_seconds": 12.5,
            "target": {
                "future_corridor_intrusion": True,
                "future_proximity_le_1_25m": True,
                "wearer_speed_mps": 0.5,
                "future_minimum_synchronized_distance_m": 1.1,
                "closest": {
                    "body": "person",
                    "role": "Visitor",
                    "time_offset_seconds": 0.3,
                    "distance_m": 1.1,
                    "longitudinal_m": 0.8,
                    "lateral_m": 0.1,
                },
                "observed_future_body_time_pairs": 10,
                "occupancy_target": [[[0]]],
            },
        }

    def test_proxy_booleans_are_withheld_from_geometry_payload(self) -> None:
        evidence = _sanitized_route_evidence(self._sample())
        self.assertNotIn("future_corridor_intrusion", evidence)
        self.assertNotIn("future_proximity_le_1_25m", evidence)
        self.assertEqual(evidence["source_native_geometry_only"], True)
        self.assertEqual(evidence["human_event_truth"], False)

    def test_round_robin_selection_keeps_sessions_distinct(self) -> None:
        rows = [
            {"candidate_id": "a1", "source_session_id": "a", "start_timestamp_ns": 1, "_route_has_intrusion_proxy": True, "_route_has_proximity_proxy": True},
            {"candidate_id": "a2", "source_session_id": "a", "start_timestamp_ns": 2, "_route_has_intrusion_proxy": False, "_route_has_proximity_proxy": False},
            {"candidate_id": "b1", "source_session_id": "b", "start_timestamp_ns": 1, "_route_has_intrusion_proxy": False, "_route_has_proximity_proxy": True},
        ]
        selected = _select_round_robin(rows, 2)
        self.assertEqual([(row["source_session_id"], row["candidate_id"]) for row in selected], [("a", "a1"), ("b", "b1")])


if __name__ == "__main__":
    unittest.main()
