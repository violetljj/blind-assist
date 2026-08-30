from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_dtr_carla_n4_multitown_frozen_replay import (  # noqa: E402
    build_event_bearing_route,
)
from dtr_carla_n3_multitown_native_dynamics import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_MAPS,
    SCENE_ORDER,
    compile_suite,
)


REGISTRY_PATH = HERE / "dtr_carla_n3_multitown_native_dynamics_registry.json"


class N3N4MultitownTest(unittest.TestCase):
    def test_compiler_freezes_exact_three_scene_surface(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        suite, plans = compile_suite(registry)
        self.assertEqual(list(SCENE_ORDER), suite["scene_order"])
        self.assertEqual(list(EXPECTED_MAPS), [value["map"] for value in suite["scenes"]])
        self.assertEqual(
            list(EXPECTED_CLASSES), [value["scenario_class"] for value in suite["scenes"]]
        )
        self.assertEqual(set(SCENE_ORDER), set(plans))
        for scene in suite["scenes"]:
            plan = plans[scene["scene_id"]]
            self.assertEqual(
                {"heavy_vehicle", "two_wheeler"},
                set(plan["suite_scene"]["required_native_vehicle_classes"]),
            )
            self.assertEqual(4, len(plan["tail_events"]))
            self.assertGreaterEqual(len(plan["walker_intents"]), 10)
            vehicle_event_actors = {
                event["primary_actor_id"]
                for event in plan["tail_events"]
                if event["type"] != "occluded_jaywalk"
            }
            self.assertEqual(1, len(vehicle_event_actors))
            self.assertEqual(
                "N3_V2_ROUTE_FEASIBILITY_DIAGNOSTIC",
                plan["suite_scene"]["event_actor_binding_authority"],
            )
            self.assertEqual(1.8, plan["suite_scene"]["maximum_wearer_speed_mps"])

    def test_event_bearing_route_centers_all_four_windows(self) -> None:
        rows = []
        actor_ids = [f"actor_{index}" for index in range(4)]
        for sample_index in range(81):
            rows.append(
                {
                    "sample_index": sample_index,
                    "time_s": sample_index * 0.05,
                    "actors": {
                        actor_id: {
                            "transform": {
                                "x": float(index * 2 + sample_index * 0.02),
                                "y": float(index),
                                "z": 0.0,
                                "yaw": float(index * 30),
                            }
                        }
                        for index, actor_id in enumerate(actor_ids)
                    },
                }
            )
        events = {
            "tail_events": [
                {
                    "event_id": f"event_{index}",
                    "type": event_type,
                    "primary_actor_id": actor_ids[index],
                    "applied_time_s": 0.4 + index * 0.9,
                    "ended_time_s": 0.8 + index * 0.9,
                }
                for index, event_type in enumerate(
                    ("occluded_jaywalk", "sudden_brake", "reverse_pullout", "door_open")
                )
            ]
        }
        route, audit = build_event_bearing_route(
            rows,
            events,
            fixed_delta_seconds=0.05,
            view_distance_m=4.5,
            maximum_event_view_range_m=12.0,
            maximum_wearer_speed_mps=1.8,
        )
        self.assertEqual(len(rows), len(route))
        self.assertEqual(4, audit["event_count"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertLessEqual(audit["maximum_bearing_error_degrees"], 1e-3)
        self.assertLessEqual(audit["maximum_route_speed_mps"], 1.8001)


if __name__ == "__main__":
    unittest.main()
