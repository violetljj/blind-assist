from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dtr_carla_n2_frozen_trace_replay as subject  # noqa: E402


PROTOCOL_PATH = HERE / "dtr_carla_n2_frozen_trace_replay_protocol.json"


def synthetic_manifest() -> dict:
    return {
        "actors": [
            {
                "actor_id": "actor_01",
                "blueprint_id": "walker.pedestrian.0001",
                "kind": "pedestrian",
            }
        ]
    }


def synthetic_rows() -> list[dict]:
    rows = []
    for sample, world_frame in enumerate((100, 101)):
        rows.append(
            {
                "schema_version": subject.TRACE_SCHEMA,
                "sample_index": sample,
                "time_s": sample * 0.05,
                "world_frame": world_frame,
                "actors": {
                    "actor_01": {
                        "actor_id": "actor_01",
                        "type_id": "walker.pedestrian.0001",
                        "kind": "pedestrian",
                        "transform": {
                            "x": float(sample),
                            "y": 0.0,
                            "z": 1.0,
                            "pitch": 0.0,
                            "yaw": 0.0,
                            "roll": 0.0,
                        },
                    }
                },
            }
        )
    return rows


class FrozenTraceReplayTest(unittest.TestCase):
    def test_production_protocol_validates(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        subject.validate_protocol(protocol)

    def test_frozen_event_bearing_route_protocol_validates(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["environment"]["map"] = "Carla/Maps/Town01"
        protocol["source"]["expected_trace_frames"] = 2
        protocol["capture"]["wearer"] = {
            "observer_mode": "frozen_event_bearing_route",
            "route_authority": "TRACE_CONDITIONED_FROZEN_BEFORE_REPLAY",
            "motion_model": "bounded_speed_planar_wearer_route",
            "maximum_speed_mps": 2.0,
            "maximum_event_view_range_m": 12.0,
            "blueprint": "walker.pedestrian.0001",
            "surface_offset_m": 0.8,
            "body_radius_m": 0.45,
            "route": [
                {"sample_index": 0, "time_s": 0.0, "x_m": 1.0, "y_m": 2.0, "yaw_degrees": 3.0},
                {"sample_index": 1, "time_s": 0.05, "x_m": 1.1, "y_m": 2.0, "yaw_degrees": 3.0},
            ],
        }
        protocol["episode"]["issued_plan_authority"] = "FROZEN_EVENT_BEARING_ROUTE"
        subject.validate_protocol(protocol)

    def test_trace_contract_accepts_contiguous_rows(self) -> None:
        summary = subject.validate_trace_rows(
            synthetic_rows(),
            synthetic_manifest(),
            expected_frames=2,
            expected_actors=1,
            fixed_delta_seconds=0.05,
        )
        self.assertEqual(2, summary["frames"])
        self.assertEqual(["actor_01"], summary["actor_ids"])

    def test_trace_contract_rejects_time_drift(self) -> None:
        rows = synthetic_rows()
        rows[1]["time_s"] = 0.06
        with self.assertRaisesRegex(ValueError, "logical time"):
            subject.validate_trace_rows(
                rows,
                synthetic_manifest(),
                expected_frames=2,
                expected_actors=1,
                fixed_delta_seconds=0.05,
            )

    def test_four_modal_alignment_requires_one_world_frame(self) -> None:
        rows = [
            {
                "sample_index": sample,
                "time_s": sample * 0.05,
                "source_world_frame": 100 + sample,
                "replay_world_frame": 500 + sample,
                "sensor_world_frames": {
                    sensor: 500 + sample for sensor in subject.SENSOR_ORDER
                },
            }
            for sample in range(2)
        ]
        receipt = subject.build_alignment_receipt("A" * 64, rows)
        self.assertEqual("SAME_WORLD_FRAME_FOUR_MODAL_REPLAY_VERIFIED", receipt["authority"])
        broken = copy.deepcopy(rows)
        broken[1]["sensor_world_frames"]["depth"] += 1
        with self.assertRaisesRegex(ValueError, "sensor world frames differ"):
            subject.build_alignment_receipt("A" * 64, broken)

    def test_c2_model_allowlist_still_rejects_actor_truth(self) -> None:
        record = {
            "schema_version": "dtr-c2-model-observation-v2",
            "episode_id": "n1_native_trace_01",
            "sample_index": 0,
            "world_frame": 1,
            "time_s": 0.0,
            "timestamp_s": 0.0,
            "wearable_rgb": {"path": "rgb.png"},
            "metric_depth": {"path": "depth.png"},
            "camera": {"K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            "wearer_pose_current": {"x": 0.0, "y": 0.0, "z": 0.8},
            "navigation": {"issued_plan": {"authority": "NO_PLAN"}},
            "frame_alignment": {"source_trace_sha256": "A" * 64},
        }
        subject.validate_model_record(record)
        record["actors"] = {}
        with self.assertRaises(ValueError):
            subject.validate_model_record(record)


if __name__ == "__main__":
    unittest.main()
