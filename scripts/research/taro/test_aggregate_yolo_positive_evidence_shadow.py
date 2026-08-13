from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro.aggregate_yolo_positive_evidence_shadow import aggregate


class AggregateYoloPositiveEvidenceShadowTest(unittest.TestCase):
    def test_frozen_four_scene_pass_is_reproduced(self) -> None:
        protocol = self.protocol()
        scenes = [
            self.scene("a", 0.2, 0.5),
            self.scene("b", 0.3, 0.7),
            self.scene("c", 0.4, 0.6),
            self.scene("d", 0.5, 0.4),
        ]

        result = self.run_aggregate(protocol, scenes)

        self.assertEqual("POSE_DIVERSE_POSITIVE_VISUAL_EVIDENCE_PASS", result["terminal"])
        self.assertEqual(3, result["pose_strict_win_scene_count"])
        self.assertEqual(120, result["evaluable_reference_count"])

    def test_model_identity_drift_is_rejected(self) -> None:
        protocol = self.protocol()
        scenes = [self.scene(name, 0.2, 0.5) for name in "abcd"]
        scenes[2]["model_sha256"] = "X" * 64

        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            self.run_aggregate(protocol, scenes)

    def test_incomplete_exact_lookup_is_rejected(self) -> None:
        protocol = self.protocol()
        scenes = [self.scene(name, 0.2, 0.5) for name in "abcd"]
        scenes[1]["exact_pose_payload_lookup_count"] = 29

        with self.assertRaisesRegex(ValueError, "pose arm is not exact-complete"):
            self.run_aggregate(protocol, scenes)

    def run_aggregate(self, protocol: dict, scenes: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            scene_paths = []
            for index, scene in enumerate(scenes):
                path = root / f"scene-{index}.json"
                path.write_text(json.dumps(scene), encoding="utf-8")
                scene_paths.append(path)
            return aggregate(protocol_path, scene_paths, root / "result.json")

    @staticmethod
    def protocol() -> dict:
        return {
            "schema": "blindassist.taro.rgb_pair_frozen_visual_evidence_backend_preflight.v1",
            "unique_successor": "TARO_RGB_PAIR_YOLO_POSITIVE_EVIDENCE_SHADOW_R0",
            "claim_ceiling": "positive-only",
            "backend_audit": {
                "selected": {
                    "model_sha256": "M" * 64,
                    "labels_sha256": "L" * 64,
                    "execution_backend": "CPU_XNNPACK",
                }
            },
            "frozen_shadow_protocol": {
                "required_distinct_scene_parents": 4,
                "minimum_evaluable_references_total": 120,
                "minimum_evaluable_references_per_scene": 20,
                "maximum_evaluable_references_per_scene": 40,
                "minimum_scene_parents_with_positive_support": 4,
                "minimum_opportunity_scene_parents": 3,
                "minimum_pose_strict_win_scene_parents": 3,
                "gates": {"maximum_unique_inference_total_latency_p95_ms": 100},
                "terminal_if_denominator_fails": "NOT_EVALUABLE",
                "terminal_if_runtime_gate_fails": "RUNTIME_FAIL",
                "terminal_if_pose_does_not_beat_passive": "POSE_FAIL",
            },
        }

    @staticmethod
    def scene(scene_id: str, passive: float, pose: float) -> dict:
        return {
            "schema": "blindassist_taro_arcore_yolo_positive_evidence_scene_v1",
            "protocol_id": "TARO_RGB_PAIR_YOLO_POSITIVE_EVIDENCE_SHADOW_R0",
            "scene_id": scene_id,
            "model_sha256": "M" * 64,
            "labels_sha256": "L" * 64,
            "execution_backend": "cpu_xnnpack",
            "availability": "SUPPORTED_INSTALLED",
            "detector_ready_at_start": True,
            "target_evaluable_references": 30,
            "evaluable_reference_count": 30,
            "positive_support_reference_count": 1,
            "opportunity_reference_count": 1,
            "passive_new_focused_token_mean": passive,
            "pose_new_focused_token_mean": pose,
            "detector_total_latency_ms": {"p95": 50},
            "structural_gate_pass": True,
            "source_identity_mismatch_count": 0,
            "selected_payload_lookup_miss_count": 0,
            "exact_passive_payload_lookup_count": 30,
            "exact_pose_payload_lookup_count": 30,
            "model_failure_counts": {},
            "resource_error_counts": {},
            "privacy": {"raw_images_persisted": False, "detections_or_boxes_persisted": False},
            "authorization": {
                "benchmark_only": True,
                "screen_space_positive_evidence_only": True,
                "absence_is_safe": False,
                "production_authorized": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
