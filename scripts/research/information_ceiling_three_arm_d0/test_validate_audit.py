from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image


MODULE_PATH = Path(__file__).with_name("validate_audit.py")
SPEC = importlib.util.spec_from_file_location("information_ceiling_validate_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateAuditTest(unittest.TestCase):
    def build_fixture(
        self,
        root: Path,
        alert_frames_by_arm: dict[str, set[str]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        dataset = root / "dataset"
        (dataset / "images" / "test").mkdir(parents=True)
        (dataset / "source_masks" / "test").mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        event_specs = (
            ("negative", False, "MID"),
            ("positive", True, "NEAR"),
            ("critical", True, "CRITICAL"),
        )
        for event_id, should_alert, distance in event_specs:
            for frame_index in range(2):
                frame_id = f"{event_id}_{frame_index:02d}"
                image_path = dataset / "images" / "test" / f"{frame_id}.png"
                mask_path = dataset / "source_masks" / "test" / f"{frame_id}.png"
                Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
                Image.new("RGB", (32, 32), (3, 0, 0)).save(mask_path)
                is_expected = should_alert and frame_index == 0
                row = {
                    "id": frame_id,
                    "image_path": f"images/test/{frame_id}.png",
                    "split": "test",
                    "width": 32,
                    "height": 32,
                    "sequence_id": event_id,
                    "frame_index": frame_index,
                    "frame_timestamp_ms": frame_index * 100,
                    "source_regions": [
                        {
                            "id": f"{frame_id}_region",
                            "class": "obstacle",
                            "sanpo_class_id": 20,
                            "instance_id": 1,
                            "bbox_xyxy": [8, 8, 24, 30],
                            "pixel_count": 256,
                        }
                    ],
                    "assist_scenario": "OUTDOOR_SLOW",
                    "expected_risk_direction": "CENTER",
                    "expected_distance_band": distance,
                    "expected_should_alert": is_expected,
                    "expected_risk_level": "HIGH" if is_expected else "LOW",
                    "expected_approach_state": "APPROACHING" if is_expected else "STABLE",
                    "expected_approach_alert": is_expected,
                    "expected_time_to_alert_frames": 0 if is_expected else None,
                    "primary_object_id": None,
                    "scene_bucket": event_id,
                    "risk_event_id": event_id,
                    "expected_event_phase": "APPROACHING" if frame_index == 0 else "PASSED",
                    "source_annotation_quality": "HUMAN_ANNOTATED",
                    "source": {
                        "official_split": "test",
                        "sha256": sha256(image_path),
                        "mask_sha256": sha256(mask_path),
                    },
                }
                rows.append(row)
        manifest = dataset / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        dataset_spec = dataset / "dataset_spec.json"
        dataset_spec.write_text("{}\n", encoding="utf-8")
        benchmark = self.make_benchmark(rows, alert_frames_by_arm)
        benchmark_path = root / "benchmark.json"
        benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
        inputs = audit.AuditInputs(
            benchmark_json=benchmark_path,
            manifest=manifest,
            output_dir=root / "output",
            expected_manifest_sha256=sha256(manifest),
            expected_dataset_spec_sha256=sha256(dataset_spec),
            expected_frame_count=6,
            expected_event_count=3,
            verify_source_hashes=True,
        )
        return inputs, rows

    def make_benchmark(
        self,
        rows: list[dict[str, Any]],
        alert_frames_by_arm: dict[str, set[str]],
    ) -> dict[str, Any]:
        models = []
        flags = {
            "A_CURRENT_YOLO": (False, True, False, True),
            "B_ORACLE_RISK_BOX": (True, False, False, False),
            "C_ORACLE_RISK_MASK": (False, False, True, False),
        }
        for arm_id in audit.ARM_IDS:
            box_oracle, include_yolo, mask_oracle, runtime_comparable = flags[arm_id]
            frames = []
            for row in rows:
                image_name = Path(row["image_path"]).name
                expected = audit.normalized_expected_truth(row)
                if arm_id == "B_ORACLE_RISK_BOX":
                    risk_inputs = audit.expected_box_inputs(row)
                else:
                    risk_inputs = []
                frames.append(
                    {
                        "image": image_name,
                        "risk_inputs": risk_inputs,
                        "expected_blindassist": expected,
                        "actual_alert": image_name in alert_frames_by_arm.get(arm_id, set()),
                        "feedback_reason": "TRIGGERED",
                        "stable_model_risk": {
                            "level": "HIGH",
                            "direction": "CENTER",
                            "proximity": row["expected_distance_band"],
                        },
                    }
                )
            models.append(
                {
                    "id": arm_id,
                    "model_asset_sha256": audit.EXPECTED_YOLO_SHA256,
                    "source_region_box_oracle": box_oracle,
                    "include_yolo_risk_inputs": include_yolo,
                    "traversability_oracle": mask_oracle,
                    "runtime_comparable": runtime_comparable,
                    "app_detector": {
                        "decision_kernel_contract_id": "assist-decision-test-v1",
                        "runs_per_image": 1,
                        "failures": [],
                        "per_image": frames,
                        "blindassist_metrics": {},
                    },
                }
            )
        return {
            "schema": "blindassist_detector_ab_device_benchmark_v2",
            "comparison_mode": audit.COMPARISON_MODE,
            "dataset_kind": "BlindAssistEvalSet",
            "risk_config": "current",
            "alert_profile": "STANDARD",
            "synthetic_clock_frame_step_ms": 100,
            "image_count": len(rows),
            "app_runs_per_image": 1,
            "decision_kernel_contract_id": "assist-decision-test-v1",
            "models": models,
        }

    @staticmethod
    def alert_names(*events: str) -> set[str]:
        return {f"{event}_00.png" for event in events}

    def evaluate_terminal(self, alerts: dict[str, set[str]]) -> str:
        with tempfile.TemporaryDirectory() as temp:
            inputs, _ = self.build_fixture(Path(temp), alerts)
            summary, validation, ledger = audit.evaluate_audit(inputs)
            self.assertEqual("PASS", validation["status"], validation["errors"])
            self.assertEqual(9, len(ledger))
            return summary["terminal"]

    def test_detector_gap_terminal(self) -> None:
        terminal = self.evaluate_terminal(
            {
                "A_CURRENT_YOLO": set(),
                "B_ORACLE_RISK_BOX": self.alert_names("positive", "critical"),
                "C_ORACLE_RISK_MASK": self.alert_names("positive", "critical"),
            }
        )
        self.assertEqual("DETECTOR_MODEL_OR_TAXONOMY_GAP_SUPPORTED", terminal)

    def test_mask_adapter_gain_terminal(self) -> None:
        terminal = self.evaluate_terminal(
            {
                "A_CURRENT_YOLO": set(),
                "B_ORACLE_RISK_BOX": set(),
                "C_ORACLE_RISK_MASK": self.alert_names("positive", "critical"),
            }
        )
        self.assertEqual("CURRENT_MASK_ADAPTER_AND_SOURCE_POLICY_GAIN_SUPPORTED", terminal)

    def test_no_oracle_gain_terminal(self) -> None:
        terminal = self.evaluate_terminal(
            {
                "A_CURRENT_YOLO": set(),
                "B_ORACLE_RISK_BOX": set(),
                "C_ORACLE_RISK_MASK": set(),
            }
        )
        self.assertEqual("DECISION_OR_MONOCULAR_OBSERVABILITY_BOTTLENECK_SUPPORTED", terminal)

    def test_mixed_terminal(self) -> None:
        terminal = self.evaluate_terminal(
            {
                "A_CURRENT_YOLO": set(),
                "B_ORACLE_RISK_BOX": self.alert_names("positive"),
                "C_ORACLE_RISK_MASK": self.alert_names("positive", "critical"),
            }
        )
        self.assertEqual("MIXED_DETECTOR_AND_REPRESENTATION_GAPS", terminal)

    def test_box_oracle_leak_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs, _ = self.build_fixture(
                Path(temp),
                {
                    "A_CURRENT_YOLO": set(),
                    "B_ORACLE_RISK_BOX": set(),
                    "C_ORACLE_RISK_MASK": set(),
                },
            )
            benchmark = json.loads(inputs.benchmark_json.read_text(encoding="utf-8"))
            box_model = next(model for model in benchmark["models"] if model["id"] == "B_ORACLE_RISK_BOX")
            box_model["app_detector"]["per_image"][0]["risk_inputs"].append(
                {
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.9,
                    "source": "OBJECT_DETECTOR",
                    "bbox_xyxy": [0, 0, 1, 1],
                    "temporal_promotion_eligible": True,
                }
            )
            inputs.benchmark_json.write_text(json.dumps(benchmark), encoding="utf-8")
            summary, validation, ledger = audit.evaluate_audit(inputs)
            self.assertEqual("FAIL", validation["status"])
            self.assertEqual("NOT_EVALUABLE", summary["terminal"])
            self.assertEqual([], ledger)
            self.assertTrue(any("expected 1 risk inputs, found 2" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
