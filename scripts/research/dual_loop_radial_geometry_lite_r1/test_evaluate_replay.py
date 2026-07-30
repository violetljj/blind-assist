from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
EVALUATOR = importlib.import_module("evaluate_replay")
GEOMETRY = importlib.import_module("radial_geometry")


def output(
    frame: int,
    arm: str,
    score: float | None,
    abstention: str | None = None,
) -> dict:
    captured = frame * 40_000_000
    components = {}
    if abstention == "FRAME_SHAPE_CHANGE":
        components = {
            "previous_frame_shape_hw": [160, 160],
            "current_frame_shape_hw": [158, 160],
        }
    return {
        "protocol_id": GEOMETRY.PROTOCOL_ID,
        "implementation_id": GEOMETRY.IMPLEMENTATION_ID,
        "parameter_sha256": GEOMETRY.PARAMETER_SHA256,
        "arm_id": arm,
        "capture_id": "REVEL_DYNAMIC_V1",
        "source_frame_id": f"f{frame}",
        "captured_at_ns": captured,
        "available_at_ns": captured,
        "target_id": "track-000",
        "track_epoch": "track-000:epoch-0001",
        "region": "CENTER",
        "signed_approach_rate_per_s": score,
        "quality": {
            "score": 1.0 if abstention is None else 0.0,
            "components": components,
        },
        "ttl_ns": 100_000_000,
        "valid_until_ns": captured + 100_000_000,
        "abstention_reason": abstention,
    }


def replay_row(frame: int) -> dict:
    return {
        "source_frame_id": f"f{frame}",
        "captured_at_ns": frame * 40_000_000,
        "target_id": "track-000",
        "track_epoch": "track-000:epoch-0001",
        "region": "CENTER",
    }


class EvaluateReplayTest(unittest.TestCase):
    def test_shape_change_requires_both_arms(self) -> None:
        rows = [
            output(0, GEOMETRY.ARM_BBOX, None, "FRAME_SHAPE_CHANGE"),
            output(0, GEOMETRY.ARM_FLOW, None, "FRAME_SHAPE_CHANGE"),
        ]
        EVALUATOR.validate_output_ledger(rows, [replay_row(0)])
        with self.assertRaisesRegex(ValueError, "both arms"):
            broken = [
                rows[0],
                output(0, GEOMETRY.ARM_FLOW, None, "FEATURES_LT_8"),
            ]
            EVALUATOR.validate_output_ledger(broken, [replay_row(0)])

    def test_shape_change_requires_exact_dimensions_and_zero_quality(self) -> None:
        rows = [
            output(0, GEOMETRY.ARM_BBOX, None, "FRAME_SHAPE_CHANGE"),
            output(0, GEOMETRY.ARM_FLOW, None, "FRAME_SHAPE_CHANGE"),
        ]
        rows[0]["quality"]["components"]["current_frame_shape_hw"] = [160, 160]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            EVALUATOR.validate_output_ledger(rows, [replay_row(0)])

    def test_r0_scientific_gate_literals_are_frozen(self) -> None:
        contract = EVALUATOR.SCIENTIFIC_GATE_CONTRACT
        self.assertEqual(EVALUATOR.PRIMARY_DEADBAND_PER_S, 0.02)
        self.assertEqual(contract["fixed_primary_event_denominator"], 469)
        self.assertEqual(
            contract["flow_over_bbox"]["maximum_evaluable_event_loss"],
            23,
        )
        self.assertEqual(
            contract["readiness_floor"]["minimum_correct_fraction"],
            0.60,
        )

    def test_abstention_remains_in_fixed_event_denominator(self) -> None:
        rows = [
            output(0, GEOMETRY.ARM_FLOW, None, "INSUFFICIENT_HISTORY"),
            output(1, GEOMETRY.ARM_FLOW, None, "FRAME_SHAPE_CHANGE"),
            output(2, GEOMETRY.ARM_FLOW, 0.1),
            output(3, GEOMETRY.ARM_FLOW, 0.1),
            output(4, GEOMETRY.ARM_FLOW, 0.1),
        ]
        event = {
            "event_id": "event-0",
            "target_id": "track-000",
            "anchor_region": "CENTER",
            "truth_state": "approaching",
            "eligible_frame_count": 5,
        }
        result = EVALUATOR._summarize_event_rows(rows, event)
        self.assertEqual(result["denominator_rows"], 5)
        self.assertEqual(result["coverage"], 0.6)
        self.assertTrue(result["evaluable"])

    def test_pretruth_gate_rejects_lock_and_nonformal_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-r1-evaluator-") as directory:
            root = Path(directory)
            lock = root / "implementation.json"
            lock.write_text(json.dumps({
                "schema": "blindassist_dual_loop_radial_geometry_implementation_lock_v1",
                "protocol_id": GEOMETRY.PROTOCOL_ID,
                "implementation_id": GEOMETRY.IMPLEMENTATION_ID,
                "parameter_sha256": GEOMETRY.PARAMETER_SHA256,
                "scientific_gate_contract_sha256": (
                    EVALUATOR.SCIENTIFIC_GATE_CONTRACT_SHA256
                ),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from activation"):
                EVALUATOR.evaluate_files(
                    lock,
                    "0" * 64,
                    root / "audit.json",
                    root / "producer.jsonl",
                    root / "receipt.json",
                    root / "replay.jsonl",
                    root / "truth.jsonl",
                    root / "events.jsonl",
                    root / "evaluation.json",
                )
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "status": "PRODUCER_COMPLETE",
                "mode": "pilot",
                "truth_joined": False,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formal terminal"):
                EVALUATOR.evaluate_files(
                    lock,
                    EVALUATOR.sha256_file(lock),
                    root / "audit.json",
                    root / "producer.jsonl",
                    receipt,
                    root / "replay.jsonl",
                    root / "truth.jsonl",
                    root / "events.jsonl",
                    root / "evaluation.json",
                )

    def test_shape_rows_bind_to_source_audit_keys_and_arm_components(self) -> None:
        rows = [
            output(0, GEOMETRY.ARM_BBOX, None, "FRAME_SHAPE_CHANGE"),
            output(0, GEOMETRY.ARM_FLOW, None, "FRAME_SHAPE_CHANGE"),
        ]
        audit = {
            "status": "SOURCE_SHAPE_AUDIT_COMPLETE",
            "truth_or_event_accessed": False,
            "shape_mismatch_pair_count": 32,
            "expected_common_shape_abstention_arm_rows": 64,
            "shape_mismatches": [{
                "current_source_frame_id": "f0",
                "target_id": "track-000",
                "previous_shape": "160x160",
                "current_shape": "158x160",
            }],
        }
        EVALUATOR.validate_shape_audit_binding(rows, audit)
        rows[1]["quality"]["components"]["current_frame_shape_hw"] = [157, 160]
        with self.assertRaisesRegex(ValueError, "components differ"):
            EVALUATOR.validate_shape_audit_binding(rows, audit)


if __name__ == "__main__":
    unittest.main()
