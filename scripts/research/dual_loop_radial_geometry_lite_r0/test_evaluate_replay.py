from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
EVALUATOR = importlib.import_module("evaluate_replay")
GEOMETRY = importlib.import_module("radial_geometry")


def output(frame: int, arm: str, score: float | None, abstention: str | None = None) -> dict:
    captured = frame * 40_000_000
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
        "quality": {"score": 1.0 if abstention is None else 0.0, "components": {}},
        "ttl_ns": 100_000_000,
        "valid_until_ns": captured + 100_000_000,
        "abstention_reason": abstention,
    }


def fixture(truth_state: str = "approaching") -> tuple[list[dict], list[dict], list[dict]]:
    truth = []
    outputs = []
    for frame in range(5):
        truth.append({
            "source_frame_id": f"f{frame}",
            "target_id": "track-000",
            "event_id": "event-0",
            "primary_event_eligible": True,
        })
        outputs.append(output(frame, GEOMETRY.ARM_BBOX, -0.10))
        outputs.append(output(frame, GEOMETRY.ARM_FLOW, 0.10))
    events = [{
        "event_id": "event-0",
        "target_id": "track-000",
        "anchor_region": "CENTER",
        "truth_state": truth_state,
        "eligible_frame_count": 5,
        "primary_event_eligible": True,
    }]
    return outputs, truth, events


def replay_row(frame: int) -> dict:
    return {
        "source_frame_id": f"f{frame}",
        "captured_at_ns": frame * 40_000_000,
        "target_id": "track-000",
        "track_epoch": "track-000:epoch-0001",
        "region": "CENTER",
    }


class EvaluateReplayTest(unittest.TestCase):
    def test_event_score_uses_fixed_denominator_and_wrong_sign(self) -> None:
        outputs, truth, events = fixture()
        bbox = EVALUATOR._summarize_event_rows(
            [row for row in outputs if row["arm_id"] == GEOMETRY.ARM_BBOX],
            events[0],
        )
        flow = EVALUATOR._summarize_event_rows(
            [row for row in outputs if row["arm_id"] == GEOMETRY.ARM_FLOW],
            events[0],
        )
        self.assertTrue(bbox["wrong_signed"])
        self.assertFalse(bbox["correct"])
        self.assertTrue(flow["correct"])
        self.assertEqual(flow["denominator_rows"], 5)

    def test_quasi_direction_error_is_not_wrong_signed(self) -> None:
        outputs, _, events = fixture("quasi_static")
        row = EVALUATOR._summarize_event_rows(
            [item for item in outputs if item["arm_id"] == GEOMETRY.ARM_FLOW],
            events[0],
        )
        self.assertFalse(row["correct"])
        self.assertFalse(row["wrong_signed"])

    def test_abstention_stays_in_denominator(self) -> None:
        rows = [
            output(0, GEOMETRY.ARM_FLOW, None, "INSUFFICIENT_HISTORY"),
            output(1, GEOMETRY.ARM_FLOW, None, "SURVIVING_TRACKS_LT_8"),
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
        self.assertTrue(result["evaluable"])
        self.assertEqual(result["coverage"], 0.6)
        rows[2] = output(2, GEOMETRY.ARM_FLOW, None, "SURVIVING_TRACKS_LT_8")
        result = EVALUATOR._summarize_event_rows(rows, event)
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["coverage"], 0.4)

    def test_full_replay_keyset_cannot_drop_nonprimary_rows(self) -> None:
        outputs, truth, events = fixture()
        replay_rows = [replay_row(frame) for frame in range(5)]
        EVALUATOR.evaluate_records(outputs, truth, events, replay_rows)
        with self.assertRaisesRegex(ValueError, "keyset drift"):
            EVALUATOR.evaluate_records(outputs[:-1], truth, events, replay_rows)

    def test_truth_is_not_opened_before_exact_keyset_validation(self) -> None:
        outputs, _, _ = fixture()
        outputs.pop()
        replay_rows = [replay_row(frame) for frame in range(5)]
        calls: list[str] = []

        def read_spy(path: Path) -> list[dict]:
            calls.append(path.name)
            if path.name == "producer.jsonl":
                return outputs
            if path.name == "replay_input.jsonl":
                return replay_rows
            self.fail("truth/event ledger opened before keyset rejection")

        with tempfile.TemporaryDirectory(prefix="radial-evaluator-") as directory:
            root = Path(directory)
            producer = root / "producer.jsonl"
            replay = root / "replay_input.jsonl"
            receipt = root / "producer.receipt.json"
            lock = root / "implementation.json"
            truth = root / "truth.jsonl"
            events = root / "natural_events.jsonl"
            for path in (producer, replay, truth, events):
                path.write_text("{}\n", encoding="utf-8")
            receipt.write_text(json.dumps({
                "output_sha256": EVALUATOR.sha256_file(producer),
                "replay_input_sha256": EVALUATOR.sha256_file(replay),
                "output_rows": 10,
                "truth_joined": False,
            }), encoding="utf-8")
            lock.write_text(json.dumps({
                "schema": "blindassist_dual_loop_radial_geometry_implementation_lock_v1",
                "producer_contract": {
                    "input_allowlist": {
                        "replay_input": {
                            "sha256": EVALUATOR.sha256_file(replay),
                            "rows": 5,
                        },
                    },
                    "output_rows_expected": 10,
                },
                "evaluator_contract": {
                    "truth": {"sha256": EVALUATOR.sha256_file(truth), "rows": 1},
                    "natural_events": {"sha256": EVALUATOR.sha256_file(events), "rows": 1},
                },
            }), encoding="utf-8")
            with mock.patch.object(EVALUATOR, "read_jsonl", side_effect=read_spy):
                with self.assertRaisesRegex(ValueError, "keyset drift"):
                    EVALUATOR.evaluate_files(
                        lock,
                        producer,
                        receipt,
                        replay,
                        truth,
                        events,
                        root / "evaluation.json",
                    )
        self.assertEqual(calls, ["producer.jsonl", "replay_input.jsonl"])

    def test_evaluation_output_cannot_overwrite_an_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "collides"):
            EVALUATOR._assert_output_distinct(
                Path("producer.jsonl"),
                [Path("producer.jsonl")],
            )

    def test_output_contract_rejects_ttl_and_quality_drift(self) -> None:
        replay_rows = [replay_row(0)]
        rows = [
            output(0, GEOMETRY.ARM_BBOX, 0.0),
            output(0, GEOMETRY.ARM_FLOW, 0.0),
        ]
        rows[0]["ttl_ns"] = 50_000_000
        rows[0]["valid_until_ns"] = rows[0]["captured_at_ns"] + rows[0]["ttl_ns"]
        with self.assertRaisesRegex(ValueError, "TTL value"):
            EVALUATOR.validate_output_ledger(rows, replay_rows)
        rows[0] = output(0, GEOMETRY.ARM_BBOX, 0.0)
        rows[0]["quality"] = {"score": 1.1, "components": {}}
        with self.assertRaisesRegex(ValueError, "quality score"):
            EVALUATOR.validate_output_ledger(rows, replay_rows)


if __name__ == "__main__":
    unittest.main()
