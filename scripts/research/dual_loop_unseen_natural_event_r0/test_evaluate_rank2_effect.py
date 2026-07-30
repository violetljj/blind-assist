from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.research.dual_loop_unseen_natural_event_r0.finalize_rank2_truth import (
    validate_emitted_ledger,
)
from scripts.research.dual_loop_unseen_natural_event_r0.evaluate_rank2_effect import (
    RANK2_PROTOCOL_PATH,
    RANK2_PROTOCOL_SHA256,
    REPO_ROOT,
    atomic_publish_json_directory,
    classify_terminal,
    first_or_none,
    sha256_file,
    triggers_in_interval,
)


class Rank2EffectEvaluatorTest(unittest.TestCase):
    def test_interval_is_closed_and_ignores_untriggered_rows(self) -> None:
        rows = [
            {"source_capture_timestamp_ns": 99, "feedback_triggered": True},
            {"source_capture_timestamp_ns": 100, "feedback_triggered": True},
            {"source_capture_timestamp_ns": 150, "feedback_triggered": False},
            {"source_capture_timestamp_ns": 200, "feedback_triggered": True},
            {"source_capture_timestamp_ns": 201, "feedback_triggered": True},
        ]
        self.assertEqual(
            triggers_in_interval(rows, 100, 200, "feedback_triggered"),
            [100, 200],
        )

    def test_candidate_miss_remains_explicit(self) -> None:
        self.assertEqual(first_or_none([300, 100, 200]), 100)
        self.assertIsNone(first_or_none([]))

    def test_terminal_precedence(self) -> None:
        passing = {
            "absolute_positive_recall_non_decreasing": True,
            "baseline_hit_retention_complete": True,
            "timely_baseline_hit_retention_complete": True,
            "induced_negative_windows_zero": True,
            "risk_invariants_preserved": True,
        }
        failed = dict(passing)
        failed["baseline_hit_retention_complete"] = False
        self.assertEqual(
            classify_terminal(failed, corrected_negative_windows=2),
            ("FIRST_UNSEEN_SOURCE_GUARDRAIL_FAILED", "ACTIVE_R1_REJECTED"),
        )
        self.assertEqual(
            classify_terminal(passing, corrected_negative_windows=0),
            (
                "FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT",
                "DENSITY_SIGNAL_ONLY",
            ),
        )
        self.assertEqual(
            classify_terminal(passing, corrected_negative_windows=1),
            (
                "FIRST_UNSEEN_SOURCE_EVENT_SIGNAL",
                "SECOND_INDEPENDENT_SESSION_REQUIRED",
            ),
        )

    def test_kotlin_and_protocol_hash_bindings_are_current(self) -> None:
        evaluator = REPO_ROOT / (
            "scripts/research/dual_loop_unseen_natural_event_r0/"
            "evaluate_rank2_effect.py"
        )
        kotlin = REPO_ROOT / (
            "device-benchmark/src/main/java/com/linnan/blindassist/benchmark/"
            "ProductionTemporalGeometryFactorialAbDeviceTest.kt"
        )
        self.assertIn(
            sha256_file(evaluator),
            kotlin.read_text(encoding="utf-8"),
        )
        self.assertEqual(sha256_file(RANK2_PROTOCOL_PATH), RANK2_PROTOCOL_SHA256)
        self.assertIn(
            RANK2_PROTOCOL_SHA256,
            kotlin.read_text(encoding="utf-8"),
        )

    def test_truth_producer_rejects_overlapping_metric_intervals(self) -> None:
        rows = [
            {
                "item_id": "P1",
                "should_alert": True,
                "start_ns": 0,
                "alertable_start_ns": 100,
                "end_ns": 200,
                "region": "center",
                "truth_confidence": 0.9,
            },
            {
                "item_id": "N1",
                "should_alert": False,
                "start_ns": 200,
                "alertable_start_ns": None,
                "end_ns": 300,
                "region": "center",
                "truth_confidence": 0.9,
            },
        ]
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / "truth.jsonl"
            ledger.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_emitted_ledger(ledger)

    def test_atomic_json_publication_is_lf_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "published"
            atomic_publish_json_directory(
                output,
                {"value.json": {"status": "COMPLETE", "count": 1}},
            )
            payload = (output / "value.json").read_bytes()
            self.assertNotIn(b"\r\n", payload)
            self.assertTrue(payload.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
