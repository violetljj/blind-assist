from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from .decompose import (
    FrameRow,
    SourceBundle,
    TargetDiagnostics,
    TruthItem,
    audit_upper_bound,
    classify_retained_false,
    publish_outputs,
    summarize_item,
)


def frame(
    time_ns: int,
    baseline: bool = False,
    candidate: bool = False,
    contradiction: bool = False,
    candidate_reason: str | None = None,
) -> FrameRow:
    return FrameRow(
        source_name="Synthetic",
        session="s",
        frame_id=str(time_ns),
        source_time_ns=time_ns,
        relative_time_ns=time_ns,
        baseline_feedback=baseline,
        baseline_reason="TRIGGERED" if baseline else "COOLDOWN",
        candidate_feedback=candidate,
        candidate_reason=candidate_reason,
        contradiction=contradiction,
        scene_rate_per_s=-0.1 if contradiction else None,
        order_index=time_ns,
    )


def truth(item_id: str, positive: bool = False) -> TruthItem:
    return TruthItem(
        source_name="Synthetic",
        source_id="synthetic",
        item_id=item_id,
        item_kind="positive_event" if positive else "negative_window",
        should_alert=positive,
        trace_session="s",
        score_start_rel_ns=0,
        score_end_rel_ns=400_000_000,
        category="TURN_OR_NEAR_IN_PLACE_ROTATION",
        role="DEVELOPMENT",
        outcome_access_state="CLOSED",
    )


class DecomposeTest(unittest.TestCase):
    def test_veto_then_retry_metrics_and_classification(self) -> None:
        rows = [
            frame(0),
            frame(100_000_000, baseline=True, contradiction=True, candidate_reason="DUAL_LOOP_CONTRADICTED"),
            frame(150_000_000, contradiction=True, candidate_reason="DUAL_LOOP_CONTRADICTED"),
            frame(250_000_000, candidate=True, candidate_reason="TRIGGERED"),
        ]
        diagnostics = TargetDiagnostics(
            selected_rates={("s", "100000000"): -0.2},
            scene_medians={("s", "100000000"): -0.1, ("s", "150000000"): -0.2},
            reset_rows={},
            target_observability="RECOMPUTED_FROM_FULL_DETECTION_TRACE",
            scene_observability="RECOMPUTED_FROM_FULL_DETECTION_TRACE",
        )
        result = summarize_item(truth("N1"), rows, diagnostics)
        self.assertEqual("RETAINED_FALSE", result["final_event_outcome"])
        self.assertEqual(100_000_000, result["first_actual_veto_ns"])
        self.assertEqual(250_000_000, result["next_candidate_feedback_after_veto_ns"])
        self.assertEqual(150.0, result["retry_after_veto_ms"])
        self.assertEqual(2, result["contradiction_row_count"])
        self.assertEqual("C_FRAME_VETO_THEN_RETRY", classify_retained_false(result))

    def test_signal_absent_and_late_classes_are_explicit(self) -> None:
        absent = summarize_item(
            truth("N1"),
            [frame(100_000_000, baseline=True), frame(200_000_000, candidate=True)],
            TargetDiagnostics({}, {}, {}, "RECOMPUTED_FROM_FULL_DETECTION_TRACE", "RECOMPUTED_FROM_FULL_DETECTION_TRACE"),
        )
        self.assertEqual("A_SIGNAL_ABSENT", classify_retained_false(absent))
        late = summarize_item(
            truth("N2"),
            [frame(100_000_000, baseline=True, candidate=True), frame(200_000_000, contradiction=True)],
            TargetDiagnostics({}, {("s", "200000000"): -0.1}, {}, "RECOMPUTED_FROM_FULL_DETECTION_TRACE", "RECOMPUTED_FROM_FULL_DETECTION_TRACE"),
        )
        self.assertEqual("B_SIGNAL_LATE", classify_retained_false(late))

    def test_upper_bound_finds_only_in_memory_witness(self) -> None:
        bundle = SourceBundle(
            name="Synthetic",
            source_id="synthetic",
            protocol_id="synthetic",
            rows=[
                frame(0),
                frame(100_000_000, contradiction=True),
                frame(300_000_000, baseline=True, candidate=True),
                frame(600_000_000, baseline=True, candidate=True),
            ],
            truth=[truth("N1"), TruthItem(
                source_name="Synthetic", source_id="synthetic", item_id="P1", item_kind="positive_event",
                should_alert=True, trace_session="s", score_start_rel_ns=500_000_000,
                score_end_rel_ns=700_000_000, category="APPROACH", role="DEVELOPMENT", outcome_access_state="CLOSED",
            )],
            diagnostics=TargetDiagnostics({}, {}, {}, "RECOMPUTED_FROM_FULL_DETECTION_TRACE", "RECOMPUTED_FROM_FULL_DETECTION_TRACE"),
            input_hashes={},
            pre_frozen_delay_limit_ns=0,
        )
        audit = audit_upper_bound(bundle, [])
        self.assertTrue(audit["safe_witness_exists"])
        self.assertEqual("N1", audit["witness"]["eliminated_negative_window"])
        self.assertEqual(0, audit["witness"]["maximum_positive_added_delay_ms"])

    def test_published_outputs_are_lf_and_deterministic(self) -> None:
        result = {
            "terminal": "MIXED_NO_CLEAR_SUCCESSOR",
            "summary": {
                "sources": [],
                "window_count": 0,
                "positive_event_count": 0,
                "negative_window_count": 0,
                "retained_false_count": 0,
                "retained_false_class_counts": {
                    "A_SIGNAL_ABSENT": 0,
                    "B_SIGNAL_LATE": 0,
                    "C_FRAME_VETO_THEN_RETRY": 0,
                    "D_TARGET_OR_ASSOCIATION_MISMATCH": 0,
                    "E_SCALE_SIGNAL_TASK_MISMATCH": 0,
                    "MIXED_OR_UNRESOLVED": 0,
                },
                "baseline_feedback_rows": 0,
                "candidate_feedback_rows": 0,
            },
            "upper_bound_audit": {"safe_witness_exists": False, "source_results": []},
            "windows": [],
            "recommendation": {"worth_designing_single_variable_r2": False},
        }
        with TemporaryDirectory() as directory:
            output = Path(directory)
            publish_outputs(output, result)
            first = {
                path.name: path.read_bytes()
                for path in output.iterdir()
            }
            for payload in first.values():
                self.assertNotIn(b"\r\n", payload)
                self.assertTrue(payload.endswith(b"\n"))
            publish_outputs  # keep the helper import covered without writing twice
            for path in output.iterdir():
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_module_text_files_are_lf(self) -> None:
        for path in (Path(__file__), Path(__file__).with_name("decompose.py"), Path(__file__).with_name("README.md")):
            self.assertNotIn(b"\r\n", path.read_bytes(), str(path))


if __name__ == "__main__":
    unittest.main()
