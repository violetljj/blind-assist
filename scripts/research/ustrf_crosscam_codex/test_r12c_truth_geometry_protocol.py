from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate_r12c_truth_geometry_consistency as subject


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/ustrf_crosscam_truth_geometry_r12c_prereg_v1.json"
R13_V2 = ROOT / "configs/ustrf_crosscam_continuous_events_r13_prereg_v2.json"
ADJUDICATION = ROOT / "artifacts.local/evidence/ustrf-crosscam-codex/truth-geometry-r12c-seen-diagnostic-v1/japan_model_adjudication_v1.json"


class R12cTruthGeometryProtocolTest(unittest.TestCase):
    def test_protocol_preserves_single_variable_and_hard_stops(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        subject.validate_protocol(protocol, ROOT)
        candidate = protocol["london_single_variable_candidate"]
        self.assertEqual("london_center_marker_intrusion", candidate["target_event_id"])
        self.assertEqual(768, candidate["input_size"])
        self.assertEqual("fp16", candidate["precision"])
        self.assertEqual("gpu_delegate", candidate["execution_backend"])
        self.assertEqual(1, protocol["execution_sequence"]["candidate_count"])
        self.assertFalse(protocol["hard_stops"]["fp16_320_allowed"])
        self.assertFalse(protocol["hard_stops"]["int8_allowed"])
        self.assertFalse(protocol["hard_stops"]["tracker_optimization_allowed"])

    def test_current_oracle_fails_closed_on_japan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subject.evaluate(PROTOCOL, output)
            sidecar = Path(str(output) + ".sha256")
            self.assertTrue(sidecar.is_file())
        self.assertEqual(6, result["positive_event_count"])
        self.assertEqual(5, result["consistent_positive_event_count"])
        self.assertEqual(["japan_path_intrusion"], result["truth_geometry_conflict_event_ids"])
        japan = next(row for row in result["events"] if row["event_id"] == "japan_path_intrusion")
        self.assertEqual(2, japan["alertable_anchor_count"])
        self.assertEqual(0, japan["alertable_robust_inside_count"])
        self.assertEqual("truth_geometry_conflict", japan["status"])
        self.assertFalse(result["authorization"]["london_768_candidate_execution_authorized"])
        self.assertFalse(result["authorization"]["r13_inventory_unlock_authorized"])

    def test_model_adjudication_excludes_japan_without_waiting_for_a_real_human(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subject.evaluate(PROTOCOL, Path(directory) / "result.json", ADJUDICATION)
        self.assertEqual([], result["unresolved_truth_geometry_conflict_event_ids"])
        self.assertEqual(1, result["adjudicated_excluded_positive_event_count"])
        self.assertEqual(5, result["eligible_positive_event_count"])
        japan = next(row for row in result["events"] if row["event_id"] == "japan_path_intrusion")
        self.assertEqual("adjudicated_exclude_from_score", japan["status"])
        self.assertEqual("reject_as_strict_positive", japan["positive_gate_decision"])
        self.assertFalse(result["authorization"]["london_768_candidate_execution_authorized"])
        self.assertEqual(
            "preregister_one_non_r13_seen_positive_with_independent_event_truth_and_route_geometry_then_rerun_r12c",
            result["next_action"],
        )

    def test_r13_v2_is_sealed_and_routes_conflicts_to_third_model_adjudication(self) -> None:
        prereg = json.loads(R13_V2.read_text(encoding="utf-8"))
        self.assertEqual(12, len(prereg["inventory"]["source_slots"]))
        self.assertFalse(prereg["novelty_and_access"]["source_discovery_authorized"])
        self.assertFalse(prereg["novelty_and_access"]["download_decode_or_detector_inference_authorized"])
        self.assertFalse(prereg["novelty_and_access"]["result_access_authorized"])
        review = prereg["event_truth_review"]
        self.assertEqual("third_independent_model_adjudication_fail_closed_no_score", review["disagreement_policy"])
        self.assertEqual("third_independent_model_adjudication_fail_closed_no_score", review["truth_geometry_conflict_policy"])
        self.assertTrue(review["model_review_may_progress_research_without_waiting_for_a_real_human"])
        self.assertFalse(review["model_review_grants_human_or_production_truth_authority"])
        self.assertFalse(review["manual_polygon_drag_or_refit_to_rescue_result_allowed"])


if __name__ == "__main__":
    unittest.main()
