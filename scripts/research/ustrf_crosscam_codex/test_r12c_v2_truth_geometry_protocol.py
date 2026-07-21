from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate_r12c_v2_truth_geometry_consistency as subject


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/ustrf_crosscam_truth_geometry_r12c_prereg_v2.json"
CONTINUOUS_V2 = ROOT / "configs/ustrf_crosscam_continuous_events_r12c_seen_v2.json"
CONTINUOUS_R12A = ROOT / "configs/ustrf_crosscam_continuous_events_r12a_seen_v1.json"
R12D_PREREG = ROOT / "configs/ustrf_crosscam_small_target_detector_r12d_prereg_v1.json"


class R12cV2TruthGeometryProtocolTest(unittest.TestCase):
    def test_materialization_changes_only_japan_to_bangkok(self) -> None:
        old = json.loads(CONTINUOUS_R12A.read_text(encoding="utf-8"))
        new = json.loads(CONTINUOUS_V2.read_text(encoding="utf-8"))
        subject.validate_continuous_inventory(new, ROOT)
        old_events = {row["event_id"]: row for row in old["events"]}
        new_events = {row["event_id"]: row for row in new["events"]}
        self.assertEqual(12, len(new_events))
        self.assertNotIn("japan_path_intrusion", new_events)
        self.assertIn("bangkok_tactile_cone_intrusion", new_events)
        for event_id in old_events.keys() - {"japan_path_intrusion"}:
            self.assertEqual(old_events[event_id], new_events[event_id])
        self.assertEqual(6, sum(row["expected_class"] == "positive" for row in new_events.values()))

    def test_protocol_preserves_authority_and_single_candidate(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        subject.validate_protocol(protocol, ROOT)
        candidate = protocol["candidate_inheritance"]
        self.assertEqual("r12c_c1_sameweights_fp16_768_gpu_london_only", candidate["candidate_id"])
        self.assertEqual(1, candidate["candidate_count"])
        self.assertFalse(candidate["additional_resolution_candidates_allowed"])
        policy = protocol["authorization_policy"]
        self.assertTrue(policy["oracle_pass_may_authorize_london_768_candidate_execution"])
        self.assertFalse(policy["oracle_pass_may_authorize_full_continuous_replay"])
        self.assertFalse(policy["oracle_pass_may_authorize_device_soak"])
        self.assertFalse(policy["oracle_pass_may_authorize_r13_inventory_unlock"])

    def test_full_six_positive_oracle_passes_with_bangkok(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subject.evaluate(PROTOCOL, output)
            self.assertTrue(Path(str(output) + ".sha256").is_file())
        self.assertEqual(6, result["positive_event_count"])
        self.assertEqual(6, result["consistent_positive_event_count"])
        self.assertEqual(6, result["eligible_positive_event_count"])
        self.assertEqual(0, result["truth_geometry_conflict_count"])
        self.assertEqual([], result["unresolved_truth_geometry_conflict_event_ids"])
        self.assertTrue(result["all_positive_truth_geometry_consistent"])
        bangkok = next(row for row in result["events"]
                       if row["event_id"] == "bangkok_tactile_cone_intrusion")
        self.assertEqual(2, bangkok["alertable_anchor_count"])
        self.assertEqual(2, bangkok["alertable_robust_inside_count"])
        self.assertTrue(result["authorization"]["london_768_candidate_execution_authorized"])
        self.assertFalse(result["authorization"]["full_continuous_replay_authorized"])
        self.assertFalse(result["authorization"]["device_soak_authorized"])
        self.assertFalse(result["authorization"]["r13_inventory_unlock_authorized"])
        self.assertFalse(result["replacement"]["r13_slot_consumed"])

    def test_inventory_validator_rejects_an_unrelated_event_change(self) -> None:
        continuous = json.loads(CONTINUOUS_V2.read_text(encoding="utf-8"))
        event = next(row for row in continuous["events"]
                     if row["event_id"] == "london_center_marker_intrusion")
        event["alertable_start_ms"] += 1
        with self.assertRaisesRegex(ValueError, "unrelated event changed"):
            subject.validate_continuous_inventory(continuous, ROOT)

    def test_failed_768_closes_resolution_search_and_preregisters_p2_hypothesis(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        prereg = json.loads(R12D_PREREG.read_text(encoding="utf-8"))
        self.assertEqual(subject.sha256_file(PROTOCOL), prereg["trigger"]["r12c_v2_protocol_sha256"])
        self.assertEqual(["london_center_marker_intrusion"], prereg["trigger"]["failed_event_ids"])
        self.assertTrue(prereg["trigger"]["resolution_search_closed"])
        hypothesis = prereg["hypothesis"]
        self.assertEqual("smallest_detection_output_stride_at_most_4",
                         hypothesis["required_architecture_property"])
        self.assertFalse(hypothesis["same_weights_resolution_candidates_allowed"])
        self.assertFalse(hypothesis["tracker_only_candidate_allowed"])
        self.assertFalse(hypothesis["threshold_rescue_allowed"])
        self.assertEqual(0, prereg["candidate_admission"]["candidate_count"])
        self.assertTrue(prereg["candidate_admission"]["training_data_requires_review_consent_and_exact_geometry_receipts"])
        self.assertTrue(all(value is False for value in prereg["authority"].values()))


if __name__ == "__main__":
    unittest.main()
