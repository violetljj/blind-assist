import json
import tempfile
import unittest
from pathlib import Path

from normalize_completed_review_fields import normalize


class NormalizeCompletedReviewFieldsTest(unittest.TestCase):
    def _row(self) -> dict[str, object]:
        return {
            "record_kind": "REVIEW_OUTPUT",
            "schema": "hftf_d7_public_real_review_output_v1",
            "candidate_id": "candidate-a",
            "event_bucket": "NORMAL_WALKABLE_NEGATIVE",
        }

    def test_model_blind_default_requires_manifest_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "review.jsonl"
            manifest = root / "manifest.jsonl"
            output.write_text(json.dumps(self._row()) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({
                    "candidate_id": "candidate-a",
                    "review_input_id": "input-a",
                    "model_output_visible": False,
                }) + "\n",
                encoding="utf-8",
            )
            result = normalize(
                output,
                1,
                "SUPPORT",
                role="RGB_REVIEWER_B",
                downgrade_incomplete_support=True,
                manifest_path=manifest,
                assume_model_blind_from_manifest=True,
                canonicalize_completed_review=True,
            )
            self.assertEqual(result["status"], "NORMALIZED")
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(row["model_output_visible"])
            self.assertEqual(row["review_input_id"], "input-a")
            self.assertEqual(row["schema"], "hftf_d7_public_real_completed_review_v1")
            self.assertEqual(row["record_kind"], "COMPLETED_REVIEW")
            self.assertTrue(row["review_completed"])
            self.assertEqual(row["decision"], "NOT_EVALUABLE")
            self.assertEqual(row["event_bucket"], "NOT_EVALUABLE")

    def test_model_blind_default_fails_without_manifest_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "review.jsonl"
            manifest = root / "manifest.jsonl"
            output.write_text(json.dumps(self._row()) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({
                    "candidate_id": "candidate-a",
                    "review_input_id": "input-a",
                    "model_output_visible": True,
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                normalize(
                    output,
                    1,
                    "SUPPORT",
                    role="RGB_REVIEWER_B",
                    manifest_path=manifest,
                    assume_model_blind_from_manifest=True,
                )

    def test_legacy_negative_interval_binds_to_manifest_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "review.jsonl"
            manifest = root / "manifest.jsonl"
            row = self._row() | {
                "decision": "SUPPORT",
                "continuous_negative_interval": {
                    "start_frame_index": 0,
                    "end_frame_index": 19,
                },
                "model_output_visible": False,
            }
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({
                    "candidate_id": "candidate-a",
                    "review_input_id": "input-a",
                    "review_index": 0,
                    "model_output_visible": False,
                    "window_start_timestamp_ns": 100,
                    "window_end_timestamp_ns": 200,
                }) + "\n",
                encoding="utf-8",
            )
            normalize(
                output,
                1,
                None,
                role="RGB_REVIEWER_B",
                downgrade_incomplete_support=True,
                manifest_path=manifest,
                canonicalize_completed_review=True,
                bind_support_intervals_from_manifest=True,
            )
            normalized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(normalized["decision"], "SUPPORT")
            self.assertEqual(
                normalized["phase_intervals"]["continuous_negative_interval"],
                {
                    "end_frame_index": 19,
                    "end_timestamp_ns": 200,
                    "interval_binding": "IMMUTABLE_REVIEW_MANIFEST_WINDOW",
                    "start_frame_index": 0,
                    "start_timestamp_ns": 100,
                },
            )

    def test_nested_negative_interval_binds_to_manifest_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "review.jsonl"
            manifest = root / "manifest.jsonl"
            row = self._row() | {
                "decision": "SUPPORT",
                "phase_intervals": {
                    "continuous_negative_interval": {
                        "start_frame_index": 0,
                        "end_frame_index": 19,
                    }
                },
                "model_output_visible": False,
            }
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest.write_text(
                json.dumps({
                    "candidate_id": "candidate-a",
                    "review_input_id": "input-a",
                    "model_output_visible": False,
                    "window_start_timestamp_ns": 100,
                    "window_end_timestamp_ns": 200,
                }) + "\n",
                encoding="utf-8",
            )
            normalize(
                output,
                1,
                None,
                role="RGB_REVIEWER_B",
                downgrade_incomplete_support=True,
                manifest_path=manifest,
                canonicalize_completed_review=True,
                bind_support_intervals_from_manifest=True,
            )
            normalized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(normalized["decision"], "SUPPORT")
            self.assertEqual(
                normalized["phase_intervals"]["continuous_negative_interval"]["start_timestamp_ns"],
                100,
            )

    def test_not_evaluable_empty_phase_list_is_canonicalized_to_null(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.jsonl"
            path.write_text(json.dumps({
                "candidate_id": "candidate-a",
                "review_input_id": "input-a",
                "decision": "NOT_EVALUABLE",
                "event_bucket": "NOT_EVALUABLE",
                "phase_intervals": [],
                "model_output_visible": False,
            }) + "\n", encoding="utf-8")
            result = normalize(
                path,
                expected_count=1,
                default_decision=None,
                role="RGB_REVIEWER_A",
                canonicalize_completed_review=True,
            )
            self.assertEqual(result["rows"], 1)
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["phase_intervals"])

    def test_terminal_alias_binds_not_evaluable_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.jsonl"
            path.write_text(json.dumps({
                "candidate_id": "candidate-a",
                "review_input_id": "input-a",
                "terminal": "NOT_EVALUABLE",
                "event_bucket": None,
                "model_output_visible": False,
            }) + "\n", encoding="utf-8")
            normalize(
                path,
                expected_count=1,
                default_decision=None,
                role="GEOMETRY_EVIDENCE_REVIEWER",
                canonicalize_completed_review=True,
            )
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["decision"], "NOT_EVALUABLE")
            self.assertEqual(row["event_bucket"], "NOT_EVALUABLE")
            self.assertEqual(row["terminal"], "NOT_EVALUABLE")

    def test_non_support_legacy_phase_list_is_canonicalized_to_null(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.jsonl"
            path.write_text(json.dumps({
                "candidate_id": "candidate-a",
                "review_input_id": "input-a",
                "decision": "REJECT",
                "event_bucket": "NORMAL_WALKABLE_NEGATIVE",
                "phase_intervals": [{"start_timestamp_ns": 0, "end_timestamp_ns": 1}],
                "model_output_visible": False,
            }) + "\n", encoding="utf-8")
            normalize(
                path,
                expected_count=1,
                default_decision=None,
                role="RGB_REVIEWER_A",
                canonicalize_completed_review=True,
            )
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["phase_intervals"])

    def test_canonicalize_can_rebind_legacy_identity_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "review.jsonl"
            manifest = root / "manifest.jsonl"
            output.write_text(json.dumps({
                "candidate_id": "candidate-a",
                "reviewer_role": "COUNTEREXAMPLE_REVIEWER",
                "review_input_id": "wrong-input",
                "decision": "NOT_EVALUABLE",
                "event_bucket": "NOT_EVALUABLE",
                "model_output_visible": False,
            }) + "\n", encoding="utf-8")
            manifest.write_text(json.dumps({
                "candidate_id": "candidate-a",
                "review_input_id": "input-a",
                "batch_id": "batch-a",
                "model_output_visible": False,
            }) + "\n", encoding="utf-8")
            normalize(
                output,
                1,
                None,
                role="COUNTEREXAMPLE_REVIEWER",
                manifest_path=manifest,
                canonicalize_completed_review=True,
                rebind_identity_from_manifest=True,
            )
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["review_role"], "COUNTEREXAMPLE_REVIEWER")
            self.assertEqual(row["review_input_id"], "input-a")
            self.assertEqual(row["batch_id"], "batch-a")
            self.assertEqual(row["review_role"], "COUNTEREXAMPLE_REVIEWER")
            self.assertTrue(row["counterexample_search_completed"])


if __name__ == "__main__":
    unittest.main()
