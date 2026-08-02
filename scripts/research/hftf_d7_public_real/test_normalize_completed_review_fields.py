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


if __name__ == "__main__":
    unittest.main()
