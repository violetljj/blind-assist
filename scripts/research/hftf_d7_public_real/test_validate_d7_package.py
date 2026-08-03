import json
import tempfile
import unittest
from pathlib import Path

from validate_d7_package import _validate_adjudicated


class ValidateD7PackageTest(unittest.TestCase):
    def test_canonical_adjudicated_event_without_decision_is_not_assignment_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adjudicated_events.jsonl"
            path.write_text(json.dumps({
                "schema": "hftf_d7_public_real_adjudicated_event_v1",
                "record_kind": "ADJUDICATED_EVENT",
                "event_id": "event-1",
                "admission_status": "ADMITTED",
                "event_bucket": "NORMAL_WALKABLE_NEGATIVE",
                "continuous_negative_interval": {
                    "start_timestamp_ns": 100,
                    "end_timestamp_ns": 200,
                },
                "review_model_output_visible": False,
                "geometry_model_output_visible": False,
            }) + "\n", encoding="utf-8")
            rows, errors = _validate_adjudicated(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
