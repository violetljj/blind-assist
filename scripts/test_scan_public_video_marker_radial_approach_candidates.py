import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_public_video_marker_radial_approach_candidates as subject


class MarkerRadialApproachCandidateScanTest(unittest.TestCase):
    def test_only_passing_events_are_emitted(self):
        # candidate_rows delegates event construction to the frozen probe; this
        # test protects the final fail-closed filter without duplicating it.
        original = subject.probe.diagnose
        subject.probe.diagnose = lambda _f, _c: [{"source_id": "s", "events": [
            {"radial_approach_passed": True, "event_entry_timestamp_ms": 1},
            {"radial_approach_passed": False, "event_entry_timestamp_ms": 2},
        ]}]
        try:
            rows = subject.candidate_rows({}, {})
        finally:
            subject.probe.diagnose = original
        self.assertEqual([1], [row["event_entry_timestamp_ms"] for row in rows[0]["events"]])


if __name__ == "__main__":
    unittest.main()
