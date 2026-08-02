from __future__ import annotations

import unittest

from materialize_thor_magni_review_bundle import _dense_sample_times, _select


class MaterializeThorMagniReviewBundleTest(unittest.TestCase):
    def _row(self, session: str, timestamp: int) -> dict[str, object]:
        return {
            "candidate_id": f"candidate-{session}-{timestamp}",
            "dataset_id": "THOR-MAGNI",
            "source_session_id": session,
            "start_timestamp_ns": timestamp,
        }

    def test_selection_is_session_stratified_and_deterministic(self) -> None:
        rows = [self._row("session-b", 20), self._row("session-a", 40), self._row("session-a", 10), self._row("session-c", 0)]
        selected = _select(rows, count=3, session_count=2)
        self.assertEqual(
            [(row["source_session_id"], row["start_timestamp_ns"]) for row in selected],
            [("session-a", 10), ("session-b", 20), ("session-a", 40)],
        )

    def test_non_thor_rows_are_excluded(self) -> None:
        selected = _select([self._row("session-a", 0), {"candidate_id": "other", "dataset_id": "EgoWalk"}], count=1, session_count=1)
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-a-0"])

    def test_dense_times_cover_source_window_with_context(self) -> None:
        row = {"candidate_id": "candidate", "start_timestamp_ns": 1_000_000_000, "end_timestamp_ns": 5_000_000_000}
        times = _dense_sample_times(row, source_start_ns=0, duration_s=10.0, sample_count=30)
        self.assertEqual(len(times), 30)
        self.assertLessEqual(times[0], 1.0)
        self.assertGreaterEqual(times[-1], 5.0)
        self.assertEqual(times, sorted(times))


if __name__ == "__main__":
    unittest.main()
