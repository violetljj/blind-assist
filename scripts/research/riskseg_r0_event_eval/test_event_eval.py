from __future__ import annotations

import unittest

from .validate_and_freeze_cohort import (
    validate_final_items,
    validate_review_coverage,
)


def candidate(candidate_id: str, session: str, start: int) -> dict:
    return {
        "event_candidate_id": candidate_id,
        "source_session_id": session,
        "sequence_id": candidate_id,
        "source_frame_start": start,
        "source_frame_end": start + 74,
        "frame_count": 50,
    }


def item(candidate_id: str, bucket: str) -> dict:
    positive = bucket.endswith("_positive")
    return {
        "parent_event_id": f"event-{candidate_id}",
        "event_candidate_id": candidate_id,
        "bucket": bucket,
        "truth_confidence": 0.9,
        "alertable_interval_frames": [5, 20] if positive else None,
        "passed_interval_frames": [30, 40] if positive else None,
        "truth_notes": "fixture",
    }


class CohortValidationTest(unittest.TestCase):
    def make_valid(self) -> tuple[list[dict], dict[str, dict]]:
        floors = {
            "blocking_obstacle_positive": 8,
            "boundary_level_change_positive": 8,
            "parallel_curb_negative": 7,
            "normal_walkable_negative": 7,
        }
        candidates: dict[str, dict] = {}
        items: list[dict] = []
        ordinal = 0
        for bucket, count in floors.items():
            for bucket_index in range(count):
                ordinal += 1
                candidate_id = f"candidate-{ordinal:02d}"
                session = f"session-{bucket_index % 8:02d}"
                start = (ordinal // 8) * 100
                candidates[candidate_id] = candidate(candidate_id, session, start)
                items.append(item(candidate_id, bucket))
        return items, candidates

    def test_exact_floor_passes(self) -> None:
        items, candidates = self.make_valid()
        buckets, sessions = validate_final_items(items, candidates, set())
        self.assertEqual(30, sum(buckets.values()))
        self.assertEqual(8, len(sessions))

    def test_excluded_session_fails(self) -> None:
        items, candidates = self.make_valid()
        with self.assertRaisesRegex(ValueError, "excluded source session"):
            validate_final_items(items, candidates, {"session-00"})

    def test_missing_bucket_event_fails(self) -> None:
        items, candidates = self.make_valid()
        items[0]["bucket"] = "normal_walkable_negative"
        items[0]["alertable_interval_frames"] = None
        items[0]["passed_interval_frames"] = None
        with self.assertRaisesRegex(ValueError, "needs 8"):
            validate_final_items(items, candidates, set())

    def test_overlapping_windows_fail(self) -> None:
        items, candidates = self.make_valid()
        first = items[0]["event_candidate_id"]
        second = items[4]["event_candidate_id"]
        candidates[second]["source_session_id"] = candidates[first]["source_session_id"]
        candidates[second]["source_frame_start"] = candidates[first]["source_frame_start"]
        candidates[second]["source_frame_end"] = candidates[first]["source_frame_end"]
        with self.assertRaisesRegex(ValueError, "event windows overlap"):
            validate_final_items(items, candidates, set())

    def test_positive_without_passed_interval_fails(self) -> None:
        items, candidates = self.make_valid()
        items[0]["passed_interval_frames"] = None
        with self.assertRaisesRegex(ValueError, "invalid frame interval"):
            validate_final_items(items, candidates, set())

    def test_review_must_cover_every_candidate_once(self) -> None:
        candidates = {
            "candidate-01": candidate("candidate-01", "session-01", 0),
            "candidate-02": candidate("candidate-02", "session-02", 0),
        }
        review = {
            "items": [
                {
                    "event_candidate_id": "candidate-01",
                    "source_session_id": "session-01",
                    "bucket": "normal_walkable_negative",
                    "confidence": 0.9,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_review_coverage(review, candidates, "fixture-review")


if __name__ == "__main__":
    unittest.main()
