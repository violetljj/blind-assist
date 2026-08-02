from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from materialize_sanpo_review_bundle import (
    _bind_pose_rows,
    _provider_destination,
    _provider_kind,
    _select_candidates,
)


class MaterializeSanpoReviewBundleTest(unittest.TestCase):
    def _row(self, session: str, start: int) -> dict[str, object]:
        return {
            "candidate_id": f"candidate-{session}-{start}",
            "dataset_id": "SANPO-Real",
            "source_id": session,
            "start_frame_index": start,
            "source_metadata": {"raw_source_session_id": session},
        }

    def test_selection_is_session_stratified_and_deterministic(self) -> None:
        rows = [
            self._row("session-b", 30),
            self._row("session-a", 60),
            self._row("session-a", 0),
            self._row("session-c", 0),
        ]
        selected = _select_candidates(rows, count=3, session_count=2)
        self.assertEqual(
            [(row["source_id"], row["start_frame_index"]) for row in selected],
            [("session-a", 0), ("session-a", 60), ("session-b", 30)],
        )

    def test_non_sanpo_rows_are_excluded(self) -> None:
        rows = [self._row("session-a", 0), {"candidate_id": "other", "dataset_id": "EgoWalk"}]
        selected = _select_candidates(rows, count=1, session_count=1)
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-a-0"])

    def test_session_offset_skips_earlier_sessions(self) -> None:
        rows = [
            self._row("session-a", 0),
            self._row("session-b", 0),
            self._row("session-c", 0),
        ]
        selected = _select_candidates(rows, count=1, session_count=1, session_offset=1)
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-b-0"])

    def test_allowed_sessions_can_fail_closed_to_complete_media_only(self) -> None:
        rows = [self._row("session-a", 0), self._row("session-b", 0)]
        selected = _select_candidates(rows, count=1, session_count=1, allowed_sessions={"session-b"})
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-b-0"])

    def test_provider_destination_is_stable_and_kind_scoped(self) -> None:
        item = {"name": "sanpo/session/video_frames/000123.png"}
        self.assertEqual(_provider_kind(str(item["name"])), "rgb")
        first = _provider_destination(Path(r"F:\\review"), item)
        second = _provider_destination(Path(r"F:\\review"), item)
        self.assertEqual(first, second)
        self.assertEqual(first.parent.name, "rgb")

    def test_pose_binding_requires_complete_source_row_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "camera_poses.csv"
            fixed = root / "fixed_camera_poses.csv"
            fields = ["tracking_state", "pos_x", "pos_y", "pos_z", "q_x", "q_y", "q_z", "q_w"]
            rows = [
                ["TrackingState.READY", "0", "1", "2", "0", "0", "0", "1"],
                ["TrackingState.READY", "1", "2", "3", "0", "0", "0", "1"],
            ]
            for path in (pose, fixed):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
            status, bound, reason = _bind_pose_rows(
                pose_path=pose,
                fixed_pose_path=fixed,
                expected_rows=2,
                frame_indices=[0, 1],
            )
            self.assertEqual(status, "FRAME_INDEX_ROW_KEYED")
            self.assertEqual(reason, "complete_frame_count_equals_pose_data_rows")
            self.assertEqual(bound[1]["camera_pose"]["position"]["pos_x"], 1.0)

            status, bound, _ = _bind_pose_rows(
                pose_path=pose,
                fixed_pose_path=fixed,
                expected_rows=3,
                frame_indices=[0, 1],
            )
            self.assertEqual(status, "NOT_EVALUABLE")
            self.assertEqual(bound, {})


if __name__ == "__main__":
    unittest.main()
