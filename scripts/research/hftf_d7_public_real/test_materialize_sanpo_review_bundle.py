from __future__ import annotations

import unittest
from pathlib import Path

from materialize_sanpo_review_bundle import _provider_destination, _provider_kind, _select_candidates


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

    def test_provider_destination_is_stable_and_kind_scoped(self) -> None:
        item = {"name": "sanpo/session/video_frames/000123.png"}
        self.assertEqual(_provider_kind(str(item["name"])), "rgb")
        first = _provider_destination(Path(r"F:\\review"), item)
        second = _provider_destination(Path(r"F:\\review"), item)
        self.assertEqual(first, second)
        self.assertEqual(first.parent.name, "rgb")


if __name__ == "__main__":
    unittest.main()
