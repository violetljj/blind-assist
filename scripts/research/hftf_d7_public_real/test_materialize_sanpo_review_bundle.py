from __future__ import annotations

import unittest
import base64
import csv
import hashlib
import tempfile
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from materialize_sanpo_review_bundle import (
    _bind_pose_rows,
    _download_object,
    _materialize_object,
    _provider_cache_destination,
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

    def test_selection_excludes_completed_candidate_ids(self) -> None:
        rows = [self._row("session-a", 0), self._row("session-a", 60)]
        selected = _select_candidates(
            rows,
            count=1,
            session_count=1,
            excluded_candidate_ids={"candidate-session-a-0"},
        )
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-a-60"])

    def test_selection_can_require_source_segmentation(self) -> None:
        rows = [
            {**self._row("session-a", 0), "source_metadata": {"raw_source_session_id": "session-a", "segmentation_complete": False}},
            {**self._row("session-b", 0), "source_metadata": {"raw_source_session_id": "session-b", "segmentation_complete": True}},
        ]
        selected = _select_candidates(rows, count=1, session_count=1, require_segmentation=True)
        self.assertEqual([row["candidate_id"] for row in selected], ["candidate-session-b-0"])

    def test_selection_can_limit_to_one_candidate_per_source_session(self) -> None:
        rows = [
            self._row("session-a", 60),
            self._row("session-a", 0),
            self._row("session-b", 0),
        ]
        selected = _select_candidates(
            rows,
            count=2,
            session_count=2,
            one_per_session=True,
        )
        self.assertEqual(
            [(row["source_id"], row["start_frame_index"]) for row in selected],
            [("session-a", 0), ("session-b", 0)],
        )

    def test_selection_can_bind_exact_candidate_ids(self) -> None:
        rows = [
            self._row("session-a", 0),
            self._row("session-b", 0),
        ]
        selected = _select_candidates(
            rows,
            count=1,
            session_count=1,
            candidate_ids=["candidate-session-b-0"],
        )
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

    def test_download_retries_transient_url_error_and_replaces_atomically(self) -> None:
        payload = b"verified sanpo object"
        item = {
            "name": "sanpo/session/camera_poses.csv",
            "size": len(payload),
            "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
            "generation": "1",
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, _size: int) -> bytes:
                if self._read:
                    return b""
                self._read = True
                return payload

            _read = False

        calls = 0

        def fake_urlopen(_request, timeout: int):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.URLError("transient EOF")
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "camera_poses.csv"
            with patch("materialize_sanpo_review_bundle.urllib.request.urlopen", side_effect=fake_urlopen), patch(
                "materialize_sanpo_review_bundle.time.sleep"
            ):
                result = _download_object(item, destination)

            self.assertEqual(calls, 2)
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name(destination.name + ".part").exists())
            self.assertTrue(result["md5_verified"])

    def test_download_uses_curl_fallback_after_urllib_retries(self) -> None:
        payload = b"curl fallback object"
        item = {
            "name": "sanpo/session/depth/000000.float16.gz",
            "size": len(payload),
            "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
        }

        def fake_curl(args, **_kwargs):
            output = Path(args[args.index("--output") + 1])
            output.write_bytes(payload)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "000000.float16.gz"
            with patch(
                "materialize_sanpo_review_bundle.urllib.request.urlopen",
                side_effect=urllib.error.URLError("persistent EOF"),
            ), patch("materialize_sanpo_review_bundle.time.sleep"), patch(
                "materialize_sanpo_review_bundle.subprocess.run", side_effect=fake_curl
            ) as run:
                result = _download_object(item, destination)

            self.assertEqual(run.call_count, 1)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertTrue(result["md5_verified"])

    def test_cache_only_reuses_verified_provider_object_without_network(self) -> None:
        payload = b"cached sanpo object"
        item = {
            "name": "sanpo/session/depth/000001.float16.gz",
            "size": len(payload),
            "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
            "generation": "2",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "provider_cache"
            cached = _provider_cache_destination(cache_root, item)
            cached.parent.mkdir(parents=True)
            cached.write_bytes(payload)
            destination = root / "new_batch" / "provider_cache" / "depth" / cached.name
            with patch(
                "materialize_sanpo_review_bundle.urllib.request.urlopen",
                side_effect=AssertionError("cache-only materialization must not access the network"),
            ):
                result = _materialize_object(item, destination, cache_root, True)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertTrue(result["md5_verified"])


if __name__ == "__main__":
    unittest.main()
