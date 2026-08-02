import json
import tempfile
import unittest
from pathlib import Path

from select_thor_magni_complete_geometry_candidates import run


class SelectThorMagniCompleteGeometryCandidatesTest(unittest.TestCase):
    def _candidate(self, candidate_id: str, *, complete: bool = True) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "dataset_id": "THOR-MAGNI",
            "source_session_id": f"session-{candidate_id}",
            "start_timestamp_ns": 1,
            "end_timestamp_ns": 4_000_000_001,
            "frame_ids": [f"frame-{candidate_id}"],
            "model_output_visible_to_selector": False,
            "native_geometry_used_for_selection": False,
            "source_metadata": {
                "camera_centroid_complete": complete,
                "scene_frame_complete": complete,
                "scene_frame_missing_row_count": 0 if complete else 1,
                "qtm_window_duplicate_row_count": 0,
            },
        }

    def test_excludes_existing_review_identity_and_keeps_complete_geometry_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text(
                "\n".join(json.dumps(row) for row in (
                    self._candidate("keep"),
                    self._candidate("already"),
                    self._candidate("incomplete", complete=False),
                )) + "\n",
                encoding="utf-8",
            )
            bundles = root / "reviews" / "input_bundles" / "batch-a"
            bundles.mkdir(parents=True)
            (bundles / "bundle_manifest.json").write_text(
                json.dumps({"candidate_ids": ["already"]}),
                encoding="utf-8",
            )
            output = root / "selected.jsonl"
            result = run(type("Args", (), {
                "run_id": "selection-test",
                "output_root": str(root),
                "candidate_artifact": str(candidate_path),
                "review_bundles_root": str(root / "reviews" / "input_bundles"),
                "output_path": str(output),
                "max_count": None,
                "min_adjacency_gap_ns": 1_000_000_000,
                "ignore_batch_id": [],
            })())
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["candidate_id"] for row in selected], ["keep"])
            self.assertEqual(result["selected_candidate_count"], 1)
            self.assertEqual(result["excluded_already_reviewed_count"], 1)

    def test_fails_when_no_candidates_remain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text(json.dumps(self._candidate("only", complete=False)) + "\n", encoding="utf-8")
            bundles = root / "reviews" / "input_bundles"
            bundles.mkdir(parents=True)
            with self.assertRaises(Exception):
                run(type("Args", (), {
                    "run_id": "selection-test-empty",
                    "output_root": str(root),
                    "candidate_artifact": str(candidate_path),
                    "review_bundles_root": str(bundles),
                    "output_path": str(root / "selected.jsonl"),
                    "max_count": None,
                    "min_adjacency_gap_ns": 1_000_000_000,
                    "ignore_batch_id": [],
                })())

    def test_excludes_reviewed_same_session_adjacency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            keep = self._candidate("keep")
            keep["source_session_id"] = "shared"
            keep["start_timestamp_ns"] = 5_000_000_000
            keep["end_timestamp_ns"] = 9_000_000_001
            old = self._candidate("old")
            old["source_session_id"] = "shared"
            old["start_timestamp_ns"] = 1
            old["end_timestamp_ns"] = 4_000_000_001
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text(
                "\n".join(json.dumps(row) for row in (keep, old)) + "\n",
                encoding="utf-8",
            )
            bundles = root / "reviews" / "input_bundles" / "old-batch"
            bundles.mkdir(parents=True)
            (bundles / "bundle_manifest.json").write_text(
                json.dumps({"candidate_ids": ["old"]}),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                run(type("Args", (), {
                    "run_id": "selection-adjacency-empty",
                    "output_root": str(root),
                    "candidate_artifact": str(candidate_path),
                    "review_bundles_root": str(root / "reviews" / "input_bundles"),
                    "output_path": str(root / "selected.jsonl"),
                    "max_count": None,
                    "min_adjacency_gap_ns": 1_000_000_000,
                    "ignore_batch_id": [],
                })())

    def test_excludes_within_batch_adjacency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._candidate("first")
            second = self._candidate("second")
            first["source_session_id"] = second["source_session_id"] = "shared"
            first["start_timestamp_ns"], first["end_timestamp_ns"] = 1, 4_000_000_001
            second["start_timestamp_ns"], second["end_timestamp_ns"] = 4_010_000_001, 8_010_000_002
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text(
                "\n".join(json.dumps(row) for row in (first, second)) + "\n",
                encoding="utf-8",
            )
            bundles = root / "reviews" / "input_bundles"
            bundles.mkdir(parents=True)
            result = run(type("Args", (), {
                "run_id": "selection-within-batch",
                "output_root": str(root),
                "candidate_artifact": str(candidate_path),
                "review_bundles_root": str(bundles),
                "output_path": str(root / "selected.jsonl"),
                "max_count": None,
                "min_adjacency_gap_ns": 1_000_000_000,
                "ignore_batch_id": [],
            })())
            selected = [json.loads(line) for line in (root / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["candidate_id"] for row in selected], ["first"])
            self.assertEqual(result["excluded_within_batch_overlap_or_adjacency_count"], 1)


if __name__ == "__main__":
    unittest.main()
