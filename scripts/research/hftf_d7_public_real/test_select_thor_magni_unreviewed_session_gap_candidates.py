import json
import tempfile
import unittest
from pathlib import Path

from select_thor_magni_unreviewed_session_gap_candidates import run


class SelectThorMagniReviewCandidatesTest(unittest.TestCase):
    def _candidate(
        self,
        candidate_id: str,
        session: str,
        start: int,
        end: int,
        *,
        dataset: str = "THOR-MAGNI",
        native_geometry: bool = True,
        timestamp_semantics: str = "SOURCE_SYNCHRONIZED_QTM_TIME",
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "dataset_id": dataset,
            "source_session_id": session,
            "source_id": f"source-{session}",
            "start_timestamp_ns": start,
            "end_timestamp_ns": end,
            "native_geometry_available": native_geometry,
            "timestamp_semantics": timestamp_semantics,
            "model_output_visible_to_selector": False,
        }

    def _args(self, root: Path, candidate_path: Path, output: Path) -> object:
        return type("Args", (), {
            "run_id": "thor-selection-test",
            "output_root": str(root),
            "candidate_artifact": str(candidate_path),
            "review_bundles_root": str(root / "reviews" / "input_bundles"),
            "output_path": str(output),
            "max_count": None,
            "min_adjacency_gap_ns": 1_000_000_000,
            "ignore_batch_id": [],
        })()

    def test_excludes_identity_temporal_neighbors_and_source_contract_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                self._candidate("old", "shared", 1, 4_000_000_001),
                self._candidate("near", "shared", 5_000_000_001, 9_000_000_001),
                self._candidate("keep", "shared", 20_000_000_001, 24_000_000_001),
                self._candidate("same-session-later", "shared", 30_000_000_001, 34_000_000_001),
                self._candidate("other", "other", 1, 4_000_000_001),
                self._candidate("no-native", "no-native", 1, 4_000_000_001, native_geometry=False),
                self._candidate("wrong-time", "wrong-time", 1, 4_000_000_001, timestamp_semantics="VIDEO_TIME"),
            ]
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            bundle = root / "reviews" / "input_bundles" / "old-batch"
            bundle.mkdir(parents=True)
            (bundle / "bundle_manifest.json").write_text(
                json.dumps({"dataset_id": "THOR-MAGNI", "candidate_ids": ["old"]}),
                encoding="utf-8",
            )
            output = root / "selected.jsonl"
            result = run(self._args(root, candidate_path, output))
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["candidate_id"] for row in selected], ["other", "keep"])
            self.assertEqual(result["selected_candidate_count"], 2)
            self.assertEqual(result["excluded_already_reviewed_count"], 1)
            self.assertEqual(result["excluded_reviewed_overlap_or_adjacency_count"], 1)
            self.assertFalse(result["event_truth_inferred"])
            self.assertFalse(result["independence_assigned"])

    def test_rejects_empty_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidate_index.jsonl"
            candidate_path.write_text(
                json.dumps(self._candidate("only", "s", 1, 4)) + "\n",
                encoding="utf-8",
            )
            bundle = root / "reviews" / "input_bundles" / "old"
            bundle.mkdir(parents=True)
            (bundle / "bundle_manifest.json").write_text(
                json.dumps({"dataset_id": "THOR-MAGNI", "candidate_ids": ["only"]}),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                run(self._args(root, candidate_path, root / "selected.jsonl"))


if __name__ == "__main__":
    unittest.main()
