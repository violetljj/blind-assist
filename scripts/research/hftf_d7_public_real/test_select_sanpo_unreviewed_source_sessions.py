import json
import tempfile
import unittest
from pathlib import Path

from select_sanpo_unreviewed_source_sessions import run


class SelectSanpoUnreviewedSourceSessionsTest(unittest.TestCase):
    def _candidate(self, candidate_id: str, raw_session: str, start: int, *, complete: bool = True) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "dataset_id": "SANPO-Real",
            "source_id": raw_session,
            "source_session_id": f"d7sess-{raw_session}",
            "start_frame_index": start,
            "end_frame_index": start + 59,
            "frame_count": 60,
            "source_metadata": {
                "raw_source_session_id": raw_session,
                "segmentation_complete": complete,
            },
        }

    def _args(self, root: Path, candidate_path: Path, inventory: Path, output: Path) -> object:
        return type("Args", (), {
            "run_id": "sanpo-selection-test",
            "output_root": str(root),
            "candidate_artifact": str(candidate_path),
            "inventory": str(inventory),
            "output_path": str(output),
            "max_count": None,
            "require_pose": True,
        })()

    def test_excludes_reviewed_raw_session_and_takes_one_per_unseen_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                self._candidate("old", "raw-old", 0),
                self._candidate("same-session", "raw-old", 60),
                self._candidate("keep-a", "raw-a", 0),
                self._candidate("later-a", "raw-a", 60),
                self._candidate("keep-b", "raw-b", 0),
                self._candidate("incomplete", "raw-c", 0, complete=False),
            ]
            candidate_path = root / "candidates.jsonl"
            candidate_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps({
                "records": [
                    {"source_session_id": raw, "media": {"complete_frame_count": 120}, "auxiliary": {"pose": [{"name": raw + ".csv"}]}}
                    for raw in ("raw-old", "raw-a", "raw-b", "raw-c")
                ]
            }), encoding="utf-8")
            bundle = root / "reviews" / "input_bundles" / "old"
            bundle.mkdir(parents=True)
            (bundle / "bundle_manifest.json").write_text(json.dumps({"dataset_id": "SANPO-Real", "candidate_ids": ["old"]}), encoding="utf-8")
            output = root / "selected.jsonl"
            result = run(self._args(root, candidate_path, inventory_path, output))
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["candidate_id"] for row in selected], ["keep-a", "keep-b"])
            self.assertEqual(result["selected_candidate_count"], 2)
            self.assertEqual(result["excluded_existing_review_session_count"], 1)

    def test_fails_when_all_candidates_are_in_reviewed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidates.jsonl"
            candidate_path.write_text(json.dumps(self._candidate("old", "raw-old", 0)) + "\n", encoding="utf-8")
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps({"records": [{"source_session_id": "raw-old", "media": {"complete_frame_count": 60}, "auxiliary": {"pose": [{"name": "pose.csv"}]}}]}), encoding="utf-8")
            bundle = root / "reviews" / "input_bundles" / "old"
            bundle.mkdir(parents=True)
            (bundle / "bundle_manifest.json").write_text(json.dumps({"dataset_id": "SANPO-Real", "candidate_ids": ["old"]}), encoding="utf-8")
            with self.assertRaises(Exception):
                run(self._args(root, candidate_path, inventory_path, root / "selected.jsonl"))


if __name__ == "__main__":
    unittest.main()
