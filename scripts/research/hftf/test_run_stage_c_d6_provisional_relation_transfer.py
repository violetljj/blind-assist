import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_provisional_relation_transfer import (
    active_at_threshold,
    collect_reviewed_negative_sources,
    collect_training_episodes,
)


class ProvisionalRelationTransferTest(unittest.TestCase):
    def test_threshold_replays_causal_confirmation(self):
        event = {
            "parent_event_id": "event-1",
            "frames": [
                {"timestamp_ms": timestamp}
                for timestamp in (0, 100, 200, 300, 400)
            ],
        }
        active = active_at_threshold(
            event,
            [0.6, 0.6, 0.7, 0.7, 0.4],
            0.5,
        )
        self.assertEqual(
            [False, False, True, True, False],
            active,
        )

    def _write_source(
        self,
        root,
        name,
        source_id,
        session_id,
        image_exists=True,
    ):
        folder = root / name
        folder.mkdir()
        image_root = folder / "images"
        image_root.mkdir()
        if image_exists:
            (image_root / "frame.png").write_bytes(b"fixture")
        manifest = {
            "source_id": source_id,
            "source": {"session_id": session_id},
            "frames": [
                {
                    "file_name": "frame.png",
                    "sha256": "a" * 64,
                }
            ],
            "promotion": {"image_root": str(image_root)},
        }
        (folder / "source.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        labels = {
            "source": {
                "source_id": source_id,
                "source_manifest_path": "source.json",
            },
            "training_execution_authorized": True,
            "episodes": [
                {
                    "episode_id": f"{name}-episode",
                    "silver_should_alert": "candidate_alert",
                    "confidence": 0.8,
                    "evidence_frame_sha256": ["a" * 64],
                    "risk_profile": {},
                }
            ],
        }
        (folder / "silver_labels_v2.json").write_text(
            json.dumps(labels),
            encoding="utf-8",
        )

    def test_collection_excludes_evaluation_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_source(
                root,
                "kept",
                "source-kept",
                "session-kept",
            )
            self._write_source(
                root,
                "overlap",
                "source-overlap",
                "session-eval",
            )
            self._write_source(
                root,
                "missing",
                "source-missing",
                "session-missing",
                image_exists=False,
            )
            episodes, exclusions = collect_training_episodes(
                root,
                {
                    "events": [
                        {"source_session_id": "session-eval"}
                    ]
                },
            )
        self.assertEqual(
            ["kept-episode"],
            [row["episode_id"] for row in episodes],
        )
        reasons = {
            row["source_id"]: row["reason"] for row in exclusions
        }
        self.assertEqual(
            "overlaps_evaluation_session",
            reasons["source-overlap"],
        )
        self.assertEqual(
            "missing_evidence_frames",
            reasons["source-missing"],
        )

    def test_normal_passage_collection_deduplicates_by_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"fixture")
            review = {
                "candidate_id": "candidate-1",
                "disposition": "reject",
                "observed_types": ["normal_passage_negative"],
            }
            (root / "luna_reviews.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )
            frame = {
                "frame_ref": str(image),
                "frame_sha256": "b" * 64,
                "timestamp_ms": 100,
            }
            report = {
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "source_id": "source-1",
                        "frame_refs": [frame, frame],
                    }
                ]
            }
            (root / "review_queue_report.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )
            episodes = collect_reviewed_negative_sources(root)
        self.assertEqual(1, len(episodes))
        self.assertEqual(1, len(episodes[0]["frames"]))
        self.assertEqual(["candidate-1"], episodes[0]["candidate_ids"])

    def test_quarantined_parallel_curb_requires_canary_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"fixture")
            review = {
                "candidate_id": "candidate-curb",
                "disposition": "quarantine",
                "observed_types": ["parallel_curb"],
                "abstained": True,
            }
            (root / "luna_reviews.jsonl").write_text(
                json.dumps(review) + "\n",
                encoding="utf-8",
            )
            report = {
                "candidates": [
                    {
                        "candidate_id": "candidate-curb",
                        "source_id": "source-curb",
                        "frame_refs": [
                            {
                                "frame_ref": str(image),
                                "frame_sha256": "c" * 64,
                                "timestamp_ms": 100,
                            }
                        ],
                    }
                ]
            }
            (root / "review_queue_report.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )
            default = collect_reviewed_negative_sources(root)
            canary = collect_reviewed_negative_sources(
                root,
                include_quarantined_parallel_curb=True,
            )
        self.assertEqual([], default)
        self.assertEqual(1, len(canary))
        self.assertEqual(
            ["parallel_curb"],
            canary[0]["risk_profile"]["observed_types"],
        )


if __name__ == "__main__":
    unittest.main()
