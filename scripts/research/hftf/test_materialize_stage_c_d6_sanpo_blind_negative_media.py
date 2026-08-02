import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d6_sanpo_blind_negative_media import (
    RGB_ROLES,
    build_materialization,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SanpoBlindNegativeMaterializationTest(unittest.TestCase):
    def test_selects_unanimous_negatives_and_deduplicates_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_rows = []
            reviews = {role: [] for role in RGB_ROLES}
            staging = root / "staging"
            for candidate_id, start, decisions in (
                ("negative-a", 0, ("REJECT",) * 3),
                ("negative-b", 2, ("REJECT",) * 3),
                ("mixed", 6, ("REJECT", "NOT_EVALUABLE", "REJECT")),
            ):
                end = start + 5
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "start_frame_index": start,
                        "end_frame_index": end,
                        "frame_count": 6,
                        "timestamp_semantics": "DERIVED_RELATIVE_NOMINAL",
                        "source_metadata": {
                            "raw_source_session_id": "raw-session",
                            "camera": "chest",
                            "view": "left",
                        },
                    }
                )
                temporal = []
                for frame_index in range(start, end + 1):
                    rgb_path = root / "rgb" / f"{frame_index:06d}.png"
                    rgb_path.parent.mkdir(parents=True, exist_ok=True)
                    if not rgb_path.exists():
                        rgb_path.write_bytes(f"frame-{frame_index}".encode())
                    temporal.append(
                        {
                            "frame_index": frame_index,
                            "nominal_time_ns": frame_index * 10,
                            "rgb_path": str(rgb_path),
                            "rgb_sha256": hashlib.sha256(
                                rgb_path.read_bytes()
                            ).hexdigest(),
                        }
                    )
                write_jsonl(
                    staging / candidate_id / "temporal_manifest.jsonl",
                    temporal,
                )
                for role, decision in zip(RGB_ROLES, decisions):
                    reviews[role].append(
                        {
                            "candidate_id": candidate_id,
                            "dataset_id": "SANPO-Real",
                            "record_kind": "COMPLETED_REVIEW",
                            "review_completed": True,
                            "review_role": role,
                            "model_output_visible": False,
                            "decision": decision,
                            "event_bucket": "NEGATIVE",
                        }
                    )

            candidate_index = root / "candidate_index.jsonl"
            write_jsonl(candidate_index, candidate_rows)
            review_paths = []
            for role, rows in reviews.items():
                path = root / f"{role}.jsonl"
                write_jsonl(path, rows)
                review_paths.append(path)

            media, intervals, windows, report = build_materialization(
                candidate_index,
                review_paths,
                staging,
            )

            self.assertEqual(len(intervals), 2)
            self.assertEqual(len(media), 8)
            self.assertEqual(len(windows), 4)
            self.assertEqual(report["overlap_observation_count"], 4)
            overlap = next(
                row for row in media if row["frame_index"] == 2
            )
            self.assertEqual(
                overlap["blind_review_candidate_ids"],
                ["negative-a", "negative-b"],
            )
            self.assertTrue(
                all(row["covering_candidate_ids"] for row in windows)
            )


if __name__ == "__main__":
    unittest.main()
