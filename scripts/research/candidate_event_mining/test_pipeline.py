from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    build_candidate_report,
    finalize_candidate_pool,
    load_contract,
    make_review_bundle,
    normalize_frame,
    normalize_frames,
    read_json,
    read_jsonl,
    sha256_file,
    validate_project_index,
    write_json,
    write_jsonl,
)

CONTRACT_PATH = REPO_ROOT / "configs" / "candidate_event_mining_contract_v1.json"


def _frame(index: int, **signals: float) -> dict[str, object]:
    values = {"yolo.coverage": 1.0}
    values.update(signals)
    return {
        "schema": "blindassist_candidate_event_mining_frame_v1",
        "source_id": "source-a",
        "session_id": "session-a",
        "frame_index": index,
        "timestamp_ms": index * 100,
        "frame_ref": f"F:/ba-data/source-a/frame-{index:04d}.jpg",
        "signals": values,
    }


def _project_index() -> dict[str, object]:
    return {
        "schema": "blindassist_candidate_event_mining_project_index_v1",
        "index_version": "r0",
        "project_id": "test",
        "data_root": r"F:\ba-data",
        "project_root": r"F:\ba-data\blindassist-candidate-event-mining",
        "sources": [
            {
                "source_id": "source-a",
                "session_id": "session-a",
                "media_path": r"F:\ba-data\source-a\video.mp4",
                "source_url": "https://example.test/source-a",
                "retrieved_at_utc": "2026-08-02T00:00:00+00:00",
                "content_sha256": "a" * 64,
                "retrieval_status": "verified",
            }
        ],
    }


class CandidateMiningPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.contract_meta = load_contract(CONTRACT_PATH)

    def test_project_index_and_frame_contract_are_strict(self) -> None:
        self.assertEqual(validate_project_index(_project_index())["sources"][0]["source_id"], "source-a")
        normalized = normalize_frame(_frame(0, **{"motion.front_approach": 0.8}))
        self.assertEqual(normalized["timestamp_ms"], 0)
        with self.assertRaises(ContractError):
            normalize_frame(_frame(0, **{"motion.front_approach": 1.2}))

    def test_extracts_taxonomy_deduplicates_and_clusters(self) -> None:
        rows = [
            _frame(0, **{"motion.front_approach": 0.8, "segmentation.risk": 0.8, "yolo.coverage": 0.0}),
            _frame(1, **{"motion.front_approach": 0.85, "segmentation.risk": 0.8, "yolo.coverage": 0.0}),
            _frame(3, **{"motion.crossing": 0.8}),
            _frame(4, **{"motion.crossing": 0.75}),
            _frame(5, **{"motion.static_obstacle_approach": 0.8}),
            _frame(6, **{"motion.static_obstacle_approach": 0.75}),
            _frame(7, **{"geometry.step_drop": 0.8}),
            _frame(8, **{"geometry.step_drop": 0.75}),
            _frame(9, **{"geometry.parallel_curb": 0.8}),
            _frame(10, **{"geometry.parallel_curb": 0.75}),
            _frame(11, **{"object.doorframe": 0.8}),
            _frame(12, **{"object.doorframe": 0.75}),
            _frame(13, **{"context.normal_passage": 0.8}),
            _frame(14, **{"context.normal_passage": 0.75}),
            _frame(15, **{"motion.head_turn": 0.8}),
            _frame(16, **{"motion.jitter": 0.75}),
            _frame(17, **{"motion.dynamic_crowd": 0.8}),
            _frame(18, **{"motion.dynamic_crowd": 0.75}),
            _frame(19, **{"hftf.future_field_change": 0.8}),
            _frame(20, **{"hftf.future_field_change": 0.75}),
            _frame(21, **{"segmentation.high_frequency_alert": 0.8}),
            _frame(22, **{"segmentation.high_frequency_alert": 0.75}),
            _frame(24, **{"motion.front_approach": 0.8}),
            _frame(25, **{"motion.front_approach": 0.8}),
            _frame(36, **{"motion.front_approach": 0.8}),
            _frame(37, **{"motion.front_approach": 0.8}),
        ]
        normalized = normalize_frames(rows)
        report = build_candidate_report(
            normalized,
            self.contract_meta,
            {"path": "project_index.json", "sha256": "b" * 64},
            {"path": "trace.jsonl", "sha256": "c" * 64},
            "test-run",
            self.contract,
        )
        types = {candidate["trigger_type"] for candidate in report["candidates"]}
        self.assertEqual(
            types,
            {
                "front_obstacle_approach",
                "crossing",
                "static_obstacle_approach",
                "step_or_drop",
                "parallel_curb",
                "doorframe_table_corner_tree_branch",
                "normal_passage_negative",
                "head_turn_or_jitter_negative",
                "dynamic_crowd",
                "yolo_miss_segmentation_or_depth_response",
                "segmentation_high_frequency_alert",
                "hftf_future_field_change",
            },
        )
        front = [row for row in report["candidates"] if row["trigger_type"] == "front_obstacle_approach"]
        self.assertEqual(len(front), 2)
        self.assertEqual(front[-1]["deduplicated_count"], 2)
        self.assertTrue(all(row["cluster_id"] for row in report["candidates"]))
        self.assertFalse(any(row["candidate_status"] != "unreviewed_trigger" for row in report["candidates"]))

    def test_review_bundle_hides_trigger_and_finalizes_luna_pool(self) -> None:
        rows = normalize_frames([
            _frame(0, **{"motion.front_approach": 0.8}),
            _frame(1, **{"motion.front_approach": 0.8}),
        ])
        report = build_candidate_report(
            rows,
            self.contract_meta,
            {"path": "project_index.json", "sha256": "b" * 64},
            {"path": "trace.jsonl", "sha256": "c" * 64},
            "review-run",
            self.contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "candidate_report.json"
            write_json(report_path, report)
            bundle_dir = root / "review-bundle"
            bundle = make_review_bundle(report, report_path, self.contract_meta, self.contract, bundle_dir)
            review_inputs = read_jsonl(bundle_dir / "review_inputs.jsonl")
            self.assertEqual(len(review_inputs), len(report["candidates"]))
            self.assertTrue(all(item["candidate_type_hidden"] for item in review_inputs))
            self.assertTrue(all("trigger_type" not in item for item in review_inputs))

            reviews_path = root / "luna_reviews.jsonl"
            reviews = [
                {
                    "schema": "blindassist_candidate_event_mining_luna_review_v1",
                    "candidate_id": candidate["candidate_id"],
                    "reviewer_id": "luna-reader-1",
                    "reviewer_type": "ai_model",
                    "reviewer_role": "luna_reader",
                    "provider": "Luna",
                    "model_version": "test",
                    "review_run_id": "luna-run-1",
                    "workflow_id": "candidate_event_mining_luna_review_v1",
                    "prompt_sha256": bundle["review_prompt_sha256"],
                    "input_sha256": bundle["review_inputs_sha256"],
                    "independent_review": True,
                    "isolated_context": True,
                    "other_review_outputs_viewed": False,
                    "candidate_output_visible": False,
                    "confidence": 0.9,
                    "abstained": False,
                    "observed_types": [candidate["trigger_type"]],
                    "disposition": "keep",
                }
                for candidate in report["candidates"]
            ]
            write_jsonl(reviews_path, reviews)
            bundle_path = bundle_dir / "review_bundle_manifest.json"
            pool = finalize_candidate_pool(
                report,
                report_path,
                bundle,
                bundle_path,
                reviews,
                reviews_path,
                self.contract,
            )
            self.assertEqual(pool["summary"], {"candidate_count": 1, "pool_count": 1, "quarantine_count": 0})
            self.assertEqual(pool["pool"][0]["candidate_status"], "luna_reviewed_keep")
            self.assertEqual(pool["authority"]["event_truth"], False)

    def test_missing_review_fails_closed(self) -> None:
        rows = normalize_frames([
            _frame(0, **{"motion.front_approach": 0.8}),
            _frame(1, **{"motion.front_approach": 0.8}),
        ])
        report = build_candidate_report(
            rows,
            self.contract_meta,
            {"path": "project_index.json", "sha256": "b" * 64},
            {"path": "trace.jsonl", "sha256": "c" * 64},
            "missing-review-run",
            self.contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "candidate_report.json"
            write_json(report_path, report)
            bundle_dir = root / "review-bundle"
            bundle = make_review_bundle(report, report_path, self.contract_meta, self.contract, bundle_dir)
            reviews_path = root / "empty.jsonl"
            reviews_path.write_text("", encoding="utf-8")
            with self.assertRaises(ContractError):
                finalize_candidate_pool(
                    report,
                    report_path,
                    bundle,
                    bundle_dir / "review_bundle_manifest.json",
                    [],
                    reviews_path,
                    self.contract,
                )


if __name__ == "__main__":
    unittest.main()
