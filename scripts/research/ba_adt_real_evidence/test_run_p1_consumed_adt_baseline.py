from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import run_p1_consumed_adt_baseline as adapter


def public_fixture() -> dict:
    return {
        "schema_version": adapter.PUBLIC_SCHEMA,
        "protocol_id": adapter.baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY",
        "tracker": adapter.TRACKER_NAME,
        "sources": [{
            "source_sequence_id": "source",
            "rgb_video_path": "video.mp4",
            "rgb_video_sha256": "0" * 64,
            "episodes": [{
                "episode_id": "p1-r0-consumed-001",
                "handoff": {
                    "status": "REFERENT_ESTABLISHED",
                    "goal_id": "consumed-goal-001",
                    "referent_id": "consumed-referent-001",
                    "grounding_provenance": {
                        "p0_decision_id": "oracle-init-001",
                        "source_frame_index": 0,
                        "authority": "P0_ESTABLISHED_REFERENT",
                    },
                },
                "frames": [
                    {"frame_index": 0, "timestamp_ms": 0, "video_frame_index": 0},
                    {"frame_index": 1, "timestamp_ms": 33, "video_frame_index": 1},
                ],
                "initial_target_bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
            }],
        }],
    }


class PublicFirewallTest(unittest.TestCase):
    def test_exact_public_surface_passes(self):
        value = public_fixture()
        self.assertIs(adapter.validate_public_input(value), value)

    def test_future_truth_field_is_rejected(self):
        value = public_fixture()
        value["sources"][0]["episodes"][0]["frames"][1]["target_bbox_xyxy"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "keys drift"):
            adapter.validate_public_input(value)

    def test_source_uid_in_public_episode_id_is_rejected(self):
        value = public_fixture()
        value["sources"][0]["episodes"][0]["episode_id"] = "p1-d0-source-4820696891314354"
        with self.assertRaisesRegex(ValueError, "opaque ordinal"):
            adapter.validate_public_input(value)

    def test_no_referent_is_not_silently_bound(self):
        value = public_fixture()
        value["sources"][0]["episodes"][0]["handoff"]["status"] = "NO_REFERENT"
        with self.assertRaisesRegex(ValueError, "established"):
            adapter.validate_public_input(value)


class TerminalRuleTest(unittest.TestCase):
    def aggregate(self, coverage: float, wrong: int = 0, switches: int = 0, false_reacq: int = 0):
        return {
            "correct_identity_coverage": {"value": coverage},
            "wrong_instance_asserted_frames": wrong,
            "identity_switches": switches,
            "false_reacquisitions": false_reacq,
        }

    def test_observation_floor_precedes_sparse_safety_events(self):
        self.assertEqual(
            adapter._terminal(self.aggregate(0.09, wrong=1)),
            "P1_R0_BASELINE_BELOW_PERSISTENCE_RESEARCH_FLOOR_OBSERVATION_FIRST",
        )

    def test_safety_headroom_requires_researchable_coverage(self):
        self.assertEqual(
            adapter._terminal(self.aggregate(0.10, false_reacq=1)),
            "REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT",
        )

    def test_clean_identity_surface_closes_headroom(self):
        self.assertEqual(
            adapter._terminal(self.aggregate(0.65)),
            "NO_MATERIAL_PERSISTENCE_HEADROOM_ON_CURRENT_DEVELOPMENT_COHORT",
        )


class TrackerMechanicsTest(unittest.TestCase):
    def test_private_bbox_matching_distinguishes_instance_from_background(self):
        source = SimpleNamespace(boxes={
            "target": {100: {"visibility_ratio": 1.0, "x_min": 10.0, "y_min": 10.0, "x_max": 30.0, "y_max": 30.0}},
            "other": {100: {"visibility_ratio": 1.0, "x_min": 50.0, "y_min": 50.0, "x_max": 70.0, "y_max": 70.0}},
        })
        self.assertEqual(adapter._match_bbox(source, 100, [11, 11, 29, 29], "ep")[0], "adt:target")
        self.assertEqual(adapter._match_bbox(source, 100, [51, 51, 69, 69], "ep")[0], "adt:other")
        self.assertEqual(adapter._match_bbox(source, 100, [75, 75, 90, 90], "ep")[0], "background:ep")

    def test_rgb_tracker_uses_one_oracle_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
            self.assertTrue(writer.isOpened())
            for frame_index in range(8):
                image = np.zeros((96, 96, 3), dtype=np.uint8)
                x = 20 + frame_index
                cv2.rectangle(image, (x, 30), (x + 24, 54), (255, 255, 255), -1)
                cv2.line(image, (x, 30), (x + 24, 54), (0, 0, 0), 2)
                writer.write(image)
            writer.release()
            episode = copy.deepcopy(public_fixture()["sources"][0]["episodes"][0])
            episode["frames"] = [
                {"frame_index": index, "timestamp_ms": index * 100, "video_frame_index": index}
                for index in range(8)
            ]
            episode["initial_target_bbox_xyxy"] = [20.0, 30.0, 45.0, 55.0]
            result = adapter.track_episode(episode, video)
            self.assertEqual(result["oracle_initializations"], 1)
            self.assertEqual(result["post_initialization_gt_reads"], 0)
            self.assertEqual(len(result["p1_output"]["frames"]), 8)
            self.assertEqual(result["candidate_bboxes"][0]["source"], "oracle_initialization")
            self.assertTrue(any(row["source"] == "sparse_lk_flow" for row in result["candidate_bboxes"][1:]))


if __name__ == "__main__":
    unittest.main()
