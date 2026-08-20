from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mine_goal_episodes import Thresholds, mine
from run_rgb_observer import TargetMemory, appearance_embedding, candidate_confirmed, choose_target, deduplicate_candidates, flow_bbox, initial_anchor_candidate, iou, two_by_two_tiles
from evaluate_rgb_observations import metrics
from build_offline_demo import copilot_state
from account_redetection_failures import account_opportunities
from run_visual_upper_bound_r5 import select_trusted_exemplar
from audit_reappearance_windows import diagnostic_class, oracle_status


def csv_bytes(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class EpisodeMinerTest(unittest.TestCase):
    def test_instance_memory_prefers_anchor_appearance(self):
        import numpy as np

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[10:50, 10:70] = (0, 100, 255)
        image[10:50, 110:170] = (255, 0, 0)
        memory = TargetMemory()
        self.assertTrue(memory.remember(image, [10, 10, 70, 50], 0, force=True))
        matching = memory.score(image, [10, 10, 70, 50], 10)
        distractor = memory.score(image, [110, 10, 170, 50], 10)
        self.assertGreater(matching["appearance"], distractor["appearance"])
        self.assertGreater(matching["score"], distractor["score"])

    def test_redetection_requires_two_compatible_hits(self):
        import numpy as np

        embedding = np.ones(4, dtype=np.float32) / 2.0
        first = {"bbox_xyxy": [10, 10, 30, 30], "embedding": embedding}
        second = {"bbox_xyxy": [11, 10, 31, 30], "embedding": embedding}
        self.assertFalse(candidate_confirmed([], first, 2))
        self.assertTrue(candidate_confirmed([first], second, 2))

    def test_only_empty_memory_accepts_unconfirmed_initial_anchor(self):
        candidates = [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.7}]
        memory = TargetMemory()
        self.assertEqual(initial_anchor_candidate(candidates, memory), candidates[0])
        memory.templates.append(object())
        self.assertIsNone(initial_anchor_candidate(candidates, memory))

    def test_sparse_flow_propagates_bbox_translation(self):
        import cv2
        import numpy as np

        first = np.zeros((80, 80), dtype=np.uint8)
        for point in ((22, 22), (30, 22), (22, 30), (30, 30)):
            cv2.circle(first, point, 2, 255, -1)
        matrix = np.float32([[1, 0, 4], [0, 1, 3]])
        second = cv2.warpAffine(first, matrix, (80, 80))
        points = np.float32([[22, 22], [30, 22], [22, 30], [30, 30]]).reshape(-1, 1, 2)
        bbox, moved = flow_bbox(first, second, points, [18, 18, 34, 34], 80, 80)
        self.assertIsNotNone(moved)
        self.assertAlmostEqual(bbox[0], 22, delta=0.6)
        self.assertAlmostEqual(bbox[1], 21, delta=0.6)

    def test_offline_copilot_state_keeps_short_dropout_uncertain(self):
        self.assertEqual(copilot_state(False, False, 99, 99, None), "SEARCHING")
        self.assertEqual(copilot_state(False, True, 8, 99, None), "UNCERTAIN")
        self.assertEqual(copilot_state(False, True, 31, 99, None), "LOST")
        self.assertEqual(copilot_state(True, True, 0, 2, None), "REACQUIRED")
        self.assertEqual(copilot_state(True, True, 0, 20, 0.01), "APPROACHING")

    def test_evaluator_reports_reacquisition_and_quality_separation(self):
        rows = []
        for index in range(50):
            gt_visible = index < 15 or index >= 25
            localized = gt_visible and index not in {25, 26}
            rows.append({
                "gt_visible": gt_visible,
                "predicted_visible": localized,
                "localized": localized,
                "iou": 0.8 if localized else 0.0,
                "bearing_error_normalized": 0.02 if localized else None,
                "predicted_scale": 0.01 * index if localized else None,
                "truth_scale": 0.01 * index if localized else None,
                "observation_quality": 0.9 if localized else 0.1,
            })
        result = metrics(rows)
        self.assertEqual(result["eligible_reacquisition_count"], 1)
        self.assertEqual(result["reacquisition_delay_frames"], [2])
        self.assertEqual(result["reacquisition_success_within_90_frames"], 1.0)
        self.assertEqual(result["median_successful_reacquisition_delay_frames"], 2)
        self.assertGreater(result["observation_quality_mean_when_localized"], result["observation_quality_mean_when_missed"])

    def test_evaluator_separates_correct_wrong_and_unresolved_redetection(self):
        rows = [
            {"gt_visible": True, "predicted_visible": True, "localized": True, "iou": 0.8, "bearing_error_normalized": 0.01, "predicted_scale": 0.1, "truth_scale": 0.1, "observation_quality": 0.9, "instance_redetection": True, "wrong_instance_redetection": False, "unresolved_redetection": False},
            {"gt_visible": True, "predicted_visible": True, "localized": False, "iou": 0.0, "bearing_error_normalized": 0.5, "predicted_scale": 0.1, "truth_scale": 0.1, "observation_quality": 0.2, "instance_redetection": True, "wrong_instance_redetection": True, "unresolved_redetection": False},
            {"gt_visible": False, "predicted_visible": True, "localized": False, "iou": 0.0, "bearing_error_normalized": None, "predicted_scale": None, "truth_scale": None, "observation_quality": 0.1, "instance_redetection": True, "wrong_instance_redetection": False, "unresolved_redetection": True},
        ]
        result = metrics(rows)
        self.assertEqual(result["correct_instance_redetection_count"], 1)
        self.assertEqual(result["wrong_instance_redetection_count"], 1)
        self.assertEqual(result["unresolved_redetection_count"], 1)
        self.assertEqual(result["id_switch_count_at_instance_redetection"], 1)

    def test_failure_accounting_separates_candidate_rejection_and_confirmation(self):
        rows = []
        for index in range(30):
            gt_visible = index < 5 or 12 <= index < 16 or index >= 22
            rows.append({"gt_visible": gt_visible, "localized": False, "search_active": gt_visible and index >= 12, "correct_candidate_present": False, "correct_top_eligible": False, "best_candidate_iou": 0.0})
        rows[13].update({"correct_candidate_present": True, "best_candidate_iou": 0.6})
        rows[22].update({"correct_candidate_present": True, "correct_top_eligible": True, "best_candidate_iou": 0.7})
        accounting = account_opportunities(rows)
        self.assertEqual(accounting["eligible_reacquisition_count"], 2)
        self.assertEqual(accounting["failure_counts"], {"NO_CANDIDATE": 0, "CANDIDATE_REJECTED": 1, "CONFIRMATION_FAILED": 1})
        self.assertEqual(accounting["opportunities"][0]["first_valid_candidate_latency_frames"], 1)

    def test_observability_class_and_one_frame_oracle_failure_are_separate(self):
        one_frame = [{"visibility_ratio": 0.1, "size_visibility_usable": False, "same_prototype_distractors": 0, "oracle_candidate_traced": True, "oracle_candidate_eligible": False}]
        self.assertEqual(diagnostic_class(one_frame)[0], "UNOBSERVABLE_OR_HEAVILY_OCCLUDED")
        self.assertEqual(oracle_status(one_frame, False)[0], "FAIL_2_OF_3_INSUFFICIENT_VISIBLE_FRAMES")

    def test_iou_association_prefers_temporal_match(self):
        previous = [0.0, 0.0, 10.0, 10.0]
        candidates = [
            {"bbox_xyxy": [0.0, 0.0, 9.0, 9.0], "confidence": 0.6},
            {"bbox_xyxy": [50.0, 50.0, 60.0, 60.0], "confidence": 0.95},
        ]
        selected, overlap = choose_target(candidates, previous)
        self.assertEqual(selected, candidates[0])
        self.assertGreater(overlap, 0.8)
        self.assertAlmostEqual(iou(previous, previous), 1.0)

    def test_tiled_search_covers_frame_and_deduplicates_overlap(self):
        import numpy as np

        tiles = two_by_two_tiles(np.zeros((100, 200, 3), dtype=np.uint8), 0.20)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0][1:], (0, 0))
        self.assertEqual(tiles[-1][1] + tiles[-1][0].shape[1], 200)
        self.assertEqual(tiles[-1][2] + tiles[-1][0].shape[0], 100)
        candidates = [
            {"bbox_xyxy": [10, 10, 30, 30], "confidence": 0.9},
            {"bbox_xyxy": [11, 10, 31, 30], "confidence": 0.8},
            {"bbox_xyxy": [80, 80, 90, 90], "confidence": 0.7},
        ]
        self.assertEqual(len(deduplicate_candidates(candidates)), 2)

    def test_r5_exemplar_is_rgb_only_first_segment_maximum(self):
        frames = [
            {"frame_index": index, "observation_source": "none", "bbox_xyxy": None, "target_confidence": 0.0}
            for index in range(8)
        ]
        frames[2].update(observation_source="detector", bbox_xyxy=[1, 2, 3, 4], target_confidence=0.71)
        frames[3].update(observation_source="detector", bbox_xyxy=[2, 3, 4, 5], target_confidence=0.85)
        frames[6].update(observation_source="detector", bbox_xyxy=[3, 4, 5, 6], target_confidence=0.99)
        result = select_trusted_exemplar({
            "frames": frames,
            "events": [
                {"frame_index": 2, "event": "ACQUIRED"},
                {"frame_index": 5, "event": "LOST"},
                {"frame_index": 6, "event": "REACQUIRED"},
            ],
        })
        self.assertEqual(result["frame_index"], 3)
        self.assertEqual(result["bbox_xyxy"], [2.0, 3.0, 4.0, 5.0])

    def test_mines_search_track_loss_reacquire_and_approach(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.zip"
            times = [index * 1_000_000_000 for index in range(40)]
            visible = set(range(6, 19)) | set(range(26, 40))
            bbox_rows = [{"stream_id": "214-1", "object_uid": "7", "timestamp[ns]": timestamp, "x_min[pixel]": 100, "x_max[pixel]": 200, "y_min[pixel]": 100, "y_max[pixel]": 200, "visibility_ratio[%]": 0.9 if index in visible else 0.0} for index, timestamp in enumerate(times)]
            trajectory_rows = [{"tracking_timestamp_us": timestamp // 1000, "tx_world_device": index * 0.1, "ty_world_device": 0, "tz_world_device": 0} for index, timestamp in enumerate(times)]
            scene_rows = [{"object_uid": "7", "timestamp[ns]": -1, "t_wo_x[m]": 5, "t_wo_y[m]": 0, "t_wo_z[m]": 0}]
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("instances.json", json.dumps({"7": {"instance_name": "chair"}}))
                zf.writestr("2d_bounding_box.csv", csv_bytes(list(bbox_rows[0]), bbox_rows))
                zf.writestr("aria_trajectory.csv", csv_bytes(list(trajectory_rows[0]), trajectory_rows))
                zf.writestr("scene_objects.csv", csv_bytes(list(scene_rows[0]), scene_rows))
            result = mine(archive, Thresholds(min_track_frames=10))
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["candidates"][0]["target_name"], "chair")
            self.assertEqual(result["candidates"][0]["phases"], ["SEARCH", "ACQUIRE", "TRACK", "LOST", "REACQUIRE", "APPROACH"])
            self.assertEqual(result["rgb_estimator_access_count"], 0)

    def test_rejects_archive_without_required_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("instances.json", "{}")
            with self.assertRaisesRegex(ValueError, "missing"):
                mine(archive, Thresholds())


if __name__ == "__main__":
    unittest.main()
