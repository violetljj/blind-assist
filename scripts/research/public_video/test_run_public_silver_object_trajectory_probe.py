#!/usr/bin/env python3
"""Pure tests for the frozen object-trajectory probe."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_object_trajectory_probe as probe


def detection(group: str, box: list[float], *, confidence: float = 0.8) -> dict[str, object]:
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0)
    return {
        "class_name": group,
        "group": group,
        "confidence": confidence,
        "box": box,
        "area": area,
        "center_x": (x0 + x1) / 2,
        "center_y": (y0 + y1) / 2,
        "bottom": y1,
        "corridor_overlap": probe.corridor_overlap(box),
        "threat": confidence * area,
    }


class PublicSilverObjectTrajectoryProbeTest(unittest.TestCase):
    def test_corridor_overlap_prefers_center_lower_box(self) -> None:
        center = probe.corridor_overlap([0.40, 0.60, 0.60, 0.95])
        side = probe.corridor_overlap([0.00, 0.60, 0.15, 0.95])
        upper = probe.corridor_overlap([0.40, 0.00, 0.60, 0.20])
        self.assertGreater(center, side)
        self.assertEqual(0.0, upper)

    def test_tracker_links_growing_same_group_and_separates_other_group(self) -> None:
        frames = [
            [detection("person", [0.45, 0.20, 0.55, 0.40])],
            [detection("person", [0.43, 0.20, 0.57, 0.50]), detection("vehicle", [0.0, 0.5, 0.2, 0.8])],
            [detection("person", [0.40, 0.18, 0.60, 0.65])],
        ]
        tracks = probe.track_detections(frames)
        lengths = sorted((len(track["detections"]), track["detections"][0]["group"]) for track in tracks)
        self.assertEqual([(1, "vehicle"), (3, "person")], lengths)
        person = next(track for track in tracks if track["detections"][0]["group"] == "person")
        vector = probe.track_vector(person, frame_count=3)
        self.assertGreater(vector[8], 0.0)
        self.assertGreater(vector[11], 0.0)

    def test_episode_vector_is_fixed_size_and_deterministic(self) -> None:
        frames = [
            [detection("person", [0.45, 0.20, 0.55, 0.40])],
            [detection("person", [0.42, 0.20, 0.58, 0.55])],
            [detection("person", [0.38, 0.18, 0.62, 0.72])],
        ]
        first, summary = probe.episode_vector(frames)
        second, _ = probe.episode_vector(frames)
        self.assertEqual(163, len(first))
        self.assertEqual(1, summary["track_count"])
        np.testing.assert_array_equal(first, second)

    def test_normalize_detection_clips_and_groups(self) -> None:
        row = probe.normalize_detection("motorcycle", 0.5, [-10, 20, 80, 120], width=100, height=100)
        self.assertEqual("vehicle", row["group"])
        self.assertEqual([0.0, 0.2, 0.8, 1.0], row["box"])
        self.assertGreater(row["threat"], 0.0)


if __name__ == "__main__":
    unittest.main()
