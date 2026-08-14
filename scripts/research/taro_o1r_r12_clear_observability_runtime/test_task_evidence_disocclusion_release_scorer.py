import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_disocclusion_release_scorer as subject


def _record(name: str, translation: float, release: float, margin: float, target: int = 0) -> SimpleNamespace:
    frame = SimpleNamespace(frame_id=name)
    pair = SimpleNamespace(
        translation_m=translation,
        rotation_deg=0.0,
        gap_s=0.5,
        neighbor=frame,
    )
    return SimpleNamespace(
        reference_id="reference",
        pair=pair,
        analytic={
            "robust_release_cell_count": release,
            "release_margin_sum": margin,
        },
        target_gain=target,
    )


class DisocclusionReleaseScorerTest(unittest.TestCase):
    def test_candidate_feature_path_is_finite_without_candidate_depth(self) -> None:
        reference = SimpleNamespace(camera_to_world=np.eye(4), frame_id="reference")
        candidate_pose = np.eye(4)
        candidate_pose[0, 3] = 0.10
        neighbor = SimpleNamespace(camera_to_world=candidate_pose, frame_id="candidate")
        pair = SimpleNamespace(
            reference=reference,
            neighbor=neighbor,
            translation_m=0.10,
            rotation_deg=0.0,
            gap_s=0.5,
        )
        query = {
            "virtual_query_frame": {
                "origin_camera_xyz": [0.0, 1.2, 0.0],
                "forward_camera_xyz": [0.0, 0.0, 1.0],
                "lateral_camera_xyz": [1.0, 0.0, 0.0],
                "gravity_up_camera_xyz": [0.0, -1.0, 0.0],
                "path_heading_camera_xyz": [0.0, 0.0, 1.0],
            },
            "path_lateral_offset_m": 0.0,
        }
        depth = np.ones((20, 20), dtype=np.float64)
        context = SimpleNamespace(
            row=SimpleNamespace(reference=reference),
            low_depth=depth,
            valid=np.ones_like(depth, dtype=bool),
            intrinsics=np.asarray([[10.0, 0.0, 9.5], [0.0, 10.0, 9.5], [0.0, 0.0, 1.0]]),
            queries=[query],
            static=np.zeros((1, 6, 3, 4), dtype=bool),
        )
        features, analytic = subject.disocclusion_candidate_features(context, pair)
        self.assertEqual((6,), features.shape)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertIn("robust_release_cell_count", analytic)

    def test_edge_search_requires_observed_non_foreground_depth(self) -> None:
        depth = np.ones((7, 9), dtype=np.float64)
        valid = np.ones_like(depth, dtype=bool)
        depth[3, 7] = 4.0
        edge = subject._foreground_edge_distances(
            depth,
            valid,
            np.asarray([4.0]),
            np.asarray([3.0]),
            np.asarray([1.0]),
            np.asarray([0.0]),
            np.asarray([3.0]),
            np.asarray([True]),
        )
        self.assertEqual(3.0, float(edge[0]))
        valid[3, 7] = False
        fail_closed = subject._foreground_edge_distances(
            depth,
            valid,
            np.asarray([4.0]),
            np.asarray([3.0]),
            np.asarray([1.0]),
            np.asarray([0.0]),
            np.asarray([3.0]),
            np.asarray([True]),
        )
        self.assertEqual(subject.MAX_FOREGROUND_EDGE_SEARCH_PX + 1, float(fail_closed[0]))

    def test_primary_overrides_only_for_one_cell_advantage(self) -> None:
        records = [
            _record("generic", 1.0, 2.0, 2.0),
            _record("task", 0.5, 3.0, 1.0),
        ]
        scores, receipt = subject.primary_selection_scores(records)
        self.assertEqual([0.0, 1.0], scores.tolist())
        self.assertEqual(1, receipt["generic_override_count"])
        records[1].analytic["robust_release_cell_count"] = 2.0
        scores, receipt = subject.primary_selection_scores(records)
        self.assertEqual([1.0, 0.0], scores.tolist())
        self.assertEqual(1, receipt["generic_fallback_count"])

    def test_selection_does_not_read_target_gain(self) -> None:
        records = [
            _record("generic", 1.0, 0.0, 0.0, target=-999),
            _record("task", 0.5, 2.0, 1.0, target=999),
        ]
        before, _ = subject.primary_selection_scores(records)
        records[0].target_gain = 999
        records[1].target_gain = -999
        after, _ = subject.primary_selection_scores(records)
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
