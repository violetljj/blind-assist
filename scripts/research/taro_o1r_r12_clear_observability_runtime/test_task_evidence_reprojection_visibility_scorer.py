import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as subject


def _record(name: str, translation: float, novel: float, strength: float, target: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        reference_id="reference",
        pair=SimpleNamespace(
            translation_m=translation,
            rotation_deg=0.0,
            gap_s=0.5,
            neighbor=SimpleNamespace(frame_id=name),
        ),
        analytic={
            "reprojection_novel_cell_count": novel,
            "novel_appearance_strength_sum": strength,
        },
        target_gain=target,
    )


class ReprojectionVisibilityScorerTest(unittest.TestCase):
    def test_identity_warp_preserves_valid_luma_and_coverage(self) -> None:
        height, width = 6, 8
        depth = np.full((height, width), 2.0, dtype=np.float64)
        luma = np.arange(height * width, dtype=np.float32).reshape(height, width) / 100.0
        camera = SimpleNamespace(camera_to_world=np.eye(4), frame_id="camera")
        context = SimpleNamespace(
            low_depth=depth,
            valid=np.ones_like(depth, dtype=bool),
            intrinsics=np.asarray([[6.0, 0.0, 3.5], [0.0, 6.0, 2.5], [0.0, 0.0, 1.0]]),
            row=SimpleNamespace(reference=camera),
        )
        pair = SimpleNamespace(neighbor=camera)
        warped, direct, explained = subject._forward_z_buffer_warp(context, pair, luma)
        np.testing.assert_allclose(luma, warped, atol=1e-6)
        self.assertTrue(np.all(direct))
        self.assertTrue(np.all(explained))

    def test_invalid_reference_depth_never_explains_candidate_pixel(self) -> None:
        depth = np.full((5, 5), 2.0, dtype=np.float64)
        valid = np.ones_like(depth, dtype=bool)
        valid[2, 2] = False
        camera = SimpleNamespace(camera_to_world=np.eye(4), frame_id="camera")
        context = SimpleNamespace(
            low_depth=depth,
            valid=valid,
            intrinsics=np.asarray([[5.0, 0.0, 2.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]]),
            row=SimpleNamespace(reference=camera),
        )
        _warped, direct, _explained = subject._forward_z_buffer_warp(
            context, SimpleNamespace(neighbor=camera), np.ones_like(depth, dtype=np.float32)
        )
        self.assertFalse(bool(direct[2, 2]))

    def test_primary_gate_and_target_firewall(self) -> None:
        records = [
            _record("generic", 1.0, 2.0, 2.0, target=-999),
            _record("task", 0.5, 3.0, 1.0, target=999),
        ]
        before, receipt = subject.primary_selection_scores(records)
        self.assertEqual([0.0, 1.0], before.tolist())
        self.assertEqual(1, receipt["generic_override_count"])
        records[0].target_gain = 999
        records[1].target_gain = -999
        after, _ = subject.primary_selection_scores(records)
        np.testing.assert_array_equal(before, after)
        records[1].analytic["reprojection_novel_cell_count"] = 2.0
        fallback, receipt = subject.primary_selection_scores(records)
        self.assertEqual([1.0, 0.0], fallback.tolist())
        self.assertEqual(1, receipt["generic_fallback_count"])


if __name__ == "__main__":
    unittest.main()
