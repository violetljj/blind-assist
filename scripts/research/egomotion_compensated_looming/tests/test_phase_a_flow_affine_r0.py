from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    load_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.sparse_flow import (
    SparseTrackResult,
    detect_fixed_grid_features,
    track_features,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    make_base_texture,
)


class PhaseAFlowAffineTest(unittest.TestCase):
    def test_sparse_lk_recovers_translation(self) -> None:
        protocol = load_protocol()
        parameters = protocol["sparse_lk"]
        image = make_base_texture(480, 360, 1000)
        matrix = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, -1.5]])
        shifted = cv2.warpAffine(
            image,
            matrix,
            (480, 360),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = np.full_like(image, 255)
        points = detect_fixed_grid_features(image, mask, parameters)
        tracks = track_features(image, shifted, points, mask, parameters)
        displacement = tracks.current_points - tracks.previous_points
        self.assertGreater(tracks.valid_count, 200)
        np.testing.assert_allclose(
            np.median(displacement, axis=0),
            np.asarray([2.0, -1.5]),
            atol=0.15,
        )

    def test_exact_affine_recovers_expansion_per_second(self) -> None:
        protocol = load_protocol()
        parameters = protocol["local_affine"]
        points: list[list[float]] = []
        for y in np.linspace(8, 352, 18):
            for x in np.linspace(8, 472, 24):
                points.append([x, y])
        previous = np.asarray(points, dtype=np.float32)
        dt = 1.0 / 30.0
        center = np.asarray([239.5, 179.5])
        expansion = 0.2
        velocity = expansion * (previous - center)
        current = previous + velocity * dt
        tracks = SparseTrackResult(
            previous_points=previous,
            current_points=current.astype(np.float32),
            forward_backward_errors=np.zeros(previous.shape[0], dtype=np.float32),
            requested_count=previous.shape[0],
        )
        cells = fit_fixed_grid_local_affine(
            tracks, dt, (360, 480), parameters
        )
        self.assertEqual(sum(cell.evaluable for cell in cells), 9)
        for cell in cells:
            self.assertIsNotNone(cell.expansion)
            self.assertAlmostEqual(cell.expansion or 0.0, expansion, places=5)

    def test_insufficient_support_is_not_evaluable(self) -> None:
        protocol = load_protocol()
        tracks = SparseTrackResult(
            previous_points=np.asarray([[10.0, 10.0]], dtype=np.float32),
            current_points=np.asarray([[10.1, 10.1]], dtype=np.float32),
            forward_backward_errors=np.zeros(1, dtype=np.float32),
            requested_count=1,
        )
        cells = fit_fixed_grid_local_affine(
            tracks,
            1.0 / 30.0,
            (360, 480),
            protocol["local_affine"],
        )
        self.assertTrue(all(not cell.evaluable for cell in cells))
        self.assertTrue(
            all(
                cell.abstention_reason == "LK_TRACK_SUPPORT_BELOW_12"
                for cell in cells
            )
        )

    def test_non_positive_dt_is_rejected(self) -> None:
        protocol = load_protocol()
        empty = SparseTrackResult(
            previous_points=np.empty((0, 2), dtype=np.float32),
            current_points=np.empty((0, 2), dtype=np.float32),
            forward_backward_errors=np.empty((0,), dtype=np.float32),
            requested_count=0,
        )
        with self.assertRaisesRegex(ValueError, "NON_POSITIVE"):
            fit_fixed_grid_local_affine(
                empty, 0.0, (360, 480), protocol["local_affine"]
            )


if __name__ == "__main__":
    unittest.main()
