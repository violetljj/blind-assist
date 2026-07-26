from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    TrialSpec,
    enumerate_trials,
    load_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    make_base_texture,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.evaluation import (
    IMPLEMENTATION_REVISION,
    run_trial,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    SparseTrackResult,
    detect_fixed_grid_features,
    track_features,
)


class PhaseACoverageRevisionR1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def _run(
        self,
        *,
        family: str,
        axis: str,
        angular: float,
        scale: float,
        fps: int,
        degradation: str,
        seed: int = 1000,
    ) -> dict:
        result, _ = run_trial(
            TrialSpec(
                trial_id=(
                    f"r1_{family}_{axis}_{angular}_{scale}_{fps}_"
                    f"{degradation}_{seed}"
                ),
                split="clean" if degradation == "clean" else "stress",
                motion_family=family,
                axis=axis,
                angular_velocity_deg_per_s=angular,
                scale_rate_per_s=scale,
                fps=fps,
                degradation=degradation,
                seed=seed,
            ),
            self.protocol,
        )
        return result

    def test_r0_protocol_and_inventory_remain_exact(self) -> None:
        self.assertEqual(
            PROTOCOL_SHA256,
            "d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502",
        )
        protocol_on_disk = json.loads(
            Path(PROTOCOL_PATH).read_text(encoding="utf-8")
        )
        self.assertEqual(protocol_on_disk, self.protocol)
        trials = enumerate_trials(self.protocol)
        self.assertEqual(len(trials), 2520)
        self.assertEqual(len({trial.trial_id for trial in trials}), 2520)
        self.assertEqual(
            self.protocol["trials"]["seeds"], list(range(1000, 1020))
        )

    def test_multilevel_backward_cycle_retains_boundary_support(self) -> None:
        image = make_base_texture(480, 360, 1000)
        shifted = np.zeros_like(image)
        shifted[18:] = image[:-18]
        valid = np.zeros_like(image)
        valid[18:] = 255
        points = detect_fixed_grid_features(
            image, np.full_like(image, 255), self.protocol["sparse_lk"]
        )
        tracks = track_features(
            image,
            shifted,
            points,
            valid,
            self.protocol["sparse_lk"],
        )
        self.assertGreater(tracks.valid_count, 300)
        displacement = tracks.current_points - tracks.previous_points
        self.assertAlmostEqual(
            float(np.median(displacement[:, 1])), 18.0, delta=0.2
        )

    def test_clean_pitch_15fps_r0_failure_cell_is_now_evaluable(self) -> None:
        result = self._run(
            family="pure_rotation",
            axis="pitch",
            angular=30.0,
            scale=0.0,
            fps=15,
            degradation="clean",
        )
        self.assertTrue(result["evaluable"], result["abstention_counts"])
        self.assertEqual(result["implementation_revision"], IMPLEMENTATION_REVISION)
        self.assertEqual(result["evaluable_pair_count"], 9)

    def test_partial_occlusion_pitch_directions_reach_frozen_pair_gate(self) -> None:
        for angular in (-30.0, 30.0):
            for family, scale in (
                ("pure_rotation", 0.0),
                ("rotation_plus_scale_up", 0.15),
            ):
                with self.subTest(angular=angular, family=family):
                    result = self._run(
                        family=family,
                        axis="pitch",
                        angular=angular,
                        scale=scale,
                        fps=30,
                        degradation="partial_occlusion",
                    )
                    self.assertTrue(
                        result["evaluable"], result["abstention_counts"]
                    )
                    self.assertGreaterEqual(
                        result["evaluable_pair_fraction"], 0.8
                    )

    def test_partial_occlusion_revision_is_deterministic(self) -> None:
        arguments = {
            "family": "pure_rotation",
            "axis": "pitch",
            "angular": -30.0,
            "scale": 0.0,
            "fps": 30,
            "degradation": "partial_occlusion",
        }
        first = self._run(**arguments)
        second = self._run(**arguments)
        self.assertEqual(first, second)

    def test_affine_consensus_rejects_occluder_tracks_without_changing_gate(
        self,
    ) -> None:
        rng = np.random.default_rng(20260726)
        points = []
        for y in np.linspace(8, 352, 18):
            for x in np.linspace(8, 472, 24):
                points.append([x, y])
        previous = np.asarray(points, dtype=np.float32)
        dt = 1.0 / 30.0
        center = np.asarray([239.5, 179.5])
        expansion = 0.2
        current = previous + expansion * (previous - center) * dt
        outliers = rng.choice(previous.shape[0], size=80, replace=False)
        current[outliers] += rng.normal(0.0, 7.0, size=(outliers.size, 2))
        tracks = SparseTrackResult(
            previous_points=previous,
            current_points=current.astype(np.float32),
            forward_backward_errors=np.zeros(previous.shape[0], dtype=np.float32),
            requested_count=previous.shape[0],
        )
        cv2.setRNGSeed(20260726)
        cells = fit_fixed_grid_local_affine(
            tracks,
            dt,
            (360, 480),
            self.protocol["local_affine"],
        )
        self.assertEqual(sum(cell.evaluable for cell in cells), 9)
        self.assertTrue(
            any(
                cell.support_count < cell.tracked_support_count
                for cell in cells
            )
        )
        for cell in cells:
            self.assertAlmostEqual(cell.expansion or 0.0, expansion, delta=0.01)
            self.assertLessEqual(
                cell.fit_residual_pixels_per_frame or 0.0, 0.75
            )

    def test_insufficient_consensus_remains_not_evaluable(self) -> None:
        previous = np.asarray(
            [[10.0 + index, 10.0 + (index % 3)] for index in range(11)],
            dtype=np.float32,
        )
        tracks = SparseTrackResult(
            previous_points=previous,
            current_points=previous + 0.1,
            forward_backward_errors=np.zeros(11, dtype=np.float32),
            requested_count=11,
        )
        cells = fit_fixed_grid_local_affine(
            tracks,
            1.0 / 30.0,
            (360, 480),
            self.protocol["local_affine"],
        )
        self.assertFalse(cells[0].evaluable)
        self.assertEqual(
            cells[0].abstention_reason, "LK_TRACK_SUPPORT_BELOW_12"
        )

    def test_clustered_consensus_still_fails_frozen_hull_gate(self) -> None:
        previous = np.asarray(
            [[20.0 + index, 20.0 + 0.05 * index] for index in range(20)],
            dtype=np.float32,
        )
        current = previous + np.asarray([0.2, 0.1], dtype=np.float32)
        tracks = SparseTrackResult(
            previous_points=previous,
            current_points=current,
            forward_backward_errors=np.zeros(20, dtype=np.float32),
            requested_count=20,
        )
        cells = fit_fixed_grid_local_affine(
            tracks,
            1.0 / 30.0,
            (360, 480),
            self.protocol["local_affine"],
        )
        self.assertFalse(cells[0].evaluable)
        self.assertEqual(
            cells[0].abstention_reason,
            "TRACK_HULL_COVERAGE_BELOW_0_10",
        )


if __name__ == "__main__":
    unittest.main()
