from __future__ import annotations

from dataclasses import dataclass
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
    load_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    generate_sequence,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    SparseTrackResult,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.support_manager import (
    CURRENT_LEG_SURVIVOR,
    GEOMETRIC_FIELD_EXIT,
    OBSERVABLE_OCCLUSION,
    ORDINARY_NEW_TRACK_FAILURE,
    ObservableTrackDiagnostics,
    activated_cell_indices,
    classify_new_track_failures,
    classify_prior_survivors,
    median_centered_patch_errors,
    select_spatial_supplements,
    track_observable_points,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0 import (
    evaluation as candidate_evaluation,
)


def diagnostics(
    points: np.ndarray,
    *,
    forward: np.ndarray | None = None,
    available: bool = False,
    accepted: bool = False,
) -> ObservableTrackDiagnostics:
    initial = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    count = initial.shape[0]
    forward_points = (
        np.asarray(forward, dtype=np.float32).reshape(-1, 2)
        if forward is not None
        else np.full((count, 2), np.nan, dtype=np.float32)
    )
    return ObservableTrackDiagnostics(
        initial_points=initial,
        forward_points=forward_points,
        forward_available=np.full(count, available, dtype=bool),
        forward_backward_errors=np.full(
            count, 0.0 if accepted else np.inf, dtype=np.float32
        ),
        forward_backward_pass=np.full(count, accepted, dtype=bool),
        source_patch_valid=np.ones(count, dtype=bool),
        target_patch_valid=np.full(count, accepted, dtype=bool),
        photometric_errors=np.full(
            count, 0.0 if accepted else np.inf, dtype=np.float32
        ),
        photometric_pass=np.full(count, accepted, dtype=bool),
        accepted=np.full(count, accepted, dtype=bool),
    )


@dataclass(frozen=True)
class Cell:
    support_count: int
    hull_fraction: float


class SupportManagerR0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_api_cannot_receive_generator_or_sequence_metadata(self) -> None:
        parameters = inspect.signature(track_observable_points).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "source_image",
                "target_image",
                "initial_points",
                "source_valid_mask",
                "target_valid_mask",
                "parameters",
                "forward_backward_threshold_pixels",
                "photometric_threshold_intensity",
            ),
        )

    def test_full_candidate_is_deterministic_and_oracle_firewalled(self) -> None:
        spec = TrialSpec(
            trial_id="unit_determinism_oracle_firewall_seed17",
            split="clean",
            motion_family="pure_rotation",
            axis="pitch",
            angular_velocity_deg_per_s=15.0,
            scale_rate_per_s=0.0,
            fps=15,
            degradation="clean",
            seed=17,
        )
        sequence = generate_sequence(spec, self.protocol)

        class PoisonSequence:
            def __init__(self) -> None:
                self.frames = sequence.frames
                self.valid_masks = sequence.valid_masks
                self.timestamps_seconds = sequence.timestamps_seconds
                self.rotation_homography_previous_to_current = (
                    sequence.rotation_homography_previous_to_current
                )
                self.base_sha256 = sequence.base_sha256
                self.sequence_sha256 = sequence.sequence_sha256

            @property
            def occlusion_masks(self) -> tuple[np.ndarray, ...]:
                raise AssertionError("ORACLE_MASK_ACCESSED")

        with patch.object(
            candidate_evaluation,
            "generate_sequence",
            side_effect=lambda _spec, _protocol: PoisonSequence(),
        ):
            first, _ = candidate_evaluation.run_trial(spec, self.protocol)
            second, _ = candidate_evaluation.run_trial(spec, self.protocol)
        self.assertEqual(first, second)

    def test_full_49_sample_patch_and_median_centering(self) -> None:
        rng = np.random.default_rng(17)
        image = rng.integers(30, 181, size=(32, 32), dtype=np.uint8)
        brightened = (image.astype(np.int16) + 20).astype(np.uint8)
        mask = np.full(image.shape, 255, dtype=np.uint8)
        errors, source_valid, target_valid = median_centered_patch_errors(
            image,
            brightened,
            np.asarray([[12.25, 15.5], [2.5, 2.5]], dtype=np.float32),
            np.asarray([[12.25, 15.5], [2.5, 2.5]], dtype=np.float32),
            mask,
            mask,
        )
        self.assertTrue(source_valid[0])
        self.assertTrue(target_valid[0])
        self.assertLess(float(errors[0]), 1e-5)
        self.assertFalse(source_valid[1])
        self.assertFalse(target_valid[1])
        self.assertTrue(np.isinf(errors[1]))

    def test_field_exit_precedes_observable_occlusion(self) -> None:
        point = np.asarray([[29.0, 16.0]], dtype=np.float32)
        current = diagnostics(point, available=False, accepted=False)
        classes = classify_prior_survivors(
            point,
            np.asarray([[4.0, 0.0]], dtype=np.float32),
            current,
            np.full((32, 32), 255, dtype=np.uint8),
            prior_dt_seconds=1.0,
            current_dt_seconds=1.0,
        )
        self.assertEqual(classes.tolist(), [GEOMETRIC_FIELD_EXIT])

    def test_prior_failure_inside_domain_is_observable_occlusion(self) -> None:
        point = np.asarray([[16.0, 16.0]], dtype=np.float32)
        current = diagnostics(
            point,
            forward=np.asarray([[16.0, 16.0]], dtype=np.float32),
            available=True,
            accepted=False,
        )
        classes = classify_prior_survivors(
            point,
            np.zeros((1, 2), dtype=np.float32),
            current,
            np.full((32, 32), 255, dtype=np.uint8),
            prior_dt_seconds=1.0,
            current_dt_seconds=1.0,
        )
        self.assertEqual(classes.tolist(), [OBSERVABLE_OCCLUSION])

    def test_new_failure_is_never_promoted_to_occlusion(self) -> None:
        point = np.asarray([[16.0, 16.0]], dtype=np.float32)
        current = diagnostics(
            point,
            forward=point,
            available=True,
            accepted=False,
        )
        classes = classify_new_track_failures(
            current, np.full((32, 32), 255, dtype=np.uint8)
        )
        self.assertEqual(classes.tolist(), [ORDINARY_NEW_TRACK_FAILURE])
        self.assertNotIn(OBSERVABLE_OCCLUSION, classes.tolist())

    def test_current_survivor_is_retained(self) -> None:
        point = np.asarray([[16.0, 16.0]], dtype=np.float32)
        current = diagnostics(
            point,
            forward=np.asarray([[16.5, 16.0]], dtype=np.float32),
            available=True,
            accepted=True,
        )
        classes = classify_prior_survivors(
            point,
            np.asarray([[0.5, 0.0]], dtype=np.float32),
            current,
            np.full((32, 32), 255, dtype=np.uint8),
            prior_dt_seconds=1.0,
            current_dt_seconds=1.0,
        )
        self.assertEqual(classes.tolist(), [CURRENT_LEG_SURVIVOR])

    def test_spatial_supplements_are_deterministic_and_exclusion_bound(self) -> None:
        rng = np.random.default_rng(20260726)
        image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        mask = np.full(image.shape, 255, dtype=np.uint8)
        existing = np.asarray([[8.0, 8.0]], dtype=np.float32)
        exclusion = np.asarray([[48.0, 48.0]], dtype=np.float32)
        first = select_spatial_supplements(
            image, mask, (0, 0, 64, 64), existing, exclusion
        )
        second = select_spatial_supplements(
            image, mask, (0, 0, 64, 64), existing, exclusion
        )
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(first.shape[0], 16)
        if first.size:
            distances = np.linalg.norm(first - exclusion[0], axis=1)
            self.assertTrue(np.all(distances >= 10.0))

    def test_only_support_or_hull_deficit_activates_manager(self) -> None:
        sufficient = [Cell(12, 0.10) for _ in range(9)]
        self.assertEqual(activated_cell_indices(sufficient, sufficient), ())
        support_deficit = sufficient.copy()
        support_deficit[2] = Cell(11, 0.50)
        hull_deficit = sufficient.copy()
        hull_deficit[7] = Cell(20, 0.09)
        self.assertEqual(
            activated_cell_indices(support_deficit, hull_deficit), (2, 7)
        )

    def test_support_below_12_remains_not_evaluable(self) -> None:
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
        self.assertEqual(cells[0].abstention_reason, "LK_TRACK_SUPPORT_BELOW_12")

    def test_clustered_support_remains_below_frozen_hull_gate(self) -> None:
        previous = np.asarray(
            [[20.0 + index, 20.0 + 0.05 * index] for index in range(20)],
            dtype=np.float32,
        )
        tracks = SparseTrackResult(
            previous_points=previous,
            current_points=previous + np.asarray([0.2, 0.1], dtype=np.float32),
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
