from __future__ import annotations

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.temporal_structure_diagnostic_r1.extract import (
    extract,
    flow_direction_metrics,
)
from scripts.research.egomotion_compensated_looming.temporal_structure_diagnostic_r1.analyze import (
    component_spectrum,
    direction_evaluable,
    event_lengths,
    flow_temporal_metrics,
    pose_cycles,
    summarize_session,
)


class TemporalStructureTest(unittest.TestCase):
    def test_extractor_rejects_sealed_session_before_file_access(self) -> None:
        with self.assertRaisesRegex(
            PermissionError, "SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN"
        ):
            extract(16, Path("missing"), Path("unused"), Path("missing"))

    def test_real_lk_metrics_recover_translation_and_radial_expansion(self) -> None:
        rng = np.random.default_rng(20260728)
        previous = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
        valid = np.full(previous.shape, 255, dtype=np.uint8)
        translated = cv2.warpAffine(
            previous,
            np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
            (320, 240),
        )
        translation = flow_direction_metrics(previous, translated, valid)
        self.assertGreaterEqual(translation["direction_track_count"], 60)
        self.assertAlmostEqual(
            translation["median_flow_dx_px"], 2.0, delta=0.2
        )
        self.assertAlmostEqual(
            translation["median_flow_dy_px"], -1.0, delta=0.2
        )
        self.assertGreater(translation["spatial_direction_resultant"], 0.95)

        center = (159.5, 119.5)
        radial = cv2.warpAffine(
            previous,
            cv2.getRotationMatrix2D(center, 0.0, 1.02),
            (320, 240),
        )
        expansion = flow_direction_metrics(previous, radial, valid)
        self.assertGreaterEqual(expansion["radial_direction_track_count"], 60)
        self.assertGreater(expansion["median_radial_flow_px"], 0.5)
        self.assertGreater(expansion["radial_direction_consistency"], 0.9)

    def test_pose_band_energy_and_dominant_frequency(self) -> None:
        timestamps = np.arange(601, dtype=np.float64) / 60.0
        values = np.sin(2.0 * math.pi * 1.5 * timestamps)
        result = component_spectrum(timestamps, values)
        self.assertGreater(result["band_energy_fraction"], 0.95)
        self.assertAlmostEqual(result["dominant_frequency_hz"], 1.5, delta=0.11)

    def test_pose_cycles_report_recurrence_and_phase_locking(self) -> None:
        timestamps = np.arange(601, dtype=np.float64) / 60.0
        bandpassed = np.sin(2.0 * math.pi * 1.5 * timestamps)
        evaluable = np.ones(601, dtype=bool)
        high_response = bandpassed > 0.9
        absolute_response = bandpassed.copy()
        result = pose_cycles(
            timestamps,
            bandpassed,
            timestamps,
            evaluable,
            high_response,
            absolute_response,
        )
        self.assertGreaterEqual(result["valid_cycle_count"], 13)
        self.assertEqual(result["high_response_cycle_fraction"], 1.0)
        self.assertGreaterEqual(
            result["longest_consecutive_high_response_cycles"], 13
        )
        self.assertGreater(
            result[
                "cycle_max_absolute_response_axial_phase_locking_value"
            ],
            0.95,
        )

    def test_invalid_cycle_breaks_consecutive_high_response_run(self) -> None:
        timestamps = np.arange(601, dtype=np.float64) / 60.0
        bandpassed = np.sin(2.0 * math.pi * timestamps)
        evaluable = np.ones(601, dtype=bool)
        evaluable[(timestamps >= 4.0) & (timestamps < 5.0)] = False
        result = pose_cycles(
            timestamps,
            bandpassed,
            timestamps,
            evaluable,
            np.ones(601, dtype=bool),
            np.abs(bandpassed),
        )
        self.assertEqual(result["valid_cycle_count"], 7)
        self.assertEqual(
            result["longest_consecutive_high_response_cycles"], 4
        )

    def test_flow_direction_periodicity_uses_response_blind_vectors(self) -> None:
        timestamps = np.arange(601, dtype=np.float64) / 60.0
        rows = []
        for timestamp in timestamps:
            rows.append(
                {
                    "direction_track_count": 100,
                    "forward_backward_consistent_fraction": 0.9,
                    "median_forward_backward_error_px": 0.2,
                    "spatial_direction_resultant": 0.9,
                    "radial_direction_track_count": 100,
                    "median_radial_flow_px": 0.0,
                    "radial_direction_consistency": 0.9,
                    "median_flow_dx_px": 2.0
                    + math.sin(2.0 * math.pi * 1.5 * timestamp),
                    "median_flow_dy_px": 0.1,
                }
            )
        result = flow_temporal_metrics(timestamps, rows, 1.5)
        result.pop("_direction_evaluable_mask")
        self.assertEqual(result["direction_evaluable_fraction"], 1.0)
        self.assertGreater(result["median_adjacent_direction_cosine"], 0.99)
        self.assertGreater(
            result["flow_periodic_r_squared_at_pose_frequency"], 0.99
        )

    def test_failure_events_preserve_temporal_runs(self) -> None:
        self.assertEqual(
            event_lengths(
                np.asarray([False, True, True, False, True, True, True])
            ),
            [2, 3],
        )

    def test_radial_structure_can_make_pair_direction_evaluable(self) -> None:
        self.assertTrue(
            direction_evaluable(
                {
                    "direction_track_count": 100,
                    "forward_backward_consistent_fraction": 0.9,
                    "median_forward_backward_error_px": 0.2,
                    "spatial_direction_resultant": 0.1,
                    "radial_direction_track_count": 80,
                    "radial_direction_consistency": 0.8,
                }
            )
        )
        self.assertFalse(
            direction_evaluable(
                {
                    "direction_track_count": 0,
                    "forward_backward_consistent_fraction": 0.0,
                    "median_forward_backward_error_px": None,
                    "spatial_direction_resultant": None,
                    "radial_direction_track_count": 0,
                    "radial_direction_consistency": None,
                }
            )
        )

    def test_full_session_routes_track_consistent_periodic_pattern(self) -> None:
        direction_rows = []
        proxy_rows = []
        r3_rows = []
        for index in range(601):
            previous = index / 60.0
            current = (index + 1) / 60.0
            center = 0.5 * (previous + current)
            sine = math.sin(2.0 * math.pi * 1.5 * center)
            direction_rows.append(
                {
                    "session": 13,
                    "pair_index": index,
                    "previous_timestamp_s": previous,
                    "current_timestamp_s": current,
                    "camera_angular_velocity_x_deg_per_s": sine,
                    "camera_angular_velocity_y_deg_per_s": 0.0,
                    "camera_angular_velocity_z_deg_per_s": 0.0,
                    "camera_translation_velocity_x_m_per_s": 0.0,
                    "camera_translation_velocity_y_m_per_s": 0.0,
                    "camera_translation_velocity_z_m_per_s": 0.0,
                    "direction_track_count": 100,
                    "forward_backward_consistent_fraction": 0.9,
                    "median_forward_backward_error_px": 0.2,
                    "spatial_direction_resultant": 0.9,
                    "radial_direction_track_count": 100,
                    "median_radial_flow_px": sine,
                    "radial_direction_consistency": 0.9,
                    "median_flow_dx_px": 2.0 + sine,
                    "median_flow_dy_px": 0.1,
                }
            )
            proxy_rows.append(
                {
                    "session": 13,
                    "pair_index": index,
                    "sharpness_laplacian_variance": 100.0,
                    "detected_features_per_valid_megapixel": 1000.0,
                    "detected_feature_count": 100,
                    "forward_backward_consistent_count": 90,
                    "forward_backward_consistent_fraction": 0.9,
                    "median_forward_backward_error_px": 0.2,
                }
            )
            r3_rows.append(
                {
                    "session": 13,
                    "pair_index": index,
                    "evaluable": True,
                    "compensated_expansion_median_per_s": sine,
                }
            )
        result = summarize_session(13, direction_rows, proxy_rows, r3_rows)
        self.assertTrue(result["valid_for_cross_session_terminal"])
        self.assertTrue(
            result["track_consistent_periodic_motion_support"], result
        )
        self.assertFalse(result["measurement_failure_support"])


if __name__ == "__main__":
    unittest.main()
