"""Evaluator-only all-oracle ceiling for the frozen SAGE-LM policy."""

from __future__ import annotations

import math

from .observation import ApertureObservation, RgbEpisodeInput, RgbEpisodeTruth


class OracleApertureObservationProvider:
    """Expose evaluator truth only to the explicitly selected diagnostic arm.

    This provider is intentionally separate from ``RgbObservationProvider`` so
    evaluator labels cannot enter the real-RGB observation surface by accident.
    """

    arm_name = "SAGE_LM_ORACLE_APERTURE_PROGRESS"

    def __init__(self, episode_input: RgbEpisodeInput, truth: RgbEpisodeTruth) -> None:
        if episode_input.episode_id != truth.episode_id:
            raise ValueError("input/truth episode mismatch")
        if not truth.camera_positions_m:
            raise ValueError("oracle diagnostic requires source camera positions")
        values = (truth.aperture_center_x_m, truth.aperture_width_m, truth.start_range_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("oracle aperture truth must be finite")
        if truth.aperture_width_m <= 0.0 or truth.start_range_m <= 0.0:
            raise ValueError("oracle width and start range must be positive")
        self.input = episode_input
        self.truth = truth
        self.diagnostics = {
            "authority": "EVALUATOR_TRUTH_ALL_ORACLE_CEILING_ONLY",
            "source_camera_positions_m": [list(row) for row in truth.camera_positions_m],
        }

    def observe(self) -> ApertureObservation:
        return ApertureObservation(
            visible=True,
            center_x_m=self.truth.aperture_center_x_m,
            width_m=self.truth.aperture_width_m,
            range_m=self.truth.start_range_m,
            boundary_confidence=1.0,
            flow_confidence=1.0,
            depth_consistency=1.0,
            geometry_confidence=1.0,
        )
