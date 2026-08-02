#!/usr/bin/env python3
"""Tests for the D11 causal kinematic information diagnostic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d11_thor_magni_kinematic_information_ceiling import (
    causal_relative_velocity,
    risk_scores,
)


class D11KinematicInformationTests(unittest.TestCase):
    def test_velocity_fit_does_not_read_future_rows(self) -> None:
        times = np.linspace(-0.8, 0.2, 101)
        camera = np.zeros((101, 3))
        body = np.zeros((101, 3))
        body[:, 0] = times
        body[times > 0.0, 0] = 1000.0
        velocity = causal_relative_velocity(
            times,
            camera,
            body,
            80,
        )
        self.assertIsNotNone(velocity)
        np.testing.assert_allclose(velocity, [1.0, 0.0], atol=1e-12)

    def test_motion_into_corridor_increases_risk(self) -> None:
        position = [np.asarray((2.0, 1.5))]
        forward = np.asarray((1.0, 0.0))
        static = risk_scores(
            position,
            [np.zeros(2)],
            forward,
        )
        moving = risk_scores(
            position,
            [np.asarray((0.0, -0.5))],
            forward,
        )
        self.assertGreater(moving[1], static[1])


if __name__ == "__main__":
    unittest.main()
