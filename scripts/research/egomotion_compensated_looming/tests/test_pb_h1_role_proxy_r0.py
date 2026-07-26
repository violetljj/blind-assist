from __future__ import annotations

import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.pb_h1_role_proxy.experiment import (
    K_BONN,
    run_controlled_fixture,
)
from scripts.research.egomotion_compensated_looming.pb_h1_role_proxy.geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


class PbH1RoleProxyTest(unittest.TestCase):
    def test_controlled_fixture_passes_all_physical_checks(self) -> None:
        fixture = run_controlled_fixture()
        self.assertTrue(fixture["physical_calibration_pass"], fixture["checks"])
        self.assertTrue(all(fixture["checks"].values()))
        motions = fixture["motions"]
        self.assertGreater(
            motions["lateral_translation"][
                "median_absolute_radial_expansion_per_s"
            ],
            motions["forward_approach"][
                "median_absolute_radial_expansion_per_s"
            ],
            "Absolute radial expansion alone must not be used as approach.",
        )

    def test_pure_rotation_has_zero_translation_term(self) -> None:
        pixels = np.asarray(((100.0, 100.0), (500.0, 350.0)))
        depth = np.asarray((2.0, 4.0))
        angle = np.deg2rad(5.0)
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        summary = summarize_translation_induced_geometry(
            translation_induced_geometry(
                pixels,
                depth,
                K_BONN,
                rotation,
                np.zeros(3),
                0.1,
                image_size_wh=(640, 480),
                zbuffer=False,
            )
        )
        self.assertEqual(
            summary["median_absolute_radial_expansion_per_s"], 0.0
        )
        self.assertLessEqual(
            summary["q90_time_normalized_parallax_rad_per_s"], 1e-7
        )

    def test_invalid_dt_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "PB_H1_NONFINITE_OR_DT"):
            translation_induced_geometry(
                np.asarray(((100.0, 100.0),)),
                np.asarray((2.0,)),
                K_BONN,
                np.eye(3),
                np.zeros(3),
                0.0,
                image_size_wh=(640, 480),
            )


if __name__ == "__main__":
    unittest.main()
