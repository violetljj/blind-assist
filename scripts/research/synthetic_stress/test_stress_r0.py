"""Contract and mutation tests for the controlled synthetic stress runner."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts.research.synthetic_stress import run_stress_r0 as stress


class SyntheticStressR0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = stress.load_json(stress.PROTOCOL_PATH)

    def test_protocol_is_development_only(self) -> None:
        authority = self.protocol["authority"]
        self.assertEqual(authority["maximum_claim"], "synthetic_mechanics_and_implementation_diagnostic_only")
        self.assertFalse(authority["confirmation"])
        self.assertFalse(authority["production"])
        self.assertFalse(authority["default_app_changed"])

    def test_case_design_is_large_and_non_cartesian(self) -> None:
        cases = stress.build_cases(self.protocol)
        self.assertGreaterEqual(len(cases), 3000)
        self.assertEqual(len(cases), len({case["case_id"] for case in cases}))
        blocks = {case["design_block"].split("_")[0] for case in cases}
        self.assertIn("motion", blocks)
        self.assertIn("one", blocks)
        self.assertIn("pairwise", blocks)
        self.assertIn("stress", blocks)

    def test_boundary_suite_passes_and_rejects_unknown_numeric_mutation(self) -> None:
        result = stress._boundary_suite(self.protocol)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["passed"], result["total"])
        mutation = next(
            item for item in result["checks"] if item["name"] == "unknown_numeric_mutation_rejected"
        )
        self.assertTrue(mutation["caught"])

    def test_d2_predicted_basis_has_expected_translation_sign(self) -> None:
        history = stress.camera_binding(np.asarray([0.0, 0.0, -0.4]), np.eye(3))
        current = stress.camera_binding(np.zeros(3), np.eye(3))
        plane = {
            "camera_ground_projection_m": [0.0, 0.0, 0.0],
            "normal_toward_camera": [0.0, 1.0, 0.0],
        }
        _, predicted, receipt = stress.predicted_bases(history, current, plane)
        self.assertGreater(receipt["tangent_translation_velocity_m_s"][2], 0.0)
        self.assertTrue(np.allclose(predicted[0.4][0], [0.0, 0.0, 0.4]))

    def test_clean_scale_case_has_signed_expansion_or_honest_terminal(self) -> None:
        motion = next(item for item in stress.motion_catalog(self.protocol) if item.family == "scale" and item.sign > 0)
        case = stress.make_case("unit", motion, stress.base_degradation(), 20260802)
        row = stress.run_case(case, self.protocol)
        self.assertIn(row["rcle"]["status"], {"EVALUABLE", "NOT_EVALUABLE_INSUFFICIENT_TRACK_SUPPORT"})
        if row["rcle"]["status"] == "EVALUABLE":
            self.assertIn("expansion_sign", row["rcle"])
            self.assertEqual(row["rcle"]["expansion_sign"]["expected"], 1)


if __name__ == "__main__":
    unittest.main()
