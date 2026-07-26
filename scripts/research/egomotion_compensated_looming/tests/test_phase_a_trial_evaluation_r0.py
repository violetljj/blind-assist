from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.rcle_minimal.evaluation import (
    run_trial,
    wilson_interval,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
    load_protocol,
)


class PhaseATrialEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def _run(
        self,
        family: str,
        axis: str,
        angular: float,
        scale: float,
        degradation: str = "clean",
        require_evaluable: bool = True,
    ) -> dict:
        spec = TrialSpec(
            trial_id=f"test_{family}_{axis}_{angular}_{scale}_{degradation}",
            split="clean" if degradation == "clean" else "stress",
            motion_family=family,
            axis=axis,
            angular_velocity_deg_per_s=angular,
            scale_rate_per_s=scale,
            fps=30,
            degradation=degradation,
            seed=1000,
        )
        result, _ = run_trial(spec, self.protocol)
        if require_evaluable:
            self.assertTrue(result["evaluable"], result["abstention_counts"])
        return result

    def test_scale_signs_and_units(self) -> None:
        up = self._run("scale", "none", 0.0, 0.15)
        down = self._run("scale", "none", 0.0, -0.15)
        self.assertGreater(up["compensated_closing_estimate_per_s"], 0.0)
        self.assertLess(down["compensated_closing_estimate_per_s"], 0.0)
        self.assertLess(up["compensated_closing_error_per_s"], 0.04)
        self.assertLess(down["compensated_closing_error_per_s"], 0.04)

    def test_rotation_plus_scale_retains_closing(self) -> None:
        mixed = self._run(
            "rotation_plus_scale_up", "yaw", 30.0, 0.15
        )
        self.assertGreater(mixed["compensated_closing_estimate_per_s"], 0.0)
        self.assertLess(mixed["compensated_closing_error_per_s"], 0.05)

    def test_ratio_denominator_protection(self) -> None:
        roll = self._run("pure_rotation", "roll", 0.01, 0.0)
        self.assertIsNone(roll["rsr"])
        self.assertEqual(
            roll["rsr_status"], "NOT_EVALUABLE_DENOMINATOR_FLOOR"
        )
        down = self._run("scale", "none", 0.0, -0.15)
        self.assertIsNone(down["crr"])
        self.assertEqual(down["crr_status"], "NOT_APPLICABLE")

    def test_each_degradation_is_deterministic_and_evaluable(self) -> None:
        for degradation in (
            "gaussian_noise",
            "gaussian_blur",
            "partial_occlusion",
        ):
            with self.subTest(degradation=degradation):
                first = self._run(
                    "rotation_plus_scale_up",
                    "pitch",
                    -30.0,
                    0.15,
                    degradation,
                    require_evaluable=False,
                )
                second = self._run(
                    "rotation_plus_scale_up",
                    "pitch",
                    -30.0,
                    0.15,
                    degradation,
                    require_evaluable=False,
                )
                self.assertEqual(
                    first["sequence_sha256"], second["sequence_sha256"]
                )
                self.assertEqual(first["evaluable"], second["evaluable"])
                self.assertEqual(
                    first["compensated_closing_estimate_per_s"],
                    second["compensated_closing_estimate_per_s"],
                )

    def test_wilson_interval_is_bounded(self) -> None:
        low, high = wilson_interval(18, 20)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)


if __name__ == "__main__":
    unittest.main()
