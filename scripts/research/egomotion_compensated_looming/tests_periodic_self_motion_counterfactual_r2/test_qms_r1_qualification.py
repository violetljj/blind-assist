from __future__ import annotations

from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    qms_r1_qualification as qms,
)


class QmsR1QualificationTests(unittest.TestCase):
    def test_new_seed_domain_is_unique(self):
        seeds = [
            qms.derive_seed(block, ordinal)
            for block in qms.BLOCKS
            for ordinal in range(4)
        ]
        self.assertEqual(len({item["token"] for item in seeds}), 16)
        self.assertEqual(
            len({item["numeric_seed_uint64"] for item in seeds}), 16
        )

    def test_new_cal_requires_four_of_four(self):
        rows = [
            {
                "block": block,
                "motion": motion,
                "sequence_pass": ordinal < 3,
            }
            for block in qms.BLOCKS
            for motion in qms.MOTIONS
            for ordinal in range(4)
        ]
        summaries, qualified = qms.summarize(rows, "new-cal")
        self.assertFalse(qualified)
        self.assertTrue(all(item["pass_count"] == 3 for item in summaries))

    def test_full_frame_metric_is_not_a_gate(self):
        self.assertNotIn(
            "descriptive_full_frame_gradient_density_ratio",
            (
                "material_residual_ratio",
                "structure_contrast_ratio",
            ),
        )

    def test_immediate_boundary_contrast_is_descriptive_not_a_gate(self):
        self.assertNotIn(
            "structure_contrast_ratio",
            ("material_residual_ratio", "geometry_identity"),
        )

    def test_firewall(self):
        source = Path(qms.__file__).read_text(encoding="utf-8")
        self.assertNotIn("rgb_algorithm_development_canary", source)
        self.assertNotIn("p3_transport", source)
        self.assertNotIn("p4_formal_runner", source)
