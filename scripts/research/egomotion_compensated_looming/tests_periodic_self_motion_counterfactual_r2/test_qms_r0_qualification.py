from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    qms_r0_qualification as qms,
)


class QmsQualificationTests(unittest.TestCase):
    def test_seed_domains_are_deterministic_and_unique(self):
        values = [
            qms.derive_qms_seed("CAL", block, ordinal)
            for block in qms.BLOCKS
            for ordinal in range(4)
        ]
        self.assertEqual(len({item["token"] for item in values}), 16)
        self.assertEqual(
            len({item["numeric_seed_uint64"] for item in values}), 16
        )

    def test_material_interior_excludes_boundaries(self):
        object_id = np.zeros((80, 80), dtype=np.int32)
        object_id[:, 40:] = 1
        interior = qms.material_interior_mask(
            object_id, np.ones((80, 80), dtype=bool)
        )
        self.assertFalse(np.any(interior[:, 37:44]))
        self.assertTrue(np.any(interior[:, :10]))
        self.assertTrue(np.any(interior[:, 70:]))

    def test_new_cal_requires_all_four_sequences(self):
        rows = []
        for block in qms.BLOCKS:
            for motion in qms.MOTIONS:
                for ordinal in range(4):
                    rows.append(
                        {
                            "block": block,
                            "motion": motion,
                            "sequence_pass": ordinal != 3,
                        }
                    )
        summaries, qualified = qms.summarize(rows, "new-cal")
        self.assertFalse(qualified)
        self.assertTrue(all(item["pass_count"] == 3 for item in summaries))

    def test_source_firewall(self):
        source = Path(qms.__file__).read_text(encoding="utf-8")
        self.assertNotIn("rgb_algorithm_development_canary", source)
        self.assertNotIn("p3_transport", source)
        self.assertNotIn("p4_formal_runner", source)
