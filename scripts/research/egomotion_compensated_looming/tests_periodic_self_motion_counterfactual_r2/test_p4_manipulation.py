from __future__ import annotations

import copy
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p4_manipulation_r0 as producer,
)


def _row(block: str, motion: str, ordinal: int, blur: bool, low: bool):
    return {
        "block": block,
        "motion": motion,
        "ordinal": ordinal,
        "blur_sequence_pass": blur,
        "low_texture_sequence_pass": low,
    }


class ManipulationMutationTests(unittest.TestCase):
    def test_subgroup_gate_requires_18_of_20_for_each_degradation(self):
        rows = []
        for block in producer.BLOCKS:
            for motion in producer.MOTIONS:
                for ordinal in range(20):
                    rows.append(
                        _row(block, motion, ordinal, ordinal < 18, ordinal >= 2)
                    )
        summary = producer._subgroups(rows)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(item["blur_pass_count"] == 18 for item in summary))
        self.assertTrue(
            all(item["low_texture_pass_count"] == 18 for item in summary)
        )
        self.assertTrue(all(item["blur_subgroup_pass"] for item in summary))
        self.assertTrue(
            all(item["low_texture_subgroup_pass"] for item in summary)
        )

    def test_subgroup_gate_mutation_17_of_20_fails(self):
        rows = []
        for block in producer.BLOCKS:
            for motion in producer.MOTIONS:
                for ordinal in range(20):
                    rows.append(_row(block, motion, ordinal, ordinal < 17, True))
        summary = producer._subgroups(rows)
        self.assertFalse(any(item["blur_subgroup_pass"] for item in summary))

    def test_subgroup_rejects_missing_sequence(self):
        rows = [
            _row(block, motion, ordinal, True, True)
            for block in producer.BLOCKS
            for motion in producer.MOTIONS
            for ordinal in range(20)
        ]
        mutated = copy.deepcopy(rows)
        mutated.pop()
        with self.assertRaisesRegex(
            producer.InvalidManipulation, "SUBGROUP_SEQUENCE_COUNT"
        ):
            producer._subgroups(mutated)

    def test_module_firewall_has_no_r3_or_p3_runtime_import(self):
        source = Path(producer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("rgb_algorithm_development_canary", source)
        self.assertNotIn("p3_runtime_preflight", source)
        self.assertNotIn("p3_transport_r0", source)
