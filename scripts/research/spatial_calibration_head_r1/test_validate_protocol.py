#!/usr/bin/env python3

import copy
import json
import unittest

from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, validate


class SpatialCalibrationHeadR1ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

    def test_frozen_protocol_is_valid(self) -> None:
        self.assertEqual(validate(self.protocol, REPO_ROOT), [])

    def test_consumed_tum_reuse_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["authority"]["historical_tum_training_or_selection_for_r1"] = True
        self.assertIn("consumed TUM must be excluded", validate(mutated, REPO_ROOT))

    def test_parameter_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["student"]["layers"] = [781, 32, 3]
        mutated["student"]["trainable_parameters"] = 25123
        self.assertIn("student layers changed", validate(mutated, REPO_ROOT))
        self.assertIn("parameter count must be 9423", validate(mutated, REPO_ROOT))

    def test_sealed_access_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["authority"]["sealed_media_access_before_activation_receipt"] = True
        self.assertIn("sealed media must remain closed", validate(mutated, REPO_ROOT))

    def test_gate_rescue_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["evaluation"]["task_gates"]["false_clear_rate_max"] = 0.10
        self.assertIn("five task gates changed", validate(mutated, REPO_ROOT))

    def test_official_split_overlap_exclusion_is_required(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["data"]["cohort"]["metadata_exclusions"] = "exclude visit_id NA"
        self.assertIn("metadata exclusions changed", validate(mutated, REPO_ROOT))


if __name__ == "__main__":
    unittest.main()
