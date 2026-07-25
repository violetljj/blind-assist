#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_r1a_oracle_signal_contract_r0 as subject  # noqa: E402


class R1AOracleSignalContractTest(unittest.TestCase):
    def test_frozen_contract_validates(self) -> None:
        subject.validate(subject.build())

    def test_missing_arm_is_rejected(self) -> None:
        receipt = copy.deepcopy(subject.build())
        receipt["arms"].pop()
        with self.assertRaisesRegex(ValueError, "arm"):
            subject.validate(receipt)

    def test_old_window_read_is_rejected(self) -> None:
        receipt = copy.deepcopy(subject.build())
        receipt["hard_boundaries"][
            "old_window_selection_tuning_acceptance_reads"
        ] = 1
        with self.assertRaisesRegex(ValueError, "old-window"):
            subject.validate(receipt)

    def test_full_6dof_cannot_gain_acceptance_authority(self) -> None:
        receipt = copy.deepcopy(subject.build())
        receipt["arms"][-1]["acceptance_authority"] = True
        with self.assertRaisesRegex(ValueError, "6DoF"):
            subject.validate(receipt)


if __name__ == "__main__":
    unittest.main()
