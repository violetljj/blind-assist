from __future__ import annotations

import copy
import unittest

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    enumerate_trials,
    load_protocol,
    validate_protocol,
)


class PhaseAProtocolTest(unittest.TestCase):
    def test_locked_protocol_and_trial_inventory(self) -> None:
        protocol = load_protocol()
        trials = enumerate_trials(protocol)
        self.assertEqual(
            PROTOCOL_SHA256,
            "d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502",
        )
        self.assertEqual(len(trials), 2520)
        self.assertEqual(len({trial.trial_id for trial in trials}), 2520)
        self.assertEqual(sum(trial.split == "clean" for trial in trials), 1680)
        self.assertEqual(sum(trial.split == "stress" for trial in trials), 840)

    def test_duplicate_seed_is_rejected(self) -> None:
        protocol = load_protocol()
        mutated = copy.deepcopy(protocol)
        mutated["trials"]["seeds"][-1] = mutated["trials"]["seeds"][0]
        with self.assertRaisesRegex(ValueError, "SEED_INVENTORY_DRIFT"):
            validate_protocol(mutated)

    def test_ratio_authority_drift_is_rejected(self) -> None:
        protocol = load_protocol()
        mutated = copy.deepcopy(protocol)
        mutated["metrics"]["ratios_are_diagnostic_only"] = False
        with self.assertRaisesRegex(ValueError, "RATIO_AUTHORITY_DRIFT"):
            validate_protocol(mutated)


if __name__ == "__main__":
    unittest.main()
