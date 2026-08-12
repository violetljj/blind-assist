from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a_recovery as runner


class PoolPhaseARecoveryTests(unittest.TestCase):
    def test_actual_candidate_completion_is_adoptable(self) -> None:
        path = runner.REPO_ROOT / runner.CANDIDATE_ROOT / "candidate-completion.json"
        completion = runner.validate_candidate_completion(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(completion["frame_count"], 402)
        self.assertEqual(completion["faro_reads"], 0)

    def test_candidate_completion_mutation_fails_closed(self) -> None:
        path = runner.REPO_ROOT / runner.CANDIDATE_ROOT / "candidate-completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        changed = copy.deepcopy(value)
        changed["frame_count"] = 401
        with self.assertRaises(Exception):
            runner.validate_candidate_completion(changed)

    def test_recovery_authority_forbids_inference_and_truth(self) -> None:
        self.assertFalse(runner.EXPECTED_AUTHORITY["candidate_inference"])
        self.assertEqual(runner.EXPECTED_AUTHORITY["candidate_inference_count"], 0)
        self.assertFalse(runner.EXPECTED_AUTHORITY["faro_payload_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["truth_scoring"])


if __name__ == "__main__":
    unittest.main()
