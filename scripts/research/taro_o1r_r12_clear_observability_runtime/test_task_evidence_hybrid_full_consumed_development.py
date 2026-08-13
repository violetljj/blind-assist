import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_development as r17
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_full_consumed_development as subject


class FullConsumedHybridDevelopmentTest(unittest.TestCase):
    def test_candidate_family_identity_is_stable(self) -> None:
        first = subject.shared.canonical_json_bytes(r17.candidate_policy_specs())
        second = subject.shared.canonical_json_bytes(r17.candidate_policy_specs())
        self.assertEqual(first, second)
        self.assertEqual(96, len(r17.candidate_policy_specs()))


if __name__ == "__main__":
    unittest.main()
