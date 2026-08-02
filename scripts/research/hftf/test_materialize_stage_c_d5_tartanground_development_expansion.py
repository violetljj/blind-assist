import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d5_tartanground_development_expansion import (
    DIAGNOSTIC_PARENT,
    GENERALIZATION_ENVIRONMENTS,
    SOURCES,
    metadata_remote,
    selection_rank,
)


class TartanGroundDevelopmentExpansionTest(unittest.TestCase):
    def test_roster_has_one_diagnostic_and_six_unique_environments(self):
        self.assertEqual(DIAGNOSTIC_PARENT, "WaterMillDay/Data_diff/P1002")
        self.assertEqual(len(GENERALIZATION_ENVIRONMENTS), 6)
        self.assertEqual(len(set(GENERALIZATION_ENVIRONMENTS)), 6)
        self.assertEqual(len(SOURCES), 7)

    def test_generalization_roster_is_in_hash_rank_order(self):
        ranks = [
            selection_rank(environment)
            for environment in GENERALIZATION_ENVIRONMENTS
        ]
        self.assertEqual(ranks, sorted(ranks))

    def test_metadata_remote_is_revision_pinned(self):
        remote = metadata_remote(DIAGNOSTIC_PARENT)
        self.assertIn("@388faf9c800568cfc6828fa47e063f8369397eb3/", remote)
        self.assertTrue(remote.endswith("/metadata.zip"))


if __name__ == "__main__":
    unittest.main()
