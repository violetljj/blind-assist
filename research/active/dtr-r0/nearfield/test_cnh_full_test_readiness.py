"""Quota-only plan checks; no UE or protected payload access."""
import unittest
from collections import Counter

from cnh_full_test_readiness import ENVIRONMENTS, FAMILIES_PER_ENVIRONMENT, slots


class FrozenTestScheduleChecks(unittest.TestCase):
    def test_exact_outdoor_quota_and_unassigned_sources(self):
        rows = slots()
        self.assertEqual(rows, slots())
        self.assertEqual(len(rows), 164)
        self.assertEqual(len({row['slot_id'] for row in rows}), 164)
        for environment in ENVIRONMENTS:
            per_env = [row for row in rows if row['environment'] == environment]
            self.assertEqual(len(per_env), 41)
            self.assertEqual(Counter(row['family'] for row in per_env), FAMILIES_PER_ENVIRONMENT)
        self.assertTrue(all(row['nominal_samples'] == 160 and row['capture_spec'] is None
                            and row['physical_site_id'] is None and row['target_source_family'] is None
                            for row in rows))


if __name__ == '__main__':
    unittest.main()
