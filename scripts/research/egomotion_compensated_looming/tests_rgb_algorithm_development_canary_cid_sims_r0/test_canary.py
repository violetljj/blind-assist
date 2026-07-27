from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0.producer import (
    _longest_trigger_run,
)
from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0.validator import (
    longest_trigger_run,
)


class CanaryAggregationTest(unittest.TestCase):
    def test_trigger_run_breaks_on_abstention(self) -> None:
        rows = [
            {
                "evaluable": True,
                "trigger": True,
                "previous_timestamp_s": 0.0,
                "current_timestamp_s": 0.1,
            },
            {
                "evaluable": True,
                "trigger": True,
                "previous_timestamp_s": 0.1,
                "current_timestamp_s": 0.2,
            },
            {
                "evaluable": False,
                "trigger": False,
                "previous_timestamp_s": 0.2,
                "current_timestamp_s": 0.3,
            },
            {
                "evaluable": True,
                "trigger": True,
                "previous_timestamp_s": 0.3,
                "current_timestamp_s": 0.4,
            },
        ]
        self.assertEqual(_longest_trigger_run(rows), (2, 0.2))
        self.assertEqual(longest_trigger_run(rows), (2, 0.2))

    def test_no_trigger(self) -> None:
        rows = [
            {
                "evaluable": True,
                "trigger": False,
                "previous_timestamp_s": 0.0,
                "current_timestamp_s": 0.1,
            }
        ]
        self.assertEqual(_longest_trigger_run(rows), (0, 0.0))
        self.assertEqual(longest_trigger_run(rows), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
