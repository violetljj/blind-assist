import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    route_target,
    stable_fold,
)


class ThorMagniLocalRouteSupervisionTest(unittest.TestCase):
    def test_future_route_intrusion_is_local_and_causal(self):
        times = np.arange(0.0, 3.0, 0.01)
        camera = np.stack(
            (times, np.zeros_like(times), np.zeros_like(times)),
            axis=1,
        )
        crossing = np.stack(
            (
                np.full_like(times, 1.5),
                2.0 - times,
                np.zeros_like(times),
            ),
            axis=1,
        )

        target = route_target(
            times,
            camera,
            {"person": crossing},
            {"person": "Visitor"},
            50,
        )

        self.assertIsNotNone(target)
        self.assertTrue(target["future_corridor_intrusion"])
        self.assertTrue(target["future_proximity_le_1_25m"])
        self.assertGreater(target["occupancy_positive_cells"], 0)
        self.assertEqual(target["closest"]["body"], "person")

    def test_lateral_noninteraction_stays_outside_corridor(self):
        times = np.arange(0.0, 3.0, 0.01)
        camera = np.stack(
            (times, np.zeros_like(times), np.zeros_like(times)),
            axis=1,
        )
        lateral = np.stack(
            (
                np.full_like(times, 1.5),
                np.full_like(times, 3.0),
                np.zeros_like(times),
            ),
            axis=1,
        )

        target = route_target(
            times,
            camera,
            {"person": lateral},
            {"person": "Visitor"},
            50,
        )

        self.assertIsNotNone(target)
        self.assertFalse(target["future_corridor_intrusion"])
        self.assertFalse(target["future_proximity_le_1_25m"])

    def test_source_fold_is_stable(self):
        self.assertEqual(stable_fold("session-a"), stable_fold("session-a"))
        self.assertIn(stable_fold("session-a"), range(5))


if __name__ == "__main__":
    unittest.main()
