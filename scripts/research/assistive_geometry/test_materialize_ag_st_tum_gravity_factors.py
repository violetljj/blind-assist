import unittest

import numpy as np

from materialize_ag_st_tum_gravity_factors import unknown_geometric_factors


class TumGravityFactorsTest(unittest.TestCase):
    def test_unknown_parent_is_fail_closed(self) -> None:
        factors = unknown_geometric_factors((3, 4))
        self.assertFalse(factors["normal_valid_hw"].any())
        self.assertFalse(factors["support_truth_valid_hw"].any())
        self.assertFalse(factors["evidence_truth_valid_hw"].any())
        self.assertTrue(np.isnan(factors["boundary_distance_px_hw"]).all())
        self.assertTrue(np.isnan(factors["height_above_support_m_hw"]).all())
        self.assertFalse(bool(factors["support_plane_valid"]))


if __name__ == "__main__":
    unittest.main()
