import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_multichannel_risk_profile_probe as probe


class MultichannelRiskProfileProbeTest(unittest.TestCase):
    def scores(self):
        background = {key: index / 10 for index, key in enumerate(probe.BACKGROUND_FIELDS, 1)}
        background.update({"transition_count": 3, "reliable_transition_count": 2})
        clearance = {key: index / 20 for index, key in enumerate(probe.CLEARANCE_FIELDS + probe.PATH_FIELDS + probe.DETOUR_FIELDS, 1)}
        return background, clearance

    def test_compact_profile_has_fixed_contract_order(self):
        background, clearance = self.scores()
        result = probe.compact_profile(background, clearance)
        self.assertEqual((16,), result.shape)
        self.assertAlmostEqual(background[probe.BACKGROUND_FIELDS[0]], result[0])
        self.assertAlmostEqual(clearance[probe.DETOUR_FIELDS[-1]], result[-1])

    def test_unreliable_background_values_are_zero_not_nan(self):
        background, clearance = self.scores()
        background[probe.BACKGROUND_FIELDS[1]] = None
        result = probe.compact_profile(background, clearance)
        self.assertEqual(0.0, result[1])
        self.assertTrue(np.isfinite(result).all())

    def test_matrix_hash_is_repeatable_and_order_sensitive(self):
        values = np.arange(32, dtype=np.float64).reshape(2, 16)
        self.assertEqual(probe.matrix_sha256(values), probe.matrix_sha256(values.copy()))
        self.assertNotEqual(probe.matrix_sha256(values), probe.matrix_sha256(values[::-1]))

    def test_directional_projection_orders_positive_delta(self):
        features = np.asarray([[0.0], [2.0], [0.2], [2.2]], dtype=np.float64)
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        episodes = [
            {"counterfactual_pair_id": "a"}, {"counterfactual_pair_id": "a"},
            {"counterfactual_pair_id": "b"}, {"counterfactual_pair_id": "b"},
        ]
        result = probe.pair_directional_projection(features, episodes, labels, ridge=1.0)
        self.assertTrue(result["all_pairs_ordered"])
        self.assertEqual(2, result["ordered_pair_count"])


if __name__ == "__main__":
    unittest.main()
