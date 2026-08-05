import unittest

from quality_gated_clearance_fusion_r0 import Evidence, Filter, State


class QualityGatedFusionTest(unittest.TestCase):
    def e(self, t, values=(2.0, 2.0, 2.0), valid=(True, True, True), tof=True, age=0.1, disagreement=0.05):
        return Evidence(t, values, valid, tof, age, disagreement)

    def test_clear_requires_two_confirmations(self):
        f = Filter()
        self.assertEqual(f.update(self.e(1_000_000_000)).state, (State.UNKNOWN,) * 3)
        self.assertEqual(f.update(self.e(1_100_000_000)).state, (State.CLEAR,) * 3)

    def test_occupied_is_immediate(self):
        f = Filter()
        self.assertEqual(f.update(self.e(1_000_000_000, (1.0, 1.0, 1.0))).state, (State.OCCUPIED,) * 3)

    def test_stale_or_disagreement_is_unknown(self):
        f = Filter()
        self.assertEqual(f.update(self.e(1_000_000_000, age=0.6)).state, (State.UNKNOWN,) * 3)
        self.assertEqual(f.update(self.e(1_100_000_000, disagreement=0.3)).state, (State.UNKNOWN,) * 3)

    def test_gap_resets_fail_closed(self):
        f = Filter()
        f.update(self.e(1_000_000_000))
        self.assertEqual(f.update(self.e(1_600_000_001)).state, (State.UNKNOWN,) * 3)

    def test_invalid_band_is_unknown(self):
        f = Filter()
        self.assertEqual(f.update(self.e(1_000_000_000, valid=(True, False, True))).state, (State.UNKNOWN,) * 3)


if __name__ == "__main__":
    unittest.main()
