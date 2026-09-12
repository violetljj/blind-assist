import unittest

from mz85_residual_fn_audit import classify_primary


class ResidualAuditTests(unittest.TestCase):
    def test_lateral_cold_start(self):
        self.assertEqual(classify_primary(family='lateral_crossing', frame_index=0,
                                         tof_known=True, radar_raw=False,
                                         radar_alert=False),
                         'LATERAL_HISTORY_COLD_START')

    def test_available_radar_evidence_not_activated(self):
        self.assertEqual(classify_primary(family='wall_multipath', frame_index=7,
                                         tof_known=False, radar_raw=True,
                                         radar_alert=False),
                         'AVAILABLE_RADAR_EVIDENCE_NOT_ACTIVATED')

    def test_dual_observation_absence(self):
        self.assertEqual(classify_primary(family='multi_target', frame_index=9,
                                         tof_known=False, radar_raw=False,
                                         radar_alert=False),
                         'DUAL_INSTANTANEOUS_OBSERVATION_ABSENCE')


if __name__ == '__main__':
    unittest.main()
