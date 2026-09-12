import unittest
import numpy as np
import mz90_observation_source as source
from mz90_final_decision_audit import trace


class TraceChecks(unittest.TestCase):
    def raw(self):
        raw=source.materialize(source.build_source()[:1],'ideal')[0]
        raw={k:v[:7].copy() for k,v in raw.items()}
        raw['tof_status'][:]=255;raw['radar_valid'][:]=False
        raw['delta_yaw'][:]=0;raw['delta_pitch'][:]=0
        return raw

    def test_radar_activation_decay_and_nonrecursive_hold(self):
        raw=self.raw()
        for i in (0,1):
            raw['radar_valid'][i,0]=True;raw['radar_range_m'][i,0]=2
            raw['radar_velocity'][i,0]=-.7;raw['radar_angle'][i,0]=0
        rows=trace(raw)[2]
        self.assertFalse(rows[0]['final'])
        self.assertEqual(rows[1]['radar_activation_seed_frames'],[0,1])
        self.assertEqual(rows[2]['evidence_mode'],'hysteresis_without_current_return')
        self.assertEqual(rows[3]['evidence_mode'],'outer_one_frame_hold')
        self.assertEqual(rows[3]['origin_sensor'],'Radar')
        self.assertFalse(rows[4]['final'])

    def test_hold_sensor_separate_from_selected_sensor(self):
        raw=self.raw();raw['tof_status'][0,0]=5;raw['tof_range_m'][0,0]=2;raw['tof_theta_deg'][0,0]=0
        rows=trace(raw)[2]
        self.assertEqual(rows[1]['selected_sensor'],'Radar')
        self.assertEqual(rows[1]['origin_sensor'],'ToF')
        self.assertTrue(rows[1]['outer_hold']);self.assertFalse(rows[2]['final'])

    def test_prefix_and_episode_reset(self):
        raw=self.raw();raw['radar_valid'][:2,0]=True
        raw['radar_range_m'][:2,0]=2;raw['radar_velocity'][:2,0]=-.7;raw['radar_angle'][:2,0]=0
        rows=trace(raw)[2]
        self.assertEqual(rows[:3],trace({k:v[:3] for k,v in raw.items()})[2])
        raw['episode_id'][2:]='next'
        self.assertFalse(trace(raw)[2][2]['final'])


if __name__=='__main__':unittest.main()
