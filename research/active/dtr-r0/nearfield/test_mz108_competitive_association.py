"""Bounded observable-only checks for reciprocal association and arm parity."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import mz107_rgb_association as previous
import mz108_competitive_association as current
from test_mz107_rgb_association import fixture


def without_competitor_metadata(result):
    result = copy.deepcopy(result)
    for association in result['associations']:
        association.pop('competing_returns', None)
    return result


def two_regions():
    row, image = fixture()
    angle = math.degrees(math.atan((560-row['rgb_intrinsics']['cx'])/row['rgb_intrinsics']['fx']))
    direction = current.ray(angle, 0., row['camera_pitch_deg'], 0.)
    row['tof64_range_m'].append(2.5/math.hypot(direction[0], direction[1]))
    row['tof64_status'].append(5)
    row['tof64_theta_deg'].append(angle)
    row['tof64_phi_deg'].append(0.)
    return row, image, [[400,110,420,230], [550,110,570,230]]


class ReciprocalAssociation(unittest.TestCase):
    def test_original_arm_exact_parity(self):
        row, image = fixture()
        cases = [(row, image), (row, None)]
        conflict = copy.deepcopy(row); conflict['tof64_range_m'] = [3.8]
        cases.append((conflict, image))
        missing = copy.deepcopy(row); missing['tof_packet_received'] = False
        cases.append((missing, image))
        duplicate = copy.deepcopy(row)
        duplicate.update(radar_range_m=[2.5,2.5], radar_angle=[0.,0.], radar_valid=[True,True])
        cases.append((duplicate, image))
        for observation, rgb in cases:
            for yaw in (-12., 0., 12.):
                with self.subTest(yaw=yaw, missing=rgb is None):
                    self.assertEqual(previous.predict_frame(observation,rgb,yaw),
                        without_competitor_metadata(current.predict_frame(observation,rgb,yaw,
                            use_regions=False,competitive=False)))

    def test_all_returns_compete_for_one_region(self):
        row, image = fixture()
        row.update(radar_range_m=[2.5,2.5],radar_angle=[0.,0.],radar_valid=[True,True])
        result = current.predict_frame(row,image,0.,use_regions=False)
        self.assertTrue(result['candidate'])
        self.assertEqual([a['state'] for a in result['associations']], ['UNKNOWN','UNKNOWN'])
        self.assertEqual([a['competing_returns'] for a in result['associations']], [2,2])

    def test_multi_region_contender_still_blocks_unique_contender(self):
        row, image, boxes = two_regions()
        row.update(radar_range_m=[2.5,2.5],radar_angle=[0.,18.],radar_valid=[True,True])
        with patch.object(current,'proposals',return_value=boxes):
            result = current.predict_frame(row,image,0.)
            self.assertEqual(result['associations'][0]['eligible'], 1)
            self.assertEqual(result['associations'][0]['competing_returns'], 2)
            self.assertEqual(result['associations'][1]['eligible'], 2)
            self.assertTrue(all(a['state']=='UNKNOWN' for a in result['associations']))
            row['radar_angle'].reverse()
            reversed_result = current.predict_frame(row,image,0.)
            self.assertEqual(result['candidate'],reversed_result['candidate'])
            self.assertEqual(result['associations'][0]['competing_returns'],
                reversed_result['associations'][1]['competing_returns'])

    def test_distinct_regions_and_invalid_returns_do_not_compete(self):
        row, image, boxes = two_regions()
        row.update(radar_range_m=[2.5,2.5,2.5],radar_angle=[0.,35.,0.],radar_valid=[True,True,False])
        with patch.object(current,'proposals',return_value=boxes):
            result = current.predict_frame(row,image,0.)
        self.assertEqual(len(result['associations']),2)
        self.assertTrue(all(a['state']=='ASSOCIATED' and a['competing_returns']==1
                            for a in result['associations']))

    def test_tof_preserved_and_rgb_disabled_in_all_arms(self):
        row, image = fixture()
        for key, value in [('tof64_range_m',2.),('tof64_status',5),
                           ('tof64_theta_deg',0.),('tof64_phi_deg',0.)]:
            row[key].append(value)
        for regions in (False,True):
            for competitive in (False,True):
                on = current.predict_frame(row,image,0.,regions,competitive)
                off = current.predict_frame(row,None,0.,regions,competitive)
                self.assertTrue(on['tof_support'] and on['candidate'])
                self.assertEqual(off['candidate'],on['baseline'])
                self.assertEqual(off['candidate'],off['baseline'])

    def test_mser_duplicates_and_contained_boxes_are_removed(self):
        class Regions:
            def detectRegions(self, gray):
                return [], np.array([[400,100,40,140],[405,105,30,130],
                                     [100,100,20,20],[430,100,20,140]])
        with patch.object(current,'otsu_proposals',return_value=[[400,100,440,240]]), \
             patch.object(current.cv2,'MSER_create',return_value=Regions()):
            result = current.proposals(np.zeros((360,640,3),np.uint8))
        self.assertEqual(result,[[400,100,440,240],[430,100,450,240],[100,100,120,120]])


if __name__ == '__main__':
    unittest.main()
