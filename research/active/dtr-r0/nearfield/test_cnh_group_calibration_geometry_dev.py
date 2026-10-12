"""Small eligibility/preservation tests; no full geometry or source generation."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import cnh_group_calibration_geometry_dev as G


def sample():
    travel=np.tile(np.eye(4),(16,1,1));travel[:,2,3]=np.arange(16)*.08
    boxes=[dict(lo=[1.2,-.1,5],hi=[1.4,.26,5.2],rho=.5)]
    physical=G.C.calibration_geometry(G.M.evaluate_geometry(boxes,travel))
    original=dict(anchor_id='u410052_c0',row_id='u410052_c0_original',tag='original',
                  unit=410052,config=0,photon_seed=123,family='none',**physical)
    return original,travel


class GroupCalibrationTest(unittest.TestCase):
    def test_original_unmodified_and_only_hb_height_changes(self):
        original,travel=sample();before=copy.deepcopy(original)
        hb=G.hb_record(original,travel)
        self.assertEqual(original,before)
        self.assertEqual(hb['tag'],'HB');self.assertEqual(hb['row_id'],'u410052_c0_HB')
        self.assertEqual(hb['photon_seed'],original['photon_seed'])
        restored=copy.deepcopy(hb['boxes']);restored[0]['lo'][1]=original['boxes'][0]['lo'][1];restored[0]['hi'][1]=original['boxes'][0]['hi'][1]
        self.assertEqual(restored,original['boxes'])
        self.assertTrue(hb['eligible_control']);self.assertIsNone(hb['deadline_index'])

    def test_hb_control_does_not_require_original_eligibility(self):
        original,travel=sample()
        # Eligibility is not an input to the physical HB evaluator.
        original['eligible_control']=False;original['evaluable']=False
        hb=G.hb_record(original,travel)
        self.assertTrue(hb['eligible_control']);self.assertTrue(hb['evaluable'])
        self.assertFalse(original['eligible_control'])

    def test_hb_overlap_is_invalid_not_negative(self):
        original,travel=sample()
        boxes=copy.deepcopy(original['boxes'])
        boxes.append(dict(lo=[1.25,.5,5.05],hi=[1.35,.7,5.15],rho=.2))
        original.update(G.C.calibration_geometry(G.M.evaluate_geometry(boxes,travel)))
        self.assertTrue(original['eligible_control'])
        hb=G.hb_record(original,travel)
        self.assertFalse(hb['valid']);self.assertFalse(hb['eligible_control'])
        self.assertIsNone(hb['contact']);self.assertIsNone(hb['control'])

    def test_unit_isolation_and_pose_loader_reuse(self):
        self.assertEqual(G.UNITS,tuple(range(410052,410100)))
        self.assertTrue(all(not value for value in G.C.unit_contract()['overlaps'].values()))
        with patch.object(G.C,'load_source',return_value={'marker':1}) as load:
            self.assertEqual(G.load_source(410052,3,G.OUT),{'marker':1})
            load.assert_called_once_with(410052,3,G.SOURCE_OUT)


if __name__=='__main__':unittest.main()
