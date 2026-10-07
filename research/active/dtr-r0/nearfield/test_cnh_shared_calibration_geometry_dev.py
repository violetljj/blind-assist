"""Focused calibration membership checks; never generate the new batch."""
import copy
import unittest
import cnh_shared_calibration_geometry_dev as C


def row(role='clear',covered=True,valid=True):
    categories=[['clear','clear'] for _ in range(13)]
    if role!='clear':categories[5][0]='contact>5cm' if role=='contact' else 'pass0-10cm'
    return dict(valid=valid,evaluable=valid and role in ('contact','clear'),
        contact=(role=='contact') if valid else None,control=(role=='clear') if valid else None,
        covered=covered,frame_category=categories,reasons=[] if valid else ['target_background_overlap'])


class SharedCalibrationTest(unittest.TestCase):
    def test_units_disjoint_and_balanced(self):
        contract=C.unit_contract()
        self.assertTrue(all(not ids for ids in contract['overlaps'].values()))
        self.assertEqual(len(contract['mode_mirror_counts']),6)
        self.assertEqual(set(contract['mode_mirror_counts'].values()),{8})
        self.assertEqual(len(C.UNITS)*40,1920)
        self.assertEqual(C.M.UNITS,tuple(range(410004,410052)))

    def test_only_strict_controls_eligible(self):
        for role,covered,valid in [('contact',True,True),('pass',True,True),('pass',False,True),('clear',True,False)]:
            original=row(role,covered,valid);before=copy.deepcopy(original)
            value=C.calibration_geometry(original)
            self.assertFalse(value['eligible_control']);self.assertFalse(value['evaluable'])
            self.assertEqual(original,before)
            self.assertEqual(value['contact'],original['contact'])
        contact=C.calibration_geometry(row('contact'))
        self.assertTrue(contact['geometry_evaluable']);self.assertTrue(contact['contact'])

    def test_censored_clear_is_still_control_and_inconsistent_control_rejected(self):
        value=C.calibration_geometry(row('clear',False))
        self.assertTrue(value['eligible_control']);self.assertTrue(value['evaluable'])
        self.assertFalse(value['covered']);self.assertIsNone(value['calibration_exclusion'])
        bad=row('clear');bad['frame_category'][3][1]='pass0-10cm'
        with self.assertRaises(ValueError):C.calibration_geometry(bad)


if __name__=='__main__':unittest.main()
