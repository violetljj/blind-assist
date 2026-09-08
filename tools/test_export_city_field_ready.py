import copy
import unittest
from export_city_field_ready import reject_reasons


class ExportPolicyTest(unittest.TestCase):
    def setUp(self):
        self.frame=dict(rgb=dict(valid=True,status='PRESENT'),depth=dict(valid=True,status='PRESENT'),
            source_load_status='READY',synchronization=dict(status='PAIRED'),pose_evidence=dict(status='MEASURED'),
            floor_probe=dict(status='CONSISTENT'),labels=dict(near=[0,1],support_unknown_cells=[100,100]),
            targets=[dict(label_status='NOT_PRESENT')])

    def test_sky_unknown_pixels_do_not_drop_valid_frame(self):
        self.assertEqual(reject_reasons(self.frame),[])

    def test_bad_active_target_dropped_not_negative(self):
        self.frame['targets']=[dict(label_status='UNKNOWN')]
        self.assertEqual(reject_reasons(self.frame),['ACTIVE_TARGET_LABEL_UNKNOWN'])

    def test_loading_floor_and_pair_errors_preserved(self):
        self.frame.update(source_load_status='UNKNOWN',floor_probe=dict(status='REVIEW'),synchronization=dict(status='MISSING'))
        self.assertEqual(set(reject_reasons(self.frame)),{'LOAD_NOT_READY','FLOOR_REVIEW','UNPAIRED'})

    def test_input_not_mutated(self):
        before=copy.deepcopy(self.frame);reject_reasons(self.frame);self.assertEqual(before,self.frame)

    def test_missing_or_unknown_near_is_not_a_negative(self):
        for near in ([],[-1,0]):
            self.frame['labels']['near']=near
            self.assertIn('NEAR_LABEL_UNKNOWN',reject_reasons(self.frame))


if __name__=='__main__':unittest.main()
