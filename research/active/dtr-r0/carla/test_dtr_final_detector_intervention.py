import copy
import unittest
import dtr_final_detector_intervention as intervention


class DetectorInterventionTest(unittest.TestCase):
    def fixture(self):
        candidates=[{'candidates':[{'class_name':'person'}]} for _ in range(40)]
        frames=[{'sample_index':i,'tracks':[{'track_id':'p','disposition':'MEASURED'}],
                 'arms':{intervention.x24.ARM_X24:{'route_risk':True,'confirmed_risk_track_ids':['p']}}} for i in range(40)]
        return candidates,{'frames':frames}

    def test_fixed_transform_preserves_raw_and_nonremoved_records(self):
        raw,credential=self.fixture(); original=copy.deepcopy(raw)
        changed,receipt=intervention.intervene_episode(raw,credential,'S03_MULTI',[[12,13],[17,18,19],list(range(23,29))])
        self.assertEqual(original,raw)
        self.assertEqual(11,receipt['removed_candidates'])
        for i in range(40):self.assertEqual(changed[i]['candidates'],[] if i in receipt['removed_frames'] else raw[i]['candidates'])

    def test_held_or_unconfirmed_credential_fails_without_relocation(self):
        raw,credential=self.fixture()
        credential['frames'][11]['tracks'][0]['disposition']='HOLD'
        with self.assertRaisesRegex(ValueError,'NO_INDEX_RESCUE'):
            intervention.intervene_episode(raw,credential,'S02_SINGLE',[[12]])

    def test_truth_is_rejected(self):
        raw,credential=self.fixture();raw[0]['truth']={}
        with self.assertRaises((ValueError,RuntimeError)):
            intervention.intervene_episode(raw,credential,'S02_SINGLE',[[12]])


if __name__=='__main__':unittest.main()
