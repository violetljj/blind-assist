import unittest
from join_dtr_final_roster_source import inapplicable_occlusion

class BridgeTest(unittest.TestCase):
    def test_does_not_claim_occlusion_pass_or_read_collision_truth(self):
        class Protected(dict):
            def __getitem__(self,key):raise AssertionError('truth accessed')
        result=inapplicable_occlusion({}, {'contract_id':'partial','episodes':['ep_09']}, {'ep_09':Protected()})
        self.assertIs(result['passed'],False)
        self.assertEqual(result['selected_indices'],{'ep_09':[]})
        self.assertTrue(result['status'].startswith('NOT_APPLICABLE'))

    def test_missing_episode_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing_occlusion_episode'):
            inapplicable_occlusion({}, {'contract_id':'partial','episodes':['ep_09']},{})

if __name__=='__main__':unittest.main()
