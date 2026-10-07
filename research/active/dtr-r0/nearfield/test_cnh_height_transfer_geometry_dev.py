"""Small physical geometry tests, without reading full anchor/data manifests."""
import copy
import unittest
import numpy as np
import cnh_height_transfer_geometry_dev as H


def source():
    scene=dict(boxes=[dict(lo=[-.1,-.1,1.8],hi=[.1,.26,2.0],rho=.73)])
    travel=np.tile(np.eye(4),(16,1,1));travel[:,2,3]=np.arange(16)*.08
    hb=copy.deepcopy(scene['boxes'][0]);hb['lo'][1]=-.1;hb['hi'][1]=.84
    _,ref=H.G.deadline_reference(H.G.corners(hb),travel[H.G.FRAMES])
    ref['deadline_index']=int(np.searchsorted(H.G.FRAMES,ref['reference_fraction']+1e-10,side='right')-1)
    return scene,travel,ref


class HeightTransferTest(unittest.TestCase):
    def test_only_target_y_changes_and_reference_consistent(self):
        scene,travel,ref=source();before=copy.deepcopy(scene)
        scene['boxes'].append(dict(lo=[2,-.5,1],hi=[3,1.5,3],rho=.2));before=copy.deepcopy(scene)
        for tag,bounds in H.HEIGHTS.items():
            v=H.validate_variant(scene,travel,'contact',tag,ref)
            self.assertTrue(v['valid'],v['reasons'])
            self.assertEqual(v['contact_query'],[True,True])
            self.assertEqual(v['boxes'][0]['rho'],.73)
            self.assertEqual(v['boxes'][1],before['boxes'][1])
            self.assertEqual([v['boxes'][0]['lo'][1],v['boxes'][0]['hi'][1]],list(bounds))
            for endpoint in ('lo','hi'):
                self.assertEqual([v['boxes'][0][endpoint][k] for k in (0,2)],
                                 [before['boxes'][0][endpoint][k] for k in (0,2)])
            self.assertTrue(v['original_reference_consistent'])
            self.assertEqual(v['deadline_index'],ref['deadline_index'])
        self.assertEqual(scene,before)

    def test_background_overlap_invalid_is_not_negative(self):
        scene,travel,ref=source()
        scene['boxes'].append(dict(lo=[-.05,.95,1.85],hi=[.05,1.2,1.95],rho=.2))
        short=H.validate_variant(scene,travel,'contact','short',ref)
        tall=H.validate_variant(scene,travel,'contact','tall',ref)
        self.assertTrue(short['valid'])
        self.assertFalse(tall['valid'])
        self.assertIn('target_background_overlap',tall['reasons'])
        self.assertEqual(tall['changed_target_background_overlaps'],[1])
        self.assertIsNone(tall['contact']);self.assertIsNone(tall['control'])
        self.assertTrue(tall['physical_contact'])

    def test_invalid_role_and_reference_change(self):
        scene,travel,ref=source()
        wrong_role=H.validate_variant(scene,travel,'clear','short',ref)
        self.assertFalse(wrong_role['valid']);self.assertIsNone(wrong_role['control'])
        changed=copy.deepcopy(ref);changed['reference_fraction']+=.1
        altered=H.validate_variant(scene,travel,'contact','short',changed)
        self.assertIn('original_reference_changed',altered['reasons'])
        self.assertIsNone(altered['contact'])
        with self.assertRaises(ValueError):H.validate_variant(scene,travel,'unknown','short',ref)

    def test_unchanged_censored_clear_valid_but_changed_reference_invalid(self):
        scene,travel,_=source()
        scene['boxes'][0]['lo']=[1.2,-.1,5.0]
        scene['boxes'][0]['hi']=[1.4,.26,5.2]
        hb=copy.deepcopy(scene['boxes'][0]);hb['hi'][1]=.84
        ranges,ref=H.G.deadline_reference(H.G.corners(hb),travel[H.G.FRAMES])
        self.assertFalse(ref['covered']);self.assertEqual(ref['censor_reason'],'right_censored')
        ref.update(deadline_index=None,frame_ranges=ranges)
        value=H.validate_variant(scene,travel,'clear','short',ref)
        self.assertTrue(value['valid'],value['reasons'])
        self.assertTrue(value['control']);self.assertFalse(value['contact'])
        self.assertIsNone(value['deadline_index'])
        self.assertEqual(value['clear_all'],[True,True])
        self.assertTrue(value['original_reference_consistent'])
        changed=copy.deepcopy(ref);changed['censor_reason']='left_censored'
        invalid=H.validate_variant(scene,travel,'clear','short',changed)
        self.assertFalse(invalid['valid'])
        self.assertIn('original_reference_changed',invalid['reasons'])
        self.assertIsNone(invalid['control'])


if __name__=='__main__':unittest.main()
