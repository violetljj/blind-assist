import unittest
import numpy as np
from cnh_street_e2e_materialize import triangle_box_hits, physical_labels, frame_identity, is_development, authored_layout_metadata


class MaterializeTests(unittest.TestCase):
    def test_alley_inherits_manifest_partition_without_mutating_spec(self):
        spec=dict(scope='ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK',
            alley_manifest=dict(physical_site_id='site',proposed_split='train'),
            layouts=[dict(layout_id='layout',physical_site_id='site')])
        self.assertEqual(authored_layout_metadata(spec)['layout']['proposed_split'],'train')
        self.assertNotIn('proposed_split',spec['layouts'][0])
        spec['layouts'][0]['split']='dev'
        with self.assertRaisesRegex(ValueError,'conflicts'):authored_layout_metadata(spec)
        del spec['layouts'][0]['split'];spec['layouts'][0]['physical_site_id']='other'
        with self.assertRaisesRegex(ValueError,'physical site'):authored_layout_metadata(spec)

    def test_materializer_accepts_both_development_sources_only(self):
        for scope in ('STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK','ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK'):
            self.assertTrue(is_development({'scope':scope}))
        self.assertFalse(is_development({'scope':'TWO_LAYOUT_SOURCE_ENGINEERING_NOT_BENCHMARK'}))
        self.assertFalse(is_development({'scope':'ALLEY_FINAL_TEST'}))

    def test_triangle_crossing_without_inside_vertices(self):
        tri=np.array([[[-2.,0.,0.],[2.,0.,0.],[0.,2.,0.]]])
        self.assertTrue(triangle_box_hits(tri,[-.1,-.1,-.1],[.1,.1,.1]))
        self.assertFalse(triangle_box_hits(tri,[1.5,1.5,-.1],[2.,2.,.1]))

    def test_unknown_never_negative_and_positive_survives(self):
        empty=np.empty((0,3,3)); tri=np.array([[[-.1,0.,1.],[.1,0.,1.],[0.,.1,1.]]])
        labels,_=physical_labels(empty,empty,np.eye(4),False,np.zeros(3),8)
        self.assertTrue((labels==-1).all())
        labels,_=physical_labels(empty,tri,np.eye(4),False,np.zeros(3),8)
        self.assertEqual(labels[2],1);self.assertEqual(labels[3],-1)

    def test_missing_bounds_and_probe_radius_block_negative(self):
        empty=np.empty((0,3,3))
        labels,_=physical_labels(empty,empty,np.eye(4),True,np.zeros(3),8)
        self.assertTrue((labels==0).all())
        labels,_=physical_labels(empty,empty,np.eye(4),True,np.zeros(3),1)
        self.assertTrue((labels==-1).all())
        labels,_=physical_labels(empty,empty,np.eye(4),True,np.zeros(3),8,[([-.1,0,.5],[.1,.1,1])])
        self.assertEqual(labels[2],-1);self.assertEqual(labels[3],0)

    def test_seed_content_identity_not_clip_label(self):
        a=frame_identity('manifest',dict(id='a',clip_id='centre'),'camera','depth')
        b=frame_identity('manifest',dict(id='a',clip_id='removed'),'camera','depth')
        self.assertEqual(a,b)
        self.assertNotEqual(a,frame_identity('manifest',dict(id='b'),'camera','depth'))


if __name__=='__main__':unittest.main()
