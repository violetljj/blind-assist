import unittest
from collections import Counter
from body_query_capture_spec import generate as fixtures
from body_query_collection_spec import generate


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=fixtures({},'train')

    def test_complete_balanced_groups_and_fixed_calibration(self):
        pose=dict(x=123.,y=-45.,yaw=90.,floor_z_m=1.2,region_id='new_region')
        spec=generate(self.fixtures,{},pose,'eval')
        self.assertEqual(len(spec['cases']),40)
        groups={}
        for c in spec['cases']:
            groups.setdefault(c['group_id'],[]).append(c)
            self.assertEqual(c['source_role'],'EVAL_ONLY')
            self.assertAlmostEqual(c['camera']['z']-c['floor_z_m'],1.7)
            self.assertEqual((c['camera']['pitch'],c['camera']['roll'],c['camera']['yaw']),(0.,0.,90.))
            self.assertAlmostEqual(c['wearer']['z'],1.2)
            self.assertEqual((c['camera']['x'],c['camera']['y']),(123.,-45.))
        self.assertEqual(len(groups),8)
        for group in groups.values():
            variants={c['variant_id'] for c in group}
            self.assertTrue({'CLEAR','BODY_ONLY','HEAD_ONLY','BOTH'}<=variants)
            self.assertEqual(len(variants),8 if group[0]['condition']['family']=='crossbar' else 4)
        self.assertEqual(Counter(c['declared_range'] for c in spec['cases']),{'near':20,'far':20})

    def test_region_placement_preserves_shared_local_fixture_geometry(self):
        a=generate(self.fixtures,{},dict(x=0.,y=0.,yaw=0.,floor_z_m=.7,region_id='a'),'train')
        b=generate(self.fixtures,{},dict(x=123.,y=-45.,yaw=90.,floor_z_m=1.2,region_id='b'),'dev')
        for ca,cb in zip(a['cases'],b['cases']):
            self.assertEqual(ca['local_geometry_sha256'],cb['local_geometry_sha256'])
            self.assertNotEqual(ca['group_id'],cb['group_id'])
            for oa,ob in zip(ca['objects'],cb['objects']):
                ax,ay,az=oa['center_m'];bx,by,bz=ob['center_m']
                self.assertAlmostEqual(bx,123.-ay)
                self.assertAlmostEqual(by,-45.+ax)
                self.assertAlmostEqual(bz,az+.5)
                self.assertEqual(set(ob['rotation_deg']),{'pitch','yaw','roll'})


if __name__=='__main__':unittest.main()
