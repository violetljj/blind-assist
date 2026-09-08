import unittest
from collections import Counter
from body_query_capture_spec import generate
from body_query_5000_spec import fixture_catalog,site_cases

class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=fixture_catalog(generate({},'train'))

    def test_500_frames_complete_groups_and_geometry_variation(self):
        region=dict(region_id='r',split='train');rows=[]
        for rank in range(25):
            rows+=site_cases(self.catalog,region,dict(site_id=f'r{rank}',camera_xy_m=[rank*6.,3.],
                floor_z_m=.7,yaw_deg=(rank%4)*90),rank)
        self.assertEqual(len(rows),500)
        groups={}
        for c in rows:
            groups.setdefault(c['group_id'],[]).append(c)
            self.assertAlmostEqual(c['camera']['z']-c['floor_z_m'],1.7)
            self.assertEqual(c['camera']['pitch'],0.)
            self.assertEqual(c['source_role'],'TRAIN_ONLY')
        self.assertEqual(len(groups),100)
        for group in groups.values():
            self.assertEqual(len({c['variant_id'] for c in group}),8 if group[0]['condition']['family']=='crossbar' else 4)
            self.assertEqual(len({str(c['camera']) for c in group}),1)
        self.assertGreater(len({c['local_geometry_sha256'] for c in rows}),400)
        self.assertEqual(Counter(c['declared_range'] for c in rows),{'near':252,'far':248})

if __name__=='__main__':unittest.main()
