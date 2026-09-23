"""Source geometry and partition checks independent of rendered outcomes."""
import unittest
from collections import Counter
from cnh_route_spec import layouts,pilot_layouts,frames,pose,COUNTS,CLIPS,FAMILIES,ENVIRONMENTS


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=layouts()

    def test_family_and_environment_counts(self):
        expected_environments={'sidewalk','intersection','plaza','alley'}
        self.assertEqual(set(ENVIRONMENTS),expected_environments)
        self.assertEqual(len(self.rows),384)
        self.assertEqual(Counter(x['environment'] for x in self.rows),
                         {name:96 for name in expected_environments})
        for split,counts in COUNTS.items():
            selected=[x for x in self.rows if x['split']==split]
            self.assertEqual(Counter(x['family'] for x in selected),dict(zip(FAMILIES,counts)))
            expected={'train':45,'dev':10,'test':41}[split]
            self.assertEqual(Counter(x['environment'] for x in selected),
                             {name:expected for name in expected_environments})
        self.assertEqual(len({x['asset_bundle_id'] for x in self.rows}),384)

    def test_pilot_has_no_dev_or_test_and_removed_exact_pose(self):
        pilot=pilot_layouts(self.rows)
        self.assertEqual(len(pilot),20)
        self.assertEqual({x['split'] for x in pilot},{'train'})
        self.assertEqual(Counter(x['environment'] for x in pilot),{name:5 for name in ENVIRONMENTS})
        self.assertEqual(Counter(x['family'] for x in pilot),dict(zip(FAMILIES,(6,5,3,3,3))))
        for row in pilot:
            for t in (0,.7,2,3.9):
                self.assertEqual(pose(row,'centre',t),pose(row,'removed',t))
        f=frames(pilot[:1]);self.assertEqual(len(f),160)
        self.assertTrue(all(not any(o['id']==1 for o in x['objects']) for x in f if x['clip_kind']=='removed'))

    def test_phase_one_illumination_and_fixtures_are_outdoor(self):
        for row in self.rows:
            self.assertIn(row['illumination']['mode'],('daylight','dusk'))
            self.assertFalse(any(o['name']=='ceiling' for o in row['backgrounds']))

    def test_positive_duration_and_path_endpoint(self):
        for row in self.rows:
            p=row['trajectory']
            self.assertLessEqual(p['start_m']-3.9*p['speed_mps'],1.+1e-12)
            if row['family']!='low':
                self.assertGreaterEqual(row['source_geometric_check']['conservative_positive_samples'],20)
                self.assertLessEqual(row['source_geometric_check']['final_target_point_axial_m'],1.)
            self.assertEqual(set(row['path_y']),set(CLIPS))


if __name__=='__main__':unittest.main()
