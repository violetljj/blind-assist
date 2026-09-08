import collections
import copy
import unittest
from headspace_spec import FAMILIES, RELATIONS, specification, target_contact


class HeadspaceTests(unittest.TestCase):
    def template(self):
        return dict(cases=[dict(name='west_sidewalk', objects=[], floor_z_m=.2,
            camera=dict(x=-85., y=9., z=1.9, yaw=0., pitch=-5., roll=0.))])

    def test_paired_worlds_and_four_computed_relations(self):
        template = self.template()
        original = copy.deepcopy(template)
        spec = specification(template)
        self.assertEqual(template, original)
        self.assertEqual(spec, specification(template))
        self.assertEqual(len(spec['cases']), 64)
        self.assertEqual(len({c['name'] for c in spec['cases']}), 64)
        groups = collections.defaultdict(list)
        worlds = collections.defaultdict(list)
        for case in spec['cases']:
            groups[case['group_id']].append(case)
            worlds[case['world_id']].append(case)
        self.assertEqual(len(groups), 16)
        self.assertEqual({c['family'] for c in spec['cases']}, set(FAMILIES))
        for cases in groups.values():
            self.assertEqual({c['expected_target_contact']['relation'] for c in cases}, set(RELATIONS))
            ref = cases[0]
            for case in cases:
                self.assertEqual(case['camera'], ref['camera'])
                for a, b in zip(case['objects'], ref['objects']):
                    self.assertEqual(a['center_m'][:2], b['center_m'][:2])
                    self.assertEqual(a['size_m'], b['size_m'])
                    self.assertEqual(a['name'], b['name'])
        for cases in worlds.values():
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]['objects'], cases[1]['objects'])

    def test_open_structure_is_not_whole_bbox(self):
        wearer = dict(x=0, y=0, z=0)
        posts = [dict(center_m=[2, y, 1.3], size_m=[.1,.1,1.5]) for y in (-.6,.6)]
        self.assertEqual(target_contact(posts, wearer)['relation'], 'CLEAR')
        self.assertEqual(target_contact([dict(center_m=[2,0,1.3], size_m=[.1,1.3,1.5])], wearer)['relation'], 'BOTH')

    def test_distance_and_closed_boundary(self):
        cube = dict(center_m=[1.,0,1.4], size_m=[.1,.1,.02])
        result = target_contact([cube], dict(x=0,y=0,z=0))
        self.assertEqual(result['relation'], 'BOTH')
        self.assertAlmostEqual(result['first_contact_distance_m'][0], .77)
        self.assertAlmostEqual(result['first_contact_distance_m'][1], .82)
        self.assertEqual(target_contact([cube], dict(x=-5,y=0,z=0))['relation'], 'CLEAR')
        self.assertEqual(target_contact([cube], dict(x=2,y=0,z=0))['relation'], 'CLEAR')
        with self.assertRaises(ValueError):
            target_contact([dict(cube, mesh_asset='anything')], dict(x=0,y=0,z=0))


if __name__ == '__main__':
    unittest.main()
