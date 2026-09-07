"""CPU-only NF-G9-A factorial source and analytic geometry invariants."""
import unittest

from grounding_spec import specification, VARIANTS
from contact_retina_spec import BODY_BOXES


def bounds(obj):
    return ([c-s/2 for c, s in zip(obj['center_m'], obj['size_m'])],
            [c+s/2 for c, s in zip(obj['center_m'], obj['size_m'])])


def intersects(a, b):
    return all(max(a[0][i], b[0][i]) < min(a[1][i], b[1][i]) for i in range(3))


class GroundingSourceTests(unittest.TestCase):
    def test_main_counts_and_static_frames(self):
        spec = specification()
        self.assertEqual(len(spec['clips']), 48)
        self.assertEqual(len(spec['samples']), 48)
        self.assertEqual(len(spec['cases']), 144)
        self.assertEqual(len({s['group_id'] for s in spec['samples']}), 12)
        self.assertEqual(len({c['name'] for c in spec['cases']}), 144)
        used = []
        for sample in spec['samples']:
            self.assertEqual(sample['split'], 'test')
            cases = [spec['cases'][i] for i in sample['frame_indices']]
            used.extend(sample['frame_indices'])
            self.assertEqual([c['frame_in_clip'] for c in cases], [0, 1, 2])
            self.assertEqual([c['time_s'] for c in cases], [0., .2, .4])
            for case in cases:
                self.assertEqual(case['clip_id'], sample['clip_id'])
                self.assertEqual(case['settling_frames'], 8)
                for key in ('camera', 'wearer', 'objects'):
                    self.assertEqual(case[key], cases[0][key])
        self.assertEqual(sorted(used), list(range(144)))
        self.assertEqual(spec, specification())

    def test_factorial_retains_exact_camera_material_and_geometry(self):
        for spec in (specification(), specification(preflight=True)):
            for group in {s['group_id'] for s in spec['samples']}:
                clips = {c['variant']: c for c in spec['clips'] if c['group_id'] == group}
                self.assertEqual(set(clips), set(VARIANTS))
                by_clip = {c['clip_id']: c for c in spec['cases'] if c['frame_in_clip'] == 2}
                cases = [by_clip[clips[v]['clip_id']] for v in VARIANTS]
                for case in cases:
                    self.assertEqual(case['camera'], cases[0]['camera'])
                    self.assertEqual(case['wearer'], cases[0]['wearer'])
                both = {o['name']: o for o in clips['both']['objects']}
                self.assertEqual(set(both), {'bar', 'box'})
                self.assertEqual(clips['bar_only']['objects'], [both['bar']])
                self.assertEqual(clips['box_only']['objects'], [both['box']])
                self.assertEqual(clips['neither']['objects'], [])

    def test_object_query_isolation(self):
        for case in specification()['cases']:
            if len(case['objects']) != 2:
                continue
            objects = {o['name']: bounds(o) for o in case['objects']}
            wearer = case['wearer']
            queries = []
            for low, high in BODY_BOXES:
                queries.append(([wearer['x']+high[0], wearer['y']+low[1], wearer['z']+low[2]],
                                [wearer['x']+high[0]+3., wearer['y']+high[1], wearer['z']+high[2]]))
            self.assertTrue(intersects(objects['box'], queries[0]))
            self.assertFalse(intersects(objects['box'], queries[1]))
            self.assertTrue(intersects(objects['bar'], queries[1]))
            self.assertFalse(intersects(objects['bar'], queries[0]))
            self.assertFalse(intersects(objects['box'], objects['bar']))
            self.assertLess(objects['box'][1][2], queries[1][0][2])
            self.assertGreater(objects['bar'][0][2], queries[0][1][2])

    def test_preflight_two_groups_two_settings_disjoint_main(self):
        preflight, main = specification(preflight=True), specification()
        self.assertEqual(len(preflight['clips']), 16)
        self.assertEqual(len(preflight['cases']), 48)
        self.assertEqual(len(preflight['samples']), 16)
        physical_groups = {s['group_id'].split('_settle')[0] for s in preflight['samples']}
        self.assertEqual(physical_groups, {'g000', 'g001'})
        self.assertTrue(physical_groups.isdisjoint({s['group_id'] for s in main['samples']}))
        self.assertEqual({c['settling_frames'] for c in preflight['cases']}, {2, 8})
        cases = {c['clip_id']: c for c in preflight['cases'] if c['frame_in_clip'] == 2}
        for group in physical_groups:
            for variant in VARIANTS:
                slow, fast = cases[f'{group}_settle8_{variant}'], cases[f'{group}_settle2_{variant}']
                for key in ('camera', 'wearer', 'objects'):
                    self.assertEqual(slow[key], fast[key])
        self.assertEqual({c['settling_frames'] for c in specification(settling=2)['cases']}, {2})


if __name__ == '__main__':
    unittest.main()
