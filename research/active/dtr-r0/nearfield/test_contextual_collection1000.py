import collections
import copy
import json
from pathlib import Path
import sys
import unittest

from contextual_collection1000 import SCHEMA, SEED, SITE_ID, collection
from contextual_scene import collection as collection128

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tools'))
from check_contextual_headspace import evaluate
from make_contextual_headspace import make


class Collection1000Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = collection({})

    def native(self, spec):
        visibility = {'CLEAR': [0, 0], 'BODY_ONLY': [1, 0],
                      'HEAD_ONLY': [0, 1], 'BOTH': [1, 1]}
        return dict(status='PASS', rows=[dict(name=c['name'],
                    body_head_visible_targets=visibility[c['geometric_contact']['relation']],
                    visible_pixels_per_height=[0, 0], unknown_depth_pixels=0)
                    for c in spec['cases']])

    def test_deterministic_balanced_and_production_contract(self):
        template = dict(map_file='pinned-map', provenance={'old': True}, cases=[{'old': True}])
        before = copy.deepcopy(template)
        generated = collection(template)
        self.assertEqual(template, before)
        self.assertEqual(generated['cases'], self.spec['cases'])
        self.assertEqual(generated['schema'], SCHEMA)
        self.assertEqual(generated['seed'], SEED)
        self.assertEqual(generated['map_file'], 'pinned-map')
        self.assertNotIn('provenance', generated)
        self.assertEqual(len(generated['cases']), 1000)
        self.assertEqual(len({c['name'] for c in generated['cases']}), 1000)
        self.assertEqual(collections.Counter(c['geometric_contact']['relation']
                                             for c in generated['cases']),
                         dict(CLEAR=250, BODY_ONLY=250, HEAD_ONLY=250, BOTH=250))
        self.assertEqual(generated['pair_export_mode'], 'native_async')
        self.assertIs(generated['export_appearance'], False)
        self.assertEqual(generated['settling_ticks'], 32)
        self.assertEqual(generated['exposure_ev100'], 12.)
        self.assertEqual(generated['suite_contract']['source_worlds'], 1)
        json.dumps(generated, allow_nan=False)

    def test_quartet_views_materials_shapes_and_supports_stay_matched(self):
        groups = collections.defaultdict(list)
        for case in self.spec['cases']:
            groups[case['group_id']].append(case)
        self.assertEqual(len(groups), 250)
        self.assertEqual(collections.Counter(cases[0]['condition']['family'] for cases in groups.values()),
                         dict(crossbar=63, cabinet=63, oblique_rod=62, hanging_sign=62))
        cameras = set()
        for cases in groups.values():
            self.assertEqual(len(cases), 4)
            self.assertEqual({c['condition']['desired_relation'] for c in cases},
                             {'CLEAR', 'BODY_ONLY', 'HEAD_ONLY', 'BOTH'})
            ref = cases[0]
            cameras.add(tuple(sorted(ref['camera'].items())))
            for case in cases:
                self.assertEqual(case['camera'], ref['camera'])
                self.assertEqual(case['wearer'], ref['wearer'])
                condition = case['condition']
                self.assertEqual(condition['split_group_id'], SITE_ID)
                self.assertEqual(condition['source_site_id'], SITE_ID)
                self.assertEqual(case['source_site_id'], SITE_ID)
                self.assertEqual(condition['counterfactual_parent_id'], case['group_id'])
                for field in ('tilt_degrees', 'thickness_m', 'distance_m', 'lateral_offset_m',
                              'eye_height_m', 'camera_pitch_deg', 'common_height_jitter_m'):
                    self.assertEqual(condition[field], ref['condition'][field])
                names = {o['name'] for o in case['objects']} | {'ground', 'existing_facade'}
                self.assertTrue(all(o['support_parent'] in names for o in case['objects']))
                self.assertTrue(any(not o['target_part'] for o in case['objects']))
                self.assertEqual(len(case['objects']), len(ref['objects']))
                for obj, first in zip(case['objects'], ref['objects']):
                    for field in ('name', 'material_asset', 'support_parent', 'target_part'):
                        self.assertEqual(obj[field], first[field])
                    self.assertEqual(obj.get('rotation_deg'), first.get('rotation_deg'))
                    self.assertEqual(obj['center_m'][:2], first['center_m'][:2])
                    if obj['name'].startswith('suspension_'):
                        self.assertEqual(obj['size_m'][:2], first['size_m'][:2])
                    else:
                        self.assertEqual(obj['size_m'], first['size_m'])
        self.assertEqual(len(cameras), 250)

    def test_camera_and_shape_parameters_respect_requested_ranges(self):
        for case in self.spec['cases']:
            c = case['condition']
            for field, low, high in (('distance_m', .8, 2.8), ('lateral_offset_m', -.1, .1),
                                      ('eye_height_m', 1.6, 1.8), ('camera_pitch_deg', -8., 6.),
                                      ('thickness_m', .03, .08), ('common_height_jitter_m', -.005, .005)):
                self.assertGreaterEqual(c[field], low)
                self.assertLessEqual(c[field], high)
            if c['family'] == 'oblique_rod':
                self.assertGreaterEqual(c['tilt_degrees'], 15.)
                self.assertLessEqual(c['tilt_degrees'], 35.)
            else:
                self.assertEqual(c['tilt_degrees'], 0.)
            self.assertAlmostEqual(case['camera']['z'] - case['floor_z_m'], c['eye_height_m'])

    def test_checker_accepts_exact_quartets_and_retains_visibility_gaps(self):
        native = self.native(self.spec)
        report = evaluate(self.spec, native)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['quartets'], 250)
        self.assertEqual(report['native_visible_agreement'], 1000)
        native['rows'][1]['body_head_visible_targets'] = [None, None]
        report = evaluate(self.spec, native)
        self.assertEqual(report['status'], 'VISIBILITY_GAP')
        self.assertEqual(report['frames'], 1000)
        self.assertEqual(len(report['rows']), 1000)
        self.assertEqual(report['native_visibility_gaps'], 1)
        self.assertEqual(report['rows'][1]['visible_relation'], 'UNKNOWN')

    def test_checker_rejects_duplicate_names_unmatched_views_and_broken_geometry(self):
        native = self.native(self.spec)
        for mutation in ('duplicate_name', 'view', 'geometry', 'intrusion', 'group'):
            spec = copy.deepcopy(self.spec)
            case = spec['cases'][0]
            if mutation == 'duplicate_name':
                case['name'] = spec['cases'][1]['name']
            elif mutation == 'view':
                case['camera']['pitch'] += 1.
            elif mutation == 'geometry':
                case['geometric_contact']['relation'] = 'BOTH'
            elif mutation == 'intrusion':
                case['geometric_intrusion']['regions']['HEAD']['lateral_coverage_ratio'] = .5
            else:
                case['group_id'] = 'unmatched-group'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluate(spec, native)

    def test_existing128_core_acceptance_is_unchanged(self):
        spec = collection128({})
        native = self.native(spec)
        # Hard-case visibility gaps do not turn the existing core-only gate into a failure.
        hard_index = next(i for i, c in enumerate(spec['cases']) if c['condition']['subset'] == 'hard')
        native['rows'][hard_index]['body_head_visible_targets'] = [None, None]
        report = evaluate(spec, native)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['frames'], 128)
        self.assertEqual(report['core_visible_agreement'], 96)
        self.assertEqual(report['hard_visible_agreement'], 31)
        native['rows'][0]['body_head_visible_targets'] = [None, None]
        self.assertEqual(evaluate(spec, native)['status'], 'CORE_VISIBILITY_GAP')

    def test_materializer_rejects_conflicting_sizes_before_file_access(self):
        with self.assertRaises(ValueError):
            make('unused-source', 'unused-output', full=True, collection_1000=True)


if __name__ == '__main__':
    unittest.main()
