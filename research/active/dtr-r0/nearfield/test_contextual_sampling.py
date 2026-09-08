import collections
import itertools
import json
import unittest

from contact_retina_spec import BODY_BOXES
from contextual_sampling import (
    DISTANCES_M, FAMILIES, HARD_COUNTS, HEIGHTS_M, POSITIONS, RELATIONS,
    conditions, specification,
)


class ContextualSamplingTests(unittest.TestCase):
    def test_deterministic_exact_factorial_and_allocation(self):
        rows = conditions()
        self.assertEqual(rows, conditions())
        self.assertEqual(len(rows), 128)
        self.assertEqual(len({r['condition_id'] for r in rows}), 128)
        core = [r for r in rows if r['subset'] == 'core']
        self.assertEqual(len(core), 96)
        self.assertEqual(collections.Counter(
            (r['family'], r['desired_relation'], r['position'], r['distance_m'])
            for r in core), collections.Counter(itertools.product(
                FAMILIES, RELATIONS, POSITIONS, DISTANCES_M)))
        self.assertEqual(collections.Counter(r['hard_kind'] for r in rows
                                            if r['subset'] == 'hard'), HARD_COUNTS)

    def test_core_matched_parents_and_split_binding(self):
        parents = collections.defaultdict(list)
        for row in conditions():
            self.assertEqual(row['split_group_id'],
                             row['source_site_id'] + ':' + row['family'])
            if row['subset'] == 'core':
                parents[row['counterfactual_parent_id']].append(row)
                self.assertEqual(row['camera_pitch_deg'], 0.)
                self.assertEqual(row['height_m'], HEIGHTS_M[row['desired_relation']])
                self.assertEqual(row['lateral_offset_m'], POSITIONS[row['position']])
                self.assertEqual(row['tilt_degrees'],
                                 30. if row['family'] == 'oblique_rod' else 0.)
                self.assertEqual(row['thickness_m'], .06)
                self.assertFalse(row['boundary'])
        self.assertEqual(len(parents), 4)
        for rows in parents.values():
            self.assertEqual(len(rows), 24)
            self.assertEqual({r['desired_relation'] for r in rows}, set(RELATIONS))
            self.assertEqual({r['position'] for r in rows}, set(POSITIONS))
            self.assertEqual({r['distance_m'] for r in rows}, set(DISTANCES_M))
        self.assertTrue({r['split_group_id'] for r in conditions('site_a')}.isdisjoint(
            {r['split_group_id'] for r in conditions('site_b')}))

    def test_hard_controls_change_only_requested_factor(self):
        groups = collections.defaultdict(list)
        for row in conditions():
            if row['subset'] == 'hard':
                groups[row['counterfactual_parent_id']].append(row)
        factor = {'thin': 'thickness_m', 'oblique': 'tilt_degrees',
                  'occlusion': 'occluder', 'light': 'lighting_profile',
                  'pitch': 'camera_pitch_deg', 'path': 'route_lateral_delta_m'}
        expected = {'thin': {.01, .02}, 'oblique': {15., 30., 45., 60.},
                    'occlusion': {False, True}, 'light': {'daylight', 'darker'},
                    'pitch': {-15., 10.}, 'path': {-.35, .35}}
        for rows in groups.values():
            kind = rows[0]['hard_kind']
            ignore = {'condition_id'}
            if kind in factor:
                ignore.add(factor[kind])
                self.assertEqual({r[factor[kind]] for r in rows}, expected[kind])
            else:
                ignore.update(('height_m', 'desired_relation'))
                midpoint = 1.4 if kind == 'body_head_boundary' else 1.85
                self.assertEqual({r['height_m'] for r in rows},
                                 {round(midpoint - .02, 2), round(midpoint + .02, 2)})
                self.assertTrue(all(r['boundary'] for r in rows))
            fixed = [{k: v for k, v in r.items() if k not in ignore} for r in rows]
            self.assertTrue(all(r == fixed[0] for r in fixed))

    def test_manifest_is_intent_only_and_does_not_mutate_body_contract(self):
        spec = specification()
        json.dumps(spec, allow_nan=False)
        self.assertEqual(spec['body_boxes'], BODY_BOXES)
        self.assertEqual([b[0][2] for b in BODY_BOXES], [.65, 1.4])
        self.assertEqual([b[1][2] for b in BODY_BOXES], [1.4, 1.85])
        spec['body_boxes'][0][0][2] = -99
        self.assertEqual(BODY_BOXES[0][0][2], .65)
        self.assertIn('NOT_GEOMETRY_TRUTH', spec['relation_authority'])
        for row in spec['conditions']:
            self.assertIn(row['desired_relation'], RELATIONS)
            self.assertFalse({'relation', 'truth', 'expected_target_contact',
                              'actual_relation'} & row.keys())
            self.assertEqual(row['boundary'], row['boundary_name'] is not None)
        self.assertTrue(any(r['desired_relation'] == 'CLEAR' and r['boundary']
                            for r in spec['conditions']))
        with self.assertRaises(ValueError):
            conditions(' ')


if __name__ == '__main__':
    unittest.main()
