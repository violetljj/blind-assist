import copy
import math
import unittest

from contextual_geometry import intrusion_metrics, target_contact


class ContextualGeometryTests(unittest.TestCase):
    wearer = dict(x=0., y=0., z=0.)

    def solid(self, name='target', center=(1., 0., 1.1), size=(.1, .1, .1), roll=0.):
        return dict(name=name, center_m=list(center), size_m=list(size),
                    rotation_deg=dict(pitch=0., yaw=0., roll=roll))

    def test_open_structure_union_does_not_fill_hole(self):
        posts = [self.solid(f'post_{y}', center=(1., y, 1.25), size=(.1, .1, 1.5))
                 for y in (-.6, .6)]
        result = target_contact(posts, self.wearer)
        self.assertEqual(result['relation'], 'CLEAR')
        self.assertEqual(result['object_witness_names'], {'BODY': [], 'HEAD': []})
        enclosing = self.solid(center=(1., 0., 1.25), size=(.1, 1.3, 1.5))
        self.assertEqual(target_contact([enclosing], self.wearer)['relation'], 'BOTH')

    def test_tilted_thin_rod_head_only_despite_body_aabb_overlap(self):
        rod = self.solid(center=(1., .4, 1.45), size=(.06, .8, .01), roll=45.)
        result = target_contact([rod], self.wearer)
        self.assertEqual(result['relation'], 'HEAD_ONLY')
        self.assertIsNone(result['first_contact_distance_m'][0])
        self.assertAlmostEqual(result['first_contact_distance_m'][1], .84)
        self.assertEqual(result['object_witness_names']['HEAD'], ['target'])
        bbox = self.solid(center=(1., .4, 1.45),
                          size=(.06, .81 / math.sqrt(2), .81 / math.sqrt(2)))
        self.assertEqual(target_contact([bbox], self.wearer)['relation'], 'BOTH')
        # UE positive roll lowers the +Y end. Reversing it changes the contacts.
        rod['rotation_deg']['roll'] = -45.
        self.assertEqual(target_contact([rod], self.wearer)['relation'], 'BODY_ONLY')

    def test_start_overlap_behind_and_closed_horizon(self):
        overlap = self.solid(center=(0., 0., 1.1))
        self.assertEqual(target_contact([overlap], self.wearer, 0.)[
            'first_contact_distance_m'], [0., None])
        behind = self.solid(center=(-.5, 0., 1.1))
        self.assertEqual(target_contact([behind], self.wearer)['relation'], 'CLEAR')
        partly_behind = self.solid(center=(-.2, 0., 1.1))
        self.assertEqual(target_contact([partly_behind], self.wearer)[
            'first_contact_distance_m'], [0., None])
        edge = self.solid(center=(3.23, 0., 1.1))
        self.assertAlmostEqual(target_contact([edge], self.wearer)[
            'first_contact_distance_m'][0], 3.)
        edge['center_m'][0] += .00001
        self.assertEqual(target_contact([edge], self.wearer)['relation'], 'CLEAR')

    def test_body_head_closed_boundary_and_world_translation(self):
        shared = self.solid(center=(1., 0., 1.4), size=(.1, .1, .02))
        result = target_contact([shared], self.wearer)
        self.assertEqual(result['relation'], 'BOTH')
        self.assertAlmostEqual(result['first_contact_distance_m'][0], .77)
        self.assertAlmostEqual(result['first_contact_distance_m'][1], .82)
        offset = (4., -3., .7)
        shifted = copy.deepcopy(shared)
        shifted['center_m'] = [v + d for v, d in zip(shared['center_m'], offset)]
        translated = target_contact([shifted], dict(zip(('x', 'y', 'z'), offset)))
        self.assertEqual(translated['relation'], 'BOTH')
        for a, b in zip(result['first_contact_distance_m'], translated['first_contact_distance_m']):
            self.assertAlmostEqual(a, b)
        touching = self.solid(center=(1., 0., 1.9), size=(.1, .1, .1))
        self.assertEqual(target_contact([touching], self.wearer)['relation'], 'HEAD_ONLY')

    def test_supports_and_names_have_no_semantic_collision_override(self):
        target = self.solid('target', center=(2., 0., 2.2))
        support = self.solid('decorative_ignore_me', center=(1., 0., 1.25), size=(.1, .1, 2.))
        support['material'] = 'transparent'
        support['role'] = 'support'
        second = self.solid('second', center=(2., 0., 1.1))
        original = copy.deepcopy([target, support, second])
        result = target_contact([target, support, second], self.wearer)
        self.assertEqual(result['relation'], 'BOTH')
        self.assertEqual(result['object_witness_names'],
                         {'BODY': ['decorative_ignore_me', 'second'],
                          'HEAD': ['decorative_ignore_me']})
        self.assertAlmostEqual(result['first_contact_distance_m'][0], .77)
        self.assertEqual([target, support, second], original)
        self.assertEqual(target_contact([], self.wearer)['relation'], 'CLEAR')

    def test_intrusion_unions_overlapping_and_disconnected_components(self):
        overlapping = [self.solid(center=(x, y, 1.65), size=(.1, .2, .1))
                       for x, y in ((1., -.05), (2., .05))]
        result = intrusion_metrics(overlapping, self.wearer)['regions']
        self.assertAlmostEqual(result['HEAD']['lateral_intrusion_width_m'], .3)
        self.assertAlmostEqual(result['HEAD']['lateral_coverage_ratio'], .3 / .36)
        self.assertEqual(result['BODY']['lateral_intrusion_width_m'], 0.)
        disconnected = [self.solid(center=(1., y, 1.65), size=(.1, .02, .1))
                        for y in (-.13, .13)]
        self.assertAlmostEqual(intrusion_metrics(disconnected, self.wearer)[
            'regions']['HEAD']['lateral_intrusion_width_m'], .04)
        full = self.solid(center=(1., 0., 1.4), size=(.1, 3., 2.))
        covered = intrusion_metrics([full, full], self.wearer)['regions']
        for region, width in (('BODY', .56), ('HEAD', .36)):
            self.assertAlmostEqual(covered[region]['lateral_intrusion_width_m'], width)
            self.assertEqual(covered[region]['lateral_coverage_ratio'], 1.)

    def test_intrusion_clips_slanted_polygon_instead_of_its_bbox(self):
        rod = self.solid(center=(1., .4, 1.45), size=(.06, .8, .01), roll=45.)
        result = intrusion_metrics([rod], self.wearer)['regions']
        self.assertEqual(result['BODY']['lateral_intrusion_width_m'], 0.)
        expected = .18 - (.4 - .405 / math.sqrt(2))
        self.assertAlmostEqual(result['HEAD']['lateral_intrusion_width_m'], expected)
        self.assertAlmostEqual(result['HEAD']['lateral_coverage_ratio'], expected / .36)
        posts = [self.solid(center=(1., y, 1.25), size=(.1, .1, 1.5))
                 for y in (-.6, .6)]
        for values in intrusion_metrics(posts, self.wearer)['regions'].values():
            self.assertEqual(values['lateral_intrusion_width_m'], 0.)
            self.assertEqual(values['lateral_coverage_ratio'], 0.)

    def test_intrusion_excludes_unreachable_solids_and_handles_empty(self):
        distant = self.solid(center=(4., 0., 1.25), size=(.1, 1., 2.))
        behind = self.solid(center=(-1., 0., 1.25), size=(.1, 1., 2.))
        for objects in ([], [distant], [behind], [distant, behind]):
            for values in intrusion_metrics(objects, self.wearer)['regions'].values():
                self.assertEqual(values['lateral_intrusion_width_m'], 0.)
                self.assertEqual(values['lateral_coverage_ratio'], 0.)
        overlap = self.solid(center=(0., 0., 1.65), size=(.1, .02, .1))
        self.assertAlmostEqual(intrusion_metrics([overlap], self.wearer, horizon=0.)[
            'regions']['HEAD']['lateral_intrusion_width_m'], .02)
        self.assertIn('NOT_VOLUME_SEVERITY_PROBABILITY', intrusion_metrics(
            [], self.wearer)['authority'])

    def test_invalid_or_unsupported_geometry_is_rejected(self):
        variants = []
        for key in ('mesh_asset', 'primitive_asset'):
            variants.append(dict(self.solid(), **{key: '/Game/mesh'}))
        for key in ('pitch', 'yaw'):
            obj = self.solid()
            obj['rotation_deg'][key] = 1.
            variants.append(obj)
        for key in ('center_m', 'size_m'):
            for bad in (math.nan, math.inf, -math.inf):
                obj = self.solid()
                obj[key][0] = bad
                variants.append(obj)
        variants.extend([dict(self.solid(), size_m=[0., .1, .1]),
                         dict(self.solid(), size_m=[-.1, .1, .1]),
                         dict(self.solid(), center_m=[1., 2.]),
                         dict(self.solid(), rotation_deg={'roll': math.nan}),
                         dict(self.solid(), rotation_deg=[0., 0., 0.])])
        for obj in variants:
            for evaluate in (target_contact, intrusion_metrics):
                with self.subTest(obj=obj, evaluator=evaluate.__name__), self.assertRaises(ValueError):
                    evaluate([obj], self.wearer)
        for bad in (-1., math.nan, math.inf):
            with self.subTest(horizon=bad), self.assertRaises(ValueError):
                target_contact([], self.wearer, bad)
        for pose in (dict(self.wearer, yaw=90.), dict(self.wearer, x=math.nan),
                     dict(self.wearer, roll=math.inf)):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                target_contact([], pose)


if __name__ == '__main__':
    unittest.main()
