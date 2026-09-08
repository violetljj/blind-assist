"""Local geometry, matching and rigid-transform checks; no UE/native inference."""
from collections import Counter
import copy
import math
import unittest
import contextual_collection1000 as original
from contextual_geometry import target_contact
from scene_diversity_templates import (AUTHORITY, SEED, generate, place_quartet,
    inverse_placement, validate_quartet, validate_placement)


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = generate({})

    def test_balanced_exact_first256_seed_and_no_capture_admission(self):
        before = original.SEED
        template = dict(map_file='unadmitted-map', cases=[{'old': True}], provenance={'status': 'PASS'})
        saved = copy.deepcopy(template)
        second = generate(template)
        self.assertEqual(template, saved)
        self.assertEqual(original.SEED, before)
        self.assertEqual(second['shared_geometry_sha256'], self.library['shared_geometry_sha256'])
        self.assertEqual(second['authority'], AUTHORITY)
        self.assertEqual(second['native_labels_status'], 'NOT_RUN')
        self.assertIs(second['capture_source'], False)
        self.assertNotIn('map_file', second)
        self.assertNotIn('status', second)
        self.assertEqual(Counter(q['family'] for q in second['quartets']),
            dict(crossbar=16, cabinet=16, oblique_rod=16, hanging_sign=16))
        self.assertEqual(Counter(c['variant_id'] for q in second['quartets'] for c in q['cases']),
            dict(CLEAR=64, BODY_ONLY=64, HEAD_ONLY=64, BOTH=64))
        try:
            original.SEED = SEED
            source = original.collection({})['cases'][:256]
        finally:
            original.SEED = before
        for raw, case in zip(source, [c for q in second['quartets'] for c in q['cases']]):
            self.assertEqual(case['source_lineage']['collection_case_name'], raw['name'])
            self.assertEqual(case['condition']['height_m'], raw['condition']['height_m'])
            self.assertEqual(case['camera']['pitch'], raw['camera']['pitch'])
            self.assertAlmostEqual(case['camera']['z'], raw['camera']['z']-raw['floor_z_m'])
            for a, b in zip(case['objects'], raw['objects']):
                self.assertEqual(a['material_asset'], b['material_asset'])
                self.assertEqual(a.get('rotation_deg'), b.get('rotation_deg'))

    def test_all_local_contacts_context_and_backboards(self):
        for quartet in self.library['quartets']:
            self.assertTrue(validate_quartet(quartet))
            for case in quartet['cases']:
                self.assertEqual(target_contact(case['objects'], case['wearer'])['relation'], case['variant_id'])
                board = [o for o in case['objects'] if o['name'] == 'coverage_grounded_backboard']
                self.assertEqual(len(board), int(quartet['family'] in ('cabinet', 'hanging_sign')))
                if board:
                    self.assertAlmostEqual(board[0]['center_m'][2]-board[0]['size_m'][2]/2, 0.)
                self.assertFalse(any(o['support_parent'] == 'existing_facade' for o in case['objects']))

    def test_nonzero_yaw_floor_inverse_all_quartets(self):
        for index, quartet in enumerate(self.library['quartets']):
            before = copy.deepcopy(quartet)
            floor = 7.25 if index % 2 else -3.125
            placed = place_quartet(quartet, [32.7, -15.2], floor, 37.5, 'arm-A-world03')
            recovered = inverse_placement(placed)
            self.assertEqual(quartet, before)
            self.assertEqual(placed['shared_geometry_sha256'], quartet['shared_geometry_sha256'])
            for raw, world, local in zip(quartet['cases'], placed['cases'], recovered):
                self.assertEqual(world['camera']['x'], 32.7)
                self.assertEqual(world['camera']['y'], -15.2)
                self.assertEqual(world['wearer']['z'], floor)
                self.assertAlmostEqual(world['camera']['z'], floor+raw['camera']['z'])
                self.assertEqual(world['geometric_contact'], raw['geometric_contact'])
                self.assertEqual(world['native_labels_status'], 'REQUIRES_NATIVE_WORLD_ACCEPTANCE')
                for key in ('camera', 'wearer'):
                    for axis in ('x', 'y', 'z', 'pitch', 'yaw', 'roll'):
                        self.assertAlmostEqual(local[key].get(axis, 0.), raw[key].get(axis, 0.), places=10)
                for ro, wo, lo in zip(raw['objects'], world['objects'], local['objects']):
                    self.assertEqual(set(wo['rotation_deg']), {'pitch', 'yaw', 'roll'})
                    for v, ref in zip(lo['center_m'], ro['center_m']): self.assertAlmostEqual(v, ref, places=10)
                    self.assertAlmostEqual(wo['center_m'][2], floor+ro['center_m'][2])
                    self.assertEqual(wo['rotation_deg']['yaw'], ro.get('rotation_deg', {}).get('yaw', 0.)+37.5)
                    for axis in ('pitch', 'roll'):
                        self.assertEqual(wo['rotation_deg'].get(axis, 0.), ro.get('rotation_deg', {}).get(axis, 0.))
                self.assertEqual(target_contact(local['objects'], local['wearer'])['relation'], raw['variant_id'])

    def test_shared_geometry_between_distinct_world_placements(self):
        q = self.library['quartets'][2]  # Tilted rod: original roll must survive yaw.
        a = place_quartet(q, [0, 0], 0., 0., 'one-world')
        b = place_quartet(q, [-104, 56], 4.8, 123., 'other-world')
        self.assertEqual(a['shared_geometry_sha256'], b['shared_geometry_sha256'])
        self.assertNotEqual(a['cases'][0]['objects'], b['cases'][0]['objects'])
        self.assertTrue(validate_placement(a, q))
        self.assertTrue(validate_placement(b, q))
        rod = next(o for o in b['cases'][0]['objects'] if o['target_part'])
        self.assertNotEqual(rod['rotation_deg']['roll'], 0.)
        self.assertEqual(rod['rotation_deg']['yaw'], 123.)
        # Analytic solver must NOT quietly claim it evaluated a yawed world.
        with self.assertRaises(ValueError):
            target_contact(b['cases'][0]['objects'], b['cases'][0]['wearer'])
        # Copying the shared hash is not enough if world geometry was altered.
        b['cases'][0]['objects'][0]['center_m'][2] += .2
        with self.assertRaisesRegex(ValueError, 'center transform mismatch'): validate_placement(b, q)

    def test_mutations_and_nonfinite_placement_rejected(self):
        q = copy.deepcopy(self.library['quartets'][0])
        q['cases'][1]['camera']['pitch'] += 1
        with self.assertRaisesRegex(ValueError, 'context changed'): validate_quartet(q)
        q = copy.deepcopy(self.library['quartets'][0])
        q['cases'][1]['geometric_contact']['relation'] = 'CLEAR'
        with self.assertRaisesRegex(ValueError, 'Stale local contact'): validate_quartet(q)
        q = self.library['quartets'][0]
        with self.assertRaisesRegex(ValueError, 'finite'): place_quartet(q, [0, 0], math.inf, 0)


if __name__ == '__main__':
    unittest.main()
