import collections
import unittest
from contextual_scene import preview, collection


class ContextSceneTests(unittest.TestCase):
    def test_supported_preview_and_fixed_views(self):
        spec=preview({})
        self.assertEqual(len(spec['cases']),10)
        for a,b in zip(spec['cases'][::2],spec['cases'][1::2]):
            self.assertEqual(a['objects'],b['objects'])
            names={o['name'] for o in a['objects']}|{'ground','existing_facade'}
            self.assertTrue(all(o['support_parent'] in names for o in a['objects']))
            self.assertTrue(all('material_asset' in o for o in a['objects']))
        window=spec['cases'][-1]
        self.assertTrue(any(o['name']=='bay_sill_wall' for o in window['objects']))

    def test_full_core_geometric_balance_and_view_pairing(self):
        spec=collection({})
        core=[c for c in spec['cases'] if c['condition']['subset']=='core']
        self.assertEqual(len(spec['cases']),128)
        self.assertEqual(collections.Counter(c['geometric_contact']['relation'] for c in core),
                         dict(CLEAR=24,BODY_ONLY=24,HEAD_ONLY=24,BOTH=24))
        for a,b in zip(core[::2],core[1::2]):
            self.assertEqual(a['objects'],b['objects'])
            self.assertNotEqual(a['camera']['x'],b['camera']['x'])
        for case in core:
            self.assertEqual(case['condition']['desired_relation'],case['geometric_contact']['relation'])

    def test_light_pair_preserves_geometry_and_path_shift_preserves_world(self):
        cases=collection({})['cases']
        for kind in ('light','path'):
            a,b=[c for c in cases if c['condition']['hard_kind']==kind]
            self.assertEqual(a['objects'],b['objects'])
        light=[c for c in cases if c['condition']['hard_kind']=='light']
        self.assertNotEqual(light[0]['sun_intensity_scale'],light[1]['sun_intensity_scale'])


if __name__=='__main__':unittest.main()
