"""G14 generator checks for reproducibility and matched height interventions."""
import collections
import json
import unittest
from worlds_spec import specification, scene_preview
from worlds_scene import world_spec


class WorldSpecTests(unittest.TestCase):
    def test_scene_preview_is_target_free_and_has_local_frame_indices(self):
        preview=scene_preview()
        self.assertEqual(len(preview['cases']),2)
        self.assertEqual({c['world_id'] for c in preview['cases']},{'sidewalk','corridor'})
        self.assertTrue(all(not c['objects'] for c in preview['cases']+preview['clips']))
        self.assertEqual([s['frame_indices'] for s in preview['samples']],[[0],[1]])
        self.assertTrue(all(c['objects'] for c in specification()['cases']))

    def test_counterfactual_changes_only_target_height(self):
        spec=specification();self.assertEqual(spec,specification())
        self.assertEqual(len(spec['samples']),48)
        groups=collections.defaultdict(list)
        for case in spec['cases']:groups[case['group_id']].append(case)
        self.assertEqual(len(groups),16)
        for cases in groups.values():
            self.assertEqual({c['relation'] for c in cases},{'BODY','HEAD','NEITHER'})
            reference=cases[0]
            for case in cases:
                self.assertEqual(case['camera'],reference['camera'])
                self.assertEqual(case['wearer'],reference['wearer'])
                obj=case['objects'][0];ref=reference['objects'][0]
                self.assertEqual({k:v for k,v in obj.items() if k!='center_m'},{k:v for k,v in ref.items() if k!='center_m'})
                self.assertEqual(obj['center_m'][:2],ref['center_m'][:2])

    def test_worlds_are_deterministic_and_use_real_assets(self):
        for world in ('sidewalk','corridor'):
            for seed in (1401,1402):
                spec=world_spec(world,seed)
                self.assertEqual(spec,world_spec(world,seed))
                encoded=json.dumps(spec,allow_nan=False)
                self.assertIn('/Game/SampleMaterialsV2/',encoded)
                self.assertEqual(spec['floor_z_m'],.12)
                self.assertLess(len(spec['objects'])+len(spec['lights']),200)


if __name__=='__main__':unittest.main()
