"""Fixed source budget, disjoint groups and retained object identity."""
import unittest
from collections import Counter
from factorial_spec import specification


class FactorialSourceTests(unittest.TestCase):
    def test_one_frame_budget_and_split(self):
        main=specification(conversion_mode='single_access')
        probe=specification(preflight=True)
        self.assertEqual(len(main['cases']),256)
        self.assertEqual(len(probe['cases']),16)
        self.assertEqual(Counter(s['split'] for s in main['samples']),dict(train=160,val=48,test=48))
        self.assertTrue(all(s['frame_indices']==[i] for i,s in enumerate(main['samples'])))
        self.assertTrue(all(c['settling_frames']==26 for c in main['cases']))
        self.assertFalse({s['group_id'] for s in main['samples']}&{s['group_id'] for s in probe['samples']})

    def test_retained_objects_and_camera_are_identical(self):
        spec=specification();groups={}
        for clip,case in zip(spec['clips'],spec['cases']):
            groups.setdefault(clip['group_id'],{})[clip['variant']]=(clip,case)
        for variants in groups.values():
            self.assertEqual(set(variants),{'both','bar_only','box_only','neither'})
            full={o['name']:o for o in variants['both'][0]['objects']}
            self.assertEqual(variants['neither'][0]['objects'],[])
            for name in ('bar','box'):
                self.assertEqual(variants[name+'_only'][0]['objects'],[full[name]])
            for _,case in variants.values():
                self.assertEqual(case['camera'],variants['both'][1]['camera'])
                self.assertEqual(case['wearer'],variants['both'][1]['wearer'])


if __name__=='__main__': unittest.main()
