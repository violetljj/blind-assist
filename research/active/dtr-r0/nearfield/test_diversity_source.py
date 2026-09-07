"""G10 fixed group/stratum allocation and factorial identity checks."""
from collections import Counter
import unittest
from diversity_spec import specification


class DiversitySourceTests(unittest.TestCase):
    def test_allocations_and_freshness(self):
        main=specification();probe=specification(True)
        self.assertEqual(len(main['cases']),864)
        self.assertEqual(len(probe['cases']),32)
        groups={c['group_id']:(c['split'],c['stratum']) for c in main['clips']}
        self.assertEqual(Counter(groups.values()),{('train','diverse'):160,('val','narrow'):12,('val','diverse'):12,('test','narrow'):16,('test','diverse'):16})
        self.assertFalse(set(groups)&{c['group_id'] for c in probe['clips']})
        self.assertTrue(all(s['frame_indices']==[i] for i,s in enumerate(main['samples'])))

    def test_retained_geometry_and_warmups(self):
        spec=specification()
        for first in range(0,len(spec['cases']),4):
            clips=spec['clips'][first:first+4];cases=spec['cases'][first:first+4]
            variants={c['variant']:c for c in clips}
            full={o['name']:o for o in variants['both']['objects']}
            self.assertEqual(variants['neither']['objects'],[])
            for name in ('bar','box'):
                self.assertEqual(variants[name+'_only']['objects'],[full[name]])
            for case in cases:
                self.assertEqual(case['camera'],cases[0]['camera'])
                self.assertEqual(case['wearer'],cases[0]['wearer'])
                self.assertEqual(case['settling_frames'],26)


if __name__=='__main__':unittest.main()
