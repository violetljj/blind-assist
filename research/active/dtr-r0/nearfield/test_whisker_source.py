"""Focused NF-G8 geometry and paired-source invariants."""
import unittest

from whisker_spec import specification
from verify_whisker import query_overlap
from contact_retina_spec import BODY_BOXES


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specification()

    def test_groups_and_causal_history(self):
        groups = {}
        for sample in self.spec['samples']:
            groups.setdefault(sample['group_id'], set()).add(sample['split'])
            cases = [self.spec['cases'][i] for i in sample['frame_indices']]
            self.assertEqual(len(cases), 3)
            self.assertEqual(len({c['clip_id'] for c in cases}), 1)
            self.assertTrue(all(a['time_s'] < b['time_s'] for a,b in zip(cases,cases[1:])))
        self.assertEqual(len(groups),128)
        self.assertTrue(all(len(s)==1 for s in groups.values()))

    def test_far_horizontal_bars_stay_above_floor(self):
        for clip in self.spec['clips']:
            if clip['control']=='far' and clip['shape']=='bar':
                obj=clip['objects'][-1]
                self.assertGreater(obj['center_m'][2]-obj['size_m'][2]/2,.12)

    def test_stopped_and_static_same_current_geometry(self):
        cases={c['name']:c for c in self.spec['cases']}
        for clip in self.spec['clips']:
            if clip['motion']=='stop':
                stop=cases[clip['clip_id']+'_f5']
                static=cases[clip['clip_id'].replace('_stop','_static')+'_f5']
                self.assertEqual(stop['camera'],static['camera'])
                self.assertEqual(stop['wearer'],static['wearer'])
                self.assertEqual(stop['speed_m_s'],0)
                self.assertEqual([query_overlap(stop,b) for b in BODY_BOXES],
                                 [query_overlap(static,b) for b in BODY_BOXES])

    def test_lateral_and_above_are_outside_queries(self):
        controls={c['clip_id']:c['control'] for c in self.spec['clips']}
        for case in self.spec['cases']:
            if controls[case['clip_id']] in ('lateral','above'):
                self.assertFalse(any(query_overlap(case,b) for b in BODY_BOXES))


if __name__=='__main__':
    unittest.main()
