import unittest
from range_pair_diagnostic import pairs_from_spec,summary,RULES
from local_stability_spec import specification
from local_rescue_source import specification as rescue_spec


class PairTest(unittest.TestCase):
    def test_complete_pairs_and_duplicates(self):
        for make in (specification,rescue_spec):
            pairs=pairs_from_spec(make())
            self.assertEqual(len(pairs),96)
            self.assertEqual(sum(r['unique'] for r in pairs),72)
            self.assertEqual(sum(r['unique'] and r['relation']!='OUTSIDE' for r in pairs),48)

    def test_scene_change_rejected(self):
        spec=specification();spec['cases'][3]['objects'][0]['size_m'][0]+=.01
        with self.assertRaises(AssertionError):pairs_from_spec(spec)

    def test_missing_is_not_success_and_exact_boundary(self):
        def row(relation,near,far):
            return dict(relation=relation,readouts={r:dict(near_m=near,far_m=far,near_target=True,far_target=True) for r in RULES})
        s=summary([row('INSIDE',3.,3.01),row('BOUNDARY',None,3.2),row('OUTSIDE',2.9,None)])
        for r in RULES:
            self.assertEqual(s[r]['correct_crossings'],1)
            self.assertEqual(s[r]['joint_rate'],.5)
            self.assertEqual(s[r]['outside_FPR'],.5)
            self.assertEqual(s[r]['missing_pairs'],1)


if __name__=='__main__':unittest.main()
