import copy
import unittest
from collections import Counter
import numpy as np
from scene_diversity_plan import allocate, exposure, schedule, validate_final_records


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = allocate(['r1','r2','r3','r4'], ['r1','r2'])

    def records(self):
        return {a: [dict(frame_index=i, source_role='TRAIN_ONLY', template_id=rs[i//4]['template_id'],
            family=rs[i//4]['family'], region_id=rs[i//4]['region_id'], relation=rs[i//4]['relations'][i%4],
            native_near=[[0,0],[1,0],[0,1],[1,1]][i%4], geometry_template_sha256=f'{i//4:064x}',
            camera_context_sha256='a'*64, native_label_sha256='b'*64) for i in range(256)]
            for a, rs in self.plan['arms'].items()}

    def test_allocation_and_sampling_balance(self):
        for a, count in [('concentrated',32),('distributed',16)]:
            rows=self.plan['arms'][a]
            self.assertEqual(set(Counter(r['region_id'] for r in rows).values()), {count})
            for region in {r['region_id'] for r in rows}:
                self.assertEqual(set(Counter(r['family'] for r in rows if r['region_id']==region).values()), {count//4})
        d=schedule(); e=exposure(self.plan,d)
        self.assertTrue(all(len(set(row))==3 for row in d['quartet_indices']))
        self.assertEqual((e['original_draws'],e['new_draws'],e['new_BODY_positive_draws'],e['new_HEAD_positive_draws']), (40000,24000,12000,12000))
        self.assertEqual(e['arms']['concentrated']['family_draws'],e['arms']['distributed']['family_draws'])
        self.assertTrue(all(np.array_equal(v,schedule()[k]) for k,v in d.items()))

    def test_invalid_subset(self):
        with self.assertRaises(ValueError): allocate(['r1','r2','r3','r4'],['r1','eval'])

    def test_capture_correspondence(self):
        rows=self.records(); self.assertEqual(validate_final_records(self.plan,rows)['status'],'PASS')
        for key,value in [('source_role','EVAL_ONLY'),('native_near',[-1,0]),('camera_context_sha256','c'*64),('geometry_template_sha256','d'*64)]:
            bad=copy.deepcopy(rows); bad['distributed'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_final_records(self.plan,bad)


if __name__=='__main__': unittest.main()
