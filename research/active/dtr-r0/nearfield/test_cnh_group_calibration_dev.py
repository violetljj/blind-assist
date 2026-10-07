"""Two-group maximum threshold, indivisible ties and exact max-score control."""
import unittest
import numpy as np
import cnh_group_calibration_dev as M


class GroupThresholds(unittest.TestCase):
    def data(self):
        tags=np.array(['original','HB','original','HB']);eligible=np.array([1,1,0,0],bool);scores={}
        for fold in range(3):
            for method in M.METHODS:
                x=np.full((4,13),np.nan);x[:2]=1000
                shift=0 if method=='original_center' else fold*.1
                x[0,2:12]=np.array([.1]*5+[.9]*5)+shift
                x[1,2:12]=np.array([.2]*5+[1.2]*5)+shift
                scores[f'fold{fold}/{method}']=x
        return scores,eligible,tags

    def test_max_group_cutoff_whole_ties_cap_and_minimality(self):
        scores,eligible,tags=self.data();ts,records,individual=M.group_thresholds(scores,eligible,tags,caps=(.25,.5))
        for key,theta in ts.items():
            fold,cap_text,method=key;cap=float(cap_text)
            self.assertEqual(theta,max(individual[tag][key] for tag in M.TAGS))
            previous=np.nextafter(theta,-np.inf);violates=[]
            for tag in M.TAGS:
                values=scores[f'fold{fold}/{method}'][eligible&(tags==tag),2:12]
                self.assertLessEqual((values>=theta).mean(),cap)
                violates.append((values>=previous).mean()>cap)
            self.assertTrue(any(violates))
        self.assertEqual(len({ts[f,'0.250','original_center'] for f in range(3)}),1)
        self.assertTrue(all(r['minimality_verified'] for r in records))

    def test_hb_binding_and_alarm_subset_of_original_only(self):
        scores,eligible,tags=self.data();ts,_,individual=M.group_thresholds(scores,eligible,tags,caps=(.25,))
        for key,theta in ts.items():
            fold,_,method=key
            self.assertGreater(theta,individual['original'][key])
            values=scores[f'fold{fold}/{method}'][eligible]
            self.assertFalse(((values>=theta)&~(values>=individual['original'][key])).any())


if __name__=='__main__':unittest.main()
