"""Fixed batch and inherited model/threshold routing; no GPU or new data fit."""
import unittest
import cnh_model_stability_dev as M


class StabilityDriver(unittest.TestCase):
    def test_full_distribution_order_and_evaluable_not_pass(self):
        rows=[dict(unit=u,config=c,tag=t,valid=True,evaluable=False,contact=False,control=False,deadline_index=None)
              for u in range(410004,410052) for c in range(40) for t in M.TAGS]
        self.assertEqual(len(M.validate_rows(rows)),48)
        rows[0].update(evaluable=True,contact=True,deadline_index=0)
        M.validate_rows(rows)
        rows[0]['contact']=False
        with self.assertRaises(AssertionError):M.validate_rows(rows)
        rows[0]['evaluable']=False
        rows[-1],rows[-2]=rows[-2],rows[-1]
        with self.assertRaises(AssertionError):M.validate_rows(rows)

    def test_same_max_requires_all_three_original_thresholds(self):
        rows=[dict(fold=f,key=f'calibrated/{cap}/{method}',threshold=f+1.)
              for f in range(3) for cap in ('0.025','0.050') for method in M.METHODS]
        ts=M.thresholds(dict(fold_records=rows))
        self.assertEqual(len(ts),24)
        self.assertEqual([ts[f,'0.025','original_center'] for f in range(3)],[1.,2.,3.])
        with self.assertRaises(AssertionError):M.thresholds(dict(fold_records=rows[1:]))
        with self.assertRaises(ValueError):M.thresholds(dict(fold_records=rows+[rows[0]]))


if __name__=='__main__':unittest.main()
