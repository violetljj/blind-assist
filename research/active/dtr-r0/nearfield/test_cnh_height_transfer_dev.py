"""Focused transfer boundary tests without rendering or model fitting."""
import unittest
import cnh_height_transfer_dev as M


class TransferContract(unittest.TestCase):
    def test_invalid_is_missing_and_common_subset_requires_all_four(self):
        anchors=[]
        for i in range(2):
            variants={tag:dict(valid=True,contact=True,control=False,physical_contact=True,physical_control=False,
                contact_query=[True,True],deadline_index=7,reasons=[]) for tag in M.TAGS}
            if i==0:variants['short'].update(valid=False,contact=None,control=None,deadline_index=None,reasons=['deadline_censored'])
            anchors.append(dict(anchor_id=str(i),unit=i,config=0,fold=0,variants=variants))
        rows=M.evaluation_layout(dict(anchors=anchors))
        self.assertEqual(len(rows),8)
        self.assertIsNone(rows[0]['contact']);self.assertIsNone(rows[0]['control'])
        self.assertEqual(rows[0]['deadline'],-1)
        self.assertEqual(rows[0]['invalid_reasons'],['deadline_censored'])
        self.assertEqual(sum(r['valid'] for r in rows),7)
        self.assertEqual(sum(r['common_valid'] for r in rows),4)
        self.assertEqual([r['tag'] for r in rows[:4]],list(M.TAGS))

    def test_only_complete_original_calibration_thresholds_inherited(self):
        records=[dict(fold=f,key=f'calibrated/{cap}/{m}',threshold=10+f) for f in range(3) for cap in ('0.025','0.050') for m in M.METHODS]
        records.append(dict(fold=0,key='matched_eval_descriptive/0.025/raw_hb_aug',threshold=-100))
        ts=M.thresholds(dict(fold_records=records))
        self.assertEqual(len(ts),42);self.assertEqual(ts[0,'0.025','raw_hb_aug'],10)
        with self.assertRaises(ValueError):M.thresholds(dict(fold_records=records+[records[0]]))
        with self.assertRaises(AssertionError):M.thresholds(dict(fold_records=records[1:]))


if __name__=='__main__':unittest.main()
