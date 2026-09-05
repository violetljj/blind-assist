"""Synthetic native-schema gate contracts only; no capture or model evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import evaluate_dtr_final_roster_source as ev

PROTOCOL=json.loads(Path(__file__).with_name('dtr_final_reckoning_roster_protocol.json').read_text(encoding='utf-8'))
IDS={s['stratum_id'][:3]:s['stratum_id'] for s in PROTOCOL['source_design']['strata']}


def fixture(code):
    rows=[]
    contact_indices=[] if code in ('S05','S08') else [40,80] if code=='S10' else [40]
    for i in range(101):
        t=i*.1
        transform={'x':7-.5*t,'y':0.,'z':0.,'yaw':0.,'pitch':0.,'roll':0.}
        if code=='S04':transform.update(x=4.,y=4.-t)
        if code=='S05':transform.update(x=abs(t-5)+.5)
        if code=='S08':transform.update(x=4.)
        actor={'transform':transform,'bounding_box':{'extent':{'x':.3,'y':.3,'z':1.}},
               'local_position':{'forward_m':transform['x'],'right_m':transform['y']},
               'command_velocity':{'x':-.5,'y':0.},'kind':'walker'}
        visible=0 if code=='S10' and 41<=i<=49 else 20 if code=='S09' and 20<=i<=25 else 200
        inst={'target':{'pixels':visible,'bbox_uv_normalized':[.1,.1,.4,.8], 'non_singleton_components':2,'connected_components':2 if code=='S09' and 20<=i<=25 else 1},
              'secondary':{'pixels':100,'bbox_uv_normalized':[.3,.1,.6,.8] if 20<=i<=25 else [.6,.1,.9,.8]}}
        ego={'x':0.,'y':0.,'z':0.,'yaw':i*.4 if code=='S08' else 0.,'pitch':0.,'roll':0.}
        rows.append({'episode_id':code,'sample_index':i,'time_s':t,'actors':{'target':actor,'secondary':copy.deepcopy(actor)},
                     'wearer_transform':ego,'camera_transform':ego.copy(),'plan_receipt_sha256':'plan','layout_receipt_sha256':'layout',
                     'instance_visibility':inst,'truth':{'current_contact':i in contact_indices,
                     'responsible_assets':['target'] if i in contact_indices else [],'minimum_distance_m':.5,
                     'future_contact_within_horizon':any(0<=j-i<=30 for j in contact_indices)}})
    annex={'target_asset':'target','secondary_asset':'secondary'}
    if code=='S02':annex['removal_windows']=[[20]]
    if code=='S03':annex['removal_windows']=[[12,13],[20,21,22],list(range(30,36))]
    if code=='S07':
        body={'schema_version':'dtr-c2-plan-receipt-v1','plan_id':'p','session_id':'s','issued_at_s':0.,'expires_at_s':10.,
              'coordinate_frame':'LAYOUT_FORWARD_RIGHT','time_parameterized_waypoints':[
                  {'time_s':0.,'forward_m':0.,'right_m':0.}, {'time_s':1.,'forward_m':1.,'right_m':0.},
                  {'time_s':2.,'forward_m':2.,'right_m':1.}]}
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest().upper()
        annex['issued_plan_receipt']=dict(body,receipt_sha256=digest)
        for r in rows:r['plan_receipt_sha256']=digest
    reference=copy.deepcopy(rows) if code in ('S09','S10') else None
    if reference:
        for r in reference:r['instance_visibility']['target']['pixels']=100
    return rows,copy.deepcopy(rows),copy.deepcopy(rows),annex,reference


class SourceGateTests(unittest.TestCase):
    def evaluate(self,code,parts):
        return ev.evaluate_episode(PROTOCOL,IDS[code],*parts)

    def test_all_ten_source_semantic_fixtures(self):
        for code in IDS:
            with self.subTest(code=code):
                r=self.evaluate(code,fixture(code))
                self.assertEqual(r['status'],ev.PASS,r)
                self.assertFalse(r['method_predictions_opened'])

    def test_removed_indices_need_positive_truth_and_adjacent_pixels(self):
        rows,ins,wit,ann,ref=fixture('S02')
        ann['removal_windows']=[[90]]
        self.assertEqual(self.evaluate('S02',(rows,ins,wit,ann,ref))['status'],ev.FAIL)
        ann['removal_windows']=[[20]];ins[19]['instance_visibility']['target']['pixels']=0
        self.assertEqual(self.evaluate('S02',(rows,ins,wit,ann,ref))['status'],ev.FAIL)

    def test_multi_drop_lengths_are_fixed_and_separate(self):
        parts=list(fixture('S03'));parts[3]['removal_windows']=[[20],[22,23],list(range(30,36))]
        self.assertEqual(self.evaluate('S03',parts)['status'],ev.FAIL)

    def test_disjoint_disappearance_and_negative_gap_fails(self):
        parts=list(fixture('S10'))
        for row in parts[1]:row['instance_visibility']['target']['pixels']=0 if 55<=row['sample_index']<=61 else 100
        r=self.evaluate('S10',parts)
        self.assertEqual(r['status'],ev.FAIL)
        self.assertIn('negative_gap_disappearance',r['reason'])

    def test_missing_fragmentation_evidence_not_a_fraction_only_pass(self):
        parts=list(fixture('S09'))
        for row in parts[1]:row['instance_visibility']['target'].pop('connected_components')
        r=self.evaluate('S09',parts)
        self.assertEqual(r['status'],ev.FAIL)
        self.assertIn('connected_components',r['reason'])

    def test_witness_geometry_mismatch_and_missing_fields_fail_closed(self):
        parts=list(fixture('S01'));parts[2][3]['actors']['target']['transform']['x']+=1
        self.assertEqual(self.evaluate('S01',parts)['status'],ev.FAIL)
        parts=list(fixture('S01'));parts[0][3]['truth'].pop('future_contact_within_horizon')
        self.assertEqual(self.evaluate('S01',parts)['status'],ev.FAIL)

    def test_right_censored_negative_tail_is_unknown(self):
        r=self.evaluate('S01',fixture('S01'))
        self.assertGreater(r['metrics']['right_censored_future_frames'],0)
        self.assertEqual(ev.evaluate_roster(PROTOCOL,{}, {})['status'],ev.FAIL)

    def test_route_bend_in_progress_before_contact_counts(self):
        parts=list(fixture('S07'))
        receipt=parts[3]['issued_plan_receipt']
        receipt['time_parameterized_waypoints'][1]['time_s']=2.
        receipt['time_parameterized_waypoints'][2]['time_s']=5.
        body={k:v for k,v in receipt.items() if k!='receipt_sha256'}
        receipt['receipt_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest().upper()
        for stream in parts[:3]:
            for row in stream:row['plan_receipt_sha256']=receipt['receipt_sha256']
        self.assertEqual(self.evaluate('S07',parts)['status'],ev.PASS)

    def test_uniform_score_start_excludes_startup_not_internal_disappearance(self):
        parts=list(fixture('S09'))
        parts[1][0]['instance_visibility']['target']['pixels']=0
        self.assertEqual(self.evaluate('S09',parts)['status'],ev.FAIL)
        parts[3]['score_start_s']=.1
        self.assertEqual(self.evaluate('S09',parts)['status'],ev.PASS)
        parts[1][13]['instance_visibility']['target']['pixels']=0
        self.assertEqual(self.evaluate('S09',parts)['status'],ev.FAIL)

    def test_receipt_geometry_hash_cannot_be_replaced_with_boolean(self):
        parts=list(fixture('S07'));parts[3]['issued_plan_receipt']['receipt_sha256']='wrong'
        self.assertEqual(self.evaluate('S07',parts)['status'],ev.FAIL)


if __name__=='__main__':unittest.main()
