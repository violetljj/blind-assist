"""Synthetic source, composite-label, public-contract and decision checks."""
import unittest

import numpy as np

from local_stability_spec import specification,check_spec,planned_bounds
from local_stability_inference import synthetic_checks,validate_identities
from local_stability_metrics import make_rows,stability,coverage,ARMS,AXES
from query_occupancy_data import geometric_labels
from ba_camera_corridor_metrics import evaluate_rows


def fixture(extra_false=0):
    spec = specification()
    ids = [{**c,'id':c['name'],'index':i} for i,c in enumerate(spec['cases'])]
    labels = dict(indices=np.arange(576),classes=np.asarray([geometric_labels(planned_bounds(c))[0] for c in spec['cases']]),
                  valid=np.ones((576,6),bool))
    positive = (labels['classes'][:,[1,4]]<6).any(1)
    base = [dict(alert=bool(positive[i] and c['frame_in_clip']>=5),unknown=True) for i,c in enumerate(spec['cases'])]
    probabilities = dict(raw=np.zeros((576,6)),local=np.zeros((576,6)))
    for i,c in enumerate(spec['cases']):
        if positive[i] and c['frame_in_clip']>=4:probabilities['local'][i,4]=.9
    for i in np.flatnonzero(~positive)[:extra_false]:probabilities['local'][i,4]=.9
    rows = make_rows(ids,base,probabilities,labels,dict(raw=.8,local=.8))
    metrics = evaluate_rows(rows,arms=ARMS)
    strata = {key:{value:evaluate_rows([r for r in rows if r[key]==value],arms=ARMS)
                   for value in {r[key] for r in rows}} for key in AXES}
    return rows,metrics,strata


class StabilityTests(unittest.TestCase):
    def test_source_count_full_composite_and_schedules(self):
        spec = specification();result = check_spec(spec)
        self.assertEqual((result['positive_frames'],result['negative_frames'],result['positive_events']),(224,352,32))
        ids = [{**c,'id':c['name'],'index':i} for i,c in enumerate(spec['cases'])]
        self.assertEqual(validate_identities(ids,set())['frames'],576)

    def test_cube_union_does_not_fill_enclosing_gap(self):
        bounds = [(np.array([.4,0.,2.]),np.array([.6,1.,2.2])),
                  (np.array([-.6,0.,2.]),np.array([.6,.1,2.2]))]
        self.assertEqual(geometric_labels(bounds)[0][1],6)
        hull = [(np.array([-.6,0.,2.]),np.array([.6,1.,2.2]))]
        self.assertLess(geometric_labels(hull)[0][1],6)

    def test_public_identity_contract_and_two_argument_extract(self):
        self.assertEqual(synthetic_checks()['cases'],10)

    def test_early_benefit_passes_without_inventing_new_events(self):
        rows,metrics,strata = fixture();gate = stability(rows,metrics,strata)
        self.assertEqual(gate['status'],'PASS')
        self.assertEqual(gate['timing']['benefit_events'],32)
        self.assertEqual(gate['timing']['recovered_events'],0)
        self.assertEqual(metrics['arms']['local']['detected_events'],32)
        self.assertEqual(coverage(rows,metrics)['local']['internal_gap_frames'],0)
        self.assertTrue(all(r['predictions']['local']['unknown'] for r in rows))

    def test_false_alert_budget_fails_despite_early_benefits(self):
        rows,metrics,strata = fixture(extra_false=3);gate = stability(rows,metrics,strata)
        self.assertEqual(gate['status'],'FAIL');self.assertFalse(gate['costs_pass'])
        self.assertEqual(gate['timing']['benefit_events'],32)

    def test_source_admission_failure_is_not_model_failure(self):
        rows,metrics,strata = fixture()
        self.assertEqual(stability(rows,metrics,strata,False)['status'],'NOT_EVALUABLE')


if __name__=='__main__':unittest.main()
