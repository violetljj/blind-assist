import unittest
import dtr_final_score_adapter as adapter


class ScoreAdapterTest(unittest.TestCase):
    def fixture(self):
        truth=[]; predictions=[]
        for i in range(61):
            t=i/10
            positive=1<=t<=4
            truth.append({'sample_index':i,'time_s':t,'truth':{
                'future_contact_within_horizon':positive,'current_contact':t==4,
                'realized_time_to_contact_seconds':4-t if positive else None}})
            predictions.append({'sample_index':i,'time_s':t,'arms':{'a':{'route_risk':None}}})
        return truth,predictions,{'score_start_s':.1,'score_end_s':3.}

    def test_tail_contact_support_survives_prefix_and_unknown_prediction(self):
        truth,pred,annex=self.fixture()
        rows=adapter.aligned_rows(truth,pred,annex,'a')
        value=adapter.metrics.evaluate_episode(rows)
        self.assertEqual(1,value['truth_event_count'])
        self.assertEqual(1,value['event_fn'])
        self.assertEqual(4,rows[-1]['truth_contact_time_s'])
        self.assertEqual(.1,rows[0]['time_s'])

    def test_missing_arm_or_identity_is_not_silent_abstention(self):
        truth,pred,annex=self.fixture()
        with self.assertRaises(KeyError):adapter.aligned_rows(truth,pred,annex,'missing')
        pred[2]['sample_index']=900
        with self.assertRaises(ValueError):adapter.aligned_rows(truth,pred,annex,'a')

    def test_fit_group_is_rejected(self):
        with self.assertRaises(ValueError):adapter.score_final({}, {'FIT_ONLY':{}},{},{})


if __name__=='__main__':unittest.main()
