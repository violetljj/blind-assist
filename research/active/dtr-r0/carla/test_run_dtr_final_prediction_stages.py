"""Mock-only ordering tests: no real prediction, truth loading or model fit."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import run_dtr_final_prediction_stages as runner


class StageOrderTests(unittest.TestCase):
    def test_all_prediction_seals_precede_access_and_final_scoring(self):
        parent=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='stage-order-',dir=parent) as temp:
            dest=Path(temp);events=[]
            prepared={g:{'contract':SimpleNamespace(episodes=[SimpleNamespace(episode_id='ep')],calibration=None),
                         'values':{'ep':[]},'annex':{},'entry':{'group':g}} for g in runner.GROUPS}
            manifest={'groups':{g:{'truth_episodes':{}} for g in runner.GROUPS}}
            def read_truth(root,entry):
                group=entry['group'];events.append('truth:'+group)
                self.assertTrue(all((dest/(g+'-nonlearned.json')).is_file() for g in runner.GROUPS))
                if group=='FIT_ONLY':self.assertTrue((dest/'FIT_ONLY-access.json').is_file())
                else:
                    self.assertTrue((dest/'FINAL-access.json').is_file())
                    self.assertTrue(all((dest/(g+'-all-arms.json')).is_file() for g in ('FINAL_A','FINAL_B')))
                return {}
            def fit(*args):events.append('fit');return {'mock':True}
            def score(*args):events.append('score');return {'mock':True}
            with mock.patch('dtr_carla_raw_kalman_baseline.predict_episode',return_value={}), \
                 mock.patch('dtr_final_classic_prediction_adapter.predict_episode',return_value={}), \
                 mock.patch('dtr_final_structural_prediction_adapter.predict_episode',return_value={}), \
                 mock.patch('dtr_final_prediction_stages.merge_nonlearned',return_value={'mock':True}), \
                 mock.patch('dtr_final_prediction_stages.fit_models',side_effect=fit), \
                 mock.patch('dtr_final_prediction_stages.apply_learned',return_value={'frames':[]}), \
                 mock.patch('dtr_final_score_adapter.score_final',side_effect=score), \
                 mock.patch.object(runner,'open_truth',side_effect=read_truth):
                result=runner.execute_phases(dest,dest,manifest,{}, {},prepared)
            self.assertEqual(events,['truth:FIT_ONLY','fit','truth:FINAL_A','truth:FINAL_B','score'])
            self.assertTrue(Path(result['path']).is_file())

    def test_existing_execution_cannot_resume(self):
        parent=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        with tempfile.TemporaryDirectory(prefix='stage-noresume-',dir=parent) as temp:
            root=Path(temp);(root/'prediction-stages').mkdir()
            with self.assertRaises(FileExistsError):runner.run(root)


if __name__=='__main__':unittest.main()
