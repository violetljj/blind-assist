"""Synthetic integration checks only; no pixels, predictions on source, or fit."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import dtr_final_classic_prediction_adapter as integration


def fixture(count=6):
    observations = tuple(SimpleNamespace(sample_index=i, time_s=i/10,
        world_frame=100+i, navigation_session_id='synthetic') for i in range(count))
    episode = SimpleNamespace(episode_id='synthetic', route_frame=None, observations=observations)
    candidates = [{'candidates': [{'x':3.-.1*i, 'y':.02*i*i}]} for i in range(count)]
    return episode, candidates


def measurement_stub(observation, candidate, calibration, anchor):
    return [integration.x24.Measurement(0,'person',.9,(.1,.1,.2,.3),
             np.asarray([row['x'],row['y']]),10) for row in candidate['candidates']]


class ClassicPredictionAdapterTests(unittest.TestCase):
    def predict(self, episode, candidates):
        with patch.object(integration.x24,'candidate_measurements',side_effect=measurement_stub), \
             patch.object(integration.x24,'wearer_anchor_state',return_value=((0.,0.),(1.,0.))), \
             patch.object(integration.x24,'load_receipt',return_value=None), \
             patch.object(integration.x24,'predict_episode',side_effect=AssertionError('derived tracks forbidden')):
            return integration.predict_episode(episode,candidates,None)

    def test_shape_and_causal_turn_history(self):
        episode,candidates=fixture()
        result=self.predict(episode,candidates)
        self.assertEqual(len(result['frames']),6)
        self.assertEqual(len(result['tiny_feature_names']),8)
        for frame in result['frames']:
            self.assertEqual(set(frame['arms']),set(integration.ARMS))
            self.assertEqual(len(frame['tiny_features']),8)
            self.assertTrue(np.isfinite(frame['tiny_features']).all())
            self.assertEqual(frame['tiny_status'],'FEATURES_ONLY_NOT_FITTED')
        self.assertTrue(any(abs(v)>0 for v in result['frames'][4]['causal_yaw_rates_rad_s'].values()))
        self.assertFalse(any(result['frames'][1]['causal_yaw_rates_rad_s'].values()))

    def test_suffix_changes_and_prefix_calls_do_not_change_history(self):
        episode,candidates=fixture()
        full=self.predict(episode,candidates)
        prefix=SimpleNamespace(episode_id=episode.episode_id,route_frame=None,
                               observations=episode.observations[:4])
        self.assertEqual(full['frames'][:4], self.predict(prefix,candidates[:4])['frames'])
        altered=candidates[:4]+[{'candidates':[{'x':90.,'y':80.}]}]*2
        self.assertEqual(full['frames'][:4], self.predict(episode,altered)['frames'][:4])
        self.assertEqual(full,self.predict(episode,candidates))

    def test_fd_does_not_emit_on_missing_measurement(self):
        episode,candidates=fixture()
        candidates[-1]={'candidates':[]}
        frame=self.predict(episode,candidates)['frames'][-1]
        self.assertEqual(frame['fd_tracks'],[])
        self.assertFalse(frame['arms'][integration.classic.ARM_FINITE_DIFFERENCE_CV]['route_risk'])
        self.assertTrue(frame['tracks'])
        self.assertEqual(frame['tiny_features'][6],0.)
        self.assertTrue(all(t['disposition']=='PREDICTED_HOLD' for t in frame['tracks']))

    def test_bad_count_and_reversed_time_rejected(self):
        episode,candidates=fixture()
        with self.assertRaisesRegex(Exception,'candidate_count'):
            self.predict(episode,candidates[:-1])
        episode.observations=tuple(reversed(episode.observations))
        with self.assertRaisesRegex(Exception,'time_order'):
            self.predict(episode,candidates)

    def test_kalman_tracks_match_locked_raw_predictor(self):
        episode,candidates=fixture()
        result=self.predict(episode,candidates)
        with patch.object(integration.x24,'candidate_measurements',side_effect=measurement_stub), \
             patch.object(integration.x24,'wearer_anchor_state',return_value=((0.,0.),(1.,0.))), \
             patch.object(integration.x24,'load_receipt',return_value=None):
            baseline=integration.kalman.predict_episode(episode,candidates,None)
        for frame,reference in zip(result['frames'],baseline['frames']):
            self.assertEqual(frame['tracks'],reference['tracks'])
            for index,arm in enumerate((integration.kalman.ARM_RADIAL,integration.kalman.ARM_ROUTE)):
                entry=reference['arms'][arm]['minimum_entry_s']
                self.assertEqual(frame['tiny_features'][index], 4. if entry is None else entry)


if __name__=='__main__':
    unittest.main()
