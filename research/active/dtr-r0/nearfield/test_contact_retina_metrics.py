"""Synthetic evaluator checks; never opens captured labels."""
import copy
import unittest

from evaluate_contact_retina import (ARMS, SEEDS, add_ensembles, evaluate, matrix,
    select_thresholds, threshold, validate_predictions)


class ContactMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples = [dict(sample_id=k, clip_id=k, group_id=k, split=split)
                        for k, split in (('tr', 'train'), ('v', 'val'), ('t', 'test'))]
        self.zero = [[0, 0, 0], [0, 0, 0]]
        self.targets = {s['sample_id']: copy.deepcopy(self.zero) for s in self.samples}
        self.rows = {s['sample_id']: dict(scores=[[.2]*3, [.2]*3]) for s in self.samples}

    def test_threshold_only_validation_negatives(self):
        self.rows['tr']['scores'] = [[.99]*3]*2
        self.rows['t']['scores'] = [[1.]*3]*2
        self.targets['v'] = [[0, 0, 1], [0, 0, 0]]
        self.rows['v']['scores'][0][2] = .95
        selected = threshold(self.rows, self.samples, self.targets)
        self.assertGreater(selected, .2)
        self.assertLess(selected, .20000000001)
        self.targets['t'] = [[1]*3]*2
        self.assertEqual(selected, threshold(self.rows, self.samples, self.targets))

    def test_repeat_threshold_reused_including_ensemble(self):
        predictions = {(a, seed): copy.deepcopy(self.rows) for a in ARMS for seed in SEEDS}
        for (arm, _), rows in predictions.items():
            if arm.endswith('_repeated_history'):
                rows['v']['scores'] = [[.9]*3]*2
        add_ensembles(predictions, self.samples)
        values = select_thresholds(predictions, self.samples, self.targets)
        for seed in (*SEEDS, 'ensemble'):
            self.assertEqual(values['temporal_structure', seed], values['temporal_structure_repeated_history', seed])
            self.assertLess(values['temporal_structure_repeated_history', seed], .3)

    def test_wrong_part_and_premature_alert_on_positive_clip(self):
        self.targets['t'] = [[0, 0, 1], [0, 0, 0]]
        self.rows['t']['scores'] = [[.9]*3, [.9]*3]
        result = evaluate(self.rows, self.samples, self.targets, {'t': [2.5, None]}, .5)
        self.assertEqual(result['recovered'], 1)
        self.assertEqual(result['all_false_alert_clips'], ['t'])
        self.assertEqual(result['pure_negative_false_alert_clips'], [])
        self.assertEqual(result['cells']['HEAD@3s']['FP'], 1)
        self.assertEqual(result['cells']['BODY@1s']['FP'], 1)
        self.assertEqual(result['cells']['BODY@3s']['TP'], 1)
        self.assertEqual(result['clips']['t']['first_alarm']['horizon_s'], 1)
        self.assertFalse(result['clips']['t']['first_alarm']['correct'])
        self.assertAlmostEqual(result['cells']['BODY@3s']['Brier_known_cells'], .01)
        self.assertEqual(result['simulated_lead_s_mean'], 2.5)

    def test_missed_lead_is_null_unknown_not_tn(self):
        self.targets['t'] = [[0, 0, 1], [0, 0, 0]]
        self.rows['t']['states'] = [['UNKNOWN']*3, ['KNOWN']*3]
        result = evaluate(self.rows, self.samples, self.targets, {'t': [2.5, None]}, .5)
        self.assertEqual(result['recovered'], 0)
        self.assertIsNone(result['simulated_lead_s_mean'])
        self.assertIsNone(result['clips']['t']['parts']['BODY']['simulated_lead_s'])
        cell = result['cells']['BODY@3s']
        self.assertEqual((cell['UNKNOWN'], cell['unknown_positive'], cell['TN'], cell['FN']), (1, 1, 0, 0))
        self.assertEqual(cell['missed_or_unknown_positive'], 1)
        self.assertEqual(cell['all_opportunity_recall'], 0)
        self.assertIsNone(cell['Brier_known_cells'])

    def test_earliest_warning_uses_simulated_time(self):
        samples = [dict(sample_id=k, clip_id='episode', group_id='g', split='test') for k in ('late', 'early')]
        rows = {k: dict(scores=[[0, 0, 1], [0, 0, 0]]) for k in ('late', 'early')}
        targets = {k: [[0, 0, 1], [0, 0, 0]] for k in rows}
        result = evaluate(rows, samples, targets, {'late': [2., None], 'early': [2.8, None]}, .5,
                          {'late': 1.5, 'early': .7})
        self.assertEqual(result['simulated_lead_s_mean'], 2.8)

    def test_coverage_split_and_monotonicity_rejected(self):
        payload = dict(rows=[dict(arm='mde_contact', seed=0, sample_id=s['sample_id'], split=s['split'], scores=self.zero)
                             for s in self.samples])
        validate_predictions(payload, self.samples, False)
        with self.assertRaises(ValueError):
            validate_predictions(dict(rows=payload['rows'][:-1]), self.samples, False)
        payload['rows'][0]['split'] = 'test'
        with self.assertRaises(ValueError):
            validate_predictions(payload, self.samples, False)
        with self.assertRaises(ValueError):
            matrix([[.8, .2, .9], [0, 0, 0]])
        with self.assertRaises(ValueError):
            matrix([[0, 0, float('nan')], [0, 0, 0]])

    def test_baseline_vocabulary_unknown_requires_zero(self):
        payload = dict(rows=[dict(arm='mde_contact', seed=0, sample_id=s['sample_id'], split=s['split'],
            scores=[[0, 0, 1], [0, 0, 0]], states=[['UNKNOWN', 'NO_NEAR_OBSERVED', 'OBSTACLE'], ['UNKNOWN']*3])
            for s in self.samples])
        validate_predictions(payload, self.samples, False)
        payload['rows'][0]['states'][0][2] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            validate_predictions(payload, self.samples, False)


if __name__ == '__main__':
    unittest.main()
