"""Focused synthetic checks; no experiment outputs are read."""
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cnh_candidate_quality_analysis as quality
import cnh_position_prior_analysis as prior


class QualityAnalysisTests(unittest.TestCase):
    def test_pack_keeps_old_module_unchanged_and_first_hit(self):
        old = prior.ARMS
        y = np.ones((9, 6), int)
        w = np.tile(np.array([.8, 1.3, .7, .6, .5, .4, .3, .3, .3])[:, None], (1, 6))
        s = {a: np.full((9, 6), -np.inf) for a in quality.ARMS}
        s['G0'][:2] = 4.
        sq = dict(unit=96, y=y, w=w, strata=np.full((9, 6), 'tiny'), s=s)
        p = quality.pack([sq], (0, 2, 4))
        self.assertIs(prior.ARMS, old)
        self.assertEqual(set(p['s']), set(quality.ARMS)|{'G4local'})
        metrics = prior.evaluate(p, 'G0', (3.,))['tiny']
        self.assertEqual(metrics['timely']['numerator'], 0)
        self.assertEqual(metrics['late']['numerator'], 3)

    def test_quality_gate_requires_retention_and_false_alert(self):
        results = {}
        for group in prior.GROUPS:
            for arm in quality.POLICIES:
                timely = .4 if arm == 'G0' else .6 if arm == 'G1' else .5
                fa = .121 if arm == 'FA3' else .1
                results[f'{group}/{arm}/0.10'] = {'eval': {
                    'tiny': {'timely': {'rate': timely, 'numerator': int(timely*100), 'denominator': 100}},
                    'all': {'false_alert': {'rate': fa, 'numerator': int(fa*1000), 'denominator': 1000}}}}
        checks = quality.quality_checks(results)
        self.assertTrue(checks['HEAD']['policies']['FA1']['meets_quality_spec'])
        self.assertFalse(checks['HEAD']['policies']['FA3']['meets_quality_spec'])
        results['HEAD/G1/0.10']['eval']['tiny']['timely']['rate'] = .3
        checks = quality.quality_checks(results)
        self.assertIsNone(checks['HEAD']['policies']['FA1']['retention_of_G1_gain'])
        self.assertIsNone(checks['HEAD']['policies']['FA1']['meets_quality_spec'])

    def test_original_incomplete_unit_rejected(self):
        @contextmanager
        def fake_load(*args, **kwargs):
            class Packed(dict):
                files = ['split']
            yield Packed(split=np.array('audit'))
        with patch.object(Path, 'glob', return_value=[Path('unit143.npz')]), patch.object(np, 'load', fake_load):
            with self.assertRaisesRegex(ValueError, 'outside authorized'):
                quality.load('unused', require_complete=False)

    def test_g4prime_fine_baseline_and_disabled_local(self):
        p = dict(y=np.array([[False, False], [True, True]]), w=np.array([[np.nan, np.nan], [1.4, .8]]),
            empty=np.array([True, False]), near=np.array([False, True]),
            s={'G0': np.array([[3.1, 0.], [3.13, 0.]]), 'G4local': np.full((2, 2), -np.inf)})
        self.assertEqual(prior.select(p, 'G4', .1), (3.125, np.inf))

    def test_failed_parity_rejected_before_analysis(self):
        @contextmanager
        def fake_load(*args, **kwargs):
            class Packed(dict):
                files = ['split', 'G0_consistency_max_abs', 'G0_consistency_max_rel']
            yield Packed(split=np.array('calib'), G0_consistency_max_abs=.001, G0_consistency_max_rel=.00002)
        with patch.object(Path, 'glob', return_value=[Path('unit96.npz')]), patch.object(np, 'load', fake_load):
            with self.assertRaisesRegex(ValueError, 'failed G0 reference parity'):
                quality.load('unused', require_complete=False)


if __name__ == '__main__':
    unittest.main()
