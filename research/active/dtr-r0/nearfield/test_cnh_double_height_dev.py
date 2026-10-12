"""No rendering: physical variants, frozen routing and causal event bookkeeping."""
import copy
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cnh_double_height_dev as D
import cnh_double_height_geometry_dev as G


def threshold_fixture():
    return {'fold_records': [dict(fold=f, key=f'in_domain/calibrated/{cap}/{method}',
                                  threshold=.2 + .1*f if method == 'pair_only' else 2.+f)
                            for f in range(3) for cap in ('0.025', '0.050')
                            for method in ('pair_only', 'original_center')]}


class DoubleHeightTests(unittest.TestCase):
    def test_thresholds_select_own_fold_method_and_exact_calibrated_caps(self):
        result = threshold_fixture()
        result['fold_records'].extend([
            dict(fold=0, key='in_domain/matched_eval_descriptive/0.025/pair_only', threshold=999.),
            dict(fold=0, key='none_to_corner/calibrated/0.025/pair_only', threshold=998.),
        ])
        ts = D.thresholds(result)
        self.assertEqual(12, len(ts))
        self.assertAlmostEqual(.4, ts[2, '0.025', 'pair_only'])
        self.assertEqual(4., ts[2, '0.050', 'original_center'])
        duplicate = threshold_fixture()
        duplicate['fold_records'].append(duplicate['fold_records'][0].copy())
        with self.assertRaises(ValueError):
            D.thresholds(duplicate)
        for replacement in ('in_domain/calibrated/0.100/pair_only',
                            'in_domain/calibrated/0.05/pair_only'):
            bad = threshold_fixture()
            bad['fold_records'][0]['key'] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                D.thresholds(bad)

    def test_deadline_uses_current_output_only_and_accounts_for_warmup(self):
        score = np.zeros(13)
        score[1] = 1.
        score[8:] = 1.
        state = D.decisions(score, .5, 7)
        self.assertTrue(state['timely'])
        self.assertFalse(state['timely_excluding_warmup'])
        self.assertEqual(4, state['false_alarm_intervals'])
        self.assertEqual(1, state['warmup_alarm_frames'])
        score[7] = .5  # alarm comparison is >= and includes the deadline frame.
        self.assertTrue(D.decisions(score, .5, 7)['timely_excluding_warmup'])
        score[:2] = 0.
        score[7] = 0.
        self.assertFalse(D.decisions(score, .5, 7)['timely'])

    def test_clear_without_contact_deadline_still_has_false_alarm_denominator(self):
        score = np.zeros(13)
        score[2:12] = 1.
        state = D.decisions(score, .5, None)
        self.assertFalse(state['timely'])
        self.assertFalse(state['timely_excluding_warmup'])
        self.assertEqual(10, state['false_alarm_intervals'])

    def test_physical_variants_are_independent_single_boxes_and_share_deadline(self):
        scene = {'boxes': [{'lo': [-.1, -.1, 2.6], 'hi': [.1, .26, 2.7], 'reflectance': .6}]}
        saved = copy.deepcopy(scene)
        travel = np.repeat(np.eye(4)[None], 16, axis=0)
        travel[:, 2, 3] = np.arange(16) * .16
        checked = G.validate_candidate(scene, travel, 'contact')
        self.assertTrue(checked['valid'], checked['reasons'])
        self.assertEqual(saved, scene)
        variants = checked['variants']
        for name, contacts in [('H', [True, False]), ('B', [False, True]), ('HB', [True, True])]:
            v = variants[name]
            self.assertEqual(1, len(v['boxes']))
            self.assertEqual(contacts, v['contact_query'])
            self.assertTrue(v['physical_contact'])
            box = v['boxes'][0]
            self.assertEqual([-.1, 2.6], [box['lo'][0], box['lo'][2]])
            self.assertEqual([.1, 2.7], [box['hi'][0], box['hi'][2]])
            self.assertEqual(.6, box['reflectance'])
        self.assertEqual(1, len({v['deadline_index'] for v in variants.values()}))
        fractions = [v['reference_fraction'] for v in variants.values()]
        np.testing.assert_allclose(fractions, fractions[0], atol=1e-10, rtol=0)
        self.assertEqual(int(np.floor(fractions[0]))-3, variants['HB']['deadline_index'])
        variants['HB']['boxes'][0]['lo'][0] = -99.
        self.assertEqual(-.1, variants['H']['boxes'][0]['lo'][0])
        self.assertEqual(saved, scene)

    def test_freeze_rejects_training_or_calibration_model_for_anchor(self):
        anchor = dict(unit=123, config=0, fold=0)
        folds = [dict(train=[123], calibration=[], evaluation=[]),
                 dict(train=[], calibration=[], evaluation=[123]),
                 dict(train=[], calibration=[123], evaluation=[])]
        def fake_read(path):
            return {'anchors': [anchor], 'hashes': {}} if Path(path).name == 'geometry_manifest.json' else {'folds': folds}
        renderer = SimpleNamespace(source_paths=lambda unit: {})
        with patch.object(D, 'read', side_effect=fake_read), patch.object(Path, 'is_file', return_value=True), \
                patch.dict('sys.modules', {'cnh_double_height_render_dev': renderer}), \
                patch.object(D, 'save') as save, patch.object(D, 'sha', return_value='f'*64):
            with self.assertRaises(AssertionError):
                D.freeze()
            save.assert_not_called()
            anchor['fold'] = 1
            folds[1]['calibration'] = [123]  # Even evaluation membership cannot excuse leakage.
            with self.assertRaises(AssertionError):
                D.freeze()
            save.assert_not_called()

    def test_analysis_routes_fold_preserves_head_body_and_counts_one_hb_event(self):
        anchor = dict(anchor_id='fixture', unit=123, config=0, fold=2, role='contact',
                      family='none', mode=0, turn='none', depth_bin=1,
                      variants={tag: {'deadline_index': 7} for tag in D.TAGS})
        smooth = {tag: np.tile([1.+j, 2.+j], (13, 1)) for j, tag in enumerate(D.TAGS)}
        calls = []
        class Model:
            def __init__(self, fold): self.fold = fold
            def predict_proba(self, x):
                calls.append((self.fold, x.copy()))
                p = np.where(x[:, 0] < x[:, 1], .9, .1)
                return np.column_stack([1-p, p])
        def fake_load(path):
            fold = int(Path(path).name.split('fold')[1][0])
            self.assertTrue(Path(path).name.startswith('in_domain_'))
            return Model(fold)
        def fake_read(path):
            name = Path(path).name
            if name == 'PLAN.json': return {'budget_analysis_cpu_seconds': 600}
            if name == 'gpu_terminal.json': return {'status': 'COMPLETE'}
            if name == 'result.json': return threshold_fixture()
            if name == 'geometry_manifest.json': return {'anchors': [anchor]}
            return {'sha256': 'f'*64}
        saved = {}
        with patch.object(D, 'read', side_effect=fake_read), patch.object(D, 'verify'), \
                patch.object(D, 'sha', return_value='f'*64), \
                patch.object(D, 'save', side_effect=lambda p, v: saved.update({Path(p).name: v})), \
                patch.object(np, 'load', side_effect=lambda p: nullcontext({'smooth': smooth[Path(p).stem]})), \
                patch.object(np, 'savez_compressed'), \
                patch.dict('sys.modules', {'joblib': SimpleNamespace(load=fake_load),
                                         'threadpoolctl': SimpleNamespace(threadpool_limits=lambda **kw: nullcontext())}), \
                redirect_stdout(StringIO()):
            D.analyze()
        self.assertEqual([2, 2, 2], [fold for fold, x in calls])
        for (_, x), tag in zip(calls, D.TAGS):
            np.testing.assert_array_equal(x, smooth[tag])
        result = saved['result.json']
        self.assertEqual(1, result['metrics']['0.025/pair_only/HB/all']['events'])
        self.assertEqual(1, result['metrics']['0.025/pair_only/HB/all']['timely'])
        self.assertEqual(1, result['metrics']['0.025/pair_only/HB/depth/1']['events'])


if __name__ == '__main__':
    unittest.main()
