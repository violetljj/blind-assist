"""Focused, read-only-input audit of the consumed sensor-translation experiment.

Loads saved estimators; never calls fit or the experiment runner. Only the audit
JSON is written. Numerical metric reduction and raw decoding are independent of
the experiment's helpers. Forward rendering is replayed, not retrained.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '4')

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import joblib
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
HOST = Path(__file__).resolve().parent
ARMS = ('geometry', 'assumed', 'empirical')
ALL_ARMS = (*ARMS, 'raw_peak')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_metrics(pred, target, width):
    pred, target = np.asarray(pred), np.asarray(target)
    if not target.size:
        return dict(n=0, mae_m=None, median_abs_m=None, p90_abs_m=None,
                    within_one_bin=None, bias_m=None, false_near=None, missed_near=None)
    delta = pred - target
    absolute = np.abs(delta)
    positive = target < 2.0
    return dict(n=int(target.size), mae_m=float(absolute.sum()/target.size),
                median_abs_m=float(np.quantile(absolute, .5)),
                p90_abs_m=float(np.quantile(absolute, .9)),
                within_one_bin=float(np.count_nonzero(absolute <= width)/target.size),
                bias_m=float(delta.sum()/target.size),
                false_near=dict(numerator=int(np.count_nonzero((pred < 2) & ~positive)),
                                denominator=int(np.count_nonzero(~positive))),
                missed_near=dict(numerator=int(np.count_nonzero((pred >= 2) & positive)),
                                 denominator=int(np.count_nonzero(positive))))


def independent_task_metrics(pred, target, weak):
    pred, target, weak = [np.asarray(x, dtype=bool) for x in (pred, target, weak)]
    return dict(n=int(target.size), positive=int(target.sum()), negative=int((~target).sum()),
                hits=int((pred & target).sum()), false_alarms=int((pred & ~target).sum()),
                misses=int((~pred & target).sum()), weak_positive=int(weak.sum()),
                weak_hits=int((pred & weak).sum()))


def equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=0, atol=1e-12))
    return a == b


def independently_decode(path):
    """Decode every valid frame using multiplication by powers of two."""
    values = {k: [] for k in ('H', 'A', 'D', 'S', 'N', 'seq', 'ms')}
    frame_keys = {'D': 'distance_mm', 'S': 'target_status', 'N': 'nb_target'}
    required = dict(hist_raw=(64, 16), hist_scaler=(64, 16),
                    ambient_raw=(64,), ambient_scaler=(64,),
                    distance_mm=(64,), target_status=(64,), nb_target=(64,))
    for line in Path(path).read_bytes().splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('type') != 'cnh_frame':
                continue
            arrays = {key: np.asarray(row[key], dtype=np.float64) for key in required}
            if any(a.shape != required[k] or not np.isfinite(a).all() for k, a in arrays.items()):
                continue
            if any(not isinstance(row[k], int) or isinstance(row[k], bool) for k in ('seq', 'ms')):
                continue
            h = arrays['hist_raw'] * np.exp2(-arrays['hist_scaler'])
            ambient = arrays['ambient_raw'] * np.exp2(-arrays['ambient_scaler'])
            if not np.isfinite(h).all() or not np.isfinite(ambient).all():
                continue
            values['H'].append(h)
            values['A'].append(ambient)
            for key, native in frame_keys.items():
                values[key].append(arrays[native])
            values['seq'].append(row['seq'])
            values['ms'].append(row['ms'])
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
    return {k: np.asarray(v) for k, v in values.items()}


def audit(run, output):
    started = time.perf_counter()
    checks, notes = [], []

    def check(name, condition, detail=None):
        item = dict(name=name, passed=bool(condition))
        if detail is not None:
            item['detail'] = detail
        checks.append(item)
        if not condition:
            raise AssertionError(name)

    payload = dict(status='RUNNING', schema='sensor-translation-independent-audit.v1',
                   checks=checks, evidence_boundaries=notes)
    try:
        result = read(run/'result.json')
        inputs = read(run/'inputs.json')
        source = run/'source.py'
        check('saved_source_hash', digest(source) == result['source_sha256'])
        # Compile the saved source using the original module location, so its
        # repository-relative constants remain correct. __main__ is never used.
        sys.path.insert(0, str(HOST))
        module = types.ModuleType('sensor_translation_saved_for_verification')
        module.__file__ = str(HOST/'sensor_translation_dev_20261015.py')
        exec(compile(source.read_text(encoding='utf-8-sig'), str(source), 'exec'), module.__dict__)
        width = float(result['bin_width_m'])
        check('bin_width_matches_frozen_source', width == module.WIDTH)
        models = {a: joblib.load(run/(a+'.joblib')) for a in ARMS}
        classifiers = {a: joblib.load(run/(a+'-near.joblib')) for a in ARMS}
        nuisance = dict(np.load(run/'nuisance.npz', allow_pickle=False))
        nuisance_receipt = read(run/'nuisance.json')
        check('calibration_roster', nuisance_receipt['fit_segments'] == ['2', '4', '4b'])
        check('pilot_fit_roles', {x['id'] for x in inputs['real'] if x['role'] == 'fit'} == {'2', '4', '4b'})
        check('pilot_check_roles', {x['id'] for x in inputs['real'] if x['role'] == 'check'} == {'1', '3', '5', '6', '7', '8', '9'})
        check('public_manifest_hash', digest(module.PUBLIC/'manifest.json') == inputs['public_selection_sha256'])
        check('real_inventory_hash', digest(module.PILOT/'inventory.json') == inputs['real_inventory_sha256'])
        manifest = read(module.PUBLIC/'manifest.json')['rows']
        train_families = {r['family'] for r in manifest[:6]}
        check_families = {r['family'] for r in manifest[6:]}
        check('public_family_split', len(train_families) == 6 and len(check_families) == 2
              and not train_families.intersection(check_families))
        check('public_roster_identity', [r['id'] for r in manifest] == [r['id'] for r in inputs['public']])
        for receipt in inputs['public']:
            check('public_depth_hash:'+receipt['id'], digest(module.DATA/receipt['depth']) == receipt['depth_sha256'])
            check('public_camera_hash:'+receipt['id'], digest(module.PUBLIC/(receipt['id']+'.npz')) == receipt['camera_sidecar_sha256'])

        cache = {}
        for receipt in inputs['real']:
            sid = receipt['id']
            path = module.PILOT/(sid+'.npz')
            check('pilot_cache_hash:'+sid, digest(path) == receipt['sha256'])
            cache[sid] = dict(np.load(path, allow_pickle=False))
        inventory = {x['id']: x for x in read(module.PILOT/'inventory.json')}
        for sid in ('2', '3'):
            raw_path = ROOT/inventory[sid]['raw_path']
            check('pilot_raw_hash:'+sid, digest(raw_path) == inventory[sid]['raw_sha256'])
            decoded = independently_decode(raw_path)
            for key in ('H', 'A', 'D', 'S', 'N', 'seq', 'ms'):
                check('raw_cache_full:'+sid+':'+key, np.array_equal(decoded[key], cache[sid][key]))

        sentinel = dict(H=np.arange(64*16).reshape(1, 64, 16), A=np.arange(64).reshape(1, 64),
                        D=np.arange(64).reshape(1, 64), S=np.arange(64).reshape(1, 64),
                        N=np.arange(64).reshape(1, 64), seq=np.array([7]))
        mapped = module.model_order(sentinel)
        for key in ('H', 'A', 'D', 'S', 'N'):
            check('all_zone_mapping:'+key, np.array_equal(mapped[key], sentinel[key][:, ::-1]))
        check('zone_mapping_preserves_sequence', np.array_equal(mapped['seq'], sentinel['seq']))
        module.check_mounting()

        block_ids = {'train': set(), 'check': set()}
        residuals = {'train': [], 'check': []}
        ambients = {'train': [], 'check': []}
        for block in nuisance_receipt['nuisance_blocks']:
            sid, start, length, role = block['segment'], block['start'], block['frames'], block['role']
            check('noise_block_role:'+sid+':'+str(start), role == ('check' if (start//8) % 2 else 'train'))
            indices = list(range(start, start+length))
            block_ids[role].update((sid, i) for i in indices)
            h = cache[sid]['H'][indices, ::-1]
            residuals[role].extend(h - np.median(h, axis=0, keepdims=True))
            ambients[role].extend(cache[sid]['A'][indices, ::-1])
        check('noise_blocks_disjoint', not block_ids['train'].intersection(block_ids['check']))
        for role, prefix in (('train', ''), ('check', 'check_')):
            check('noise_bank_exact:'+role, np.array_equal(np.asarray(residuals[role]), nuisance[prefix+'residuals']))
            check('ambient_bank_exact:'+role, np.array_equal(np.asarray(ambients[role]), nuisance[prefix+'ambients']))
        notes.append('Noise blocks are disjoint within the same two consumed recordings; random frame draws preserve spatial/bin patterns, not temporal correlation.')

        recomputed, coverage = {'pilot': {}, 'phone': {}}, {}
        real_input_receipts = []
        prepared = read(module.PHONE/'prepared.json')
        if isinstance(prepared, dict):
            prepared = prepared.get('sessions', prepared.get('rows', prepared))
        phone_ids = [x if isinstance(x, str) else x['session'] for x in prepared]
        check('phone_roster_18', len(phone_ids) == 18 and set(phone_ids) == set(result['phone']))
        real_input_receipts.append(dict(path=str(module.PHONE/'prepared.json'), sha256=digest(module.PHONE/'prepared.json')))
        for group in ('pilot', 'phone'):
            for sid, reported in result[group].items():
                stored = dict(np.load(run/(group+'-'+sid+'.npz'), allow_pickle=False))
                if group == 'pilot':
                    native = cache[sid]
                else:
                    raw_path = module.PHONE/'raw'/sid/'tof.raw.jsonl'
                    native = independently_decode(raw_path)
                    real_input_receipts.append(dict(path=str(raw_path), sha256=digest(raw_path)))
                expected_d = native['D'][:, ::-1]/1000
                mask = ((native['S'][:, ::-1] == 5) & (native['N'][:, ::-1] > 0)
                        & (expected_d >= width) & (expected_d < 4.5))
                check('weak_reference_exact:'+group+':'+sid, np.array_equal(expected_d, stored['weak_reference_m']))
                check('weak_mask_exact:'+group+':'+sid, np.array_equal(mask, stored['weak_valid']))
                check('coverage_exact:'+group+':'+sid,
                      reported['frames'] == len(mask) and reported['all_zone_frames'] == mask.size
                      and reported['weak_valid_zone_frames'] == int(mask.sum())
                      and abs(reported['weak_valid_fraction']-mask.mean()) < 1e-12)
                h = native['H'][:, ::-1].astype(float, copy=True)
                h[..., 0] = -np.inf
                peak = (np.argmax(h, axis=-1)+.5)*width
                check('raw_peak_exact:'+group+':'+sid, np.array_equal(peak, stored['raw_peak']))
                recomputed[group][sid] = {}
                for arm in ALL_ARMS:
                    check('prediction_shape_finite:'+group+':'+sid+':'+arm,
                          stored[arm].shape == mask.shape and np.isfinite(stored[arm]).all())
                    metric = independent_metrics(stored[arm][mask], expected_d[mask], width)
                    check('metric_reduction:'+group+':'+sid+':'+arm, equal(metric, reported['by_arm'][arm]))
                    recomputed[group][sid][arm] = metric
            selected = [s for s in result[group] if group == 'phone' or result[group][s]['role'] == 'check']
            usable = [s for s in selected if recomputed[group][s]['geometry']['n']]
            key = 'phone_macro' if group == 'phone' else 'pilot_check_macro'
            for arm in ALL_ARMS:
                mae = np.mean([recomputed[group][s][arm]['mae_m'] for s in usable])
                check('macro_reduction:'+group+':'+arm,
                      result[key][arm]['segment_count'] == len(usable)
                      and abs(result[key][arm]['mae_m']-mae) < 1e-12)
            coverage[group] = dict(roster=len(selected), evaluable_segments=len(usable),
                                   valid_zone_frames=sum(result[group][s]['weak_valid_zone_frames'] for s in selected),
                                   all_zone_frames=sum(result[group][s]['all_zone_frames'] for s in selected))

        specimen = {k: v[:2].copy() for k, v in module.model_order(cache['3']).items()}
        base_features = module.features(specimen['H'], specimen['A'])
        for b0 in (-1e12, 1e12):
            changed = specimen['H'].copy()
            changed[..., 0] = b0
            check('bin0_feature_invariance:'+str(b0), np.array_equal(base_features, module.features(changed, specimen['A'])))
        original_report, original_pred = module.evaluate(models, specimen)
        invalid = {**specimen, 'D': np.full_like(specimen['D'], -1),
                   'S': np.full_like(specimen['S'], 5), 'N': np.ones_like(specimen['N'])}
        invalid_report, invalid_pred = module.evaluate(models, invalid)
        check('all_invalid_D_zero_evaluable', invalid_report['weak_valid_zone_frames'] == 0)
        for arm in ALL_ARMS:
            check('invalid_D_not_FREE:'+arm, invalid_report['by_arm'][arm]['n'] == 0
                  and invalid_report['by_arm'][arm]['false_near'] is None
                  and invalid_report['by_arm'][arm]['missed_near'] is None)
            check('D_poison_prediction_invariance:'+arm, np.array_equal(original_pred[arm], invalid_pred[arm]))
        poison = {**specimen, 'D': np.full_like(specimen['D'], 3500),
                  'S': np.zeros_like(specimen['S']), 'N': np.zeros_like(specimen['N'])}
        _, poison_pred = module.evaluate(models, poison)
        for arm in ALL_ARMS:
            check('DSN_poison_prediction_invariance:'+arm, np.array_equal(original_pred[arm], poison_pred[arm]))
        zero_features = module.features(np.zeros((64, 16)), np.zeros(64))
        payload['zero_input_predictions_m'] = {a: np.quantile(m.predict(zero_features), [0, .5, 1]).tolist()
                                               for a, m in models.items()}
        notes.append('Saved regressors always emit a distance, including zero/no-support input. n=0 is NOT_EVALUABLE, not FREE; no deployment or UNKNOWN policy is validated.')

        rows, _ = module.geometry()
        check('generated_family_roles', {r['family'] for r in rows if r['role'] == 'train'} == train_families
              and {r['family'] for r in rows if r['role'] == 'check'} == check_families)
        for arm in ARMS:
            h, ambient = module.render(rows[0], arm, nuisance, np.random.default_rng(987), check_noise=True)
            check('render_float_finite:'+arm, h.dtype.kind == 'f' and ambient.dtype.kind == 'f'
                  and np.isfinite(h).all() and np.isfinite(ambient).all())
            # This assignment is the exact operation that failed for run-b.
            h[:, 0] = -np.inf
            check('render_accepts_peak_exclusion:'+arm, np.isneginf(h[:, 0]).all())
        for arm in ARMS:
            check('training_counts:'+arm,
                  result['training'][arm]['samples'] == 8*sum(np.isfinite(r['y']).sum() for r in rows if r['role'] == 'train')
                  and result['training'][arm]['near_task_samples'] == 8*sum(r['task_valid'].sum() for r in rows if r['role'] == 'train')
                  and result['training'][arm]['near_task_positive'] == 8*sum((r['task_valid'] & r['near_target']).sum() for r in rows if r['role'] == 'train'))
        check_rows = [r for r in rows if r['role'] == 'check']
        payload['public_denominators_before_noise_repeats'] = dict(
            families=len(check_families), source_frames=len(check_families),
            zone_scale_regression=sum(int(np.isfinite(r['y']).sum()) for r in check_rows),
            zone_scale_task=sum(int(r['task_valid'].sum()) for r in check_rows),
            zone_scale_positive=sum(int((r['task_valid'] & r['near_target']).sum()) for r in check_rows),
            zone_scale_weak=sum(int(r['weak_near'].sum()) for r in check_rows), noise_repeats=4)
        notes.append('Public check is two source families/frames, six scale variants and four noise repeats; zone-scale/repeat counts are not independent scenes. Weak near presence is not body/head collision truth.')

        # Observe the actual producer evaluation function: every estimator must
        # receive identical input bytes for each domain/row/repeat.
        seen = {kind: {a: [] for a in ARMS} for kind in ('regression', 'classifier')}
        class Spy:
            def __init__(self, arm, kind):
                self.arm, self.kind = arm, kind
            def remember(self, x):
                seen[self.kind][self.arm].append(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
            def predict(self, x):
                self.remember(x)
                return np.zeros(len(x))
            def predict_proba(self, x):
                self.remember(x)
                return np.tile([.5, .5], (len(x), 1))
        module.public_checks({a: Spy(a, 'regression') for a in ARMS},
                             {a: Spy(a, 'classifier') for a in ARMS}, check_rows[:1], nuisance)
        for kind in seen:
            check('producer_shared_domain_inputs:'+kind,
                  len(seen[kind]['geometry']) == 12
                  and seen[kind]['geometry'] == seen[kind]['assumed'] == seen[kind]['empirical'])
        check('regressor_classifier_same_features', seen['regression'] == seen['classifier'])

        # Replay saved models on the declared shared target-domain inputs, but
        # reduce every metric with the independent functions above.
        for domain in ARMS:
            rng = np.random.default_rng(module.SEED+1)
            predictions = {a: [] for a in ALL_ARMS}
            decisions = {a: [] for a in ALL_ARMS}
            targets, labels, weak = [], [], []
            for row in check_rows:
                for _ in range(4):
                    h, ambient = module.render(row, domain, nuisance, rng, check_noise=True)
                    x = module.features(h, ambient)
                    valid, task = np.isfinite(row['y']), row['task_valid']
                    hp = h.copy(); hp[:, 0] = -np.inf
                    peak = (np.argmax(hp, -1)+.5)*width
                    targets.extend(row['y'][valid]); labels.extend(row['near_target'][task]); weak.extend(row['weak_near'][task])
                    predictions['raw_peak'].extend(peak[valid]); decisions['raw_peak'].extend((peak < 2)[task])
                    for arm in ARMS:
                        predictions[arm].extend(models[arm].predict(x)[valid])
                        decisions[arm].extend((classifiers[arm].predict_proba(x)[:, 1] >= .5)[task])
            for arm in ALL_ARMS:
                check('public_regression_replay:'+domain+':'+arm,
                      equal(independent_metrics(predictions[arm], targets, width), result['public_check'][domain][arm]))
                check('public_task_replay:'+domain+':'+arm,
                      equal(independent_task_metrics(decisions[arm], labels, weak), result['public_near_task'][domain][arm]))
        task_rows = read(run/'public-task-rows.json')
        for arm in ARMS:
            subset = [r for r in task_rows if r['arm'] == arm]
            sums = {k: sum(r[k] for r in subset) for k in ('n', 'positive', 'negative', 'hits', 'false_alarms', 'misses', 'weak_positive', 'weak_hits')}
            check('public_task_details_sum:'+arm, equal(sums, result['public_near_task']['empirical'][arm]))

        payload.update(status='PASS', coverage=coverage, raw_input_receipts=real_input_receipts,
                       source_sha256=digest(source), verifier_sha256=digest(__file__),
                       run_result_sha256=digest(run/'result.json'))
    except Exception as exc:
        payload.update(status='FAIL', error=dict(type=type(exc).__name__, message=str(exc)))
    payload.update(assertions=len(checks), passed=sum(x['passed'] for x in checks), seconds=time.perf_counter()-started)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({k: payload[k] for k in ('status', 'assertions', 'passed', 'seconds')}, ensure_ascii=False))
    if payload['status'] != 'PASS':
        print(payload.get('error'))
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or args.run/'verification.json'
    if target.exists():
        raise FileExistsError('Preserve the prior audit; choose another --output')
    audit(args.run, target)
