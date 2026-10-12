"""Independent per-height joint-score replay on consumed Development.

Loads only exact-hash locally generated HGB models; reconstructs score inputs,
fit eligibility/weights, separated cutoffs, grades and complete event accounting.
The pure runtime peak descriptors have a separate feature audit.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import audit_cnh_graded_corridor_eval_dev as A
import audit_cnh_graded_peak_alert_dev as P

OUT = A.ROOT / 'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
PARAMS = dict(learning_rate=.1, max_iter=100, max_leaf_nodes=7, max_depth=3,
              min_samples_leaf=50, l2_regularization=1., early_stopping=False,
              random_state=20261010)
BANKS = ['score_only', 'score_current', 'score_temporal']
BUDGETS = [[5, 32], [15, 64], [30, 128]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spatial(split, ids):
    with np.load(OUT / 'features' / f'{split}_features.npz', allow_pickle=False) as a:
        np.testing.assert_array_equal(a['scene_ids'], ids)
        np.testing.assert_array_equal(a['frames'], np.arange(3, 16))
        np.testing.assert_array_equal(a['queries'], ['HEAD', 'BODY'])
        values = {}
        for name, count in (('current', 22), ('temporal', 57)):
            x = a[name + '_features'].astype(np.float64)
            valid = a[name + '_valid'] & np.isfinite(x)
            assert x.shape == (384, 4, 13, 2, count)
            values[name] = np.concatenate((np.where(valid, x, np.nan),
                                          np.logical_not(valid).astype(float)), -1)
        return values


def comparison(before, after, category):
    pure_clear = (category == 'clear').all(-1)
    pure_pass = (category == 'pass').any(-1) & ~(category == 'contact').any(-1)
    costs = {}
    for name, mask in (('clear', pure_clear), ('pass', pure_pass)):
        costs[name] = {}
        for kind in ('any', 'light'):
            before_flags = (before > 0).any(-1) if kind == 'any' else before.max(-1) == 1
            after_flags = (after > 0).any(-1) if kind == 'any' else after.max(-1) == 1
            b, a = A.costs(before_flags[mask]), A.costs(after_flags[mask])
            costs[name][kind] = dict(before=b, after=a,
                                     delta={key: a[key]-b[key] for key in
                                            ('slots', 'clips', 'segments', 'longest_run_frames')})
    return dict(paired_timely=A.paired(before > 0, after > 0, category),
                before=A.basic(before > 0, category), after=A.basic(after > 0, category),
                costs=costs)


def run():
    began = time.monotonic()
    target = OUT / 'audit'
    target.mkdir(exist_ok=True)
    if (target / 'result.json').exists():
        raise FileExistsError('Preserve prior joint-score audit')
    plan = A.read(OUT / 'PLAN.json')
    A.eq(plan['params'], PARAMS, 'HGB fixed params')
    A.eq(plan['banks'], BANKS, 'fixed banks')
    A.eq(plan['budgets'], BUDGETS, 'fixed budgets')
    A.eq(plan['dimensions'], dict(score_only=3, score_current=47, score_temporal=117), 'dimensions')
    A.eq(sha(OUT / 'TASK.json'), plan['task_record_sha256'], 'task hash')
    for path, digest in plan['inputs_sha256'].items():
        A.eq(sha(A.ROOT / path), digest, 'input hash')
    producer = Path(__file__).with_name('cnh_graded_peak_joint_dev.py')
    A.eq(sha(producer), plan['source_sha256'], 'producer hash')
    tree = ast.parse(producer.read_text(encoding='utf8'))
    for name, expected in (
        ('fit_height', ["tensor['cal'][bank]", "data['cal']['category']",
                        "refs['cal']['baseline']", 'fit_mask', 'q']),
        ('cutoff', ["score['cal']", "eligible['cal']", "data['cal']['category']",
                    '~fit_mask', 'cc', 'pc'])):
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == name]
        assert len(calls) == 1
        A.eq([ast.unparse(arg) for arg in calls[0].args], expected, name + '/cal-only')
    trainer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit_height')
    trainer_fit = [n for n in ast.walk(trainer) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit']
    assert len(trainer_fit) == 1
    A.eq([ast.unparse(arg) for arg in trainer_fit[0].args], ['x[mask, ..., q, :][eligible]', 'y'], 'fit matrix')
    A.eq({n.arg: ast.unparse(n.value) for n in trainer_fit[0].keywords}, {'sample_weight': 'weight'}, 'fit weight input')
    ds = A.load();cal = ds['cal']
    families = {}
    for row in cal['rows']:
        families.setdefault(row['background_family'], set()).add(row['background_id'])
    assert all(len(ids) == 2 for ids in families.values())
    fit_ids = {min(ids) for ids in families.values()}
    cutoff_ids = {row['background_id'] for row in cal['rows']} - fit_ids
    validation_ids = {row['background_id'] for row in ds['validation']['rows']}
    assert not (cutoff_ids & fit_ids or validation_ids & (cutoff_ids | fit_ids))
    A.eq(A.read(OUT / 'cal_partition.json'),
         dict(fit_background_ids=sorted(fit_ids), calibrate_background_ids=sorted(cutoff_ids)), 'cal partition')
    fit_mask = np.array([row['background_id'] in fit_ids for row in cal['rows']])
    cut_mask = ~fit_mask
    descriptors = {split: spatial(split, d['ids']) for split, d in ds.items()}
    thresholds = A.read(A.ROOT / 'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    cuts = A.read(OUT / 'calibrations.json');metrics = A.read(OUT / 'metrics.json')
    records = A.read(OUT / 'models.json')
    saved, score_saved = {}, {}
    for split in ds:
        with np.load(OUT / f'{split}_grades.npz', allow_pickle=False) as a:
            saved[split] = dict(zip(a['keys'].tolist(), a['grades']))
        with np.load(OUT / f'{split}_scores.npz', allow_pickle=False) as a:
            score_saved[split] = dict(zip(a['keys'].tolist(), a['scores']))
    reports, bases, firsts, details, model_checks = {}, {}, {}, {}, {}
    for seed in A.SEEDS:
        baseline, refs, tensors = {}, {}, {}
        for split, d in ds.items():
            th = thresholds[str(seed)]
            ordinary = A.smooth(d['raw'][seed])
            base = A.score_features(d['raw'][seed], th['single'])
            tensors[split] = dict(score_only=base,
                                 score_current=np.concatenate((base, descriptors[split]['current']), -1),
                                 score_temporal=np.concatenate((base, descriptors[split]['temporal']), -1))
            old = (d['m3'] >= .9404184587540165) | (d['local'] >= 4.625390338985158)
            strong = old | (ordinary >= th['addition'])
            prior = strong | (ordinary >= th['single'])
            baseline[split] = np.where(strong, 2, np.where(prior, 1, 0)).astype(np.int8)
            refs[split] = dict(prior_light=prior, ordinary_OR=strong,
                               M3=d['m3'] >= .8557642486787612, old_fusion=old)
            bases[(split, seed)] = A.describe(baseline[split], d, refs[split])
        for bank in BANKS:
            scores = {split: np.empty((384, 4, 13, 2), float) for split in ds}
            for q, height in enumerate(('HEAD', 'BODY')):
                if time.monotonic() - began >= 120:
                    raise TimeoutError('120 second audit cap')
                key = f'{seed}/{bank}/{height}';record = records[key]
                model_path = (OUT / record['path']).resolve()
                assert model_path.parent == (OUT / 'models').resolve()
                digest = sha(model_path)
                A.eq(digest, record['sha256'], 'model hash/' + key)
                model = pickle.loads(model_path.read_bytes())
                assert type(model) is HistGradientBoostingClassifier
                A.eq({k: model.get_params()[k] for k in PARAMS}, PARAMS, 'model params/' + key)
                A.eq(model.n_features_in_, plan['dimensions'][bank], 'model dimensions')
                A.eq(record['dimensions'], plan['dimensions'][bank], 'record dimensions')
                A.eq(model.n_iter_, 100, 'model iterations')
                A.eq(record['iterations'], 100, 'record iterations')
                np.testing.assert_array_equal(model.classes_, [0, 1])
                eligible = baseline['cal'][fit_mask, ..., q] == 0
                frame_count = eligible.sum(-1, keepdims=True)
                weight = np.broadcast_to(np.divide(1., frame_count,
                                          out=np.zeros_like(frame_count, dtype=float),
                                          where=frame_count > 0), eligible.shape)[eligible]
                weight *= len(weight) / weight.sum()
                y = np.broadcast_to((cal['category'][fit_mask, q] == 'contact')[:, None, None],
                                    eligible.shape)[eligible]
                A.eq(len(y), record['rows'], 'fit rows/' + key)
                A.eq(int(y.sum()), record['contact_rows'], 'fit contacts/' + key)
                A.eq(int((frame_count > 0).sum()), record['eligible_scene_replica_groups'], 'fit episode groups')
                assert abs(weight.sum() - record['weight_sum']) < 1e-8
                prevalence = float(np.sum(weight * y) / weight.sum())
                logit = float(np.log(prevalence / (1 - prevalence)))
                assert abs(model._baseline_prediction.item() - logit) < 1e-12
                assert abs(record['baseline_logit'] - logit) < 1e-12
                model_checks[key] = dict(sha256=digest, rows=len(y), contact_rows=int(y.sum()),
                                         eligible_groups=int((frame_count > 0).sum()),
                                         weight_sum=float(weight.sum()), weighted_contact_fraction=prevalence,
                                         baseline_logit=logit)
                for split in ds:
                    x = tensors[split][bank][..., q, :]
                    scores[split][..., q] = model.decision_function(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])
            for split in ds:
                np.testing.assert_array_equal(scores[split], score_saved[split][f'{seed}/{bank}'])
                assert np.isfinite(scores[split]).all()
            for clear_cap, pass_cap in BUDGETS:
                key = f'{seed}/{bank}/c{clear_cap}_p{pass_cap}'
                score = scores['cal'][cut_mask]
                eligible = (baseline['cal'][cut_mask] == 0) & np.isfinite(score)
                candidate = np.where(eligible, score, -np.inf).max(-1)
                category = cal['category'][cut_mask]
                clear = (category == 'clear').all(-1)
                passed = (category == 'pass').any(-1) & ~(category == 'contact').any(-1)
                theta = max(P.cutoff(candidate[clear], clear_cap), P.cutoff(candidate[passed], pass_cap))
                flags = eligible & (score >= theta)
                ac = int(flags[clear].any(-1).sum());ap = int(flags[passed].any(-1).sum())
                A.eq(cuts[key], dict(theta=None if theta == -np.inf else theta,
                                      nonbinding=bool(theta == -np.inf), clear_cap=clear_cap, pass_cap=pass_cap,
                                      actual_extra_candidate_clear_slots=ac,
                                      actual_extra_candidate_pass_slots=ap,
                                      clear_unused=clear_cap-ac, pass_unused=pass_cap-ap,
                                      clear_slot_denominator=int(clear.sum())*4*13,
                                      pass_slot_denominator=int(passed.sum())*4*13), 'cal/' + key)
                assert ac <= clear_cap and ap <= pass_cap
                for split, d in ds.items():
                    score = scores[split]
                    added = (baseline[split] == 0) & np.isfinite(score) & (score >= theta)
                    grade = np.where(added, 1, baseline[split]).astype(np.int8)
                    np.testing.assert_array_equal(grade, saved[split][key])
                    np.testing.assert_array_equal(grade == 2, baseline[split] == 2)
                    assert np.all(grade >= baseline[split])
                    report = A.describe(grade, d, refs[split])
                    A.eq(report, metrics[split + '/' + key], 'metrics/' + split + '/' + key)
                    assert all(row['loss'] == 0 and row['later'] == 0
                               for row in report['paired_any']['prior_light'])
                    reports[(split, key)] = report
                    details[(split, key)] = P.added_costs(added, d['category'])
                    if split == 'validation':
                        firsts[key] = (A.first(baseline[split] > 0), A.first(grade > 0))
    seen = set()
    with (OUT / 'summary.csv').open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            seed = int(row['seed']);split = row['split']
            key = f"{seed}/{row['bank']}/c{row['clear_cap']}_p{row['pass_cap']}"
            identity = (split, key);assert identity not in seen;seen.add(identity)
            r, b = reports[identity], bases[(split, seed)]
            p = r['paired_any']['prior_light'];a = details[identity]
            expected = dict(HEAD=r['any']['counts'][0], BODY=r['any']['counts'][1],
                            HEAD_rescue=p[0]['rescue'], BODY_rescue=p[1]['rescue'],
                            HEAD_loss=p[0]['loss'], BODY_loss=p[1]['loss'],
                            HEAD_earlier=p[0]['earlier'], BODY_earlier=p[1]['earlier'],
                            HEAD_later=p[0]['later'], BODY_later=p[1]['later'],
                            clear_slots=r['any']['clear_slots'], pass_clips=r['any']['pass_clips'],
                            extra_total_clear_slots=r['any']['clear_slots']-b['any']['clear_slots'],
                            extra_total_pass_clips=r['any']['pass_clips']-b['any']['pass_clips'],
                            extra_candidate_clear_slots=a['clear']['slots'],
                            extra_candidate_pass_slots=a['pass']['slots'],
                            light_pass_slots=r['joint_costs']['pass']['light']['slots'],
                            light_pass_longest_frames=r['joint_costs']['pass']['light']['longest_run_frames'])
            for name, value in expected.items():
                A.eq(int(row[name]), value, 'summary/' + name)
            assert expected['extra_total_clear_slots'] <= expected['extra_candidate_clear_slots']
            assert expected['extra_total_pass_clips'] <= a['pass']['clips']
    A.eq(len(seen), 54, 'summary count')
    ledger_seen = set();lookup = {int(v):i for i,v in enumerate(ds['validation']['ids'])}
    with (OUT / 'event_ledger.csv').open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['key'];i = lookup[int(row['scene'])];k = int(row['replica'])
            q = ('HEAD', 'BODY').index(row['height'])
            identity = (key, i, k, q);assert identity not in ledger_seen;ledger_seen.add(identity)
            before, after = firsts[key]
            A.eq(int(row['before_first']), int(before[i,k,q]), 'ledger/before')
            A.eq(int(row['after_first']), int(after[i,k,q]), 'ledger/after')
            A.eq(row['category'], ds['validation']['category'][i,q], 'ledger/category')
            A.eq(row['family'], ds['validation']['rows'][i]['shape_family'], 'ledger/family')
            A.eq(row['background_family'], ds['validation']['rows'][i]['background_family'], 'ledger/background')
            assert before[i,k,q] < 0 or (0 <= after[i,k,q] <= before[i,k,q])
    A.eq(len(ledger_seen), 82944, 'ledger count')
    A.eq(len(records), 18, 'model count');A.eq(len(cuts), 27, 'cutoff count')
    old_root = A.ROOT / 'artifacts.local/work/cnh-graded-peak-track-dev-20261010/alert'
    old_path = old_root / 'validation_grades.npz'
    with np.load(old_path, allow_pickle=False) as archive:
        old_grades = dict(zip(archive['keys'].tolist(), archive['grades']))
    comparisons = {}
    d = ds['validation']
    for seed in A.SEEDS:
        for cc, pc in BUDGETS:
            suffix = f'c{cc}_p{pc}'
            banks = {bank: saved['validation'][f'{seed}/{bank}/{suffix}'] for bank in BANKS}
            pairs = (('current_vs_score_only', banks['score_only'], banks['score_current']),
                     ('temporal_vs_current', banks['score_current'], banks['score_temporal']),
                     ('current_vs_raw_frame', old_grades[f'{seed}/frame_peak/{suffix}'], banks['score_current']),
                     ('temporal_vs_raw_frame', old_grades[f'{seed}/frame_peak/{suffix}'], banks['score_temporal']))
            for name, before, after in pairs:
                key = f'{seed}/{suffix}/{name}'
                comparisons[key] = comparison(before, after, d['category'])
                if cc == 15:
                    comparisons[key]['groups'] = {group: comparison(before[mask], after[mask], d['category'][mask])
                                                  for group, mask in sorted(d['groups'].items())
                                                  if group.startswith(('family:', 'background_family:'))}
    (target / 'comparisons.json').write_text(json.dumps(dict(
        limitation='Consumed exploratory validation. Identical cal candidate-slot budgets do not imply equal validation costs. Raw frame comparator has no scalar scorer fitting, while joint scorers use cal8/10. Paired comparisons include both gains and losses; no rule selection or refit.',
        inputs_sha256={str(old_path.relative_to(A.ROOT)):sha(old_path),
                       str((OUT/'validation_grades.npz').relative_to(A.ROOT)):sha(OUT/'validation_grades.npz')},
        comparisons=comparisons), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    seconds = time.monotonic() - began
    assert seconds <= 120
    result = dict(status='PASS', cells=54, model_checks=18, calibrations=27,
                  event_rows=len(ledger_seen), scalar_checks=A.CHECKS, CPU_seconds=seconds,
                  CPU_seconds_cap=120, GPU_seconds=0, fit_background_ids=sorted(fit_ids),
                  cutoff_background_ids=sorted(cutoff_ids), validation_background_ids=sorted(validation_ids),
                  producer_source_sha256=sha(producer), audit_source_sha256=sha(Path(__file__)),
                  independent_metric_helpers_sha256=sha(Path(A.__file__)),
                  independent_cutoff_cost_helpers_sha256=sha(Path(P.__file__)),
                  independence='No producer helpers or model refits. Independent raw smoothing/score descriptors, feature adaptation, exact-hash HGB params/predictions, cal-only fit mask/weights/baseline logit, complete-tie cutoffs, grade preservation, all metrics/timing/costs and ledger. Pure peak features audited separately.')
    (target / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (target / 'fit_model_checks.json').write_text(json.dumps(model_checks, indent=2)+'\n', encoding='utf8')
    (target / 'added_candidate_duration.json').write_text(json.dumps({split+'/'+key:value for (split,key),value in details.items()}, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
