"""Saved-output scoring and scalar audit for the two frozen MZ47 fits."""
import argparse
from pathlib import Path
import time
import numpy as np

from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz40_evaluate import metrics, paired
from mz45_object_transfer import Bindings, CONDITIONS

METHODS = ('rgb', 'tof', 'MZ5', 'MZ43_ENSEMBLE') + tuple(
    arm + '/' + method for arm in ('FROZEN', 'REPLAY', 'ENRICH')
    for method in ('local_before', 'local_after', 'MZ28', 'MZ35', 'MZ37'))


def scalar_check(scores, truth, known, expected):
    counts = {k: [0] * 4 for k in ('tp', 'fp', 'fn', 'tn', 'known', 'unknown')}
    exact = 0
    for i in range(len(truth)):
        correct = True
        for q in range(4):
            if not known[i, q]:
                counts['unknown'][q] += 1; correct = False; continue
            counts['known'][q] += 1
            p, y = bool(scores[i, q] >= 0), bool(truth[i, q])
            counts[('tp' if y else 'fp') if p else ('fn' if y else 'tn')][q] += 1
            correct &= p == y
        exact += correct
    assert all(value == expected[key] for key, value in counts.items())
    assert exact == expected['exact_frames']
    return int(known.sum())


def run(task):
    started = time.perf_counter(); source = task / 'run-v1'; out = task / 'score-v1'
    out.mkdir(exist_ok=False); bind = Bindings()
    r = read(bind(source / 'receipt.json')); assert r['status'] == 'PASS' and r['total_steps'] == 600
    for name, digest in r['outputs'].items(): bind(source / name, digest)
    p = load_npz(source / 'predictions.npz'); rows = read(source / 'groups.json')['records']
    np.testing.assert_array_equal(p['rich/frame_ids'], [row['frame_id'] for row in rows])
    mask = dict(fit=np.array([row['block'] == 'near' and row['family'] in ('pipe', 'ladder', 'pouch') for row in rows]),
        form_transfer=np.array([row['family'] == 'birch' for row in rows]),
        distance_transfer=np.array([row['block'] == 'far' and row['family'] in ('pipe', 'ladder', 'pouch') for row in rows]),
        context_control=np.array([row['family'] == 'oblique_rod' for row in rows]),
        unsupported=np.array([row['block'] == 'unsupported' for row in rows]))
    assert [int(mask[k].sum()) for k in ('fit', 'form_transfer', 'distance_transfer', 'context_control')] == [12, 8, 12, 12]
    mask['nonfit'] = ~mask['fit']; mask['all'] = np.ones(44, bool)
    assert (mask['fit'].astype(int) + mask['form_transfer'] + mask['distance_transfer'] + mask['context_control'] == 1).all()
    schedule = load_npz(source / 'schedule.npz')
    assert schedule['old'].shape == (300, 16) and schedule['replay'].shape == schedule['enrich'].shape == (300, 4)
    assert np.isin(schedule['old'], schedule['train_ids']).all() and np.isin(schedule['replay'], schedule['train_ids']).all()
    np.testing.assert_array_equal(np.unique(schedule['enrich']), np.flatnonzero(mask['fit']))
    for arm in ('REPLAY', 'ENRICH'):
        prefix = 'DEV/DROP_CLOSE/'
        cut = cutoff_zero_added(p[prefix + arm + '/raw'], p[prefix + arm + '/support'], p[prefix + 'MZ5'], p['DEV/truth'])
        np.testing.assert_array_equal(cut, np.load(source / (arm + '-cutoff.npy')))
        for condition in CONDITIONS:
            assert not ((p[prefix + arm + '/local_before'] >= 0) & (p[prefix + 'MZ5'] < 0) & ~p['DEV/truth']).any()
    results, comparison, failures = {}, {}, []; audited = 0
    for cohort in ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36'):
        truth, known = p[cohort + '/truth'], p[cohort + '/known']
        if cohort == 'mz36': truth, known = p[cohort + '/attempted_truth'], p[cohort + '/attempted_known']
        groups = mask if cohort == 'rich' else dict(all=np.ones(len(truth), bool))
        results[cohort], comparison[cohort] = {}, {}
        for condition in CONDITIONS:
            a = {method: p[cohort + '/' + condition + '/' + method] for method in METHODS}
            if cohort == 'mz36':
                expanded = {}
                for method, value in a.items():
                    expanded[method] = np.full((400, 4), np.nan)
                    expanded[method][p['mz36/admitted_index']] = value
                a = expanded
                assert len(truth) == 400 and int((~known).sum()) == 80
            results[cohort][condition], comparison[cohort][condition] = {}, {}
            for group, selected in groups.items():
                results[cohort][condition][group] = {method: metrics(scores[selected], truth[selected], known[selected]) for method, scores in a.items()}
                comparison[cohort][condition][group] = {}
                for before in ('REPLAY/MZ37', 'FROZEN/MZ37', 'MZ43_ENSEMBLE'):
                    change = paired(a['ENRICH/MZ37'][selected], a[before][selected], truth[selected], known[selected])
                    change['net_tp'] = sum(change['tp_gained']) - sum(change['tp_lost'])
                    change['net_fp'] = sum(change['fp_added']) - sum(change['fp_removed'])
                    change['enrichment_supported'] = change['net_tp'] > 0 and sum(change['fp_added']) == 0
                    comparison[cohort][condition][group][before] = change
            # Audit full unique cohorts once, with explicit unknown skipping.
            for method in METHODS:
                audited += scalar_check(a[method], truth, known, results[cohort][condition]['all'][method])
            for i in range(len(truth)):
                for q in range(4):
                    if not known[i, q]: continue
                    before, after = a['REPLAY/MZ37'][i, q] >= 0, a['ENRICH/MZ37'][i, q] >= 0
                    if before != after:
                        ids = p['mz36/attempted_frame_ids'] if cohort == 'mz36' else p[cohort + '/frame_ids']
                        failures.append(dict(cohort=cohort, condition=condition, frame_id=str(ids[i]), query=q,
                            truth=bool(truth[i, q]), before=bool(before), after=bool(after),
                            fit=bool(mask['fit'][i]) if cohort == 'rich' else False))
    witnesses = {}
    for condition in ('IDEAL', 'DROP_CLOSE'):
        witnesses[condition] = {}
        for group, selected in mask.items():
            y = p['rich/truth'][selected]
            witnesses[condition][group] = {stage: (y & ~p['rich/' + condition + '/ENRICH/' + key][selected]).sum(0).tolist()
                for stage, key in [('no_candidate', 'support'), ('no_witness_before_bank', 'native_witness_before'), ('no_witness_after_bank', 'native_witness_after')]}
    primary = comparison['rich']['DROP_CLOSE']['nonfit']['REPLAY/MZ37']
    regressions = {c + '/' + condition: comparison[c][condition]['all']['FROZEN/MZ37']
        for c in ('DEV', 'relation10000', 'distance5000', 'mz36') for condition in CONDITIONS}
    legacy_cost = any(value['net_fp'] > 0 or value['net_tp'] < 0 for value in regressions.values())
    result = dict(status='PASS', primary_condition='DROP_CLOSE', conditions=results, comparisons=comparison,
        gates=dict(primary_nonfit_enrichment=primary['enrichment_supported'], any_legacy_regression=legacy_cost,
            automatic_replacement=primary['enrichment_supported'] and not legacy_cost),
        native_witness_gaps=witnesses, fitted_new_frames=12, unique_rich_frames=44,
        scope='Consumed Development; two single fits; unchanged non-bitwise CUDA training; no fresh or hardware claim')
    write(out / 'result.json', result)
    write(out / 'changes.json', dict(records=failures))
    write(out / 'audit.json', dict(status='PASS', scalar_known_bits=audited, mz36_unknown_bits=80,
        original_train_membership=True, fixed_extra_ids=True, cutoffs_independently_recomputed=2,
        original_fits=2, extra_fit_steps=0, merge_labels_not_inherited=True))
    bind(Path(__file__)); bind.check()
    write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, seconds=time.perf_counter() - started,
        backend='CPU scalar scoring; TASK_NOT_GPU_SUITABLE', outputs={f.name: sha(f) for f in out.iterdir() if f.is_file()}))
    print('GATES', result['gates'])
    for condition in CONDITIONS:
        print('RICH', condition, {group: {m: [sum(results['rich'][condition][group][m][k]) for k in ('tp', 'fp', 'fn')]
            for m in ('FROZEN/MZ37', 'REPLAY/MZ37', 'ENRICH/MZ37')} for group in ('fit', 'form_transfer', 'distance_transfer', 'context_control', 'nonfit')})
    print('LEGACY', {k: [v['net_tp'], v['net_fp']] for k, v in regressions.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task.resolve())
