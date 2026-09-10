"""Fixed-cutoff scalar audit, paired errors and local witness accounting."""
import argparse
from pathlib import Path
import time

import numpy as np

from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz40_evaluate import metrics, paired
from mz45_object_transfer import Bindings, CONDITIONS
from mz47_score import scalar_check
from mz50_train import split_source, READOUTS, BASELINES

METHODS = BASELINES + tuple(name + '/candidate' for name in READOUTS)


def run(task):
    source, out = task / 'run-v1', task / 'score-v1'
    out.mkdir(exist_ok=False); started = time.perf_counter(); bind = Bindings()
    receipt = read(bind(source / 'receipt.json'))
    assert receipt['status'] == 'PASS' and receipt['total_steps'] == 600 and receipt['fits'] == 1
    for name, digest in receipt['outputs'].items(): bind(source / name, digest)
    p = load_npz(source / 'predictions.npz')
    records = read(source / 'groups.json')['records']; groups = split_source(records)
    batches = np.load(source / 'batches.npy', allow_pickle=False)
    np.testing.assert_array_equal(batches, np.random.default_rng(150).choice(groups['fit'], size=(600, 16), replace=True))
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    for name in READOUTS:
        def collect(key):
            return np.concatenate([p['DEV/DROP_CLOSE/' + key], p['mz48/DROP_CLOSE/' + key][groups['calibration']]])
        y = np.concatenate([p['DEV/truth'], p['mz48/truth'][groups['calibration']]])
        cut = cutoff_zero_added(collect(name + '/raw'), collect(name + '/support'), collect('MZ37'), y)
        np.testing.assert_array_equal(cut, np.load(source / (name + '-cutoff.npy')))
        accepted = (collect(name + '/candidate') >= 0) & (collect('MZ37') < 0)
        assert not (accepted & ~y).any()
    scores, changes, local, context = {}, {}, {}, {}
    audited = 0
    noncal = dict(OPEN=[0, 0], GATED=[0, 0])
    primary = dict(OPEN=[0, 0], GATED=[0, 0])
    for cohort in ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48'):
        truth, known = p[cohort + '/truth'], p[cohort + '/known']
        if cohort == 'mz36': truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
        cohort_groups = {'all': np.arange(len(truth))}
        if cohort == 'mz48':
            cohort_groups.update(groups)
            cohort_groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))
            for field in ('family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                for value in sorted({str(r[field]) for r in records}):
                    cohort_groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
        scores[cohort], changes[cohort] = {}, {}
        for condition in CONDITIONS:
            prefix = cohort + '/' + condition + '/'
            a = {name: p[prefix + name] for name in METHODS}
            if cohort == 'mz36':
                expanded = {}
                for name, value in a.items():
                    expanded[name] = np.full((400, 4), np.nan)
                    expanded[name][p['mz36/admitted_index']] = value
                a = expanded
                assert int((~known).sum()) == 80
            scores[cohort][condition], changes[cohort][condition] = {}, {}
            for group, ids in cohort_groups.items():
                scores[cohort][condition][group] = {name: metrics(value[ids], truth[ids], known[ids]) for name, value in a.items()}
                changes[cohort][condition][group] = {name: paired(a[name + '/candidate'][ids], a['MZ37'][ids], truth[ids], known[ids]) for name in READOUTS}
                changes[cohort][condition][group]['OPEN_vs_GATED'] = paired(a['OPEN/candidate'][ids], a['GATED/candidate'][ids], truth[ids], known[ids])
                for name in READOUTS:
                    assert not any(changes[cohort][condition][group][name]['tp_lost'])
                    assert not any(changes[cohort][condition][group][name]['fp_removed'])
                    if condition == 'DROP_CLOSE' and ((cohort in ('relation10000', 'distance5000', 'rich', 'mz36') and group == 'all') or (cohort == 'mz48' and group in ('fit', 'nonfit'))):
                        row = changes[cohort][condition][group][name]
                        noncal[name][0] += sum(row['tp_gained']); noncal[name][1] += sum(row['fp_added'])
                    if cohort == 'mz48' and group == 'nonfit' and condition == 'DROP_CLOSE':
                        row = changes[cohort][condition][group][name]
                        primary[name] = [sum(row['tp_gained']), sum(row['fp_added'])]
            for name in METHODS:
                audited += scalar_check(a[name], truth, known, scores[cohort][condition]['all'][name])
            if cohort == 'mz48':
                local[condition] = {}
                for group in ('fit', 'calibration', 'heldout_site', 'nonfit_family', 'nonfit'):
                    ids = cohort_groups[group]; local[condition][group] = {}
                    for name in READOUTS:
                        gained = (a[name + '/candidate'] >= 0) & (a['MZ37'] < 0) & truth & known
                        winner = p[prefix + name + '/winning_native']
                        local[condition][group][name] = dict(
                            gained_tp=gained[ids].sum(0).tolist(),
                            gained_tp_winning_native=(gained & winner)[ids].sum(0).tolist(),
                            gained_tp_without_gated_candidate=(gained & ~p[prefix + 'GATED/support'])[ids].sum(0).tolist())
                pairs = {}
                for i, row in enumerate(records): pairs.setdefault(row['pair_id'], []).append(i)
                pair_indices = np.array(list(pairs.values())); assert pair_indices.shape == (1280, 2)
                np.testing.assert_array_equal(truth[pair_indices[:, 0]], truth[pair_indices[:, 1]])
                context[condition] = {name: ((a[name][pair_indices[:, 0]] >= 0) != (a[name][pair_indices[:, 1]] >= 0)).sum(0).tolist() for name in METHODS}
    gates = {name: dict(new_nonfit_added_tp=primary[name][0], new_nonfit_added_fp=primary[name][1],
        all_noncal_added_tp=noncal[name][0], all_noncal_added_fp=noncal[name][1],
        useful_component=primary[name][0] > 0 and noncal[name][1] == 0) for name in READOUTS}
    result = dict(status='PASS', primary_condition='DROP_CLOSE', conditions=scores, comparisons=changes,
                  local_witnesses=local, context_pair_changed_decisions=context, gates=gates,
                  scope='Single controlled Development fit; consumed sites; no hardware, fresh blind or safety claim')
    write(out / 'result.json', result)
    write(out / 'audit.json', dict(status='PASS', scalar_known_bits=audited, mz36_attempts=400, mz36_unknown_bits=80,
        source_frames=2560, schedule_exact=True, no_site_or_family_holdout_fit=True,
        cutoffs_recomputed=2, prior_mz37_positive_retention=True, new_fits=0, new_inference_frames=0))
    bind(Path(__file__)); bind.check()
    write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, seconds=time.perf_counter() - started,
        backend='CPU saved-output scalar scoring; TASK_NOT_GPU_SUITABLE', outputs={f.name: sha(f) for f in out.iterdir() if f.is_file()}))
    print('GATES', gates)
    for condition in CONDITIONS:
        print('MZ48', condition, {g: {name: [sum(scores['mz48'][condition][g][name][k]) for k in ('tp', 'fp', 'fn')]
            for name in ('MZ37', 'OPEN/candidate', 'GATED/candidate')} for g in ('fit', 'calibration', 'heldout_site', 'nonfit_family')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task.resolve())
