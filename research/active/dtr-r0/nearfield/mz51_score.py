"""CPU saved-output scoring of the two fixed MZ51 negative-replay arms."""
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


ARMS = ('NEW_NEG', 'OLD_NEG')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
LEGACY_NONCAL = ('relation10000', 'distance5000', 'rich', 'mz36')
READOUT_KEYS = READOUTS + tuple(arm + '/' + mode for arm in ARMS for mode in READOUTS)
METHODS = BASELINES + tuple(key + '/candidate' for key in READOUT_KEYS)


def input_path(receipt, suffix, bind):
    matches = [(Path(path), digest) for path, digest in receipt['inputs'].items()
               if path.replace('\\', '/').endswith(suffix)]
    assert len(matches) == 1, suffix
    return bind(*matches[0])


def validate_schedule(source, receipt, p, groups, bind):
    schedule = load_npz(source / 'schedule.npz')
    assert set(schedule) == {'shared', 'query', 'NEW_NEG', 'OLD_NEG', 'train_ids', 'fit_ids'}
    for name in ('shared', 'query', 'NEW_NEG', 'OLD_NEG'):
        assert schedule[name].shape == (600, 8) and schedule[name].dtype.kind in 'iu'
    np.testing.assert_array_equal(schedule['fit_ids'], groups['fit'])
    assert len(schedule['train_ids']) == len(np.unique(schedule['train_ids'])) == 7562
    # Read just the small truth/train-ID members. No dense mmap or local label
    # counts are materialized, and no model module function is executed.
    truths = []
    for suffix in ('mz8-attribution-20260910/cache-v5/evaluator.npz',
                   'mz15-shared-support-20260910/cache-v1/evaluator.npz'):
        with np.load(input_path(receipt, suffix, bind), allow_pickle=False) as archive:
            truths.append(archive['truth'])
    truth_old = np.concatenate(truths)
    with np.load(input_path(receipt, 'mz30-branch-responsibility-20260910/run-v1/branches.npz', bind),
                 allow_pickle=False) as archive:
        np.testing.assert_array_equal(schedule['train_ids'], archive['train_ids'])
    for cohort in COHORTS[:3]:
        ids = p[cohort + '/frame_ids']
        assert ids.dtype.kind in 'iu' and not np.isin(schedule['train_ids'], ids).any()
        np.testing.assert_array_equal(truth_old[ids], p[cohort + '/truth'])
    generator = np.random.default_rng(151)
    np.testing.assert_array_equal(schedule['shared'], generator.choice(groups['fit'], size=(600, 8), replace=True))
    query = np.tile(np.repeat(np.arange(4), 2), (600, 1))
    np.testing.assert_array_equal(schedule['query'], query)
    summary = {}
    for arm, truth, ids in (('NEW_NEG', p['mz48/truth'], groups['fit']),
                            ('OLD_NEG', truth_old, schedule['train_ids'])):
        pools = [ids[~truth[ids, q]] for q in range(4)]
        assert all(len(pool) for pool in pools)
        expected = np.stack([np.concatenate([generator.choice(pool, 2, replace=True) for pool in pools])
                             for _ in range(600)])
        np.testing.assert_array_equal(schedule[arm], expected)
        assert np.isin(schedule[arm], ids).all() and not truth[schedule[arm], query].any()
        count = [int((query == q).sum()) for q in range(4)]
        assert count == receipt['fits'][arm]['negative_presentations_per_query'] == [1200] * 4
        unique = int(len(np.unique(schedule[arm])))
        assert unique == receipt['fits'][arm]['unique_extra_frames']
        summary[arm] = dict(negative_presentations=4800, per_query=count, unique_frames=unique,
                            negative_query_truth_verified=True, allowed_source_membership=True)
    return dict(status='PASS', seed=151, schedule_exact=True, shared_presentations=4800,
                shared_unique_frames=int(len(np.unique(schedule['shared']))), arms=summary,
                old_train_ids=7562, new_fit_ids=1280, old_train_disjoint_from_three_DEV_cohorts=True)


def comparisons(a, truth, known):
    result = {key: paired(a[key + '/candidate'], a['MZ37'], truth, known) for key in READOUT_KEYS}
    result['OPEN_vs_GATED'] = paired(a['OPEN/candidate'], a['GATED/candidate'], truth, known)
    for arm in ARMS:
        result[arm + '/OPEN_vs_GATED'] = paired(a[arm + '/OPEN/candidate'], a[arm + '/GATED/candidate'], truth, known)
        for mode in READOUTS:
            result[arm + '/' + mode + '_vs_MZ50'] = paired(a[arm + '/' + mode + '/candidate'],
                                                         a[mode + '/candidate'], truth, known)
    for mode in READOUTS:
        result['OLD_NEG_vs_NEW_NEG/' + mode] = paired(a['OLD_NEG/' + mode + '/candidate'],
                                                     a['NEW_NEG/' + mode + '/candidate'], truth, known)
    for key in READOUT_KEYS:
        assert not any(result[key]['tp_lost']) and not any(result[key]['fp_removed'])
    return result


def run(task):
    started = time.perf_counter()
    source, out = task / 'run-v1', task / 'score-v1'
    assert not out.exists(), 'Keep an existing sealed score intact'
    bind = Bindings()
    receipt = read(bind(source / 'receipt.json'))
    assert receipt['status'] == 'PASS' and receipt['total_steps'] == 1200 and receipt['steps_per_arm'] == 600
    assert set(receipt['fits']) == set(ARMS) and all(receipt['fits'][arm]['steps'] == 600 for arm in ARMS)
    assert receipt['exact_initial_weights'] and receipt['new_baseline_inferences'] == 0
    for name, digest in receipt['outputs'].items():
        bind(source / name, digest)
    assert read(source / 'parity.json')['status'] == 'PASS'
    p = load_npz(source / 'predictions.npz')
    stored_groups = read(source / 'groups.json')
    records = stored_groups['records']
    groups = split_source(records)
    for key, ids in groups.items():
        np.testing.assert_array_equal(ids, stored_groups['groups'][key])
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    assert len(set(p['mz48/frame_ids'])) == 2560
    groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))
    schedule_audit = validate_schedule(source, receipt, p, groups, bind)

    original_path = input_path(receipt, 'mz50-echo-independent-local-20260911/run-v1/predictions.npz', bind)
    original = load_npz(original_path)
    assert set(original) <= set(p)
    for key, array in original.items():
        np.testing.assert_array_equal(p[key], array, err_msg='Frozen MZ50 changed: ' + key)
    frozen_arrays = len(original)
    del original
    cuts = {}
    for arm in ARMS:
        for mode in READOUTS:
            key = arm + '/' + mode

            def collect(suffix):
                return np.concatenate([p['DEV/DROP_CLOSE/' + suffix],
                                       p['mz48/DROP_CLOSE/' + suffix][groups['calibration']]])

            truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][groups['calibration']]])
            known = np.concatenate([p['DEV/known'], p['mz48/known'][groups['calibration']]])
            assert truth.shape == (1256, 4) and known.all()
            checked_cutoff = cutoff_zero_added(collect(key + '/raw'), collect(key + '/support'), collect('MZ37'), truth)
            saved = np.load(source / (arm + '-' + mode + '-cutoff.npy'), allow_pickle=False)
            np.testing.assert_array_equal(checked_cutoff, saved)
            np.testing.assert_array_equal(saved, receipt['fits'][arm][mode + '_cutoff'])
            cuts[key] = saved
            added = (collect(key + '/candidate') >= 0) & (collect('MZ37') < 0)
            assert not (added & ~truth).any()

    scores, changes, local, context, cached = {}, {}, {}, {}, {}
    audited = 0
    for cohort in COHORTS:
        truth, known = p[cohort + '/truth'], p[cohort + '/known']
        if cohort == 'mz36':
            truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
            admitted = p['mz36/admitted_index']
            assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80
            assert len(admitted) == len(np.unique(admitted)) == 380
            np.testing.assert_array_equal(p['mz36/attempted_frame_ids'][admitted], p['mz36/frame_ids'])
            np.testing.assert_array_equal(truth[admitted], p['mz36/truth'])
        cohort_groups = {'all': np.arange(len(truth))}
        if cohort == 'mz48':
            cohort_groups.update(groups)
            for field in ('family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                for value in sorted({str(r[field]) for r in records}):
                    cohort_groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
        scores[cohort], changes[cohort] = {}, {}
        for condition in CONDITIONS:
            prefix = cohort + '/' + condition + '/'
            a = {name: p[prefix + name] for name in METHODS}
            for arm in ARMS:
                for mode in READOUTS:
                    key = arm + '/' + mode
                    raw, support, candidate = (p[prefix + key + '/' + suffix] for suffix in ('raw', 'support', 'candidate'))
                    np.testing.assert_array_equal(support, p[prefix + mode + '/support'])
                    assert raw.shape == support.shape == candidate.shape == p[cohort + '/truth'].shape
                    assert np.isfinite(raw).all() and np.isfinite(candidate).all()
                    margin = raw.astype(float) - cuts[key]
                    accepted = (a['MZ37'] < 0) & support & (margin >= 0)
                    np.testing.assert_array_equal(candidate, np.where(accepted, margin, a['MZ37']))
                    winner = p[prefix + key + '/winner']
                    assert winner.shape == raw.shape and winner.min() >= 0 and winner.max() < 3136
            if cohort == 'mz36':
                expanded = {}
                for name, value in a.items():
                    expanded[name] = np.full((400, 4), np.nan)
                    expanded[name][admitted] = value
                a = expanded
            scores[cohort][condition], changes[cohort][condition] = {}, {}
            for group, ids in cohort_groups.items():
                aa = {name: value[ids] for name, value in a.items()}
                scores[cohort][condition][group] = {name: metrics(value, truth[ids], known[ids]) for name, value in aa.items()}
                changes[cohort][condition][group] = comparisons(aa, truth[ids], known[ids])
            for name, value in a.items():
                audited += scalar_check(value, truth, known, scores[cohort][condition]['all'][name])
            cached.setdefault(condition, {})[cohort] = (a, truth, known)
            if cohort == 'mz48':
                local[condition] = {}
                for group in groups:
                    ids = groups[group]
                    local[condition][group] = {}
                    for key in READOUT_KEYS:
                        gated_key = key.rsplit('/', 1)[0] + '/GATED' if '/' in key else 'GATED'
                        gain = (a[key + '/candidate'] >= 0) & (a['MZ37'] < 0) & truth & known
                        winner = p[prefix + key + '/winning_native']
                        assert winner.shape == truth.shape and winner.dtype == np.bool_
                        assert not (winner & ~p[prefix + key + '/support']).any()
                        local[condition][group][key] = dict(gained_tp=gain[ids].sum(0).tolist(),
                            gained_tp_winning_native=(gain & winner)[ids].sum(0).tolist(),
                            gained_tp_without_gated_candidate=(gain & ~p[prefix + gated_key + '/support'])[ids].sum(0).tolist())
                pairs = {}
                for i, row in enumerate(records):
                    pairs.setdefault(row['pair_id'], []).append(i)
                pair_indices = np.array(list(pairs.values()))
                assert pair_indices.shape == (1280, 2)
                np.testing.assert_array_equal(truth[pair_indices[:, 0]], truth[pair_indices[:, 1]])
                context[condition] = {name: ((value[pair_indices[:, 0]] >= 0) != (value[pair_indices[:, 1]] >= 0)).sum(0).tolist()
                                      for name, value in a.items()}

    aggregates, aggregate_changes = {}, {}
    for condition in CONDITIONS:
        data = cached[condition]
        pieces = [(data[name], np.arange(len(data[name][1]))) for name in LEGACY_NONCAL]
        selectors = dict(legacy_noncal=pieces, mz48_fit=[(data['mz48'], groups['fit'])],
                         mz48_nonfit=[(data['mz48'], groups['nonfit'])])
        selectors['all_noncal'] = pieces + selectors['mz48_fit'] + selectors['mz48_nonfit']
        aggregates[condition], aggregate_changes[condition] = {}, {}
        for group, selected in selectors.items():
            y = np.concatenate([source_truth[ids] for (_, source_truth, _), ids in selected])
            k = np.concatenate([source_known[ids] for (_, _, source_known), ids in selected])
            a = {name: np.concatenate([arrays[name][ids] for (arrays, _, _), ids in selected]) for name in METHODS}
            aggregates[condition][group] = {name: metrics(value, y, k) for name, value in a.items()}
            aggregate_changes[condition][group] = comparisons(a, y, k)
    drop, delta = aggregates['DROP_CLOSE'], aggregate_changes['DROP_CLOSE']
    old_key, new_key = 'OLD_NEG/GATED', 'NEW_NEG/GATED'
    old_fp = sum(delta['legacy_noncal'][old_key]['fp_added'])
    new_fp = sum(delta['legacy_noncal'][new_key]['fp_added'])
    reference_fp = sum(delta['legacy_noncal']['GATED']['fp_added'])
    reference_nonfit = drop['mz48_nonfit']['GATED/candidate']
    assert reference_fp == 19 and sum(reference_nonfit['tp']) == 453 and sum(reference_nonfit['fp']) == 21
    target = drop['mz48_nonfit'][old_key + '/candidate']
    gate = dict(OLD_NEG_GATED_old_noncal_added_fp=old_fp, NEW_NEG_GATED_old_noncal_added_fp=new_fp,
                fixed_MZ50_GATED_old_noncal_added_fp=reference_fp,
                OLD_NEG_GATED_nonfit_tp=sum(target['tp']), OLD_NEG_GATED_nonfit_fp=sum(target['fp']),
                old_fp_below_19=old_fp < 19, old_fp_below_NEW_NEG=old_fp < new_fp,
                nonfit_tp_at_least_453=sum(target['tp']) >= 453,
                nonfit_fp_at_most_21=sum(target['fp']) <= 21)
    gate['passes_primary'] = all(gate[name] for name in ('old_fp_below_19', 'old_fp_below_NEW_NEG',
                                                       'nonfit_tp_at_least_453', 'nonfit_fp_at_most_21'))
    result = dict(status='PASS', primary_condition='DROP_CLOSE', methods=list(METHODS),
                  conditions=scores, comparisons=changes, aggregates=aggregates, aggregate_comparisons=aggregate_changes,
                  local_witnesses=local, context_pair_changed_decisions=context, primary_gate=gate,
                  existing_cutoffs={key: value.tolist() for key, value in cuts.items()},
                  aggregation_scope=dict(legacy_noncal='relation2000 + distance1000 + rich44 + MZ36 attempts400; DEV excluded',
                    mz48_fit='1280 training frames, reported separately', mz48_nonfit='640 heldout-site +384 withheld-family',
                    all_noncal='legacy_noncal + mz48_fit + mz48_nonfit, each once; includes fit and is not wholly held-out evidence'),
                  scope='Two fixed continued-learning arms; controlled Development on consumed sites; no hardware, fresh blind or safety claim')
    audit = dict(status='PASS', schedule=schedule_audit, frozen_MZ50_arrays_equal=frozen_arrays,
                 scalar_known_bits=audited, scalar_methods=len(METHODS), observation_profiles=len(CONDITIONS),
                 mz36_attempts=400, mz36_unknown_bits=80, source_frames=2560, cutoffs_recomputed_for_parity=4,
                 new_cutoffs=0, threshold_searches=0, prior_MZ37_positive_retention=True,
                 all_noncal_fit_frames=1280, aggregate_groups_counted_once=True,
                 winning_native='Source-bound saved flags; not independently rederived from native depth',
                 dense_feature_reads=0, model_checkpoint_loads=0, new_fits=0, new_inference_frames=0)
    for name in ('mz51_score.py', 'mz50_score.py', 'mz47_score.py', 'mz40_evaluate.py', 'mz15_train.py'):
        bind(Path(__file__).with_name(name))
    input_path(receipt, 'nearfield/mz51_training_coverage.py', bind)
    bind.check()
    out.mkdir(exist_ok=False)
    write(out / 'result.json', result)
    write(out / 'audit.json', audit)
    write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, seconds=time.perf_counter() - started,
        backend='CPU saved-output scalar scoring; TASK_NOT_GPU_SUITABLE', new_fits=0, new_inference_frames=0,
        new_cutoffs=0, outputs={name: sha(out / name) for name in ('result.json', 'audit.json')}))
    print('PRIMARY_GATE', gate)
    print('DROP_TOTALS', {group: {key: [sum(values[term]) for term in ('tp', 'fp', 'fn')]
        for key, values in methods.items() if key == 'MZ37' or key.endswith('/candidate')}
        for group, methods in drop.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task.resolve())
