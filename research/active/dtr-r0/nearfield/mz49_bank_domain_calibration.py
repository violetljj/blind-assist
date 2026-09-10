"""CPU-only calibration-domain alignment with fixed MZ47 scores."""
import argparse
from pathlib import Path
import time
import numpy as np
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz40_evaluate import metrics, paired
from mz45_object_transfer import Bindings, CONDITIONS
from mz47_local_enrichment import composed
from mz47_score import scalar_check

ARMS = ('REPLAY', 'ENRICH')


def run(root, task):
    out = task / 'run-v1'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); bind = Bindings(); source = root / 'artifacts.local/work/mz47-local-enrichment-20260911/run-v1'
    receipt = read(bind(source / 'receipt.json')); assert receipt['status'] == 'PASS'
    predictions = load_npz(bind(source / 'predictions.npz', receipt['outputs']['predictions.npz']))
    rows = read(bind(source / 'groups.json', receipt['outputs']['groups.json']))['records']
    bank_run = root / 'artifacts.local/work/mz35-responsibility-convergence-20260910/run-v1'
    restore_run = root / 'artifacts.local/work/mz37-positive-restoration-20260910/run-v1'
    selector_cut = np.load(bind.checkpoint(bank_run, 'cutoff.npy'))
    restore_cut = np.load(bind.checkpoint(restore_run, 'cutoff.npy'))
    for name in (Path(__file__).name, 'MZ49_BANK_DOMAIN_CALIBRATION_20260911.md', 'mz47_local_enrichment.py', 'mz15_train.py', 'mz37_restore.py'):
        bind(Path(__file__).with_name(name))
    write(out / 'start.json', dict(status='STARTED', inputs=bind.inputs.copy(), training_steps=0, gpu_jobs=0, new_cutoffs=2))
    new_cuts, old_cuts, details = {}, {}, {}
    prefix = 'DEV/DROP_CLOSE/'
    for arm in ARMS:
        old = np.load(bind(source / (arm + '-cutoff.npy'), receipt['outputs'][arm + '-cutoff.npy']))
        raw, support = (predictions[prefix + arm + '/' + k] for k in ('restricted_raw', 'restricted_support'))
        baseline, truth = predictions[prefix + 'MZ5'], predictions['DEV/truth']
        new = cutoff_zero_added(raw, support, baseline, truth); assert (new <= old).all()
        providers = []
        for q in range(4):
            negative = support[:, q] & (baseline[:, q] < 0) & ~truth[:, q]
            ids = np.flatnonzero(negative)
            providers.append(predictions['DEV/frame_ids'][ids[raw[ids, q].argmax()]].item() if len(ids) else None)
        old_cuts[arm], new_cuts[arm] = old, new
        details[arm] = dict(old=old.tolist(), new=new.tolist(), maximum_negative_global_ids=providers)
        np.save(out / (arm + '-cutoff.npy'), new)
    summary, changes, arrays = {}, {}, {}; audited = parity_values = 0
    rich_fit = np.array([r['block'] == 'near' and r['family'] in ('pipe', 'ladder', 'pouch') for r in rows])
    for cohort in ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36'):
        truth, known = predictions[cohort + '/truth'], predictions[cohort + '/known']
        if cohort == 'mz36': truth, known = predictions['mz36/attempted_truth'], predictions['mz36/attempted_known']
        arrays[cohort + '/truth'], arrays[cohort + '/known'] = truth, known
        masks = dict(all=np.ones(len(truth), bool))
        if cohort == 'rich': masks.update(fit=rich_fit, nonfit=~rich_fit)
        summary[cohort], changes[cohort] = {}, {}
        for condition in CONDITIONS:
            prefix = cohort + '/' + condition + '/'
            a = {k: predictions[prefix + k] for k in ('rgb', 'tof', 'MZ5', 'confidence')}
            values = {}
            for arm in ARMS:
                local = {k: predictions[prefix + arm + '/' + k] for k in ('raw', 'support', 'restricted_raw', 'restricted_support')}
                old = composed(a, local, old_cuts[arm], restore_cut, selector_cut)
                for method in ('MZ28', 'MZ35', 'MZ37'):
                    np.testing.assert_array_equal(old[method], predictions[prefix + arm + '/' + method])
                    parity_values += old[method].size
                new = composed(a, local, new_cuts[arm], restore_cut, selector_cut)
                before, after = old['MZ37'], new['MZ37']
                assert not ((before >= 0) & (after < 0)).any()
                if cohort == 'mz36':
                    expanded = []
                    for score in (before, after):
                        full = np.full((400, 4), np.nan); full[predictions['mz36/admitted_index']] = score; expanded.append(full)
                    before, after = expanded
                values[arm + '/before'], values[arm + '/after'] = before, after
                arrays[prefix + arm + '/before'], arrays[prefix + arm + '/after'] = before, after
            summary[cohort][condition], changes[cohort][condition] = {}, {}
            for group, mask in masks.items():
                summary[cohort][condition][group] = {label: metrics(value[mask], truth[mask], known[mask]) for label, value in values.items()}
                changes[cohort][condition][group] = {arm: paired(values[arm + '/after'][mask], values[arm + '/before'][mask], truth[mask], known[mask]) for arm in ARMS}
                changes[cohort][condition][group]['ENRICH_vs_REPLAY_after'] = paired(values['ENRICH/after'][mask], values['REPLAY/after'][mask], truth[mask], known[mask])
            for label, value in values.items(): audited += scalar_check(value, truth, known, summary[cohort][condition]['all'][label])
    gates = {}
    for arm in ARMS:
        groups = [changes[c]['DROP_CLOSE']['nonfit' if c == 'rich' else 'all'][arm] for c in ('relation10000', 'distance5000', 'rich', 'mz36')]
        gained = sum(sum(g['tp_gained']) for g in groups); added = sum(sum(g['fp_added']) for g in groups)
        lost = sum(sum(g['tp_lost']) for g in groups)
        assert sum(changes['DEV']['DROP_CLOSE']['all'][arm]['fp_added']) == 0
        gates[arm] = dict(nonfit_noncalibration_tp_gained=gained, noncalibration_fp_added=added,
            tp_lost=lost, useful_no_added_false_events=gained > 0 and added == 0 and lost == 0)
    result = dict(status='PASS', cutoffs=details, metrics=summary, paired=changes, gates=gates,
        attempts_per_condition=4444, mz36_unknown_bits=80, no_fits=True, no_inference=True,
        scope='Consumed Development; candidate-domain change only, two fixed single calibrations')
    np.savez_compressed(out / 'predictions.npz', **arrays); write(out / 'result.json', result)
    write(out / 'audit.json', dict(status='PASS', scalar_known_bits=audited, exact_old_composition_values=parity_values,
        unchanged_input_scores=True, all_old_positive_decisions_retained=True, calibration_added_fp=0, mz36_unknown_bits=80))
    bind.check(); write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, seconds=time.perf_counter() - started,
        backend='CPU saved-output arithmetic; TASK_NOT_GPU_SUITABLE', training_steps=0, model_inference_frames=0,
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print('CUTOFFS', details); print('GATES', gates)
    for cohort in changes:
        print(cohort, {c: {arm: [sum(changes[cohort][c]['nonfit' if cohort == 'rich' else 'all'][arm][k]) for k in ('tp_gained', 'fp_added')]
            for arm in ARMS} for c in CONDITIONS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); root, task = args.root.resolve(), args.task.resolve()
    assert task.is_relative_to((root / 'artifacts.local').resolve()); run(root, task)
