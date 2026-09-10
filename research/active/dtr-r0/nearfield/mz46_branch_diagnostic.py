"""Independently audit saved MZ46 branch swaps; never invoke a model or scorer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
METHODS = ('rgb', 'tof', 'MZ5', 'MZ28', 'MZ35', 'MZ37', 'MZ43_TOF',
           'MZ43_ENSEMBLE', 'IDEAL_FUSION', 'MZ43_FUSION')
EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
COMBOS = ('old_rgb_old_tof', 'old_rgb_new_tof', 'new_rgb_old_tof', 'new_rgb_new_tof')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metrics(logits, truth, known):
    out = {key: [0] * 4 for key in ('tp', 'fp', 'fn', 'tn', 'known', 'unknown')}
    exact = 0
    for i in range(4):
        correct = True
        for q in range(4):
            if not bool(known[i, q]):
                out['unknown'][q] += 1
                correct = False
                continue
            out['known'][q] += 1
            predicted, target = float(logits[i, q]) >= 0., bool(truth[i, q])
            key = ('tp' if target else 'fp') if predicted else ('fn' if target else 'tn')
            out[key][q] += 1
            correct &= predicted == target
        exact += int(correct)
    out['exact_frames'] = exact
    out['totals'] = {key: sum(out[key]) for key in ('tp', 'fp', 'fn', 'tn')}
    return out


def diagnostic(task):
    paths = ('prepared-v1/evaluator.npz', 'prepared-v1/native-pairs.json',
             'inference-v1/predictions.npz', 'inference-v1/receipt.json',
             'inference-v1/parity.json', 'score-v1/result.json',
             'score-v1/paired-logits.json', 'score-v1/audit.json')
    bindings = {name: sha(task / name) for name in paths}
    labels = np.load(task / paths[0], allow_pickle=False)
    data = np.load(task / paths[2], allow_pickle=False)
    native = read(task / paths[1])
    inference_receipt = read(task / paths[3])
    for name in ('predictions.npz', 'parity.json'):
        assert sha(task / 'inference-v1' / name) == inference_receipt['outputs'][name]
    parity = read(task / paths[4])
    upstream_audit = read(task / paths[7])
    assert parity['status'] == upstream_audit['status'] == 'PASS'
    scored = read(task / paths[5])
    saved_pairs = read(task / paths[6])['records']
    assert labels['truth'].shape == labels['known'].shape == (8, 4)
    np.testing.assert_array_equal(data['frame_ids'], labels['frame_ids'])
    np.testing.assert_array_equal(labels['truth'][:4], labels['truth'][4:])
    np.testing.assert_array_equal(labels['known'][:4], labels['known'][4:])
    assert labels['known'].all()
    identifiers = [str(x).split('/', 1)[1] for x in data['frame_ids'][:4]]
    assert identifiers == [str(x).split('/', 1)[1] for x in data['frame_ids'][4:]]
    assert len(native['pairs']) == 4
    for row, identifier in zip(native['pairs'], identifiers):
        assert row['original_frame_id'] == identifier and row['target_dictionary_equal']
        assert row['native_labels_equal'] and row['changed_event_mask_pixels'] == [0] * 4
        assert row['event_depth_max_abs_m'] == 0 and row['context_attribution_eligible']
    pair_map = {(p['condition'], p['method'], p['original_frame_id']): p for p in saved_pairs}
    assert len(pair_map) == 120
    results, invariance, changed = {}, {}, []
    for condition in CONDITIONS:
        invariance[condition] = {}
        for name in ('rgb', 'tof', 'MZ43_TOF'):
            actual, hybrid = data[condition + '/' + name], data['hybrid/' + condition + '/' + name]
            expected = actual if name == 'rgb' else np.concatenate((actual[4:], actual[:4]))
            np.testing.assert_array_equal(hybrid, expected)
            invariance[condition][name + '_unrelated_branch_swap_exact'] = True
        np.testing.assert_array_equal(data[condition + '/rgb'], data['IDEAL/rgb'])
        for prefix in (condition + '/', 'hybrid/' + condition + '/'):
            for ensemble, tof in (('MZ5', 'tof'), ('MZ43_ENSEMBLE', 'MZ43_TOF')):
                expected = (data[prefix + 'rgb'] + data[prefix + tof]) * np.float32(.5)
                np.testing.assert_array_equal(expected, data[prefix + ensemble])
        invariance[condition]['both_fixed_ensembles_exact_all_actual_hybrid'] = True
        results[condition] = {}
        for method in METHODS:
            actual, hybrid = data[condition + '/' + method], data['hybrid/' + condition + '/' + method]
            assert actual.shape == hybrid.shape == (8, 4)
            assert np.isfinite(actual).all() and np.isfinite(hybrid).all()
            arrays = (actual[:4], hybrid[:4], hybrid[4:], actual[4:])
            result = {name: metrics(a, labels['truth'][:4], labels['known'][:4]) for name, a in zip(COMBOS, arrays)}
            for name, reported in (('old_rgb_old_tof', 'supported'), ('new_rgb_new_tof', 'unsupported')):
                reference = scored['conditions'][condition]['methods'][method][reported]
                for key in ('tp', 'fp', 'fn', 'tn', 'known', 'unknown', 'exact_frames'):
                    assert result[name][key] == reference[key], (condition, method, name, key)
            changes = {'lost_true_alarms': [], 'new_false_alarms': []}
            for i, identifier in enumerate(identifiers):
                logged = pair_map[(condition, method, identifier)]
                for combo, a in zip(COMBOS, arrays):
                    np.testing.assert_array_equal(a[i], logged['logits'][combo])
                for q, event in enumerate(EVENTS):
                    pred = [float(a[i, q]) >= 0. for a in arrays]
                    target = bool(labels['truth'][i, q])
                    kind = 'lost_true_alarms' if target and pred[0] and not pred[3] else (
                        'new_false_alarms' if not target and not pred[0] and pred[3] else None)
                    if kind:
                        adverse = (lambda x: not x) if target else bool
                        rgb_alone, tof_alone = adverse(pred[2]), adverse(pred[1])
                        category = ('either_branch_alone_sufficient' if rgb_alone and tof_alone else
                                    'rgb_alone_sufficient' if rgb_alone else 'tof_alone_sufficient' if tof_alone else
                                    'joint_change_required_at_threshold')
                        entry = dict(original_frame_id=identifier, event=event, category=category,
                                     logits={name: float(a[i, q]) for name, a in zip(COMBOS, arrays)})
                        changes[kind].append(entry)
                        changed.append(dict(condition=condition, method=method, change=kind, **entry))
            result['paired_adverse_changes'] = changes
            results[condition][method] = result
    output = dict(status='PASS_SAVED_OUTPUT_ONLY', inputs=bindings, code_sha256=sha(__file__),
        methods=list(METHODS), conditions=list(CONDITIONS), event_order=list(EVENTS), combinations=list(COMBOS),
        scalar_scored_event_bits=3 * 10 * 4 * 4 * 4, actual_score_comparison='All60 actual method/state metrics match saved scores',
        hybrid_logit_comparison='All120 paired-logit records match saved arrays exactly',
        invariance=invariance, native_evidence='Consumed hashed native-pairs report: allfour event masks and event-surface depths unchanged; this analysis does not re-render/recompute native geometry',
        metrics=results, adverse_event_changes=changed,
        findings=['RGB branch loses allthree formerly detected positive bits with only the RGB source changed; exact packet-swap invariance holds',
            'ToF is independently context-sensitive: MZ43_TOF true positives change3to2 IDEAL,4to1 MERGE_CLOSE,4to0 DROP_CLOSE; exact RGB-swap invariance holds',
            'MZ43_ENSEMBLE IDEAL loses two true alarms with RGB-only swap, but the third requires both branch changes to cross threshold',
            'MZ43_ENSEMBLE MERGE_CLOSE/DROP_CLOSE: either branch change alone loses the same HEAD_FAR alarm; three remaining alarms require joint changes to cross threshold',
            'The complete collapse of MZ43_ENSEMBLE is therefore not RGB-only. Joint threshold crossings of two degraded branch logits occur even though fixed ensemble arithmetic is exactly additive',
            'MZ35/MZ37 retain allthree supported-source true alarms in every condition; ideal/merged each add one FP and dropped adds none. Context sensitivity does not collapse every tested method'],
        causal_scope=['Saved synthetic branch interventions identify sensitivity of these frozen models on four matched cases, not a population average causal effect',
            'RGB-only swaps include all visual effects of support removal, including changed lighting/shadows/occlusion; they do not identify a learned semantic scaffold detector',
            'ToF-only swaps include changed out-of-corridor supports and resulting zone summaries; unchanged target pixels do not imply unchanged model observation',
            'Threshold sufficiency can differ by event and condition; neither total TP changes nor logit changes identify a unique internal mechanism',
            'One target geometry, size, site, camera and four states; cannot settle new-form size-versus-shape failure or deployment/sensor behavior'],
        training_steps=0, model_inference_frames=0, capture_frames=0)
    path = task / 'branch-diagnostic.json'
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': output['status'], 'sha256': sha(path), 'MZ43_ENSEMBLE': {
        c: {k: results[c]['MZ43_ENSEMBLE'][k]['totals'] for k in COMBOS} for c in CONDITIONS}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    diagnostic(args.task.resolve())
