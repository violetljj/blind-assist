"""Independent saved-array MZ40 audit; no inference or mutation implementation imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


ARMS = ('MERGE_CLOSE', 'DROP_CLOSE')
MODELS = ('rgb', 'tof', 'MZ5', 'MZ28', 'MZ30', 'MZ35', 'MZ37')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def equal(a, b):
    np.testing.assert_array_equal(a, b)


def scalar_metrics(scores, truth, known):
    result = {key: [0] * 4 for key in ('tp', 'fp', 'fn', 'tn', 'known', 'unknown')}
    exact = 0
    for i in range(len(scores)):
        correct = 0
        for q in range(4):
            if not bool(known[i, q]):
                result['unknown'][q] += 1
                assert np.isnan(scores[i, q]), (i, q, 'excluded score must stay NaN')
                continue
            assert np.isfinite(scores[i, q]), (i, q)
            result['known'][q] += 1
            positive, target = float(scores[i, q]) >= 0., bool(truth[i, q])
            key = ('tp' if target else 'fp') if positive else ('fn' if target else 'tn')
            result[key][q] += 1
            correct += positive == target
        exact += correct == 4
    result['exact_frames'] = exact
    result['attempted_frames'] = len(scores)
    result['complete_frames'] = sum(all(row) for row in known)
    return result


def scalar_pairs(scores, baseline, truth, known):
    counts = {key: [0] * 4 for key in ('tp_gained', 'tp_lost', 'fp_added', 'fp_removed')}
    counts.update(exact_gained=0, exact_lost=0)
    for i in range(len(scores)):
        if all(known[i]):
            after_exact = all((float(scores[i, q]) >= 0.) == bool(truth[i, q]) for q in range(4))
            before_exact = all((float(baseline[i, q]) >= 0.) == bool(truth[i, q]) for q in range(4))
            counts['exact_gained'] += after_exact and not before_exact
            counts['exact_lost'] += before_exact and not after_exact
        for q in range(4):
            if not bool(known[i, q]):
                continue
            positive, before = float(scores[i, q]) >= 0., float(baseline[i, q]) >= 0.
            if positive != before:
                key = ('tp_gained' if positive else 'tp_lost') if truth[i, q] else ('fp_added' if positive else 'fp_removed')
                counts[key][q] += 1
    return counts


def audit(task, packet_only=False):
    task = task.resolve()
    output = task / ('packet-audit.json' if packet_only else 'audit.json')
    if output.exists():
        raise FileExistsError(output)
    checked = {}
    checks = []

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = checked.get(str(path)) or sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        checked[str(path)] = digest
        return path

    def receipt(directory, inputs=True):
        document = read(bind(directory / 'receipt.json'))
        assert document['status'] == 'PASS', str(directory)
        if inputs:
            for path, digest in document.get('inputs', {}).items():
                bind(path, digest)
        for name, digest in document.get('outputs', {}).items():
            bind(directory / name, digest)
        return document

    try:
        prior = task.parent / 'mz36-new-source-20260910'
        packets = task / 'packets-v1'
        receipt(packets)
        old_receipt = receipt(prior / 'inference-v1')
        old = arrays(bind(prior / 'inference-v1/predictions.npz'))
        manifest = read(bind(packets / 'manifest.json'))
        original_manifest = read(bind(prior / 'admission-v1/predictor-manifest.json'))
        assert set(manifest) == {'camera', 'frames'}
        assert manifest['camera'] == original_manifest['camera']
        assert len(manifest['frames']) == len(old['frame_ids']) == 380
        rgb_bindings = {str(Path(p).resolve()): h for p, h in old_receipt['inputs'].items()}
        for i, (row, original) in enumerate(zip(manifest['frames'], original_manifest['frames'])):
            assert set(row) == {'frame_id', 'rgb_path'}
            assert row == {key: original[key] for key in row}
            assert row['frame_id'] == old['frame_ids'][i]
            rgb_path = Path(row['rgb_path']).resolve()
            bind(rgb_path, rgb_bindings[str(rgb_path)])
        parity_manifest = read(bind(packets / 'parity-manifest.json'))
        assert parity_manifest == dict(camera=manifest['camera'], frames=manifest['frames'][:16])
        parity = arrays(bind(packets / 'PARITY.npz'))
        for key in ('frame_ids', 'ranges', 'valid'):
            equal(parity[key], old[key][:16])
        ambiguity = arrays(bind(packets / 'ambiguity.npz'))
        equal(ambiguity['frame_ids'], old['frame_ids'])
        original_ranges, original_valid = old['ranges'], old['valid']
        assert original_ranges.shape == original_valid.shape == (380, 64, 2)
        assert original_ranges.dtype == np.float32 and original_valid.dtype == np.bool_
        expected_close = np.zeros((380, 64), dtype=bool)
        for i in range(380):
            for zone in range(64):
                valid = [bool(original_valid[i, zone, slot]) for slot in range(2)]
                first, last = (float(original_ranges[i, zone, slot]) for slot in range(2))
                expected_close[i, zone] = all(valid) and last - first < .600
        equal(ambiguity['unresolved'], expected_close)
        assert sum(bool(v) for row in expected_close for v in row) == 4234
        assert sum(any(row) for row in expected_close) == 346
        packet_result = read(bind(packets / 'result.json'))
        packet_stats = {}
        for arm, expected_hist in zip(ARMS, ([16592, 7069, 659], [20826, 2835, 659])):
            actual = arrays(bind(packets / (arm + '.npz')))
            equal(actual['frame_ids'], old['frame_ids'])
            assert actual['ranges'].dtype == np.float32 and actual['valid'].dtype == np.bool_
            assert actual['ranges'].shape == actual['valid'].shape == (380, 64, 2)
            histogram, missing = [0, 0, 0], 0
            for i in range(380):
                frame_returns = 0
                for zone in range(64):
                    expected_range = [float(v) for v in original_ranges[i, zone]]
                    expected_valid = [bool(v) for v in original_valid[i, zone]]
                    if expected_close[i, zone]:
                        if arm == 'MERGE_CLOSE':
                            expected_range = [float(np.float32(sum(expected_range) / 2.)), 0.]
                            expected_valid = [True, False]
                        else:
                            expected_range, expected_valid = [0., 0.], [False, False]
                    count = sum(expected_valid)
                    histogram[count] += 1
                    frame_returns += count
                    for slot in range(2):
                        assert float(actual['ranges'][i, zone, slot]) == expected_range[slot], (arm, i, zone, slot)
                        assert bool(actual['valid'][i, zone, slot]) == expected_valid[slot]
                        assert expected_valid[slot] or expected_range[slot] == 0.
                missing += frame_returns == 0
            assert histogram == expected_hist
            stats = dict(zone_return_counts=histogram, valid_slots=histogram[1] + 2 * histogram[2],
                         all_tof_missing_frames=missing, changed_zones=4234, changed_frames=346)
            assert stats == packet_result['arms'][arm]
            packet_stats[arm] = stats
        frozen = read(bind(prior / 'frozen-models.json'))['frozen']
        for path, digest in frozen.items():
            bind(path, digest)
        checks += ['Independent scalar replay of 48640 arm-zone mutations and close identities',
                   'Original MZ36 packets, frame order, 380 RGB bytes and camera unchanged',
                   '16-frame parity inputs identical; zero evaluator fields in predictor manifests',
                   'Original frozen model, normalization, cutoff and bank hashes']
        result = dict(status='PASS',scope='PACKETS_ONLY' if packet_only else 'FULL_SAVED_OUTPUT_AUDIT',
                      packet_stats=packet_stats,checks=checks,training_steps=0,model_inference_frames=0)
        if not packet_only:
            receipt(task / 'score-v1')
            scored = arrays(bind(task / 'score-v1/scored.npz'))
            reported = read(bind(task / 'score-v1/result.json'))
            assert reported['status'] == 'PASS'
            source = arrays(bind(prior / 'admission-v1/evaluator.npz'))
            receipt(prior / 'admission-v1')
            for key in ('frame_ids', 'truth', 'known'):
                equal(scored[key], source[key])
            ids, truth, known = (scored[key] for key in ('frame_ids', 'truth', 'known'))
            assert len(ids) == len(set(ids.tolist())) == 400 and known.shape == (400, 4)
            assert known.sum() == 1520 and (~known).sum() == 80
            assert sum(all(row) for row in known) == 380
            assert sum(not any(row) for row in known) == 20
            lookup = {frame: i for i, frame in enumerate(ids.tolist())}
            positions = [lookup[frame] for frame in old['frame_ids'].tolist()]
            for model in MODELS[:-1]:
                equal(scored['IDEAL/' + model][positions], old[model])
            restoration = task.parent / 'mz37-positive-restoration-20260910/run-v1'
            receipt(restoration)
            restore = arrays(bind(restoration / 'predictions.npz'))
            for key in ('frame_ids', 'truth', 'known'):
                equal(scored[key], restore['MZ36/' + key])
            equal(scored['IDEAL/MZ37'], restore['MZ36/candidate'])
            frozen_expected = {str(Path(p).resolve()): h for p, h in old_receipt['frozen'].items()}
            for name in ('receipt.json', 'cutoff.npy'):
                path = bind(restoration / name)
                frozen_expected[str(path)] = checked[str(path)]
            for stage in ('parity-v1',) + ARMS:
                current = receipt(task / stage)
                assert current['training_steps'] == 0
                frozen_actual = {str(Path(p).resolve()): h for p, h in current['frozen'].items()}
                assert frozen_actual == frozen_expected, stage
                for path, digest in frozen_actual.items():
                    bind(path, digest)
                for name, digest in old_receipt['code_sha256'].items():
                    if name != 'mz36_frozen_inference.py':
                        assert current['code_sha256'][name] == digest, (stage, name)
                stage_inputs = {str(Path(p).resolve()): h for p, h in current['inputs'].items()}
                for path in stage_inputs:
                    assert 'depth' not in Path(path).name.lower()
                    assert '/evaluator/' not in path.replace('\\', '/')
                packet_name = 'PARITY.npz' if stage == 'parity-v1' else stage + '.npz'
                manifest_name = 'parity-manifest.json' if stage == 'parity-v1' else 'manifest.json'
                for name in (packet_name, manifest_name):
                    path = bind(packets / name)
                    assert stage_inputs[str(path)] == checked[str(path)]
                emitted = arrays(bind(task / stage / 'predictions.npz'))
                injected = arrays(bind(packets / packet_name))
                for key in ('frame_ids', 'ranges', 'valid'):
                    equal(emitted[key], injected[key])
                if stage == 'parity-v1':
                    for key, values in old.items():
                        equal(emitted[key], values[:16])
                    equal(emitted['MZ37'], restore['MZ36/candidate'][positions[:16]])
                    equal(emitted['MZ37/added'], restore['MZ36/added'][positions[:16]])
                else:
                    for key in ('rgb', 'visual'):
                        equal(emitted[key], old[key])
                    for model in MODELS:
                        equal(scored[stage + '/' + model][positions], emitted[model])
            receipt(task / 'parity-check-v1')
            assert read(bind(task / 'parity-check-v1/result.json'))['status'] == 'PASS'
            counts = {}
            for arm in ('IDEAL',) + ARMS:
                counts[arm] = {}
                for model in MODELS:
                    values = scored[arm + '/' + model]
                    measured = scalar_metrics(values, truth, known)
                    for key, value in measured.items():
                        assert reported['methods'][arm][model][key] == value, (arm, model, key)
                    counts[arm][model] = measured
                    if arm != 'IDEAL':
                        assert scalar_pairs(values, scored['IDEAL/' + model], truth, known) == reported['paired_vs_ideal'][arm][model]
                    assert scalar_pairs(values, scored['IDEAL/rgb'], truth, known) == reported['paired_vs_rgb'][arm][model]
                equal(scored[arm + '/rgb'], scored['IDEAL/rgb'])
            assert reported['attempted_frames'] == 400 and reported['admitted_frames'] == 380
            assert reported['excluded_frames'] == 20 and reported['unknown_by_query'] == [20] * 4
            for arm in ARMS:
                assert reported['coverage'][arm] == {key: value for key, value in packet_stats[arm].items()
                                                     if key not in ('changed_zones', 'changed_frames')}
            ideal_hist = [0, 0, 0]
            for row in original_valid:
                for zone in row:
                    ideal_hist[sum(bool(v) for v in zone)] += 1
            assert reported['coverage']['IDEAL'] == dict(zone_return_counts=ideal_hist,
                valid_slots=ideal_hist[1] + 2 * ideal_hist[2],
                all_tof_missing_frames=sum(not any(bool(v) for zone in row for v in zone) for row in original_valid))
            admission = read(bind(prior / 'admission-v1/result.json'))
            records = admission['records']
            equal(ids, [record['frame_id'] for record in records])
            for field in ('region_id', 'family', 'site_id'):
                labels = sorted({record[field] for record in records})
                assert sorted(reported['by'][field]) == labels
                for label in labels:
                    take = [i for i, record in enumerate(records) if record[field] == label]
                    for arm in ('IDEAL',) + ARMS:
                        for model in MODELS:
                            expected = scalar_metrics(scored[arm + '/' + model][take], truth[take], known[take])
                            assert expected == reported['by'][field][label][arm][model], (field, label, arm, model)
            assert len(admission['groups']) == len(reported['groups']) == 80
            group_indices = []
            for original_group, group in zip(admission['groups'], reported['groups']):
                take = original_group['frame_indices']
                group_indices.extend(take)
                complete = all(all(known[i]) for i in take)
                for field in ('region_id', 'group_id', 'site_id', 'family', 'attempted_frames'):
                    assert group[field] == original_group[field]
                assert group['admitted'] == complete == original_group['accepted']
                for arm in ('IDEAL',) + ARMS:
                    for model in MODELS:
                        expected = scalar_metrics(scored[arm + '/' + model][take], truth[take], known[take])
                        exact_group = complete and expected['exact_frames'] == len(take)
                        assert group['methods'][arm][model] == dict(metrics=expected, exact_group=exact_group)
            assert sorted(group_indices) == list(range(400))
            checks += ['400 frame identities/truth; 20 excluded frames and 80 UNKNOWN bits unchanged',
                       'Scalar confusion, exact frames and every paired bit change',
                       'Original six ideal methods and MZ37 restoration scores unchanged',
                       'Frozen bindings, shared inference code and 16-frame exact adapter parity',
                       'Emitted injected packets and RGB/visual arrays unchanged where required',
                       'RGB output scores unchanged in all arms',
                       'Region/family/site metrics, 80 complete-group decisions and packet coverage']
            result.update(methods=counts,scored_known_bits=1520 * 21,attempted_frames=400)
        result.update(inputs=checked,code_sha256=sha(__file__),
                      limits='Saved-output audit; does not establish physical sensor fidelity or distinguish information loss from training mismatch')
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(json.dumps({key: result[key] for key in ('status', 'scope', 'checks')}))
    except Exception:
        failure = task / ('audit-failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        failure.write_text(json.dumps(dict(status='FAIL',packet_only=packet_only,checks_completed=checks,
                                          inputs=checked,code_sha256=sha(__file__),error=traceback.format_exc()),indent=2) + '\n',
                           encoding='utf-8',newline='\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--packet-only', action='store_true')
    arguments = parser.parse_args()
    audit(arguments.task, arguments.packet_only)
