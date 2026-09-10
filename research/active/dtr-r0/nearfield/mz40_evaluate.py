"""Identity/parity checks and saved-output scoring for the fixed MZ40 arms."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from mz5_ensemble_readout import read, write, sha, load_npz

MODELS = ('rgb', 'tof', 'MZ5', 'MZ28', 'MZ30', 'MZ35', 'MZ37')
ARMS = ('MERGE_CLOSE', 'DROP_CLOSE')
EVENT_ORDER = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def metrics(logits, truth, known):
    pred = logits >= 0
    return dict(attempted_frames=len(truth), known=known.sum(0).tolist(),
        unknown=(~known).sum(0).tolist(), tp=(pred & truth & known).sum(0).tolist(),
        fp=(pred & ~truth & known).sum(0).tolist(), fn=(~pred & truth & known).sum(0).tolist(),
        tn=(~pred & ~truth & known).sum(0).tolist(), complete_frames=int(known.all(1).sum()),
        exact_frames=int(((pred == truth).all(1) & known.all(1)).sum()))


def paired(after, before, truth, known):
    a, b = after >= 0, before >= 0
    ae = (a == truth).all(1) & known.all(1)
    be = (b == truth).all(1) & known.all(1)
    return dict(tp_gained=(a & ~b & truth & known).sum(0).tolist(),
        tp_lost=(~a & b & truth & known).sum(0).tolist(),
        fp_added=(a & ~b & ~truth & known).sum(0).tolist(),
        fp_removed=(~a & b & ~truth & known).sum(0).tolist(),
        exact_gained=int((ae & ~be).sum()), exact_lost=int((~ae & be).sum()))


def run(root, mode, output):
    root = root.resolve(); work = root/'artifacts.local/work'
    task = work/'mz40-l8cx-constrained-20260910'; prior = work/'mz36-new-source-20260910'
    output = output.resolve()
    assert output.is_relative_to((root/'artifacts.local').resolve()) and not output.exists()
    output.mkdir(parents=True); inputs = {}; started = time.perf_counter()

    def bind(path, expected=None):
        digest = sha(path); assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest; return path

    def receipt(folder):
        r = read(bind(folder/'receipt.json')); assert r['status'] == 'PASS'
        for name, digest in r['outputs'].items(): bind(folder/name, digest)
        return r

    def frozen_equal(actual, expected):
        a = {str(Path(p).resolve()): h for p, h in actual['frozen'].items()}
        for path, digest in expected['frozen'].items():
            assert a.get(str(Path(path).resolve())) == digest, path

    try:
        bind(Path(__file__).with_name('MZ40_L8CX_CONSTRAINED_PROTOCOL_20260910.md'))
        original_receipt = receipt(prior/'inference-v1')
        original = load_npz(prior/'inference-v1/predictions.npz')
        restoration_dir = work/'mz37-positive-restoration-20260910/run-v1'
        restore_receipt = receipt(restoration_dir)
        restored = load_npz(restoration_dir/'predictions.npz')
        restored_index = {v: i for i, v in enumerate(restored['MZ36/frame_ids'])}
        packet_receipt = receipt(task/'packets-v1')
        packets_manifest = read(task/'packets-v1/manifest.json')
        np.testing.assert_array_equal(original['frame_ids'], [r['frame_id'] for r in packets_manifest['frames']])

        def prediction(folder, packet_name):
            r = receipt(folder); assert r['training_steps'] == 0
            frozen_equal(r, original_receipt)
            for name in ('receipt.json', 'cutoff.npy'):
                path = restoration_dir/name
                declared = {str(Path(p).resolve()): h for p, h in r['frozen'].items()}
                assert declared.get(str(path.resolve())) == sha(path)
            normalized_inputs = {str(Path(p).resolve()): h for p, h in r['inputs'].items()}
            for name in (packet_name, 'parity-manifest.json' if mode == 'parity' else 'manifest.json'):
                path = task/'packets-v1'/name
                assert normalized_inputs.get(str(path.resolve())) == sha(path)
            assert not any('depth' in Path(p).name.lower() or '/evaluator/' in p.replace('\\', '/')
                           for p in r['inputs'])
            # The external adapter must preserve every shared frozen implementation.
            for name, digest in original_receipt['code_sha256'].items():
                if name != 'mz36_frozen_inference.py': assert r['code_sha256'][name] == digest, name
            return load_npz(folder/'predictions.npz'), r

        if mode == 'parity':
            a, r = prediction(task/'parity-v1', 'PARITY.npz')
            assert len(a['frame_ids']) == 16
            checked = {}
            for key in original:
                np.testing.assert_array_equal(a[key], original[key][:16], err_msg=key)
                checked[key] = int(a[key].size)
            take = [restored_index[v] for v in a['frame_ids']]
            for key, source in [('MZ37', 'candidate'), ('MZ37/added', 'added')]:
                np.testing.assert_array_equal(a[key], restored['MZ36/'+source][take], err_msg=key)
                checked[key] = int(a[key].size)
            result = dict(status='PASS', frames=16, exact_arrays=checked,
                checkpoint_and_shared_code_identity=True, training_steps=0,
                rule='Exact equality for all original saved arrays and frozen MZ37 candidate/added; no tolerance widening')
        else:
            assert read(bind(task/'parity-check-v1/result.json'))['status'] == 'PASS'
            receipt(prior/'admission-v1')
            admission = read(prior/'admission-v1/result.json')
            evaluator = load_npz(prior/'admission-v1/evaluator.npz')
            ids, truth, known = (evaluator[k] for k in ('frame_ids', 'truth', 'known'))
            records = admission['records']; groups = admission['groups']; ix = np.flatnonzero(known.all(1))
            assert len(ids) == 400 and len(ix) == 380 and (~known).sum() == 80
            np.testing.assert_array_equal(ids, [r['frame_id'] for r in records])
            np.testing.assert_array_equal(ids[ix], original['frame_ids'])
            np.testing.assert_array_equal(evaluator['labels'], np.where(known, truth.astype(np.int8), -1))
            logits = {'IDEAL': {}}
            for name in MODELS:
                arr = np.full((400, 4), np.nan)
                if name == 'MZ37': arr[ix] = restored['MZ36/candidate'][[restored_index[v] for v in ids[ix]]]
                else: arr[ix] = original[name]
                logits['IDEAL'][name] = arr
            coverage = {}; runtimes = {}; change_records = []
            for arm in ARMS:
                a, r = prediction(task/arm, arm+'.npz')
                np.testing.assert_array_equal(a['frame_ids'], ids[ix])
                packet = load_npz(task/'packets-v1'/(arm+'.npz'))
                for key in ('ranges', 'valid'): np.testing.assert_array_equal(a[key], packet[key])
                for key in ('rgb', 'visual'): np.testing.assert_array_equal(a[key], original[key])
                declared = {str(Path(p).resolve()): h for p, h in r['inputs'].items()}
                for row in records:
                    if row['group_accepted']: assert declared.get(str(Path(row['rgb_path']).resolve())) == row['rgb_sha256']
                logits[arm] = {}
                for name in MODELS:
                    assert a[name].shape == (380, 4) and np.isfinite(a[name]).all()
                    logits[arm][name] = np.full((400, 4), np.nan); logits[arm][name][ix] = a[name]
                    changed = (logits[arm][name] >= 0) != (logits['IDEAL'][name] >= 0)
                    for i, q in np.argwhere(changed & known):
                        change_records.append(dict(arm=arm, model=name, frame_id=str(ids[i]), query=EVENT_ORDER[q],
                            truth=bool(truth[i, q]), before=float(logits['IDEAL'][name][i, q]),
                            after=float(logits[arm][name][i, q]), family=records[i]['family'],
                            region_id=records[i]['region_id'], site_id=records[i]['site_id'], group_id=records[i]['group_id']))
                count = a['valid'].sum(2)
                coverage[arm] = dict(zone_return_counts=[int((count == n).sum()) for n in range(3)],
                    valid_slots=int(a['valid'].sum()), all_tof_missing_frames=int((~a['valid'].any((1, 2))).sum()))
                runtimes[arm] = read(task/arm/'result.json')['total_seconds']
            count = original['valid'].sum(2)
            coverage['IDEAL'] = dict(zone_return_counts=[int((count == n).sum()) for n in range(3)],
                valid_slots=int(original['valid'].sum()), all_tof_missing_frames=int((~original['valid'].any((1, 2))).sum()))
            methods = {arm: {n: metrics(z, truth, known) for n, z in values.items()} for arm, values in logits.items()}
            pairs = {arm: {n: paired(logits[arm][n], logits['IDEAL'][n], truth, known) for n in MODELS} for arm in ARMS}
            rgb_pairs = {arm: {n: paired(z, logits['IDEAL']['rgb'], truth, known) for n, z in values.items()} for arm, values in logits.items()}
            by = {}
            for field in ('region_id', 'family', 'site_id'):
                by[field] = {}
                for value in sorted({r[field] for r in records}):
                    take = np.array([i for i, r in enumerate(records) if r[field] == value])
                    by[field][value] = {arm: {n: metrics(z[take], truth[take], known[take]) for n, z in values.items()} for arm, values in logits.items()}
            group_results = []
            for g in groups:
                take = np.array(g['frame_indices']); complete = bool(known[take].all())
                group_results.append(dict(region_id=g['region_id'], group_id=g['group_id'], site_id=g['site_id'],
                    family=g['family'], admitted=complete, attempted_frames=len(take),
                    methods={arm: {n: dict(metrics=metrics(z[take], truth[take], known[take]),
                        exact_group=bool(complete and ((z[take] >= 0) == truth[take]).all())) for n, z in values.items()} for arm, values in logits.items()}))
            result = dict(status='PASS', attempted_frames=400, admitted_frames=380, excluded_frames=20,
                unknown_by_query=(~known).sum(0).tolist(), event_order=EVENT_ORDER, methods=methods,
                paired_vs_ideal=pairs, paired_vs_rgb=rgb_pairs, by=by, groups=group_results, coverage=coverage,
                inference_seconds=runtimes, training_steps=0, model_inference_frames=0,
                scope='Consumed controlled MZ36 Development; named unresolved-return proxies, ideal-trained frozen weights/bank; information loss and training mismatch are not separated')
            np.savez_compressed(output/'scored.npz', frame_ids=ids, truth=truth, known=known,
                **{arm+'/'+n: z for arm, values in logits.items() for n, z in values.items()})
            write(output/'changes.json', change_records)
        write(output/'result.json', result)
        for path, digest in inputs.items(): assert sha(path) == digest, path
        write(output/'receipt.json', dict(status='PASS', mode=mode, inputs=inputs,
            code_sha256={Path(__file__).name: sha(Path(__file__))}, training_steps=0, model_inference_frames=0,
            backend='CPU saved-array evaluation; TASK_NOT_GPU_SUITABLE', seconds=time.perf_counter()-started,
            outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
        print('PASS', mode, flush=True)
        if mode == 'score':
            for arm, values in methods.items():
                print(arm, {n: dict(fp=sum(m['fp']), fn=sum(m['fn']), exact=m['exact_frames']) for n, m in values.items()}, flush=True)
    except Exception:
        write(output/'failure.json', dict(status='FAIL', mode=mode, inputs=inputs, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('parity', 'score'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.mode, a.output)
