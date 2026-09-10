"""Fixed-model paired support removal, with extraction-free native checks."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from data_lightweight import CompactSource
from multizone64_observation import observe
from mz5_ensemble_readout import read, write, sha, load_npz
from mz36_frozen_inference import CAMERA
from mz40_packets import constrain
from mz40_evaluate import metrics, paired
from mz45_object_transfer import Bindings, FrozenModels, CONDITIONS, METHODS, setup

CONTEXTS = ('supported', 'unsupported')
COMBINATIONS = ('old_rgb_old_tof', 'new_rgb_new_tof', 'old_rgb_new_tof', 'new_rgb_old_tof')
BRIEF = 'MZ46_SCAFFOLD_CONTEXT_20260911.md'


def bind_code(bind):
    for name in (Path(__file__).name, BRIEF, 'mz45_object_transfer.py',
                 'mz40_packets.py', 'multizone64_observation.py', 'data_lightweight.py'):
        bind(Path(__file__).with_name(name))


def native_masks(depth):
    """Independent NumPy reconstruction of frozen native query geometry."""
    yy, xx = np.indices((360, 640))
    focal = 320 / np.tan(np.deg2rad(50))
    right, up = (xx - 319.5) / focal, -(yy - 179.5) / focal
    d = depth.astype(float)
    lateral, height = d * right, 1.7 + d * up
    valid = np.isfinite(d) & (d > 0) & (d < 100) & (d * np.sqrt(1 + right**2 + up**2) <= 4)
    masks = []
    for front, width, low, high in ((.18, .28, .65, 1.4), (.13, .18, 1.4, 1.85)):
        for half in (0, 1):
            masks.append(valid & (d >= front + 1.5 * half)
                & ((d < front + 1.5) if half == 0 else (d <= front + 3))
                & (lateral >= -width) & (lateral <= width) & (height >= low) & (height <= high))
    return np.stack(masks)


def prepare(root, task):
    out = task / 'prepared-v1'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); setup(); bind = Bindings(); bind_code(bind)
    source_dirs = (root / 'artifacts.local/work/mz44-rich-far-objects-20260911', task)
    manifest = read(bind(task / 'spec-v1/manifest.json'))
    frozen = read(bind(task / 'spec-v1/spec.json', manifest['spec_sha256']))
    cases = frozen['cases']; assert len(cases) == 4
    case_ids = [c['name'] for c in cases]
    frames, groups, truths, knowns, depths, masks, ranges, validity = ([] for _ in range(8))
    specs = []
    for context, folder in zip(CONTEXTS, source_dirs):
        returned = folder / 'returned-v1'
        packing = read(bind(returned / 'pack.json')); assert packing['status'] == 'PASS'
        verification = read(bind(returned / 'verify.json'))
        assert verification['status'] == 'PASS' and verification['all_sha256_match']
        archive = bind(returned / 'capture.source.zip', packing['archive_sha256'])
        receipt = read(bind(returned / 'dataset-v1/receipt.json')); assert receipt['status'] == 'PASS'
        dataset = read(bind(returned / 'dataset-v1/result.json', receipt['outputs']['result.json']))
        labels = load_npz(bind(returned / 'dataset-v1/labels.npz', receipt['outputs']['labels.npz']))
        np.testing.assert_array_equal(labels['frame_ids'], [r['frame_id'] for r in dataset['records']])
        with CompactSource(archive) as compact:
            assert compact.read_json('source-integrity.json')['unchanged']
            assert compact.read_json('render-resource-health.json')['ready_data_eligible']
            spec = compact.read_json('source/spec.json'); specs.append(spec)
            lookup = {c['name']: c for c in spec['cases']}
            for case_id in case_ids:
                index = list(labels['frame_ids']).index(case_id)
                row, case = dataset['records'][index], lookup[case_id]
                assert case['camera']['pitch'] == case['camera']['roll'] == case['camera']['yaw'] == 0
                assert abs(case['camera']['z'] - row['floor']['declared_floor_z_m'] - 1.7) < 1e-6
                for key in ('rgb', 'native'):
                    assert compact.entries[row[key]]['sha256'] == row[key + '_sha256']
                depth = compact.load_array(row['native'])
                assert depth.dtype == np.float32 and depth.shape == (360, 640)
                mask = native_masks(depth); count = mask.sum((1, 2))
                np.testing.assert_array_equal(count, row['event_counts'])
                np.testing.assert_array_equal(count >= 3, labels['truth'][index])
                frame_id = context + '/' + case_id
                frames.append(dict(frame_id=frame_id, archive=str(archive), rgb=row['rgb']))
                groups.append(dict(frame_id=frame_id, original_frame_id=case_id, context=context,
                    variant=case['range_variant'], native_event_counts=count.tolist()))
                truths.append(labels['truth'][index]); knowns.append(labels['known'][index])
                depths.append(depth); masks.append(mask)
            with torch.inference_mode():
                packet = observe(torch.from_numpy(np.stack(depths[-4:])).cuda(), readout='multi_surface')
            ranges.append(torch.nan_to_num(packet['range_m']).float().cpu().numpy())
            validity.append(packet['valid'].cpu().numpy())
    before, after = specs
    assert after == frozen
    assert {k: v for k, v in before.items() if k != 'cases'} == {k: v for k, v in after.items() if k != 'cases'}
    old_cases = {c['name']: c for c in before['cases']}
    comparisons = []
    for i, case in enumerate(cases):
        old = old_cases[case['name']]
        assert {k: v for k, v in case.items() if k != 'objects'} == {k: v for k, v in old.items() if k != 'objects'}
        target = [o for o in old['objects'] if o['name'] == 'adjustable_cross_member']
        assert case['objects'] == target and len(old['objects']) == 13 and len(target) == 1
        changed = (masks[i] != masks[i + 4]).sum((1, 2))
        union = (masks[i] | masks[i + 4]).any(0)
        delta = np.abs(depths[i].astype(float) - depths[i + 4].astype(float))
        same_labels = bool(np.array_equal(truths[i], truths[i + 4]))
        eligible = bool(np.array_equal(knowns[i], knowns[i + 4]) and np.all(knowns[i]) and same_labels and not changed.any())
        comparisons.append(dict(original_frame_id=case['name'], target_dictionary_equal=True,
            native_labels_equal=same_labels, changed_event_mask_pixels=changed.tolist(),
            event_depth_max_abs_m=float(delta[union].max()) if union.any() else 0.,
            native_exact_pixels=int((depths[i] == depths[i + 4]).sum()),
            context_attribution_eligible=eligible))
    rr, vv = np.concatenate(ranges), np.concatenate(validity)
    assert rr.shape == vv.shape == (8, 64, 2)
    packets = dict(frame_ids=np.array([r['frame_id'] for r in frames]), ranges=rr, valid=vv)
    for condition in CONDITIONS[1:]:
        altered, _ = constrain(rr, vv, condition)
        packets.update({condition + '/' + k: v for k, v in altered.items()})
    np.savez_compressed(out / 'packets.npz', **packets)
    np.savez_compressed(out / 'evaluator.npz', frame_ids=packets['frame_ids'], truth=np.stack(truths), known=np.stack(knowns))
    write(out / 'predictor.json', dict(camera=CAMERA, frames=frames, replay_frame_ids=case_ids))
    write(out / 'groups.json', dict(records=groups))
    write(out / 'native-pairs.json', dict(status='PASS', independently_recounted_event_bits=32, pairs=comparisons))
    bind.check()
    write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, frames=8, new_frames=4,
        backend='CUDA packet generation; independent NumPy native audit', device=torch.cuda.get_device_name(),
        seconds=time.perf_counter() - started, training_steps=0, extracted_files=0,
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print('PREPARED', comparisons, flush=True)


def infer(root, task):
    prepared = task / 'prepared-v1'; out = task / 'inference-v1'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); setup(); bind = Bindings(); bind_code(bind)
    pr = read(bind(prepared / 'receipt.json')); assert pr['status'] == 'PASS'
    manifest = read(bind(prepared / 'predictor.json', pr['outputs']['predictor.json']))
    packets = load_npz(bind(prepared / 'packets.npz', pr['outputs']['packets.npz']))
    assert manifest['camera'] == CAMERA and len(manifest['frames']) == 8
    np.testing.assert_array_equal(packets['frame_ids'], [r['frame_id'] for r in manifest['frames']])
    previous = root / 'artifacts.local/work/mz45-object-transfer-20260911/inference-v1'
    previous_receipt = read(bind(previous / 'receipt.json')); assert previous_receipt['status'] == 'PASS'
    reference = load_npz(bind(previous / 'predictions.npz', previous_receipt['outputs']['predictions.npz']))
    take = [list(reference['frame_ids']).index(name) for name in manifest['replay_frame_ids']]
    models = None
    try:
        models = FrozenModels(root, bind)
        write(out / 'start.json', dict(status='STARTED', inputs=bind.inputs.copy(), frames=8,
            new_rendered_frames=4, fitting_steps=0, device=torch.cuda.get_device_name(),
            observation_contract='RGB plus range/validity and fixed calibration; no evaluator/groups/native-pairs reads',
            hybrid_contract='Prespecified old/new RGB x old/new ToF; diagnostic only'))
        with torch.inference_mode(), ExitStack() as stack:
            sources = {p: stack.enter_context(CompactSource(bind(p))) for p in {r['archive'] for r in manifest['frames']}}
            images = [sources[r['archive']].load_image(r['rgb']) for r in manifest['frames']]
            torch.cuda.synchronize(); tick = time.perf_counter(); visual, detail = models.visual(images)
            torch.cuda.synchronize(); visual_seconds = time.perf_counter() - tick
            arrays = dict(frame_ids=packets['frame_ids']); parity = {}; readout_seconds = 0.
            for condition in CONDITIONS:
                prefix = '' if condition == 'IDEAL' else condition + '/'
                rr, vv = packets[prefix + 'ranges'], packets[prefix + 'valid']
                torch.cuda.synchronize(); tick = time.perf_counter()
                actual = models.predict(visual, detail, rr, vv)
                swapped = models.predict(visual, detail, rr[[4, 5, 6, 7, 0, 1, 2, 3]], vv[[4, 5, 6, 7, 0, 1, 2, 3]])
                torch.cuda.synchronize(); readout_seconds += time.perf_counter() - tick
                for key, value in actual.items():
                    np.testing.assert_array_equal(value[:4], reference[condition + '/' + key][take], err_msg=condition + '/' + key)
                    arrays[condition + '/' + key] = value
                for key, value in swapped.items():
                    arrays['hybrid/' + condition + '/' + key] = value
                parity[condition] = dict(exact_arrays=len(actual), exact_elements=sum(v[:4].size for v in actual.values()))
            np.savez_compressed(out / 'predictions.npz', **arrays)
            write(out / 'parity.json', dict(status='PASS', retained_frames=4, conditions=parity))
        bind.check()
        write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, frames=8, training_steps=0,
            backend='CUDA', device=torch.cuda.get_device_name(), visual_seconds=visual_seconds,
            readout_seconds=readout_seconds, total_seconds=time.perf_counter() - started,
            evaluator_labels_read=False, native_arrays_read=False, extracted_files=0, permanent_dense_cache=False,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
        print('INFERENCE PASS', visual_seconds, readout_seconds, flush=True)
    except Exception:
        write(out / 'failure.json', dict(status='FAILED', error=traceback.format_exc())); raise
    finally:
        del models; torch.cuda.empty_cache()


def score(root, task):
    out = task / 'score-v1'; out.mkdir(exist_ok=False); bind = Bindings(); bind_code(bind)
    prepared, inference = task / 'prepared-v1', task / 'inference-v1'
    pr = read(bind(prepared / 'receipt.json')); ir = read(bind(inference / 'receipt.json'))
    assert pr['status'] == ir['status'] == 'PASS'
    data = load_npz(bind(inference / 'predictions.npz', ir['outputs']['predictions.npz']))
    labels = load_npz(bind(prepared / 'evaluator.npz', pr['outputs']['evaluator.npz']))
    pairs = read(bind(prepared / 'native-pairs.json', pr['outputs']['native-pairs.json']))['pairs']
    groups = read(bind(prepared / 'groups.json', pr['outputs']['groups.json']))['records']
    np.testing.assert_array_equal(labels['frame_ids'], data['frame_ids'])
    truth, known = labels['truth'], labels['known']
    eligible = np.array([p['context_attribution_eligible'] for p in pairs])
    pair_known = known[:4] & known[4:] & eligible[:, None]
    result, records, scalar_bits = {}, [], 0
    for condition in CONDITIONS:
        methods = {}
        for method in METHODS:
            actual, hybrid = data[condition + '/' + method], data['hybrid/' + condition + '/' + method]
            before, after = actual[:4], actual[4:]
            methods[method] = dict(supported=metrics(before, truth[:4], known[:4]),
                unsupported=metrics(after, truth[4:], known[4:]),
                invariant_pairs=paired(after, before, truth[:4], pair_known))
            for i in range(4):
                scores = (before[i], after[i], hybrid[i], hybrid[i + 4])
                records.append(dict(original_frame_id=pairs[i]['original_frame_id'], variant=groups[i]['variant'],
                    condition=condition, method=method, eligible=bool(eligible[i]),
                    truth_before=truth[i].tolist(), truth_after=truth[i + 4].tolist(),
                    logits={name: value.tolist() for name, value in zip(COMBINATIONS, scores)}))
            for context, logits, y, k in zip(CONTEXTS, (before, after), (truth[:4], truth[4:]), (known[:4], known[4:])):
                counts = {name: [0] * 4 for name in ('tp', 'fp', 'fn', 'tn')}
                for i in range(4):
                    for q in range(4):
                        if not k[i, q]: continue
                        pred, target = bool(logits[i, q] >= 0), bool(y[i, q])
                        key = ('tp' if target else 'fp') if pred else ('fn' if target else 'tn')
                        counts[key][q] += 1; scalar_bits += 1
                assert all(counts[key] == methods[method][context][key] for key in counts)
        prefix = '' if condition == 'IDEAL' else condition + '/'
        # Packet differences are observations, not proof of target-return identity.
        rr, vv = data[condition + '/ranges'], data[condition + '/valid']
        result[condition] = dict(methods=methods, changed_range_slots=int((rr[:4] != rr[4:]).sum()),
            changed_valid_slots=int((vv[:4] != vv[4:]).sum()))
    write(out / 'result.json', dict(status='PASS', attempted_frames=8, newly_rendered_frames=4,
        known_event_bits=int(known.sum()), unknown_event_bits=int((~known).sum()),
        context_eligible_pairs=int(eligible.sum()), conditions=result, training_steps=0,
        scope='One-site paired support-removal diagnosis; hybrid inputs are synthetic branch interventions'))
    write(out / 'paired-logits.json', dict(records=records))
    write(out / 'audit.json', dict(status='PASS', scalar_scored_bits=scalar_bits, all_attempts_retained=True,
        native_event_bits=32, exact_old_model_parity=read(inference / 'parity.json')))
    bind.check()
    write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, backend='CPU scalar scoring; TASK_NOT_GPU_SUITABLE',
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print('SCORED', {c: {m: result[c]['methods'][m] for m in ('rgb', 'MZ5', 'MZ43_ENSEMBLE', 'MZ28')} for c in CONDITIONS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'infer', 'score'), required=True)
    args = parser.parse_args(); root, task = args.root.resolve(), args.task.resolve()
    assert task.is_relative_to((root / 'artifacts.local').resolve())
    globals()[args.phase](root, task)
