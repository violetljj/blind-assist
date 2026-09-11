"""Matched old-only versus diverse-shape learning with four input profiles."""
import argparse
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha
from mz15_train import cutoff_zero_added
from mz45_object_transfer import Bindings, setup
from mz51_training_coverage import negative_loss
from mz54_full_rgb import make_encoder, parameters_sha
from mz54_full_rgb_model import loss_for
from mz54_full_rgb_source import RGBStore
from mz56_global_anchor import tensors, saved_outputs
from mz56_global_anchor_model import AnchorQuery
from mz62_profile_coverage import candidate_values, array_identity
from mz63_geometry_transfer import packet
from mz64_geometry_learning import missing_check

TASK = 'mz70-diverse-learning-20260911'
ARMS = ('CONTROL', 'DIVERSE')
STEPS = 4096
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE', 'ALL_INVALID')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55', 'mz61', 'mz67')
INITIAL_SHA = 'fb318f03a598d00b89fc011eb5b1a368873f9ac801d1dc0710618ba119f4d4aa'


def profiles(cohort):
    return PROFILES if cohort in ('mz61', 'mz67') else PROFILES[:3]


def load_heads(root, bind):
    work = root / 'artifacts.local/work'
    p54, p64 = work/'mz54-full-rgb-20260911/run-v1', work/'mz64-geometry-learning-20260911/run-v1'
    r54 = read(bind(p54/'receipt.json'))
    r64 = read(bind(p64/'receipt.json', '3e917c4e2d6d56f134ab52633e572ae528e75487bc68405d2ce5a6a247d2508e'))
    assert r54['status'] == r64['status'] == 'PASS' and r64['outputs']['GEOMETRY.pt'] == INITIAL_SHA
    geometry = torch.load(bind(p54/'FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt']), map_location='cpu', weights_only=True)
    initial_path = bind(p64/'GEOMETRY.pt', INITIAL_SHA)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    heads = {}
    for arm in ARMS:
        torch.manual_seed(151)
        head = AnchorQuery(geometry, 'GLOBAL_ANCHOR')
        head.load_state_dict(initial, strict=True)
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial[name].cpu(), rtol=0, atol=0)
        assert sum(p.numel() for p in head.parameters()) == 11020
        heads[arm] = head.cuda().eval()
    identities = {arm: parameters_sha(head) for arm, head in heads.items()}
    assert len(set(identities.values())) == 1
    return heads, identities, initial_path


def native_chunks(arm, schedule, step):
    chunks = [('mz48', schedule['mz48'][step], np.arange(4))]
    if arm == 'CONTROL':
        chunks.append(('mz61', schedule['CONTROL'][step], np.arange(4, 8)))
    else:
        assert arm == 'DIVERSE'
        tags, ids = schedule['DIVERSE_source'][step], schedule['DIVERSE'][step]
        assert np.isin(tags, (0, 1)).all()
        for tag, source in enumerate(('mz61', 'mz67')):
            positions = np.flatnonzero(tags == tag)
            if len(positions): chunks.append((source, ids[positions], positions + 4))
    assert sorted(int(i) for _, _, positions in chunks for i in positions) == list(range(8))
    return chunks


def native_batch(arm, bundle, features, step):
    """Keep geometry slot order identical across the paired source substitution."""
    batch = None
    for source, ids, positions in native_chunks(arm, bundle['schedule'], step):
        observed, labels = bundle['cohorts'][source], bundle['native_labels'][source]
        values = (features.training(source, ids), observed['ranges'][ids], observed['valid'][ids],
            labels['cell_truth'][ids], labels['cell_known'][ids],
            bundle['predictions'][source+'/truth'][ids], bundle['predictions'][source+'/known'][ids])
        if batch is None: batch = [np.empty((8, *v.shape[1:]), dtype=v.dtype) for v in values]
        for target, value in zip(batch, values): target[positions] = value
    return batch


def fit(head, arm, bundle, features, mean, std, out):
    schedule = bundle['schedule']
    torch.manual_seed(151)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    torch.cuda.synchronize()
    start, history, audit = time.perf_counter(), [], []
    for step in range(STEPS):
        assert schedule['profile_index'][step] == step % 4
        profile = PROFILES[step % 4]
        dense, rr, vv, cell, local_known, truth, known = native_batch(arm, bundle, features, step)
        obs = packet(rr, vv, profile)
        pred = head.inspect(*tensors(dense, obs['ranges'], obs['valid'], mean, std))
        native, parts = loss_for(pred, torch.from_numpy(cell).cuda(), torch.from_numpy(local_known).cuda(),
            torch.from_numpy(truth).cuda(), torch.from_numpy(known).cuda())
        old_ids = schedule['OLD_NEG'][step]
        old = packet(bundle['old_ranges'][old_ids], bundle['old_valid'][old_ids], profile)
        extra = head.inspect(*tensors(features.training('old', old_ids), old['ranges'], old['valid'], mean, std))
        negative = negative_loss(extra, torch.from_numpy(schedule['query'][step]).cuda())
        if profile == 'ALL_INVALID':
            assert not obs['ranges'].any() and not obs['valid'].any()
            assert not old['ranges'].any() and not old['valid'].any()
            for output in (pred, extra):
                assert not output['anchor_available'].any()
                assert torch.count_nonzero(output['anchor_vector']).item() == 0
        audit.append(dict(step=step+1, profile=profile, native_valid_slots=int(obs['valid'].sum()),
            old_valid_slots=int(old['valid'].sum()),
            geometry_mz67=0 if arm == 'CONTROL' else int(schedule['DIVERSE_source'][step].sum()),
            null_anchor_zero_checked=profile == 'ALL_INVALID'))
        loss = native + .25 * negative
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 256 == 0 or step == STEPS-1:
            row = dict(step=step+1, profile=profile, loss=float(loss.detach()), native=float(native.detach()),
                negative=float(negative.detach()), **{k:float(v.detach()) for k,v in parts.items()})
            history.append(row)
            write(out/'progress.json', dict(stage='fit', arm=arm, **row))
            print('FIT', arm, row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    head.eval()
    torch.save({k:v.detach().cpu() for k,v in head.state_dict().items()}, out/(arm+'.pt'))
    write(out/(arm+'-training-steps.json'), dict(status='PASS', steps=audit,
        profile_counts={p:sum(row['profile'] == p for row in audit) for p in PROFILES}))
    return dict(steps=STEPS, seconds=seconds, history=history, optimizer_reset=True,
        target_presentations=STEPS*4, mz48_presentations=STEPS*4, old_negative_presentations=STEPS*8,
        native_presentations=STEPS*8, all_invalid_presentations=STEPS*4,
        geometry_by_source={'mz61':STEPS*4} if arm == 'CONTROL' else {'mz61':STEPS*2, 'mz67':STEPS*2},
        training_profile_counts={p:STEPS//4 for p in PROFILES})


def initialization_check(heads, bundle, features, mean, std):
    rows, selected = [], {}
    with torch.inference_mode():
        for cohort in ('mz61', 'mz67'):
            ids = np.sort(bundle[cohort+'_groups']['fit'])[:16]
            selected[cohort] = ids.tolist()
            dense, data = features.training(cohort, ids), bundle['cohorts'][cohort]
            for profile in PROFILES:
                obs = packet(data['ranges'][ids], data['valid'][ids], profile)
                args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
                prefix = cohort+'/'+profile+'/MZ64/GEOMETRY/'
                for arm, head in heads.items():
                    actual = saved_outputs(head, args)
                    expected = bundle['predictions'][prefix+'raw'][ids]
                    np.testing.assert_allclose(actual['raw'], expected, atol=2e-5, rtol=1e-6)
                    np.testing.assert_array_equal(actual['support'], bundle['predictions'][prefix+'support'][ids])
                    rows.append(dict(cohort=cohort, profile=profile, arm=arm, max_abs=float(np.abs(actual['raw']-expected).max())))
    return dict(status='PASS', unique_train_frames=32, selected=selected, comparisons=rows,
        atol=2e-5, rtol=1e-6, baseline_cohort_replays=0)


def evaluate(heads, bundle, features, mean, std, out):
    p, outputs = bundle['predictions'], {}
    start = time.perf_counter()
    with torch.inference_mode():
        for cohort in COHORTS:
            data = bundle['cohorts'][cohort]
            for begin in range(0, len(data['ranges']), 16):
                ids = np.arange(begin, min(begin+16, len(data['ranges'])))
                dense = features.evaluation(cohort, ids)
                for profile in profiles(cohort):
                    obs = packet(data['ranges'][ids], data['valid'][ids], profile)
                    args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
                    for arm, head in heads.items():
                        prefix = cohort+'/'+profile+'/MZ70/'+arm+'/'
                        for key, value in saved_outputs(head, args).items():
                            outputs.setdefault(prefix+key, []).append(value)
                if begin % 1024 == 0 or begin+len(ids) == len(data['ranges']):
                    row = dict(stage='evaluation', cohort=cohort, frames=begin+len(ids), total=len(data['ranges']))
                    write(out/'progress.json', row); print('EVAL', row, flush=True)
    for key, chunks in outputs.items():
        assert key not in p
        p[key] = np.concatenate(chunks)
    calibration = bundle['groups']['calibration']
    truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][calibration]])
    known = np.concatenate([p['DEV/known'], p['mz48/known'][calibration]])
    assert truth.shape == known.shape == (1256, 4) and known.all()
    def collect(name):
        return np.concatenate([p['DEV/DROP_CLOSE/'+name], p['mz48/DROP_CLOSE/'+name][calibration]])
    cuts = {}
    for arm in ARMS:
        key = 'MZ70/'+arm+'/'
        cuts[arm] = cutoff_zero_added(collect(key+'raw'), collect(key+'support'), collect('MZ37'), truth)
        np.save(out/(arm+'-cutoff.npy'), cuts[arm])
        for cohort in COHORTS:
            for profile in profiles(cohort):
                prefix = cohort+'/'+profile+'/'
                p[prefix+key+'candidate'] = candidate_values(p[prefix+key+'raw'], p[prefix+key+'support'], p[prefix+'MZ37'], cuts[arm])
                old = np.maximum(p[prefix+'OLD_NEG/OPEN/candidate'], p[prefix+'OLD_NEG/GATED/candidate'])
                np.testing.assert_array_equal(old, p[prefix+'OLD_NEG/UNION'])
                p[prefix+key+'UNION'] = np.maximum(old, p[prefix+key+'candidate'])
    return p, cuts, time.perf_counter()-start


def run(root, task):
    from mz70_diverse_source import prepare, FeatureStore
    assert task == (root/'artifacts.local/work'/TASK).resolve()
    registration = read(task/'registration.json')
    assert registration['status'] == 'ACTIVE' and registration['experiment_id'] == TASK
    assert sha(registration['inputs']['path']) == registration['inputs']['sha256']
    out = task/'run-v1'; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    bind = Bindings(); bundle = features = store = heads = None
    try:
        inputs = read(bind(task/'inputs.json', registration['inputs']['sha256']))
        bind(task/'registration.json')
        for item in inputs['bound_files']: bind(item['path'], item['sha256'])
        definition = read(bind(task/'primary-definition.json'))
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['arms'] == list(ARMS) and definition['steps_per_arm'] == STEPS
        assert definition['initial_checkpoint_sha256'] == INITIAL_SHA and definition['training_profiles'] == list(PROFILES)
        bundle = prepare(root, bind)
        inherited = tuple(bundle['predictions'])
        original_arrays = array_identity(bundle['predictions'])
        np.savez_compressed(out/'schedule.npz', **bundle['schedule'])
        groups = {key:bundle[key] for key in ('records', 'mz55_records', 'mz61_records', 'mz67_records')}
        for key in ('groups', 'mz55_groups', 'mz61_groups', 'mz67_groups'):
            groups[key] = {name:np.asarray(ids).tolist() for name,ids in bundle[key].items()}
        write(out/'groups.json', groups)
        setup()
        heads, identities, initial_path = load_heads(root, bind)
        base, mean, std = make_encoder(root, bind)
        store = RGBStore(); features = FeatureStore(base, store, bundle, task, bind)
        write(out/'start.json', dict(status='STARTED', arms=ARMS, steps_per_arm=STEPS,
            initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            schedule_sha256=sha(out/'schedule.npz'), primary_definition=definition,
            source_info=bundle['source_info'], inputs=bind.inputs, device=torch.cuda.get_device_name()))
        tick = time.perf_counter(); features.build(); feature_seconds = time.perf_counter()-tick
        write(out/'initialization-parity.json', initialization_check(heads, bundle, features, mean, std))
        write(out/'initial-missing.json', missing_check(heads, bundle))
        fits = {arm:fit(heads[arm], arm, bundle, features, mean, std, out) for arm in ARMS}
        write(out/'learned-missing.json', missing_check(heads, bundle))
        p, cuts, evaluation_seconds = evaluate(heads, bundle, features, mean, std, out)
        for name in ('crop_mask', 'sensor_coverage', 'rays'):
            p['mz70/'+name] = getattr(heads['CONTROL'], name).cpu().numpy()
        assert array_identity({key:p[key] for key in inherited}) == original_arrays
        np.savez_compressed(out/'predictions.npz', **p)
        bind.check()
        receipt = dict(status='PASS', inputs=bind.inputs, arms=ARMS, steps_per_arm=STEPS, total_steps=STEPS*len(ARMS),
            fits=fits, source_info=bundle['source_info'], primary_definition=definition,
            loss_contract='original_MZ59_balanced_local_frame_loss', positive_pool='original_OPEN_max',
            exact_initial_weights=True, initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            trainable_parameters_per_arm=11020, baseline_arrays_preserved=list(inherited), baseline_array_identities=original_arrays,
            original_baseline_cohort_inferences=0, initial_train_parity_unique_frames=32,
            original_calibration_frames=1256, mz55_calibration_rows_used=0, mz61_calibration_rows_used=0, mz67_calibration_rows_used=0,
            new_cutoffs=2, threshold_searches=0, training_profile_counts_per_arm={p:STEPS//4 for p in PROFILES},
            all_invalid_training_steps_per_arm=STEPS//4, all_invalid_training_presentations_per_arm=STEPS*4,
            feature_build_seconds=feature_seconds, evaluation_seconds=evaluation_seconds,
            feature_counts=dict(features.stats), rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            source_unknown_preserved=True, native_depth_reads=0, optimizer_change_between_arms='none',
            input_change_between_arms='Half geometry exposure allocated to MZ67 instead of second MZ61 repeat',
            backend='CUDA frozen RGB encoder and Adam head; CPU source I/O', device=torch.cuda.get_device_name(),
            seconds=time.perf_counter()-start, consumed_development=True,
            outputs={path.name:sha(path) for path in out.iterdir() if path.is_file()})
        write(out/'receipt.json', receipt)
        print('PASS', {k:receipt[k] for k in ('seconds', 'feature_build_seconds', 'evaluation_seconds', 'feature_counts')}, flush=True)
    except BaseException:
        write(out/'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise
    finally:
        if features is not None: features.close()
        if store is not None: store.close()
        heads = None; gc.collect()
        write(task/'handle-release.json', dict(status='PASS', feature_handles_closed=True,
            compact_handles_closed=True, scratch_retained_until_score=True, process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); run(args.root.resolve(), args.task.resolve())
