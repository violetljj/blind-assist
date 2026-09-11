"""Matched adaptation to MZ61 geometry with original readout/loss/calibration."""
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

TASK = 'mz64-geometry-learning-20260911'
ARMS = ('CONTROL', 'GEOMETRY')
STEPS = 1536
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55', 'mz61')
INITIAL_SHA = 'b7106449e4246b7bd7933659cbf0dd815691072c03a8b522633b7aee9a456d2e'


def profiles(cohort):
    return PROFILES + ('ALL_INVALID',) if cohort == 'mz61' else PROFILES


def load_heads(root, bind):
    work = root / 'artifacts.local/work'
    geometry_dir = work / 'mz54-full-rgb-20260911/run-v1'
    initial_dir = work / 'mz62-profile-coverage-20260911/run-v1'
    r54, r62 = (read(bind(p / 'receipt.json')) for p in (geometry_dir, initial_dir))
    assert r54['status'] == r62['status'] == 'PASS'
    geometry_path = bind(geometry_dir / 'FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt'])
    assert r62['outputs']['CONTROL.pt'] == INITIAL_SHA
    initial_path = bind(initial_dir / 'CONTROL.pt', INITIAL_SHA)
    geometry = torch.load(geometry_path, map_location='cpu', weights_only=True)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    models = {}
    for arm in ARMS:
        torch.manual_seed(151)
        model = AnchorQuery(geometry, 'GLOBAL_ANCHOR')
        model.load_state_dict(initial, strict=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial[key].cpu(), rtol=0, atol=0)
        assert sum(p.numel() for p in model.parameters()) == 11020
        models[arm] = model.cuda().eval()
    identities = {arm: parameters_sha(model) for arm, model in models.items()}
    assert len(set(identities.values())) == 1
    return models, identities, initial_path


def fit(model, arm, bundle, features, mean, std, out):
    schedule = bundle['schedule']
    target = 'mz55' if arm == 'CONTROL' else 'mz61'
    assert arm in ARMS and schedule[arm].shape == (STEPS, 4)
    torch.manual_seed(151)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    torch.cuda.synchronize()
    tick, history = time.perf_counter(), []
    for step in range(STEPS):
        assert schedule['profile_index'][step] == step % 3
        condition = PROFILES[step % 3]
        dense, rr, vv, cell, cknown, truth, known = [], [], [], [], [], [], []
        for source, ids in (('mz48', schedule['mz48'][step]), (target, schedule[arm][step])):
            data, labels = bundle['cohorts'][source], bundle['native_labels'][source]
            dense.append(features.training(source, ids))
            rr.append(data['ranges'][ids]); vv.append(data['valid'][ids])
            cell.append(labels['cell_truth'][ids]); cknown.append(labels['cell_known'][ids])
            truth.append(bundle['predictions'][source + '/truth'][ids])
            known.append(bundle['predictions'][source + '/known'][ids])
        observed = packet(np.concatenate(rr), np.concatenate(vv), condition)
        prediction = model.inspect(*tensors(np.concatenate(dense), observed['ranges'], observed['valid'], mean, std))
        native, parts = loss_for(prediction, torch.from_numpy(np.concatenate(cell)).cuda(),
            torch.from_numpy(np.concatenate(cknown)).cuda(), torch.from_numpy(np.concatenate(truth)).cuda(),
            torch.from_numpy(np.concatenate(known)).cuda())
        old_ids = schedule['OLD_NEG'][step]
        old = packet(bundle['old_ranges'][old_ids], bundle['old_valid'][old_ids], condition)
        extra = model.inspect(*tensors(features.training('old', old_ids), old['ranges'], old['valid'], mean, std))
        negative = negative_loss(extra, torch.from_numpy(schedule['query'][step]).cuda())
        loss = native + .25 * negative
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == STEPS - 1:
            row = dict(step=step+1, condition=condition, loss=float(loss.detach()),
                native=float(native.detach()), negative=float(negative.detach()),
                **{k: float(v.detach()) for k,v in parts.items()})
            history.append(row)
            write(out/'progress.json', dict(stage='fit', arm=arm, **row))
            print('FIT', arm, row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - tick
    model.eval()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out/(arm+'.pt'))
    return dict(steps=STEPS, seconds=seconds, history=history, target_source=target,
        target_presentations=6144, mz48_presentations=6144, old_negative_presentations=12288,
        native_presentations=12288, all_invalid_presentations=0, optimizer_reset=True)


def initialization_check(models, bundle, features, mean, std):
    comparisons, selections = [], {}
    with torch.inference_mode():
        for cohort, key in (('mz48','mz48'), ('mz61','GEOMETRY')):
            ids = np.unique(bundle['schedule'][key])[:16]
            assert len(ids) == 16
            selections[cohort] = ids.tolist()
            dense = features.training(cohort, ids)
            data = bundle['cohorts'][cohort]
            for profile in profiles(cohort):
                observed = packet(data['ranges'][ids], data['valid'][ids], profile)
                args = tensors(dense, observed['ranges'], observed['valid'], mean, std)
                prefix = cohort+'/'+profile+'/MZ62/CONTROL/'
                for arm, model in models.items():
                    actual = saved_outputs(model, args)
                    expected = bundle['predictions'][prefix+'raw'][ids]
                    np.testing.assert_allclose(actual['raw'], expected, atol=2e-5, rtol=1e-6)
                    np.testing.assert_array_equal(actual['support'], bundle['predictions'][prefix+'support'][ids])
                    comparisons.append(dict(cohort=cohort, profile=profile, arm=arm,
                        max_abs=float(np.abs(actual['raw']-expected).max())))
    return dict(status='PASS', unique_train_frames=32, ids=selections, comparisons=comparisons,
        atol=2e-5, rtol=1e-6, full_baseline_cohort_replays=0)


def missing_check(models, bundle):
    rr = torch.from_numpy(bundle['old_ranges'][bundle['schedule']['OLD_NEG'][0]]).cuda()
    vv = torch.zeros_like(rr, dtype=torch.bool)
    with torch.inference_mode():
        for model in models.values():
            context, available = model.encode_anchor(rr, vv)
            assert not available.any() and torch.count_nonzero(context).item() == 0
    return dict(status='PASS', frames_per_arm=len(rr), all_missing_context_exactly_zero=True)


def evaluate(models, bundle, features, mean, std, out):
    p, outputs = bundle['predictions'], {}
    tick = time.perf_counter()
    with torch.inference_mode():
        for cohort in COHORTS:
            data = bundle['cohorts'][cohort]
            for begin in range(0, len(data['ranges']), 16):
                ids = np.arange(begin, min(begin+16, len(data['ranges'])))
                dense = features.evaluation(cohort, ids)
                for profile in profiles(cohort):
                    obs = packet(data['ranges'][ids], data['valid'][ids], profile)
                    args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
                    for arm, model in models.items():
                        prefix = cohort+'/'+profile+'/MZ64/'+arm+'/'
                        for key, value in saved_outputs(model, args).items():
                            outputs.setdefault(prefix+key, []).append(value)
                if begin % 512 == 0 or begin+len(ids) == len(data['ranges']):
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
        key = 'MZ64/'+arm+'/'
        cuts[arm] = cutoff_zero_added(collect(key+'raw'), collect(key+'support'), collect('MZ37'), truth)
        np.save(out/(arm+'-cutoff.npy'), cuts[arm])
        for cohort in COHORTS:
            for profile in profiles(cohort):
                prefix = cohort+'/'+profile+'/'
                p[prefix+key+'candidate'] = candidate_values(p[prefix+key+'raw'], p[prefix+key+'support'], p[prefix+'MZ37'], cuts[arm])
                old = np.maximum(p[prefix+'OLD_NEG/OPEN/candidate'], p[prefix+'OLD_NEG/GATED/candidate'])
                np.testing.assert_array_equal(old, p[prefix+'OLD_NEG/UNION'])
                p[prefix+key+'UNION'] = np.maximum(old, p[prefix+key+'candidate'])
    return p, cuts, time.perf_counter()-tick


def run(root, task):
    from mz64_geometry_source import prepare, FeatureStore
    assert task == (root/'artifacts.local/work'/TASK).resolve()
    out = task/'run-v1'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bind = Bindings(); bundle = features = store = None; models = {}
    try:
        definition = read(bind(task/'primary-definition.json'))
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['steps_per_arm'] == STEPS and definition['arms'] == list(ARMS)
        assert definition['initial_checkpoint_sha256'] == INITIAL_SHA
        for name in ('MZ64_GEOMETRY_LEARNING_20260911.md', 'mz64_geometry_learning.py', 'mz64_geometry_source.py',
                     'mz64_score.py', 'mz62_profile_coverage.py', 'mz63_geometry_transfer.py', 'mz56_global_anchor.py',
                     'mz56_global_anchor_model.py', 'mz54_full_rgb.py', 'mz54_full_rgb_model.py',
                     'mz54_full_rgb_source.py', 'mz51_training_coverage.py', 'mz15_train.py',
                     'mz47_local_enrichment.py', 'mz36_frozen_inference.py'):
            bind(Path(__file__).with_name(name))
        bundle = prepare(root, bind)
        assert bundle['old_maps'] is None
        original_arrays = array_identity(bundle['predictions'])
        inherited = tuple(bundle['predictions'])
        groups = dict(records=bundle['records'], groups={k:np.asarray(v).tolist() for k,v in bundle['groups'].items()})
        for cohort in ('mz55','mz61'):
            groups[cohort+'_records'] = bundle[cohort+'_records']
            groups[cohort+'_groups'] = {k:np.asarray(v).tolist() for k,v in bundle[cohort+'_groups'].items()}
        write(out/'groups.json', groups)
        (out/'schedule.npz').write_bytes(Path(bundle['candidate_schedule_path']).read_bytes())
        setup()
        models, identities, initial_path = load_heads(root, bind)
        base, mean, std = make_encoder(root, bind)
        store = RGBStore(); features = FeatureStore(base, store, bundle, task, bind)
        write(out/'start.json', dict(status='STARTED', arms=ARMS, steps_per_arm=STEPS, total_steps=STEPS*2,
            initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            schedule_sha256=sha(out/'schedule.npz'), primary_definition=definition,
            source_info=bundle['source_info'], inputs=bind.inputs, device=torch.cuda.get_device_name()))
        tick = time.perf_counter(); features.build(); feature_seconds = time.perf_counter()-tick
        write(out/'initialization-parity.json', initialization_check(models, bundle, features, mean, std))
        write(out/'initial-missing.json', missing_check(models, bundle))
        fits = {arm:fit(model, arm, bundle, features, mean, std, out) for arm,model in models.items()}
        write(out/'learned-missing.json', missing_check(models, bundle))
        p, cuts, evaluation_seconds = evaluate(models, bundle, features, mean, std, out)
        for name in ('crop_mask','sensor_coverage','rays'):
            p['mz64/'+name] = getattr(models['CONTROL'], name).cpu().numpy()
        assert array_identity({key:p[key] for key in inherited}) == original_arrays
        np.savez_compressed(out/'predictions.npz', **p)
        bind.check()
        receipt = dict(status='PASS', inputs=bind.inputs, arms=ARMS, steps_per_arm=STEPS, total_steps=STEPS*2,
            fits=fits, source_info=bundle['source_info'], primary_definition=definition,
            loss_contract='original_MZ59_balanced_local_frame_loss', positive_pool='original_OPEN_max',
            exact_initial_weights=True, initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            trainable_parameters=11020, baseline_arrays_preserved=list(inherited), baseline_array_identities=original_arrays,
            original_baseline_cohort_inferences=0, initial_train_parity_unique_frames=32,
            original_calibration_frames=1256, mz55_calibration_rows_used=0, mz61_calibration_rows_used=0,
            all_invalid_training_presentations=0, new_cutoffs=2, threshold_searches=0,
            feature_build_seconds=feature_seconds, evaluation_seconds=evaluation_seconds,
            feature_counts=dict(features.stats), rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            source_unknown_preserved=True, native_depth_reads=0,
            backend='CUDA frozen RGB encoder and two matched source-adaptation heads; CPU source I/O',
            device=torch.cuda.get_device_name(), seconds=time.perf_counter()-started, consumed_development=True,
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        write(out/'receipt.json', receipt)
        print('PASS', {k:receipt[k] for k in ('seconds','feature_build_seconds','evaluation_seconds','feature_counts')}, flush=True)
    except BaseException:
        write(out/'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise
    finally:
        if features is not None: features.close()
        if store is not None: store.close()
        if bundle is not None and bundle.get('old_maps') is not None: bundle['old_maps']._mmap.close()
        models.clear(); gc.collect()
        write(task/'handle-release.json', dict(status='PASS', feature_handles_closed=True,
            compact_handles_closed=True, scratch_retained_until_score=True, process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); run(args.root.resolve(), args.task.resolve())
