"""Equal-budget MZ59 coverage contrast; never change frozen model/source code."""
import argparse
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz45_object_transfer import Bindings, CONDITIONS, setup
from mz47_local_enrichment import packet
from mz51_training_coverage import negative_loss
from mz54_full_rgb import make_encoder, parameters_sha
from mz54_full_rgb_model import loss_for
from mz54_full_rgb_source import RGBStore
from mz56_global_anchor import tensors, saved_outputs
from mz56_global_anchor_model import AnchorQuery

ARMS = ('CONTROL', 'DIVERSE')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55')


def load_heads(root, bind):
    work = root / 'artifacts.local/work'
    initial_dir = work / 'mz54-full-rgb-20260911/run-v1'
    previous_dir = work / 'mz56-global-anchor-20260911/run-v2'
    r54 = read(bind(initial_dir / 'receipt.json'))
    r56 = read(bind(previous_dir / 'receipt.json'))
    assert r54['status'] == r56['status'] == 'PASS'
    geometry_path = bind(initial_dir / 'FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt'])
    state_path = bind(previous_dir / 'GLOBAL_ANCHOR.pt', r56['outputs']['GLOBAL_ANCHOR.pt'])
    geometry = torch.load(geometry_path, map_location='cpu', weights_only=True)
    initial = torch.load(state_path, map_location='cpu', weights_only=True)
    models = {}
    for arm in ARMS:
        torch.manual_seed(151)
        model = AnchorQuery(geometry, 'GLOBAL_ANCHOR')
        model.load_state_dict(initial, strict=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial[key].cpu(), atol=0, rtol=0)
        assert sum(x.numel() for x in model.parameters()) == 11020
        models[arm] = model.cuda().eval()
    identities = {arm: parameters_sha(model) for arm, model in models.items()}
    assert len(set(identities.values())) == 1
    return models, identities, state_path


def fit(model, arm, bundle, features, mean, std, out):
    schedule = bundle['schedule']
    torch.manual_seed(151)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    torch.cuda.synchronize()
    tick, history = time.perf_counter(), []
    for step, old_native_ids in enumerate(schedule['shared']):
        condition = CONDITIONS[step % 3]
        if arm == 'CONTROL':
            slices = [('mz48', old_native_ids)]
        else:
            slices = [('mz48', old_native_ids[:4]), ('mz55', schedule['mz55_diverse'][step])]
        dense, rr, vv, cell, cknown, truth, known = [], [], [], [], [], [], []
        for source, ids in slices:
            data = bundle['cohorts'][source]
            labels = bundle['native_labels'][source]
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
        obs_old = packet(bundle['old_ranges'][old_ids], bundle['old_valid'][old_ids], condition)
        extra = model.inspect(*tensors(features.training('old', old_ids), obs_old['ranges'], obs_old['valid'], mean, std))
        negative = negative_loss(extra, torch.from_numpy(schedule['query'][step]).cuda())
        loss = native + .25 * negative
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 599:
            row = dict(step=step+1, condition=condition, loss=float(loss.detach()),
                native=float(native.detach()), negative=float(negative.detach()),
                **{k: float(v.detach()) for k, v in parts.items()})
            history.append(row)
            write(out / 'progress.json', dict(stage='fit', arm=arm, **row))
            print('FIT', arm, row, flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - tick
    model.eval()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / (arm + '.pt'))
    return dict(steps=600, seconds=seconds, history=history, native_presentations=4800,
                mz55_presentations=2400 if arm == 'DIVERSE' else 0, old_negative_presentations=4800)


def evaluate(models, bundle, features, mean, std, out):
    p = bundle['predictions']
    inherited = tuple(p)
    outputs = {}
    tick = time.perf_counter()
    with torch.inference_mode():
        for cohort in COHORTS:
            data = bundle['cohorts'][cohort]
            for begin in range(0, len(data['ranges']), 16):
                ids = np.arange(begin, min(begin + 16, len(data['ranges'])))
                dense = features.evaluation(cohort, ids)
                for profile in CONDITIONS:
                    obs = packet(data['ranges'][ids], data['valid'][ids], profile)
                    args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
                    for arm, model in models.items():
                        prefix = cohort + '/' + profile + '/MZ59/' + arm + '/'
                        for key, value in saved_outputs(model, args).items():
                            outputs.setdefault(prefix + key, []).append(value)
                if begin % 512 == 0 or begin + len(ids) == len(data['ranges']):
                    row = dict(stage='evaluation', cohort=cohort, frames=begin+len(ids), total=len(data['ranges']))
                    write(out / 'progress.json', row); print('EVAL', row, flush=True)
    for key, value in outputs.items():
        assert key not in p
        p[key] = np.concatenate(value)
    cuts = {}
    calibration = bundle['groups']['calibration']
    truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][calibration]])
    assert len(truth) == 1256
    for arm in ARMS:
        key = 'MZ59/' + arm + '/'
        def collect(name):
            return np.concatenate([p['DEV/DROP_CLOSE/' + name], p['mz48/DROP_CLOSE/' + name][calibration]])
        cuts[arm] = cutoff_zero_added(collect(key + 'raw'), collect(key + 'support'), collect('MZ37'), truth)
        np.save(out / (arm + '-cutoff.npy'), cuts[arm])
        for cohort in COHORTS:
            for profile in CONDITIONS:
                prefix = cohort + '/' + profile + '/'
                baseline = p[prefix + 'MZ37']
                margin = p[prefix + key + 'raw'].astype(float) - cuts[arm]
                accepted = (baseline < 0) & p[prefix + key + 'support'] & (margin >= 0)
                p[prefix + key + 'candidate'] = np.where(accepted, margin, baseline)
    return p, cuts, inherited, time.perf_counter() - tick


def missing_check(models, bundle):
    rr = torch.from_numpy(bundle['old_ranges'][bundle['schedule']['OLD_NEG'][0]]).cuda()
    vv = torch.zeros_like(rr, dtype=torch.bool)
    with torch.inference_mode():
        for model in models.values():
            context, available = model.encode_anchor(rr, vv)
            assert not available.any() and torch.count_nonzero(context).item() == 0
    return dict(status='PASS', frames_per_arm=len(rr), all_missing_context_exactly_zero=True)


def initialization_check(models, bundle, features, mean, std):
    ids = np.unique(bundle['schedule']['shared'])[:16]
    dense = features.training('mz48', ids)
    data, p = bundle['cohorts']['mz48'], bundle['predictions']
    comparisons = []
    with torch.inference_mode():
        for profile in CONDITIONS:
            obs = packet(data['ranges'][ids], data['valid'][ids], profile)
            args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
            for arm, model in models.items():
                actual = saved_outputs(model, args)
                key = 'mz48/' + profile + '/MZ56/GLOBAL_ANCHOR/'
                np.testing.assert_allclose(actual['raw'], p[key + 'raw'][ids], atol=2e-5, rtol=1e-6)
                np.testing.assert_array_equal(actual['support'], p[key + 'support'][ids])
                comparisons.append(dict(arm=arm, profile=profile,
                    max_abs=float(np.abs(actual['raw'] - p[key + 'raw'][ids]).max())))
    return dict(status='PASS', unique_train_frames=16, ids=ids.tolist(),
        comparisons=comparisons, atol=2e-5, rtol=1e-6, baseline_cohort_replay_frames=0)


def run(root, task):
    from mz59_source import prepare, FeatureStore
    out = task / 'run-v1'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bundle = features = store = None
    models = {}
    try:
        bind = Bindings()
        definition = read(bind(task / 'primary-definition.json'))
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        for name in ('MZ59_TRAINING_DIVERSITY_20260911.md', 'mz59_training_diversity.py', 'mz59_source.py',
                     'mz56_global_anchor.py', 'mz56_global_anchor_model.py', 'mz54_full_rgb.py',
                     'mz54_full_rgb_model.py', 'mz54_full_rgb_source.py', 'mz51_training_coverage.py',
                     'mz15_train.py', 'mz47_local_enrichment.py', 'mz36_frozen_inference.py'):
            bind(Path(__file__).with_name(name))
        for eid in ('mz56-global-anchor-20260911', 'mz57-complementary-scale-union-20260911',
                    'mz58-diverse-transfer-20260911'):
            folder = root / 'artifacts.local/work' / eid / 'score-v1'
            previous = read(bind(folder / 'receipt.json'))
            assert previous['status'] == 'PASS'
            bind(folder / 'result.json', previous['outputs']['result.json'])
        bundle = prepare(root, bind)
        if bundle.get('old_maps') is not None:
            bundle['old_maps']._mmap.close()
            bundle['old_maps'] = None
        bundle['native_labels'] = {
            'mz48': dict(cell_truth=bundle['cell_truth'], cell_known=bundle['cell_known']),
            'mz55': dict(cell_truth=bundle['mz55_cell_truth'], cell_known=bundle['mz55_cell_known'])}
        for name in ('truth', 'known'):
            key = 'mz55/' + name
            if key in bundle['predictions']:
                np.testing.assert_array_equal(bundle['predictions'][key], bundle['mz55_' + name])
            else:
                bundle['predictions'][key] = bundle['mz55_' + name]
        groups = dict(records=bundle['records'], groups={k: np.asarray(v).tolist() for k,v in bundle['groups'].items()},
            mz55_records=bundle['mz55_records'], mz55_groups={k: np.asarray(v).tolist() for k,v in bundle['mz55_groups'].items()})
        write(out / 'groups.json', groups)
        np.savez_compressed(out / 'schedule.npz', **bundle['schedule'])
        setup()
        models, identities, state_path = load_heads(root, bind)
        base, mean, std = make_encoder(root, bind)
        store = RGBStore()
        features = FeatureStore(base, store, bundle, task, bind)
        write(out / 'start.json', dict(status='STARTED', arms=ARMS, steps_per_arm=600, total_steps=1200,
            initial_checkpoint_sha256=sha(state_path), initial_parameter_sha256=identities,
            schedule_sha256=sha(out / 'schedule.npz'), primary_definition=definition,
            source_info=bundle['source_info'], inputs=bind.inputs, device=torch.cuda.get_device_name()))
        tick = time.perf_counter()
        features.build()
        feature_build_seconds = time.perf_counter() - tick
        write(out / 'initialization-parity.json', initialization_check(models, bundle, features, mean, std))
        write(out / 'initial-missing.json', missing_check(models, bundle))
        fits = {arm: fit(model, arm, bundle, features, mean, std, out) for arm, model in models.items()}
        write(out / 'learned-missing.json', missing_check(models, bundle))
        p, cuts, inherited, evaluation_seconds = evaluate(models, bundle, features, mean, std, out)
        for cohort in COHORTS:
            for profile in CONDITIONS:
                prefix = cohort + '/' + profile + '/'
                old_key = prefix + 'OLD_NEG/UNION'
                old_union = np.maximum(p[prefix+'OLD_NEG/OPEN/candidate'], p[prefix+'OLD_NEG/GATED/candidate'])
                if old_key in p:
                    np.testing.assert_array_equal(p[old_key], old_union)
                else:
                    p[old_key] = old_union
                for arm in ARMS:
                    p[prefix+'MZ59/'+arm+'/UNION'] = np.maximum(old_union, p[prefix+'MZ59/'+arm+'/candidate'])
        for name in ('crop_mask', 'sensor_coverage', 'rays'):
            p['mz59/' + name] = getattr(models['CONTROL'], name).cpu().numpy()
        np.savez_compressed(out / 'predictions.npz', **p)
        bind.check()
        stats = dict(features.stats)
        receipt = dict(status='PASS', inputs=bind.inputs, arms=ARMS, steps_per_arm=600, total_steps=1200,
            fits=fits, source_info=bundle['source_info'], primary_definition=definition,
            exact_initial_weights=True, initial_parameter_sha256=identities, trainable_parameters=11020,
            baseline_arrays_preserved=list(inherited), original_baseline_cohort_inferences=0,
            initial_train_parity_unique_frames=16, original_calibration_frames=1256,
            mz55_calibration_rows_used=0, new_cutoffs=2, threshold_searches=0,
            feature_build_seconds=feature_build_seconds, evaluation_seconds=evaluation_seconds,
            feature_counts=stats, rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            source_unknown_preserved=True, native_depth_reads=0,
            backend='CUDA frozen RGB encoder and matched heads; CPU source I/O', device=torch.cuda.get_device_name(),
            seconds=time.perf_counter()-started, consumed_development=True,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()})
        write(out / 'receipt.json', receipt)
        print('PASS', {k:receipt[k] for k in ('seconds','feature_build_seconds','evaluation_seconds','feature_counts')}, flush=True)
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', error=traceback.format_exc()))
        raise
    finally:
        if features is not None:
            features.close()
        if store is not None:
            store.close()
        if bundle is not None and bundle.get('old_maps') is not None:
            bundle['old_maps']._mmap.close()
        models.clear()
        gc.collect()
        write(task / 'handle-release.json', dict(status='PASS', feature_handles_closed=True,
            compact_handles_closed=True, scratch_retained_until_score=True, process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.task.resolve())
