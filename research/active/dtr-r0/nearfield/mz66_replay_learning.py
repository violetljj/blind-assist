"""One source-identical geometry fit with an OLD_NEG Adam-step constraint."""
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
from mz64_geometry_learning import profiles, initialization_check, missing_check
from mz66_replay_projection import project_adam_delta
from mz66_replay_source import TASK, prepare, FeatureStore

STEPS = 1536
ARMS = ('PROJECT',)
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55', 'mz61')
INITIAL_SHA = 'b7106449e4246b7bd7933659cbf0dd815691072c03a8b522633b7aee9a456d2e'
KEY = 'MZ66/PROJECT/'


def load_head(root, bind):
    work = root / 'artifacts.local/work'
    r54 = read(bind(work/'mz54-full-rgb-20260911/run-v1/receipt.json'))
    r62 = read(bind(work/'mz62-profile-coverage-20260911/run-v1/receipt.json'))
    assert r54['status'] == r62['status'] == 'PASS'
    geometry_path = bind(work/'mz54-full-rgb-20260911/run-v1/FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt'])
    assert r62['outputs']['CONTROL.pt'] == INITIAL_SHA
    initial_path = bind(work/'mz62-profile-coverage-20260911/run-v1/CONTROL.pt', INITIAL_SHA)
    geometry = torch.load(geometry_path, map_location='cpu', weights_only=True)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    torch.manual_seed(151)
    model = AnchorQuery(geometry, 'GLOBAL_ANCHOR')
    model.load_state_dict(initial, strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), initial[key].cpu(), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 11020
    return model.cuda().eval(), initial_path


def flat(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def assign(parameters, vector):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(vector[offset:offset+count].view_as(parameter))
            offset += count
    assert offset == vector.numel()


def fit(model, bundle, features, mean, std, out):
    schedule = bundle['schedule']
    assert schedule['GEOMETRY'].shape == (STEPS, 4)
    torch.manual_seed(151)
    model.train()
    parameters = tuple(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.001)
    torch.cuda.synchronize()
    tick, history, steps = time.perf_counter(), [], []
    audit_steps = set(np.linspace(0, STEPS-1, 16, dtype=int).tolist())
    vectors = {key: [] for key in ('step', 'before', 'proposed', 'gradient', 'assigned')}
    for step in range(STEPS):
        assert schedule['profile_index'][step] == step % 3
        condition = PROFILES[step % 3]
        dense, rr, vv, cell, cknown, truth, known = [], [], [], [], [], [], []
        for source, ids in (('mz48', schedule['mz48'][step]), ('mz61', schedule['GEOMETRY'][step])):
            data, labels = bundle['cohorts'][source], bundle['native_labels'][source]
            dense.append(features.training(source, ids))
            rr.append(data['ranges'][ids]); vv.append(data['valid'][ids])
            cell.append(labels['cell_truth'][ids]); cknown.append(labels['cell_known'][ids])
            truth.append(bundle['predictions'][source+'/truth'][ids])
            known.append(bundle['predictions'][source+'/known'][ids])
        observed = packet(np.concatenate(rr), np.concatenate(vv), condition)
        prediction = model.inspect(*tensors(np.concatenate(dense), observed['ranges'], observed['valid'], mean, std))
        native, parts = loss_for(prediction, torch.from_numpy(np.concatenate(cell)).cuda(),
            torch.from_numpy(np.concatenate(cknown)).cuda(), torch.from_numpy(np.concatenate(truth)).cuda(),
            torch.from_numpy(np.concatenate(known)).cuda())
        old_ids = schedule['OLD_NEG'][step]
        old = packet(bundle['old_ranges'][old_ids], bundle['old_valid'][old_ids], condition)
        replay_args = tensors(features.training('old', old_ids), old['ranges'], old['valid'], mean, std)
        query = torch.from_numpy(schedule['query'][step]).cuda()
        extra = model.inspect(*replay_args)
        negative = negative_loss(extra, query)
        loss = native + .25 * negative
        assert torch.isfinite(loss)
        reference = torch.autograd.grad(negative, parameters, retain_graph=True, allow_unused=True)
        gradient = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                              for p, g in zip(parameters, reference)])
        before = flat(parameters)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        proposed = flat(parameters)
        assigned, diagnostics = project_adam_delta(before, proposed, gradient)
        ideal = diagnostics.pop('theta_ideal64')
        assert torch.isfinite(ideal).all()
        assign(parameters, assigned)
        torch.testing.assert_close(flat(parameters), assigned, rtol=0, atol=0)
        actual_dot = float(torch.dot(gradient.double(), assigned.double()-before.double()))
        assert abs(actual_dot-diagnostics['post_dot']) <= diagnostics['arithmetic_roundoff_bound']
        bound = diagnostics['residual_bound']
        assert actual_dot <= bound, (step, actual_dot, bound)
        if step in audit_steps:
            vectors['step'].append(step+1)
            for key, value in (('before', before), ('proposed', proposed), ('gradient', gradient), ('assigned', assigned)):
                vectors[key].append(value.detach().cpu().numpy().copy())
        with torch.no_grad():
            post_negative = float(negative_loss(model.inspect(*replay_args), query))
        row = dict(step=step+1, condition=condition, loss=float(loss.detach()),
            native=float(native.detach()), negative=float(negative.detach()),
            post_negative=post_negative, actual_assigned_dot=actual_dot, allowed_rounding_bound=bound,
            **diagnostics, **{k: float(v.detach()) for k,v in parts.items()})
        steps.append(row)
        if step % 100 == 0 or step == STEPS-1:
            history.append(row)
            write(out/'progress.json', dict(stage='fit', arm='PROJECT', **row))
            print('FIT PROJECT', {k:row[k] for k in ('step','condition','loss','negative','post_negative','projected','post_dot')}, flush=True)
        del reference, gradient, before, proposed, assigned, ideal, loss, native, negative, prediction, extra
    torch.cuda.synchronize()
    seconds = time.perf_counter()-tick
    model.eval()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out/'PROJECT.pt')
    np.savez_compressed(out/'projection-vectors.npz', **{k:np.asarray(v) for k,v in vectors.items()})
    write(out/'projection-steps.json', dict(status='PASS', steps=steps,
        projected_steps=sum(r['projected'] for r in steps),
        nonlinear_replay_increases=sum(r['post_negative'] > r['negative'] for r in steps),
        interpretation='Batch-average first-order constraint only; actual nonlinear replay changes are descriptive'))
    return dict(steps=STEPS, seconds=seconds, history=history, target_source='mz61',
        target_presentations=6144, mz48_presentations=6144, old_negative_presentations=12288,
        native_presentations=12288, all_invalid_presentations=0, optimizer_reset=True,
        projected_steps=sum(r['projected'] for r in steps), post_replay_measurements=STEPS)


def evaluate(model, bundle, features, mean, std, out):
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
                    prefix = cohort+'/'+profile+'/'+KEY
                    values = saved_outputs(model, tensors(dense, obs['ranges'], obs['valid'], mean, std))
                    for key, value in values.items():
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
    cut = cutoff_zero_added(collect(KEY+'raw'), collect(KEY+'support'), collect('MZ37'), truth)
    np.save(out/'PROJECT-cutoff.npy', cut)
    for cohort in COHORTS:
        for profile in profiles(cohort):
            prefix = cohort+'/'+profile+'/'
            p[prefix+KEY+'candidate'] = candidate_values(p[prefix+KEY+'raw'], p[prefix+KEY+'support'], p[prefix+'MZ37'], cut)
            old = np.maximum(p[prefix+'OLD_NEG/OPEN/candidate'], p[prefix+'OLD_NEG/GATED/candidate'])
            np.testing.assert_array_equal(old, p[prefix+'OLD_NEG/UNION'])
            p[prefix+KEY+'UNION'] = np.maximum(old, p[prefix+KEY+'candidate'])
    return p, cut, time.perf_counter()-tick


def run(root, task):
    assert task == (root/'artifacts.local/work'/TASK).resolve()
    assert read(task/'registration.json')['status'] == 'ACTIVE'
    out = task/'run-v1'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bind = Bindings(); bundle = features = store = model = None
    try:
        inputs = read(task/'inputs.json')
        for item in inputs['bound_files']:
            bind(item['path'], item['sha256'])
        definition = read(bind(task/'primary-definition.json'))
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['steps_per_arm'] == STEPS and definition['arms'] == list(ARMS)
        assert definition['initial_checkpoint_sha256'] == INITIAL_SHA
        bundle = prepare(root, bind)
        original_arrays = array_identity(bundle['predictions'])
        inherited = tuple(bundle['predictions'])
        old_run = root/'artifacts.local/work/mz64-geometry-learning-20260911/run-v1'
        for name in ('groups.json','schedule.npz'):
            (out/name).write_bytes((old_run/name).read_bytes())
        setup()
        model, initial_path = load_head(root, bind)
        identities = {'PROJECT': parameters_sha(model)}
        base, mean, std = make_encoder(root, bind)
        store = RGBStore(); features = FeatureStore(base, store, bundle, task, bind)
        write(out/'start.json', dict(status='STARTED', arms=ARMS, steps_per_arm=STEPS,
            initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            schedule_sha256=sha(out/'schedule.npz'), primary_definition=definition,
            source_info=bundle['source_info'], inputs=bind.inputs, device=torch.cuda.get_device_name()))
        tick = time.perf_counter(); features.build(); feature_seconds = time.perf_counter()-tick
        write(out/'initialization-parity.json', initialization_check({'PROJECT':model}, bundle, features, mean, std))
        write(out/'initial-missing.json', missing_check({'PROJECT':model}, bundle))
        fits = {'PROJECT': fit(model, bundle, features, mean, std, out)}
        write(out/'learned-missing.json', missing_check({'PROJECT':model}, bundle))
        p, cut, evaluation_seconds = evaluate(model, bundle, features, mean, std, out)
        for name in ('crop_mask','sensor_coverage','rays'):
            p['mz66/'+name] = getattr(model, name).cpu().numpy()
        assert array_identity({key:p[key] for key in inherited}) == original_arrays
        np.savez_compressed(out/'predictions.npz', **p)
        bind.check()
        receipt = dict(status='PASS', inputs=bind.inputs, arms=ARMS, steps_per_arm=STEPS, total_steps=STEPS,
            fits=fits, source_info=bundle['source_info'], primary_definition=definition,
            loss_contract='original_MZ59_balanced_local_frame_loss', positive_pool='original_OPEN_max',
            exact_initial_weights=True, initial_checkpoint_sha256=sha(initial_path), initial_parameter_sha256=identities,
            trainable_parameters=11020, baseline_arrays_preserved=list(inherited), baseline_array_identities=original_arrays,
            original_baseline_cohort_inferences=0, initial_train_parity_unique_frames=32,
            original_calibration_frames=1256, mz55_calibration_rows_used=0, mz61_calibration_rows_used=0,
            all_invalid_training_presentations=0, new_cutoffs=1, threshold_searches=0,
            feature_build_seconds=feature_seconds, evaluation_seconds=evaluation_seconds,
            feature_counts=dict(features.stats), rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            source_unknown_preserved=True, native_depth_reads=0,
            optimizer_change=definition['optimizer_change'], post_replay_measurements=STEPS,
            backend='CUDA frozen RGB encoder and actual-Adam-step constrained head; CPU source I/O',
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
        model = None; gc.collect()
        write(task/'handle-release.json', dict(status='PASS', feature_handles_closed=True,
            compact_handles_closed=True, scratch_retained_until_score=True, process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); run(args.root.resolve(), args.task.resolve())
