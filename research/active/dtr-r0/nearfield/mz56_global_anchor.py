"""Registered MZ56 continuations using read-only, handed-off MZ54 features.

No training feature extraction or baseline cohort replay. The 16 TRAIN-row
initialization check is engineering parity, not an evaluated comparator.
"""
import argparse
import gc
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import Bindings, CONDITIONS, setup
from mz47_local_enrichment import packet
from mz51_training_coverage import negative_loss
from mz54_full_rgb import make_encoder, parameters_sha
from mz54_full_rgb_model import RasterQuery, loss_for
from mz54_full_rgb_source import prepare, RGBStore
from mz56_global_anchor_model import AnchorQuery

ARMS = ('LOCAL_ONLY', 'GLOBAL_ANCHOR')
MODES = ARMS + ('GLOBAL_SUPPRESSED',)
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
PRIMARY_DEFINITION = ('Pre-fit operational clarification: count nonfit BODY_NEAR true events added over '
    'MZ37 by this arm passing its existing cutoff, with a native winning cell outside the measured '
    '45-degree field. Compare GLOBAL_ANCHOR > LOCAL_ONLY, with no higher nonfit or legacy noncal FP. '
    'Inherited MZ37 positives do not count as new-path evidence; report all final TP separately.')


def file_identity(path):
    def command(*args):
        run = subprocess.run(['fsutil', *args, str(path)], check=True, capture_output=True, text=True)
        return run.stdout.strip()
    identity = command('file', 'queryfileid')
    links = command('hardlink', 'list').splitlines()
    return dict(file_id=identity, links=[line.strip() for line in links if line.strip()])


class ReusedFeatures:
    """Only mmap mode r; the handed-off array is never assigned or flushed."""
    def __init__(self, bundle, base, store, task, prior, bind):
        self.bundle, self.base, self.store = bundle, base, store
        handoff = read(bind(task / 'cache-handoff.json'))
        assert handoff['status'] == 'PASS' and handoff['owner'] == task.name
        assert handoff['source_experiment'] == 'mz54-full-rgb-20260911'
        assert handoff['new_physical_dense_copy'] is False
        self.path = Path(handoff['path']).resolve(strict=True)
        assert self.path == (task / 'scratch-v1/training-full.npy').resolve(strict=True)
        assert not self.path.is_symlink() and not (self.path.lstat().st_file_attributes & 0x400)
        assert self.path.stat().st_size == handoff['bytes'] == 4408012928
        self.identity = file_identity(self.path)
        assert re.search(r'0x[0-9a-f]+', self.identity['file_id']).group() == re.search(r'0x[0-9a-f]+', handoff['file_id']).group()
        assert len(self.identity['links']) == 1, 'MZ56 must own the sole remaining cache link'
        bind(self.path, handoff['sha256'])
        assert sha(prior / 'receipt.json') == handoff['source_receipt_sha256']
        index = load_npz(bind(handoff['source_index_path'], handoff['source_index_sha256']))
        self.new_ids, self.old_ids = index['new_ids'], index['old_ids']
        self.new_lookup, self.old_lookup = index['new_lookup'], index['old_lookup']
        np.testing.assert_array_equal(self.new_ids, np.unique(bundle['schedule']['shared']))
        np.testing.assert_array_equal(self.old_ids, np.unique(bundle['schedule']['OLD_NEG']))
        assert (len(self.new_ids), len(self.old_ids)) == (1250, 3533)
        new_lookup = np.full(2560, -1, np.int32)
        old_lookup = np.full(len(bundle['old_truth']), -1, np.int32)
        new_lookup[self.new_ids] = np.arange(1250)
        old_lookup[self.old_ids] = np.arange(3533) + 1250
        np.testing.assert_array_equal(self.new_lookup, new_lookup)
        np.testing.assert_array_equal(self.old_lookup, old_lookup)
        self.full = np.load(self.path, mmap_mode='r', allow_pickle=False)
        assert self.full.shape == (4783, 64, 45, 80) and self.full.dtype == np.float32
        assert not self.full.flags.writeable and self.full.mode == 'r'
        self.index = index
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0, full_cache_eval_hits=0)

    def training(self, ids, old=False):
        ids = np.asarray(ids)
        lookup = self.old_lookup if old else self.new_lookup
        assert (lookup[ids] >= 0).all()
        return np.array(self.full[lookup[ids]])

    def extract(self, images):
        assert 0 < len(images) <= 16
        rgb = torch.from_numpy(np.stack([np.asarray(im, dtype=np.uint8) for im in images]))
        rgb = rgb.permute(0, 3, 1, 2).cuda().float() / 255
        with torch.inference_mode():
            dense = fixed_batch_dense(self.base, rgb).cpu().numpy()
        assert dense.shape == (len(images), 64, 45, 80) and np.isfinite(dense).all()
        self.stats['full_extracted_frames'] += len(images)
        return dense

    def evaluation(self, cohort, ids):
        dense = np.empty((len(ids), 64, 45, 80), np.float32)
        missing, images = [], []
        rows = self.bundle['cohorts'][cohort]
        try:
            for j, i in enumerate(ids):
                cache = -1
                if cohort in COHORTS[:3]:
                    cache = self.old_lookup[int(self.bundle['predictions'][cohort + '/frame_ids'][i])]
                elif cohort == 'mz48':
                    cache = self.new_lookup[i]
                if cache >= 0:
                    dense[j] = self.full[cache]
                    self.stats['full_cache_eval_hits'] += 1
                else:
                    missing.append(j)
                    images.append(self.store.load(rows['rgb_refs'][int(i)]))
            if missing:
                dense[missing] = self.extract(images)
            return dense
        finally:
            for image in images:
                image.close()

    def close(self):
        if getattr(self, 'full', None) is not None:
            self.full._mmap.close()
            self.full = None


def tensors(dense, ranges, valid, mean, std):
    return ((torch.from_numpy(dense).cuda() - mean) / std,
            torch.from_numpy(ranges).cuda(), torch.from_numpy(valid).cuda())


def initialization_check(initial, models, features, bundle, mean, std):
    # Reconstruct the completed frozen MZ54 head, not an earlier checkpoint.
    bridge = dict(initial)
    bridge['head.0.weight'] = initial['head.0.weight'][:, :40]
    reference = RasterQuery(bridge, 'FULL_RASTER')
    reference.load_state_dict(initial, strict=True)
    reference.cuda().eval()
    ids = features.old_ids[:16]
    dense = features.training(ids, old=True)
    evidence, matched_arms = [], []
    with torch.inference_mode():
        for condition in CONDITIONS:
            obs = packet(bundle['old_ranges'][ids], bundle['old_valid'][ids], condition)
            args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
            expected = reference.inspect(*args)
            actual = {arm: model.inspect(*args) for arm, model in models.items()}
            for arm, value in actual.items():
                for key in ('field',):
                    left, right = value[key].cpu().numpy(), expected[key].cpu().numpy()
                    np.testing.assert_allclose(left, right, atol=2e-5, rtol=1e-6)
                raw, raw_ref = value['OPEN']['raw'].cpu().numpy(), expected['OPEN']['raw'].cpu().numpy()
                np.testing.assert_allclose(raw, raw_ref, atol=2e-5, rtol=1e-6)
                np.testing.assert_array_equal(value['OPEN']['support'].cpu(), expected['OPEN']['support'].cpu())
                evidence.append(dict(arm=arm, profile=condition, field_values=int(left.size),
                    field_max_abs=float(np.abs(left-right).max()), raw_max_abs=float(np.abs(raw-raw_ref).max())))
            left = actual[ARMS[0]]['field'].cpu().numpy()
            right = actual[ARMS[1]]['field'].cpu().numpy()
            np.testing.assert_allclose(left, right, atol=2e-5, rtol=1e-6)
            matched_arms.append(dict(profile=condition, max_abs=float(np.abs(left-right).max()),
                                     exact=bool(np.array_equal(left, right))))
    del reference
    return dict(status='PASS', source='Completed MZ54 FULL checkpoint on handed-off TRAIN features',
        train_ids=ids.tolist(), unique_frames=16, profiles=3, atol=2e-5, rtol=1e-6,
        two_initial_arms_within_tolerance=True,
        two_initial_arms_exactly_equal=all(row['exact'] for row in matched_arms),
        matched_initial_arm_comparisons=matched_arms, original_checkpoint_comparisons=evidence,
        new_RGB_extractions=0, new_training_feature_extractions=0)


def missing_check(models, bundle, ids):
    ranges = torch.from_numpy(bundle['old_ranges'][ids]).cuda()
    valid = torch.zeros_like(ranges, dtype=torch.bool)
    answer = {}
    with torch.inference_mode():
        for arm, model in models.items():
            anchor, available = model.encode_anchor(ranges, valid)
            assert not available.any() and torch.count_nonzero(anchor).item() == 0
            answer[arm] = dict(frames=len(ids), zero_values=anchor.numel(), all_missing_anchor_exactly_zero=True)
    return answer


def saved_outputs(model, args, suppress=False):
    result = model.inspect(*args, suppress_global=suppress)
    available = result['anchor_available']
    used = result['anchor_vector']
    assert torch.count_nonzero(used[~available]).item() == 0
    if model.arm == 'LOCAL_ONLY' or suppress:
        assert torch.count_nonzero(used).item() == 0
    return dict(raw=result['OPEN']['raw'].cpu().numpy(), support=result['OPEN']['support'].cpu().numpy(),
        winner=result['field'].flatten(1, 2).argmax(1).cpu().numpy().astype(np.int16),
        anchor_available=available.cpu().numpy(), anchor_vector=used.cpu().numpy())


def run(root, task, run_name='run-v1'):
    started = time.perf_counter()
    assert run_name in ('run-v1', 'run-v2')
    out = task / run_name
    out.mkdir(parents=True, exist_ok=False)
    bundle = features = store = None
    try:
        bind = Bindings()
        setup()
        if run_name == 'run-v2':
            failed = read(bind(task / 'preflight-failure-v1/receipt.json'))
            assert failed['completed_fit_steps'] == failed['training_feature_extractions'] == failed['evaluation_feature_extractions'] == 0
            for name, digest in failed['run_v1_untouched'].items():
                bind(task / 'run-v1' / name, digest)
            for name, digest in failed['files'].items():
                bind(task / 'preflight-failure-v1' / name, digest)
        for name in ('MZ56_GLOBAL_ANCHOR_20260911.md', 'mz56_global_anchor.py', 'mz56_global_anchor_model.py',
                     'mz54_full_rgb.py', 'mz54_full_rgb_model.py', 'mz54_full_rgb_source.py',
                     'mz51_training_coverage.py', 'mz36_frozen_inference.py', 'mz15_train.py', 'mz40_packets.py'):
            bind(Path(__file__).with_name(name))
        work = root / 'artifacts.local/work'
        prior = work / 'mz54-full-rgb-20260911/run-v1'
        r54 = read(bind(prior / 'receipt.json'))
        assert r54['status'] == 'PASS'
        bundle = prepare(root, bind)
        # The source adapter preserves MZ51; this continuation preserves MZ54 too.
        p = load_npz(bind(prior / 'predictions.npz', r54['outputs']['predictions.npz']))
        for key, value in bundle['predictions'].items():
            np.testing.assert_array_equal(p[key], value)
        inherited_keys = tuple(p)
        bundle['predictions'] = p
        groups, schedule = bundle['groups'], bundle['schedule']
        for key, value in load_npz(bind(prior / 'schedule.npz', r54['outputs']['schedule.npz'])).items():
            np.testing.assert_array_equal(schedule[key], value)
        grouping = read(bind(prior / 'groups.json', r54['outputs']['groups.json']))
        assert grouping == dict(records=bundle['records'], groups={k: np.asarray(v).tolist() for k, v in groups.items()})
        # The old crop mmap is unused; release it immediately after metadata preparation.
        bundle['old_maps']._mmap.close()
        bundle['old_maps'] = None
        for eid in ('mz51-training-coverage-20260911', 'mz53-dual-readout-union-20260911', 'mz54-full-rgb-20260911'):
            rec = read(bind(work / eid / 'score-v1/receipt.json'))
            assert rec['status'] == 'PASS'
            bind(work / eid / 'score-v1/result.json', rec['outputs']['result.json'])
        np.savez_compressed(out / 'schedule.npz', **schedule)
        write(out / 'groups.json', grouping)
        initial_path = bind(prior / 'FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt'])
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        models = {}
        for arm in ARMS:
            torch.manual_seed(151)
            models[arm] = AnchorQuery(initial, arm).cuda().eval()
        identities = {arm: parameters_sha(model) for arm, model in models.items()}
        assert len(set(identities.values())) == 1
        assert all(sum(v.numel() for v in model.parameters()) == 11020 for model in models.values())
        for model in models.values():
            assert torch.count_nonzero(model.head[0].weight[:, 41:]).item() == 0
            np.testing.assert_array_equal(model.head[0].weight[:, :41].detach().cpu(), initial['head.0.weight'])
        torch.save({k: v.detach().cpu() for k, v in models[ARMS[0]].named_parameters()}, out / 'initial-weights.pt')
        base, mean, std = make_encoder(root, bind)
        store = RGBStore()
        features = ReusedFeatures(bundle, base, store, task, prior, bind)
        np.savez_compressed(out / 'feature-cache-index.npz', **features.index)
        write(out / 'start.json', dict(status='STARTED', run_name=run_name, steps_per_arm=600, total_steps=1200, arms=ARMS,
            initial_parameter_sha256=identities, initial_checkpoint_sha256=sha(initial_path),
            primary_operational_definition=PRIMARY_DEFINITION, cache_identity=features.identity,
            training_cache_mode='r', training_feature_extractions=0, backend='CUDA',
            device=torch.cuda.get_device_name(), inputs=bind.inputs))
        parity = initialization_check(initial, models, features, bundle, mean, std)
        parity['initial_all_missing'] = missing_check(models, bundle, features.old_ids[:16])
        write(out / 'initialization-parity.json', parity)
        fits = {}
        for arm, model in models.items():
            torch.manual_seed(151)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            torch.cuda.synchronize()
            tick, history = time.perf_counter(), []
            for step, ids in enumerate(schedule['shared']):
                condition = CONDITIONS[step % 3]
                def forward(ii, old=False):
                    rr, vv = ((bundle['old_ranges'], bundle['old_valid']) if old else
                              (bundle['cohorts']['mz48']['ranges'], bundle['cohorts']['mz48']['valid']))
                    obs = packet(rr[ii], vv[ii], condition)
                    return model.inspect(*tensors(features.training(ii, old), obs['ranges'], obs['valid'], mean, std))
                result = forward(ids)
                native, parts = loss_for(result, torch.from_numpy(bundle['cell_truth'][ids]).cuda(),
                    torch.from_numpy(bundle['cell_known'][ids]).cuda(), torch.from_numpy(p['mz48/truth'][ids]).cuda(),
                    torch.from_numpy(p['mz48/known'][ids]).cuda())
                extra = forward(schedule['OLD_NEG'][step], old=True)
                negative = negative_loss(extra, torch.from_numpy(schedule['query'][step]).cuda())
                loss = native + .25 * negative
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 100 == 0 or step == 599:
                    row = dict(step=step+1, condition=condition, loss=float(loss.detach()), native=float(native.detach()),
                        negative=float(negative.detach()), **{k: float(v.detach()) for k, v in parts.items()})
                    history.append(row)
                    write(out / 'progress.json', dict(stage='fit', arm=arm, **row))
                    print(arm, row, flush=True)
            torch.cuda.synchronize()
            model.eval()
            fits[arm] = dict(steps=600, seconds=time.perf_counter()-tick, history=history)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / (arm + '.pt'))
            del optimizer, result, extra, loss, native, negative
        learned_missing = missing_check(models, bundle, features.old_ids[:16])
        write(out / 'learned-all-missing.json', dict(status='PASS', arms=learned_missing))
        p['mz56/zone_angles'] = models['GLOBAL_ANCHOR'].zone_angles.cpu().numpy()
        prediction_parts = {}
        tick = time.perf_counter()
        with torch.inference_mode():
            for cohort in COHORTS:
                data = bundle['cohorts'][cohort]
                n = len(data['ranges'])
                for begin in range(0, n, 16):
                    ids = np.arange(begin, min(begin+16, n))
                    dense = features.evaluation(cohort, ids)
                    for condition in CONDITIONS:
                        obs = packet(data['ranges'][ids], data['valid'][ids], condition)
                        args = tensors(dense, obs['ranges'], obs['valid'], mean, std)
                        for mode in MODES:
                            arm = 'GLOBAL_ANCHOR' if mode == 'GLOBAL_SUPPRESSED' else mode
                            vals = saved_outputs(models[arm], args, suppress=mode == 'GLOBAL_SUPPRESSED')
                            available = (obs['valid'] & np.isfinite(obs['ranges']) & (obs['ranges'] > 0) & (obs['ranges'] <= 4)).any((1, 2))
                            np.testing.assert_array_equal(vals['anchor_available'], available)
                            prefix = cohort + '/' + condition + '/MZ56/' + mode + '/'
                            for key, value in vals.items():
                                prediction_parts.setdefault(prefix + key, []).append(value)
                    if begin % 512 == 0 or begin + len(ids) == n:
                        row = dict(stage='streamed-evaluation', cohort=cohort, frames=begin+len(ids), total=n)
                        write(out / 'progress.json', row)
                        print('EVAL', row, flush=True)
        inference_seconds = time.perf_counter() - tick
        for key, values in prediction_parts.items():
            assert key not in p
            p[key] = np.concatenate(values)
        del prediction_parts
        flat = bundle['cell_truth'].reshape(2560, 3600, 4)
        for condition in CONDITIONS:
            for mode in MODES:
                key = 'mz48/' + condition + '/MZ56/' + mode + '/'
                p[key + 'winning_native'] = np.take_along_axis(flat, p[key + 'winner'][:, None], 1)[:, 0] & p[key + 'support']
        cutoffs = {}
        truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][groups['calibration']]])
        for arm in ARMS:
            def collect(key):
                return np.concatenate([p['DEV/DROP_CLOSE/' + key], p['mz48/DROP_CLOSE/' + key][groups['calibration']]])
            key = 'MZ56/' + arm + '/'
            cutoffs[arm] = cutoff_zero_added(collect(key + 'raw'), collect(key + 'support'), collect('MZ37'), truth)
            np.save(out / (arm + '-cutoff.npy'), cutoffs[arm])
            fits[arm]['cutoff'] = cutoffs[arm].tolist()
        for cohort in COHORTS:
            for condition in CONDITIONS:
                prefix = cohort + '/' + condition + '/'
                for mode in MODES:
                    key = prefix + 'MZ56/' + mode + '/'
                    cut = cutoffs['GLOBAL_ANCHOR' if mode == 'GLOBAL_SUPPRESSED' else mode]
                    margin = p[key + 'raw'].astype(float) - cut
                    accepted = (p[prefix + 'MZ37'] < 0) & p[key + 'support'] & (margin >= 0)
                    p[key + 'candidate'] = np.where(accepted, margin, p[prefix + 'MZ37'])
        assert features.stats == dict(training_extracted_frames=0, full_extracted_frames=5734, full_cache_eval_hits=1250)
        assert store.loads == 5734
        original = load_npz(prior / 'predictions.npz')
        assert tuple(original) == inherited_keys
        for key, value in original.items():
            np.testing.assert_array_equal(p[key], value)
        del original
        np.savez_compressed(out / 'predictions.npz', **p)
        bind.check()
        identity_after = file_identity(features.path)
        assert identity_after == features.identity
        receipt = dict(status='PASS', run_name=run_name, inputs=bind.inputs, steps_per_arm=600, total_steps=1200, fits=fits,
            exact_initial_weights=True, initial_parameter_sha256=identities, trainable_parameters=11020,
            primary_operational_definition=PRIMARY_DEFINITION, preserved_MZ54_arrays=len(inherited_keys),
            initialization_reference_frames=16, initial_logits_parity=parity,
            learned_all_missing=learned_missing, seconds=time.perf_counter()-started,
            training_feature_seconds=0, streamed_inference_seconds=inference_seconds,
            backend='CUDA; read-only inherited feature mmap plus streamed uncached RGB', device=torch.cuda.get_device_name(),
            feature_counts=features.stats, rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            inherited_full_cache_bytes=int(features.full.nbytes), new_physical_dense_copy=False,
            training_cache_mode='r', cache_identity_before=features.identity, cache_identity_after=identity_after,
            cache_sha256_unchanged=True, original_baseline_cohort_inferences=0, new_training_feature_extractions=0,
            new_cutoffs=2, threshold_searches=0, native_depth_reads=0, permanent_dense_cache=False,
            full_candidate_cells=3600, original_sensor_field_degrees=45, source_unknown_preserved=True,
            GLOBAL_SUPPRESSED_cutoff_source='GLOBAL_ANCHOR-cutoff.npy', inherited_single_fit_nondeterminism=True,
            outputs={f.name: sha(f) for f in out.iterdir() if f.is_file()})
        write(out / 'receipt.json', receipt)
        print('PASS', dict(seconds=receipt['seconds'], fits=fits), flush=True)
    except BaseException:
        write(task / 'failure.json', dict(status='FAIL', error=traceback.format_exc()))
        raise
    finally:
        if features is not None:
            features.close()
        if store is not None:
            store.close()
        if bundle is not None and hasattr(bundle.get('old_maps'), '_mmap'):
            bundle['old_maps']._mmap.close()
        gc.collect()
        write(task / 'handle-release.json', dict(status='PASS', mmap_handles_closed=True, rgb_store_closed=True,
            scratch_retained_for_root_owned_cleanup=str(task / 'scratch-v1'), process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--run-name', choices=('run-v1', 'run-v2'), default='run-v1')
    args = parser.parse_args()
    run(args.root.resolve(), args.task.resolve(), args.run_name)
