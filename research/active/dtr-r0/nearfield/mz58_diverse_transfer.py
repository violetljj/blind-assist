"""Fixed-checkpoint MZ55 transfer; requires complete source and root GPU GO.

The runner reads observable compact members only. Evaluator labels and source
grouping are opened separately by mz58_score.py after predictions are sealed.
"""
import argparse
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from data_lightweight import CompactSource
from mz5_ensemble_readout import read, write, sha, load_npz
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import Bindings, FrozenModels, CONDITIONS, setup
from mz47_local_enrichment import packet
from mz50_echo_independent import SpatialQuery
from mz50_train import infer as infer_spatial
from mz54_full_rgb_source import RGBRef, RGBStore
from mz56_global_anchor import saved_outputs
from mz56_global_anchor_model import AnchorQuery

MODES = ('LOCAL_ONLY', 'GLOBAL_ANCHOR', 'GLOBAL_SUPPRESSED')
LOCAL_KEYS = ('OLD_NEG/OPEN', 'OLD_NEG/GATED') + tuple('MZ56/' + mode for mode in MODES)
UNIONS = tuple('MZ57/' + mode + '/UNION' for mode in MODES)
METHODS = ('MZ37', 'OLD_NEG/OPEN/candidate', 'OLD_NEG/GATED/candidate', 'OLD_NEG/UNION') + tuple('MZ56/' + mode + '/candidate' for mode in MODES) + UNIONS
CALIBRATION = dict(native_size=[640, 360], horizontal_fov_degrees=100, eye_height_m=1.7, zone_crop_degrees=[45, 45])


def bound_ref(source, ref, bind):
    path = (source / ref['path']).resolve(strict=True)
    assert path.is_relative_to(source.resolve()), path
    return bind(path, ref['sha256'])


def load_predictor(source, bind):
    index_path = bind(source / 'source-index.json')
    index = read(index_path)
    assert index['schema'] == 'mz55-diverse-source-v1' and index['status'] == 'COMPLETE'
    assert index['frames'] == 2560 and index['source_role'] == 'CONSUMED_DEVELOPMENT'
    assert len(index['shards']) == 10
    combined = index['combined']
    receipt = read(bound_ref(source, combined['receipt.json'], bind))
    assert receipt['status'] == 'PASS' and receipt['original_sources_unchanged']
    for name in ('packets.npz', 'rgb-refs.json'):
        assert receipt['outputs'][name] == combined[name]['sha256']
    packets = load_npz(bound_ref(source, combined['packets.npz'], bind))
    references = read(bound_ref(source, combined['rgb-refs.json'], bind))['frames']
    assert set(packets) == {'frame_ids', 'ranges', 'valid'}
    assert packets['ranges'].shape == packets['valid'].shape == (2560, 64, 2)
    assert packets['ranges'].dtype == np.float32 and packets['valid'].dtype == np.bool_
    assert len(references) == 2560 and len(set(packets['frame_ids'])) == 2560
    np.testing.assert_array_equal(packets['frame_ids'], [row['frame_id'] for row in references])
    expected = {(row['archive']['path'], row['archive']['sha256']) for row in index['shards']}
    actual = {(row['archive']['path'], row['archive']['sha256']) for row in references}
    assert expected == actual and len(actual) == 10
    archives = {}
    for relative, digest in sorted(actual):
        path = bound_ref(source, dict(path=relative, sha256=digest), bind)
        with CompactSource(path) as archive:
            declaration = archive.read_json('model/predictor.json')
            assert declaration['schema'] == 'mz55-observable-input-v1'
            assert declaration['calibration'] == CALIBRATION
            assert declaration['packets'] == 'model/packets.npz'
            selected = [row for row in references if row['archive']['path'] == relative]
            assert declaration['frames'] == len(selected)
            assert declaration['rgb_files'] == [row['member'] for row in selected]
            for row in selected:
                assert row['member'].startswith('model/') and row['member'].endswith('.png')
                assert archive.entries[row['member']]['sha256'] == row['sha256']
        archives[relative] = path
    refs = [RGBRef(str(archives[row['archive']['path']]), row['sha256'], row['member']) for row in references]
    return dict(frame_ids=packets['frame_ids'], ranges=packets['ranges'], valid=packets['valid'], rgb_refs=refs), index_path


def load_state(model, state):
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    assert set(actual) == set(state)
    for key, expected in state.items():
        assert actual[key].shape == expected.shape and actual[key].dtype == expected.dtype
        torch.testing.assert_close(actual[key].detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    return model.cuda().eval().requires_grad_(False)


def load_models(root, bind, out):
    work = root / 'artifacts.local/work'
    models = FrozenModels(root, bind)
    r51_path = work / 'mz51-training-coverage-20260911/run-v1'
    r51 = read(bind(r51_path / 'receipt.json'))
    r54_path = work / 'mz54-full-rgb-20260911/run-v1'
    r54 = read(bind(r54_path / 'receipt.json'))
    r56_path = work / 'mz56-global-anchor-20260911/run-v2'
    r56 = read(bind(r56_path / 'receipt.json'))
    assert r51['status'] == r54['status'] == r56['status'] == 'PASS'
    # Freeze the actual inherited anchor implementation, not merely its class name.
    expected_model = [(p, h) for p, h in r56['inputs'].items() if p.replace('\\', '/').endswith('/mz56_global_anchor_model.py')]
    assert len(expected_model) == 1
    bind(*expected_model[0])
    old_path = bind(r51_path / 'OLD_NEG.pt', r51['outputs']['OLD_NEG.pt'])
    spatial_state = torch.load(old_path, map_location='cpu', weights_only=True)
    spatial = load_state(SpatialQuery(models.rank), spatial_state)
    initial_path = bind(r54_path / 'FULL_RASTER.pt', r54['outputs']['FULL_RASTER.pt'])
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    anchors, states, checkpoint_refs = {}, {}, dict(OLD_NEG=dict(path=str(old_path), sha256=sha(old_path)))
    for mode in MODES[:2]:
        path = bind(r56_path / (mode + '.pt'), r56['outputs'][mode + '.pt'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        anchors[mode] = load_state(AnchorQuery(initial, mode), state)
        states[mode] = len(state)
        checkpoint_refs[mode] = dict(path=str(path), sha256=sha(path))
    cuts, cut_refs = {}, {}
    for key, folder, receipt, filename in (
        ('OLD_NEG/OPEN', r51_path, r51, 'OLD_NEG-OPEN-cutoff.npy'),
        ('OLD_NEG/GATED', r51_path, r51, 'OLD_NEG-GATED-cutoff.npy'),
        ('MZ56/LOCAL_ONLY', r56_path, r56, 'LOCAL_ONLY-cutoff.npy'),
        ('MZ56/GLOBAL_ANCHOR', r56_path, r56, 'GLOBAL_ANCHOR-cutoff.npy')):
        path = bind(folder / filename, receipt['outputs'][filename])
        value = np.load(path, allow_pickle=False)
        assert value.shape == (4,) and np.isfinite(value).all()
        cuts[key] = value
        np.save(out / filename, value)
        cut_refs[key] = dict(path=str(path), sha256=sha(path), filename=filename, values=value.tolist())
    cuts['MZ56/GLOBAL_SUPPRESSED'] = cuts['MZ56/GLOBAL_ANCHOR']
    cut_refs['MZ56/GLOBAL_SUPPRESSED'] = cut_refs['MZ56/GLOBAL_ANCHOR']
    for eid in ('mz51-training-coverage-20260911', 'mz56-global-anchor-20260911', 'mz57-complementary-scale-union-20260911'):
        folder = work / eid / 'score-v1'
        receipt = read(bind(folder / 'receipt.json'))
        assert receipt['status'] == 'PASS'
        bind(folder / 'result.json', receipt['outputs']['result.json'])
    return models, spatial, anchors, cuts, dict(checkpoints=checkpoint_refs, cutoffs=cut_refs,
        strict_state_tensors=dict(OLD_NEG=len(spatial_state), **states), original_model_inference_parity_frames=0)


def run(root, task):
    started = time.perf_counter()
    out = task / 'run-v1'
    out.mkdir(parents=True, exist_ok=False)
    store = models = spatial = anchors = None
    try:
        bind = Bindings()
        for name in ('MZ58_DIVERSE_TRANSFER_20260911.md', 'mz58_diverse_transfer.py', 'mz55_index.py',
                     'mz45_object_transfer.py', 'mz50_train.py', 'mz50_echo_independent.py',
                     'mz54_full_rgb_source.py', 'mz56_global_anchor.py', 'mz56_global_anchor_model.py',
                     'mz36_frozen_inference.py', 'mz47_local_enrichment.py', 'mz40_packets.py', 'data_lightweight.py'):
            bind(Path(__file__).with_name(name))
        source = root / 'artifacts.local/work/mz55-diverse-mesh-source-20260911'
        predictor, index_path = load_predictor(source, bind)
        setup()
        models, spatial, anchors, cuts, frozen = load_models(root, bind, out)
        write(out / 'start.json', dict(status='STARTED', frames=2560, methods=METHODS, profiles=CONDITIONS,
            training_steps=0, new_cutoffs=0, source_index_sha256=sha(index_path), inputs=bind.inputs,
            fixed_batch_size=16, encoder_views=['global256x144_BOX', 'crop224x224', 'full640x360'],
            predictor_inputs=['RGB-derived features', 'observed ranges/validity', 'fixed calibration'],
            evaluator_labels_read=False, old_dense_cache_reads=0, permanent_dense_cache=False,
            device=torch.cuda.get_device_name(), frozen=frozen))
        store = RGBStore()
        parts, timing = {}, dict(visual_views=0., fixed_readouts=0.)
        with torch.inference_mode():
            for begin in range(0, 2560, 16):
                ids = np.arange(begin, min(begin+16, 2560))
                images = []
                try:
                    images = [store.load(predictor['rgb_refs'][int(i)]) for i in ids]
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    visual, detail = models.visual(images)
                    full_rgb = torch.from_numpy(np.stack([np.asarray(im, np.uint8) for im in images]))
                    full_rgb = full_rgb.permute(0, 3, 1, 2).cuda().float() / 255
                    full = fixed_batch_dense(models.context.base, full_rgb)
                    assert full.shape == (len(ids), 64, 45, 80) and detail.shape == (len(ids), 64, 28, 28)
                    normalized_full = (full - models.detail_mean) / models.detail_std
                    detail_cpu = detail.cpu().numpy()
                    torch.cuda.synchronize()
                    timing['visual_views'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    for profile in CONDITIONS:
                        observed = packet(predictor['ranges'][ids], predictor['valid'][ids], profile)
                        baseline = models.predict(visual, detail, observed['ranges'], observed['valid'])
                        old = infer_spatial(spatial, detail_cpu, observed['ranges'], observed['valid'], models)
                        row = dict(MZ37=baseline['MZ37'], ranges=observed['ranges'], valid=observed['valid'])
                        for key, value in old.items():
                            row['OLD_NEG/' + key] = value
                        args = (normalized_full, torch.from_numpy(observed['ranges']).cuda(), torch.from_numpy(observed['valid']).cuda())
                        for mode in MODES:
                            model = anchors['GLOBAL_ANCHOR' if mode == 'GLOBAL_SUPPRESSED' else mode]
                            values = saved_outputs(model, args, suppress=mode == 'GLOBAL_SUPPRESSED')
                            for key, value in values.items():
                                row['MZ56/' + mode + '/' + key] = value
                        for key in LOCAL_KEYS:
                            margin = row[key + '/raw'].astype(float) - cuts[key]
                            accepted = (row['MZ37'] < 0) & row[key + '/support'] & (margin >= 0)
                            row[key + '/candidate'] = np.where(accepted, margin, row['MZ37'])
                        row['OLD_NEG/UNION'] = np.maximum(row['OLD_NEG/OPEN/candidate'], row['OLD_NEG/GATED/candidate'])
                        for mode, union in zip(MODES, UNIONS):
                            row[union] = np.maximum(row['OLD_NEG/UNION'], row['MZ56/' + mode + '/candidate'])
                        for key, value in row.items():
                            parts.setdefault(profile + '/' + key, []).append(value)
                    torch.cuda.synchronize()
                    timing['fixed_readouts'] += time.perf_counter() - tick
                finally:
                    for image in images:
                        image.close()
                if begin % 512 == 0 or begin + len(ids) == 2560:
                    progress = dict(stage='fixed-transfer', frames=begin+len(ids), total=2560)
                    write(out / 'progress.json', progress)
                    print('TRANSFER', progress, flush=True)
        predictions = {key: np.concatenate(values) for key, values in parts.items()}
        predictions['frame_ids'] = predictor['frame_ids']
        for key in ('crop_mask', 'sensor_coverage', 'rays'):
            predictions['geometry/' + key] = getattr(anchors['GLOBAL_ANCHOR'], key).cpu().numpy()
        predictions['geometry/angular_rays'] = spatial.rays.cpu().numpy()
        predictions['geometry/zone_angles'] = anchors['GLOBAL_ANCHOR'].zone_angles.cpu().numpy()
        np.savez_compressed(out / 'predictions.npz', **predictions)
        assert store.loads == 2560
        bind.check()
        receipt = dict(status='PASS', inputs=bind.inputs, source_index_sha256=sha(index_path),
            source_task=str(source.resolve()), frames=2560, methods=METHODS, profiles=CONDITIONS,
            fixed_checkpoints=True, frozen=frozen, training_steps=0, new_cutoffs=0,
            existing_cutoff_vectors=4, source_calibration_rows_used=0, old_cohort_replay_frames=0,
            evaluator_labels_read=False, native_depth_reads=0, old_dense_cache_reads=0, permanent_dense_cache=False,
            encoder_frames_per_view=dict(global256x144_BOX=2560, crop224x224=2560, full640x360=2560),
            fixed_encoder_batch=16, rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            seconds=time.perf_counter()-started, timing_seconds=timing,
            backend='CUDA frozen inference; CPU compact-source I/O', device=torch.cuda.get_device_name(),
            all_roles_are_descriptive=True, source_role='CONSUMED_DEVELOPMENT',
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()})
        write(out / 'receipt.json', receipt)
        print('PASS', {k: receipt[k] for k in ('frames', 'seconds', 'timing_seconds', 'rgb_loads')}, flush=True)
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', error=traceback.format_exc()))
        raise
    finally:
        if store is not None:
            store.close()
        del models, spatial, anchors
        gc.collect()
        write(task / 'handle-release.json', dict(status='PASS', compact_handles_closed=True,
            dense_files_created=0, process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.task.resolve())
