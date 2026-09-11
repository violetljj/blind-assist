"""MZ69 frozen MZ67 topology transfer; no execution until registered/root GPU GO.

The runner reads observable compact members only. Evaluator labels and source
grouping are opened separately by the separately authorized MZ69 scorer after predictions are sealed.
"""
from pathlib import Path
import argparse
import sys
ROOT = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path("E:/linnan/linnan")
sys.path.insert(0, str(ROOT / "research/active/dtr-r0/nearfield"))
import gc
import time
import traceback

import numpy as np
import torch

from data_lightweight import CompactSource
from mz5_ensemble_readout import read, write, sha, load_npz
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import Bindings, FrozenModels, setup
from mz47_local_enrichment import packet as original_packet
from mz50_echo_independent import SpatialQuery
from mz50_train import infer as infer_spatial
from mz54_full_rgb_source import RGBRef, RGBStore
from mz56_global_anchor import saved_outputs
from mz56_global_anchor_model import AnchorQuery

CONDITIONS = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE', 'ALL_INVALID')


def packet(ranges, valid, profile):
    if profile == 'ALL_INVALID':
        return dict(ranges=np.zeros_like(ranges), valid=np.zeros_like(valid, dtype=bool))
    assert profile in CONDITIONS[:3]
    return original_packet(ranges, valid, profile)


HEADS = ('MZ64/GEOMETRY', 'MZ68/NULL_COVERAGE')
LOCAL_KEYS = ('OLD_NEG/OPEN', 'OLD_NEG/GATED') + HEADS
UNIONS = tuple(key + '/UNION' for key in HEADS)
METHODS = ('MZ37', 'OLD_NEG/UNION') + tuple(k + '/candidate' for k in LOCAL_KEYS) + UNIONS
CALIBRATION = dict(native_size=[640, 360], horizontal_fov_degrees=100, eye_height_m=1.7, zone_crop_degrees=[45, 45])


def bound_ref(source, ref, bind):
    path = (source / ref['path']).resolve(strict=True)
    assert path.is_relative_to(source.resolve()), path
    return bind(path, ref['sha256'])


def validate_index_header(index):
    assert index['schema'] == 'mz67-topology-source-v1' and index['status'] == 'COMPLETE'
    assert index['frames'] == 4096 and index['source_role'] == 'CONSUMED_DEVELOPMENT'
    assert len(index['shards']) == 10
    assert index['manifest']['sha256'] == '851482eba0c6bc6c201897ef090e2b29d223542be94fef30b7735e24ab10b99a'


def load_predictor(source, bind, expected_index_sha256):
    index_path = bind(source / 'source-index.json', expected_index_sha256)
    index = read(index_path)
    validate_index_header(index)
    combined = index['combined']
    receipt = read(bound_ref(source, combined['receipt.json'], bind))
    assert receipt['status'] == 'PASS' and receipt['original_sources_unchanged']
    for name in ('packets.npz', 'rgb-refs.json'):
        assert receipt['outputs'][name] == combined[name]['sha256']
    packets = load_npz(bound_ref(source, combined['packets.npz'], bind))
    references = read(bound_ref(source, combined['rgb-refs.json'], bind))['frames']
    assert set(packets) == {'frame_ids', 'ranges', 'valid'}
    assert packets['ranges'].shape == packets['valid'].shape == (4096, 64, 2)
    assert packets['ranges'].dtype == np.float32 and packets['valid'].dtype == np.bool_
    assert len(references) == 4096 and len(set(packets['frame_ids'])) == 4096
    np.testing.assert_array_equal(packets['frame_ids'], [row['frame_id'] for row in references])
    assert np.isfinite(packets['ranges']).all() and (packets['ranges'][~packets['valid']] == 0).all()
    assert ((packets['ranges'][packets['valid']] > 0) & (packets['ranges'][packets['valid']] <= 4)).all()
    positions = {str(v): i for i, v in enumerate(packets['frame_ids'])}
    expected = {(row['archive']['path'], row['archive']['sha256']) for row in index['shards']}
    actual = {(row['archive']['path'], row['archive']['sha256']) for row in references}
    assert expected == actual and len(actual) == 10
    archives = {}
    for relative, digest in sorted(actual):
        path = bound_ref(source, dict(path=relative, sha256=digest), bind)
        with CompactSource(path) as archive:
            declaration = archive.read_json('model/predictor.json')
            assert declaration['schema'] == 'mz67-observable-input-v1'
            assert declaration['calibration'] == CALIBRATION
            assert declaration['packets'] == 'model/packets.npz'
            selected = [row for row in references if row['archive']['path'] == relative]
            assert declaration['frames'] == len(selected)
            assert declaration['rgb_files'] == [row['member'] for row in selected]
            with archive.open('model/packets.npz') as stream, np.load(stream, allow_pickle=False) as stored:
                assert set(stored.files) == {'ranges', 'valid'}
                ii = [positions[row['frame_id']] for row in selected]
                for key in ('ranges', 'valid'):
                    np.testing.assert_array_equal(stored[key], packets[key][ii])
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


class DesignBindings(Bindings):
    def __init__(self, design):
        super().__init__()
        self.expected = {str(Path(p).resolve()): h for p, h in design['frozen_models_dependency_graph']['graph'].items()}
        for section in ('code', 'checkpoints', 'cutoffs', 'receipts'):
            for ref in design[section].values():
                self.expected[str(Path(ref['path']).resolve())] = ref['sha256']
    def __call__(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        frozen = self.expected.get(str(path))
        if frozen is not None:
            assert expected is None or expected == frozen, str(path)
            expected = frozen
        return super().__call__(path, expected)


def load_models(root, bind, out, design):
    # Same FrozenModels construction graph; no pruning or refitting.
    for ref in design['code'].values():
        bind(ref['path'], ref['sha256'])
    for path, digest in design['frozen_models_dependency_graph']['graph'].items():
        if path.endswith('.py'):
            bind(path, digest)
    for path, digest in design['frozen_models_dependency_graph']['graph'].items():
        if not path.endswith('.py'):
            bind(path, digest)
    models = FrozenModels(root, bind)
    def state(key):
        ref = design['checkpoints'][key]
        return torch.load(bind(ref['path'], ref['sha256']), map_location='cpu', weights_only=True)
    spatial_state = state('OLD_NEG')
    spatial = load_state(SpatialQuery(models.rank), spatial_state)
    initial = state('MZ54_constructor')
    anchors, states = {}, {}
    for key in HEADS:
        checkpoint = state(key)
        anchors[key] = load_state(AnchorQuery(initial, 'GLOBAL_ANCHOR'), checkpoint)
        states[key] = len(checkpoint)
    cuts, cut_refs = {}, {}
    for key in LOCAL_KEYS:
        ref = design['cutoffs'][key]
        path = bind(ref['path'], ref['sha256'])
        value = np.load(path, allow_pickle=False)
        assert value.shape == (4,) and np.isfinite(value).all()
        np.testing.assert_array_equal(value, ref['values'])
        filename = key.replace('/', '-') + '-cutoff.npy'
        np.save(out / filename, value)
        assert sha(out / filename) == ref['sha256']
        cuts[key] = value
        cut_refs[key] = dict(ref, filename=filename)
    return models, spatial, anchors, cuts, dict(checkpoints=design['checkpoints'], cutoffs=cut_refs,
        strict_state_tensors=dict(OLD_NEG=len(spatial_state), **states), original_model_inference_parity_frames=0)


def validate_authority(registration, protocol, design_path, design_sha, index_sha):
    """Root registration locks scientific inputs before any CUDA or output creation."""
    record = read(registration)
    assert record['experiment_id'] == 'mz69-topology-transfer-20260911'
    assert record['status'] == 'ACTIVE'
    assert index_sha == '23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b'
    assert record['source_index_sha256'] == index_sha
    assert record['design_sha256'] == design_sha == sha(design_path)
    assert record['runner_sha256'] == sha(Path(__file__))
    assert record['protocol_sha256'] == sha(protocol)
    inputs_ref = record['inputs']
    assert sha(inputs_ref['path']) == inputs_ref['sha256']
    manifest = read(inputs_ref['path'])
    assert manifest['experiment_id'] == record['experiment_id']
    assert manifest['status'] == 'FROZEN_BEFORE_INFERENCE'
    assert manifest['frames'] == 4096 and manifest['training_steps'] == 0
    assert manifest['calibration_rows'] == 0 and tuple(manifest['profiles']) == CONDITIONS
    assert manifest['source_index']['sha256'] == index_sha
    assert manifest['design']['sha256'] == design_sha
    assert Path(manifest['design']['path']).resolve() == design_path.resolve()
    expected_code = {Path(__file__).name: record['runner_sha256'],
                     protocol.name: record['protocol_sha256'],
                     'mz69_score.py': record['scorer_sha256']}
    assert {Path(v['path']).name: v['sha256'] for v in manifest['code']} == expected_code
    paths = {inputs_ref['path']: inputs_ref['sha256']}
    for item in manifest['code'] + [manifest['root_review'], manifest['source_index'], manifest['design']]:
        assert sha(item['path']) == item['sha256']
        paths[item['path']] = item['sha256']
    assert read(manifest['root_review']['path'])['status'] == 'PASS_READY_FOR_REGISTERED_TRANSFER'
    record['_verified_paths'] = paths
    return record


def run(root, task, source, index_sha, design_path, design_sha, protocol, registration):
    assert len(index_sha) == 64 and all(c in '0123456789abcdef' for c in index_sha)
    assert task.is_relative_to((root / 'artifacts.local/work').resolve())
    assert not task.is_relative_to(source) and not source.is_relative_to(task)
    assert protocol.is_file()
    authority = validate_authority(registration, protocol, design_path, design_sha, index_sha)
    started = time.perf_counter()
    out = task / 'run-v1'
    out.mkdir(parents=True, exist_ok=False)
    store = models = spatial = anchors = None
    try:
        design = read(design_path)
        assert design['schema'] == 'mz69-fixed-transfer-inputs-v1' and design['status'] == 'FROZEN_PREPARATION'
        assert design['source_index']['sha256'] == index_sha
        assert Path(design['source_index']['path']).resolve() == source / 'source-index.json'
        assert tuple(design['methods']) == METHODS and tuple(design['profiles']) == CONDITIONS
        bind = DesignBindings(design)
        bind(design_path, design_sha); bind(protocol); bind(Path(__file__)); bind(registration)
        for path, digest in authority['_verified_paths'].items():
            bind(path, digest)
        predictor, index_path = load_predictor(source, bind, index_sha)
        setup()
        models, spatial, anchors, cuts, frozen = load_models(root, bind, out, design)
        write(out / 'start.json', dict(status='STARTED', frames=4096, methods=METHODS, profiles=CONDITIONS,
            training_steps=0, new_cutoffs=0, source_index_sha256=sha(index_path), inputs=bind.inputs,
            fixed_batch_size=16, encoder_views=['global256x144_BOX', 'crop224x224', 'full640x360'],
            predictor_inputs=['RGB-derived features', 'observed ranges/validity', 'fixed calibration'],
            evaluator_labels_read=False, old_dense_cache_reads=0, permanent_dense_cache=False,
            device=torch.cuda.get_device_name(), frozen=frozen))
        store = RGBStore()
        parts, timing = {}, dict(rgb_decode=0., visual_views=0., fixed_readouts=0.)
        with torch.inference_mode():
            for begin in range(0, 4096, 16):
                ids = np.arange(begin, min(begin+16, 4096))
                images = []
                try:
                    decode_tick = time.perf_counter()
                    for i in ids:
                        images.append(store.load(predictor['rgb_refs'][int(i)]))
                    timing['rgb_decode'] += time.perf_counter() - decode_tick
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
                        for head in HEADS:
                            values = saved_outputs(anchors[head], args, suppress=False)
                            available = observed['valid'].any((1, 2))
                            np.testing.assert_array_equal(values['anchor_available'], available)
                            assert values['anchor_vector'].shape == (len(ids), 8)
                            assert (values['anchor_vector'][~available] == 0).all()
                            for key, value in values.items():
                                row[head + '/' + key] = value
                        for key in LOCAL_KEYS:
                            margin = row[key + '/raw'].astype(float) - cuts[key]
                            accepted = (row['MZ37'] < 0) & row[key + '/support'] & (margin >= 0)
                            row[key + '/candidate'] = np.where(accepted, margin, row['MZ37'])
                        row['OLD_NEG/UNION'] = np.maximum(row['OLD_NEG/OPEN/candidate'], row['OLD_NEG/GATED/candidate'])
                        for head, union in zip(HEADS, UNIONS):
                            row[union] = np.maximum(row['OLD_NEG/UNION'], row[head + '/candidate'])
                        for key, value in row.items():
                            parts.setdefault(profile + '/' + key, []).append(value)
                    torch.cuda.synchronize()
                    timing['fixed_readouts'] += time.perf_counter() - tick
                finally:
                    for image in images:
                        image.close()
                if begin % 512 == 0 or begin + len(ids) == 4096:
                    progress = dict(stage='fixed-transfer', frames=begin+len(ids), total=4096)
                    write(out / 'progress.json', progress)
                    print('TRANSFER', progress, flush=True)
        predictions = {key: np.concatenate(values) for key, values in parts.items()}
        predictions['frame_ids'] = predictor['frame_ids']
        for key in ('crop_mask', 'sensor_coverage', 'rays'):
            predictions['geometry/' + key] = getattr(anchors[HEADS[0]], key).cpu().numpy()
        predictions['geometry/angular_rays'] = spatial.rays.cpu().numpy()
        predictions['geometry/zone_angles'] = anchors[HEADS[0]].zone_angles.cpu().numpy()
        np.savez_compressed(out / 'predictions.npz', **predictions)
        assert store.loads == 4096
        bind.check()
        receipt = dict(status='PASS', inputs=bind.inputs, source_index_sha256=sha(index_path),
            source_task=str(source.resolve()), design_input=dict(path=str(design_path), sha256=sha(design_path)), protocol=dict(path=str(protocol), sha256=sha(protocol)), frames=4096, methods=METHODS, profiles=CONDITIONS,
            fixed_checkpoints=True, frozen=frozen, training_steps=0, new_cutoffs=0,
            existing_cutoff_vectors=4, source_calibration_rows_used=0, old_cohort_replay_frames=0,
            evaluator_labels_read=False, native_depth_reads=0, old_dense_cache_reads=0, permanent_dense_cache=False,
            encoder_frames_per_view=dict(global256x144_BOX=4096, crop224x224=4096, full640x360=4096),
            fixed_encoder_batch=16, rgb_loads=store.loads, rgb_bytes_read=store.bytes_read,
            seconds=time.perf_counter()-started, timing_seconds=timing,
            backend='CUDA frozen inference; CPU compact-source I/O', device=torch.cuda.get_device_name(),
            added_condition='ALL_INVALID zero ranges/validity; no residual ranges, no failure-probability claim',
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
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-index-sha256', required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--design-sha256', required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(strict=True), args.task.resolve(), args.source.resolve(strict=True),
        args.source_index_sha256, args.design.resolve(strict=True), args.design_sha256,
        args.protocol.resolve(strict=True), args.registration.resolve(strict=True))
