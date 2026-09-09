"""Independent MZ5 geometry, readout, schedule, and saved-result audit.

No source images/native depth are opened and no optimizer steps are performed.
--self-check uses synthetic CPU tensors. A completed run is replayed only with
the requested device; the reference forward never calls SpatialFusion.forward.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
ARMS = ('POOLED_FUSION', 'LOCAL_FUSION', 'LOCAL_RGB_ONLY')
ROLES = ('TRAIN_ONLY', 'DEV_ONLY', 'EVAL_ONLY')
EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
BUFFERS = ('valid', 'xyz', 'zone', 'covered', 'radial', 'grid')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def state_sha(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        for part in (key.encode(), str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            digest.update(part)
    return digest.hexdigest()


def reference_lattice():
    """Reconstruct the declared physical query lattice, without project imports."""
    coordinates = []
    for front, half_width, bottom, top in ((.18, .28, .65, 1.4), (.13, .18, 1.4, 1.85)):
        for distance in range(2):
            for lateral in range(3):
                sample = []
                for fx in (.15, .5, .85):
                    for fy in (.15, .5, .85):
                        for fz in (.15, .5, .85):
                            sample.append((front + 1.5 * (distance + fx),
                                half_width * (-1 + 2 * (lateral + fy) / 3),
                                bottom + (top - bottom) * fz))
                coordinates.append(sample)
    return np.asarray(coordinates, dtype=np.float64)


def zone_mapping(relative):
    # The MZ0 packet uses separate pinhole horizontal/vertical angles, not
    # spherical elevation atan(z / hypot(x,y)). Rows increase down the image.
    angles = np.rad2deg(np.arctan(relative[..., 1:] / relative[..., :1]))
    azimuth, elevation = angles[..., 0], angles[..., 1]
    col = np.clip(np.floor((azimuth + 22.5) / 5.625).astype(np.int64), 0, 7)
    row = np.clip(np.floor((22.5 - elevation) / 5.625).astype(np.int64), 0, 7)
    inside = (np.abs(angles) <= 22.5).all(-1) & (relative[..., 0] > 0)
    return 8 * row + col, inside


def verify_geometry(values):
    arrays = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
              for key, value in values.items()}
    physical = arrays['xyz'].astype(np.float64) * np.array([3.18, .28, 1.85])
    expected = reference_lattice()
    np.testing.assert_allclose(physical, expected, rtol=0, atol=1.8e-7)
    relative = physical - [0., 0., 1.7]
    tangent = np.tan(np.deg2rad(50.))
    projection = np.stack((relative[..., 1] / (relative[..., 0] * tangent),
        -relative[..., 2] / (relative[..., 0] * tangent * 360 / 640)), -1)
    np.testing.assert_allclose(arrays['grid'], projection, rtol=0, atol=4e-7)
    rgb_valid = (np.abs(projection) <= 1).all(-1)
    np.testing.assert_array_equal(arrays['valid'], rgb_valid)
    zones, in_fov = zone_mapping(relative)
    np.testing.assert_array_equal(arrays['zone'], zones)
    np.testing.assert_array_equal(arrays['covered'], in_fov & rgb_valid)
    np.testing.assert_allclose(arrays['radial'], np.sqrt((relative ** 2).sum(-1)) / 4,
                               rtol=0, atol=5e-8)
    # Round-trip every declared zone center and both image/FoV directions.
    centers = []
    for row in range(8):
        for col in range(8):
            centers.append([1., np.tan(np.deg2rad(-22.5 + (col + .5) * 5.625)),
                np.tan(np.deg2rad(22.5 - (row + .5) * 5.625))])
    center_zones, center_valid = zone_mapping(np.asarray(centers))
    np.testing.assert_array_equal(center_zones, np.arange(64))
    assert center_valid.all()
    outside = np.array([[1., np.tan(np.deg2rad(a)), 0.] for a in (-22.51, 22.51)] +
                       [[1., 0., np.tan(np.deg2rad(a))] for a in (-22.51, 22.51)])
    assert not zone_mapping(outside)[1].any()
    return dict(query_points=324, rgb_valid=int(rgb_valid.sum()), tof_covered=int((in_fov & rgb_valid).sum()),
        rgb_valid_by_query=rgb_valid.sum(1).tolist(), tof_covered_by_query=(in_fov & rgb_valid).sum(1).tolist(),
        physical_coordinate_max_error_m=float(np.abs(physical - expected).max()),
        projection_max_error=float(np.abs(arrays['grid'] - projection).max()),
        zone_centers_verified=64, outside_fov_rays_verified=4)


def reference_weights():
    """Same declared initialization order, using only elementary layers."""
    torch.manual_seed(53)
    point = nn.Sequential(nn.Linear(74, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
    readout = nn.Sequential(nn.Linear(1412, 88), nn.ReLU(), nn.Linear(88, 4))
    result = {f'point.{key}': value for key, value in point.state_dict().items()}
    result.update({f'readout.{key}': value for key, value in readout.state_dict().items()})
    assert sum(value.numel() for value in result.values()) == 131580
    assert 64 + 3 + 2 + 2 + 2 + 1 == 74 and 12 * 32 + 772 + 256 == 1412
    return result


def point_channels(points, tof, geometry, arm):
    """Build the reference 74-channel tensor with flat packet indexing."""
    count = len(points)
    mask = geometry['valid'].reshape(1, 12, 27, 1)
    if arm == 'POOLED_FUSION':
        means = (points * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        points = means[:, :, None, :].expand(count, 12, 27, 64)
    observation = torch.zeros_like(tof) if arm == 'LOCAL_RGB_ONLY' else tof
    slots = (geometry['zone'][..., None] * 2 + torch.arange(2, device=points.device)).reshape(-1)
    ranges = observation[:, slots].reshape(count, 12, 27, 2)
    validity = observation[:, slots + 128].reshape(count, 12, 27, 2)
    validity = validity * geometry['covered'][None, ..., None]
    ranges = ranges * validity
    difference = (ranges - geometry['radial'][None, ..., None]) * validity
    channels = torch.cat([points, geometry['xyz'][None].expand(count, -1, -1, -1),
        ranges, validity, difference, geometry['covered'][None, ..., None].expand(count, -1, -1, -1)], -1)
    assert channels.shape == (count, 12, 27, 74)
    return channels, observation


def reference_forward(state, points, visual, tof, arm):
    channels, observation = point_channels(points, tof, state, arm)
    local = F.relu(F.linear(channels, state['point.0.weight'], state['point.0.bias']))
    local = F.relu(F.linear(local, state['point.2.weight'], state['point.2.bias']))
    mask = state['valid'].reshape(1, 12, 27, 1)
    pooled = (local * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
    combined = torch.cat([visual, observation, pooled.reshape(len(points), 384)], dim=1)
    hidden = F.relu(F.linear(combined, state['readout.0.weight'], state['readout.0.bias']))
    return F.linear(hidden, state['readout.2.weight'], state['readout.2.bias'])


def independent_permutation(valid):
    generator = np.random.default_rng(83)
    result = np.tile(np.arange(27), (12, 1))
    for query, mask in enumerate(valid):
        indices = np.where(mask)[0]
        result[query, indices] = generator.permutation(indices)
    return result


def source_contract():
    fusion_path = Path(__file__).with_name('mz5_spatial_fusion.py')
    cache_path = Path(__file__).with_name('mz5_spatial_features.py')
    fusion, cache = fusion_path.read_text(encoding='utf-8'), cache_path.read_text(encoding='utf-8')
    trees = [ast.parse(text) for text in (fusion, cache)]
    forward = next(node for node in ast.walk(trees[0]) if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    assert [arg.arg for arg in forward.args.args] == ['self', 'points', 'visual', 'tof']
    assert not ({'truth', 'counts', 'role', 'region_id', 'condition', 'family', 'native'} &
                {node.id for node in ast.walk(forward) if isinstance(node, ast.Name)})
    row_keys = {node.slice.value for node in ast.walk(trees[1]) if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id in ('r', 'row')
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}
    assert row_keys <= {'rgb_file', 'rgb_sha256', 'status', 'source_role'}
    assert "data['truth'][train].astype(np.float32), device=dev" in fusion
    assert "data['truth'][train[:128]].astype(np.float32)" in fusion
    assert 'range(STEPS)' in fusion and 'lr=.001, weight_decay=.0001' in fusion
    assert 'binary_cross_entropy_with_logits(logits, target[target_indices[step]])' in fusion
    assert "torch.backends.cudnn.allow_tf32 = False" in fusion
    assert "torch.backends.cudnn.allow_tf32 = True" in cache
    return dict(model_arguments=['points', 'visual', 'tof'], cache_row_keys=sorted(row_keys),
        label_device_paths='TRAIN_ONLY subset and TRAIN_ONLY workload probe only; scoring remains CPU',
        cached_visual_lineage='frozen RGB encoder samples and RGB-only JOINT predictions',
        native_depth_reads_in_recache=False, geometry_is_fixed=True,
        training_cudnn_tf32=False, recache_cudnn_tf32=True,
        source_sha256={fusion_path.name: sha(fusion_path), cache_path.name: sha(cache_path)},
        limit='Code-path and interface review; this does not independently replay optimizer history')


def self_check():
    # Import production only here, to compare synthetic invariants against the
    # independently constructed reference. Saved-run replay does not import it.
    from mz5_spatial_fusion import SpatialFusion
    torch.set_num_threads(1)
    expected_weights = reference_weights()
    torch.manual_seed(53)
    base = SpatialFusion()
    state = base.state_dict()
    for key, expected in expected_weights.items():
        torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
    geometry_result = verify_geometry(state)
    rng = np.random.default_rng(1701)
    points = torch.from_numpy(rng.normal(size=(7, 12, 27, 64)).astype(np.float32))
    visual = torch.from_numpy(rng.normal(size=(7, 772)).astype(np.float32))
    tof = torch.from_numpy(np.concatenate([rng.uniform(0, 1, (7, 128)), rng.integers(0, 2, (7, 128))], 1).astype(np.float32))
    tof[:, :128] *= tof[:, 128:]
    other_tof = 1 - tof
    permutation = independent_permutation(state['valid'].numpy())
    shuffled = points[:, np.arange(12)[:, None], permutation]
    np.testing.assert_array_equal(permutation[~state['valid'].numpy()], np.tile(np.arange(27), (12, 1))[~state['valid'].numpy()])
    altered_invalid = points.clone()
    altered_invalid[:, ~state['valid']] = 1000
    errors = {}
    with torch.inference_mode():
        for arm in ARMS:
            model = SpatialFusion(arm)
            model.load_state_dict(state)
            actual = model(points, visual, tof)
            expected = reference_forward(state, points, visual, tof, arm)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(model(altered_invalid, visual, tof), actual, rtol=0, atol=0)
            errors[arm] = float((actual - expected).abs().max())
            if arm == 'POOLED_FUSION':
                torch.testing.assert_close(model(shuffled, visual, tof), actual, rtol=2e-6, atol=2e-6)
            if arm == 'LOCAL_RGB_ONLY':
                torch.testing.assert_close(model(points, visual, other_tof), actual, rtol=0, atol=0)
                channels, global_tof = point_channels(points, tof, state, arm)
                assert not channels[..., 67:73].count_nonzero() and not global_tof.count_nonzero()
        channels, _ = point_channels(points, tof, state, 'LOCAL_FUSION')
        assert not channels[:, ~state['covered'], 67:73].count_nonzero()
        assert not channels[..., 67:69][channels[..., 69:71] == 0].count_nonzero()
        assert not channels[..., 71:73][channels[..., 69:71] == 0].count_nonzero()
    return dict(status='PASS', backend='CPU', synthetic_rows=7, geometry=geometry_result,
        trainable_parameters=131580, point_channels=74, readout_channels=1412,
        initial_seed=53, initial_parameters_bitwise_equal=True, reference_forward_max_errors=errors,
        pooled_valid_permutation_invariant=True, invalid_rgb_samples_masked=True,
        rgb_only_local_and_global_tof_ablation=True, invalid_and_outside_tof_local_channels_masked=True,
        source_contract=source_contract(), training_steps=0, source_images_read=0, native_depth_reads=0)


def scalar_metrics(pred, truth):
    """Scalar row/event implementation, independent of the vectorized scorer."""
    result = {key: dict(numerator=0, denominator=0) for key in
        ('spatial_exact', 'body_head_accuracy', 'near_far_accuracy', 'wrong_far', 'body_to_head', 'head_to_body')}
    result['event_confusion'] = {name: dict(tp=0, fp=0, fn=0, tn=0) for name in EVENTS}
    for p, t in zip(pred.tolist(), truth.tolist()):
        for key, passed in (
            ('spatial_exact', p == t),
            ('body_head_accuracy', [any(p[:2]), any(p[2:])] == [any(t[:2]), any(t[2:])]),
            ('near_far_accuracy', [p[0] or p[2], p[1] or p[3]] == [t[0] or t[2], t[1] or t[3]])):
            result[key]['numerator'] += int(passed)
            result[key]['denominator'] += 1
        for first in (0, 2):
            if t[first] and not t[first + 1]:
                result['wrong_far']['denominator'] += 1
                result['wrong_far']['numerator'] += int(p[first + 1])
        for key, source, other in (('body_to_head', slice(0, 2), slice(2, 4)),
                                    ('head_to_body', slice(2, 4), slice(0, 2))):
            if any(t[source]) and not any(t[other]):
                result[key]['denominator'] += 1
                result[key]['numerator'] += int(any(p[other]))
        for index, name in enumerate(EVENTS):
            label = 'tp' if p[index] and t[index] else 'fp' if p[index] else 'fn' if t[index] else 'tn'
            result['event_confusion'][name][label] += 1
    return result


def audit_run(run, cache, device):
    run, cache = Path(run).resolve(), Path(cache).resolve()
    source = WORK / 'mz1-tiny-fusion-20260910'
    receipt, cache_receipt, feature_receipt = read(run/'receipt.json'), read(cache/'receipt.json'), read(source/'cache-v1/features-receipt.json')
    assert receipt['status'] == cache_receipt['status'] == feature_receipt['status'] == 'PASS'
    expected_hashes = {run/'predictions.npz': receipt['predictions_sha256'],
        run/'metrics.json': receipt['metrics_sha256'], run/'source.py': receipt['source_sha256'],
        run/'protocol.md': receipt['protocol_sha256'], cache/'receipt.json': receipt['cache_receipt_sha256'],
        cache/'points.npy': cache_receipt['points_sha256'], cache/'geometry.npz': cache_receipt['geometry_sha256'],
        source/'cache-v1/features.npz': receipt['source_features_sha256']}
    for path, expected in expected_hashes.items():
        assert sha(path) == expected, str(path)
    assert receipt['source_features_sha256'] == cache_receipt['original_features_sha256'] == feature_receipt['feature_sha256']
    assert receipt['source_index_sha256'] == cache_receipt['source_index_sha256'] == feature_receipt['source_index_sha256']
    assert cache_receipt['frames'] == cache_receipt['original_alert_parity'] == cache_receipt['joint_flag_parity'] == 5000
    assert cache_receipt['averaged_feature_max_error'] < 1e-4 and cache_receipt['training_steps'] == 0
    assert cache_receipt['native_depth_read'] is False
    features = load_npz(source/'cache-v1/features.npz')
    prediction, reported = load_npz(run/'predictions.npz'), read(run/'metrics.json')
    role = features['role']
    assert {name: int(np.sum(role == name)) for name in ROLES} == dict(TRAIN_ONLY=2500, DEV_ONLY=1000, EVAL_ONLY=1500)
    for name in ('role', 'truth', 'joint', 'original_alerts'):
        np.testing.assert_array_equal(prediction[name], features[name])
    assert features['truth'].dtype == np.bool_ and features['original_alerts'].dtype == np.bool_
    assert np.isfinite(features['visual']).all() and np.isfinite(features['tof']).all()
    assert np.isin(features['tof'][:, 128:], [0, 1]).all()
    assert np.all(features['tof'][:, :128][features['tof'][:, 128:] == 0] == 0)
    assert np.all((features['tof'][:, :128] >= 0) & (features['tof'][:, :128] <= 1))
    train = np.flatnonzero(role == 'TRAIN_ONLY')
    schedule = np.random.default_rng(59).choice(train, size=(300, 128), replace=True)
    np.testing.assert_array_equal(np.load(run/'schedule.npy', allow_pickle=False), schedule)
    np.testing.assert_array_equal(np.load(source/'run-v1/schedule.npy', allow_pickle=False), schedule)
    assert np.all(role[schedule] == 'TRAIN_ONLY')
    initial = torch.load(run/'initial.pt', map_location='cpu', weights_only=True)
    expected_weights = reference_weights()
    assert set(initial) == set(expected_weights) | set(BUFFERS)
    for name, value in expected_weights.items():
        torch.testing.assert_close(initial[name], value, rtol=0, atol=0)
    assert state_sha(initial) == receipt['initial_state_sha256']
    geom_result = verify_geometry(initial)
    cached_geometry = load_npz(cache/'geometry.npz')
    for name in ('valid', 'xyz', 'grid'):
        np.testing.assert_array_equal(cached_geometry[name], initial[name].numpy())
    permutation = independent_permutation(initial['valid'].numpy())
    np.testing.assert_array_equal(np.load(run/'permutation.npy', allow_pickle=False), permutation)
    assert {name: receipt[name] for name in ('parameters', 'initial_seed', 'schedule_seed', 'permutation_seed',
        'steps_per_arm', 'batch_size', 'learning_rate', 'weight_decay')} == dict(parameters=131580, initial_seed=53,
        schedule_seed=59, permutation_seed=83, steps_per_arm=300, batch_size=128, learning_rate=.001, weight_decay=.0001)
    points = np.load(cache/'points.npy', mmap_mode='r', allow_pickle=False)
    assert points.shape == (5000, 12, 27, 64) and points.dtype == np.float32
    mask = initial['valid'].numpy()[None, ..., None]
    mean_error = 0.
    for begin in range(0, 5000, 128):
        chunk = np.asarray(points[begin:begin+128])
        assert np.isfinite(chunk).all()
        means = (chunk * mask).sum(2) / np.maximum(mask.sum(2), 1)
        mean_error = max(mean_error, float(np.abs(means.reshape(len(chunk), 768) - features['visual'][begin:begin+128, :768]).max()))
    assert mean_error < 1e-4
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        assert torch.cuda.is_available()
    reconstructed, all_metrics = {}, {}
    for arm in ARMS:
        arm_receipt = receipt['arms'][arm]
        assert arm_receipt['steps'] == 300 and len(arm_receipt['train_losses']) == 300
        assert np.isfinite(arm_receipt['train_losses']).all()
        assert arm_receipt['initial_state_sha256'] == receipt['initial_state_sha256']
        assert sha(run/f'{arm}.pt') == arm_receipt['checkpoint_sha256']
        state = torch.load(run/f'{arm}.pt', map_location='cpu', weights_only=True)
        assert set(state) == set(initial) and state_sha(state) == arm_receipt['final_state_sha256']
        for name in BUFFERS:
            torch.testing.assert_close(state[name], initial[name], rtol=0, atol=0)
        state = {key: tensor.to(device) for key, tensor in state.items()}
        logits = []
        with torch.inference_mode():
            for begin in range(0, 5000, 128):
                batch = slice(begin, begin+128)
                inputs = [torch.as_tensor(np.array(value[batch]), device=device) for value in
                          (points, features['visual'], features['tof'])]
                logits.append(reference_forward(state, *inputs, arm).cpu().numpy())
        logits = np.concatenate(logits)
        np.testing.assert_allclose(logits, prediction[arm+'_logits'], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(logits >= 0, prediction[arm+'_flags'])
        assert prediction[arm+'_flags'].dtype == np.bool_
        reconstructed[arm] = dict(rows=5000, max_logit_error=float(np.abs(logits-prediction[arm+'_logits']).max()))
        all_metrics[arm] = {}
        for split in ROLES:
            selected = role == split
            value = scalar_metrics(prediction[arm+'_flags'][selected], features['truth'][selected])
            assert value == reported[arm][split]
            all_metrics[arm][split] = value
        if arm == 'LOCAL_FUSION':
            eval_rows = np.flatnonzero(role == 'EVAL_ONLY')
            outputs = []
            with torch.inference_mode():
                for begin in range(0, 1500, 128):
                    rows = eval_rows[begin:begin+128]
                    sampled = np.array(points[rows])[:, np.arange(12)[:, None], permutation]
                    inputs = [torch.as_tensor(value, device=device) for value in
                              (sampled, features['visual'][rows], features['tof'][rows])]
                    outputs.append(reference_forward(state, *inputs, arm).cpu().numpy())
            permuted = np.concatenate(outputs)
            np.testing.assert_allclose(permuted, prediction['LOCAL_PERMUTED_logits'], rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(permuted >= 0, prediction['LOCAL_PERMUTED_flags'])
            reconstructed['LOCAL_PERMUTED'] = dict(rows=1500,
                max_logit_error=float(np.abs(permuted-prediction['LOCAL_PERMUTED_logits']).max()))
            all_metrics['LOCAL_PERMUTED'] = scalar_metrics(permuted >= 0, features['truth'][eval_rows])
        del state
    # Protect all durable inputs, including the large recache, across the audit.
    for path, expected in expected_hashes.items():
        assert sha(path) == expected, str(path)
    del points
    return dict(status='PASS', backend=device, geometry=geom_result, trainable_parameters=131580,
        schedule_rows=300, batch_size=128, schedule_equal_to_MZ1=True, initial_parameters_bitwise_equal=True,
        averaged_recache_max_error=mean_error, original_alert_parity=5000,
        replay=reconstructed, independent_scalar_metric_groups=9, metrics=all_metrics,
        source_contract=source_contract(), training_steps=0, source_images_read=0, native_depth_reads=0,
        optimizer_history='Not rerun. Fixed settings, TRAIN_ONLY schedule, initial state, and 300 recorded losses verified.',
        input_hashes={str(path): digest for path, digest in expected_hashes.items()},
        input_hashes_unchanged=True)


def scalar_compare(candidate, baseline, truth, ids):
    result = dict(n=len(truth), gained=0, lost=0, both_correct=0, both_wrong=0,
                  gained_frame_ids=[], lost_frame_ids=[])
    result['errors'] = {kind: dict(introduced=0, removed=0, retained=0, candidate=0, baseline=0)
                        for kind in ('wrong_far', 'cross_body')}
    for c, b, t, frame in zip(candidate.tolist(), baseline.tolist(), truth.tolist(), ids.tolist()):
        cc, bc = c == t, b == t
        label = 'both_correct' if cc and bc else 'gained' if cc else 'lost' if bc else 'both_wrong'
        result[label] += 1
        if label in ('gained', 'lost'):
            result[label + '_frame_ids'].append(frame)
        def row_errors(p):
            return dict(wrong_far=[p[k+1] and t[k] and not t[k+1] for k in (0, 2)],
                cross_body=[any(t[:2]) and not any(t[2:]) and any(p[2:]),
                            any(t[2:]) and not any(t[:2]) and any(p[:2])])
        ce, be = row_errors(c), row_errors(b)
        for kind, counts in result['errors'].items():
            for c_error, b_error in zip(ce[kind], be[kind]):
                counts['introduced'] += int(c_error and not b_error)
                counts['removed'] += int(b_error and not c_error)
                counts['retained'] += int(c_error and b_error)
                counts['candidate'] += int(c_error)
                counts['baseline'] += int(b_error)
    return result


def audit_analysis(run):
    """Audit metadata stratification and paired results without opening sources."""
    run = Path(run).resolve()
    receipt, analysis = read(run/'receipt.json'), read(run/'analysis.json')
    old_root = WORK/'mz1-tiny-fusion-20260910/run-v1'
    ensemble_root = WORK/'mz5-fixed-ensemble-20260910/run-v1'
    index = WORK/'body-query-5000-20260909/dataset-v1/index.json'
    hashes = {run/'predictions.npz': receipt['predictions_sha256'],
        old_root/'predictions.npz': read(old_root/'receipt.json')['predictions_sha256'],
        ensemble_root/'predictions.npz': read(ensemble_root/'receipt.json')['output_sha256']['predictions.npz'],
        index: receipt['source_index_sha256'], run/'analysis.json': sha(run/'analysis.json')}
    for path, expected in hashes.items():
        assert sha(path) == expected, str(path)
    assert analysis['prediction_sha256'] == receipt['predictions_sha256']
    assert analysis['source_index_sha256'] == receipt['source_index_sha256']
    arrays, old, ensemble = (load_npz(path) for path in
        (run/'predictions.npz', old_root/'predictions.npz', ensemble_root/'predictions.npz'))
    rows = [row for row in read(index)['frames'] if row['source_role'] == 'EVAL_ONLY']
    assert len(rows) == 1500
    ids = np.asarray([row['frame_id'] for row in rows])
    ev = arrays['role'] == 'EVAL_ONLY'
    truth = arrays['truth'][ev]
    np.testing.assert_array_equal(truth, [[sum(row['counts'][start:start+3]) >= 3
                                          for start in (0, 3, 6, 9)] for row in rows])
    for name in ('role', 'truth', 'original_alerts', 'joint'):
        np.testing.assert_array_equal(arrays[name], old[name])
    for name, expected in (('truth', truth), ('frame_id', ids), ('original_alerts', arrays['original_alerts'][ev])):
        np.testing.assert_array_equal(ensemble[name], expected)
    predictions = {arm: arrays[arm+'_flags'][ev] for arm in ARMS}
    predictions.update(LOCAL_PERMUTED=arrays['LOCAL_PERMUTED_flags'], MZ1_FUSION=old['FUSION_flags'][ev],
        MZ1_TOF=old['TOF_ONLY_flags'][ev], MZ1_RGB=old['RGB_ONLY_flags'][ev],
        ENSEMBLE=ensemble['ENSEMBLE_flags'], MZ0=ensemble['MZ0_flags'])
    scores = {arm: scalar_metrics(flags, truth) for arm, flags in predictions.items()}
    assert scores == analysis['scores']
    strata = {key: np.asarray([row[key] for row in rows]) for key in ('condition', 'region_id', 'family')}
    strata['condition_range'] = np.asarray([row['condition']+'/'+row['declared_range'] for row in rows])
    strata['negative'] = np.asarray(['POSITIVE' if any(row) else 'NEGATIVE' for row in truth])
    stratum_metrics, stratum_groups = {}, 0
    for key, values in strata.items():
        stratum_metrics[key] = {}
        for value in sorted(set(values)):
            mask = values == value
            result = {arm: scalar_metrics(flags[mask], truth[mask]) for arm, flags in predictions.items()}
            assert result == analysis['strata_scores'][key][value]
            stratum_metrics[key][value] = result
            stratum_groups += len(result)
    paired = {}
    assert set(analysis['comparisons']) == {
        'LOCAL_FUSION_vs_' + baseline for baseline in
        ('POOLED_FUSION', 'LOCAL_RGB_ONLY', 'ENSEMBLE', 'MZ1_FUSION', 'MZ0')
    } | {'LOCAL_PERMUTED_vs_LOCAL_FUSION'}
    for pair, expected in analysis['comparisons'].items():
        candidate, baseline = pair.split('_vs_')
        result = scalar_compare(predictions[candidate], predictions[baseline], truth, ids)
        assert result == expected['overall']
        paired[pair] = result
        for key, values in strata.items():
            for value in sorted(set(values)):
                mask = values == value
                assert scalar_compare(predictions[candidate][mask], predictions[baseline][mask], truth[mask], ids[mask]) == expected['strata'][key][value]
    local, pooled, rgb = [scores[arm] for arm in ('LOCAL_FUSION', 'POOLED_FUSION', 'LOCAL_RGB_ONLY')]
    head_near = stratum_metrics['condition_range']['HEAD_ONLY/near']
    cross = lambda values: values['body_to_head']['numerator'] + values['head_to_body']['numerator']
    criteria = dict(exact_above_pooled=local['spatial_exact']['numerator'] > pooled['spatial_exact']['numerator'],
        exact_above_rgb=local['spatial_exact']['numerator'] > rgb['spatial_exact']['numerator'],
        head_only_near_above_pooled=head_near['LOCAL_FUSION']['spatial_exact']['numerator'] > head_near['POOLED_FUSION']['spatial_exact']['numerator'],
        wrong_far_not_worse_than_pooled=local['wrong_far']['numerator'] <= pooled['wrong_far']['numerator'],
        cross_body_not_worse_than_pooled=cross(local) <= cross(pooled))
    criteria['all'] = all(criteria.values())
    assert criteria == analysis['predeclared_necessary_criterion']
    for path, expected in hashes.items():
        assert sha(path) == expected, str(path)
    return dict(status='PASS', backend='CPU', analysis_rows=1500, original_alert_parity=5000,
        scalar_overall_metric_groups=len(scores), scalar_stratum_metric_groups=stratum_groups,
        paired_comparisons=paired, all_paired_strata_verified=True, predeclared_necessary_criterion=criteria,
        summary={arm: dict(exact=values['spatial_exact']['numerator'], wrong_far=values['wrong_far']['numerator'],
            cross_body=cross(values), head_only_near=head_near[arm]['spatial_exact']['numerator']) for arm, values in scores.items()},
        input_hashes={str(path): digest for path, digest in hashes.items()}, input_hashes_unchanged=True,
        training_steps=0, model_forward_passes=0, source_images_read=0, native_depth_reads=0,
        scope='Consumed Development metadata and stored predictions only; zeros do not assert CLEAR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.self_check and (args.run or args.analysis_only)) or (not args.self_check and not args.run) or (args.run and not args.analysis_only and not args.cache):
        parser.error('Choose --self-check, --run with --cache, or --analysis-only with --run')
    output = args.output.resolve()
    allowed = (WORK/'mz5-spatial-fusion-20260910').resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('Use a fresh child of the task artifact directory')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'audit-source.py')
    started = time.perf_counter()
    write(output/'start-receipt.json', dict(status='STARTED', utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(__file__), mode='CPU_SYNTHETIC' if args.self_check else 'CPU_ANALYSIS' if args.analysis_only else 'SAVED_RUN_REPLAY'))
    try:
        result = self_check() if args.self_check else audit_analysis(args.run) if args.analysis_only else audit_run(args.run, args.cache, args.device)
        result.update(seconds=time.perf_counter()-started, audit_source_sha256=sha(__file__))
        write(output/'audit.json', result)
        print(json.dumps({key: result[key] for key in ('status', 'backend', 'seconds', 'training_steps')}))
    except Exception as error:
        write(output/'failure.json', dict(status='FAILED', error=repr(error), audit_source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
