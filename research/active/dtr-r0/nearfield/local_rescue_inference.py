"""Frozen public RGB/ToF prediction on new-dimension composite rescue confirmation.

Only RGB and ToF reach the unchanged feature extractor. Metadata validates the
source factorial; labels/native arrays are never opened. No fitting or selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from inherit_spatial_model import ARMS, FEATURE_SIZE, QUERIES, extract
from local_transfer_inference import frozen_bundle, FROZEN_MODELS, FROZEN_SELECTION, A_THRESHOLD
from query_occupancy_data import read, write, sha, new_stage_directory
from tof_corridor_calibration import score_frame, decide

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

SHAPES = dict(body_protruding_plane='L_plate', body_suspended_solid='depth_step',
              head_hanging_plane='inverted_L_plate', head_horizontal='T_bar')
TRAJECTORIES = ('approach_return', 'approach_dwell_return')
RELATIONS = ('INSIDE', 'BOUNDARY', 'OUTSIDE')


def validate_identities(ids, old_groups):
    required = {'index', 'id', 'split', 'clip_id', 'frame_in_clip', 'time_s', 'base_group_id',
                'type_id', 'layer', 'layout_relation', 'trajectory', 'shape', 'size_level'}
    if len(ids) != 576 or any(not required.issubset(r) for r in ids) or len({r['id'] for r in ids}) != 576:
        raise ValueError('Need 576 unique frames with complete public identity fields')
    if any({'appearance', 'appearance_pair_id', 'geometry_pair_id', 'pair_id'} & r.keys() for r in ids):
        raise ValueError('Appearance/pair identities do not belong to this source')
    groups = {r['base_group_id'] for r in ids}
    if len(groups) != 8 or groups & old_groups or any(not g.startswith('local_rescue_fresh_') for g in groups):
        raise ValueError('Need eight new local_rescue_fresh_ groups, disjoint from all old model groups')
    if {r['type_id'] for r in ids} != set(SHAPES) or {r['split'] for r in ids} != {'evaluation'}:
        raise ValueError('Expected evaluation-only source with all four original families')
    clips = {}
    for i, row in enumerate(ids):
        if type(row['index']) is not int or row['index'] != i:
            raise ValueError('Public identity index mismatch')
        if row['shape'] != SHAPES[row['type_id']] or row['size_level'] not in ('small', 'large'):
            raise ValueError('Unexpected composite shape or size level')
        if row['layer'] != ('BODY' if row['type_id'].startswith('body_') else 'HEAD'):
            raise ValueError('Family/layer identity mismatch')
        if row['trajectory'] not in TRAJECTORIES or row['layout_relation'] not in RELATIONS:
            raise ValueError('Unexpected trajectory or lateral relation')
        clips.setdefault(row['clip_id'], []).append(row)
    if len(clips) != 48:
        raise ValueError('Expected 48 complete clips')
    stable = ('base_group_id', 'type_id', 'layer', 'layout_relation', 'trajectory', 'shape', 'size_level')
    for clip in clips.values():
        ordered = sorted(clip, key=lambda r: r['frame_in_clip'])
        if [r['frame_in_clip'] for r in ordered] != list(range(12)):
            raise ValueError('Each clip needs twelve contiguous frames')
        if any(not np.isfinite(r['time_s']) or abs(r['time_s']-.2*r['frame_in_clip']) > 1e-8 for r in ordered):
            raise ValueError('Unexpected posed-sample clock')
        if any(len({r[k] for r in clip}) != 1 for k in stable):
            raise ValueError('Clip metadata must remain constant')
    for group in groups:
        part = [r for r in ids if r['base_group_id'] == group]
        if len(part) != 72 or len({(r['type_id'], r['shape'], r['size_level']) for r in part}) != 1:
            raise ValueError('Each base group must preserve one family/shape/size')
        if {(r['trajectory'], r['layout_relation']) for r in part} != {(t, r) for t in TRAJECTORIES for r in RELATIONS}:
            raise ValueError('Incomplete trajectory/relation factorial')
    for family in SHAPES:
        pairs = {(r['base_group_id'], r['size_level']) for r in ids if r['type_id'] == family}
        if len(pairs) != 2 or {size for _, size in pairs} != {'small', 'large'}:
            raise ValueError('Each family needs one small and one large group')
    return dict(status='PASS', frames=576, clips=48, groups=sorted(groups), shapes=SHAPES,
                trajectories=list(TRAJECTORIES), size_levels=['small', 'large'], split='evaluation',
                old_model_groups_disjoint=True, previous_transfer_prefix_rejected=True,
                limitation='Identity validation only; source geometry disjointness belongs to the source check')


def require_governed(result):
    artifact = (ROOT/'artifacts.local').resolve()
    path = Path(os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL', '')).resolve()
    if not result.resolve().is_relative_to(artifact) or not path.is_relative_to(artifact) or not path.is_file():
        raise ValueError('Use active governed research-ue with canonical artifact output')
    journal = read(path)
    outputs = [(Path(r['path']) if Path(r['path']).is_absolute() else artifact/r['path']).resolve()
               for r in journal.get('outputs', [])]
    if journal.get('schema') != 'blindassist-asset-run-journal-v1' or journal.get('state') != 'running' or result.resolve() not in outputs:
        raise ValueError('Active governed run must own the exact result')
    return dict(journal=str(path), run_id=journal['id'])


def extract_public(rgb, tof):
    return extract(rgb, tof)


def predict(args):
    start, governance = time.perf_counter(), require_governed(args.result)
    old, cutoffs = frozen_bundle(args.frozen_run)
    manifest = read(args.materialization)
    if not np.allclose(manifest['queries'], QUERIES, rtol=0, atol=3e-8):
        raise ValueError('Original fixed queries changed')
    hashes = {'observations/'+name: sha(args.observations/name) for name in ('rgb.npy', 'tof.npy', 'identities.json')}
    if any(digest != manifest['hashes'][name] for name, digest in hashes.items()):
        raise ValueError('Public input identity mismatch')
    ids = read(args.observations/'identities.json')
    old_sets = read(args.frozen_run/'feature-seal.json')['groups']
    identity_check = validate_identities(ids, set().union(*map(set, old_sets.values())))
    rgb, tof = (np.load(args.observations/name, mmap_mode='r', allow_pickle=False) for name in ('rgb.npy', 'tof.npy'))
    if rgb.shape != (576, 3, 180, 320) or rgb.dtype != np.uint8 or tof.shape != (576, 64, 6) or tof.dtype != np.float32:
        raise ValueError('Unexpected public RGB/ToF tensor contract')
    out = args.result.parent.resolve(); new_stage_directory(out)
    own = [Path(__file__), HERE/'local_transfer_inference.py', HERE/'local_transfer_metrics.py', ROOT/'tools/research_backend.py']
    sources = {**old['sources'], **{p.relative_to(ROOT).as_posix(): sha(p) for p in own}}
    for name, digest in sources.items():
        dest = out/'source-snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        if sha(dest) != digest:
            raise ValueError('Frozen source snapshot mismatch')
    freeze = dict(source_hashes=sources, input_hashes=hashes, governance=governance,
        materialization_sha256=sha(args.materialization), protocol_sha256=sha(args.protocol),
        expected_evaluation_labels_sha256=manifest['hashes']['labels/evaluation.npz'],
        frozen_model_hashes=FROZEN_MODELS, frozen_selection_sha256=FROZEN_SELECTION,
        frozen_model_seal_sha256=sha(args.frozen_run/'model-seal.json'), thresholds=cutoffs,
        A_threshold=A_THRESHOLD, query_boxes=QUERIES.tolist(), groups=identity_check['groups'],
        feature_width=FEATURE_SIZE, evaluation_labels_opened=False, fits=0, cutoff_selections=0)
    write(out/'freeze.json', freeze); write(out/'public-identity-checks.json', identity_check)
    shutil.copyfile(args.frozen_run/'selection.json', out/'selection.json')
    select_backend('model-inference', cpu=BackendCandidate('frozen-hgb-rescue-fresh-cpu', 'cpu',
        lambda: extract_public(rgb[0], tof[0]), lambda _: DeviceObservation('cpu', platform.processor() or 'host CPU',
        'NumPy '+np.__version__+' sklearn '+sklearn.__version__)), cpu_reason='TASK_NOT_GPU_SUITABLE',
        record_path=out/'backend.json', capabilities=dict(reason='Unchanged tabular HGB and interval/RGB statistics',
        python_executable=sys.executable, threads=4, fitting=False))
    x = np.empty((576, 6, 2, FEATURE_SIZE), np.float32)
    baseline, feature_times, baseline_times = [], [], []
    for i in range(576):
        tick = time.perf_counter(); x[i] = extract_public(rgb[i], tof[i]); feature_times.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        values = np.where(tof[i, :, 1] == 1, tof[i, :, 0]*8, np.nan)
        boxes = np.rint(tof[i, :, 2:]*[192, 256, 192, 256]).astype(int)
        base = decide(score_frame(boxes, values), A_THRESHOLD)
        if 'baseline' in ids[i]:
            prior = ids[i]['baseline']
            if any(base[k] != prior[k] for k in ('alert', 'unknown', 'ambiguous', 'valid_zones', 'definite_zones')) or abs(base['score']-prior['score']) > 1e-12:
                raise ValueError('Materialization A parity failed')
        baseline.append(base); baseline_times.append(time.perf_counter()-tick)
    np.savez_compressed(out/'features.npz', raw=x[:, :, 0], local=x[:, :, 1])
    probabilities, times = {}, {}
    with threadpool_limits(limits=4):
        for j, arm in enumerate(ARMS):
            with (args.frozen_run/(arm+'.pkl')).open('rb') as stream:
                model = pickle.load(stream)
            if list(model.classes_) != [0, 1] or model.n_features_in_ != FEATURE_SIZE:
                raise ValueError('Unexpected frozen classifier schema')
            tick = time.perf_counter()
            probabilities[arm] = model.predict_proba(x[:, :, j].reshape(-1, FEATURE_SIZE))[:, 1].reshape(576, 6)
            times[arm] = time.perf_counter()-tick
    np.savez_compressed(out/'probabilities.npz', indices=np.arange(576), **probabilities)
    write(out/'baseline.json', baseline); write(out/'identities.json', ids)
    costs = dict(frames=576, fits=0, cutoff_selections=0, feature_seconds=sum(feature_times),
        feature_p50_s=float(np.median(feature_times)), feature_p95_s=float(np.quantile(feature_times, .95)),
        baseline_seconds=sum(baseline_times), batch_inference_seconds=times, total_seconds=time.perf_counter()-start,
        scope='Host CPU including input verification; excludes capture/PNG decode; not endpoint latency')
    write(out/'costs.json', costs)
    for key, digest in hashes.items():
        if sha(args.observations/Path(key).name) != digest:
            raise ValueError('Public input changed during prediction')
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source changed during prediction')
    seal = dict(status='PASS', frames=576, queries=3456, evaluation_labels_opened=False,
        fits=0, cutoff_selections=0, thresholds=cutoffs,
        hashes={name: sha(out/name) for name in ('freeze.json', 'features.npz', 'probabilities.npz',
            'baseline.json', 'identities.json', 'selection.json', 'costs.json', 'public-identity-checks.json', 'backend.json')})
    write(out/'prediction-seal.json', seal)
    answer = dict(status='PASS', phase='SEALED_PREDICTIONS', frames=576, fits=0,
        prediction_seal_sha256=sha(out/'prediction-seal.json'), thresholds=cutoffs,
        evaluation_labels_opened=False, resource_state='CPU command completed; no worker or capture created')
    write(args.result, answer); print(json.dumps(answer), flush=True)
    return answer


def synthetic_checks():
    ids = []
    for family, shape in SHAPES.items():
        for size in ('small', 'large'):
            group = 'local_rescue_fresh_'+family+'_'+size
            for trajectory in TRAJECTORIES:
                for relation in RELATIONS:
                    clip = group+'_'+trajectory+'_'+relation
                    for frame in range(12):
                        ids.append(dict(index=len(ids), id=clip+str(frame), split='evaluation', clip_id=clip,
                            frame_in_clip=frame, time_s=frame*.2, base_group_id=group, type_id=family,
                            shape=shape, size_level=size, trajectory=trajectory, layout_relation=relation,
                            layer='BODY' if family.startswith('body_') else 'HEAD'))
    assert validate_identities(ids, {'old_group'})['frames'] == 576
    for key, value in (('index', 9), ('split', 'train'), ('shape', 'cube'), ('trajectory', 'old_path'),
                       ('size_level', 'medium'), ('appearance', 'base'), ('time_s', .01)):
        wrong = copy.deepcopy(ids); wrong[0][key] = value
        try:
            validate_identities(wrong, set())
        except ValueError:
            continue
        raise AssertionError('Invalid metadata accepted: '+key)
    try:
        validate_identities(ids, {ids[0]['base_group_id']})
    except ValueError:
        pass
    else:
        raise AssertionError('Old group admitted')
    from unittest.mock import patch
    image, sensor = object(), object()
    with patch(__name__+'.extract', return_value='sentinel') as probe:
        assert extract_public(image, sensor) == 'sentinel'
        probe.assert_called_once_with(image, sensor)
    return dict(status='PASS', cases=10, scientific_inputs_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest='phase', required=True)
    command = commands.add_parser('predict')
    for name in ('observations', 'materialization', 'frozen-run', 'protocol', 'result'):
        command.add_argument('--'+name, type=Path, required=True)
    predict(parser.parse_args())
