"""Describe frozen M3/S on genuine 16-frame synthetic-CNH sequences.

Input NPZ: hist[N,16,8,8,16], ambient[N,16,8,8] (or [16,8,8]),
sensor[N,16,4,4] sensor-to-world poses, public_query[N,16,4,4]
current sensor-to-query transforms, timestamps[N,16]. No labels are read.
The frozen two HEAD/BODY corridor outputs are NOT the common 27-query grid.
Never repeat sparse frames or invent motion to satisfy the temporal interface.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / 'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json'


def prepare(scene, destination):
    """First chronological complete window; official camera-to-world convention."""
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from sync_rgb_tof_pilot_dev import render_registered_depth
    started = time.monotonic()
    paths = sorted((scene / 'highres_depth').glob('*.png'))
    times = np.array([float(p.stem.rsplit('_', 1)[1]) for p in paths])
    order = np.argsort(times)
    times, paths = times[order], [paths[j] for j in order]
    selected = None
    for start in times:
        targets = start + .2 * np.arange(16)
        indices = abs(times[:, None] - targets).argmin(0)
        values = times[indices]
        if (len(set(indices)) == 16 and abs(values-targets).max() <= .051
                and (np.diff(values) >= .15).all() and (np.diff(values) <= .25).all()):
            selected = indices
            break
    if selected is None:
        raise ValueError('No complete 16-frame nominal 5Hz window')
    trajectory = np.loadtxt(scene / 'lowres_wide.traj')
    hist, ambient, poses, delta = [], [], [], []
    for fi, idx in enumerate(selected):
        path = paths[idx]
        with Image.open(path) as image:
            depth = np.asarray(image).astype(np.float64) / 1000
        w, h, fx, fy, cx, cy = np.loadtxt(scene / 'wide_intrinsics' / (path.stem + '.pincam'))
        sx, sy = depth.shape[1] / w, depth.shape[0] / h
        intrinsic = np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1.]])
        rendered = render_registered_depth(depth, intrinsic, 2026101050 + fi)
        hist.append(rendered['hist'])
        ambient.append(rendered['ambient'])
        ti = abs(trajectory[:, 0]-times[idx]).argmin()
        delta.append(float(trajectory[ti, 0]-times[idx]))
        if abs(delta[-1]) > .051:
            raise ValueError('Trajectory mismatch exceeds51ms')
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Rotation.from_rotvec(trajectory[ti, 1:4]).as_matrix()
        extrinsic[:3, 3] = trajectory[ti, 4:7]
        # Identical convention to official TrajStringToMatrix, CV right/down/forward.
        poses.append(np.linalg.inv(extrinsic))
    payload = dict(hist=np.array(hist)[None], ambient=np.array(ambient)[None],
        sensor=np.array(poses)[None], public_query=np.broadcast_to(np.eye(4), (1,16,4,4)).copy(),
        timestamps=times[selected][None])
    validate(payload)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(destination, **payload)
    destination.with_suffix('.receipt.json').write_text(json.dumps(dict(
        seconds=time.monotonic()-started, scene=str(scene), selected_timestamps=times[selected].tolist(),
        actual_step_seconds=np.diff(times[selected]).tolist(), trajectory_delta_seconds=delta,
        selection='First chronological complete16-frame window, no score selection',
        public_query='Identity: sensor-relative original two corridors, no body/world height claim',
        geometry='FARO-rendered registered highres2.5D; constant rho=.3',
        rng_seeds=list(range(2026101050,2026101066))), indent=2)+'\n', encoding='utf8')


def validate(data):
    hist = data['hist']
    n = len(hist)
    if hist.shape != (n, 16, 8, 8, 16):
        raise ValueError('Frozen temporal contract requires 16 real sampled frames per clip')
    for name in ('sensor', 'public_query'):
        if data[name].shape != (n, 16, 4, 4) or not np.isfinite(data[name]).all():
            raise ValueError(name + ' must contain finite per-clip 4x4 transforms')
    times = data['timestamps']
    if times.shape != (n, 16) or not np.isfinite(times).all():
        raise ValueError('Missing actual timestamps')
    dt = np.diff(times, axis=1)
    if not ((dt >= .15) & (dt <= .25)).all():
        raise ValueError('Not a contiguous nominal 5Hz sequence; no frame repetition allowed')
    if data['ambient'].shape not in ((16, 8, 8), (n, 16, 8, 8)):
        raise ValueError('ambient shape mismatch')
    return n


def run(source, output, budget_seconds):
    import cnh_counterfactual_common_dev as C
    import cnh_counterfactual_eval_dev as E
    import cnh_graded_corridor_eval_dev as Corr
    import cnh_frozen_e2e_tof_20261010 as F
    import cnh_task_cost_train_20261010 as T
    start = time.monotonic()
    def check():
        if time.monotonic() - start >= budget_seconds:
            raise TimeoutError('Frozen description command-wall budget reached')
    with np.load(source, allow_pickle=False) as z:
        data = {key: z[key] for key in ('hist', 'ambient', 'sensor', 'public_query', 'timestamps')}
    n = validate(data)
    manifest = C.read(MANIFEST)
    bindings = (manifest['models'] + manifest['features']['implementation']
                + manifest['strong_source']['frozen_dependencies'])
    for entry in bindings:
        if C.sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen dependency hash mismatch: ' + entry['path'])
    tau = manifest['threshold']['tau']
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for i in range(n):
        check()
        clip = output / f'clip{i:03d}'
        folder = clip / 'data' / 'pilot'
        folder.mkdir(parents=True)
        np.save(folder / 'hist.npy', data['hist'][i:i+1, None])
        ambient = data['ambient'][i] if data['ambient'].ndim == 4 else data['ambient']
        np.save(folder / 'ambient.npy', ambient)
        np.savez_compressed(folder / 'geometry.npz', sensor=data['sensor'][i], public_query=data['public_query'][i])
        C.save(clip / 'PLAN.json', dict(input=str(source), input_sha256=C.sha(source), clip_index=i,
                                      labels=False, query_mapping='two native corridors only'))
        C.save(clip / 'render_receipt.json', dict(source='External synthetic CNH, already rendered',
            outputs_sha256={f'data/pilot/{name}': C.sha(folder / name) for name in ('hist.npy', 'ambient.npy')}))
        F.S = SimpleNamespace(SPLITS=('pilot',), K=1)
        F.scientific(clip, {'pilot': [{}]}, check)
        with np.load(folder / 'scores.npz') as raw:
            ordinary = raw['ordinary_raw']
            current = raw['current'].astype(float)
            valid = raw['current_valid'] & np.isfinite(current)
            m3 = E.smooth(raw['m3_raw'])
            old5 = E.old_fusion(m3, E.smooth(raw['local_raw']))
        spatial = np.concatenate((np.where(valid, current, np.nan), (~valid).astype(float)), -1)
        cuts = C.read(F.THRESHOLDS)
        features = np.array([np.concatenate((Corr.build_score_features(
            ordinary[si], E.smooth(ordinary[si]), cuts[str(seed)]['single']), spatial), -1)
            for si, seed in enumerate(T.SEEDS)], np.float32)
        scores = T.predict_s(ROOT / 'artifacts.local/work/cnh-task-cost-retrain-dev-20261010', features)
        mean = scores.mean(0)
        grade = np.where(old5, 2, np.where(mean >= tau, 1, 0)).astype(np.int8)
        results.append(dict(m3_smoothed=m3[0, 0], old5=old5[0, 0], seed_scores=scores[:, 0, 0],
                            s_mean=mean[0, 0], s_grade=grade[0, 0]))
        check()
    merged = {key: np.stack([r[key] for r in results]) for key in results[0]}
    np.savez_compressed(output / 'frozen_description.npz', **merged, timestamps=data['timestamps'][:, 3:])
    C.save(output / 'receipt.json', dict(status='DESCRIPTIVE_ONLY', seconds=time.monotonic()-start,
        input_sha256=C.sha(source), manifest_sha256=C.sha(MANIFEST), threshold=tau,
        query_unit='HEAD/BODY frozen corridors; no mapping to 27 point queries',
        trajectory_domain='ARKit handheld scan differs from co-directional walking simulation',
        clips=n, output_frames_per_clip=13, old5_strong_slots=merged['old5'].sum(axis=(0, 1)).tolist(),
        m3_positive_slots=(merged['m3_smoothed'] >= E.M3_THETA).sum(axis=(0, 1)).tolist(),
        s_light_slots=(merged['s_grade'] == 1).sum(axis=(0, 1)).tolist(),
        training=0, protected_access=0, labels_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--budget-seconds', type=float, default=120)
    parser.add_argument('--prepare', action='store_true', help='Source is scene directory; output is input NPZ')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source, args.output)
    else:
        run(args.source, args.output, args.budget_seconds)
