"""Original Nymeria sample structure/clock/offline-registration audit.

No BodyDataProvider corrections, inference, slow-walk coverage or benefit claim.
XSens segment6 Head/0 Pelvis WXYZ; MPS poses XYZW/device tracking clock.
Independent NumPy/SciPy implementation of official stride2 hand-eye equations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    x = np.asarray(values, float).ravel(); good = x[np.isfinite(x)]
    return dict(count=len(x), nonfinite=int(len(x)-len(good)),
        min=float(good.min()) if len(good) else None,
        median=float(np.median(good)) if len(good) else None,
        p95=float(np.quantile(good, .95)) if len(good) else None,
        max=float(good.max()) if len(good) else None)


def ns_from_us(t):
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.integer):
        return t.astype(np.int64)*1000
    return np.rint(t*1000).astype(np.int64)


def raw_contiguous_runs(t_us):
    """Partition original row order at every raw240Hz anomaly; never repair."""
    t = np.asarray(t_us)
    dt = np.diff(t.astype(float))
    bad = ~np.isfinite(dt) | (np.abs(dt-1e6/240.) > 1000)
    bounds = np.r_[0, np.flatnonzero(bad)+1, len(t)]
    result = []
    for index, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = t[begin:end]
        good = bool(np.isfinite(part).all() and (np.diff(part) > 0).all())
        result.append(dict(run_index=index, start_frame=int(begin), end_frame_exclusive=int(end),
            frames=int(end-begin), timestamp_finite_and_monotonic=good,
            duration_seconds=float((part[-1]-part[0])/1e6) if good and len(part) else None,
            raw_timestamp_start_us=int(part[0]) if len(part) and np.isfinite(part[0]) else None,
            raw_timestamp_end_us=int(part[-1]) if len(part) and np.isfinite(part[-1]) else None,
            preceding_anomalous_dt_us=float(dt[begin-1]) if begin and np.isfinite(dt[begin-1]) else None,
            status='NOT_RUN', reason='Only largest usable raw continuous segment is eligible for offline registration'))
    return result


def inverse(T):
    out = np.zeros_like(T); out[..., :3, :3] = T[..., :3, :3].swapaxes(-1, -2)
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', out[..., :3, :3], T[..., :3, 3])
    out[..., 3, 3] = 1
    return out


def poses(q, p, order):
    q = np.asarray(q, float); p = np.asarray(p, float)
    if order == 'WXYZ':
        q = q[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or not np.isfinite(p).all() or (norms < .1).any():
        raise ValueError('Invalid raw quaternion/position; no previous-frame repair applied')
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T[:, :3, 3] = p
    return T


def nearest_index(t, query):
    """Sorted timestamps, closest match; exact distance tie goes right."""
    t = np.asarray(t)
    if len(t) < 1 or (np.diff(t) <= 0).any():
        raise ValueError('Nearest timeline must be strictly increasing')
    r = np.searchsorted(t, query).clip(0, len(t)-1)
    l = (r-1).clip(0, len(t)-1)
    return np.where(np.abs(t[l]-query) < np.abs(t[r]-query), l, r)


def map_clock(provider, timecode_ns, check=lambda: None):
    device = np.empty(len(timecode_ns), np.int64); back = device.copy()
    for i, t in enumerate(timecode_ns):
        if i % 512 == 0:
            check()
        device[i] = provider.convert_from_timecode_to_device_time_ns(int(t))
        back[i] = -1 if device[i] < 0 else provider.convert_from_device_time_to_timecode_ns(int(device[i]))
    return device, back


def handeye(A, B, stride=2):
    """Official SO3xR3 SVD and JTJ least-squares, without online claim."""
    if len(A) != len(B) or len(A) <= stride:
        raise ValueError('Insufficient aligned poses')
    da = inverse(A[:-stride]) @ A[stride:]
    db = inverse(B[:-stride]) @ B[stride:]
    va = Rotation.from_matrix(da[:, :3, :3]).as_rotvec().T
    vb = Rotation.from_matrix(db[:, :3, :3]).as_rotvec().T
    U, singular, Vh = np.linalg.svd(vb @ va.T)
    RX = Vh.T @ U.T
    reflected = np.linalg.det(RX) < 0
    if reflected:
        RX[2, :] *= -1  # reproduce official HandEyeSolver, not alternate Kabsch
    J = (da[:, :3, :3]-np.eye(3)).reshape(-1, 3)
    rhs = (np.einsum('ij,nj->ni', RX, db[:, :3, 3])-da[:, :3, 3]).reshape(-1)
    jtj = J.T @ J
    tx = np.linalg.lstsq(jtj, J.T @ rhs, rcond=None)[0]
    X = np.eye(4); X[:3, :3] = RX; X[:3, 3] = tx
    error = inverse(da @ X) @ (X @ db)
    condition = np.linalg.cond(jtj)
    report = dict(stride=stride, pairs=len(da), X_device_head=X.tolist(),
        status='DESCRIPTIVE_OFFLINE' if np.linalg.matrix_rank(jtj) == 3 else 'NOT_EVALUABLE',
        observability='Translation normal matrix rank reported; rank deficiency leaves hand-eye translation underconstrained.',
        rotation_cross_covariance_singular_values=singular.tolist(),
        translation_normal_matrix_rank=int(np.linalg.matrix_rank(jtj)),
        translation_condition_number=float(condition) if np.isfinite(condition) else None,
        official_reflection_correction=bool(reflected),
        rotation_motion_residual_deg=stats(np.rad2deg(Rotation.from_matrix(error[:, :3, :3]).magnitude())),
        translation_motion_residual_m=stats(np.linalg.norm(error[:, :3, 3], axis=1)),
        interpretation='Offline fit using full aligned sequence. Residual/conditioning are diagnostics, not drift or synchronization certification.')
    return X, report


def continuity(T, t_ns):
    dt = np.diff(t_ns)/1e9
    distance = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    angle = np.rad2deg(Rotation.from_matrix((inverse(T[:-1]) @ T[1:])[:, :3, :3]).magnitude())
    positive = dt > 0
    return dict(adjacent_pairs=len(dt), dt_seconds=stats(dt), translation_step_m=stats(distance),
        rotation_step_deg=stats(angle), nonpositive_dt_pairs=int((~positive).sum()),
        translation_speed_mps=stats(distance[positive]/dt[positive]),
        rotation_speed_deg_s=stats(angle[positive]/dt[positive]),
        interpretation='Adjacent raw frames, no interpolation or missing-frame bridge repair; descriptive extrema are not pass thresholds.')


def past_displacement(t_ns, position, window_ns=500_000_000, graph=None):
    """Only a past endpoint difference in MPS world/odometry XYZ, no body input.

    Endpoint search uses [t-window,t], never a future sample. Zero displacement
    yields undefined direction, explicitly NaN. Not a slow-walk heading model.
    """
    t = np.asarray(t_ns); p = np.asarray(position, float)
    if p.shape != (len(t), 3) or (np.diff(t) <= 0).any():
        raise ValueError('Monotonic pose timestamps required')
    begin = np.searchsorted(t, t-window_ns, side='left')
    end = np.arange(len(t)); delta = p-p[begin]
    length = np.linalg.norm(delta, axis=1)
    valid = (begin < end) & np.isfinite(delta).all(1) & (length > 0)
    if graph is not None:
        graph = np.asarray(graph)
        changes = np.r_[0, np.cumsum(graph[1:] != graph[:-1])]
        valid &= changes[begin] == changes[end]
    direction = np.full_like(delta, np.nan)
    direction[valid] = delta[valid]/length[valid, None]
    return direction, valid, length


def causal_check(t, p, graph=None):
    full = past_displacement(t, p, graph=graph)
    tested = []
    for cut in sorted(set((max(1, len(t)//4), max(1, len(t)//2), len(t)-2))):
        if not 1 <= cut < len(t)-1:
            continue
        prefix = past_displacement(t[:cut+1], p[:cut+1], graph=None if graph is None else graph[:cut+1])
        changed = p.copy(); changed[cut+1:] += np.array([1234., -5678., 9012.])
        future = past_displacement(t, changed, graph=graph)
        for expected, truncated, perturbed in zip(full, prefix, future):
            if not np.array_equal(expected[:cut+1], truncated, equal_nan=True) or not np.array_equal(expected[:cut+1], perturbed[:cut+1], equal_nan=True):
                raise AssertionError('Future influenced past proxy output')
        tested.append(cut)
    return dict(status='PASS' if tested else 'NOT_EVALUABLE', actual_prefix_last_indices=tested,
        checks=['Actual prefix rerun matches all retained outputs', 'Large future-position perturbation leaves all retained outputs unchanged'],
        scope='Causality of this displacement implementation only. Offline MPS generation may use future observations; this does not establish online MPS causality.')


def read_mps(path, check):
    """Read original CSV with explicit axis fields; no clock or pose correction."""
    with path.open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames or []
        world = 'world' if 'tx_world_device' in fields else 'odometry'
        cols = ['tracking_timestamp_us'] + [f'{a}_{world}_device' for a in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')]
        uid_column = 'graph_uid' if 'graph_uid' in fields else 'session_uid' if 'session_uid' in fields else None
        missing = set(cols)-set(fields)
        if missing:
            raise ValueError('Missing CSV columns: '+repr(sorted(missing)))
        values = []; graphs = []
        for i, row in enumerate(reader):
            if i % 4096 == 0:
                check()
            values.append([float(row[k]) for k in cols])
            if uid_column is not None:
                graphs.append(row[uid_column])
    a = np.asarray(values, float)
    if a.ndim != 2 or len(a) < 2:
        raise ValueError('Insufficient MPS rows')
    return dict(t_us=a[:, 0], position=a[:, 1:4], quaternion=a[:, 4:8],
        graph=np.asarray(graphs) if graphs else None, frame=world, columns=cols,
        frame_uid_column=uid_column)


def inspect_sample(sample, sdk_path=None, cap_seconds=180.):
    if not 0 < cap_seconds <= 180:
        raise ValueError('Sample analysis cap must be in(0,180] seconds')
    sample = Path(sample); began = time.monotonic()
    report = dict(analysis_status='INCOMPLETE', sample=str(sample.resolve()),
        missing_files=[], sections={}, boundaries=[
            'Original raw timestamps/quaternions retained; official provider may replace invalid intervals/quaternions, but is not called.',
            'Hand-eye and nearest closed-loop MPS association are offline and can use future information.',
            'Per-frame registered Head=device@X is a construction identity, never an independent drift/synchronization pass.',
            'No global PASS, slow-walk coverage, heading error benefit or product claim.'])
    def check():
        if time.monotonic()-began >= cap_seconds:
            raise TimeoutError('Sample analysis budget reached')
    needed = ['metadata.json', 'body/xdata.npz', 'recording_head/data/motion.vrs',
        'recording_head/mps/slam/closed_loop_trajectory.csv', 'recording_head/mps/slam/open_loop_trajectory.csv']
    paths = {p: sample/p for p in needed}
    report['missing_files'] = [p for p, f in paths.items() if not f.is_file()]
    report['files'] = {p: dict(bytes=f.stat().st_size) for p, f in paths.items() if f.is_file()}
    try:
        if paths['metadata.json'].exists():
            metadata = json.loads(paths['metadata.json'].read_text(encoding='utf8'))
            report['metadata'] = metadata
        body = None; t_ns = None; H = None; P = None; provider = None; device = None; runs = []
        if paths['body/xdata.npz'].exists():
            check()
            with np.load(paths['body/xdata.npz'], allow_pickle=False) as z:
                available_keys = list(z.files)
                body = {k: z[k] for k in ('timestamps_us', 'frameCount', 'frameRate', 'segment_tXYZ', 'segment_qWXYZ') if k in z.files}
            info = dict(arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                nonfinite=int((~np.isfinite(v)).sum()) if np.issubdtype(v.dtype, np.number) else None) for k, v in body.items()})
            info['all_npz_keys'] = available_keys
            info['array_audit_scope'] = 'Only timestamps/frameCount/frameRate/segment positions/quaternions loaded; acceleration and other fields not inspected.'
            required = {'timestamps_us', 'frameCount', 'frameRate', 'segment_tXYZ', 'segment_qWXYZ'}
            if not required.issubset(body):
                raise ValueError('Missing body fields: '+repr(sorted(required-set(body))))
            t = np.asarray(body['timestamps_us']).reshape(-1); n = len(t)
            q = np.asarray(body['segment_qWXYZ']).reshape(n, 23, 4)
            pos = np.asarray(body['segment_tXYZ']).reshape(n, 23, 3)
            dt = np.diff(t.astype(float)); norms = np.linalg.norm(q, axis=-1)
            invalid_interval = ~np.isfinite(dt) | (np.abs(dt-1e6/240.) > 1000)
            corrected_dt = np.where(invalid_interval, int(1e6/240.), dt)
            correction = np.r_[0., np.cumsum(corrected_dt)] + float(t[0])-t.astype(float)
            info.update(rows=n, declared_frame_count=np.asarray(body['frameCount']).tolist(),
                declared_frame_count_matches_rows=bool(np.asarray(body['frameCount']).size == 1 and int(np.asarray(body['frameCount']).item()) == n),
                declared_frame_rate=np.asarray(body['frameRate']).tolist(), timestamp_us=stats(t), dt_us=stats(dt),
                nonpositive_dt=int((dt <= 0).sum()), raw_invalid_240Hz_interval_count=int(invalid_interval.sum()),
                quaternion_norm=stats(norms), quaternion_norm_below_point1=int((norms < .1).sum()),
                head_quaternion_norm=stats(norms[:, 6]), pelvis_quaternion_norm=stats(norms[:, 0]),
                official_timestamp_repair_if_applied=dict(intervals_replaced=int(invalid_interval.sum()),
                    maximum_change_us=float(np.abs(correction).max()), final_change_us=float(correction[-1]),
                    official_endpoint_10ms_rejection=bool(abs(correction[-1]) > 10000), applied=False),
                official_quaternion_repair_if_triggered=dict(trigger_any_norm_below_point1=bool((norms < .1).any()),
                    potential_norm_below_point5=int((norms < .5).sum()), applied=False))
            report['sections']['raw_body'] = info
            runs = raw_contiguous_runs(t)
            report['sections']['raw_contiguous_runs'] = dict(runs=runs,
                scope='Original frame order partitioned at all anomalous240Hz dt; all frames retained. No timestamp repair or cross-run bridging.')
            if not np.isfinite(t).all():
                report['sections']['registration'] = dict(status='NOT_EVALUABLE', reason='Nonfinite raw body timestamps cannot enter integer SDK mapping; no correction')
            else:
                t_ns = ns_from_us(t)
                try:
                    H = poses(q[:, 6], pos[:, 6], 'WXYZ'); P = poses(q[:, 0], pos[:, 0], 'WXYZ')
                    report['sections']['original_head_continuity'] = continuity(H, t_ns)
                    report['sections']['original_head_body_distance_m'] = stats(np.linalg.norm(pos[:, 6]-pos[:, 0], axis=1))
                except ValueError as exc:
                    report['sections']['registration'] = dict(status='NOT_EVALUABLE', reason=str(exc))
        trajectories = {}
        for name in ('closed', 'open'):
            path = paths[f'recording_head/mps/slam/{name}_loop_trajectory.csv']
            if not path.exists():
                report['sections'][name+'_MPS'] = dict(status='NOT_EVALUABLE', reason='Missing CSV')
                continue
            try:
                d = read_mps(path, check); trajectories[name] = d
                norms = np.linalg.norm(d['quaternion'], axis=1)
                report['sections'][name+'_MPS'] = dict(rows=len(d['t_us']), frame=d['frame'],
                    frame_uid_column=d['frame_uid_column'],
                    columns=d['columns'], timestamp_nonfinite=int((~np.isfinite(d['t_us'])).sum()),
                    dt_us=stats(np.diff(d['t_us'])), nonpositive_dt=int((np.diff(d['t_us']) <= 0).sum()),
                    position_nonfinite=int((~np.isfinite(d['position'])).sum()),
                    quaternion_norm=stats(norms), invalid_quaternion_count=int((norms < .1).sum()),
                    graph_changes=int((d['graph'][1:] != d['graph'][:-1]).sum()) if d['graph'] is not None else None)
            except (ValueError, KeyError) as exc:
                report['sections'][name+'_MPS'] = dict(status='NOT_EVALUABLE', reason=str(exc))
        if sdk_path:
            sys.path.insert(0, str(Path(sdk_path)))
        if paths['recording_head/data/motion.vrs'].exists() and t_ns is not None:
            try:
                from projectaria_tools.core import data_provider
                from projectaria_tools.core.sensor_data import TimeDomain
                check(); provider = data_provider.create_vrs_data_provider(str(paths['recording_head/data/motion.vrs']))
                if provider is None:
                    raise ValueError('VRS SDK did not create provider')
                streams = provider.get_all_streams()
                supported = {str(s): bool(provider.supports_time_domain(s, TimeDomain.TIME_CODE)) for s in streams}
                if not any(supported.values()):
                    raise ValueError('TIME_CODE not supported by any VRS stream')
                vrs_start = int(provider.get_first_time_ns_all_streams(TimeDomain.TIME_CODE))
                vrs_end = int(provider.get_last_time_ns_all_streams(TimeDomain.TIME_CODE))
                device, back = map_clock(provider, t_ns, check)
                valid_conversion = (device >= 0) & (back >= 0)
                report['sections']['clock'] = dict(status='MAPPED', frames=len(t_ns), TIME_CODE_supported_streams=supported,
                    valid_conversion_count=int(valid_conversion.sum()),
                    roundtrip_error_ns=stats((back-t_ns)[valid_conversion]),
                    unsupported_conversion_count=int((~valid_conversion).sum()),
                    device_conversion_negative_count=int((device < 0).sum()),
                    reverse_conversion_negative_count=int((back < 0).sum()),
                    body_samples_outside_VRS_supported_span=int(((t_ns < vrs_start) | (t_ns > vrs_end)).sum()),
                    device_dt_ns=stats(np.diff(device)),
                    nonpositive_device_dt=int((np.diff(device) <= 0).sum()),
                    VRS_timecode_start_ns=vrs_start, VRS_timecode_end_ns=vrs_end,
                    interpretation='MAPPED means SDK numerical conversion attempted; small roundtrip does not establish physical synchronization. Unsupported/out-of-span raw frames retained, never repaired.')
            except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
                device = None
                report['sections']['clock'] = dict(status='NOT_EVALUABLE', reason=repr(exc), repair_applied=False)
        else:
            report['sections']['clock'] = dict(status='NOT_EVALUABLE', reason='Missing VRS or usable raw body timestamps')
        if device is not None and H is not None and P is not None and 'closed' in trajectories:
            try:
                d = trajectories['closed']
                if not np.isfinite(d['t_us']).all():
                    raise ValueError('Nonfinite closed-loop timestamps; no clock repair')
                mt = ns_from_us(d['t_us'])
                ix = nearest_index(mt, device); diff = mt[ix]-device
                valid_clock = device >= 0
                in_span = valid_clock & (device >= mt[0]) & (device <= mt[-1])
                report['sections']['nearest_closed_MPS'] = dict(frames=len(ix), dt_ns=stats(diff[valid_clock]),
                    absolute_dt_ns=stats(np.abs(diff[valid_clock])), in_span_dt_ns=stats(diff[in_span]),
                    valid_device_clock_frames=int(valid_clock.sum()), in_span_frames=int(in_span.sum()),
                    future_matches=int(((diff > 0) & valid_clock).sum()),
                    outside_trajectory_span=int((~in_span).sum()), nearest_may_use_future=True)
                Dall = poses(d['quaternion'], d['position'], 'XYZW'); D = Dall[ix]
                # Preserve every run in the ledger; this fit intentionally
                # restricts scope to the largest usable original continuous run.
                for run in runs:
                    a, b = run['start_frame'], run['end_frame_exclusive']
                    run['valid_device_clock_frames'] = int(valid_clock[a:b].sum())
                    run['in_closed_MPS_span_frames'] = int(in_span[a:b].sum())
                usable = [run for run in runs if run['timestamp_finite_and_monotonic'] and run['frames'] >= 6 and run['in_closed_MPS_span_frames'] >= 6]
                if not usable:
                    raise ValueError('No usable raw continuous segment')
                largest = max(usable, key=lambda r: (r['frames'], -r['run_index']))
                largest['status'] = 'SELECTED_OFFLINE_ONLY'
                largest['reason'] = 'Largest usable original raw continuous segment, selected by frame count before residual fitting'
                a, b = largest['start_frame'], largest['end_frame_exclusive']
                raw_selected = np.arange(a, b)
                clock = report['sections']['clock']
                lo = max(int(t_ns[a]), clock['VRS_timecode_start_ns'])+2_000_000_000
                hi = min(int(t_ns[b-1]), clock['VRS_timecode_end_ns'])-2_000_000_000
                keep = (t_ns[raw_selected] >= lo) & (t_ns[raw_selected] <= hi)
                keep &= (device[raw_selected] >= mt[0]+2_000_000_000) & (device[raw_selected] <= mt[-1]-2_000_000_000)
                selected = raw_selected[keep]
                if len(selected) < 6 or not in_span[selected].all():
                    largest['status'] = 'NOT_EVALUABLE'
                    raise ValueError('Largest usable raw run has insufficient clock/closed coverage after2s trimming')
                if (np.diff(selected) != 1).any():
                    largest['status'] = 'NOT_EVALUABLE'
                    raise ValueError('Clipped largest segment has interior unsupported frames; no bridging')
                if np.any(np.abs(np.diff(t_ns[selected])-1e9/240.) > 1_000_000):
                    raise ValueError('Raw missing/irregular body frames in registration span; no repair or bridging')
                if d['graph'] is not None and len(np.unique(d['graph'][ix[selected]])) != 1:
                    raise ValueError('Registration span crosses closed-loop graph frames')
                X, fit = handeye(D[selected], H[selected], 2)
                fit['scope'] = 'largest raw continuous segment only'
                fit['original_run_index'] = largest['run_index']
                fit['original_run_range_half_open'] = [a, b]
                fit['original_run_frames'] = b-a
                fit['registration_clip_rule'] = 'Intersect largest run with VRS TIME_CODE and closed DEVICE spans; trim2seconds at ends. Other raw runs NOT_RUN. No bridging or timestamp correction.'
                fit['registered_frames'] = len(selected)
                largest['status'] = fit['status']
                fit['selected_original_frame_range'] = [int(selected[0]), int(selected[-1])]
                registered_pelvis = D[selected] @ X @ inverse(H[selected]) @ P[selected]
                fit['registered_pelvis_continuity'] = continuity(registered_pelvis, t_ns[selected])
                fit['registered_head_body_distance_m'] = stats(np.linalg.norm((D[selected] @ X)[:, :3, 3]-registered_pelvis[:, :3, 3], axis=1))
                fit['head_anchor_constraint'] = 'RegisteredHead = devicePose @ X by construction; do not use relative constancy as independent drift pass.'
                blocks = []
                for part in np.array_split(selected, 3):
                    check()
                    if len(part) < 6:
                        blocks.append(dict(status='NOT_EVALUABLE', reason='Too few block frames')); continue
                    bx, br = handeye(D[part], H[part], 2)
                    delta = inverse(X) @ bx
                    blocks.append(dict(first_frame=int(part[0]), last_frame=int(part[-1]),
                        transform_translation_difference_m=float(np.linalg.norm(delta[:3, 3])),
                        transform_rotation_difference_deg=float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())),
                        translation_normal_matrix_rank=br['translation_normal_matrix_rank']))
                fit['offline_block_transform_consistency'] = blocks
                report['sections']['registration'] = fit
            except (ValueError, np.linalg.LinAlgError) as exc:
                report['sections']['registration'] = dict(status='NOT_EVALUABLE', reason=str(exc), scope='largest raw continuous segment only; no repair/bridging')
        else:
            report['sections'].setdefault('registration', dict(status='NOT_EVALUABLE', reason='Missing usable body/clock/closed-loop poses'))
        if 'open' in trajectories:
            try:
                d = trajectories['open']
                if not np.isfinite(d['t_us']).all():
                    raise ValueError('Nonfinite open-loop timestamps; no clock repair')
                t = ns_from_us(d['t_us'])
                check(); direction, valid, distance = past_displacement(t, d['position'], graph=d['graph'])
                report['sections']['pure_past_openloop_proxy'] = dict(window_seconds=.5, output_rows=len(t),
                    defined_direction_rows=int(valid.sum()), undefined_direction_rows=int((~valid).sum()),
                    displacement_m=stats(distance), causal_check=causal_check(t, d['position'], d['graph']),
                    interpretation='3D odometry displacement proxy only; no mocap, future labels or walking-heading accuracy claim; offline openloop source does not prove full MPS online causality.')
            except ValueError as exc:
                report['sections']['pure_past_openloop_proxy'] = dict(status='NOT_EVALUABLE', reason=str(exc))
        else:
            report['sections']['pure_past_openloop_proxy'] = dict(status='NOT_EVALUABLE', reason='Missing usable open-loop CSV')
        check(); report['analysis_status'] = 'COMPLETE'
    except BaseException as exc:
        report['error'] = repr(exc)
        report['analysis_status'] = 'BUDGET_STOP' if isinstance(exc, TimeoutError) else 'INCOMPLETE'
    finally:
        report['seconds'] = time.monotonic()-began
        report['cap_seconds'] = cap_seconds
        report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sdk-path', type=Path)
    parser.add_argument('--cap-seconds', type=float, default=180.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit output')
    receipt = inspect_sample(args.sample, args.sdk_path, args.cap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(analysis_status=receipt['analysis_status'], seconds=receipt['seconds'], missing_files=receipt['missing_files'])))
