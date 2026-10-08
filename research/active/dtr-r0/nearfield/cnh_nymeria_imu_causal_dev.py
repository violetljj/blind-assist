"""Nymeria factory-only IMU attitude dependency probe; not a position/E1 estimator.

Raw input is native DEVICE_TIME, accel in m/s^2 and gyro in rad/s. Factory
intrinsics and extrinsics are parsed independently of MPS/online calibration.
The initial up estimate assumes gyro-compensated mean specific force over the
first 2 seconds approximates gravity. This is not a validated stationary period.
All output is UNKNOWN until that window closes; gyro bias is not fit from motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

INIT_NS = 2_000_000_000
SPAN_NS = 60_000_000_000


def factory_parameters(factory_json):
    from projectaria_tools.core import calibration
    c = calibration.device_calibration_from_json_string(factory_json)
    if c is None:
        raise ValueError('Factory JSON is not a DeviceCalibration')
    imu = c.get_imu_calib('imu-right')
    if imu is None:
        raise ValueError('Factory lacks imu-right')
    params = dict(
        accel_rectification=np.asarray(imu.get_accel_model().get_rectification()),
        accel_bias=np.asarray(imu.get_accel_model().get_bias()),
        gyro_rectification=np.asarray(imu.get_gyro_model().get_rectification()),
        gyro_bias=np.asarray(imu.get_gyro_model().get_bias()),
        device_imu=imu.get_transform_device_imu().to_matrix(),
        device_cpf=c.get_transform_device_cpf().to_matrix())
    if not all(np.isfinite(v).all() for v in params.values()):
        raise ValueError('Nonfinite factory parameters')
    return params, imu


def heading(forward):
    horizontal = np.array([forward[0], 0., forward[2]])
    norm = np.linalg.norm(horizontal)
    return horizontal / norm if norm >= .1 else np.full(3, np.nan)


def pipeline(raw, factory_json):
    """Reparse factory and reinitialize for every prefix; output at each record."""
    p, _ = factory_parameters(factory_json)
    t, accel, gyro, valid = (raw[k] for k in ('capture_ns', 'accel', 'gyro', 'valid'))
    R_di = p['device_imu'][:3, :3]
    a = np.linalg.solve(p['accel_rectification'], (accel-p['accel_bias']).T).T @ R_di.T
    w = np.linalg.solve(p['gyro_rectification'], (gyro-p['gyro_bias']).T).T @ R_di.T
    forward = p['device_cpf'][:3, 2]
    result = np.full((len(t), 3), np.nan)
    R = Rotation.identity()
    accumulation = np.zeros(3)
    initialization_count = 0
    up_rotation = None
    initialization_ns = None
    # Invalid records or a nonpositive/>20ms gap reset rather than bridge time.
    start_ns = None
    last_ns = None
    for i, ts in enumerate(t):
        if not valid[i] or (last_ns is not None and not 0 < ts-last_ns <= 20_000_000):
            R = Rotation.identity()
            accumulation[:] = 0
            initialization_count = 0
            start_ns = last_ns = None
            up_rotation = None
            initialization_ns = None
            continue
        if start_ns is None:
            start_ns = int(ts)
        if last_ns is not None:
            dt = float(ts-last_ns)/1e9
            R = R * Rotation.from_rotvec((w[i-1]+w[i])*.5*dt)
        last_ns = int(ts)
        if up_rotation is None:
            accumulation += R.apply(a[i])
            initialization_count += 1
            if ts-start_ns < INIT_NS:
                continue
            norm = np.linalg.norm(accumulation)
            if not np.isfinite(norm) or norm < 1e-8:
                continue
            up = accumulation/norm
            up_rotation = Rotation.align_vectors([[0.,1.,0.]], [up])[0]
            initialization_ns = int(ts)
        result[i] = heading((up_rotation * R).apply(forward))
    return result, dict(initialization_capture_ns=initialization_ns,
                       final_initialization_count=initialization_count)


def read_raw(provider, sid, cutoff_ns, timestamp='capture_timestamp_ns'):
    rows = []
    # Index order is preserved; no sorting, interpolation or TIME_CODE conversion.
    for i in range(provider.get_num_data(sid)):
        d = provider.get_imu_data_by_index(sid, i)
        if getattr(d, timestamp) > cutoff_ns:
            break
        a, w = np.asarray(d.accel_msec2), np.asarray(d.gyro_radsec)
        rows.append((d.capture_timestamp_ns, d.arrival_timestamp_ns, a, w,
                     bool(d.accel_valid and d.gyro_valid and np.isfinite(a).all() and np.isfinite(w).all())))
    return dict(capture_ns=np.array([v[0] for v in rows], dtype=np.int64),
                arrival_ns=np.array([v[1] for v in rows], dtype=np.int64),
                accel=np.array([v[2] for v in rows]), gyro=np.array([v[3] for v in rows]),
                valid=np.array([v[4] for v in rows], dtype=bool))


def run(sample, sdk, output, cap_seconds):
    began = time.monotonic()
    if (output/'causal_probe.json').exists():
        raise FileExistsError('Preserve prior result: choose a new output directory')
    sys.path.insert(0, str(sdk))
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    provider = data_provider.create_vrs_data_provider(str(sample/'recording_head/data/motion.vrs'))
    sid = StreamId('1202-1')
    if provider.get_label_from_stream_id(sid) != 'imu-right':
        raise ValueError('Unexpected stream label')
    cfg = provider.get_imu_configuration(sid)
    if cfg.online_calibration:
        raise ValueError('Configuration unexpectedly contains online calibration')
    params, imu = factory_parameters(cfg.factory_calibration)
    first = provider.get_imu_data_by_index(sid, 0).capture_timestamp_ns
    raw = read_raw(provider, sid, first+SPAN_NS)
    count = len(raw['capture_ns'])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'raw_right_60s.npz', **raw)
    # Verify explicit rectification formula against SDK on actual measurements.
    for k in ('accel','gyro'):
        x = raw[k][0]
        formula = np.linalg.solve(params[k+'_rectification'], x-params[k+'_bias'])
        sdk_value = getattr(imu, 'raw_to_rectified_'+k)(x)
        if not np.allclose(formula, sdk_value, atol=1e-12, rtol=1e-12):
            raise ValueError('Factory rectification disagrees with SDK')
    full, meta = pipeline(raw, cfg.factory_calibration)
    available_ns = np.maximum.accumulate(raw['arrival_ns'])
    cuts = []
    for seconds in (1., 1.999, 2.002, 3., 10., 30., 60.):
        cutoff = first+round(seconds*1e9)
        n = int(np.searchsorted(raw['capture_ns'], cutoff, side='right'))
        prefix = {k: v[:n] for k,v in raw.items()}
        value, _ = pipeline(prefix, cfg.factory_calibration)
        same = np.array_equal(value, full[:n], equal_nan=True)
        # Entire raw ingestion repeated from VRS through calibration/init/filter.
        streamed = read_raw(provider, sid, cutoff)
        streamed_value, _ = pipeline(streamed, cfg.factory_calibration)
        streaming_same = np.array_equal(streamed_value, full[:n], equal_nan=True)
        # HOST_TIME recorded-arrival prefix is a separate replay clock, not
        # application reception. Never subtract it from DEVICE_TIME as latency.
        host_cut = min(int(raw['arrival_ns'][0])+round(seconds*1e9), int(raw['arrival_ns'][-1]))
        received = read_raw(provider, sid, host_cut, 'arrival_timestamp_ns')
        received_value, _ = pipeline(received, cfg.factory_calibration)
        received_same = np.array_equal(received_value, full[:len(received_value)], equal_nan=True)
        # Change every future record (including validity); earlier outputs invariant.
        changed = {k: v.copy() for k,v in raw.items()}
        changed['gyro'][n:] += 100.
        changed['accel'][n:] *= -50.
        changed['valid'][n:] = False
        mutated, _ = pipeline(changed, cfg.factory_calibration)
        mutation_same = np.array_equal(mutated[:n], full[:n], equal_nan=True)
        changed_host = {k:v.copy() for k,v in raw.items()}
        m = len(received_value)
        changed_host['gyro'][m:] += 100.
        changed_host['accel'][m:] *= -50.
        changed_host['valid'][m:] = False
        mutated_host, _ = pipeline(changed_host, cfg.factory_calibration)
        host_mutation_same = np.array_equal(mutated_host[:m], full[:m], equal_nan=True)
        cuts.append(dict(seconds=seconds, records=n, prefix_equal=same,
                         vrs_adapter_equal=streaming_same, future_mutation_equal=mutation_same,
                         recorded_host_cut_ns=host_cut, recorded_arrival_records=m,
                         arrival_prefix_equal=received_same, arrival_future_mutation_equal=host_mutation_same))
        if time.monotonic()-began > cap_seconds:
            raise TimeoutError('CPU diagnostic cap reached')
    timestamp_difference = (raw['arrival_ns']-raw['capture_ns'])/1e6
    result = dict(status='ATTITUDE_PREFIX_PASS' if all(all(v[k] for k in ('prefix_equal','vrs_adapter_equal','future_mutation_equal','arrival_prefix_equal','arrival_future_mutation_equal')) for v in cuts) else 'ATTITUDE_PREFIX_FAIL',
      sequence=sample.name, records=count, valid_records=int(raw['valid'].sum()),
      output_records=int(np.isfinite(full).all(axis=1).sum()),
      capture_span_seconds=float((raw['capture_ns'][-1]-raw['capture_ns'][0])/1e9),
      recorded_arrival_minus_capture_ms_quantiles=np.quantile(timestamp_difference,[0,.5,.95,1]).tolist(),
      arrival_nonpositive_intervals=int((np.diff(raw['arrival_ns']) <= 0).sum()),
      recorded_arrival_less_than_capture=int((timestamp_difference < 0).sum()),
      factory_sha256=hashlib.sha256(cfg.factory_calibration.encode()).hexdigest(),
      calibration={k:v.tolist() for k,v in params.items()},
      initialization=meta, prefix_checks=cuts,
      output_availability='HOST_TIME recorded-arrival prefix proxy; not app/device live reception or latency',
      E1_status='NOT_EVALUABLE_NO_CAUSAL_POSITION', position_estimator='NOT_IMPLEMENTED',
      limits=['Factory JSON is a fixed recording configuration; physical VRS file truncation and live transport are not tested.',
              'Prefix tests cover raw-record ingestion, factory parsing, initialization, gyro propagation and CPF projection on first60s.',
              'Bootstrapped up and gyro attitude are unvalidated; no heading accuracy or drift claim.',
              'Recorded HOST_TIME and DEVICE_TIME are distinct domains; numeric timestamp differences are not end-to-end latency.',
              'No MPS, mocap, timecode, future pelvis, fitted initial velocity or position is an estimator input.'],
      seconds=time.monotonic()-began,
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez(output/'cpf_attitude_60s.npz', capture_ns=raw['capture_ns'],
             available_ns=available_ns, horizontal_forward=full)
    with (output/'causal_probe.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--sdk-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cap-seconds',type=float,default=160.)
    args = parser.parse_args()
    r = run(args.sample,args.sdk_path,args.output,args.cap_seconds)
    print(json.dumps({k:r[k] for k in ('status','records','valid_records','output_records','initialization','prefix_checks','E1_status','seconds')}))
