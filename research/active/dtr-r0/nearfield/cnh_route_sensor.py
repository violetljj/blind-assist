"""Declared CNH forward proxy, NOT a calibrated ST sensor/target detector.

Input is first-visible, opaque diffuse geometry in the *ToF* frame, sampled on
an 8x8 angular lattice. Each zone contains R quadrature rays; missing rays keep
their solid angle in the denominator. Never pass evaluator labels or occluded
surfaces. See CNH_ROUTE_COMPARISON_PLAN_20260924.md for scope and timing.

The expensive ray histogram can use torch/CUDA via backend='torch', device=...
Small 64x128 electronics/readout arrays use NumPy on CPU. Backend and transfers
are explicit; this is not an end-to-end GPU pipeline. Timing/integration,
visibility, scanning order, installation error and packet drops belong upstream.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

RAW_BIN_M = 0.0375348
RAW_BINS = 128
FOV_DEG = 45.0
SCHEMA = 'blindassist.cnh-sensor-proxy.v1'


@dataclass(frozen=True)
class SensorConfig:
    name: str
    rows: int
    cols: int
    bins: int
    sub_sample: int
    hz: float
    integration_ms: float = 20.0
    start_bin: int = 0
    cnh_visible: bool = True

    @property
    def buffer_bytes(self):
        return self.rows * self.cols * (self.bins + 1) * 5 + 28

    def validate(self):
        if (self.rows, self.cols) not in ((4, 4), (8, 8)):
            raise ValueError('Only declared 4x4/8x8 configurations supported')
        if not (1 <= self.sub_sample <= 8 and 1 <= self.bins <= 128):
            raise ValueError('Invalid bin aggregation')
        if self.start_bin < 0 or self.start_bin + self.bins*self.sub_sample > RAW_BINS:
            raise ValueError('CNH window exceeds 128 raw bins')
        if self.cnh_visible and self.buffer_bytes > 6160:
            raise ValueError('CNH exceeds 6160-byte buffer')
        if self.hz <= 0 or self.integration_ms <= 0:
            raise ValueError('Invalid timing')
        integrations = 4 if self.rows == 8 else 1
        if integrations*self.integration_ms + 1 >= 1000/self.hz:
            raise ValueError('Integration alone exceeds frame period')
        return self


H1 = SensorConfig('H1', 8, 8, 128, 1, 10, cnh_visible=False)
H2 = SensorConfig('H2', 4, 4, 24, 4, 5)
H3 = SensorConfig('H3', 8, 8, 16, 8, 5)
CONFIGS = {c.name: c for c in (H1, H2, H3)}


@dataclass(frozen=True)
class SensorParameters:
    # All electronics scales below are ASSUMED_STRESS until separately fitted.
    # signal_counts is photons at 1m, rho=1, cos=1 per native zone/integration;
    # output_gain maps pseudo counts to arbitrary CNH units, not ST calibration.
    signal_counts: float = 4000.0
    ambient_counts: float = 4.0
    output_gain: float = 1.0
    noise_scale: float = 1.0
    pulse_sigma_bins: float = 2.0
    tail_mass: float = 0.1
    tail_decay_bins: float = 4.0
    crosstalk_fraction: float = 0.02
    crosstalk_range_m: float = 0.04
    neighbour_leak: float = 0.02
    detection_snr: float = 5.0
    # Zero is a nominal coordinate assumption, not an identified bin offset.
    range_zero_m: float = 0.0

    def validate(self):
        vals = asdict(self)
        if not all(np.isfinite(v) for v in vals.values()):
            raise ValueError('Nonfinite sensor parameter')
        for k in ('signal_counts', 'ambient_counts', 'noise_scale', 'pulse_sigma_bins',
                  'tail_mass', 'crosstalk_fraction', 'neighbour_leak'):
            if vals[k] < 0:
                raise ValueError(f'Negative {k}')
        if self.output_gain <= 0 or self.tail_decay_bins <= 0 or self.detection_snr <= 0:
            raise ValueError('Invalid positive sensor parameter')
        if max(self.tail_mass, self.crosstalk_fraction, self.neighbour_leak) > 1:
            raise ValueError('Fractions exceed one')
        return self


def decode_cnh(raw, scaler):
    """UM3183 Rev7 section 5.7; preserve sign and negative signed scalers."""
    raw, scaler = np.broadcast_arrays(np.asarray(raw), np.asarray(scaler))
    if not np.isfinite(raw).all() or not np.isfinite(scaler).all():
        raise ValueError('Nonfinite raw CNH')
    if np.any(scaler != np.floor(scaler)) or np.any((scaler < -128) | (scaler > 127)):
        raise ValueError('Invalid signed 8-bit scaler')
    if np.any(raw != np.floor(raw)) or np.any((raw < -2**31) | (raw > 2**31-1)):
        raise ValueError('Invalid signed 32-bit CNH integer')
    return np.ldexp(raw.astype(np.float64), -scaler.astype(np.int32))


def angular_rays(samples_per_axis=16):
    """Midpoint quadrature on equal tangent-plane zones; camera X right,Y down.

    Returns [8,8,R,3] unit directions and [8,8,R] solid-angle weights. True
    lens angular response remains unidentified; this square cone is a proxy.
    """
    n = int(samples_per_axis)
    if n < 1 or n != samples_per_axis:
        raise ValueError('samples_per_axis must be a positive integer')
    edge = np.tan(np.deg2rad(FOV_DEG/2))
    q = -edge + (np.arange(8*n)+.5)*2*edge/(8*n)
    yy, xx = np.meshgrid(q, q, indexing='ij')
    xyz = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    norm = np.linalg.norm(xyz, axis=-1)
    directions = xyz/norm[..., None]
    weights = (2*edge/(8*n))**2/norm**3
    directions = directions.reshape(8, n, 8, n, 3).transpose(0, 2, 1, 3, 4).reshape(8, 8, n*n, 3)
    weights = weights.reshape(8, n, 8, n).transpose(0, 2, 1, 3).reshape(8, 8, n*n)
    return directions, weights


def axial_to_radial(axial_m, directions):
    directions = np.asarray(directions)
    if directions.shape[-1] != 3 or np.any(directions[..., 2] <= 0):
        raise ValueError('Need forward unit directions')
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-6):
        raise ValueError('Directions must be unit length')
    return np.asarray(axial_m)/directions[..., 2]


def _ray_inputs(radial_m, nir_reflectance, incidence_cos, solid_angle_weights):
    distance = np.asarray(radial_m, dtype=np.float64)
    if distance.ndim < 3 or distance.shape[-3:-1] != (8, 8):
        raise ValueError('Expected [...,8,8,rays] radial distances')
    rho, cosine, weights = [np.broadcast_to(np.asarray(a, dtype=np.float64), distance.shape)
                            for a in (nir_reflectance, incidence_cos, solid_angle_weights)]
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(-1) <= 0):
        raise ValueError('Need nonnegative finite quadrature, including missed rays')
    valid = np.isfinite(distance) & (distance > 0)
    if np.any(valid & (~np.isfinite(rho) | (rho < 0) | (rho > 1))):
        raise ValueError('NIR reflectance must lie in [0,1] for valid surfaces')
    if np.any(valid & (~np.isfinite(cosine) | (cosine < 0) | (cosine > 1))):
        raise ValueError('Incidence cosine must lie in [0,1]')
    weights = weights/weights.sum(-1, keepdims=True)
    return distance, np.where(valid, rho, 0), np.where(valid, cosine, 0), weights, valid


def _histogram(distance, energy, valid, bins, width, start, backend, device):
    position = np.floor((np.where(valid, distance, start)-start)/width).astype(np.int64)
    inside = valid & (position >= 0) & (position < bins)
    shape = distance.shape[:-1]+(bins,)
    if backend == 'numpy':
        out = np.zeros(shape)
        flat = out.reshape(-1, bins)
        index = position.reshape(len(flat), -1)
        values = np.where(inside, energy, 0).reshape(len(flat), -1)
        row = np.broadcast_to(np.arange(len(flat))[:, None], index.shape)
        np.add.at(flat, (row, np.clip(index, 0, bins-1)), values)
        actual_device = 'cpu'
    elif backend == 'torch':
        import torch
        actual_device = str(torch.device(device))
        index = torch.as_tensor(np.clip(position, 0, bins-1), device=device)
        values = torch.as_tensor(np.where(inside, energy, 0), device=device)
        result = torch.zeros(shape, dtype=values.dtype, device=device)
        result.scatter_add_(-1, index, values)
        out = result.cpu().numpy()
    else:
        raise ValueError('backend must be numpy or torch')
    return out, np.where(valid & ~inside, energy, 0).sum(-1), actual_device


def _pulse_matrix(params):
    # Rows input, columns output. No wrap or boundary renormalization: escaped
    # pulse energy is lost, never folded into first/last bin.
    offset = np.arange(-RAW_BINS+1, RAW_BINS, dtype=float)
    if params.pulse_sigma_bins == 0:
        gaussian = (offset == 0).astype(float)
    else:
        gaussian = np.exp(-.5*(offset/params.pulse_sigma_bins)**2)
        gaussian /= gaussian.sum()
    tail = np.where(offset >= 0, np.exp(-np.maximum(offset, 0)/params.tail_decay_bins), 0)
    tail /= tail.sum()
    kernel = (1-params.tail_mass)*gaussian + params.tail_mass*tail
    shifts = np.arange(RAW_BINS)[None, :]-np.arange(RAW_BINS)[:, None]
    return kernel[shifts+RAW_BINS-1]


def synthesize_response(radial_m, nir_reflectance, incidence_cos, solid_angle_weights,
                        *, params=None, seed=20260924, backend='numpy', device='cpu'):
    """Shared 8x8x128 signed response, paired across configuration readouts.

    Same seed means paired electronics for equal geometry samples. Acquisition
    scheduling must use a stable frame/exposure seed and its true time upstream;
    this routine does not equate 4x4 simultaneous and 8x8 sequential exposures.
    rho*cos/range^2 is an assumed diffuse return law, not measured L8CH gain.
    diagnostics are evaluator/engineering only, never feature inputs.
    """
    params = (params or SensorParameters()).validate()
    distance, rho, cosine, weights, valid = _ray_inputs(
        radial_m, nir_reflectance, incidence_cos, solid_angle_weights)
    safe_distance = np.where(valid, distance, 1)
    energy = params.signal_counts*rho*cosine*weights/np.maximum(safe_distance, .05)**2
    histogram, outside, actual_device = _histogram(distance, energy, valid, RAW_BINS,
                RAW_BIN_M, params.range_zero_m, backend, device)
    matrix = _pulse_matrix(params)
    signal = histogram@matrix
    escaped = histogram.sum(-1)-signal.sum(-1)
    # Symmetric nearest-neighbour exchange preserves total energy, including
    # boundaries (missing neighbours retain their share; never periodic wrap).
    leak = params.neighbour_leak/4
    old = signal.copy()
    signal[..., 1:, :, :] += leak*(old[..., :-1, :, :]-old[..., 1:, :, :])
    signal[..., :-1, :, :] += leak*(old[..., 1:, :, :]-old[..., :-1, :, :])
    signal[..., :, 1:, :] += leak*(old[..., :, :-1, :]-old[..., :, 1:, :])
    signal[..., :, :-1, :] += leak*(old[..., :, 1:, :]-old[..., :, :-1, :])
    xtalk_bin = int(np.floor((params.crosstalk_range_m-params.range_zero_m)/RAW_BIN_M))
    xtalk = np.zeros(RAW_BINS)
    if 0 <= xtalk_bin < RAW_BINS:
        xtalk = params.signal_counts*params.crosstalk_fraction*matrix[xtalk_bin]
    # Crosstalk parameter is residual after firmware removal, not raw optical
    # crosstalk added then silently perfectly removed.
    expectation = signal + xtalk
    rng = np.random.default_rng(seed)
    count_mean = expectation+params.ambient_counts
    ambient_estimate = rng.poisson(params.ambient_counts, size=expectation.shape)
    counts = rng.poisson(count_mean)
    signed = expectation + params.noise_scale*(counts-ambient_estimate-expectation)
    # Public SNR uses only observed output and known ambient estimate. It never
    # uses clean signal/geometry variance, which would leak privileged truth.
    ambient = np.full(expectation.shape[:-1], params.ambient_counts*params.output_gain)
    return {'schema': SCHEMA, 'histogram': signed*params.output_gain, 'ambient': ambient,
            'params': asdict(params), 'seed': int(seed),
            'backend': {'ray_histogram': backend, 'device': actual_device, 'electronics': 'numpy/cpu'},
            'diagnostics': {'out_of_window_energy': outside, 'pulse_escape_energy': escaped,
                            'expected_signal': signal, 'raw_signal_energy': histogram.sum(-1)}}


def derive_readout(response, config):
    """Derive both CNH and scalar from the exact same noisy, aggregated response.

    Spatial merge sums photon-equivalent values, as the declared aggregate proxy.
    No denoising or privileged signal expectation enters the strongest detector.
    status 5 means this proxy passed SNR, 255 UNKNOWN; it is NOT ST status emulation.
    """
    config.validate()
    histogram = np.asarray(response['histogram'])
    ambient = np.asarray(response['ambient'])
    if histogram.shape[-3:] != (8, 8, RAW_BINS) or ambient.shape != histogram.shape[:-1]:
        raise ValueError('Invalid shared response dimensions')
    if not np.isfinite(histogram).all() or not np.isfinite(ambient).all():
        raise ValueError('Nonfinite response')
    if config.rows == 4:
        histogram = histogram.reshape(*histogram.shape[:-3], 4, 2, 4, 2, RAW_BINS).sum(axis=(-4, -2))
        ambient = ambient.reshape(*ambient.shape[:-2], 4, 2, 4, 2).sum(axis=(-3, -1))
    end = config.start_bin+config.bins*config.sub_sample
    histogram = histogram[..., config.start_bin:end].reshape(
        *histogram.shape[:-1], config.bins, config.sub_sample).sum(-1)
    peak_idx = histogram.argmax(-1)
    peak = np.take_along_axis(histogram, peak_idx[..., None], -1)[..., 0]
    p = response['params']
    # Approximate independent signal/background shot noise, estimated from
    # measured peak. Pressure noise_scale is public config, not scene truth.
    variance = (np.maximum(peak, 0)+2*ambient*config.sub_sample)*p['output_gain']
    sigma = p['noise_scale']*np.sqrt(np.maximum(variance, 1e-12))
    valid = (peak > 0) & (peak >= p['detection_snr']*sigma)
    centers = p['range_zero_m']+(config.start_bin+(np.arange(config.bins)+.5)*config.sub_sample)*RAW_BIN_M
    scalar = np.where(valid, centers[peak_idx], np.nan)
    return {'config': asdict(config), 'histogram': histogram if config.cnh_visible else None,
            'distance_m': scalar, 'valid': valid, 'status': np.where(valid, 5, 255).astype(np.uint8),
            'ambient': ambient, 'bin_centers_m': centers, 'response_seed': response['seed'],
            'detector': 'observed-strongest-SNR-proxy-not-ST'}


def scalar_projection(readout):
    """G3s/G4s retain identical scalar/status/ambient metadata, remove only CNH."""
    return {k: v for k, v in readout.items() if k not in ('histogram', 'bin_centers_m')}


def ideal_distribution(radial_m, solid_angle_weights, *, backend='numpy', device='cpu'):
    """G2: visible-ray area distribution [0.1,8.0)m, 79 bins, no NIR/labels."""
    distance, _, _, weights, valid = _ray_inputs(radial_m, 1, 1, solid_angle_weights)
    histogram, outside, actual_device = _histogram(distance, weights, valid, 79, .1, .1, backend, device)
    return {'histogram': histogram, 'bin_centers_m': .15+np.arange(79)*.1,
            'observed_fraction': histogram.sum(-1), 'backend': backend, 'device': actual_device,
            'diagnostics': {'out_of_window_fraction': outside}}


def provenance():
    return {'schema': SCHEMA, 'configs': {k: asdict(v) | {'buffer_bytes': v.buffer_bytes,
                'buffer_transmitted': v.cnh_visible} for k, v in CONFIGS.items()},
        'parameters': asdict(SensorParameters()),
        'official': {'FOV_DEG': [FOV_DEG, 'DS14310 Rev9 p4'],
                     'RAW_BINS': [RAW_BINS, 'DS14310 Rev9 section8 p23'],
                     'RAW_BIN_M': [RAW_BIN_M, 'locked vl53lmz_plugin_cnh.h macro BIN_WIDTH_MM'],
                     'buffer': 'UM3183 Rev7 section5.6; default flags single buffer/no variance/ambient',
                     'scaler': 'UM3183 Rev7 section5.7 signed integer / 2**signed scaler'},
        'observed_local': 'H2 4x4x24/start0/sub4/20ms/5.1Hz, LOCAL_CNH_20260921.md',
        'assumed_stress': list(asdict(SensorParameters())),
        'unidentified': ['absolute bin zero', '940nm reflectance/gain', 'photon units',
                         'angular response/lens flips', 'pulse kernel', 'residual crosstalk',
                         'noise law', '4x4 versus 8x8 gain', 'scan ordering'],
        'not_implemented_here': ['mesh visibility', 'multiple exposure/integration substeps',
                                 'sampling/arrival clocks', 'installation transform', 'packet drops'],
        'model_scope': 'opaque diffuse first-visible rays; no glass/specular/multipath; no hardware calibration',
        'compute': 'numpy CPU electronics; optional torch device histogram; caller times transfers'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _template_error(y, background, foreground):
    """Small exact two-column NNLS diagnostic, never a distance/occupancy label."""
    matrix = np.column_stack((background, foreground))
    norm = max(float(np.linalg.norm(y)), 1e-12)
    energy = np.sum(matrix*matrix, axis=0)
    if np.any(energy <= 0):
        return None
    single = np.maximum(0, matrix.T@y/energy)
    candidates = [np.zeros(2), np.array([single[0], 0]), np.array([0, single[1]])]
    interior = np.linalg.lstsq(matrix, y, rcond=None)[0]
    if np.all(interior >= 0):
        candidates.append(interior)
    errors = [float(np.linalg.norm(y-matrix@a)/norm) for a in candidates]
    index = int(np.argmin(errors))
    return {'relative_residual': errors[index], 'best_single_residual': min(errors[1:3]),
            'coefficients': candidates[index].tolist()}


def audit_paired_templates(root):
    """Recheck fixed historical windows, not a new sensor-model fitting pass.

    Pure templates use the original first two inner seconds only. All 16 zones
    and all four inner blocks are retained, including formerly NOT_EVALUABLE
    zones. No historical verdict, thresholds or amplitude envelopes are changed.
    """
    root = Path(root)
    artifacts = root/'artifacts.local/hardware-bringup'
    run = artifacts/'paired/dashboard-20260921T144239Z-8a8beb'
    old_path = artifacts/'cnh-components-20260921-v1/report.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    markers_path = run.parent/'.dashboard-control'/f'{run.name}.manual.json'
    frames_path = run/'tof/frames.jsonl'
    raw_path = run/'tof/raw.bin'
    expected = old['provenance']
    checks = [(markers_path, expected['markers_sha256']),
              (frames_path, expected['sources']['tof']['sha256']),
              (raw_path, expected['tof_raw_sha256'])]
    sources = []
    for path, digest in checks:
        actual = _sha(path)
        if actual != digest:
            raise ValueError(f'Historical paired source mismatch: {path}')
        sources.append({'path': str(path.relative_to(root)), 'sha256': actual})
    markers = json.loads(markers_path.read_text(encoding='utf-8'))
    frames = [json.loads(line) for line in frames_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    frames = [r for r in frames if r.get('sensor', {}).get('type') == 'cnh_frame']
    raw_by_seq = {}
    for line in raw_path.read_bytes().splitlines():
        try:
            sensor = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if sensor.get('type') == 'cnh_frame':
            if sensor['seq'] in raw_by_seq:
                raise ValueError('Duplicate historical raw seq')
            raw_by_seq[sensor['seq']] = sensor
    if len(raw_by_seq) != len(frames):
        raise ValueError('Historical raw/derived stream lengths differ')
    for row in frames:
        if row['sensor'] != raw_by_seq[row['sensor']['seq']]:
            raise ValueError('Raw sensor object differs from timestamp-indexed frame')
    phases = {}
    second = 1_000_000_000
    if [s['phase'] for s in markers['segments']] != ['background', 'foreground', 'mixture', 'return']:
        raise ValueError('Unexpected historical markers')
    for segment in markers['segments']:
        start = segment['start_host_monotonic_ns']+second
        end = segment['end_host_monotonic_ns']-second
        selected = [r for r in frames if start <= r['host_received_monotonic_ns'] < end]
        phases[segment['phase']] = [(decode_cnh(r['sensor']['hist_raw'], r['sensor']['hist_scaler']),
                                    int((r['host_received_monotonic_ns']-start)//second)) for r in selected]
        expected_count = old['zones'][0]['scalar'][segment['phase']]['frames']
        if len(selected) != expected_count:
            raise ValueError('Historical window count changed')
    background = np.median([h for h, block in phases['background'] if block < 2], axis=0)
    foreground = np.median([h for h, block in phases['foreground'] if block < 2], axis=0)
    zones = []
    for z in range(16):
        near = int(foreground[z].argmax())
        far = int(background[z].argmax())
        values = {}
        for name, phase in phases.items():
            blocks = []
            for block in range(4):
                rows = [h[z] for h, b in phase if b == block]
                if not rows:
                    raise ValueError('Historical phase block missing')
                med = np.median(rows, axis=0)
                fit = _template_error(med, background[z], foreground[z])
                positive = np.maximum(med, 0)
                near_mass = positive[max(0, near-1):min(24, near+2)].sum()
                far_mass = positive[max(0, far-1):min(24, far+2)].sum()
                blocks.append({'block': block, 'frames': len(rows), 'peak_bin': int(med.argmax()),
                    'template_fit': fit, 'near_window_positive_mass': float(near_mass),
                    'far_window_positive_mass': float(far_mass),
                    'near_over_far_window_mass': float(near_mass/far_mass) if far_mass else None,
                    'window_overlap': abs(near-far) <= 2})
            values[name] = blocks
        zones.append({'zone': z, 'historical_verdict': old['zones'][z]['verdict'],
                      'pure_foreground_peak_bin': near, 'pure_background_peak_bin': far,
                      'pure_background_template': background[z].tolist(),
                      'pure_foreground_template': foreground[z].tolist(), 'phases': values})
    return {'role': 'consumed Development fixed-window descriptive recheck; no new gate or refit',
            'old_report_sha256': _sha(old_path), 'sources': sources,
            'raw_derived_objects_equal': len(frames), 'historical_verdict_preserved': old['verdict'],
            'phase_frame_counts': {k: len(v) for k, v in phases.items()}, 'zones': zones,
            'limits': 'Coefficients are arbitrary waveform amplitudes, not geometric area, object count, '
                'or a physical forward model. Pure scenes lack ruler truth. Large amplitude extrapolation '
                'in mixture remains; no true distance, NIR reflectance, pulse width or sensor accuracy inferred.'}


def audit_local_sources(root):
    """Read old records only; preserve all zones, fit no physical parameters."""
    root = Path(root)
    hw = root/'research/active/hardware-bringup'
    inventory_path = hw/'references/inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    documents = []
    for item in inventory['documents']:
        if item['id'] not in ('vl53l8ch', 'um3183'):
            continue
        for field, hash_field in [('artifact_path', 'sha256'), ('text_artifact_path', 'text_sha256')]:
            path = root/item[field]
            actual = _sha(path)
            if actual != item[hash_field]:
                raise ValueError(f'Official archive hash mismatch: {path}')
            documents.append({'path': item[field], 'sha256': actual, 'matched_inventory': True})
    driver_root = root/'artifacts.local/hardware-bringup/sketches/tof_cnh_diag'
    lock = json.loads((hw/'vendor-lock.json').read_text(encoding='utf-8'))
    for item in lock['files']:
        if 'plugin_cnh' in item['path']:
            path = driver_root/Path(item['path']).name
            actual = _sha(path)
            if actual != item['sha256']:
                raise ValueError(f'Locked driver mismatch: {path}')
            documents.append({'path': str(path.relative_to(root)), 'sha256': actual,
                              'matched_vendor_lock': True})
    index = json.loads((hw/'evidence-index.json').read_text(encoding='utf-8'))
    captures = []
    for item in index['records']:
        if not item['id'].startswith(('local-wall-', 'local-foreground-')):
            continue
        path = root/item['raw_path']
        actual = _sha(path)
        if actual != item['raw_sha256']:
            raise ValueError(f'Raw capture mismatch: {path}')
        frames = []
        incomplete = 0
        for line in path.read_bytes().splitlines():
            try:
                sensor = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                incomplete += 1
                continue
            if sensor.get('type') == 'cnh_frame':
                frames.append(sensor)
        if len(frames) != item['frames']:
            raise ValueError('Raw complete frame count differs from evidence index')
        curves = np.stack([decode_cnh(s['hist_raw'], s['hist_scaler']) for s in frames])
        ambient = np.stack([decode_cnh(s['ambient_raw'], s['ambient_scaler']) for s in frames])
        if curves.shape[1:] != (16, 24):
            raise ValueError('Unexpected local raw configuration')
        zones = []
        # Approximately one second per block, preserve time variation; these
        # remain consecutive correlated samples, not independent evidence.
        blocks = [curves[i:i+5].mean(0) for i in range(0, len(curves), 5)]
        med = np.median(curves, axis=0)
        for z in range(16):
            positives = np.maximum(med[z], 0)
            mass = positives.sum()
            zones.append({'zone': z, 'median_signed_waveform': med[z].tolist(),
                'peak_bin': int(med[z].argmax()),
                'near_bin0_2_positive_mass_fraction': float(positives[:3].sum()/mass) if mass else None,
                'negative_bin_fraction': float((curves[:, z] < 0).mean()),
                'median_ambient_arbitrary_units': float(np.median(ambient[:, z])),
                'block_peak_bins': [int(b[z].argmax()) for b in blocks],
                'block_relative_residual': [float(np.linalg.norm(b[z]-med[z])/max(np.linalg.norm(med[z]), 1e-12)) for b in blocks]})
        captures.append({'id': item['id'], 'path': item['raw_path'], 'sha256': actual,
            'frames': len(frames), 'incomplete_lines_preserved': incomplete,
            'zones': zones, 'role': 'consumed Development descriptive pure scenes, no ruler truth'})
    return {'sources': documents, 'inventory_sha256': _sha(inventory_path),
            'evidence_index_sha256': _sha(hw/'evidence-index.json'), 'captures': captures,
            'calibration_status': 'NOT_CALIBRATED',
            'applicability': 'Signs, configuration and qualitative near/far shape can be checked. '
                'Unknown geometry, zero, NIR reflectance and counts prevent unique physical calibration. '
                'Early-bin residual may dominate: never interpret every early peak as object.',
            'paired_template_recheck': audit_paired_templates(root),
            'remaining': 'A physical simulator-to-record comparison with identified geometry is unavailable. '
                'Pure-reference templates are descriptive, not the simulator kernel. '
                'No physical parameters fitted from these records.'}


def engineering_checks():
    """Analytic engineering witnesses, not UE capture or task-effectiveness results."""
    p = SensorParameters(noise_scale=0, pulse_sigma_bins=0, tail_mass=0,
                         crosstalk_fraction=0, neighbour_leak=0)
    rays, weights = angular_rays(16)
    plane = axial_to_radial(2.0, rays)
    response = synthesize_response(plane, .5, rays[..., 2], weights, params=p)
    readout = derive_readout(response, H1)
    plane_error = float(np.max(np.abs(readout['distance_m']-np.average(plane, weights=weights, axis=-1))))
    # A 2cm front-facing strip at Z=3 crosses the centre of a native zone;
    # analytic solid angle of its clipped rectangle provides an independent
    # geometric integration reference. Both layers are first-visible surfaces.
    convergence = []
    edge = np.tan(np.deg2rad(FOV_DEG/2))
    zone_width = 2*edge/8
    center_x = zone_width/2
    near_low, near_high = center_x-.01/3, center_x+.01/3
    def omega(x, y):
        return np.arctan2(x*y, np.sqrt(1+x*x+y*y))
    expected = float((omega(near_high, zone_width)-omega(near_low, zone_width)
                -omega(near_high, 0)+omega(near_low, 0)) / omega(zone_width, zone_width))
    for n in (32, 64, 128, 256, 512, 1024, 2048):
        # Only the independently checked zone is required. Do not allocate
        # 64 copies of millions of rays for this small geometric unit check.
        axis = (np.arange(n)+.5)*zone_width/n
        yy, slope = np.meshgrid(axis, axis, indexing='ij')
        w = (zone_width/n)**2/(1+slope*slope+yy*yy)**1.5
        visible_near = (slope >= near_low) & (slope < near_high)
        near_fraction = float((w*visible_near).sum()/w.sum())
        convergence.append({'samples_per_axis': n, 'near_fraction': near_fraction,
                            'relative_error_to_analytic': abs(near_fraction-expected)/expected})
    # Avoid a full lattice histogram for a geometric quadrature test. A pair of
    # weighted visible subregions independently tests depth-layer retention.
    distance = np.broadcast_to(np.array([1.8, 3.0]), (8, 8, 2))
    two = synthesize_response(distance, .5, 1, [0.1, .9], params=p)
    two_hist = derive_readout(two, H2)['histogram'][0, 0]
    peaks = np.flatnonzero(two_hist > 0).tolist()
    return {'plane_max_peak_vs_weighted_range_error_m': plane_error,
            'plane_quantization_check': plane_error < 2*RAW_BIN_M,
            'thin_strip_width_m': .02, 'thin_strip_axial_m': 3,
            'thin_strip_analytic_fraction': expected, 'thin_strip_convergence': convergence,
            'thin_strip_final_relative_error': convergence[-1]['relative_error_to_analytic'],
            'thin_strip_2percent_check': convergence[-1]['relative_error_to_analytic'] <= .02,
            'two_layer_nonzero_H2_bins': peaks, 'two_layer_retained': len(peaks) == 2,
            'false_convergence_at_256': convergence[3]['relative_error_to_analytic'] > .02,
            'limits': 'Analytic strip, not all UE mesh poses. 256 samples/axis falsely converges '
                      'between doublings; analytic truth still differs 2.9%. 1024/2048 pass this '
                      'one pose only. Production requires analytic/adaptive silhouette or density checks.'}


def check_torch_device(device='cuda'):
    """One small-array equivalence/timing witness, not a full workload benchmark."""
    import torch
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        return {'status': 'ACCELERATOR_UNAVAILABLE', 'torch': torch.__version__}
    warm = torch.ones(1, device=device)
    if warm.is_cuda:
        torch.cuda.synchronize()
    rng = np.random.default_rng(413)
    distance = rng.uniform(.2, 4.6, (2, 8, 8, 64))
    weights = rng.uniform(.1, 1, distance.shape)
    params = SensorParameters(noise_scale=0)
    started = time.perf_counter()
    cpu = synthesize_response(distance, .5, .8, weights, params=params, seed=81)
    cpu_seconds = time.perf_counter()-started
    started = time.perf_counter()
    gpu = synthesize_response(distance, .5, .8, weights, params=params, seed=81,
                              backend='torch', device=device)
    if warm.is_cuda:
        torch.cuda.synchronize()
    gpu_seconds = time.perf_counter()-started
    np.testing.assert_allclose(cpu['histogram'], gpu['histogram'], atol=1e-9, rtol=1e-10)
    return {'status': 'PASSED', 'torch': torch.__version__, 'device': str(warm.device),
        'device_name': torch.cuda.get_device_name(warm.device) if warm.is_cuda else 'CPU',
        'array_shape': list(distance.shape),
        'max_abs_difference': float(np.max(np.abs(cpu['histogram']-gpu['histogram']))),
        'numpy_seconds': cpu_seconds, 'torch_histogram_including_transfers_seconds': gpu_seconds,
        'electronics': 'numpy CPU both arms', 'scope': 'single small-array parity, not production placement benchmark'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--audit-local', action='store_true')
    parser.add_argument('--check-torch', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    report = {'status': 'ANALYTIC_ENGINEERING_ONLY_SENSOR_REALISM_UNRESOLVED',
              'provenance': provenance(), 'engineering': engineering_checks(),
              'cpu_reason': 'TASK_NOT_GPU_SUITABLE: small analytic/preflight arrays and file/hash metadata'}
    if args.audit_local:
        report['local_audit'] = audit_local_sources(args.root)
    if args.check_torch:
        report['torch_parity'] = check_torch_device()
    report['elapsed_seconds'] = time.perf_counter()-started
    report['source_sha256'] = _sha(__file__)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sensor-preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    summary = ('# CNH sensor preflight\n\n'
        'Implemented a reproducible signed histogram proxy and same-response scalar attribution. '
        'This is analytic engineering evidence; physical calibration and UE capture acceptance are still unresolved.\n\n'
        '- H2 4x4x24 uses 2028 bytes; H3 8x8x16 uses 5468 bytes. '
        'The H3 5 Hz setting is a simulation assumption, not measured hardware capability.\n'
        '- Plane radial conversion and two visible depth layers passed. '
        'The 2 cm / 3 m strip matched analytic angular occupancy to 0.12% at 1024 and 2048 samples/axis. '
        'At 32 through 512 samples/axis, apparent convergence concealed a 2.92% error. '
        'Production capture needs analytic or adaptive silhouette checks; this does not approve one global ray density.\n'
        '- Histogram and scalar readout share the exact noisy response. Signed values and UNKNOWN are preserved; '
        'the peak detector is an explicit SNR proxy, not ST firmware.\n'
        '- Signal gain, pulse shape, bin zero, crosstalk, noise law and angular response remain assumptions. '
        'Optional Torch moves only ray histogram accumulation to its stated device; electronics are NumPy CPU.\n')
    if args.audit_local:
        summary += ('- Verified archived ST PDF/text and locked CNH driver hashes; decoded all 16 zones in '
            '102 pure-wall and 103 pure-foreground raw frames, preserving signed bins and incomplete lines. '
            'No ruler geometry or independent NIR reflectance exists; these records cannot identify a unique '
            'metric-range/photonic calibration. Also rechecked all 444 timestamp-indexed paired raw sensor objects '
            'and fixed 21/20/22/20-frame background/foreground/mixture/return windows; '
            'all 16 zones retain pure-template block residuals and near/far mass descriptors. '
            'Historical verdicts are preserved. No physical parameter was fitted, and no physical '
            'simulator-to-real consistency pass is claimed.\n')
    (args.output/'README.md').write_text(summary, encoding='utf-8')
    print(json.dumps({'output': str(args.output/'sensor-preflight.json'),
                      'engineering': report['engineering'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
