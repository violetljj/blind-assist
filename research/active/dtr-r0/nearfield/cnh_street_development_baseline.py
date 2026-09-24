"""CPU Development interface smoke from SceneDepth; no admission or accuracy claim.

Uniform reflectance and incidence are deliberately assumed placeholders. Output
is neither calibrated CNH nor hardware multi-return, and never declares clearance.
Evaluator labels/instance IDs/geometry are not read by the baseline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cnh_route_sensor import angular_rays, axial_to_radial, synthesize_response, derive_readout, H3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_depth(folder):
    """Prefer canonical EXR; transport is a compatibility fallback only."""
    folder = Path(folder)
    canonical = folder/'depth_left.exr'
    if canonical.exists():
        import OpenEXR
        with OpenEXR.File(str(canonical), separate_channels=True) as image:
            depth = image.channels()['Z'].pixels.copy()
        mask = np.load(folder/'depth_left_valid.npy', allow_pickle=False)
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError('Canonical depth validity mask differs')
        return np.where(mask, depth, np.nan), canonical
    transport = folder/'depth_left.transport.npy'
    depth = np.load(transport, allow_pickle=False)
    return np.where(np.isfinite(depth) & (depth > 0) & (depth < 100), depth, np.nan), transport


def sample_depth(depth, camera, samples_per_axis=16):
    depth = np.asarray(depth)
    if depth.shape != (camera['height'], camera['width']) or depth.ndim != 2:
        raise ValueError('Depth shape differs from camera')
    k = np.asarray(camera['K'], dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics')
    rays, weights = angular_rays(samples_per_axis)
    x = np.rint(k[0, 0]*rays[..., 0]/rays[..., 2]+k[0, 2]).astype(int)
    y = np.rint(k[1, 1]*rays[..., 1]/rays[..., 2]+k[1, 2]).astype(int)
    if np.any((x < 0) | (x >= depth.shape[1]) | (y < 0) | (y >= depth.shape[0])):
        raise ValueError('Camera does not cover the declared 45 degree square cone')
    axial = depth[y, x].astype(float)
    axial = np.where(np.isfinite(axial) & (axial > 0), axial, np.nan)
    pixel_rays = np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones_like(x)), axis=-1)
    pixel_rays /= np.linalg.norm(pixel_rays, axis=-1, keepdims=True)
    return axial_to_radial(axial, pixel_rays), weights


def fixed_peaks(readout, params, max_returns=2):
    """Up to two local maxima passing the existing observed-response SNR rule.

    Plateau tie goes to its first bin. The first return is strongest, not nearest.
    This fixed detector does not inspect known geometry or expected signal.
    """
    h = np.asarray(readout['histogram'])
    ambient = np.asarray(readout['ambient'])
    variance = (np.maximum(h, 0)+2*ambient[..., None]*H3.sub_sample)*params['output_gain']
    sigma = params['noise_scale']*np.sqrt(np.maximum(variance, 1e-12))
    left = np.concatenate([np.full_like(h[..., :1], -np.inf), h[..., :-1]], axis=-1)
    right = np.concatenate([h[..., 1:], np.full_like(h[..., :1], -np.inf)], axis=-1)
    eligible = (h > left) & (h >= right) & (h > 0) & (h >= params['detection_snr']*sigma)
    order = np.argsort(-np.where(eligible, h, -np.inf), axis=-1, kind='stable')[..., :max_returns]
    valid = np.take_along_axis(eligible, order, -1)
    return np.where(valid, np.asarray(readout['bin_centers_m'])[order], np.nan), valid


def run(capture, output, max_frames=2, seed=20260924):
    capture = Path(capture).resolve(strict=True)
    output = Path(output)
    if max_frames < 1:
        raise ValueError('Positive bounded frame limit required')
    manifest_path = capture/'raw-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if str(manifest.get('split', '')).lower() in ('test', 'locked_test', 'blind'):
        raise ValueError('Protected test data prohibited')
    format_path = capture/'format-receipt.json'
    transport = json.loads(format_path.read_text(encoding='utf-8-sig'))
    if transport.get('status') != 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Finalized seven-pass source transport required')
    rows = manifest['frames'][:max_frames]
    if not rows:
        raise ValueError('No captured frames')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='cnh-street-development-baseline-v1', status='RUNNING',
        evidence='DEVELOPMENT_INTERFACE_SMOKE_ONLY', source_manifest_sha256=sha(manifest_path),
        source_format_sha256=sha(format_path),
        code_sha256=sha(__file__), sensor_code_sha256=sha(Path(__file__).with_name('cnh_route_sensor.py')),
        assumptions=dict(nir_reflectance=0.5, incidence_cos=1.0,
            timing='STATIC_SINGLE_EXPOSURE_NO_8X8_SEQUENTIAL_TIMING',
            sampling='NEAREST_PIXEL_16X16_PER_ZONE_NOT_CONVERGENCE_VALIDATED'),
        clearance='UNKNOWN', geometry_admission='NOT_EVALUATED', energy_convergence='NOT_EVALUATED',
        label_precision='NOT_EVALUATED', asset_isolation='NOT_EVALUATED',
        hardware_validity='NOT_ESTABLISHED', performance_metrics=None, frames=[])
    try:
        for index, row in enumerate(rows):
            if str(row.get('split', '')).lower() in ('test', 'locked_test', 'blind'):
                raise ValueError('Protected test data prohibited')
            folder = (capture/row['folder']).resolve(strict=True)
            if not folder.is_relative_to(capture):
                raise ValueError('Frame folder escapes capture')
            camera_path = folder/'camera.json'
            camera = json.loads(camera_path.read_text(encoding='utf-8-sig'))
            depth, depth_path = load_depth(folder)
            radial, weights = sample_depth(depth, camera)
            response = synthesize_response(radial, .5, 1., weights, seed=seed+index)
            readout = derive_readout(response, H3)
            peaks, valid = fixed_peaks(readout, response['params'])
            target = output/f'frame-{index:06d}.npz'
            np.savez_compressed(target, histogram=readout['histogram'], ambient=readout['ambient'],
                bin_centers_m=readout['bin_centers_m'], single_distance_m=readout['distance_m'],
                single_status=readout['status'], multi_distance_m=peaks, multi_valid=valid)
            receipt['frames'].append(dict(frame_id=row['id'], layout_id=row['layout_id'],
                depth_file=depth_path.name, depth_sha256=sha(depth_path), camera_sha256=sha(camera_path), seed=seed+index,
                depth_valid_sha256=sha(folder/'depth_left_valid.npy') if depth_path.suffix=='.exr' else None,
                output=target.name, output_sha256=sha(target),
                single_valid_zones=int(readout['valid'].sum()), two_peak_zones=int(valid.all(-1).sum())))
        receipt['status'] = 'PASS_INTERFACE_SMOKE_ONLY'
    except Exception as error:
        receipt.update(status='FAIL_INTERFACE_SMOKE', error=str(error))
        raise
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.capture, args.output, args.max_frames), indent=2))
