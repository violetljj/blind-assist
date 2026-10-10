"""Causal signed native paths around current observed candidates, Development.

Public sensor motion backtraces a static-world bin-center hypothesis. It is
not target tracking or a visibility guarantee. The static arm uses the same
kernel/window and preserves the current native cell in all historical frames.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import cnh_graded_corridor_features_dev as C
import cnh_graded_peak_tracks_dev as P

ROOT = C.ROOT
OUTPUT = ROOT / 'artifacts.local/work/cnh-graded-motion-path-dev-20261010/features'
WINDOW = 8
TOP = 8
METRICS = ('current_signed_z', 'signed_mean', 'independent_noise_sum_z',
           'positive_mass', 'negative_mass', 'maximum_signed_z',
           'minimum_signed_z', 'signed_std', 'positive_frame_fraction',
           'visible_frames', 'visible_span', 'signed_slope_per_frame',
           'current_minus_signed_mean', 'kernel_square_sum',
           'current_public_membership', 'current_public_inner_share')
NAMES = tuple(f'rank{rank + 1}_{name}' for rank in range(TOP) for name in METRICS)


def sensor_centers():
    edge = float(np.tan(np.pi / 8))
    slope = -edge + (np.arange(8, dtype=np.float64) + .5) * (2 * edge / 8)
    yy, xx = np.meshgrid(slope, slope, indexing='ij')
    rays = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(64, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    radius = (np.arange(16) + .5) * float(C.WIDTH)
    return (rays[:, None] * radius[None, :, None]).reshape(1024, 3)


def trilinear(points):
    """Fixed cell-center interpolation; edge cells extend to physical FOV edge.

    An observation outside angular FOV, positive-z halfspace or radial range is
    missing. Inside a boundary half-cell, duplicate clamped neighbors are merged
    when computing sum(kernel squared), avoiding overstated noise reduction.
    """
    edge = float(np.tan(np.pi / 8))
    z = points[..., 2]
    sx = np.divide(points[..., 0], z, out=np.zeros_like(z), where=z != 0)
    sy = np.divide(points[..., 1], z, out=np.zeros_like(z), where=z != 0)
    radius = np.linalg.norm(points, axis=-1)
    valid = ((z > 0) & (abs(sx) <= edge) & (abs(sy) <= edge)
             & (radius >= 0) & (radius < 16 * float(C.WIDTH)))
    coords = np.stack(((sy + edge) * 8 / (2 * edge) - .5,
                       (sx + edge) * 8 / (2 * edge) - .5,
                       radius / float(C.WIDTH) - .5), -1)
    # Exact current native centers should have exactly a one-cell kernel.
    near = np.rint(coords)
    coords = np.where(abs(coords - near) < 1e-10, near, coords)
    low = np.floor(coords).astype(np.int64)
    frac = coords - low
    indices, weights = [], []
    for dy in (0, 1):
        for dx in (0, 1):
            for dr in (0, 1):
                offset = np.array((dy, dx, dr))
                cells = np.clip(low + offset, 0, np.array((7, 7, 15)))
                idx = (cells[..., 0] * 8 + cells[..., 1]) * 16 + cells[..., 2]
                w = np.prod(np.where(offset, frac, 1 - frac), -1)
                indices.append(idx.astype(np.int16))
                weights.append(np.where(valid, w, 0).astype(np.float32))
    index = np.stack(indices, -1)
    weight = np.stack(weights, -1)
    square = np.zeros(valid.shape, np.float32)
    for k in range(8):
        square += weight[..., k] ** 2
        for j in range(k):
            square += 2 * weight[..., k] * weight[..., j] * (index[..., k] == index[..., j])
    return index, weight, valid, square, coords.astype(np.float32)


def make_maps(sensor):
    centers = sensor_centers()
    inverse = np.linalg.inv(sensor.astype(np.float64))
    maps = {}
    for arm in ('aligned', 'static'):
        indices, weights, valid, square, coords, history = [], [], [], [], [], []
        for frame in C.FRAMES:
            h = np.arange(int(frame) - WINDOW + 1, int(frame) + 1)
            available = h >= 0
            if arm == 'aligned':
                transforms = inverse[np.maximum(h, 0)] @ sensor[int(frame)].astype(np.float64)
                points = np.einsum('tij,bj->tbi', transforms[:, :3, :3], centers)
                points += transforms[:, None, :3, 3]
            else:
                points = np.broadcast_to(centers, (WINDOW, *centers.shape))
            idx, w, v, s, c = trilinear(points)
            v &= available[:, None]
            w *= available[:, None, None]
            s *= available[:, None]
            indices.append(idx); weights.append(w); valid.append(v)
            square.append(s); coords.append(c); history.append(h)
        for key, values in (('index', indices), ('weight', weights), ('visible', valid),
                            ('kernel_square', square), ('coordinates', coords), ('history_frame', history)):
            maps[f'{arm}_{key}'] = np.stack(values)
    return maps


def extract(normalized, candidates, maps, arm):
    """Candidate rank stays separate; read normalized signed evidence, no labels."""
    raw = normalized.astype(np.float32).reshape(len(normalized), 16, 1024)
    n = len(raw)
    path = np.full((n, 13, 2, TOP, WINDOW), np.nan, np.float32)
    visibility = np.zeros(path.shape, bool)
    squares = np.zeros(path.shape, np.float32)
    for fj in range(13):
        seed = np.maximum(candidates['native_index'][:, fj], 0)
        for t in range(WINDOW):
            h = int(maps[f'{arm}_history_frame'][fj, t])
            if h < 0:
                continue
            index = maps[f'{arm}_index'][fj, t][seed]
            weight = maps[f'{arm}_weight'][fj, t][seed]
            valid = maps[f'{arm}_visible'][fj, t][seed] & candidates['valid'][:, fj]
            sampled = (raw[np.arange(n)[:, None, None, None], h, index] * weight).sum(-1)
            path[:, fj, :, :, t] = np.where(valid, sampled, np.nan)
            visibility[:, fj, :, :, t] = valid
            squares[:, fj, :, :, t] = np.where(valid, maps[f'{arm}_kernel_square'][fj, t][seed], 0)
    count = visibility.sum(-1)
    den = np.maximum(count, 1)
    values = np.where(visibility, path, 0)
    signed_sum = values.sum(-1)
    mean = signed_sum / den
    kernel_square_sum = squares.sum(-1)
    noise_sum_z = signed_sum / np.sqrt(np.maximum(kernel_square_sum, 1e-20))
    positive = np.maximum(values, 0).sum(-1)
    negative = np.maximum(-values, 0).sum(-1)
    maximum = np.where(visibility, path, -np.inf).max(-1)
    minimum = np.where(visibility, path, np.inf).min(-1)
    std = np.sqrt(((np.where(visibility, path - mean[..., None], 0)) ** 2).sum(-1) / den)
    fraction = ((path > 0) & visibility).sum(-1) / den
    times = np.arange(WINDOW, dtype=np.float32)
    first = np.where(visibility, times, np.inf).min(-1)
    last = np.where(visibility, times, -np.inf).max(-1)
    span = np.where(count > 0, last - first + 1, 0)
    mean_t = (visibility * times).sum(-1) / den
    centered_t = np.where(visibility, times - mean_t[..., None], 0)
    slope_den = (centered_t ** 2).sum(-1)
    slope = np.divide((centered_t * values).sum(-1), slope_den,
                      out=np.full(mean.shape, np.nan, np.float32), where=slope_den > 0)
    current = path[..., -1]
    features = np.stack((current, mean, noise_sum_z, positive, negative, maximum, minimum,
                         std, fraction, count, span, slope, current - mean, kernel_square_sum,
                         candidates['membership'], candidates['inner_share']), -1).astype(np.float32)
    features = np.where(candidates['valid'][..., None] & np.isfinite(features), features, np.nan)
    flat = features.reshape(n, 13, 2, -1)
    return dict(features=flat, valid=np.isfinite(flat), path_values=path,
                path_visible=visibility, path_kernel_square=squares)


def run(output):
    start = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((ROOT / 'artifacts.local').resolve()):
        raise ValueError('Use canonical artifacts.local')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve prior feature run')
    output.mkdir(parents=True, exist_ok=True)
    phase = 'inputs'
    try:
        bias = np.load(C.BIAS).astype(np.float32)
        inputs, summaries = {}, {}
        for split in ('cal', 'validation'):
            directory = C.SOURCE / 'data' / split
            with np.load(directory / 'geometry.npz', allow_pickle=False) as a:
                sensor, query, ids, uids = (a[k] for k in ('sensor', 'public_query', 'scene_ids', 'scene_uids'))
            with np.load(directory / 'physics.npz', allow_pickle=False) as a:
                ambient = a['ambient']
                np.testing.assert_array_equal(a['sensor'], sensor)
                np.testing.assert_array_equal(a['public_query'], query)
            maps = make_maps(sensor)
            np.savez_compressed(output / f'{split}_maps.npz', **maps, sensor=sensor,
                                public_query=query, sensor_centers=sensor_centers())
            cache = P.OUTPUT / f'{split}_tracks.npz'
            with np.load(cache, allow_pickle=False) as a:
                candidates = {key: a[f'candidate_{key}'].reshape(1536, 13, 2, *a[f'candidate_{key}'].shape[4:])
                              for key in ('native_index', 'valid', 'membership', 'inner_share')}
                np.testing.assert_array_equal(a['scene_ids'], ids)
                np.testing.assert_array_equal(a['scene_uids'], uids)
            hist = np.load(directory / 'hist.npy', mmap_mode='r', allow_pickle=False)
            assert hist.shape == (384, 4, 16, 8, 8, 16)
            chunks = {'aligned': [], 'static': []}
            for begin in range(0, 384, 16):
                if time.monotonic() - start >= 175:
                    raise TimeoutError('Extraction command wall budget180s reached')
                phase = f'{split} batch {begin}'
                normalized = C.normalize(np.array(hist[begin:begin + 16], copy=True), ambient, bias).reshape(-1, 16, 8, 8, 16)
                c = {key: value[begin * 4:(begin + 16) * 4] for key, value in candidates.items()}
                for arm in chunks:
                    chunks[arm].append(extract(normalized, c, maps, arm))
            merged = {arm: {key: np.concatenate([item[key] for item in batches]).reshape(384, 4, *batches[0][key].shape[1:])
                            for key in batches[0]} for arm, batches in chunks.items()}
            phase = f'{split} parity and prefix'
            public = P.native_geometry(sensor, query)
            prefix = []
            for i, replica, through in ((0, 0, 7), (383, 3, 11)):
                normalized = C.normalize(np.array(hist[i, replica], copy=True), ambient, bias)[None]
                own = P.current_candidates(normalized, public)
                row = i * 4 + replica
                for key, cached in candidates.items():
                    np.testing.assert_array_equal(own[key][0], cached[row])
                altered = normalized.copy(); altered[:, through + 1:] = -10
                changed = P.current_candidates(altered, public)
                for arm in chunks:
                    checked = extract(altered, changed, maps, arm)
                    for key in checked:
                        np.testing.assert_array_equal(checked[key][0, :through - 2], merged[arm][key][i, replica, :through - 2])
                prefix.append(dict(scene=int(ids[i]), replica=replica, through_frame=through, exact=True))
            with np.load(directory / 'rows.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_id'], np.repeat(ids, 4 * 13))
                np.testing.assert_array_equal(a['replica'], np.tile(np.repeat(np.arange(4), 13), 384))
                np.testing.assert_array_equal(a['frame'], np.tile(C.FRAMES, 384 * 4))
            arrays = {arm: values['features'] for arm, values in merged.items()}
            arrays.update({f'{arm}_valid': values['valid'] for arm, values in merged.items()})
            arrays.update({f'{arm}_{key}': value for arm, values in merged.items() for key, value in values.items()
                           if key not in ('features', 'valid')})
            arrays.update({f'candidate_{key}': value.reshape(384, 4, *value.shape[1:]) for key, value in candidates.items()})
            np.savez_compressed(output / f'{split}_features.npz', **arrays, names=np.array(NAMES),
                                scene_ids=ids, scene_uids=uids, frames=C.FRAMES, queries=np.array(['HEAD', 'BODY']),
                                replica=np.arange(4), cache_row_index=np.arange(19968).reshape(384, 4, 13))
            # The current sample uses native identity under both motion choices.
            np.testing.assert_array_equal(merged['aligned']['path_values'][..., -1], merged['static']['path_values'][..., -1])
            summaries[split] = dict(shape=list(arrays['aligned'].shape), future_prefix=prefix,
                valid_candidates=int(candidates['valid'].sum()),
                aligned_visible_path_frames=int(merged['aligned']['path_visible'].sum()),
                static_visible_path_frames=int(merged['static']['path_visible'].sum()),
                current_path_parity='exact', candidate_cache_parity='representative exact')
            inputs[split] = {str(path): C.sha(path) for path in
                             (directory / 'hist.npy', directory / 'geometry.npz', directory / 'physics.npz', directory / 'rows.npz', cache)}
            del hist
        C.save(output / 'schema.json', dict(names=NAMES, metrics=METRICS, window=WINDOW, ranks=TOP,
            shape_axes=['scene384', 'replica4', 'frame13', 'query2', 'feature128'],
            source_sha256=C.sha(__file__), bias_sha256=C.sha(C.BIAS), inputs_sha256=inputs,
            public_mapping='inv(sensor[h]) @ sensor[f] applied to current native-bin center; static arm identity; h<=f only',
            interpolation='Fixed trilinear in angular slopes y/x and radial centers; edge clamping inside physical FOV; duplicated corners merged for kernel variance',
            history='Past<=8 inclusive current, includes f0..2 at early outputs; left padding missing',
            noise_statistic='sum signed normalized samples / sqrt(sum over time of sum merged cell kernel squares); temporal/bin covariance ignored, diagnostic only',
            missing='NaN and valid false for unavailable candidate or undefined slope; path NaN for absent history or FOV/range loss',
            hypotheses='Static-world current native center, public sensor motion; neither object identity nor true target support; weak negative fluctuations retained',
            backend='CPU vectorized 8-corner gather, bounded small fixed native tensors; no model inference or targettruth input'))
        C.save(output / 'result.json', dict(status='COMPLETE', seconds=time.monotonic() - start,
            source_sha256=C.sha(__file__), summaries=summaries, physical_output=str(output)))
    except BaseException as error:
        C.save(output / f'failure_{time.time_ns()}.json', dict(phase=phase, error=repr(error),
               traceback=traceback.format_exc(), seconds=time.monotonic() - start))
        raise
    finally:
        C.save(output / 'execution_receipt.json', dict(seconds=time.monotonic() - start,
            command=[sys.executable, *sys.argv], source_sha256=C.sha(__file__), GPU_seconds=0,
            outputs_sha256={p.name: C.sha(p) for p in output.iterdir() if p.is_file() and p.name != 'execution_receipt.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run(parser.parse_args().output)
