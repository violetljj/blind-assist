"""Existing alley Development depth inputs and visible-witness audit.

The depth arrays are first-visible SceneDepth only. The label authority is the
separate physical triangle/box intersection; this code never reads triangles
or instance IDs. It cannot turn invisible surfaces into an observable oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cnh_rgb_dev_comparison import checked, read_inputs, sha
from cnh_rgb_alley_v2 import COLLECTION_SHA256, PARTITION_SHA256, QUERY_NAMES
from cnh_street_development_baseline import sample_depth
from cnh_street_e2e_materialize import BOXES
from cnh_route_sensor import RAW_BIN_M, H3


def visible_witness(depth: np.ndarray, camera: dict) -> np.ndarray:
    """Six booleans: a first-visible pixel centre lies inside each closed box."""
    height, width = depth.shape
    if (width, height) != (camera['width'], camera['height']):
        raise ValueError('Depth/camera dimensions differ')
    k = np.asarray(camera['K'], dtype=np.float64)
    y, x = np.indices(depth.shape)
    z = depth
    x_m = (x - k[0, 2]) * z / k[0, 0]
    y_m = (y - k[1, 2]) * z / k[1, 1]
    valid = np.isfinite(z) & (z > 0)
    return np.asarray([
        np.any(valid & (z >= lo[2]) & (z <= hi[2]) &
               (x_m >= lo[0]) & (x_m <= hi[0]) &
               (y_m >= lo[1]) & (y_m <= hi[1]))
        for lo, hi in BOXES], dtype=np.bool_)


def perfect_h3_histogram(depth: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted, noise-free visible-ray distribution on H3's 16 bins.

    Out-of-window or invalid rays retain their area in the denominator.
    The extra nearest-visible radial scalar is derived only from these rays.
    """
    radial, weights = sample_depth(depth, camera, 16)
    if radial.shape != (8, 8, 256) or weights.shape != radial.shape:
        raise ValueError('Unexpected frozen 16x16 angular sample shape')
    width = RAW_BIN_M * H3.sub_sample
    ids = np.floor(np.where(np.isfinite(radial), radial, -1) / width).astype(np.int32)
    good = np.isfinite(radial) & (radial >= 0) & (ids >= 0) & (ids < H3.bins)
    histogram = np.zeros((64, H3.bins), dtype=np.float32)
    flat_weights = np.broadcast_to(weights, radial.shape).reshape(64, -1)
    flat_ids = ids.reshape(64, -1)
    flat_good = good.reshape(64, -1)
    for zone in range(64):
        histogram[zone] = np.bincount(flat_ids[zone, flat_good[zone]],
                                      weights=flat_weights[zone, flat_good[zone]],
                                      minlength=H3.bins).astype(np.float32)
    total = flat_weights.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError('Nonpositive angular area')
    histogram /= total[:, None]
    nearest = np.where(good, radial, np.inf).reshape(64, -1).min(axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, np.nan).astype(np.float32)
    valid_fraction = good.reshape(64, -1).sum(axis=1).astype(np.float32) / 256
    return histogram, nearest, valid_fraction


def load_scene_depth(frame: dict) -> tuple[np.ndarray, dict]:
    # Every source identity comes from the immutable train/dev overlay.
    depth_path = checked(frame['original_files']['depth_left.exr'])
    mask_path = checked(frame['original_files']['depth_left_valid.npy'])
    camera_path = checked(frame['original_files']['camera.json'])
    import OpenEXR
    with OpenEXR.File(str(depth_path), separate_channels=True) as image:
        depth = image.channels()['Z'].pixels.copy()
    mask = np.load(mask_path, allow_pickle=False)
    camera = json.loads(camera_path.read_text(encoding='utf-8-sig'))
    if depth.shape != (360, 640) or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError('Unexpected canonical depth or validity shape')
    if not np.array_equal(mask, np.isfinite(depth) & (depth > 0) & (depth < 100)):
        raise ValueError('EXR and frozen validity mask differ')
    return np.where(mask, depth, np.nan), camera


def run(collection: Path, partition: Path, output: Path, *, limit: int | None = None) -> dict:
    if sha(collection) != COLLECTION_SHA256 or sha(partition) != PARTITION_SHA256:
        raise ValueError('Frozen Development overlay or partition hash differs')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh/empty output required')
    data = read_inputs(collection, partition)
    if tuple(data['query_names']) != QUERY_NAMES:
        raise ValueError('Frozen query order differs')
    frames = []
    for item in data['collection']['layouts']:
        overlay = json.loads(checked({'path': item['overlay'], 'sha256': item['overlay_sha256']}).read_text(encoding='utf-8-sig'))
        frames.extend(overlay['frames'])
    if len(frames) != 960 or [r['frame_key'] for r in frames] != [r['frame_key'] for r in data['rows']]:
        raise ValueError('Overlay/materialized frame keys differ')
    n = len(frames) if limit is None else limit
    if not 1 <= n <= 960:
        raise ValueError('Limit must be 1..960')
    output.mkdir(parents=True, exist_ok=True)
    # Native float32 resolution and NaNs are retained; no resize/quantization.
    depth_array = np.lib.format.open_memmap(output/'visible-depth-f32.npy', mode='w+',
                                            dtype=np.float32, shape=(n, 360, 640))
    histogram = np.empty((n, 64, 16), np.float32)
    nearest = np.empty((n, 64), np.float32)
    fraction = np.empty((n, 64), np.float32)
    witness = np.empty((n, 6), np.bool_)
    for index, frame in enumerate(frames[:n]):
        depth, camera = load_scene_depth(frame)
        depth_array[index] = depth
        witness[index] = visible_witness(depth, camera)
        histogram[index], nearest[index], fraction[index] = perfect_h3_histogram(depth, camera)
        if (index + 1) % 160 == 0:
            print(json.dumps({'materialized': index + 1, 'total': n}), flush=True)
    depth_array.flush()
    del depth_array
    np.savez_compressed(output/'perfect-tof-h3.npz', frame_key=[r['frame_key'] for r in frames[:n]],
                        histogram=histogram, nearest_m=nearest,
                        valid_in_window_fraction=fraction, visible_witness=witness)
    labels = data['labels'][:n]
    summaries = {}
    for key, selected in [('all', np.ones(n, dtype=bool)),
                          ('train', data['train'][:n]), ('dev', data['dev'][:n])]:
        if not selected.any():
            continue
        y, w = labels[selected], witness[selected]
        summaries[key] = dict(frames=int(selected.sum()), positive=int((y == 1).sum()),
                              negative=int((y == 0).sum()), unknown=int((y < 0).sum()),
                              positive_visible_witness=int(((y == 1) & w).sum()),
                              positive_without_visible_witness=int(((y == 1) & ~w).sum()),
                              negative_with_visible_witness=int(((y == 0) & w).sum()))
    report = dict(status='COMPLETE_VISIBLE_DEPTH_DEVELOPMENT_AUDIT' if n == 960 else 'CANARY_ONLY',
                  claim_limit='Visible first-surface pixels are not complete physical occupancy. A positive without a sampled visible witness may be occluded, subpixel, or discrepant; neither direction alone proves label error.',
                  frame_count=n, train_frames=int(data['train'][:n].sum()), dev_frames=int(data['dev'][:n].sum()),
                  query_names=QUERY_NAMES, source='First-visible native 640x360 EXR Z and validity, exact camera K; labels only for audit after feature derivation; no triangle/instance/target input',
                  perfect_tof='16 H3-width radial bins, 16x16 rays per 8x8 zone, angular-area histogram; no NIR/shot noise/electronics; out-of-window and invalid rays stay in denominator; nearest visible scalar from same rays',
                  collection_overlay_sha256=sha(collection), partition_sha256=sha(partition),
                  files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                  code_sha256=sha(__file__), summaries=summaries)
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.collection_overlay, args.partition_plan, args.output, limit=args.limit), indent=2))
