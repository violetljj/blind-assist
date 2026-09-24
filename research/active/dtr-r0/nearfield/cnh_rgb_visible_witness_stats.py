"""Independent first-visible pixel witnesses for frozen alley failure slicing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cnh_rgb_alley_v2 import COLLECTION_SHA256, PARTITION_SHA256
from cnh_rgb_dev_comparison import checked, sha
from cnh_rgb_visible_depth_audit import load_scene_depth
from cnh_street_e2e_materialize import BOXES


def statistics(depth: np.ndarray, camera: dict) -> tuple[np.ndarray, ...]:
    y, x = np.indices(depth.shape)
    z = depth
    k = np.asarray(camera['K'], dtype=np.float64)
    x_m = (x-k[0, 2])*z/k[0, 0]
    y_m = (y-k[1, 2])*z/k[1, 1]
    valid = np.isfinite(z) & (z > 0)
    transform = np.linalg.inv(np.asarray(camera['T_camera_tof'], dtype=np.float64))
    edges = np.linspace(-22.5, 22.5, 9)
    count = np.zeros(6, dtype=np.int32)
    min_z = np.full(6, np.nan, dtype=np.float32)
    median_z = np.full(6, np.nan, dtype=np.float32)
    min_edge = np.full(6, np.nan, dtype=np.float32)
    near_edge_fraction = np.full(6, np.nan, dtype=np.float32)
    tof_fov_fraction = np.full(6, np.nan, dtype=np.float32)
    for q, (lo, hi) in enumerate(BOXES):
        mask = (valid & (z >= lo[2]) & (z <= hi[2]) &
                (x_m >= lo[0]) & (x_m <= hi[0]) &
                (y_m >= lo[1]) & (y_m <= hi[1]))
        count[q] = int(mask.sum())
        if count[q] == 0:
            continue
        zx, xx, yy = z[mask], x_m[mask], y_m[mask]
        min_z[q] = zx.min()
        median_z[q] = np.median(zx)
        points = np.stack((xx, yy, zx), axis=1) @ transform[:3, :3].T + transform[:3, 3]
        ax = np.degrees(np.arctan2(points[:, 0], points[:, 2]))
        ay = np.degrees(np.arctan2(points[:, 1], points[:, 2]))
        edge_x = np.abs(ax[:, None]-edges).min(axis=1)
        edge_y = np.abs(ay[:, None]-edges).min(axis=1)
        margin = np.minimum(edge_x, edge_y)
        min_edge[q] = margin.min()
        near_edge_fraction[q] = np.mean(margin <= 1.0)
        tof_fov_fraction[q] = np.mean((np.abs(ax) <= 22.5) & (np.abs(ay) <= 22.5))
    return count, min_z, median_z, min_edge, near_edge_fraction, tof_fov_fraction


def run(collection: Path, partition: Path, prepared: Path, output: Path) -> dict:
    if sha(collection) != COLLECTION_SHA256 or sha(partition) != PARTITION_SHA256:
        raise ValueError('Frozen Development input differs')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh/empty output required')
    prepared_report = json.loads((prepared/'result.json').read_text(encoding='utf-8'))
    if prepared_report['status'] != 'COMPLETE_VISIBLE_DEPTH_DEVELOPMENT_AUDIT':
        raise ValueError('Full depth audit required')
    collection_obj = json.loads(collection.read_text(encoding='utf-8-sig'))
    frames, dev = [], []
    for layout in collection_obj['layouts']:
        overlay = json.loads(checked({'path': layout['overlay'], 'sha256': layout['overlay_sha256']}).read_text(encoding='utf-8-sig'))
        frames.extend(overlay['frames'])
        dev.extend([layout['split'] == 'dev'] * len(overlay['frames']))
    if len(frames) != 960 or sum(dev) != 480:
        raise ValueError('Six-layout frozen split required')
    with np.load(prepared/'perfect-tof-h3.npz', allow_pickle=False) as check:
        if list(check['frame_key'].astype(str)) != [r['frame_key'] for r in frames]:
            raise ValueError('Prepared frame keys differ')
        expected = check['visible_witness'].copy()
    results = [np.empty((960, 6), dtype=dtype) for dtype in
               (np.int32, np.float32, np.float32, np.float32, np.float32, np.float32)]
    for i, frame in enumerate(frames):
        depth, camera = load_scene_depth(frame)
        values = statistics(depth, camera)
        for dest, value in zip(results, values):
            dest[i] = value
        if not np.array_equal(results[0][i] > 0, expected[i]):
            raise ValueError('Visible witness differs from independent first audit')
        if (i+1) % 160 == 0:
            print(json.dumps({'processed': i+1, 'total': 960}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    keys = np.array([r['frame_key'] for r in frames])
    np.savez_compressed(output/'witness-stats.npz', frame_key=keys, dev=np.asarray(dev),
                        pixel_count=results[0], min_z_m=results[1], median_z_m=results[2],
                        min_tof_zone_edge_deg=results[3], fraction_within_1deg_of_tof_zone_edge=results[4],
                        fraction_inside_tof_45deg_fov=results[5])
    np.savez_compressed(output/'direct-depth-rule-dev.npz', frame_key=keys[np.asarray(dev)],
                        prediction=(results[0] > 0)[np.asarray(dev)])
    report = dict(status='COMPLETE_VISIBLE_WITNESS_STATS_DEVELOPMENT',
                  count=960, dev=480, query_count=6, source='Original 640x360 EXR Z, validity, camera K/T and six fixed query boxes only',
                  target_instance_attribution='UNAVAILABLE_FROM_DEPTH_AND_CAMERA_ALONE',
                  label_source_triangles_read=False,
                  distance_definition='Axial camera Z of first-visible pixels whose backprojected centres lie in each closed query box',
                  tof_boundary_definition='Minimum angular distance to any edge of public 8x8 45deg zone grid; fraction <=1 degree and fraction inside full ToF FOV also saved',
                  prepared_result_sha256=sha(prepared/'result.json'), collection_overlay_sha256=sha(collection),
                  partition_sha256=sha(partition), code_sha256=sha(__file__),
                  files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.collection_overlay, args.partition_plan, args.prepared, args.output), indent=2))
