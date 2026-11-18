"""Rebuild the Oxford low-obstacle RGB, metric depth and ground manifests without model outputs.

Public source ori-drs/oxford_spires_dataset, revision 03f4382308333aa70c3253f12acd3fbf0c7c4a15.
Use sequences/<recording>/raw/images.zip (cam0/*.jpg) and processed/trajectory/gt-tum.txt,
ground_truth_map/<site>/merged-cloud-1cm.pcd (Observatory: merged-clouds-1cm-no-colour.pcd),
and the released cam0 / LiDAR / IMU calibration used by spires_insert_dev_20261117.py.
The existing --spires layout is map/<site>.pcd and seq/<recording>/gt-tum.txt; each --images
root may contain <recording>/cam0/*.jpg or <recording>/*.jpg. One raw cam0 exposure is absent
at the fixed first-window grid position 14. The complete reconstructible subset has 81 + 12 rows.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import spires_insert_dev_20261117 as spires

BLENHEIM = '2024-03-14-blenheim-palace-01'
OBSERVATORY = '2024-03-13-observatory-quarter-01'
# (recording, window suffix, first 5 Hz grid timestamp, grid length, global row ID offset, TLS crop radius)
WINDOWS = (
    (BLENHEIM, 'win0', 1710406816.3083062 - 4, 41, 0, 7.),
    (BLENHEIM, 'win1', 1710406862.9074843 - 4, 41, 100, 7.),
    (BLENHEIM, 'negative', 1710406761.7092786, 6, 1000, 5.),
    (OBSERVATORY, 'negative', 1710338156.2424626, 6, 1006, 5.),
)


def raw_images(roots, seq):
    photos = {}
    for root in roots:
        for folder in (root / seq / 'cam0', root / seq):
            photos.update((float(p.stem), p) for p in folder.glob('*.jpg'))
    if not photos:
        raise ValueError(f'No cam0 images for {seq}')
    return np.array(sorted(photos)), photos


def window_images(roots, spec):
    seq, suffix, t0, count, offset, _ = spec
    times, photos = raw_images(roots, seq)
    chosen = []
    for index, nominal in enumerate(t0 + np.arange(count) / 5):
        t = times[np.argmin(np.abs(times - nominal))]
        if abs(t - nominal) > .027:
            if suffix == 'win0' and index == 14:
                continue
            raise ValueError(f'Missing required raw image for {seq} at {nominal:.6f}')
        chosen.append((offset + index, t, photos[t]))
    return chosen


def render_window(spec, source, points, transforms, destination):
    seq, suffix = spec[:2]
    folder = destination / (seq + '__' + suffix)
    folder.mkdir(parents=True, exist_ok=True)
    voxel_keys, counts = np.unique(np.floor(points / spires.S.VOX).astype(np.int32), axis=0, return_counts=True)
    floor_grid = spires.S.Grid(voxel_keys, counts)
    K = [[spires.F, 0, spires.CX], [0, spires.F, spires.CY], [0, 0, 1.]]
    rows = []
    for (i, t, original), transform in zip(source, transforms):
        position = transform[:3, 3]
        floor = spires.S.ground_at(floor_grid, position[:2], position[2])
        if not np.isfinite(floor):
            raise ValueError(f'No surveyed floor for {original.name}')
        up = transform[:3, :3].T @ np.array([0., 0., 1.])
        height = float(position[2] - floor)
        rgb, depth = folder / (original.stem + '_rgb.jpg'), folder / (original.stem + '_z.png')
        image = spires.pinhole_rgb(original)
        z_m = spires.tls_depth(points, transform).astype(np.float32) / 1000.
        mm = np.clip(np.nan_to_num(z_m) * 1000., 0, 65535).astype(np.uint16)
        if not cv2.imwrite(str(rgb), image, [cv2.IMWRITE_JPEG_QUALITY, 94]) or not cv2.imwrite(str(depth), mm):
            raise OSError(f'Could not save paired images for {original}')
        rows.append(dict(seq=folder.name, i=int(i), t=float(t), rgb=rgb.relative_to(destination).as_posix(),
                         depth=depth.relative_to(destination).as_posix(), ref_scale=.001, down=(-up).tolist(),
                         K=K, size=[spires.W, spires.H], mount='head', ref_grade='sparse_geometry',
                         eye_height=height, ground=[up.tolist(), height]))
    return rows


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--spires', type=Path, required=True)
    args.add_argument('--images', type=Path, action='append', required=True)
    args.add_argument('--out', type=Path, required=True)
    opt = args.parse_args()
    dest = opt.out.resolve()
    spires.F = 280.  # Same 480x640 and 6-degree pitch as the project preset, wider rectified view.
    spires._MAP = None
    windows = []
    for spec in WINDOWS:
        chosen = window_images(opt.images, spec)
        trajectory = spires.Traj(opt.spires / 'seq' / spec[0] / 'gt-tum.txt')
        transforms = trajectory.T([entry[1] for entry in chosen])
        windows.append((spec, chosen, transforms))
    manifests = {'positive': [], 'negative': []}
    for site, radius in dict.fromkeys((spires.S.site_of(v[0][0]), v[0][-1]) for v in windows):
        group = [v for v in windows if spires.S.site_of(v[0][0]) == site and v[0][-1] == radius]
        centres = [v[2][:, :3, 3].mean(axis=0)[:2] for v in group]
        for (spec, src, poses), pts in zip(group, spires.map_crop(opt.spires, site, centres, rad=radius)):
            label = 'negative' if spec[1] == 'negative' else 'positive'
            manifests[label].extend(render_window(spec, src, pts, poses, dest))
    for label, rows in manifests.items():
        file = dest / ('filled_generic_ground.json' if label == 'positive' else 'negative_generic_ground.json')
        data = dict(dataset='oxford_spires_blenheim_pilot' if label == 'positive' else 'oxford_spires_local_negative',
                    data_root=str(dest), frames=rows)
        file.write_text(json.dumps(data, indent=1, allow_nan=False), encoding='utf-8')
        print(file, len(rows), flush=True)


if __name__ == '__main__':
    main()
