"""Independent consumed-Development CNH/readout accounting; no model execution."""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from cnh_pose_expected_only import S


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def load_depth(path):
    if Path(path).suffix == '.npy':
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as f:
        return f['depth']


def interval(x, y, q):
    lower = np.full(x.shape, q['low'][2], dtype=float)
    upper = np.full(x.shape, q['high'][2], dtype=float)
    for ray, lo, hi in zip([x, y], q['low'][:2], q['high'][:2]):
        nz = abs(ray) > 1e-12
        aa = np.divide(lo, ray, out=np.zeros_like(ray), where=nz)
        bb = np.divide(hi, ray, out=np.zeros_like(ray), where=nz)
        lower = np.maximum(lower, np.where(nz, np.minimum(aa, bb), -np.inf))
        upper = np.minimum(upper, np.where(nz, np.maximum(aa, bb), np.inf))
        upper[(~nz) & ((lo > 0) | (hi < 0))] = -np.inf
    return lower, upper, lower <= upper + 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--pilot', default='pilot')
    ap.add_argument('--fov-manifest', type=Path, action='append', default=[])
    a = ap.parse_args()
    start = time.perf_counter()
    run, pilot = a.run, a.run / a.pilot
    repo = Path(__file__).resolve().parents[4]
    old = repo / 'artifacts.local/work/rgb-near-readout-scale-dev-20261010'
    result = read(pilot / 'results.json')
    seal = read(run / 'implementation_seal.json')
    rows = table(pilot / 'query_table.csv')
    index = {(r['scan'], int(r['frame']), r['query'], r['arm']): r for r in rows}
    assert len(index) == len(rows)
    pred = {model: {(r['scan'], r['frame']): r for r in read(old / model / 'predictions.json')['rows']}
            for model in ['dav2', 'unidepth', 'depthpro']}
    manifests = {s: read(old / 'new-eval-sensor' / ('new_arkit_' + s) / 'dataset_manifest.json')
                 for s in ['41069021', '41069048']}
    edge = np.tan(np.deg2rad(22.5))
    checked = 0
    coverage, snrs, errors = [], [], []
    native_halfangles = []
    geometry_checks = []
    native_zone_medians = {}
    directions, _ = S.angular_rays(16)
    for fr in result['frames']:
        scan, fi = str(fr['scan']), int(fr['frame'])
        m = manifests[scan]
        ref = next(r for r in m['rows'] if r['frame'] == fi)
        with np.load(fr['sample'], allow_pickle=False) as f:
            d = dict(f)
        traj = np.loadtxt(run / 'sources' / scan / 'lowres_wide.traj')
        source_time = float(Path(fr['source']).stem.rsplit('_', 1)[1])
        poses = []
        for timestamp, stored in [(source_time, d['source_pose']), (ref['timestamp_s'], d['target_pose'])]:
            line = traj[np.argmin(abs(traj[:, 0]-timestamp))]
            ext = np.eye(4)
            ext[:3, :3] = Rotation.from_rotvec(line[1:4]).as_matrix()
            ext[:3, 3] = line[4:7]
            pose = np.linalg.inv(ext)
            assert np.allclose(pose, stored)
            poses.append(pose)
        # Independent homogeneous-coordinate warp and z-buffer from source FARO only.
        source_z = np.asarray(Image.open(fr['source']), dtype=float)/1000.
        sy, sx = np.nonzero(np.isfinite(source_z) & (source_z > 0))
        depth = source_z[sy, sx]
        source_xyz = np.linalg.inv(d['geometry_K']) @ np.stack([sx*depth, sy*depth, depth])
        world = poses[0] @ np.vstack([source_xyz, np.ones(len(depth))])
        target_xyz = (np.linalg.inv(poses[1]) @ world)[:3]
        proj = d['target_K'] @ target_xyz
        px, py = np.rint(proj[:2]/proj[2]).astype(int)
        shape = tuple(ref['depth_shape'])
        inside = (target_xyz[2] > 0) & (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0])
        reconstructed = np.full(np.prod(shape), np.inf)
        np.minimum.at(reconstructed, py[inside]*shape[1]+px[inside], target_xyz[2, inside])
        reconstructed = reconstructed.reshape(shape)
        assert np.array_equal(np.isfinite(reconstructed), np.isfinite(d['registered_z']))
        finite = np.isfinite(reconstructed)
        assert np.allclose(reconstructed[finite], d['registered_z'][finite], atol=1e-10)
        # Verify quadrature coverage directly on that registered geometry.
        tk = d['target_K']
        uu = np.rint(tk[0, 0]*directions[..., 0]/directions[..., 2]+tk[0, 2]).astype(int)
        vv = np.rint(tk[1, 1]*directions[..., 1]/directions[..., 2]+tk[1, 2]).astype(int)
        validray = (uu >= 0) & (uu < shape[1]) & (vv >= 0) & (vv < shape[0])
        rz = reconstructed[vv.clip(0, shape[0]-1), uu.clip(0, shape[1]-1)]
        assert np.allclose((validray & np.isfinite(rz) & (rz > 0)).mean(-1), d['coverage'])
        geometry_checks.append(dict(scan=scan, frame=fi, registered_pixels=int(finite.sum()),
                                    exact_geometry_timestamp=abs(source_time-ref['timestamp_s']) < 1e-6))
        h = d['hist']
        assert h.shape == (8, 8, 16)
        assert np.array_equal(h, d['counts'] - d['background'])
        peak = h.argmax(-1)
        height = np.take_along_axis(h, peak[..., None], -1)[..., 0]
        bg = np.take_along_axis(d['background'], peak[..., None], -1)[..., 0]
        snr = height / np.sqrt(np.maximum(height, 0) + 2 * bg + 1)
        distance = (peak + .5) * .3002784
        assert np.allclose(snr, d['snr']) and np.allclose(distance, d['peak_range'])
        assert np.all((d['coverage'] >= 0) & (d['coverage'] <= 1))
        coverage.extend(d['coverage'].ravel().tolist()); snrs.extend(snr.ravel().tolist())
        k = np.asarray(ref['depth_K'])
        yy, xx = np.indices(ref['depth_shape'])
        x, y = (xx-k[0, 2])/k[0, 0], (yy-k[1, 2])/k[1, 1]
        native_halfangles.extend(np.rad2deg(np.arctan([-x.min(), x.max(), -y.min(), y.max()])).tolist())
        ix = np.clip(np.floor((x+edge)/(2*edge)*8).astype(int), 0, 7)
        iy = np.clip(np.floor((y+edge)/(2*edge)*8).astype(int), 0, 7)
        nz = d['native_z']
        nrad = nz * np.sqrt(1+x*x+y*y)
        for zy in range(8):
            for zx in range(8):
                nm = (ix == zx) & (iy == zy) & (abs(x) <= edge) & (abs(y) <= edge) & np.isfinite(nz) & (nz > 0)
                native_zone_medians[scan, fi, zy, zx] = None if not nm.any() else float(np.median(nrad[nm]))
        valid = (abs(x) <= edge) & (abs(y) <= edge) & (d['coverage'][iy, ix] >= .75) & (snr[iy, ix] >= 3)
        tof = distance[iy, ix] / np.sqrt(1+x*x+y*y)
        vals = {name: load_depth(pred[model][(ref['scan'], fi)]['path'])
                for name, model in [('dav', 'dav2'), ('uni', 'unidepth'), ('public', 'depthpro')]}
        public = np.isfinite(vals['public']) & (vals['public'] > 0)
        with np.load(ref['reference_path'], allow_pickle=False) as f:
            labels = f['labels']
        for j, q in enumerate(m['queries']):
            lower, upper, domain = interval(x, y, q)
            near = q['high'][2] <= .8
            masks = {}
            for arm, depth, ok, cut, active in [
                ('tof', tof, valid, 0., True),
                ('dav', vals['dav'], np.isfinite(vals['dav']) & (vals['dav'] > 0), .24403834342956543, not near),
                ('uni', vals['uni'], np.isfinite(vals['uni']) & (vals['uni'] > 0), .09616100788116455, near)]:
                masks[arm] = public & domain & ok & (np.minimum(depth-lower, upper-depth) >= cut) & active
            rgb = masks['uni' if near else 'dav']
            masks.update(rgb_banded=rgb, tof_OR_rgb_banded=masks['tof'] | rgb, tof_AND_rgb_banded=masks['tof'] & rgb,
                         tof_OR_dav=masks['tof'] | masks['dav'], tof_AND_dav=masks['tof'] & masks['dav'],
                         tof_OR_uni=masks['tof'] | masks['uni'], tof_AND_uni=masks['tof'] & masks['uni'])
            for arm, mask in masks.items():
                r = index[scan, fi, q['name'], arm]
                n, w = int(mask.sum()), int((mask & (labels[j] == 1)).sum())
                assert int(r['support_pixels']) == n and int(r['positive_pixels']) == w, (scan, fi, q['name'], arm)
                assert (r['support'] == 'True') == (n >= 16)
                assert (r['witness'] == 'True') == (w >= 16)
                assert r['state'] == ref['queries'][j]['state']
                checked += 1
    sums = []
    for s in result['summary']:
        rr = [r for r in rows if r['arm'] == s['arm'] and r['band'] == s['band']]
        pos = [r for r in rr if r['state'] == 'POSITIVE']
        free = [r for r in rr if r['state'] == 'FREE_ON_SAMPLED_RAYS']
        unk = [r for r in rr if r['state'] == 'UNKNOWN']
        calc = dict(queries=len(rr), POS=len(pos), W=sum(r['witness'] == 'True' for r in pos),
                    FREE=len(free), F=sum(r['support'] == 'True' for r in free),
                    UNKNOWN=len(unk), U=sum(r['support'] == 'True' for r in unk))
        for key, value in calc.items():
            assert s[key] == value, (s['arm'], s['band'], key)
        sums.append(dict(arm=s['arm'], band=s['band'], **calc))
    zones = table(pilot / 'zones.csv')
    for z in zones:
        measured = native_zone_medians[z['scan'], int(z['frame']), int(z['zone_y']), int(z['zone_x'])]
        assert (measured is None) == (not z['native_radial_median'])
        if z['native_radial_median']:
            assert abs(measured-float(z['native_radial_median'])) < 1e-10
            err = float(z['peak_range']) - float(z['native_radial_median'])
            assert abs(err-float(z['peak_radial_error'])) < 1e-10
            errors.append(err)
    with np.load(run / 'frozen_readout/frozen_description.npz', allow_pickle=False) as f:
        frozen = dict(f)
    frozen_receipt = read(run / 'frozen_readout/receipt.json')
    assert (frozen['s_grade'][frozen['old5']] == 2).all()
    assert frozen['old5'].sum(axis=(0, 1)).tolist() == frozen_receipt['old5_strong_slots']
    assert (frozen['s_grade'] == 1).sum(axis=(0, 1)).tolist() == frozen_receipt['s_light_slots']
    assert np.allclose(frozen['seed_scores'].mean(axis=1), frozen['s_mean'])
    with np.load(run / 'frozen_window.npz', allow_pickle=False) as f:
        steps = np.diff(f['timestamps'], axis=1)
    assert ((steps >= .15) & (steps <= .25)).all()
    fov_checks = []
    for path in a.fov_manifest:
        m = read(path)
        for r in m['rows']:
            k = np.asarray(r['depth_K'])
            h, w = r['depth_shape']
            bounds = [k[0, 2]/k[0, 0], (w-1-k[0, 2])/k[0, 0], k[1, 2]/k[1, 1], (h-1-k[1, 2])/k[1, 1]]
            angles = np.rad2deg(np.arctan(bounds))
            fov_checks.append(dict(scan=r['scan'], frame=r['frame'], min_halfangle_deg=float(angles.min()), covered=bool((angles >= 22.5).all())))
    sanity = read(pilot / 'sanity.json')
    for s in sanity['error_by_native_radial_band']:
        lo, hi = map(float, s['band'].removesuffix('m').split('-'))
        rr = [z for z in zones if z['native_radial_median'] and lo <= float(z['native_radial_median']) < hi and (s['subset'] == 'all' or z['accepted'] == 'True')]
        assert len(rr) == s['zones']
        if rr:
            ee = np.array([float(z['peak_radial_error']) for z in rr])
            assert np.isclose(np.median(abs(ee)), s['median_absolute_error_m'])
            assert np.isclose(np.quantile(abs(ee), .9), s['p90_absolute_error_m'])
    output = dict(status='PASS', seconds=time.perf_counter()-start, frames=len(result['frames']),
                  query_rows_recomputed=checked, summary=sums, zone_rows=len(zones),
                  coverage_quantiles=np.quantile(coverage, [0, .1, .5, .9, 1]).tolist(),
                  snr_quantiles=np.quantile(snrs, [0, .1, .5, .9, 1]).tolist(),
                  radial_error_abs_median=None if not errors else float(np.median(np.abs(errors))),
                  native_min_boundary_halfangle_deg=min(native_halfangles),
                  geometry_reprojection_checks=geometry_checks,
                  additional_native_fov_checks=fov_checks,
                  bin_width_m=.3002784, implementation_seal=seal,
                  frozen_counts=frozen_receipt, frozen_step_range_s=[float(steps.min()), float(steps.max())],
                  scope='Independent histogram, pixel-mask W/F and aggregation recomputation; no training, model execution or protected access',
                  evidence_limits=['FARO acquisition independent; high/low-depth filtering induces selection correlation',
                      'Angle-expanded zone support is not 16 independent measurements',
                      'Static reprojection cannot remove dynamic-scene or pose error',
                      'Sparse anchors are not a continuous 5Hz synchronized benchmark'])
    (run / 'independent_audit.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: output[k] for k in ['status', 'seconds', 'frames', 'query_rows_recomputed', 'zone_rows']}))


if __name__ == '__main__':
    main()
