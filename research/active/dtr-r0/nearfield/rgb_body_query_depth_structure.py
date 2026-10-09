"""CPU-only descriptive depth distribution and spatial-structure diagnosis.

All frames are retained. Reference depth is used only for a same-domain
descriptive comparison, never as an observation or a resolution causal test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from rgb_body_query_3rscan import color_coordinates, sample_prediction


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def describe(z, domain):
    z = np.asarray(z, dtype=np.float64)
    valid = domain & np.isfinite(z) & (z > 0)
    values = z[valid]
    result = dict(domain_pixels=int(domain.sum()), finite_positive_pixels=int(valid.sum()),
                  clamp_tail_ge1000m=int((valid & (z >= 1000)).sum()))
    if not values.size:
        return result
    quant = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    result.update(dict(zip(('min_m', 'p01_m', 'p05_m', 'p25_m', 'p50_m', 'p75_m',
                           'p95_m', 'p99_m', 'max_m'), quant.tolist())))
    result.update(mean_m=float(values.mean()), std_m=float(values.std()),
                  iqr_m=float(quant[5]-quant[3]),
                  log_iqr=float(np.quantile(np.log(values), .75)-np.quantile(np.log(values), .25)))
    logz = np.log(np.where(valid, z, 1.))
    for name, left, right, delta in (
        ('horizontal', valid[:, :-1], valid[:, 1:], np.diff(logz, axis=1)),
        ('vertical', valid[:-1], valid[1:], np.diff(logz, axis=0)),
    ):
        d = np.abs(delta[left & right])
        result[name+'_adjacent_pairs'] = int(d.size)
        result[name+'_log_change_mean'] = float(d.mean()) if d.size else None
        result[name+'_log_change_median'] = float(np.median(d)) if d.size else None
        result[name+'_log_change_p95'] = float(np.quantile(d, .95)) if d.size else None
        result[name+'_exact_equal_pairs'] = int((d == 0).sum())
    blocks = []
    h, w = z.shape
    for iy in range(8):
        for ix in range(8):
            sl = (slice(iy*h//8, (iy+1)*h//8), slice(ix*w//8, (ix+1)*w//8))
            vals = z[sl][valid[sl]]
            if vals.size:
                blocks.append(float(np.median(vals)))
    result.update(blocks_observed=len(blocks),
                  block_median_min_m=min(blocks) if blocks else None,
                  block_median_max_m=max(blocks) if blocks else None,
                  block_median_log_std=float(np.std(np.log(blocks))) if blocks else None)
    return result


def fixture():
    z = np.full((16, 16), 2., dtype=np.float32)
    a = describe(z, np.ones_like(z, dtype=bool))
    assert a['iqr_m'] == a['log_iqr'] == 0
    assert a['horizontal_log_change_mean'] == a['vertical_log_change_mean'] == 0
    varied = np.exp(np.arange(256).reshape(16, 16)/256.)
    b = describe(varied, np.ones_like(z, dtype=bool))
    assert b['iqr_m'] > 0 and b['horizontal_log_change_mean'] > 0 and b['vertical_log_change_mean'] > 0
    return dict(status='PASS', checks=['constant has zero IQR and local log gradient',
                                     'varying array has nonzero IQR and both local gradients'])


def analyze(sensor, dataset, split, budget_deadline, output):
    manifest_path = sensor/'dataset_manifest.json'
    pred_path = sensor/'depthpro/predictions.json'
    manifest = json.loads(manifest_path.read_text('utf-8-sig'))
    predictions = json.loads(pred_path.read_text('utf-8-sig'))
    assert predictions['status'] == 'COMPLETE'
    priors = {(r['scan'], r['frame']): r for r in predictions['rows']}
    rows, samples = [], []
    sources = []
    for index, r in enumerate(manifest['rows']):
        if split and r['split'] != split:
            continue
        if time.perf_counter() > budget_deadline:
            raise TimeoutError('CPU allocation exceeded')
        p = priors[r['scan'], r['frame']]
        assert sha(p['path']) == p['sha256'] and p['rgb_sha256'] == r['rgb_sha256']
        assert sha(r['reference_path']) == r['reference_sha256']
        with np.load(p['path'], allow_pickle=False) as a:
            native = a['depth']
        with np.load(r['reference_path'], allow_pickle=False) as a:
            measured = a['depth']
            observed = a['observed']
            dk, ck = a['depth_K'], a['color_K']
            ref_mx, ref_my = a['map_x'], a['map_y']
        assert np.allclose(dk, r['depth_K']) and np.allclose(ck, r['color_K'])
        assert list(measured.shape) == r['depth_shape']
        assert list(native.shape) == r['color_shape']
        mx, my = color_coordinates(dk, ck, measured.shape)
        assert np.max(np.abs(mx-ref_mx)) < 1e-5 and np.max(np.abs(my-ref_my)) < 1e-5
        # Floating-point K projection can put an exact border ray a few
        # millionths of a pixel outside. This tolerance restores that boundary,
        # rather than treating it as loss of physical camera coverage.
        tol = 1e-5
        inside = (mx >= -tol) & (my >= -tol) & (mx <= native.shape[1]-1+tol) & (my <= native.shape[0]-1+tol)
        pred = sample_prediction(native, mx, my)
        common = inside & observed & np.isfinite(measured) & (measured > 0) & np.isfinite(pred) & (pred > 0)
        domains = [
            ('prediction_native_all', native, np.ones_like(native, dtype=bool)),
            ('prediction_public_ray_inside', pred, inside),
            ('prediction_common_sensor', pred, common),
            ('sensor_common_prediction', measured, common),
            ('prediction_common_sensor_tail_excluded', pred, common & (pred < 1000)),
            ('sensor_common_prediction_tail_excluded', measured, common & (pred < 1000)),
        ]
        for domain_name, z, domain in domains:
            rows.append(dict(dataset=dataset, index=index, environment=r['environment'],
                             scan=r['scan'], frame=r['frame'], split=r['split'], domain=domain_name,
                             height=z.shape[0], width=z.shape[1], inside_pixels=int(inside.sum()),
                             total_sensor_pixels=inside.size, **describe(z, domain)))
        sources.append(dict(scan=r['scan'], frame=r['frame'], prediction_path=p['path'],
                            prediction_sha256=p['sha256'], reference_path=r['reference_path'],
                            reference_sha256=r['reference_sha256'], inside_pixels=int(inside.sum()),
                            common_pixels=int(common.sum()), native_shape=list(native.shape)))
        if dataset == 'ARKitScenes' and index in (0, 5, 10, 15):
            assert sha(r['rgb_path']) == r['rgb_sha256']
            samples.append(dict(index=index, row=r, native=native, pred=pred, measured=measured,
                                common=common, inside=inside))
    return rows, samples, dict(dataset=dataset, frames=len(sources),
        manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
        prediction_manifest_path=str(pred_path), prediction_manifest_sha256=sha(pred_path), sources=sources)


def figure(samples, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    norm = LogNorm(vmin=.2, vmax=10)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgray')
    for i, sample in enumerate(samples):
        axes[i, 0].imshow(Image.open(sample['row']['rgb_path'])); axes[i, 0].set_title(f"Fixed index {sample['index']} / RGB 256 x 192")
        for j, key in ((1, 'pred'), (2, 'measured')):
            z = sample[key]
            shown = np.where(sample['inside'] & np.isfinite(z) & (z > 0), z, np.nan)
            axes[i, j].imshow(shown, norm=norm, cmap=cmap, interpolation='nearest')
            tail = sample['inside'] & np.isfinite(z) & (z >= 1000)
            if tail.any():
                overlay = np.zeros(z.shape+(4,)); overlay[tail] = [1, 0, 1, 1]
                axes[i, j].imshow(overlay, interpolation='nearest')
            axes[i, j].set_title(('Raw Depth Pro' if j == 1 else 'Sensor optical Z')+f" / tail >=1000m: {int(tail.sum())}")
        for ax in axes[i]: ax.set_axis_off()
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=axes[:, 1:].ravel().tolist(), shrink=.8, label='Optical Z (m), fixed log color .2-10m; magenta >=1000m')
    fig.suptitle('ARKitScenes 47333462: fixed frame indices, descriptive structure only')
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--transfer', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget-s', type=float, default=240)
    args = p.parse_args()
    started = time.perf_counter(); deadline = started+args.budget_s
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError('Preserve previous results')
    write(out/'plan.json', dict(status='PLANNED', cpu_budget_s=args.budget_s, gpu_budget_s=0,
        downloads=0, units='optical Z metres; log statistics use natural log',
        frame_selection='all 16 ARKit; all 64 new 3RScan; all 24 old validation',
        contact_figure_fixed_indices=[0, 5, 10, 15], tail_threshold_m=1000,
        inside_boundary_tolerance_pixels=1e-5,
        main_domains='native all; public-ray inside; intersection with observed positive sensor depth',
        tail_exclusion='supplement only; no input or primary statistic deleted',
        interpretation='descriptive spatial variation; no resolution causality or metric accuracy claim'))
    checks = fixture(); rows, samples, sources = [], [], []
    for sensor, dataset, split in (
        (args.transfer/'camera-sensor', 'ARKitScenes', None),
        (args.transfer/'sensor', 'new_3RScan', None),
        (args.old/'sensor', 'old_3RScan_validation', 'validation'),
    ):
        r, s, m = analyze(sensor.resolve(), dataset, split, deadline, out)
        rows.extend(r); samples.extend(s); sources.append(m)
    keys = sorted({k for r in rows for k in r})
    with (out/'all_frame_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
    figure(samples, out/'fixed_frame_contact.png')
    summaries = []
    for dataset in ('ARKitScenes', 'new_3RScan', 'old_3RScan_validation'):
        for domain in sorted({r['domain'] for r in rows}):
            r = [x for x in rows if x['dataset'] == dataset and x['domain'] == domain]
            fields = ('p50_m', 'iqr_m', 'log_iqr', 'horizontal_log_change_mean',
                      'vertical_log_change_mean', 'block_median_log_std')
            summaries.append(dict(dataset=dataset, domain=domain, frames=len(r),
                domain_pixels=sum(x['domain_pixels'] for x in r),
                finite_positive_pixels=sum(x['finite_positive_pixels'] for x in r),
                clamp_tail_ge1000m=sum(x['clamp_tail_ge1000m'] for x in r),
                per_frame_ranges={k:dict(min=min(x[k] for x in r if x.get(k) is not None),
                                        median=float(np.median([x[k] for x in r if x.get(k) is not None])),
                                        max=max(x[k] for x in r if x.get(k) is not None)) for k in fields}))
    shutil.copyfile(__file__, out/'executed_depth_structure.py')
    result = dict(status='COMPLETE', sources=sources, summaries=summaries,
        source_sha256=sha(__file__), statistics_csv_sha256=sha(out/'all_frame_statistics.csv'),
        figure_sha256=sha(out/'fixed_frame_contact.png'), focused_checks=checks,
        shape_map_sha_checks='PASS for all selected frames', wall_s=time.perf_counter()-started,
        cpu_budget_s=args.budget_s, gpu_s=0, downloads_bytes=0,
        limitations='All ARKit frames are one correlated capture. Sensor depth is diagnostic reference, not metrology. Spatial variation alone does not establish useful or accurate depth; no resolution intervention performed.')
    write(out/'result.json', result)
    write(out/'focused_check.json', checks)
    print(json.dumps(dict(status=result['status'], wall_s=result['wall_s'], summaries=summaries[:6])))


if __name__ == '__main__':
    main()
