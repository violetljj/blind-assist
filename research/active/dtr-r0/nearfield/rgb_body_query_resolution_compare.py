"""Independent same-frame resolution comparison with frozen query scoring.

Native VGA versus its BOX downsample only; no original-stream substitution.
The common reference domain excludes the same boundary rays in both arms.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from rgb_body_query_depth_structure import describe
from rgb_body_query_hires_prepare import resize_k
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_scene_diagnostic import paired_counts, write_csv
from rgb_body_query_transfer_audit import load, score


def depth_quality(z, ref, domain):
    v = domain & np.isfinite(z) & (z > 0) & np.isfinite(ref) & (ref > 0)
    ratio = z[v].astype(np.float64)/ref[v]
    al = np.abs(np.log(ratio)); ar = np.abs(ratio-1)
    return dict(domain_pixels=int(domain.sum()), metric_pixels=int(v.sum()),
        mean_abs_log=float(al.mean()), median_abs_log=float(np.median(al)),
        mean_abs_rel=float(ar.mean()), median_abs_rel=float(np.median(ar)),
        ratio_p05=float(np.quantile(ratio, .05)), ratio_p50=float(np.median(ratio)),
        ratio_p95=float(np.quantile(ratio, .95)), sum_abs_log=float(al.sum()),
        sum_abs_rel=float(ar.sum()), clamp_tail_ge1000m=int((z[v] >= 1000).sum()))


def aggregate_pairs(records, fields):
    groups = defaultdict(list)
    for r in records: groups[tuple(r[f] for f in fields)].append(r)
    out = []
    for group, rs in groups.items():
        count_keys = ('positive', 'free', 'rgb_tp', 'geometry_tp', 'rgb_fp', 'geometry_fp',
                      'positive_rescue', 'positive_loss', 'free_rescue', 'free_added')
        counts = {k:sum(r[k] for r in rs) for k in count_keys}
        assert counts['positive_rescue']-counts['positive_loss'] == counts['rgb_tp']-counts['geometry_tp']
        assert counts['free_added']-counts['free_rescue'] == counts['rgb_fp']-counts['geometry_fp']
        out.append(dict(zip(fields, group), **counts, records=len(rs),
                        native_recall=counts['rgb_tp']/counts['positive'] if counts['positive'] else None,
                        derived_recall=counts['geometry_tp']/counts['positive'] if counts['positive'] else None,
                        native_fpr=counts['rgb_fp']/counts['free'] if counts['free'] else None,
                        derived_fpr=counts['geometry_fp']/counts['free'] if counts['free'] else None))
    return out


def contact(samples, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgray')
    norm = LogNorm(.2, 10)
    for i, s in enumerate(samples):
        axes[i, 0].imshow(Image.open(s['rgb_path'])); axes[i, 0].set_title('Fixed pair '+str(s['pair_index'])+' native RGB')
        for j, field in ((1, 'native_depth'), (2, 'derived_depth'), (3, 'reference')):
            z = np.where(s['common'] & np.isfinite(s[field]) & (s[field] > 0), s[field], np.nan)
            axes[i, j].imshow(z, cmap=cmap, norm=norm, interpolation='nearest')
            tail = np.isfinite(z) & (z >= 1000)
            if tail.any():
                overlay = np.zeros(z.shape+(4,)); overlay[tail] = [1, 0, 1, 1]
                axes[i, j].imshow(overlay, interpolation='nearest')
            axes[i, j].set_title({'native_depth':'DP from 640 x 480', 'derived_depth':'DP from derived 256 x 192',
                                 'reference':'Same sensor optical Z'}[field]+f' / tail {int(tail.sum())}')
        for ax in axes[i]: ax.set_axis_off()
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, 1:].ravel().tolist(),
                 shrink=.75, label='Optical Z metres; fixed .2-10m log; magenta >=1000m')
    fig.suptitle('Same-frame BOX resolution intervention: fixed pairs 0, 2, 4, 7; common scoring rays')
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frozen-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--budget-s', type=float, default=240)
    args = p.parse_args(); start = time.perf_counter(); deadline = start+args.budget_s
    root, source, out = args.root.resolve(), args.frozen_source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'comparison.json').exists(): raise FileExistsError('Preserve completed comparison')
    write(out/'plan.json', dict(cpu_budget_s=args.budget_s, gpu_budget_s=0, downloads=0,
        frame_selection='all 8 prepared pairs; fixed figure pair indices 0,2,4,7',
        metric_domain='same paired_common_image_rays and finite positive sensor/prediction; full tail retained',
        attribution='resolution in this same-frame paired condition only; not cause of old capture regression'))
    audits = {}
    for arm in ('native_vga', 'derived_256'):
        audit_dir = out/('audit_'+arm)
        score(source, root/'paired-sensor'/arm, root/'paired-readout'/arm,
              audit_dir, max(1, deadline-time.perf_counter()))
        audits[arm] = load(audit_dir/'score.json')
    manifests = {a:load(root/'paired-sensor'/a/'dataset_manifest.json') for a in audits}
    prediction = {a:load(root/'paired-readout'/a/'predictions.json') for a in audits}
    evs = {a:load(root/'paired-readout'/a/'evaluation.json') for a in audits}
    assert all(len(m['rows']) == 8 for m in manifests.values())
    assert manifests['native_vga']['queries'] == manifests['derived_256']['queries']
    qs = manifests['native_vga']['queries']; rows, pairs, samples, identities, boundary_rows = [], [], [], [], []
    metric_arrays = {a:[] for a in audits}
    for i in range(8):
        if time.perf_counter() > deadline: raise TimeoutError('CPU budget reached')
        rs = {a:m['rows'][i] for a,m in manifests.items()}
        high, low = rs['native_vga'], rs['derived_256']
        assert all(high[k] == low[k] for k in ('source_id', 'timestamp_s', 'scan', 'frame', 'environment', 'depth_K', 'queries'))
        assert np.array_equal(np.array(low['color_K']), resize_k(high['color_K'], high['color_shape'], low['color_shape']))
        imh = Image.open(high['rgb_path']); iml = Image.open(low['rgb_path'])
        assert np.array_equal(np.array(imh.resize((256, 192), Image.Resampling.BOX)), np.array(iml))
        refs, depths, scores = {}, {}, {}
        for a, r in rs.items():
            assert sha(r['rgb_path']) == r['rgb_sha256'] and sha(r['reference_path']) == r['reference_sha256']
            with np.load(r['reference_path'], allow_pickle=False) as z: refs[a] = {k:z[k] for k in z.files}
            pr = prediction[a]['rows'][i]
            assert sha(pr['predicted_depth_path']) == pr['predicted_depth_sha256']
            with np.load(pr['predicted_depth_path'], allow_pickle=False) as z: depths[a] = z['depth']
            scores[a] = {name:np.load(item['path'], allow_pickle=False) for name,item in pr['predictions'].items()}
        for field in ('labels', 'depth', 'observed', 'depth_K', 'paired_common_image_rays'):
            assert np.array_equal(refs['native_vga'][field], refs['derived_256'][field]), field
        assert np.array_equal(scores['native_vga']['geometry'], scores['derived_256']['geometry'])
        common = refs['native_vga']['paired_common_image_rays']; ref = refs['native_vga']['depth']
        assert int(common.sum()) == 48705 and int((~common).sum()) == 447
        assert np.all(refs['native_vga']['labels'][:, ~common] == 255)
        valid = common & np.isfinite(ref) & (ref > 0)
        for d in depths.values(): valid &= np.isfinite(d) & (d > 0)
        for a, d in depths.items():
            quality = depth_quality(d, ref, valid)
            rows.append(dict(pair_index=i, source_id=high['source_id'], timestamp_s=high['timestamp_s'],
                input_arm=a, environment=high['environment'], frame=high['frame'],
                distribution=describe(d, valid), **quality))
            ratio = d[valid].astype(np.float64)/ref[valid]
            metric_arrays[a].append(ratio)
        rx, ry = rays(high['depth_K'], high['depth_shape']); labels = refs['native_vga']['labels']
        for j, q in enumerate(qs):
            entry, exit, domain = ray_interval(rx, ry, q)
            preds = {}
            for a in audits:
                preds[a] = {name:(probs[j] >= prediction[a]['cutoffs'][name]) & domain for name,probs in scores[a].items()}
                for name, factor in (('depthpro_raw', 1.), ('depthpro_cal_scale', 1.012320716490867)):
                    z = depths[a]*factor
                    preds[a][name] = domain & np.isfinite(z) & (z >= entry-1e-12) & (z <= exit+1e-12)
                for name, pred in preds[a].items():
                    full_support = int(pred.sum()); common_support = int((pred & common).sum())
                    boundary_rows.append(dict(input_arm=a, model_arm=name, source_id=high['source_id'], frame=high['frame'],
                        query=q['name'], reference_state=high['queries'][j]['state'], full_reachable_support=full_support,
                        common_ray_support=common_support, excluded_boundary_support=full_support-common_support,
                        full_support_hit=full_support >= 16, common_support_hit=common_support >= 16,
                        known_tp=int(((labels[j] == 1) & pred).sum())))
            for model_arm in preds['native_vga']:
                c = paired_counts(labels[j], preds['native_vga'][model_arm], preds['derived_256'][model_arm])
                pairs.append(dict(arm=model_arm, environment=high['environment'], source_id=high['source_id'],
                    frame=high['frame'], query=q['name'], band=f'{q["low"][2]:g}-{q["high"][2]:g}m', **c))
        if i in (0, 2, 4, 7):
            samples.append(dict(pair_index=i, rgb_path=high['rgb_path'], native_depth=depths['native_vga'],
                                derived_depth=depths['derived_256'], reference=ref, common=common))
        identities.append(dict(pair_index=i, source_id=high['source_id'], timestamp_s=high['timestamp_s'],
                              common_pixels=int(common.sum()), shape_K_labels_timestamp_rgb_BOX_verified=True,
                              geometry_probabilities_exact_equal=True))
    summaries = {}
    for a, arrays in metric_arrays.items():
        ratio = np.concatenate(arrays)
        summaries[a] = dict(metric_pixels=int(ratio.size), mean_abs_log=float(np.abs(np.log(ratio)).mean()),
            median_abs_log=float(np.median(np.abs(np.log(ratio)))),
            mean_abs_rel=float(np.abs(ratio-1).mean()), median_abs_rel=float(np.median(np.abs(ratio-1))),
            ratio_p05=float(np.quantile(ratio, .05)), ratio_p50=float(np.median(ratio)), ratio_p95=float(np.quantile(ratio, .95)))
    write(out/'depth_quality_all_frames.json', rows)
    flat = [{**{k:v for k,v in r.items() if k != 'distribution'}, **{'prediction_'+k:v for k,v in r['distribution'].items()}} for r in rows]
    write_csv(out/'depth_quality_all_frames.csv', flat); write_csv(out/'frame_query_paired_increments.csv', pairs)
    write_csv(out/'full_vs_common_query_support.csv', boundary_rows)
    increments = {}
    for name, fields in (('all', ['arm']), ('environment', ['arm', 'environment']), ('band', ['arm', 'band']),
                         ('query', ['arm', 'query']), ('environment_band', ['arm', 'environment', 'band'])):
        increments[name] = aggregate_pairs(pairs, fields); write_csv(out/f'paired_increments_{name}.csv', increments[name])
    for a in audits:
        for model_arm in evs[a]['arms']:
            write_csv(out/f'{a}_{model_arm}_all_frame_query.csv', evs[a]['arms'][model_arm]['records'])
    contact(samples, out/'fixed_pair_contact.png')
    shutil.copyfile(__file__, out/'executed_resolution_compare.py')
    result = dict(status='PASS', independent_audits=audits, identities=identities, depth_quality=summaries,
        paired_increments=increments, scores={a:{model:v['summary'] for model,v in ev['arms'].items()} for a,ev in evs.items()},
        frozen_cutoffs=prediction['native_vga']['cutoffs'], source_sha256=sha(__file__), wall_s=time.perf_counter()-start,
        boundary_support=[dict(input_arm=a, model_arm=model,
            excluded_boundary_support=sum(r['excluded_boundary_support'] for r in boundary_rows if r['input_arm']==a and r['model_arm']==model),
            support_decisions_changed=sum(r['full_support_hit'] != r['common_support_hit'] for r in boundary_rows if r['input_arm']==a and r['model_arm']==model))
            for a in audits for model in evs[a]['arms']],
        cpu_budget_s=args.budget_s, gpu_s=0, downloads_bytes=0,
        limitations='8 correlated frames from one capture; 447 boundary rays/frame excluded equally. Full reachable support can include UNKNOWN, so use known_witness_hits for measured positive hits. Same-frame improvement supports resolution effect here only; no improvement does not rule out interactions. Not body-volume clearance, old-16 comparison or broad camera generalization.')
    write(out/'comparison.json', result)
    print(json.dumps(dict(status=result['status'], quality=summaries, scores=result['scores'], wall_s=result['wall_s'])))


if __name__ == '__main__': main()
