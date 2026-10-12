"""Agent-reviewed visible-interior traces, not human/whole-object ground truth.

Coordinates are chosen from fixed RGB images before stereo/model inspection.
The original six boxes and paired-reference rules are inherited. Nonempty
negative units refer only to the sampled trace, never full scene clearance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from rgb_body_query_geometry import camera_queries, resize_intrinsics
from rgb_body_query_input_diagnostic import sha, source_frames, write
from rgb_body_query_reference_eval import (
    MIN_SUPPORT, NEGATIVE, POSITIVE, UNKNOWN, TARGET_CLASSES,
    confusion, interval_labels, load_depth, ratios, ray_interval, rays,
)


def raster_trace(trace, shape):
    points = np.asarray(trace['points_native'], float)
    width = int(trace['width_native_px'])
    h, w = shape
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or width < 1:
        raise ValueError('Need a positive-width visible polyline')
    if not np.isfinite(points).all() or (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError('Trace outside source RGB')
    image = Image.new('L', (w, h))
    ImageDraw.Draw(image).line([tuple(p) for p in points], fill=1, width=width)
    return np.asarray(image, dtype=bool)


def core_state(record):
    if record['state'] == 'POSITIVE':
        return 'POSITIVE_CORE_INTRUSION'
    if record['state'].startswith('NEGATIVE') and record['domain_pixels'] >= MIN_SUPPORT:
        return 'NEGATIVE_CORE_INTRUSION'
    return 'UNKNOWN'


def scaled_depth(array, shape):
    with np.load(array, allow_pickle=False) as a:
        z = a['depth'].astype(np.float32)
        valid = a['valid_mask'] if 'valid_mask' in a else np.isfinite(z) & (z > 0)
    # Never fill invalid predictions with interpolated valid neighbouring depth.
    depth = np.array(Image.fromarray(np.where(valid, z, 0.)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
    weight = np.asarray(Image.fromarray(valid.astype(np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
    keep = (weight >= 1.-1e-6) & np.isfinite(depth) & (depth > 0)
    depth[~keep] = np.nan
    return depth, keep


def sources(repo):
    frames = {(r['session'], r['camera'], r['frame']): r for r in source_frames(repo)}
    seq = repo/'artifacts.local/work/rgb-body-query-dev-20261009/sequence'
    plan = json.loads((seq/'planned_files.json').read_text('utf-8-sig'))
    receipt = json.loads((seq/'acquisition_receipt.json').read_text('utf-8-sig'))
    files = {(r['frame'], r['kind']): r for r in receipt['verified_files']}
    desc = next(r for r in plan['references'] if r['path'].endswith('description.json'))
    for frame in plan['frames']:
        row = dict(session=plan['session'], camera=plan['camera'], frame=frame,
                   description_path=Path(desc['path']), description_sha256=desc['sha256'])
        for alias, kind in [('rgb', 'video_frames'), ('mask', 'segmentation_masks'), ('cres', 'depth_maps'), ('zed', 'zed_depth_maps')]:
            row[alias+'_path'] = Path(files[frame, kind]['path'])
            row[alias+'_sha256'] = files[frame, kind]['sha256']
        frames[row['session'], row['camera'], frame] = row
    old_data = repo/'artifacts.local/datasets/sanpo-real-coverage-20260927'
    old_receipt = json.loads((repo/'artifacts.local/work/sanpo-coverage-20260927/acquisition_receipt.json').read_text('utf-8-sig'))
    hashes = {r['path'].replace('\\', '/'): r['sha256'] for r in old_receipt['files']}
    for row in frames.values():
        if 'cres_path' in row:
            continue
        base = f"{row['session']}/{row['camera']}/left"
        for alias, kind in [('cres', 'depth_maps'), ('zed', 'zed_depth_maps')]:
            relative = f"{base}/{kind}/{row['frame']:06d}.png"
            # Official SANPO depth payloads are gzip, even with .png suffix.
            if relative not in hashes:
                candidates = [p for p in hashes if p.startswith(base+'/'+kind+'/') and Path(p).name.split('.')[0] == f"{row['frame']:06d}"]
                if len(candidates) != 1:
                    raise ValueError(f'Missing unique cached {alias} for {relative}')
                relative = candidates[0]
            row[alias+'_path'] = old_data/relative
            row[alias+'_sha256'] = hashes[relative]
    return frames


def predictions(repo):
    by_frame = defaultdict(list)
    stage1 = repo/'artifacts.local/work/rgb-body-query-dev-20261009'
    selection = json.loads((stage1/'baseline_selection.json').read_text('utf-8-sig'))
    for i, r in enumerate(selection['rows']):
        for arm, suffix in [('DP-smoke-native', 'native'), ('DP-smoke-low128', 'low128')]:
            path = stage1/f'depthpro_{i:02d}_{suffix}.npz'
            by_frame[r['session'], r['camera'], r['frame']].append(dict(arm=arm, path=path, sha256=sha(path), rgb_sha256=r['rgb_sha256']))
    stage2 = repo/'artifacts.local/work/rgb-body-query-eval-dev-20261009'
    for folder in ('depthpro', 'vda'):
        manifest = json.loads((stage2/folder/'prediction_manifest.json').read_text('utf-8-sig'))
        for r in manifest['rows']:
            by_frame[r['session'], r['camera'], r['frame']].append(dict(arm=r['arm'], path=Path(r['depth_path']), sha256=r['depth_sha256'], rgb_sha256=r['rgb_sha256'], K=r['public_K']))
    return by_frame


def run(repo, manifests, output, budget_s):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    for path in manifests:
        traces.extend(json.loads(path.read_text('utf-8-sig'))['traces'])
    names = [(t['session'], t['camera'], t['frame'], t['name']) for t in traces]
    if len(names) != len(set(names)):
        raise ValueError('Trace identity duplicated')
    source, pred_sources = sources(repo), predictions(repo)
    grouped = defaultdict(list)
    for t in traces:
        if t['annotation_provenance'] != 'agent_visual_review':
            raise ValueError('This adapter requires explicit agent-review provenance')
        grouped[t['session'], t['camera'], t['frame']].append(t)
    records, trace_rows, frame_rows = [], [], []
    for key, tt in grouped.items():
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('CPU label completion budget reached')
        src = source[key]
        for alias in ('rgb', 'mask', 'description', 'cres', 'zed'):
            if sha(src[alias+'_path']) != src[alias+'_sha256']:
                raise ValueError('Changed source '+alias)
        with Image.open(src['rgb_path']) as image:
            native = image.size[1], image.size[0]
            rgb = image.convert('RGB')
        cres = load_depth(src['cres_path'])
        zed = np.asarray(Image.fromarray(load_depth(src['zed_path'])).resize((cres.shape[1], cres.shape[0]), Image.Resampling.NEAREST))
        with Image.open(src['mask_path']) as image:
            semantic = np.asarray(image.resize((cres.shape[1], cres.shape[0]), Image.Resampling.NEAREST))[..., 0]
        desc = json.loads(src['description_path'].read_text('utf-8-sig'))
        c = desc['session_camera_details'][desc['session_camera_location'].index(src['camera'])]['left_camera_params']
        k = resize_intrinsics([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]], native, cres.shape)
        rx, ry = rays(k, cres.shape)
        pp = []
        for p in pred_sources[key]:
            if p['rgb_sha256'] != src['rgb_sha256'] or sha(p['path']) != p['sha256']:
                raise ValueError('Prediction input identity differs')
            if 'K' in p and not np.allclose(p['K'], k, atol=1e-10, rtol=1e-10):
                raise ValueError('Common prediction K differs')
            z, valid = scaled_depth(p['path'], cres.shape)
            pp.append((p, z, valid))
        union = np.zeros(cres.shape, bool)
        overlay = rgb.copy()
        draw = ImageDraw.Draw(overlay)
        for ti, t in enumerate(tt):
            if t['rgb_sha256'] != src['rgb_sha256']:
                raise ValueError('Coordinates were reviewed on different RGB')
            mask = raster_trace(t, native)
            target = np.asarray(Image.fromarray(mask).resize((cres.shape[1], cres.shape[0]), Image.Resampling.NEAREST), bool)
            union |= target
            labels, qs = [], []
            for q in camera_queries():
                label, r = interval_labels(cres, zed, target, rx, ry, q)
                state = core_state(r)
                r.update(core_state=state)
                labels.append(label); qs.append(r)
                entry, exit, reachable = ray_interval(rx, ry, q)
                for p, z, valid in pp:
                    predicted = valid & reachable & (z >= entry-1e-12) & (z <= exit+1e-12)
                    n = int((predicted & target).sum())
                    records.append(dict(session=key[0], camera=key[1], frame=key[2], trace=t['name'],
                                        arm=p['arm'], query=q['name'], reference_state=state,
                                        prediction=n >= MIN_SUPPORT, oracle_support_pixels=n,
                                        reference_domain_pixels=r['domain_pixels'], **confusion(label, predicted)))
            name = f"{src['session']}_{src['frame']:06d}_{ti:02d}"
            path = output/(name+'.npz')
            np.savez(path, labels=np.stack(labels), core=target, K=k)
            trace_rows.append(dict(**t, path=str(path.resolve()), sha256=sha(path),
                                   core_grid_pixels=int(target.sum()), semantic_counts=dict(Counter(map(int, semantic[target]))),
                                   official_pool_covered_pixels=int((target & np.isin(semantic,TARGET_CLASSES)).sum()), queries=qs))
            draw.line([tuple(p) for p in t['points_native']], fill='#ff3366', width=max(5,t['width_native_px']))
            x,y=t['points_native'][0]; draw.text((x+6,y), t['name'], fill='yellow')
        overlay.thumbnail((1400,1400))
        overlay.save(output/f'{src["session"]}_{src["frame"]:06d}_overlay.png')
        augmented_labels, augmented_queries = [], []
        for q in camera_queries():
            label, record = interval_labels(cres,zed,np.isin(semantic,TARGET_CLASSES)|union,rx,ry,q)
            augmented_labels.append(label); augmented_queries.append(record)
        frame_ref = output/f'{src["session"]}_{src["frame"]:06d}_augmented.npz'
        np.savez(frame_ref, labels=np.stack(augmented_labels), target=np.isin(semantic,TARGET_CLASSES)|union, K=k)
        frame_rows.append(dict(session=key[0], camera=key[1], frame=key[2],
                               rgb_sha256=src['rgb_sha256'], core_pixels=int(union.sum()),
                               path=str(frame_ref.resolve()), sha256=sha(frame_ref),
                               annotation_type='MIXED_OFFICIAL_HUMAN_AGENT_CORE',
                               annotation_provenance='Official anchor semantic plus supplementary agent-reviewed traces',
                               queries=augmented_queries, augmented_queries=augmented_queries))
    summaries={}
    for arm in sorted({r['arm'] for r in records}):
        rr=[r for r in records if r['arm']==arm]
        c={name:sum(r[name] for r in rr) for name in ('tp','fn','fp','tn')}
        qc=dict(tp=0,fn=0,fp=0,tn=0)
        for r in rr:
            if r['reference_state']=='UNKNOWN':continue
            field=('tp' if r['prediction'] else 'fn') if r['reference_state']=='POSITIVE_CORE_INTRUSION' else ('fp' if r['prediction'] else 'tn')
            qc[field]+=1
        summaries[arm]=dict(frames=len({(r['session'],r['frame']) for r in rr}),
                            pixel_confusion=c,pixel_metrics=ratios(c),core_query_confusion=qc,
                            unknown_core_queries=sum(r['reference_state']=='UNKNOWN' for r in rr))
    result=dict(status='COMPLETE',annotation_provenance='agent_visual_review; no human verification of supplemental coordinates',
                source_manifests=[dict(path=str(p.resolve()),sha256=sha(p)) for p in manifests],
                frames=len(frame_rows),sessions=len({r['session'] for r in frame_rows}),traces=len(trace_rows),
                state_counts=dict(Counter(q['core_state'] for t in trace_rows for q in t['queries'])),
                empty_angular_domains=sum(q['domain_pixels']==0 for t in trace_rows for q in t['queries']),
                reference_grid_lost_traces=sum(t['core_grid_pixels']==0 for t in trace_rows),
                unit='sampled visible-interior trace x original camera query; closed boxes, support16 and original agreement',
                query_negatives='nonempty>=16 ray pixels, no positive or unresolved core ray; never full-scene clearance',
                scope='GT core localization oracle; excludes untraced structure/boundaries, not independent RGB detection',
                seconds=time.perf_counter()-start,summaries=summaries,trace_rows=trace_rows,frame_rows=frame_rows,rows=records)
    write(output/'labels_evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','trace_rows','frame_rows')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--annotations',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=600)
    a=p.parse_args();run(a.repo.resolve(),a.annotations,a.output.resolve(),a.budget_s)
