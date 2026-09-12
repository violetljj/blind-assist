"""Fixed observable ZNCC features, then separate consumed-data attribution.

No production filter is installed. Envelope cutoffs are posthoc diagnostic bounds.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import mz101_spatial as m
from run_mz101_spatial import sha, metrics, observation_contract
from run_mz103_depth_frontend import PANELS, event_compare, write

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, Workload, select_backend, torch_observation

TASK = ROOT / 'artifacts.local/work/mz105-residual-matching-20260912'
REF = ROOT / 'artifacts.local/work/mz103-depth-frontend-20260912/native-v1'
SIGNALS = ('assigned_zncc', 'far_margin', 'competitor_margin')
CRITICAL = ('thin_left', 'thin_right', 'small_head', 'occluded_thin')
FOCAL_BASELINE = m.RIG['width'] / (2 * math.tan(math.radians(35))) * .1


def membership(depth, pose):
    yy, xx = np.nonzero(np.isfinite(depth))
    points = m.depth_points(depth)
    fov = ((np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0]))) <= 22.5)
           & (np.abs(np.degrees(np.arctan2(points[:, 2], points[:, 0]))) <= 20.))
    body = points @ m.rotation(pose['yaw'], pose['pitch'], pose['roll']).T + pose['camera_in_body_m']
    inside = np.stack([fov & ((body >= lo) & (body <= hi)).all(1) for lo, hi in m.BOXES], 1)
    return yy, xx, inside


@torch.no_grad()
def zncc_volume(left, right, device):
    """Return actual-device cost volume; full windows, no wraparound matches."""
    # Float64 avoids catastrophic variance cancellation in flat rendered patches.
    l = torch.as_tensor(left, device=device, dtype=torch.float64)[None, None] - 128.
    r = torch.as_tensor(right, device=device, dtype=torch.float64)[None, None] - 128.
    h, w = left.shape
    disp = torch.arange(96, device=device)[:, None, None]
    xx = torch.arange(w, device=device)[None, None, :]
    yy = torch.arange(h, device=device)[None, :, None]
    xr = (xx - disp).expand(-1, h, -1)
    shifted = r[0, 0][yy.expand(96, -1, w), xr.clamp(0, w-1)][:, None]
    avg = lambda v: F.avg_pool2d(v, 5, stride=1, padding=2)
    lm, rm = avg(l), avg(shifted)
    lv = (avg(l*l) - lm*lm).clamp_min(0)
    rv = (avg(shifted*shifted) - rm*rm).clamp_min(0)
    denom = torch.sqrt(lv*rv)
    score = ((avg(l*shifted) - lm*rm) / denom.clamp_min(1e-10)).clamp(-1, 1)[:, 0]
    ok = ((xr >= 2) & (xr < w-2) & (xx >= 2) & (xx < w-2)
          & (yy >= 2) & (yy < h-2) & (denom[:, 0] > 1e-8))
    return torch.where(ok, score, -torch.inf)


def point_features(volume, yy, xx, depth):
    device = volume.device
    d = torch.as_tensor(np.rint(FOCAL_BASELINE/depth[yy, xx]).astype('int64'), device=device)
    assert bool(((d >= 0) & (d < 96)).all())
    costs = volume[:, torch.as_tensor(yy, device=device), torch.as_tensor(xx, device=device)]
    assigned = costs[d, torch.arange(len(d), device=device)]
    far = costs[:math.ceil(FOCAL_BASELINE/4)].max(0).values
    competitors = torch.abs(torch.arange(96, device=device)[:, None]-d[None]) > 1
    other = torch.where(competitors, costs, -torch.inf).max(0).values
    out = torch.stack([assigned, assigned-far, assigned-other], 1)
    return torch.where(torch.isfinite(out), out, -torch.inf).cpu().numpy()


def verify_receipt(folder):
    receipt = json.loads((folder/'receipt.json').read_text())
    for name, digest in receipt['hashes'].items():
        assert sha(folder/name) == digest, (folder, name)
    return receipt


def produce():
    output = TASK/'features-v3'
    if output.exists():
        raise ValueError('Features output already exists; preserve it')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    first = ROOT/'artifacts.local/work'/PANELS['mz101'][0]/'capture-v1/frame/head_bar_textured_00'
    left, right = [cv2.imread(str(first/(view+'.png')), 0) for view in ('left', 'right')]
    cpu = BackendCandidate('torch-cpu', 'cpu', lambda: zncc_volume(left, right, 'cpu'),
                           lambda x: torch_observation(output=x))
    gpu = (BackendCandidate('torch-cuda', 'cuda', lambda: zncc_volume(left, right, 'cuda'),
                            lambda x: torch_observation(output=x), torch.cuda.synchronize)
           if torch.cuda.is_available() else None)
    backend = select_backend(Workload.BATCH_TENSOR, cpu=cpu, gpu=gpu,
        cpu_reason='ACCELERATOR_UNAVAILABLE' if gpu is None else None,
        record_path=TASK/'backend.json', warmups=1, repeats=2)
    device = backend['selected_device_type']
    if gpu is not None:
        a, b = zncc_volume(left, right, 'cpu'), zncc_volume(left, right, 'cuda').cpu()
        torch.testing.assert_close(torch.isfinite(a), torch.isfinite(b))
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)
    inputs, timing = {}, []
    verify_receipt(REF)
    for panel, (name, old_dir, depth_dir, _) in PANELS.items():
        folder = ROOT/'artifacts.local/work'/name
        old, capture = folder/old_dir, folder/'capture-v1'
        cached_receipt = verify_receipt(old)
        captured = json.loads((capture/'receipt.json').read_text())
        obs = json.loads((REF/panel/'observations.json').read_text())
        saved = np.load(REF/panel/'predictions.npz')
        dest = output/panel
        dest.mkdir()
        write(dest/'observations.json', obs)
        for i, o in enumerate(obs):
            started = time.perf_counter()
            raw = capture/'frame'/o['id']
            for view in ('left', 'right', 'tof-range', 'tof-valid'):
                filename = view + ('.png' if view in ('left', 'right') else '.npy')
                path = raw/filename
                digest = sha(path)
                assert digest == captured['hashes'][f"frame/{o['id']}/{filename}"]
                inputs[str(path.relative_to(ROOT))] = digest
            path = old/depth_dir/(o['id']+'.npy')
            inputs[str(path.relative_to(ROOT))] = sha(path)
            depth = np.load(path)
            # Original summary receipts do not include cached-depth payloads.
            # Recompute the unchanged CPU-only OpenCV implementation for exact
            # cache identity, including LR validity and connected components.
            rgb = [cv2.imread(str(raw/(v+'.png'))) for v in ('left', 'right')]
            replayed, _ = m.stereo_depth(*rgb)
            np.testing.assert_array_equal(depth, replayed)
            stereo = m.readout(m.depth_points(depth), o['pose'])[0]
            tof = m.readout(m.tof_points(np.load(raw/'tof-range.npy'), np.load(raw/'tof-valid.npy')), o['pose'])[0]
            np.testing.assert_array_equal(stereo, saved['sgbm_support'][i])
            np.testing.assert_array_equal(tof, saved['tof_support'][i])
            yy, xx, inside = membership(depth, o['pose'])
            keep = inside.any(1)
            yy, xx, inside = yy[keep], xx[keep], inside[keep]
            np.testing.assert_array_equal(inside.any(0), stereo > 0)
            left, right = [cv2.imread(str(raw/(v+'.png')), 0) for v in ('left', 'right')]
            volume = zncc_volume(left, right, device)
            scores = point_features(volume, yy, xx, depth) if len(yy) else np.empty((0, 3), 'float32')
            np.savez_compressed(dest/(o['id']+'.npz'), yy=yy, xx=xx, inside=inside,
                                scores=scores, stereo=stereo, tof=tof)
            del volume
            timing.append(time.perf_counter()-started)
            if (i+1) % 48 == 0:
                print(f'{panel} features {i+1}/288', flush=True)
    write(output/'seal.json', dict(status='FEATURES_SEALED_BEFORE_EVALUATOR_ACCESS',
        inputs=inputs, source_sha256=sha(Path(__file__)),
        hashes={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
        frames=len(timing), mean_full_frame_s=float(np.mean(timing)),
        p95_full_frame_s=float(np.percentile(timing, 95)), signals=SIGNALS,
        backend=backend['selected_backend'], baseline_support_parity=True))
    print('FEATURES SEALED', flush=True)


def counts(p, gt):
    return dict(TP=int((p & gt).sum()), FP=int((p & ~gt).sum()), FN=int((~p & gt).sum()))


def summarize(preds, data):
    rows = {}
    for panel, d in data.items():
        p, gt, base = preds[panel], d['gt'], d['base']
        final = m.hysteresis(p, [o['episode'] for o in d['obs']])
        scored = metrics(final, gt, d['obs'])
        old = metrics(d['final'], gt, d['obs'])
        critical = {}
        for family in CRITICAL:
            target = base & gt & (d['family'] == family)[:, None]
            critical[family] = dict(retained=int((p & target).sum()), total=int(target.sum()))
        rows[panel] = dict(raw=counts(p, gt), final={k:v for k,v in scored.items() if k!='event_details'},
            transitions=dict(lost_raw_tp=int((base & ~p & gt).sum()), removed_raw_fp=int((base & ~p & ~gt).sum()),
                lost_final_tp=int((d['final'] & ~final & gt).sum()), new_final_tp=int((~d['final'] & final & gt).sum())),
            critical=critical, event_comparison=event_compare(scored, old), unknown=int((~p).sum()))
    return rows


def evaluate():
    features, output = TASK/'features-v3', TASK/'evaluation-v1'
    if output.exists():
        raise ValueError('Evaluation already exists; preserve it')
    seal = json.loads((features/'seal.json').read_text())
    assert seal['frames'] == 576
    for name, digest in seal['hashes'].items():
        assert sha(features/name) == digest, name
    output.mkdir()
    data, origins, query_rows = {}, {}, []
    for panel, (name, _, _, _) in PANELS.items():
        capture = ROOT/'artifacts.local/work'/name/'capture-v1'
        captured = json.loads((capture/'receipt.json').read_text())
        obs = json.loads((features/panel/'observations.json').read_text())
        spec_path = capture/'spec.json'
        assert sha(spec_path) == captured['spec_sha256']
        spec = json.loads(spec_path.read_text())
        assert observation_contract(spec) == obs
        gt = np.load(REF/panel/'truth.npy')
        old = np.load(REF/panel/'predictions.npz')
        base = old['sgbm_union_support'] > 0
        scores = np.full((288, 2, 3), -np.inf, dtype='float32')
        c = Counter()
        for i, o in enumerate(obs):
            f = np.load(features/panel/(o['id']+'.npz'))
            native_path = capture/'frame'/o['id']/'native-left-depth.npy'
            assert sha(native_path) == captured['hashes'][f"frame/{o['id']}/native-left-depth.npy"]
            native = np.load(native_path)
            yy, xx, inside = f['yy'], f['xx'], f['inside']
            truth_depth = native[yy, xx]
            reference = np.full_like(native, np.nan)
            eligible = np.isfinite(truth_depth) & (truth_depth >= .5) & (truth_depth <= 4)
            reference[yy[eligible], xx[eligible]] = truth_depth[eligible]
            ny, nx, ninside = membership(reference, o['pose'])
            native_inside = np.zeros((*native.shape, 2), bool)
            native_inside[ny, nx] = ninside
            for j, part in enumerate(m.PARTS):
                ix = inside[:, j]
                if ix.any():
                    scores[i, j] = np.max(f['scores'][ix], axis=0)
                row = dict(panel=panel, id=o['id'], part=part, truth=bool(gt[i,j]),
                    baseline=bool(base[i,j]), stereo=bool(f['stereo'][j]), tof=bool(f['tof'][j]),
                    family=spec['frames'][i]['family'], scores=[float(x) if np.isfinite(x) else None for x in scores[i,j]])
                if base[i,j] and not gt[i,j]:
                    n = truth_depth[ix]
                    far = np.isfinite(n) & (n > 4)
                    near = np.isfinite(n) & (n >= .5) & (n <= 4)
                    ins = native_inside[yy[ix], xx[ix], j]
                    detail = dict(pixels=int(ix.sum()), far=int(far.sum()),
                        near_inside=int((near & ins).sum()), near_outside=int((near & ~ins).sum()),
                        below_range=int((np.isfinite(n) & (n < .5)).sum()), unavailable=int((~np.isfinite(n)).sum()))
                    row['origin'] = detail
                    c['raw_false_queries'] += 1
                    c['tof_supported'] += bool(f['tof'][j])
                    c['stereo_only'] += bool(f['stereo'][j]) and not bool(f['tof'][j])
                    c['tof_only'] += bool(f['tof'][j]) and not bool(f['stereo'][j])
                    c['both'] += bool(f['tof'][j]) and bool(f['stereo'][j])
                    c['any_far'] += bool(far.any())
                    c['only_far'] += bool(len(n) and far.all())
                    c['only_far_stereo_only'] += bool(len(n) and far.all() and not f['tof'][j])
                    for k,v in detail.items():
                        c['pixels_'+k] += v
                query_rows.append(row)
        origins[panel] = dict(c)
        data[panel] = dict(gt=gt, base=base, final=old['sgbm_union'], obs=obs, scores=scores,
            tof=old['tof_support'] > 0, family=np.array([f['family'] for f in spec['frames']]))
    results = {}
    for k, name in enumerate(SIGNALS):
        vals = np.unique(np.concatenate([d['scores'][:,:,k].ravel() for d in data.values()]))
        # Include accept-all (-inf) and reject-all stereo (+inf); availability-only is explicit.
        thresholds = np.r_[vals, np.inf]
        best = {}
        for mode in ('all_raw_tp', 'relaxed_95_percent'):
            selected = None
            for threshold in thresholds:
                pred = {p: d['tof'] | (d['base'] & (d['scores'][:,:,k] >= threshold)) for p,d in data.items()}
                ok = True
                for p,d in data.items():
                    target = d['base'] & d['gt']
                    required = 1. if mode == 'all_raw_tp' else .95
                    ok &= (pred[p] & target).sum() >= math.ceil(required*target.sum())
                    for fam in CRITICAL:
                        group = target & (d['family'] == fam)[:,None]
                        ok &= (pred[p] & group).sum() >= math.ceil(required*group.sum())
                if ok:
                    selected = (threshold, pred)
            threshold, pred = selected
            best[mode] = dict(threshold=float(threshold) if np.isfinite(threshold) else str(threshold),
                authority='POSTHOC_ATTAINABLE_ENVELOPE_NOT_SELECTED_METHOD', panels=summarize(pred, data))
        finite_pred = {p: d['tof'] | (d['base'] & np.isfinite(d['scores'][:,:,k])) for p,d in data.items()}
        best['finite_score_only'] = summarize(finite_pred, data)
        best['score_quantiles_stereo_only'] = {}
        for p,d in data.items():
            groups = {}
            for label, mask in [('TP', d['base'] & d['gt'] & ~d['tof']), ('FP', d['base'] & ~d['gt'] & ~d['tof'])]:
                v = d['scores'][:,:,k][mask]
                finite = v[np.isfinite(v)]
                groups[label] = dict(total=len(v), unavailable=int((~np.isfinite(v)).sum()),
                    p0_p25_p50_p75_p100=np.percentile(finite,[0,25,50,75,100]).tolist() if len(finite) else [])
            best['score_quantiles_stereo_only'][p] = groups
        results[name] = best
    write(output/'queries.json', query_rows)
    write(output/'summary.json', dict(authority='CONSUMED_DEVELOPMENT_DIAGNOSTIC_NO_INTEGRATION',
        frames=576, origins=origins, baseline=summarize({p:d['base'] for p,d in data.items()}, data),
        signals=results, feature_seal_sha256=sha(features/'seal.json')))
    write(output/'receipt.json', dict(status='PASS', hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(origins, indent=2), flush=True)
    for name, result in results.items():
        print(name, {mode: {p:r['raw'] for p,r in result[mode]['panels'].items()}
              for mode in ('all_raw_tp','relaxed_95_percent')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('produce', 'evaluate'))
    args = parser.parse_args()
    TASK.mkdir(parents=True, exist_ok=True)
    produce() if args.mode == 'produce' else evaluate()
