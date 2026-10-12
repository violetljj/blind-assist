"""Cached scene diagnostics; no image encoding or threshold fitting.

Cross-scene pairs retain fixed public queries and all frame Cartesian pairs.
Exact native-slot matching is reported separately from approximate public-ray
matching. Query-ray pairs are correlated observations, never independent N.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval


COUNT_KEYS = ('positive', 'free', 'rgb_tp', 'geometry_tp', 'rgb_fp', 'geometry_fp',
              'positive_rescue', 'positive_loss', 'free_rescue', 'free_added',
              'positive_queries', 'rgb_query_support', 'geometry_query_support',
              'rgb_known_witness', 'geometry_known_witness')
MARGIN_KEYS = ('margin_ge_0p01','margin_le_minus_0p01','margin_abs_lt_0p01')
HISTOGRAM_KEYS = tuple(f'margin_hist_bin_{i}' for i in range(8))
MARGIN_BINS = [-float('inf'), -.1, -.01, -.001, 0., .001, .01, .1, float('inf')]


def paired_counts(labels, rgb, geometry):
    pos, free = labels == 1, labels == 0
    result = dict(positive=int(pos.sum()), free=int(free.sum()),
                  rgb_tp=int((pos & rgb).sum()), geometry_tp=int((pos & geometry).sum()),
                  rgb_fp=int((free & rgb).sum()), geometry_fp=int((free & geometry).sum()),
                  positive_rescue=int((pos & rgb & ~geometry).sum()),
                  positive_loss=int((pos & ~rgb & geometry).sum()),
                  free_rescue=int((free & ~rgb & geometry).sum()),
                  free_added=int((free & rgb & ~geometry).sum()))
    assert result['positive_rescue'] - result['positive_loss'] == result['rgb_tp'] - result['geometry_tp']
    assert result['free_added'] - result['free_rescue'] == result['rgb_fp'] - result['geometry_fp']
    return result


def rate(n, d):
    return n/d if d else None


def metrics(row):
    return dict(row, rgb_recall=rate(row['rgb_tp'], row['positive']),
                geometry_recall=rate(row['geometry_tp'], row['positive']),
                rgb_fpr=rate(row['rgb_fp'], row['free']),
                geometry_fpr=rate(row['geometry_fp'], row['free']))


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key not in groups:
            groups[key] = dict(zip(keys, key), **{k: 0 for k in COUNT_KEYS}, frame_query_records=0, _frames=set())
        for k in COUNT_KEYS:
            groups[key][k] += row[k]
        groups[key]['frame_query_records'] += 1
        groups[key]['_frames'].add((row['scan'],row['frame']))
    for row in groups.values():
        row['frames'] = len(row.pop('_frames'))
    return [metrics(row) for row in groups.values()]


def correspondence(k1, k2, shape, mapped, tolerance):
    """Fixed public-K-only nearest correspondence; exclude beyond tolerance.

    Difference is Euclidean norm of normalized optical (x/z,y/z) directions.
    The mapped view is approximate and does not claim identical directions.
    """
    rx, ry = rays(k1, shape)
    h, w = shape
    yy, xx = np.indices(shape)
    if mapped:
        hom = np.stack([rx, ry, np.ones_like(rx)], -1) @ np.asarray(k2).T
        mx = np.rint(hom[..., 0]/hom[..., 2]).astype(int)
        my = np.rint(hom[..., 1]/hom[..., 2]).astype(int)
    else:
        mx, my = xx, yy
    valid = (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
    mx, my = np.clip(mx, 0, w-1), np.clip(my, 0, h-1)
    bx, by = rays(k2, shape)
    delta = np.hypot(rx-bx[my, mx], ry-by[my, mx])
    accepted = valid & (delta <= tolerance)
    indices = (my*w+mx).reshape(-1)
    return indices, accepted.reshape(-1), dict(
        direction_tolerance=tolerance, valid_projection_pixels=int(valid.sum()),
        accepted_pixels=int(accepted.sum()), total_native_pixels=h*w,
        minimum_direction_difference=float(delta[valid].min()),
        maximum_direction_difference=float(delta[valid].max()),
        accepted_maximum_direction_difference=float(delta[accepted].max()) if accepted.any() else None,
        exact_equal_direction_pixels=int((valid & (delta == 0)).sum()))


def concordance(a, b, mask, sign):
    delta = (a-b)[mask] * sign[mask]
    histogram = np.histogram(delta,bins=MARGIN_BINS)[0]
    return dict(pairs=int(delta.size), concordant=int((delta > 0).sum()),
                discordant=int((delta < 0).sum()), ties=int((delta == 0).sum()),
                score_delta_sum=float(delta.sum(dtype=np.float64)),
                max_absolute_delta=float(np.abs(delta).max()) if delta.size else None,
                margin_ge_0p01=int((delta>=.01).sum()),margin_le_minus_0p01=int((delta<=-.01).sum()),
                margin_abs_lt_0p01=int((np.abs(delta)<.01).sum()),
                **dict(zip(HISTOGRAM_KEYS,map(int,histogram))))


def finish_concordance(rows):
    total = {k: sum(row[k] for row in rows) for k in ('pairs', 'concordant', 'discordant', 'ties', 'score_delta_sum')}
    total['conditional_pair_auc'] = rate(total['concordant'] + .5*total['ties'], total['pairs'])
    total['mean_occupied_minus_free_score'] = rate(total['score_delta_sum'], total['pairs'])
    total['maximum_absolute_delta'] = max((r.get('max_absolute_delta', r.get('maximum_absolute_delta')) or 0 for r in rows), default=0)
    for k in MARGIN_KEYS + HISTOGRAM_KEYS:
        total[k] = sum(row.get(k,0) for row in rows)
    return total


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def canonical_geometry(source, model, k, shape, queries):
    """CPU-only frozen head readout at exactly one shared public ray input."""
    import torch
    torch.set_num_threads(4)
    checkpoint = source/model/'geometry_final.pt'
    receipt = json.loads((source/model/'receipt.json').read_text('utf-8-sig'))
    if sha(checkpoint) != receipt['stages']['train']['arms']['geometry']['sha256']:
        raise ValueError('Frozen geometry checkpoint identity changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
    rx, ry = rays(k, shape)
    directions = torch.tensor(np.stack([rx,ry],-1).reshape(-1,2), dtype=torch.float32)
    bounds = torch.tensor([q['low']+q['high'] for q in queries], dtype=torch.float32)
    bounds /= torch.tensor([.9,.55,6.,.9,.55,6.])
    if model == 'pilot':
        head = torch.nn.Sequential(torch.nn.Linear(40,64),torch.nn.ReLU(),
            torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        head.load_state_dict(state)
        features = torch.zeros((len(directions),32))
    else:
        head = torch.nn.Sequential(torch.nn.Linear(56,64),torch.nn.ReLU(),
            torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        head.load_state_dict({key.removeprefix('query.'):value for key,value in state.items() if key.startswith('query.')})
        features = torch.cat([torch.relu(state['low.bias']),torch.relu(state['high.bias'])])[None].expand(len(directions),-1)
    head.eval()
    output = []
    with torch.inference_mode():
        for q in bounds:
            x = torch.cat([features, directions, q[None].expand(len(directions),-1)],1)
            a = head(x).sigmoid().numpy()[:,0]
            b = head(x).sigmoid().numpy()[:,0]
            if not np.array_equal(a,b):
                raise ValueError('Identical geometry input produced different CPU score')
            output.append(a)
    return np.stack(output), dict(checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        same_input_repeated_calls=2*len(queries), exact_score_equality=True,
        maximum_score_difference=0., note='Identical public ray/query/zero-RGB input, same frozen head; no environment input')


def run(source, output, budget_s=240.):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'receipt.json').exists():
        raise FileExistsError('Preserve run evidence; choose new output')
    sensor = source/'sensor'
    manifest = json.loads((sensor/'dataset_manifest.json').read_text('utf-8-sig'))
    queries, rows = manifest['queries'], manifest['rows']
    selected = [i for i,r in enumerate(rows) if r['split'] in ('cal', 'validation')]
    provenance = [dict(path=str(sensor/'dataset_manifest.json'), sha256=sha(sensor/'dataset_manifest.json'))]
    labels, domains = {}, {}
    def check():
        if time.perf_counter()-started > budget_s:
            raise TimeoutError('Scene diagnostic budget reached')
    for i in selected:
        r = rows[i]
        if sha(r['reference_path']) != r['reference_sha256']:
            raise ValueError('Reference identity changed')
        with np.load(r['reference_path'], allow_pickle=False) as ref:
            labels[i] = np.array(ref['labels']).reshape(len(queries), -1)
        rx, ry = rays(r['depth_K'], r['depth_shape'])
        domains[i] = np.stack([ray_interval(rx,ry,q)[2] for q in queries]).reshape(len(queries), -1)
        provenance.append(dict(path=r['reference_path'], sha256=r['reference_sha256']))
    increment_rows, pair_rows, grid_rows, arm_results, canonical_checks = [], [], [], {}, {}
    val = [i for i in selected if rows[i]['split'] == 'validation']
    envs = sorted({rows[i]['environment'] for i in val})
    canonical_row = next(i for i in val if rows[i]['environment']==envs[0])
    canonical_k = rows[canonical_row]['depth_K']
    canonical_shape = rows[canonical_row]['depth_shape']
    for model in ('pilot', 'multiscale'):
        check()
        path = source/model/'evaluation.json'
        evaluation = json.loads(path.read_text('utf-8-sig'))
        provenance.append(dict(path=str(path), sha256=sha(path)))
        predictions = {}
        cutoffs = {}
        for arm, info in evaluation['arms'].items():
            cutoffs[arm] = info['cutoff']
            predictions[arm] = {}
            for p in info['predictions']:
                i = p['row']
                if i not in selected:
                    continue
                if sha(p['path']) != p['sha256']:
                    raise ValueError('Prediction identity changed')
                predictions[arm][i] = np.load(p['path'], allow_pickle=False).reshape(len(queries), -1)
                provenance.append(dict(path=p['path'], sha256=p['sha256']))
        for arm in ('rgb', 'rgb_shuffled'):
            for i, score in predictions[arm].items():
                r = rows[i]
                pred = (score >= cutoffs[arm]) & domains[i]
                control = (predictions['geometry'][i] >= cutoffs['geometry']) & domains[i]
                for qi, q in enumerate(queries):
                    counts = paired_counts(labels[i][qi], pred[qi], control[qi])
                    positive = r['queries'][qi]['state'] == 'POSITIVE'
                    counts.update(positive_queries=int(positive),
                                  rgb_query_support=int(positive and pred[qi].sum() >= 16),
                                  geometry_query_support=int(positive and control[qi].sum() >= 16),
                                  rgb_known_witness=int(positive and counts['rgb_tp'] >= 16),
                                  geometry_known_witness=int(positive and counts['geometry_tp'] >= 16))
                    increment_rows.append(dict(model=model, arm=arm, split=r['split'], environment=r['environment'],
                        scan=r['scan'], frame=r['frame'], query=q['name'], band=f"{q['low'][2]:g}-{q['high'][2]:g}m", **counts))
        canonical_scores, canonical_checks[model] = canonical_geometry(source,model,canonical_k,canonical_shape,queries)
        np.save(output/f'{model}_geometry_canonical.npy', canonical_scores, allow_pickle=False)
        for env_a, env_b in combinations(envs, 2):
            a = [i for i in val if rows[i]['environment'] == env_a]
            b = [i for i in val if rows[i]['environment'] == env_b]
            if len(a) != 8 or len(b) != 8:
                raise ValueError('Expected all eight fixed frames per environment')
            if any(rows[i]['depth_K'] != rows[a[0]]['depth_K'] for i in a) or any(rows[i]['depth_K'] != rows[b[0]]['depth_K'] for i in b):
                raise ValueError('Frame-varying K requires explicit matching')
            for method in ('same_native_slot', 'public_ray_nearest_approximate'):
                if method == 'same_native_slot':
                    indices, allowed, geometry = correspondence(rows[a[0]]['depth_K'], rows[b[0]]['depth_K'],
                        rows[a[0]]['depth_shape'], False, 1e-3)
                    indices_a = np.arange(indices.size)
                else:
                    indices_a, allowed_a, geometry_a = correspondence(canonical_k, rows[a[0]]['depth_K'], canonical_shape, True, 1e-3)
                    indices, allowed_b, geometry_b = correspondence(canonical_k, rows[b[0]]['depth_K'], canonical_shape, True, 1e-3)
                    # Require each scene ray close to canonical AND the actual pair close to each other.
                    arx,ary = rays(rows[a[0]]['depth_K'],canonical_shape)
                    brx,bry = rays(rows[b[0]]['depth_K'],canonical_shape)
                    direct_delta = np.hypot(arx.reshape(-1)[indices_a]-brx.reshape(-1)[indices],
                        ary.reshape(-1)[indices_a]-bry.reshape(-1)[indices])
                    allowed = allowed_a & allowed_b & (direct_delta <= 1e-3)
                    geometry = dict(direction_tolerance=1e-3, accepted_pixels=int(allowed.sum()),
                        total_native_pixels=int(allowed.size), canonical_a=geometry_a, canonical_b=geometry_b,
                        actual_pair_maximum_direction_difference=float(direct_delta[allowed].max()) if allowed.any() else None,
                        maximum_angle_error_bound_radians=float(direct_delta[allowed].max()) if allowed.any() else None,
                        position_difference_bound_at_6m=float(6*direct_delta[allowed].max()) if allowed.any() else None)
                grid_rows.append(dict(model=model, environment_a=env_a, environment_b=env_b, method=method, **geometry))
                for qi,q in enumerate(queries):
                    local = {arm: [] for arm in predictions}
                    if method != 'same_native_slot':
                        local['geometry_canonical_exact_input'] = []
                    for ai in a:
                        for bi in b:
                            check()
                            la, lb = labels[ai][qi,indices_a], labels[bi][qi, indices]
                            mask = allowed & (la < 2) & (lb < 2) & (la != lb)
                            sign = np.where(la == 1, 1., -1.)
                            for arm in predictions:
                                local[arm].append(concordance(predictions[arm][ai][qi,indices_a], predictions[arm][bi][qi, indices], mask, sign))
                            if method != 'same_native_slot':
                                local['geometry_canonical_exact_input'].append(concordance(canonical_scores[qi],canonical_scores[qi],mask,sign))
                    for arm, counters in local.items():
                        pair_rows.append(dict(model=model, arm=arm, environment_a=env_a, environment_b=env_b,
                            method=method, query=q['name'], frame_cartesian_pairs=len(a)*len(b), **finish_concordance(counters)))
        arm_results[model] = dict(cutoffs=cutoffs, thresholds_refitted=False)
    aggregates = {name: aggregate(increment_rows, keys) for name,keys in {
        'all': ['model','arm','split'],
        'environment': ['model','arm','split','environment'],
        'band': ['model','arm','split','band'],
        'environment_band': ['model','arm','split','environment','band'],
        'environment_query': ['model','arm','split','environment','query'],
    }.items()}
    pairs_summary = []
    for model in ('pilot','multiscale'):
        for arm in ('rgb','geometry','rgb_shuffled','geometry_canonical_exact_input'):
            for method in ('same_native_slot','public_ray_nearest_approximate'):
                selected_pairs = [r for r in pair_rows if r['model']==model and r['arm']==arm and r['method']==method]
                pairs_summary.append(dict(model=model, arm=arm, method=method, **finish_concordance(selected_pairs)))
    result = dict(status='COMPLETE', increments=aggregates, pair_summary=pairs_summary,
        public_grid_correspondence=grid_rows, thresholds=arm_results, canonical_geometry_checks=canonical_checks,
        canonical_grid=dict(environment=rows[canonical_row]['environment'],depth_K=canonical_k,shape=canonical_shape),
        fixed_margin_diagnostic=dict(threshold=.01,histogram_edges=['-inf',-.1,-.01,-.001,0.,.001,.01,.1,'inf'],
            histogram_interval='numpy left-inclusive/right-exclusive, final bin right-inclusive',
            contract='All arms and all pairs; cached-score diagnostic only, not a new classifier threshold'),
        contracts=['cal and validation are separate; training predictions are not cached',
                   'Every fixed query and every environment-pair 8x8 frame Cartesian combination retained',
                   'Pairs require opposite known sensor labels; UNKNOWN excluded',
                   'Exact same native slot and public-K nearest approximate correspondence reported separately',
                   'Approximate match uses fixed 1e-3 normalized optical-direction tolerance; no result-tuned matching',
                   'Conditional pair AUC is a concordance fraction on correlated query-ray pairs; no independence or significance claim',
                   'Query support need not have a known positive witness; both diagnostics retained',
                   'All original cal cutoffs retained; no inference, fitting, training or downloads'])
    write(output/'scene_diagnostic.json', result)
    write_csv(output/'frame_query_increments.csv', increment_rows)
    write_csv(output/'paired_scene_query.csv', pair_rows)
    for name, data in aggregates.items():
        write_csv(output/f'increments_{name}.csv', data)
    write(output/'receipt.json', dict(status='COMPLETE', wall_s=time.perf_counter()-started,
        cpu_execution_budget_s=budget_s, gpu_s=0, download_bytes=0, script_sha256=sha(Path(__file__)),
        sources=provenance, identities_verified=True, resource_release='No workers or service allocated',
        paired_increment_identities='rescue-loss=TPdelta; added-rescue=FPdelta checked for every frame/query'))
    print(json.dumps(dict(wall_s=time.perf_counter()-started, increments=aggregates['all'], paired=pairs_summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240.)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.budget_s)
