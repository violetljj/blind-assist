"""Post-hoc audit of frozen MZ10; no model fitting or deployed rule selection.

Scalar threshold frontiers use consumed DEV truth and are descriptive only.
They are not independent validation or a general separability bound.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble, load_npz, read, sha, write
from mz9_source_readout import SourceReadout


def describe(values):
    return None if not len(values) else np.quantile(values, [0, .25, .5, .75, 1]).tolist()


def auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def frontier(score, truth, available, baseline):
    """Enumerate monotone SOURCE thresholds, keeping the fallback fixed."""
    budget = int(((baseline >= 0) & ~truth).sum())
    fallback_fp = int(((baseline >= 0) & ~truth & ~available).sum())
    fallback_tp = int(((baseline >= 0) & truth & ~available).sum())
    rows = []
    for threshold in np.r_[-np.inf, np.unique(score[available]), np.inf]:
        pred = available & (score >= threshold)
        rows.append(dict(threshold=float(threshold) if np.isfinite(threshold) else str(threshold),
                         tp=int((pred & truth).sum()) + fallback_tp,
                         fp=int((pred & ~truth).sum()) + fallback_fp))
    admissible = [r for r in rows if r['fp'] <= budget]
    best = max(admissible, key=lambda r: (r['tp'], -r['fp']))
    return dict(budget=budget, fallback_fp=fallback_fp, source_budget=budget-fallback_fp,
                baseline_tp=int(((baseline >= 0) & truth).sum()),
                best_descriptive_point=best, points=rows)


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    run = root/'artifacts.local/work/mz10-availability-20260910/run-v2'
    cache = root/'artifacts.local/work/mz8-attribution-20260910/cache-v5'
    receipt = read(run/'receipt.json')
    assert sha(run/'predictions.npz') == receipt['outputs']['predictions.npz']
    cache_receipt = read(cache/'receipt.json')
    assert sha(cache/'receipt.json') == receipt['cache_receipt_sha256']
    assert sha(cache/'observations.npz') == cache_receipt['files']['observations.npz']
    data = load_npz(run/'predictions.npz')
    obs = load_npz(cache/'observations.npz')
    checkpoint = root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint) == 'ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    fixed = CompactEnsemble.from_checkpoint(checkpoint).cuda()
    geometry = SourceReadout().cuda().eval()
    summaries, records, arrays = {}, [], {}
    for cohort in ['DEV', 'clean', 'stress']:
        ids = data['DEV/cache_index'] if cohort == 'DEV' else np.arange(3500, 3700)
        ranges = obs['stress_ranges'] if cohort == 'stress' else obs['ranges'][ids]
        valid = obs['stress_valid'] if cohort == 'stress' else obs['valid'][ids]
        rgb, tof, counts, zones, returns = [], [], [], [], []
        with torch.inference_mode():
            for begin in range(0, len(ids), 32):
                end = begin+32
                r = torch.from_numpy(ranges[begin:end]).cuda()
                v = torch.from_numpy(valid[begin:end]).cuda()
                packed = torch.cat([r.flatten(1)/4, v.flatten(1).float()], 1).float()
                a, b = fixed.branches(torch.from_numpy(obs['visual'][ids[begin:end]]).cuda(), packed)
                rgb.append(a.cpu().numpy()); tof.append(b.cpu().numpy())
                eligible = geometry.eligibility(r, v)[2]
                counts.append(eligible.sum((1, 2, 3)).cpu().numpy())
                zones.append(eligible.any(3).any(2).sum(1).cpu().numpy())
                returns.append(eligible.any(3).sum((1, 2)).cpu().numpy())
        features = dict(rgb=np.concatenate(rgb), tof=np.concatenate(tof),
                        candidate_count=np.concatenate(counts), zone_count=np.concatenate(zones),
                        return_count=np.concatenate(returns))
        b, s, c, t, a = [data[cohort+'/'+k] for k in ['baseline', 'source', 'composite', 'truth', 'available']]
        assert np.array_equal(features['candidate_count'] > 0, a)
        assert np.array_equal((features['rgb'] + features['tof']) >= 0, b >= 0)
        features.update(baseline=b, source=s, source_minus_baseline=s-b,
                        rgb_minus_tof=features['rgb']-features['tof'])
        groups = dict(old_fp=(b>=0)&~t, new_fp=(c>=0)&(b<0)&~t,
                      gained_tp=(c>=0)&(b<0)&t, lost_tp=(c<0)&(b>=0)&t)
        summaries[cohort] = {}
        for q, name in enumerate(['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']):
            summary = dict(groups={}, univariate_auc={}, frontier=frontier(s[:, q], t[:, q], a[:, q], b[:, q]))
            for group, mask in groups.items():
                m = mask[:, q]
                summary['groups'][group] = dict(count=int(m.sum()), source_selected=int((m&a[:,q]).sum()),
                    features={k:describe(v[m & a[:,q] if k in ['source','source_minus_baseline'] else m,q]) for k,v in features.items()})
            for key, values in features.items():
                summary['univariate_auc'][key] = auc(values[groups['gained_tp'][:,q],q], values[groups['new_fp'][:,q],q])
            summaries[cohort][name] = summary
            for i in range(len(t)):
                records.append(dict(cohort=cohort,index=i,query=name,truth=bool(t[i,q]),available=bool(a[i,q]),
                    groups=[k for k,m in groups.items() if m[i,q]],
                    features={k:float(v[i,q]) if a[i,q] or k not in ['source','source_minus_baseline'] else None for k,v in features.items()}))
        arrays.update({cohort+'/'+k:v for k,v in features.items()})
    write(output/'summary.json', summaries)
    write(output/'records.json', records)
    np.savez_compressed(output/'features.npz', **arrays)
    write(output/'receipt.json', dict(status='PASS',kind='EXISTING_MZ10_POSTHOC_AUDIT',training_steps=0,
        deployed_threshold_changes=0,independent_confirmation=False,backend='CUDA geometry/branches; CPU scalar summaries',
        device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),
        mz10_receipt_sha256=sha(run/'receipt.json'),cache_receipt_sha256=sha(cache/'receipt.json'),
        outputs={p.name:sha(p) for p in output.iterdir()}))
    print({cohort:{q:{'counts':{g:v['count'] for g,v in x['groups'].items()},
        'frontier':{k:v for k,v in x['frontier'].items() if k!='points'},'auc':x['univariate_auc']}
        for q,x in value.items()} for cohort,value in summaries.items()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.output)
