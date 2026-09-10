"""Independent saved-output, cutoff and exported residual-network audit."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz11_audit import scalar_metrics
from mz22_evidence_arbiter import EvidenceArbiter


def optimal_cutoff(scores, truth, budget):
    if not len(scores):
        return 0.
    levels = sorted(set(map(float, scores)), reverse=True)
    best = (0, 0, float(np.nextafter(np.float64(max(levels)), np.inf)))
    tp = fp = 0
    for level in levels:
        labels = truth[scores == level]
        tp += int(labels.sum())
        fp += int((~labels).sum())
        if fp <= budget:
            best = max(best, (tp, -fp, level))
    return best[2]


def main(root, run):
    receipt, start, result = (read(run / name) for name in ['receipt.json','start.json','result.json'])
    assert receipt['status'] == 'PASS'
    assert not (run / 'audit.json').exists(), 'Do not replace a prior audit'
    for name, digest in receipt['outputs'].items():
        assert sha(run / name) == digest, name
    for name, digest in start['inputs'].items():
        assert sha(name) == digest, name
    for name, digest in start['code_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert sha(Path(__file__).with_name('MZ22_EVIDENCE_ARBITRATION_PROTOCOL_20260910.md')) == start['protocol_sha256']
    work = root / 'artifacts.local/work'
    rank = work / 'mz20-rank-objective-20260910/run-v1'
    assert sha(run / 'batches.npy') == sha(rank / 'batches.npy')
    batches = np.load(run / 'batches.npy')
    assert batches.shape == (1200, 16) and len(np.unique(batches)) == start['unique_train'] == 7562
    old = load_npz(work / 'mz8-attribution-20260910/cache-v5/observations.npz')
    oldtruth = load_npz(work / 'mz8-attribution-20260910/cache-v5/evaluator.npz')['truth']
    newcache = work / 'mz15-shared-support-20260910/cache-v1'
    newtruth = load_npz(newcache / 'evaluator.npz')['truth']
    rows = read(newcache / 'selected.json')
    train_allowed = np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'), np.arange(3500,3700), np.arange(3700,11200)]
    assert np.isin(batches, train_allowed).all()
    assert not np.isin(batches, np.flatnonzero(old['role']=='DEV_ONLY')).any()
    assert (batches < 11200).all()
    assert all(r['role']=='TRAIN_ONLY' for r in rows[:7500])
    assert all(r['role']=='DEV_ONLY' for r in rows[7500:])
    p11 = load_npz(work / 'mz11-selective-addition-20260910/run-v1/predictions.npz')
    p13 = load_npz(work / 'mz13-training-coverage-20260910/features-v1/features.npz')
    np.testing.assert_array_equal(p11['TRAIN/truth'],oldtruth[np.flatnonzero(old['role']=='TRAIN_ONLY')])
    np.testing.assert_array_equal(p13['truth'],newtruth[:7500])
    np.testing.assert_array_equal(p13['dataset'],[r['dataset'] for r in rows[:7500]])
    assert (p13['dataset'][:5000]=='relation').all() and (p13['dataset'][5000:]=='distance').all()
    a, ref = load_npz(run / 'predictions.npz'), load_npz(rank / 'predictions.npz')
    norm = load_npz(run / 'normalization.npz')
    assert norm['mean'].shape == norm['std'].shape == (1,4,18)
    assert np.isfinite(norm['mean']).all() and (norm['std']>=.1).all()
    torch.set_num_threads(1)
    initial, final = EvidenceArbiter(), EvidenceArbiter()
    initial.load_state_dict(torch.load(run / 'initial.pt',map_location='cpu',weights_only=True))
    final.load_state_dict(torch.load(run / 'arbiter.pt',map_location='cpu',weights_only=True))
    initial.eval(); final.eval()
    for head in initial.heads:
        assert torch.count_nonzero(head[-1].weight)==0 and torch.count_nonzero(head[-1].bias)==0
    assert sum(p.numel() for p in final.parameters()) == result['fit']['parameters']
    cuts = np.load(run / 'cutoff.npy')
    calibration = []
    for q in range(4):
        support, target = a['DEV/support'][:,q], a['DEV/truth'][:,q]
        budget = int(((a['DEV/baseline'][:,q]>=0)&support&~target).sum())
        expected = optimal_cutoff(a['DEV/score'][support,q],target[support],budget)
        assert cuts[q] == expected, (q,cuts[q],expected)
        calibration.append(dict(query=q,budget=budget,cutoff=expected))
    bits=0; max_error=0.; cpu_sign_differences={}; effects={}
    cohort_ids=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),
        stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    alltruth=np.concatenate([oldtruth,newtruth])
    for name,m in result['metrics'].items():
        v={k:a[name+'/'+k] for k in ['features','baseline','truth','support','score','candidate','raw','mz20']}
        normal=name.removesuffix('_wrong')
        np.testing.assert_array_equal(v['truth'],alltruth[cohort_ids[normal]])
        for key in ['baseline','truth','support','raw']:
            if key=='raw':np.testing.assert_allclose(v[key],ref['BODY_RANK/'+name+'/'+key],atol=2e-5,rtol=1e-5)
            else:np.testing.assert_array_equal(v[key],ref['BODY_RANK/'+name+'/'+key])
        np.testing.assert_array_equal(v['mz20'],ref['BODY_RANK/'+name+'/candidate'])
        assert v['features'].shape==(len(v['truth']),4,18) and np.isfinite(v['features']).all()
        for q in range(4):
            np.testing.assert_array_equal(v['features'][:,q,:4],v['baseline'].astype(np.float32))
            np.testing.assert_array_equal(v['features'][:,q,4:8],np.where(v['support'],v['raw'],0))
        np.testing.assert_array_equal(v['features'][~v['support'],15:18],0)
        x=torch.from_numpy((v['features']-norm['mean'])/norm['std'])
        b=torch.from_numpy(v['baseline'].astype(np.float32));s=torch.from_numpy(v['support'])
        with torch.inference_mode():
            assert torch.equal(initial(x,b,s),b)
            replay=final(x,b,s).numpy()
        error=float(np.abs(replay-v['score']).max());max_error=max(max_error,error)
        np.testing.assert_allclose(replay,v['score'],atol=1e-5,rtol=1e-6)
        cpu_candidate=np.where(v['support'],replay.astype(float)-cuts,v['baseline'])
        cpu_sign_differences[name]=int(((cpu_candidate>=0)!=(v['candidate']>=0)).sum())
        z=np.where(v['support'],v['score'].astype(float)-cuts,v['baseline'])
        np.testing.assert_array_equal(z,v['candidate'])
        np.testing.assert_array_equal(z[~v['support']],v['baseline'][~v['support']])
        assert scalar_metrics(z,v['truth'])==m['candidate']
        assert scalar_metrics(v['baseline'],v['truth'])==m['baseline']
        assert scalar_metrics(v['mz20'],v['truth'])==m['mz20']
        p,base,t=z>=0,v['baseline']>=0,v['truth'];change=p!=base
        effect=dict(gained_tp=(change&p&t).sum(0).tolist(),lost_tp=(change&~p&t).sum(0).tolist(),
            added_fp=(change&p&~t).sum(0).tolist(),removed_fp=(change&~p&~t).sum(0).tolist(),
            unsupported_changed=int((change&~v['support']).sum()))
        assert all(m[k]==val for k,val in effect.items());effects[name]=effect;bits+=z.size
        if name in result['thin']:assert scalar_metrics(z[100:125],t[100:125])==result['thin'][name]
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name]
        for field,units in result['groups'][name].items():
            assert set(units)=={r[field] for r in rr}
            for unit,metrics in units.items():
                ii=[i for i,r in enumerate(rr) if r[field]==unit]
                assert metrics==dict(frames=len(ii),baseline=scalar_metrics(a[name+'/baseline'][ii],a[name+'/truth'][ii]),
                    candidate=scalar_metrics(a[name+'/candidate'][ii],a[name+'/truth'][ii]))
    m=result['metrics'];normal=list(cohort_ids)
    gates=dict(no_fp_increase=all(all(m[n]['candidate']['fp'][q]<=m[n]['baseline']['fp'][q] for q in range(4)) for n in normal),
        near_retained=all(all(m[n]['candidate']['tp'][q]>=m[n]['baseline']['tp'][q] for q in [0,2]) for n in normal),
        placement_far_gain=all(sum(m[n]['candidate']['tp'][1::2])>sum(m[n]['baseline']['tp'][1::2]) for n in ['relation10000','distance5000']),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
    gates['useful_effect']=all(gates.values());assert gates==result['gates']
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),scalar_bits=bits,calibration=calibration,
        cutoff_check='Independent descending tied-score cumulative optimization: TP, fewer FP, higher cutoff',
        exported_arbiter_cpu_max_abs_error=max_error,cpu_replay_sign_differences=cpu_sign_differences,
        unsupported_identity='exact',initial_identity='exact on every saved feature cohort',
        batch_draws=int(batches.size),unique_train=int(len(np.unique(batches))),train_calibration_placement_disjoint=True,
        source_truth_and_baseline_alignment='saved MZ11 oldTRAIN and MZ13 newTRAIN rows checked; every MZ20 cohort checked',
        groups='family/group/site exact',effects=effects,gates=gates,
        limits=['TRAIN raw arbiter features were not saved; TRAIN-only normalization verified by code and ID routing, not numeric recomputation.',
                'CPU replay starts from saved observable features; it is not a new frozen RGB/model feature extraction or total runtime benchmark.',
                'MZ20 training predictions are in-sample stacking inputs; sequence200 was in training; all placements are consumed Development.'])
    write(run/'audit.json',audit)
    print('PASS',bits,'bits','CPU max error',max_error,'sign differences',cpu_sign_differences,'gates',gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.run)
