"""Independent AP implementation and scalar replay of the witness-objective contrast."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_audit import scalar_metrics


def main(root,run):
    receipt,start,result=(read(run/n) for n in ['receipt.json','start.json','result.json'])
    for n,h in receipt['outputs'].items():assert sha(run/n)==h
    for n,h in start['code_sha256'].items():assert sha(Path(__file__).with_name(n))==h
    assert sha(Path(__file__).with_name('MZ20_RANK_OBJECTIVE_PROTOCOL_20260910.md'))==start['protocol_sha256']
    prior=root/'artifacts.local/work/mz15-shared-support-20260910/run-v1'
    assert sha(run/'batches.npy')==sha(prior/'batches.npy')==start['train_batch_sha256']
    a=load_npz(run/'predictions.npz');local=load_npz(run/'local_samples.npz');bits=0;local_cells=0
    init=[torch.load(run/'BODY_RANK-initial.pt',weights_only=True),torch.load(root/'artifacts.local/work/mz16-visual-detail-20260910/run-v1/HIGH_DETAIL-initial.pt',weights_only=True)]
    for key in init[0]:torch.testing.assert_close(init[0][key],init[1][key],rtol=0,atol=0)
    rows=read(root/'artifacts.local/work/mz15-shared-support-20260910/cache-v1/selected.json')
    for arm in ['BODY_RANK']:
        cut=np.load(run/(arm+'-cutoff.npy'));lc=np.load(run/(arm+'-local-cutoff.npy'))
        for q in range(4):
            raw,base,truth,sup=(a[arm+'/DEV/'+k] for k in ['raw','baseline','truth','support'])
            elig=(base[:,q]<0)&sup[:,q];neg=elig&~truth[:,q]
            expected=np.nextafter(np.float64(raw[neg,q].max()),np.inf) if neg.any() else (-np.inf if elig.any() else np.inf)
            assert cut[q]==expected
            s,y=(local[arm+f'/DEV/{q}/'+k] for k in ['score','label'])
            negative=np.sort(s[~y]);budget=len(negative)//100
            expected=np.nextafter(np.float64(negative[-budget-1]),np.inf) if len(negative) else -np.inf
            assert lc[q]==expected
        for name,m in result['metrics'][arm].items():
            v={k:a[arm+'/'+name+'/'+k] for k in ['raw','support','margin','baseline','truth','candidate','accepted']}
            for i in range(len(v['truth'])):
                for q in range(4):
                    margin=float(v['raw'][i,q])-cut[q] if v['support'][i,q] else -1e6
                    accept=v['baseline'][i,q]<0 and v['support'][i,q] and margin>=0
                    assert margin==v['margin'][i,q] and accept==v['accepted'][i,q]
                    assert v['candidate'][i,q]==(margin if accept else v['baseline'][i,q]);bits+=1
            assert scalar_metrics(v['baseline'],v['truth'])==m['baseline']
            assert scalar_metrics(v['candidate'],v['truth'])==m['candidate']
            assert (v['accepted']&v['truth']).sum(0).tolist()==m['added_tp']
            assert (v['accepted']&~v['truth']).sum(0).tolist()==m['added_fp']
            if name in result['thin'][arm]:assert scalar_metrics(v['candidate'][100:125],v['truth'][100:125])==result['thin'][arm][name]
        for name,m in result['local'][arm].items():
            aps=[]
            for q,row in enumerate(m['per_query']):
                s,y,f=(local[arm+'/'+name+f'/{q}/'+k] for k in ['score','label','frame'])
                local_cells+=len(s);ap=float(average_precision_score(y,s)) if y.any() else None
                if ap is not None:
                    np.testing.assert_allclose(ap,row['ap'],atol=1e-12,rtol=0);aps.append(ap)
                else:assert row['ap'] is None
                p=s.astype(np.float64)>=lc[q]
                counts=a[arm+'/'+name+'/local_counts'][:,q]
                for j,mask in enumerate([p&y,p&~y,~p&y,y,~y]):
                    np.testing.assert_array_equal(np.bincount(f[mask],minlength=len(counts)),counts[:,j])
                assert counts.sum(0).tolist()==[row[k] for k in ['tp','fp','fn','positive','negative']]
                if name=='DEV':assert row['fp']<=row['negative']//100
            np.testing.assert_allclose(np.mean(aps),m['macro_ap'],atol=1e-12,rtol=0)
        for name in ['relation10000','distance5000']:
            rr=[r for r in rows if r['dataset']==name]
            for field,units in result['groups'][arm][name].items():
                for unit,m in units.items():
                    ii=[i for i,r in enumerate(rr) if r[field]==unit]
                    assert m==dict(frames=len(ii),baseline=scalar_metrics(a[arm+'/'+name+'/baseline'][ii],a[arm+'/'+name+'/truth'][ii]),candidate=scalar_metrics(a[arm+'/'+name+'/candidate'][ii],a[arm+'/'+name+'/truth'][ii]))
        m,thin=result['metrics'][arm],result['thin'][arm];names=list(result['local'][arm])
        g=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in names),placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),thin_clean=sum(thin['clean']['tp'][1::2])>=48,thin_stress=sum(thin['stress']['tp'][1::2])>=48,near_retained=all(all(m[n]['candidate']['tp'][q]>=m[n]['baseline']['tp'][q] for q in [0,2]) for n in names))
        g['useful_effect']=all(g.values());assert g==result['gates'][arm]
    reference=read(root/'artifacts.local/work/mz16-visual-detail-20260910/run-v1/result.json')
    for k in ['metrics','local','thin','gates']:assert result['comparator'][k]==reference[k]['HIGH_DETAIL']
    local_gain=all(result['local']['BODY_RANK'][n]['macro_ap']>reference['local']['HIGH_DETAIL'][n]['macro_ap'] for n in ['relation10000','distance5000'])
    assert local_gain==result['witness_local_ap_gain']
    write(run/'audit.json',dict(status='PASS',scalar_bits=bits,local_candidate_bits=local_cells,local_ap_check='Independent sklearn average_precision_score with tied-score comparison',batch_and_initialization='identical',calibration='oldDEV only; local FPR<=1%',groups='family/group/site exact',gates=result['gates'],local_gain=local_gain))
    print('PASS',bits,'output bits',local_cells,'local candidate bits',result['gates'],local_gain)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run)
