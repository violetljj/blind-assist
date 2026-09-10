"""Stored-array audit; no model fit or extra cutoff selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_audit import scalar_metrics


def main(root,features,run):
    receipt=read(run/'receipt.json');start=read(run/'start-receipt.json');result=read(run/'result.json')
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest
    assert sha(Path(__file__).with_name('mz13_coverage_fit.py'))==start['source_sha256']
    assert sha(Path(__file__).with_name('MZ13_TRAINING_COVERAGE_PROTOCOL_20260910.md'))==start['protocol_sha256']
    for name,digest in start['inputs'].items():assert sha(name)==digest
    selected=read(features/'selected.json')
    evaluation=read(root/'artifacts.local/work/mz12-existing-data-20260910/inventory-v1/selected.json')
    assert len(selected)==7500 and all(r['role']=='TRAIN_ONLY' for r in selected)
    assert not {r['site'] for r in selected}&{r['site'] for r in evaluation}
    assert not {r['rgb_sha'] for r in selected}&{r['rgb_sha'] for r in evaluation}
    assert start['steps']==600 and start['parameters']==20
    arrays=load_npz(run/'predictions.npz');cutoff=np.load(run/'threshold.npy',allow_pickle=False)
    checkpoint=torch.load(run/'gate.pt',map_location='cpu',weights_only=True)
    weight=checkpoint['weight'].numpy();bias=checkpoint['bias'].numpy()
    cohorts=list(dict.fromkeys(k.split('/')[0] for k in arrays));bits=0;maxerr=0
    for cohort in cohorts:
        a={k.split('/',1)[1]:v for k,v in arrays.items() if k.startswith(cohort+'/')}
        b,s,t=a['baseline'],a['source'],a['truth'];new=a['candidate'];available=a['available']
        count=0
        for i in range(len(t)):
            for q in range(4):
                eligible=b[i,q]<0 and s[i,q]>=0 and available[i,q]
                accepted=eligible and a['score'][i,q]>=cutoff[q]
                assert eligible==a['eligible'][i,q] and accepted==a['accepted'][i,q]
                assert new[i,q]==(s[i,q] if accepted else b[i,q])
                if b[i,q]>=0:assert new[i,q]==b[i,q]
                replay=sum(float(a['features'][i,q,j])*float(weight[q,j]) for j in range(4))+float(bias[q])
                maxerr=max(maxerr,abs(replay-float(a['score'][i,q])));bits+=1
                count+=int(accepted)
        m=result['metrics'][cohort]
        assert scalar_metrics(new,t)==m['candidate'];assert scalar_metrics(b,t)==m['baseline']
        assert (a['accepted']&t).sum(0).tolist()==m['added_tp']
        assert (a['accepted']&~t).sum(0).tolist()==m['added_fp']
        if 'previous' in a:
            assert scalar_metrics(a['previous'],t)==m['previous']
            c,p=new>=0,a['previous']>=0
            expected=dict(tp_gained=(c&~p&t).sum(0).tolist(),tp_lost=(~c&p&t).sum(0).tolist(),
                fp_added=(c&~p&~t).sum(0).tolist(),fp_removed=(~c&p&~t).sum(0).tolist())
            assert expected==m['vs_previous']
    for q in range(4):
        eligible=arrays['DEV/eligible'][:,q];negative=eligible&~arrays['DEV/truth'][:,q]
        expected=np.nextafter(arrays['DEV/score'][negative,q].max(),np.float32(np.inf)) if negative.any() else (-np.inf if eligible.any() else np.inf)
        assert cutoff[q]==expected
    assert maxerr<1e-4
    m=result['metrics'];sign_fp=0
    for cohort in ['relation10000','distance5000']:
        rows=[r for r in evaluation if r['dataset']==cohort]
        for field in ['site','group']:
            units={}
            for i,row in enumerate(rows):units.setdefault(row[field],[]).append(i)
            expected=dict(total=len(units))
            for arm in ['candidate','previous']:
                expected[arm]=sum(all(bool(arrays[cohort+'/'+arm][i,q]>=0)==bool(arrays[cohort+'/truth'][i,q])
                    for i in ids for q in range(4)) for ids in units.values())
            assert expected==result['groups'][cohort][field]
        for family in sorted({r['family'] for r in rows}):
            mask=np.array([r['family']==family for r in rows]);t=arrays[cohort+'/truth'][mask]
            f=result['families'][cohort][family]
            assert scalar_metrics(arrays[cohort+'/candidate'][mask],t)==f['candidate']
            added=(arrays[cohort+'/accepted'][mask]&~t).sum(0).tolist();assert added==f['added_fp']
            if family=='hanging_sign':sign_fp+=sum(added)
    assert sign_fp==result['sign_added_fp']
    clip=load_npz(root/'artifacts.local/work/mz11-selective-addition-20260910/run-v1/predictions.npz')['clip']
    mask=clip=='thin_pole_background_target'
    for cohort in ['clean','stress','clean_wrong','stress_wrong']:
        assert scalar_metrics(arrays[cohort+'/candidate'][mask],arrays[cohort+'/truth'][mask])==result['thin'][cohort]['candidate']
    gate=dict(sign_fp_reduced=sign_fp<17,
        transfer_fp_not_increased=all(all(x<=y for x,y in zip(m[k]['candidate']['fp'],m[k]['previous']['fp'])) for k in ['relation10000','distance5000']),
        transfer_far_not_lost=all(sum(m[k]['candidate']['tp'][1::2])>=sum(m[k]['previous']['tp'][1::2]) for k in ['relation10000','distance5000']),
        old_dev_near_not_lost=sum(m['DEV']['added_tp'][::2])>=15,old_dev_far_not_lost=sum(m['DEV']['added_tp'][1::2])>=17,
        thin_clean_not_lost=sum(result['thin']['clean']['candidate']['tp'][1::2])>=46,
        thin_stress_not_lost=sum(result['thin']['stress']['candidate']['tp'][1::2])>=44,
        baseline_positives_preserved=all(v['baseline_positive_lost']==0 for v in m.values()))
    assert result['gate']==gate
    assert result['favorable_challenger']==all(result['gate'].values())
    write(run/'audit.json',dict(status='PASS',scalar_bits=bits,gate_score_max_error=maxerr,
        training_rows=7500,training_eval_site_overlap=0,training_eval_rgb_overlap=0,
        sign_added_fp=sign_fp,gate=result['gate'],favorable_challenger=result['favorable_challenger'],
        source_sha256=sha(Path(__file__))))
    print('AUDIT PASS',bits,'signFP',sign_fp,'favorable',result['favorable_challenger'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','features','run']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.root,a.features,a.run)
