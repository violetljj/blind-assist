"""Independent NumPy geometry and exported-row recount of MZ29; no inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from contact_retina_spec import BODY_BOXES
from mz5_ensemble_readout import read,write,sha,load_npz
from mz15_evaluator import align_stress_labels


def numpy_geometry(ranges,valid,rays):
    r=ranges.astype(np.float32);good=valid&np.isfinite(r)&(r>0)&(r<=4)
    xyz=np.where(good,r,np.float32(0))[...,None,None]*rays[None,:,None]+np.array([0,0,1.7],np.float32)
    masks=[]
    for lower,upper in BODY_BOXES:
        for half in [0,1]:
            start=np.float32(upper[0]+1.5*half);end=np.float32(upper[0]+1.5*(half+1))
            depth=(xyz[...,0]>=start)&((xyz[...,0]<end) if half==0 else (xyz[...,0]<=end))
            cross=(xyz[...,1]>=np.float32(lower[1]))&(xyz[...,1]<=np.float32(upper[1]))&(xyz[...,2]>=np.float32(lower[2]))&(xyz[...,2]<=np.float32(upper[2]))
            masks.append(depth&cross&good[...,None])
    return np.stack(masks,-1)


def main(root,run):
    assert not (run/'audit.json').exists();receipt,start,result=(read(run/n) for n in ['receipt.json','start.json','result.json'])
    assert receipt['status']==result['status']=='PASS' and result['training_steps']==result['new_inference_frames']==0
    for n,h in receipt['outputs'].items():assert sha(run/n)==h,n
    for p,h in start['inputs'].items():assert sha(p)==h,p
    for n,h in start['code_sha256'].items():assert sha(Path(__file__).with_name(n))==h,n
    assert start['protocol_sha256']==sha(Path(__file__).with_name('MZ29_BASELINE_SUPPORT_PROTOCOL_20260910.md'))
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz');metadata=read(newpath/'selected.json')
    ol=load_npz(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz');nl=load_npz(newpath/'evaluator.npz')
    source=np.concatenate([ol['source_counts']>0,nl['source_presence']]);query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol
    oldtruthpath=oldpath/'evaluator.npz';assert sha(oldtruthpath)==read(oldpath/'receipt.json')['files']['evaluator.npz']
    truth=np.concatenate([load_npz(oldtruthpath)['truth'],nl['truth']]);del nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    pred=load_npz(rank/'predictions.npz');p26=load_npz(work/'mz26-deterministic-convergence-20260910/run-v1/step4800-predictions.npz')
    p28=load_npz(work/'mz28-packet-availability-20260910/run-v1/predictions.npz');part=load_npz(run/'partitions.npz')
    rays=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True)['rays'].numpy();cut=np.load(rank/'BODY_RANK-cutoff.npy')
    cohorts=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    records=result['records'];keys=[(r['cohort'],r['frame'],r['query']) for r in records];assert len(keys)==len(set(keys))==3355
    recordlookup=dict(zip(keys,records));checked=0;geometry_bits=0;flags=['geometry','learned','packet','actual_query','eligible_actual_query','mz20_local','mz26_local','mz28_local']
    recounted={};coverage={}
    for name,ids in cohorts.items():
        prefix='BODY_RANK/'+name+'/';b=pred[prefix+'baseline'];target=pred[prefix+'truth'];np.testing.assert_array_equal(target,truth[ids])
        for p in [p26,p28]:
            np.testing.assert_array_equal(p[name+'/baseline'],b);np.testing.assert_array_equal(p[name+'/truth'],target)
            np.testing.assert_array_equal(p[name+'/original_support'],pred[prefix+'support'])
        rr,vv=(old['stress_ranges'],old['stress_valid']) if name=='stress' else (ranges[ids],valid[ids]);ss,qq=source[ids],query[ids]
        if name=='stress':ss,qq,_=align_stress_labels(ss,qq,ranges[ids],valid[ids],rr,vv)
        actual=qq.sum((1,2,3));eligible=np.zeros_like(actual);geo=np.zeros_like(target)
        for startrow in range(0,len(ids),64):
            end=min(startrow+64,len(ids));g=numpy_geometry(rr[startrow:end],vv[startrow:end],rays);geometry_bits+=g.size
            geo[startrow:end]=g.any((1,2,3));eligible[startrow:end]=(g&qq[startrow:end]).sum((1,2,3))
        np.testing.assert_array_equal(geo,pred[prefix+'support'])
        valid_zones=(vv&np.isfinite(rr)&(rr>0)&(rr<=4)).any(2).sum(1)
        expected=dict(global_ids=ids,baseline=b,truth=target,actual_query_contributors=actual,eligible_actual_contributors=eligible,valid_zones=valid_zones,
            geometry=geo,learned=p26[name+'/support'],packet=p28[name+'/support'],actual_query=actual>0,eligible_actual_query=eligible>0,
            mz20_local=(pred[prefix+'raw'].astype(float)>=cut)&geo,mz26_local=(p26[name+'/raw'].astype(float)>=cut)&p26[name+'/support'],
            mz28_local=(p28[name+'/raw'].astype(float)>=cut)&p28[name+'/support'])
        for k,v in expected.items():np.testing.assert_array_equal(part[name+'/'+k],v)
        for frame,q in np.argwhere(b>=0):
            r=recordlookup[(name,int(frame),int(q))];checked+=1;gid=int(ids[frame]);assert r['global_id']==gid and r['truth']==bool(target[frame,q])
            assert r['baseline_margin']==float(b[frame,q])
            for k,p in [('mz20',pred),('mz26',p26),('mz28',p28)]:
                raw=p[prefix+'raw'] if k=='mz20' else p[name+'/raw'];assert r[k+'_margin']==float(raw[frame,q]-cut[q])
            assert r['actual_query_contributors']==int(actual[frame,q]) and r['eligible_actual_contributors']==int(eligible[frame,q]) and r['valid_zones']==int(valid_zones[frame])
            for k in flags:assert r[k]==bool(expected[k][frame,q])
            identity={k:metadata[gid-3700][k] for k in ['site','group','family','dataset']} if gid>=3700 else {}
            assert r['identity']==identity
        local=[r for r in records if r['cohort']==name];recounted[name]={}
        for positive,label in [(False,'baseline_false'),(True,'baseline_true')]:
            rows=[r for r in local if r['truth']==positive];s={}
            s['total']=[sum(r['query']==q for r in rows) for q in range(4)]
            for flag in flags:s[flag]={yn:[sum(r['query']==q and r[flag]==value for r in rows) for q in range(4)] for yn,value in [('yes',True),('no',False)]}
            s['no_geometry_with_actual_query']=[sum(r['query']==q and not r['geometry'] and r['actual_query'] for r in rows) for q in range(4)]
            s['no_geometry_no_actual_query']=[sum(r['query']==q and not r['geometry'] and not r['actual_query'] for r in rows) for q in range(4)]
            assert s==result['summaries'][name][label];recounted[name][label]=s
        coverage[name]=dict(frames=len(ids),baseline_positive_bits=len(local))
    assert checked==3355 and set(result['summaries'])==set(cohorts)
    placement=[r for r in records if r['cohort'] in ['relation10000','distance5000'] and not r['truth']]
    assert len(placement)==81 and sum(not r['geometry'] for r in placement)==80
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),receipt_sha256=sha(run/'receipt.json'),inputs=start['inputs'],
        additional_inputs={str(oldtruthpath):sha(oldtruthpath)},dependencies={name:sha(Path(__file__).with_name(name)) for name in ['mz15_evaluator.py','contact_retina_spec.py','mz5_ensemble_readout.py']},
        training_steps=0,new_inference_frames=0,baseline_positive_rows=checked,geometry_candidate_bits=geometry_bits,coverage=coverage,summaries=recounted,
        supported_placement_false_rows=[r for r in placement if r['geometry']],
        scope='Independent NumPy float32 geometry and exact saved-row/partition recount; no candidate changes. Actual query contributors count selected-return subcell incidences, not all native pixels.')
    write(run/'audit.json',audit);print('PASS',checked,'rows',geometry_bits,'geometry bits',coverage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run)
