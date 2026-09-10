"""Saved-array baseline support/source partition; no learned inference or fit."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz24_audit import geometry
from mz15_evaluator import align_stress_labels


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    rank=work/'mz20-rank-objective-20260910/run-v1';run26=work/'mz26-deterministic-convergence-20260910/run-v1';run28=work/'mz28-packet-availability-20260910/run-v1'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1';lp=work/'mz9-source-supervision-20260910/labels-v1'
    for folder,names in [(rank,['predictions.npz','BODY_RANK.pt','BODY_RANK-cutoff.npy']),
                         (run26,['step4800-predictions.npz']),(run28,['predictions.npz'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['outputs'][name])
    for folder,names in [(oldpath,['observations.npz']),(newpath,['observations.npz','evaluator.npz','selected.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['files'][name])
    bind(lp/'receipt.json');bind(lp/'evaluator.npz',read(lp/'receipt.json')['labels_sha256'])
    a=load_npz(rank/'predictions.npz');b=load_npz(run26/'step4800-predictions.npz');c=load_npz(run28/'predictions.npz')
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz')
    ol,nl=load_npz(lp/'evaluator.npz'),load_npz(newpath/'evaluator.npz')
    source=np.concatenate([ol['source_counts']>0,nl['source_presence']]);query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']]);rows=read(newpath/'selected.json')
    rays=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True)['rays'];torch.set_num_threads(1);cut=np.load(rank/'BODY_RANK-cutoff.npy')
    cohorts=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    code={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'mz24_audit.py','mz15_evaluator.py']}
    write(output/'start.json',dict(status='STARTED',training_steps=0,new_inference_frames=0,inputs=inputs,code_sha256=code,
        protocol_sha256=sha(Path(__file__).with_name('MZ29_BASELINE_SUPPORT_PROTOCOL_20260910.md'))))
    records=[];summaries={};arrays={}
    for name,ids in cohorts.items():
        prefix='BODY_RANK/'+name+'/';base=a[prefix+'baseline'];truth=a[prefix+'truth'];geo=a[prefix+'support']
        for d in [b,c]:
            np.testing.assert_array_equal(base,d[name+'/baseline']);np.testing.assert_array_equal(truth,d[name+'/truth']);np.testing.assert_array_equal(geo,d[name+'/original_support'])
        rr,vv=(old['stress_ranges'],old['stress_valid']) if name=='stress' else (ranges[ids],valid[ids]);ss,qq=source[ids],query[ids]
        if name=='stress':ss,qq,moved=align_stress_labels(ss,qq,ranges[ids],valid[ids],rr,vv)
        actual=qq.sum((1,2,3));eligible=np.zeros_like(actual);zone_valid=(vv&np.isfinite(rr)&(rr>0)&(rr<=4)).any(2).sum(1)
        for begin in range(0,len(ids),64):
            end=min(begin+64,len(ids));e=geometry(rr[begin:end],vv[begin:end],rays)
            np.testing.assert_array_equal(e.any((1,2,3)),geo[begin:end]);eligible[begin:end]=(e&qq[begin:end]).sum((1,2,3))
        flags=dict(geometry=geo,learned=b[name+'/support'],packet=c[name+'/support'],actual_query=actual>0,eligible_actual_query=eligible>0,
            mz20_local=(a[prefix+'raw'].astype(float)>=cut)&geo,
            mz26_local=(b[name+'/raw'].astype(float)>=cut)&b[name+'/support'],mz28_local=(c[name+'/raw'].astype(float)>=cut)&c[name+'/support'])
        summaries[name]={}
        for label,title in [(False,'baseline_false'),(True,'baseline_true')]:
            take=(base>=0)&(truth==label);parts={}
            for flag,mask in flags.items():parts[flag]=dict(yes=(take&mask).sum(0).tolist(),no=(take&~mask).sum(0).tolist())
            parts['total']=take.sum(0).tolist();parts['no_geometry_with_actual_query']=(take&~geo&(actual>0)).sum(0).tolist();parts['no_geometry_no_actual_query']=(take&~geo&(actual==0)).sum(0).tolist()
            summaries[name][title]=parts
        for i,q in np.argwhere(base>=0):
            gid=int(ids[i]);identity={}
            if gid>=3700:
                row=rows[gid-3700];identity={k:row[k] for k in ['site','group','family','dataset']}
            records.append(dict(cohort=name,frame=int(i),global_id=gid,query=int(q),truth=bool(truth[i,q]),baseline_margin=float(base[i,q]),
                mz20_margin=float(a[prefix+'raw'][i,q]-cut[q]),mz26_margin=float(b[name+'/raw'][i,q]-cut[q]),mz28_margin=float(c[name+'/raw'][i,q]-cut[q]),
                actual_query_contributors=int(actual[i,q]),eligible_actual_contributors=int(eligible[i,q]),valid_zones=int(zone_valid[i]),
                **{key:bool(value[i,q]) for key,value in flags.items()},identity=identity))
        for key,value in dict(global_ids=ids,baseline=base,truth=truth,actual_query_contributors=actual,eligible_actual_contributors=eligible,valid_zones=zone_valid,**flags).items():arrays[name+'/'+key]=value
        print(name,'FP',summaries[name]['baseline_false']['total'],'FPgeom',summaries[name]['baseline_false']['geometry'],'TPmissinggeom',summaries[name]['baseline_true']['geometry']['no'],flush=True)
    assert sum(sum(summaries[n]['baseline_false']['total']) for n in ['relation10000','distance5000'])==81
    np.savez_compressed(output/'partitions.npz',**arrays)
    write(output/'result.json',dict(status='PASS',training_steps=0,new_inference_frames=0,summaries=summaries,baseline_positive_rows=len(records),records=records,
        limits=['Native source masks and task truth are evaluator-only partitions, not inference signals.','Missing geometric/local support is not CLEAR.','Repeated consumed Development; no candidate or cutoff changed.']))
    write(output/'receipt.json',dict(status='PASS',training_steps=0,new_inference_frames=0,backend='CPU saved arrays and analytic geometry',reason='TASK_NOT_GPU_SUITABLE',seconds=time.perf_counter()-started,inputs=inputs,code_sha256=code,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('PASS',len(records),'baseline positive rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
