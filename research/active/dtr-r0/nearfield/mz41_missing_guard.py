"""MZ41: protect existing RGB-positive alerts only when query zones lack returns.

Frozen EXPLORE rule: project each existing body/head near/far box to an angular
bounding rectangle. A query lacks ToF observation when none of the intersecting
8x8 zones has a finite valid return in (0,4]m. Undo MZ35 removals of RGB-positive,
ToF-negative MZ28 alerts only there, preserving MZ37 elsewhere. Broad protection
of every such removal is the control. No threshold search or learned parameters.
This mask describes availability, not sensor visibility or obstacle clearance.
"""
import argparse
from itertools import product
from pathlib import Path
import time
import numpy as np
from contact_retina_spec import BODY_BOXES
from mz5_ensemble_readout import read, write, sha, load_npz
from mz40_evaluate import metrics, paired


def zone_mask():
    masks=[]
    for low,high in BODY_BOXES:
        for start in (high[0],high[0]+1.5):
            xyz=np.array(list(product((start,start+1.5),(low[1],high[1]),(low[2]-1.7,high[2]-1.7))))
            az=np.degrees(np.arctan2(xyz[:,1],xyz[:,0]));el=np.degrees(np.arctan2(xyz[:,2],xyz[:,0]))
            row,col=np.divmod(np.arange(64),8)
            left=-22.5+col*45/8;right=left+45/8;top=22.5-row*45/8;bottom=top-45/8
            masks.append((left<az.max())&(right>az.min())&(bottom<el.max())&(top>el.min()))
    return np.array(masks).T


def compose(ranges,valid,rgb,tof,mz28,mz37):
    assert ranges.shape==valid.shape and ranges.shape[1:]==(64,2)
    mask=zone_mask();good=valid&np.isfinite(ranges)&(ranges>0)&(ranges<=4)
    counts=good.any(2).astype(np.int64)@mask.astype(np.int64)
    removable=(mz28>=0)&(mz37<0)&(rgb>=0)&(tof<0)
    protected=removable&(counts==0)
    guard=np.where(protected,mz28,mz37);broad=np.where(removable,mz28,mz37)
    np.testing.assert_array_equal(guard[mz37>=0],mz37[mz37>=0])
    np.testing.assert_array_equal(guard[~protected],mz37[~protected])
    assert not ((guard<0)&(mz37>=0)).any()
    return dict(guard=guard,broad=broad,protected=protected,removable=removable,valid_zone_counts=counts)


def run(root,output):
    root=root.resolve();work=root/'artifacts.local/work';output=output.resolve()
    assert output.is_relative_to((root/'artifacts.local').resolve()) and not output.exists()
    inputs={};started=time.perf_counter()
    def bind(p,h=None):
        digest=sha(p);assert h is None or digest==h,str(p);inputs[str(p)]=digest;return p
    def receipt(folder):
        r=read(bind(folder/'receipt.json'));assert r['status']=='PASS';return r
    source=work/'mz40-l8cx-constrained-20260910'
    bind(Path(__file__));bind(Path(__file__).with_name('contact_retina_spec.py'));bind(Path(__file__).with_name('mz40_evaluate.py'))
    assert read(bind(source/'audit.json'))['status']=='PASS'
    sr=receipt(source/'score-v1');scored=load_npz(bind(source/'score-v1/scored.npz',sr['outputs']['scored.npz']))
    ids,truth,known=(scored[k] for k in ('frame_ids','truth','known'));take=np.flatnonzero(known.all(1))
    assert len(ids)==400 and len(take)==380 and (~known).sum()==80
    mask=zone_mask();assert mask.sum(0).tolist()==[24,16,64,12]
    result=dict(status='PASS',scope='Consumed controlled MZ36 Development; ideal and MZ40 proxies; no device/new-source conclusion',
        query_zone_counts=mask.sum(0).tolist(),attempted_frames=400,admitted_frames=380,unknown_by_query=(~known).sum(0).tolist(),arms={})
    saved=dict(frame_ids=ids,truth=truth,known=known,query_zone_mask=mask);audit_counts=0
    for arm in ('IDEAL','MERGE_CLOSE','DROP_CLOSE'):
        folder=work/'mz36-new-source-20260910/inference-v1' if arm=='IDEAL' else source/arm
        rr=receipt(folder);raw=load_npz(bind(folder/'predictions.npz',rr['outputs']['predictions.npz']))
        np.testing.assert_array_equal(raw['frame_ids'],ids[take])
        old=scored[arm+'/MZ37'][take]
        for key in ('rgb','tof','MZ28'):np.testing.assert_array_equal(raw[key],scored[arm+'/'+key][take])
        out=compose(raw['ranges'],raw['valid'],raw['rgb'],raw['tof'],raw['MZ28'],old)
        # Independent scalar replay of actual guard actions and outcome counts.
        expected=np.array(old,copy=True);broad_expected=np.array(old,copy=True)
        for i in range(380):
            for q in range(4):
                no_return=not any(bool(raw['valid'][i,z,k]) and np.isfinite(raw['ranges'][i,z,k])
                    and 0<float(raw['ranges'][i,z,k])<=4 for z in range(64) if mask[z,q] for k in range(2))
                removed=float(raw['MZ28'][i,q])>=0 and float(old[i,q])<0 and float(raw['rgb'][i,q])>=0 and float(raw['tof'][i,q])<0
                if removed:broad_expected[i,q]=raw['MZ28'][i,q]
                if removed and no_return:expected[i,q]=raw['MZ28'][i,q]
                audit_counts+=1
        np.testing.assert_array_equal(expected,out['guard']);np.testing.assert_array_equal(broad_expected,out['broad'])
        methods={};pairs={}
        for name in ('guard','broad'):
            full=np.full((400,4),np.nan);full[take]=out[name];saved[arm+'/'+name]=full
            methods[name]=metrics(full,truth,known);pairs[name]=paired(full,scored[arm+'/MZ37'],truth,known)
            scalar={key:[0]*4 for key in ('tp','fp','fn','tn')};exact=0
            for i in take:
                exact+=all((float(full[i,q])>=0)==bool(truth[i,q]) for q in range(4))
                for q in range(4):
                    pred=float(full[i,q])>=0;target=bool(truth[i,q]);key=('tp' if target else 'fp') if pred else ('fn' if target else 'tn')
                    scalar[key][q]+=1
            assert methods[name]['exact_frames']==exact
            assert all(methods[name][key]==value for key,value in scalar.items())
        for key in ('protected','removable','valid_zone_counts'):saved[arm+'/'+key]=out[key]
        comparisons={name:metrics(scored[arm+'/'+name],truth,known) for name in ('rgb','tof','MZ5','MZ28','MZ37')}
        result['arms'][arm]=dict(methods=methods,baselines=comparisons,paired_vs_mz37=pairs,
            unobserved_query_bits=(out['valid_zone_counts']==0).sum(0).tolist(),protected_bits=out['protected'].sum(0).tolist(),
            changed_frames=int(out['protected'].any(1).sum()))
    # Diagnostic criteria declared by the run registration; no retrospective tuning.
    result['criteria']=dict(recovers_dropout_true_bits=sum(result['arms']['DROP_CLOSE']['paired_vs_mz37']['guard']['tp_gained'])>0,
        adds_no_fp=all(sum(a['paired_vs_mz37']['guard']['fp_added'])==0 for a in result['arms'].values()),
        retains_prior_positives=True,scalar_audit_query_actions=audit_counts)
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',**saved);write(output/'result.json',result)
    for p,h in inputs.items():assert sha(p)==h,p
    write(output/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in output.iterdir()},
        seconds=time.perf_counter()-started,backend='CPU saved-scalar replay; TASK_NOT_GPU_SUITABLE',
        training_steps=0,model_inference_frames=0,checks='4560 scalar query actions, independent confusion/exact counts, input identity and unchanged positive scores'))
    for arm,a in result['arms'].items():print(arm,{n:dict(fp=sum(m['fp']),fn=sum(m['fn']),exact=m['exact_frames']) for n,m in {**a['baselines'],**a['methods']}.items()},'protected',a['protected_bits'])
    print(result['criteria'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
