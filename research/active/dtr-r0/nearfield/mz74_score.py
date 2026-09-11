"""Independent MZ74 saved-output score; no model loading, fitting or cut search.

CLI --root ROOT --task TASK; --self-test opens only synthetic arrays.
Real scoring requires score-go-v1.json status ROOT_SCORE_GO, run_exit_code0,
run_receipt_sha256 and scorer_sha256. Output score-v1 must not exist.
Runner receipt: status PASS, training_steps0/new_cutoffs0, inputs/outputs hashes,
frozen.cutoffs references for OLD_NEG/OPEN,GATED and MZ70/CONTROL,DIVERSE,
frozen.checkpoints references for CONTROL,DIVERSE (additional refs allowed).
All references are {path,sha256}; relative paths resolve from run-v1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from mz58_score import metrics, scalar_metrics

TASK='mz74-return-survival-20260911'
SOURCES=('mz61','mz67')
OLD=('IDEAL','MERGE_CLOSE','DROP_CLOSE')
NEW=('CLOSEST_REPORTED_PROXY','FARTHEST_REPORTED_PROXY')
ARMS=('CONTROL','DIVERSE')
EVENTS=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
BASE='OLD_NEG/UNION'
METHODS=('MZ37','OLD_NEG/OPEN/candidate','OLD_NEG/GATED/candidate',BASE)+tuple(
    'MZ70/'+a+'/'+m for a in ARMS for m in ('candidate','UNION'))
PRIOR_METRIC_METHODS=tuple(m for m in METHODS if not m.startswith(('OLD_NEG/OPEN/','OLD_NEG/GATED/')))
RUN70_SHA='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
SCORE70_SHA='2fe35ce61155ba80c7197400d4331f60410159cc550270dca623153e2f23f84a'
OLD_CUT_SHA={'OPEN':'52c6e987069f2c12b71fa55cdcc32c074ce9a1565824d9a47802373d651fc8cc',
             'GATED':'939bfe5557ab09a9dc4907b7cfc308014caede0ef60351bdb200c23df7f41472'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    assert a.tobytes()==b.tobytes(), 'Byte identity mismatch'

def proxy(original,flags,mode):
    """Independent scalar unresolved-pair projection; no producer import."""
    assert mode in NEW and original.dtype==np.float32 and flags.dtype==bool
    assert original.shape==flags.shape and original.shape[1:]==(64,2)
    assert np.isfinite(original).all() and (original[~flags]==0).all()
    assert ((original[flags]>0)&(original[flags]<=4)).all()
    rr=original.copy();vv=flags.copy();changed=np.zeros(original.shape[:2],bool)
    for i in range(len(rr)):
        for z in range(64):
            if flags[i,z].all():
                lo,hi=float(original[i,z,0]),float(original[i,z,1]);assert hi>=lo
                if hi-lo<.6:
                    changed[i,z]=True
                    dropped=1 if mode==NEW[0] else 0
                    rr[i,z,dropped]=0.;vv[i,z,dropped]=False
    assert not (vv & ~flags).any()
    exact(rr[~changed],original[~changed]);exact(vv[~changed],flags[~changed])
    return rr,vv,changed

def candidate(raw,support,base,cut):
    assert raw.shape==support.shape==base.shape and raw.shape[1]==4
    assert raw.dtype==np.float32 and support.dtype==bool and cut.shape==(4,)
    assert np.isfinite(raw).all() and np.isfinite(base).all() and not np.isnan(cut).any()
    out=np.empty(base.shape,np.float64)
    for i in range(len(raw)):
        for q in range(4):
            margin=float(raw[i,q])-float(cut[q])
            out[i,q]=margin if base[i,q]<0 and support[i,q] and margin>=0 else float(base[i,q])
    return out

def exchange(after,before,truth,known,ids,indices):
    a,b=after>=0,before>=0
    masks=dict(tp_gained=a&~b&truth&known,tp_lost=~a&b&truth&known,
               fp_added=a&~b&~truth&known,fp_removed=~a&b&~truth&known)
    return {name:dict(count=mask[indices].sum(0).tolist(),
        ids=[[str(ids[i]) for i in indices if mask[i,q]] for q in range(4)]) for name,mask in masks.items()}

def native_partition(winner,truth,known,counts,valid_counts):
    assert winner.shape==truth.shape and winner.dtype.kind in 'iu'
    assert ((winner>=0)&(winner<3600)).all()
    n=len(truth);ii=np.arange(n)[:,None];qq=np.arange(4)[None,:]
    local_known=valid_counts.reshape(n,3600)[ii,winner]>0
    native=counts.reshape(n,3600,4)[ii,winner,qq]>0
    assert not (native & ~local_known).any()
    return dict(native=native,known_wrong=local_known&~native,UNKNOWN=~local_known)

def summarize_native(parts,selected,indices,ids):
    result={}
    for name,mask in parts.items():
        take=mask&selected
        result[name]=dict(count=take[indices].sum(0).tolist(),
            ids=[[str(ids[i]) for i in indices if take[i,q]] for q in range(4)])
    assert np.array_equal(sum(np.array(v['count']) for v in result.values()),selected[indices].sum(0))
    return result

def score(root,task):
    root=root.resolve();task=task.resolve();assert task==(root/'artifacts.local/work'/TASK).resolve()
    out=task/'score-v1';assert not out.exists()
    started=time.perf_counter();inputs={};audit=dict(scalar_candidates=0,scalar_metric_bits=0,packet_slots=0,
        prior_metric_tables=0,native_winner_lookups=0,dense_argmax_recomputed=False)
    def bind(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p)
        inputs[str(p)]=h;return p
    run=task/'run-v1';go=read(bind(task/'score-go-v1.json'))
    assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0
    assert go['scorer_sha256']==sha(__file__)
    rr=read(bind(run/'receipt.json',go['run_receipt_sha256']))
    assert rr['status']=='PASS' and rr['training_steps']==rr['new_cutoffs']==0
    out.mkdir()
    try:
        bind(__file__);bind(Path(__file__).with_name('mz58_score.py'))
        for p,h in rr['inputs'].items():bind(p,h)
        for name,h in rr['outputs'].items():bind(run/name,h)
        assert rr['new_inference_frames']==8192 and rr['initial_parity_frames']==32 and rr['profile_frame_evaluations']==16384
        parity=read(run/'initial-parity.json')
        assert parity['status']=='PASS' and parity['frames']==32 and parity['decision_signs_exact']
        assert tuple(parity['profiles'])==OLD and parity['atol']==2e-5 and parity['rtol']==1e-6
        oldrun=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1'
        r70=read(bind(oldrun/'receipt.json',RUN70_SHA))
        score70=oldrun.parent/'score-v1'
        sr=read(bind(score70/'receipt.json',SCORE70_SHA))
        prior_metrics=read(bind(score70/'result.json',sr['outputs']['result.json']))
        groups=read(bind(oldrun/'groups.json',r70['outputs']['groups.json']))
        def ref(v):
            p=Path(v['path']);return bind(p if p.is_absolute() else run/p,v['sha256'])
        cuts={}
        for key in ('OLD_NEG/OPEN','OLD_NEG/GATED','MZ70/CONTROL','MZ70/DIVERSE'):
            cuts[key]=np.load(ref(rr['frozen']['cutoffs'][key]),allow_pickle=False)
            assert cuts[key].shape==(4,) and cuts[key].dtype==np.float64
            if key.startswith('MZ70/'):
                arm=key.split('/')[1]
                exact(cuts[key],np.load(bind(oldrun/(arm+'-cutoff.npy'),r70['outputs'][arm+'-cutoff.npy']),allow_pickle=False))
                assert rr['frozen']['checkpoints'][arm]['sha256']==r70['outputs'][arm+'.pt']
                ref(rr['frozen']['checkpoints'][arm])
            else:
                mode=key.split('/')[1]
                original=root/'artifacts.local/work/mz51-training-coverage-20260911/run-v1'/('OLD_NEG-'+mode+'-cutoff.npy')
                assert rr['frozen']['cutoffs'][key]['sha256']==OLD_CUT_SHA[mode]
                exact(cuts[key],np.load(bind(original,OLD_CUT_SHA[mode]),allow_pickle=False))
        results={};pairs={};native_report={};coverage={};contexts={}
        with np.load(bind(oldrun/'predictions.npz',r70['outputs']['predictions.npz']),allow_pickle=False) as oldp, \
             np.load(run/'predictions.npz',allow_pickle=False) as newp:
            for source in SOURCES:
                ids=oldp[source+'/frame_ids'];truth=oldp[source+'/truth'];known=oldp[source+'/known']
                assert ids.shape==(4096,) and len(set(ids.tolist()))==4096
                assert truth.shape==known.shape==(4096,4) and truth.dtype==known.dtype==bool
                exact(newp[source+'/frame_ids'],ids)
                records=groups[source+'_records'];assert [r['frame_id'] for r in records]==ids.tolist()
                subsets={'all':np.arange(4096)}
                for field in ('role','family','range','support_context'):
                    for value in sorted({r[field] for r in records}):
                        subsets[field+'/'+value]=np.array([i for i,r in enumerate(records) if r[field]==value])
                assert [len(subsets['role/'+r]) for r in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')]==[2048,1024,1024]
                label=r70['source_info']['label_refs'][source]
                with np.load(bind(label['path'],label['sha256']),allow_pickle=False) as z:
                    counts=z['fullframe_event_counts'];vc=z['valid_counts'];exact(z['frame_ids'],ids)
                assert counts.shape==(4096,45,80,4) and vc.shape==(4096,45,80)
                assert counts.dtype==vc.dtype==np.uint8 and (counts<=vc[...,None]).all() and (vc<=64).all()
                np.testing.assert_array_equal(counts.sum((1,2))>=3,truth)
                values={};parts={};results[source]={};pairs[source]={};native_report[source]={};coverage[source]={};contexts[source]={}
                original_r=oldp[source+'/IDEAL/ranges'];original_v=oldp[source+'/IDEAL/valid']
                for profile in OLD+NEW:
                    prefix=source+'/'+profile+'/'
                    src=oldp if profile in OLD else newp
                    if profile in NEW:
                        er,ev,changed=proxy(original_r,original_v,profile)
                        exact(src[prefix+'ranges'],er);exact(src[prefix+'valid'],ev)
                        audit['packet_slots']+=er.size
                        coverage[source][profile]=dict(changed_zones=int(changed.sum()),changed_frames=int(changed.any(1).sum()),
                            valid_slots=int(ev.sum()),valid_zones=int(ev.any(2).sum()),all_missing_frames=int((~ev.any((1,2))).sum()),
                            unchanged_outside_close=True,never_added_valid=True)
                    else:er,ev=src[prefix+'ranges'],src[prefix+'valid']
                    v={m:src[prefix+m] for m in METHODS}
                    for key in ('OLD_NEG/OPEN','OLD_NEG/GATED')+tuple('MZ70/'+a for a in ARMS):
                        if profile in NEW:
                            calc=candidate(src[prefix+key+'/raw'],src[prefix+key+'/support'],v['MZ37'],cuts[key])
                            np.testing.assert_array_equal(calc,v[key+'/candidate']);audit['scalar_candidates']+=calc.size
                    np.testing.assert_array_equal(v[BASE],np.maximum(v['OLD_NEG/OPEN/candidate'],v['OLD_NEG/GATED/candidate']))
                    for arm in ARMS:
                        key='MZ70/'+arm
                        np.testing.assert_array_equal(v[key+'/UNION'],np.maximum(v[BASE],v[key+'/candidate']))
                        available=src[prefix+key+'/anchor_available'];anchor=src[prefix+key+'/anchor_vector']
                        np.testing.assert_array_equal(available,ev.any((1,2)))
                        assert anchor.shape==(4096,8) and np.isfinite(anchor).all() and not anchor[~available].any()
                        parts[profile,arm]=native_partition(src[prefix+key+'/winner'],truth,known,counts,vc)
                        audit['native_winner_lookups']+=truth.size
                    values[profile]=v;results[source][profile]={};native_report[source][profile]={}
                    for group,ii in subsets.items():
                        table={}
                        for method,score in v.items():
                            assert score.shape==truth.shape and np.isfinite(score).all()
                            m=metrics(score[ii],truth[ii],known[ii]);table[method]=m
                            if group=='all':
                                assert m==scalar_metrics(score,truth,known);audit['scalar_metric_bits']+=score.size
                            if profile in OLD and method in PRIOR_METRIC_METHODS:
                                assert m==prior_metrics['conditions'][source][profile][group][method]
                                audit['prior_metric_tables']+=1
                        results[source][profile][group]=table
                        native_report[source][profile][group]={arm:summarize_native(parts[profile,arm],
                            (v['MZ70/'+arm+'/candidate']>=0)&(v[BASE]<0)&truth&known,ii,ids) for arm in ARMS}
                    ctx={}
                    for i,r in enumerate(records):ctx.setdefault(r['pair_id'],[]).append(i)
                    assert len(ctx)==2048 and all(len(ii)==2 for ii in ctx.values())
                    contexts[source][profile]={}
                    for method,vv in v.items():
                        changed_ids=[]
                        for pair,ii in ctx.items():
                            if not np.array_equal(vv[ii[0]]>=0,vv[ii[1]]>=0):changed_ids.append(pair)
                        contexts[source][profile][method]=dict(pairs=2048,changed_decision_pairs=len(changed_ids),pair_ids=changed_ids)
                for profile in NEW:
                    pairs[source][profile]={}
                    for before in OLD:
                        comparison={}
                        for group,ii in subsets.items():
                            comparison[group]={}
                            for arm in ARMS:
                                for decision in ('candidate','UNION'):
                                    key='MZ70/'+arm+'/'+decision
                                    ex=exchange(values[profile][key],values[before][key],truth,known,ids,ii)
                                    gained=(values[profile][key]>=0)&(values[before][key]<0)&truth&known
                                    lost=(values[profile][key]<0)&(values[before][key]>=0)&truth&known
                                    ex['gained_winner']=summarize_native(parts[profile,arm],gained,ii,ids)
                                    ex['lost_winner']=summarize_native(parts[before,arm],lost,ii,ids)
                                    ex['winner_credit_boundary']='Winner location only; an OR transition can be caused by OLD_NEG. Not branch-causal attribution.'
                                    if decision=='UNION':
                                        old_path=gained&(values[profile][BASE]>=0)
                                        head_only=gained&(values[profile][BASE]<0)
                                        assert np.array_equal(old_path|head_only,gained)
                                        ex['gained_OLD_NEG_already_positive']=dict(count=old_path[ii].sum(0).tolist(),
                                            ids=[[str(ids[i]) for i in ii if old_path[i,q]] for q in range(4)])
                                        ex['gained_head_only_beyond_OLD_NEG']=summarize_native(parts[profile,arm],head_only,ii,ids)
                                    comparison[group][key]=ex
                        pairs[source][profile][before]=comparison
        result=dict(status='PASS',conditions=results,coverage=coverage,native_additions_beyond_profile_OLD_NEG=native_report,
            support_context_pairs=contexts,profiles=list(OLD+NEW),methods=METHODS,query_order=EVENTS,
            claim='Frozen-model unresolved-return sensitivity: keep closest/farthest reported geometric slot; neither is ST strongest-target behavior.',
            selection='No model/profile winner selected; no new cutoff or fit.',
            limitations=['Original native counts used only after prediction sealing. UNKNOWN retained.',
                'Native argmax not independently recomputed; saved winner cell lookup only.',
                'Old/new input profiles change original OLD_NEG and new-head paths together; paired OR gain is not automatically new-head contribution.',
                'CONTROL/DIVERSE weights come from prior matched training; this run measures frozen input sensitivity.'])
        write(out/'result.json',result);write(out/'paired-events.json',pairs);write(out/'audit.json',dict(status='PASS',**audit))
        lines=['# MZ74 frozen-model return survival sensitivity','',result['claim'],'',
            'Each HELD source has1024 attempted frames; query-specific known/UNKNOWN and all source roles remain in result.json. Values are final OLD_NEG OR candidate. No profile/model selection.', '',
            '| Source/profile | Query | CONTROL TP/FP/FN | DIVERSE TP/FP/FN |','|---|---|---:|---:|']
        for s in SOURCES:
            for p in OLD+NEW:
                table=results[s][p]['role/HELDOUT_GEOMETRY']
                for q,event in enumerate(EVENTS):
                    a,b=(table['MZ70/'+arm+'/UNION'] for arm in ARMS)
                    lines.append(f"| {s}/{p} | {event} | {a['tp'][q]}/{a['fp'][q]}/{a['fn'][q]} | {b['tp'][q]}/{b['fp'][q]}/{b['fn'][q]} |")
        lines+=['','Detailed candidate/OR metrics, input coverage and native/UNKNOWN attribution: result.json. Exact TP gained/lost and FP added/removed IDs against each original input: paired-events.json. Both directions are retained, including net-zero swaps. Audit: audit.json. No hardware or safety claim.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        for p,h in inputs.items():assert sha(p)==h,p
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            seconds=time.perf_counter()-started,training_steps=0,new_cutoffs=0,model_inference_frames=0,dense_native_reads=0))
        print(json.dumps(dict(status='PASS',seconds=time.perf_counter()-started,audit=audit)))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs));raise

def self_test():
    rr=np.zeros((2,64,2),np.float32);vv=np.zeros_like(rr,bool)
    rr[0,0]=[1.,1.2];vv[0,0]=True;rr[1,1]=[1.,2.];vv[1,1]=True
    near,nv,changed=proxy(rr,vv,NEW[0]);far,fv,_=proxy(rr,vv,NEW[1])
    assert near[0,0,0]==rr[0,0,0] and far[0,0,1]==rr[0,0,1] and nv.sum()==fv.sum()==3
    assert not fv[0,0,0] and far[0,0,0]==0 and not nv[0,0,1]
    assert changed.sum()==1 and not near[~nv].any()
    for corruption in ('dtype','order','nan','invalid_nonzero'):
        bad=rr.copy()
        if corruption=='dtype':bad=bad.astype(np.float64)
        elif corruption=='order':bad[0,0]=[2.,1.]
        elif corruption=='nan':bad[0,0,0]=np.nan
        else:bad[0,1,0]=1.
        try:proxy(bad,vv,NEW[0])
        except AssertionError:pass
        else:raise AssertionError('Corrupt input accepted: '+corruption)
    raw=np.array([[1.,-1.,2.,0.],[0.,2.,3.,-4.]],np.float32);sup=np.ones_like(raw,bool)
    base=np.full_like(raw,-2.);base[0,2]=1.;cut=np.array([1.,0.,1.,0.])
    value=candidate(raw,sup,base,cut)
    np.testing.assert_array_equal(value,np.where((base<0)&sup&(raw-cut>=0),raw.astype(float)-cut,base))
    t=np.array([[True,False,False,True],[False,True,False,False]]);k=np.ones_like(t);k[1,2]=False
    assert metrics(value,t,k)==scalar_metrics(value,t,k)
    ex=exchange(value,base,t,k,np.array(['a','b']),np.arange(2))
    assert ex['tp_gained']['count']==[1,1,0,1] and ex['fp_added']['count']==[0,0,0,0]
    counts=np.zeros((2,45,80,4),np.uint8);vc=np.zeros((2,45,80),np.uint8)
    counts[0,0,0,0]=vc[0,0,0]=1;vc[1,0,0]=1
    parts=native_partition(np.zeros((2,4),np.int64),t,k,counts,vc)
    assert parts['native'][0,0] and parts['known_wrong'][1].all()
    assert np.array_equal(sum(v.astype(int) for v in parts.values()),np.ones((2,4),int))
    print('PASS synthetic scalar proxy preservation/candidate boundary/UNKNOWN/exchange checks; no experiment input opened')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--task',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        assert a.root is None and a.task is None;self_test()
    else:
        assert a.root is not None and a.task is not None;score(a.root,a.task)
