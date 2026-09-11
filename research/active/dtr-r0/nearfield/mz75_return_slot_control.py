"""MZ75 prepare/run/score: fixed far-return values, one slot representation change.

prepare and self-test are CPU-only. run requires ACTIVE registration. score
requires ROOT_SCORE_GO. Models are imported only inside run; labels only in score.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

TASK='mz75-return-slot-control-20260911'
FAR='FARTHEST_REPORTED_PROXY'
PACKED='FAR_PACKED_SLOT0'
ARMS=('CONTROL','DIVERSE')
SOURCES=('mz61','mz67')
EVENTS=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
METHODS=('MZ37','OLD_NEG/OPEN/candidate','OLD_NEG/GATED/candidate','OLD_NEG/UNION')+tuple(
    'MZ70/'+arm+'/'+kind for arm in ARMS for kind in ('candidate','UNION'))
R70_SHA='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
R74_SHA='c2a67b4c7326f9beb9f30d12552c7ab493445b6c4053c00edfa577da40baf79d'
S74_SHA='5e720c247bb3df8ec511b2d6ad23f3c4eedae83caf69185a138d796eed22b1bb'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def make_bind():
    inputs={}
    def bind(p,h=None):
        p=Path(p).resolve();v=sha(p);assert h is None or v==h,str(p);inputs[str(p)]=v;return p
    return bind,inputs

def pack(ranges,valid):
    assert ranges.dtype==np.float32 and valid.dtype==bool and ranges.shape==valid.shape and ranges.shape[1:]==(64,2)
    assert np.isfinite(ranges).all() and (ranges[~valid]==0).all()
    assert ((ranges[valid]>0)&(ranges[valid]<=4)).all()
    mask=~valid[:,:,0]&valid[:,:,1]
    rr=ranges.copy();vv=valid.copy()
    rr[:,:,0][mask]=ranges[:,:,1][mask];rr[:,:,1][mask]=0
    vv[:,:,0][mask]=True;vv[:,:,1][mask]=False
    exact(rr[~mask],ranges[~mask]);exact(vv[~mask],valid[~mask])
    exact(np.sort(np.where(vv,rr,np.inf),axis=2),np.sort(np.where(valid,ranges,np.inf),axis=2))
    exact(vv.any(2),valid.any(2))
    return rr,vv,mask

def scalar_pack(ranges,valid):
    """Scorer reconstruction independent of vector pack above."""
    rr=ranges.copy();vv=valid.copy();changes=0
    for i in range(len(rr)):
        for z in range(64):
            if not valid[i,z,0] and valid[i,z,1]:
                rr[i,z,0]=ranges[i,z,1];rr[i,z,1]=0
                vv[i,z,0]=True;vv[i,z,1]=False;changes+=1
    return rr,vv,changes

def prepare(root,task):
    out=task/'preparation-v1';out.mkdir(parents=True,exist_ok=False)
    bind,inputs=make_bind();tick=time.perf_counter();bind(__file__)
    for name in ['mz74_return_survival.py','mz74_score.py','mz58_score.py','mz71_missing_state_calibration.py']:
        bind(Path(__file__).with_name(name))
    t70=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1'
    r70=read(bind(t70/'receipt.json',R70_SHA));g=read(bind(t70/'groups.json',r70['outputs']['groups.json']))
    selection=dict(status='FROZEN_BEFORE_MZ75_INFERENCE',criterion='role HELDOUT_GEOMETRY, original order; parity sorted original TRAIN fit IDs first16/source',sources={})
    with np.load(bind(t70/'schedule.npz',r70['outputs']['schedule.npz'])) as schedule:
        for c in SOURCES:
            records=g[c+'_records'];held=[i for i,r in enumerate(records) if r['role']=='HELDOUT_GEOMETRY']
            train=np.sort(schedule[c+'_fit_ids'])[:16].tolist()
            assert len(held)==1024 and len(train)==16 and not set(held)&set(train)
            assert all(records[i]['role']=='TRAIN_CANDIDATE' for i in train)
            selection['sources'][c]=dict(held_ids=held,held_frame_ids=[records[i]['frame_id'] for i in held],
                parity_ids=train,parity_frame_ids=[records[i]['frame_id'] for i in train])
    t74=root/'artifacts.local/work/mz74-return-survival-20260911/run-v1'
    r74=read(bind(t74/'receipt.json',R74_SHA));bind(t74/'predictions.npz',r74['outputs']['predictions.npz'])
    write(task/'selection.json',selection)
    self_test()
    write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={'selection.json':sha(task/'selection.json')},
        new_inference_frames=2048,parity_train_frames=32,training_steps=0,new_cutoffs=0,seconds=time.perf_counter()-tick))
    print('PREPARE PASS',sha(__file__),flush=True)

def run(root,task):
    # All GPU/model imports occur after explicit run stage is selected.
    import torch
    from mz74_return_survival import infer_batch,SOURCES as SOURCE_REFS,load_models,setup,DesignBindings,RGBStore,DESIGN_SHA
    reg=read(task/'registration.json');assert reg['status']=='ACTIVE' and reg['experiment_id']==TASK
    inp=read(task/'inputs.json');assert sha(task/'inputs.json')==reg['inputs']['sha256']
    dpath=root/'artifacts.local/work/mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(dpath)==DESIGN_SHA;design=read(dpath);bind=DesignBindings(design)
    bind(dpath,DESIGN_SHA);bind(__file__);bind(task/'registration.json');bind(task/'inputs.json',reg['inputs']['sha256'])
    for ref in inp['bound_files']:bind(ref['path'],ref['sha256'])
    for ref in design['code'].values():bind(ref['path'],ref['sha256'])
    for p,h in design['frozen_models_dependency_graph']['graph'].items():bind(p,h)
    prep=read(bind(task/'preparation-v1/receipt.json'));assert prep['status']=='PASS'
    for p,h in prep['inputs'].items():bind(p,h)
    selection=read(bind(task/'selection.json',prep['outputs']['selection.json']))
    prior=root/'artifacts.local/work/mz74-return-survival-20260911/run-v1'
    oldreceipt=read(bind(prior/'receipt.json',R74_SHA));oldpath=bind(prior/'predictions.npz',oldreceipt['outputs']['predictions.npz'])
    cohorts={c:loader(root/'artifacts.local/work'/name,bind,h)[0] for c,(name,h,loader) in SOURCE_REFS.items()}
    out=task/'run-v1';out.mkdir(exist_ok=False);start=time.perf_counter();models=spatial=heads=store=None
    try:
        setup();models,spatial,heads,cuts,checkpoints=load_models(root,bind,design,out);store=RGBStore()
        timing=dict(rgb_decode=0.,shared_visual_views=0.,readouts=0.)
        write(out/'start.json',dict(status='STARTED',frames=2048,parity_frames=32,device=torch.cuda.get_device_name()))
        predictions={};checks=[];packing={}
        with np.load(oldpath) as old,torch.inference_mode():
            for c,data in cohorts.items():
                exact(data['frame_ids'],old[c+'/frame_ids']);sel=selection['sources'][c]
                # Only saved observable FAR packet substitutes the original packet. No label member opened.
                far=dict(data,ranges=old[c+'/'+FAR+'/ranges'],valid=old[c+'/'+FAR+'/valid'])
                ids=np.array(sel['parity_ids']);np.testing.assert_array_equal(data['frame_ids'][ids],sel['parity_frame_ids'])
                row=infer_batch(far,ids,('IDEAL',),models,spatial,heads,cuts,store,timing)['IDEAL']
                for key,value in row.items():
                    expected=old[c+'/'+FAR+'/'+key][ids];item=dict(source=c,key=key)
                    if key.endswith('/winner'):item['winner_equal']=bool(np.array_equal(value,expected))
                    elif value.dtype.kind=='f':
                        np.testing.assert_allclose(value,expected,atol=2e-5,rtol=1e-6)
                        if key=='MZ37' or key.endswith(('/candidate','/UNION')):np.testing.assert_array_equal(value>=0,expected>=0)
                        item['max_abs']=float(np.abs(value-expected).max())
                    else:exact(value,expected)
                    checks.append(item)
                ids=np.array(sel['held_ids']);np.testing.assert_array_equal(data['frame_ids'][ids],sel['held_frame_ids'])
                rr,vv,mask=pack(far['ranges'][ids],far['valid'][ids])
                held=dict(frame_ids=data['frame_ids'][ids],ranges=rr,valid=vv,rgb_refs=[data['rgb_refs'][i] for i in ids])
                packing[c]=dict(frames=len(ids),zones_changed=int(mask.sum()),frames_changed=int(mask.any(1).sum()))
                predictions[c+'/frame_ids']=held['frame_ids'];predictions[c+'/original_indices']=ids;chunks={}
                for begin in range(0,len(ids),16):
                    row=infer_batch(held,np.arange(begin,min(begin+16,len(ids))),('IDEAL',),models,spatial,heads,cuts,store,timing)['IDEAL']
                    for key,value in row.items():chunks.setdefault(c+'/'+PACKED+'/'+key,[]).append(value)
                    if begin%256==0:write(out/'progress.json',dict(source=c,frames=begin+len(row['ranges']),total=len(ids)))
                for key,value in chunks.items():predictions[key]=np.concatenate(value)
        assert store.loads==2080
        write(out/'initial-parity.json',dict(status='PASS',frames=32,ids={c:selection['sources'][c]['parity_ids'] for c in SOURCES},
            comparisons=checks,atol=2e-5,rtol=1e-6,decision_signs_exact=True,winner_equality_descriptive=True))
        np.savez_compressed(out/'predictions.npz',**predictions)
        cutrefs={key:dict(path=str(out/(key.replace('/','-')+'-cutoff.npy')),sha256=sha(out/(key.replace('/','-')+'-cutoff.npy')),values=value.tolist()) for key,value in cuts.items()}
        bind.check()
        write(out/'receipt.json',dict(status='PASS',inputs=bind.inputs,frozen=dict(checkpoints=checkpoints,cutoffs=cutrefs),
            new_inference_frames=2048,initial_parity_frames=32,rgb_loads=store.loads,rgb_bytes_read=store.bytes_read,
            training_steps=0,new_cutoffs=0,native_depth_reads=0,evaluator_label_reads=0,packing=packing,
            timing_seconds=timing,seconds=time.perf_counter()-start,permanent_dense_cache=False,
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print('RUN PASS',time.perf_counter()-start,flush=True)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=bind.inputs));raise
    finally:
        if store is not None:store.close()
        models=spatial=heads=None;gc.collect()
        write(task/'handle-release.json',dict(status='PASS',compact_handles_closed=True,permanent_dense_cache=False,process_exit_required=True))

def score(root,task):
    from mz58_score import metrics,scalar_metrics
    from mz74_score import candidate,exchange
    bind,inputs=make_bind();go=read(bind(task/'score-go-v1.json'))
    assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0 and go['scorer_sha256']==sha(__file__)
    runpath=task/'run-v1';r=read(bind(runpath/'receipt.json',go['run_receipt_sha256']))
    assert r['status']=='PASS' and r['new_inference_frames']==2048 and r['initial_parity_frames']==32 and r['rgb_loads']==2080
    assert r['training_steps']==r['new_cutoffs']==r['evaluator_label_reads']==r['native_depth_reads']==0
    out=task/'score-v1';out.mkdir(exist_ok=False);tick=time.perf_counter()
    try:
        bind(__file__);bind(Path(__file__).with_name('mz74_score.py'));bind(Path(__file__).with_name('mz58_score.py'))
        for p,h in r['inputs'].items():bind(p,h)
        for name,h in r['outputs'].items():bind(runpath/name,h)
        parity=read(runpath/'initial-parity.json');assert parity['status']=='PASS' and parity['frames']==32 and parity['decision_signs_exact']
        selection=read(bind(task/'selection.json'));prep=read(bind(task/'preparation-v1/receipt.json'))
        assert sha(task/'selection.json')==prep['outputs']['selection.json']
        p74=root/'artifacts.local/work/mz74-return-survival-20260911';r74=read(bind(p74/'run-v1/receipt.json',R74_SHA))
        sr74=read(bind(p74/'score-v1/receipt.json',S74_SHA));previous=read(bind(p74/'score-v1/result.json',sr74['outputs']['result.json']))
        p70=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1';r70=read(bind(p70/'receipt.json',R70_SHA))
        groups=read(bind(p70/'groups.json',r70['outputs']['groups.json']))
        cuts={}
        assert set(r['frozen']['cutoffs'])=={'OLD_NEG/OPEN','OLD_NEG/GATED','MZ70/CONTROL','MZ70/DIVERSE'}
        for key,ref in r['frozen']['cutoffs'].items():
            assert ref['sha256']==r74['frozen']['cutoffs'][key]['sha256']
            cuts[key]=np.load(bind(ref['path'],ref['sha256']));assert cuts[key].dtype==np.float64 and cuts[key].shape==(4,)
        for arm in ARMS:assert r['frozen']['checkpoints'][arm]['sha256']==r70['outputs'][arm+'.pt']
        result={};pairs={};audit=dict(packet_scalars=0,candidate_scalars=0,metric_bits=0,native_argmax_recomputed=False)
        with np.load(runpath/'predictions.npz') as new,np.load(bind(p74/'run-v1/predictions.npz',r74['outputs']['predictions.npz'])) as old74,np.load(bind(p70/'predictions.npz',r70['outputs']['predictions.npz'])) as old70:
            for c in SOURCES:
                rec=groups[c+'_records'];ii=np.array([i for i,row in enumerate(rec) if row['role']=='HELDOUT_GEOMETRY'])
                assert len(ii)==1024;np.testing.assert_array_equal(ii,selection['sources'][c]['held_ids']);exact(ii,new[c+'/original_indices'])
                ids=old70[c+'/frame_ids'][ii];exact(ids,new[c+'/frame_ids']);exact(old74[c+'/frame_ids'][ii],ids)
                t=old70[c+'/truth'][ii];k=old70[c+'/known'][ii];prefix=c+'/'+PACKED+'/'
                farprefix=c+'/'+FAR+'/';rr=old74[farprefix+'ranges'][ii];vv=old74[farprefix+'valid'][ii]
                er,ev,count=scalar_pack(rr,vv);exact(er,new[prefix+'ranges']);exact(ev,new[prefix+'valid']);audit['packet_scalars']+=er.size
                exact(np.sort(np.where(ev,er,np.inf),axis=2),np.sort(np.where(vv,rr,np.inf),axis=2));exact(ev.any(2),vv.any(2))
                a={m:new[prefix+m] for m in METHODS}
                for key,cut in cuts.items():
                    calc=candidate(new[prefix+key+'/raw'],new[prefix+key+'/support'],a['MZ37'],cut)
                    np.testing.assert_array_equal(calc,a[key+'/candidate']);audit['candidate_scalars']+=calc.size
                np.testing.assert_array_equal(a['OLD_NEG/UNION'],np.maximum(a['OLD_NEG/OPEN/candidate'],a['OLD_NEG/GATED/candidate']))
                for arm in ARMS:
                    key='MZ70/'+arm;np.testing.assert_array_equal(a[key+'/UNION'],np.maximum(a['OLD_NEG/UNION'],a[key+'/candidate']))
                    exact(new[prefix+key+'/anchor_available'],ev.any((1,2)))
                conditions={PACKED:a,FAR:{m:old74[farprefix+m][ii] for m in METHODS}}
                for profile in ['CLOSEST_REPORTED_PROXY','IDEAL','DROP_CLOSE']:
                    source=old74 if profile.startswith('CLOSEST') else old70
                    conditions[profile]={m:source[c+'/'+profile+'/'+m][ii] for m in METHODS}
                result[c]=dict(frames=1024,packing_zones=count,conditions={});pairs[c]={}
                for profile,values in conditions.items():
                    table={m:metrics(v,t,k) for m,v in values.items()};result[c]['conditions'][profile]=table
                    for m,v in values.items():assert table[m]==scalar_metrics(v,t,k);audit['metric_bits']+=v.size
                    if profile!=PACKED:
                        for m in METHODS:assert table[m]==previous['conditions'][c][profile]['role/HELDOUT_GEOMETRY'][m]
                        pairs[c][profile]={m:exchange(a[m],values[m],t,k,ids,np.arange(1024)) for m in METHODS}
                # Separate inherited OLD_NEG gains from head-only OR gains; no new native attribution.
                for arm in ARMS:
                    key='MZ70/'+arm+'/UNION';gain=(a[key]>=0)&(conditions[FAR][key]<0)&t&k
                    pairs[c][FAR][key]['gain_OLD_NEG_positive']=(gain&(a['OLD_NEG/UNION']>=0)).sum(0).tolist()
                    pairs[c][FAR][key]['gain_head_only']=(gain&(a['OLD_NEG/UNION']<0)).sum(0).tolist()
        write(out/'result.json',dict(status='PASS',sources=result,query_order=EVENTS,primary_reference=FAR,
            interpretation='Slot-representation control only. All distances and availability preserved. No profile/model winner or hardware claim.'))
        write(out/'paired-events.json',pairs);write(out/'audit.json',dict(status='PASS',**audit))
        lines=['# MZ75 return-slot control','', 'Same retained far distances; slot 1 versus slot 0. Original cutoffs and weights. Each source: 1,024 HELD frames.', '',
            '| Source | Arm | Query | FAR TP/FP/FN | Packed TP/FP/FN |','|---|---|---|---:|---:|']
        for c in SOURCES:
            for arm in ARMS:
                key='MZ70/'+arm+'/UNION';a=result[c]['conditions'][FAR][key];b=result[c]['conditions'][PACKED][key]
                for q,event in enumerate(EVENTS):lines.append(f"| {c} | {arm} | {event} | {a['tp'][q]}/{a['fp'][q]}/{a['fn'][q]} | {b['tp'][q]}/{b['fp'][q]}/{b['fn'][q]} |")
        lines+=['','All eight decisions, query known/UNKNOWN denominators and descriptive original IDEAL/CLOSEST/DROP comparisons: result.json. Exact gain/loss and added/removed false-bit IDs: paired-events.json. Native winners were not reattributed; no native source or model execution in score.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        for p,h in inputs.items():assert sha(p)==h,p
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            seconds=time.perf_counter()-tick,model_inference=0,training_steps=0,new_cutoffs=0,native_reads=0))
        print('SCORE PASS',time.perf_counter()-tick,flush=True)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs));raise

def self_test():
    r=np.zeros((2,64,2),np.float32);v=np.zeros_like(r,bool)
    r[0,0,1]=1.8;v[0,0,1]=True;r[0,1]=[1,2];v[0,1]=True;r[1,0,0]=.7;v[1,0,0]=True
    rr,vv,mask=pack(r,v);er,ev,n=scalar_pack(r,v);exact(rr,er);exact(vv,ev);assert n==mask.sum()==1
    assert rr[0,0,0]==r[0,0,1] and not vv[0,0,1]
    exact(pack(rr,vv)[0],rr);exact(pack(rr,vv)[1],vv)
    for bad in [r.astype(np.float64),np.where(v,r,np.float32(1))]:
        try:pack(bad,v)
        except AssertionError:pass
        else:raise AssertionError('Invalid dtype/canonical missing range accepted')
    from mz74_score import candidate,exchange
    from mz58_score import metrics,scalar_metrics
    raw=np.array([[1,-1,3,0]],np.float32);support=np.ones((1,4),bool);base=np.full((1,4),-1.,np.float32);cut=np.array([1,0,2,0.])
    a=candidate(raw,support,base,cut);t=np.array([[1,0,0,1]],bool);k=np.array([[1,1,0,1]],bool)
    assert metrics(a,t,k)==scalar_metrics(a,t,k)
    ex=exchange(a,base,t,k,np.array(['x']),np.array([0]));assert ex['tp_gained']['count']==[1,0,0,1]
    print('SELFTEST PASS: packing identity/idempotence/scalar parity; exact cutoff boundary and UNKNOWN; no model imported')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','self-test','run','score'],required=True)
    parser.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'));parser.add_argument('--task',type=Path)
    args=parser.parse_args();root=args.root.resolve();task=(args.task or root/'artifacts.local/work'/TASK).resolve()
    assert task==(root/'artifacts.local/work'/TASK).resolve()
    if args.stage=='self-test':self_test()
    else:globals()[args.stage](root,task)
