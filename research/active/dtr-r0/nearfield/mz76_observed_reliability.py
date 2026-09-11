"""Registered, fixed two-source256-step continuation; no execution on import."""
import argparse,gc,time,traceback
from pathlib import Path
import sys
ROOT_DEFAULT=Path('E:/linnan/linnan')
sys.path.insert(0,str(ROOT_DEFAULT/'research/active/dtr-r0/nearfield'))
import numpy as np
import torch
from mz74_return_survival import SOURCES,setup,load_models,DesignBindings,RGBStore,DESIGN_SHA
from mz5_ensemble_readout import read,write,sha
from mz36_frozen_inference import fixed_batch_dense
from mz50_train import infer as infer_spatial
from mz56_global_anchor import saved_outputs
from mz54_full_rgb import parameters_sha
from mz62_profile_coverage import candidate_values
from tof_return_sensitivity import constrain
from tof_model_adapter import legacy_batch_inputs
from reliability_anchor_model import ReliabilityAnchorQuery,loss_for

TASK='mz76-observed-reliability-20260911'
PROFILES=('IDEAL','CLOSEST_REPORTED_PROXY','FARTHEST_REPORTED_PROXY')
ARMS=('BASE','RELIABILITY')
R70_SHA='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'

def observation(r,v,profile):
    if profile!='IDEAL':values,_=constrain(r,v,profile);r,v=values['ranges'],values['valid']
    return legacy_batch_inputs(r,v)[0]

def open_images(refs,store):
    images=[]
    try:
        for ref in refs:images.append(store.load(ref))
        return images
    except BaseException:
        for im in images:im.close()
        raise

def full_features(images,models):
    rgb=torch.from_numpy(np.stack([np.asarray(im,np.uint8) for im in images])).permute(0,3,1,2).cuda().float()/255
    with torch.no_grad():dense=fixed_batch_dense(models.context.base,rgb)
    return (dense-models.detail_mean)/models.detail_std

def eval_batch(data,ids,models,spatial,original,newheads,cuts,store,timing):
    tick=time.perf_counter()
    images=open_images([data['rgb_refs'][int(i)] for i in ids],store)
    timing['evaluation_decode']+=time.perf_counter()-tick
    try:
        tick=time.perf_counter();visual,detail=models.visual(images);dense=full_features(images,models)
        detail_cpu=detail.cpu().numpy();torch.cuda.synchronize();timing['evaluation_views']+=time.perf_counter()-tick
        rows={}
        for profile in PROFILES:
            obs=observation(data['ranges'][ids],data['valid'][ids],profile);rr,vv=obs['ranges'],obs['valid']
            row=dict(ranges=rr,valid=vv,MZ37=models.predict(visual,detail,rr,vv)['MZ37'])
            row.update({'OLD_NEG/'+k:v for k,v in infer_spatial(spatial,detail_cpu,rr,vv,models).items()})
            args=(dense,torch.from_numpy(rr).cuda(),torch.from_numpy(vv).cuda())
            for key,head in [('MZ70/DIVERSE',original)]+[('MZ76/'+a,newheads[a]) for a in ARMS]:
                row.update({key+'/'+k:v for k,v in saved_outputs(head,args).items()})
            for key in ('OLD_NEG/OPEN','OLD_NEG/GATED'):
                row[key+'/candidate']=candidate_values(row[key+'/raw'],row[key+'/support'],row['MZ37'],cuts[key])
            row['OLD_NEG/UNION']=np.maximum(row['OLD_NEG/OPEN/candidate'],row['OLD_NEG/GATED/candidate'])
            for key in ('MZ70/DIVERSE','MZ76/BASE','MZ76/RELIABILITY'):
                row[key+'/candidate']=candidate_values(row[key+'/raw'],row[key+'/support'],row['MZ37'],cuts['MZ70/DIVERSE'])
                row[key+'/UNION']=np.maximum(row['OLD_NEG/UNION'],row[key+'/candidate'])
            rows[profile]=row
        return rows
    finally:
        for im in images:im.close()

def run(root,task):
    assert task==(root/'artifacts.local/work'/TASK).resolve()
    reg=read(task/'registration.json');assert reg['status']=='ACTIVE' and reg['experiment_id']==TASK
    assert sha(task/'inputs.json')==reg['inputs']['sha256'];inp=read(task/'inputs.json')
    dp=root/'artifacts.local/work/mz69-topology-transfer-20260911/design-inputs.json';assert sha(dp)==DESIGN_SHA
    design=read(dp);bind=DesignBindings(design);bind(dp,DESIGN_SHA);bind(__file__)
    bind(task/'registration.json');bind(task/'inputs.json',reg['inputs']['sha256'])
    for ref in inp['bound_files']:bind(ref['path'],ref['sha256'])
    for ref in design['code'].values():bind(ref['path'],ref['sha256'])
    for p,h in design['frozen_models_dependency_graph']['graph'].items():bind(p,h)
    prep=read(bind(task/'preparation-v1/receipt.json'));assert prep['status']=='PASS_PREPARED'
    for p,h in prep['inputs'].items():bind(p,h)
    for name,h in prep['outputs'].items():
        path=task/'preparation-v1'/name
        if not path.exists():path=task/name
        bind(path,h)
    selection=read(task/'selection.json')['sources'];schedule=dict(np.load(task/'schedule.npz'))
    labels=dict(np.load(task/'preparation-v1/train-labels.npz'))
    assert np.array_equal(schedule['profile_index'],np.arange(256)%3)
    cohorts={c:loader(root/'artifacts.local/work'/name,bind,h)[0] for c,(name,h,loader) in SOURCES.items()}
    lookup={}
    for c,data in cohorts.items():
        sel=selection[c];fit=np.asarray(sel['fit_ids'],np.int64);assert len(fit)==2048 and (np.diff(fit)>0).all()
        assert schedule[c].shape==(256,8) and schedule[c].dtype==np.int64
        np.testing.assert_array_equal(np.sort(schedule[c].ravel()),fit)
        np.testing.assert_array_equal(data['frame_ids'][fit],labels[c+'/frame_ids'])
        np.testing.assert_array_equal(data['frame_ids'][fit],sel['fit_frame_ids'])
        assert len(sel['held_ids'])==1024 and not np.isin(sel['held_ids'],fit).any()
        assert len(sel['parity_ids'])==16 and np.isin(sel['parity_ids'],fit).all()
        for key,shape in [('cell_truth',(2048,45,80,4)),('cell_known',(2048,45,80)),('truth',(2048,4)),('known',(2048,4))]:
            assert labels[c+'/'+key].shape==shape and labels[c+'/'+key].dtype==bool
        assert not (labels[c+'/cell_truth']&~labels[c+'/cell_known'][...,None]).any()
        lookup[c]={int(i):j for j,i in enumerate(fit)}
    oldpath=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1'
    r70=read(bind(oldpath/'receipt.json',R70_SHA));oldprediction=bind(oldpath/'predictions.npz',r70['outputs']['predictions.npz'])
    out=task/'run-v1';out.mkdir(exist_ok=False);start=time.perf_counter();store=models=spatial=loaded=newheads=optimizers=None
    try:
        setup();models,spatial,loaded,cuts,checkpoints=load_models(root,bind,design,out);original=loaded['DIVERSE']
        geometry=torch.load(bind(design['checkpoints']['MZ54_constructor']['path'],design['checkpoints']['MZ54_constructor']['sha256']),map_location='cpu',weights_only=True)
        newheads={}
        for arm in ARMS:
            torch.manual_seed(176);newheads[arm]=ReliabilityAnchorQuery(geometry,original.state_dict(),arm).cuda().requires_grad_(True)
            assert sum(p.numel() for p in newheads[arm].parameters())==11116
        initial={a:parameters_sha(h) for a,h in newheads.items()};assert len(set(initial.values()))==1
        for h in newheads.values():h.eval()
        store=RGBStore();timing=dict(training_decode=0.,training_encode=0.,training_updates=0.,evaluation_decode=0.,evaluation_views=0.)
        write(out/'start.json',dict(status='STARTED',steps_per_arm=256,arms=ARMS,profiles=PROFILES,device=torch.cuda.get_device_name(),initial_parameters_sha256=initial))
        parity=[]
        with torch.inference_mode(),np.load(oldprediction) as prior:
            for c,data in cohorts.items():
                ids=np.array(selection[c]['parity_ids']);rows=eval_batch(data,ids,models,spatial,original,newheads,cuts,store,timing)
                for profile,row in rows.items():
                    if profile=='IDEAL':
                        old=prior[c+'/IDEAL/MZ70/DIVERSE/raw'][ids]
                        np.testing.assert_allclose(row['MZ70/DIVERSE/raw'],old,atol=2e-5,rtol=1e-6)
                        np.testing.assert_array_equal(row['MZ70/DIVERSE/support'],prior[c+'/IDEAL/MZ70/DIVERSE/support'][ids])
                        np.testing.assert_array_equal(row['MZ70/DIVERSE/candidate']>=0,prior[c+'/IDEAL/MZ70/DIVERSE/candidate'][ids]>=0)
                    for arm in ARMS:
                        key='MZ76/'+arm;expected=row['MZ70/DIVERSE/raw']
                        np.testing.assert_allclose(row[key+'/raw'],expected,atol=2e-5,rtol=1e-6)
                        np.testing.assert_array_equal(row[key+'/support'],row['MZ70/DIVERSE/support'])
                        np.testing.assert_array_equal(row[key+'/candidate']>=0,row['MZ70/DIVERSE/candidate']>=0)
                        parity.append(dict(source=c,profile=profile,arm=arm,max_abs=float(np.abs(row[key+'/raw']-expected).max())))
        write(out/'initial-parity.json',dict(status='PASS',frames=32,comparisons=parity,atol=2e-5,rtol=1e-6,decision_signs_exact=True,ids={c:selection[c]['parity_ids'] for c in SOURCES}))
        optimizers={a:torch.optim.Adam(h.parameters(),lr=.0001) for a,h in newheads.items()}
        for h in newheads.values():h.train()
        torch.manual_seed(176);steps=[]
        for step in range(256):
            refs=[];rr=[];vv=[];ys={k:[] for k in ('cell_truth','cell_known','truth','known')}
            for c,data in cohorts.items():
                ids=schedule[c][step];local=np.array([lookup[c][int(i)] for i in ids])
                refs.extend(data['rgb_refs'][i] for i in ids);rr.append(data['ranges'][ids]);vv.append(data['valid'][ids])
                for k in ys:ys[k].append(labels[c+'/'+k][local])
            tick=time.perf_counter();images=open_images(refs,store)
            timing['training_decode']+=time.perf_counter()-tick;tick=time.perf_counter()
            try:dense=full_features(images,models)
            finally:
                for im in images:im.close()
            torch.cuda.synchronize();timing['training_encode']+=time.perf_counter()-tick
            assert not dense.requires_grad and not torch.is_inference(dense)
            profile=PROFILES[step%3];obs=observation(np.concatenate(rr),np.concatenate(vv),profile)
            args=(dense,torch.from_numpy(obs['ranges']).cuda(),torch.from_numpy(obs['valid']).cuda())
            truth_args=[torch.from_numpy(np.concatenate(ys[k])).cuda() for k in ('cell_truth','cell_known','truth','known')]
            tick=time.perf_counter();row=dict(step=step+1,profile=profile,frames=16,arms={})
            for arm,h in newheads.items():
                loss,parts=loss_for(h.inspect(*args),*truth_args);assert torch.isfinite(loss)
                optimizers[arm].zero_grad();loss.backward();optimizers[arm].step()
                row['arms'][arm]=dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
            torch.cuda.synchronize();timing['training_updates']+=time.perf_counter()-tick;steps.append(row)
            del dense,args,truth_args,loss,parts
            if step%32==0 or step==255:write(out/'progress.json',dict(stage='fit',**row));print('FIT',step+1,flush=True)
        write(out/'training-audit.json',dict(status='PASS',steps=steps))
        for arm,h in newheads.items():h.eval();torch.save({k:v.detach().cpu() for k,v in h.state_dict().items()},out/(arm+'.pt'))
        predictions={}
        with torch.inference_mode():
            for c,data in cohorts.items():
                selected=np.array(selection[c]['held_ids']);np.testing.assert_array_equal(data['frame_ids'][selected],selection[c]['held_frame_ids'])
                predictions[c+'/frame_ids']=data['frame_ids'][selected];predictions[c+'/original_indices']=selected;chunks={}
                for begin in range(0,len(selected),16):
                    ids=selected[begin:begin+16];rows=eval_batch(data,ids,models,spatial,original,newheads,cuts,store,timing)
                    for profile,row in rows.items():
                        for key,value in row.items():chunks.setdefault(c+'/'+profile+'/'+key,[]).append(value)
                    if begin%256==0:write(out/'progress.json',dict(stage='eval',source=c,frames=begin+len(ids)))
                for key,value in chunks.items():predictions[key]=np.concatenate(value)
        assert store.loads==32+4096+2048
        np.savez_compressed(out/'predictions.npz',**predictions);bind.check()
        cutrefs={key:dict(path=str(out/(key.replace('/','-')+'-cutoff.npy')),sha256=sha(out/(key.replace('/','-')+'-cutoff.npy')),values=cut.tolist()) for key,cut in cuts.items()}
        write(out/'receipt.json',dict(status='PASS',inputs=bind.inputs,arms=ARMS,profiles=PROFILES,
            steps_per_arm=256,training_steps_per_arm=256,trainable_parameters_per_arm=11116,initial_checkpoint=checkpoints['DIVERSE'],initial_parameters_sha256=initial,
            fits={a:dict(steps=256,training_frames=4096,source_frames=dict(mz61=2048,mz67=2048)) for a in ARMS},
            old_negative_replay_frames=0,new_cutoffs=0,cutoff_authority='Original MZ70 DIVERSE unchanged for both continuations',
            frozen=dict(checkpoints=checkpoints,cutoffs=cutrefs),initial_parity_frames=32,evaluation_frames=2048,evaluation_profiles=3,
            rgb_loads=store.loads,rgb_bytes_read=store.bytes_read,native_depth_reads=0,HELD_label_reads=0,
            permanent_dense_cache=False,training_feature_peak_frames=16,training_feature_bytes=16*64*45*80*4,
            timing_seconds=timing,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),
            claim='Matched two-source continuation only; observable geometry proxies, not hardware quality or general old-scene retention.',
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print('PASS',time.perf_counter()-start,flush=True)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=bind.inputs));raise
    finally:
        if store is not None:store.close()
        models=spatial=loaded=newheads=optimizers=None;gc.collect()
        write(task/'handle-release.json',dict(status='PASS',compact_handles_closed=True,permanent_dense_cache=False,process_exit_required=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT_DEFAULT);p.add_argument('--task',type=Path)
    a=p.parse_args();root=a.root.resolve();run(root,(a.task or root/'artifacts.local/work'/TASK).resolve())
