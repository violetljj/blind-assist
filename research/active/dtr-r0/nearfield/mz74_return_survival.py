"""Frozen MZ70 models under paired surviving-return proxies; no fit or recut."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import setup
from mz50_train import infer as infer_spatial
from mz54_full_rgb_source import RGBStore
from mz56_global_anchor import saved_outputs
from mz62_profile_coverage import candidate_values
from mz63_geometry_transfer import load_predictor as load61, packet as old_packet
from mz69_topology_transfer import load_predictor as load67, DesignBindings
from mz71_missing_state_calibration import load_models, ARMS, MZ70_RECEIPT, DESIGN_SHA
from tof_return_sensitivity import MODES, constrain

TASK='mz74-return-survival-20260911'
SOURCES={'mz61':('mz61-geometry-source-20260911','6f9e4be717b7fe23d66af58a9002a667d7acff304ec0c0af58497dfc502c72c7',load61),
         'mz67':('mz67-topology-source-20260911','23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b',load67)}
PARITY_PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE')


def infer_batch(data, ids, profiles, models, spatial, heads, cuts, store, timing):
    images=[]
    try:
        tick=time.perf_counter()
        for i in ids:images.append(store.load(data['rgb_refs'][int(i)]))
        timing['rgb_decode']+=time.perf_counter()-tick
        torch.cuda.synchronize();tick=time.perf_counter()
        visual,detail=models.visual(images)
        rgb=torch.from_numpy(np.stack([np.asarray(im,np.uint8) for im in images])).permute(0,3,1,2).cuda().float()/255
        dense=fixed_batch_dense(models.context.base,rgb)
        normalized=(dense-models.detail_mean)/models.detail_std
        detail_cpu=detail.cpu().numpy()
        torch.cuda.synchronize();timing['shared_visual_views']+=time.perf_counter()-tick
        result={};tick=time.perf_counter()
        for profile in profiles:
            rr,vv=data['ranges'][ids],data['valid'][ids]
            obs=constrain(rr,vv,profile)[0] if profile in MODES else old_packet(rr,vv,profile)
            rr,vv=obs['ranges'],obs['valid']
            row=dict(ranges=rr,valid=vv,MZ37=models.predict(visual,detail,rr,vv)['MZ37'])
            row.update({'OLD_NEG/'+k:v for k,v in infer_spatial(spatial,detail_cpu,rr,vv,models).items()})
            args=(normalized,torch.from_numpy(rr).cuda(),torch.from_numpy(vv).cuda())
            for arm,head in heads.items():
                row.update({'MZ70/'+arm+'/'+k:v for k,v in saved_outputs(head,args).items()})
            for key,cut in cuts.items():
                row[key+'/candidate']=candidate_values(row[key+'/raw'],row[key+'/support'],row['MZ37'],cut)
            row['OLD_NEG/UNION']=np.maximum(row['OLD_NEG/OPEN/candidate'],row['OLD_NEG/GATED/candidate'])
            for arm in ARMS:
                row['MZ70/'+arm+'/UNION']=np.maximum(row['OLD_NEG/UNION'],row['MZ70/'+arm+'/candidate'])
            result[profile]=row
        torch.cuda.synchronize();timing['readouts']+=time.perf_counter()-tick
        return result
    finally:
        for im in images:im.close()


def run(root,task):
    assert task==(root/'artifacts.local/work'/TASK).resolve()
    reg=read(task/'registration.json')
    assert reg['status']=='ACTIVE' and reg['experiment_id']==TASK
    assert sha(task/'inputs.json')==reg['inputs']['sha256']
    inputs=read(task/'inputs.json');definition=read(task/'primary-definition.json')
    assert definition['new_inference_frames']==8192 and definition['initial_parity_frames']==32
    assert definition['training_steps']==definition['new_cutoffs']==0
    design_path=root/'artifacts.local/work/mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(design_path)==DESIGN_SHA
    design=read(design_path);bind=DesignBindings(design)
    bind(design_path,DESIGN_SHA);bind(task/'inputs.json',reg['inputs']['sha256']);bind(task/'registration.json')
    for ref in inputs['bound_files']:bind(ref['path'],ref['sha256'])
    for ref in design['code'].values():bind(ref['path'],ref['sha256'])
    for p,h in design['frozen_models_dependency_graph']['graph'].items():bind(p,h)
    prior=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1'
    original=read(bind(prior/'receipt.json',MZ70_RECEIPT))
    schedule_path=bind(prior/'schedule.npz',original['outputs']['schedule.npz'])
    with np.load(schedule_path,allow_pickle=False) as z:
        parity_ids={c:np.sort(z[c+'_fit_ids'])[:16] for c in SOURCES}
    previous_path=bind(prior/'predictions.npz',original['outputs']['predictions.npz'])
    cohorts={c:loader(root/'artifacts.local/work'/name,bind,digest)[0] for c,(name,digest,loader) in SOURCES.items()}
    out=task/'run-v1';out.mkdir(exist_ok=False);started=time.perf_counter()
    models=spatial=heads=store=None
    try:
        setup();models,spatial,heads,cuts,checkpoints=load_models(root,bind,design,out)
        store=RGBStore();timing=dict(rgb_decode=0.,shared_visual_views=0.,readouts=0.)
        write(out/'start.json',dict(status='STARTED',frames=8192,profiles=MODES,parity_frames=32,
            device=torch.cuda.get_device_name(),checkpoints=checkpoints,inputs=bind.inputs,
            source_arrays={c:dict(frames=len(d['ranges']),valid_slots=int(d['valid'].sum())) for c,d in cohorts.items()}))
        comparisons=[]
        with np.load(previous_path,allow_pickle=False) as old, torch.inference_mode():
            for c,data in cohorts.items():
                ids=parity_ids[c]
                rows=infer_batch(data,ids,PARITY_PROFILES,models,spatial,heads,cuts,store,timing)
                for profile,row in rows.items():
                    for key,value in row.items():
                        name=c+'/'+profile+'/'+key
                        if key in ('ranges','valid'):continue
                        expected=old[name][ids]
                        check=dict(source=c,profile=profile,key=key)
                        if key.endswith('/winner'):
                            check['winner_equal']=bool(np.array_equal(value,expected))
                        elif value.dtype.kind=='f':
                            np.testing.assert_allclose(value,expected,atol=2e-5,rtol=1e-6)
                            check['max_abs']=float(np.abs(value-expected).max())
                            if key=='MZ37' or key.endswith(('/candidate','/UNION')):
                                np.testing.assert_array_equal(value>=0,expected>=0)
                        else:
                            np.testing.assert_array_equal(value,expected);check['exact']=True
                        comparisons.append(check)
        write(out/'initial-parity.json',dict(status='PASS',frames=32,profiles=PARITY_PROFILES,
            ids={c:ids.tolist() for c,ids in parity_ids.items()},comparisons=comparisons,
            atol=2e-5,rtol=1e-6,decision_signs_exact=True))
        predictions={};frames=0
        with torch.inference_mode():
            for c,data in cohorts.items():
                parts={};predictions[c+'/frame_ids']=data['frame_ids']
                for begin in range(0,len(data['ranges']),16):
                    ids=np.arange(begin,min(begin+16,len(data['ranges'])))
                    rows=infer_batch(data,ids,MODES,models,spatial,heads,cuts,store,timing)
                    for profile,row in rows.items():
                        for key,value in row.items():parts.setdefault(c+'/'+profile+'/'+key,[]).append(value)
                    frames+=len(ids)
                    if begin%512==0 or begin+len(ids)==len(data['ranges']):
                        progress=dict(stage='return-survival',source=c,frames=begin+len(ids),total=len(data['ranges']))
                        write(out/'progress.json',progress);print('INFER',progress,flush=True)
                for key,chunks in parts.items():predictions[key]=np.concatenate(chunks)
        assert frames==8192 and store.loads==8224
        np.savez_compressed(out/'predictions.npz',**predictions)
        bind.check()
        cutrefs={key:dict(path=str(out/(key.replace('/','-')+'-cutoff.npy')),
                         sha256=sha(out/(key.replace('/','-')+'-cutoff.npy')),values=value.tolist())
                 for key,value in cuts.items()}
        receipt=dict(status='PASS',inputs=bind.inputs,frozen=dict(checkpoints=checkpoints,cutoffs=cutrefs),
            arms=ARMS,profiles=MODES,cohorts=list(SOURCES),training_steps=0,new_cutoffs=0,
            new_inference_frames=frames,initial_parity_frames=32,profile_frame_evaluations=frames*2,
            rgb_loads=store.loads,rgb_bytes_read=store.bytes_read,
            native_depth_reads=0,evaluator_label_reads=0,permanent_dense_cache=False,
            backend='CUDA frozen model inference; CPU compact I/O and scalar packet transforms',
            device=torch.cuda.get_device_name(),timing_seconds=timing,seconds=time.perf_counter()-started,
            predictor_inputs=['RGB features','ranges','valid','fixed calibration'],
            limits='Consumed Development sensitivity only; not strongest-target physics or measured sensor reliability',
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        write(out/'receipt.json',receipt);print('PASS',dict(frames=frames,seconds=receipt['seconds']),flush=True)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=bind.inputs));raise
    finally:
        if store is not None:store.close()
        models=spatial=heads=None;gc.collect()
        write(task/'handle-release.json',dict(status='PASS',compact_handles_closed=True,permanent_dense_cache=False,process_exit_required=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.task.resolve())
