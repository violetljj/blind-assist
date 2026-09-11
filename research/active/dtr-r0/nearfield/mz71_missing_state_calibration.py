"""Frozen MZ70 readouts with calibration routed by observed missing state."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from mz5_ensemble_readout import read, write, sha
from mz15_train import cutoff_zero_added
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import FrozenModels, setup
from mz50_echo_independent import SpatialQuery
from mz50_train import infer as infer_spatial
from mz54_full_rgb_source import RGBStore
from mz56_global_anchor import saved_outputs
from mz56_global_anchor_model import AnchorQuery
from mz62_profile_coverage import array_identity, candidate_values
from mz69_topology_transfer import DesignBindings, load_state

TASK='mz71-missing-state-calibration-20260911'
ARMS=('CONTROL','DIVERSE')
COHORTS=('DEV','relation10000','distance5000','rich','mz36','mz48','mz55','mz61','mz67')
PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE','ALL_INVALID')
MZ70_RECEIPT='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
DESIGN_SHA='f4d32fe0b8b0a7e16f96ccae1ab0ea1f05f4921fd6bc99e0e4335c344c020ca5'


def routed_candidate(raw,support,base,original,missing,cut):
    """Keep every nonmissing row byte-exact; no source/profile/label chooses state."""
    assert raw.shape==support.shape==base.shape==original.shape
    assert missing.dtype==bool and missing.shape==(len(raw),)
    result=original.copy()
    result[missing]=candidate_values(raw[missing],support[missing],base[missing],cut)
    np.testing.assert_array_equal(result[~missing],original[~missing])
    return result


def load_models(root,bind,design,out):
    models=FrozenModels(root,bind)
    def state(ref):return torch.load(bind(ref['path'],ref['sha256']),map_location='cpu',weights_only=True)
    spatial=load_state(SpatialQuery(models.rank),state(design['checkpoints']['OLD_NEG']))
    initial=state(design['checkpoints']['MZ54_constructor'])
    old=root/'artifacts.local/work/mz70-diverse-learning-20260911/run-v1'
    receipt=read(bind(old/'receipt.json',MZ70_RECEIPT))
    heads={};fixed={};refs={}
    for arm in ARMS:
        path=bind(old/(arm+'.pt'),receipt['outputs'][arm+'.pt'])
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        heads[arm]=load_state(AnchorQuery(initial,'GLOBAL_ANCHOR'),checkpoint)
        refs[arm]=dict(path=str(path),sha256=sha(path),strict_state_tensors=len(checkpoint))
        path=bind(old/(arm+'-cutoff.npy'),receipt['outputs'][arm+'-cutoff.npy'])
        fixed['MZ70/'+arm]=np.load(path,allow_pickle=False)
        np.save(out/('MZ70-'+arm+'-cutoff.npy'),fixed['MZ70/'+arm])
        assert sha(out/('MZ70-'+arm+'-cutoff.npy'))==sha(path)
    for key in ('OLD_NEG/OPEN','OLD_NEG/GATED'):
        ref=design['cutoffs'][key];path=bind(ref['path'],ref['sha256'])
        fixed[key]=np.load(path,allow_pickle=False)
        np.testing.assert_array_equal(fixed[key],ref['values'])
        np.save(out/(key.replace('/','-')+'-cutoff.npy'),fixed[key])
        assert sha(out/(key.replace('/','-')+'-cutoff.npy'))==ref['sha256']
    return models,spatial,heads,fixed,refs


def infer_batch(data,ids,observed,models,spatial,heads,fixed,store,timing):
    images=[]
    try:
        tick=time.perf_counter()
        for i in ids:images.append(store.load(data['rgb_refs'][int(i)]))
        timing['rgb_decode']+=time.perf_counter()-tick
        torch.cuda.synchronize();tick=time.perf_counter()
        visual,detail=models.visual(images)
        full_rgb=torch.from_numpy(np.stack([np.asarray(im,np.uint8) for im in images])).permute(0,3,1,2).cuda().float()/255
        full=fixed_batch_dense(models.context.base,full_rgb)
        assert full.shape==(len(ids),64,45,80)
        normalized=(full-models.detail_mean)/models.detail_std
        detail_cpu=detail.cpu().numpy()
        torch.cuda.synchronize();timing['visual_views']+=time.perf_counter()-tick
        tick=time.perf_counter()
        rr,vv=observed['ranges'][ids],observed['valid'][ids]
        assert not rr.any() and not vv.any()
        baseline=models.predict(visual,detail,rr,vv)
        row=dict(MZ37=baseline['MZ37'])
        for key,value in infer_spatial(spatial,detail_cpu,rr,vv,models).items():row['OLD_NEG/'+key]=value
        args=(normalized,torch.from_numpy(rr).cuda(),torch.from_numpy(vv).cuda())
        for arm,head in heads.items():
            values=saved_outputs(head,args)
            assert not values['anchor_available'].any() and not values['anchor_vector'].any()
            row.update({'MZ70/'+arm+'/'+key:value for key,value in values.items()})
        for key,cut in fixed.items():row[key+'/candidate']=candidate_values(row[key+'/raw'],row[key+'/support'],row['MZ37'],cut)
        row['OLD_NEG/UNION']=np.maximum(row['OLD_NEG/OPEN/candidate'],row['OLD_NEG/GATED/candidate'])
        for arm in ARMS:row['MZ70/'+arm+'/UNION']=np.maximum(row['OLD_NEG/UNION'],row['MZ70/'+arm+'/candidate'])
        torch.cuda.synchronize();timing['readouts']+=time.perf_counter()-tick
        return row
    finally:
        for im in images:im.close()


def initial_parity(bundle,models,spatial,heads,fixed,store,timing):
    comparisons=[]
    with torch.inference_mode():
        for cohort in ('mz61','mz67'):
            ids=np.asarray(bundle['initial_parity_ids'][cohort],np.int64)
            assert len(ids)==16
            prefix=cohort+'/ALL_INVALID/'
            row=infer_batch(bundle['cohorts'][cohort],ids,bundle['observed_packets'][cohort+'/ALL_INVALID'],models,spatial,heads,fixed,store,timing)
            for key,value in row.items():
                expected=bundle['predictions'][prefix+key][ids]
                if key.endswith('/winner'):
                    # Winner ties can differ without changing pooled output; native
                    # attribution for replayed sources remains their sealed array.
                    comparisons.append(dict(cohort=cohort,key=key,winner_equal=bool(np.array_equal(value,expected))))
                elif value.dtype.kind=='f':
                    np.testing.assert_allclose(value,expected,atol=2e-5,rtol=1e-6)
                    if key=='MZ37' or key.endswith(('/candidate','/UNION')):np.testing.assert_array_equal(value>=0,expected>=0)
                    comparisons.append(dict(cohort=cohort,key=key,max_abs=float(np.abs(value-expected).max())))
                else:
                    np.testing.assert_array_equal(value,expected)
                    comparisons.append(dict(cohort=cohort,key=key,exact=True))
    return dict(status='PASS',unique_train_frames=32,selected={c:np.asarray(ids).tolist() for c,ids in bundle['initial_parity_ids'].items()},profiles=['ALL_INVALID'],atol=2e-5,rtol=1e-6,comparisons=comparisons,scientific_source_predictions_replaced=False)


def run(root,task):
    from mz71_missing_state_source import prepare
    assert task==(root/'artifacts.local/work'/TASK).resolve()
    registration=read(task/'registration.json')
    assert registration['status']=='ACTIVE' and registration['experiment_id']==TASK
    assert sha(task/'inputs.json')==registration['inputs']['sha256']
    inputs=read(task/'inputs.json')
    design_path=root/'artifacts.local/work/mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(design_path)==DESIGN_SHA
    design=read(design_path);bind=DesignBindings(design)
    bind(design_path,DESIGN_SHA);bind(task/'inputs.json',registration['inputs']['sha256']);bind(task/'registration.json')
    for ref in design['code'].values():bind(ref['path'],ref['sha256'])
    for path,digest in design['frozen_models_dependency_graph']['graph'].items():bind(path,digest)
    for ref in inputs['bound_files']:bind(ref['path'],ref['sha256'])
    definition=read(bind(task/'primary-definition.json'))
    assert definition['status']=='FROZEN_BEFORE_INFERENCE' and definition['training_steps']==0 and definition['new_cutoff_vectors']==2
    assert definition['new_inference_frames']==9544 and definition['initial_parity_frames']==32
    out=task/'run-v1';out.mkdir(exist_ok=False);started=time.perf_counter()
    store=models=spatial=heads=None
    try:
        bundle=prepare(root,bind);p=bundle['predictions'];inherited=tuple(p);identity=array_identity(p)
        write(out/'groups.json',bundle['original_groups_document'])
        setup();models,spatial,heads,fixed,checkpoints=load_models(root,bind,design,out)
        store=RGBStore();timing=dict(rgb_decode=0.,visual_views=0.,readouts=0.)
        write(out/'start.json',dict(status='STARTED',training_steps=0,new_inference_frames=9544,initial_parity_frames=32,checkpoints=checkpoints,inputs=bind.inputs,device=torch.cuda.get_device_name(),source_info=bundle['source_info']))
        write(out/'initial-parity.json',initial_parity(bundle,models,spatial,heads,fixed,store,timing))
        new_frames=0
        with torch.inference_mode():
            for cohort in COHORTS[:7]:
                data=bundle['cohorts'][cohort];parts={};prefix=cohort+'/ALL_INVALID/'
                for begin in range(0,len(data['ranges']),16):
                    ids=np.arange(begin,min(begin+16,len(data['ranges'])))
                    row=infer_batch(data,ids,bundle['observed_packets'][cohort+'/ALL_INVALID'],models,spatial,heads,fixed,store,timing)
                    for key,value in row.items():parts.setdefault(key,[]).append(value)
                    new_frames+=len(ids)
                    if begin%1024==0 or begin+len(ids)==len(data['ranges']):
                        progress=dict(stage='missing-inference',cohort=cohort,frames=begin+len(ids),total=len(data['ranges']))
                        write(out/'progress.json',progress);print('INFER',progress,flush=True)
                for key,values in parts.items():
                    assert prefix+key not in p
                    p[prefix+key]=np.concatenate(values)
        assert new_frames==9544 and store.loads==9576
        for prefix,observed in bundle['observed_packets'].items():
            for name,value in observed.items():
                key=prefix+'/'+name
                if key in p:np.testing.assert_array_equal(p[key],value)
                else:p[key]=value
            missing=~observed['valid'].any((1,2))
            np.testing.assert_array_equal(missing,bundle['missing_masks'][prefix])
            p[prefix+'/MZ71/missing_mask']=missing
        cal=bundle['groups']['calibration']
        truth=np.concatenate([p['DEV/truth'],p['mz48/truth'][cal]])
        known=np.concatenate([p['DEV/known'],p['mz48/known'][cal]])
        assert truth.shape==known.shape==(1256,4) and known.all()
        def collect(key):return np.concatenate([p['DEV/ALL_INVALID/'+key],p['mz48/ALL_INVALID/'+key][cal]])
        new_cuts={};routing={}
        for arm in ARMS:
            oldkey='MZ70/'+arm;newkey='MZ71/'+arm
            new_cuts[arm]=cutoff_zero_added(collect(oldkey+'/raw'),collect(oldkey+'/support'),collect('MZ37'),truth)
            assert np.isfinite(new_cuts[arm]).all()
            np.save(out/(arm+'-cutoff.npy'),new_cuts[arm])
            for cohort in COHORTS:
                for profile in PROFILES:
                    prefix=cohort+'/'+profile+'/'
                    missing=p[prefix+'MZ71/missing_mask']
                    value=routed_candidate(p[prefix+oldkey+'/raw'],p[prefix+oldkey+'/support'],p[prefix+'MZ37'],p[prefix+oldkey+'/candidate'],missing,new_cuts[arm])
                    p[prefix+newkey+'/candidate']=value
                    union=np.maximum(p[prefix+'OLD_NEG/UNION'],value)
                    p[prefix+newkey+'/UNION']=union
                    np.testing.assert_array_equal(union[~missing],p[prefix+oldkey+'/UNION'][~missing])
                    routing.setdefault(cohort,{})[profile]=dict(frames=len(missing),missing=int(missing.sum()),nonmissing=int((~missing).sum()))
        assert array_identity({key:p[key] for key in inherited})==identity
        np.savez_compressed(out/'predictions.npz',**p)
        write(out/'routing.json',routing)
        bind.check()
        receipt=dict(status='PASS',inputs=bind.inputs,source_info=bundle['source_info'],checkpoints=checkpoints,
            arms=ARMS,training_steps=0,new_cutoffs=2,threshold_sweeps=0,original_calibration_frames=1256,calibration_profile='ALL_INVALID',
            mz55_calibration_rows_used=0,mz61_calibration_rows_used=0,mz67_calibration_rows_used=0,
            observed_state_route='no valid measured slots; source/profile names are not prediction inputs',
            raw_scores_changed=False,nonmissing_rows_byte_exact=True,baseline_arrays_preserved=list(inherited),baseline_array_identities=identity,
            new_inference_cohorts=list(COHORTS[:7]),new_inference_frames=new_frames,initial_parity_unique_train_frames=32,
            source61_67_cohort_reinference_frames=0,rgb_loads=store.loads,rgb_bytes_read=store.bytes_read,
            encoder_frames_per_view={v:store.loads for v in ('global256x144_BOX','crop224x224','full640x360')},
            original_mz70_receipt_sha256=MZ70_RECEIPT,old_negative_cutoffs_unchanged=True,fixed_checkpoints=True,
            native_depth_reads=0,permanent_dense_cache=False,old_dense_cache_reads=0,source_unknown_preserved=True,
            backend='CUDA frozen inference; CPU compact I/O and two fixed calibration rules',device=torch.cuda.get_device_name(),
            timing_seconds=timing,seconds=time.perf_counter()-started,consumed_development=True,
            outputs={q.name:sha(q) for q in out.iterdir() if q.is_file()})
        write(out/'receipt.json',receipt);print('PASS',dict(frames=new_frames,seconds=receipt['seconds'],routing=routing),flush=True)
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=bind.inputs));raise
    finally:
        if store is not None:store.close()
        models=spatial=heads=None;gc.collect()
        write(task/'handle-release.json',dict(status='PASS',compact_handles_closed=True,permanent_dense_cache=False,process_exit_required=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.task.resolve())
