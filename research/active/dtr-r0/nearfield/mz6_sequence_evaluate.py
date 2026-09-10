"""Frozen MZ5 short-sequence pilot: separate synthetic sensor, inference and scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
import time
import shutil

import cv2
import numpy as np
from PIL import Image
import torch

from mz5_fixed_ensemble import EVENTS, load, metric, read, sha, write
from mz6_causal_packet import CausalPacket, mean3, HZ


def prepare(root, capture, output):
    from body_query_context_evidence import ContextEvidence
    from body_query_fresh_size_eval import FROZEN
    from multizone64_observation import observe, native_events, geometry
    work=root/'artifacts.local/work'
    manifest=read(capture/'model/sensor_manifest.json')
    assert set(manifest)=={'calibration','frames'}
    assert manifest['calibration']['eye_height_m']==1.7
    assert manifest['calibration']['pitch_degrees']==0
    frames=manifest['frames']; n=len(frames)
    assert n==200 and len(set(f['clip_id'] for f in frames))==8
    assert all(set(f)=={'sample_index','clip_id','rgb_path','nominal_time_s'} for f in frames)
    receipt=read(capture/'receipt.json')
    assert receipt['status']=='PASS' and receipt['source_unchanged']
    assert read(capture/'source-admission.json')['status']=='PASS'
    assert read(capture/'process-release.json')['released']
    base=work/'body-query-10000-b-20260909/run-v1'
    decoder=work/'body-query-context-decoder-20260909/run-v1'
    hashes={name:sha((base if name=='NEW-step2000.pt' else decoder)/name) for name in FROZEN}
    assert hashes==FROZEN
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    source_folder=output/'source';source_folder.mkdir()
    source_paths=[Path(__file__),Path(__file__).with_name('mz6_causal_packet.py'),
        Path(__file__).with_name('MZ6_SHORT_SEQUENCE_PROTOCOL_20260910.md'),
        Path(__file__).with_name('multizone64_observation.py'),
        Path(__file__).with_name('mz5_ensemble_readout.py')]
    source_hashes={p.name:sha(p) for p in source_paths}
    for path in source_paths:shutil.copyfile(path,source_folder/path.name)
    model=ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
    captured=[]
    hook=model.decoder.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    visual=[];alerts=[];ranges=[];valid=[];counts=[];crop_counts=[];unknown=[];grays=[];small_fraction=[]
    input_hashes={}
    g=geometry('cuda',8)
    # This loop is the simulation/extraction boundary. Native data is never sent to an RGB model.
    try:
        with torch.inference_mode():
            for begin in range(0,n,16):
                images=[];depths=[]
                for row in frames[begin:begin+16]:
                    rgb_path=capture/'model'/row['rgb_path']
                    native_path=capture/f"evaluator/native/{row['sample_index']:04d}.npy"
                    input_hashes[str(rgb_path)]=sha(rgb_path);input_hashes[str(native_path)]=sha(native_path)
                    with Image.open(rgb_path) as im:
                        rgb=im.convert('RGB')
                        images.append(np.array(rgb.resize((256,144),Image.Resampling.BOX)))
                        grays.append(cv2.cvtColor(np.array(rgb.resize((320,180),Image.Resampling.BOX)),cv2.COLOR_RGB2GRAY))
                    depths.append(np.load(native_path,allow_pickle=False))
                x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
                out=model(x); assert len(captured)==1
                visual.append(torch.cat((captured.pop().flatten(1),out['range_probabilities'].flatten(1)),1).cpu().numpy())
                alerts.append(out['alerts'].cpu().numpy())
                depth=torch.from_numpy(np.stack(depths)).cuda()
                packet=observe(depth,8,'multi_surface')
                ranges.append(torch.nan_to_num(packet['range_m'],nan=0).cpu().numpy())
                valid.append(packet['valid'].cpu().numpy())
                full=native_events(depth,False);crop=native_events(depth,True)
                counts.append(full['counts'].cpu().numpy());crop_counts.append(crop['counts'].cpu().numpy())
                unknown.append(((depth<=0)|~torch.isfinite(depth)).sum((1,2)).cpu().numpy())
                radial=depth.double()*g['radial_factor']
                known=torch.isfinite(depth)&(depth>0)&(depth<100)&(radial<=4)
                fractions=[]
                for z in range(64):
                    mask=g['zone_ids']==z
                    r=radial[:,mask];v=known[:,mask]
                    # First/last observed bin means determine occupied bins, no object-ID lookup.
                    first=packet['range_m'][:,z,0]
                    bin_id=torch.floor(torch.nan_to_num(first,nan=0)*10).clamp(0,39).long()
                    sample_bin=torch.floor(r*10).clamp(0,39).long()
                    member=v&(sample_bin==bin_id[:,None])
                    fractions.append((member.sum(1)/v.sum(1).clamp_min(1)).cpu().numpy())
                small_fraction.append(np.stack(fractions,1))
    finally:
        hook.remove()
    arrays=dict(visual=np.concatenate(visual),original_alerts=np.concatenate(alerts),
        ranges=np.concatenate(ranges),valid=np.concatenate(valid),gray=np.stack(grays),
        clip=np.array([r['clip_id'] for r in frames]),index=np.tile(np.arange(25),8),
        time_s=np.array([r['nominal_time_s'] for r in frames]))
    fraction=np.concatenate(small_fraction)
    eligible=arrays['valid'].all(2)&((arrays['ranges'][:,:,1]-arrays['ranges'][:,:,0])>=.30)&(fraction<=.15)
    deleted=eligible&(np.random.default_rng(107).random(eligible.shape)<.5)
    stressed=arrays['ranges'].copy();stressed_valid=arrays['valid'].copy()
    stressed[deleted,0]=stressed[deleted,1];stressed[deleted,1]=0
    stressed_valid[deleted,0]=True;stressed_valid[deleted,1]=False
    arrays.update(stress_ranges=stressed,stress_valid=stressed_valid)
    np.savez_compressed(output/'observations.npz',**arrays)
    np.savez_compressed(output/'evaluator.npz',truth=np.concatenate(counts)>=3,
        native_counts=np.concatenate(counts),crop_counts=np.concatenate(crop_counts),
        unknown_pixels=np.concatenate(unknown),eligible=eligible,deleted=deleted,
        first_bin_fraction=fraction)
    write(output/'prepare-receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-started,frozen_hashes=hashes,input_sha256=input_hashes,
        manifest_sha256=sha(capture/'model/sensor_manifest.json'),capture_receipt_sha256=sha(capture/'receipt.json'),
        source_admission_sha256=sha(capture/'source-admission.json'),
        source_sha256=source_hashes,
        observations_sha256=sha(output/'observations.npz'),evaluator_sha256=sha(output/'evaluator.npz'),
        eligible_zones=int(eligible.sum()),deleted_zones=int(deleted.sum()),training_steps=0))
    print('PREPARE PASS',n,'frames',int(eligible.sum()),'eligible',int(deleted.sum()),'deleted',flush=True)


def episodes(pred, truth, clips):
    records=[]
    # Raw event bits plus BODY/HEAD unions avoid mistaking far->near for alert clearance.
    pred=np.concatenate((pred,np.stack((pred[:,:2].any(1),pred[:,2:].any(1)),1)),1)
    truth=np.concatenate((truth,np.stack((truth[:,:2].any(1),truth[:,2:].any(1)),1)),1)
    names=(*EVENTS,'BODY_ANY','HEAD_ANY')
    for clip in dict.fromkeys(clips.tolist()):
        p,t=pred[clips==clip],truth[clips==clip]
        for k,name in enumerate(names):
            start=0
            while start<len(t):
                if not t[start,k]:
                    start+=1;continue
                end=start+1
                while end<len(t) and t[end,k]:end+=1
                hits=np.flatnonzero(p[start:end,k])
                next_positive=np.flatnonzero(t[end:,k])
                negative_end=end+int(next_positive[0]) if len(next_positive) else len(t)
                clear=next((j for j in range(end,negative_end-1) if not p[j,k] and not p[j+1,k]),None)
                records.append(dict(clip=clip,event=name,start=start,end_exclusive=end,
                    left_censored=start==0,positive_frames=end-start,hits=int(p[start:end,k].sum()),
                    first_hit=None if not len(hits) else start+int(hits[0]),
                    delay_frames=None if not len(hits) else int(hits[0]),
                    delay_nominal_s=None if not len(hits) else int(hits[0])/HZ,
                    clearance_evaluable=negative_end-end>=2,
                    clearance_delay_frames=None if clear is None else clear-end,
                    clearance_delay_nominal_s=None if clear is None else (clear-end)/HZ,
                    clearance_confirm_delay_frames=None if clear is None else clear+1-end,
                    clearance_confirm_delay_nominal_s=None if clear is None else (clear+1-end)/HZ,
                    clearance_status='RIGHT_CENSORED' if negative_end-end<2 else 'UNCLEARED' if clear is None else 'CLEARED'))
                start=end
    return records


def compare_episodes(candidate,baseline):
    assert len(candidate)==len(baseline)
    keys=['clip','event','start','end_exclusive']
    improved=delayed=clear_delayed=0
    for a,b in zip(candidate,baseline):
        assert all(a[k]==b[k] for k in keys)
        av=a['delay_frames'] if a['delay_frames'] is not None else float('inf')
        bv=b['delay_frames'] if b['delay_frames'] is not None else float('inf')
        improved+=av<bv;delayed+=av>bv
        if a['clearance_evaluable']:
            av=a['clearance_delay_frames'] if a['clearance_delay_frames'] is not None else float('inf')
            bv=b['clearance_delay_frames'] if b['clearance_delay_frames'] is not None else float('inf')
            clear_delayed+=av>bv
    return dict(earlier_or_recovered_episodes=improved,delayed_or_lost_episodes=delayed,clearance_worse_episodes=clear_delayed)


def score(root,output):
    from mz5_ensemble_readout import CompactEnsemble
    data=load(output/'observations.npz'); receipt=read(output/'prepare-receipt.json')
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in receipt['source_sha256'].items())
    assert sha(output/'observations.npz')==receipt['observations_sha256']
    compact=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1'
    checkpoint=compact/'compact.pt'
    export=read(compact/'export-receipt.json')
    # Record and compare the actual compact checkpoint against its completed parity receipt.
    parity=read(compact/'parity-receipt.json')
    digest=sha(checkpoint)
    assert digest == export['compact_sha256'] == parity['exported_weights_sha256']
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    model=CompactEnsemble.from_checkpoint(checkpoint).cuda().eval()
    visual=torch.from_numpy(data['visual']).cuda()
    def infer(ranges,valid):
        features=np.concatenate((ranges.astype(np.float32).reshape(len(ranges),128)/4,
                                  valid.astype(np.float32).reshape(len(ranges),128)),1)
        with torch.inference_mode():
            return model(visual,torch.from_numpy(features).cuda()).cpu().numpy()
    started=time.perf_counter(); predictions={}; completions={}; timings={}
    # Prediction API receives only RGB, calibration-derived zones and raw packets.
    for condition in ('clean','stress'):
        ranges=data['ranges' if condition=='clean' else 'stress_ranges']
        valid=data['valid' if condition=='clean' else 'stress_valid']
        logits=infer(ranges,valid)
        predictions[condition+'/CURRENT']=logits
        predictions[condition+'/MEAN3']=mean3(logits,data['clip'])
        for name,zero in [('COMPATIBLE',False),('ZERO_MOTION',True)]:
            clock=time.perf_counter();state=CausalPacket(zero)
            new_ranges=[];new_valid=[];changes=[]
            for i in range(len(ranges)):
                r,v,c=state.step(str(data['clip'][i]),int(data['index'][i]),data['gray'][i],ranges[i],valid[i])
                new_ranges.append(r);new_valid.append(v)
                changes.append(dict(row=i,clip=str(data['clip'][i]),completions=c))
            timings[condition+'/'+name]=time.perf_counter()-clock
            predictions[condition+'/'+name]=infer(np.stack(new_ranges),np.stack(new_valid))
            completions[condition+'/'+name]=changes
    np.savez_compressed(output/'predictions.npz',**predictions)
    write(output/'completion-hypotheses.json',completions)
    # Evaluator truth is loaded only after predictions have been written.
    assert sha(output/'evaluator.npz')==receipt['evaluator_sha256']
    evaluator=load(output/'evaluator.npz');truth=evaluator['truth']
    results={}; empty=np.array(['empty_target' in clip for clip in data['clip']])
    for name,logits in predictions.items():
        p=logits>=0
        results[name]=dict(metrics=metric(p,truth),episodes=episodes(p,truth,data['clip']),
            empty_target_control=dict(frames=int(empty.sum()),false_event_bits=int((p[empty]&~truth[empty]).sum()),
                false_event_frames=int((p[empty]&~truth[empty]).any(1).sum()),
                actual_positive_bits=int(truth[empty].sum())),
            by_clip={clip:metric(p[data['clip']==clip],truth[data['clip']==clip]) for clip in dict.fromkeys(data['clip'].tolist())})
    comparisons={}
    for condition in ('clean','stress'):
        baseline=predictions[condition+'/CURRENT']>=0
        for method in ('MEAN3','COMPATIBLE','ZERO_MOTION'):
            name=condition+'/'+method;candidate=predictions[name]>=0
            c,b=(candidate==truth).all(1),(baseline==truth).all(1)
            comparisons[name]=dict(exact_gains=int((c&~b).sum()),exact_losses=int((~c&b).sum()),
                far_tp_gains=int((truth[:,[1,3]]&candidate[:,[1,3]]&~baseline[:,[1,3]]).sum()),
                far_tp_losses=int((truth[:,[1,3]]&~candidate[:,[1,3]]&baseline[:,[1,3]]).sum()),
                **compare_episodes(results[name]['episodes'],results[condition+'/CURRENT']['episodes']))
    passed=[]
    for condition in ('clean','stress'):
        candidate=results[condition+'/COMPATIBLE'];baseline=results[condition+'/CURRENT']
        comp=comparisons[condition+'/COMPATIBLE']
        passed.append(comp['far_tp_losses']==0 and comp['delayed_or_lost_episodes']==0 and comp['clearance_worse_episodes']==0
            and candidate['empty_target_control']['false_event_bits']<=baseline['empty_target_control']['false_event_bits']
            and all(candidate['metrics'][k]['numerator']<=baseline['metrics'][k]['numerator'] for k in ('wrong_far','body_to_head','head_to_body')))
    c=comparisons['stress/COMPATIBLE']
    improvement=c['far_tp_gains']>0 or c['earlier_or_recovered_episodes']>0
    status=('NOT_EVALUABLE_PRESSURE' if not evaluator['deleted'].any() else
            'RETAIN_CANDIDATE_CONTROLLED' if all(passed) and improvement else 'RETAIN_CURRENT_NO_CANDIDATE_UPGRADE')
    result=dict(status=status,scope='Controlled posed Willow Development; raw visible-support metrics, not clearance safety',
        frames=len(truth),training_steps=0,source_unknown=dict(frames_with_invalid_native_pixels=int((evaluator['unknown_pixels']>0).sum()),
            total_invalid_native_pixels=int(evaluator['unknown_pixels'].sum()),
            semantics='Unknown rays are not CLEAR; false truth bits mean fewer than3 visible positive pixels only'),
        pressure=dict(eligible_zone_frames=int(evaluator['eligible'].sum()),deleted_zone_frames=int(evaluator['deleted'].sum()),
            by_clip={clip:dict(eligible=int(evaluator['eligible'][data['clip']==clip].sum()),deleted=int(evaluator['deleted'][data['clip']==clip].sum())) for clip in dict.fromkeys(data['clip'].tolist())}),
        arms=results,comparisons=comparisons,retention_checks=passed,required_gain=improvement,
        hypotheses={name:dict(completed_zone_frames=sum(len(r['completions']) for r in records),
            affected_frames=sum(bool(r['completions']) for r in records)) for name,records in completions.items()})
    write(output/'result.json',result)
    write(output/'score-receipt.json',dict(status='PASS',checkpoint_sha256=digest,observations_sha256=receipt['observations_sha256'],
        original_alerts_container_sha256=sha(output/'observations.npz'),original_alerts_modified=False,
        readout_backend='CUDA',device=torch.cuda.get_device_name(),correspondence_backend='CPU',
        correspondence_placement_reason='GPU_BACKEND_UNAVAILABLE',cv2_cuda_devices=cv2.cuda.getCudaEnabledDeviceCount(),
        correspondence_seconds=timings,total_seconds=time.perf_counter()-started,training_steps=0,
        output_sha256={name:sha(output/name) for name in ('predictions.npz','result.json','completion-hypotheses.json')}))
    print('SCORE',status,flush=True)
    print({name:dict(exact=x['metrics']['spatial_exact'],tp=[v['tp'] for v in x['metrics']['event_confusion'].values()],
        fp=[v['fp'] for v in x['metrics']['event_confusion'].values()]) for name,x in results.items()},flush=True)
    print(comparisons,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','score'],required=True)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    allowed=(root/'artifacts.local/work/mz6-short-sequence-20260910').resolve()
    assert output.is_relative_to(allowed) and output!=allowed
    if args.phase=='prepare':output.mkdir(parents=True,exist_ok=False)
    else:assert output.is_dir() and not (output/'predictions.npz').exists()
    try:
        if args.phase=='prepare':prepare(root,args.capture.resolve(),output)
        else:score(root,output)
    except Exception as exc:
        write(output/f'failure-{args.phase}.json',dict(status='FAILED',error=repr(exc),source_sha256=sha(__file__)))
        raise
