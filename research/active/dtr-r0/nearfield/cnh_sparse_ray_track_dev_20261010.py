"""Fixed causal ray-anchored sparse peak integration, consumed Development.

Current gated radial maxima define candidate bin-center points. Each candidate
is reprojected through observed poses into up to four observed frames, where a
fixed three-bin signed kernel is integrated. No labels, authored boxes, learned
model, noise draw, threshold selection or notification code enters this path.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

import cnh_graded_corridor_features_dev as C
import cnh_graded_peak_tracks_dev as P

ROOT=C.ROOT
SOURCE=C.SOURCE
DEFAULT_OUT=ROOT/'artifacts.local/work/cnh-sparse-ray-track-dev-20261010'
FRAMES=np.arange(3,16)
WINDOW=4
KERNEL=np.array([.25,.5,.25],np.float32)
OFFSETS=np.arange(-1,2)
ARMS=('aligned_adaptive','unaligned_adaptive','current_only','aligned_fixed4')
PREFIX_WINDOWS=(1,2,4)
WIDTH=float(C.WIDTH)
EDGE=float(np.tan(np.pi/8))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    if path.exists(): raise FileExistsError('Preserve existing output: '+str(path))
    C.save(path,value)


def logical_path(path):
    """Keep canonical artifact labels across the E: junction and F: target."""
    path=Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return 'artifacts.local/'+path.relative_to((ROOT/'artifacts.local').resolve()).as_posix()
        except ValueError:
            return str(path)


def native_centers():
    slopes=-EDGE+(np.arange(8)+.5)*(2*EDGE/8)
    yy,xx=np.meshgrid(slopes,slopes,indexing='ij')
    rays=np.stack((xx,yy,np.ones_like(xx)),-1).reshape(64,3)
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    return (rays[:,None]*(np.arange(16)+.5)[None,:,None]*WIDTH).reshape(1024,3)


def reprojection(sensor, query):
    """Observation-independent point maps; histories begin at observed f3."""
    centers=native_centers()
    maps=np.zeros((13,4,1024,3),np.int16)
    valid=np.zeros_like(maps,bool)
    weights=np.zeros_like(maps,np.float32)
    lengths=np.minimum(np.arange(13)+1,WINDOW).astype(np.int8)
    history_frames=np.full((13,4),-1,np.int8)
    world=np.empty((13,1024,3),np.float64)
    current_query=np.empty_like(world)
    for fj,f in enumerate(FRAMES):
        world[fj]=centers@sensor[f,:3,:3].T+sensor[f,:3,3]
        current_query[fj]=centers@query[f,:3,:3].T+query[f,:3,3]
        begin=max(3,int(f)-WINDOW+1)
        for slot,h in enumerate(range(begin,int(f)+1),start=WINDOW-(f-begin+1)):
            history_frames[fj,slot]=h
            inverse=np.linalg.inv(sensor[h])
            points=world[fj]@inverse[:3,:3].T+inverse[:3,3]
            radius=np.linalg.norm(points,axis=-1)
            slopes_xy=np.divide(points[:,:2],points[:,2,None],out=np.zeros((1024,2)),where=points[:,2,None]>0)
            ij=np.floor((slopes_xy+EDGE)/(2*EDGE)*8).astype(int)
            bins=np.floor(radius/WIDTH).astype(int)
            inside=(points[:,2]>0)&(ij>=0).all(1)&(ij<8).all(1)
            zone=ij[:,1].clip(0,7)*8+ij[:,0].clip(0,7)
            radial=bins[:,None]+OFFSETS
            eligible=inside[:,None]&(radial>=0)&(radial<16)
            maps[fj,slot]=(zone[:,None]*16+radial.clip(0,15)).astype(np.int16)
            valid[fj,slot]=eligible
            weights[fj,slot]=eligible*KERNEL
    # Current exposure maps each anchor back into its own central zone/bin.
    np.testing.assert_array_equal(maps[:,-1,:,1],np.broadcast_to(np.arange(1024),(13,1024)))
    np.testing.assert_array_equal(valid[:,-1,:,1],True)
    return dict(indices=maps,valid=valid,kernel_weights=weights,
                available_frames=lengths,history_frames=history_frames,
                anchor_world_xyz=world,anchor_query_xyz=current_query)


def extract(z, geometry, maps):
    """Return two max-over-peak scores and public diagnostic argmax metadata."""
    z=np.asarray(z,np.float32).reshape(-1,16,1024)
    count=len(z)
    scores=np.full((len(ARMS),count,13,2),np.nan,np.float32)
    anchors=np.full(scores.shape,-1,np.int16)
    anchor_membership=np.zeros_like(scores)
    anchor_inner_share=np.zeros_like(scores)
    history_positive_frames=np.zeros_like(scores,np.int8)
    selected_window=np.zeros_like(scores,np.int8)
    history_native_index=np.full((*scores.shape,WINDOW),-1,np.int16)
    anchor_count=np.zeros((count,13,2),np.int16)
    for fj,f in enumerate(FRAMES):
        current=z[:,f].reshape(count,64,16)
        signed=np.sign(current)*np.log1p(abs(current))
        gated=np.maximum(signed[:,None]*geometry['membership'][fj].reshape(1,2,64,16),0)
        left=np.concatenate((np.full((*gated.shape[:-1],1),-np.inf,np.float32),gated[...,:-1]),-1)
        right=np.concatenate((gated[...,1:],np.full((*gated.shape[:-1],1),-np.inf,np.float32)),-1)
        peaks=((gated>0)&(gated>left)&(gated>=right)).reshape(count,2,1024)
        anchor_count[:,fj]=peaks.sum(-1)
        aligned=np.zeros((count,WINDOW,1024),np.float32)
        unaligned=np.zeros_like(aligned)
        variance_aligned=np.zeros((WINDOW,1024),np.float32)
        variance_unaligned=np.zeros_like(variance_aligned)
        current_indices=maps['indices'][fj,-1]
        current_kernel=maps['kernel_weights'][fj,-1]
        for slot,h in enumerate(maps['history_frames'][fj]):
            if h<0: continue
            kernel=maps['kernel_weights'][fj,slot]
            values=z[:,h,maps['indices'][fj,slot]]
            aligned[:,slot]=(values*kernel[None]).sum(-1)
            unaligned[:,slot]=(z[:,h,current_indices]*current_kernel[None]).sum(-1)
            variance_aligned[slot]=(kernel*kernel).sum(-1)
            variance_unaligned[slot]=(current_kernel*current_kernel).sum(-1)
        def prefix_values(values,variance):
            result=[]; positive=[]
            for window in PREFIX_WINDOWS:
                numerator=values[:,-window:].sum(1)
                v=variance[-window:].sum(0)
                result.append(np.divide(numerator,np.sqrt(v)[None],out=np.zeros_like(numerator),where=v[None]>0))
                positive.append(((values[:,-window:]>0)&(variance[None,-window:]>0)).sum(1))
            return np.stack(result),np.stack(positive)
        av,ap=prefix_values(aligned,variance_aligned)
        uv,up=prefix_values(unaligned,variance_unaligned)
        aligned_choice=av.argmax(0); unaligned_choice=uv.argmax(0)
        av_best=np.take_along_axis(av,aligned_choice[None],axis=0)[0]
        uv_best=np.take_along_axis(uv,unaligned_choice[None],axis=0)[0]
        for ai,(value,choices,positive_windows) in enumerate((
                (av_best,aligned_choice,ap),(uv_best,unaligned_choice,up),
                (av[0],np.zeros_like(aligned_choice),ap),(av[2],np.full_like(aligned_choice,2),ap))):
            weighted=value[:,None]*geometry['membership'][fj][None]
            selected=np.where(peaks,weighted,-np.inf)
            index=selected.argmax(-1)
            valid=peaks.any(-1)
            score=np.take_along_axis(selected,index[...,None],axis=-1)[...,0]
            scores[ai,:,fj]=np.where(valid,score,np.nan)
            anchors[ai,:,fj]=np.where(valid,index,-1)
            anchor_membership[ai,:,fj]=np.where(valid,geometry['membership'][fj][np.arange(2)[None],index],0)
            anchor_inner_share[ai,:,fj]=np.where(valid,geometry['inner_share'][fj][np.arange(2)[None],index],0)
            picked_choice=np.take_along_axis(choices[:,None].repeat(2,1),index[...,None],-1)[...,0]
            actual_windows=np.minimum(np.asarray(PREFIX_WINDOWS)[picked_choice],maps['available_frames'][fj])
            selected_window[ai,:,fj]=np.where(valid,actual_windows,0)
            sample=np.arange(count)[:,None]; queries=np.arange(2)[None]
            history_positive_frames[ai,:,fj]=np.where(valid,positive_windows[picked_choice,sample,index],0)
            # A selected anchor/window plus the saved kernel geometry fully
            # specifies the signed evidence used. For the unaligned control the
            # same current native center is repeated at each observed exposure.
            centers=maps['indices'][fj,:,:,1].T if ai!=1 else np.broadcast_to(np.arange(1024)[:,None],(1024,WINDOW))
            center_valid=maps['valid'][fj,:,:,1].T if ai!=1 else np.ones((1024,WINDOW),bool)
            path=centers[index]
            slots=np.arange(WINDOW)[None,None]
            active=slots>=WINDOW-actual_windows[...,None]
            active&=(maps['history_frames'][fj]>=0)[None,None]
            active&=center_valid[index]&valid[...,None]
            history_native_index[ai,:,fj]=np.where(active,path,-1)
    return dict(scores=scores,anchor_native_index=anchors,anchor_membership=anchor_membership,
                anchor_inner_share=anchor_inner_share,history_positive_frames=history_positive_frames,
                selected_window=selected_window,anchor_history_native_center_index=history_native_index,
                current_radial_peak_count=anchor_count)


def run(out, cap, batch):
    out=out.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()): raise ValueError('Canonical artifacts.local required')
    if not (out/'PLAN.json').exists(): raise ValueError('Controller must declare PLAN first')
    if (out/'scientific_receipt.json').exists(): raise FileExistsError('Preserve completed run')
    began=time.monotonic(); previous=sum(read(p)['seconds'] for p in out.glob('scientific_failure_*.json'))
    def check():
        if previous+time.monotonic()-began>=cap: raise TimeoutError('Cumulative scientific cap reached')
    phase='inputs'
    paths=[Path(__file__),Path(C.__file__),Path(P.__file__),C.BIAS,out/'PLAN.json']
    try:
        plan=read(out/'PLAN.json')
        # Input hashes, if declared, are validated without opening evaluator truth.
        for path,digest in plan.get('scientific_input_sha256',{}).items():
            if C.sha(ROOT/path)!=digest: raise ValueError('Declared input drift: '+path)
        bias=np.load(C.BIAS).astype(np.float32)
        parities=[]; prefix=[]; outputs=[]
        for split in ('cal','validation'):
            check(); folder=SOURCE/'data'/split
            with np.load(folder/'geometry.npz',allow_pickle=False) as a:
                sensor,query,ids,uids=(a[name] for name in ('sensor','public_query','scene_ids','scene_uids'))
            with np.load(folder/'physics.npz',allow_pickle=False) as a:
                ambient=a['ambient']
                np.testing.assert_array_equal(sensor,a['sensor']); np.testing.assert_array_equal(query,a['public_query'])
            paths.extend([folder/'geometry.npz',folder/'physics.npz',folder/'hist.npy'])
            geom=P.native_geometry(sensor,query)
            maps=reprojection(sensor,query)
            # The public gate equals the retained current-query geometry.
            prior=C.public_geometry(sensor,query)
            np.testing.assert_array_equal(geom['membership'],prior['membership'][:,:,1,-1])
            parities.append(dict(split=split,current_public_gate_exact=True,self_projection_exact=True))
            geometry_path=out/f'{split}_public_geometry.npz'
            if geometry_path.exists(): raise FileExistsError('Preserve partial geometry')
            np.savez_compressed(geometry_path,**maps,membership=geom['membership'],inner_share=geom['inner_share'],
                sensor=sensor,public_query=query,frames=FRAMES,kernel=KERNEL,offsets=OFFSETS)
            outputs.append(geometry_path)
            hist=np.load(folder/'hist.npy',mmap_mode='r',allow_pickle=False)
            if hist.shape!=(384,4,16,8,8,16): raise ValueError('Only declared consumed Development expected')
            batches=[]
            for start in range(0,len(hist),batch):
                check(); phase=f'{split}/batch{start}'
                observed=np.array(hist[start:start+batch],copy=True)
                z=C.normalize(observed,ambient,bias).reshape(-1,16,8,8,16)
                result=extract(z,geom,maps)
                batches.append(result)
            merged={}
            for key in batches[0]:
                axis=1 if key!='current_radial_peak_count' else 0
                data=np.concatenate([b[key] for b in batches],axis=axis)
                merged[key]=data.reshape((len(ARMS),384,4,*data.shape[2:]) if axis==1 else (384,4,*data.shape[1:]))
            score_path=out/f'{split}_scores.npz'
            if score_path.exists(): raise FileExistsError('Preserve partial scores')
            np.savez_compressed(score_path,**merged,arm_names=np.array(ARMS),scene_ids=ids,scene_uids=uids,
                frames=FRAMES,replica=np.arange(4),queries=np.array(['HEAD','BODY']),
                window_frames=maps['available_frames'])
            outputs.append(score_path)
            # Future alterations cannot change an existing prefix. Deliberately
            # include warmup and a full four-exposure history boundary.
            for scene,k,through in ((0,0,3),(383,3,7)):
                z=C.normalize(np.array(hist[scene,k],copy=True),ambient,bias)[None]
                altered=z.copy(); altered[:,through+1:]=np.float16(-100)
                rebuilt=extract(altered,geom,maps)
                for key,value in rebuilt.items():
                    actual=value[:,:,:through-2] if key!='current_radial_peak_count' else value[:,:through-2]
                    expected=merged[key][:,scene:scene+1,k,:through-2] if key!='current_radial_peak_count' else merged[key][scene:scene+1,k,:through-2]
                    np.testing.assert_array_equal(actual,expected)
                prefix.append(dict(split=split,scene=scene,replica=k,through=through,exact=True))
            print('SPARSE_RAY_DONE',split,round(time.monotonic()-began,3),flush=True)
        check(); phase='receipt'
        hashes={logical_path(p):C.sha(p) for p in paths}
        receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,previous_failure_seconds=previous,
            CPU_cap_seconds=cap,GPU_seconds=0,input_sha256=hashes,outputs_sha256={p.name:C.sha(p) for p in outputs},
            source_sha256=C.sha(Path(__file__)),arms=ARMS,window=WINDOW,kernel=KERNEL.tolist(),offsets=OFFSETS.tolist(),
            current_public_gate_parity=parities,prefix_checks=prefix,observations_per_split=384*4*13*2,
            schema='scores[arm4,scene384,replica4,frame13,query2]; no anchors -> NaN and index-1; aligned_fixed4 descriptive only',
            mechanism='Current positive gated radial maxima over all native bins; central-ray bin-center world anchor; known observed poses backproject to native history; signed triangular radial kernel; sum/sqrt(sum kernel weights squared); max across available causal prefix windows1/2/4, current expanded membership multiplication, max across anchors',
            history='f3 is first observed sample; causal past<=4, left warmup1/2/3/4. No prior f0..2 or future input.',
            validity='Reprojection center points are evidence association hypotheses, not object IDs or target attribution; no motion estimate. Triangular bin-tolerance kernel is not calibrated physical pulse. Reference unit-variance score is not signal-conditional SNR; adaptive scale/anchor search must be cost-calibrated. Fixed4 is descriptive, not candidate selection.',
            detector_thresholds=0,training=0,model_forward=0,new_photons=0,protected_access=0,
            evaluator_fields_accessed=False,persistent_resources=0,command=[sys.executable,*sys.argv])
        save_new(out/'scientific_receipt.json',receipt)
        print('SPARSE_RAY_COMPLETE',round(time.monotonic()-began,3),flush=True)
    except BaseException as error:
        save_new(out/f'scientific_failure_{time.time_ns()}.json',dict(status='FAILED',phase=phase,
            error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began,previous_seconds=previous))
        raise


def finalize_existing(out,cap):
    """Recover receipt serialization without regenerating any readout arrays."""
    out=out.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()): raise ValueError('Canonical artifacts.local required')
    if (out/'scientific_receipt.json').exists(): raise FileExistsError('Preserve completed receipt')
    began=time.monotonic()
    failures=[(p,read(p)) for p in out.glob('scientific_failure_*.json')]
    if not failures or failures[-1][1]['phase']!='receipt': raise ValueError('Only receipt-stage failures may finalize existing arrays')
    previous=sum(value['seconds'] for _,value in failures)
    def check():
        if previous+time.monotonic()-began>=cap: raise TimeoutError('Cumulative scientific cap reached')
    plan=read(out/'PLAN.json')
    original=out/'source_before_receipt_repair.py'
    source_path=logical_path(Path(__file__))
    original_sha=plan['scientific_input_sha256'][source_path]
    if C.sha(original)!=original_sha: raise ValueError('Preserved compute source does not match frozen PLAN')
    before=ast.parse(original.read_text(encoding='utf8'))
    after=ast.parse(Path(__file__).read_text(encoding='utf8'))
    numerical_names={'native_centers','reprojection','extract'}
    configuration_names={'FRAMES','WINDOW','KERNEL','OFFSETS','ARMS','PREFIX_WINDOWS','WIDTH','EDGE'}
    def numerical_contract(module):
        selected=[]
        for node in module.body:
            if isinstance(node,ast.FunctionDef) and node.name in numerical_names: selected.append(ast.dump(node))
            elif isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in configuration_names for target in node.targets): selected.append(ast.dump(node))
        return selected
    if numerical_contract(before)!=numerical_contract(after): raise ValueError('Numerical mechanism/configuration changed during receipt repair')
    input_hashes={}
    for name,digest in plan['scientific_input_sha256'].items():
        if name==source_path:
            input_hashes[name]=digest
            continue
        if C.sha(ROOT/name)!=digest: raise ValueError('Frozen input drift: '+name)
        input_hashes[name]=digest
    outputs=[]; parities=[]; prefix=[]
    bias=np.load(C.BIAS).astype(np.float32)
    for split in ('cal','validation'):
        check(); folder=SOURCE/'data'/split
        with np.load(folder/'geometry.npz') as a:
            sensor,query,ids,uids=(a[name] for name in ('sensor','public_query','scene_ids','scene_uids'))
        with np.load(folder/'physics.npz') as a: ambient=a['ambient']
        geom=P.native_geometry(sensor,query); maps=reprojection(sensor,query)
        public_path=out/f'{split}_public_geometry.npz'
        with np.load(public_path) as a:
            for name,value in maps.items(): np.testing.assert_array_equal(value,a[name])
            np.testing.assert_array_equal(a['membership'],geom['membership'])
            np.testing.assert_array_equal(a['inner_share'],geom['inner_share'])
            np.testing.assert_array_equal(a['sensor'],sensor); np.testing.assert_array_equal(a['public_query'],query)
        prior=C.public_geometry(sensor,query)
        np.testing.assert_array_equal(geom['membership'],prior['membership'][:,:,1,-1])
        parities.append(dict(split=split,current_public_gate_exact=True,self_projection_exact=True,saved_geometry_exact=True))
        hist=np.load(folder/'hist.npy',mmap_mode='r')
        score_path=out/f'{split}_scores.npz'
        with np.load(score_path) as a:
            np.testing.assert_array_equal(a['arm_names'],ARMS)
            np.testing.assert_array_equal(a['scene_ids'],ids); np.testing.assert_array_equal(a['scene_uids'],uids)
            np.testing.assert_array_equal(a['frames'],FRAMES)
            if a['scores'].shape!=(len(ARMS),384,4,13,2): raise ValueError('Score schema changed')
            np.testing.assert_array_equal(np.isnan(a['scores']),a['anchor_native_index']<0)
            for scene,k,through in ((0,0,3),(383,3,7)):
                z=C.normalize(np.array(hist[scene,k],copy=True),ambient,bias)[None]
                altered=z.copy(); altered[:,through+1:]=np.float16(-100)
                rebuilt=extract(altered,geom,maps)
                for name,value in rebuilt.items():
                    actual=value[:,:,:through-2] if name!='current_radial_peak_count' else value[:,:through-2]
                    expected=a[name][:,scene:scene+1,k,:through-2] if name!='current_radial_peak_count' else a[name][scene:scene+1,k,:through-2]
                    np.testing.assert_array_equal(actual,expected)
                prefix.append(dict(split=split,scene=scene,replica=k,through=through,exact=True))
        outputs.extend([public_path,score_path])
    check()
    finalized=time.monotonic()-began
    receipt=dict(status='COMPLETE',seconds=previous+finalized,original_compute_seconds=previous,
        finalize_only_seconds=finalized,CPU_cap_seconds=cap,GPU_seconds=0,input_sha256=input_hashes,
        outputs_sha256={p.name:C.sha(p) for p in outputs},
        source_sha256=C.sha(Path(__file__)),original_compute_source_sha256=original_sha,
        repaired_receipt_helper_source_sha256=C.sha(Path(__file__)),numerical_functions_and_configuration_AST_equal=True,
        original_compute_source_snapshot=logical_path(original),arms=ARMS,window=WINDOW,
        prefix_windows=PREFIX_WINDOWS,kernel=KERNEL.tolist(),offsets=OFFSETS.tolist(),
        current_public_gate_parity=parities,prefix_checks=prefix,observations_per_split=384*4*13*2,
        schema='scores[arm4,scene384,replica4,frame13,query2]; aligned_fixed4 descriptor only; NaN iff anchor index-1',
        mechanism=plan['runtime_contract'],history='Reset at observed f3, no earlier/future exposure',
        validity='Bin-center association hypotheses; triangular bin-tolerance kernel is not calibrated physical pulse; reference-noise norm is not conditional SNR; no object identity or target attribution.',
        failure_lineage=[dict(path=logical_path(p),sha256=C.sha(p),seconds=value['seconds'],phase=value['phase']) for p,value in failures],
        repair='Canonical F: junction resolved path serialized as logical artifacts.local; completed scores unchanged, no full-cohort rerun',
        training=0,model_forward=0,new_photons=0,protected_access=0,evaluator_fields_accessed=False,persistent_resources=0,
        command=[sys.executable,*sys.argv])
    save_new(out/'scientific_receipt.json',receipt)
    print('SPARSE_RAY_RECEIPT_RECOVERED',round(finalized,3),'cumulative',round(previous+finalized,3),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--cap',type=float,default=900.)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--finalize-existing',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.finalize_existing: finalize_existing(args.out,args.cap)
        else: run(args.out,args.cap,args.batch)
