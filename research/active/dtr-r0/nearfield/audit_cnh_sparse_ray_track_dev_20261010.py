"""Focused independent scalar and causality audit of fixed sparse ray readout.

Only declared consumed observations, public poses/gates and scientific scores.
No authored boxes, labels, new photon draws, model forward or GPU computation.
"""
import argparse
import ast
import json
from pathlib import Path
import time
import traceback

import numpy as np
import cnh_sparse_ray_track_dev_20261010 as M


def native_point(index):
    zone,radial=divmod(int(index),16);row,col=divmod(zone,8)
    edge=np.tan(np.pi/8)
    ray=np.array([-edge+(col+.5)*2*edge/8,-edge+(row+.5)*2*edge/8,1.])
    return ray/np.linalg.norm(ray)*(radial+.5)*float(M.WIDTH)


def history_kernel(index,now,past,sensor,aligned):
    """Independent scalar coordinate chain, with explicit row/column axes."""
    if aligned:
        anchor=native_point(index)
        world=sensor[now,:3,:3]@anchor+sensor[now,:3,3]
        local=sensor[past,:3,:3].T@(world-sensor[past,:3,3])
        if local[2]<=0:return [],[], -1
        edge=np.tan(np.pi/8)
        xy=np.floor((local[:2]/local[2]+edge)/(2*edge)*8).astype(int)
        if np.any(xy<0) or np.any(xy>=8):return [],[],-1
        zone=int(xy[1]*8+xy[0]);radial=int(np.floor(np.linalg.norm(local)/float(M.WIDTH)))
    else:
        zone,radial=divmod(int(index),16)
    cells=[];weights=[]
    for offset,weight in zip((-1,0,1),(.25,.5,.25)):
        b=radial+offset
        if 0<=b<16:cells.append(zone*16+b);weights.append(weight)
    center=zone*16+radial if 0<=radial<16 else -1
    return cells,weights,center


def scalar_slot(z,sensor,membership,inner_share,frame,query,arm):
    """Enumerate all current peaks and declared windows in scalar order."""
    current=z[frame].reshape(64,16)
    gated=np.maximum((np.sign(current)*np.log1p(np.abs(current)))*membership.reshape(64,16),0)
    candidates=[]
    for zone in range(64):
        for radial in range(16):
            value=gated[zone,radial]
            left=gated[zone,radial-1] if radial>0 else -np.inf
            right=gated[zone,radial+1] if radial<15 else -np.inf
            if value>0 and value>left and value>=right:candidates.append(zone*16+radial)
    windows=(1,) if arm=='current_only' else (4,) if arm=='aligned_fixed4' else (1,2,4)
    winner=None
    for index in candidates:
        local_best=None
        for requested in windows:
            begin=max(3,frame-requested+1)
            perframe=[];variance=[];path=[];positive=0
            for h in range(begin,frame+1):
                cells,weights,center=history_kernel(index,frame,h,sensor,arm!='unaligned_adaptive')
                weights=np.asarray(weights,np.float32)
                weighted=np.sum(z[h,cells]*weights,dtype=np.float32) if len(cells) else np.float32(0)
                var=np.sum(weights*weights,dtype=np.float32)
                perframe.append(weighted);variance.append(var);path.append(center)
                positive+=int(weighted>0 and var>0)
            total=np.sum(perframe,dtype=np.float32);v=np.sum(variance,dtype=np.float32)
            score=float(total/np.sqrt(v)*membership[index]) if v>0 else 0.
            result=dict(score=score,index=index,window=frame-begin+1,positive=positive,
                membership=float(membership[index]),inner_share=float(inner_share[index]),
                path=[-1]*(4-len(path))+path)
            if local_best is None or result['score']>local_best['score']:local_best=result
        if winner is None or local_best['score']>winner['score']:winner=local_best
    if winner is None:winner=dict(score=np.nan,index=-1,window=0,positive=0,membership=0.,inner_share=0.,path=[-1]*4)
    winner['peak_count']=len(candidates)
    return winner


def synthetic_alignment():
    """A known stationary point with independently authored moving range bins."""
    sensor=np.repeat(np.eye(4)[None],16,axis=0)
    sensor[:,2,3]=(np.arange(16)-3)*.60
    query=np.repeat(np.eye(4)[None],16,axis=0)
    maps=M.reprojection(sensor,query)
    z=np.zeros((1,16,1024),np.float32)
    end=6;anchor=(3*8+3)*16+3
    world=sensor[end,:3,:3]@native_point(anchor)+sensor[end,:3,3]
    bins=[]
    for h in range(3,end+1):
        local=sensor[h,:3,:3].T@(world-sensor[h,:3,3])
        edge=np.tan(np.pi/8);xy=np.floor((local[:2]/local[2]+edge)/(2*edge)*8).astype(int)
        radial=int(np.floor(np.linalg.norm(local)/float(M.WIDTH)))
        index=int((xy[1]*8+xy[0])*16+radial);z[0,h,index]=8;bins.append(index)
    geometry=dict(membership=np.ones((13,2,1024),np.float32),inner_share=np.ones((13,2,1024),np.float32))
    result=M.extract(z,geometry,maps)
    values={arm:float(result['scores'][i,0,end-3,0]) for i,arm in enumerate(M.ARMS)}
    if not values['aligned_adaptive']>values['unaligned_adaptive']*1.8:
        raise AssertionError('Known moving point did not benefit from correct ray association')
    if values['unaligned_adaptive']!=values['current_only']:
        raise AssertionError('Native unaligned control should select current exposure here')
    np.testing.assert_allclose(values['aligned_adaptive']/values['current_only'],2.,atol=1e-6)
    np.testing.assert_array_equal(result['anchor_history_native_center_index'][0,0,end-3,0],bins)
    return dict(status='PASS',authored_native_indices=bins,scores=values,
        aligned_over_current_ratio=values['aligned_adaptive']/values['current_only'],
        purpose='Engineering sign/direction check, deterministic observation fixture, no scientific photon generation')


def runtime_boundary():
    path=Path(M.__file__);tree=ast.parse(path.read_text(encoding='utf-8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    literal_strings={n.value for n in ast.walk(run) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    prohibited={'category','frame_category','target_box','background_boxes','boxes','hold','physical_key'}
    accessed=sorted(literal_strings & prohibited)
    if accessed:raise AssertionError('Evaluator fields unexpectedly accessed by runtime '+repr(accessed))
    # Source boundary is also enforced by the runtime source/input manifest.
    return dict(status='PASS',forbidden_evaluator_field_literals=accessed,
        reviewed_semantics='All-bin current peaks; signed z integration; current public query gate; pose-only backprojection; history begins f3; no posterior truth, fit or model forward')


def audit(out,cap):
    began=time.monotonic();previous=sum(M.read(p)['seconds'] for p in out.glob('audit_failure_*.json'))
    def check():
        if previous+time.monotonic()-began>=cap:raise TimeoutError('Cumulative focused audit cap reached')
    if (out/'audit_receipt.json').exists():raise FileExistsError('Preserve completed audit')
    phase='receipt'
    try:
        receipt=M.read(out/'scientific_receipt.json')
        if receipt['status']!='COMPLETE':raise ValueError('Scientific run must complete first')
        if receipt['source_sha256']!=M.C.sha(Path(M.__file__)):raise ValueError('Runtime source drift')
        original_compute_sha=receipt.get('original_compute_source_sha256',receipt['source_sha256'])
        snapshot=receipt.get('original_compute_source_snapshot')
        if snapshot:
            original=M.ROOT/snapshot
            if M.C.sha(original)!=original_compute_sha:raise ValueError('Original compute snapshot drift')
            names={'native_centers','reprojection','extract','FRAMES','WINDOW','KERNEL','OFFSETS','ARMS','PREFIX_WINDOWS','WIDTH','EDGE'}
            def numerical_AST(path):
                result=[]
                for node in ast.parse(path.read_text(encoding='utf-8')).body:
                    if isinstance(node,ast.FunctionDef) and node.name in names:result.append(ast.dump(node))
                    elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):result.append(ast.dump(node))
                return result
            if numerical_AST(original)!=numerical_AST(Path(M.__file__)):
                raise ValueError('Receipt repair changed numerical mechanism')
        for path,digest in receipt['input_sha256'].items():
            if (M.ROOT/path).resolve()==Path(M.__file__).resolve() and digest==original_compute_sha and snapshot:
                continue
            if M.C.sha(M.ROOT/path)!=digest:raise ValueError('Scientific input drift '+path)
        for path,digest in receipt['outputs_sha256'].items():
            if M.C.sha(out/path)!=digest:raise ValueError('Scientific score/map drift '+path)
        boundary=runtime_boundary();synthetic=synthetic_alignment();bias=np.load(M.C.BIAS).astype(np.float32)
        scalar_rows=[];causality=[];normalizer=[];max_score_error=0.;maps_checked=0
        for split in ('cal','validation'):
            check();phase=split;source=M.SOURCE/'data'/split
            with np.load(source/'geometry.npz') as a:sensor=a['sensor'];query=a['public_query']
            with np.load(source/'physics.npz') as a:ambient=a['ambient']
            with np.load(out/f'{split}_scores.npz') as a:saved={key:a[key] for key in a.files}
            geometry=M.P.native_geometry(sensor,query);maps=M.reprojection(sensor,query)
            hist=np.load(source/'hist.npy',mmap_mode='r',allow_pickle=False)
            # Two spread-out observations; three slots span warmup/full history/
            # timely endpoint. Every arm, both public height queries are checked.
            for scene,replica in ((0,0),(383,3)):
                check();raw=np.array(hist[scene,replica],copy=True)
                expected_z=((raw.astype(np.float32)-bias)/np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))).astype(np.float16)
                z=M.C.normalize(raw,ambient,bias).astype(np.float32).reshape(16,1024)
                np.testing.assert_array_equal(z,expected_z.astype(np.float32).reshape(16,1024))
                if not np.any(z<0):raise AssertionError('Representative signed history has no negative bins')
                normalizer.append(dict(split=split,scene=scene,replica=replica,signed_negative_bins=int((z<0).sum()),FP16_exact=True))
                rebuilt=M.extract(z[None],geometry,maps)
                for key,value in rebuilt.items():
                    expected=saved[key][:,scene:scene+1,replica] if key!='current_radial_peak_count' else saved[key][scene:scene+1,replica]
                    np.testing.assert_array_equal(value,expected)
                for through in (3,7):
                    altered=z.copy();altered[:3]=100;altered[through+1:]=-100
                    altered_sensor=sensor.copy();altered_sensor[through+1:,:3,3]+=np.array([3.,-2.,1.])
                    altered_query=query.copy();altered_query[through+1:,:3,3]+=np.array([-1.,1.,.5])
                    altered_maps=M.reprojection(altered_sensor,altered_query)
                    altered_geometry=M.P.native_geometry(altered_sensor,altered_query)
                    changed=M.extract(altered[None],altered_geometry,altered_maps)
                    for key,value in changed.items():
                        sl=(slice(None),slice(None),slice(0,through-2)) if key!='current_radial_peak_count' else (slice(None),slice(0,through-2))
                        np.testing.assert_array_equal(value[sl],rebuilt[key][sl])
                    causality.append(dict(split=split,scene=scene,replica=replica,through=through,
                        future_hist_pose_query_and_pre_f3_changed=True,all_prefix_descriptors_exact=True))
                for frame in (3,7,13):
                    fj=frame-3
                    for q in (0,1):
                        for ai,arm in enumerate(M.ARMS):
                            check();r=scalar_slot(z,sensor,geometry['membership'][fj,q],geometry['inner_share'][fj,q],frame,q,arm)
                            actual=float(saved['scores'][ai,scene,replica,fj,q]);error=abs(actual-r['score']) if np.isfinite(actual) else 0.
                            np.testing.assert_allclose(actual,r['score'],atol=2e-5,rtol=2e-6,equal_nan=True)
                            if r['index']!=int(saved['anchor_native_index'][ai,scene,replica,fj,q]):
                                raise AssertionError('Independent scalar winning anchor mismatch')
                            for key,name in (('selected_window','window'),('history_positive_frames','positive'),('anchor_membership','membership'),('anchor_inner_share','inner_share')):
                                np.testing.assert_allclose(saved[key][ai,scene,replica,fj,q],r[name],atol=1e-7,rtol=0)
                            np.testing.assert_array_equal(saved['anchor_history_native_center_index'][ai,scene,replica,fj,q],r['path'])
                            np.testing.assert_array_equal(saved['current_radial_peak_count'][scene,replica,fj,q],r['peak_count'])
                            max_score_error=max(max_score_error,error);scalar_rows.append(dict(split=split,scene=scene,replica=replica,frame=frame,query=q,arm=arm,max_abs_error=error))
                # Independently verify mappings including radial endpoints and
                # a lateral point, not only the winning observation anchors.
                for frame in (3,7,13):
                    fj=frame-3
                    for index in (0,511,1023):
                        for slot,h in enumerate(maps['history_frames'][fj]):
                            if h<0:continue
                            cells,weights,_=history_kernel(index,frame,int(h),sensor,True)
                            actual_indices=maps['indices'][fj,slot,index][maps['valid'][fj,slot,index]]
                            actual_weights=maps['kernel_weights'][fj,slot,index][maps['valid'][fj,slot,index]]
                            np.testing.assert_array_equal(actual_indices,cells);np.testing.assert_array_equal(actual_weights,weights);maps_checked+=1
        check()
        result=dict(status='PASS',seconds=time.monotonic()-began,previous_failure_seconds=previous,
            scalar_slot_checks=len(scalar_rows),max_scalar_score_abs_error=max_score_error,
            independent_point_mapping_checks=maps_checked,prefix_checks=causality,
            normalizer_checks=normalizer,runtime_boundary=boundary,synthetic_alignment=synthetic,
            source_sha256=M.C.sha(Path(__file__)),scientific_source_sha256=receipt['source_sha256'],
            original_compute_source_sha256=original_compute_sha,
            repaired_receipt_helper_source_sha256=receipt['source_sha256'],
            numerical_AST_unchanged_during_receipt_repair=True if snapshot else None,
            no_new_photons=True,no_GPU=True,no_model_forward=True,no_protected_or_new_hold_access=True,
            limitations='Focused engineering verification; does not validate simulator physics, false-positive calibration transfer or real-world detection',
            scalar_checks=scalar_rows)
        M.save_new(out/'audit_receipt.json',result)
        print('SPARSE_AUDIT_PASS',len(scalar_rows),'score-error',max_score_error,'seconds',round(result['seconds'],3),flush=True)
    except BaseException as error:
        M.save_new(out/f'audit_failure_{time.time_ns()}.json',dict(error=repr(error),phase=phase,
            traceback=traceback.format_exc(),seconds=time.monotonic()-began))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=M.DEFAULT_OUT)
    parser.add_argument('--cap',type=float,default=300.)
    args=parser.parse_args();audit(args.out,args.cap)
