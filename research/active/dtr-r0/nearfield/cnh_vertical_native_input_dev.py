"""Reconstruct selected controlled native M3 inputs; no network forward."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-vertical-native-input-dev-20261009'
OLD=WORK/'cnh-vertical-controlled-pair-dev-20261009'
SHAPE=WORK/'cnh-aligned-shapes-dev-20261008'

def save(name,x):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def csvsave(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    t=time.monotonic();p=json.loads((OLD/'PLAN.json').read_text());scene=json.loads((SHAPE/'PLAN.json').read_text())['scene_rows']
    pairs=[r for r in p['pairs'] if r['angle']==3 and r['depth']=='in_1cm' and r['height']=='BODY' and scene[r['contact_scene']]['group']=='BOTH' and scene[r['contact_scene']]['rho']==.65]
    assert len(pairs)==4
    ids=sorted({s for r in pairs for s in (r['clear_scene'],r['contact_scene'])});assert len(ids)==6
    inputs=[OLD/'PLAN.json',OLD/'coupled_physical.npz',OLD/'responses.npz',SHAPE/'PLAN.json',SHAPE/'geometry.npz']
    save('PLAN.json',dict(task='CNH_VERTICAL_NATIVE_INPUT_DEV_20261009',lane='EXPLORE selected consumed simulated Development',base_commit='b87b0010',
        authorization='User 推进 preceding proposal to reconstruct selected complete controlled features and inspect input/raw response correspondence',
        budgets_stage_wall_seconds=dict(cpu_prepare_analysis_checks=120,gpu_projection_native_transform_IO=120),preliminary_cpu_command_seconds=8.8796546,
        scene_ids=ids,pairs=pairs,K=4,angle=3,frames=list(range(6,14)),feature_examples=192,
        goal='Distinguish coarse BODY support from actual signedlog native representation and spatial/height layout; no policy/readout benefit claim',
        H1='Positive BODY current/summed raw support contrasts may reverse after native signedlog; compare exactly both representations rather than density proxy alone',
        H2='Contact positive support may be more concentrated than mirror clear and differences outside BODY may remain; inspect public partition and participation proxies, not target attribution or network importance',
        metrics='Raw tot/current and actual FP32 signedlog tot/current: signed/positive/negative sum, positive weight, weighted positive participation and positive peak. Regions fixed before results: x4 public bands, HEAD/BODY/remaining y, before/query/beyond z (36 additive cells), plus original public HEAD/BODY/complement diagnostics.',
        contract='All region fractions in original public query coordinates, not target side alignment. cnt and querymask channels identical across arms. Model receives entire5channel cube; no new CNN/classifier forward, original raw values only. Conditional shared Poisson coupling and target occlusion kept.',
        decision_check='4 relations/width conditions x8frames xK4=128 paired physical windows, 64near/64earlier; grid decompositions are repeated views. No numeric success cutoff, dprime/performance bound or threshold tuning. If native sum still positive with lower raw, investigate richer layout/learned response rather than claim compression explains it.',
        adjustable_scope='Implementation repair and focused checks only; no extra angle/frame/scene/draw/gate/feature ablation/model inference',
        stop='Complete fixed projection, spatial comparisons and focused checks or stage120s cumulative including failures. Prior policies and stop recipes frozen.',
        inputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},source_sha256=sha(Path(__file__)),prepare_seconds=time.monotonic()-t))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('PREPARED',ids,'192 features',flush=True)

def run():
    t=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());engine=None
    previous=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('gpu_failure_*.json'))
    def check():
        if previous+time.monotonic()-t>120:raise TimeoutError('cumulative GPU stage120s')
    try:
        for path,h in p['inputs_sha256'].items():assert sha(ROOT/path)==h,path
        repairs=sorted(OUT.glob('implementation_repair*.json'))
        expected=json.loads(repairs[-1].read_text())['new_source_sha256'] if repairs else p['source_sha256']
        assert sha(Path(__file__))==expected
        import cnh_aligned_boundary_dev as B
        import cnh_bar_local_readout_dev as L
        import cnh_bar_pose_transfer_dev as T
        A,_,_,normal=B.imports();A.OUT=OUT/'runtime';A.setup_gpu()
        import torch
        from cnh_cvr_v2_materialize import BatchedProjector
        from cnh_cvr_projection import SHAPE as GRID_SHAPE,query_masks,grid
        from cnh_temporal_readout_model import prepare_voxels
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        engine=SimpleNamespace(torch=torch,projector=BatchedProjector())
        with np.load(OLD/'coupled_physical.npz') as z:
            original_ids=z['scene_ids'];ix={int(s):i for i,s in enumerate(original_ids)};hist=z['hist'][[ix[s] for s in p['scene_ids']]];ambient=z['ambient']
        with np.load(SHAPE/'geometry.npz') as z:sensor=z['sensor'];q=z['public_query']
        with np.load(OLD/'responses.npz') as z:oldv=z['values'][1,[ix[s] for s in p['scene_ids']]];oldraw=z['raw'][1,[ix[s] for s in p['scene_ids']]]
        z=normal(hist.reshape(24,16,8,8,16),np.broadcast_to(ambient,(24,*ambient.shape)))
        masks=query_masks();inner=masks.astype(float);outer=np.zeros_like(inner);yz=inner.sum(1)/12;outer[:,:6]=yz[:,None];outer[:,18:]=yz[:,None]
        mm=np.stack([inner[0],outer[0],inner[1],outer[1]]).reshape(4,-1)
        features=np.empty((6,4,8,3,*GRID_SHAPE),np.float16);native=np.empty((6,4,8,5,*GRID_SHAPE),np.float32)
        maximum=0.;parity=[]
        for j,f in enumerate(p['frames']):
            check();estimated,goal=T.injected(sensor,q,dict(kind='query',degrees=3,sign=1));start=max(0,f-7)
            m=(goal[f]@np.linalg.inv(estimated[f]))[None]@estimated[start:f+1];cp=L.Projection(engine,m)
            xx=cp(z[:,start:f+1]).cpu().numpy().astype(np.float16)
            if j in (0,7):
                ref=A.Engine.project(engine,z[:1,start:f+1],m[None]).cpu().numpy().astype(np.float16)
                np.testing.assert_array_equal(xx[:1],ref);parity.append(dict(frame=f,first_record_FP16_bitwise=True))
            features[:,:,j]=xx.reshape(6,4,3,*GRID_SHAPE)
            native[:,:,j]=prepare_voxels(torch.from_numpy(xx).cuda(),torch.as_tensor(masks,device='cuda')).cpu().numpy().reshape(6,4,5,*GRID_SHAPE)
            total=xx[:,0].astype(float).reshape(24,-1);current=xx[:,2].astype(float).reshape(24,-1)
            coverage=mm@xx[0,1].astype(float).ravel();ts=total@mm.T;cs=current@mm.T
            ha=np.abs(total-current)@mm.T;ca=np.abs(current)@mm.T;pos=np.maximum(current,0)@mm.T
            for qi in (0,1):
                i,o=2*qi,2*qi+1;values=np.stack([ha[:,i]/(ha[:,i]+ca[:,i]),pos[:,i]/(pos[:,i]+pos[:,o]),ts[:,i]/coverage[i],cs[:,i]/coverage[i]],1).reshape(6,4,4)
                old=oldv[:,:,f-3,qi];diff=float(np.max(np.abs(values-old)));maximum=max(maximum,diff)
                np.testing.assert_array_equal(values,old)
            del cp,xx,total,current
        check();centers,_=grid()
        np.savez_compressed(OUT/'native_inputs.npz',scene_ids=p['scene_ids'],frames=p['frames'],feature_fp16=features,native_fp32=native,query_masks=masks,centers=centers,original_raw=oldraw[:,:,3:11])
        save('run_result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,device=torch.cuda.get_device_name(),feature_examples=192,reconstructed_old_4metric_values=1536,max_metric_difference=maximum,reference_projection_checks=parity,new_projection=True,new_inference=0,new_samples=0,training=0))
        print('RECONSTRUCTED192 native inputs, 1536 oldmetric bitwise',round(time.monotonic()-t,3),flush=True)
    except BaseException as e:save('gpu_failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise
    finally:
        if engine is not None:engine.projector=None;engine.torch.cuda.empty_cache()

def regions(masks,centers):
    def frac(axis,lo,hi,step):
        c=centers[...,axis];return np.maximum(np.minimum(c+step/2,hi)-np.maximum(c-step/2,lo),0)/step
    xs=[frac(0,a,b,.05) for a,b in ((-.6,-.3),(-.3,0),(0,.3),(.3,.6))]
    ys=[frac(1,-.2,.42,.1),frac(1,.42,.9,.1)];ys.append(1-ys[0]-ys[1])
    zs=[frac(2,0,.3,.1),frac(2,.3,3.,.1),frac(2,3.,3.3,.1)]
    grid=[(f'x{x}_y{y}_z{z}',a*b*c) for x,a in enumerate(xs) for y,b in enumerate(ys) for z,c in enumerate(zs)]
    np.testing.assert_allclose(sum(m for n,m in grid),1,atol=1e-12,rtol=0)
    query=[('public_HEAD',masks[0]),('public_BODY',masks[1]),('public_complement',1-masks.sum(0))]
    return grid+query

def measure(x,mask):
    p=np.maximum(x,0);positive=float(np.sum(mask*p));second=float(np.sum(mask*p*p))
    return dict(signed_sum=float(np.sum(mask*x)),positive_sum=positive,negative_abs_sum=float(np.sum(mask*np.maximum(-x,0))),positive_weight=float(np.sum(mask*(x>0))),positive_participation=positive*positive/second if second>0 else None,positive_peak=float(p[mask>0].max()))

def analyze():
    t=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());rows=[]
    with np.load(OUT/'native_inputs.npz') as z:ids=z['scene_ids'];f=z['feature_fp16'];native=z['native_fp32'];masks=z['query_masks'];centers=z['centers'];raw=z['original_raw']
    index={int(s):i for i,s in enumerate(ids)};parts=regions(masks,centers)
    for pair in p['pairs']:
        ci,pi=index[pair['clear_scene']],index[pair['contact_scene']]
        width=4 if pair['contact_scene']==65 else 10
        for j,frame in enumerate(p['frames']):
            for k in range(4):
                for name,mask in parts:
                    row=dict(width_cm=width,relation=pair['relation'],clear_scene=pair['clear_scene'],contact_scene=pair['contact_scene'],frame=frame,replica=k,region=name,region_weight=float(mask.sum()),original_raw_BODY_delta=float(raw[pi,k,j,1]-raw[ci,k,j,1]))
                    for channel,representation in ((0,'raw_tot'),(2,'raw_current'),(0,'native_log_tot'),(2,'native_log_current')):
                        source=f if representation.startswith('raw') else native
                        clear=measure(source[ci,k,j,channel].astype(float),mask);contact=measure(source[pi,k,j,channel].astype(float),mask)
                        for metric in clear:
                            row['clear_'+representation+'_'+metric]=clear[metric];row['contact_'+representation+'_'+metric]=contact[metric]
                            row['delta_'+representation+'_'+metric]=contact[metric]-clear[metric] if clear[metric] is not None and contact[metric] is not None else None
                    rows.append(row)
    csvsave('paired_region_metrics.csv',rows)
    summary=[]
    for width in (4,10):
        for relation in ('same_side','mirrored_side'):
            for period in ('earlier','near'):
                for name,_ in parts:
                    rr=[r for r in rows if r['width_cm']==width and r['relation']==relation and r['region']==name and ((r['frame']>=10)==(period=='near'))]
                    x=dict(width_cm=width,relation=relation,period=period,region=name,paired_windows=len(rr),original_raw_BODY_delta_mean=float(np.mean([r['original_raw_BODY_delta'] for r in rr])))
                    for key in [k for k in rr[0] if k.startswith('delta_')]:
                        v=np.array([r[key] for r in rr if r[key] is not None],float)
                        x[key+'_finite']=len(v);x[key+'_mean']=float(v.mean()) if len(v) else None;x[key+'_negative']=int((v<0).sum());x[key+'_positive']=int((v>0).sum())
                    summary.append(x)
    csvsave('region_summary.csv',summary)
    spent=p['preliminary_cpu_command_seconds']+p['prepare_seconds']+time.monotonic()-t
    if spent>95:raise TimeoutError('CPU120s reserve25s audit')
    save('analysis_result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,paired_physical_windows=128,paired_region_rows=len(rows),region_summary_rows=len(summary),partition_cells=36,additional_overlapping_public_regions=3,new_inference=0,limits='Weighted support/participation summaries, not actual CNN features or network causal contribution; raw outputs inherited, no candidate or costmatched gain'))
    print('ANALYZED128 pairs/4992 region rows',round(time.monotonic()-t,3),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
