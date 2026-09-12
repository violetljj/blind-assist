"""One fixed causal near/far reprojection contradiction gate; no tuning."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import mz101_spatial as m
from mz105_residual_matching import membership
from run_mz101_spatial import sha, metrics, observation_contract
from run_mz103_depth_frontend import PANELS, write, event_compare

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT/'artifacts.local/work/mz106-temporal-geometry-20260912'
REF = ROOT/'artifacts.local/work/mz103-depth-frontend-20260912/native-v1'
PREVIOUS_SEAL = ROOT/'artifacts.local/work/mz105-residual-matching-20260912/features-v3/seal.json'
FOCAL = 640/(2*math.tan(math.radians(35)))
LK = dict(winSize=(21,21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,.01))
CRITICAL = ('thin_left','thin_right','small_head','occluded_thin')
sys.path.insert(0,str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, Workload, select_backend


def camera_to_world(camera):
    basis = m.rotation(camera['yaw'],camera['pitch'],camera.get('roll',0.))
    # CV right/down/forward -> UE world; capture left is the rig origin.
    return basis[:,[1,2,0]] * np.array([1.,-1.,1.])


def current_to_past(current, past):
    a,b = camera_to_world(current),camera_to_world(past)
    pa,pb = [np.array([c[k] for k in ('x','y','z')]) for c in (current,past)]
    return b.T@a, b.T@(pa-pb)


def track(current, past, xy):
    n = len(xy)
    if not n:
        return np.empty((0,2)), np.zeros(0,bool), np.empty(0), np.empty(0)
    p = np.asarray(xy,dtype='float32').reshape(-1,1,2)
    q,status,error = cv2.calcOpticalFlowPyrLK(current,past,p,None,**LK)
    back,back_status,_ = cv2.calcOpticalFlowPyrLK(past,current,q,None,**LK)
    q,back = q.reshape(-1,2),back.reshape(-1,2)
    cycle = np.linalg.norm(back-xy,axis=1)
    h,w = current.shape
    interior = lambda v: ((v[:,0]>=10)&(v[:,0]<w-10)&(v[:,1]>=10)&(v[:,1]<h-10))
    valid = (status.ravel().astype(bool)&back_status.ravel().astype(bool)
             & np.isfinite(q).all(1)&np.isfinite(cycle)&(cycle<=1.)
             & (error.ravel()<=20.) & interior(xy)&interior(q))
    return q,valid,cycle,error.ravel()


def geometric_contradiction(xy, depth, tracked, tracking_valid, rotation, translation):
    """Far alternatives are the projected ray segment from4m to infinity."""
    xy = np.asarray(xy).reshape(-1,2)
    ray = np.column_stack(((xy[:,0]-320)/FOCAL,(xy[:,1]-180)/FOCAL,np.ones(len(xy))))
    direction = ray@np.asarray(rotation).T
    near = direction*np.asarray(depth)[:,None]+translation
    far4 = direction*4.+translation
    project = lambda x: x[:,:2]/np.maximum(x[:,2:],1e-9)*FOCAL + [320.,180.]
    pn,p4,pinf = project(near),project(far4),project(direction)
    segment = pinf-p4
    alpha = np.sum((tracked-p4)*segment,axis=1)/np.maximum(np.sum(segment**2,axis=1),1e-15)
    closest = p4+np.clip(alpha,0.,1.)[:,None]*segment
    en = np.linalg.norm(tracked-pn,axis=1)
    ef = np.linalg.norm(tracked-closest,axis=1)
    gap = np.linalg.norm(pn-p4,axis=1)
    usable = (tracking_valid & (near[:,2]>0)&(far4[:,2]>0)&(direction[:,2]>0)
              & np.isfinite(en)&np.isfinite(ef)&(gap>=1.))
    if np.linalg.norm(translation)<.10:
        usable[:] = False
    rejected = usable & (en>2.) & (ef<=1.) & ((en-ef)>=1.)
    return rejected,usable,np.column_stack((en,ef,gap))


def prepare():
    """Bind observable paths and separately labeled ideal-pose metadata."""
    out=TASK/'inputs.json'
    if out.exists():
        raise ValueError('Inputs already bound')
    oldseal=json.loads(PREVIOUS_SEAL.read_text())
    panels={}
    for panel,(name,old_dir,depth_dir,_) in PANELS.items():
        folder=ROOT/'artifacts.local/work'/name
        cap=folder/'capture-v1'
        receipt=json.loads((cap/'receipt.json').read_text())
        specpath=cap/'spec.json'
        assert sha(specpath)==receipt['spec_sha256']
        # Only camera metadata is forwarded; objects and task labels are excluded.
        spec=json.loads(specpath.read_text())
        obs=json.loads((REF/panel/'observations.json').read_text())
        assert [f['id'] for f in spec['frames']]==[o['id'] for o in obs]
        entries=[]
        for i,o in enumerate(obs):
            paths=dict(left=cap/'frame'/o['id']/'left.png',depth=folder/old_dir/depth_dir/(o['id']+'.npy'))
            hashes={}
            for key,path in paths.items():
                hashes[key]=sha(path)
                assert hashes[key]==oldseal['inputs'][str(path.relative_to(ROOT))]
            entries.append(dict(**o,paths={k:str(v) for k,v in paths.items()},hashes=hashes,
                ideal_camera=spec['frames'][i]['camera']))
        del spec
        panels[panel]=dict(frames=entries,spec_sha256=receipt['spec_sha256'])
    write(out,dict(authority='RGB_DEPTH_OBSERVATIONS_PLUS_SEPARATE_IDEAL_CAMERA_METADATA',
        panels=panels,prior_depth_seal_sha256=sha(PREVIOUS_SEAL)))
    print(out,flush=True)


def produce(pose_file):
    output=TASK/'verifier-v1'
    if output.exists():
        raise ValueError('Preserve existing verifier outputs')
    inputs=json.loads((TASK/'inputs.json').read_text())
    pose=json.loads(pose_file.read_text())
    pose_seal=json.loads((pose_file.parent/'seal.json').read_text())
    assert pose_seal['authority']=='RGB_ONLY_NO_NATIVE_DEPTH_NO_LABELS_NO_WORLD_POSE'
    assert sha(pose_file)==pose_seal['estimates_sha256']
    for name,digest in pose_seal['inputs'].items():assert sha(Path(name))==digest
    estimated={(x['panel'],x['id']):x for x in pose}
    assert len(estimated)==576
    output.mkdir()
    cv2.setNumThreads(4)
    first=inputs['panels']['mz101']['frames']
    l0,l1=[cv2.imread(o['paths']['left'],0) for o in first[:2]]
    y,x=np.mgrid[30:330:12,30:610:12]
    sample=np.column_stack((x.ravel(),y.ravel())).astype('float32')
    # This OpenCV build exposes no CUDA SparsePyrLK implementation.
    has_gpu=hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create') and cv2.cuda.getCudaEnabledDeviceCount()>0
    if has_gpu:
        raise RuntimeError('CUDA SparsePyrLK available: backend comparison required')
    backend=select_backend(Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate('opencv-cpu-lk','cpu',lambda:track(l1,l0,sample),
            lambda _:DeviceObservation('cpu','host CPU',f'OpenCV-{cv2.__version__}')),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=TASK/'backend.json')
    timing=[];pixel_rows=[]
    for panel,entry in inputs['panels'].items():
        dest=output/panel;dest.mkdir()
        frames=entry['frames'];old=np.load(REF/panel/'predictions.npz')
        obs=[{k:o[k] for k in ('id','episode','time_s','pose')} for o in frames]
        write(dest/'observations.json',obs)
        supports={name:[] for name in ('baseline','ideal','rgb')}
        for i,o in enumerate(frames):
            started=time.perf_counter()
            for key,path in o['paths'].items():
                assert sha(Path(path))==o['hashes'][key]
            depth=np.load(o['paths']['depth'])
            base=m.readout(m.depth_points(depth),o['pose'])[0]
            np.testing.assert_array_equal(base,old['sgbm_support'][i])
            supports['baseline'].append(base)
            yy,xx,inside=membership(depth,o['pose']);selected=inside.any(1)
            yy,xx,inside=yy[selected],xx[selected],inside[selected]
            xy=np.column_stack((xx,yy)).astype('float32')
            rejection={k:np.zeros(len(xy),bool) for k in ('ideal','rgb')}
            usable={k:np.zeros(len(xy),bool) for k in ('ideal','rgb')}
            geometry={k:np.full((len(xy),3),np.nan) for k in ('ideal','rgb')}
            tracked=np.full_like(xy,np.nan);valid=np.zeros(len(xy),bool)
            row=dict(panel=panel,id=o['id'],candidate_pixels=len(xy),has_past=False)
            if i and frames[i-1]['episode']==o['episode']:
                prev=frames[i-1]
                assert abs(o['time_s']-prev['time_s']-.25)<1e-6
                row['has_past']=True
                current=cv2.imread(o['paths']['left'],0);past=cv2.imread(prev['paths']['left'],0)
                tracked,valid,_,_=track(current,past,xy)
                ir,it=current_to_past(o['ideal_camera'],prev['ideal_camera'])
                row['ideal_baseline_m']=float(np.linalg.norm(it))
                rejection['ideal'],usable['ideal'],geometry['ideal']=geometric_contradiction(xy,depth[yy,xx],tracked,valid,ir,it)
                e=estimated[(panel,o['id'])]
                assert e['past_id']==prev['id']
                row['rgb_pose_valid']=bool(e['valid'])
                if e['valid']:
                    # PnP describes past->current; the verifier uses current->past.
                    er=np.asarray(e['R']).T;et=-er@np.asarray(e['t']).reshape(3)
                    row['rgb_baseline_m']=float(np.linalg.norm(et))
                    rejection['rgb'],usable['rgb'],geometry['rgb']=geometric_contradiction(xy,depth[yy,xx],tracked,valid,er,et)
            row['tracked_pixels']=int(valid.sum())
            for name in ('ideal','rgb'):
                filtered=depth.copy();filtered[yy[rejection[name]],xx[rejection[name]]]=np.nan
                support=m.readout(m.depth_points(filtered),o['pose'])[0]
                assert np.all(support<=base)
                supports[name].append(support)
                row[name]=dict(usable_pixels=int(usable[name].sum()),rejected_pixels=int(rejection[name].sum()),
                    query_usable=inside[usable[name]].any(0).tolist(),
                    query_rejected=inside[rejection[name]].any(0).tolist())
            np.savez_compressed(dest/(o['id']+'.npz'),yy=yy,xx=xx,inside=inside,tracked=tracked,
                tracking_valid=valid,**{k+'_reject':v for k,v in rejection.items()},
                **{k+'_usable':v for k,v in usable.items()},**{k+'_geometry':v for k,v in geometry.items()})
            pixel_rows.append(row);timing.append(time.perf_counter()-started)
            if (i+1)%48==0:print(panel,i+1,'/288',flush=True)
        union={name:np.asarray(v)+old['tof_support'] for name,v in supports.items()}
        episodes=[o['episode'] for o in obs]
        final={name:m.hysteresis(v,episodes) for name,v in union.items()}
        np.testing.assert_array_equal(union['baseline'],old['sgbm_union_support'])
        np.testing.assert_array_equal(final['baseline'],old['sgbm_union'])
        np.savez_compressed(dest/'predictions.npz',**final,**{k+'_support':v for k,v in union.items()})
    write(output/'pixel-summary.json',pixel_rows)
    write(output/'seal.json',dict(status='PREDICTIONS_SEALED_BEFORE_TASK_TRUTH',
        authority=dict(rgb='RGB_STEREO_PNP',ideal='SIMULATOR_IDEAL_CAMERA_POSE_DIAGNOSTIC'),
        source_sha256=sha(Path(__file__)),inputs_sha256=sha(TASK/'inputs.json'),pose_sha256=sha(pose_file),
        hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        frames=len(timing),mean_full_frame_s=float(np.mean(timing)),p95_full_frame_s=float(np.percentile(timing,95)),
        backend=backend['selected_backend'],pose_estimation_cost_excluded=True))
    print('SEALED',flush=True)


def evaluate():
    source=TASK/'verifier-v1';out=TASK/'evaluation-v1'
    if out.exists():raise ValueError('Evaluation exists')
    seal=json.loads((source/'seal.json').read_text())
    for name,digest in seal['hashes'].items():assert sha(source/name)==digest
    inputs=json.loads((TASK/'inputs.json').read_text());pixels=json.loads((source/'pixel-summary.json').read_text())
    out.mkdir();panels={}
    for panel,(folder,_,_,_) in PANELS.items():
        pred=np.load(source/panel/'predictions.npz')
        obs=json.loads((source/panel/'observations.json').read_text())
        specpath=ROOT/'artifacts.local/work'/folder/'capture-v1/spec.json'
        assert sha(specpath)==inputs['panels'][panel]['spec_sha256']
        spec=json.loads(specpath.read_text());assert observation_contract(spec)==obs
        gt=np.load(REF/panel/'truth.npy')
        raw={k:pred[k+'_support']>0 for k in ('baseline','ideal','rgb')}
        scores={stage:{k:metrics(p,gt,obs) for k,p in pp.items()} for stage,pp in
                [('raw',raw),('final',{k:pred[k] for k in raw})]}
        results={}
        for name in ('ideal','rgb'):
            b,p=raw['baseline'],raw[name]
            critical={}
            for family in CRITICAL:
                target=b & gt & np.array([f['family']==family for f in spec['frames']])[:,None]
                critical[family]=dict(total=int(target.sum()),retained=int((target & p).sum()))
            events=event_compare(scores['final'][name],scores['final']['baseline'])
            gates=dict(fewer_raw_fp=scores['raw'][name]['FP']<scores['raw']['baseline']['FP'],
                no_more_raw_fn=scores['raw'][name]['FN']<=scores['raw']['baseline']['FN'],
                critical_retention=all(v['retained']>=math.ceil(.95*v['total']) for v in critical.values()),
                no_lost_events=not events['lost_events'],
                fewer_final_fp=scores['final'][name]['FP']<scores['final']['baseline']['FP'],
                delay_within_frame=events['max_extra_delay_s'] is None or events['max_extra_delay_s']<=.25,
                no_more_false_segments=scores['final'][name]['false_sessions']<=scores['final']['baseline']['false_sessions'])
            transitions={}
            for stage,base,other in [('raw',b,p),('final',pred['baseline'],pred[name])]:
                transitions[stage]=dict(lost_tp=int((base & ~other & gt).sum()),new_tp=int((~base & other & gt).sum()),
                    removed_fp=int((base & ~other & ~gt).sum()),new_fp=int((~base & other & ~gt).sum()))
            rows=[r for r in pixels if r['panel']==panel]
            query_usable=np.array([r[name]['query_usable'] for r in rows])
            results[name]=dict(gates=gates,critical=critical,event_comparison=events,transitions=transitions,
                unknown_query_frames=int((~p).sum()),coverage=dict(
                    candidate_pixels=sum(r['candidate_pixels'] for r in rows),
                    tracked_pixels=sum(r['tracked_pixels'] for r in rows),
                    usable_pixels=sum(r[name]['usable_pixels'] for r in rows),
                    rejected_pixels=sum(r[name]['rejected_pixels'] for r in rows),
                    baseline_raw_tp_with_any_usable=int((query_usable & b & gt).sum()),
                    baseline_raw_fp_with_any_usable=int((query_usable & b & ~gt).sum())))
        family={fam:{stage:{k:metrics(v,gt,obs,[f['family']==fam for f in spec['frames']]) for k,v in pp.items()}
            for stage,pp in [('raw',raw),('final',{k:pred[k] for k in raw})]}
            for fam in dict.fromkeys(f['family'] for f in spec['frames'])}
        detail=dict(scores=scores,results=results,families=family)
        write(out/(panel+'.json'),detail);panels[panel]=detail
    pooled={stage:{name:{key:sum(p['scores'][stage][name][key] for p in panels.values()) for key in
            ('TP','FP','FN','missed_events','false_sessions')} for name in ('baseline','ideal','rgb')} for stage in ('raw','final')}
    summary=dict(authority='CONSUMED_RENDERED_DEVELOPMENT',pooled_descriptive=pooled,
        panels={p:d['results'] for p,d in panels.items()},source_seal_sha256=sha(source/'seal.json'))
    write(out/'summary.json',summary)
    write(out/'receipt.json',dict(status='PASS',hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','produce','evaluate'))
    parser.add_argument('--pose-file',type=Path);args=parser.parse_args()
    TASK.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='produce':produce(args.pose_file)
    else:evaluate()
