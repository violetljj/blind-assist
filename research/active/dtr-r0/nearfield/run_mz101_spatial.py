"""Seal observable predictions, then evaluate and render one MZ101 comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
import mz101_spatial as m

ROOT=Path(__file__).resolve().parents[4]
TASK=ROOT/'artifacts.local/work/mz101-stereo-tof-spatial-20260912'
METHODS=('tof','stereo','union')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def observation_contract(spec):
    return [dict(id=f['id'],episode=f['episode'],time_s=f['time_s'],pose=dict(
        yaw=f['camera']['yaw'],pitch=f['camera']['pitch'],roll=f['camera']['roll'],
        camera_in_body_m=[f['camera'][k]-f['body_origin_m'][i] for i,k in enumerate(('x','y','z'))]))
        for f in spec['frames']]


def segments(a):
    x=np.pad(np.asarray(a,bool).astype(int),(1,1));d=np.diff(x)
    return list(zip(np.flatnonzero(d==1).tolist(),np.flatnonzero(d==-1).tolist()))


def metrics(pred,gt,obs,selection=None):
    ids=np.array([o['episode'] for o in obs]);times=np.array([o['time_s'] for o in obs])
    sel=np.ones(len(gt),bool) if selection is None else np.asarray(selection,bool)
    p,g=pred[sel],gt[sel]
    tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
    detail=[];false_sessions=0;false_segments=0;fragments=0;events=0;misses=0;delays=[]
    for ep in dict.fromkeys(ids[sel]):
        ix=np.flatnonzero((ids==ep)&sel)
        for part in range(2):
            e=segments(gt[ix,part]);ps=segments(pred[ix,part]);false_segments+=len(segments(pred[ix,part]&~gt[ix,part]))
            false_sessions+=sum(not gt[ix[s:t],part].any() for s,t in ps)
            for s,t in e:
                events+=1;hit=np.flatnonzero(pred[ix[s:t],part]);misses+=not len(hit)
                fragments+=max(0,len(segments(pred[ix[s:t],part]))-1)
                delay=float(times[ix[s+hit[0]]]-times[ix[s]]) if len(hit) else None
                if delay is not None:delays.append(delay)
                detail.append(dict(episode=str(ep),part=m.PARTS[part],start_s=float(times[ix[s]]),
                    end_s=float(times[ix[t-1]]),left_censored=s==0,right_censored=t==len(ix),
                    detected=bool(len(hit)),first_correct_delay_s=delay,
                    carried_at_start=bool(s>0 and pred[ix[s-1],part] and pred[ix[s],part])))
    return dict(TP=tp,FP=fp,FN=fn,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        events=events,missed_events=int(misses),false_sessions=int(false_sessions),
        false_segments=false_segments,fragments=fragments,false_duration_s=fp*.25,
        first_correct_delay_mean_s=float(np.mean(delays)) if delays else None,
        first_correct_delay_max_s=max(delays) if delays else None,event_details=detail)


def ground_truth(spec):
    truth=[]
    for f in spec['frames']:
        row=[]
        for low,high in m.BOXES:
            supported=False
            for obj in f['objects']:
                a=np.array(obj['center_m'])-np.array(obj['size_m'])/2-np.array(f['body_origin_m'])
                b=np.array(obj['center_m'])+np.array(obj['size_m'])/2-np.array(f['body_origin_m'])
                supported |= bool(((b>=low)&(a<=high)).all())
            row.append(supported)
        truth.append(row)
    return np.asarray(truth,bool)


def draw(frame,left,depth,ranges,valid,pred,gt,out):
    canvas=np.zeros((460,1280,3),np.uint8)
    canvas[:360,:640]=left
    mapped=np.uint8(np.clip(np.nan_to_num(depth,nan=0.)/4*255,0,255))
    color=cv2.applyColorMap(mapped,cv2.COLORMAP_TURBO);color[~np.isfinite(depth)]=0
    canvas[:360,640:]=color
    f=640/(2*np.tan(np.radians(35)))
    pose=dict(yaw=frame['camera']['yaw'],pitch=frame['camera']['pitch'],roll=frame['camera']['roll'],
        camera_in_body_m=[frame['camera'][k]-frame['body_origin_m'][i] for i,k in enumerate(('x','y','z'))])
    camera_points=m.depth_points(depth)
    body=camera_points@m.rotation(pose['yaw'],pose['pitch'],pose['roll']).T+pose['camera_in_body_m']
    for part,(low,high) in enumerate(m.BOXES):
        q=camera_points[((body>=low)&(body<=high)).all(1)]
        q=q[::max(1,len(q)//300)]
        tint=(255,230,0) if part==0 else (255,0,230)
        for x,y,z in q:
            u,v=int(320+f*y/x),int(180-f*z/x)
            if 0<=u<640 and 0<=v<360:cv2.circle(canvas,(u,v),1,tint,-1)
    for x,y,z in m.tof_points(ranges,valid):
        u,v=int(320+f*y/x),int(180-f*z/x)
        if 0<=u<640 and 0<=v<360:cv2.circle(canvas,(u,v),4,(0,200,255),1)
    cv2.putText(canvas,'ToF orange; stereo BODY cyan / HEAD pink',(10,25),0,.50,(255,255,255),1)
    cv2.putText(canvas,'Computed stereo depth; black=UNKNOWN',(650,25),0,.58,(255,255,255),2)
    cv2.putText(canvas,frame['id']+f"  t={frame['time_s']:.2f}s",(10,385),0,.55,(255,255,255),1)
    cv2.putText(canvas,f"GT current corridor BODY/HEAD {gt.astype(int).tolist()}",(10,410),0,.55,(220,220,220),1)
    cv2.putText(canvas,'  '.join(f'{name}: {pred[name].astype(int).tolist()}' for name in METHODS),(10,440),0,.60,(0,220,220),1)
    return canvas


def run(capture,out,spec_path):
    assert not out.exists() and out.resolve().parent==TASK.resolve()
    out.mkdir(parents=True);start=time.perf_counter()
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    assert len(spec['frames'])==288 and spec['rig']==m.RIG
    capture_receipt=json.loads((capture/'receipt.json').read_text())
    assert capture_receipt['status']=='PASS' and capture_receipt['frames']==288
    assert sha(spec_path)==capture_receipt['spec_sha256']
    for name,digest in capture_receipt['hashes'].items():assert sha(capture/name)==digest,name
    observations=observation_contract(spec);write(out/'observations.json',observations)
    for name in ('mz101_spatial.py','mz101_spatial_spec.py','run_mz101_spatial.py','MZ101_STEREO_TOF_PROTOCOL_20260912.md'):
        shutil.copyfile(Path(__file__).parent/name,out/name)
    rig=spec['rig'];ids=[o['episode'] for o in observations]
    counts={view:{name:[] for name in METHODS} for view in ('common','native','proxy_common')}
    proxy_rng=np.random.default_rng(101031)
    timings=[];input_hashes={};depth_dir=out/'stereo-depth';depth_dir.mkdir()
    for i,o in enumerate(observations):
        folder=capture/'frame'/o['id']
        paths={name:folder/name for name in ('left.png','right.png','tof-range.npy','tof-valid.npy')}
        for name,p in paths.items():input_hashes[str(p.relative_to(capture))]=sha(p)
        left,right=[cv2.imread(str(paths[n])) for n in ('left.png','right.png')]
        if left is None or right is None:raise ValueError(f'Missing stereo {o["id"]}')
        depth,timing=m.stereo_depth(left,right,rig);timings.append(timing)
        np.save(depth_dir/(o['id']+'.npy'),depth)
        ranges=np.load(paths['tof-range.npy']).reshape(64);valid=np.load(paths['tof-valid.npy']).reshape(64).astype(bool)
        noisy=ranges+proxy_rng.normal(0,.02,64)
        proxy_valid=valid&(proxy_rng.random(64)>=.15)
        # One declared two-packet blackout in every episode, no target dependence.
        if i%12 in (5,6):proxy_valid[:]=False
        points=dict(tof=m.tof_points(ranges,valid),stereo=m.depth_points(depth),proxy=m.tof_points(noisy,proxy_valid))
        for view in counts:
            c={name:m.readout(points['proxy' if view=='proxy_common' and name=='tof' else name],o['pose'],common_fov=view!='native')[0] for name in ('tof','stereo')}
            c['union']=c['tof']+c['stereo']
            for name in METHODS:counts[view][name].append(c[name])
        if (i+1)%24==0:print(f'predicted {i+1}/{len(observations)}',flush=True)
    preds={}
    for view in counts:
        for name in METHODS:
            c=np.array(counts[view][name]);counts[view][name]=c
            preds[f'{view}_{name}']=m.hysteresis(c,ids)
    np.savez_compressed(out/'predictions.npz',**preds,**{f'{v}_{n}_support':counts[v][n] for v in counts for n in METHODS})
    write(out/'prediction-seal.json',dict(status='PREDICTIONS_SEALED_BEFORE_EVALUATOR_SCORING_AND_NATIVE_DEPTH_ACCESS',
        observation_sha256=sha(out/'observations.json'),prediction_sha256=sha(out/'predictions.npz'),
        spec_sha256=sha(spec_path),input_hashes=input_hashes,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    # Evaluator-only access begins here; no predictions are changed below.
    manifest=json.loads((capture/'manifest.json').read_text())
    assert len(manifest['frames'])==len(spec['frames'])
    for f,row in zip(spec['frames'],manifest['frames']):
        assert f['id']==row['id']
        assert len(f['objects'])==len(row['native_bounds'])
        for obj,bounds in zip(f['objects'],row['native_bounds']):
            assert obj['name']==bounds['name']
            np.testing.assert_allclose(obj['center_m'],bounds['center_m'],atol=1e-5)
            np.testing.assert_allclose(np.array(obj['size_m'])/2,bounds['extent_m'],atol=1e-5)
    # Check every non-task surface is outside each GT corridor.
    for f in spec['frames']:
        for name in ('background','floor'):
            surface=capture_receipt[name]
            if surface is None:continue
            a=np.array(surface['center_m'])-np.array(surface['size_m'])/2-np.array(f['body_origin_m'])
            b=np.array(surface['center_m'])+np.array(surface['size_m'])/2-np.array(f['body_origin_m'])
            assert not any(((b>=low)&(a<=high)).all() for low,high in m.BOXES),(name,f['id'])
    gt=ground_truth(spec);np.save(out/'truth.npy',gt)
    results={key:metrics(p,gt,observations) for key,p in preds.items()}
    slices={family:{key:metrics(p,gt,observations,[f['family']==family for f in spec['frames']])
        for key,p in preds.items() if key.startswith('common')} for family in dict.fromkeys(f['family'] for f in spec['frames'])}
    appearances={app:{key:metrics(p,gt,observations,[f['appearance']==app for f in spec['frames']])
        for key,p in preds.items() if key.startswith('common')} for app in ('textured','flat')}
    diagnostic=[];video=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(1280,460))
    if not video.isOpened():raise RuntimeError('Video writer unavailable')
    snapshots=[]
    try:
        for i,f in enumerate(spec['frames']):
            folder=capture/'frame'/f['id'];depth=np.load(depth_dir/(f['id']+'.npy'))
            native=np.load(folder/'native-left-depth.npy')
            valid=np.isfinite(depth)&np.isfinite(native)&(native>=.5)&(native<=4)
            errors=np.abs(depth[valid]-native[valid])
            diagnostic.append(dict(id=f['id'],valid_pixels=int(np.isfinite(depth).sum()),
                paired_depth_pixels=int(valid.sum()),depth_MAE_m=float(errors.mean()) if len(errors) else None,
                depth_median_error_m=float(np.median(errors)) if len(errors) else None,
                tof_supported=counts['common']['tof'][i].tolist(),stereo_supported=counts['common']['stereo'][i].tolist()))
            canvas=draw(f,cv2.imread(str(folder/'left.png')),depth,np.load(folder/'tof-range.npy'),np.load(folder/'tof-valid.npy'),
                {n:preds['common_'+n][i] for n in METHODS},gt[i],out)
            video.write(canvas)
            if i%12==7:
                p=out/(f['episode']+'.jpg');cv2.imwrite(str(p),canvas)
                snapshots.append(cv2.resize(canvas,(640,230)))
    finally:video.release()
    for app,indices in [('textured',range(0,24,2)),('flat',range(1,24,2))]:
        tiles=[snapshots[i] for i in indices]
        sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
        cv2.imwrite(str(out/f'contact-sheet-{app}.jpg'),sheet)
    baseline=results['common_tof'];union=results['common_union']
    byevent=lambda r:{(e['episode'],e['part'],e['start_s']):e for e in r['event_details']}
    be,ue=byevent(baseline),byevent(union)
    new=[k for k,e in be.items() if not e['detected'] and ue[k]['detected']]
    lost=[k for k,e in be.items() if e['detected'] and not ue[k]['detected']]
    delays=[ue[k]['first_correct_delay_s']-e['first_correct_delay_s'] for k,e in be.items() if e['detected'] and ue[k]['detected']]
    gates=dict(new_tof_missed_event=len(new)>0,no_lost_tof_events=not lost,
        false_sessions_not_increased=union['false_sessions']<=baseline['false_sessions'],
        false_duration_not_increased=union['false_duration_s']<=baseline['false_duration_s'],
        first_correct_not_later=not delays or max(delays)<=.25)
    summary=dict(frames=len(gt),episodes=len(set(ids)),geometry_families=12,
        target='CURRENT_FORWARD_BODY_HEAD_CORRIDOR',results=results,slices=slices,appearances=appearances,
        per_part={part:{key:dict(TP=int((p[:,j]&gt[:,j]).sum()),FP=int((p[:,j]&~gt[:,j]).sum()),
            FN=int((~p[:,j]&gt[:,j]).sum())) for key,p in preds.items()} for j,part in enumerate(m.PARTS)},
        gates=gates,component_gate_pass=all(gates.values()),new_events=new,lost_events=lost,
        direct_support_unknown={f'{v}_{n}':int((counts[v][n]==0).sum()) for v in counts for n in METHODS},
        backend=dict(opencv=cv2.__version__,stereo_device='CPU',reason='GPU_BACKEND_UNAVAILABLE',
            mean_stereo_seconds=float(np.mean([t['seconds'] for t in timings])),
            total_stereo_seconds=sum(t['seconds'] for t in timings)),
        elapsed_s=time.perf_counter()-start)
    write(out/'depth-diagnostic.json',diagnostic);write(out/'summary.json',summary)
    write(out/'receipt.json',dict(status='PASS',hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in summary.items() if k in ('frames','episodes','gates','component_gate_pass','backend')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();run(a.capture,a.output,a.spec)
