"""Local RGB flow association of unchanged autonomous depth-ridge proposals.

Consumed Development diagnostic. No label/seed input, no missing-frame fill.
Local translation uses forward/backward checked LK points near each prior line;
unreliable local flow falls back to the original whole-frame affine estimate.
"""
from __future__ import annotations
import json
import math
import time
import sys
from collections import Counter
import cv2
import numpy as np
from atom_camera import AtomCamera, ROOT
from thin_structure_candidates_dev_20261013 import line_parts

SOURCE = ROOT / 'artifacts.local/work/thin-structure-dev-20261013/detection_ridge_frames.jsonl'
OUT = ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013'
RAW = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw'
CONFIG = dict(distance_px=14,angle_deg=15,overlap=.3,stable_hits=3,max_gap_s=.5,
              local_radius_px=10,max_features=16,min_flow_points=4,fb_px=1.5,
              residual_px=3,min_inlier_fraction=.6,min_line_span=.25)
TRACK_FIELDS = {'track_id','hits','stable','track_flow','association'}

def geometry(c):
    return line_parts([c['x1'],c['y1'],c['x2'],c['y2']])

def local_predictions(prev, gray, candidates, fallback):
    """One batched LK call in each direction; one robust translation per line."""
    if not candidates: return [],[]
    all_pts=[]; groups=[]
    h,w=gray.shape
    for c in candidates:
        p,m,u,L=geometry(c)
        lo=np.maximum(0,np.floor(p.min(axis=0)-12)).astype(int)
        hi=np.minimum([w,h],np.ceil(p.max(axis=0)+13)).astype(int)
        roi=prev[lo[1]:hi[1],lo[0]:hi[0]]
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        normal=np.array([-u[1],u[0]])
        mask=(np.abs((xx-m[0])*normal[0]+(yy-m[1])*normal[1])<=10).astype('uint8')*255
        corners=cv2.goodFeaturesToTrack(roi,16,.01,3,mask=mask)
        pts=np.empty((0,2),np.float32) if corners is None else corners[:,0]+lo
        # Explicit along-line samples allow a thin texture-poor shaft a chance;
        # LK's minimum eigenvalue and the FB check still reject aperture failures.
        seeds=p[0]+np.linspace(.05,.95,9)[:,None]*(p[1]-p[0])
        pts=np.concatenate([pts,seeds]).astype(np.float32)
        groups.append((len(all_pts),len(all_pts)+len(pts)))
        all_pts.extend(pts)
    a=np.array(all_pts,np.float32).reshape(-1,1,2)
    b,s1,_=cv2.calcOpticalFlowPyrLK(prev,gray,a,None,winSize=(15,15),maxLevel=3)
    back,s2,_=cv2.calcOpticalFlowPyrLK(gray,prev,b,None,winSize=(15,15),maxLevel=3)
    aa=a[:,0]; bb=b[:,0]
    fb=np.linalg.norm(back[:,0]-aa,axis=1)
    ok=(s1[:,0]>0)&(s2[:,0]>0)&np.isfinite(bb).all(axis=1)&(fb<=1.5)
    ok&=(bb[:,0]>=0)&(bb[:,0]<w)&(bb[:,1]>=0)&(bb[:,1]<h)
    predictions=[]; qualities=[]
    for c,(lo,hi) in zip(candidates,groups):
        p,m,u,L=geometry(c)
        good=np.flatnonzero(ok[lo:hi])+lo
        quality=dict(valid=False,reason='few_fb_points',points=hi-lo,fb_points=len(good))
        pred=np.c_[p,np.ones(2)]@fallback.T
        if len(good)>=4:
            delta=bb[good]-aa[good]; shift=np.median(delta,axis=0)
            inlier=np.linalg.norm(delta-shift,axis=1)<=3
            ng=int(inlier.sum())
            span=float(np.ptp(aa[good[inlier]]@u)/max(L,1)) if ng else 0
            quality.update(inliers=ng,line_span=span,median_fb=float(np.median(fb[good])),reason='incoherent_or_short_span')
            if ng>=4 and ng/len(good)>=.6 and span>=.25:
                shift=np.median(delta[inlier],axis=0)
                pred=p+shift
                quality.update(valid=True,reason='local_translation',shift=shift.tolist())
        predictions.append(line_parts(pred)); qualities.append(quality)
    return predictions,qualities

def associate(current, previous, predictions, qualities, next_id):
    used=set()
    for c in current:
        p,m,u,L=geometry(c); best=None
        for k,(q,n,v,K) in enumerate(predictions):
            if k in used or abs(u@v)<math.cos(math.radians(15)): continue
            normal=np.array([-u[1],u[0]])
            distance=abs(float((m-n)@normal))
            overlap=max(0,min(max(p@u),max(q@u))-max(min(p@u),min(q@u)))/max(1,min(L,K))
            if distance<=14 and overlap>=.3:
                cost=distance+.1*abs(float((m-n)@u))
                if best is None or cost<best[0]: best=(cost,k,distance,overlap)
        if best is None:
            c.update(track_id=next_id,hits=1,track_flow=None,association=None); next_id+=1
        else:
            cost,k,distance,overlap=best; used.add(k); d=previous[k]
            c.update(track_id=d['track_id'],hits=d['hits']+1,track_flow=qualities[k],
                     association=dict(previous_rank=k+1,cost=cost,distance=distance,overlap=overlap))
        c['stable']=c['hits']>=3
    assert len({c['track_id'] for c in current})==len(current)
    return next_id

def self_check():
    rng=np.random.default_rng(42)
    a=cv2.GaussianBlur(rng.integers(0,256,(160,160),dtype=np.uint8),(3,3),0)
    matrix=np.array([[1.,0.,7.],[0.,1.,-4.]])
    b=cv2.warpAffine(a,matrix,(160,160))
    c=dict(x1=70.,y1=30.,x2=70.,y2=125.,track_id=10,hits=2)
    pred,quality=local_predictions(a,b,[c],np.eye(2,3))
    assert quality[0]['valid'],quality
    error=float(np.max(np.abs(pred[0][0]-(geometry(c)[0]+[7,-4]))))
    assert error<.5,error
    cur=[dict(x1=77.,y1=26.,x2=77.,y2=121.),dict(x1=77.,y1=26.,x2=77.,y2=121.)]
    associate(cur,[c],pred,quality,11)
    assert cur[0]['track_id']==10 and cur[0]['stable'] and cur[1]['hits']==1
    return dict(translation_max_error_px=error,one_to_one=True)

def main():
    start=time.perf_counter(); cv2.setNumThreads(2); OUT.mkdir(parents=True,exist_ok=True)
    checks=self_check(); camera=AtomCamera(); old_key=None; previous=[]; prev_gray=None; prev_t=None
    summary=dict(config=CONFIG,checks=checks,sessions={},gpu_seconds=0,complete=False,
                 evidence='consumed Development; association only; no labels or manual seed')
    seen=set(); unchanged=0
    try:
        with SOURCE.open() as source,(OUT/'local_frames.jsonl').open('w',encoding='utf8') as output:
            for line in source:
                if time.perf_counter()-start>365: raise TimeoutError('365s internal execution cap')
                row=json.loads(line); key=row['key']; identity=(key,row['i'])
                assert identity not in seen; seen.add(identity)
                if key!=old_key:
                    folder=RAW/row['session']; rgb=[json.loads(q) for q in (folder/'rgb.jsonl').read_text().splitlines()]
                    previous=[]; prev_gray=None; prev_t=None; next_id=1
                    counts=Counter(); summary['sessions'][key]=counts; old_key=key
                gray=cv2.cvtColor(camera.undistort_upright(cv2.imread(str(folder/rgb[row['rgb_index']]['file']))),cv2.COLOR_BGR2GRAY)
                gap=prev_t is None or row['rgb_t_ns']<=prev_t or (row['rgb_t_ns']-prev_t)/1e9>.5
                if gap: previous=[]; prev_gray=None
                fallback=np.array(row['motion'].get('matrix',[[1.,0.,0.],[0.,1.,0.]]),float)
                predictions,qualities=local_predictions(prev_gray,gray,previous,fallback)
                original=[{k:v for k,v in c.items() if k not in TRACK_FIELDS} for c in row['candidates']]
                next_id=associate(row['candidates'],previous,predictions,qualities,next_id)
                assert original==[{k:v for k,v in c.items() if k not in TRACK_FIELDS} for c in row['candidates']]
                unchanged+=len(original)
                row['local_flow']=dict(attempted=len(previous),valid=sum(q['valid'] for q in qualities),
                                       reasons=dict(Counter(q['reason'] for q in qualities)),prior_quality=qualities)
                row['temporal_reset']=gap
                output.write(json.dumps(row,allow_nan=False)+'\n')
                counts.update(frames=1,candidates=len(original),depth_frames=int(row['depth_available']),
                              stable=sum(c['stable'] for c in row['candidates']),local_attempted=len(previous),
                              local_valid=sum(q['valid'] for q in qualities),resets=int(gap),
                              matched=sum(c['hits']>1 for c in row['candidates']),
                              matched_local=sum(c['track_flow'] is not None and c['track_flow']['valid'] for c in row['candidates']))
                previous=row['candidates']; prev_gray=gray; prev_t=row['rgb_t_ns']
        summary.update(complete=True,unique_frame_keys=len(seen),unchanged_candidates=unchanged)
    finally:
        summary['wall_s']=time.perf_counter()-start
        (OUT/'local_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        print(json.dumps(summary,indent=2),flush=True)

def camera_clock(chain=False):
    """Separate state-clock ablation, preserving the strict-reset output above.

    A duplicate image never supplies another hit. On depth-missing images only
    internal line positions are transported; exported observations stay empty.
    Tracks expire 0.5s after an actual candidate, and stable requires three
    distinct advancing RGB observations in the trailing 0.5s window.
    """
    start=time.perf_counter(); cv2.setNumThreads(2); camera=AtomCamera()
    stem='camera_chain' if chain else 'camera_clock'
    summary=dict(config=CONFIG,variant=stem+' + local flow',sessions={},gpu_seconds=0,
        plan='Duplicate already-observed RGB: no hits/reset. Missing depth: internal transport only. '
             'Expire after 0.5s no actual candidate. Stable: '+
             ('3 independent candidate detections along a chain with adjacent gaps <=0.5s; first detection on a previously depth-missing RGB counts once.' if chain else
              '3 independent advancing RGB detections in trailing 0.5s.'),
        definition_correction=bool(chain),
        complete=False)
    old_key=None; seen=set(); unchanged=0; duplicate_checks=0
    with SOURCE.open() as source,(OUT/(stem+'_frames.jsonl')).open('w',encoding='utf8') as output:
        for line in source:
            if time.perf_counter()-start>250: raise TimeoutError('camera-clock 250s cap')
            row=json.loads(line); key=row['key']; t=row['rgb_t_ns']; identity=(key,row['i'])
            assert identity not in seen; seen.add(identity)
            if key!=old_key:
                folder=RAW/row['session']; rgb=[json.loads(q) for q in (folder/'rgb.jsonl').read_text().splitlines()]
                state=[]; prev_gray=None; prev_t=None; next_id=1
                counts=Counter(); summary['sessions'][key]=counts; old_key=key
            gray=cv2.cvtColor(camera.undistort_upright(cv2.imread(str(folder/rgb[row['rgb_index']]['file']))),cv2.COLOR_BGR2GRAY)
            reset=prev_t is None or t<prev_t or (t-prev_t)/1e9>.5
            duplicate=t==prev_t
            if reset: state=[]
            state=[c for c in state if 0<=t-c['_last_detection_ns']<=500_000_000]
            fallback=np.array(row['motion'].get('matrix',[[1.,0.,0.],[0.,1.,0.]]),float)
            if duplicate:
                predictions=[geometry(c) for c in state]
                qualities=[dict(valid=False,reason='duplicate_no_motion') for c in state]
            else:
                predictions,qualities=local_predictions(prev_gray,gray,state,fallback)
            original=[{k:v for k,v in c.items() if k not in TRACK_FIELDS} for c in row['candidates']]
            next_id=associate(row['candidates'],state,predictions,qualities,next_id)
            used=set(); next_state=[]
            for c in row['candidates']:
                link=c['association']; history=[]
                if link is not None:
                    k=link['previous_rank']-1; used.add(k)
                    history=list(state[k]['_hit_times']) if chain else [v for v in state[k]['_hit_times'] if 0<=t-v<=500_000_000]
                prior_count=len(history); already_seen=t in history
                counted=(not already_seen) if chain else (not duplicate and (not history or history[-1]!=t))
                if counted: history.append(t)
                if already_seen:
                    assert len(history)==prior_count
                    duplicate_checks+=1
                if chain:
                    assert all(0<b-a<=500_000_000 for a,b in zip(history,history[1:])),history
                c['hits']=len(history); c['stable']=len(history)>=3
                internal=dict(c,_hit_times=history,_last_detection_ns=t)
                next_state.append(internal)
                c['independent_hit_t_ns']=history
                c['counted_new_image']=counted
            # Retention enables future real observations to reconnect but is
            # never exported as an observation or counted toward hits.
            for k,c in enumerate(state):
                if k in used: continue
                p=predictions[k][0]; carried=dict(c)
                carried.update(x1=float(p[0,0]),y1=float(p[0,1]),x2=float(p[1,0]),y2=float(p[1,1]))
                if not chain: carried['_hit_times']=[v for v in carried['_hit_times'] if 0<=t-v<=500_000_000]
                carried['hits']=len(carried['_hit_times']); carried['stable']=carried['hits']>=3
                next_state.append(carried)
            assert original==[{k:v for k,v in c.items() if k not in TRACK_FIELDS|{'independent_hit_t_ns','counted_new_image'}} for c in row['candidates']]
            assert all(len(c['independent_hit_t_ns'])==len(set(c['independent_hit_t_ns'])) for c in row['candidates'])
            assert all(not c['stable'] or (len(c['independent_hit_t_ns'])>=3 and
                       (chain or t-c['independent_hit_t_ns'][0]<=500_000_000)) for c in row['candidates'])
            unchanged+=len(original)
            row.update(temporal_reset=reset,duplicate_rgb=duplicate,
                       local_flow=dict(attempted=len(state),valid=sum(q['valid'] for q in qualities),
                                       reasons=dict(Counter(q['reason'] for q in qualities)),prior_quality=qualities),
                       internal_retained_tracks=len(next_state)-len(row['candidates']))
            output.write(json.dumps(row,allow_nan=False)+'\n')
            counts.update(frames=1,candidates=len(original),stable=sum(c['stable'] for c in row['candidates']),
                          local_attempted=len(state),local_valid=sum(q['valid'] for q in qualities),resets=int(reset),
                          duplicate_rgb=int(duplicate),missing_depth=int(not row['depth_available']),
                          candidate_empty=int(not row['candidates']))
            state=next_state; prev_gray=gray; prev_t=t
    summary.update(complete=True,unique_frame_keys=len(seen),unchanged_candidates=unchanged,
                   wall_s=time.perf_counter()-start,checks=dict(unique_keys=True,unchanged_candidates=True,
                   independent_timestamps=True,stable_window_or_chain=True,no_internal_track_export=True,
                   repeated_image_no_increment=duplicate_checks))
    (OUT/(stem+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    if '--camera-chain' in sys.argv: camera_clock(chain=True)
    elif '--camera-clock' in sys.argv: camera_clock()
    else: main()
