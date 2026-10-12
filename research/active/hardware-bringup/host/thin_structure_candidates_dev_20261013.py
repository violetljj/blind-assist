"""Label-free line proposals on consumed Development RGB; not an obstacle detector.

Coordinates are AtomCamera undistorted upright 480x640. Fixed proposal cutpoints
are exploratory mechanism choices, not calibrated safety thresholds. Cached DAV2
availability follows the earlier pose/time gate; absent depth is UNKNOWN.
"""
from __future__ import annotations
import json
import time
import math
from pathlib import Path
import cv2
import numpy as np
from atom_camera import AtomCamera, ROOT

OUT = ROOT / 'artifacts.local/work/thin-structure-dev-20261013'
CACHE = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
RAW = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw'
CONFIG = dict(min_length_px=25, max_edges=160, pair_width_px=[2,18],
              pair_angle_deg=8, min_pair_overlap_px=25, top_k=40,
              depth_cut_iqr=0.02, track_distance_px=14, track_angle_deg=15,
              track_min_overlap=0.3, track_max_gap_s=0.5)

def line_parts(p):
    p = np.asarray(p, float).reshape(2,2)
    v = p[1]-p[0]; length = float(np.linalg.norm(v))
    u = v/max(length,1e-9)
    if u[1]<0 or (abs(u[1])<1e-8 and u[0]<0):
        p = p[::-1].copy(); u = -u
    return p, (p[0]+p[1])/2, u, length

def proposals(gray, disp, lsd, return_all=False):
    found = lsd.detect(gray)[0]
    # LSD's subpixel refinement can extend endpoints beyond the raster edge.
    # Clamp before all scoring/association so exported geometry is in-frame.
    bounds=np.array([gray.shape[1]-1,gray.shape[0]-1])
    edges = [] if found is None else [line_parts(np.clip(q.reshape(2,2),0,bounds)) for q in found[:,0]]
    edges = sorted((e for e in edges if e[3]>=25), key=lambda e:-e[3])[:160]
    raw = [(e[0],0.,'single') for e in edges]
    for a in range(len(edges)):
        p,m,u,L = edges[a]; normal = np.array([-u[1],u[0]])
        for b in range(a+1,len(edges)):
            q,n,v,K = edges[b]
            if abs(float(u@v))<math.cos(math.radians(8)): continue
            width=abs(float((n-m)@normal))
            if not 2<=width<=18: continue
            lo=max(float(p[0]@u),float(np.min(q@u)))
            hi=min(float(p[1]@u),float(np.max(q@u)))
            if hi-lo<25: continue
            offset=float((m+n)@normal/2)
            raw.append((np.array([u*lo+normal*offset,u*hi+normal*offset]),width,'pair'))
    iqr = None if disp is None else max(float(np.percentile(disp,75)-np.percentile(disp,25)),1e-5)
    candidates=[]
    for p,width,kind in raw:
        p,m,u,L=line_parts(np.clip(p,0,bounds)); normal=np.array([-u[1],u[0]])
        if L<25: continue
        prom=None; rank=None
        if disp is not None:
            t=np.linspace(.1,.9,16)[:,None]; pts=p[0]+t*(p[1]-p[0])
            off=max(6,width/2+4)
            strips=np.stack([pts,pts+normal*off,pts-normal*off])
            xx=np.rint(strips[:,:,0]).astype(int); yy=np.rint(strips[:,:,1]).astype(int)
            valid=(xx>=0)&(xx<disp.shape[1])&(yy>=0)&(yy<disp.shape[0])
            valid=valid.all(axis=0)
            if valid.sum()>=8:
                val=disp[yy[:,valid],xx[:,valid]]
                bg=np.maximum(val[1],val[2])
                prom=float(np.median(val[0]-bg)/iqr)
                rank=float(np.mean(val[0]>bg))
        # Positive local prominence is evidence, absent/negative remains a proposal.
        score=float(math.log1p(L)+(0.45 if kind=='pair' else 0)+2*max(0,min(prom or 0,1)))
        candidates.append(dict(x1=float(p[0,0]),y1=float(p[0,1]),x2=float(p[1,0]),y2=float(p[1,1]),
                               length=L,width=float(width),kind=kind,score=score,
                               depth_prominence=prom,depth_rank=rank,depth_available=prom is not None,
                               positive_depth=prom is not None and prom>=.02))
    candidates.sort(key=lambda c:-c['score'])
    stats=dict(edges=len(edges),raw_proposals=len(raw),raw_pairs=sum(c[2]=='pair' for c in raw))
    if return_all:
        for k,c in enumerate(candidates): c['rank_before_dedup']=k+1
        return candidates,stats
    kept=[]
    for c in candidates:
        p,m,u,L=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
        duplicate=False
        for d in kept:
            q,n,v,K=line_parts([d['x1'],d['y1'],d['x2'],d['y2']])
            if abs(u@v)>.99 and np.linalg.norm(m-n)<5 and abs(L-K)<.25*max(L,K):
                duplicate=True; break
        if not duplicate: kept.append(c)
        if len(kept)>=40: break
    for k,c in enumerate(kept): c['rank']=k+1
    return kept,stats

def motion(prev,gray):
    identity=np.array([[1.,0.,0.],[0.,1.,0.]])
    if prev is None: return identity,dict(valid=False,reason='first')
    a=cv2.goodFeaturesToTrack(prev,160,.02,10)
    if a is None or len(a)<8: return identity,dict(valid=False,reason='few_features')
    b,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,a,None,winSize=(21,21),maxLevel=3)
    ok=status[:,0].astype(bool)
    if ok.sum()<8: return identity,dict(valid=False,reason='few_flow')
    matrix,inliers=cv2.estimateAffinePartial2D(a[ok],b[ok],method=cv2.RANSAC,ransacReprojThreshold=3)
    if matrix is None or int(inliers.sum())<8: return identity,dict(valid=False,reason='ransac')
    scale=float(np.linalg.norm(matrix[:,0]))
    if not .8<=scale<=1.25: return identity,dict(valid=False,reason='scale')
    return matrix,dict(valid=True,inliers=int(inliers.sum()),flow_points=int(ok.sum()),matrix=matrix.tolist())

def associate(candidates,previous,matrix,next_id):
    pool=[]
    for c in previous:
        p=np.array([[c['x1'],c['y1'],1],[c['x2'],c['y2'],1]])@matrix.T
        pool.append((c,line_parts(p)))
    used=set()
    for c in candidates:
        p,m,u,L=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
        best=None
        for k,(d,(q,n,v,K)) in enumerate(pool):
            if k in used or abs(u@v)<math.cos(math.radians(15)): continue
            normal=np.array([-u[1],u[0]])
            distance=abs(float((m-n)@normal))
            overlap=max(0,min(max(p@u),max(q@u))-max(min(p@u),min(q@u)))/max(1,min(L,K))
            if distance<=14 and overlap>=.3:
                cost=distance+abs(float((m-n)@u))*.1
                if best is None or cost<best[0]: best=(cost,k,d)
        if best is None:
            c.update(track_id=next_id,hits=1); next_id+=1
        else:
            _,k,d=best; used.add(k); c.update(track_id=d['track_id'],hits=d['hits']+1)
        c['stable']=c['hits']>=3
    return next_id

def main():
    start=time.perf_counter(); cv2.setNumThreads(2); OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((CACHE/'cache_manifest.json').read_text())
    camera=AtomCamera(); lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    summary=dict(config=CONFIG,coordinate_system='undistorted upright x-right y-down 480x640',
                 evidence='consumed Development; label-free proposals; no obstacle or safety verdict',sessions={},gpu_seconds=0)
    try:
        with (OUT/'detection_frames.jsonl').open('w',encoding='utf8') as output:
            for key,session in manifest['sessions'].items():
                folder=RAW/session['session']; rgb=[json.loads(s) for s in (folder/'rgb.jsonl').read_text().splitlines()]
                previous=[]; prev_gray=None; prev_t=None; next_id=1; counts=dict(frames=0,depth_frames=0,candidates=0,positive=0,stable=0,motion_valid=0)
                for row in session['frames']:
                    if time.perf_counter()-start>540: raise TimeoutError('540 second internal reserve reached')
                    image=cv2.imread(str(folder/rgb[row['rgb_index']]['file']))
                    upright=camera.undistort_upright(image)
                    gray=cv2.cvtColor(upright,cv2.COLOR_BGR2GRAY)
                    disp=None
                    if row.get('cache'):
                        with np.load(CACHE/row['cache']) as z: disp=z['disp'].copy()
                    candidates,stats=proposals(gray,disp,lsd)
                    gap=prev_t is None or (row['rgb_t_ns']-prev_t)/1e9>.5 or row['rgb_t_ns']<=prev_t
                    if gap: previous=[]; prev_gray=None
                    matrix,quality=motion(prev_gray,gray)
                    next_id=associate(candidates,previous,matrix,next_id)
                    result=dict(key=key,session=session['session'],i=row['i'],rgb_index=row['rgb_index'],
                                rgb_t_ns=row['rgb_t_ns'],t_ns=row['t_ns'],depth_available=disp is not None,
                                candidates=candidates,proposal_stats=stats,motion=quality,temporal_reset=gap)
                    output.write(json.dumps(result,allow_nan=False)+'\n')
                    counts['frames']+=1; counts['depth_frames']+=disp is not None; counts['candidates']+=len(candidates)
                    counts['positive']+=sum(c['positive_depth'] for c in candidates)
                    counts['stable']+=sum(c['stable'] for c in candidates); counts['motion_valid']+=quality['valid']
                    previous=candidates; prev_gray=gray; prev_t=row['rgb_t_ns']
                summary['sessions'][key]=counts
                print(key,counts,round(time.perf_counter()-start,2),flush=True)
        summary['complete']=True
    finally:
        summary['wall_s']=time.perf_counter()-start
        (OUT/'detection_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

def export_all():
    """Diagnostic proposal upper bound; does not alter top40 or track results."""
    start=time.perf_counter(); cv2.setNumThreads(2)
    manifest=json.loads((CACHE/'cache_manifest.json').read_text())
    camera=AtomCamera(); lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    total=0; count=0
    with (OUT/'detection_all_proposals.jsonl').open('w',encoding='utf8') as out:
        for key,session in manifest['sessions'].items():
            folder=RAW/session['session']; rgb=[json.loads(s) for s in (folder/'rgb.jsonl').read_text().splitlines()]
            for row in session['frames']:
                if time.perf_counter()-start>300: raise TimeoutError('300s proposal diagnostic reserve')
                up=camera.undistort_upright(cv2.imread(str(folder/rgb[row['rgb_index']]['file'])))
                gray=cv2.cvtColor(up,cv2.COLOR_BGR2GRAY); disp=None
                if row.get('cache'):
                    with np.load(CACHE/row['cache']) as z: disp=z['disp'].copy()
                candidates,stats=proposals(gray,disp,lsd,return_all=True)
                out.write(json.dumps(dict(key=key,i=row['i'],rgb_index=row['rgb_index'],
                                          candidates=candidates,proposal_stats=stats),allow_nan=False)+'\n')
                total+=len(candidates); count+=1
            print(key,count,total,round(time.perf_counter()-start,2),flush=True)
    (OUT/'detection_all_summary.json').write_text(json.dumps(dict(frames=count,proposals=total,
        wall_s=time.perf_counter()-start,note='Before dedup/top40; same fixed 160-longest-edge cap; no tracking',gpu_seconds=0),indent=2))

def export_diverse():
    """Fixed direction/spatial coverage control using unchanged proposal scores."""
    start=time.perf_counter(); previous=[]; old_key=None; next_id=1; count=0
    with (OUT/'detection_all_proposals.jsonl').open() as source, (OUT/'detection_frames.jsonl').open() as base, (OUT/'detection_diverse_frames.jsonl').open('w') as out:
        for raw,original in zip(source,base):
            r=json.loads(raw); b=json.loads(original)
            assert (r['key'],r['i'])==(b['key'],b['i'])
            buckets={}
            for c in r['candidates']:
                p,m,u,L=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
                angle=math.atan2(float(u[1]),float(u[0]))%math.pi
                bucket=(min(3,int(angle/math.pi*4)),min(2,int(m[0]/480*3)),min(2,int(m[1]/640*3)))
                if bucket not in buckets: buckets[bucket]=c
            selected=sorted(buckets.values(),key=lambda c:-c['score'])
            ids={id(c) for c in selected}
            for c in r['candidates']:
                if len(selected)>=40: break
                if id(c) in ids: continue
                p,m,u,L=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
                duplicate=False
                for d in selected:
                    q,n,v,K=line_parts([d['x1'],d['y1'],d['x2'],d['y2']])
                    if abs(u@v)>.99 and np.linalg.norm(m-n)<5 and abs(L-K)<.25*max(L,K):
                        duplicate=True; break
                if not duplicate: selected.append(c)
            selected.sort(key=lambda c:-c['score'])
            for k,c in enumerate(selected): c['rank']=k+1
            if r['key']!=old_key: previous=[]; next_id=1
            if b['temporal_reset']: previous=[]
            matrix=np.array(b['motion'].get('matrix',[[1.,0.,0.],[0.,1.,0.]]))
            next_id=associate(selected,previous,matrix,next_id)
            b['candidates']=selected; b['selection']='4 direction x 3x3 midpoint bins, best per occupied bin then score fill to40'
            b['occupied_buckets']=len(buckets); out.write(json.dumps(b,allow_nan=False)+'\n')
            previous=selected; old_key=r['key']; count+=1
    assert count==1677
    (OUT/'detection_diverse_summary.json').write_text(json.dumps(dict(frames=count,wall_s=time.perf_counter()-start,gpu_seconds=0,
        config='4 direction x 3x3 midpoint bins, best per occupied bin then score fill to40; reuse image-only affine'),indent=2))

if __name__=='__main__':
    import sys
    if '--all-proposals' in sys.argv: export_all()
    elif '--diverse' in sys.argv: export_diverse()
    else: main()
