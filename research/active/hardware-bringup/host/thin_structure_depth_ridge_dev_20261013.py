"""Posthoc mechanism test: autonomous relative-disparity ridge proposals.

No labels read. Depth UNKNOWN stays UNKNOWN. Not metric distance or alarm.
"""
import json
import time
import math
import cv2
import numpy as np
from thin_structure_candidates_dev_20261013 import OUT, CACHE, associate, line_parts


def propose(d):
    iqr=max(float(np.percentile(d,75)-np.percentile(d,25)),1e-5)
    candidates=[]
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        response=np.full(d.shape,-np.inf,np.float32)
        for distance in (4,8,12):
            ox,oy=dx*distance,dy*distance
            p=np.roll(d,(oy,ox),(0,1));q=np.roll(d,(-oy,-ox),(0,1))
            r=(d-np.maximum(p,q))/iqr
            r[:abs(oy)+1,:]=-np.inf;r[-abs(oy)-1:,:]=-np.inf
            r[:,:abs(ox)+1]=-np.inf;r[:,-abs(ox)-1:]=-np.inf
            response=np.maximum(response,r)
        mask=(response>=.02).astype(np.uint8)*255
        lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=20,minLineLength=25,maxLineGap=8)
        if lines is None:continue
        expected=np.array([-dy,dx],float);expected/=np.linalg.norm(expected)
        for line in lines[:,0]:
            p,m,u,length=line_parts(line)
            if abs(u@expected)<math.cos(math.radians(15)):continue
            pts=p[0]+np.linspace(0,1,24)[:,None]*(p[1]-p[0]);xy=np.rint(pts).astype(int)
            values=response[xy[:,1],xy[:,0]]
            support=float(np.mean(values>=.02))
            if support<.6:continue
            prominence=float(np.median(values))
            if not np.isfinite(prominence):continue
            candidates.append(dict(x1=float(p[0,0]),y1=float(p[0,1]),x2=float(p[1,0]),y2=float(p[1,1]),length=length,width=0.,kind='ridge',score=float(math.log1p(length)+2*min(max(prominence,0),1)),depth_prominence=prominence,depth_available=True,positive_depth=prominence>=.02,support=support))
    kept=[]
    for c in sorted(candidates,key=lambda c:-c['score']):
        p,m,u,l=line_parts([c['x1'],c['y1'],c['x2'],c['y2']])
        if any(abs(u@line_parts([k['x1'],k['y1'],k['x2'],k['y2']])[2])>.99 and np.linalg.norm(m-line_parts([k['x1'],k['y1'],k['x2'],k['y2']])[1])<12 for k in kept):continue
        c['rank']=len(kept)+1;kept.append(c)
        if len(kept)==40:break
    return kept,len(candidates)


def main():
    start=time.perf_counter();cv2.setNumThreads(2);cv2.setRNGSeed(0)
    manifest=json.loads((CACHE/'cache_manifest.json').read_text())
    original=[json.loads(s) for s in (OUT/'detection_frames.jsonl').read_text().splitlines()]
    lookup={(key,r['i']):r for key,s in manifest['sessions'].items() for r in s['frames']}
    last=None;previous=[];next_id=1;total=0
    with (OUT/'detection_ridge_frames.jsonl').open('w',encoding='utf8') as f:
        for r in original:
            if time.perf_counter()-start>240:raise TimeoutError('ridge budget reserve')
            if last!=r['key']:previous=[];next_id=1
            if r['temporal_reset']:previous=[]
            cache=lookup[r['key'],r['i']].get('cache')
            if cache:
                with np.load(CACHE/cache) as z:cs,n=propose(z['disp'])
            else:cs,n=[],0
            matrix=np.array(r['motion'].get('matrix',[[1.,0.,0.],[0.,1.,0.]]))
            next_id=associate(cs,previous,matrix,next_id)
            r=dict(r,candidates=cs,proposal_stats=dict(ridge_before_top40=n),source='autonomous cached disparity ridge; missing depth -> no proposals')
            f.write(json.dumps(r,allow_nan=False)+'\n');previous=cs;last=r['key'];total+=len(cs)
    summary=dict(frames=len(original),candidates=total,wall_s=time.perf_counter()-start,gpu_s=0,config=dict(side_offsets=[4,8,12],normal_directions=4,minimum_prominence_iqr=.02,hough_threshold=20,min_length=25,max_gap=8,min_support=.6,top_k=40),evidence='posthoc Development autonomous candidate; not annotation initialized')
    (OUT/'ridge_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
