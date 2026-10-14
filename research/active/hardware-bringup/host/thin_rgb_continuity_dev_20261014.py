"""Label-blind full-RGB time ablation and observed supporting-line continuity.

No weak-label/scene class/ToF/metric distance input. Existing autonomous ridge
candidate geometry is copied exactly. FLOW_SUPPORTED uses current real pixels;
COASTED is internal state only. Neither supplies object purity or safety truth.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from atom_camera import AtomCamera, ROOT
from thin_structure_candidates_dev_20261013 import line_parts, motion
from thin_structure_local_track_dev_20261013 import local_predictions, associate

IN = ROOT/'artifacts.local/work/thin-continuity-dev-20261014/input'
OUT = ROOT/'artifacts.local/work/thin-continuity-dev-20261014/tracking'
CONFIG = dict(max_clock_gap_s=.5,max_coast_s=.5,max_flow_only_s=2.,normal_distance_px=14.,
              direction_deg=15.,appearance_weight=3.,reference_y=320.,reference_x=240.,
              fb_px=1.5,min_flow_points=6,min_flow_span=.3,max_flow_fit_residual_px=1.5,
              min_flow_profile_similarity=.5,ambiguity_cost_margin=.75,
              max_reacquire_gap_s=2.,reacquire_profile_similarity=.8,reacquire_normal_px=6.,
              reacquire_direction_deg=8.,reacquire_unique_margin=2.)
GROUP_CONFIG=dict(direction_deg=3.,normal_distance_px=2.,min_overlap_px=10.,
                  min_shorter_overlap=.5,min_profile_similarity=.8,linkage='complete')


def read(path):
    return [json.loads(q) for q in path.read_text(encoding='utf-8-sig').splitlines() if q.strip()]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def geom(c):return line_parts([c['x1'],c['y1'],c['x2'],c['y2']])


def with_segment(c,p):
    c=dict(c);c.update(x1=float(p[0,0]),y1=float(p[0,1]),x2=float(p[1,0]),y2=float(p[1,1]));return c


def line_reference(c,k):
    p,m,u,length=geom(c);n=np.array([-u[1],u[0]]);rho=float(n@m)
    if abs(u[1])>=abs(u[0]):
        y=CONFIG['reference_y'];x=float(p[0,0]+(y-p[0,1])*u[0]/u[1]);inside=min(p[:,1])<=y<=max(p[:,1])
        bearing=float(np.rad2deg(np.arctan((x-k[0,2])/k[0,0])))
        axis='x_at_y';ref=y;value=x
    else:
        x=CONFIG['reference_x'];y=float(p[0,1]+(x-p[0,0])*u[1]/u[0]);inside=min(p[:,0])<=x<=max(p[:,0])
        bearing=float(np.rad2deg(np.arctan((y-k[1,2])/k[1,1])))
        axis='y_at_x';ref=x;value=y
    support=c.get('supporting_subsegments',c.get('members'))
    current_support=None
    if support is not None:
        current_support=any((min(q['y1'],q['y2'])<=ref<=max(q['y1'],q['y2'])) if axis=='x_at_y' else (min(q['x1'],q['x2'])<=ref<=max(q['x1'],q['x2'])) for q in support)
    return dict(normal_xy=n.tolist(),normal_offset_px=rho,direction_xy=u.tolist(),
                axis=axis,reference_coordinate_px=ref,line_coordinate_px=value,bearing_deg=bearing,
                reference_inside_visible_segment=bool(inside),
                reference_inside_current_member_support=current_support,
                validity='IN_VISIBLE_SEGMENT' if inside else 'EXTRAPOLATED_LINE_ONLY',
                scope='supporting-line proxy at common image reference; not a physical point or TTC')


def appearance(gray,c):
    p,m,u,length=geom(c);n=np.array([-u[1],u[0]])
    centers=p[0]+np.linspace(.1,.9,13)[:,None]*(p[1]-p[0])
    offsets=np.arange(-10,11,dtype=float)
    xy=centers[:,None,:]+offsets[None,:,None]*n
    valid=((xy[:,:,0]>=1)&(xy[:,:,0]<gray.shape[1]-2)&(xy[:,:,1]>=1)&(xy[:,:,1]<gray.shape[0]-2)).all(1)
    if valid.sum()<4:return dict(valid=False,reason='few_valid_cross_sections',profile=[])
    val=cv2.remap(gray,xy[:,:,0].astype('float32'),xy[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
    profile=np.median(val[valid],axis=0);gradient=np.diff(profile);energy=float(np.linalg.norm(gradient))
    if energy<3:return dict(valid=False,reason='flat_normal_profile',profile=[],energy=energy)
    return dict(valid=True,profile=(gradient/energy).tolist(),energy=energy,sections=int(valid.sum()))


def profile_similarity(a,b):
    if not a.get('valid') or not b.get('valid'):return None
    x=np.asarray(a['profile']);y=np.asarray(b['profile'])
    # A one-pixel subpixel ridge shift must not invert a cross-section identity.
    sims=[float(x@y)]
    for shift in (-1,1):
        xx=x[1:] if shift>0 else x[:-1];yy=y[:-1] if shift>0 else y[1:]
        sims.append(float(xx@yy/max(np.linalg.norm(xx)*np.linalg.norm(yy),1e-8)))
    return max(sims)


def group_candidate_families(candidates,gray):
    """Complete-link families of overlapping duplicate line fragments.

    No transitive bridge, no spatial gap fill, no physical-object assertion.
    Original member segments and ranks are preserved for evaluation.
    """
    apps=[appearance(gray,c) for c in candidates];groups=[]
    def compatible(i,j):
        p,m,u,length=geom(candidates[i]);q,n,v,qlen=geom(candidates[j])
        angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),0,1))))
        if angle>GROUP_CONFIG['direction_deg']:return False
        lo=max(min(p@u),min(q@u));hi=min(max(p@u),max(q@u));overlap=max(0.,hi-lo)
        if overlap<GROUP_CONFIG['min_overlap_px'] or overlap/max(1,min(length,qlen))<GROUP_CONFIG['min_shorter_overlap']:return False
        # Compare at the shared projected support, avoiding midpoint offsets
        # of differently cropped visible intervals and rejecting nearby lines.
        s=(lo+hi)/2;point=p[0]+u*(s-p[0]@u)
        normal=np.array([-v[1],v[0]]);distance=abs(float((point-q[0])@normal))
        sim=profile_similarity(apps[i],apps[j])
        return distance<=GROUP_CONFIG['normal_distance_px'] and sim is not None and sim>=GROUP_CONFIG['min_profile_similarity']
    for j in range(len(candidates)):
        matched=next((g for g in groups if all(compatible(i,j) for i in g)),None)
        if matched is None:groups.append([j])
        else:matched.append(j)
    families=[]
    for fid,indices in enumerate(groups,1):
        longest=max(indices,key=lambda i:geom(candidates[i])[3]);p,m,u,length=geom(candidates[longest]);n=np.array([-u[1],u[0]])
        mids=np.array([geom(candidates[i])[1] for i in indices]);rho=float(np.median(mids@n))
        endpoints=np.concatenate([geom(candidates[i])[0] for i in indices]);lo=float(np.min(endpoints@u));hi=float(np.max(endpoints@u))
        rep=np.array([u*lo+n*rho,u*hi+n*rho])
        members=[];states=[];intervals=[]
        for i in indices:
            c=copy.deepcopy(candidates[i]);c['source_candidate_rank']=c['rank'];members.append(c)
            states.append(dict(c,_appearance=apps[i]))
            pp=geom(c)[0];intervals.append(dict(member_rank=c['rank'],lo=float(min(pp@u)),hi=float(max(pp@u)),segment=[c[k] for k in ('x1','y1','x2','y2')]))
        family=dict(x1=float(rep[0,0]),y1=float(rep[0,1]),x2=float(rep[1,0]),y2=float(rep[1,1]),
            rank=fid,family_id_within_frame=fid,members=members,support_intervals=intervals,
            family_member_ranks=[c['rank'] for c in members],source='CURRENT_OVERLAPPING_AUTONOMOUS_FRAGMENT_FAMILY',
            family_scope='structure hypothesis; representative line for association only; target support is member segments',
            _member_current=states,_appearance=appearance(gray,dict(x1=rep[0,0],y1=rep[0,1],x2=rep[1,0],y2=rep[1,1])))
        families.append(family)
    return families


def family_predictions(prevgray,gray,state,fallback):
    predictions,qualities=support_predictions(prevgray,gray,state,fallback)
    if not state:return predictions,qualities
    flat=[];bounds=[]
    for c in state:
        members=c['_member_current'];bounds.append((len(flat),len(flat)+len(members)));flat.extend(members)
    mp,mq=support_predictions(prevgray,gray,flat,fallback)
    for i,(lo,hi) in enumerate(bounds):
        predictions[i]['_member_current']=mp[lo:hi]
        sub=[]
        for p,q in zip(mp[lo:hi],mq[lo:hi]):
            if not q.get('pixel_supported'):continue
            sub.append(dict(x1=p['x1'],y1=p['y1'],x2=p['x2'],y2=p['y2'],
                member_anchor_source_candidate_id=p.get('source_candidate_id'),member_anchor_rank=p['rank'],
                current_rgb_support=q,scope='current per-member RGB point-fit segment; object purity UNKNOWN'))
        qualities[i]['supporting_subsegments']=sub
        qualities[i]['pixel_supported']=bool(qualities[i].get('pixel_supported') and sub)
    return predictions,qualities


def support_predictions(prevgray,gray,state,fallback):
    if not state:return [],[]
    points=[];groups=[]
    for c in state:
        p,m,u,length=geom(c);h,w=gray.shape
        lo=np.maximum(0,np.floor(p.min(0)-8)).astype(int);hi=np.minimum([w,h],np.ceil(p.max(0)+9)).astype(int)
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];n=np.array([-u[1],u[0]])
        mask=(np.abs((xx-m[0])*n[0]+(yy-m[1])*n[1])<=5).astype('uint8')*255
        roi=prevgray[lo[1]:hi[1],lo[0]:hi[0]]
        corners=cv2.goodFeaturesToTrack(roi,20,.01,3,mask=mask) if roi.size else None
        pts=np.empty((0,2),np.float32) if corners is None else corners[:,0]+lo
        pts=np.r_[pts,p[0]+np.linspace(.05,.95,11)[:,None]*(p[1]-p[0])].astype('float32')
        groups.append((len(points),len(points)+len(pts)));points.extend(pts)
    a=np.array(points,np.float32).reshape(-1,1,2)
    b,s1,_=cv2.calcOpticalFlowPyrLK(prevgray,gray,a,None,winSize=(15,15),maxLevel=3)
    back,s2,_=cv2.calcOpticalFlowPyrLK(gray,prevgray,b,None,winSize=(15,15),maxLevel=3)
    aa=a[:,0];bb=b[:,0];fb=np.linalg.norm(back[:,0]-aa,axis=1)
    good=(s1[:,0]>0)&(s2[:,0]>0)&np.isfinite(bb).all(1)&(fb<=CONFIG['fb_px'])
    good&=(bb[:,0]>=0)&(bb[:,0]<gray.shape[1])&(bb[:,1]>=0)&(bb[:,1]<gray.shape[0])
    predictions=[];qualities=[]
    for c,(lo,hi) in zip(state,groups):
        p,m,u,length=geom(c);idx=np.flatnonzero(good[lo:hi])+lo
        pred=np.c_[p,np.ones(2)]@fallback.T
        q=dict(valid=False,reason='few_fb_points',seeded=hi-lo,fb_points=len(idx))
        if len(idx)>=CONFIG['min_flow_points']:
            x=aa[idx];y=bb[idx];t=(x-m)@u/max(length,1);design=np.c_[t,np.ones(len(t))]
            weights=np.ones(len(t))
            for _ in range(3):
                coef=np.linalg.lstsq(design*weights[:,None],y*weights[:,None],rcond=None)[0]
                residual=np.linalg.norm(design@coef-y,axis=1);weights=np.minimum(1.,1.5/np.maximum(residual,1e-6))
            keep=residual<=3;span=float(np.ptp(t[keep])) if keep.any() else 0
            q.update(inliers=int(keep.sum()),line_span=span,median_residual_px=float(np.median(residual)),
                     median_fb_px=float(np.median(fb[idx])),reason='incoherent_or_short_line_span')
            if keep.sum()>=CONFIG['min_flow_points'] and span>=CONFIG['min_flow_span'] and np.median(residual)<=CONFIG['max_flow_fit_residual_px']:
                ends=(p-m)@u/max(length,1);pred=np.c_[ends,np.ones(2)]@coef
                if np.linalg.norm(pred[1]-pred[0])>=10:
                    q.update(valid=True,reason='local_supporting_line_fit')
        pred=np.clip(pred,[0,0],[gray.shape[1]-1,gray.shape[0]-1])
        prediction=with_segment(c,pred);currentapp=appearance(gray,prediction)
        sim=profile_similarity(c['_appearance'],currentapp)
        q.update(profile_similarity=sim,current_profile_valid=currentapp['valid'],
                 pixel_supported=bool(q['valid'] and sim is not None and sim>=CONFIG['min_flow_profile_similarity']))
        predictions.append(prediction);qualities.append(q)
    return predictions,qualities


def continuous_association(candidates,state,predictions,qualities,gray,next_id):
    currentapps=[appearance(gray,c) for c in candidates];nr,nc=len(state),len(candidates)
    costs=np.full((nr,nc),1e6);details={}
    for i,(old,pred,q) in enumerate(zip(state,predictions,qualities)):
        p,m,u,length=geom(pred)
        for j,c in enumerate(candidates):
            cp,cm,v,clen=geom(c);angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),0,1))))
            if angle>CONFIG['direction_deg']:continue
            normal=np.array([-u[1],u[0]]);distance=abs(float((cm-m)@normal))
            if distance>CONFIG['normal_distance_px']:continue
            overlap=max(0.,min(max(p@u),max(cp@u))-max(min(p@u),min(cp@u)))/max(1.,min(length,clen))
            sim=profile_similarity(old['_appearance'],currentapps[j])
            # No endpoint overlap is admissible only with measured local pixel
            # support AND an appearance match; proximity alone cannot join it.
            if overlap<.1 and not (q.get('pixel_supported') and sim is not None and sim>=.5):continue
            if sim is not None and sim<-.2:continue
            appcost=1. if sim is None else 1.-sim
            cost=distance+.2*angle+CONFIG['appearance_weight']*appcost
            costs[i,j]=cost
            details[i,j]=dict(normal_distance_px=distance,direction_delta_deg=angle,visible_overlap=overlap,
                              profile_similarity=sim,cost=cost,flow=q)
    links={}
    if nr and nc:
        ii,jj=linear_sum_assignment(costs)
        for i,j in zip(ii,jj):
            if costs[i,j]>=1e5:continue
            detail=details[i,j];alternatives=[]
            for oi in range(nr):
                if oi!=i and costs[oi,j]<1e5:alternatives.append((float(costs[oi,j]),'other_track',oi))
            for oj in range(nc):
                if oj!=j and costs[i,oj]<1e5:alternatives.append((float(costs[i,oj]),'other_candidate',oj))
            best_alt=min(alternatives,default=None)
            margin=None if best_alt is None else best_alt[0]-float(costs[i,j])
            detail.update(previous_track_id=state[i]['track_id'],previous_state_index=int(i),
                          best_competitor=None if best_alt is None else dict(cost=best_alt[0],kind=best_alt[1],index=best_alt[2]),
                          competitor_margin=margin,ambiguous=bool(margin is not None and margin<CONFIG['ambiguity_cost_margin']))
            links[int(j)]=(int(i),detail)
    used=set()
    for j,c in enumerate(candidates):
        if j in links:
            i,d=links[j];used.add(i);old=state[i]
            c.update(track_id=old['track_id'],association=d,
                     identity_status='AMBIGUOUS' if d['ambiguous'] else 'SUPPORTED_CURRENT_CANDIDATE',
                     identity_ambiguous_history=bool(old.get('identity_ambiguous_history') or d['ambiguous']))
        else:
            c.update(track_id=next_id,association=None,identity_status='NEW_AUTONOMOUS_CANDIDATE',identity_ambiguous_history=False);next_id+=1
        c['_appearance']=currentapps[j]
    assert len({c['track_id'] for c in candidates})==len(candidates)
    return next_id,used


def clean(c):return {k:v for k,v in c.items() if not k.startswith('_')}


def reacquire_current(candidates,dormant,predictions,qualities,currentgray,previous_rgb_index,gap_s):
    """Recognize a dormant identity only from unique current image evidence.

    Episodes always split at the gap. No observation is inserted into the gap.
    Indistinguishable alternatives retain the current fresh identity.
    """
    nr,nc=len(dormant),len(candidates);costs=np.full((nr,nc),1e6);detail={}
    for i,(old,pred,q) in enumerate(zip(dormant,predictions,qualities)):
        if not q.get('pixel_supported'):continue
        p,m,u,length=geom(pred)
        for j,c in enumerate(candidates):
            cp,cm,v,clen=geom(c);angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),0,1))))
            normal=np.array([-u[1],u[0]]);dist=abs(float((cm-m)@normal))
            sim=profile_similarity(old['_appearance'],c['_appearance'])
            if angle>CONFIG['reacquire_direction_deg'] or dist>CONFIG['reacquire_normal_px'] or sim is None or sim<CONFIG['reacquire_profile_similarity']:continue
            cost=dist+.2*angle+3*(1-sim);costs[i,j]=cost
            detail[i,j]=dict(cost=cost,normal_distance_px=dist,direction_delta_deg=angle,profile_similarity=sim,
                             pixel_support=q,previous_track_id=old['track_id'],previous_rgb_index=previous_rgb_index,
                             gap_s=gap_s,previous_segment=[old[k] for k in ('x1','y1','x2','y2')],
                             current_segment=[c[k] for k in ('x1','y1','x2','y2')],source='DORMANT_ID_CURRENT_RGB_CANDIDATE_AND_FB_PROFILE')
    events=[]
    if not nr or not nc:return events
    ii,jj=linear_sum_assignment(costs)
    for i,j in zip(ii,jj):
        if costs[i,j]>=1e5:continue
        d=detail[i,j];other=[costs[oi,j] for oi in range(nr) if oi!=i and costs[oi,j]<1e5]
        other+=[costs[i,oj] for oj in range(nc) if oj!=j and costs[i,oj]<1e5]
        margin=None if not other else float(min(other)-costs[i,j])
        accepted=margin is None or margin>=CONFIG['reacquire_unique_margin']
        d.update(unique_margin=margin,accepted=bool(accepted),candidate_index=int(j),
                 current_fresh_track_id=candidates[j]['track_id'],current_track_id=int(dormant[i]['track_id']) if accepted else candidates[j]['track_id'],
                 episode_continuous=False,gap_observations_added=0,
                 status='REACQUIRED_CURRENT_CANDIDATE' if accepted else 'AMBIGUOUS_REACQUISITION_NEW_ID')
        if accepted:
            old=dormant[i];candidates[j].update(track_id=old['track_id'],identity_status=d['status'],
                identity_ambiguous_history=bool(old.get('identity_ambiguous_history')),reacquisition=d,
                association=dict(previous_state_index=None,previous_track_id=old['track_id'],source=d['source'],reacquisition=True),
                _reacquired_old=old,_episode_id=old.get('_episode_id',0)+1)
        else:candidates[j]['reacquisition']=d
        events.append(d)
    assert len({c['track_id'] for c in candidates})==len(candidates)
    return events


def run(device_primary=False,reacquire=False,grouped=False):
    start=time.perf_counter();cv2.setNumThreads(2);cv2.setRNGSeed(0);OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((IN/'manifest.json').read_text(encoding='utf8'))
    producer_receipt=json.loads((IN/'receipt.json').read_text(encoding='utf8'))
    assert producer_receipt.get('complete'),producer_receipt
    source=IN/'frames.jsonl';rows=read(source);assert len(rows)==654,len(rows)
    cam=AtomCamera();k=cam.K_upright
    summary=dict(config=CONFIG,frames=len(rows),source_sha256=sha(source),source_code_sha256=sha(Path(__file__)),
                 grouping=GROUP_CONFIG if grouped else None,
                 clock_policy='device capture primary; receipt gap flagged' if device_primary else 'strict device and receipt gaps',
                 identity_gap_policy='unique current-pixel dormant reacquisition; episode always split' if reacquire else 'identity reset at capture gap',
                 backend='TASK_NOT_GPU_SUITABLE: CPU sparse LK + small one-to-one assignment',gpu_s=0,
                 labels_read=False,sessions={},complete=False)
    states={'grouped':[]} if grouped else ({'reacquire':[]} if reacquire else ({'device_primary':[]} if device_primary else {'baseline':[],'continuous':[]}))
    nextids={name:1 for name in states};oldkey=None;prev=None;prevgray=None
    handles={name:(OUT/f'tracking_{name}.jsonl').open('w',encoding='utf8') for name in states}
    try:
        for row in rows:
            if time.perf_counter()-start>600:raise TimeoutError('600s execution reserve')
            key=row['key'];t=int(row['device_capture_ns']);rt=int(row['received_elapsed_ns'])
            if key!=oldkey:
                states={name:[] for name in states};nextids={name:1 for name in states};prev=None;prevgray=None;oldkey=key
                summary['sessions'][key]={name:Counter() for name in states}
            gray=cv2.cvtColor(cam.undistort_upright(cv2.imread(row['rgb_file'])),cv2.COLOR_BGR2GRAY)
            duplicate=prev is not None and t==prev['device_capture_ns']
            dt=None if prev is None else (t-prev['device_capture_ns'])/1e9
            dr=None if prev is None else (rt-prev['received_elapsed_ns'])/1e9
            reset=prev is None or row['boot_id']!=prev['boot_id'] or dt<0 or dr<0 or dt>.5 or (dr>.5 and not device_primary)
            dormant=[];dormantgray=None
            if reacquire and reset and prev is not None and row['boot_id']==prev['boot_id'] and .5<dt<=CONFIG['max_reacquire_gap_s'] and dr>=0:
                dormant=[copy.deepcopy(c) for c in states[next(iter(states))] if c['_last_observed_ns']==prev['device_capture_ns']]
                dormantgray=prevgray
            if reset:states={name:[] for name in states};prevgray=None
            fallback,globalq=motion(prevgray,gray) if not duplicate else (np.eye(2,3),dict(valid=False,reason='duplicate'))
            for name in states:
                state=[c for c in states[name] if 0<=t-c['_last_observed_ns']<=500000000]
                rawcurrent=copy.deepcopy(row['candidates'])
                for c in rawcurrent:c['source_candidate_id']=f'{key}:{row["rgb_index"]}:{c["rank"]}'
                current=group_candidate_families(rawcurrent,gray) if grouped else copy.deepcopy(row['candidates'])
                flowobs=[];reacquisitionevents=[]
                if duplicate:
                    predictions=[dict(c) for c in state];qualities=[dict(valid=False,pixel_supported=False,reason='duplicate_no_motion') for c in state]
                elif name=='baseline':
                    pp,qualities=local_predictions(prevgray,gray,state,fallback)
                    predictions=[with_segment(c,p[0]) for c,p in zip(state,pp)]
                elif grouped:predictions,qualities=family_predictions(prevgray,gray,state,fallback)
                else:predictions,qualities=support_predictions(prevgray,gray,state,fallback)
                if name=='baseline':
                    geo=[geom(c) for c in predictions]
                    nextids[name]=associate(current,state,geo,qualities,nextids[name])
                    used=set()
                    for c in current:
                        link=c['association']
                        if link is None:c.update(identity_status='NEW_AUTONOMOUS_CANDIDATE',identity_ambiguous_history=False)
                        else:
                            used.add(link['previous_rank']-1)
                            c.update(identity_status='BASELINE_GREEDY_ASSOCIATION',identity_ambiguous_history=False)
                            link['previous_state_index']=link['previous_rank']-1
                else:nextids[name],used=continuous_association(current,state,predictions,qualities,gray,nextids[name])
                if reacquire and dormant and current:
                    dp,dq=(family_predictions if grouped else support_predictions)(dormantgray,gray,dormant,np.eye(2,3))
                    reacquisitionevents=reacquire_current(current,dormant,dp,dq,gray,prev['rgb_index'],dt)
                nextstate=[]
                for j,c in enumerate(current):
                    link=c.get('association');pi=link['previous_state_index'] if link is not None else None
                    ancestor=c.pop('_reacquired_old',None)
                    if ancestor is None and pi is not None:ancestor=state[pi]
                    history=[] if ancestor is None else list(ancestor['_hit_times'])
                    reacquired=bool(c.get('reacquisition',{}).get('accepted'))
                    episodehistory=[] if ancestor is None or reacquired else list(ancestor.get('_episode_hit_times',ancestor['_hit_times']))
                    new=not duplicate and t not in history
                    if new:history.append(t)
                    if not duplicate and t not in episodehistory:episodehistory.append(t)
                    episodestable=len(episodehistory)>=3 and c['identity_status']!='AMBIGUOUS'
                    c.update(object_independent_hits=len(history),object_established=len(history)>=3,
                        episode_independent_hits=len(episodehistory),episode_stable=episodestable,
                        independent_hits=len(episodehistory),hits=len(episodehistory),stable=episodestable,
                        counted_new_image=new,observation_kind='CURRENT_CANDIDATE_FAMILY' if grouped else 'CURRENT_AUTONOMOUS_CANDIDATE',
                        source_candidate_id=f'{key}:{row["rgb_index"]}:{c["rank"]}',line_reference=line_reference(c,k),
                        source_rgb_index=row['rgb_index'],source_frame_id=row['source_frame_id'],
                        episode_id=c.get('_episode_id',0 if ancestor is None else ancestor.get('_episode_id',0)))
                    c.update(_hit_times=history,_last_observed_ns=t,_last_autonomous_ns=t,
                             _appearance=c.get('_appearance',appearance(gray,c)),_episode_id=c['episode_id'],_episode_hit_times=episodehistory)
                    nextstate.append(c)
                for i,(old,pred,q) in enumerate(zip(state,predictions,qualities)):
                    if i in used:continue
                    observed=name!='baseline' and not duplicate and q.get('pixel_supported') and 0<=t-old['_last_autonomous_ns']<=int(CONFIG['max_flow_only_s']*1e9)
                    carried=dict(pred);carried['_appearance']=old['_appearance']
                    if observed:
                        history=list(old['_hit_times'])
                        if t not in history:history.append(t)
                        episodehistory=list(old.get('_episode_hit_times',old['_hit_times']))
                        if t not in episodehistory:episodehistory.append(t)
                        if grouped:carried['supporting_subsegments']=q['supporting_subsegments']
                        carried.update(_hit_times=history,_episode_hit_times=episodehistory,_last_observed_ns=t,
                            object_independent_hits=len(history),object_established=len(history)>=3,
                            episode_independent_hits=len(episodehistory),episode_stable=False,
                            independent_hits=len(episodehistory),hits=len(episodehistory),stable=False,
                            counted_new_image=True,observation_kind='FLOW_SUPPORTED',
                            identity_status='UNKNOWN_FLOW_OBJECT_PURITY',flow_support=q,
                            association=dict(previous_track_id=old['track_id'],source='CURRENT_RGB_LOCAL_FLOW',
                                             autonomous_candidate_present=False,object_purity='UNKNOWN'),
                            line_reference=line_reference(carried,k),source_rgb_index=row['rgb_index'],source_frame_id=row['source_frame_id'],
                            appearance_validity='matched existing normalized line-normal profile; object purity UNKNOWN')
                        if grouped:
                            carried['supporting_subsegments']=q['supporting_subsegments']
                            carried['family_scope']='only supporting_subsegments may enter target geometry evaluation; representative span is not filled observation'
                        carried['candidate_metadata_scope']='LAST_AUTONOMOUS_CANDIDATE_PROVENANCE_NOT_CURRENT_DEPTH_SUPPORT'
                        carried['current_depth_support']='UNKNOWN_NOT_TESTED'
                        carried['geometry_scope']='CURRENT_PIXEL_LINE_FIT; PROPAGATED_VISIBLE_REFERENCE_ENDPOINTS'
                        flowobs.append(clean(carried))
                    # COASTED is an unobserved prediction, never exported or counted.
                    nextstate.append(carried)
                assert len({c['track_id'] for c in nextstate})==len(nextstate)
                export=[clean(c) for c in current]
                original_fields=set(row['candidates'][0]) if row['candidates'] else set()
                if not grouped:assert all(all(a[field]==b[field] for field in original_fields) for a,b in zip(row['candidates'],export))
                if grouped:
                    familyexport=export;export=rawcurrent
                    membership={rank:f for f in familyexport for rank in f['family_member_ranks']}
                    for c in export:
                        f=membership[c['rank']]
                        c.update(member_observation_id=c['source_candidate_id'],track_id=c['source_candidate_id'],
                            candidate_family_id=f['family_id_within_frame'],family_track_id=f['track_id'],object_id=f['track_id'],
                            episode_id=f['episode_id'],independent_hits=f['independent_hits'],hits=f['independent_hits'],stable=f['stable'],
                            object_independent_hits=f['object_independent_hits'],object_established=f['object_established'],
                            episode_independent_hits=f['episode_independent_hits'],episode_stable=f['episode_stable'],
                            counted_new_image=f['counted_new_image'],observation_kind='CURRENT_AUTONOMOUS_FAMILY_MEMBER',
                            identity_status=f['identity_status'],family_membership_scope='overlapping structure hypothesis, not physical object truth')
                result={key:val for key,val in row.items() if key!='candidates'}
                result.update(arm=name,candidates=export,flow_observations=flowobs,temporal_reset=reset,duplicate_capture=duplicate,
                    device_dt_s=dt,receipt_dt_s=dr,global_motion=globalq,
                    receipt_transport_gap=bool(dr is not None and dr>.5 and dt is not None and 0<dt<=.5),
                    clock_policy=summary['clock_policy'],
                    internal_coasted_tracks=len(nextstate)-len(current)-len(flowobs),
                    reacquisition_edges=reacquisitionevents,
                    observation_contract='current autonomous candidates or separately identified measured RGB flow; unobserved prediction is not exported')
                if grouped:result['candidate_families']=familyexport
                handles[name].write(json.dumps(result,allow_nan=False)+'\n')
                counts=summary['sessions'][key][name]
                counts.update(frames=1,candidates=len(current),new_tracks=sum(c.get('association') is None for c in current),
                    autonomous_observations=sum(c['counted_new_image'] for c in current),flow_observations=len(flowobs),
                    ambiguous=sum(c['identity_status']=='AMBIGUOUS' for c in current),resets=int(reset),duplicate=int(duplicate),
                    coasted=len(nextstate)-len(current)-len(flowobs),stable=sum(c['stable'] for c in current))
                counts.update(reacquisition_accepted=sum(e['accepted'] for e in reacquisitionevents),
                              reacquisition_ambiguous=sum(not e['accepted'] for e in reacquisitionevents))
                if grouped:counts.update(original_member_candidates=len(rawcurrent),families=len(current))
                states[name]=nextstate
            prev=row;prevgray=gray
        summary['complete']=True
    finally:
        for handle in handles.values():handle.close()
        summary['wall_s']=time.perf_counter()-start
        receipt='tracking_receipt_grouped.json' if grouped else ('tracking_receipt_reacquire.json' if reacquire else ('tracking_receipt_device_primary.json' if device_primary else 'tracking_receipt.json'))
        save(receipt,summary);print(json.dumps(summary,indent=2))


def controls():
    """No scene/target labels: endpoint cropping and ambiguous identity fixture."""
    cv2.setNumThreads(2);rng=np.random.default_rng(2)
    gray=cv2.GaussianBlur(rng.integers(0,40,(256,256),dtype=np.uint8),(3,3),0)
    shaft=200+rng.integers(0,40,(256,7),dtype=np.uint8)
    gray[:,77:84]=shaft
    prior=dict(x1=80.,y1=20.,x2=80.,y2=160.,rank=1,track_id=1,hits=3,
               _appearance=appearance(gray,dict(x1=80.,y1=20.,x2=80.,y2=160.)))
    current=[dict(x1=80.,y1=130.,x2=80.,y2=235.,rank=1)]
    predictions,q=support_predictions(gray,gray,[prior],np.eye(2,3))
    # The crop has no sufficient baseline endpoint overlap; measured support
    # and shared supporting line can still preserve identity in the new arm.
    assert q[0]['pixel_supported'],q
    nid,used=continuous_association(current,[prior],predictions,q,gray,2)
    assert current[0]['track_id']==1 and len(used)==1,(current,used,q)
    # Two indistinguishable prior line hypotheses compete for one current line.
    duplicated=[dict(prior),dict(prior,track_id=2)]
    competing=[dict(x1=80.,y1=130.,x2=80.,y2=235.,rank=1)]
    _,used=continuous_association(competing,duplicated,[dict(prior),dict(prior)],[dict(q[0]),dict(q[0])],gray,3)
    assert len(used)==1 and competing[0]['identity_status']=='AMBIGUOUS',competing
    # Driver timestamp rule is explicitly tested; a duplicate supplies no hit,
    # and the only retained state has a <=0.5 s last-observed expiry.
    history=[1000000000];duplicate=1000000000
    if duplicate not in history:history.append(duplicate)
    assert len(history)==1
    expired=not 0<=1600000001-history[-1]<=500000000
    assert expired
    _,blankq=support_predictions(gray,np.zeros_like(gray),[prior],np.eye(2,3))
    assert not blankq[0]['pixel_supported'],blankq
    reacq=[dict(x1=80.,y1=130.,x2=80.,y2=235.,rank=1,track_id=3,_appearance=appearance(gray,current[0]))]
    revents=reacquire_current(reacq,[prior],predictions,q,gray,0,1.)
    assert revents and revents[0]['accepted'] and reacq[0]['track_id']==1 and revents[0]['gap_observations_added']==0,revents
    competingreacq=[dict(x1=80.,y1=130.,x2=80.,y2=235.,rank=1,track_id=3,_appearance=appearance(gray,current[0]))]
    ambiguousrevents=reacquire_current(competingreacq,duplicated,[dict(predictions[0]),dict(predictions[0])],[dict(q[0]),dict(q[0])],gray,0,1.)
    assert ambiguousrevents and not ambiguousrevents[0]['accepted'] and competingreacq[0]['track_id']==3,ambiguousrevents
    cropgroup=group_candidate_families([dict(x1=80.,y1=20.,x2=80.,y2=160.,rank=1),dict(x1=80.,y1=100.,x2=80.,y2=210.,rank=2)],gray)
    assert len(cropgroup)==1 and len(cropgroup[0]['members'])==2,cropgroup
    parallelgroup=group_candidate_families([dict(x1=80.,y1=20.,x2=80.,y2=160.,rank=1),dict(x1=84.,y1=20.,x2=84.,y2=160.,rank=2)],gray)
    assert len(parallelgroup)==2,parallelgroup
    crossinggroup=group_candidate_families([dict(x1=80.,y1=20.,x2=80.,y2=160.,rank=1),dict(x1=20.,y1=100.,x2=140.,y2=100.,rank=2)],gray)
    assert len(crossinggroup)==2,crossinggroup
    supportref=line_reference(dict(x1=80.,y1=20.,x2=80.,y2=400.,
        supporting_subsegments=[dict(x1=80.,y1=20.,x2=80.,y2=160.)]),AtomCamera().K_upright)
    assert supportref['reference_inside_visible_segment'] and not supportref['reference_inside_current_member_support'],supportref
    save('controls.json',dict(status='PASS',endpoint_crop_retained_track=1,endpoint_crop_visible_overlap=30/105,
        competing_identity_status=competing[0]['identity_status'],one_to_one=True,duplicate_hit_count=len(history),
        large_unobserved_gap_expired=expired,blank_frame_not_flow_observation=True,
        unique_current_pixel_reacquisition=True,reacquisition_added_gap_observations=0,
        ambiguous_reacquisition_keeps_fresh_id=True,
        overlapping_cropped_fragments_grouped=True,nearby_parallel_4px_not_grouped=True,crossing_not_grouped=True,
        family_composite_reference_not_falsely_marked_current_pixel_support=True,
        scope='mechanical textured-shaft support with endpoint crop and ambiguous competing hypothesis; not real object correctness',
        flow_quality_fixture=q[0]))


def summarize():
    summaries={};cam=AtomCamera();panelgroups=[]
    for name in ('baseline','continuous','device_primary','reacquire','grouped'):
        if not (OUT/f'tracking_{name}.jsonl').exists():continue
        rows=read(OUT/f'tracking_{name}.jsonl');groups={};obscount=Counter()
        for r in rows:
            for c in r.get('candidate_families',r['candidates'])+r['flow_observations']:
                if not c['counted_new_image']:continue
                groups.setdefault((r['key'],c['track_id']),[]).append((r,c));obscount[c['observation_kind']]+=1
        chains=[]
        for (key,tid),records in groups.items():
            chunks=[];chunk=[]
            for r,c in records:
                if chunk and (r['device_capture_ns']-chunk[-1][0]['device_capture_ns']>500000000 or (name not in ('device_primary','reacquire','grouped') and r['received_elapsed_ns']-chunk[-1][0]['received_elapsed_ns']>500000000)):
                    chunks.append(chunk);chunk=[]
                chunk.append((r,c))
            if chunk:chunks.append(chunk)
            for ci,chunk in enumerate(chunks):
                rr=[r for r,c in chunk];cc=[c for r,c in chunk];ang=[c['line_reference']['bearing_deg'] for c in cc]
                chains.append(dict(key=key,track_id=tid,chain=ci,independent_images=len(chunk),
                    duration_s=(rr[-1]['device_capture_ns']-rr[0]['device_capture_ns'])/1e9,
                    first_rgb_index=rr[0]['rgb_index'],last_rgb_index=rr[-1]['rgb_index'],
                    line_bearing_span_deg=max(ang)-min(ang),first_to_last_line_bearing_deg=ang[-1]-ang[0],
                    source_counts=dict(Counter(c['observation_kind'] for c in cc)),
                    ambiguous_frames=sum(c.get('identity_status')=='AMBIGUOUS' for c in cc),
                    ambiguity_history=any(c.get('identity_ambiguous_history') for c in cc),
                    reference_extrapolated=sum(not c['line_reference']['reference_inside_visible_segment'] for c in cc),
                    observations=[dict(rgb_index=r['rgb_index'],device_capture_ns=r['device_capture_ns'],received_elapsed_ns=r['received_elapsed_ns'],
                        source_candidate_id=c.get('source_candidate_id'),observation_kind=c['observation_kind'],identity_status=c['identity_status'],
                        segment=[c[k] for k in ('x1','y1','x2','y2')],line_reference=c['line_reference'],
                        actual_support_segments=[{k:s[k] for k in ('x1','y1','x2','y2')} for s in c.get('supporting_subsegments',c.get('members',[c]))]) for r,c in chunk]))
        (OUT/f'track_chains_{name}.jsonl').write_text('\n'.join(json.dumps(c) for c in chains)+'\n',encoding='utf8')
        summary={}
        for key in sorted({r['key'] for r in rows}):
            cs=[c for c in chains if c['key']==key]
            summary[key]=dict(frames=sum(r['key']==key for r in rows),chains=len(cs),multiimage_chains=sum(c['independent_images']>=2 for c in cs),
                longest_any_chain_s=max(c['duration_s'] for c in cs),
                note='unlabeled all-structure summary; longest chain is not target correctness')
            # One longest all-structure chain per arm/clip: diagnostic selection
            # is label-blind and may show background, never a target score.
            chosen=max(cs,key=lambda c:c['duration_s']);obs=chosen['observations'];tiles=[]
            for j in (0,len(obs)//2,len(obs)-1):
                o=obs[j];r=next(r for r in rows if r['key']==key and r['rgb_index']==o['rgb_index'])
                im=cam.undistort_upright(cv2.imread(r['rgb_file']));p=np.asarray(o['segment']).astype(int).reshape(2,2)
                for s in o['actual_support_segments']:
                    pp=np.array([s['x1'],s['y1'],s['x2'],s['y2']]).astype(int).reshape(2,2)
                    cv2.line(im,tuple(pp[0]),tuple(pp[1]),(0,255,0),2)
                banner=np.zeros((50,480,3),np.uint8);cv2.putText(banner,f'{name} {key} tid={chosen["track_id"]} rgb={o["rgb_index"]}',(4,21),0,.5,(255,255,255),1)
                cv2.putText(banner,o['observation_kind'],(4,42),0,.45,(255,255,255),1);tiles.append(np.vstack((banner,im)))
            cv2.imwrite(str(OUT/f'longest_unlabeled_{name}_{key}.jpg'),np.hstack(tiles))
        summaries[name]=dict(sessions=summary,observation_counts=dict(obscount))
    save('track_summary.json',summaries);print(json.dumps(summaries,indent=2))


def verify():
    original=read(IN/'frames.jsonl');n=0
    for name in ('baseline','continuous','device_primary','reacquire','grouped'):
        if not (OUT/f'tracking_{name}.jsonl').exists():continue
        rows=read(OUT/f'tracking_{name}.jsonl');assert len(rows)==len(original)==654
        for a,b in zip(original,rows):
            assert (a['key'],a['rgb_index'])==(b['key'],b['rgb_index'])
            assert len(a['candidates'])==len(b['candidates'])
            for old,new in zip(a['candidates'],b['candidates']):
                assert all(old[k]==new[k] for k in old)
                n+=1
            observations=b.get('candidate_families',b['candidates'])+b['flow_observations']
            assert len({c['track_id'] for c in observations})==len(observations)
            if b['duplicate_capture']:assert not any(c['counted_new_image'] for c in b['candidates']) and not b['flow_observations']
            if name=='grouped':
                memberranks=[m['rank'] for f in b['candidate_families'] for m in f['members']]
                assert len(memberranks)==len(set(memberranks))==len(a['candidates'])
                originals={m['rank']:m for m in a['candidates']}
                for f in b['candidate_families']:
                    for member in f['members']:assert all(originals[member['rank']][k]==member[k] for k in originals[member['rank']])
                for f in b['flow_observations']:
                    assert f['supporting_subsegments']
                    assert all(s['current_rgb_support']['pixel_supported'] for s in f['supporting_subsegments'])
            if name in ('device_primary','reacquire','grouped'):
                if b['device_dt_s'] is not None and b['device_dt_s']>.5:
                    assert b['temporal_reset'] and not b['flow_observations']
                if b['receipt_transport_gap']:
                    assert 0<b['device_dt_s']<=.5 and b['receipt_dt_s']>.5 and not b['temporal_reset']
                for e in b.get('reacquisition_edges',[]):
                    assert .5<e['gap_s']<=2. and e['gap_observations_added']==0 and not e['episode_continuous']
                    if e['accepted']:
                        assert e['pixel_support']['pixel_supported'] and e['profile_similarity']>=.8
                        assert e['unique_margin'] is None or e['unique_margin']>=2.
        for chain in read(OUT/f'track_chains_{name}.jsonl'):
            obs=chain['observations'];tt=[o['device_capture_ns'] for o in obs]
            assert len(tt)==len(set(tt))
            assert all(0<q-p<=500000000 for p,q in zip(tt,tt[1:]))
        backup=OUT/f'tracking_initial_{name}.jsonl'
        if backup.exists():
            for old,new in zip(read(backup),rows):
                assert old['candidates']==new['candidates']
                assert len(old['flow_observations'])==len(new['flow_observations'])
                for a,b in zip(old['flow_observations'],new['flow_observations']):
                    assert all(a[k]==b[k] for k in a)
        episodebackup=OUT/f'tracking_pre_episode_{name}.jsonl'
        if episodebackup.exists():
            allowed={'object_independent_hits','object_established','episode_independent_hits','episode_stable','independent_hits','hits','stable','count_semantics'}
            for old,new in zip(read(episodebackup),rows):
                assert all(old[k]==new[k] for k in old if k not in ('candidates','candidate_families','flow_observations'))
                for kind in ('candidates','candidate_families','flow_observations'):
                    if kind not in old:continue
                    for a,b in zip(old[kind],new[kind]):assert all(a[k]==b[k] for k in a if k not in allowed)
                obs=new.get('candidate_families',new['candidates'])+new['flow_observations']
                for c in obs:
                    assert c['independent_hits']==c['hits']==c['episode_independent_hits']
                    assert c['stable']==c['episode_stable'] and c['object_independent_hits']>=c['episode_independent_hits']
                for edge in new.get('reacquisition_edges',[]):
                    if edge['accepted']:
                        c=next(c for c in obs if c['track_id']==edge['current_track_id'])
                        assert c['episode_independent_hits']==1 and not c['episode_stable']
    assert sha(OUT/'source_initial.py')==json.loads((OUT/'tracking_receipt.json').read_text(encoding='utf8'))['source_code_sha256']
    hashes={p.name:sha(p) for p in OUT.glob('*.json*') if p.name!='verification.json'}
    save('verification.json',dict(status='PASS',frames_each_arm=654,unchanged_candidate_checks=n,source_code_sha256=sha(Path(__file__)),
         input_sha256=sha(IN/'frames.jsonl'),payload_hashes=hashes,one_to_one=True,duplicate_not_counted=True,
         all_exported_observations_are_current_candidates_or_current_rgb_flow=True))


def repair_lineage():
    """Mechanical output metadata repair; never changes geometry or identity.

    Initial files retain exact bytes. Inherited ridge fields belong to the
    ancestor candidate, not to a new measurement of depth at the flow location.
    """
    changes=[]
    for name in ('continuous','device_primary'):
        path=OUT/f'tracking_{name}.jsonl';original=path.read_bytes();backup=OUT/f'tracking_initial_{name}.jsonl'
        if not backup.exists():backup.write_bytes(original)
        rows=read(path);n=0
        for r in rows:
            for c in r['flow_observations']:
                c['candidate_metadata_scope']='LAST_AUTONOMOUS_CANDIDATE_PROVENANCE_NOT_CURRENT_DEPTH_SUPPORT'
                c['current_depth_support']='UNKNOWN_NOT_TESTED'
                c['geometry_scope']='CURRENT_PIXEL_LINE_FIT; PROPAGATED_VISIBLE_REFERENCE_ENDPOINTS'
                n+=1
        path.write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows)+'\n',encoding='utf8')
        changes.append(dict(arm=name,initial_sha256=hashlib.sha256(original).hexdigest(),final_sha256=sha(path),
                            current_flow_metadata_records=n,geometry_identity_clock_unchanged=True))
    save('lineage_repair.json',dict(status='COMPLETE',changes=changes,
          reason='Explicitly distinguish inherited last-autonomous ridge fields from current measured RGB flow; no depth remeasurement or algorithm change'))


def repair_episode_metadata():
    """Separate cumulative object evidence from each gap-delimited episode.

    No inference, association, geometry, identity, episode ID, timestamp or gap
    changes. Pre-repair files and receipts retain exact bytes.
    """
    affected=[]
    for arm in ('reacquire','grouped'):
        path=OUT/f'tracking_{arm}.jsonl';receiptpath=OUT/f'tracking_receipt_{arm}.json'
        backup=OUT/f'tracking_pre_episode_{arm}.jsonl';receiptbackup=OUT/f'tracking_receipt_pre_episode_{arm}.json'
        if not backup.exists():backup.write_bytes(path.read_bytes())
        if not receiptbackup.exists():receiptbackup.write_bytes(receiptpath.read_bytes())
        oldrows=read(backup);rows=copy.deepcopy(oldrows);objects={};episodes={};changed=0;first=[]
        for r in rows:
            observations=r.get('candidate_families',r['candidates'])+r['flow_observations']
            for c in observations:
                obj=(r['key'],c['track_id']);epi=(*obj,int(c.get('episode_id',0)));t=r['device_capture_ns']
                objecttimes=objects.setdefault(obj,set());episodetimes=episodes.setdefault(epi,set())
                if c['counted_new_image']:objecttimes.add(t);episodetimes.add(t)
                count=len(episodetimes);isflow=c['observation_kind']=='FLOW_SUPPORTED'
                stable=bool(count>=3 and not isflow and c['identity_status']!='AMBIGUOUS')
                if (c['independent_hits'],c['hits'],c['stable'])!=(count,count,stable):changed+=1
                c.update(object_independent_hits=len(objecttimes),object_established=len(objecttimes)>=3,
                    episode_independent_hits=count,episode_stable=stable,
                    independent_hits=count,hits=count,stable=stable,
                    count_semantics='hits/stable are current episode only; cumulative object evidence is separately named')
                if any(e['accepted'] and e['current_track_id']==c['track_id'] for e in r.get('reacquisition_edges',[])):
                    assert count==1 and not stable,(r['key'],r['rgb_index'],c)
                    first.append(dict(key=r['key'],rgb_index=r['rgb_index'],track_id=c['track_id'],episode_id=c.get('episode_id'),
                        episode_hits=count,episode_stable=stable,object_hits=len(objecttimes)))
            if arm=='grouped':
                families={c['track_id']:c for c in r['candidate_families']}
                for c in r['candidates']:
                    f=families[c['family_track_id']]
                    for field in ('object_independent_hits','object_established','episode_independent_hits','episode_stable','independent_hits','hits','stable','count_semantics'):
                        c[field]=f[field]
        allowed={'object_independent_hits','object_established','episode_independent_hits','episode_stable','independent_hits','hits','stable','count_semantics'}
        for old,new in zip(oldrows,rows):
            assert all(old[k]==new[k] for k in old if k not in ('candidates','candidate_families','flow_observations'))
            for kind in ('candidates','candidate_families','flow_observations'):
                if kind not in old:continue
                assert len(old[kind])==len(new[kind])
                for a,b in zip(old[kind],new[kind]):assert all(a[k]==b[k] for k in a if k not in allowed)
        path.write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows)+'\n',encoding='utf8')
        receipt=json.loads(receiptbackup.read_text(encoding='utf8'))
        for key in receipt['sessions']:
            stablecount=sum(c['stable'] for r in rows if r['key']==key for c in r.get('candidate_families',r['candidates']))
            receipt['sessions'][key][arm]['stable']=stablecount
        receipt['episode_metadata_repair']=dict(source_code_sha256=sha(Path(__file__)),output_sha256=sha(path),
            pre_repair_output_sha256=sha(backup),geometry_ids_timestamps_gaps_unchanged=True,
            aliases='hits/independent_hits/stable now refer only to current episode')
        receiptpath.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        affected.append(dict(arm=arm,changed_observation_metadata=changed,first_reacquired_observations=first,
            pre_repair_sha256=sha(backup),final_sha256=sha(path),geometry_identity_clock_unchanged=True))
    save('episode_metadata_repair.json',dict(status='COMPLETE',affected=affected,no_new_inference=True,
         reason='Old cumulative object hits made a reacquired episode first image appear stable; current episode count starts at one'))
    print(json.dumps(affected,indent=2))


def reacquisition_pairs():
    cam=AtomCamera()
    for arm in ('reacquire','grouped'):
        if not (OUT/f'tracking_{arm}.jsonl').exists():continue
        rows=read(OUT/f'tracking_{arm}.jsonl');lookup={(r['key'],r['rgb_index']):r for r in rows};pairs=[]
        folder=OUT/f'reacquisition_pairs_{arm}';folder.mkdir(exist_ok=True)
        for r in rows:
          for event in r['reacquisition_edges']:
            before=lookup[r['key'],event['previous_rgb_index']];tiles=[]
            color=(0,255,0) if event['accepted'] else (0,200,255)
            for name,row,segment in (('before',before,event['previous_segment']),('after',r,event['current_segment'])):
                im=cam.undistort_upright(cv2.imread(row['rgb_file']));p=np.asarray(segment).astype(int).reshape(2,2)
                support=None
                if arm=='grouped':
                    if name=='before':
                        observed=next((c for c in row['candidate_families']+row['flow_observations'] if c['track_id']==event['previous_track_id']),None)
                    else:observed=row['candidate_families'][event['candidate_index']]
                    if observed is not None:support=observed.get('supporting_subsegments',observed.get('members'))
                if support is None:cv2.line(im,tuple(p[0]),tuple(p[1]),color,2)
                else:
                    for s in support:
                        pp=np.array([s[k] for k in ('x1','y1','x2','y2')]).astype(int).reshape(2,2)
                        cv2.line(im,tuple(pp[0]),tuple(pp[1]),color,2)
                banner=np.zeros((55,480,3),np.uint8)
                cv2.putText(banner,f'{name} {r["key"]} rgb={row["rgb_index"]}',(4,22),0,.6,(255,255,255),1)
                cv2.putText(banner,f'gap {event["gap_s"]:.3f}s accepted={event["accepted"]}',(4,45),0,.5,(255,255,255),1)
                tiles.append(np.vstack((banner,im)))
            filename=f'{r["key"]}_{r["rgb_index"]:04d}_{event["previous_track_id"]}_{event["candidate_index"]}.jpg'
            cv2.imwrite(str(folder/filename),np.hstack(tiles))
            pairs.append(dict(key=r['key'],rgb_index=r['rgb_index'],file=str(folder/filename),event=event))
        save(f'reacquisition_pairs_{arm}.json',dict(pairs=pairs,accepted=sum(p['event']['accepted'] for p in pairs),
                                       ambiguous=sum(not p['event']['accepted'] for p in pairs),
                                       selection='all proposed current-image reacquisition assignment edges; no labels'))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','run-device','run-reacquire','run-grouped','controls','summarize','verify','repair-lineage','repair-episodes','reacquisition-pairs']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'run':run,'run-device':lambda:run(device_primary=True),'run-reacquire':lambda:run(device_primary=True,reacquire=True),'run-grouped':lambda:run(device_primary=True,reacquire=True,grouped=True),'controls':controls,'summarize':summarize,'verify':verify,'repair-lineage':repair_lineage,'repair-episodes':repair_episode_metadata,'reacquisition-pairs':reacquisition_pairs}[args.stage]()
