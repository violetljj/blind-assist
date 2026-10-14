"""Generic current RGB appearance and immutable-anchor identity diagnostics.

No labels, target IDs, hand boxes, scene class or metric depth are read. All
autonomous candidates survive identity rejection. Causal output separately
identifies current measured flow and unobserved coast; capture gaps split
episodes even when an object hypothesis is uniquely recognized later.
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
import thin_rgb_continuity_dev_20261014 as C
import thin_appearance_features_dev_20261014 as F
import thin_appearance_track_fast_dev_20261014 as Fast
import thin_appearance_partial_dev_20261014 as Partial

IN=ROOT/'artifacts.local/work/thin-continuity-dev-20261014/input'
FEATURES=ROOT/'artifacts.local/work/thin-appearance-dev-20261014/diagnostic'
OUT=ROOT/'artifacts.local/work/thin-appearance-dev-20261014/tracking'
ARMS=('local_lab','local_strip','anchor_strip')
TAG=''
SESSIONS=None
WALL_CAP=240.
CONFIG=dict(max_capture_gap_s=.5,max_coast_s=.5,max_flow_without_native_s=2.,
    normal_px=14.,direction_deg=15.,max_core_delta_lab=18.,max_contrast_delta_lab=18.,
    min_centered_profile_similarity=.65,ambiguity_margin=.75,
    reacquire_max_gap_s=2.,reacquire_normal_px=6.,reacquire_direction_deg=8.,
    reacquire_unique_margin=2.,anchor_establishment='first native descriptor AVAILABLE; quality is separate',
    missing_appearance='UNKNOWN_NOT_IDENTITY_SUPPORT',lookahead_frames=0)


def read(path):return [json.loads(q) for q in path.read_text(encoding='utf-8-sig').splitlines() if q.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def clean(c):return {k:v for k,v in c.items() if not k.startswith('_')}
def desc_hash(d):return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest() if d is not None else None


def compatible(a,b,arm):
    if arm=='local_lab_partial':
        if a is None or b is None:return dict(accepted=False,status='UNKNOWN',reason='UNKNOWN_APPEARANCE_NO_IDENTITY_SUPPORT')
        if a.get('status')=='PARTIAL_CORE_AVAILABLE' or b.get('status')=='PARTIAL_CORE_AVAILABLE':
            q=Partial.compare_partial_core(b,a)
            if q['status']!='AVAILABLE':return dict(q,accepted=False,reason='UNKNOWN_PARTIAL_CORE_NO_IDENTITY_SUPPORT')
            if q['lab_core_delta']>CONFIG['max_core_delta_lab']:return dict(q,accepted=False,reason='CORE_LAB_INCOMPATIBLE')
            if q['lab_contrast_delta'] is None:return dict(q,accepted=False,status='UNKNOWN',reason='NO_COMMON_OBSERVED_FLANK_CONTRAST')
            if q['lab_contrast_delta']>CONFIG['max_contrast_delta_lab']:return dict(q,accepted=False,reason='CONTRAST_LAB_INCOMPATIBLE')
            return dict(q,accepted=True,reason='PARTIAL_CURRENT_COLOR_AND_COMMON_FLANK_COMPATIBLE')
        arm='local_lab'
    if a is None or b is None or a.get('status')!='AVAILABLE' or b.get('status')!='AVAILABLE':
        return dict(accepted=False,status='UNKNOWN',reason='UNKNOWN_APPEARANCE_NO_IDENTITY_SUPPORT')
    q=Fast.compare(a,b)
    if q.get('status')!='AVAILABLE':return dict(q,accepted=False,reason='UNKNOWN_APPEARANCE_NO_IDENTITY_SUPPORT')
    if q['lab_core_delta']>CONFIG['max_core_delta_lab']:
        return dict(q,accepted=False,reason='CORE_LAB_INCOMPATIBLE')
    if q['lab_contrast_delta']>CONFIG['max_contrast_delta_lab']:
        return dict(q,accepted=False,reason='CONTRAST_LAB_INCOMPATIBLE')
    if arm!='local_lab':
        if arm=='anchor_finite' and a.get('quality',{}).get('stripe_reliable') and not b.get('quality',{}).get('stripe_reliable'):
            return dict(q,accepted=False,status='UNKNOWN',reason='FINITE_ANCHOR_CURRENT_BILATERAL_SUPPORT_UNKNOWN')
        s=q.get('normalized_shape_similarity')
        if s is None:return dict(q,accepted=False,status='UNKNOWN',reason='UNKNOWN_PROFILE_NO_IDENTITY_SUPPORT')
        if s<CONFIG['min_centered_profile_similarity']:return dict(q,accepted=False,reason='CENTERED_PROFILE_INCOMPATIBLE')
        qa=a.get('quality',{});qb=b.get('quality',{})
        if (qa.get('stripe_reliable') and qb.get('strip_state')=='SINGLE_STEP_LIKE') or (qb.get('stripe_reliable') and qa.get('strip_state')=='SINGLE_STEP_LIKE'):
            return dict(q,accepted=False,reason='FINITE_STRIP_VS_SINGLE_STEP')
    return dict(q,accepted=True,reason='CURRENT_COLOR_PROFILE_COMPATIBLE')


def reference_descriptor(state,arm):
    return state['_anchor'] if arm in ('anchor_strip','anchor_finite') else state['_rolling']


def predict(prevgray,gray,prepared,state,fallback,arm):
    predictions,qualities=C.support_predictions(prevgray,gray,state,fallback)
    for old,pred,q in zip(state,predictions,qualities):
        d=F.extract_appearance(prepared,pred)
        if arm=='local_lab_partial' and d['status']!='AVAILABLE':d=Partial.extract_partial_core(prepared,pred)
        cmp=compatible(reference_descriptor(old,arm),d,arm)
        q['new_appearance']=cmp;q['current_descriptor_status']=d['status']
        # The old normalized gray gradient is retained only as a diagnostic.
        # Its old sim=None cost path cannot authorize identity support here.
        q['pixel_supported']=bool(q['valid'] and cmp['accepted'])
        pred['_current_descriptor']=d
    return predictions,qualities


def assign(current,descriptors,state,predictions,qualities,next_id,arm,reacquire=False,gap=None):
    nr,nc=len(state),len(current);costs=np.full((nr,nc),1e6);details={};rejected=[]
    for i,(old,pred,flow) in enumerate(zip(state,predictions,qualities)):
        p,m,u,length=C.geom(pred)
        for j,c in enumerate(current):
            cp,cm,v,clen=C.geom(c);angle=float(np.rad2deg(np.arccos(np.clip(abs(u@v),0,1))))
            normal=np.array([-u[1],u[0]]);distance=abs(float((cm-m)@normal))
            maxdist=CONFIG['reacquire_normal_px'] if reacquire else CONFIG['normal_px']
            maxangle=CONFIG['reacquire_direction_deg'] if reacquire else CONFIG['direction_deg']
            if distance>maxdist or angle>maxangle:continue
            cmp=compatible(reference_descriptor(old,arm),descriptors[j],arm)
            if not cmp['accepted']:
                rejected.append(dict(candidate_index=j,previous_track_id=old['track_id'],normal_distance_px=distance,
                    direction_deg=angle,comparison=cmp,reason=cmp['reason']));continue
            overlap=max(0.,min(max(p@u),max(cp@u))-max(min(p@u),min(cp@u)))/max(1.,min(length,clen))
            if (reacquire or overlap<.1) and not flow['pixel_supported']:
                rejected.append(dict(candidate_index=j,previous_track_id=old['track_id'],normal_distance_px=distance,
                    direction_deg=angle,comparison=cmp,reason='NO_CURRENT_FB_PIXEL_SUPPORT'));continue
            s=cmp.get('normalized_shape_similarity') if arm not in ('local_lab','local_lab_partial') else 1.
            shape_cost=0. if s is None else 3*(1-s)
            cost=distance+.2*angle+.3*cmp['lab_core_delta']+.2*cmp['lab_contrast_delta']+shape_cost
            costs[i,j]=cost;details[i,j]=dict(previous_state_index=i,previous_track_id=old['track_id'],
                normal_distance_px=distance,direction_deg=angle,visible_overlap=overlap,cost=cost,
                appearance=cmp,flow=flow,reacquisition=reacquire,gap_s=gap)
    links={};events=[]
    if nr and nc:
        ii,jj=linear_sum_assignment(costs)
        for i,j in zip(ii,jj):
            if costs[i,j]>=1e5:continue
            others=[costs[k,j] for k in range(nr) if k!=i and costs[k,j]<1e5]
            others+=[costs[i,k] for k in range(nc) if k!=j and costs[i,k]<1e5]
            margin=None if not others else float(min(others)-costs[i,j])
            minimum=CONFIG['reacquire_unique_margin'] if reacquire else CONFIG['ambiguity_margin']
            d=details[i,j];d.update(unique_margin=margin,ambiguous=bool(margin is not None and margin<minimum))
            if reacquire:
                d.update(accepted=not d['ambiguous'],candidate_index=int(j),previous_segment=[state[i][k] for k in ('x1','y1','x2','y2')],
                    current_segment=[current[j][k] for k in ('x1','y1','x2','y2')],
                    previous_rgb_index=state[i]['source_rgb_index'],episode_continuous=False,gap_observations_added=0)
                events.append(d)
            if d['ambiguous']:
                rejected.append(dict(candidate_index=int(j),previous_track_id=state[i]['track_id'],normal_distance_px=d['normal_distance_px'],
                    direction_deg=d['direction_deg'],comparison=d['appearance'],reason='AMBIGUOUS_LINK_NO_IDENTITY_UPDATE',unique_margin=margin))
            else:links[int(j)]=(int(i),d)
    used=set()
    for j,c in enumerate(current):
        d=descriptors[j]
        if j in links:
            i,link=links[j];old=state[i];used.add(i)
            c.update(track_id=old['track_id'],association=link,identity_status='REACQUIRED_NATIVE_APPEARANCE' if reacquire else 'APPEARANCE_SUPPORTED_NATIVE',
                _anchor=old['_anchor'],_anchor_id=old['_anchor_id'],_anchor_rgb_index=old['_anchor_rgb_index'],
                _anchor_hash=old['_anchor_hash'],_object_times=list(old['_object_times']),
                _episode_times=[] if reacquire else list(old['_episode_times']),episode_id=old['episode_id']+int(reacquire))
        else:
            anchor=copy.deepcopy(d) if d['status'] in ('AVAILABLE','PARTIAL_CORE_AVAILABLE') else None
            c.update(track_id=next_id,association=None,identity_status='NEW_NATIVE_AVAILABLE' if anchor is not None else 'NEW_NATIVE_UNKNOWN_APPEARANCE',
                _anchor=anchor,_anchor_id=None,_anchor_rgb_index=None,_anchor_hash=desc_hash(anchor),
                _object_times=[],_episode_times=[],episode_id=0);next_id+=1
        c['_rolling']=copy.deepcopy(d) if d['status'] in ('AVAILABLE','PARTIAL_CORE_AVAILABLE') else None
    assert len({c['track_id'] for c in current})==len(current)
    # Keep at most three nearby rejected predecessors per current candidate.
    selected=[]
    for j in range(len(current)):
        rr=[r for r in rejected if r['candidate_index']==j]
        selected.extend(sorted(rr,key=lambda r:r['normal_distance_px'])[:3])
    return next_id,used,selected,events


def count_observation(c,t,kind):
    fresh=t not in c['_object_times']
    if fresh:c['_object_times'].append(t)
    if t not in c['_episode_times']:c['_episode_times'].append(t)
    c.update(object_id=c['track_id'],object_independent_hits=len(c['_object_times']),object_established=len(c['_object_times'])>=3,
        episode_independent_hits=len(c['_episode_times']),independent_hits=len(c['_episode_times']),hits=len(c['_episode_times']),
        episode_stable=kind!='FLOW_SUPPORTED' and len(c['_episode_times'])>=3,
        stable=kind!='FLOW_SUPPORTED' and len(c['_episode_times'])>=3,counted_new_image=fresh,observation_kind=kind,
        anchor_descriptor_id=c['_anchor_id'],anchor_rgb_index=c['_anchor_rgb_index'],anchor_sha256=c['_anchor_hash'],
        anchor_status='AVAILABLE' if c['_anchor'] is not None else 'UNKNOWN_NOT_ESTABLISHED',
        anchor_descriptor_status=None if c['_anchor'] is None else c['_anchor']['status'],
        anchor_stripe_reliable=False if c['_anchor'] is None else bool(c['_anchor'].get('quality',{}).get('stripe_reliable')),
        anchor_strip_state=None if c['_anchor'] is None else c['_anchor'].get('quality',{}).get('strip_state'),
        appearance_scope='generic RGB compatibility and object hypothesis; no target/purity/distance/safety truth')


def run():
    start=time.perf_counter();cv2.setNumThreads(2);cv2.setRNGSeed(0);OUT.mkdir(parents=True,exist_ok=True)
    feature_receipt=json.loads((FEATURES/'receipt.json').read_text(encoding='utf8'))
    assert feature_receipt['complete'] and feature_receipt['source_sha256']==sha(F.__file__)
    rows=read(IN/'frames.jsonl');assert len(rows)==654
    if SESSIONS:rows=[r for r in rows if r['key'] in SESSIONS]
    feature_rows=read(FEATURES/'features.jsonl');lookup={(r['key'],r['rgb_index']):{c['rank']:c['feature'] for c in r['features']} for r in feature_rows}
    if 'local_lab_partial' in ARMS:
        for r in read(FEATURES/'partial_core_features.jsonl'):
            for d in r['partial_features']:
                if lookup[r['key'],r['rgb_index']][d['rank']]['status']!='AVAILABLE':lookup[r['key'],r['rgb_index']][d['rank']]=d['feature']
    summary=dict(config=CONFIG,arms=ARMS,frames=len(rows),source_code_sha256=sha(__file__),input_sha256=sha(IN/'frames.jsonl'),
        feature_source_sha256=sha(F.__file__),feature_cache_sha256=sha(FEATURES/'features.jsonl'),fast_source_sha256=sha(Fast.__file__),
        partial_source_sha256=sha(Partial.__file__),partial_cache_sha256=sha(FEATURES/'partial_core_features.jsonl'),
        labels_read=False,lookahead_frames=0,gpu_s=0,backend='TASK_NOT_GPU_SUITABLE sparse CPU LK and line-local color/profile',sessions={},complete=False)
    cam=AtomCamera();states={a:[] for a in ARMS};nextids={a:1 for a in ARMS};oldkey=None;prev=None;prevgray=None
    handles={a:(OUT/f'tracking_{a}.jsonl').open('w',encoding='utf8') for a in ARMS}
    anchors=(OUT/f'anchors{("_"+TAG) if TAG else ""}.jsonl').open('w',encoding='utf8')
    try:
        for row in rows:
            if time.perf_counter()-start>WALL_CAP:raise TimeoutError(f'{WALL_CAP}s scoped command cap')
            key=row['key'];t=row['device_capture_ns'];rt=row['received_elapsed_ns']
            if key!=oldkey:
                states={a:[] for a in ARMS};nextids={a:1 for a in ARMS};prev=None;prevgray=None;oldkey=key
                summary['sessions'][key]={a:Counter() for a in ARMS}
            bgr=cam.undistort_upright(cv2.imread(row['rgb_file']));prepared=F.prepare_image(bgr);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            duplicate=prev is not None and t==prev['device_capture_ns'];dt=None if prev is None else (t-prev['device_capture_ns'])/1e9
            dr=None if prev is None else (rt-prev['received_elapsed_ns'])/1e9
            reset=prev is None or row['boot_id']!=prev['boot_id'] or dt<0 or dr<0 or dt>.5
            dormant={a:[] for a in ARMS};dormantgray=None
            if reset and prev is not None and row['boot_id']==prev['boot_id'] and .5<dt<=CONFIG['reacquire_max_gap_s']:
                dormant={a:[copy.deepcopy(c) for c in states[a] if c['_last_observed_ns']==prev['device_capture_ns']] for a in ARMS};dormantgray=prevgray
            if reset:states={a:[] for a in ARMS};prevgray=None
            fallback,motionq=C.motion(prevgray,gray) if not duplicate else (np.eye(2,3),dict(valid=False,reason='duplicate'))
            descriptors=[lookup[key,row['rgb_index']][c['rank']] for c in row['candidates']]
            for arm in ARMS:
                state=[c for c in states[arm] if 0<=t-c['_last_observed_ns']<=500000000];current=copy.deepcopy(row['candidates'])
                if duplicate:
                    predictions=[dict(c) for c in state];qualities=[dict(valid=False,pixel_supported=False,reason='DUPLICATE_NO_OBSERVATION') for c in state]
                else:predictions,qualities=predict(prevgray,gray,prepared,state,fallback,arm)
                nextids[arm],used,rejections,events=assign(current,descriptors,state,predictions,qualities,nextids[arm],arm)
                if dormant[arm] and current:
                    dp,dq=predict(dormantgray,gray,prepared,dormant[arm],np.eye(2,3),arm)
                    fresh_next=nextids[arm]
                    nextids[arm],unused,rejected_reacq,events=assign(current,descriptors,dormant[arm],dp,dq,nextids[arm],arm,True,dt)
                    rejections+=rejected_reacq
                nextstate=[];flowobs=[]
                for c,d in zip(current,descriptors):
                    identifier=f'{key}:{row["rgb_index"]}:{c["rank"]}'
                    if c['_anchor_id'] is None and c['_anchor'] is not None:
                        c['_anchor_id']=identifier;c['_anchor_rgb_index']=row['rgb_index']
                        anchors.write(json.dumps(dict(arm=arm,key=key,track_id=c['track_id'],descriptor_id=identifier,anchor_sha256=c['_anchor_hash'],descriptor=c['_anchor']),allow_nan=False)+'\n')
                    count_observation(c,t,'CURRENT_AUTONOMOUS_CANDIDATE')
                    c.update(source_candidate_id=identifier,source_rgb_index=row['rgb_index'],source_frame_id=row['source_frame_id'],
                        descriptor_id=identifier,current_descriptor_status=d['status'],current_strip_state=d.get('quality',{}).get('strip_state'),
                        line_reference=C.line_reference(c,cam.K_upright),_last_observed_ns=t,_last_native_ns=t,_appearance=C.appearance(gray,c))
                    nextstate.append(c)
                for i,(old,pred,q) in enumerate(zip(state,predictions,qualities)):
                    if i in used:continue
                    carried=dict(pred)
                    observed=not duplicate and q['pixel_supported'] and t-old['_last_native_ns']<=int(CONFIG['max_flow_without_native_s']*1e9)
                    if observed:
                        carried['_object_times']=list(old['_object_times']);carried['_episode_times']=list(old['_episode_times'])
                        # Rolling-color ablations update only from real current
                        # pixels; the frozen identity anchor never changes.
                        if arm not in ('anchor_strip','anchor_finite'):carried['_rolling']=copy.deepcopy(pred['_current_descriptor'])
                        count_observation(carried,t,'FLOW_SUPPORTED')
                        carried.update(identity_status='CURRENT_RGB_FLOW_APPEARANCE_OBJECT_PURITY_UNKNOWN',flow_support=q,
                            association=dict(source='CURRENT_RGB_FB_AND_APPEARANCE',previous_track_id=old['track_id']),
                            source_rgb_index=row['rgb_index'],source_frame_id=row['source_frame_id'],_last_observed_ns=t,
                            line_reference=C.line_reference(carried,cam.K_upright),current_depth_support='UNKNOWN_NOT_MEASURED',
                            candidate_metadata_scope='LAST_NATIVE_CANDIDATE_PROVENANCE_ONLY',current_descriptor_status=pred['_current_descriptor']['status'])
                        flowobs.append(clean(carried))
                    nextstate.append(carried)
                assert len({c['track_id'] for c in nextstate})==len(nextstate)
                export=[clean(c) for c in current]
                assert all(all(c[k]==d[k] for k in c) for c,d in zip(row['candidates'],export))
                result={k:v for k,v in row.items() if k!='candidates'}
                result.update(arm=arm,candidates=export,flow_observations=flowobs,identity_rejections=rejections,reacquisition_edges=events,
                    temporal_reset=reset,duplicate_capture=duplicate,device_dt_s=dt,receipt_dt_s=dr,
                    receipt_transport_gap=bool(dr is not None and dr>.5 and dt is not None and 0<dt<=.5),
                    global_motion=motionq,internal_coast_tracks=len(nextstate)-len(current)-len(flowobs),lookahead_frames=0,
                    anchor_update_contract='immutable after first native AVAILABLE descriptor; ambiguous/flow never update anchor')
                handles[arm].write(json.dumps(result,allow_nan=False)+'\n')
                summary['sessions'][key][arm].update(frames=1,candidates=len(current),native_observations=len(current),flow_observations=len(flowobs),
                    new_id=sum(c['association'] is None for c in current),identity_rejections=len(rejections),
                    unknown_descriptor=sum(d['status']!='AVAILABLE' for d in descriptors),
                    reacquisition_accepted=sum(e['accepted'] for e in events),reacquisition_ambiguous=sum(not e['accepted'] for e in events))
                states[arm]=nextstate
            prev=row;prevgray=gray
        summary['complete']=True
    finally:
        for f in handles.values():f.close()
        anchors.close();summary['wall_s']=time.perf_counter()-start;save(f'receipt{("_"+TAG) if TAG else ""}.json',summary);print(json.dumps(summary,indent=2))


def controls():
    available=dict(status='AVAILABLE',core_lab=[50.,0.,0.],contrast_lab=[-10.,0.,0.],lab_profile=[[float(50-10*(abs(i-16)<3)),0.,0.] for i in range(33)],
        old_gray_gradient=[0.]*20,quality=dict(stripe_reliable=True,strip_state='FINITE_BILATERAL_STRIP'),scales=[])
    unknown=dict(status='UNKNOWN')
    assert not compatible(available,unknown,'anchor_strip')['accepted']
    # Anchor memory update is exercised explicitly independently of the image
    # descriptor controls owned by the feature module.
    oldanchor=copy.deepcopy(available);state=dict(_anchor=copy.deepcopy(oldanchor),_anchor_id='first',_anchor_rgb_index=0,_anchor_hash=desc_hash(oldanchor),
        _rolling=copy.deepcopy(available),_object_times=[1,2],_episode_times=[],track_id=1,episode_id=1)
    count_observation(state,3,'CURRENT_AUTONOMOUS_CANDIDATE')
    assert state['object_independent_hits']==3 and state['episode_independent_hits']==1 and not state['stable']
    current=copy.deepcopy(state);current['_rolling']=unknown;count_observation(current,4,'FLOW_SUPPORTED')
    assert current['_anchor']==oldanchor and current['_anchor_hash']==desc_hash(oldanchor)
    assert not current['stable']
    save('controls.json',dict(status='PASS',unknown_not_identity_support=True,first_reacquired_episode_count=1,
        first_reacquired_episode_stable=False,immutable_anchor_after_flow=True,anchor_quality_is_not_purity=True))


def verify():
    original=read(IN/'frames.jsonl');anchors={}
    for r in read(OUT/f'anchors{("_"+TAG) if TAG else ""}.jsonl'):anchors[r['arm'],r['key'],r['track_id']]=r
    n=0;hashes={}
    for arm in ARMS:
        rows=read(OUT/f'tracking_{arm}.jsonl');assert len(rows)==len(original)==654
        times={};episodes={}
        for old,r in zip(original,rows):
            assert len(old['candidates'])==len(r['candidates'])
            for c,d in zip(old['candidates'],r['candidates']):assert all(c[k]==d[k] for k in c);n+=1
            observed=r['candidates']+r['flow_observations'];assert len({c['track_id'] for c in observed})==len(observed)
            for c in observed:
                identity=(r['key'],c['track_id']);epi=(*identity,c['episode_id']);t=r['device_capture_ns']
                tt=times.setdefault(identity,set());ee=episodes.setdefault(epi,set());tt.add(t);ee.add(t)
                assert c['object_independent_hits']==len(tt) and c['episode_independent_hits']==len(ee)
                assert c['hits']==c['independent_hits']==c['episode_independent_hits']
                if c['association'] and c['association'].get('reacquisition'):assert c['episode_independent_hits']==1 and not c['stable']
                if c['anchor_status']=='AVAILABLE':
                    a=anchors[arm,r['key'],c['track_id']];assert a['anchor_sha256']==c['anchor_sha256'] and a['descriptor_id']==c['anchor_descriptor_id']
                if c['observation_kind']=='FLOW_SUPPORTED':assert c['flow_support']['pixel_supported'] and c['current_depth_support']=='UNKNOWN_NOT_MEASURED'
        hashes[arm]=sha(OUT/f'tracking_{arm}.jsonl')
    save(f'verification{("_"+TAG) if TAG else ""}.json',dict(status='PASS',frames_each_arm=654,unchanged_candidate_checks=n,source_code_sha256=sha(__file__),
        feature_source_sha256=sha(F.__file__),input_sha256=sha(IN/'frames.jsonl'),output_hashes=hashes,
        anchor_immutable=True,current_pixel_flow_only=True,episode_separate_from_object=True,labels_read=False,lookahead_frames=0))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','controls','verify']);ap.add_argument('--arms');ap.add_argument('--tag',default='');ap.add_argument('--sessions');ap.add_argument('--out');ap.add_argument('--wall-cap',type=float,default=240.);a=ap.parse_args()
    if a.arms:ARMS=tuple(a.arms.split(','))
    TAG=a.tag
    SESSIONS=set(a.sessions.split(',')) if a.sessions else None
    WALL_CAP=a.wall_cap
    if a.out:OUT=Path(a.out)
    OUT.mkdir(parents=True,exist_ok=True)
    {'run':run,'controls':controls,'verify':verify}[a.stage]()
