"""Correct descriptive no-endcap sensitivity; preserve frozen main result.

Clip the entire finite capsule tube by its start/end tangent half-spaces.
Internal rounded joins remain, including between the closely spaced saved poses.
This fixes internal vertex disks leaking across a globally flat endpoint.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import cnh_turn_path_diagnostic as D

def _clip(poly,normal,bound):
    if not len(poly):return poly
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        da=float(a@normal-bound);db=float(b@normal-bound)
        ia=da>=0.;ib=db>=0.
        if ia:result.append(a)
        if ia!=ib:result.append(a+(b-a)*(da/(da-db)))
    return np.asarray(result).reshape(-1,2)

def _point_segment(p,a,b):
    v=b-a;den=float(v@v)
    t=0. if den<1e-30 else float(np.clip((p-a)@v/den,0.,1.))
    return float(np.linalg.norm(p-(a+t*v)))

def _cross(a,b):return float(a[0]*b[1]-a[1]*b[0])

def _segments_distance(a,b,c,d):
    v=b-a;w=d-c;den=_cross(v,w)
    if abs(den)>1e-15:
        t=_cross(c-a,w)/den;u=_cross(c-a,v)/den
        if 0<=t<=1 and 0<=u<=1:return 0.
    return min(_point_segment(a,c,d),_point_segment(b,c,d),_point_segment(c,a,b),_point_segment(d,a,b))

def _inside(p,poly):return all(_cross(b-a,p-a)>=-1e-12 for a,b in zip(poly,np.roll(poly,-1,axis=0)))

def flat_endcaps_hit(segments,box,radius=.30):
    moving=[s for s in segments if s['length']>0]
    if not moving:return False
    start=np.asarray(moving[0]['start'])[[0,2]];end=np.asarray(moving[-1]['end'])[[0,2]]
    first=np.asarray(moving[0]['end'])[[0,2]]-start;first/=np.linalg.norm(first)
    last=end-np.asarray(moving[-1]['start'])[[0,2]];last/=np.linalg.norm(last)
    lo=np.asarray(box['lo'])[[0,2]];hi=np.asarray(box['hi'])[[0,2]]
    poly=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    poly=_clip(poly,first,float(start@first));poly=_clip(poly,-last,float(-end@last))
    if len(poly)<3:return False
    area=abs(sum(_cross(a,b) for a,b in zip(poly,np.roll(poly,-1,axis=0))))/2
    if area<=D.EPS**2:return False
    for seg in segments:
        a=np.asarray(seg['start'])[[0,2]];b=np.asarray(seg['end'])[[0,2]]
        if _inside(a,poly) or _inside(b,poly):return True
        if min(_segments_distance(a,b,c,d) for c,d in zip(poly,np.roll(poly,-1,axis=0)))<radius-D.EPS:return True
    return False

def run():
    tick=time.monotonic();plan_hash=D.sha(D.OUT/'PLAN.json');original_hash=D.sha(D.OUT/'path_result.json')
    assert time.time()<D.read(D.OUT/'PLAN.json')['deadline_unix']
    import cnh_cvr_pilot as CP
    trajectory={};geometry={};rows=[]
    for batch in (96000,97000):
        _,scenes,_=D.load_batch(batch)
        for arm in ('single','OR'):
            for row in D.read(D.OUT/f'events{batch}_{arm}.json')['events']:
                u,c,f,q=row['unit'],row['config'],row['frame'],row['query'];key=(u,c,f)
                if u not in trajectory:trajectory[u]=CP.motion_metadata(u,0)[1]
                if key not in geometry:
                    segments=D.future_segments(trajectory[u],f,3.);geometry[key]=[flat_endcaps_hit(segments,b) for b in scenes[(u,c)]['boxes']]
                hit=geometry[key];same=any(v and o['height_overlaps'][q] for v,o in zip(hit,row['objects']));other=any(v and o['height_overlaps'][1-q] for v,o in zip(hit,row['objects']))
                rows.append(dict(batch=batch,arm=arm,ledger_index=row['ledger_index'],unit=u,config=c,query=q,frame=f,
                    OR_added_mode2=row['OR_added_mode2'],previous_no_endcaps=row['union_no_endcaps'],same_query_flat=same,other_height_flat=other,
                    union_flat=same or other,changed=row['union_no_endcaps']!=(same or other)))
    primary=[r for r in rows if r['OR_added_mode2']]
    summary={str(batch):dict(n=sum(r['batch']==batch for r in primary),union_flat=sum(r['batch']==batch and r['union_flat'] for r in primary)) for batch in (96000,97000)}
    summary['pooled']=dict(n=len(primary),union_flat=sum(r['union_flat'] for r in primary),changed_primary=sum(r['changed'] for r in primary))
    assert D.sha(D.OUT/'path_result.json')==original_hash and D.sha(D.OUT/'PLAN.json')==plan_hash
    result=dict(status='COMPLETE',scope='Descriptive no-endcaps sensitivity amendment only; frozen main capsule result unchanged',
        semantics='Entire radius.30 continuous tube intersect start-tangent forward half-space and final-tangent backward half-space; internal joins preserved.',
        reason='Unioning internal vertex disks can leak beyond global endpoint planes when pose spacing .16m < radius.30m.',
        primary=summary,all_event_rows=len(rows),all_changed=sum(r['changed'] for r in rows),events=rows,seconds=time.monotonic()-tick,
        main_result_sha256=original_hash,plan_sha256=plan_hash,script_sha256=D.sha(__file__),main_geometry_script_sha256=D.sha(D.__file__))
    D.save(D.OUT/'sensitivity_flat_endcaps_amendment.json',result)
    print(json.dumps(dict(primary=summary,all_changed=result['all_changed'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':run()
