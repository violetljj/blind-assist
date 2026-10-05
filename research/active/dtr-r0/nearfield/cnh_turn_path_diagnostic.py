"""Frozen retrospective finite future-path diagnostic; evaluator truth only.

Preserves original query labels, scores and verdict. No inference or rendering.
Horizontal collision is an exact piecewise-linear path / rounded-AABB test,
with a 0.30 m circular footprint and the frozen HEAD/BODY height intervals.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-turn-path-diagnostic-20261005'
PREVIOUS=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005'
OLD=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural'
SOURCE=ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
EPS=1e-8
HEIGHTS=((-0.2,0.42),(0.42,0.9))
THRESHOLD=.8557642486787612
FRAMES=np.arange(3,16)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def future_segments(travel,frame,horizon=3.):
    """Known current disk, saved future chords, then finite final-heading line.

    Segments expose start/end xyz, start arclength, length and source. The initial
    zero-length segment marks the known current footprint even at frame15.
    """
    travel=np.asarray(travel,float);positions=travel[:,:3,3]
    assert travel.shape==(16,4,4) and 0<=frame<16 and horizon>0
    assert np.max(abs(positions[:,1]-positions[frame,1]))<1e-12
    current=positions[frame].copy();distance=0.;result=[dict(start=current,end=current,length=0.,s0=0.,source='recorded')]
    for f in range(frame+1,16):
        nxt=positions[f];delta=nxt-current;length=float(np.linalg.norm(delta[[0,2]]))
        if length<=0:continue
        take=min(length,horizon-distance);end=current+delta*(take/length)
        result.append(dict(start=current.copy(),end=end.copy(),length=take,s0=distance,source='recorded'))
        distance+=take;current=end
        if distance>=horizon-1e-12:return result
    direction=travel[-1,:3,2].copy();direction[1]=0.;direction/=np.linalg.norm(direction)
    np.testing.assert_allclose(direction,[0,0,1],atol=1e-12,rtol=0)
    length=horizon-distance
    if length>1e-12:result.append(dict(start=current.copy(),end=current+length*direction,length=length,s0=distance,source='extrapolated'))
    return result

def _rect_first(p,v,lo,hi):
    t0,t1=0.,1.
    for axis in range(2):
        if abs(v[axis])<1e-15:
            if p[axis]<=lo[axis] or p[axis]>=hi[axis]:return None
        else:
            a=(lo[axis]-p[axis])/v[axis];b=(hi[axis]-p[axis])/v[axis]
            t0=max(t0,min(a,b));t1=min(t1,max(a,b))
            if t0>=t1:return None
    return t0 if t0<1 and t1>0 else None

def _disk_first(p,v,center,radius):
    d=p-center;aa=float(v@v);cc=float(d@d-radius**2)
    if cc<0:return 0.
    if aa<1e-30:return None
    bb=2*float(d@v);disc=bb**2-4*aa*cc
    if disc<=0:return None
    a=(-bb-np.sqrt(disc))/(2*aa);b=(-bb+np.sqrt(disc))/(2*aa)
    return max(0.,a) if b>0 and a<1 else None

def _rounded_box_first(p,v,lo,hi,radius):
    # AABB Minkowski disk = two orthogonal strips plus four corner disks.
    rr=radius-EPS
    vals=[_rect_first(p,v,lo-[rr,0],hi+[rr,0]),_rect_first(p,v,lo-[0,rr],hi+[0,rr])]
    vals += [_disk_first(p,v,np.array([x,z]),rr) for x in (lo[0],hi[0]) for z in (lo[1],hi[1])]
    valid=[float(t) for t in vals if t is not None]
    return min(valid) if valid else None

def tube_first_hit(segments,box,radius=.30):
    """Continuous earliest collision, plus nonexclusive saved/extrapolated flags."""
    lo=np.asarray(box['lo'])[[0,2]];hi=np.asarray(box['hi'])[[0,2]];hits=[]
    for seg in segments:
        p=np.asarray(seg['start'])[[0,2]];v=(np.asarray(seg['end'])-seg['start'])[[0,2]]
        t=_rounded_box_first(p,v,lo,hi,radius)
        if t is not None:hits.append(dict(arclength_m=seg['s0']+t*seg['length'],source=seg['source']))
    if not hits:return dict(invasion=False,recorded=False,extrapolated=False,extrapolated_only=False,first_arclength_m=None,first_time_s=None)
    first=min(hits,key=lambda h:h['arclength_m']);recorded=any(h['source']=='recorded' for h in hits);extrapolated=any(h['source']=='extrapolated' for h in hits)
    return dict(invasion=True,recorded=recorded,extrapolated=extrapolated,extrapolated_only=not recorded,
        first_arclength_m=first['arclength_m'],first_time_s=first['arclength_m']/.8,first_source=first['source'])

def _rect_strip_overlap(p,end,lo,hi,radius):
    v=end-p;length=float(np.linalg.norm(v))
    if length<1e-15:return False
    tangent=v/length;normal=np.array([-tangent[1],tangent[0]])
    corners=np.array([[x,z] for x in (lo[0],hi[0]) for z in (lo[1],hi[1])])
    strip=np.array([p-radius*normal,p+radius*normal,end-radius*normal,end+radius*normal])
    for axis in (np.array([1.,0.]),np.array([0.,1.]),tangent,normal):
        a=corners@axis;b=strip@axis
        if min(a.max(),b.max())-max(a.min(),b.min())<=EPS:return False
    return True

def no_endcaps_hit(segments,box,radius=.30):
    """Segment normal strips and internal rounded joins, without two endcaps."""
    segs=[s for s in segments if s['length']>0];lo=np.asarray(box['lo'])[[0,2]];hi=np.asarray(box['hi'])[[0,2]]
    for seg in segs:
        p=np.asarray(seg['start'])[[0,2]];end=np.asarray(seg['end'])[[0,2]]
        if _rect_strip_overlap(p,end,lo,hi,radius-EPS):return True
    for seg in segs[:-1]:
        p=np.asarray(seg['end'])[[0,2]];delta=np.maximum(0,np.maximum(lo-p,p-hi))
        if np.linalg.norm(delta)<radius-EPS:return True
    return False

def kind(index,n):
    return 'target' if index==0 else 'floor' if index==n-2 else 'back_wall' if index==n-1 else 'mixed_surface_panel'

def classify_boxes(boxes,travel,frame,query,horizon=3.):
    segments=future_segments(travel,frame,horizon);short=future_segments(travel,frame,2.5);objects=[]
    origin_y=float(travel[frame,1,3])
    for b,box in enumerate(boxes):
        vertical=[min(box['hi'][1],origin_y+y1)-max(box['lo'][1],origin_y+y0)>EPS for y0,y1 in HEIGHTS]
        hit=tube_first_hit(segments,box);small=tube_first_hit(short,box);no_caps=no_endcaps_hit(segments,box)
        objects.append(dict(box_index=b,type=kind(b,len(boxes)),height_overlaps=vertical,
            same_query_invasion=bool(vertical[query] and hit['invasion']),other_height_invasion=bool(vertical[1-query] and hit['invasion']),
            same_query_2p5=bool(vertical[query] and small['invasion']),other_height_2p5=bool(vertical[1-query] and small['invasion']),
            same_query_no_endcaps=bool(vertical[query] and no_caps),other_height_no_endcaps=bool(vertical[1-query] and no_caps),horizontal_path=hit))
    def flag(field):return any(o[field] for o in objects)
    same,other=flag('same_query_invasion'),flag('other_height_invasion')
    colliding=[o for o in objects if o['same_query_invasion'] or o['other_height_invasion']]
    first=min((o['horizontal_path']['first_arclength_m'] for o in colliding),default=None)
    return dict(same_query_path=same,other_height_path=other,union_path_invasion=same or other,
        exclusive_class='SAME_QUERY_PATH' if same else 'OTHER_HEIGHT_ONLY' if other else 'PATH_CLEAR',
        recorded_path_invasion=any(o['horizontal_path']['recorded'] for o in colliding),
        extrapolated_path_invasion=any(o['horizontal_path']['extrapolated'] for o in colliding),
        extrapolated_only_invasion=bool(colliding and not any(o['horizontal_path']['recorded'] for o in colliding)),
        first_invasion_arclength_m=first,first_invasion_time_s=None if first is None else first/.8,
        union_2p5=flag('same_query_2p5') or flag('other_height_2p5'),union_no_endcaps=flag('same_query_no_endcaps') or flag('other_height_no_endcaps'),
        path_recorded_length_m=sum(s['length'] for s in segments if s['source']=='recorded'),
        path_extrapolated_length_m=sum(s['length'] for s in segments if s['source']=='extrapolated'),objects=objects)

def load_batch(batch):
    paths=[]
    if batch==96000:
        ledger=OLD/'ledger.npz';manifest=SOURCE/'scene_manifest.json';geometry=ROOT/'artifacts.local/work/cnh-observed-sequence-20261002/geometry.npz'
        with np.load(ledger) as d:g={k:d[k].copy() for k in d.files}
        g['sensor']=g['sensor_scores'];g['ranges']=g['frame_ranges']
        with np.load(geometry) as d:
            keep=d['split']=='evaluation';cat=d['frame_category'][keep].copy()
            for k in ('unit','config','query'):np.testing.assert_array_equal(d[k][keep],g[k])
    else:
        ledger=PREVIOUS/'fusion/natural97000_ledger.npz';manifest=PREVIOUS/'natural97000/scene_manifest.json';geometry=PREVIOUS/'natural97000/geometry.npz'
        with np.load(ledger) as d:g={k:d[k].copy() for k in d.files}
        with np.load(geometry) as d:
            cat=d['frame_category'].copy()
            for k in ('unit','config','query'):np.testing.assert_array_equal(d[k],g[k])
    paths=[ledger,manifest,geometry]
    scenes={(r['unit'],r['config']):r for r in read(manifest) if r.get('split','evaluation')=='evaluation'}
    g['frame_category']=cat
    assert np.array_equal(g['query'].reshape(-1,2),np.tile([0,1],(len(g['query'])//2,1)))
    for k in ('unit','config'):np.testing.assert_array_equal(g[k][::2],g[k][1::2])
    return g,scenes,{str(p):sha(p) for p in paths}

def summarize(rows):
    n=len(rows);union=sum(r['union_path_invasion'] for r in rows)
    ratio=union/n if n else None
    branch='NOT_EVALUABLE' if not n else 'TURN_COST_MOSTLY_LABEL' if ratio>=.5 else 'TURN_COST_REAL' if ratio<=.2 else 'MIXED'
    return dict(n=n,union_path_invasion=union,union_rate=ratio,judgment=branch,
        exclusive=dict(Counter(r['exclusive_class'] for r in rows)),
        nonexclusive={field:sum(r[field] for r in rows) for field in ('same_query_path','other_height_path','recorded_path_invasion','extrapolated_path_invasion','extrapolated_only_invasion','union_2p5','union_no_endcaps','other_height_contact_ever','other_height_nonclear_ever')},
        colliding_object_types={typ:sum(any(o['type']==typ and (o['same_query_invasion'] or o['other_height_invasion']) for o in r['objects']) for r in rows) for typ in ('target','mixed_surface_panel','floor','back_wall')})

def engineering():
    # Frozen geometry cases, independent of any scientific scores/counts.
    path=[dict(start=np.array([0.,0.,0.]),end=np.array([0.,0.,3.]),length=3.,s0=0.,source='recorded')]
    box=lambda x,z:dict(lo=[x,-.1,z],hi=[x+.1,.2,z+.1])
    assert tube_first_hit(path,box(.29,1))['invasion']
    assert not tube_first_hit(path,box(.30,1))['invasion']
    assert not tube_first_hit(path,box(0,3.31))['invasion']
    assert tube_first_hit(path,box(0,3.1))['invasion'] and not no_endcaps_hit(path,box(0,3.1))
    assert abs(tube_first_hit(path,box(0,1))['first_arclength_m']-(1-.3+EPS))<1e-10
    save(OUT/'path_engineering.json',dict(status='PASS',cases=5,source_sha256=sha(__file__)))

def run():
    tick=time.monotonic();plan=read(OUT/'PLAN.json');plan_hash=sha(OUT/'PLAN.json')
    assert plan['path']['horizon_arclength_m']==3 and plan['threshold']==THRESHOLD
    assert time.time()<plan['deadline_unix'];engineering()
    import cnh_cvr_pilot as CP
    trajectory={};classified={};allrows=[];hashes={};batch_summary={}
    for batch in (96000,97000):
        g,scenes,input_hash=load_batch(batch);hashes.update(input_hash)
        ss=(g['single']>=THRESHOLD).any(1);ors=g['sensor'].max(1)
        for arm,scores in [('single',g['single']),('OR',ors)]:
            alarm=scores>=THRESHOLD;stopped=alarm.any(1);first=alarm.argmax(1);selected=np.flatnonzero(g['clear']&stopped)
            rows=[]
            for i in selected:
                u,c,q=int(g['unit'][i]),int(g['config'][i]),int(g['query'][i]);f=int(FRAMES[first[i]])
                if u not in trajectory:trajectory[u]=CP.motion_metadata(u,0)[1]
                key=(u,c,q,f)
                if key not in classified:classified[key]=classify_boxes(scenes[(u,c)]['boxes'],trajectory[u],f,q)
                obj=classified[key];other=i+1 if q==0 else i-1
                othercat=g['frame_category'][other]
                branch='single' if arm=='single' else 'L' if g['sensor'][i,0,first[i]]>g['sensor'][i,1,first[i]] else 'R' if g['sensor'][i,1,first[i]]>g['sensor'][i,0,first[i]] else 'tie'
                row=dict(batch=batch,arm=arm,ledger_index=int(i),unit=u,config=c,query=q,query_name=('HEAD','BODY')[q],mode=u%3,
                    frame=f,first_branch=branch,first_range_m=float(g['ranges'][i,first[i]]),
                    OR_added_mode2=bool(arm=='OR' and u%3==2 and not ss[i]),other_query_all_clear=bool(g['clear'][other]),
                    other_height_nonclear_ever=bool(np.any(othercat!='clear')),other_height_contact_ever=bool(np.char.startswith(othercat,'contact').any()),
                    other_height_first_frame_category=str(othercat[first[i]]),other_height_frame_categories=othercat.tolist(),**obj)
                rows.append(row);allrows.append(row)
            save(OUT/f'events{batch}_{arm}.json',dict(status='COMPLETE',batch=batch,arm=arm,threshold=THRESHOLD,plan_sha256=plan_hash,events=rows))
        batch_rows=[r for r in allrows if r['batch']==batch]
        batch_summary[str(batch)]=dict(primary_OR_added_mode2=summarize([r for r in batch_rows if r['OR_added_mode2']]),
            allmodes={arm:{str(mode):summarize([r for r in batch_rows if r['arm']==arm and r['mode']==mode]) for mode in range(3)} for arm in ('single','OR')},
            all={arm:summarize([r for r in batch_rows if r['arm']==arm]) for arm in ('single','OR')})
        print('batch',batch,json.dumps(batch_summary[str(batch)]['primary_OR_added_mode2']),flush=True)
        assert time.time()<plan['deadline_unix']
    for p,h in hashes.items():assert sha(p)==h
    assert sha(OUT/'PLAN.json')==plan_hash
    primary=[r for r in allrows if r['OR_added_mode2']]
    result=dict(status='COMPLETE',scope='Retrospective supplemental evaluator-only synthetic EXPLORE; original labels and GATING_NOT_SUPPORTED unchanged',
        plan_sha256=plan_hash,primary_pooled=summarize(primary),batches=batch_summary,event_rows=len(allrows),
        inputs_sha256=hashes,script_sha256=sha(__file__),seconds=time.monotonic()-tick,
        limits=['Finite3m centerline horizon plus circular endpoint caps(radius.30), not infinite collision. No full forward query box was swept.',
            'Saved continuous piecewise-linear trajectory and finite final-heading extrapolation have separate flags. The extrapolation is a diagnostic assumption, not observed future walking.',
            'Actual other-height path collision is distinct from other query not-clear(pass can be non-contact).',
            'Floor/back retained by true boxes; height mismatch is not collision. Static solid-volume future collision differs from original forward surface labels.',
            'True future path and boxes are evaluator-only; neither is deployable nor enters fusion/calibration. No counterfactual proves which object caused M3 alarm.'])
    save(OUT/'path_result.json',result);print('COMPLETE',json.dumps(result['primary_pooled']),result['seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','engineering'),default='run',nargs='?');args=parser.parse_args()
    if args.stage=='run':run()
    else:engineering()
