"""Independent CPU verification of the finite future-path diagnostic.

Uses segment/rectangle distances and rounded-rectangle entry, independent of
the diagnostic's intersection routine. No renderer, model or calibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-turn-path-diagnostic-20261005'
sys.pycache_prefix=str(OUT/'pycache')
RADIUS=.30-1e-8
HEIGHTS=((-0.2,.42),(.42,.9))
THRESHOLD=.8557642486787612


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(name,value):
    p=OUT/'checks'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def independent_travel(unit):
    yaw=np.linspace(-20,0,16) if unit%3==2 else np.zeros(16)
    points=np.zeros((16,2))
    for f in range(1,16):
        a=np.deg2rad((yaw[f-1]+yaw[f])/2)
        points[f]=points[f-1]+.16*np.array([np.sin(a),np.cos(a)])
    return points-points[-1]


def independent_segments(unit,frame,horizon=3.):
    positions=independent_travel(unit)
    segments=[];arc=0.
    for f in range(frame,15):
        p,q=positions[f],positions[f+1];length=float(np.linalg.norm(q-p))
        remaining=horizon-arc
        if remaining<=0:break
        if remaining<length:q=p+(q-p)*remaining/length;length=remaining
        segments.append((p,q,arc,length,True));arc+=length
    if arc<horizon:
        p=positions[-1];length=horizon-arc
        segments.append((p,p+[0,length],arc,length,False))
    return segments


def point_rect_distance(p,lo,hi):
    return float(np.linalg.norm(np.maximum(0,np.maximum(lo-p,p-hi))))


def point_segment_distance(p,a,b):
    d=b-a;t=float(np.clip((p-a)@d/(d@d),0,1))
    return float(np.linalg.norm(p-a-t*d))


def slab_entry(a,b,lo,hi):
    lower,upper=0.,1.;d=b-a
    for i in range(2):
        if abs(d[i])<1e-15:
            if a[i]<lo[i] or a[i]>hi[i]:return None
        else:
            t1,t2=sorted(((lo[i]-a[i])/d[i],(hi[i]-a[i])/d[i]))
            lower=max(lower,t1);upper=min(upper,t2)
            if lower>upper:return None
    return lower if lower<=upper and upper>=0 and lower<=1 else None


def segment_rect_distance(a,b,lo,hi):
    if slab_entry(a,b,lo,hi) is not None:return 0.
    corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])
    return min(point_rect_distance(a,lo,hi),point_rect_distance(b,lo,hi),
               *(point_segment_distance(p,a,b) for p in corners))


def rounded_rect_entry(a,b,lo,hi,r=RADIUS):
    """First center entry into rect ⊕ disk, without sampling a trajectory."""
    entries=[]
    # The rounded rectangle is the union of two strips and four disks.
    for l,h in ((lo-[r,0],hi+[r,0]),(lo-[0,r],hi+[0,r])):
        t=slab_entry(a,b,l,h)
        if t is not None:entries.append(t)
    d=b-a;dd=float(d@d)
    for c in np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]]):
        v=a-c;vv=float(v@v-r*r)
        if vv<0:entries.append(0.);continue
        linear=float(v@d);disc=linear*linear-dd*vv
        if disc>0:
            first=(-linear-np.sqrt(disc))/dd;last=(-linear+np.sqrt(disc))/dd
            if last>=0 and first<=1:entries.append(float(max(0,first)))
    return min(entries) if entries else None


def path_box_hit(unit,frame,box,horizon=3.):
    lo=np.asarray(box['lo'])[[0,2]];hi=np.asarray(box['hi'])[[0,2]]
    hits=[]
    for a,b,arc,length,observed in independent_segments(unit,frame,horizon):
        distance=segment_rect_distance(a,b,lo,hi)
        if distance<RADIUS:
            t=rounded_rect_entry(a,b,lo,hi)
            assert t is not None
            hits.append(dict(arc=arc+t*length,observed=observed))
    if not hits:return None
    current=independent_travel(unit)[frame]
    return dict(arc=min(h['arc'] for h in hits),recorded=point_rect_distance(current,lo,hi)<RADIUS or any(h['observed'] for h in hits),
                extrapolated=any(not h['observed'] for h in hits))


def classify(unit,frame,boxes,query,horizon=3.):
    flags=[False,False];records=[]
    for i,box in enumerate(boxes):
        hit=path_box_hit(unit,frame,box,horizon)
        if hit is None:continue
        for q,(low,high) in enumerate(HEIGHTS):
            if min(float(box['hi'][1]),high)-max(float(box['lo'][1]),low)>1e-8:
                flags[q]=True;records.append(dict(box=i,query=q,**hit))
    name='SAME_QUERY_PATH' if flags[query] else 'OTHER_HEIGHT_ONLY' if flags[1-query] else 'PATH_CLEAR'
    return name,flags,records


def input_check():
    plan=read(OUT/'PLAN.json')
    assert plan['budget_seconds']==2400 and not any(plan[k] for k in ('training','rendering','fusion_search'))
    assert plan['threshold']==THRESHOLD and plan['path']['horizon_arclength_m']==3.
    hashes=read(OUT/'INITIAL_SOURCE_HASHES.json')
    for p,h in hashes.items():assert sha(ROOT/p).lower()==h.lower(),p
    return dict(status='PASS',initial_payloads_preserved=len(hashes),plan_sha256=sha(OUT/'PLAN.json'))


def travel_matrix(unit):
    positions=independent_travel(unit);yaw=np.linspace(-20,0,16) if unit%3==2 else np.zeros(16)
    result=np.repeat(np.eye(4)[None],16,0);result[:,[0,2],3]=positions
    for f,a in enumerate(np.deg2rad(yaw)):
        c,s=np.cos(a),np.sin(a);result[f,:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    return result


def geometry_check():
    import cnh_turn_path_diagnostic as D
    rng=np.random.default_rng(2026100548);tested=0;collisions=0;error=0.
    for unit in (96000,96001,96002):
        travel=travel_matrix(unit)
        for frame in (3,8,15):
            for horizon in (2.5,3.):
                owner=D.future_segments(travel,frame,horizon);own=independent_segments(unit,frame,horizon)
                nonzero=[s for s in owner if s['length']>0]
                assert len(own)==len(nonzero)
                for (a,b,arc,length,observed),s in zip(own,nonzero):
                    np.testing.assert_allclose(s['start'][[0,2]],a,atol=1e-14,rtol=0)
                    np.testing.assert_allclose(s['end'][[0,2]],b,atol=1e-14,rtol=0)
                    assert abs(s['length']-length)<1e-14 and abs(s['s0']-arc)<1e-14
                    assert (s['source']=='recorded')==observed
                boxes=[]
                for _ in range(40):
                    low=rng.uniform([-1,-1,-3],[1,1,4]);size=rng.uniform(.01,.8,3)
                    boxes.append(dict(lo=low.tolist(),hi=(low+size).tolist()))
                # Explicit lateral tangent and initial/endcap corner cases.
                boxes.extend([dict(lo=[.3,-.1,.5],hi=[.4,.2,1]),dict(lo=[-.1,-.1,3.1],hi=[.1,.2,3.2])])
                for box in boxes:
                    actual=D.tube_first_hit(owner,box);expected=path_box_hit(unit,frame,box,horizon)
                    assert actual['invasion']==(expected is not None),(unit,frame,horizon,box,actual,expected)
                    if expected is not None:
                        assert actual['recorded']==expected['recorded'] and actual['extrapolated']==expected['extrapolated']
                        err=abs(actual['first_arclength_m']-expected['arc']);assert err<1e-10;error=max(error,err);collisions+=1
                    tested+=1
    # This tests the alternative minimum-distance method directly.
    assert segment_rect_distance(np.array([0.,0]),np.array([0.,3]),np.array([.3,1]),np.array([.4,1.1]))==.3
    return dict(status='PASS',capsule_segment_box_cases=tested,colliding_cases=collisions,
                independently_reconstructed_trajectories=3,frames=[3,8,15],horizons=[2.5,3],
                first_invasion_arc_max_abs_error=error,zero_current_recorded_disk_checked=True,
                diagnostic_source_sha256=sha(D.__file__))


def load_independent(batch):
    old=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural'
    previous=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005'
    p=old/'ledger.npz' if batch==96000 else previous/'fusion/natural97000_ledger.npz'
    with np.load(p) as z:d={k:z[k] for k in z.files}
    if batch==96000:
        d['sensor']=d['sensor_scores'];d['ranges']=d['frame_ranges']
        gp=ROOT/'artifacts.local/work/cnh-observed-sequence-20261002/geometry.npz'
        with np.load(gp) as z:
            keep=z['split']=='evaluation';d['frame_category']=z['frame_category'][keep]
            for k in ('unit','config','query'):np.testing.assert_array_equal(d[k],z[k][keep])
        mp=ROOT/'artifacts.local/work/cnh-margin-confirm-20261002/scene_manifest.json'
    else:
        gp=previous/'natural97000/geometry.npz'
        with np.load(gp) as z:
            d['frame_category']=z['frame_category']
            for k in ('unit','config','query'):np.testing.assert_array_equal(d[k],z[k])
        mp=previous/'natural97000/scene_manifest.json'
    scenes={(r['unit'],r['config']):r for r in read(mp) if r.get('split','evaluation')=='evaluation'}
    return d,scenes


def first_flags(score,ranges):
    n=len(score);stopped=np.zeros(n,bool);first=np.full(n,-1);timely=stopped.copy()
    for i in range(n):
        frames=[f for f,x in enumerate(score[i]) if x>=THRESHOLD]
        if frames:
            first[i]=frames[0];stopped[i]=True;timely[i]=ranges[i,frames[0]]>=.9
    return stopped,timely,first


def verify_summary(rows,cell):
    from collections import Counter
    union=sum(r['union_path_invasion'] for r in rows);n=len(rows)
    ratio=union/n if n else None
    branch='NOT_EVALUABLE' if not n else 'TURN_COST_MOSTLY_LABEL' if ratio>=.5 else 'TURN_COST_REAL' if ratio<=.2 else 'MIXED'
    assert (cell['n'],cell['union_path_invasion'],cell['union_rate'],cell['judgment'])==(n,union,ratio,branch)
    assert cell['exclusive']==dict(Counter(r['exclusive_class'] for r in rows))
    for field,num in cell['nonexclusive'].items():assert num==sum(r[field] for r in rows),field
    for typ,num in cell['colliding_object_types'].items():
        assert num==sum(any(o['type']==typ and (o['same_query_invasion'] or o['other_height_invasion']) for o in r['objects']) for r in rows)
    return branch


def outcome_check():
    result=read(OUT/'path_result.json');plan_sha=sha(OUT/'PLAN.json');all_rows=[];checked=0;objects=0
    assert result['plan_sha256']==plan_sha
    for batch in (96000,97000):
        d,scenes=load_independent(batch)
        ss,_,_=first_flags(d['single'],d['ranges'])
        scores={'single':d['single'],'OR':d['sensor'].max(1)}
        for arm,score in scores.items():
            stopped,_,first=first_flags(score,d['ranges'])
            payload=read(OUT/f'events{batch}_{arm}.json');rows=payload['events']
            assert payload['threshold']==THRESHOLD and payload['plan_sha256']==plan_sha
            expected=np.flatnonzero(d['clear']&stopped).tolist()
            assert [r['ledger_index'] for r in rows]==expected
            for row in rows:
                i=row['ledger_index'];u,c,q=(int(d[k][i]) for k in ('unit','config','query'));f=int(first[i])+3
                assert (row['batch'],row['arm'],row['unit'],row['config'],row['query'],row['mode'],row['frame'])==(batch,arm,u,c,q,u%3,f)
                assert row['first_range_m']==d['ranges'][i,first[i]]
                added=bool(arm=='OR' and u%3==2 and not ss[i]);assert row['OR_added_mode2']==added
                boxes=scenes[(u,c)]['boxes'];name,flags,recs=classify(u,f,boxes,q)
                assert (row['exclusive_class'],row['same_query_path'],row['other_height_path'],row['union_path_invasion'])==(name,flags[q],flags[1-q],any(flags))
                relevant=[r for r in recs if r['query'] in (q,1-q)]
                recorded=any(r['recorded'] for r in relevant);extrapolated=any(r['extrapolated'] for r in relevant)
                assert row['recorded_path_invasion']==recorded and row['extrapolated_path_invasion']==extrapolated
                assert row['extrapolated_only_invasion']==bool(relevant and not recorded)
                arc=min((r['arc'] for r in relevant),default=None)
                if arc is None:assert row['first_invasion_arclength_m'] is None
                else:
                    assert abs(row['first_invasion_arclength_m']-arc)<1e-10
                    assert abs(row['first_invasion_time_s']-arc/.8)<1e-10
                _,short_flags,_=classify(u,f,boxes,q,horizon=2.5)
                assert row['union_2p5']==any(short_flags)
                other=i+1 if q==0 else i-1;cats=d['frame_category'][other]
                assert row['other_query_all_clear']==bool(d['clear'][other])
                assert row['other_height_nonclear_ever']==bool((cats!='clear').any())
                assert row['other_height_contact_ever']==any(str(x).startswith('contact') for x in cats)
                for b,box in enumerate(boxes):
                    horizontal=path_box_hit(u,f,box)
                    entry=row['objects'][b];hit=entry['horizontal_path']
                    vertical=[min(box['hi'][1],high)-max(box['lo'][1],low)>1e-8 for low,high in HEIGHTS]
                    assert entry['height_overlaps']==vertical
                    assert hit['invasion']==(horizontal is not None)
                    if horizontal is not None:
                        assert hit['recorded']==horizontal['recorded'] and hit['extrapolated']==horizontal['extrapolated']
                        assert abs(hit['first_arclength_m']-horizontal['arc'])<1e-10
                    objects+=1
                checked+=1
            all_rows.extend(rows)
        group=[r for r in all_rows if r['batch']==batch];cells=result['batches'][str(batch)]
        for arm in ('single','OR'):
            verify_summary([r for r in group if r['arm']==arm],cells['all'][arm])
            for mode in range(3):verify_summary([r for r in group if r['arm']==arm and r['mode']==mode],cells['allmodes'][arm][str(mode)])
        verify_summary([r for r in group if r['OR_added_mode2']],cells['primary_OR_added_mode2'])
    primary=[r for r in all_rows if r['OR_added_mode2']]
    branch=verify_summary(primary,result['primary_pooled'])
    assert len(all_rows)==result['event_rows']
    return dict(status='PASS',clear_event_memberships_checked=checked,continuous_object_intersections_checked=objects,
                primary_added_mode2_n=len(primary),primary_union=sum(r['union_path_invasion'] for r in primary),
                primary_branch=branch,samequery=sum(r['same_query_path'] for r in primary),
                otherheight=sum(r['other_height_path'] for r in primary),
                recorded=sum(r['recorded_path_invasion'] for r in primary),extrapolated_only=sum(r['extrapolated_only_invasion'] for r in primary),
                preserved_otherheight_contact_vs_nonclear_distinction=True,
                result_sha256=sha(OUT/'path_result.json'),sensitivity_no_endcaps='Original implementation only; flat-cap amendment checked separately')


def fullpass_check():
    value=read(OUT/'full_object_pass_sensitivity.json');rows=value['events'];checked=0;objects=0
    assert value['contract_sha256']==sha(OUT/'SUPPLEMENT_FULL_PASS_PLAN.json')
    assert value['main_result_sha256']==sha(OUT/'path_result.json')
    for batch in (96000,97000):
        _,scenes=load_independent(batch)
        for arm in ('single','OR'):
            main=read(OUT/f'events{batch}_{arm}.json')['events']
            part=[r for r in rows if r['batch']==batch and r['arm']==arm]
            assert [r['ledger_index'] for r in main]==[r['ledger_index'] for r in part]
            for old,row in zip(main,part):
                u,c,q,f=row['unit'],row['config'],row['query'],row['frame'];boxes=scenes[(u,c)]['boxes'][:-2]
                assert row['union_main3m']==old['union_path_invasion'] and row['OR_added_mode2']==old['OR_added_mode2']
                records=[];flags=[False,False]
                assert len(boxes)==len(row['objects'])
                for b,(box,actual) in enumerate(zip(boxes,row['objects'])):
                    horizon=.16*(15-f)+max(0.,box['hi'][2]+.30)
                    assert abs(horizon-actual['horizon_arclength_m'])<1e-12
                    assert abs(actual['endpoint_world_z']-max(0.,box['hi'][2]+.30))<1e-12
                    hit=path_box_hit(u,f,box,horizon)
                    assert actual['horizontal_path']['invasion']==(hit is not None)
                    vertical=[min(box['hi'][1],high)-max(box['lo'][1],low)>1e-8 for low,high in HEIGHTS]
                    assert actual['same_query_invasion']==bool(hit and vertical[q])
                    assert actual['other_height_invasion']==bool(hit and vertical[1-q])
                    if hit is not None:
                        assert abs(actual['horizontal_path']['first_arclength_m']-hit['arc'])<1e-10
                        assert abs(actual['horizontal_path']['first_time_s']-hit['arc']/.8)<1e-10
                        assert actual['horizontal_path']['recorded']==hit['recorded']
                        assert actual['horizontal_path']['extrapolated']==hit['extrapolated']
                        for j in range(2):flags[j]|=vertical[j]
                        if any(vertical):records.append(hit)
                    objects+=1
                assert row['same_query_full_pass']==flags[q] and row['other_height_full_pass']==flags[1-q]
                assert row['union_full_pass']==any(flags)
                assert row['recorded_full_pass']==any(x['recorded'] for x in records)
                assert row['extrapolated_only_full_pass']==bool(records and not any(x['recorded'] for x in records))
                assert row['exclusive_full_pass']==('SAME_QUERY_PATH' if flags[q] else 'OTHER_HEIGHT_ONLY' if flags[1-q] else 'PATH_CLEAR')
                checked+=1
    primary=[r for r in rows if r['OR_added_mode2']]
    for key,selected in [('96000',[r for r in primary if r['batch']==96000]),('97000',[r for r in primary if r['batch']==97000]),('pooled',primary)]:
        cell=value['primary'][key];assert cell['n']==len(selected)
        for field,source in [('union_full_pass','union_full_pass'),('same_query','same_query_full_pass'),('other_height','other_height_full_pass'),('recorded_path','recorded_full_pass'),('extrapolated_only','extrapolated_only_full_pass')]:assert cell[field]==sum(r[source] for r in selected)
        assert cell['beyond_main3m_added']==sum(r['union_full_pass'] and not r['union_main3m'] for r in selected)
        assert cell['rate']==cell['union_full_pass']/cell['n']
    return dict(status='PASS',event_rows=checked,finite_object_intersections=objects,primary_n=len(primary),primary_union=sum(r['union_full_pass'] for r in primary),beyond3m_added=sum(r['union_full_pass'] and not r['union_main3m'] for r in primary),source_sha256=sha(OUT/'full_object_pass_sensitivity.json'))


def independent_clip(poly,normal):
    out=[]
    for i in range(len(poly)):
        a,b=poly[i],poly[(i+1)%len(poly)];da=float(normal@a);db=float(normal@b)
        if da<=0:out.append(a)
        if (da<=0)!=(db<=0):out.append(a+(b-a)*da/(da-db))
    return np.asarray(out).reshape(-1,3)


def independent_fov_surface(box,rotation,position):
    lo,hi=np.asarray(box['lo']),np.asarray(box['hi']);edge=np.tan(np.pi/8);total=0.;cut=0.
    for fixed in range(3):
        other=[j for j in range(3) if j!=fixed]
        for side in (lo[fixed],hi[fixed]):
            face=[]
            for a,b in ((0,0),(1,0),(1,1),(0,1)):
                p=np.zeros(3);p[fixed]=side;p[other[0]]=(lo,hi)[a][other[0]];p[other[1]]=(lo,hi)[b][other[1]];face.append(p)
            poly=(np.asarray(face)-position)@rotation;total+=np.prod((hi-lo)[other])
            for normal in ([1,0,-edge],[-1,0,-edge],[0,1,-edge],[0,-1,-edge],[0,0,-1]):
                if not len(poly):break
                poly=independent_clip(poly,np.asarray(normal))
            if len(poly)>=3:cut+=sum(np.linalg.norm(np.cross(poly[j]-poly[0],poly[j+1]-poly[0]))/2 for j in range(1,len(poly)-1))
    return total,cut


def shallow_check():
    import csv
    import cnh_cvr_pilot as CP
    import cnh_dual_gated_geometry as GF
    work=OUT/'shallow';value=read(work/'result.json');all_cases=[];frames_checked=0;area_error=0.;scores_checked=0
    assert value['source_sha256']==sha(Path(__file__).with_name('cnh_turn_path_shallow.py'))
    for batch in (96000,97000):
        d,scenes=load_independent(batch);ss,st,sf=first_flags(d['single'],d['ranges']);os,ot,of=first_flags(d['sensor'].max(1),d['ranges'])
        shallow=d['shallow'] if 'shallow' in d else d['ref_category']=='contact0-2cm'
        den=(d['unit']%3==2)&d['covered']&shallow;chosen=np.flatnonzero(den&st&~ot)
        summaries=value['batches'][str(batch)]
        assert (summaries['mode2_covered_shallow_n'],summaries['single_timely'],summaries['OR_timely'],summaries['single_timely_OR_untimely'],summaries['OR_rescues'])==(int(den.sum()),int((den&st).sum()),int((den&ot).sum()),len(chosen),int((den&~st&ot).sum()))
        expected={(int(d['unit'][i]),int(d['config'][i]),int(d['query'][i])) for i in chosen}
        declared={(r['unit'],r['config'],('HEAD','BODY').index(r['query'])) for r in value['cases'] if r['batch']==batch};assert expected==declared
        for i in chosen:
            u,c,q=(int(d[k][i]) for k in ('unit','config','query'));case=read(work/f'batch{batch}_unit{u}_config{c}_query{q}.json');all_cases.append(case)
            box=scenes[(u,c)]['boxes'][0];lo,hi=np.asarray(box['lo']),np.asarray(box['hi'])
            points=np.asarray([[lo[j] if not (k>>j)&1 else hi[j] for j in range(3)] for k in range(8)])
            sensor,travel,noisy=CP.motion_metadata(u,c);np.testing.assert_allclose(travel,travel_matrix(u),atol=1e-12,rtol=0)
            coverage=GF.query_coverage(GF.estimated_query_poses(noisy))[:,:,q]
            if batch==96000:
                root=ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
                with np.load(root/'frame_scores_M3_early.npz') as e,np.load(root/'frame_scores_M3.npz') as l:rawsingle=np.concatenate((e[str(u)][c],l[str(u)][c]),axis=0)
                with np.load(ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural/scores'/f'unit{u}.npz') as z:rawbranch=z['raw'][:,c]
            else:
                with np.load(ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005/natural97000/scores'/f'unit{u}.npz') as z:rawsingle=z['raw'][0,c];rawbranch=z['raw'][1:,c]
            fraction=case['deadline']['reference_fraction'];left=int(np.floor(fraction));alpha=fraction-left
            position=(1-alpha)*travel[left,:3,3]+alpha*travel[left+1,:3,3];yaw=np.deg2rad(-20+fraction*20/15)
            reference=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
            assert abs(((points-position)@reference)[:,2].min()-.9)<1e-10
            assert case['original_single_first_frame']==sf[i]+3 and case['original_OR_first_frame']==of[i]+3
            for t,row in enumerate(case['frames']):
                f=t+3;start=max(0,t-4);w=2.**np.arange(t-start+1)
                smoothsingle=float(np.average(rawsingle[start:t+1,q],weights=w));smoothbranch=np.average(rawbranch[:,start:t+1,q],axis=1,weights=w)
                np.testing.assert_allclose([row['single_logit'],row['L_logit'],row['R_logit']],[smoothsingle,*smoothbranch],atol=1e-12,rtol=0);scores_checked+=3
                np.testing.assert_allclose([row['f_L'],row['f_R']],coverage[:,t],atol=1e-12,rtol=0)
                assert row['predeadline']==(f<=fraction+1e-12)
                for branch,name in enumerate(('single','L','R')):
                    if branch==0:rotation=sensor[f,:3,:3]
                    else:
                        a=np.deg2rad((-15,15)[branch-1]);ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
                        p=np.deg2rad(-10);pitch=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
                        ext=pitch.T@ry@pitch;rotation=sensor[f,:3,:3]@ext
                    geom=row['sensors'][name];local=(points-sensor[f,:3,3])@rotation;center=((lo+hi)/2-sensor[f,:3,3])@rotation
                    horizontal=np.rad2deg(np.arctan2(center[0],center[2]));vertical=np.rad2deg(np.arctan2(center[1],center[2]));margin=22.5-max(abs(horizontal),abs(vertical))
                    np.testing.assert_allclose([geom['center_horizontal_deg'],geom['center_vertical_deg'],geom['center_margin_deg']],[horizontal,vertical,margin],atol=1e-10,rtol=0)
                    inside=(local[:,2]>0)&(abs(local[:,0])<=local[:,2]*np.tan(np.pi/8)+1e-12)&(abs(local[:,1])<=local[:,2]*np.tan(np.pi/8)+1e-12)
                    assert geom['corner_inside_count']==int(inside.sum())
                    total,cut=independent_fov_surface(box,rotation,sensor[f,:3,3]);err=abs(cut-geom['surface_area_in_fov_m2']);area_error=max(area_error,err);assert err<1e-10
                    assert abs(total-geom['total_surface_area_m2'])<1e-10 and geom['surface_any_positive_area']==(cut>1e-12)
                    frames_checked+=1
            assert q==scenes[(u,c)]['group'] and case['nominal_target_reference_category']=='contact0-2cm'
    with (work/'frames.csv').open(encoding='utf8',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==frames_checked==234
    common=value['common_feature_counts']['counts'];assert common['dual_has_geometric_surface_support_every_predeadline_frame']==sum(all(any(r['sensors'][b]['surface_any_positive_area'] for b in ('L','R')) for r in c['frames'] if r['predeadline']) for c in all_cases)
    assert common['at_single_first_report_target_surface_inside_both_branches']==sum(all(c['frames'][c['original_single_first_frame']-3]['sensors'][b]['surface_any_positive_area'] for b in ('L','R')) for c in all_cases)
    return dict(status='PASS',selected_cases=len(all_cases),all_motion_mode=2,frame_sensor_geometry_rows=frames_checked,raw_smoothed_scores_checked=scores_checked,independent_face_clipping_max_error=area_error,public_coverage_recomputed=True,deadline_fractional_pose_checked=True,firsthit_occupancy='Analytic ray/AABB implementation inspected; independent occupancy checks in followup receipt',result_sha256=sha(work/'result.json'))


def polygon_clip_2d(poly,normal,bound):
    lifted=np.column_stack((poly,np.ones(len(poly))))
    clipped=independent_clip(lifted,np.array([-normal[0],-normal[1],bound]))
    return clipped[:,:2]


def flat_hit_independent(unit,frame,box):
    segs=independent_segments(unit,frame);start=segs[0][0];end=segs[-1][1]
    first=segs[0][1]-start;first/=np.linalg.norm(first);last=end-segs[-1][0];last/=np.linalg.norm(last)
    lo=np.asarray(box['lo'])[[0,2]];hi=np.asarray(box['hi'])[[0,2]]
    poly=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    poly=polygon_clip_2d(poly,first,float(first@start))
    if not len(poly):return False
    poly=polygon_clip_2d(poly,-last,float(-last@end))
    if len(poly)<3:return False
    area=sum(np.cross(poly[j]-poly[0],poly[j+1]-poly[0]) for j in range(1,len(poly)-1))/2
    if abs(area)<=1e-16:return False
    def inside(p):
        cross=[float(np.cross(poly[(j+1)%len(poly)]-poly[j],p-poly[j])) for j in range(len(poly))]
        return min(cross)>=-1e-12
    def edge_distance(a,b,c,d):
        v=b-a;w=d-c;cross=float(np.cross(v,w))
        if abs(cross)>1e-15:
            t=float(np.cross(c-a,w))/cross;s=float(np.cross(c-a,v))/cross
            if 0<=t<=1 and 0<=s<=1:return 0.
        return min(point_segment_distance(a,c,d),point_segment_distance(b,c,d),point_segment_distance(c,a,b),point_segment_distance(d,a,b))
    for a,b,*_ in segs:
        if inside(a) or inside(b):return True
        if min(edge_distance(a,b,poly[j],poly[(j+1)%len(poly)]) for j in range(len(poly)))<RADIUS:return True
    return False


def sensitivity_check():
    value=read(OUT/'sensitivity_flat_endcaps_amendment.json');checked=0;objects=0
    assert value['main_result_sha256']==sha(OUT/'path_result.json')
    for batch in (96000,97000):
        _,scenes=load_independent(batch)
        for row in (r for r in value['events'] if r['batch']==batch):
            flags=[False,False]
            for box in scenes[(row['unit'],row['config'])]['boxes']:
                hit=flat_hit_independent(row['unit'],row['frame'],box)
                for q,(lo,hi) in enumerate(HEIGHTS):flags[q]|=hit and min(box['hi'][1],hi)-max(box['lo'][1],lo)>1e-8
                objects+=1
            q=row['query'];assert (row['same_query_flat'],row['other_height_flat'],row['union_flat'])==(flags[q],flags[1-q],any(flags));checked+=1
    primary=[r for r in value['events'] if r['OR_added_mode2']]
    assert value['primary']['pooled']['union_flat']==sum(r['union_flat'] for r in primary)==35
    assert not any(r['changed'] for r in value['events'])
    # A backward object behind a closely spaced first interior vertex must be excluded.
    assert not flat_hit_independent(96000,15,dict(lo=[-.05,0,-.2],hi=[.05,.1,-.1]))
    return dict(status='PASS',event_rows=checked,object_intersections=objects,primary_n=len(primary),flat_union=35,changed=0,global_backwards_endplane_case=True,amendment_sha256=sha(OUT/'sensitivity_flat_endcaps_amendment.json'))


def occupancy_check():
    import cnh_cvr_pilot as CP
    value=read(OUT/'shallow/result.json');count=0;ray_cases=0;nearest_error=0.
    edge=np.tan(np.pi/8);axis=(np.arange(129)+.5)*2*edge/129-edge
    xx,yy=np.meshgrid(axis,axis);local=np.column_stack((xx.ravel(),yy.ravel(),np.ones(xx.size)));local/=np.linalg.norm(local,axis=1)[:,None]
    for brief in value['cases']:
        case=read(OUT/'shallow'/brief['payload']);u,c=case['unit'],case['config'];_,scenes=load_independent(case['batch']);boxes=scenes[(u,c)]['boxes'];sensor=CP.motion_metadata(u,c)[0]
        for row in case['frames']:
            if not row['predeadline']:continue
            f=row['frame']
            for branch,name in enumerate(('single','L','R')):
                yaw=np.deg2rad(-20+f*20/15+(0,-15,15)[branch]);pitch=np.deg2rad(-10)
                ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
                rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
                direction=local@(ry@rx).T;origin=sensor[f,:3,3];allhits=[]
                for box in boxes:
                    lower=np.full(len(local),-np.inf);upper=np.full(len(local),np.inf)
                    for j in range(3):
                        parallel=abs(direction[:,j])<1e-14
                        safe=np.where(parallel,1.,direction[:,j]);a=(box['lo'][j]-origin[j])/safe;b=(box['hi'][j]-origin[j])/safe
                        within=box['lo'][j]<=origin[j]<=box['hi'][j]
                        lower=np.maximum(lower,np.where(parallel,-np.inf if within else np.inf,np.minimum(a,b)))
                        upper=np.minimum(upper,np.where(parallel,np.inf if within else -np.inf,np.maximum(a,b)))
                    allhits.append(np.where((upper>=np.maximum(lower,0))&(upper>0),np.maximum(lower,0),np.inf))
                allhits=np.array(allhits);nearest=allhits.min(0);target=np.isfinite(allhits[0]);visible=target&(allhits.argmin(0)==0);actual=row['sensors'][name]['ray_occupancy']
                assert actual['target_first_hit_rays']==int(visible.sum())
                assert actual['target_ray_intersection_without_occlusion_rays']==int(target.sum())
                assert actual['target_occluded_ray_count']==int((target&~visible).sum())
                assert actual['target_first_hit_within_2p5m_rays']==int((visible&(nearest<=2.5)).sum())
                if visible.any():
                    err=abs(actual['target_nearest_first_hit_range_m']-float(nearest[visible].min()));assert err<1e-10;nearest_error=max(nearest_error,err)
                else:assert actual['target_nearest_first_hit_range_m'] is None
                count+=1;ray_cases+=len(local)
    assert count==204
    return dict(status='PASS',predeadline_frame_sensor_rows=count,ray_cases=ray_cases,nearest_range_max_error=nearest_error,semantics='Independent physical yaw/pitch and all-box first-positive slab intersection; analytic finite grid, not photon returns')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['input','geometry','outcome','fullpass','shallow','sensitivity','occupancy']);args=parser.parse_args()
    tick=time.monotonic();value={'input':input_check,'geometry':geometry_check,'outcome':outcome_check,'fullpass':fullpass_check,'shallow':shallow_check,'sensitivity':sensitivity_check,'occupancy':occupancy_check}[args.stage]();value['elapsed_seconds']=time.monotonic()-tick
    value['check_source_sha256']=sha(__file__);record(args.stage+'.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
