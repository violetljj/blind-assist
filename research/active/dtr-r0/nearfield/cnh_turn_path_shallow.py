"""CPU-only turn-path shallow-loss diagnosis on sealed96000/97000.

This post-hoc evaluator reads physical boxes only to diagnose selected events.
No new rendering, observations, training, thresholds or fusion rules are made.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-turn-path-diagnostic-20261005'
WORK=OUT/'shallow'
GATED=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005'
OLD=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural'
SOURCE=ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
FRAMES=np.arange(3,16)
THRESHOLD=.8557642486787612
EDGE=float(np.tan(np.pi/8))
NS=np.array([[1,0,-EDGE],[-1,0,-EDGE],[0,1,-EDGE],[0,-1,-EDGE],[0,0,-1.]])

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def deadline():
    plan=read(OUT/'PLAN.json')
    if time.time()>=plan['deadline_unix']:raise TimeoutError('Parent40min CPU budget reached')

def freeze():
    deadline();WORK.mkdir(parents=True,exist_ok=True)
    p=WORK/'FRAME_GEOMETRY_CONTRACT.json'
    if p.exists():return read(p)
    import cnh_sequence_observed_geometry as GE
    import cnh_dual_gated_geometry as GF
    result=dict(parent_plan_sha256=sha(OUT/'PLAN.json'),frozen_unix=time.time(),threshold=THRESHOLD,
        selection='Each batch mode2 AND covered AND all-object reference category contact0-2cm; original single timely AND original two-branch OR untimely; counts independently reconstructed from raw, never forced to3+3',
        smoothing='Original causal lastup-to5 retained decision-frame logits with weights1,2,4,8,16; separately per single/L/R, max forOR afterwards',
        frames='All saved physicalframes3..15; public cached coverage retained; evaluator true current sensor axes for target geometry only',
        nominal_target='boxes[0]; its all-object-category contribution and target group are checked separately; shallow reference may come from another object',
        bearings='atan2(sensor-x,sensor-z) and atan2(sensor-y,sensor-z); square45deg FOV signed angularmargin22.5-max(abs(horizontal),abs(vertical))',
        distinctions=['Eightcorners inside count and bearing ranges','Geometric box-center inside and signed angularmargin','Actual12triangles positive-area surface clipped byfive opticalFOV halfspaces; fraction of total target surface area, including hidden faces'],
        predeadline='Savedframes whosephysicalframe<=exact fractional0.9m target-front crossing; no extrapolated frame outcome',
        deadline='Exact original eight-corner minz, linear position and yaw interpolation plus50 bisection steps; saved geometry reference cross-checked',
        ray_occupancy='Only predeadline frames: fixed129x129 normalized optical midpoint rays, first positive ray/AABB hit across every scene object; report target first-hit count and nearest radialrange. Finite-grid occupancy is not photon visibility or causalM3 evidence',
        support='Positive target-surface intersection with FOV; separately center/corners/first-hit sampled support. Surface support may be occluded; lastsupport distance/time and edgegap reported',
        source_sha256={str(Path(__file__)):sha(__file__),str(Path(GE.__file__)):sha(GE.__file__),str(Path(GF.__file__)):sha(GF.__file__)})
    write(p,result);return result

def smooth(raw):
    raw=np.asarray(raw,float)
    if raw.shape[-2:]!=(13,2) or not np.isfinite(raw).all():raise ValueError('13 retained finite rawframes required')
    out=np.empty_like(raw)
    for t in range(13):
        ids=np.arange(max(0,t-4),t+1);w=2.**np.arange(len(ids))
        out[...,t,:]=sum(raw[...,i,:]*v for i,v in zip(ids,w))/w.sum()
    return out

def event(scores,ranges):
    alarm=scores>=THRESHOLD;stopped=alarm.any(-1);first=alarm.argmax(-1)
    distance=np.take_along_axis(ranges,first[...,None],axis=-1)[...,0]
    return dict(stopped=stopped,first=first,first_range=distance,timely=stopped&(distance>=.9))

def corners(box):
    lo=np.asarray(box['lo']);hi=np.asarray(box['hi'])
    return np.array([[lo[j] if not (i>>j)&1 else hi[j] for j in range(3)] for i in range(8)])

def mesh(box):
    v=corners(box);faces=[(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    return np.array([v[list(t)] for a,b,c,d in faces for t in [(a,b,c),(a,c,d)]])

def clip_polygon(poly):
    poly=np.asarray(poly,float)
    for n in NS:
        if not len(poly):break
        result=[];last=poly[-1];dl=float(last@n)
        for cur in poly:
            dc=float(cur@n);il=dl<=1e-12;ic=dc<=1e-12
            if il!=ic:
                result.append(last+dl/(dl-dc)*(cur-last))
            if ic:result.append(cur)
            last,dl=cur,dc
        poly=np.asarray(result).reshape(-1,3)
    return poly

def area(poly):
    if len(poly)<3:return 0.
    return float(sum(np.linalg.norm(np.cross(poly[i]-poly[0],poly[i+1]-poly[0]))/2 for i in range(1,len(poly)-1)))

def point_geometry(points):
    points=np.asarray(points,float)
    h=np.rad2deg(np.arctan2(points[...,0],points[...,2]));v=np.rad2deg(np.arctan2(points[...,1],points[...,2]))
    margin=22.5-np.maximum(abs(h),abs(v));inside=(points[...,2]>0)&(margin>=-1e-10)
    return h,v,margin,inside

def target_fov(box,pose):
    verts=(corners(box)-pose[:3,3])@pose[:3,:3]
    center=((np.asarray(box['lo'])+np.asarray(box['hi']))/2-pose[:3,3])@pose[:3,:3]
    ch,cv,cm,ci=point_geometry(verts);hh,vv,mm,ii=point_geometry(center)
    triangles=(mesh(box)-pose[:3,3])@pose[:3,:3]
    polys=[clip_polygon(t) for t in triangles];total=sum(area(t) for t in triangles);cut=sum(area(p) for p in polys)
    support=cut>1e-12
    inside_vertices=np.concatenate([p for p in polys if area(p)>1e-12]) if support else np.empty((0,3))
    gap=float(np.maximum(0,point_geometry(inside_vertices)[2]).min()) if support else None
    return dict(center_xyz_m=center.tolist(),center_horizontal_deg=float(hh),center_vertical_deg=float(vv),center_margin_deg=float(mm),center_inside=bool(ii),
        corner_xyz_m=verts.tolist(),corner_horizontal_deg=ch.tolist(),corner_vertical_deg=cv.tolist(),corner_margin_deg=cm.tolist(),corner_inside=ci.tolist(),corner_inside_count=int(ci.sum()),
        corner_horizontal_span_deg=[float(ch.min()),float(ch.max())],corner_vertical_span_deg=[float(cv.min()),float(cv.max())],
        any_corner_inside=bool(ci.any()),all_corners_inside=bool(ci.all()),surface_any_positive_area=support,
        total_surface_area_m2=total,surface_area_in_fov_m2=cut,surface_fraction_in_fov=cut/total,
        nearest_inside_surface_edge_margin_deg=gap,surface_boundary_crossing=bool(support and gap<=1e-9))

def first_hit_occupancy(boxes,pose,n=129):
    a=(np.arange(n)+.5)/n*2*EDGE-EDGE;x,y=np.meshgrid(a,a)
    rays=np.stack((x,y,np.ones_like(x)),-1).reshape(-1,3);rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    directions=rays@pose[:3,:3].T;origin=pose[:3,3];dist=[]
    for box in boxes:
        low=np.asarray(box['lo']);high=np.asarray(box['hi']);parallel=abs(directions)<1e-14
        safe=np.where(parallel,1.,directions)
        aa=(low-origin)/safe;bb=(high-origin)/safe
        near=np.minimum(aa,bb);far=np.maximum(aa,bb)
        inparallel=(origin>=low)&(origin<=high)
        near=np.where(parallel,np.where(inparallel,-np.inf,np.inf),near)
        far=np.where(parallel,np.where(inparallel,np.inf,-np.inf),far)
        entry=near.max(1);exit=far.min(1);hit=(exit>=np.maximum(entry,0))&(exit>0)
        dist.append(np.where(hit,np.maximum(entry,0),np.inf))
    dist=np.asarray(dist);near=dist.min(0);hit=np.isfinite(near)&(dist.argmin(0)==0)
    unobscured=np.isfinite(dist[0]);num=int(hit.sum())
    return dict(grid_n=n,grid_rays=n*n,target_first_hit_rays=num,target_first_hit_grid_fraction=num/(n*n),
        target_ray_intersection_without_occlusion_rays=int(unobscured.sum()),target_occluded_ray_count=int((unobscured&~hit).sum()),
        target_first_hit_within_2p5m_rays=int((hit&(near<=2.5)).sum()),target_nearest_first_hit_range_m=float(near[hit].min()) if num else None)

def load_batch(batch):
    """Score and metadata materialization only after frozen parent/frame contracts."""
    import cnh_sequence_observed_geometry as GE
    inputs={}
    if batch==96000:
        lp=OLD/'ledger.npz';gp=GE.OUT/'geometry.npz';mp=SOURCE/'scene_manifest.json';fp=GATED/'diagnostic/coverage_96000.npz'
        with np.load(lp,allow_pickle=False) as z:ledger={k:z[k].copy() for k in z.files}
        with np.load(gp,allow_pickle=False) as z:
            keep=(z['split']=='evaluation')&np.isin(z['unit'],ledger['unit'])
            geometry={k:z[k][keep].copy() for k in z.files if z[k].shape[:1]==keep.shape}
        with np.load(fp,allow_pickle=False) as z:
            coverage=z['coverage'].copy()
            for k in ('unit','config','query'):np.testing.assert_array_equal(z[k],ledger[k])
        manifests={(r['unit'],r['config']):r for r in read(mp) if r['split']=='evaluation'}
        single=ledger['single'];branches=ledger['sensor_scores']
    else:
        lp=GATED/'fusion/natural97000_ledger.npz';gp=GATED/'natural97000/geometry.npz';mp=GATED/'natural97000/scene_manifest.json';fp=lp
        with np.load(lp,allow_pickle=False) as z:ledger={k:z[k].copy() for k in z.files}
        with np.load(gp,allow_pickle=False) as z:geometry={k:z[k].copy() for k in z.files if k!='frames'}
        coverage=ledger['coverage'];manifests={(r['unit'],r['config']):r for r in read(mp)};single=ledger['single'];branches=ledger['sensor']
    for k in ('unit','config','query'):np.testing.assert_array_equal(geometry[k],ledger[k])
    ranges=geometry['frame_ranges'];units=np.unique(ledger['unit']);raw_single=[];raw_branch=[]
    if batch==96000:
        early=SOURCE/'frame_scores_M3_early.npz';late=SOURCE/'frame_scores_M3.npz'
        inputs[str(early)]=sha(early);inputs[str(late)]=sha(late)
        with np.load(early,allow_pickle=False) as e,np.load(late,allow_pickle=False) as l:
            for u in units:
                raw_single.append(np.concatenate((e[str(u)],l[str(u)]),axis=1))
                p=OLD/'scores'/f'unit{u}.npz'
                with np.load(p,allow_pickle=False) as z:raw_branch.append(z['raw'].copy())
                inputs[str(p)]=sha(p)
    else:
        for u in units:
            p=GATED/'natural97000/scores'/f'unit{u}.npz'
            with np.load(p,allow_pickle=False) as z:raw_single.append(z['raw'][0].copy());raw_branch.append(z['raw'][1:].copy())
            inputs[str(p)]=sha(p)
    rebuilt_single=smooth(np.stack(raw_single)).transpose(0,1,3,2).reshape(-1,13)
    rebuilt_branch=smooth(np.stack(raw_branch)).transpose(0,2,4,1,3).reshape(-1,2,13)
    np.testing.assert_allclose(rebuilt_single,single,atol=1e-12,rtol=0)
    np.testing.assert_allclose(rebuilt_branch,branches,atol=1e-12,rtol=0)
    single_ev=event(rebuilt_single,ranges);or_ev=event(rebuilt_branch.max(1),ranges)
    den=(ledger['unit']%3==2)&geometry['covered']&(geometry['ref_category']=='contact0-2cm')
    selected=den&single_ev['timely']&~or_ev['timely']
    for p in (lp,gp,mp,fp):inputs[str(p)]=sha(p)
    return dict(batch=batch,ledger=ledger,geometry=geometry,manifest=manifests,single=rebuilt_single,branches=rebuilt_branch,coverage=coverage,
        single_ev=single_ev,or_ev=or_ev,selected=np.flatnonzero(selected),den=den,inputs=inputs,
        parity=dict(single_raw_smoothed_max_abs=float(abs(rebuilt_single-single).max()),branches_raw_smoothed_max_abs=float(abs(rebuilt_branch-branches).max())))

def brief_support(frames,sensor):
    f=[r for r in frames if r['predeadline']];supported=[r for r in f if r['sensors'][sensor]['surface_any_positive_area']];visible=[r for r in f if r['sensors'][sensor]['ray_occupancy']['target_first_hit_rays']>0]
    result=dict(predeadline_frames=len(f),center_inside_frames=sum(r['sensors'][sensor]['center_inside'] for r in f),
        any_corner_inside_frames=sum(r['sensors'][sensor]['any_corner_inside'] for r in f),all_corners_inside_frames=sum(r['sensors'][sensor]['all_corners_inside'] for r in f),
        positive_surface_frames=len(supported),sampled_first_hit_frames=len(visible),surface_boundary_crossing_frames=sum(r['sensors'][sensor]['surface_boundary_crossing'] for r in f))
    for name,rows in [('surface',supported),('sampled_first_hit',visible)]:
        result['last_'+name+'_frame']=rows[-1]['frame'] if rows else None
        result['last_'+name+'_front_range_m']=rows[-1]['target_front_range_m'] if rows else None
    if supported:result.update(minimum_surface_fraction=min(r['sensors'][sensor]['surface_fraction_in_fov'] for r in supported),nearest_edge_margin_deg=min(r['sensors'][sensor]['nearest_inside_surface_edge_margin_deg'] for r in supported))
    return result

def run():
    tick=time.monotonic();plan=freeze();deadline()
    if (WORK/'result.json').exists():raise FileExistsError('Completed diagnosis immutable')
    import cnh_cvr_pilot as CP
    import cnh_dual_gated_geometry as GF
    import cnh_sequence_observed_geometry as GE
    cases=[];batch_summaries={};inputs={};csvrows=[]
    for batch in (96000,97000):
        deadline();d=load_batch(batch);g=d['geometry'];l=d['ledger'];inputs.update(d['inputs'])
        den=d['den'];batch_summaries[str(batch)]=dict(mode2_covered_shallow_n=int(den.sum()),single_timely=int((den&d['single_ev']['timely']).sum()),
            OR_timely=int((den&d['or_ev']['timely']).sum()),single_timely_OR_untimely=int(len(d['selected'])),OR_rescues=int((den&~d['single_ev']['timely']&d['or_ev']['timely']).sum()),parity=d['parity'])
        for i in d['selected']:
            deadline();u,c,q=int(l['unit'][i]),int(l['config'][i]),int(l['query'][i]);scene=d['manifest'][(u,c)];boxes=scene['boxes'];target=boxes[0]
            sensor,travel,noisy=CP.motion_metadata(u,c);ranges,reference=GE.deadline_reference(corners(target),travel[FRAMES])
            np.testing.assert_allclose(ranges,g['frame_ranges'][i],atol=1e-12,rtol=0)
            for key in ('reference_pose','reference_fraction','reference_time_s','interpolation_alpha'):np.testing.assert_allclose(reference[key],g[key][i],atol=1e-12,rtol=0)
            physical=np.stack([sensor,sensor@GF.extrinsic(-15),sensor@GF.extrinsic(15)])
            recomputed=GF.query_coverage(GF.estimated_query_poses(noisy))[:,:,q]
            np.testing.assert_allclose(recomputed,d['coverage'][i],atol=1e-12,rtol=0)
            reference_categories=[GE.surface_category(GE.transform(mesh(b),reference['reference_pose']),q) for b in boxes]
            frames=[]
            for t,f in enumerate(FRAMES):
                pre=bool(f<=reference['reference_fraction']+1e-12)
                row=dict(frame=int(f),time_from_first_decision_s=float((f-3)*.2),time_to_deadline_s=float((reference['reference_fraction']-f)*.2),
                    predeadline=pre,target_front_range_m=float(ranges[t]),single_logit=float(d['single'][i,t]),L_logit=float(d['branches'][i,0,t]),R_logit=float(d['branches'][i,1,t]),
                    OR_logit=float(d['branches'][i,:,t].max()),f_L=float(d['coverage'][i,0,t]),f_R=float(d['coverage'][i,1,t]),
                    frame_actual_query_category=str(g['frame_category'][i,t]),sensors={})
                for b,name in enumerate(('single','L','R')):
                    geom=target_fov(target,physical[b,f])
                    if pre:geom['ray_occupancy']=first_hit_occupancy(boxes,physical[b,f])
                    row['sensors'][name]=geom
                    flat=dict(batch=batch,unit=u,config=c,query=q,frame=int(f),sensor=name,predeadline=pre,range_m=float(ranges[t]),
                        single_logit=row['single_logit'],L_logit=row['L_logit'],R_logit=row['R_logit'],f_L=row['f_L'],f_R=row['f_R'],
                        horizontal_deg=geom['center_horizontal_deg'],vertical_deg=geom['center_vertical_deg'],center_margin_deg=geom['center_margin_deg'],
                        center_inside=geom['center_inside'],corner_inside_count=geom['corner_inside_count'],surface_any=geom['surface_any_positive_area'],surface_fraction=geom['surface_fraction_in_fov'],
                        nearest_surface_edge_margin_deg=geom['nearest_inside_surface_edge_margin_deg'],target_first_hit_rays=geom['ray_occupancy']['target_first_hit_rays'] if pre else '')
                    csvrows.append(flat)
                frames.append(row)
            support={b:brief_support(frames,b) for b in ('single','L','R')}
            single_t=int(d['single_ev']['first'][i]);or_stopped=bool(d['or_ev']['stopped'][i]);or_t=int(d['or_ev']['first'][i]) if or_stopped else None
            case=dict(batch=batch,unit=u,config=c,query=q,query_name=('HEAD','BODY')[q],nominal_target_group=int(scene['group']),query_matches_nominal_target_group=q==int(scene['group']),
                nominal_target=target,scene_condition=scene.get('cond'),all_object_reference_category=str(g['ref_category'][i]),nominal_target_reference_category=reference_categories[0],
                object_reference_categories=reference_categories,
                deadline={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in reference.items()},
                original_single_first_frame=int(FRAMES[single_t]),original_single_first_range_m=float(ranges[single_t]),
                original_OR_stopped=or_stopped,original_OR_first_frame=int(FRAMES[or_t]) if or_stopped else None,original_OR_first_range_m=float(ranges[or_t]) if or_stopped else None,
                predeadline_support=support,frames=frames)
            name=f'batch{batch}_unit{u}_config{c}_query{q}.json';write(WORK/name,case);case['payload']=name;cases.append(case)
    common={
        'query_is_HEAD':sum(c['query']==0 for c in cases),
        'query_matches_nominal_target_group':sum(c['query_matches_nominal_target_group'] for c in cases),
        'nominal_target_itself_shallow_at_reference':sum(c['nominal_target_reference_category']=='contact0-2cm' for c in cases),
        'dual_has_geometric_surface_support_every_predeadline_frame':sum(all(any(r['sensors'][b]['surface_any_positive_area'] for b in ('L','R')) for r in c['frames'] if r['predeadline']) for c in cases),
        'dual_has_sampled_target_firsthit_every_predeadline_frame':sum(all(any(r['sensors'][b]['ray_occupancy']['target_first_hit_rays']>0 for b in ('L','R')) for r in c['frames'] if r['predeadline']) for c in cases),
        'at_single_first_report_target_surface_inside_both_branches':sum(all(c['frames'][c['original_single_first_frame']-3]['sensors'][b]['surface_any_positive_area'] for b in ('L','R')) for c in cases)}
    keys=list(csvrows[0]) if csvrows else []
    with (WORK/'frames.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(csvrows)
    brief=[]
    for c in cases:
        first=c['frames'][c['original_single_first_frame']-3]
        last=max((r for r in c['frames'] if r['predeadline']),key=lambda r:r['frame'])
        brief.append(dict(batch=c['batch'],unit=c['unit'],config=c['config'],query=c['query_name'],scene_condition=c['scene_condition'],
            target_group_matches=c['query_matches_nominal_target_group'],target_reference_category=c['nominal_target_reference_category'],
            single_first_frame=c['original_single_first_frame'],single_first_range_m=c['original_single_first_range_m'],OR_first_frame=c['original_OR_first_frame'],OR_first_range_m=c['original_OR_first_range_m'],
            deadline_fractional_frame=c['deadline']['reference_fraction'],single_first_logits=[first['single_logit'],first['L_logit'],first['R_logit']],
            last_predeadline_logits=[last['single_logit'],last['L_logit'],last['R_logit']],single_first_f=[first['f_L'],first['f_R']],
            predeadline_support=c['predeadline_support'],payload=c['payload']))
    result=dict(status='COMPLETE',scope='Posthoc consumed synthetic geometry/score diagnosis; no new rendering/observations/training/fusion selection',
        batches=batch_summaries,total_selected_cases=len(cases),expected_count_note='3+3 was a prior expectation; actual counts independently reconstructed, not imposed',
        common_feature_counts=dict(denominator=len(cases),counts=common),cases=brief,
        contract_sha256=sha(WORK/'FRAME_GEOMETRY_CONTRACT.json'),parent_plan_sha256=sha(OUT/'PLAN.json'),input_sha256=inputs,source_sha256=sha(__file__),frames_csv_sha256=sha(WORK/'frames.csv'),seconds=time.monotonic()-tick,
        limits=['Physical sensor/boxes are evaluator-only diagnosis, not deployable inputs','Geometric target surface inFOV does not establish detected photons, signal quality or causalM3attribution','129x129 first-hit rays are analytic finite-grid geometry; very thin surfaces can be missed by sampling','Total surface fraction includes occluded/back-facing box faces; first-hit occupancy separately accounts for box occlusion','No model/threshold/fusion change or new evaluation claim; six selected events do not establish population prevalence'])
    write(WORK/'result.json',result);print(json.dumps(dict(batches=batch_summaries,common=result['common_feature_counts'],seconds=result['seconds']),ensure_ascii=False),flush=True)
    return result

def check():
    p=np.eye(4);inside=dict(lo=[-.01,-.01,1.],hi=[.01,.01,1.1]);outside=dict(lo=[1.,-.01,1.],hi=[1.1,.01,1.1])
    i=target_fov(inside,p);o=target_fov(outside,p)
    assert i['all_corners_inside'] and abs(i['surface_fraction_in_fov']-1)<1e-12
    assert not o['surface_any_positive_area'] and o['surface_fraction_in_fov']==0
    spanning=dict(lo=[-1.,-1.,1.],hi=[1.,1.,1.1]);s=target_fov(spanning,p)
    assert s['surface_any_positive_area'] and not s['any_corner_inside'] and s['surface_boundary_crossing']
    near=dict(lo=[-.05,-.05,.5],hi=[.05,.05,.6]);v=first_hit_occupancy([inside],p);blocked=first_hit_occupancy([inside,near],p)
    assert v['target_first_hit_rays']>0 and blocked['target_first_hit_rays']==0 and blocked['target_occluded_ray_count']>0
    sample=np.arange(13,dtype=float)[None,:,None]+np.zeros((1,13,2));sm=smooth(sample)
    assert sm[0,0,0]==0 and abs(sm[0,4,0]-98/31)<1e-12
    print('PASS exact inside/outside, surface crossing withoutcorner inclusion, firsthit occlusion, original causal smoothing')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','check','run']);args=p.parse_args()
    {'freeze':freeze,'check':check,'run':run}[args.stage]()
