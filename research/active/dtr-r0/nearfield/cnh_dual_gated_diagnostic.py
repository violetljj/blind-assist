"""Read-only natural96000 diagnostic. Evaluator boxes never enter fusion rules.

Nearest visible box is a geometric co-occurrence, not a causal M3 attribution.
First-hit ray/AABB visibility uses a fixed 129x129 angular midpoint grid.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005/natural'
MANIFEST = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002/scene_manifest.json'
OUT = ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005/diagnostic'
THRESHOLD = .8557642486787612
FRAMES = np.arange(3,16)
QUERY = ('HEAD','BODY')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Preserve existing evidence: '+str(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def summary(values):
    a=np.asarray(values,float)
    return dict(n=len(a),minimum=float(a.min()),q10=float(np.quantile(a,.1)),median=float(np.median(a)),q90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(n=0)

def rays(n=129):
    edge=np.tan(np.pi/8); a=(np.arange(n)+.5)/n*2*edge-edge
    x,y=np.meshgrid(a,a);r=np.stack((x,y,np.ones_like(x)),-1).reshape(-1,3)
    return r/np.linalg.norm(r,axis=1,keepdims=True)

def box_kind(box, index, n):
    if index==0:return 'boundary_target'
    if index==n-2:return 'floor'
    if index==n-1:return 'back_wall'
    return 'mixed_surface_panel'

def visible_boxes(boxes, sensor, travel, n=129, max_range=2.5):
    direction=rays(n)@sensor[:3,:3].T; origin=sensor[:3,3]
    dist=[]
    for box in boxes:
        lo=np.asarray(box['lo']);hi=np.asarray(box['hi'])
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(lo-origin)/direction;b=(hi-origin)/direction
        entry=np.minimum(a,b).max(1);exit=np.maximum(a,b).min(1)
        hit=(exit>=np.maximum(entry,0))&(exit>0)
        dist.append(np.where(hit,np.maximum(entry,0),np.inf))
    dist=np.asarray(dist);near=dist.min(0);chosen=dist.argmin(0);records=[]
    for b,box in enumerate(boxes):
        valid=(chosen==b)&(near<=max_range)
        if not valid.any():continue
        ray=int(np.flatnonzero(valid)[np.argmin(near[valid])]); point=origin+direction[ray]*near[ray]
        local=(point-travel[:3,3])@travel[:3,:3]
        corners=np.asarray(list(itertools.product(*zip(box['lo'],box['hi']))))
        corridor=(corners-travel[:3,3])@travel[:3,:3]
        xl,xh=float(corridor[:,0].min()),float(corridor[:,0].max())
        closest=0. if xl<=0<=xh else min(abs(xl),abs(xh))
        center=(xl+xh)/2
        side='spanning' if xl<=-.29 and xh>=.29 else 'R' if center>0 else 'L' if center<0 else 'center'
        records.append(dict(box_index=b,type=box_kind(box,b,len(boxes)),nearest_radial_m=float(near[ray]),
            first_hit_ray_count=int(valid.sum()),first_hit_ray_fraction=float(valid.mean()),
            nearest_point_corridor_xyz_m=local.tolist(),side=side,turn_side='inner' if side=='R' else 'outer' if side=='L' else side,
            lateral_extent_m=[xl,xh],query_corridor_edge_gap_m=closest-.29,query_corridor_overlap=closest<.29,
            nearest_point_edge_gap_m=abs(float(local[0]))-.29))
    return sorted(records,key=lambda x:x['nearest_radial_m'])

def run_geometry():
    tick=time.monotonic()
    import cnh_cvr_pilot as CP
    from cnh_dual_sensor_envelope_natural import extrinsic
    result=read(OLD/'result.json')
    with np.load(OLD/'ledger.npz') as d:ledger={k:d[k].copy() for k in d.files}
    manifest={(r['unit'],r['config']):r for r in read(MANIFEST) if r['split']=='evaluation'}
    single=ledger['single']>=THRESHOLD;branches=ledger['sensor_scores'];dual=branches.max(1)>=THRESHOLD
    ss=single.any(1);ds=dual.any(1);clear=ledger['clear'];mode2=ledger['unit']%3==2
    added=mode2&clear&ds&~ss;removed=mode2&clear&ss&~ds
    all_first=dual.argmax(1);ix=np.arange(len(dual));picks=branches[ix,:,all_first]
    winners=np.where(picks[:,0]>picks[:,1],0,np.where(picks[:,1]>picks[:,0],1,2))
    counts=Counter('L' if w==0 else 'R' if w==1 else 'tie' for w in winners[mode2&ds])
    assert counts==Counter(result['cells']['dual15_original_threshold']['mode2']['first_report_source'])
    rows=[];cached={}
    for i in np.flatnonzero(added):
        u=int(ledger['unit'][i]);c=int(ledger['config'][i]);q=int(ledger['query'][i]);f=int(FRAMES[all_first[i]])
        if (u,c) not in cached:cached[(u,c)]=CP.motion_metadata(u,c)
        sensor,travel,noisy=cached[(u,c)];branch=int(winners[i]);angle=(-15,15)[branch] if branch<2 else 0
        current=sensor[f]@extrinsic(angle);optical=travel[f,:3,:3].T@current[:3,2]
        signed_yaw=float(np.rad2deg(np.arctan2(optical[0],optical[2])))
        # In the frozen source, all mode2 turn yaws increase, i.e. toward +x (right).
        forward=travel[:,:3,2];yaw=np.unwrap(np.arctan2(forward[:,0],forward[:,2]));rate=float(np.rad2deg((yaw[f]-yaw[f-1])/.2))
        objects=visible_boxes(manifest[(u,c)]['boxes'],current,travel[f]);nearest=objects[0] if objects else None
        rows.append(dict(ledger_index=int(i),unit=u,config=c,query=QUERY[q],frame=f,first_branch=('L','R','tie')[branch],
            turn_yaw_rate_deg_s=rate,turn_direction='right' if rate>0 else 'left' if rate<0 else 'stationary',
            optical_axis_relative_travel_deg=signed_yaw,first_stop_range_m=float(ledger['frame_ranges'][i,all_first[i]]),
            first_scores_L_R=picks[i].tolist(),category=str(ledger['ref_category'][i]),family=manifest[(u,c)]['cond'],
            target_final_side='R' if sum(manifest[(u,c)]['boxes'][0]['lo'][0:1]+manifest[(u,c)]['boxes'][0]['hi'][0:1])>0 else 'L',
            nearest_visible_within_2p5m=nearest,visible_objects_within_2p5m=objects))
    aggregate=dict(n=len(rows),first_branch=dict(Counter(r['first_branch'] for r in rows)),query=dict(Counter(r['query'] for r in rows)),
        nearest_type=dict(Counter(r['nearest_visible_within_2p5m']['type'] if r['nearest_visible_within_2p5m'] else 'none_within_2p5m' for r in rows)),
        nearest_turn_side=dict(Counter(r['nearest_visible_within_2p5m']['turn_side'] if r['nearest_visible_within_2p5m'] else 'none' for r in rows)),
        family=dict(Counter(r['family'] for r in rows)),first_stop_range_m=summary([r['first_stop_range_m'] for r in rows]))
    cells={}
    for branch in ('L','R'):
        part=[r for r in rows if r['first_branch']==branch]
        cells[branch]=dict(n=len(part),nearest_type=dict(Counter(r['nearest_visible_within_2p5m']['type'] if r['nearest_visible_within_2p5m'] else 'none_within_2p5m' for r in part)),
            nearest_side=dict(Counter(r['nearest_visible_within_2p5m']['side'] if r['nearest_visible_within_2p5m'] else 'none' for r in part)),
            family=dict(Counter(r['family'] for r in part)),query=dict(Counter(r['query'] for r in part)),target_final_side=dict(Counter(r['target_final_side'] for r in part)))
    layout={}
    for family in ('none','corner'):
        layout[family]=dict(Counter('R' if r['boxes'][0]['lo'][0]+r['boxes'][0]['hi'][0]>0 else 'L' for r in manifest.values() if r['unit']%3==2 and r['cond']==family))
    save(OUT/'events.json',dict(status='COMPLETE',events=rows,sha256={str(OLD/'ledger.npz'):sha(OLD/'ledger.npz'),str(MANIFEST):sha(MANIFEST)},ray_grid=[129,129]))
    output=dict(status='COMPLETE',threshold=THRESHOLD,mode2_clear=dict(n=int((mode2&clear).sum()),single_stops=int((mode2&clear&ss).sum()),dual_stops=int((mode2&clear&ds).sum()),added=int(added.sum()),removed=int(removed.sum())),
        mode2_all_first_sources=dict(counts),added_summary=aggregate,added_by_branch=cells,mode2_target_final_layout=layout,
        turn_distribution=dict(units=32,configurations=1280,right_turn=1280,left_turn=0,yaw_start_deg=-20,yaw_final_deg=0,yaw_rate_deg_s=20/3),
        limits=['Nearest visible object co-occurs with alarm; no counterfactual or M3 attention proof of cause.',
            'No literal lateral wall boxes exist in this natural generator. Corner distractors are finite mixed-surface panels; floor/back are separate.',
            '129x129 first-hit rays approximate visibility; subray objects can be missing. Unknown nearest is retained.',
            'Mode2 contains only right turns, so turn direction and side are confounded. Layout is balanced only at final target side, not each evolving view.',
            'Query-clear and HEAD/BODY count separate episodes; this is consumed synthetic Development.'],seconds=time.monotonic()-tick)
    save(OUT/'geometry_result.json',output);print(json.dumps(output,ensure_ascii=False),flush=True)
    return output

def run_coverage():
    tick=time.monotonic()
    import cnh_dual_gated_geometry as F
    events=read(OUT/'events.json')['events']
    with np.load(OLD/'ledger.npz') as d:g={k:d[k].copy() for k in d.files}
    geometry_path=ROOT/'artifacts.local/work/cnh-observed-sequence-20261002/geometry.npz'
    with np.load(geometry_path) as d:
        keep=d['split']=='evaluation'
        for key in ('unit','config','query'):np.testing.assert_array_equal(d[key][keep],g[key])
        frame_contact=np.isin(d['frame_category'][keep],['contact0-2cm','contact2-5cm','contact>5cm'])
    # Config-specific estimated pose noise is part of public geometry.
    keys=list(dict.fromkeys((int(u),int(c)) for u,c in zip(g['unit'],g['config'])))
    public_geometry={}
    for k,(u,c) in enumerate(keys):
        public_geometry[(u,c)]=F.natural_coverage(u,c,angles=(-15,15))
        if (k+1)%400==0:print('coverage configs',k+1,'/',len(keys),'seconds',round(time.monotonic()-tick,1),flush=True)
    coverage=np.asarray([public_geometry[(int(u),int(c))][:,:,int(q)] for u,c,q in zip(g['unit'],g['config'],g['query'])])
    assert coverage.shape==g['sensor_scores'].shape
    sensor=g['sensor_scores'];alarms=sensor>=THRESHOLD;dual=alarms.any(1);first=dual.argmax(1)
    ss=(g['single']>=THRESHOLD).any(1);ds=dual.any(1);mode2=g['unit']%3==2
    added=mode2&g['clear']&ds&~ss;true=g['covered']&np.isin(g['ref_category'],['contact0-2cm','contact2-5cm','contact>5cm'])
    parts={}
    first_contact=frame_contact[np.arange(len(g['unit'])),first]
    for name,selection in [('new_mode2_clear_first',added),('true_contact_at_first_alarm',ds&first_contact),('mode2_true_contact_at_first_alarm',mode2&ds&first_contact),('contact_episode_first',true&ds)]:
        parts[name]={}
        ids=np.flatnonzero(selection); t=first[ids];win=sensor[ids,:,t].argmax(1)
        parts[name]['winner']=summary(coverage[ids,win,t])
        for b,label in enumerate(('L','R')):parts[name][label]=summary(coverage[ids,b,t])
    parts['all_true_contact_alarm_frames']={label:summary(coverage[:,b][frame_contact&alarms[:,b]]) for b,label in enumerate(('L','R'))}
    parts['mode2_true_contact_alarm_frames']={label:summary(coverage[:,b][mode2[:,None]&frame_contact&alarms[:,b]]) for b,label in enumerate(('L','R'))}
    save(OUT/'coverage_result.json',dict(status='COMPLETE',distribution=parts,
        geometry_note='Public causal 0.6s displacement yaw estimate, current estimated pose, installation extrinsic, world-gravity axis and public query volume only. Config-specific pose noise retained.',
        mode2_L_R_coverage_max_abs_difference=float(np.max(abs(coverage[mode2,0]-coverage[mode2,1]))),
        coverage_script_sha256=sha(F.__file__),frame_geometry_sha256=sha(geometry_path),seconds=time.monotonic()-tick))
    path=OUT/'coverage_96000.npz';assert not path.exists();np.savez_compressed(path,coverage=coverage,unit=g['unit'],config=g['config'],query=g['query'],frames=FRAMES)
    for event in events:
        i=event['ledger_index'];t=event['frame']-3;event['coverage_L_R']=coverage[i,:,t].tolist()
        import cnh_cvr_pilot as CP
        poses=F.estimated_query_poses(CP.motion_metadata(event['unit'],event['config'])[2])
        event['estimated_optical_axis_relative_travel_L_R_deg']=[float(np.rad2deg(np.arctan2(p[t,0,2],p[t,2,2]))) for p in poses]
    save(OUT/'events_with_coverage.json',dict(status='COMPLETE',events=events))
    print(json.dumps(parts),flush=True)

def run_context():
    """Additional descriptive context when the required 2.5m window is empty."""
    import cnh_cvr_pilot as CP
    from cnh_dual_sensor_envelope_natural import extrinsic
    events=read(OUT/'events.json')['events']
    manifest={(r['unit'],r['config']):r for r in read(MANIFEST) if r['split']=='evaluation'}
    allrows=[];added_by_key={(r['unit'],r['config'],r['query']):r for r in events}
    with np.load(OLD/'ledger.npz') as d:g={k:d[k].copy() for k in d.files}
    scores=g['sensor_scores'];dual=scores.max(1)>=THRESHOLD;first=dual.argmax(1)
    chosen=np.flatnonzero((g['unit']%3==2)&dual.any(1))
    for i in chosen:
        u,c,q=int(g['unit'][i]),int(g['config'][i]),int(g['query'][i]);f=int(FRAMES[first[i]])
        sensor,travel,_=CP.motion_metadata(u,c);v=scores[i,:,first[i]];b=int(v.argmax());cur=sensor[f]@extrinsic((-15,15)[b])
        boxes=manifest[(u,c)]['boxes'];t=boxes[0];center=(np.asarray(t['lo'])+np.asarray(t['hi']))/2
        local=(center-travel[f,:3,3])@travel[f,:3,:3];visible=visible_boxes(boxes,cur,travel[f],max_range=4.8)
        near=visible[0] if visible else None
        row=dict(unit=u,config=c,query=QUERY[q],branch=('L','R')[b],frame=f,clear=bool(g['clear'][i]),
            added=(u,c,QUERY[q]) in added_by_key,range_m=float(g['frame_ranges'][i,first[i]]),
            target_current_corridor_xyz_m=local.tolist(),target_current_side='R' if local[0]>0 else 'L',
            target_current_query_edge_gap_m=abs(float(local[0]))-.29,nearest_visible_within_4p8m=near)
        allrows.append(row)
    cells={}
    for added in (False,True):
        key='added_clear' if added else 'all_first_stops';sel=[r for r in allrows if r['added']] if added else allrows
        cells[key]={}
        for branch in ('L','R'):
            part=[r for r in sel if r['branch']==branch]
            cells[key][branch]=dict(n=len(part),target_current_side=dict(Counter(r['target_current_side'] for r in part)),
                target_current_x_m=summary([r['target_current_corridor_xyz_m'][0] for r in part]),
                target_current_query_edge_gap_m=summary([r['target_current_query_edge_gap_m'] for r in part]),
                nearest_4p8m_type=dict(Counter(r['nearest_visible_within_4p8m']['type'] if r['nearest_visible_within_4p8m'] else 'none' for r in part)),
                nearest_4p8m_turn_side=dict(Counter(r['nearest_visible_within_4p8m']['turn_side'] if r['nearest_visible_within_4p8m'] else 'none' for r in part)))
    save(OUT/'full_range_context.json',dict(status='COMPLETE',cells=cells,rows=allrows,
        limits=['4.8m diagnostic is an additional context window, separate from the requested <=2.5m nearest-box field.','No mirrored-turn or object-removal counterfactual performed.']))
    print(json.dumps(cells),flush=True)

def run_side_correction():
    """Preserve initial diagnostic and explicitly fix corridor-spanning surfaces."""
    events=read(OUT/'events.json')['events'];context=read(OUT/'full_range_context.json')['rows']
    def corrected(obj):
        if obj is not None and obj['lateral_extent_m'][0]<=-.29 and obj['lateral_extent_m'][1]>=.29:
            obj['side']='spanning';obj['turn_side']='spanning'
        return obj
    for row in events:
        corrected(row['nearest_visible_within_2p5m'])
        for obj in row['visible_objects_within_2p5m']:corrected(obj)
    for row in context:corrected(row['nearest_visible_within_4p8m'])
    summaries={}
    for name,rows,field in [('within_2p5m',events,'nearest_visible_within_2p5m'),('within_4p8m',[r for r in context if r['added']],'nearest_visible_within_4p8m')]:
        summaries[name]=dict(type=dict(Counter(r[field]['type'] if r[field] else 'none' for r in rows)),
            turn_side=dict(Counter(r[field]['turn_side'] if r[field] else 'none' for r in rows)),
            by_type={typ:dict(Counter(r[field]['turn_side'] for r in rows if r[field] and r[field]['type']==typ)) for typ in ('boundary_target','mixed_surface_panel','floor')})
    save(OUT/'side_correction.json',dict(status='COMPLETE',reason='A surface spanning both query edges has no unique inner/outer side; initial floor midpoint side fields are superseded.',summary=summaries,events=events,context=context))
    print(json.dumps(summaries),flush=True)

def run_query_overlap():
    import cnh_cvr_pilot as CP
    events=read(OUT/'events.json')['events'];manifest={(r['unit'],r['config']):r for r in read(MANIFEST)}
    rows=[]
    for event in events:
        nearest=event['nearest_visible_within_2p5m']
        if nearest is None:continue
        u,c,f=event['unit'],event['config'],event['frame'];box=manifest[(u,c)]['boxes'][nearest['box_index']]
        _,travel,_=CP.motion_metadata(u,c);corners=np.asarray(list(itertools.product(*zip(box['lo'],box['hi']))))
        local=(corners-travel[f,:3,3])@travel[f,:3,:3]
        yl,yh=(-.2,.42) if event['query']=='HEAD' else (.42,.9)
        overlap=min(yh,float(local[:,1].max()))-max(yl,float(local[:,1].min()))
        rows.append(dict(unit=u,config=c,query=event['query'],first_branch=event['first_branch'],type=nearest['type'],
            height_extent_m=[float(local[:,1].min()),float(local[:,1].max())],query_height_overlap_m=max(0.,overlap),
            query_height_overlaps=overlap>0,query_lateral_edge_gap_m=nearest['query_corridor_edge_gap_m']))
    summary_by_type={typ:dict(n=sum(r['type']==typ for r in rows),query_height_overlaps=sum(r['type']==typ and r['query_height_overlaps'] for r in rows),
        lateral_outside=sum(r['type']==typ and r['query_lateral_edge_gap_m']>=0 for r in rows)) for typ in ('boundary_target','mixed_surface_panel')}
    save(OUT/'query_overlap.json',dict(status='COMPLETE',rows=rows,summary=summary_by_type,n_nearest_present=len(rows),n_nearest_absent=len(events)-len(rows)))
    print(json.dumps(summary_by_type),flush=True)

def run_joint_clear():
    with np.load(OLD/'ledger.npz') as d:g={k:d[k].copy() for k in d.files}
    assert np.array_equal(g['query'].reshape(-1,2),np.tile([0,1],(len(g['query'])//2,1)))
    for key in ('unit','config'):assert np.array_equal(g[key][::2],g[key][1::2])
    ss=(g['single']>=THRESHOLD).any(1);ds=(g['sensor_scores'].max(1)>=THRESHOLD).any(1)
    bothclear=g['clear'].reshape(-1,2).all(1);mode2=g['unit']%3==2
    added=mode2&g['clear']&ds&~ss
    added_joint=added&np.repeat(bothclear,2);mixed=added&~np.repeat(bothclear,2)
    joint={}
    for group,keep in [('all',bothclear),('mode2',bothclear&(g['unit'][::2]%3==2))]:
        n=int(keep.sum());minutes=n*2.6/60;cell={}
        for name,flag in [('single',ss),('OR_original_threshold',ds)]:
            num=int((keep&flag.reshape(-1,2).any(1)).sum());cell[name]=dict(stops=num,n=n,proxy_minutes=minutes,rate_per_proxy_minute=num/minutes)
        joint[group]=cell
    save(OUT/'joint_clear_context.json',dict(status='COMPLETE',added_mode2_query_clear=dict(n=int(added.sum()),other_query_all_clear=int(added_joint.sum()),
        other_query_not_all_clear=int(mixed.sum()),unique_joint_clear_configs=int(np.any(added_joint.reshape(-1,2),axis=1).sum()),
        by_branch={branch:dict(other_query_all_clear=int(sum((row['first_branch']==branch) and added_joint[row['ledger_index']] for row in read(OUT/'events.json')['events'])),
            other_query_not_all_clear=int(sum((row['first_branch']==branch) and mixed[row['ledger_index']] for row in read(OUT/'events.json')['events']))) for branch in ('L','R')}),
        joint_clear=joint,interpretation='Per-query clear is the frozen primary gate. An alarm for a clear query while its other height query is not clear is not a demonstrated physical unnecessary stop.'))
    print(json.dumps(dict(added_joint_query_n=int(added_joint.sum()),added_mixed_query_n=int(mixed.sum()),joint=joint)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('geometry','coverage','context','side-correction','query-overlap','joint-clear'));args=parser.parse_args()
    if args.stage=='geometry':run_geometry()
    elif args.stage=='coverage':run_coverage()
    elif args.stage=='context':run_context()
    elif args.stage=='side-correction':run_side_correction()
    elif args.stage=='query-overlap':run_query_overlap()
    else:run_joint_clear()
