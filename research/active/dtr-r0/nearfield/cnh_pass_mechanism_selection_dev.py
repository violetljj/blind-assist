"""Freeze purposive saved-crossing cases and all-event side/rho strata.

CPU metadata/CSV analysis only: no inference, new threshold, photons or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009'
CROSS=ROOT/'artifacts.local/work/cnh-pass-pose-diagnostic-dev-20261009/crossings/crossings_events.csv'
DATA=ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
HEIGHTS=('HEAD','BODY')
SIDES=('negative_x','positive_x')
STATES=('rescue','loss','both_miss')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<22),b''):digest.update(block)
    return digest.hexdigest()


def csvwrite(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def typed(row):
    strings={'arm','point','height','side','background_family','shape_family','placement','crossing'}
    ints={'seed','scene_id','replica','side_sign','background_id','target_id','ideal_peak_frame','yaw_peak_frame',
          'ideal_detected','yaw_detected'}
    return {k:v if k in strings else int(v) if k in ints else float(v) for k,v in row.items()}


def run(output=OUT,crossings=CROSS,data_root=DATA):
    start=time.monotonic();output,crossings,data_root=map(lambda p:Path(p).resolve(),(output,crossings,data_root))
    plan=read(output/'PLAN.json');cap=float(plan['allocations']['selection_strata_cpu'])
    if cap>50:raise ValueError('Selection/strata allocation exceeds authorized 50 seconds')
    names=('selection.json','selected_events.csv','strata.csv','strata.json','strata_events.csv',
           'side_comparability.csv','selection_strata_receipt.json')
    if any((output/n).exists() for n in names):raise FileExistsError('Preserve previous frozen selection/strata')
    receipt=dict(status='RUNNING',stage='selection_strata',cpu_only=True,training=0,inference=0,
                 photons=0,new_thresholds=0,source_sha256=sha(__file__),plan_sha256=sha(output/'PLAN.json'))
    def check():
        if time.monotonic()-start>=cap:raise TimeoutError('Selection/strata CPU allocation reached')
    try:
        scene_path=data_root/'scene_rows.json';scene_rows=read(scene_path)['validation']
        metadata={int(r['scene_id']):r for r in scene_rows}
        seed,point=int(plan['seed']),plan['point']
        events={}
        with crossings.open(encoding='utf8',newline='') as handle:
            for row in csv.DictReader(handle):
                if int(row['seed'])!=seed or row['point']!=point or row['arm'] not in plan['arms']:continue
                row=typed(row);meta=metadata[row['scene_id']]
                if row['placement']!='in1cm' or meta['placement']!='in1cm':raise ValueError('Only original all-1cm events expected')
                if row['height']!=HEIGHTS[meta['group']] or row['side_sign']!=meta['side']:
                    raise ValueError('Crossing metadata identity mismatch')
                for field in ('shape_family','background_id','background_family','target_id'):
                    if row[field]!=meta[field]:raise ValueError('Crossing metadata mismatch '+field)
                row['rho']=float(meta['rho'])
                key=(row['arm'],row['scene_id'],row['replica'],row['height'])
                if key in events:raise ValueError('Duplicate source crossing identity')
                events[key]=row
        check()
        per_arm={arm:[r for r in events.values() if r['arm']==arm] for arm in plan['arms']}
        for arm,rows in per_arm.items():
            if len(rows)!=256 or Counter(r['height'] for r in rows)!=dict(HEAD=128,BODY=128):
                raise ValueError('All original 256 one-cm events/model must remain')
        identities={arm:{(r['scene_id'],r['replica'],r['height']) for r in rr} for arm,rr in per_arm.items()}
        if identities['control']!=identities['weak_pass']:raise ValueError('Models need same selected-event identities')
        groups,selected=[],[]
        for crossing,height,side in itertools.product(STATES,HEIGHTS,SIDES):
            candidates=sorted((r for r in per_arm['weak_pass'] if (r['crossing'],r['height'],r['side'])==
                               (crossing,height,side)),key=lambda r:(r['scene_id'],r['replica']))
            picked=[];seen=set()
            for event in candidates:
                if event['scene_id'] in seen:continue
                seen.add(event['scene_id']);picked.append(event)
                if len(picked)==2:break
            group_id=crossing+'/'+height+'/'+side
            group=dict(group_id=group_id,weak_crossing=crossing,height=height,side=side,
                       available_events=len(candidates),available_distinct_scenes=len({r['scene_id'] for r in candidates}),
                       requested_distinct_scenes=2,selected_count=len(picked),shortage=2-len(picked),
                       status='COMPLETE' if len(picked)==2 else 'EMPTY' if not picked else 'SHORTAGE',selection_ids=[])
            for row in picked:
                meta=metadata[row['scene_id']];q=HEIGHTS.index(height)
                event=dict(selection_id=len(selected),scene_id=row['scene_id'],replica=row['replica'],height=height,
                           query_index=q,side=side,side_sign=row['side_sign'],rho=row['rho'],
                           shape_family=row['shape_family'],background_id=row['background_id'],
                           background_family=row['background_family'],target_id=row['target_id'],
                           target_box=meta['target_box'],weak_crossing=crossing,group_id=group_id,
                           models={arm:events[arm,row['scene_id'],row['replica'],height] for arm in plan['arms']})
                group['selection_ids'].append(event['selection_id']);selected.append(event)
            groups.append(group)
        if len(selected)>24:raise AssertionError('Frozen selection max24 violated')
        selection=dict(status='FROZEN',seed=seed,point=point,policy=plan['policy'],branches=plan['branches'],
                       arms=plan['arms'],timely_frames=plan['timely_frames'],events=selected,groups=groups,
                       event_count=len(selected),source_crossings=str(crossings),source_crossings_sha256=sha(crossings),
                       source_metadata_sha256=sha(scene_path),plan_sha256=receipt['plan_sha256'],
                       rule='weak crossing x height x side; first2 distinct scenes, sort(scene,replica); preserve empty/short groups',
                       scope='Seed956 purposive prior maximum HEAD-loss case; selected cases are mechanism exploration, not representative confirmation')
        save(output/'selection.json',selection)
        flat=[]
        for e in selected:
            r={k:v for k,v in e.items() if k not in ('models','target_box')}
            for arm,row in e['models'].items():
                for field in ('ideal_detected','yaw_detected','crossing','threshold','ideal_smoothed_peak',
                              'yaw_smoothed_peak','ideal_margin','yaw_margin','ideal_peak_frame','yaw_peak_frame'):
                    r[arm+'_'+field]=row[field]
            flat.append(r)
        csvwrite(output/'selected_events.csv',flat)
        csvwrite(output/'strata_events.csv',sorted(events.values(),key=lambda r:(r['arm'],r['scene_id'],r['replica'],r['height'])))
        shapes=sorted({r['shape_family'] for r in per_arm['weak_pass']})
        bgs=sorted({r['background_id'] for r in per_arm['weak_pass']})
        rhos=sorted({r['rho'] for r in per_arm['weak_pass']})
        bgfamily={r['background_id']:r['background_family'] for r in per_arm['weak_pass']}
        strata=[]
        for arm,height,side,rho,shape,bg in itertools.product(plan['arms'],HEIGHTS,SIDES,rhos,shapes,bgs):
            rr=[r for r in per_arm[arm] if (r['height'],r['side'],r['rho'],r['shape_family'],r['background_id'])==
                (height,side,rho,shape,bg)]
            n=len(rr);margins=[r['ideal_margin'] for r in rr]
            strata.append(dict(arm=arm,seed=seed,point=point,policy=plan['policy'],branch='ideal',height=height,
                side=side,rho=rho,shape_family=shape,background_id=bg,background_family=bgfamily[bg],
                status='OBSERVED' if n else 'EMPTY_CELL',event_denominator=n,distinct_scenes=len({r['scene_id'] for r in rr}),
                ideal_detected=sum(r['ideal_detected'] for r in rr),
                ideal_detection_rate=sum(r['ideal_detected'] for r in rr)/n if n else None,
                ideal_margin_mean=float(np.mean(margins)) if n else None,
                ideal_margin_min=min(margins) if n else None,ideal_margin_max=max(margins) if n else None,
                threshold=per_arm[arm][0]['threshold'],yaw_detected=sum(r['yaw_detected'] for r in rr),
                rescue=sum(r['crossing']=='rescue' for r in rr),loss=sum(r['crossing']=='loss' for r in rr),
                both_miss=sum(r['crossing']=='both_miss' for r in rr)))
        csvwrite(output/'strata.csv',strata)
        comparisons=[]
        for height,rho,shape,bg in itertools.product(HEIGHTS,rhos,shapes,bgs):
            counts={side:sum((r['height'],r['rho'],r['shape_family'],r['background_id'],r['side'])==
                            (height,rho,shape,bg,side) for r in per_arm['weak_pass']) for side in SIDES}
            comparisons.append(dict(height=height,rho=rho,shape_family=shape,background_id=bg,background_family=bgfamily[bg],
                                    negative_x_events=counts['negative_x'],positive_x_events=counts['positive_x'],
                                    same_height_rho_shape_background_both_sides=all(counts.values()),
                                    status='COMPARABLE_SIDE_EXPOSURE' if all(counts.values()) else 'MISSING_OPPOSITE_SIDE_CELL'))
        csvwrite(output/'side_comparability.csv',comparisons)
        check()
        physics_path=data_root/'data/validation/physics.npz'
        with np.load(physics_path,allow_pickle=False) as physics:
            sensor=physics['sensor'];query=physics['public_query']
        pose=[]
        for f in plan['timely_frames']:
            wq=query[f]@np.linalg.inv(sensor[f])
            pose.append(dict(frame=f,sensor_to_query_x_row=query[f,0].tolist(),sensor_to_query_z_row=query[f,2].tolist(),
                             world_to_query_x_row=wq[0].tolist(),shared_sensor_to_query_for_HEAD_BODY=True))
        f=10;angle=np.deg2rad(3);c,s=np.cos(angle),np.sin(angle);wq=query[f]@np.linalg.inv(sensor[f])
        lateral=[]
        for height,side in itertools.product(HEIGHTS,SIDES):
            coords=[]
            for meta in metadata.values():
                if meta['placement']!='in1cm' or HEIGHTS[meta['group']]!=height or meta['side']!=(-1 if side=='negative_x' else 1):continue
                center=(np.array(meta['target_box']['lo'])+np.array(meta['target_box']['hi']))/2
                coord=wq@np.r_[center,1.];coords.append((coord[0],coord[2],c*coord[0]+s*coord[2]-coord[0]))
            coords=np.asarray(coords)
            lateral.append(dict(frame=f,height=height,side=side,fixture_count=len(coords),
                                ideal_query_x_range=[float(coords[:,0].min()),float(coords[:,0].max())],
                                ideal_query_z_range=[float(coords[:,1].min()),float(coords[:,1].max())],
                                yaw3_query_x_delta_range=[float(coords[:,2].min()),float(coords[:,2].max())]))
        side_rho={side:sorted({r['rho'] for r in per_arm['weak_pass'] if r['side']==side}) for side in SIDES}
        overlap=set(side_rho['negative_x'])&set(side_rho['positive_x'])
        conditional_rho=[]
        for height,shape,bg in itertools.product(HEIGHTS,shapes,bgs):
            values={side:sorted({r['rho'] for r in per_arm['weak_pass'] if
                                (r['height'],r['shape_family'],r['background_id'],r['side'])==
                                (height,shape,bg,side)}) for side in SIDES}
            shared=sorted(set(values['negative_x'])&set(values['positive_x']))
            conditional_rho.append(dict(height=height,shape_family=shape,background_id=bg,
                side_rho=values,shared_rho_values=shared,side_rho_confounded=not bool(shared)))
        result=dict(status='COMPLETE',seed=seed,point=point,policy=plan['policy'],events_per_model=256,
                    HEAD_events_per_model=128,BODY_events_per_model=128,all_missed_events_retained=True,
                    rows=strata,side_comparability=comparisons,side_rho=side_rho,
                    shared_rho_values_across_sides=sorted(overlap),
                    comparable_side_cells=sum(r['same_height_rho_shape_background_both_sides'] for r in comparisons),
                    side_rho_confounded_at_fixed_height_shape_background=all(r['side_rho_confounded'] for r in conditional_rho),
                    conditional_side_rho=conditional_rho,
                    marginal_rho_overlap_is_not_side_comparability=True,
                    query_lateral_metadata=dict(poses=pose,target_center_f10=lateral,
                       note='public_query is the shared sensor-to-query transform; query heights use distinct query geometry. Ry(+3) left multiply increases x for forward centers. Coordinate fact, not evidence-score causality.'),
                    limits='Descriptive fixedseed956 point30 ideal strata with small correlated cells; side/rho confounding does not prove rho caused a missing side effect. No shared-rho counterpart is generated or fabricated.')
        save(output/'strata.json',result)
        snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
        shutil.copyfile(__file__,snapshot/Path(__file__).name)
        inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (crossings,scene_path,physics_path,output/'PLAN.json')}
        save(output/'selection_input_manifest.json',inputs)
        receipt.update(status='COMPLETE',selected_events=len(selected),selection_groups=len(groups),
                       empty_groups=[g['group_id'] for g in groups if not g['selected_count']],
                       short_groups=[g['group_id'] for g in groups if g['shortage']],
                       strata_rows=len(strata),observed_strata=sum(r['event_denominator']>0 for r in strata),
                       side_comparability_cells=len(comparisons),comparable_side_cells=result['comparable_side_cells'],
                       side_rho=side_rho,selection_sha256=sha(output/'selection.json'),strata_sha256=sha(output/'strata.json'))
        check()
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:
        receipt.update(seconds=time.monotonic()-start,allocated_cpu_seconds=cap,total_task_cpu_cap_seconds=plan['cpu_analysis_cap_seconds'])
        save(output/'selection_strata_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--crossings',type=Path,default=CROSS)
    parser.add_argument('--data-root',type=Path,default=DATA)
    args=parser.parse_args();run(args.output,args.crossings,args.data_root)
