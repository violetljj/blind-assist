"""Frozen finite-object passage sensitivity, separate from main3m diagnostic.

For each target/panel individually, continue the same final straight path only
until the circular footprint completely passes that object's rear z face.
No floor/back objects, new scores, rendering or changed primary judgment.
"""
from __future__ import annotations
import time
import json
import numpy as np
import cnh_turn_path_diagnostic as D

def summarize(rows):
    n=len(rows);union=sum(r['union_full_pass'] for r in rows)
    return dict(n=n,union_full_pass=union,rate=None if not n else union/n,
        same_query=sum(r['same_query_full_pass'] for r in rows),other_height=sum(r['other_height_full_pass'] for r in rows),
        exclusive={label:sum(r['exclusive_full_pass']==label for r in rows) for label in ('SAME_QUERY_PATH','OTHER_HEIGHT_ONLY','PATH_CLEAR')},
        beyond_main3m_added=sum(r['union_full_pass'] and not r['union_main3m'] for r in rows),
        recorded_path=sum(r['recorded_full_pass'] for r in rows),extrapolated_only=sum(r['extrapolated_only_full_pass'] for r in rows),
        target=sum(any(o['type']=='target' and (o['same_query_invasion'] or o['other_height_invasion']) for o in r['objects']) for r in rows),
        panel=sum(any(o['type']=='mixed_surface_panel' and (o['same_query_invasion'] or o['other_height_invasion']) for o in r['objects']) for r in rows))

def run():
    tick=time.monotonic();contract=D.OUT/'SUPPLEMENT_FULL_PASS_PLAN.json';plan=D.read(contract);plan_hash=D.sha(contract)
    assert D.sha(D.OUT/'path_result.json')==plan['main_result_sha256']
    assert time.time()<D.read(D.OUT/'PLAN.json')['deadline_unix']
    import cnh_cvr_pilot as CP
    trajectory={};classification={};rows=[]
    for batch in (96000,97000):
        _,scenes,_=D.load_batch(batch)
        for arm in ('single','OR'):
            events=D.read(D.OUT/f'events{batch}_{arm}.json')['events']
            for event in events:
                u,c,f,q=event['unit'],event['config'],event['frame'],event['query'];key=(u,c,f)
                if u not in trajectory:trajectory[u]=CP.motion_metadata(u,0)[1]
                travel=trajectory[u]
                if key not in classification:
                    objects=[];future_saved=float(np.linalg.norm(np.diff(travel[f:,:3,3],axis=0)[:,[0,2]],axis=1).sum())
                    for b,box in enumerate(scenes[(u,c)]['boxes'][:-2]):
                        endpoint=max(float(travel[-1,2,3]),float(box['hi'][2])+.30)
                        horizon=future_saved+endpoint-float(travel[-1,2,3])
                        if horizon<=0:raise ValueError('No positive finite passage horizon')
                        hit=D.tube_first_hit(D.future_segments(travel,f,horizon),box)
                        objects.append(dict(box_index=b,type=D.kind(b,len(scenes[(u,c)]['boxes'])),object_hi_z=float(box['hi'][2]),
                            endpoint_world_z=endpoint,horizon_arclength_m=horizon,horizontal_path=hit))
                    classification[key]=objects
                objs=[]
                for item in classification[key]:
                    b=item['box_index'];vertical=event['objects'][b]['height_overlaps'];hit=item['horizontal_path']
                    obj=dict(**item,same_query_invasion=bool(vertical[q] and hit['invasion']),other_height_invasion=bool(vertical[1-q] and hit['invasion']))
                    if event['objects'][b]['same_query_invasion'] or event['objects'][b]['other_height_invasion']:
                        assert obj['same_query_invasion'] or obj['other_height_invasion'],'Full finite passage must preserve main collision for finite objects'
                    objs.append(obj)
                same=any(o['same_query_invasion'] for o in objs);other=any(o['other_height_invasion'] for o in objs)
                colliding=[o for o in objs if o['same_query_invasion'] or o['other_height_invasion']]
                recorded=any(o['horizontal_path']['recorded'] for o in colliding)
                rows.append(dict(batch=batch,arm=arm,ledger_index=event['ledger_index'],unit=u,config=c,query=q,frame=f,mode=u%3,
                    OR_added_mode2=event['OR_added_mode2'],union_main3m=event['union_path_invasion'],
                    same_query_full_pass=same,other_height_full_pass=other,union_full_pass=same or other,
                    exclusive_full_pass='SAME_QUERY_PATH' if same else 'OTHER_HEIGHT_ONLY' if other else 'PATH_CLEAR',
                    recorded_full_pass=recorded,extrapolated_only_full_pass=bool(colliding and not recorded),objects=objs))
    primary=[r for r in rows if r['OR_added_mode2']]
    summary={str(batch):summarize([r for r in primary if r['batch']==batch]) for batch in (96000,97000)}
    summary['pooled']=summarize(primary)
    firsts=[o['horizontal_path']['first_arclength_m'] for r in primary for o in r['objects'] if o['same_query_invasion'] or o['other_height_invasion']]
    horizons=[o['horizon_arclength_m'] for r in primary for o in r['objects']]
    assert D.sha(contract)==plan_hash and D.sha(D.OUT/'path_result.json')==plan['main_result_sha256']
    result=dict(status='COMPLETE',scope='Fixed supplemental full finite object passage; no replacement primary judgment',contract_sha256=plan_hash,
        main_result_sha256=plan['main_result_sha256'],primary=summary,event_rows=len(rows),events=rows,
        primary_object_horizon_m=dict(minimum=min(horizons),median=float(np.median(horizons)),maximum=max(horizons)),
        primary_colliding_first_arrival_m=dict(n=len(firsts),minimum=min(firsts) if firsts else None,median=float(np.median(firsts)) if firsts else None,maximum=max(firsts) if firsts else None),
        seconds=time.monotonic()-tick,script_sha256=D.sha(__file__),geometry_script_sha256=D.sha(D.__file__),
        limits=['The finite horizon ends after each target/panel rear face+.30m, rather than arbitrary infinite future. Floor/back excluded only in this supplemental.',
            'Future after frame15 remains assumed finalyaw0 straight. This is not observed human walking or a deployed clearance estimate.',
            'Same-query all-frames clear generally leaves the target outside final lateral corridor; under final straight continuation, extending range alone cannot bring that target into the same height path.',
            'Main finite3m criterion and TURN_COST_REAL are retained with their stated horizon; this wider supplemental can change interpretation without changing the frozen judgment.'])
    D.save(D.OUT/'full_object_pass_sensitivity.json',result)
    print(json.dumps(dict(primary=summary,horizon=result['primary_object_horizon_m'],arrival=result['primary_colliding_first_arrival_m'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':run()
