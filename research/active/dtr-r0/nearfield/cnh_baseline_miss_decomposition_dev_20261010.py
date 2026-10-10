"""Consumed Development explanation only: expectation rendering and frozen caches.

No photon draws, model forward, fit, threshold selection or protected split access.
Physical scene hashes retain replica correlation; fixed SNR bands are descriptive.
"""
import argparse
from collections import defaultdict
import csv
import json
import os
import subprocess
from pathlib import Path
import time
import traceback

import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_counterfactual_eval_dev as E
import cnh_head_thin_signal_oracle_20261010 as O
import cnh_frozen_e2e_metrics_20261010 as M

ROOT=C.ROOT
OUT=ROOT/'artifacts.local/work/cnh-baseline-miss-decomposition-dev-20261010'
E2E=ROOT/'artifacts.local/work/cnh-frozen-e2e-20261010'
OLD=C.OUT
JOINT=ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
COHORTS={'e2e_hold':(E2E,'hold'), 'old_ideal_validation':(OLD,'validation')}


def save(path,value):
    if path.exists(): raise FileExistsError(str(path))
    C.save(path,value)


def csv_new(path,rows):
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inputs(cohort):
    root,split=COHORTS[cohort]
    rows=C.read(root/'scene_rows.json')[split]
    with np.load(root/'data'/split/'geometry.npz') as z:
        sensor=z['sensor'];cat=z['category']
    return rows,sensor,cat


def visibility(engine,target):
    """Same sub16 ray quadrature as oracle; also separate geometric FOV loss."""
    cp=engine.cp
    d=engine.directions.reshape(engine.npose,engine.nray,3)
    origin=engine.poses[:,:3,3][:,None,:]
    lo,hi=cp.asarray(target['lo']),cp.asarray(target['hi'])
    parallel=cp.abs(d)<1e-14; safe=cp.where(parallel,1.,d)
    a,b=(lo-origin)/safe,(hi-origin)/safe
    enter=cp.where(parallel,-cp.inf,cp.minimum(a,b)).max(-1)
    leave=cp.where(parallel,cp.inf,cp.maximum(a,b)).min(-1)
    dist=cp.where(enter>1e-10,enter,leave)
    hit=(~(parallel&((origin<lo)|(origin>hi))).any(-1)&(leave>=cp.maximum(enter,0))&(dist>1e-10)&cp.isfinite(dist))
    visible=hit&(dist<=engine.distance.reshape(engine.npose,engine.nray))
    import cnh_location_reference_gpu as G
    rawbin=cp.floor((cp.where(visible,dist,0)-engine.params.range_zero_m)/G.SENSOR.RAW_BIN_M)
    valid=visible&(rawbin>=0)&(rawbin<128)
    mass=(visible*(engine.fractions*engine.weights)[None,:]).reshape(engine.npose,8,8,engine.sub**2).sum(-1)
    return cp.asnumpy(mass),cp.asnumpy(valid.sum(-1)),cp.asnumpy(hit.sum(-1))


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    if not OUT.resolve().is_relative_to((ROOT/'artifacts.local').resolve()): raise ValueError('Artifact route')
    if not (OUT/'PLAN.json').exists():
        save(OUT/'PLAN.json',dict(task='BASELINE_MISS_DECOMPOSITION',date='2026-10-10',lane='EXPLORE',
            goal='Explain frozen contact misses and negative notification geometry on consumed data',
            cpu_command_wall_cap_s=1500,gpu_expectation_wall_cap_s=600,scientific_stop='Caps or complete both declared cohorts; retain partials',
            cohorts=list(COHORTS),snr_bands=['<2','2-4','4-8','>=8'],timely=[3,13],full=[3,15],
            training=0,model_forward=0,photon_sampling=0,threshold_selection=0,protected_access=0,
            source_commits=['ff63612d','8d08aad9'],base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    prior=sum(C.read(x)['seconds'] for x in OUT.glob('render_*receipt.json'))+sum(C.read(x)['seconds'] for x in OUT.glob('render_failure_*.json'))
    began=time.monotonic();engine=None;handle=None
    env={k:os.environ.get(k) for k in ('PATH','CUPY_CACHE_DIR','TEMP','TMP')}
    def check():
        if prior+time.monotonic()-began>=590: raise TimeoutError('GPU rendering shared cap (10s startup reserve)')
    try:
        for k in ('CUPY_CACHE_DIR','TEMP','TMP'):
            p=OUT/'runtime'/k.lower();p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
        import torch
        dll=Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        if os.name=='nt' and dll.is_dir():
            handle=os.add_dll_directory(str(dll));os.environ['PATH']=str(dll)+os.pathsep+os.environ['PATH']
        _,_,G,_=D.frozen_imports()
        for cohort in COHORTS:
            if (OUT/f'{cohort}_expectations.npz').exists(): continue
            rows,sensor,_=inputs(cohort); n=len(rows)
            target=np.zeros((n,16,8,8,16));present=np.empty_like(target);absent=np.empty_like(target)
            occupancy=np.zeros((n,16,8,8));visible_counts=np.zeros((n,16),np.int32);fov_counts=np.zeros_like(visible_counts)
            groups=defaultdict(list)
            for i,r in enumerate(rows):groups[r['background_id']].append((i,r))
            metadata=[]
            for bg,group in groups.items():
                check();engine=G.ExpectedRenderer(sensor,group[0][1]['background_boxes'])
                try:
                    background=engine.background_expected(return_device=False);ambient=engine.ambient.copy()
                    metadata.append(dict(background_id=bg,**engine.metadata))
                    active=[]
                    for i,r in group:
                        absent[i]=background
                        if r['presence']: active.append((i,r))
                        else: present[i]=background
                    for start,endpoints in engine.iter_render([r['target_box'] for _,r in active],candidate_batch=8,pose_batch=16,deadline_check=check):
                        for j,e in enumerate(endpoints):
                            i,r=active[start+j]; target[i]=r['rho']/G.ENDPOINT_RHO*(e[1]-e[0]);present[i]=e[0]+target[i]
                            occupancy[i],visible_counts[i],fov_counts[i]=visibility(engine,r['target_box'])
                finally:engine.close();engine=None
                print('EXPECTATION',cohort,bg,round(time.monotonic()-began,2),flush=True)
            np.savez_compressed(OUT/f'{cohort}_expectations.npz',target=target,present=present,absent=absent,ambient=ambient,
                occupancy=occupancy,visible_counts=visible_counts,fov_counts=fov_counts,sensor=sensor,scene_ids=np.arange(n))
            save(OUT/f'render_{cohort}_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
                backend=metadata,physical_scenes=len({r['physical_key'] for r in rows}),nominal_rows=n,
                payload_sha256=C.sha(OUT/f'{cohort}_expectations.npz')))
            prior+=time.monotonic()-began;began=time.monotonic()
    except BaseException as err:
        save(OUT/f'render_failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,error=repr(err),traceback=traceback.format_exc()));raise
    finally:
        if engine is not None:engine.close()
        if handle is not None:handle.close()
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def frozen(cohort):
    if cohort=='e2e_hold':
        with np.load(E2E/'data/hold/scores.npz') as z: m3=E.smooth(z['m3_raw']);local=E.smooth(z['local_raw'])
        with np.load(E2E/'hold_grades_notifications.npz') as z:
            keys=z['keys'].tolist();grade={arm:z['grades'][keys.index('2026100955/'+arm)] for arm in ('m3','old5','both')}
    else:
        with np.load(OLD/'baselines/validation_ideal.npz') as z:m3=E.smooth(z['m3_raw']);local=E.smooth(z['local_raw'])
        grade={'m3':2*(m3>=E.M3_THETA).astype(np.int8),'old5':2*E.old_fusion(m3,local).astype(np.int8)}
        with np.load(JOINT/'validation_grades.npz') as z: grade['both']=z['grades'][z['keys'].tolist().index('2026100955/score_current/c15_p64')]
    return grade,m3,local


def band(s): return '<2' if s<2 else '2-4' if s<4 else '4-8' if s<8 else '>=8'


def gap(box): return max(box['lo'][0],-box['hi'][0])-.30


def dist_stats(vals):
    vals=np.asarray(vals,float)
    return {k:float(v) for k,v in zip(('min','p25','median','p75','max'),np.quantile(vals,[0,.25,.5,.75,1]))} if len(vals) else None


def analyze():
    began=time.monotonic();events=[];clips=[];frames=[]
    manifest={};frozen_files=[]
    for cohort in COHORTS:
        root,split=COHORTS[cohort];rows,sensor,category=inputs(cohort)
        grade,m3,local=frozen(cohort);emit={arm:M.replay_gap1(g) for arm,g in grade.items()}
        np.savez_compressed(OUT/f'{cohort}_frozen.npz',category=category,m3=m3,local=local,**{f'{a}_grade':g for a,g in grade.items()},**{f'{a}_notice':g for a,g in emit.items()})
        with np.load(OUT/f'{cohort}_expectations.npz') as z:
            t=z['target'];p=z['present'];a=z['absent'];ambient=z['ambient'];occ=z['occupancy'];vis=z['visible_counts'];fov=z['fov_counts']
        rootpaths=[root/'scene_rows.json',root/'data'/split/'geometry.npz',OUT/f'{cohort}_expectations.npz',OUT/f'{cohort}_frozen.npz']
        rootpaths += [E2E/'data/hold/scores.npz',E2E/'hold_grades_notifications.npz'] if cohort=='e2e_hold' else [OLD/'baselines/validation_ideal.npz',JOINT/'validation_grades.npz']
        manifest.update({str(p.relative_to(ROOT)):C.sha(p) for p in rootpaths})
        for i,r in enumerate(rows):
            metrics=O.frame_metrics(t[i],p[i],a[i],ambient,a[i]-(p[i]-t[i]),occ[i]);snr=metrics['target_best_bin_snr'];mx=float(snr[3:14].max())
            dims=np.asarray(r['target_box']['hi'])-r['target_box']['lo'];vv=np.where(vis[i]>0)[0];fv=np.where(fov[i]>0)[0]
            exits=np.where((fov[i][1:]==0)&(fov[i][:-1]>0))[0]+1
            exits_v=np.where((vis[i][1:]==0)&(vis[i][:-1]>0))[0]+1
            base=dict(cohort=cohort,scene_id=i,scene_uid=r['scene_uid'],physical_key=r['physical_key'],shape=r['shape_family'],
                target_height=('HEAD','BODY')[r['group']],presence=int(r['presence']),rho=r['rho'],side=r['side'],placement=r['placement'],
                width_m=round(float(dims[0]),6),height_m=round(float(dims[1]),6),depth_m=round(float(dims[2]),6),thickness_m=round(float(dims.min()),6),
                signed_target_clearance_m=gap(r['target_box']) if r['presence'] else None,
                min_scene_clearance_m=min(gap(b) for b in r['boxes']),background_family=r['background_family'],background_id=r['background_id'],
                max_best_bin_snr=mx,snr_band=band(mx),frames_snr_ge2=int((snr[3:14]>=2).sum()),frames_snr_ge4=int((snr[3:14]>=4).sum()),frames_snr_ge8=int((snr[3:14]>=8).sum()),
                first_visible_frame=int(vv[0]) if len(vv) else -1,first_visible_timely_frame=int(np.where(vis[i][3:14]>0)[0][0]+3) if np.any(vis[i][3:14]>0) else -1,
                exit_fov_frame=int(exits[0]) if len(exits) else -1,exit_visible_frame=int(exits_v[0]) if len(exits_v) else -1,
                permanent_exit_fov_frame=int(fv[-1]+1) if len(fv) and fv[-1]<15 else -1,
                fov_reentry_after_exit=int(bool(len(exits) and np.any(fov[i,int(exits[0])+1:]>0))),
                visible_timely_frames=int((vis[i][3:14]>0).sum()),max_zone_visible_fraction=float(occ[i,3:14].max()),
                max_total_zone_visible_fraction=float(occ[i,3:14].sum((-1,-2)).max()/64),max_visible_zone_share=float((occ[i,3:14]>0).mean((-1,-2)).max()))
            for f in range(16):
                frames.append(dict(cohort=cohort,scene_id=i,physical_key=r['physical_key'],frame=f,best_bin_snr=float(snr[f]),
                    target_counts=float(metrics['target_counts'][f]),best_snr_flatbin=int(metrics['best_snr_flatbin'][f]),visible_counts=int(vis[i,f]),fov_counts=int(fov[i,f]),
                    max_zone_visible_fraction=float(occ[i,f].max()),total_zone_visible_fraction=float(occ[i,f].sum()/64)))
            for k in range(m3.shape[1]):
                outcomes={}
                for q,height in enumerate(('HEAD','BODY')):
                    outcomes[q]={}
                    for arm in grade:
                        ix=np.where(emit[arm][i,k,:,q]>0)[0];first=int(ix[0]+3) if len(ix) else -1
                        outcomes[q][arm+'_first_frame']=first;outcomes[q][arm+'_outcome']='silent' if first<0 else 'timely' if first<=13 else 'late'
                    if category[i,q]=='contact':
                        events.append(dict(**base,replica=k,height=height,**outcomes[q],
                            m3_margin=float((m3[i,k,:11,q]-E.M3_THETA).max()),m3_raised_margin=float((m3[i,k,:11,q]-E.OLD_RAISED).max()),
                            local_margin=float((local[i,k,:11,q]-E.OLD_LOCAL).max())))
                kind='clear' if np.all(category[i]=='clear') else 'purepass' if np.any(category[i]=='pass') and not np.any(category[i]=='contact') else None
                if kind:
                    item=dict(**base,replica=k,kind=kind)
                    for arm in grade:
                        notice=emit[arm][i,k].max(-1)
                        item.update({arm+'_notifications':int((notice>0).sum()),arm+'_light':int((notice==1).sum()),arm+'_strong':int((notice==2).sum()),arm+'_max_grade':int(notice.max())})
                        for q,height in enumerate(('HEAD','BODY')):
                            item[arm+'_'+height+'_notifications']=int((emit[arm][i,k,:,q]>0).sum())
                            item[arm+'_'+height+'_max_grade']=int(emit[arm][i,k,:,q].max())
                    clips.append(item)
    csv_new(OUT/'event_ledger.csv',events);csv_new(OUT/'clip_ledger.csv',clips);csv_new(OUT/'frame_ledger.csv',frames)
    groups=[];dimensions=['shape','thickness_m','rho','side','signed_target_clearance_m','background_family']
    for cohort in COHORTS:
        for height in ('HEAD','BODY'):
            selected=[e for e in events if e['cohort']==cohort and e['height']==height]
            for dimension in ['all',*dimensions,'shape_rho','shape_size_rho_background']:
                subsets=defaultdict(list)
                for e in selected:
                    value='all' if dimension=='all' else '/'.join(str(e[k]) for k in ('shape','rho')) if dimension=='shape_rho' else '/'.join(str(e[k]) for k in ('shape','thickness_m','rho','background_family')) if dimension=='shape_size_rho_background' else str(e[dimension])
                    subsets[value].append(e)
                for value,sub in subsets.items():
                    for snrband in ('<2','2-4','4-8','>=8'):
                        bandrows=[e for e in sub if e['snr_band']==snrband]
                        for arm in ('m3','old5','both'):
                            for outcome in ('timely','late','silent'):
                                cell=[e for e in bandrows if e[arm+'_outcome']==outcome]
                                groups.append(dict(cohort=cohort,height=height,dimension=dimension,group=value,snr_band=snrband,arm=arm,outcome=outcome,
                                    events=len(cell),physical_scenes=len({e['physical_key'] for e in cell}),group_events=len(sub),group_physical_scenes=len({e['physical_key'] for e in sub})))
    csv_new(OUT/'cross_table.csv',groups)
    summary={};ranks=[];cost=[]
    for cohort in COHORTS:
        summary[cohort]={}
        for height in ('HEAD','BODY'):
            sub=[e for e in events if e['cohort']==cohort and e['height']==height];miss=[e for e in sub if e['old5_outcome']!='timely']
            hi=[e for e in miss if e['max_best_bin_snr']>=4];lo=[e for e in miss if e['max_best_bin_snr']<2];mid=[e for e in miss if 2<=e['max_best_bin_snr']<4]
            summary[cohort][height]=dict(events=len(sub),physical_scenes=len({e['physical_key'] for e in sub}),misses=len(miss),high=len(hi),low=len(lo),intermediate=len(mid),
                high_scene_any=len({e['physical_key'] for e in hi}),low_scene_any=len({e['physical_key'] for e in lo}),
                outcomes={arm:{o:sum(e[arm+'_outcome']==o for e in sub) for o in ('timely','late','silent')} for arm in ('m3','old5','both')})
            for shape in D.SHAPES:
                sg=[e for e in sub if e['shape']==shape];hm=[e for e in sg if e['old5_outcome']!='timely' and e['max_best_bin_snr']>=4]
                ranks.append(dict(cohort=cohort,height=height,shape=shape,denominator=len(sg),physical_scenes=len({e['physical_key'] for e in sg}),high_miss=len(hm),high_miss_scenes=len({e['physical_key'] for e in hm}),
                    high_late=sum(e['old5_outcome']=='late' for e in hm),high_silent=sum(e['old5_outcome']=='silent' for e in hm),
                    low_miss=sum(e['old5_outcome']!='timely' and e['max_best_bin_snr']<2 for e in sg),
                    m3_margin=dist_stats([e['m3_margin'] for e in hm]),local_margin=dist_stats([e['local_margin'] for e in hm]),
                    high_snr=dist_stats([e['max_best_bin_snr'] for e in hm]),ge4_frames=dist_stats([e['frames_snr_ge4'] for e in hm])))
        for kind in ('purepass','clear'):
            sub=[c for c in clips if c['cohort']==cohort and c['kind']==kind]
            for arm in ('old5','both','m3'):
                for notified in (True,False):
                    ss=[c for c in sub if bool(c[arm+'_notifications'])==notified];dist=[c['signed_target_clearance_m'] for c in ss if c['presence']]
                    cost.append(dict(cohort=cohort,kind=kind,arm=arm,notified=notified,clips=len(ss),denominator=len(sub),physical_scenes=len({c['physical_key'] for c in ss}),
                        notifications=sum(c[arm+'_notifications'] for c in ss),light=sum(c[arm+'_light'] for c in ss),strong=sum(c[arm+'_strong'] for c in ss),
                        present_clips=len(dist),absent_clips=sum(not c['presence'] for c in ss),distance=dist_stats(dist),near10cm=sum(d<.1 for d in dist),
                        snr=dist_stats([c['max_best_bin_snr'] for c in ss if c['presence']]),
                        distance_strata={str(d):dict(clips=sum(c['signed_target_clearance_m']==d for c in ss),notifications=sum(c[arm+'_notifications'] for c in ss if c['signed_target_clearance_m']==d)) for d in sorted(set(dist))}))
    save(OUT/'summary.json',dict(contact=summary,shape_groups=sorted(ranks,key=lambda r:-r['high_miss']),cost=cost))
    save(OUT/'input_manifest.json',manifest)
    save(OUT/'analysis_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,events=len(events),clips=len(clips),frames=len(frames),training=0,model_forward=0,sampling=0,threshold_changes=0))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('render','analyze'));args=p.parse_args()
    render() if args.phase=='render' else analyze()
