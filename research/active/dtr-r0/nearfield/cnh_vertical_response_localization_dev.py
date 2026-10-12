"""Localize frozen score-sequence contrasts using existing controlled responses."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-vertical-response-localization-dev-20261009'
OLD=WORK/'cnh-vertical-controlled-pair-dev-20261009'
SHAPE=WORK/'cnh-aligned-shapes-dev-20261008'

def save(name,x):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2)

def csvsave(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def props(pair,scene):
    c,s=scene[pair['clear_scene']],scene[pair['contact_scene']];a=pair['angle']
    return dict(width_cm=round((s['hi'][0]-s['lo'][0])*100),rho=s['rho'],span=s['group'],variant=s['variant'],
        contact_side=s['side'],clear_side=c['side'],contact_direction='away' if a*s['side']>0 else 'towards',clear_direction='away' if a*c['side']>0 else 'towards')

def common_key(r):
    return tuple(r[k] for k in ('width_cm','rho','span','variant','height','relation','depth','contact_direction','clear_direction'))

def prepare():
    t=time.monotonic();p=json.loads((OLD/'PLAN.json').read_text());scene=json.loads((SHAPE/'PLAN.json').read_text())['scene_rows']
    keys={a:{common_key(dict(**r,**props(r,scene))) for r in p['pairs'] if r['angle']==a} for a in (-3,3)}
    inputs=[OLD/'PLAN.json',OLD/'responses.npz',OLD/'paired_windows.csv',OLD/'last5_decomposition.csv',SHAPE/'PLAN.json',SHAPE/'geometry.npz']
    save('PLAN.json',dict(task='CNH_VERTICAL_RESPONSE_LOCALIZATION_DEV_20261009',lane='EXPLORE cached selected Development',base_commit='5b750872',
        authorization='User 继续 preceding recommendation to localize prior-four raw-score contributions by width/rho/height/distance, then decide whether equal-cost readout comparison is worthwhile',
        budget_seconds=60,budget_unit='Cumulative CPU cache read/connection/repair/focused checks wall seconds; no prior budget carry-over; Git network delivery and discussion excluded',
        preliminary_command_seconds=1.5782146,
        scope='All 64 existing angle/relation/depth/height conditions x4 near frames xK4=1024 height windows, five lag appearances each=5120; no new photons/projection/inference/training, no gate/threshold selection',
        views='Condition summaries; source-frame continuous scores/support; evaluation-frame weighted components; same-contact mirror contrast; strict common-condition ±3 comparison with unmatched rows retained',
        contract='Metadata width/rho/height/side/distance are evaluator grouping only. Source-frame distance from existing scene/sensor, not output-frame distance assigned to lags. raw still contains past8. No physical history attribution.',
        angle_comparison=dict(common_conditions=len(keys[-3]&keys[3]),minus_unmatched=len(keys[-3]-keys[3]),plus_unmatched=len(keys[3]-keys[-3])),
        decision_check='Original theta only describes existing M3 alarms. No candidate, dprime, inferential test or numeric success/stop cutoff. Score reversal alone not benefit; concentration/dispersal alone not launch/stop. Use named support and response pattern to propose the next falsifiable question.',
        identities='same-contact mirror delta minus same-side delta equals clear_same minus clear_mirror; distinct angles compared only at common width/rho/span/variant/height/relation/depth/relative-side and same output-frame/K',
        adjustable_scope='Implementation repair and focused arithmetic/source checks only; no more conditions/draws/angles/model/smoothing recipes',
        stop='Complete fixed cache summaries and focused audit or60s including failures. All new readout comparisons remain NOT_RUN.',
        inputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},source_sha256=sha(Path(__file__)),prepare_seconds=time.monotonic()-t))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('PREPARED common',len(keys[-3]&keys[3]),flush=True)

def summarize(rows,fields,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[f] for f in fields)].append(r)
    result=[]
    for key,rr in sorted(groups.items()):
        x=dict(zip(fields,key));x['windows']=len(rr)
        for m in metrics:
            values=np.array([r[m] for r in rr if r[m] is not None],float)
            x[m+'_finite']=len(values);x[m+'_mean']=float(values.mean()) if len(values) else None
            x[m+'_median']=float(np.median(values)) if len(values) else None
            x[m+'_negative']=int((values<0).sum());x[m+'_positive']=int((values>0).sum())
        result.append(x)
    return result

def run():
    t=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text());spent=p['preliminary_command_seconds']+p['prepare_seconds']
    spent+=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('failure_*.json'))
    try:
        for path,h in p['inputs_sha256'].items():assert sha(ROOT/path)==h,path
        assert sha(Path(__file__))==p['source_sha256']
        old=json.loads((OLD/'PLAN.json').read_text());scene=json.loads((SHAPE/'PLAN.json').read_text())['scene_rows']
        with np.load(OLD/'responses.npz') as z:ids=z['scene_ids'];raw=z['raw'];v=z['values']
        with np.load(SHAPE/'geometry.npz') as z:sensor=z['sensor']
        index={int(s):i for i,s in enumerate(ids)};near=[];lags=[];sources={}
        for pair in old['pairs']:
            meta=props(pair,scene);ai=old['angles'].index(pair['angle']);ci=index[pair['clear_scene']];pi=index[pair['contact_scene']];qi=int(pair['height']=='BODY')
            for f in old['evaluation_frames']:
                j=f-3
                for k in range(4):
                    base=dict(**pair,**meta,frame=f,replica=k,output_distance_m=float(scene[pair['contact_scene']]['lo'][2]-sensor[f,2,3]))
                    contrasts=[];prior=[]
                    for lag in range(5):
                        sf=f-lag;sj=sf-3;weight=2.**(4-lag)/31
                        c=float(raw[ai,ci,k,sj,qi]);s=float(raw[ai,pi,k,sj,qi]);d=s-c
                        mass=float(v[ai,pi,k,sj,qi,3]-v[ai,ci,k,sj,qi,3])
                        total=float(v[ai,pi,k,sj,qi,2]-v[ai,ci,k,sj,qi,2])
                        r=dict(**base,lag=lag,source_frame=sf,source_distance_m=float(scene[pair['contact_scene']]['lo'][2]-sensor[sf,2,3]),weight=weight,clear_raw=c,contact_raw=s,raw_delta=d,weighted_delta=d*weight,current_scaled_mass_delta=mass,total_density_delta=total)
                        lags.append(r);contrasts.append(d*weight)
                        if lag:prior.append(d*weight)
                        source_key=(pair['angle'],pair['relation'],pair['depth'],pair['clear_scene'],pair['contact_scene'],pair['height'],k,sf)
                        sources[source_key]={key:value for key,value in r.items() if key not in ('lag','weight','weighted_delta','frame','output_distance_m')}
                    near.append(dict(**base,raw_delta=contrasts[0]*31/16,current_weighted_delta=contrasts[0],preceding4_weighted_delta=sum(prior),last5_delta=sum(contrasts),raw_positive_last5_negative=int(contrasts[0]>0 and sum(contrasts)<0)))
        condition_fields=('angle','relation','depth','width_cm','rho','span','variant','height','contact_side','clear_side','contact_direction','clear_direction')
        condition=summarize(near,condition_fields,('raw_delta','current_weighted_delta','preceding4_weighted_delta','last5_delta','raw_positive_last5_negative'))
        frame=summarize(near,('angle','relation','depth','frame','output_distance_m'),('current_weighted_delta','preceding4_weighted_delta','last5_delta','raw_positive_last5_negative'))
        source=summarize(list(sources.values()),('angle','relation','depth','source_frame','source_distance_m'),('clear_raw','contact_raw','raw_delta','current_scaled_mass_delta','total_density_delta'))
        same={};mirror={}
        for r in near:
            key=(r['angle'],r['depth'],r['contact_scene'],r['height'],r['frame'],r['replica'])
            (same if r['relation']=='same_side' else mirror)[key]=r
        assert same.keys()==mirror.keys()
        contrasts=[]
        for key,s in sorted(same.items()):
            m=mirror[key];row={f:s[f] for f in condition_fields if f not in ('relation','clear_side','clear_direction')}
            row.update(frame=s['frame'],replica=s['replica'],contact_scene=s['contact_scene'],same_clear_scene=s['clear_scene'],mirror_clear_scene=m['clear_scene'])
            for metric in ('raw_delta','preceding4_weighted_delta','last5_delta'):row['mirror_minus_same_'+metric]=m[metric]-s[metric]
            # Contact is identical; difference is entirely a contrast of clear arms.
            ai=old['angles'].index(s['angle']);qi=int(s['height']=='BODY');k=s['replica'];j=s['frame']-3
            cd=float(raw[ai,index[s['clear_scene']],k,j,qi])-float(raw[ai,index[m['clear_scene']],k,j,qi])
            assert abs(cd-row['mirror_minus_same_raw_delta'])<1e-12
            row['clear_same_minus_mirror_raw']=cd;contrasts.append(row)
        angle_index=defaultdict(dict)
        for r in near:angle_index[r['angle']][(*common_key(r),r['frame'],r['replica'])]=r
        angle_pairs=[];unmatched=[]
        for key in sorted(angle_index[-3].keys()|angle_index[3].keys()):
            minus=angle_index[-3].get(key);plus=angle_index[3].get(key)
            if minus is None or plus is None:
                unmatched.append(minus if minus is not None else plus);continue
            row={f:minus[f] for f in ('width_cm','rho','span','variant','height','relation','depth','contact_direction','clear_direction','frame','replica')}
            for metric in ('raw_delta','preceding4_weighted_delta','last5_delta'):
                row['minus_'+metric]=minus[metric];row['plus_'+metric]=plus[metric];row['plus_minus_difference_'+metric]=plus[metric]-minus[metric]
            angle_pairs.append(row)
        mirror_summary=summarize(contrasts,('angle','depth'),('mirror_minus_same_raw_delta','mirror_minus_same_preceding4_weighted_delta','mirror_minus_same_last5_delta'))
        angle_summary=summarize(angle_pairs,('relation','depth'),('minus_raw_delta','plus_raw_delta','minus_preceding4_weighted_delta','plus_preceding4_weighted_delta','minus_last5_delta','plus_last5_delta'))
        if spent+time.monotonic()-t>50:raise TimeoutError('Reserve10s final focused audit within60s')
        for name,rr in [('near_windows.csv',near),('lag_contributions.csv',lags),('condition_summary.csv',condition),('evaluation_frame_summary.csv',frame),('source_frame_summary.csv',source),('same_contact_mirror_contrast.csv',contrasts),('matched_angle_windows.csv',angle_pairs),('unmatched_angle_windows.csv',unmatched)]:csvsave(name,rr)
        save('result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,near_windows=len(near),lag_appearances=len(lags),unique_source_windows=len(sources),condition_rows=len(condition),same_contact_contrasts=len(contrasts),matched_angle_windows=len(angle_pairs),unmatched_angle_windows=len(unmatched),mirror_summary=mirror_summary,angle_summary=angle_summary,new_inference=0,new_projection=0,new_samples=0,training=0,limits='Algebraic selected-condition localization; repeated coupled K/window/height correlations, not causal echo attribution or equal-cost readout benefit'))
        print('COMPLETE near',len(near),'lags',len(lags),'angle matched/unmatched',len(angle_pairs),len(unmatched),flush=True)
    except BaseException as e:save('failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise

if __name__=='__main__':globals()[sys.argv[1]]()
