"""Posthoc Development pilot: remove gait-scale position oscillation from the gate.

No new data or M3 variant. Four fixed configs are selected only on nominal P01..P05.
P06..P10 were inspected in the previous pilot; this is not fresh confirmation.
"""
import argparse
import json
import time

import numpy as np

import cnh_torso_bias_dev as T

OUT = T.OUT/'gait_mechanism'
CONFIGS = [dict(tau_seconds=t, straight_rate_deg_s=r, torso_rate_deg_s=12.,
    min_speed=.3,max_bias_rate_deg_s=3.,boxcar_frames=24,midpoint_frames=42,
    earliest_update_frame=90) for t in (2.,4.) for r in (3.,6.)]
ARMS = ('e1','torso','corrected_gait','oracle')


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def smooth_past(p,width=24):
    """Current-inclusive boxcar, partial observed prefixes; no zero padding."""
    p=np.asarray(p,float)[:,:2];c=np.vstack([np.zeros((1,2)),np.cumsum(p,axis=0)])
    end=np.arange(1,len(p)+1);start=np.maximum(end-width,0)
    return (c[end]-c[start])/(end-start)[:,None]


def causal_correct(pelvis,torso,config):
    p=smooth_past(pelvis,config['boxcar_frames']);torso=np.asarray(torso,float)
    n=len(p);ids=np.arange(config['earliest_update_frame'],n)
    chord=p[ids]-p[ids-60];old=p[ids-30]-p[ids-90]
    rate=np.abs(T.B.wrap(T.B.yaw(chord)-T.B.yaw(old)))/.5
    speed=np.linalg.norm(chord,axis=1)
    body_rate=np.abs(T.B.wrap(torso[ids]-torso[ids-60]))
    speed_ok=speed>=config['min_speed'];direction_ok=rate<=config['straight_rate_deg_s']
    body_ok=body_rate<=config['torso_rate_deg_s']
    eligible=speed_ok&direction_ok&body_ok
    residual=T.B.wrap(torso[ids-config['midpoint_frames']]-T.B.yaw(chord))
    alpha=1-np.exp(-1/(60*config['tau_seconds']));cap=config['max_bias_rate_deg_s']/60
    bias=np.zeros(n);up=np.zeros(n,bool);state=0.
    for j,i in enumerate(ids):
        if eligible[j]:
            state=float(T.B.wrap(state+np.clip(alpha*T.B.wrap(residual[j]-state),-cap,cap)))
            up[i]=True
        bias[i]=state
    gate=dict(ids=ids,speed_ok=speed_ok,direction_ok=direction_ok,body_ok=body_ok,rate=rate,residual=residual)
    return T.B.wrap(torso-bias),bias,up,gate


def errors(x,config):
    lab=T.labels(x);torso=T.torso_yaw(x);corrected,bias,up,gate=causal_correct(x[:,0],torso,config)
    ids=np.arange(len(x));e1=T.B.yaw(x[:,6,:2]-x[np.maximum(ids-60,0),6,:2])
    error=T.B.wrap(torso-lab['truth']);m=lab['valid']
    oracle_bias=float(np.degrees(np.arctan2(np.sin(np.radians(error[m])).mean(),
        np.cos(np.radians(error[m])).mean()))) if m.any() else 0.
    return dict(e1=T.B.wrap(e1-lab['truth']),torso=error,corrected_gait=T.B.wrap(corrected-lab['truth']),
        oracle=T.B.wrap(torso-oracle_bias-lab['truth']),bias_deg=bias,updated=up,**lab),gate


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((T.OUT/'PLAN.json').read_text(encoding='utf8'))
    prior=json.loads((T.OUT/'mechanism_onset_repair.json').read_text(encoding='utf8'))
    save('PLAN.json',dict(lane='EXPLORE posthoc consumed Development',
        goal='Changed signal, unchanged3/6deg/s limits: causal gait averaging gate and aligned bias residual',
        budget=dict(original_estimator_cpu_wall_seconds=600,prior_recorded_seconds=prior['combined_estimator_diagnostic_seconds'],
            prior_reserved_upper_seconds=20,pilot_max_seconds=580,no_new_gpu_budget=True),
        configs=CONFIGS,development=list(T.DEV),evaluation=list(T.EVAL),
        selection='Lowest equal-nominal-participant corrected RMS on P01..P05, all-valid original labels; tie list order. No evaluation/alarm selection.',
        algorithm='24frame current-inclusive causal mean pelvis with partial-prefix seed. Ended1s chord directions t,t-30 differ/.5. Speed>=.3 and past1s torso<=12deg gate. Earliestupdate t90. Bias residual torso[t42back] minus filtered chord yaw; actual boxcar delay11.5frames, integer midpoint choice42=30+12. Lowpass tau2/4s,max3deg/s,zero reset perclip.',
        frozen_support='Every original file, labels/valid masks/turn groups and original windows retained; no clipping, no session concatenation. No-update preserved.',
        limits='Source Pxx mapping/upstream processing mode unknown; shared Xsens model positions, relative yaw only. Evaluation IDs already inspected, no unseen claim. True future path labels only evaluation. No M3 promotion.',
        source_sha256=T.sha(__file__),parent_plan_sha256=T.sha(T.OUT/'PLAN.json'),
        windows_sha256=T.sha(T.OUT/'windows.json'),files=original['files']))
    with (OUT/'windows.json').open('xb') as stream:
        stream.write((T.OUT/'windows.json').read_bytes())
    assert T.sha(OUT/'windows.json')==T.sha(T.OUT/'windows.json')
    print('FROZEN',len(original['files']),'files',flush=True)


def self_check():
    cfg=CONFIGS[1];n=600;t=np.arange(n)/60
    p=np.c_[.8*t,np.zeros(n)];torso=np.full(n,10.)
    c,b,u,g=causal_correct(p,torso,cfg)
    assert not u[:90].any() and u[90:].all()
    assert b[-1]>5 and abs(c[-1])<abs(torso[-1])
    pivot=220;p2=p.copy();p2[pivot:,0]=p[pivot,0];p2[pivot:,1]=.8*(t[pivot:]-t[pivot])
    torso2=torso.copy();torso2[pivot:]=100.
    c2,b2,u2,g2=causal_correct(p2,torso2,cfg)
    assert not u2[pivot:pivot+60].any(), 'Known body turn freezes update'
    changed=p.copy();changed[300:]+=np.c_[np.arange(n-300),-np.arange(n-300)]
    changed_torso=torso.copy();changed_torso[300:]=-120.
    a,bb,uu,gg=causal_correct(changed,changed_torso,cfg)
    assert np.array_equal(c[:300],a[:300]) and np.array_equal(b[:300],bb[:300]) and np.array_equal(u[:300],uu[:300])
    print('SELF_CHECK prefix future invariance, known straight bias, explicit turn freeze PASS')


def summary(ds):
    arms={k:np.concatenate([d[k] for d in ds]) for k in ARMS}
    v=np.concatenate([d['valid'] for d in ds]);g=np.concatenate([d['turn_group'] for d in ds])
    up=np.concatenate([d['updated'] for d in ds]);bias=np.concatenate([d['bias_deg'] for d in ds])
    seen=np.concatenate([np.cumsum(d['updated'])>0 for d in ds]);age=np.concatenate([np.arange(len(d['valid'])) for d in ds])
    masks=dict(all=v,straight=v&(g==0),slowturn=v&(g==1),onset=v&(g==2),otherturn=v&(g==3),
        cold1to2=v&(age<120),age2to4=v&(age>=120)&(age<240),age4plus=v&(age>=240),never_updated_yet=v&~seen)
    groups={name:{k:T.measures(e[mask]) for k,e in arms.items()} for name,mask in masks.items()}
    return dict(clips=len(ds),valid_frames=int(v.sum()),invalid_frames=int((~v).sum()),
        updated_valid_frames=int((up&v).sum()),updated_valid_fraction=float(up[v].mean()) if v.any() else None,
        never_updated_yet_valid_frames=int((v&~seen).sum()),no_update_clips=sum(not d['updated'].any() for d in ds),
        groups=groups,abs_bias_valid=T.measures(bias[v]),
        persistent={k:dict(runs_ge1s=sum(T.persistent(d[k],d['valid'])['runs_ge1s'] for d in ds),
            frames_in_runs_ge1s=sum(T.persistent(d[k],d['valid'])['frames_in_runs_ge1s'] for d in ds)) for k in ARMS})


def run():
    tick=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    if (OUT/'result_angles.json').exists():raise FileExistsError('Preserve pilot result')
    assert T.sha(__file__)==plan['source_sha256']
    data=[]
    for r in plan['files']:
        path=T.B.SRC/r['name'];assert T.sha(path)==r['sha256'];data.append(np.load(path))
    def check():
        if time.monotonic()-tick>plan['budget']['pilot_max_seconds']:raise TimeoutError('Original CPU ceiling')
    selection=[]
    for config in CONFIGS:
        per={}
        for r,x in zip(plan['files'],data):
            if r['pid'] not in T.DEV:continue
            check();d,_=errors(x,config);per.setdefault(r['pid'],[]).append(d['corrected_gait'][d['valid']])
        bypid={pid:T.measures(np.concatenate(v))['rms_deg'] for pid,v in per.items()}
        selection.append(dict(config=config,participant_rms=bypid,equal_participant_rms=float(np.mean(list(bypid.values())))))
        print('SELECT_DEV',config,selection[-1]['equal_participant_rms'],flush=True)
    idx=min(range(len(selection)),key=lambda i:selection[i]['equal_participant_rms']);config=CONFIGS[idx]
    packed={k:[] for k in (*ARMS,'valid','turn_group','updated','bias_deg','file_index','frame_index')}
    per={};onsets=[];gate_counts={};gate_groups={};clip_rows=[]
    for fi,(r,x) in enumerate(zip(plan['files'],data)):
        check();d,gate=errors(x,config);per.setdefault(r['pid'],[]).append(d);n=len(x)
        for k in (*ARMS,'valid','turn_group','updated','bias_deg'):packed[k].append(d[k])
        packed['file_index'].append(np.full(n,fi,np.int32));packed['frame_index'].append(np.arange(n,dtype=np.int16))
        if r['pid'] not in T.EVAL:continue
        ids=gate['ids'];vv=d['valid'][ids];eligible=d['updated'][ids]
        masks={'all':vv,**{name:vv&(d['turn_group'][ids]==gid) for name,gid in [('straight',0),('slowturn',1),('onset',2),('otherturn',3)]}}
        for name,mask in masks.items():
            row=gate_groups.setdefault(name,dict(frames=0,speed_pass=0,direction_pass=0,torso_pass=0,all_pass=0,rates=[]))
            for key,ok in [('frames',np.ones(len(ids),bool)),('speed_pass',gate['speed_ok']),('direction_pass',gate['direction_ok']),('torso_pass',gate['body_ok']),('all_pass',eligible)]:row[key]+=int((ok&mask).sum())
            row['rates'].extend(gate['rate'][mask].tolist())
        clip_rows.append(dict(clip=r['name'],valid_frames=int(d['valid'].sum()),updated_frames=int(d['updated'].sum()),
            final_bias_deg=float(d['bias_deg'][-1]),first_update_frame=int(np.flatnonzero(d['updated'])[0]) if d['updated'].any() else None))
        changes=np.diff(np.r_[0.,d['bias_deg']])
        for onset in d['onsets']:
            before=slice(max(0,onset-120),onset);after=slice(onset,min(n,onset+60));valid=d['valid'][after]
            onsets.append(dict(pid=r['pid'],clip=r['name'],frame=onset,before_available_frames=min(onset,120),full_pre2s=bool(onset>=120),
                pre2s_update_fraction=float(d['updated'][before].mean()),pre2s_bias_change_deg=float(changes[before].sum()),
                pre2s_abs_bias_change_deg=float(np.abs(changes[before]).sum()),bias_at_onset_deg=float(d['bias_deg'][onset]),
                after={k:T.measures(d[k][after][valid]) for k in ARMS}))
    arrays={k:np.concatenate(v) for k,v in packed.items()}
    with (OUT/'series.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    original=dict(np.load(T.OUT/'series.npz'))
    for k in ('e1','torso','oracle','valid','turn_group','file_index','frame_index'):assert np.array_equal(arrays[k],original[k]),k
    assert T.sha(OUT/'windows.json')==plan['windows_sha256']
    summaries={}
    for role,pids in [('development',T.DEV),('evaluation',T.EVAL)]:
        participants={pid:summary(per[pid]) for pid in pids};pooled=summary([d for pid in pids for d in per[pid]])
        macro={name:{k:dict(equal_participant_rms_deg=float(np.mean([participants[pid]['groups'][name][k]['rms_deg']
            for pid in pids if participants[pid]['groups'][name][k] is not None]))) for k in ARMS} for name in pooled['groups']}
        summaries[role]=dict(participant=participants,pooled=pooled,participant_macro=macro)
    for g in gate_groups.values():
        rates=np.asarray(g.pop('rates'));g['median_rate_deg_s']=float(np.median(rates)) if len(rates) else None
        g['p90_rate_deg_s']=float(np.percentile(rates,90)) if len(rates) else None
    onset_stats={}
    for label,rows in [('all',onsets),('full_pre2s',[r for r in onsets if r['full_pre2s']]),('earlyclip',[r for r in onsets if not r['full_pre2s']])]:
        deltas=[r['after']['corrected_gait']['rms_deg']-r['after']['torso']['rms_deg'] for r in rows if r['after']['torso'] is not None]
        onset_stats[label]=dict(events=len(rows),anyupdate=sum(r['pre2s_update_fraction']>0 for r in rows),
            median_abs_bias_change_deg=float(np.median([r['pre2s_abs_bias_change_deg'] for r in rows])) if rows else None,
            mean_post_rms_delta_deg=float(np.mean(deltas)) if deltas else None,
            improved=sum(v<-1e-9 for v in deltas),worsened=sum(v>1e-9 for v in deltas))
    check();save('turn_onsets.json',onsets);save('clip_diagnostics.json',clip_rows)
    save('result_angles.json',dict(status='POSTHOC_DEVELOPMENT_PILOT_COMPLETE',seconds=time.monotonic()-tick,
        prior_reserved_upper_seconds=20,selected_index=idx,config=config,selection=selection,summaries=summaries,
        gate_groups=gate_groups,onset_stats=onset_stats,windows_sha256=T.sha(OUT/'windows.json'),series_sha256=T.sha(OUT/'series.npz'),
        plan_sha256=T.sha(OUT/'PLAN.json'),limits=plan['limits']))
    print('RESULT',json.dumps(dict(config=config,summary=summaries['evaluation']['pooled'],macro=summaries['evaluation']['participant_macro'],
        gate=gate_groups,onsets=onset_stats),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','self-check','run']);args=parser.parse_args()
    {'freeze':freeze,'self-check':self_check,'run':run}[args.stage]()
