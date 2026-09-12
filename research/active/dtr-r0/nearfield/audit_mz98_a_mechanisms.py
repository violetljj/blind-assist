"""Consumed replay instrumentation and fixed exclusions; no frozen policy edits."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import mz94_radar_horizon as a
import run_mz96_decision_head as common
import mz97_residual_suppression as paired
import mz96_ue_decision_capture as capture_code

ROOT=common.ROOT;old=common.old
TASK=ROOT/'artifacts.local/work/mz98-a-mechanism-audit-20260912'
PANELS={'mz96':'mz96-ue-decision-20260912','mz97':'mz97-residual-suppression-20260912'}


def classify(history,original=None):
    _,r,angle,vr=history[-1];theta=np.radians(angle)
    raw=bool(r*np.cos(theta)>.2 and r<3.6 and abs(angle)<=12)
    result=bool((original or a.trajectory_admission)(history))
    mean=None
    if raw:branch='R_near' if r<3.18 else 'R_extended'
    elif result:
        mean,_=a.localize(history,0.,True)
        fr,_,fa,_=mean;x,z=fr*np.sin(np.radians(fa)),fr*np.cos(np.radians(fa))
        branch='F' if abs(x)<=np.tan(np.radians(12))*z else 'P'
    else:branch='REJECTED'
    return result,branch,mean


def instrument(obs):
    records=[];used=Counter();tags=np.full(obs['radar_valid'].shape,'INVALID',dtype='<U12')
    original=a.trajectory_admission
    # Serial, scoped observer: same original histories and matching, restored in finally.
    def observer(history):
        i=int(history[-1][0]);slot=int(np.flatnonzero(obs['radar_valid'][i])[used[i]]);used[i]+=1
        result,branch,mean=classify(history,original);tags[i,slot]=branch
        _,r,angle,vr=history[-1]
        records.append(dict(frame=i,slot=slot,branch=branch,admitted=result,history=[list(v) for v in history],
            fitted_mean=None if mean is None else mean.tolist(),legacy_range_fail=r>=3.18,
            legacy_bearing_fail=abs(angle)>20,legacy_doppler_fail=vr>-.35))
        return result
    a.trajectory_admission=observer
    try:admitted,length=a.horizon_admission(obs)
    finally:
        a.trajectory_admission=original
    assert np.array_equal(admitted,np.isin(tags,['R_near','R_extended','F','P']))
    return tags,admitted,records


def staged(obs,q,tags):
    controls=a.contract.simple_controls({**obs,'radar_valid':np.zeros_like(q)})
    tof=controls['matched_hard'];direct=controls['current_geometry']
    current=q.any(1);radar=a.hysteresis(current,obs['episode_id'])
    hard=tof|(~obs['tof_known']&radar);final=a.assemble_admission(obs,q)[0]
    rows=[];latest=-1
    for i in range(len(final)):
        if i==0 or obs['episode_id'][i]!=obs['episode_id'][i-1]:latest=-1
        if current[i]:latest=i
        origin=i;stage='current';sensor='none';mix='none'
        if hard[i]:
            if tof[i]:sensor='ToF';stage='direct' if direct[i] else 'temporal'
            else:
                sensor='Radar';stage='current' if current[i] else 'hysteresis'
                origin=i if current[i] else latest
                assert origin>=0
                mix='+'.join(sorted(set(tags[origin][q[origin]])))
        elif final[i]:
            previous=rows[i-1];sensor=previous['sensor'];origin=previous['origin_frame'];mix=previous['branch_mix'];stage='outer_hold'
            assert hard[i-1]
        rows.append(dict(sensor=sensor,evidence_stage=stage,origin_frame=origin,branch_mix=mix))
    return final,rows


def group_outcomes(mask,truth,future,keys):
    groups={}
    for i in np.flatnonzero(mask):
        key=keys[i];d=groups.setdefault(key,dict(frames=0,TP=0,FP=0,future_only_TP=0))
        d['frames']+=1;d['TP']+=int(truth[i]);d['FP']+=int(not truth[i]);d['future_only_TP']+=int(future[i])
    return groups


def verified_inputs(panel):
    base=ROOT/'artifacts.local/work'/PANELS[panel];comparison=base/'comparison-v1';capture=base/'capture-v1'
    for path in (comparison,capture):
        receipt=json.loads((path/'receipt.json').read_text())
        assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert old.sha(path/name)==digest,(path,name)
    seal=json.loads((comparison/'test-prediction-seal.json').read_text())
    for key,path in (('model_sha256','model.ubj'),('selection_sha256','selection.json'),('prediction_sha256','test/predictions.npz')):
        assert old.sha(comparison/path)==seal[key]
    for name,digest in seal['implementation_sha256'].items():assert old.sha(ROOT/name)==digest,name
    return comparison,capture


def audit(panel,out):
    source,capture=verified_inputs(panel);raw=dict(np.load(source/'test/raw.npz'));labels=dict(np.load(source/'test/labels.npz'))
    frozen=dict(np.load(source/'test/predictions.npz'));obs=a.contract.adapt(raw);ids=obs['episode_id']
    tags,q,returns=instrument(obs);r=np.isin(tags,['R_near','R_extended']);f=tags=='F'
    variant_q=dict(full_A=q,raw_current_only=r,no_strict_future=r|f,
        raw_current_legacy_range=r&(obs['radar_range_m']<3.18),
        raw_current_keep_closing=r&(obs['radar_velocity']<=-.35),
        raw_current_legacy_both=r&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35))
    values={'matched_hold':a.contract.simple_controls(obs)['matched_hold']}
    assert np.array_equal(values['matched_hold'],frozen['matched_hold'])
    traces={}
    for name,m in variant_q.items():values[name],traces[name]=staged(obs,m,tags)
    assert np.array_equal(values['full_A'],frozen['horizon_A'])
    legacy=obs['radar_valid']&(obs['radar_range_m']<3.18)&(np.abs(obs['radar_angle'])<=20)&(obs['radar_velocity']<=-.35)
    legacyfinal,legacytrace=staged(obs,legacy,np.full(tags.shape,'legacy'))
    assert np.array_equal(legacyfinal,values['matched_hold'])
    metrics,events=common.score(ids,labels,values)
    pairs={name:paired.compare(ids,labels,values[b],values[c]) for name,b,c in (
        ('A_vs_baseline','matched_hold','full_A'),('raw_vs_baseline','matched_hold','raw_current_only'),
        ('fitted_current_increment','raw_current_only','no_strict_future'),('strict_future_increment','no_strict_future','full_A'),
        ('raw_range_increment','raw_current_legacy_range','raw_current_only'),
        ('raw_drop_closing_increment','raw_current_keep_closing','raw_current_only'),
        ('narrow_bearing_vs_baseline','matched_hold','raw_current_legacy_both'))}
    evaluator=[json.loads(line) for line in (capture/'evaluator.jsonl').read_text().splitlines()]
    geometry=[json.loads(line) for line in (capture/'native-geometry.jsonl').read_text().splitlines()]
    ek={(v['episode_id'],round(v['time_s'],5)):v for v in evaluator}
    gk={(v['episode_id'],round(v['time_s'],5)):v for v in geometry}
    current_extended=[];family=[];ghost=[]
    for i,ep in enumerate(ids):
        key=(str(ep),round(float(raw['time_s'][i]),5));e=ek[key];g=gk[key]
        gt=[capture_code.hazard(b['x'],b['z'],b['vx'],b['vz']) for b in g['native_bounds']]
        assert any(v[0] for v in gt)==bool(labels['truth'][i])
        assert any(v[1] for v in gt)==bool(labels['any_direct_truth'][i])
        current_extended.append(any(b['z']>.2 and np.hypot(b['x'],b['z'])<3.6 and abs(np.degrees(np.arctan2(b['x'],b['z'])))<=12 for b in g['native_bounds']))
        family.append(e['family']);ghost.append('ghost_scene' if e['persistent_ghost_scene'] else 'no_ghost_scene')
    truth=labels['truth'];future=labels['future_only_truth'];current_extended=np.array(current_extended)
    future_split={}
    for name,mask in (('current_extended_wedge',future&current_extended),('strict_future_only',future&~current_extended)):
        future_split[name]=dict(positive=int(mask.sum()),TP={k:int((v&mask).sum()) for k,v in values.items()})
    keys=['|'.join(row[k] for k in ('sensor','evidence_stage','branch_mix')) for row in traces['full_A']]
    cohorts={'A_all_alerts':values['full_A'],'A_added_alerts':values['full_A']&~values['matched_hold'],
        'baseline_alerts_removed_by_A':values['matched_hold']&~values['full_A']}
    path_groups={}
    for name,mask in cohorts.items():
        usekeys=keys if name!='baseline_alerts_removed_by_A' else ['|'.join(row[k] for k in ('sensor','evidence_stage','branch_mix')) for row in legacytrace]
        path_groups[name]=group_outcomes(mask,truth,future,usekeys)
    raw_groups={}
    for row in returns:
        i=row['frame'];key=row['branch'];d=raw_groups.setdefault(key,dict(returns=0,on_positive_frames=0,on_negative_frames=0,on_future_only_frames=0,
            legacy_range_fail=0,legacy_bearing_fail=0,legacy_doppler_fail=0))
        d['returns']+=1;d['on_positive_frames']+=int(truth[i]);d['on_negative_frames']+=int(not truth[i]);d['on_future_only_frames']+=int(future[i])
        for key in ('legacy_range_fail','legacy_bearing_fail','legacy_doppler_fail'):d[key]+=int(row[key])
    frames=[]
    for i in range(len(ids)):
        frames.append(dict(frame=i,episode_id=str(ids[i]),time_s=float(raw['time_s'][i]),truth=bool(truth[i]),future_only=bool(future[i]),
            evaluator_current_extended=bool(current_extended[i]),family=family[i],ghost_scene=ghost[i],
            predictions={k:bool(v[i]) for k,v in values.items()},a_path=traces['full_A'][i],baseline_path=legacytrace[i],
            return_branches=tags[i].tolist()))
    result=dict(metrics=metrics,pairs=pairs,future_GT_split=future_split,raw_return_groups=raw_groups,final_path_groups=path_groups,
        family_A_alerts=group_outcomes(values['full_A'],truth,future,family),ghost_scene_A_alerts=group_outcomes(values['full_A'],truth,future,ghost),
        frozen_prediction_parity=True,frames=len(ids),source_receipt_sha256=old.sha(source/'receipt.json'),events=events)
    out.mkdir();old.write(out/'result.json',result)
    for name,rows in (('frames',frames),('returns',returns)):
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    np.savez_compressed(out/'diagnostic-predictions.npz',**values)
    return result


def run(out):
    assert not out.exists() and out.resolve().parent==TASK.resolve();out.mkdir(parents=True)
    for name in ('audit_mz98_a_mechanisms.py','MZ98_A_MECHANISM_AUDIT_20260912.md'):shutil.copyfile(common.previous.HERE/name,out/name)
    results={panel:audit(panel,out/panel) for panel in PANELS}
    aggregate={}
    for name in results['mz96']['metrics']:
        row={k:sum(r['metrics'][name][k] for r in results.values()) for k in ('TP','FP','FN','future_only_TP','future_only_positive')}
        row['F1']=2*row['TP']/(2*row['TP']+row['FP']+row['FN']);aggregate[name]=row
    old.write(out/'summary.json',dict(decision='A_MECHANISM_AUDIT_COMPLETE',descriptive_aggregate=aggregate,authority='CONSUMED_DIAGNOSTIC_NOT_PROMOTION'))
    old.write(out/'receipt.json',dict(status='PASS',hashes={str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()},
        backend='CPU_SCALAR_DIAGNOSTIC_NO_TRAINING',new_collection=False,policy_promotion=False))
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
