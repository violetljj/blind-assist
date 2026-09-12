"""One consumed anchor feasibility test; downstream score only after fixed stage1 gates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import mz100_causal_angle_anchor as anchor
import run_mz99_angle_information as previous

ROOT=previous.ROOT
BASE=ROOT/'artifacts.local/work/mz99-angle-information-20260912'
TASK=ROOT/'artifacts.local/work/mz100-causal-angle-anchor-20260912'


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def r_predict(raw):
    obs=previous.audit.a.contract.adapt(raw)
    angle=np.radians(obs['radar_angle'])
    q=obs['radar_valid'] & (obs['radar_range_m']<3.6) & (obs['radar_range_m']*np.cos(angle)>.2) & (abs(obs['radar_angle'])<=12)
    return previous.audit.a.assemble_admission(obs,q)[0]


def run(out):
    assert not out.exists() and out.resolve().parent==TASK.resolve();out.mkdir(parents=True);start=time.perf_counter()
    for folder in (BASE/'capture-v1',BASE/'comparison-v1'):
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(folder/name)==digest,name
    for name in ('mz100_causal_angle_anchor.py','run_mz100_causal_angle_anchor.py','MZ100_CAUSAL_ANGLE_ANCHOR_PROTOCOL_20260912.md'):
        shutil.copyfile(Path(__file__).parent/name,out/name)
    raw=dict(np.load(BASE/'comparison-v1/measured-raw.npz'));assert set(raw)==previous.audit.a.contract.RAW_KEYS
    trace,records=anchor.estimate(raw);corrected=anchor.correct(raw,trace)
    for k in raw:
        if k!='radar_angle':np.testing.assert_array_equal(raw[k],corrected[k])
    np.testing.assert_array_equal(raw['radar_angle'][~trace['applied']],corrected['radar_angle'][~trace['applied']])
    np.savez_compressed(out/'anchor-estimates.npz',**trace);np.savez_compressed(out/'corrected-raw.npz',**corrected)
    write(out/'anchor-records.json',records)
    write(out/'estimator-seal.json',dict(status='ESTIMATES_AND_CORRECTED_INPUT_SEALED_BEFORE_EVALUATOR_ACCESS',
        payload_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        input_sha256=sha(BASE/'comparison-v1/measured-raw.npz'),
        implementation_sha256={str(Path(m.__file__).resolve().relative_to(ROOT.resolve())):sha(Path(m.__file__))
            for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file()
            and Path(m.__file__).resolve().is_relative_to(ROOT.resolve()) and 'artifacts.local' not in Path(m.__file__).parts}))
    # Evaluator-only fields below. No feedback to estimator or individual applications.
    provenance=previous.jsonlines(BASE/'capture-v1/provenance.jsonl')
    spec=json.loads((BASE/'capture-v1/spec.json').read_text());byid={s['episode_id']:s for s in spec['scenes']}
    anchor_kinds=Counter();validated_kinds=Counter()
    for row in records:
        for pair in row['pairs']:
            kind=provenance[row['frame']]['radar_slots'][pair['slot']]['kind']
            anchor_kinds[kind]+=1
            if row['status']=='validated_anchor':validated_kinds[kind]+=1
    ids=raw['episode_id'];truthbias=np.array([byid[e]['angle_bias'] for e in ids]);applied=trace['applied']
    obs=previous.audit.a.contract.adapt(raw);eligible=~obs['tof_known'] & obs['radar_valid'].any(1)
    err=np.abs(anchor.wrap(trace['bias_deg'][applied]-truthbias[applied]));before=[];after=[];counts=Counter()
    for i in np.flatnonzero(applied):
        assert provenance[i]['episode_id']==ids[i] and provenance[i]['time_s']==raw['time_s'][i]
        for k,p in enumerate(provenance[i]['radar_slots']):
            if p is None or not obs['radar_valid'][i,k]:continue
            counts[p['kind']]+=1
            if p['kind']=='real_actor':
                before.append(abs(anchor.wrap(raw['radar_angle'][i,k]-p['exact_angle_deg'])))
                after.append(abs(anchor.wrap(corrected['radar_angle'][i,k]-p['exact_angle_deg'])))
    mean=lambda x:float(np.mean(x)) if len(x) else None
    episodes=[]
    for ep in np.unique(ids):
        m=(ids==ep)&applied;diff=np.abs(anchor.wrap(trace['bias_deg'][m]-truthbias[m]));ix=np.flatnonzero(ids==ep)
        episodes.append(dict(episode_id=str(ep),family=byid[ep]['family'],true_bias=byid[ep]['angle_bias'],
            eligible_frames=int(eligible[ix].sum()),applied_frames=int(m.sum()),bias_MAE=mean(diff),within5=mean(diff<=5),
            anchor_statuses=dict(Counter(records[i]['status'] for i in ix))))
    coverage=float(applied.sum()/eligible.sum()) if eligible.any() else 0.
    gates=dict(coverage_ge_10pct=coverage>=.1,bias_within5_ge_90pct=bool(len(err) and (err<=5).mean()>=.9),
        real_return_MAE_decreases=bool(len(before) and np.mean(after)<np.mean(before)))
    diag=dict(eligible_frames=int(eligible.sum()),applied_frames=int(applied.sum()),coverage=coverage,
        applied_episodes=sum(e['applied_frames']>0 for e in episodes),bias_MAE=mean(err),bias_within5=mean(err<=5),
        real_return_slots=len(before),measured_angle_MAE=mean(before),corrected_angle_MAE=mean(after),
        median_age_s=float(np.median(trace['age_s'][applied])) if applied.any() else None,
        return_kinds=dict(counts),anchor_pair_kinds=dict(anchor_kinds),validated_frame_pair_kinds=dict(validated_kinds),
        statuses=dict(Counter(r['status'] for r in records)),episodes=episodes)
    write(out/'stage1-diagnostic.json',diag)
    result=dict(stage1_gates=gates,stage1_pass=all(gates.values()),downstream_scored=False,
        decision='CAUSAL_ANGLE_ANCHOR_FEASIBILITY_FAILED',frames=len(ids),episodes=len(episodes))
    if all(gates.values()):
        values=dict(R=r_predict(raw),anchored_R=r_predict(corrected))
        np.testing.assert_array_equal(values['R'],np.load(BASE/'comparison-v1/measured-predictions.npz')['R'])
        np.savez_compressed(out/'predictions.npz',**values)
        write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.npz'),status='SEALED_BEFORE_TASK_LABELS'))
        labels=dict(np.load(BASE/'comparison-v1/evaluation-labels.npz'));tasks={}
        for task,future in (('current_route',np.zeros(len(ids),bool)),('body_1s',labels['strict_future_contact']),('legacy',labels['legacy_future'])):
            tasklabels=dict(truth=labels[task],future_only_truth=future)
            pair=previous.audit.paired.compare(ids,tasklabels,values['R'],values['anchored_R'])
            sessions={n:previous.evaluator.session_metrics(v,ids,labels[task]) for n,v in values.items()}
            tasks[task]=dict(paired=pair,sessions=sessions)
        p=tasks['current_route']['paired'];b=p['metrics']['reference'];c=p['metrics']['candidate']
        finalgates=dict(retention=p['TP_retention']>=.98,no_lost_event=p['lost_reference_events']==0,
            delay=p['max_added_delay_s']<=.2+1e-9,fp_reduced=c['FP']<b['FP'],
            false_segments=c['false_segments']<=b['false_segments'],fragments=c['fragments']<=b['fragments'])
        contacts=json.loads((BASE/'comparison-v1/contact-events.json').read_text())
        result.update(downstream_scored=True,tasks=tasks,final_gates=finalgates,
            contact_timing={n:previous.evaluator.timing_metrics(v,ids,raw['time_s'],contacts) for n,v in values.items()},
            decision='CAUSAL_ANGLE_ANCHOR_CONSUMED_GAIN' if all(finalgates.values()) else 'CAUSAL_ANGLE_ANCHOR_FINAL_GATE_NOT_MET')
    result['seconds']=time.perf_counter()-start;write(out/'result.json',result)
    write(out/'receipt.json',dict(status='PASS',hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        backend='CPU_NO_TRAINING_CONSUMED_MZ99',decision=result['decision']))
    print(json.dumps(dict(result=result,stage1={k:v for k,v in diag.items() if k!='episodes'}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
