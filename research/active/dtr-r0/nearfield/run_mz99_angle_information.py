"""Freeze measured/corrected-angle predictions before separate target scoring."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import audit_mz98_a_mechanisms as audit
import mz99_angle_information_eval as evaluator

old=audit.old;ROOT=audit.ROOT
TASK=ROOT/'artifacts.local/work/mz99-angle-information-20260912'
SPEC_SHA='1fd71e296d6cb3716286463c4ab2a64698b10b315bde7ded063e1d88e97e1153'


def jsonlines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def materialize(capture):
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['frames']==5120
    assert old.sha(capture/'spec.json')==SPEC_SHA
    for name,digest in receipt['hashes'].items():assert old.sha(capture/name)==digest
    spec=json.loads((capture/'spec.json').read_text());assert spec['seed']==99013 and len(spec['scenes'])==128
    rows=jsonlines(capture/'raw.jsonl');provenance=jsonlines(capture/'provenance.jsonl');assert len(rows)==len(provenance)==5120
    assert all(set(r)==audit.a.contract.RAW_KEYS for r in rows)
    assert [(r['episode_id'],r['time_s']) for r in rows]==[(r['episode_id'],r['time_s']) for r in provenance]
    assert set(r['episode_id'] for r in rows)==set(s['episode_id'] for s in spec['scenes'])
    for start in range(0,len(rows),40):
        block=rows[start:start+40];assert len({v['episode_id'] for v in block})==1
        assert sum(v['episode_id']==block[0]['episode_id'] for v in rows)==40
        assert np.allclose([v['time_s'] for v in block],np.arange(40)*.1)
    boolean={'tof_packet_received','radar_packet_received','radar_valid','imu_valid'}
    raw={k:np.asarray([r[k] for r in rows],dtype=str if k=='episode_id' else bool if k in boolean else np.uint8 if k=='tof_status' else float) for k in rows[0]}
    corrected={k:v.copy() for k,v in raw.items()};kinds=Counter();changed=Counter()
    for i,row in enumerate(provenance):
        assert len(row['radar_slots'])==4
        for k,item in enumerate(row['radar_slots']):
            assert (item is not None)==bool(raw['radar_valid'][i,k])
            if item is None:continue
            assert item['observed_triple']==[float(raw['radar_range_m'][i,k]),float(raw['radar_angle'][i,k]),float(raw['radar_velocity'][i,k])]
            kind=item['kind'];kinds[kind]+=1
            if kind in ('real_actor','persistent_ghost'):
                angle=item['exact_angle_deg'];assert angle is not None and np.isfinite(angle)
                corrected['radar_angle'][i,k]=angle
            else:assert kind=='transient' and item['exact_angle_deg'] is None
            changed[kind]+=int(corrected['radar_angle'][i,k]!=raw['radar_angle'][i,k])
    for k in raw:
        if k!='radar_angle':assert np.array_equal(raw[k],corrected[k],equal_nan=True) if raw[k].dtype.kind=='f' else np.array_equal(raw[k],corrected[k])
    assert changed['transient']==0
    return raw,corrected,provenance,dict(valid_returns=dict(kinds),changed_angles=dict(changed),only_radar_angle_changes=True,
        source_sha256=SPEC_SHA,capture_receipt_sha256=old.sha(capture/'receipt.json'))


def predict(raw):
    obs=audit.a.contract.adapt(raw);tags,q,records=audit.instrument(obs);r=np.isin(tags,['R_near','R_extended'])
    legacy=obs['radar_valid']&(obs['radar_range_m']<3.18)&(np.abs(obs['radar_angle'])<=20)&(obs['radar_velocity']<=-.35)
    values={};traces={};qs=dict(baseline=legacy,R=r,RF=r|(tags=='F'),A=q)
    for name,mask in qs.items():
        values[name],traces[name]=audit.staged(obs,mask,np.full(tags.shape,'legacy') if name=='baseline' else tags)
    assert np.array_equal(values['baseline'],audit.a.contract.simple_controls(obs)['matched_hold'])
    assert np.array_equal(values['A'],audit.a.assemble_admission(obs,audit.a.horizon_admission(obs)[0])[0])
    return obs,values,traces,qs


def source_groups(values,traces,qs,provenance,truth):
    groups={}
    for method,pred in values.items():
        rows={}
        for i in np.flatnonzero(pred):
            trace=traces[method][i];kind='ToF_geometry'
            if trace['sensor']=='Radar':
                origin=trace['origin_frame'];slots=np.flatnonzero(qs[method][origin])
                kind='+'.join(sorted({provenance[origin]['radar_slots'][k]['kind'] for k in slots}))
            key='|'.join((trace['sensor'],trace['evidence_stage'],kind));row=rows.setdefault(key,dict(TP=0,FP=0))
            row['TP' if truth[i] else 'FP']+=1
        groups[method]=rows
    return groups


def run(capture,out):
    assert not out.exists() and out.resolve().parent==TASK.resolve();out.mkdir(parents=True);started=time.perf_counter()
    for name in ('run_mz99_angle_information.py','mz99_angle_information_eval.py','MZ99_ANGLE_INFORMATION_PROTOCOL_20260912.md'):
        shutil.copyfile(audit.common.previous.HERE/name,out/name)
    raw,corrected,provenance,pairing=materialize(capture);old.write(out/'pairing.json',pairing)
    inputs=dict(measured=raw,angle_corrected=corrected);allvalues={};observations={};traces={};qs={}
    for arm,packets in inputs.items():
        np.savez_compressed(out/f'{arm}-raw.npz',**packets)
        observations[arm],allvalues[arm],traces[arm],qs[arm]=predict(packets)
        np.savez_compressed(out/f'{arm}-predictions.npz',**allvalues[arm])
    old.write(out/'prediction-seal.json',dict(status='BOTH_INPUTS_ALL_PREDICTIONS_SEALED_BEFORE_SCORING',
        payload_hashes={p.name:old.sha(p) for p in out.glob('*.npz')},pairing_sha256=old.sha(out/'pairing.json'),
        implementation_sha256={str(Path(m.__file__).resolve().relative_to(ROOT.resolve())):old.sha(Path(m.__file__))
            for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file()
            and Path(m.__file__).resolve().is_relative_to(ROOT.resolve()) and 'artifacts.local' not in Path(m.__file__).parts}))
    # No task labels were scored or used in predictions above.
    geometry=jsonlines(capture/'native-geometry.jsonl');legacy=jsonlines(capture/'evaluator.jsonl')
    ids=raw['episode_id'];times=raw['time_s'];keys=list(zip(ids,times))
    assert keys==[(v['episode_id'],v['time_s']) for v in geometry]==[(v['episode_id'],v['time_s']) for v in legacy]
    labels=evaluator.labels_from_geometry(geometry)
    labels['legacy']=np.array([v['truth'] for v in legacy],bool);labels['legacy_future']=np.array([v['future_only_truth'] for v in legacy],bool)
    np.savez_compressed(out/'evaluation-labels.npz',**labels)
    contacts=evaluator.contact_events(ids,times,geometry);old.write(out/'contact-events.json',contacts)
    results={};pairs={};provenance_results={};bootstrap={};rng=np.random.default_rng(99013)
    ep_rows=[np.flatnonzero(ids==ep) for ep in np.unique(ids)];draws=rng.integers(0,len(ep_rows),(1000,len(ep_rows)))
    for task,future in (('current_route',np.zeros(len(ids),bool)),('body_1s',labels['strict_future_contact']),('legacy',labels['legacy_future'])):
        tasklabels=dict(truth=labels[task],future_only_truth=future);results[task]={};pairs[task]={};provenance_results[task]={};bootstrap[task]={}
        for arm,values in allvalues.items():
            metrics,events=audit.common.score(ids,tasklabels,values)
            for method,pred in values.items():
                obs=observations[arm]
                metrics[method].update(UNKNOWN=int((~obs['tof_known']&~pred).sum()),
                    positive_UNKNOWN=int((labels[task]&~obs['tof_known']&~pred).sum()),
                    sessions=evaluator.session_metrics(pred,ids,labels[task]))
            results[task][arm]=metrics
            provenance_results[task][arm]=source_groups(values,traces[arm],qs[arm],provenance,labels[task])
        for method in allvalues['measured']:
            measured=allvalues['measured'][method];corrected_pred=allvalues['angle_corrected'][method]
            pairs[task][method]=audit.paired.compare(ids,tasklabels,measured,corrected_pred)
            counts=[]
            for ep in ep_rows:
                counts.append([int((labels[task][ep]&p[ep]).sum()) for p in (measured,corrected_pred)]+
                    [int((~labels[task][ep]&p[ep]).sum()) for p in (measured,corrected_pred)]+[int(labels[task][ep].sum())])
            counts=np.array(counts);sums=counts[draws].sum(1);deltas=[]
            for k in (0,1):deltas.append(np.divide(2*sums[:,k],sums[:,k]+sums[:,2+k]+sums[:,4],out=np.zeros(1000),where=(sums[:,k]+sums[:,2+k]+sums[:,4])>0))
            bootstrap[task][method]=np.quantile(deltas[1]-deltas[0],[.025,.975]).tolist()
    timing={arm:{method:evaluator.timing_metrics(pred,ids,times,contacts) for method,pred in values.items()} for arm,values in allvalues.items()}
    m=results['current_route']['measured']['R'];c=results['current_route']['angle_corrected']['R'];p=pairs['current_route']['R']
    gates=dict(fp_reduction_ge_20pct=c['FP']<=.8*m['FP'] and m['FP']>0,
        paired_tp_retention_ge_98=p['retained_TP']>=.98*p['reference_TP'],no_lost_route_interval=p['lost_reference_events']==0)
    result=dict(decision='ANGLE_INFORMATION_BENEFIT_ON_CURRENT_ROUTE' if all(gates.values()) else 'ANGLE_INFORMATION_TRADEOFF_OR_GATE_NOT_MET',
        primary_gates=gates,task_metrics=results,paired_changes=pairs,contact_timing=timing,
        label_counts={k:int(v.sum()) for k,v in labels.items()},bootstrap_F1_difference_95pct=bootstrap,
        frames=len(ids),episodes=len(ep_rows),minutes=len(ids)*.1/60,seconds=time.perf_counter()-started)
    old.write(out/'result.json',result);old.write(out/'source-paths.json',provenance_results)
    old.write(out/'receipt.json',dict(status='PASS',hashes={str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()},
        backend='CPU_SCALAR_GEOMETRY_NO_TRAINING',prediction_condition_count=8,only_angle_intervention=True,new_task_not_historical_relabel_gain=True))
    print(json.dumps(dict(decision=result['decision'],primary_gates=gates,label_counts=result['label_counts'],
        compact_metrics={t:{a:{m:{k:v for k,v in r.items() if k in ('TP','FP','FN','F1','future_only_TP','future_only_positive')} for m,r in d.items()} for a,d in ar.items()} for t,ar in results.items()}),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.capture,args.output)
