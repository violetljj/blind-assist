"""Single fixed Development comparison on sealed MZ90 inputs."""
import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
import mz92_cross_sensor_calibration as model
import mz90_radar_fallback_audit as audit

previous=model.contract.previous
old=model.contract.old
TASK='mz92-cross-sensor-calibration-20260912'


def metrics(ids, evaluator, values, unknown):
    truth=evaluator['truth']; events=previous.event_rows(ids,truth,values)
    rows={}
    for name,pred in values.items():
        row=old.metrics(truth,pred)
        row.update(UNKNOWN=int(unknown[name].sum()),positive_UNKNOWN=int((unknown[name]&truth).sum()),
            false_segments=sum(len(previous.intervals(pred[ids==ep]&~truth[ids==ep])) for ep in np.unique(ids)),
            missed_events=sum(e['arms'][name]['first'] is None for e in events),
            fragments=sum(e['arms'][name]['fragments'] for e in events),
            future_only_TP=int((pred&evaluator['future_only_truth']).sum()))
        rows[name]=row
    comparison={}
    for name in ('mz91_control','mz92_calibrated'):
        b,c=values['matched_hold'],values[name]
        delays=[]; missed=0
        for event in events:
            bf,cf=event['arms']['matched_hold']['first'],event['arms'][name]['first']
            if bf is not None:
                if cf is None: missed+=1
                else: delays.append((cf-bf)*.1)
        comparison[name]=dict(removed_FP=int((b&~c&~truth).sum()),lost_TP=int((b&~c&truth).sum()),
            added_FP=int((~b&c&~truth).sum()),added_TP=int((~b&c&truth).sum()),
            missed_baseline_events=missed,max_added_delay_s=max(delays,default=0))
    b,c=rows['matched_hold'],rows['mz92_calibrated']; paired=comparison['mz92_calibrated']
    gates=dict(tp_loss_le_two=c['TP']>=b['TP']-2,fp_strictly_lower=c['FP']<b['FP'],f1_no_lower=c['F1']>=b['F1'],
        false_segments_no_increase=c['false_segments']<=b['false_segments'],fragments_no_increase=c['fragments']<=b['fragments'],
        no_missed_baseline_event=paired['missed_baseline_events']==0,delay_le_0p2=paired['max_added_delay_s']<=.2+1e-9)
    return dict(metrics=rows,paired=comparison,gates=gates,pass_all=all(gates.values()),events=events)


def provenance(obs, values, truth, origins):
    q=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    rows=[]
    for i in np.flatnonzero(values['matched_hold']):
        category='valid_tof'
        if not obs['tof_known'][i]:
            category='hold_without_recent_radar'
            for s in range(i,max(-1,i-3),-1):
                if obs['episode_id'][s]!=obs['episode_id'][i]: break
                if q[s].any():
                    kinds=set('real' if label.startswith('real:') else label for label in origins[s,q[s]])
                    category=next(iter(kinds))+'_only' if len(kinds)==1 else 'mixed'
                    break
        rows.append(dict(frame=int(i),category=category,outcome='TP' if truth[i] else 'FP',
                         control_retained=bool(values['mz91_control'][i]),calibrated_retained=bool(values['mz92_calibrated'][i])))
    result={}
    for outcome in ('TP','FP'):
        group=[r for r in rows if r['outcome']==outcome]
        result[outcome]={name:dict(Counter(r['category'] for r in group if name=='baseline' or not r[name+'_retained']))
                         for name in ('baseline','control','calibrated')}
    return rows,result


def run(root,out):
    if out.exists() or out.resolve().parent!=(previous.ROOT/'artifacts.local/work'/TASK).resolve():
        raise ValueError('Use a new child of the canonical task artifact root')
    receipt=json.loads((root/'receipt.json').read_text())
    for name,digest in receipt['hashes'].items(): assert old.sha(root/name)==digest,name
    control_root=previous.ROOT/'artifacts.local/work/mz91-radar-localization-20260912/run-v1'
    control_receipt=json.loads((control_root/'receipt.json').read_text())
    for name,digest in control_receipt['hashes'].items(): assert old.sha(control_root/name)==digest,name
    out.mkdir(parents=True); started=time.perf_counter()
    for name in ('mz92_cross_sensor_calibration.py','run_mz92_cross_sensor_calibration.py','test_mz92_cross_sensor_calibration.py','MZ92_CROSS_SENSOR_CALIBRATION_20260912.md'):
        shutil.copyfile(previous.HERE/name,out/name)
    for regime in ('ideal','sensor_proxy'):
        target=out/regime; target.mkdir()
        raw=dict(np.load(root/regime/'raw-observations.npz'))
        returned=[]
        def probe():
            returned.append(model.predict(raw)); return returned[-1]
        previous.select_backend('scalar-scoring',cpu=previous.BackendCandidate('numpy-cpu','cpu',probe,
            lambda _:previous.DeviceObservation('cpu','host CPU','numpy')),record_path=target/'backend.json',
            capabilities={'numpy':np.__version__,'scope':'small scalar track scoring'})
        obs,values,unknown,debug=returned[0]
        baseline=dict(np.load(root/regime/'predictions.npz'))
        np.testing.assert_array_equal(values['mz91_control'],np.load(control_root/regime/'predictions.npz')['multi_frame'])
        np.testing.assert_array_equal(model.contract.simple_controls(obs)['matched_hold'],baseline['matched_hold'])
        np.testing.assert_array_equal(model.assemble(obs,np.ones_like(obs['radar_range_m']))[0],baseline['matched_hold'])
        values['matched_hold']=baseline['matched_hold']; unknown['matched_hold']=baseline['matched_hold_unknown']
        for name in ('mz91_control','mz92_calibrated'):
            assert not (values[name]&~values['matched_hold']).any()
            assert not (values[name]&unknown[name]).any()
        np.savez_compressed(target/'predictions.npz',**values,**{n+'_unknown':v for n,v in unknown.items()},**debug)
        old.write(target/'prediction-receipt.json',dict(status='SEALED_BEFORE_EVALUATOR_ACCESS',sha256=old.sha(target/'predictions.npz')))
    result={}
    for regime in ('ideal','sensor_proxy'):
        pred=dict(np.load(out/regime/'predictions.npz'))
        obs=dict(np.load(root/regime/'adapted-observations.npz'))
        evaluator=dict(np.load(root/regime/'evaluator.npz'))
        names=('matched_hold','mz91_control','mz92_calibrated')
        result[regime]=metrics(obs['episode_id'],evaluator,{n:pred[n] for n in names},{n:pred[n+'_unknown'] for n in names})
    namespace={'__name__':'mz91_posthoc_provenance'}
    exec(compile(audit.instrument((root/'mz90_observation_source.py').read_text()),'<sealed-source-provenance>','exec'),namespace)
    raw,evaluator,origins=namespace['materialize'](json.loads((root/'source.json').read_text()),'sensor_proxy')
    audit.parity(raw,dict(np.load(root/'sensor_proxy/raw-observations.npz')))
    audit.parity(evaluator,dict(np.load(root/'sensor_proxy/evaluator.npz')))
    rows,slices=provenance(obs,{n:pred[n] for n in names},evaluator['truth'],origins)
    old.write(out/'provenance-rows.json',rows); result['sensor_proxy']['provenance']=slices
    result['decision']='CALIBRATION_DEVELOPMENT_GAIN' if result['sensor_proxy']['pass_all'] else 'CALIBRATION_GATE_NOT_MET'
    old.write(out/'result.json',result)
    old.write(out/'receipt.json',dict(status='PASS',decision=result['decision'],source_run=str(root),
        sealed_input_hashes_verified=len(receipt['hashes']),baseline_parity=True,provenance_replay_parity=True,
        seconds=time.perf_counter()-started,training_steps=0,threshold_searches=0,
        hashes={str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps({k:({n:v for n,v in row.items() if n!='events'} if isinstance(row,dict) else row) for k,row in result.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.source_run,args.output)
