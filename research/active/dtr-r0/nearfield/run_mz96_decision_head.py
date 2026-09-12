"""One fresh scene-disjoint UE comparison; test labels opened after prediction seal."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import mz96_decision_features as features

previous=features.a.contract.previous
old=features.a.contract.old
ROOT=previous.ROOT
TASK=ROOT/'artifacts.local/work/mz96-ue-decision-20260912'
sys.path.insert(0,str(TASK/'python-deps'))


def score(ids,evaluator,values):
    truth=evaluator['truth'];events=previous.event_rows(ids,truth,values);result={}
    for name,v in values.items():
        row=old.metrics(truth,v)
        row.update(false_segments=sum(len(previous.intervals(v[ids==ep]&~truth[ids==ep])) for ep in np.unique(ids)),
            fragments=sum(e['arms'][name]['fragments'] for e in events),missed_events=sum(e['arms'][name]['first'] is None for e in events),
            future_only_TP=int((v&evaluator['future_only_truth']).sum()))
        result[name]=row
    return result,events


def materialize(capture,out):
    assert old.sha(capture/'spec.json')=='a87173e43287eb6f5cb18cf3adf101337d6b7a52a7817a198734faab3266fb06'
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['frames']==5120
    for name,digest in receipt['hashes'].items():assert old.sha(capture/name)==digest
    spec=json.loads((capture/'spec.json').read_text());splits={s['episode_id']:s['split'] for s in spec['scenes']}
    assert spec['seed']==96013
    assert len(splits)==len(spec['scenes'])==128
    assert {s:list(splits.values()).count(s) for s in ('train','validation','test')}==dict(train=80,validation=16,test=32)
    rawrows=[json.loads(line) for line in (capture/'raw.jsonl').read_text().splitlines()]
    assert len(rawrows)==5120 and all(set(r)==features.a.contract.RAW_KEYS for r in rawrows)
    assert set(r['episode_id'] for r in rawrows)==set(splits)
    for start in range(0,5120,40):
        block=rawrows[start:start+40]
        assert len({r['episode_id'] for r in block})==1
        assert sum(r['episode_id']==block[0]['episode_id'] for r in rawrows)==40
        assert np.allclose([r['time_s'] for r in block],np.arange(40)*.1)
    # Mechanical split routing only: do not aggregate or select on held-out labels here.
    labelrows=[json.loads(line) for line in (capture/'evaluator.jsonl').read_text().splitlines()]
    assert [(r['episode_id'],r['time_s']) for r in rawrows]==[(r['episode_id'],r['time_s']) for r in labelrows]
    boolean={'tof_packet_received','radar_packet_received','radar_valid','imu_valid'}
    for split,size in (('train',3200),('validation',640),('test',1280)):
        indexes=[i for i,r in enumerate(rawrows) if splits[r['episode_id']]==split];assert len(indexes)==size
        raw={k:np.asarray([rawrows[i][k] for i in indexes],dtype=str if k=='episode_id' else bool if k in boolean else np.uint8 if k=='tof_status' else float) for k in rawrows[0]}
        labels={k:np.asarray([labelrows[i][k] for i in indexes],bool) for k in ('truth','any_direct_truth','future_only_truth')}
        target=out/split;target.mkdir()
        np.savez_compressed(target/'raw.npz',**raw);np.savez_compressed(target/'labels.npz',**labels)
    old.write(out/'data-receipt.json',dict(source_sha256=old.sha(capture/'spec.json'),capture_receipt_sha256=old.sha(capture/'receipt.json'),
        split_sizes=dict(train=3200,validation=640,test=1280),test_labels_sha256=old.sha(out/'test/labels.npz')))


def run(capture,out):
    import xgboost as xgb
    if out.exists() or out.resolve().parent!=TASK.resolve():raise ValueError('New canonical output required')
    out.mkdir(parents=True);started=time.perf_counter()
    for name in ('mz96_decision_features.py','run_mz96_decision_head.py','MZ96_UE_DECISION_HEAD_20260912.md'):
        shutil.copyfile(previous.HERE/name,out/name)
    materialize(capture,out)
    inputs={};rulevalues={};observations={}
    for split in ('train','validation','test'):
        raw=dict(np.load(out/split/'raw.npz'))
        obs,X,names,rules=features.build(raw)
        inputs[split]=X;rulevalues[split]=rules;observations[split]=obs
    y=np.load(out/'train/labels.npz')['truth'];validation=dict(np.load(out/'validation/labels.npz'))
    models={}
    kwargs=dict(n_estimators=300,max_depth=3,learning_rate=.05,min_child_weight=5,subsample=.8,colsample_bytree=.8,
        reg_lambda=5,random_state=96013,tree_method='hist',objective='binary:logistic',n_jobs=4)
    def fit(device):
        model=xgb.XGBClassifier(**kwargs,device=device);model.fit(inputs['train'],y)
        models[device]=model;return model
    def observe(model):
        device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        return previous.DeviceObservation('cuda' if device.startswith('cuda') else 'cpu',device,'xgboost')
    backend=previous.select_backend('batch-tensor',cpu=previous.BackendCandidate('xgboost-cpu','cpu',lambda:fit('cpu'),observe),
        gpu=previous.BackendCandidate('xgboost-cuda','cuda',lambda:fit('cuda'),observe),warmups=0,repeats=1,
        record_path=out/'backend.json',capabilities={'xgboost':xgb.__version__,'work':'equivalent fixed training fits; no validation selection'})
    chosen='cuda' if backend['selected_device_type']=='cuda' else 'cpu';model=models[chosen]
    model.save_model(out/'model.ubj')
    probabilities={s:model.predict_proba(X)[:,1] for s,X in inputs.items()}
    ids=observations['validation']['episode_id']
    candidates=[]
    for threshold in np.arange(.1,.901,.05):
        p=features.a.hysteresis(probabilities['validation']>=threshold,ids)
        metrics=old.metrics(validation['truth'],p)
        candidates.append((metrics['F1'],-metrics['FP'],float(threshold)))
    threshold=max(candidates)[2]
    valmetrics,_=score(ids,validation,rulevalues['validation'])
    rule_names=list(rulevalues['validation'])
    reference=max(rule_names,key=lambda name:(valmetrics[name]['F1'],-valmetrics[name]['FP'],-rule_names.index(name)))
    old.write(out/'selection.json',dict(threshold=threshold,reference_rule=reference,validation_rule_metrics=valmetrics,
        threshold_candidates=[dict(F1=f,FP=-fp,threshold=t) for f,fp,t in candidates],feature_names=names,model_config=kwargs,
        train_rows=len(y),validation_rows=len(ids),test_labels_opened_for_selection=False))
    for split in ('validation','test'):
        obs=observations[split];values=rulevalues[split]
        values['xgboost']=features.a.hysteresis(probabilities[split]>=threshold,obs['episode_id'])
        np.savez_compressed(out/split/'predictions.npz',**values,xgboost_score=probabilities[split])
    old.write(out/'test-prediction-seal.json',dict(status='SEALED_BEFORE_TEST_EVALUATION',
        model_sha256=old.sha(out/'model.ubj'),selection_sha256=old.sha(out/'selection.json'),prediction_sha256=old.sha(out/'test/predictions.npz')))
    # Only now can held-out labels affect any reported performance or conclusion.
    test=dict(np.load(out/'test/labels.npz'));obs=observations['test'];values=rulevalues['test'];ids=obs['episode_id']
    metrics,events=score(ids,test,values)
    for name,v in values.items():metrics[name].update(UNKNOWN=int((~obs['tof_known']&~v).sum()),positive_UNKNOWN=int((test['truth']&~obs['tof_known']&~v).sum()))
    ref,learned=metrics[reference],metrics['xgboost'];delays=[];lost=0
    for e in events:
        first=e['arms'][reference]['first'];now=e['arms']['xgboost']['first']
        if first is not None:
            if now is None:lost+=1
            else:delays.append((now-first)*.1)
    gates=dict(f1_gain_ge_0p02=learned['F1']>=ref['F1']+.02,tp_no_lower=learned['TP']>=ref['TP'],fp_no_higher=learned['FP']<=ref['FP'],
        no_lost_reference_event=lost==0,delay_le_0p2=max(delays,default=0)<=.2+1e-9,
        false_segments_no_increase=learned['false_segments']<=ref['false_segments'],fragments_no_increase=learned['fragments']<=ref['fragments'])
    spec=json.loads((capture/'spec.json').read_text())
    ghost_ids={s['episode_id'] for s in spec['scenes'] if s.get('radar_ghost') is not None}
    ghost=np.array([str(ep) in ghost_ids for ep in ids]);slices={}
    for label,mask in (('ghost_present',ghost),('ghost_absent',~ghost)):
        slices[label]={name:old.metrics(test['truth'][mask],v[mask]) for name,v in values.items()}
    rng=np.random.default_rng(96013);episodes=np.unique(ids);bootstrap=[]
    episode_rows=[np.flatnonzero(ids==ep) for ep in episodes]
    for _ in range(1000):
        ix=np.concatenate([episode_rows[k] for k in rng.integers(0,len(episodes),len(episodes))])
        bootstrap.append(old.metrics(test['truth'][ix],values['xgboost'][ix])['F1']-old.metrics(test['truth'][ix],values[reference][ix])['F1'])
    booster=model.get_booster();gain=booster.get_score(importance_type='gain')
    shap=booster.predict(xgb.DMatrix(inputs['test']),pred_contribs=True)
    old.write(out/'feature-importance.json',dict(gain={names[int(k[1:])]:v for k,v in gain.items()},
        mean_abs_treeshap={name:float(v) for name,v in zip(names,np.abs(shap[:,:-1]).mean(0))},claim='associations_not_causal'))
    result=dict(metrics=metrics,reference_rule=reference,threshold=threshold,gates=gates,
        decision='FRESH_UE_TREE_DEVELOPMENT_GAIN' if all(gates.values()) else 'FRESH_UE_TREE_GATE_NOT_MET',
        slices=slices,paired_episode_bootstrap_F1_difference_95pct=np.quantile(bootstrap,[.025,.975]).tolist(),
        missed_reference_events=lost,max_added_delay_s=max(delays,default=0),events=events,
        test_truth_positive=int(test['truth'].sum()),test_frames=len(ids),test_episodes=len(episodes),seconds=time.perf_counter()-started)
    old.write(out/'result.json',result)
    old.write(out/'receipt.json',dict(status='PASS',hashes={str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()},
        training_configurations=1,backend_fit_count=2,test_driven_retries=0,source='fresh_UE_geometry_with_hypothetical_Radar'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','slices')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.capture,args.output)
