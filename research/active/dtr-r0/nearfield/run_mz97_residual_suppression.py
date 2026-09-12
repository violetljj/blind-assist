"""One fixed A-conditioned false-alert fit on fresh UE scene splits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import mz97_residual_suppression as residual

common=residual.common
features=common.features
old=common.old
previous=common.previous
ROOT=common.ROOT
TASK=ROOT/'artifacts.local/work/mz97-residual-suppression-20260912'
SPEC_SHA='c236a8d102f2cd7fd3e3219746cae4cc4ba14538d767c2a436760c6f2e933411'


def materialize(capture,out):
    assert old.sha(capture/'spec.json')==SPEC_SHA
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['frames']==5120
    for name,digest in receipt['hashes'].items():assert old.sha(capture/name)==digest
    spec=json.loads((capture/'spec.json').read_text());splits={s['episode_id']:s['split'] for s in spec['scenes']}
    assert spec['seed']==97013 and len(splits)==len(spec['scenes'])==128
    assert {s:list(splits.values()).count(s) for s in ('train','validation','test')}==dict(train=80,validation=16,test=32)
    rawrows=[json.loads(line) for line in (capture/'raw.jsonl').read_text().splitlines()]
    assert len(rawrows)==5120 and all(set(r)==features.a.contract.RAW_KEYS for r in rawrows)
    assert set(r['episode_id'] for r in rawrows)==set(splits)
    for start in range(0,5120,40):
        block=rawrows[start:start+40]
        assert len({r['episode_id'] for r in block})==1
        assert sum(r['episode_id']==block[0]['episode_id'] for r in rawrows)==40
        assert np.allclose([r['time_s'] for r in block],np.arange(40)*.1)
    # Mechanical label routing only; no held-out outcome inspection or selection.
    labelrows=[json.loads(line) for line in (capture/'evaluator.jsonl').read_text().splitlines()]
    assert [(r['episode_id'],r['time_s']) for r in rawrows]==[(r['episode_id'],r['time_s']) for r in labelrows]
    boolean={'tof_packet_received','radar_packet_received','radar_valid','imu_valid'}
    for split,size in (('train',3200),('validation',640),('test',1280)):
        ix=[i for i,r in enumerate(rawrows) if splits[r['episode_id']]==split];assert len(ix)==size
        raw={k:np.asarray([rawrows[i][k] for i in ix],dtype=str if k=='episode_id' else bool if k in boolean else np.uint8 if k=='tof_status' else float) for k in rawrows[0]}
        labels={k:np.asarray([labelrows[i][k] for i in ix],bool) for k in ('truth','any_direct_truth','future_only_truth')}
        target=out/split;target.mkdir()
        np.savez_compressed(target/'raw.npz',**raw);np.savez_compressed(target/'labels.npz',**labels)
    old.write(out/'data-receipt.json',dict(source_sha256=SPEC_SHA,capture_receipt_sha256=old.sha(capture/'receipt.json'),
        split_sizes=dict(train=3200,validation=640,test=1280),test_labels_sha256=old.sha(out/'test/labels.npz')))


def run(capture,out):
    import xgboost as xgb
    if out.exists() or out.resolve().parent!=TASK.resolve():raise ValueError('New canonical output required')
    out.mkdir(parents=True);started=time.perf_counter()
    for name in ('mz97_residual_suppression.py','run_mz97_residual_suppression.py','MZ97_RESIDUAL_SUPPRESSION_20260912.md'):
        shutil.copyfile(previous.HERE/name,out/name)
    materialize(capture,out)
    inputs={};rules={};observations={}
    for split in ('train','validation','test'):
        obs,X,names,values=features.build(dict(np.load(out/split/'raw.npz')))
        inputs[split]=X;rules[split]=values;observations[split]=obs
    mask=rules['train']['horizon_A'];truth=np.load(out/'train/labels.npz')['truth']
    X=inputs['train'][mask];y=~truth[mask]
    assert len(np.unique(y))==2, 'Training candidate classes insufficient; no substitute fit permitted'
    kwargs=dict(n_estimators=300,max_depth=3,learning_rate=.05,min_child_weight=5,subsample=.8,colsample_bytree=.8,
        reg_lambda=5,random_state=97013,tree_method='hist',objective='binary:logistic',n_jobs=4)
    models={}
    def fit(device):
        model=xgb.XGBClassifier(**kwargs,device=device);model.fit(X,y);models[device]=model;return model
    def observe(model):
        device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        return previous.DeviceObservation('cuda' if device.startswith('cuda') else 'cpu',device,'xgboost')
    backend=previous.select_backend('batch-tensor',cpu=previous.BackendCandidate('xgboost-cpu','cpu',lambda:fit('cpu'),observe),
        gpu=previous.BackendCandidate('xgboost-cuda','cuda',lambda:fit('cuda'),observe),warmups=0,repeats=1,
        record_path=out/'backend.json',capabilities={'xgboost':xgb.__version__,'work':'equivalent fixed A-positive candidate fits; no predictive device selection'})
    model=models['cuda' if backend['selected_device_type']=='cuda' else 'cpu'];model.save_model(out/'model.ubj')
    probabilities={s:model.predict_proba(matrix)[:,1] for s,matrix in inputs.items()}
    val=dict(np.load(out/'validation/labels.npz'));ids=observations['validation']['episode_id']
    selection=residual.select(ids,val,rules['validation']['horizon_A'],probabilities['validation'])
    selection.update(model_config=kwargs,features=names,train_candidates=len(y),train_false_candidates=int(y.sum()),
        train_total_frames=len(mask),test_labels_used_for_selection=False)
    old.write(out/'selection.json',selection)
    for split in ('validation','test'):
        values=rules[split]
        values['residual']=residual.apply(values['horizon_A'],probabilities[split],selection['threshold'])
        assert not (values['residual']&~values['horizon_A']).any()
        rejected=values['horizon_A']&~values['residual']
        policy_unknown=(~observations[split]['tof_known']&~values['residual'])|rejected
        np.savez_compressed(out/split/'predictions.npz',**values,false_alert_score=probabilities[split],
            residual_abstention=rejected,residual_policy_UNKNOWN=policy_unknown)
    old.write(out/'test-prediction-seal.json',dict(status='SEALED_BEFORE_TEST_EVALUATION',
        model_sha256=old.sha(out/'model.ubj'),selection_sha256=old.sha(out/'selection.json'),prediction_sha256=old.sha(out/'test/predictions.npz'),
        implementation_sha256={str(Path(m.__file__).resolve().relative_to(ROOT.resolve())):old.sha(Path(m.__file__))
            for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file()
            and Path(m.__file__).resolve().is_relative_to(ROOT.resolve()) and 'artifacts.local' not in Path(m.__file__).parts}))
    # All selection and predictions are now frozen. Test scoring starts here.
    labels=dict(np.load(out/'test/labels.npz'));obs=observations['test'];ids=obs['episode_id'];values=rules['test']
    metrics,events=common.score(ids,labels,values)
    for name,v in values.items():metrics[name].update(UNKNOWN=int((~obs['tof_known']&~v).sum()),positive_UNKNOWN=int((labels['truth']&~obs['tof_known']&~v).sum()))
    rejected=values['horizon_A']&~values['residual'];policy_unknown=(~obs['tof_known']&~values['residual'])|rejected
    metrics['residual'].update(abstentions=int(rejected.sum()),policy_UNKNOWN=int(policy_unknown.sum()),
        positive_policy_UNKNOWN=int((labels['truth']&policy_unknown).sum()))
    pairs={name:residual.compare(ids,labels,values[ref],values[candidate]) for name,ref,candidate in (
        ('A_vs_matched_hold','matched_hold','horizon_A'),('AB_vs_A','horizon_A','A_plus_B'),('residual_vs_A','horizon_A','residual'))}
    result_pair=pairs['residual_vs_A'];gates={**result_pair['gates'],
        'active':not selection['disabled'],'strictly_fewer_FP':metrics['residual']['FP']<metrics['horizon_A']['FP'],
        'strictly_higher_F1':metrics['residual']['F1']>metrics['horizon_A']['F1']}
    spec=json.loads((capture/'spec.json').read_text());ghost_ids={s['episode_id'] for s in spec['scenes'] if s.get('radar_ghost') is not None}
    ghost=np.array([str(ep) in ghost_ids for ep in ids]);slices={}
    for label,mask in (('ghost_present',ghost),('ghost_absent',~ghost)):
        slices[label]={name:old.metrics(labels['truth'][mask],v[mask]) for name,v in values.items()}
    rng=np.random.default_rng(97013);episode_rows=[np.flatnonzero(ids==ep) for ep in np.unique(ids)];draws={k:[] for k in pairs}
    comparisons=(('A_vs_matched_hold','matched_hold','horizon_A'),('AB_vs_A','horizon_A','A_plus_B'),('residual_vs_A','horizon_A','residual'))
    for _ in range(1000):
        ix=np.concatenate([episode_rows[k] for k in rng.integers(0,len(episode_rows),len(episode_rows))])
        for name,ref,candidate in comparisons:draws[name].append(old.metrics(labels['truth'][ix],values[candidate][ix])['F1']-old.metrics(labels['truth'][ix],values[ref][ix])['F1'])
    old.write(out/'feature-importance.json',dict(gain={names[int(k[1:])]:v for k,v in model.get_booster().get_score(importance_type='gain').items()},claim='descriptive_associations_not_causal'))
    result=dict(decision='RESIDUAL_DEVELOPMENT_GAIN' if all(gates.values()) else 'RESIDUAL_DISABLED_NO_VALIDATION_GAIN' if selection['disabled'] else 'RESIDUAL_GATE_NOT_MET',
        metrics=metrics,pairs=pairs,gates=gates,disabled=selection['disabled'],threshold=selection['threshold'],
        validation_selected=selection['selected'],slices=slices,events=events,
        bootstrap_F1_difference_95pct={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()},
        test_frames=len(ids),test_episodes=len(episode_rows),test_positive=int(labels['truth'].sum()),seconds=time.perf_counter()-started)
    old.write(out/'result.json',result)
    old.write(out/'receipt.json',dict(status='PASS',hashes={str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()},
        training_configurations=1,backend_fit_count=2,test_driven_retries=0,source='fresh_UE_geometry_with_hypothetical_Radar'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','slices','events','validation_selected')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.capture,args.output)
