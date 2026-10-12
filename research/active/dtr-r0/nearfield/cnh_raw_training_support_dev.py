"""Fixed current-raw input by matched training-support interaction probe."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cnh_direction_information_dev as I
import cnh_querywise_calibration_dev as Q
import cnh_training_support_dev as B

R,P=I.R,I.P
OUT=R.WORK/'cnh-raw-training-support-dev-20261007'
RAW=R.WORK/'cnh-raw-radial-information-dev-20261007'
ARMS=('raw_source_repeat','raw_hb_aug')
BASE_MAP={'original_center':'original_center','pair_only':'pair_only',
          'pair_source_repeat':'source_repeat','pair_hb_aug':'hb_aug'}
METHODS=(*BASE_MAP,'raw_current',*ARMS)


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def assemble_current(pair,payload,unit,config,anchor_id=None,tag=None):
    for key,value in [('unit',unit),('config',config),('anchor_id',anchor_id),('tag',tag)]:
        if value is not None:np.testing.assert_array_equal(payload[key],value,err_msg=key+' order')
    current=np.asarray(payload['current']);pair=np.asarray(pair)
    if pair.shape!=(len(unit),13,2) or current.shape!=(len(unit),13,136):raise ValueError('Expected pair2+current136')
    if not np.isfinite(pair).all() or not np.isfinite(current).all():raise ValueError('Current inputs must be finite')
    # Exactly the same concatenate as the frozen raw probe, preserving pair dtype.
    return np.concatenate((pair,current),axis=-1)


def load_current(o,h,hrows):
    with np.load(RAW/'features.npz') as z:x=assemble_current(o['pair'],z,o['unit'],o['config'])
    with np.load(RAW/'hb_features.npz') as z:
        hx=assemble_current(h['pair'],z,h['unit'],h['config'],np.array([v['anchor_id'] for v in hrows]),h['tag'])
    return x,hx


def freeze():
    import sklearn
    parent=R.read(B.OUT/'PLAN.json');rp=R.read(RAW/'PLAN.json');assert parent['folds']==rp['folds']
    hashes=dict(parent['hashes'])
    paths=[Path(__file__),Path(B.__file__),Path(__file__).with_name('cnh_raw_radial_information_dev.py'),
        Path(__file__).with_name('cnh_raw_training_support_contrasts_dev.py'),Path(__file__).with_name('cnh_training_support_contrasts_dev.py'),
        B.OUT/'PLAN.json',B.OUT/'result.json',B.OUT/'ledger.npz',B.OUT/'hb_ledger.npz',B.OUT/'augmentation_inputs.npz',B.OUT/'augmentation_rows.json',
        RAW/'PLAN.json',RAW/'result.json',RAW/'ledger.npz',RAW/'hb_ledger.npz',RAW/'features.npz',RAW/'hb_features.npz',RAW/'features_metadata.json']
    paths.extend(B.OUT/f'fold{f}_training_schedule.npz' for f in range(3))
    paths.extend(RAW/'models'/f'fold{f}_current.joblib' for f in range(3))
    for path in paths:hashes[str(path.relative_to(R.ROOT))]=R.sha(path)
    OUT.mkdir(parents=True,exist_ok=True)
    save('PLAN.json',dict(phase='POSTHOC EXPLORE; original and HB Development consumed, HB structure deliberately exposed in training',
        authorization='Continuous algorithm exploration; raw input by training-support interaction',
        budget_cpu_analysis_wall_seconds=300,no_gpu=True,threads=4,compute_reason='TASK_NOT_GPU_SUITABLE: small fixed HGB has no GPU backend',
        one_fit_and_evaluation_run=True,arms=list(ARMS),methods=list(METHODS),model=parent['model'],seed='I.SEED+fold',folds=parent['folds'],
        input='Exactly rawprobe current: pair2+current136, identical dtype/order; only observation features, no labels/identity/time inputs',
        training='Directly reuse B fold_training_schedule ri/ti/y/weight/selected_anchor_index and original source indices; source_repeat adds original raw source once; hb_aug adds true HB current raw. No weight recomputation or renormalization.',
        factorial='Existing pair_source_repeat/pair_hb_aug and new raw_source_repeat/raw_hb_aug form a fixed2x2 input/support comparison. raw_current and pair_only retain unaugmented references.',
        calibration='Only new arms fit; thresholds only original calibration controls2.5/5%. All5 existing methods inherit scores and thresholds exactly. Eval-control matched cap is descriptive.',
        hb='Each anchor only original OOF model and source-cal threshold; unit absent from its train/cal. HB structures from other training units were exposed.216variants,72anchors,36event/36control per structure; no fresh-validation claim.',
        metrics='Timely excluding first2 outputs, actualFA[2:12], warmupFA, any-before timely; paired single-H/B-any-timely but HB-miss. Original5760/369event/1255control.',
        decision='Does the HB augmentation effect survive current raw input, while retaining original-domain detection and actualFA? Preserve all counts/rates; thresholds do not promise equalFA across structures.',
        limitations=['Posthoc consumed synthetic Development, not confirmation','Only17/19/22 added train HB positives; fixed tree capacity and no tuning',
                    '2x2 compares fixed learned recipes; unequal actualFA and finite-data uncertainty remain','Current raw gain need not identify a physical mechanism'],
        sklearn=sklearn.__version__,numpy=np.__version__,hashes=hashes))
    print('FROZEN raw training support CPU300s, two fixed new arms',flush=True)


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=R.read(OUT/'PLAN.json')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('One run only; preserve payload')
    save('RUN_STARTED.json',dict(unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_cpu_analysis_wall_seconds']:raise TimeoutError('300s CPU budget reached')
    try:
        for path,expected in plan['hashes'].items():assert R.sha(R.ROOT/path)==expected,path
        o,h,hrows,source,augrows=B.load_inputs();x,hx=load_current(o,h,hrows)
        source_hb_indices=np.array([3*i+('H','B','HB').index(a['source_variant']) for i,a in enumerate(augrows)])
        np.testing.assert_array_equal(x[source],hx[source_hb_indices])
        uid,cfg,contact,control,deadline,cq=[o[k] for k in ('unit','config','contact','control','deadline','contact_query')];n=len(uid)
        assert n==5760 and contact.sum()==369 and control.sum()==1255 and len(hrows)==216
        with np.load(B.OUT/'augmentation_inputs.npz') as z:
            np.testing.assert_array_equal(source,z['source_row_index']);np.testing.assert_array_equal(o['pair'][source],z['source_repeat'])
            np.testing.assert_array_equal(h['pair'][2::3],z['hb_aug'])
        inherited={};hinherited={};oldthreshold={};oldfitfa={}
        for directory,mapping in [(B.OUT,BASE_MAP),(RAW,{'raw_current':'current'})]:
            with np.load(directory/'ledger.npz') as z:
                for k in ('unit','config','fold','contact','control','deadline'):np.testing.assert_array_equal(z[k],o[k])
                for name,old in mapping.items():inherited[name]=z['score/'+old]
            with np.load(directory/'hb_ledger.npz') as z:
                for k in ('unit','config','fold','contact','control','deadline','tag'):np.testing.assert_array_equal(z[k],h[k])
                for name,old in mapping.items():hinherited[name]=z['score/'+old]
            for record in R.read(directory/'result.json')['fold_records']:
                mode,cap,old=record['key'].split('/')
                for name,original_name in mapping.items():
                    if old==original_name:
                        oldthreshold[record['fold'],f'{mode}/{cap}/{name}']=float(record['threshold'])
                        oldfitfa[record['fold'],f'{mode}/{cap}/{name}']=record['fit_false_alarm']
        scores={k:v.copy() for k,v in inherited.items()};scores.update({k:np.zeros((n,13)) for k in ARMS})
        alarms={};models={};thresholds={};records=[];training=[];schedules=[];roleledger={};foldidx=np.full(n,-1,int)
        (OUT/'models').mkdir(exist_ok=True)
        with threadpool_limits(limits=4):
            for fold,roles in enumerate(plan['folds']):
                check();masks={k:np.isin(uid,v) for k,v in roles.items()};tr,ca,te=[masks[k] for k in ('train','calibration','evaluation')]
                assert not ((tr&ca)|(tr&te)|(ca&te)).any() and (foldidx[te]==-1).all();foldidx[te]=fold
                roleledger.update({f'fold{fold}/{k}':v for k,v in masks.items()})
                schedule_path=B.OUT/f'fold{fold}_training_schedule.npz'
                with np.load(schedule_path) as z:
                    selected=z['selected_anchor_index'];ri=z['row_index'];ti=z['time_index'];y=z['label'];w=z['weight']
                    source_rows=z['original_source_row_index'];anchor_indices=z['anchor_index']
                assert tr[source_rows].all() and np.isin(np.array([a['unit'] for a in augrows])[selected],roles['train']).all()
                np.testing.assert_array_equal(np.concatenate((np.arange(n),source[selected]))[ri],source_rows)
                np.testing.assert_array_equal(np.concatenate((np.full(n,-1,int),selected))[ri],anchor_indices)
                assert int(contact[source[selected]].sum())==[17,19,22][fold]
                schedules.append(dict(fold=fold,path=str(schedule_path.relative_to(R.ROOT)),sha256=R.sha(schedule_path),
                    selected_anchor_index=selected.tolist(),source_row_index=source[selected].tolist(),samples=len(y),positive_mass=float(w[y==1].sum()),negative_mass=float(w[y==0].sum())))
                oldmodel=joblib.load(RAW/'models'/f'fold{fold}_current.joblib')
                np.testing.assert_array_equal(oldmodel.predict_proba(x[te].reshape(-1,138))[:,1].reshape(te.sum(),13),inherited['raw_current'][te])
                hm=h['fold']==fold
                assert np.isin(h['unit'][hm],roles['evaluation']).all() and not np.isin(h['unit'][hm],roles['train']+roles['calibration']).any()
                np.testing.assert_array_equal(oldmodel.predict_proba(hx[hm].reshape(-1,138))[:,1].reshape(hm.sum(),13),hinherited['raw_current'][hm])
                pred={k:v for k,v in inherited.items()}
                for name,added in [('raw_source_repeat',x[source]),('raw_hb_aug',hx[2::3])]:
                    check();t=time.monotonic();joint=np.concatenate((x,added[selected]),axis=0)
                    model=HistGradientBoostingClassifier(**{k:v for k,v in plan['model'].items() if k!='name'},random_state=I.SEED+fold)
                    model.fit(joint[ri,ti],y,sample_weight=w);models[fold,name]=model
                    pred[name]=model.predict_proba(x.reshape(-1,138))[:,1].reshape(n,13)
                    joblib.dump(model,OUT/'models'/f'fold{fold}_{name}.joblib')
                    training.append(dict(fold=fold,name=name,dimensions=138,schedule_sha256=R.sha(schedule_path),added_anchors=len(selected),
                        positive_events=int(y.sum()),samples=len(y),positive_mass=float(w[y==1].sum()),negative_mass=float(w[y==0].sum()),seconds=time.monotonic()-t))
                    print('FIT',fold,name,round(training[-1]['seconds'],2),flush=True)
                for name,score in pred.items():
                    if name in ARMS:
                        scores[name][te]=score[te]
                        for role,mask in [('calibration',ca),('evaluation',te)]:
                            np.savez_compressed(OUT/f'fold{fold}_{name}_{role}.npz',row_index=np.flatnonzero(mask),unit=uid[mask],config=cfg[mask],score=score[mask])
                    for cap in (.025,.05):
                        for mode,fitmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                            key=f'{mode}/{cap:.3f}/{name}'
                            theta=P.threshold_at_cap(score[fitmask&control,2:12],cap) if name in ARMS else oldthreshold[fold,key]
                            # Old scores are OOF; their cal-fold predictions are deliberately not reinterpreted as current-fold cal scores.
                            fitfa=float((score[fitmask&control,2:12]>=theta).mean()) if name in ARMS else oldfitfa[fold,key]
                            assert fitfa<=cap+1e-12
                            al=score>=theta;alarms.setdefault(key,np.zeros((n,13),bool))[te]=al[te];thresholds[fold,key]=theta
                            records.append(dict(fold=fold,key=key,threshold=theta,fit_false_alarm=fitfa,inherited=name not in ARMS,**Q.metrics(al,contact,control,deadline,te)))
        np.testing.assert_array_equal(foldidx,o['fold']);check()
        groups={'all':np.ones(n,bool),'head_contact':cq[:,0]|control,'body_contact':cq[:,1]|control,'family_none':cfg<20,'family_corner':cfg>=20}
        aggregate={k:{g:Q.metrics(v,contact,control,deadline,m) for g,m in groups.items()} for k,v in alarms.items()}
        hscores={k:v.copy() for k,v in hinherited.items()};halarms={};hmetrics={};paired={}
        with threadpool_limits(limits=4):
            for name in ARMS:
                hscores[name]=np.zeros(hx.shape[:2])
                for fold in range(3):
                    mask=h['fold']==fold;hscores[name][mask]=models[fold,name].predict_proba(hx[mask].reshape(-1,138))[:,1].reshape(mask.sum(),13)
        for name,score in hscores.items():
            for cap in (.025,.05):
                key=f'calibrated/{cap:.3f}/{name}';theta=np.array([thresholds[f,key] for f in h['fold']]);al=score>=theta[:,None];halarms[key]=al
                hmetrics[key]={tag:Q.metrics(al,h['contact'],h['control'],h['deadline'],h['tag']==tag) for tag in ('H','B','HB')}
                timely,post=Q.event_states(al,h['contact'],h['deadline'])
                paired[key]=dict(any_before=B.paired_structure_counts(timely,h['contact'],h['tag']),excluding_warmup=B.paired_structure_counts(post,h['contact'],h['tag']))
        check();save('shared_training_schedules.json',schedules)
        np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=foldidx,contact=contact,control=control,contact_query=cq,deadline=deadline,
            **{'role/'+k:v for k,v in roleledger.items()},**{'score/'+k:v for k,v in scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:Q.event_states(v,contact,deadline)[0] for k,v in alarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,contact,deadline)[1] for k,v in alarms.items()})
        np.savez_compressed(OUT/'hb_ledger.npz',**{k:h[k] for k in ('unit','config','fold','contact','control','deadline','tag')},anchor_id=np.array([r['anchor_id'] for r in hrows]),
            **{'score/'+k:v for k,v in hscores.items()},**{'alarm/'+k:v for k,v in halarms.items()},
            **{'timely/'+k:Q.event_states(v,h['contact'],h['deadline'])[0] for k,v in halarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,h['contact'],h['deadline'])[1] for k,v in halarms.items()})
        save('result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,sequences=n,units=len(np.unique(uid)),events=int(contact.sum()),controls=int(control.sum()),
            training=training,shared_schedules=schedules,fold_records=records,metrics=aggregate,hb_metrics=hmetrics,hb_paired=paired,
            plan_sha256=R.sha(OUT/'PLAN.json'),ledger_sha256=R.sha(OUT/'ledger.npz'),hb_ledger_sha256=R.sha(OUT/'hb_ledger.npz')))
        save('TERMINAL.json',dict(status='complete',seconds=time.monotonic()-start))
        for k,v in aggregate.items():
            if k.startswith('calibrated/0.025/'):print('ORIGINAL',k,v['all'],flush=True)
        for k,v in hmetrics.items():
            if k.startswith('calibrated/0.025/'):print('HB',k,v,paired[k],flush=True)
        print('COMPLETE',round(time.monotonic()-start,2),'seconds',flush=True)
    except BaseException as exc:
        save('TERMINAL.json',dict(status='failed_or_budget_stop',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
