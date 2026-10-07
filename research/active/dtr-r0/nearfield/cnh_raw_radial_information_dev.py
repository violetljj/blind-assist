"""Fixed CPU raw-radial information probe on consumed Development inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import cnh_direction_information_dev as I
import cnh_querywise_calibration_dev as Q

R,E,P=I.R,I.E,I.P
OUT=R.WORK/'cnh-raw-radial-information-dev-20261007'
BASE=Q.BASE
HB=Q.HB
ARMS=('current','current_support','temporal')
METHODS=('original_center','pair_only',*ARMS)
DIMS={'current':136,'support':24,'temporal_delta':384}


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def assemble_features(pair,payload):
    """Use explicit observation-only keys, never metadata or evaluator labels."""
    pair=np.asarray(pair)
    if pair.ndim!=3 or pair.shape[1:]!=(13,2) or not np.isfinite(pair).all():
        raise ValueError('Finite N by13 by2 paired scores required')
    arrays={}
    for key,dim in DIMS.items():
        x=np.asarray(payload[key])
        valid=not np.isinf(x).any() if key=='temporal_delta' else np.isfinite(x).all()
        if x.shape!=(*pair.shape[:2],dim) or not valid:
            raise ValueError(f'Invalid finite feature schema: {key}')
        arrays[key]=x
    current=np.concatenate((pair,arrays['current']),axis=-1)
    support=np.concatenate((current,arrays['support']),axis=-1)
    return dict(current=current,current_support=support,
                temporal=np.concatenate((support,arrays['temporal_delta']),axis=-1))


def check_alignment(payload,unit,config,anchor_id=None,tag=None):
    for key,expected in [('unit',unit),('config',config),('anchor_id',anchor_id),('tag',tag)]:
        if expected is not None:np.testing.assert_array_equal(payload[key],expected,err_msg=key+' row order')


def freeze():
    import sklearn
    import cnh_raw_radial_features_dev as F
    parent=R.read(BASE/'PLAN.json');hashes=dict(parent['hashes'])
    paths=[Path(__file__),Path(F.__file__),Path(I.__file__),Path(Q.__file__),
           Path(__file__).with_name('cnh_raw_radial_contrasts_dev.py'),
           BASE/'PLAN.json',BASE/'result.json',BASE/'ledger.npz',Q.OUT/'PLAN.json',Q.OUT/'ledger.npz',
           HB/'PLAN.json',HB/'geometry_manifest.json',HB/'score_ledger.npz']
    paths.extend(BASE/'models'/f'in_domain_fold{f}_pair_only.joblib' for f in range(3))
    for p in paths:hashes[str(p.relative_to(R.ROOT))]=R.sha(p)
    OUT.mkdir(parents=True,exist_ok=True)
    save('PLAN.json',dict(phase='POSTHOC EXPLORE; original and HB Development already consumed',
        authorization='Continuous algorithm exploration; bounded raw radial information mechanism probe',
        budget_extract_wall_seconds=900,budget_fit_eval_wall_seconds=900,no_gpu=True,threads=4,
        compute_reason='TASK_NOT_GPU_SUITABLE: small fixed HGB has no GPU backend; archive IO and tiny8x8 warps dominate extraction',
        one_fit_and_evaluation_run=True,folds=parent['folds'],model=parent['model'],seed='I.SEED+fold',
        arms=list(ARMS),feature_schema=dict(current=136,support=24,temporal_delta=384,
            final_current=138,final_current_support=162,final_temporal=546),
        extractor_schema=F.SCHEMA,
        input_boundary='New raw features contain only past/current observation-derived content and causal pose metadata, with no public query input. Existing M3 pair retains its upstream public-query conditioning and observation history. Labels/deadline/unit/config/family/time/frame are never explicit model inputs.',
        primary_comparisons=['temporal versus current_support isolates added radial history content conditional on visibility/rotation metadata',
                             'current versus pair_only tests access to current radial information beyond compressed M3 outputs'],
        supervision='Parent training_rows: one last-deadline positive per event; strict both-query-clear controls matched to positive deadline histogram and total mass; all other rows ignored.',
        calibration='Original train only fits; original calibration controls select each model threshold at2.5/5% cap. Max/pair_only inherit old thresholds. Eval-control-matched thresholds are descriptive frontiers only.',
        metrics='Primary timely excluding first2 outputs and actualFA[2:12]; any-before timely and warmupFA[:2] retained. Original5760 sequences/369 events/1255 controls; no dropped groups.',
        hb='216 consumed check-only H/B/HB sequences,72 anchors39units; each structure36 events36 controls. Use original OOF model and source-cal threshold, never HB fit/calibrate.',
        decision='Only stable gain across folds, main comparisons and actualFA with HB capability retained warrants a later validation proposal; no automatic baseline promotion.',
        weaknesses=['Fixed shallow learners give a bounded information probe, not an information-theoretic ceiling',
                    'Temporal gain supports added past observation content, not uniquely approaching motion; denoising or memory are alternative mechanisms',
                    'Lag5 is unavailable at native3/4, the first2 outputs; primary no-warmup timely begins native5 to remove this startup advantage',
                    'Current+history dimensions differ despite identical tree capacity',
                    'Synthetic family/template shortcuts and consumed Development remain possible',
                    'HB is an already observed structural counterexample, not independent confirmation'],
        extraction_provenance='Frozen extractor and input parents before extraction; generated feature hashes and extractor receipt recorded before fitting. Large observation sources may be hashed once in extraction receipts.',
        sklearn=sklearn.__version__,numpy=np.__version__,hashes=hashes))
    print('FROZEN raw radial CPU probe; extraction900s and fitting/evaluation900s',flush=True)


def load_hb(plan):
    rows=[];pair=[];scores={k:[] for k in ('original_center','pair_only')}
    with np.load(HB/'score_ledger.npz') as z:
        for a in R.read(HB/'geometry_manifest.json')['anchors']:
            roles=plan['folds'][a['fold']]
            assert a['unit'] in roles['evaluation'] and a['unit'] not in roles['train']+roles['calibration']
            for tag in ('H','B','HB'):
                v=a['variants'][tag];deadline=-1 if v['deadline_index'] is None else int(v['deadline_index'])
                if v['contact']:assert 0<=deadline<13
                rows.append(dict(anchor_id=a['anchor_id'],unit=a['unit'],config=a['config'],fold=a['fold'],tag=tag,
                    contact=bool(v['contact']),control=bool(v['control']),contact_query=v['contact_query'],deadline=deadline))
                prefix=a['anchor_id']+'/'+tag+'/'
                pair.append(z[prefix+'x'])
                for name in scores:scores[name].append(z[prefix+name])
    return rows,np.asarray(pair),{k:np.asarray(v) for k,v in scores.items()}


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=R.read(OUT/'PLAN.json')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('One run only; preserve previous payload')
    save('RUN_STARTED.json',dict(unix=time.time(),budget_seconds=900))
    def check():
        if time.monotonic()-start>plan['budget_fit_eval_wall_seconds']:raise TimeoutError('900s fit/evaluation budget reached')
    try:
        for path,h in plan['hashes'].items():assert R.sha(R.ROOT/path)==h,path
        feature_paths=[OUT/'features.npz',OUT/'hb_features.npz',OUT/'features_metadata.json']
        provenance={p.name:R.sha(p) for p in feature_paths}
        receipt=R.read(OUT/'features_metadata.json')
        assert receipt['status']=='complete'
        assert receipt['plan_sha256']==R.sha(OUT/'PLAN.json')
        save('fit_input_receipt.json',dict(generated_feature_hashes=provenance,extraction=receipt))
        with np.load(Q.OUT/'ledger.npz') as z:
            uid=z['unit'];cfg=z['config'];contact=z['contact'];control=z['control'];cq=z['contact_query'];deadline=z['deadline']
            pair=z['query_score/original_center'];old_pair_oof=z['score/pair_only']
        with np.load(OUT/'features.npz') as z:
            check_alignment(z,uid,cfg);features=assemble_features(pair,z)
        n=len(uid);assert n==5760 and contact.sum()==369 and control.sum()==1255
        old=R.read(BASE/'result.json');oldrecords={(v['fold'],v['key']):v for v in old['fold_records']}
        scores={k:np.zeros((n,13)) for k in METHODS};alarms={};foldidx=np.full(n,-1,int)
        thresholds={};models={};records=[];training=[];roles_ledger={}
        (OUT/'models').mkdir(exist_ok=True)
        with threadpool_limits(limits=4):
            for fold,roles in enumerate(plan['folds']):
                check();masks={k:np.isin(uid,v) for k,v in roles.items()};tr,ca,te=[masks[k] for k in ('train','calibration','evaluation')]
                assert not ((tr&ca)|(tr&te)|(ca&te)).any() and (foldidx[te]==-1).all();foldidx[te]=fold
                roles_ledger.update({f'fold{fold}/{k}':v for k,v in masks.items()})
                ri,ti,y,w=I.training_rows(contact,control,deadline,tr)
                oldmodel=joblib.load(BASE/'models'/f'in_domain_fold{fold}_pair_only.joblib')
                pred={'original_center':pair.max(-1),'pair_only':oldmodel.predict_proba(pair.reshape(-1,2))[:,1].reshape(n,13)}
                np.testing.assert_array_equal(pred['pair_only'][te],old_pair_oof[te])
                for name,x in features.items():
                    check();t=time.monotonic()
                    model=HistGradientBoostingClassifier(**{k:v for k,v in plan['model'].items() if k!='name'},random_state=I.SEED+fold)
                    model.fit(x[ri,ti],y,sample_weight=w);models[fold,name]=model
                    pred[name]=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(n,13)
                    joblib.dump(model,OUT/'models'/f'fold{fold}_{name}.joblib')
                    training.append(dict(fold=fold,name=name,dimensions=x.shape[-1],samples=len(y),positive_events=int(y.sum()),
                        positive_mass=float(w[y==1].sum()),negative_mass=float(w[y==0].sum()),
                        positive_deadline_histogram=np.bincount(ti[y==1],minlength=13).tolist(),seconds=time.monotonic()-t))
                    print('FIT',fold,name,round(training[-1]['seconds'],2),flush=True);check()
                for name,score in pred.items():
                    scores[name][te]=score[te]
                    for role,mask in [('calibration',ca),('evaluation',te)]:
                        np.savez_compressed(OUT/f'fold{fold}_{name}_{role}.npz',row_index=np.flatnonzero(mask),score=score[mask],unit=uid[mask],config=cfg[mask])
                    for cap in (.025,.05):
                        for mode,fitmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                            key=f'{mode}/{cap:.3f}/{name}'
                            theta=float(oldrecords[fold,'in_domain/'+key]['threshold']) if name in ('original_center','pair_only') else P.threshold_at_cap(score[fitmask&control,2:12],cap)
                            al=score>=theta;fitfa=float(al[fitmask&control,2:12].mean());assert fitfa<=cap+1e-12
                            thresholds[fold,key]=theta;alarms.setdefault(key,np.zeros((n,13),bool))[te]=al[te]
                            records.append(dict(fold=fold,key=key,threshold=theta,fit_false_alarm=fitfa,**Q.metrics(al,contact,control,deadline,te)))
        assert (foldidx>=0).all();check()
        groups={'all':np.ones(n,bool),'head_contact':cq[:,0]|control,'body_contact':cq[:,1]|control,
            'both_queries_contact':cq.all(1)|control,'family_none':cfg<20,'family_corner':cfg>=20}
        aggregate={key:{g:Q.metrics(al,contact,control,deadline,mask) for g,mask in groups.items()} for key,al in alarms.items()}
        # HB is opened only after all original-domain fitting/calibration is complete.
        hrows,hpair,hscores=load_hb(plan)
        hunit=np.array([r['unit'] for r in hrows]);hcfg=np.array([r['config'] for r in hrows]);htag=np.array([r['tag'] for r in hrows])
        hanchor=np.array([r['anchor_id'] for r in hrows]);hfold=np.array([r['fold'] for r in hrows])
        hc=np.array([r['contact'] for r in hrows]);hn=np.array([r['control'] for r in hrows]);hd=np.array([r['deadline'] for r in hrows])
        with np.load(OUT/'hb_features.npz') as z:
            check_alignment(z,hunit,hcfg,hanchor,htag);hfeatures=assemble_features(hpair,z)
        with threadpool_limits(limits=4):
            for name,x in hfeatures.items():
                hscores[name]=np.zeros(x.shape[:2])
                for fold in range(3):
                    mask=hfold==fold;hscores[name][mask]=models[fold,name].predict_proba(x[mask].reshape(-1,x.shape[-1]))[:,1].reshape(mask.sum(),13)
        halarms={};hmetrics={}
        for name,score in hscores.items():
            for cap in (.025,.05):
                key=f'calibrated/{cap:.3f}/{name}';theta=np.array([thresholds[f,key] for f in hfold])
                al=score>=theta[:,None];halarms[key]=al
                hmetrics[key]={tag:Q.metrics(al,hc,hn,hd,htag==tag) for tag in ('H','B','HB')}
        check()
        np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=foldidx,contact=contact,control=control,contact_query=cq,deadline=deadline,
            **{'role/'+k:v for k,v in roles_ledger.items()},**{'score/'+k:v for k,v in scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:Q.event_states(v,contact,deadline)[0] for k,v in alarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,contact,deadline)[1] for k,v in alarms.items()})
        np.savez_compressed(OUT/'hb_ledger.npz',unit=hunit,config=hcfg,fold=hfold,anchor_id=hanchor,tag=htag,contact=hc,control=hn,deadline=hd,
            **{'score/'+k:v for k,v in hscores.items()},**{'alarm/'+k:v for k,v in halarms.items()},
            **{'timely/'+k:Q.event_states(v,hc,hd)[0] for k,v in halarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,hc,hd)[1] for k,v in halarms.items()})
        save('hb_rows.json',hrows)
        save('result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,units=len(np.unique(uid)),sequences=n,
            events=int(contact.sum()),controls=int(control.sum()),training=training,fold_records=records,metrics=aggregate,hb_metrics=hmetrics,
            plan_sha256=R.sha(OUT/'PLAN.json'),feature_provenance=provenance,ledger_sha256=R.sha(OUT/'ledger.npz'),hb_ledger_sha256=R.sha(OUT/'hb_ledger.npz')))
        save('TERMINAL.json',dict(status='complete',seconds=time.monotonic()-start))
        for key,v in aggregate.items():
            if key.startswith('calibrated/0.025/'):print('ORIGINAL',key,v['all'],flush=True)
        for key,v in hmetrics.items():
            if key.startswith('calibrated/0.025/'):print('HB',key,v,flush=True)
        print('COMPLETE',round(time.monotonic()-start,2),'seconds',flush=True)
    except BaseException as exc:
        save('TERMINAL.json',dict(status='failed_or_budget_stop',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
