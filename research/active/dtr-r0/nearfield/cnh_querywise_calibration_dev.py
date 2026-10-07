"""Fixed query/event isotonic recipes, consumed Development; CPU only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import cnh_direction_information_dev as I

R,E,P=I.R,I.E,I.P
OUT=R.WORK/'cnh-querywise-calibration-dev-20261007'
BASE=R.WORK/'cnh-center-readout-ablation-dev-20261007'
HB=R.WORK/'cnh-double-height-dev-20261007'
ARMS=('query_isotonic','event_isotonic')
METHODS=('original_center','pair_only',*ARMS)


def supervision_masks(arm,q,contact_query,contact,control):
    cq=np.asarray(contact_query,bool);contact=np.asarray(contact,bool);control=np.asarray(control,bool)
    if cq.shape!=(len(contact),2) or control.shape!=contact.shape or q not in (0,1):
        raise ValueError('N by2 contact_query and aligned contact/control masks required')
    if not np.array_equal(cq.any(1),contact):raise ValueError('Event contact must equal either-query contact')
    if arm not in ARMS:raise ValueError('Unknown supervision recipe')
    positive=cq[:,q].copy() if arm=='query_isotonic' else contact.copy()
    negative=control.copy()
    if (negative&contact).any():raise ValueError('Strict controls cannot contain either-query contact')
    return positive,negative,~(positive|negative)


def training_rows_for_arm(arm,q,contact_query,contact,control,deadline,trainmask):
    positive,negative,_=supervision_masks(arm,q,contact_query,contact,control)
    return I.training_rows(positive,negative,deadline,trainmask)


def predict_query_models(models,pair_scores):
    x=np.asarray(pair_scores)
    if x.ndim<1 or x.shape[-1]!=2 or len(models)!=2 or not np.isfinite(x).all():
        raise ValueError('Finite paired query scores and two models required')
    per_query=np.stack([models[q].predict(x[...,q].reshape(-1)).reshape(x.shape[:-1]) for q in (0,1)],axis=-1)
    return per_query,per_query.max(-1)


def raw_cutoff(model,theta):
    """Equivalent 1D raw cutoff; direct model prediction is numerically authoritative."""
    x,y=model.X_thresholds_,model.y_thresholds_
    if theta>y[-1]:return dict(mode='never_alarm',raw_cutoff=None)
    if theta<=y[0]:return dict(mode='always_alarm',raw_cutoff=None)
    j=int(np.searchsorted(y,theta,side='left'))
    cutoff=float(x[j-1]+(theta-y[j-1])/(y[j]-y[j-1])*(x[j]-x[j-1]))
    return dict(mode='raw_at_or_above',raw_cutoff=cutoff,
                numeric_note='Piecewise-linear cutoff; direct predict>=theta decides floating ties')


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def freeze():
    import sklearn
    parent=R.read(BASE/'PLAN.json');hashes=dict(parent['hashes'])
    paths=[Path(__file__),Path(I.__file__),BASE/'PLAN.json',BASE/'result.json',BASE/'ledger.npz',
           HB/'PLAN.json',HB/'geometry_manifest.json',HB/'score_ledger.npz',HB/'result.json']
    paths.extend(BASE/'models'/f'in_domain_fold{f}_pair_only.joblib' for f in range(3))
    for path in paths:hashes[str(path.relative_to(R.ROOT))]=R.sha(path)
    OUT.mkdir(parents=True,exist_ok=True)
    save('PLAN.json',dict(phase='POSTHOC EXPLORE; original and HB Development previously consumed',
        authorization='Continuous algorithm exploration; fixed two-recipe querywise calibration probe',
        budget_cpu_analysis_wall_seconds=300,no_gpu=True,one_fit_and_evaluation_run=True,
        folds=parent['folds'],arms=list(ARMS),
        model=dict(name='IsotonicRegression',increasing=True,out_of_bounds='clip',y_min=0,y_max=1),
        input='Only the two original smoothed center HEAD/BODY scores; each query fits independently; max of calibrated scores alarms',
        supervision={'query_isotonic':'Own g.contact_query[:,q] positives; g.control strict both-query-clear negatives; all others ignored',
                     'event_isotonic':'g.contact event positives for each query; same strict g.control negatives'},
        weights='Parent training_rows separately per recipe/query: one last-deadline sample per positive, weight1; controls weighted to that positive frame histogram and total mass',
        supervision_limit='Recipe comparison also changes positive count, deadline histogram and negative time weights; not a pure label-semantics intervention',
        score_limit='Class-balanced monotonic training scores, not physical collision probabilities',
        calibration='New models fit only original train; new thresholds only original calibration controls at2.5/5%. Original max/pair_only inherit prior fold thresholds. Original eval-control-matched cap is descriptive only.',
        ties='Keep isotonic plateaus and exact >= semantics; no relaxed thresholds. Report unique scores, plateau mass and theta>1/never-alarm cases',
        primary='Timely excluding first2 outputs and actual false-alarm time[2:12]; retain any-before timely and warmupFA separately; original event denominators unchanged',
        hb='All216 existing H/B/HB sequences are check-only. Model and threshold from anchor original evaluation fold; no HB training/calibration, no new structures',
        diagnostic='Each fold/query save knots/fit counts/weights; calibration own-query and overlap FA; common threshold equivalent raw cutoffs. Increasing each input cannot lower max-isotonic score; this does not guarantee upstream HB evidence or timely recovery.',
        delivery='PLAN, model files, cal/eval per-query and max predictions, per-event ledgers, original-domain query subgroups and HB metrics, terminal/time',
        sklearn=sklearn.__version__,numpy=np.__version__,hashes=hashes))
    print('FROZEN querywise/event isotonic, CPU300s, two fixed recipes',flush=True)


def event_states(alarm,contact,deadline):
    rows=np.flatnonzero(contact);timely=np.zeros(len(contact),bool);post=timely.copy()
    timely[rows]=np.maximum.accumulate(alarm,axis=1)[rows,deadline[rows]]
    late=alarm.copy();late[:,:2]=False
    post[rows]=np.maximum.accumulate(late,axis=1)[rows,deadline[rows]]
    return timely,post


def metrics(alarm,contact,control,deadline,mask):
    timely,post=event_states(alarm,contact,deadline);ec=mask&contact;ct=mask&control
    return dict(events=int(ec.sum()),controls=int(ct.sum()),timely=int((timely&ec).sum()),
        timely_excluding_warmup=int((post&ec).sum()),
        false_alarm_intervals=int(alarm[ct,2:12].sum()),control_intervals=int(ct.sum())*10,
        false_alarm=float(alarm[ct,2:12].mean()) if ct.any() else None,
        warmup_alarm_frames=int(alarm[ct,:2].sum()),warmup_control_frames=int(ct.sum())*2,
        warmup_false_alarm=float(alarm[ct,:2].mean()) if ct.any() else None)


def run():
    import joblib
    from sklearn.isotonic import IsotonicRegression
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=R.read(OUT/'PLAN.json')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('One run only; retain prior partial/result')
    save('RUN_STARTED.json',dict(unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_cpu_analysis_wall_seconds']:raise TimeoutError('300s CPU budget reached')
    try:
        for path,h in plan['hashes'].items():assert R.sha(R.ROOT/path)==h,path
        rows,g,features,base,_,_=I.load_data();check()
        pair=features['center_current'][...,:2];n=len(rows)
        uid=np.array([r['unit'] for r in rows]);cfg=np.array([r['config'] for r in rows])
        contact,control,cq=g['contact'],g['control'],g['contact_query'];deadline=E.causal_index(g['fraction'])
        old_result=R.read(BASE/'result.json');old_records={(v['fold'],v['key']):v for v in old_result['fold_records']}
        scores={m:np.zeros((n,13)) for m in METHODS};query_scores={m:np.zeros((n,13,2)) for m in ('original_center',*ARMS)}
        alarms={};query_alarms={};folds=np.full(n,-1,int);training=[];thresholds={};fold_records=[];models={};role_ledger={}
        (OUT/'models').mkdir(exist_ok=True)
        with threadpool_limits(limits=4):
            for fold,roles in enumerate(plan['folds']):
                check();masks={role:np.isin(uid,units) for role,units in roles.items()};tr,ca,te=[masks[k] for k in ('train','calibration','evaluation')]
                assert not ((tr&ca)|(tr&te)|(ca&te)).any() and (folds[te]==-1).all();folds[te]=fold
                role_ledger.update({f'fold{fold}/{k}':v for k,v in masks.items()})
                pred={'original_center':pair.max(-1)};qp={'original_center':pair}
                old_model=joblib.load(BASE/'models'/f'in_domain_fold{fold}_pair_only.joblib')
                pred['pair_only']=old_model.predict_proba(pair.reshape(-1,2))[:,1].reshape(n,13)
                with np.load(BASE/'ledger.npz') as z:np.testing.assert_array_equal(pred['pair_only'][te],z['score/in_domain/pair_only'][te])
                for arm in ARMS:
                    fitted=[]
                    for q in (0,1):
                        positive,negative,ignored=supervision_masks(arm,q,cq,contact,control)
                        ri,ti,y,w=training_rows_for_arm(arm,q,cq,contact,control,deadline,tr)
                        model=IsotonicRegression(increasing=True,out_of_bounds='clip',y_min=0,y_max=1)
                        model.fit(pair[ri,ti,q],y,sample_weight=w);fitted.append(model)
                        joblib.dump(model,OUT/'models'/f'fold{fold}_{arm}_q{q}.joblib')
                        training.append(dict(fold=fold,arm=arm,query=q,positive_sequences=int((positive&tr).sum()),
                            negative_sequences=int((negative&tr).sum()),ignored_sequences=int((ignored&tr).sum()),
                            samples=len(y),positive_mass=float(w[y==1].sum()),negative_mass=float(w[y==0].sum()),
                            positive_deadline_histogram=np.bincount(deadline[positive&tr],minlength=13).tolist(),
                            X_thresholds=model.X_thresholds_.tolist(),y_thresholds=model.y_thresholds_.tolist()))
                    models[fold,arm]=fitted;qp[arm],pred[arm]=predict_query_models(fitted,pair)
                    print('FIT',fold,arm,[training[-2]['positive_sequences'],training[-1]['positive_sequences']],flush=True)
                for name,score in pred.items():
                    scores[name][te]=score[te]
                    if name in qp:query_scores[name][te]=qp[name][te]
                    for role,mask in [('calibration',ca),('evaluation',te)]:
                        payload=dict(row_index=np.flatnonzero(mask),score=score[mask],unit=uid[mask],config=cfg[mask])
                        if name in qp:payload['per_query']=qp[name][mask]
                        np.savez_compressed(OUT/f'fold{fold}_{name}_{role}.npz',**payload)
                    for cap in (.025,.05):
                        for mode,fitmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                            key=f'{mode}/{cap:.3f}/{name}'
                            if name in ('original_center','pair_only'):
                                theta=float(old_records[fold,'in_domain/'+key]['threshold'])
                            else:theta=P.threshold_at_cap(score[fitmask&control,2:12],cap)
                            al=score>=theta;fitvals=score[fitmask&control,2:12];fitfa=float((fitvals>=theta).mean())
                            assert fitfa<=cap+1e-12
                            alarms.setdefault(key,np.zeros((n,13),bool))[te]=al[te]
                            diag=dict(threshold=theta,fit_false_alarm=fitfa,unique_fit_scores=int(len(np.unique(fitvals))),
                                largest_fit_plateau_intervals=int(np.unique(fitvals,return_counts=True)[1].max()),
                                fit_intervals=int(fitvals.size),theta_above_one=bool(theta>1) if name in ARMS else None,
                                zero_alert_on_fit=bool(not (fitvals>=theta).any()))
                            if name in qp:
                                qa=qp[name]>=theta;query_alarms.setdefault(key,np.zeros((n,13,2),bool))[te]=qa[te]
                                fc=qa[fitmask&control,2:12]
                                diag.update(query_false_alarm_intervals=fc.sum((0,1)).tolist(),
                                    query_overlap_false_alarm_intervals=int(fc.all(-1).sum()),
                                    query_union_false_alarm_intervals=int(fc.any(-1).sum()))
                            if name in ARMS:diag['raw_cutoffs']=[raw_cutoff(m,theta) for m in models[fold,name]]
                            thresholds[fold,key]=theta
                            fold_records.append(dict(fold=fold,key=key,**diag,**metrics(al,contact,control,deadline,te)))
                check()
        assert (folds>=0).all()
        groups={'all':np.ones(n,bool),'head_contact':cq[:,0]|control,'body_contact':cq[:,1]|control,
                'both_queries_contact':cq.all(1)|control,'head_only':(cq[:,0]&~cq[:,1])|control,
                'body_only':(cq[:,1]&~cq[:,0])|control,'family_none':cfg<20,'family_corner':cfg>=20}
        aggregate={key:{group:metrics(al,contact,control,deadline,mask) for group,mask in groups.items()} for key,al in alarms.items()}
        query_metrics={}
        for key,qa in query_alarms.items():
            query_metrics[key]={}
            for q in (0,1):query_metrics[key][str(q)]=metrics(qa[...,q],cq[:,q],control,deadline,np.ones(n,bool))
        # Only after all original fitting/calibration has completed, inspect HB data.
        hbmanifest=R.read(HB/'geometry_manifest.json');hblist=[];hbpair=[];hbbase={k:[] for k in ('original_center','pair_only')}
        with np.load(HB/'score_ledger.npz') as z:
            for a in hbmanifest['anchors']:
                f=plan['folds'][a['fold']]
                assert a['unit'] in f['evaluation'] and a['unit'] not in f['train']+f['calibration']
                for tag in ('H','B','HB'):
                    v=a['variants'][tag];prefix=a['anchor_id']+'/'+tag+'/'
                    hd_value=-1 if v['deadline_index'] is None else int(v['deadline_index'])
                    if v['contact']:assert 0<=hd_value<13,'Contact requires a valid causal deadline'
                    hblist.append(dict(anchor_id=a['anchor_id'],unit=a['unit'],config=a['config'],fold=a['fold'],tag=tag,
                        contact=bool(v['contact']),control=bool(v['control']),deadline=hd_value,contact_query=v['contact_query']))
                    hbpair.append(z[prefix+'x'])
                    for name in hbbase:hbbase[name].append(z[prefix+name])
        hx=np.asarray(hbpair);hc=np.array([v['contact'] for v in hblist]);hn=np.array([v['control'] for v in hblist]);hd=np.array([v['deadline'] for v in hblist]);hf=np.array([v['fold'] for v in hblist])
        hs={k:np.asarray(v) for k,v in hbbase.items()};hq={};ha={};hbmetrics={};domination={}
        for arm in ARMS:
            hq[arm]=np.zeros(hx.shape);hs[arm]=np.zeros(hx.shape[:2])
            for fold in range(3):hq[arm][hf==fold],hs[arm][hf==fold]=predict_query_models(models[fold,arm],hx[hf==fold])
        for name,score in hs.items():
            for cap in (.025,.05):
                key=f'calibrated/{cap:.3f}/{name}'
                theta=np.array([thresholds[f,key] for f in hf]);al=score>=theta[:,None];ha[key]=al
                hbmetrics[key]={tag:metrics(al,hc,hn,hd,np.array([v['tag']==tag for v in hblist])) for tag in ('H','B','HB')}
        for arm in ARMS:
            eligible=violations=0
            for i in range(0,len(hblist),3):
                for single in (i,i+1):
                    comparable=(hx[i+2]>=hx[single]).all(-1);eligible+=int(comparable.sum())
                    violations+=int((comparable&(hs[arm][i+2]<hs[arm][single]-1e-12)).sum())
            domination[arm]=dict(raw_both_logits_nondecreasing_frames=eligible,score_decrease_frames=violations)
            assert violations==0
        check()
        np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=folds,contact=contact,control=control,contact_query=cq,deadline=deadline,
            **{'role/'+k:v for k,v in role_ledger.items()},**{'score/'+k:v for k,v in scores.items()},
            **{'query_score/'+k:v for k,v in query_scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:event_states(v,contact,deadline)[0] for k,v in alarms.items()},
            **{'timely_excluding_warmup/'+k:event_states(v,contact,deadline)[1] for k,v in alarms.items()})
        np.savez_compressed(OUT/'hb_ledger.npz',unit=np.array([v['unit'] for v in hblist]),config=np.array([v['config'] for v in hblist]),fold=hf,
            contact=hc,control=hn,deadline=hd,tag=np.array([v['tag'] for v in hblist]),pair=hx,
            **{'score/'+k:v for k,v in hs.items()},**{'query_score/'+k:v for k,v in hq.items()},**{'alarm/'+k:v for k,v in ha.items()})
        save('hb_rows.json',hblist)
        result=dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,units=len(np.unique(uid)),sequences=n,
            events=int(contact.sum()),controls=int(control.sum()),training=training,fold_records=fold_records,
            metrics=aggregate,query_metrics=query_metrics,hb_metrics=hbmetrics,hb_monotonicity_check=domination,
            plan_sha256=R.sha(OUT/'PLAN.json'),ledger_sha256=R.sha(OUT/'ledger.npz'),hb_ledger_sha256=R.sha(OUT/'hb_ledger.npz'))
        save('result.json',result);save('TERMINAL.json',dict(status='complete',seconds=time.monotonic()-start))
        for key,value in aggregate.items():
            if '/0.025/' in key:print('ORIGINAL',key,value['all'],flush=True)
        for key,value in hbmetrics.items():
            if '/0.025/' in key:print('HB',key,value,flush=True)
        print('COMPLETE',round(time.monotonic()-start,2),'seconds',flush=True)
    except BaseException as exc:
        save('TERMINAL.json',dict(status='failed_or_budget_stop',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
