"""Matched source-repeat/HB augmentation of paired M3 scores; consumed Development."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cnh_direction_information_dev as I
import cnh_querywise_calibration_dev as Q

R,P=I.R,I.P
OUT=R.WORK/'cnh-training-support-dev-20261007'
ARMS=('source_repeat','hb_aug')
METHODS=('original_center','pair_only',*ARMS)


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def augmented_training_rows(contact,control,deadline,trainmask,source_indices,anchor_units,train_units):
    """A single shared row/label/weight schedule for both observation arms."""
    selected=np.flatnonzero(np.isin(anchor_units,train_units))
    source=np.asarray(source_indices)[selected]
    assert np.asarray(trainmask)[source].all(),'Augmentation source must belong to original train role'
    c=np.concatenate((contact,np.asarray(contact)[source]));n=np.concatenate((control,np.asarray(control)[source]))
    d=np.concatenate((deadline,np.asarray(deadline)[source]));mask=np.concatenate((trainmask,np.ones(len(source),bool)))
    ri,ti,y,w=I.training_rows(c,n,d,mask)
    return selected,ri,ti,y,w


def paired_structure_counts(timely,contact,tags):
    t=np.asarray(timely).reshape(-1,3);c=np.asarray(contact).reshape(-1,3)
    assert np.array_equal(np.asarray(tags).reshape(-1,3),np.tile(['H','B','HB'],(len(t),1)))
    assert (c==c[:,0,None]).all()
    eligible=c[:,0];single=t[:,:2].any(1)
    return dict(contact_anchors=int(eligible.sum()),single_any_timely=int((eligible&single).sum()),
        hb_timely=int((eligible&t[:,2]).sum()),single_any_timely_hb_miss=int((eligible&single&~t[:,2]).sum()),
        hb_timely_both_single_miss=int((eligible&~single&t[:,2]).sum()))


def load_inputs():
    with np.load(Q.OUT/'ledger.npz') as z:
        original={k:z[k] for k in ('unit','config','fold','contact','control','contact_query','deadline')}
        original['pair']=z['query_score/original_center']
        original['baseline']={k:z['score/'+k] for k in ('original_center','pair_only')}
    with np.load(Q.OUT/'hb_ledger.npz') as z:
        hb={k:z[k] for k in ('unit','config','fold','contact','control','deadline','tag','pair')}
        hb['baseline']={k:z['score/'+k] for k in ('original_center','pair_only')}
    hrows=R.read(Q.OUT/'hb_rows.json');anchors=R.read(Q.HB/'geometry_manifest.json')['anchors']
    indices={(int(u),int(c)):i for i,(u,c) in enumerate(zip(original['unit'],original['config']))}
    source=[];receipts=[]
    for ai,a in enumerate(anchors):
        si=indices[a['unit'],a['config']];source.append(si)
        for j,tag in enumerate(('H','B','HB')):
            hi=3*ai+j;row=hrows[hi];v=a['variants'][tag]
            assert (row['anchor_id'],row['tag'],int(hb['unit'][hi]),int(hb['config'][hi]))==(a['anchor_id'],tag,a['unit'],a['config'])
            assert bool(v['contact'])==bool(original['contact'][si]) and bool(v['control'])==bool(original['control'][si])
            assert bool(hb['contact'][hi])==bool(v['contact']) and bool(hb['control'][hi])==bool(v['control'])
            if v['deadline_index'] is not None:assert v['deadline_index']==int(original['deadline'][si])
            if v['contact']:assert v['deadline_index']==int(original['deadline'][si])==int(hb['deadline'][hi])
        source_tag=a['source_variant'];source_hi=3*ai+('H','B','HB').index(source_tag)
        np.testing.assert_array_equal(hb['pair'][source_hi],original['pair'][si])
        receipts.append(dict(anchor_index=ai,anchor_id=a['anchor_id'],unit=a['unit'],config=a['config'],original_fold=a['fold'],
            source_variant=source_tag,source_row_index=si,contact=bool(original['contact'][si]),control=bool(original['control'][si]),
            training_deadline=int(original['deadline'][si]),hb_physical_deadline=a['variants']['HB']['deadline_index'],
            noncontact_deadline_representation_diff=bool(not original['contact'][si] and a['variants']['HB']['deadline_index'] is None)))
    return original,hb,hrows,np.array(source),receipts


def freeze():
    import sklearn
    parent=R.read(Q.BASE/'PLAN.json');hashes=dict(parent['hashes'])
    paths=[Path(__file__),Path(I.__file__),Path(Q.__file__),Path(__file__).with_name('cnh_training_support_contrasts_dev.py'),Q.BASE/'PLAN.json',Q.BASE/'result.json',
        Q.OUT/'PLAN.json',Q.OUT/'result.json',Q.OUT/'ledger.npz',Q.OUT/'hb_ledger.npz',Q.OUT/'hb_rows.json',Q.HB/'geometry_manifest.json']
    paths.extend(Q.BASE/'models'/f'in_domain_fold{f}_pair_only.joblib' for f in range(3))
    for p in paths:hashes[str(p.relative_to(R.ROOT))]=R.sha(p)
    original,hb,_,source,receipts=load_inputs()
    support=[]
    for fold,roles in enumerate(parent['folds']):
        au=np.array([a['unit'] for a in receipts]);sel=np.isin(au,roles['train']);ev=np.array([a['original_fold']==fold for a in receipts])
        support.append(dict(fold=fold,train_anchors=int(sel.sum()),train_positive_anchors=int(original['contact'][source[sel]].sum()),
            evaluation_positive_anchors=int(original['contact'][source[ev]].sum())))
    assert [x['train_positive_anchors'] for x in support]==[17,19,22]
    assert [x['evaluation_positive_anchors'] for x in support]==[12,12,12]
    OUT.mkdir(parents=True,exist_ok=True)
    save('PLAN.json',dict(phase='POSTHOC EXPLORE; all original and HB Development consumed',
        authorization='Continuous algorithm exploration; matched training-support intervention',budget_cpu_analysis_wall_seconds=300,
        no_gpu=True,threads=4,compute_reason='TASK_NOT_GPU_SUITABLE: small fixed HGB, no GPU backend',
        one_fit_and_evaluation_run=True,arms=list(ARMS),model=parent['model'],seed='I.SEED+fold',folds=parent['folds'],support=support,
        inputs='Only2 existing smoothed center HEAD/BODY M3 scores; no raw features, identity, geometry labels or time as model inputs',
        augmentation='Select only anchors with unit in fold train role. source_repeat adds exact original source pair once per selected anchor; hb_aug adds its HB pair once. Same selected anchors, event/control labels and training deadlines.',
        weighting='Joint original+added sequence pool passed once to training_rows for both arms: each positive last-deadline sample weight1, not renormalized; original and added controls receive equal per-frame weight matching the combined positive-time histogram and total mass.',
        clear_deadline='32 noncontact anchors have original deadline12 versus HB None; both training arms retain source12 solely as an unused placeholder. Controls use positive-time histogram and never their own deadline. HB evaluation retains None->-1. All contact deadlines match exactly.',
        calibration='New models train only original train role plus train-role augmentations. Original calibration controls only choose2.5/5% thresholds. Existing max/pair scores and thresholds inherited. Eval-control matched cap is descriptive only.',
        evaluation='Original5760 sequences369events1255controls; HB216 variants72anchors39units, each structure36events36controls. HB each anchor evaluated only by its original OOF fold; that unit absent from train/cal. No HB calibration.',
        exposure='Unlike earlier check-only probes, HB structure is deliberately exposed to training in other units; no fresh or independent-confirmation claim.',
        primary='Timely excluding first2 outputs and actualFA[2:12], plus warmupFA and any-before timely. Report paired single-H/B-any-timely but HB-miss count at each inherited-source-cal threshold.',
        decision='hb_aug versus matched source_repeat: more structural detections with retained original-domain performance and no large actualFA increase warrants later validation; preserve counts and actualFA, no promised equalFA.',
        limitations=['Only17/19/22 added train HB positives, fixed min_samples_leaf20; negative result is not a universal rejection of training support',
                    'Same consumed simulator families; changed training support and class mass, isolated between new arms by the repeat control',
                    'Positive result cannot establish real-scene or safety performance'],
        sklearn=sklearn.__version__,numpy=np.__version__,hashes=hashes))
    print('FROZEN matched training support, CPU300s',flush=True)


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    start=time.monotonic();plan=R.read(OUT/'PLAN.json')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('One run only; preserve prior payload')
    save('RUN_STARTED.json',dict(unix=time.time(),budget_seconds=300))
    def check():
        if time.monotonic()-start>plan['budget_cpu_analysis_wall_seconds']:raise TimeoutError('300s CPU analysis budget reached')
    try:
        for path,h in plan['hashes'].items():assert R.sha(R.ROOT/path)==h,path
        o,h,hrows,source,augrows=load_inputs();uid=o['unit'];cfg=o['config'];pair=o['pair'];n=len(uid)
        contact,control,deadline,cq=[o[k] for k in ('contact','control','deadline','contact_query')]
        assert n==5760 and contact.sum()==369 and control.sum()==1255 and len(hrows)==216
        au=np.array([r['unit'] for r in augrows]);hbp=h['pair'][2::3];repeat=pair[source]
        np.savez_compressed(OUT/'augmentation_inputs.npz',source_row_index=source,unit=au,anchor_id=np.array([r['anchor_id'] for r in augrows]),
            source_repeat=repeat,hb_aug=hbp,contact=contact[source],control=control[source],training_deadline=deadline[source])
        save('augmentation_rows.json',augrows)
        oldrecords={(v['fold'],v['key']):v for v in R.read(Q.BASE/'result.json')['fold_records']}
        scores={k:np.zeros((n,13)) for k in METHODS};alarms={};foldidx=np.full(n,-1,int);records=[];training=[];thresholds={};models={};roleledger={}
        (OUT/'models').mkdir(exist_ok=True)
        with threadpool_limits(limits=4):
            for fold,roles in enumerate(plan['folds']):
                check();masks={k:np.isin(uid,v) for k,v in roles.items()};tr,ca,te=[masks[k] for k in ('train','calibration','evaluation')]
                assert not ((tr&ca)|(tr&te)|(ca&te)).any() and (foldidx[te]==-1).all();foldidx[te]=fold
                roleledger.update({f'fold{fold}/{k}':v for k,v in masks.items()})
                selected,ri,ti,y,w=augmented_training_rows(contact,control,deadline,tr,source,au,roles['train'])
                assert int(contact[source[selected]].sum())==[17,19,22][fold]
                joint_source=np.concatenate((np.arange(n),source[selected]));joint_anchor=np.concatenate((np.full(n,-1,int),selected))
                np.savez_compressed(OUT/f'fold{fold}_training_schedule.npz',selected_anchor_index=selected,row_index=ri,time_index=ti,label=y,weight=w,
                    original_source_row_index=joint_source[ri],anchor_index=joint_anchor[ri])
                oldmodel=joblib.load(Q.BASE/'models'/f'in_domain_fold{fold}_pair_only.joblib')
                pred={'original_center':pair.max(-1),'pair_only':oldmodel.predict_proba(pair.reshape(-1,2))[:,1].reshape(n,13)}
                for name in pred:np.testing.assert_array_equal(pred[name][te],o['baseline'][name][te])
                hm=h['fold']==fold
                np.testing.assert_array_equal(h['pair'][hm].max(-1),h['baseline']['original_center'][hm])
                np.testing.assert_array_equal(oldmodel.predict_proba(h['pair'][hm].reshape(-1,2))[:,1].reshape(hm.sum(),13),h['baseline']['pair_only'][hm])
                for name,added in [('source_repeat',repeat),('hb_aug',hbp)]:
                    check();t=time.monotonic();x=np.concatenate((pair,added[selected]),axis=0)
                    model=HistGradientBoostingClassifier(**{k:v for k,v in plan['model'].items() if k!='name'},random_state=I.SEED+fold)
                    model.fit(x[ri,ti],y,sample_weight=w);models[fold,name]=model
                    pred[name]=model.predict_proba(pair.reshape(-1,2))[:,1].reshape(n,13)
                    joblib.dump(model,OUT/'models'/f'fold{fold}_{name}.joblib')
                    training.append(dict(fold=fold,name=name,added_anchors=len(selected),added_positives=int(contact[source[selected]].sum()),
                        added_controls=int(control[source[selected]].sum()),positive_events=int(y.sum()),samples=len(y),
                        positive_mass=float(w[y==1].sum()),negative_mass=float(w[y==0].sum()),
                        positive_deadline_histogram=np.bincount(ti[y==1],minlength=13).tolist(),seconds=time.monotonic()-t))
                    print('FIT',fold,name,round(training[-1]['seconds'],2),flush=True)
                for name,score in pred.items():
                    scores[name][te]=score[te]
                    for role,mask in [('calibration',ca),('evaluation',te)]:
                        np.savez_compressed(OUT/f'fold{fold}_{name}_{role}.npz',row_index=np.flatnonzero(mask),unit=uid[mask],config=cfg[mask],score=score[mask])
                    for cap in (.025,.05):
                        for mode,fitmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                            key=f'{mode}/{cap:.3f}/{name}'
                            theta=float(oldrecords[fold,'in_domain/'+key]['threshold']) if name in ('original_center','pair_only') else P.threshold_at_cap(score[fitmask&control,2:12],cap)
                            al=score>=theta;fitfa=float(al[fitmask&control,2:12].mean());assert fitfa<=cap+1e-12
                            thresholds[fold,key]=theta;alarms.setdefault(key,np.zeros((n,13),bool))[te]=al[te]
                            records.append(dict(fold=fold,key=key,threshold=theta,fit_false_alarm=fitfa,**Q.metrics(al,contact,control,deadline,te)))
        assert (foldidx>=0).all();np.testing.assert_array_equal(foldidx,o['fold']);check()
        groups={'all':np.ones(n,bool),'head_contact':cq[:,0]|control,'body_contact':cq[:,1]|control,'family_none':cfg<20,'family_corner':cfg>=20}
        aggregate={k:{g:Q.metrics(v,contact,control,deadline,m) for g,m in groups.items()} for k,v in alarms.items()}
        hscores={k:v.copy() for k,v in h['baseline'].items()};halarms={};hmetrics={};paired={}
        with threadpool_limits(limits=4):
            for name in ARMS:
                hscores[name]=np.zeros(h['pair'].shape[:2])
                for fold,roles in enumerate(plan['folds']):
                    mask=h['fold']==fold
                    assert np.isin(h['unit'][mask],roles['evaluation']).all() and not np.isin(h['unit'][mask],roles['train']+roles['calibration']).any()
                    hscores[name][mask]=models[fold,name].predict_proba(h['pair'][mask].reshape(-1,2))[:,1].reshape(mask.sum(),13)
        for name,score in hscores.items():
            for cap in (.025,.05):
                key=f'calibrated/{cap:.3f}/{name}';theta=np.array([thresholds[f,key] for f in h['fold']]);al=score>=theta[:,None];halarms[key]=al
                hmetrics[key]={tag:Q.metrics(al,h['contact'],h['control'],h['deadline'],h['tag']==tag) for tag in ('H','B','HB')}
                timely,post=Q.event_states(al,h['contact'],h['deadline'])
                paired[key]=dict(any_before=paired_structure_counts(timely,h['contact'],h['tag']),
                    excluding_warmup=paired_structure_counts(post,h['contact'],h['tag']))
        check()
        np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=foldidx,contact=contact,control=control,contact_query=cq,deadline=deadline,
            **{'role/'+k:v for k,v in roleledger.items()},**{'score/'+k:v for k,v in scores.items()},**{'alarm/'+k:v for k,v in alarms.items()},
            **{'timely/'+k:Q.event_states(v,contact,deadline)[0] for k,v in alarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,contact,deadline)[1] for k,v in alarms.items()})
        np.savez_compressed(OUT/'hb_ledger.npz',**{k:h[k] for k in ('unit','config','fold','contact','control','deadline','tag')},
            anchor_id=np.array([r['anchor_id'] for r in hrows]),**{'score/'+k:v for k,v in hscores.items()},**{'alarm/'+k:v for k,v in halarms.items()},
            **{'timely/'+k:Q.event_states(v,h['contact'],h['deadline'])[0] for k,v in halarms.items()},
            **{'timely_excluding_warmup/'+k:Q.event_states(v,h['contact'],h['deadline'])[1] for k,v in halarms.items()})
        save('result.json',dict(status='EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,units=len(np.unique(uid)),sequences=n,
            events=int(contact.sum()),controls=int(control.sum()),training=training,fold_records=records,metrics=aggregate,hb_metrics=hmetrics,hb_paired=paired,
            plan_sha256=R.sha(OUT/'PLAN.json'),ledger_sha256=R.sha(OUT/'ledger.npz'),hb_ledger_sha256=R.sha(OUT/'hb_ledger.npz')))
        save('TERMINAL.json',dict(status='complete',seconds=time.monotonic()-start))
        for key,v in aggregate.items():
            if key.startswith('calibrated/0.025/'):print('ORIGINAL',key,v['all'],flush=True)
        for key,v in hmetrics.items():
            if key.startswith('calibrated/0.025/'):print('HB',key,v,paired[key],flush=True)
        print('COMPLETE',round(time.monotonic()-start,2),'seconds',flush=True)
    except BaseException as exc:
        save('TERMINAL.json',dict(status='failed_or_budget_stop',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
