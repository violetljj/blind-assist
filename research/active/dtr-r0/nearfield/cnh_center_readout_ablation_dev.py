"""Posthoc CPU ablation of the center-output readout; consumed Development.

Uses the frozen information-probe folds and training recipe. No new M3 inference.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import cnh_direction_information_dev as I

R, E, P = I.R, I.E, I.P
OUT = R.WORK/'cnh-center-readout-ablation-dev-20261007'
METHODS = ('max_only', 'pair_only', 'max_width', 'pair_width', 'pair_width_monotonic')
SEED = 2026100717


def save(name, value):
    with (OUT/name).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def freeze():
    import sklearn
    parent = R.read(I.OUT/'PLAN.json')
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = dict(parent['hashes'])
    for p in (Path(__file__), Path(I.__file__), Path(P.__file__), Path(I.Q.__file__),
              Path(I.H.__file__), I.OUT/'PLAN.json', I.OUT/'result.json', I.OUT/'ledger.npz'):
        hashes[str(p.relative_to(R.ROOT))] = R.sha(p)
    save('PLAN.json', dict(phase='POSTHOC EXPLORE; previously consumed Development',
        authorization='User continuous algorithm exploration; scoped center-readout ablation delegated by parent',
        budget_cpu_analysis_wall_seconds=1200, no_gpu=True, threads=4,
        parent_plan=str((I.OUT/'PLAN.json').relative_to(R.ROOT)),
        parent_plan_sha256=R.sha(I.OUT/'PLAN.json'), model=parent['model'], folds=parent['folds'],
        model_seed='Parent I.SEED + fold, unchanged to reproduce center_current',
        methods=list(METHODS), baseline='Original center=max(HEAD,BODY), threshold only',
        inputs={'max_only':['center max'], 'pair_only':['HEAD','BODY'],
                'max_width':['center max','adaptive width','fixed width'],
                'pair_width':['HEAD','BODY','adaptive width','fixed width'],
                'pair_width_monotonic':['HEAD','BODY','adaptive width','fixed width']},
        monotonic_constraints={'pair_width_monotonic':[1,1,0,0]},
        experiments={'in_domain':list(METHODS), 'none_to_corner':['pair_only','pair_width_monotonic'],
                     'corner_to_none':['pair_only','pair_width_monotonic']},
        transfer='Same unit folds; training/calibration source family only, evaluation target family only. none=config<20, corner=config>=20. No refit to target labels.',
        supervision=parent['supervision'],
        threshold='Separate model and fold thresholds, calibration-only 2.5%/5% control time cap. Evaluation-control matched cap secondary descriptive.',
        metrics='Original deadline any-before, actualFA[2:12], warmupFA[:2], timely excluding warmup, silent/unknown with original common gate/max-clear tau. Event denominator unchanged.',
        comparisons='Paired whole-unit bootstrap: pair_width-max_only, pair_width-pair_only; same predictions, thresholds and fits held fixed. Not training uncertainty.',
        unsupported='Missing train classes, calibration controls, evaluation events or controls makes that fold not evaluable; report missing coverage, never change split.',
        limitations=['Posthoc mechanism probe, not confirmation', 'Same simulator families and consumed inputs',
                     'Current M3 scores already contain upstream observation history',
                     'Family transfer diagnostic is not real-scene or hardware validation',
                     'Warmup is outside primary control-time FA and must be reported explicitly'],
        sklearn=sklearn.__version__, numpy=np.__version__, hashes=hashes))
    print('FROZEN posthoc CPU ablation, 1200s, no GPU', flush=True)


def score_states(score, theta, contact, deadline, gate, clear_score, tau):
    alarm = score >= theta
    clear = ~alarm & gate & (clear_score <= tau)
    rr = np.flatnonzero(contact)
    timely = np.zeros(len(score), bool)
    timely[rr] = np.maximum.accumulate(alarm, axis=1)[rr, deadline[rr]]
    post = alarm.copy(); post[:, :2] = False
    no_warmup = np.zeros(len(score), bool)
    no_warmup[rr] = np.maximum.accumulate(post, axis=1)[rr, deadline[rr]]
    silent = np.zeros(len(score), bool)
    silent[rr] = ~timely[rr] & clear[rr, deadline[rr]]
    return dict(timely=timely, timely_excluding_warmup=no_warmup, silent=silent,
                alarm=alarm, unknown=~alarm & ~clear)


def metrics(s, mask, contact, control):
    ec, ct = mask & contact, mask & control
    return dict(events=int(ec.sum()), controls=int(ct.sum()),
        timely=int((s['timely'] & ec).sum()),
        timely_excluding_warmup=int((s['timely_excluding_warmup'] & ec).sum()),
        silent=int((s['silent'] & ec).sum()),
        unknown_miss=int((ec & ~s['timely'] & ~s['silent']).sum()),
        false_alarm=float(s['alarm'][ct, 2:12].mean()) if ct.any() else None,
        warmup_false_alarm=float(s['alarm'][ct, :2].mean()) if ct.any() else None,
        unknown_time=float(s['unknown'][ct, 2:12].mean()) if ct.any() else None)


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits
    start = time.monotonic(); plan = R.read(OUT/'PLAN.json')
    if (OUT/'result.json').exists() or (OUT/'ledger.npz').exists():
        raise FileExistsError('Preserve completed payload')
    for path, expected in plan['hashes'].items():
        assert R.sha(R.ROOT/path) == expected, path
    def check():
        if time.monotonic()-start > plan['budget_cpu_analysis_wall_seconds']:
            raise TimeoutError('1200s CPU analysis budget exhausted')
    rows, g, parent_features, base, gate, tau = I.load_data(); check()
    pair_width = parent_features['center_current']; pair = pair_width[..., :2]
    max_score = base['original_center']; widths = pair_width[..., 2:]
    features = dict(max_only=max_score[...,None], pair_only=pair,
        max_width=np.concatenate((max_score[...,None], widths), -1),
        pair_width=pair_width, pair_width_monotonic=pair_width)
    uid = np.array([r['unit'] for r in rows]); cfg = np.array([r['config'] for r in rows]); n=len(rows)
    contact, control = g['contact'], g['control']; deadline = E.causal_index(g['fraction'])
    roles = {}; folds_idx = np.full(n, -1, int)
    cases = {'in_domain':(np.ones(n,bool),np.ones(n,bool)),
             'none_to_corner':(cfg<20,cfg>=20), 'corner_to_none':(cfg>=20,cfg<20)}
    outputs = {}; scores = {}; records=[]; training=[]; unsupported=[]; support={}
    (OUT/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for fold, f in enumerate(plan['folds']):
            rm = {role:np.isin(uid, units) for role, units in f.items()}
            assert not ((rm['train']&rm['calibration'])|(rm['train']&rm['evaluation'])|(rm['calibration']&rm['evaluation'])).any()
            assert (folds_idx[rm['evaluation']] == -1).all()
            folds_idx[rm['evaluation']] = fold
            for role, mask in rm.items(): roles[f'fold{fold}/{role}']=mask
            for case, (source, target) in cases.items():
                check(); tr=rm['train']&source; ca=rm['calibration']&source; te=rm['evaluation']&target
                supportkey=f'{case}/fold{fold}'
                support[supportkey]={role:dict(sequences=int(mask.sum()), units=len(np.unique(uid[mask])),
                    events=int((mask&contact).sum()), controls=int((mask&control).sum()))
                    for role,mask in [('train',tr),('calibration',ca),('evaluation',te)]}
                missing=[]
                for label, mask in [('train_contacts',tr&contact),('train_controls',tr&control),
                                    ('calibration_controls',ca&control),('evaluation_contacts',te&contact),('evaluation_controls',te&control)]:
                    if not mask.any(): missing.append(label)
                if missing:
                    unsupported.append(dict(case=case,fold=fold,missing=missing)); continue
                ri, ti, y, weights = I.training_rows(contact,control,deadline,tr)
                assert np.isclose(weights[y==0].sum(), weights[y==1].sum())
                predictions={'original_center':max_score}
                for name in plan['experiments'][case]:
                    check(); fit_start=time.monotonic(); x=features[name]
                    kwargs={k:v for k,v in plan['model'].items() if k!='name'}
                    if name=='pair_width_monotonic': kwargs['monotonic_cst']=[1,1,0,0]
                    model=HistGradientBoostingClassifier(**kwargs,random_state=I.SEED+fold)
                    model.fit(x[ri,ti],y,sample_weight=weights)
                    predictions[name]=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(n,13)
                    training.append(dict(case=case,fold=fold,name=name,positive_events=int(y.sum()),
                        samples=len(y),positive_weight=float(weights[y==1].sum()),negative_weight=float(weights[y==0].sum()),
                        train_auc=float(roc_auc_score(y,model.predict_proba(x[ri,ti])[:,1],sample_weight=weights)),
                        seconds=time.monotonic()-fit_start))
                    joblib.dump(model,OUT/'models'/f'{case}_fold{fold}_{name}.joblib')
                    print('FIT',case,fold,name,round(training[-1]['train_auc'],3),round(training[-1]['seconds'],2),flush=True)
                for name, score in predictions.items():
                    skey=f'{case}/{name}'
                    scores.setdefault(skey,np.full((n,13),np.nan))[te]=score[te]
                    for cap in (.025,.05):
                        for mode, fitmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                            theta=P.threshold_at_cap(score[fitmask&control,2:12],cap)
                            fitfa=float((score[fitmask&control,2:12]>=theta).mean())
                            assert fitfa<=cap+1e-12
                            st=score_states(score,theta,contact,deadline,gate,base['adaptive_max'],tau)
                            key=f'{case}/{mode}/{cap:.3f}/{name}'
                            entry=outputs.setdefault(key,dict(valid=np.zeros(n,bool),
                                **{k:np.zeros_like(v) for k,v in st.items()}))
                            assert not entry['valid'][te].any()
                            entry['valid'][te]=True
                            for k,v in st.items(): entry[k][te]=v[te]
                            records.append(dict(key=key,fold=fold,threshold=theta,fit_false_alarm=fitfa,
                                **metrics(st,te,contact,control)))
                np.savez_compressed(OUT/f'{case}_fold{fold}_checkpoint.npz',evaluation=te,
                    **{k:v[te] for k,v in predictions.items()})
    assert (folds_idx>=0).all(); check()
    groups={'all':np.ones(n,bool),'none':cfg<20,'corner':cfg>=20}
    groups.update({f'mode{k}':np.array([r['mode']==k for r in rows]) for k in range(3)})
    aggregate={key:{group:metrics(st,st['valid']&mask,contact,control) for group,mask in groups.items()}
               for key,st in outputs.items()}
    units,inv=np.unique(uid,return_inverse=True); boot=np.zeros((1000,len(units)),int); rng=np.random.default_rng(SEED)
    strata={r['unit']:(r['batch'],r['mode'],r['turn']) for r in rows}
    for stratum in sorted(set(strata.values())):
        ids=np.array([j for j,u in enumerate(units) if strata[u]==stratum])
        for j,draw in enumerate(rng.integers(0,len(ids),(1000,len(ids)))):
            boot[j,ids]=np.bincount(draw,minlength=len(ids))
    differences={}
    for mode in ('calibrated','matched_eval_descriptive'):
        for cap in (.025,.05):
            prefix=f'in_domain/{mode}/{cap:.3f}/'
            for a,b in [('pair_width','max_only'),('pair_width','pair_only'),('pair_only','max_only'),
                        ('pair_width_monotonic','pair_width'),('pair_width_monotonic','original_center')]:
                aa,bb=outputs[prefix+a],outputs[prefix+b]
                assert np.array_equal(aa['valid'],bb['valid'])
                out={}
                for measure in ('timely','timely_excluding_warmup','silent'):
                    delta=(aa[measure].astype(int)-bb[measure].astype(int))*(contact&aa['valid'])
                    sums=np.bincount(inv,weights=delta,minlength=len(units))
                    out[measure]=dict(diff=int(sums.sum()),ci95=np.quantile(boot@sums,[.025,.975]).tolist())
                differences[prefix+a+'-'+b]=out
    with np.load(I.OUT/'ledger.npz') as old:
        reproduction=float(np.max(np.abs(scores['in_domain/pair_width']-old['score/center_current'])))
    assert reproduction<1e-12, reproduction
    check()
    np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=folds_idx,contact=contact,control=control,
        deadline=deadline,gate=gate,clear_score=base['adaptive_max'],
        **{'role/'+k:v for k,v in roles.items()}, **{'score/'+k:v for k,v in scores.items()},
        **{field+'/'+key:st[field] for key,st in outputs.items() for field in ('valid','timely','timely_excluding_warmup','silent')})
    save('result.json',dict(status='POSTHOC_EXPLORATORY_COMPLETE',seconds=time.monotonic()-start,
        units=len(units),sequences=n,events=int(contact.sum()),controls=int(control.sum()),
        unsupported_folds=unsupported,fold_support=support,fold_records=records,metrics=aggregate,
        diffs=differences,training=training,pair_width_reproduction_max_abs=reproduction,
        clear_tau=tau,plan_sha256=R.sha(OUT/'PLAN.json'),ledger_sha256=R.sha(OUT/'ledger.npz')))
    for key,value in aggregate.items():
        if '/0.025/' in key: print('RESULT',key,value['all'],flush=True)
    print('COMPLETE',round(time.monotonic()-start,2),'seconds; pair_width reproduction',reproduction,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
