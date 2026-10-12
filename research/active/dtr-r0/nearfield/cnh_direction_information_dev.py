"""Cross-fitted causal M3 output information probe; CPU, consumed Development.

Learns deadline-event discrimination, not a new physical sensor model. Five-query
inputs are noisy-heading queries only; oracle direction is evaluation reference.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import cnh_probability_query_dev as P

R, E, H, Q = P.R, P.E, P.H, P.Q
OUT = R.WORK/'cnh-direction-information-dev-20261007'
SEED = 2026100713
sys.path.insert(0, str(OUT/'runtime'))


def causal_features(data, lags):
    data = np.asarray(data)
    return np.concatenate([data[:, np.maximum(np.arange(data.shape[1])-lag, 0)] for lag in lags], axis=-1)


def make_folds(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['batch'], r['mode'], r['turn']), set()).add(r['unit'])
    rng = np.random.default_rng(SEED); partitions = {}
    for key in sorted(groups):
        ids = rng.permutation(sorted(groups[key]))
        partitions[key] = [ids[k::3].tolist() for k in range(3)]
    folds = []
    for k in range(3):
        roles = dict(train=[], calibration=[], evaluation=[])
        for key in sorted(groups):
            parts = partitions[key]; roles['evaluation'].extend(parts[k])
            other = rng.permutation(parts[(k+1)%3]+parts[(k+2)%3]).tolist()
            nc = max(1, len(other)//4)
            roles['calibration'].extend(other[:nc]); roles['train'].extend(other[nc:])
        folds.append({key: sorted(value) for key, value in roles.items()})
    return folds


def training_rows(contact, control, deadline, trainmask):
    contact, control, trainmask = [np.asarray(x, bool) for x in (contact, control, trainmask)]
    deadline = np.asarray(deadline, int)
    if (contact & control).any():
        raise ValueError('contact/control labels overlap')
    pos = np.flatnonzero(contact & trainmask); neg = np.flatnonzero(control & trainmask)
    if not len(pos) or not len(neg) or not np.all((deadline[pos] >= 0) & (deadline[pos] < 13)):
        raise ValueError('positive deadlines and negative controls required')
    counts = np.bincount(deadline[pos], minlength=13)
    ts = np.flatnonzero(counts); nr = np.repeat(neg, len(ts)); nt = np.tile(ts, len(neg))
    return (np.r_[pos, nr], np.r_[deadline[pos], nt],
            np.r_[np.ones(len(pos)), np.zeros(len(nr))], np.r_[np.ones(len(pos)), counts[nt]/len(neg)])


def save(path, value):
    with (OUT/path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def freeze():
    import sklearn
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in R.read(H.R3/'rows.json') if r['unit'] in Q.A.UNITS]
    paths = [Path(__file__), Path(P.__file__), Path(R.__file__), Path(E.__file__),
             H.R3/'rows.json', H.R3/'online.npz', *E.GEOMETRY,
             *sorted((Q.OUT/'units').glob('unit*.npz'))]
    save('PLAN.json', dict(phase='EXPLORE; previously consumed Development', goal='Test whether directional/extra-score-history evidence adds event discrimination beyond supervised center readout',
        authorization='User active goal: continue algorithm exploration; new mechanism, not continuation of stopped R/T2 or max/weight search',
        budget_cpu_analysis_wall_seconds=1200, threads=4, no_gpu=True,
        model=dict(name='HistGradientBoostingClassifier', max_iter=150, learning_rate=.05,
                   max_leaf_nodes=7, max_depth=3, min_samples_leaf=20, l2_regularization=5., early_stopping=False),
        sklearn=sklearn.__version__, numpy=np.__version__, folds=make_folds(rows),
        features='Head estimate, single sensor only: smoothed HEAD/BODY logits at center, +/-adaptive and +/-fixed. Current or lags0,2,4 outputs (0/.4/.8s); all main models include same two causal query widths. No exact score/true direction/config/mode/frame/deadline in main input.',
        models=['center_current', 'center_history', 'five_current', 'five_history', 'metadata_only', 'shuffled_sequence'],
        supervision='One last causal output per train contact event, weight1; clear_all negatives weighted to identical frame histogram and total mass. Unknown/uncovered rows not negatives.',
        folds_note='3 outer folds; each complete unit evaluated once. Within remaining strata ~1/4 calibration, rest training. No automatic early stopping or parameter/model selection on heldout.',
        calibration='Control-time FA caps2.5%,5%; threshold only from calibration. Secondary test-control matched cap is descriptive; retain every fold actual rate.',
        baselines=['original center', 'adaptive max', 'three-way mean', 'exact direction reference'],
        negative_controls='Metadata-only frame/batch/mode/turn/family is privileged diagnostic, not candidate. Shuffle whole feature sequences within train stratum/family without label access. Scramble side-history vectors within role/stratum/family/time and matching current/lag adaptive widths (rounded .001 degree), keeping center and widths; report fraction with donors.',
        metrics='Original 0.9m event deadline any-before; unknown/silent using same conservative three-direction clear gate/tau; report mode/family, train fit, warmup-control FA and timely excluding warmup.',
        interpretation='Five-history vs center-history isolates extra direction under same learning recipe; current already includes M3 historical input. Unit split shares simulator templates; gains are not new-structure transfer.',
        decision='A >0 paired-unit CI for five_history minus center_history at descriptive matched2.5% both with and without warmup, concordant cal-only improvement with actualFA and warmupFA increase<=0.5pp, and loss on width-matched side pairing disruption motivates transfer test. Otherwise inspect which representation changes matter; never infer physical absence of information from probe failure.',
        decision_check='369 events1255controls across144 units; original head direction has >100-event headroom. Every unit evaluated once; bootstrap unit, not config/frame. One event smallest count change.',
        hashes={str(p.relative_to(R.ROOT)): R.sha(p) for p in paths}))
    print('PLAN saved', sklearn.__version__, [(len(f['train']), len(f['calibration']), len(f['evaluation'])) for f in make_folds(rows)], flush=True)


def load_data():
    allrows = R.read(H.R3/'rows.json'); keep = [i for i, r in enumerate(allrows) if r['unit'] in Q.A.UNITS]
    rows = [allrows[i] for i in keep]; n = len(rows)
    g = {k: v[keep] for k, v in E.load_geometry(allrows).items()}
    with np.load(H.R3/'online.npz') as z:
        gate, tau = z['gate'][keep, :, 1, 0], float(z['thresholds'][-1])
    d = np.zeros((n, 13, 10)); widths = np.zeros((n, 13, 2)); exact = np.zeros((n, 13)); cache = {}
    for i, r in enumerate(rows):
        u, c = r['unit'], r['config']
        if u not in cache:
            with np.load(Q.OUT/'units'/f'unit{u}.npz') as z:
                cache = {u: dict(z)}
        z = cache[u]
        d[i] = np.concatenate([R.smooth(z['head_'+key+'_raw'][0, c]) for key in ('c','ap','am','fp','fm')], axis=-1)
        widths[i, :, 0] = z['head_ap_k'][c, 3:]; widths[i, :, 1] = z['head_fp_k'][c, 3:]
        exact[i] = R.smooth(z['exact_raw'][0, c]).max(-1)
    center = np.concatenate([d[..., :2], widths], -1); five = np.concatenate([d, widths], -1)
    metadata = np.zeros((n, 13, 7))
    for i, r in enumerate(rows):
        metadata[i] = np.array([0, int(r['batch']), int(r['mode']),
                                int(r['turn']=='left'), int(r['turn']=='right'), int(r['turn']=='none'), int(r['config']>=20)])
    metadata[:, :, 0] = np.arange(13)
    features = dict(center_current=center, center_history=causal_features(center,(0,2,4)),
                    five_current=five, five_history=causal_features(five,(0,2,4)), metadata_only=metadata)
    triplet = d[..., :6].reshape(n,13,3,2).max(-1)
    base = dict(original_center=triplet[...,0], adaptive_max=triplet.max(-1), mean=triplet.mean(-1), exact=exact)
    return rows, g, features, base, gate, tau


def donors(rows, rolemask, seed):
    groups = {}; permutation = np.arange(len(rows)); rng = np.random.default_rng(seed)
    for i, r in enumerate(rows):
        if rolemask[i]:
            groups.setdefault((r['batch'],r['mode'],r['turn'],r['config']>=20), []).append(i)
    for key in sorted(groups):
        ids = np.array(groups[key]); permutation[ids] = rng.permutation(ids)
    return permutation


def scramble_sides(x, rows, rolemask, seed):
    """Swap coherent side-history vectors at matched causal width coordinates."""
    out=x.copy(); rng=np.random.default_rng(seed); changed=0; total=int(np.sum(rolemask))*13
    for t in range(13):
        groups={}
        for i in np.flatnonzero(rolemask):
            r=rows[i]
            key=(r['batch'],r['mode'],r['turn'],r['config']>=20,*np.round(x[i,t,[10,22,34]],3))
            groups.setdefault(key,[]).append(i)
        for ids in groups.values():
            if len(ids)<2:
                continue
            ids=rng.permutation(ids); donor=np.roll(ids,1); changed+=len(ids)
            for off in (0,12,24):
                out[ids,t,off+2:off+10]=x[donor,t,off+2:off+10]
    return out,dict(matched_changed_frames=changed,total_frames=total)


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits
    tick=time.monotonic(); plan=R.read(OUT/'PLAN.json')
    if (OUT/'result.json').exists():
        raise FileExistsError('Preserve result')
    for p,h in plan['hashes'].items():
        assert R.sha(R.ROOT/p)==h,p
    rows,g,features,base,gate,tau=load_data(); n=len(rows)
    uid=np.array([r['unit'] for r in rows]); cfg=np.array([r['config'] for r in rows])
    deadline=E.causal_index(g['fraction']); contact,control=g['contact'],g['control']; rr=np.flatnonzero(contact)
    model_names=list(features)+['shuffled_sequence','five_history_side_scrambled']
    keys=list(base)+model_names
    oof={key:np.zeros((n,13)) for key in keys}; folds_idx=np.full(n,-1,int)
    decisions={}; fold_records=[]; metrics={}; events={}; events_post={}; training=[]; pairing=[]
    signals={}
    (OUT/'models').mkdir(exist_ok=True)
    def check():
        if time.monotonic()-tick>plan['budget_cpu_analysis_wall_seconds']:
            raise TimeoutError('CPU analysis budget exhausted')

    def eval_score(score,theta,mask):
        alarm=score>=theta; clear=~alarm & gate & (base['adaptive_max']<=tau)
        tm=np.zeros(n,bool); tm[rr]=np.maximum.accumulate(alarm,axis=1)[rr,deadline[rr]]
        sl=np.zeros(n,bool); sl[rr]=~tm[rr]&clear[rr,deadline[rr]]
        late=alarm.copy(); late[:,:2]=False
        post=np.zeros(n,bool); post[rr]=np.maximum.accumulate(late,axis=1)[rr,deadline[rr]]
        ct=mask&control; ec=mask&contact
        return dict(events=int(ec.sum()),controls=int(ct.sum()), timely=int((tm&ec).sum()),
            silent=int((sl&ec).sum()), unknown_miss=int((ec&~tm&~sl).sum()),
            false_alarm=float(alarm[ct,2:12].mean()),unknown_time=float((~alarm&~clear)[ct,2:12].mean()),
            warmup_false_alarm=float(alarm[ct,:2].mean()), timely_excluding_warmup=int((post&ec).sum())),tm,sl,alarm,~alarm&~clear

    with threadpool_limits(limits=4):
        for fold,f in enumerate(plan['folds']):
            check(); masks={r:np.isin(uid,units) for r,units in f.items()}; tr,ca,te=[masks[k] for k in ('train','calibration','evaluation')]
            assert not ((tr&ca)|(tr&te)|(ca&te)).any(); folds_idx[te]=fold
            ri,ti,y,w=training_rows(contact,control,deadline,tr)
            predictions={k:v for k,v in base.items()}; fitted={}
            for name in list(features)+['shuffled_sequence']:
                check(); start=time.monotonic()
                x=features['five_history' if name=='shuffled_sequence' else name]
                trainx=x
                if name=='shuffled_sequence':
                    trainx=x[donors(rows,tr&(contact|control),SEED+fold+100)]
                model=HistGradientBoostingClassifier(**{k:v for k,v in plan['model'].items() if k!='name'},random_state=SEED+fold)
                model.fit(trainx[ri,ti],y,sample_weight=w)
                predictions[name]=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(n,13)
                fitted[name]=model
                training.append(dict(fold=fold,name=name,positive_events=int(y.sum()),samples=len(y),
                    train_auc=float(roc_auc_score(y,model.predict_proba(trainx[ri,ti])[:,1],sample_weight=w)),
                    seconds=time.monotonic()-start))
                joblib.dump(model,OUT/'models'/f'fold{fold}_{name}.joblib')
                print('fit',fold,name,round(training[-1]['train_auc'],3),round(training[-1]['seconds'],2),flush=True)
            # Width-matched pairing ablation; no label or future-width access.
            x=features['five_history']; ab=x.copy()
            for role,mask in masks.items():
                changed,record=scramble_sides(x,rows,mask,SEED+fold+200)
                ab[mask]=changed[mask];pairing.append(dict(fold=fold,role=role,**record))
            predictions['five_history_side_scrambled']=fitted['five_history'].predict_proba(ab.reshape(-1,ab.shape[-1]))[:,1].reshape(n,13)
            for name,score in predictions.items():
                oof[name][te]=score[te]
                for target in (.025,.05):
                    for kind,cmask in [('calibrated',ca),('matched_eval_descriptive',te)]:
                        tag=f'{kind}/{target:.3f}/{name}'
                        theta=P.threshold_at_cap(score[cmask&control,2:12],target)
                        met,tm,sl,al,unk=eval_score(score,theta,te)
                        fold_records.append(dict(fold=fold,key=tag,threshold=theta,fit_fa=float((score[cmask&control,2:12]>=theta).mean()),**met))
                        events.setdefault(tag,np.zeros(n,bool))[te]=tm[te]
                        late=al.copy();late[:,:2]=False
                        post=np.zeros(n,bool);post[rr]=np.maximum.accumulate(late,axis=1)[rr,deadline[rr]]
                        events_post.setdefault(tag,np.zeros(n,bool))[te]=post[te]
                        entry=signals.setdefault(tag,dict(silent=np.zeros(n,bool),alarm=np.zeros((n,13),bool),unknown=np.zeros((n,13),bool)))
                        entry['silent'][te]=sl[te]; entry['alarm'][te]=al[te]; entry['unknown'][te]=unk[te]
            np.savez_compressed(OUT/f'fold{fold}_checkpoint.npz',evaluation=te,**{k:v[te] for k,v in predictions.items()})
    assert (folds_idx>=0).all()
    groups={'all':np.ones(n,bool),'none':cfg<20,'corner':cfg>=20}
    groups.update({f'mode{k}':np.array([r['mode']==k for r in rows]) for k in range(3)})
    for tag,tm in events.items():
        s=signals[tag]; metrics[tag]={}
        for group,mask in groups.items():
            ec=mask&contact; ct=mask&control
            post=s['alarm'].copy();post[:,:2]=False
            late=np.zeros(n,bool);late[rr]=np.maximum.accumulate(post,axis=1)[rr,deadline[rr]]
            metrics[tag][group]=dict(events=int(ec.sum()),controls=int(ct.sum()),timely=int((tm&ec).sum()),
                silent=int((s['silent']&ec).sum()),unknown_miss=int((ec&~tm&~s['silent']).sum()),
                false_alarm=float(s['alarm'][ct,2:12].mean()),unknown_time=float(s['unknown'][ct,2:12].mean()),
                warmup_false_alarm=float(s['alarm'][ct,:2].mean()),timely_excluding_warmup=int((late&ec).sum()))
    units,inv=np.unique(uid,return_inverse=True); boot=np.zeros((1000,len(units)),int);rng=np.random.default_rng(SEED+300)
    strata={r['unit']:(r['batch'],r['mode'],r['turn']) for r in rows}
    for st in sorted(set(strata.values())):
        ids=np.array([j for j,u in enumerate(units) if strata[u]==st])
        for j,draw in enumerate(rng.integers(0,len(ids),(1000,len(ids)))):
            boot[j,ids]=np.bincount(draw,minlength=len(ids))
    diffs={}
    for kind in ('calibrated','matched_eval_descriptive'):
        for target in (.025,.05):
            prefix=f'{kind}/{target:.3f}/'
            for a,b in [('five_history','center_history'),('five_current','center_current'),('center_history','center_current'),
                        ('five_history','five_current'),('five_history','original_center'),('five_history','shuffled_sequence'),
                        ('five_history','five_history_side_scrambled')]:
                delta=events[prefix+a].astype(int)-events[prefix+b].astype(int)
                sums=np.bincount(inv,weights=delta,minlength=len(units))
                diffs[prefix+a+'-'+b]=dict(diff=int(sums.sum()),ci95=np.quantile(boot@sums,[.025,.975]).tolist())
            delta=events_post[prefix+'five_history'].astype(int)-events_post[prefix+'center_history'].astype(int)
            sums=np.bincount(inv,weights=delta,minlength=len(units))
            diffs[prefix+'nowarmup/five_history-center_history']=dict(diff=int(sums.sum()),ci95=np.quantile(boot@sums,[.025,.975]).tolist())
    m='matched_eval_descriptive/0.025/';c='calibrated/0.025/'
    decisions['directional_gain_candidate']=bool(diffs[m+'five_history-center_history']['ci95'][0]>0 and
        diffs[m+'nowarmup/five_history-center_history']['ci95'][0]>0 and
        diffs[c+'five_history-center_history']['diff']>0 and metrics[c+'five_history']['all']['false_alarm']<=metrics[c+'center_history']['all']['false_alarm']+.005 and
        metrics[c+'five_history']['all']['warmup_false_alarm']<=metrics[c+'center_history']['all']['warmup_false_alarm']+.005 and
        diffs[m+'five_history-five_history_side_scrambled']['diff']>0)
    check()
    np.savez_compressed(OUT/'ledger.npz',unit=uid,config=cfg,fold=folds_idx,contact=contact,control=control,deadline=deadline,gate=gate,
        **{'score/'+k:v for k,v in oof.items()},**{'timely/'+k:v for k,v in events.items()},
        **{'silent/'+k:v['silent'] for k,v in signals.items()})
    save('result.json',dict(status='EXPLORATORY_COMPLETE',decision=decisions,seconds=time.monotonic()-tick,
        units=len(units),sequences=n,events=int(contact.sum()),controls=int(control.sum()),fold_records=fold_records,
        metrics=metrics,diffs=diffs,training=training,side_pairing=pairing,clear_tau=tau,plan_sha256=R.sha(OUT/'PLAN.json')))
    for name in keys:
        print('RESULT',name,metrics[c+name]['all'],metrics[m+name]['all'],flush=True)
    print('DECISION',decisions,'seconds',round(time.monotonic()-tick,2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.stage=='freeze' else run()
