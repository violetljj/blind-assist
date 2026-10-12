"""Consumed Development only: fixed visit CV, mixed-quality fusion diagnostic.

Uses cached observation CNH expectations, never evaluator geometry as a feature.
Each command has a wall cap; all heavyweight outputs remain in artifacts.local.
"""
from __future__ import annotations
import argparse, csv, json, time, hashlib, shutil, warnings
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sync_fusion_features_v1 import FEATURES, BANDS, features_for_depth, quality
from sync_fusion_v1_train_cal import HGB, SEEDS, enumerate_ties, select
from sync_rgb_tof_dataset_v1 import peak_readout, sample
from rgb_body_query_reference_eval import rays, ray_interval

REPO = Path(__file__).resolve().parents[4]
BASE = REPO/'artifacts.local/work'
ROOT = BASE/'sync-fusion-robust-dev-20261011'
PLAN = Path(__file__).with_name('SYNC_FUSION_ROBUST_PLAN_DEV_20261011.json')
ARMS = ('native_perturbed','faro_rho015_ambient1','faro_rho060_ambient1',
        'faro_rho030_ambient3','faro_rho030_ambient1','faro_rho030_ambient10')
MODELS = ('A','B','C','D','rgb','tof')
INTERACTIONS = ('tof_margin_x_unknown','tof_margin_x_support','tof_margin_x_peak_snr_mean')
TARGETS = (.02,.05,.10)

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n',encoding='utf8')
def table(p,rows):
    with Path(p).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def data_roots():
    return ((BASE/'sync-rgb-tof-dataset-v1-dev-20261011',
             BASE/'sync-rgb-tof-dataset-v1-1-dev-20261011',BASE/'sync-fusion-v1-dev-20261011'),
            (BASE/'sync-fusion-confirm-v2-dev-20261011',)*3)

def make_plan():
    assert not (ROOT/'PLAN.json').exists(), 'Do not replace a fixed PLAN'
    visits=[];bindings=[];roster=[]
    for publicroot,synthroot,_ in data_roots():
        p=publicroot/'public_roster.json';r=load(p)['rows'];roster.extend(r)
        for name,root in [('public_roster.json',publicroot),('synthesis_manifest.json',synthroot),
                          ('input_coverage_gate.csv',synthroot),('frozen_readout_seal.json',synthroot)]:
            path=root/name;bindings.append(dict(path=str(path.resolve()),sha256=sha(path)))
    visits=sorted({str(r['visit_id']) for r in roster});assert len(visits)==30 and len(roster)==960
    assert all(sum(str(r['visit_id'])==v for r in roster)==32 for v in visits)
    ordered=np.random.default_rng(20261011).permutation(visits).tolist();folds=[]
    for i in range(5):
        ev=ordered[i*6:(i+1)*6];remaining=sorted(set(visits)-set(ev))
        cal=np.random.default_rng(2026101100+i).permutation(remaining)[:5].tolist()
        folds.append(dict(fold=i,eval=sorted(ev),cal=sorted(cal),train=sorted(set(remaining)-set(cal))))
    p=dict(task='SYNC_FUSION_ROBUST_DEV_20261011',phase='EXPLORE_CONSUMED_DEVELOPMENT',
      parent_commit='1d16bde2',budget=dict(CPU_command_wall_s=1800,GPU_wall_s=300,
      allocations=dict(build=650,fit=450,audit=180,delivery_and_reserve=520)),
      data=dict(visits=visits,frames=960,queries_per_frame=27,fold_seed=20261011,folds=folds,
      inner_cal='5 of remaining24 visits: nearest integer to one fifth; train19. No stratification by labels/results.',
      bindings=bindings,protected_access=False),
      arms=list(ARMS),arm_contract=dict(native='Inherited v1.1/v2 native_perturbed K2, residual pool unchanged',
        quality_arms='Inherited v1.1 FARO geometry, all960 frames including missing geometry/pose; no native fill',
        FARO_gate='Only faro_rho030_ambient1: frozen K0 joint_pass_zones>=52, other arms unfiltered',
        rho015='rho=.15 ambient1; expectation=.5 times cached FARO rho.3 expectation',
        rho060='rho=.6 ambient1; expectation=2 times cached FARO rho.3 expectation',
        ambient3='rho=.3 ambient3; cached expectation unchanged and ambient times3',
        ambient10='rho=.3 ambient10; cached expectation unchanged and ambient times10; no artificial dropout. Stress severity measured, not assumed.',
        sampling='Frozen sample: signed Poisson(E+8ambient)-Poisson(8ambient), same per-frame/K seed as source; rho/ambient linear scaling is exact at noise_scale0. Old existing arms reused byte-identically.',
        K='K0/K1 arithmetic mean features; raw margin missing if either K missing; one query row per arm',
        readout='Frozen peak_readout SNR>=3 coverage>=.75 and16th query margin; RGB cached raw DAV/Uni unchanged'),
      features=dict(base=list(FEATURES),C_extra=list(INTERACTIONS),
        C_definition='Clipped model ToF margin times respectively unknown_fraction, support_pixels, mean peak SNR across all64 zones then K mean (same quality function); only these products added, no standalone peak quality added.',
        excluded='Reference states/depths, geometric truth, expectation/ambient/source identity, visit/frame IDs never features'),
      models=dict(A='Logistic v2 recipe, native-only training',B='Same Logistic, six arms mixed',
        C='B plus fixed3 interactions',D=dict(recipe='HGB mixed',parameters=HGB),seeds=list(SEEDS),
        logistic=dict(C=1,solver='lbfgs',max_iter=1000,class_weight=None),
        weight='Known POS/FREE only: A weight1; B/C/D each arm sum=Nknown/6; per-row weight=Nknown/(6*nknown_arm), mean1. Each query contributes once per available arm. UNKNOWN excluded.',
        standardizer='Train-known-only StandardScaler; B/C weighted with same arm weights; A unweighted. Three seed positive probabilities averaged; no hyperparameter selection.'),
      calibration=dict(targets=list(TARGETS),mixture='Concatenate all six available arms, raw query counts, same for all models and both single sources; no per-arm cuts',
        rule='All finite ties plus +inf; F<=floor(target*FREE), maximizeW then minimizeF then highest threshold; zeroFREE/POS => reject all.',
        baseline='RGB-only Uni near/DAV midfar and ToF-only raw mean margin; same fold/band mixture thresholds as fusion. Existing sensor readout unchanged; fusion operating points newly CV calibrated.'),
      metrics=dict(W='Prediction positive AND query POS',F='Prediction positive AND strict sampled FREE',
        paired='Each A/B/C/D vs RGB and same-arm ToF: POS rescue/loss/net and FREE added/removed',
        bootstrap=dict(seed=20261011,replicates=2000,cluster='All30 visits pooled;6 visits per fold; zero-row FARO visits retained, zero denominators omitted with valid count',
          scope='Resample fixed OOF predictions, not full retraining uncertainty',interval='percentile2.5/97.5 for W/POS,F/FREE,net count/net pp'),
        best='Fixed pooled native middle best single by higherW, then fewerF, thenRGB; same comparator for all model retention and bootstrap',
        retention='Delta middle to fixed best single divided by A delta on same OOF data; if A<=0 NOT_EVALUABLE. Per-fold own best identity and A denominator; no averaging fold ratios.',
        candidate='B/C/D: pooled FARO near net vsRGB>=0 AND ambient10 near net vsRGB>=0 AND native middle delta>=.8*A delta; descriptive, not confirmatory/noninferiority gate'),
      interpretation='Existing residuals and reference-enriched consumed visits predate CV; outer split isolates current fits/calibration only. New exploratory method choice cannot alter v2 confirmation. No hardware/safety claim.',
      decision_check='RGB headroom from v2 FARO near53%/mid62%; smallest event1. Count gains always accompanied by FREE cost/CI. Revisit next fresh confirmation only if descriptive robust candidate; no per-arm threshold, subset or feature retuning on OOF.')
    if PLAN.exists():assert load(PLAN)==p, 'Mechanical plan recovery only'
    else:save(PLAN,p)
    ROOT.mkdir(parents=True,exist_ok=True);shutil.copyfile(PLAN,ROOT/'PLAN.json')
    table(ROOT/'folds.csv',[dict(visit_id=v,fold=f['fold'],role=role) for f in folds for role in ('train','cal','eval') for v in f[role]])
    print(json.dumps(dict(plan_sha256=sha(PLAN),folds=folds)))

def cached_features(root,roles):
    bysource={}
    for source in ('native_perturbed','faro_rho030_ambient1'):
        out={}
        for role in roles:
            path=root/'features'/source/(role+'.npz')
            with np.load(path,allow_pickle=False) as z:
                assert tuple(z['feature_names'])==FEATURES
                cached={k:z[k] for k in ('X','tof_score','rgb_score','frame_id','query_id')}
            for j,key in enumerate(zip(cached['frame_id'],cached['query_id'])):
                assert key not in out
                out[key]={k:cached[k][j] for k in ('X','tof_score','rgb_score')}
        bysource[source]=out
    return bysource

def build(check):
    p=load(PLAN);assert sha(PLAN)==sha(ROOT/'PLAN.json');rows=[];inputs=[];sim=[];parity=0
    def record(path):
        path=Path(path);inputs.append(dict(path=str(path.resolve()),sha256=sha(path)))
    for bi,(publicroot,synthroot,featureroot) in enumerate(data_roots()):
        public=load(publicroot/'public_roster.json');queries=public['queries'];pub={r['source_id']:r for r in public['rows']}
        refs={r['source_id']:r for r in load(publicroot/'reference_roster.json')['rows']}
        record(publicroot/'reference_roster.json');record(publicroot/'public_roster.json')
        synth=load(synthroot/'synthesis_manifest.json');seal=load(synthroot/'frozen_readout_seal.json')
        with (synthroot/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:
            allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
        cache=cached_features(featureroot,('train','cal','eval') if bi==0 else ('cal','eval'))
        for role in (('train','cal','eval') if bi==0 else ('cal','eval')):
            for source in ('native_perturbed','faro_rho030_ambient1'):record(featureroot/'features'/source/(role+'.npz'))
        for fi,frame in enumerate(synth['frames']):
            check();r=pub[frame['source_id']];K=np.asarray(r['depth_K']);shape=tuple(r['depth_shape']);rx,ry=rays(K,shape)
            intervals=[ray_interval(rx,ry,q) for q in queries]
            labels={q['name']:q for q in refs[frame['source_id']]['queries']}
            for arm in ARMS:
                if arm=='faro_rho030_ambient1' and frame['frame_id'] not in allowed:continue
                observations=[];qualities=[];validzones=[]
                for repeat in (0,1):
                    existing=next((o for o in frame['arms'] if o['arm']==arm and o['repeat']==repeat),None)
                    source=existing or next(o for o in frame['arms'] if o['arm']=='faro_rho030_ambient1' and o['repeat']==repeat)
                    record(source['path'])
                    with np.load(source['path'],allow_pickle=False) as z:f={k:z[k] for k in z.files}
                    if existing is None:
                        signal_scale=.5 if arm=='faro_rho015_ambient1' else 2 if arm=='faro_rho060_ambient1' else 1
                        ambient_scale=3 if arm=='faro_rho030_ambient3' else 10 if arm=='faro_rho030_ambient10' else 1
                        hist,counts,bg=sample(f['expectation'][None]*signal_scale,f['ambient'][None]*ambient_scale,int(f['seed']))
                        f.update(hist=hist[0],counts=counts[0],background=bg[0],expectation=f['expectation']*signal_scale,ambient=f['ambient']*ambient_scale)
                        path=ROOT/'simulated'/frame['source_id']/(arm+f'_k{repeat}.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        if path.exists():
                            with np.load(path,allow_pickle=False) as previous:
                                for name,value in f.items():np.testing.assert_array_equal(previous[name],value)
                        else:np.savez_compressed(path,**f)
                        sim.append(dict(path=str(path.resolve()),sha256=sha(path),source=source['path'],source_sha256=source['sha256'],arm=arm,signal_scale=signal_scale,ambient_scale=ambient_scale,seed=int(f['seed'])))
                    else:assert sha(source['path'])==source['sha256']
                    if np.isfinite(f['grid_pose']).all():
                        depth,snr,radial,valid=peak_readout(f['hist'],f['background'],f['coverage'],f['grid_K'],shape,f['grid_pose'],K,shape,f['rgb_pose'],seal)
                        validzones.append(float(valid.mean()))
                    else:depth=np.full(shape,np.inf);validzones.append(0.)
                    qualities.append(quality(f['hist'],f['background'],f['coverage']))
                    observations.append([features_for_depth(depth,*it) for it in intervals])
                for j,q in enumerate(queries):
                    key=(frame['source_id'],q['name']);cached=cache['native_perturbed'][key];s0,v0=observations[0][j];s1,v1=observations[1][j]
                    both=np.isfinite(s0) and np.isfinite(s1);ts=(s0+s1)/2 if both else -np.inf;tv=(v0+v1)/2;tv[0]=ts if both else -10.;tv[1]=float(not both)
                    X=cached['X'].copy();X[:9]=tv[[0,1,4,3,6,5,7,8,9]];X[9]=int(np.isfinite(s0))+int(np.isfinite(s1))
                    for c,name in enumerate(FEATURES):
                        if name.endswith('_m'):X[c]=np.clip(X[c],-10,10)
                    if arm in cache:
                        np.testing.assert_array_equal(X,cache[arm][key]['X']);assert ts==cache[arm][key]['tof_score'];parity+=1
                    qr=labels[q['name']];state=qr['state'];y=1 if state in ('POS','POSITIVE') else 0 if state in ('FREE','FREE_ON_SAMPLED_RAYS') else -1
                    if state=='FREE_ON_SAMPLED_RAYS':assert qr['unknown_pixels']==0 and qr['positive_pixels']==0 and qr['free_ray_pixels']==qr['domain_pixels']
                    rows.append(dict(X=X,arm=arm,visit_id=str(frame['visit_id']),frame_id=frame['source_id'],query_id=q['name'],band=int(X[16]),y=y,tof_score=ts,rgb_score=cached['rgb_score'],peak_quality=float(np.mean(qualities,axis=0)[0]),accepted_zone_fraction=float(np.mean(validzones))))
            if fi%64==0:print('features',bi,fi,len(rows),flush=True)
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};assert len(set(zip(arrays['arm'],arrays['frame_id'],arrays['query_id'])))==len(rows)
    assert set(arrays['visit_id'])==set(p['data']['visits']);np.savez_compressed(ROOT/'features.npz',**arrays,feature_names=np.asarray(FEATURES))
    unique={r['path']:r for r in inputs};save(ROOT/'feature_manifest.json',dict(rows=len(rows),arms={a:int((arrays['arm']==a).sum()) for a in ARMS},parity_queries=parity,features_sha256=sha(ROOT/'features.npz'),inputs=list(unique.values()),simulations=sim,PLAN_sha256=sha(PLAN)))
    print('features COMPLETE',len(rows),'parity',parity,flush=True)

def design(d,model):
    X=d['X']
    if model=='C':X=np.column_stack((X,X[:,0]*X[:,3],X[:,0]*X[:,2],X[:,0]*d['peak_quality']))
    return X

def fit(check):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    with np.load(ROOT/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    p=load(PLAN);n=len(d['y']);oof=np.full((n,len(MODELS)),np.nan);pred=np.zeros_like(oof,dtype=bool);foldid=np.full(n,-1,int);cuts=[];entries=[];coefs=[];influence=[]
    with threadpool_limits(limits=2):
        for fold in p['data']['folds']:
            f=fold['fold'];ev=np.flatnonzero(np.isin(d['visit_id'],fold['eval']));cal=np.flatnonzero(np.isin(d['visit_id'],fold['cal']));tr=np.flatnonzero(np.isin(d['visit_id'],fold['train'])&(d['y']>=0));foldid[ev]=f
            folder=ROOT/f'fold{f}';folder.mkdir(exist_ok=False);cal_scores=np.empty((len(cal),len(MODELS)));eval_scores=np.empty((len(ev),len(MODELS)))
            for mi,model in enumerate(MODELS):
                check()
                if model in ('rgb','tof'):
                    s=d[model+'_score'];cal_scores[:,mi]=s[cal];eval_scores[:,mi]=s[ev];continue
                use=tr[d['arm'][tr]==ARMS[0]] if model=='A' else tr
                weights=np.ones(len(use))
                if model!='A':
                    for arm in ARMS:
                        ix=d['arm'][use]==arm;assert ix.any();weights[ix]=len(use)/(len(ARMS)*ix.sum())
                np.savez_compressed(folder/(model+'_training.npz'),indices=use,weights=weights)
                X=design(d,model);scaler=None
                if model!='D':scaler=StandardScaler().fit(X[use],sample_weight=weights);xt=scaler.transform(X[use]);xc=scaler.transform(X[cal]);xe=scaler.transform(X[ev])
                else:xt=X[use];xc=X[cal];xe=X[ev]
                cs=[];es=[];models=[]
                for seed in SEEDS:
                    check();m=HistGradientBoostingClassifier(**HGB,random_state=seed) if model=='D' else LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,class_weight=None,random_state=seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');m.fit(xt,d['y'][use],sample_weight=weights)
                    path=folder/(model+f'_{seed}.joblib');joblib.dump(dict(estimator=m,scaler=scaler,features=list(FEATURES)+ (list(INTERACTIONS) if model=='C' else [])),path)
                    entries.append(dict(fold=f,model=model,seed=seed,path=str(path.resolve()),sha256=sha(path),known_rows=len(use),weight_sum=float(weights.sum()),arm_weight_sums={a:float(weights[d['arm'][use]==a].sum()) for a in ARMS},warnings=[str(w.message) for w in caught]))
                    cs.append(m.predict_proba(xc)[:,1]);es.append(m.predict_proba(xe)[:,1]);models.append(m)
                    if model!='D':
                        for j,name in enumerate(list(FEATURES)+(list(INTERACTIONS) if model=='C' else [])):
                            coefs.append(dict(fold=f,model=model,seed=seed,feature=name,standardized_coefficient=float(m.coef_[0,j]),raw_coefficient=float(m.coef_[0,j]/scaler.scale_[j])))
                cal_scores[:,mi]=np.mean(cs,axis=0);eval_scores[:,mi]=np.mean(es,axis=0)
                # Fixed, observation-only local response: increase modeled margin by .01m;
                # for C recompute products. Quality correlations do not prove physical gating.
                xd=X[ev].copy();xd[:,0]+= .01
                if model=='C':xd[:,-3]=xd[:,0]*xd[:,3];xd[:,-2]=xd[:,0]*xd[:,2];xd[:,-1]=xd[:,0]*d['peak_quality'][ev]
                response=np.mean([m.predict_proba(scaler.transform(xd) if scaler else xd)[:,1] for m in models],axis=0)-eval_scores[:,mi]
                for qi,qname in ((3,'unknown_fraction'),(2,'support_pixels')):
                    vals=d['X'][ev,qi];lo,hi=np.quantile(vals,[.25,.75])
                    for group,mask in [('low',vals<=lo),('high',vals>=hi)]:
                        influence.append(dict(fold=f,model=model,quality=qname,group=group,cut=float(lo if group=='low' else hi),rows=int(mask.sum()),mean_probability_delta_per_01m=float(response[mask].mean())))
                vals=d['peak_quality'][ev];lo,hi=np.quantile(vals,[.25,.75])
                for group,mask in [('low',vals<=lo),('high',vals>=hi)]:influence.append(dict(fold=f,model=model,quality='peak_snr_mean',group=group,cut=float(lo if group=='low' else hi),rows=int(mask.sum()),mean_probability_delta_per_01m=float(response[mask].mean())))
                if model=='D':
                    # Permutation importance on inner calibration only, weighted logloss,
                    # fixed RNG/no selection. Native HGB has no coef_ or impurity importance.
                    known=d['y'][cal]>=0;yy=d['y'][cal][known];orig=cal_scores[known,mi];rng=np.random.default_rng(20261011+f)
                    def loss(s):s=np.clip(s,1e-12,1-1e-12);return float(-np.mean(yy*np.log(s)+(1-yy)*np.log(1-s)))
                    base=loss(orig)
                    for j,name in enumerate(FEATURES):
                        check();xx=xc[known].copy();xx[:,j]=xx[rng.permutation(len(xx)),j];ss=np.mean([m.predict_proba(xx)[:,1] for m in models],axis=0)
                        coefs.append(dict(fold=f,model=model,seed=-1,feature=name,standardized_coefficient=None,raw_coefficient=None,cal_permutation_logloss_increase=loss(ss)-base))
                print('fit',f,model,len(use),flush=True)
            for mi,model in enumerate(MODELS):
                for b in range(3):
                    take=d['band'][cal]==b;ties=enumerate_ties(cal_scores[take,mi],d['y'][cal][take]);cut=select(ties,d['y'][cal][take],TARGETS[b]);cuts.append(dict(fold=f,model=model,band=b,**cut))
                    save(folder/f'ties_{model}_{b}.json',ties);t=np.inf if cut['threshold'] is None else cut['threshold'];pred[ev[d['band'][ev]==b],mi]=eval_scores[d['band'][ev]==b,mi]>=t
            np.savez_compressed(folder/'cal_scores.npz',indices=cal,scores=cal_scores,models=np.asarray(MODELS));oof[ev]=eval_scores
    assert (foldid>=0).all() and not np.isnan(oof).any()
    np.savez_compressed(ROOT/'oof.npz',indices=np.arange(n),scores=oof,pred=pred,fold=foldid,models=np.asarray(MODELS))
    save(ROOT/'cuts.json',cuts);save(ROOT/'models_manifest.json',entries);save(ROOT/'coefficients.json',coefs);table(ROOT/'influence.csv',influence)
    with (ROOT/'per_query.csv').open('w',newline='',encoding='utf8') as stream:
        names=['fold','arm','visit_id','frame_id','query_id','band','y','peak_quality','accepted_zone_fraction']+[m+'_score' for m in MODELS]+[m+'_pred' for m in MODELS]
        w=csv.DictWriter(stream,fieldnames=names);w.writeheader()
        for i in range(n):
            r={k:d[k][i].item() for k in names[:9] if k!='fold'};r['fold']=int(foldid[i]);r.update({m+'_score':float(oof[i,j]) for j,m in enumerate(MODELS)});r.update({m+'_pred':int(pred[i,j]) for j,m in enumerate(MODELS)});w.writerow(r)
    print('OOF COMPLETE',n,flush=True)

def ci(values):
    finite=np.isfinite(values);a=values[finite]
    return dict(lo=float(np.quantile(a,.025)) if len(a) else None,hi=float(np.quantile(a,.975)) if len(a) else None,valid=int(len(a)))
def summarize(check):
    with np.load(ROOT/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT/'oof.npz',allow_pickle=False) as z:o={k:z[k] for k in z.files}
    pred=o['pred'];metrics=[];pairs=[];candidates=[];retention=[]
    for fold in (-1,0,1,2,3,4):
        visits=load(PLAN)['data']['visits'] if fold==-1 else load(PLAN)['data']['folds'][fold]['eval']
        rng=np.random.default_rng(20261011);samples=rng.integers(0,len(visits),size=(2000,len(visits)));mult=np.eye(len(visits),dtype=int)[samples].sum(1)
        def boot(mask,values):
            totals=np.array([values[mask&(d['visit_id']==v)].sum() for v in visits]);return mult@totals
        maskfold=np.ones(len(d['y']),bool) if fold==-1 else o['fold']==fold
        for arm in ARMS:
            for b in range(3):
                mask=maskfold&(d['arm']==arm)&(d['band']==b);pos=mask&(d['y']==1);free=mask&(d['y']==0);unknown=mask&(d['y']==-1)
                pb=boot(mask,(d['y']==1).astype(int));fb=boot(mask,(d['y']==0).astype(int))
                for mi,m in enumerate(MODELS):
                    check();p=pred[:,mi];wb=boot(mask,(p&(d['y']==1)).astype(int));ffb=boot(mask,(p&(d['y']==0)).astype(int))
                    with np.errstate(divide='ignore',invalid='ignore'):wc=ci(100*wb/pb);fc=ci(100*ffb/fb)
                    metrics.append(dict(fold=fold,arm=arm,band=b,model=m,POS=int(pos.sum()),W=int(p[pos].sum()),FREE=int(free.sum()),F=int(p[free].sum()),UNKNOWN=int(unknown.sum()),U=int(p[unknown].sum()),W_rate_ci=wc,F_rate_ci=fc))
                    if m in ('rgb','tof'):continue
                    for base in ('rgb','tof'):
                        bp=pred[:,MODELS.index(base)];res=pos&p&~bp;loss=pos&~p&bp;delta=((p.astype(int)-bp.astype(int))*(d['y']==1));bb=boot(mask,delta)
                        with np.errstate(divide='ignore',invalid='ignore'):bc=ci(100*bb/pb)
                        pairs.append(dict(fold=fold,arm=arm,band=b,model=m,baseline=base,rescue=int(res.sum()),loss=int(loss.sum()),net=int(res.sum()-loss.sum()),FREE_added=int((free&p&~bp).sum()),FREE_removed=int((free&~p&bp).sum()),net_count_ci=ci(bb),net_pp_ci=bc))
        middle=[r for r in metrics if r['fold']==fold and r['arm']==ARMS[0] and r['band']==1]
        singles=[r for r in middle if r['model'] in ('rgb','tof')];best=max(singles,key=lambda r:(r['W'],-r['F'],r['model']=='rgb'))['model']
        pp=[r for r in pairs if r['fold']==fold and r['arm']==ARMS[0] and r['band']==1 and r['baseline']==best];a=next(r['net'] for r in pp if r['model']=='A')
        for r in pp:
            ratio=r['net']/a if a>0 else None
            # Paired bootstrap ratio, comparator identity fixed on full fold/pooled data.
            mask=maskfold&(d['arm']==ARMS[0])&(d['band']==1);bp=pred[:,MODELS.index(best)];mb=boot(mask,((pred[:,MODELS.index(r['model'])].astype(int)-bp.astype(int))*(d['y']==1)));ab=boot(mask,((pred[:,0].astype(int)-bp.astype(int))*(d['y']==1)))
            with np.errstate(divide='ignore',invalid='ignore'):rat=np.where(ab>0,mb/ab,np.nan)
            retention.append(dict(fold=fold,model=r['model'],best_single=best,A_net=a,model_net=r['net'],retention=ratio,retention_ci=ci(rat)))
            if fold==-1:
                nets={arm:next(x['net'] for x in pairs if x['fold']==-1 and x['arm']==arm and x['band']==0 and x['model']==r['model'] and x['baseline']=='rgb') for arm in (ARMS[4],ARMS[5])}
                candidates.append(dict(model=r['model'],FARO_near_net=nets[ARMS[4]],stress_near_net=nets[ARMS[5]],native_middle_net=r['net'],A_middle_net=a,retention=ratio,robust_candidate=r['model']!='A' and all(x>=0 for x in nets.values()) and ratio is not None and ratio>=.8))
    save(ROOT/'summary.json',dict(metrics=metrics,paired=pairs,retention=retention,candidates=candidates,PLAN_sha256=sha(PLAN),features_sha256=sha(ROOT/'features.npz'),oof_sha256=sha(ROOT/'oof.npz'),per_query_sha256=sha(ROOT/'per_query.csv')))
    table(ROOT/'metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in metrics]);table(ROOT/'paired.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in pairs]);print(json.dumps(candidates),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('plan','build','fit','summarize'));ap.add_argument('--budget-s',type=float,default=600);a=ap.parse_args();started=time.monotonic()
    def check():
        if time.monotonic()-started>=a.budget_s-5:raise TimeoutError('Stage command-wall cap')
    if a.stage=='plan':make_plan();return
    try:
        {'build':build,'fit':fit,'summarize':summarize}[a.stage](check)
        save(ROOT/(a.stage+'_terminal.json'),dict(status='COMPLETE',seconds=time.monotonic()-started,GPU_s=0,source_sha256=sha(__file__)))
    except Exception as exc:
        save(ROOT/(a.stage+'_failure.json'),dict(error=repr(exc),seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
