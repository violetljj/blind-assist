"""Post-result, consumed-Development addendum: prior baseline and local evidence.

Human's additional request is fixed separately; original A-D results untouched.
"""
import argparse,csv,json,time,hashlib,warnings
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sync_fusion_robust_dev import ROOT,PLAN,ARMS,MODELS,FEATURES,data_roots,load,save,sha,table
from sync_fusion_v1_train_cal import SEEDS,enumerate_ties,select
from sync_rgb_tof_dataset_v1 import peak_readout
from rgb_body_query_reference_eval import rays,ray_interval

ADDPLAN=Path(__file__).with_name('SYNC_FUSION_ROBUST_EVIDENCE_ADDENDUM_PLAN_DEV_20261011.json')
CATEGORIES=('neither','tof_only','rgb_only','both')
def fix_plan():
    assert not ADDPLAN.exists()
    p=dict(task='SYNC_FUSION_ROBUST_EVIDENCE_ADDENDUM_DEV_20261011',phase='POST_RESULT_EXPLORE',
      authorization=dict(thread='01a12702-557c-7823-8791-6182ba3e4ce1',human_turn='01a12714-4cf8-79b2-abce-85a6c3387ed5',
        scope='Human requested unfinished robust run add same prior baseline and evidence split; direct human message independently read. No no-position/gated model refit requested in this robust task.'),
      original=dict(PLAN_sha256=sha(PLAN),features_sha256=sha(ROOT/'features.npz'),oof_sha256=sha(ROOT/'oof.npz'),summary_sha256=sha(ROOT/'summary.json')),
      budget=dict(original_CPU_command_wall_limit_s=1800,original_GPU_wall_limit_s=300,additional_cap_s=350,audit_cap_s=90,no_budget_expansion=True),
      prior=dict(features=list(FEATURES[16:20]),indices=[16,17,18,19],recipe='B logistic C1/L2/lbfgs1000, seeds955/956/957; same train-known row indices/weights/scaler and allarm inner-cal ties/targets; no sensor feature',
        evaluation='Same five fixed outer folds; allarms use one same per-fold/band prior cut; original A-D models/cuts/predictions never changed'),
      evidence=dict(primary='ToF: each K own frozen projected peak depth has>=16 interval-inside pixel rays, require both K0/K1. RGB: selected Uni near/DAV midfar raw16th margin>=0. Four mutually exclusive classes per query.',
        K2_alternative='Also split using K2 arithmetic mean raw16th margin>=0; averaging can pass when only one K meets>=16. This is the zero-cut version of aggregated ToF-only score, different from strict bothK.',
        calibration_alternative='Also split using original fold ToF/RGB frozen cal boolean decisions; negative cal margin cuts may support absent raw interval intrusion; never conflate with raw minimum evidence.',
        source='Frozen original/generated CNH hist/background/coverage/poses; no reference pixels or new RGB inference; query ray intervals unchanged',
        interpretation='Neither raw minimum evidence is absent local evidence, not proof of pure prior causality. Coarse fractions/relative depths/other RGB model remain possible evidence. Pure prior baseline separately quantifies predictive scene-prior opportunity.'),
      outputs=['prior15models/prior_cuts','per_query_evidence.csv','prior_metrics','fourclass_count_tables including paired rescue/loss','independent_addon_audit'],
      restrictions='No old conclusion/model/cut/readout edits; no protected480/test; label y used only for own train/cal/evaluation roles already fixed. Post-result addition is not preregistered confirmation.')
    save(ADDPLAN,p);save(ROOT/'evidence_addendum_PLAN.json',p);print('ADDPLAN',sha(ADDPLAN))

def run(budget):
    started=time.monotonic();p=load(ADDPLAN)
    def check():
        if time.monotonic()-started>=budget-5:raise TimeoutError('Addendum shared budget cap')
    for name in ('features','oof','summary'):assert sha(ROOT/(name+'.npz' if name!='summary' else 'summary.json'))==p['original'][name+'_sha256']
    with np.load(ROOT/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT/'oof.npz',allow_pickle=False) as z:o={k:z[k] for k in z.files}
    n=len(d['y']);counts=np.zeros((n,2),int);groups={};sample_frames=[];source_paths=[]
    for i,key in enumerate(zip(d['frame_id'],d['arm'])):groups.setdefault(key,[]).append(i)
    for publicroot,synthroot,_ in data_roots():
        pub=load(publicroot/'public_roster.json');public={r['source_id']:r for r in pub['rows']};qs=pub['queries'];seal=load(synthroot/'frozen_readout_seal.json')
        for fi,frame in enumerate(load(synthroot/'synthesis_manifest.json')['frames']):
            check();r=public[frame['source_id']];K=np.asarray(r['depth_K']);shape=tuple(r['depth_shape']);rx,ry=rays(K,shape);intervals=[ray_interval(rx,ry,q) for q in qs]
            for arm in ARMS:
                ids=groups.get((frame['source_id'],arm))
                if ids is None:continue
                assert len(ids)==27 and d['query_id'][ids].tolist()==[q['name'] for q in qs]
                for repeat in (0,1):
                    obs=next((a for a in frame['arms'] if a['arm']==arm and a['repeat']==repeat),None)
                    path=Path(obs['path']) if obs else ROOT/'simulated'/frame['source_id']/(arm+f'_k{repeat}.npz')
                    source_paths.append(dict(frame_id=frame['source_id'],arm=arm,K=repeat,path=str(path.resolve()),sha256=sha(path)))
                    with np.load(path,allow_pickle=False) as z:
                        depth=peak_readout(z['hist'],z['background'],z['coverage'],z['grid_K'],shape,z['grid_pose'],K,shape,z['rgb_pose'],seal)[0] if np.isfinite(z['grid_pose']).all() else np.full(shape,np.inf)
                    for j,(entry,exit,domain) in enumerate(intervals):counts[ids[j],repeat]=int((domain&np.isfinite(depth)&(depth>0)&(depth>=entry)&(depth<=exit)).sum())
                sample_frames.append(dict(frame_id=frame['source_id'],arm=arm,indices=ids,K=K.tolist(),shape=list(shape),queries=qs,seal=seal))
            if fi%128==0:print('evidence',publicroot.name,fi,flush=True)
    np.testing.assert_array_equal(counts.mean(1),d['X'][:,2]);raw_tof=(counts>=16).all(1);raw_rgb=d['rgb_score']>=0;mean_tof=d['tof_score']>=0
    assert np.all(~raw_tof|mean_tof)
    classes=dict(raw_bothK=raw_tof.astype(int)+2*raw_rgb.astype(int),raw_mean_margin=mean_tof.astype(int)+2*raw_rgb.astype(int),cal_decisions=o['pred'][:,5].astype(int)+2*o['pred'][:,4].astype(int))
    np.savez_compressed(ROOT/'evidence.npz',support_counts=counts,**classes)
    save(ROOT/'evidence_sources.json',dict(ADDPLAN_sha256=sha(ADDPLAN),paths=source_paths,sample_frames=sample_frames))
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    X=d['X'][:,16:20];scores=np.full(n,np.nan);pred=np.zeros(n,bool);cuts=[];entries=[];out=ROOT/'prior';out.mkdir(exist_ok=False)
    with threadpool_limits(limits=2):
        for fold in load(PLAN)['data']['folds']:
            f=fold['fold'];check();cal=np.flatnonzero(np.isin(d['visit_id'],fold['cal']));ev=np.flatnonzero(o['fold']==f)
            with np.load(ROOT/f'fold{f}'/'B_training.npz',allow_pickle=False) as z:train=z['indices'];weights=z['weights']
            scaler=StandardScaler().fit(X[train],sample_weight=weights);cs=[];es=[]
            for seed in SEEDS:
                model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,class_weight=None,random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(scaler.transform(X[train]),d['y'][train],sample_weight=weights)
                path=out/f'fold{f}_{seed}.joblib';joblib.dump(dict(estimator=model,scaler=scaler,features=list(FEATURES[16:20])),path);entries.append(dict(fold=f,seed=seed,path=str(path.resolve()),sha256=sha(path),warnings=[str(w.message) for w in caught]))
                cs.append(model.predict_proba(scaler.transform(X[cal]))[:,1]);es.append(model.predict_proba(scaler.transform(X[ev]))[:,1])
            cc=np.mean(cs,axis=0);scores[ev]=np.mean(es,axis=0);np.savez_compressed(out/f'cal{f}.npz',indices=cal,scores=cc)
            for b,target in enumerate((.02,.05,.10)):
                mask=d['band'][cal]==b;ties=enumerate_ties(cc[mask],d['y'][cal][mask]);cut=select(ties,d['y'][cal][mask],target);cuts.append(dict(fold=f,band=b,**cut));save(out/f'ties{f}_{b}.json',ties)
                threshold=np.inf if cut['threshold'] is None else cut['threshold'];take=ev[d['band'][ev]==b];pred[take]=scores[take]>=threshold
    assert np.isfinite(scores).all();np.savez_compressed(out/'oof.npz',scores=scores,pred=pred,fold=o['fold']);save(out/'models.json',entries);save(out/'cuts.json',cuts)
    from sync_fusion_robust_dev import ci
    metrics=[];split=[]
    for fold in (-1,0,1,2,3,4):
        check();visits=load(PLAN)['data']['visits'] if fold==-1 else load(PLAN)['data']['folds'][fold]['eval'];vi={v:j for j,v in enumerate(visits)}
        vindex=np.array([vi.get(v,-1) for v in d['visit_id']]);in_fold=vindex>=0;rng=np.random.default_rng(20261011);samples=rng.integers(0,len(visits),size=(2000,len(visits)));mult=np.eye(len(visits),dtype=int)[samples].sum(1)
        def bootstrap(mask,values):return mult@np.bincount(vindex[mask],weights=values[mask],minlength=len(visits))
        for arm in ARMS:
            for b in range(3):
                mask=in_fold&(d['arm']==arm)&(d['band']==b);pos=mask&(d['y']==1);free=mask&(d['y']==0);pb=bootstrap(mask,(d['y']==1).astype(int));fb=bootstrap(mask,(d['y']==0).astype(int));wb=bootstrap(mask,(pred&(d['y']==1)).astype(int));ffb=bootstrap(mask,(pred&(d['y']==0)).astype(int))
                with np.errstate(divide='ignore',invalid='ignore'):wc=ci(100*wb/pb);fc=ci(100*ffb/fb)
                metrics.append(dict(fold=fold,arm=arm,band=b,model='prior',POS=int(pos.sum()),W=int(pred[pos].sum()),FREE=int(free.sum()),F=int(pred[free].sum()),W_rate_ci=wc,F_rate_ci=fc))
                for variant,cl in classes.items():
                    for mi,m in enumerate(MODELS[:4]):
                        mp=o['pred'][:,mi]
                        for c in range(4):
                            pp=pos&(cl==c);ff=free&(cl==c);rgb=o['pred'][:,4];tof=o['pred'][:,5]
                            split.append(dict(fold=fold,arm=arm,band=b,model=m,variant=variant,category=CATEGORIES[c],POS=int(pp.sum()),W=int((pp&mp).sum()),FREE=int(ff.sum()),F=int((ff&mp).sum()),
                              rescue_rgb=int((pp&mp&~rgb).sum()),loss_rgb=int((pp&~mp&rgb).sum()),rescue_tof=int((pp&mp&~tof).sum()),loss_tof=int((pp&~mp&tof).sum())))
    save(ROOT/'evidence_summary.json',dict(prior_metrics=metrics,evidence_split=split,ADDPLAN_sha256=sha(ADDPLAN),bothK_missing_but_mean_supported=int((~raw_tof&mean_tof).sum()),raw_neither_queries=int((classes['raw_bothK']==0).sum()),original_results_unchanged=True))
    table(ROOT/'prior_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in metrics]);table(ROOT/'evidence_split.csv',split)
    rows=[]
    for i in range(n):rows.append(dict(index=i,fold=int(o['fold'][i]),arm=str(d['arm'][i]),visit_id=str(d['visit_id'][i]),frame_id=str(d['frame_id'][i]),query_id=str(d['query_id'][i]),band=int(d['band'][i]),y=int(d['y'][i]),tof_K0_support=int(counts[i,0]),tof_K1_support=int(counts[i,1]),RGB_raw_minimum=bool(raw_rgb[i]),raw_bothK_category=CATEGORIES[classes['raw_bothK'][i]],raw_mean_margin_category=CATEGORIES[classes['raw_mean_margin'][i]],cal_category=CATEGORIES[classes['cal_decisions'][i]],prior_score=float(scores[i]),prior_pred=bool(pred[i])))
    table(ROOT/'per_query_evidence.csv',rows)
    for name in ('features','oof','summary'):assert sha(ROOT/(name+'.npz' if name!='summary' else 'summary.json'))==p['original'][name+'_sha256']
    save(ROOT/'evidence_addendum_terminal.json',dict(status='COMPLETE',seconds=time.monotonic()-started,GPU_s=0,source_sha256=sha(__file__),ADDPLAN_sha256=sha(ADDPLAN)))
    print('ADDENDUM COMPLETE',json.dumps([r for r in metrics if r['fold']==-1 and r['arm']==ARMS[0]]),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('plan','run'));ap.add_argument('--budget-s',type=float,default=350);a=ap.parse_args()
    if a.stage=='plan':fix_plan()
    else:run(a.budget_s)
