"""Train/cal-only fixed small-model fitting and exhaustive operating-point seal.

No eval input is accepted by this entrypoint. K2 observations must already be
reduced to one independent frame/query row. W means query-level POS alert, not
the earlier native-pixel witness intersection metric.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

SEEDS=(955,956,957)
METHODS=('hgb','logit','tof','rgb','or','and')
HGB=dict(max_iter=100,learning_rate=.1,max_leaf_nodes=7,max_depth=3,
         min_samples_leaf=50,l2_regularization=1.,early_stopping=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def threshold_record(value):
    return dict(threshold=None,threshold_kind='positive_infinity') if np.isposinf(value) else dict(threshold=float(value),threshold_kind='finite')


def enumerate_ties(scores,labels):
    """Only finite observable scores can alert; +inf is explicit all-reject."""
    scores=np.asarray(scores,np.float64);labels=np.asarray(labels,np.int8)
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Scores may be finite or -inf only')
    rows=[dict(tie_index=0,**threshold_record(np.inf),W=0,F=0,U=0)]
    finite=np.flatnonzero(np.isfinite(scores))
    order=finite[np.argsort(-scores[finite],kind='stable')]
    if not len(order): return rows
    ordered=scores[order];ends=np.r_[np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(order)]
    cumulative={name:np.cumsum(labels[order]==value) for name,value in (('W',1),('F',0),('U',-1))}
    for i,end in enumerate(ends,1):
        rows.append(dict(tie_index=i,**threshold_record(ordered[end-1]),
            **{name:int(values[end-1]) for name,values in cumulative.items()}))
    return rows


def select(ties,labels,target):
    nfree=int((labels==0).sum());npos=int((labels==1).sum());nunknown=int((labels==-1).sum())
    if nfree==0:
        chosen=ties[0];status='NOT_CALIBRATABLE'
    elif npos==0:
        chosen=ties[0];status='NO_CAL_POS'
    else:
        feasible=[row for row in ties if row['F'] <= int(np.floor(target*nfree))]
        def rank(row):
            threshold=np.inf if row['threshold_kind']=='positive_infinity' else row['threshold']
            return row['W'],-row['F'],threshold
        chosen=max(feasible,key=rank);status='ADOPTED'
    return dict(chosen,target=target,status=status,POS_denominator=npos,FREE_denominator=nfree,
        UNKNOWN_denominator=nunknown,F_fraction=chosen['F']/nfree if nfree else None,
        W_fraction=chosen['W']/npos if npos else None)


def fit_models(X,y,source,out,check):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    known=y>=0
    if set(y[known].tolist())!={0,1}:
        raise ValueError('Train must contain POS and strictFREE')
    if not np.isfinite(X).all():
        raise ValueError('Features must contain predeclared missing encoding and flags, not NaN')
    models={};entries=[]
    for method in ('hgb','logit'):
        models[method]=[]
        for seed in SEEDS:
            check()
            model=(HistGradientBoostingClassifier(**HGB,random_state=seed) if method=='hgb' else
                make_pipeline(StandardScaler(),LogisticRegression(C=1.,penalty='l2',solver='lbfgs',
                    max_iter=1000,class_weight=None,random_state=seed)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(X[known],y[known])
            path=out/'models'/source/f'{method}_{seed}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(path)
            joblib.dump(model,path)
            entries.append(dict(source=source,method=method,seed=seed,path=str(path.resolve()),
                sha256=sha(path),warnings=[str(w.message) for w in caught],train_known_rows=int(known.sum())))
            models[method].append(model)
    return models,entries


def load_features(entry):
    path=Path(entry['path'])
    if sha(path)!=entry['sha256']:raise ValueError('Feature hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        data={key:z[key] for key in ('X','feature_names','frame_id','query_id','band','visit_id',
                                    'tof_score','rgb_score','or_score','and_score')}
    ids=list(zip(data['frame_id'].astype(str),data['query_id'].astype(str)))
    if len(set(ids))!=len(ids):raise ValueError('K2 must not duplicate independent queries')
    if data['X'].shape!=(len(ids),len(data['feature_names'])):raise ValueError('Feature axes mismatch')
    return data,ids


def run(args):
    started=time.monotonic()
    def check():
        if time.monotonic()-started>=args.budget_s-5:raise TimeoutError('Train/cal CPU command-wall budget')
    manifest=json.loads(args.features.read_text(encoding='utf-8-sig'))
    plan=json.loads(args.plan.read_text(encoding='utf-8-sig'))
    band_targets=plan['calibration']['band_targets']
    # Contract: this file is explicitly train/cal-only, not a full dataset roster.
    refs=json.loads(args.references.read_text(encoding='utf-8-sig'))['rows']
    if any(r['role'] not in ('train','cal') for r in refs):raise ValueError('Eval references forbidden')
    refs_by_role={role:{} for role in ('train','cal')}
    state_map={'POS':1,'POSITIVE':1,'FREE':0,'FREE_ON_SAMPLED_RAYS':0,'UNKNOWN':-1}
    for row in refs:
        # The authorized v1 roster embeds all27 query states per frame, so no
        # pixel-reference NPZ (and never the total/eval roster) is needed here.
        queries=row.get('queries',[row])
        for query in queries:
            key=(str(row.get('frame_id',row.get('source_id'))),str(query.get('query_id',query.get('name'))))
            if key in refs_by_role[row['role']]:raise ValueError('Duplicate reference query')
            if query['state']=='FREE_ON_SAMPLED_RAYS' and (
                    query['unknown_pixels']!=0 or query['positive_pixels']!=0 or
                    query['free_ray_pixels']!=query['domain_pixels']):
                raise ValueError('ReferenceFREE is not strict full-domain sampledFREE')
            refs_by_role[row['role']][key]=state_map[query['state']]
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,args.output/'executed_train_cal.py')
    cells=[];models_manifest=[];feature_contracts={};best=[]
    with threadpool_limits(limits=1):
        for source,roles in manifest['sources'].items():
            if set(roles)!={'train','cal'}:raise ValueError('Feature manifest must be train/cal-only')
            train,train_ids=load_features(roles['train']);cal,cal_ids=load_features(roles['cal'])
            if train['feature_names'].tolist()!=cal['feature_names'].tolist():raise ValueError('Feature names changed')
            ytrain=np.array([refs_by_role['train'][key] for key in train_ids],np.int8)
            ycal=np.array([refs_by_role['cal'][key] for key in cal_ids],np.int8)
            feature_contracts[source]=train['feature_names'].tolist()
            models,entries=fit_models(train['X'],ytrain,source,args.output,check)
            models_manifest.extend(entries)
            scores={method:np.mean([m.predict_proba(cal['X'])[:,1] for m in ensemble],axis=0,dtype=np.float64)
                    for method,ensemble in models.items()}
            scores.update({name:cal[name+'_score'].astype(np.float64) for name in ('tof','rgb','or','and')})
            for band in sorted(set(cal['band'].astype(str))):
                take=cal['band'].astype(str)==band
                for method in METHODS:
                    check();ties=enumerate_ties(scores[method][take],ycal[take])
                    path=args.output/'cal_ties'/f'{source}__{method}__{band.replace("/","_")}.json'
                    save(path,dict(source=source,method=method,band=band,
                        POS_denominator=int((ycal[take]==1).sum()),FREE_denominator=int((ycal[take]==0).sum()),
                        W_definition='Predicted query positive AND referencePOS, not pixel witness',rows=ties))
                    for target in (band_targets[band],):
                        cells.append(dict(cell_id=f'{source}/{method}/{band}/{target:.2f}',
                            source=source,method=method,band=band,**select(ties,ycal[take],target),
                            all_ties_path=str(path.resolve()),all_ties_sha256=sha(path)))
            np.savez_compressed(args.output/f'{source}_cal_scores.npz',**scores,band=cal['band'],
                frame_id=cal['frame_id'],query_id=cal['query_id'])
    # Best-single is selected separately per band AND F target, using cal only.
    for source in feature_contracts:
        for band in sorted({cell['band'] for cell in cells if cell['source']==source}):
            for target in (band_targets[band],):
                pair=[cell for cell in cells if cell['source']==source and cell['band']==band
                      and cell['target']==target and cell['method'] in ('tof','rgb')]
                chosen=max(pair,key=lambda cell:(cell['W'],-cell['F'],cell['method']=='rgb'))
                best.append(dict(source=source,band=band,target=target,method=chosen['method'],
                    status=chosen['status'],rule='Auxiliary only: maxcalW, then smallerF, then fixedRGB tie priority'))
    seal=dict(status='CAL_SEALED_EVAL_NOT_OPENED',feature_contracts=feature_contracts,
        feature_names=feature_contracts,plan_sha256=sha(args.plan),
        model_entries=models_manifest,cells=cells,cal_best_single=best,seeds=list(SEEDS),
        ensemble='Arithmetic mean of float64 positive-class predict_proba values',
        input_bindings={str(args.features):sha(args.features),str(args.references):sha(args.references)},
        source_sha256=sha(__file__),W_definition='query alert AND POS, not pixel-overlap witness',
        K2='Feature producer averaged repeats before fitting; unique frame/query enforced',
        hgb_recipe=HGB,logit_recipe=dict(scaler='StandardScaler fittrain only',C=1,L2=True,solver='lbfgs',max_iter=1000,class_weight=None),
        CPU_wall_s=time.monotonic()-started,eval_reads=0,GPU_s=0)
    save(args.output/'cal_seal.json',seal)
    print(json.dumps(dict(status=seal['status'],sources=list(feature_contracts),models=len(models_manifest),
        cells=len(cells),CPU_wall_s=seal['CPU_wall_s'],eval_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=330)
    args=p.parse_args()
    began=time.monotonic()
    try:
        run(args)
    except BaseException as error:
        save(args.output/f'failure_{time.time_ns()}.json',dict(error=repr(error),
            traceback=traceback.format_exc(),CPU_wall_s=time.monotonic()-began,eval_reads=0,GPU_s=0))
        raise
