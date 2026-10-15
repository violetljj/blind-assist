#!/usr/bin/env python3
"""Frozen CPU quality learning; fresh prediction never reads evaluator truth."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
import cv2
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import evaluate as ev
from run_baseline import ACTIONS
torch.set_num_threads(2)
cv2.setNumThreads(2)

SEEDS = (20261015, 20261016, 20261017)

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def model(d):
    return torch.nn.Sequential(torch.nn.Linear(d,32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))

def pair_loss(logits, y):
    a,b=torch.triu_indices(6,6,1)
    gaps=y[:,a]-y[:,b]
    return (gaps.abs()*torch.nn.functional.softplus(-gaps.sign()*(logits[:,a]-logits[:,b]))).mean()

def train_family(X, y, directory, budget_start):
    """Matched initialization and batch order for MSE and MSE+rank."""
    directory.mkdir(parents=True,exist_ok=False)
    mean=X.reshape(-1,X.shape[-1]).mean(0)
    std=X.reshape(-1,X.shape[-1]).std(0)
    std=np.where(std<1e-6,1.,std)
    Z=((X-mean)/std).astype(np.float32)
    np.savez(directory/'standardizer.npz',mean=mean,std=std)
    x=torch.from_numpy(Z); target=torch.from_numpy(y.astype(np.float32))
    receipts=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial=model(X.shape[-1]).state_dict()
        batches=np.random.default_rng(seed).integers(0,len(X),size=(600,32))
        torch.save(initial,directory/f'initial_{seed}.pt')
        np.save(directory/f'batches_{seed}.npy',batches)
        for method in ('mse','rank'):
            tick=time.perf_counter();cpu=time.process_time()
            net=model(X.shape[-1]);net.load_state_dict(copy.deepcopy(initial))
            optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
            history=[]
            for step,indices in enumerate(batches):
                if time.process_time()-budget_start>600: raise TimeoutError('CPU training budget 600 s')
                logits=net(x[indices]).squeeze(-1)
                mse=torch.nn.functional.mse_loss(torch.sigmoid(logits),target[indices])
                ranking=pair_loss(logits,target[indices])
                loss=mse+(.25*ranking if method=='rank' else 0.)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if step in (0,99,299,599):history.append(dict(step=step+1,loss=float(loss.detach()),mse=float(mse.detach()),pair=float(ranking.detach())))
            torch.save(net.state_dict(),directory/f'{method}_{seed}.pt')
            receipts.append(dict(method=method,seed=seed,steps=600,wall_s=time.perf_counter()-tick,cpu_s=time.process_time()-cpu,history=history))
    for name, estimator in (
        ('ridge',Ridge(alpha=1.)),
        ('hgb',HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.,min_samples_leaf=20,early_stopping=False,random_state=20261015))):
        tick=time.perf_counter();cpu=time.process_time()
        estimator.fit(Z.reshape(-1,Z.shape[-1]),y.reshape(-1))
        with (directory/f'{name}.pkl').open('wb') as stream:pickle.dump(estimator,stream)
        receipts.append(dict(method=name,wall_s=time.perf_counter()-tick,cpu_s=time.process_time()-cpu))
    write_json(directory/'training_receipt.json',receipts)
    return receipts

def predict_family(X,directory):
    scaler=np.load(directory/'standardizer.npz',allow_pickle=False)
    Z=((X-scaler['mean'])/scaler['std']).astype(np.float32)
    values={};per_seed={}
    for method in ('mse','rank'):
        scores=[]
        for seed in SEEDS:
            net=model(X.shape[-1]);net.load_state_dict(torch.load(directory/f'{method}_{seed}.pt',map_location='cpu',weights_only=True));net.eval()
            with torch.inference_mode(): scores.append(torch.sigmoid(net(torch.from_numpy(Z))).squeeze(-1).numpy())
        values[method]=np.mean(scores,axis=0)
        per_seed[method]=np.asarray(scores)
    for name in ('ridge','hgb'):
        with (directory/f'{name}.pkl').open('rb') as stream:estimator=pickle.load(stream)
        values[name]=np.clip(estimator.predict(Z.reshape(-1,Z.shape[-1])).reshape(Z.shape[:2]),0.,1.)
    return values,per_seed

FEATURES=['x/W','y/H','is_refined','proposal_index/2','crop_x0/W','crop_y0/H','crop_x1/W','crop_y1/H','seed_refined_dist/diag',
          'B(point)','L(point)','B(seed)','L(seed)','B(refined)','L(refined)','base_confidence','local_confidence']
FEATURES += ['GAP_raw_'+a for a in ACTIONS]+['point_raw_'+a for a in ACTIONS]
FEATURES += ['query_GAP_raw','query_point_raw','query_GAP_softmax','query_point_softmax']
assert len(FEATURES)==93

def softmax(v):
    z=np.exp(v-v.max());return z/z.sum()

def extract(obs,family):
    """Observation-only. Pixel-center transform: p in ROI maps to p*14/size."""
    w,h=obs['width'],obs['height'];views=obs['views']
    B=ev.official_minmax(ev.load_map(views['base']['heads']['ree']),(w,h))
    bc=ev.spatial_confidence(ev.normalize_mass(B)) or 0.
    xs=[];points=[]
    for i,spec in enumerate(obs[family]):
        view=views[spec['name']];x0,y0,x1,y1=view['bbox_xyxy'];cw,ch=x1-x0,y1-y0
        L=ev.official_minmax(ev.load_map(view['heads']['ree']),(cw,ch))
        if family=='candidates': seed=spec['seed_xy']
        else:
            p=ev.argmax_point(B[y0:y1,x0:x1]);seed=[p[0]+x0,p[1]+y0]
        q=ev.argmax_point(L);refined=[q[0]+x0,q[1]+y0]
        local_seed=[seed[0]-x0,seed[1]-y0]
        lc=ev.spatial_confidence(ev.normalize_mass(L)) or 0.
        logits=np.load(view['all_action_logits_path'],allow_pickle=False).astype(np.float64)
        if logits.shape!=(36,14,14) or not np.isfinite(logits).all():raise ValueError('Invalid local class maps')
        gap=logits.mean((1,2));gap_sm=softmax(gap);ai=ACTIONS.index(obs['action'])
        for is_refined,point in enumerate((seed,refined)):
            lp=[point[0]-x0,point[1]-y0]
            grid=[lp[0]*14/cw,lp[1]*14/ch]
            pv=np.asarray([ev.bilinear_value(z,grid) for z in logits]);psm=softmax(pv)
            features=[point[0]/w,point[1]/h,is_refined,i/2,x0/w,y0/h,x1/w,y1/h,math.dist(seed,refined)/math.hypot(w,h),
                      ev.bilinear_value(B,point),ev.bilinear_value(L,lp),ev.bilinear_value(B,seed),ev.bilinear_value(L,local_seed),
                      ev.bilinear_value(B,refined),ev.bilinear_value(L,q),bc,lc,*gap,*pv,gap[ai],pv[ai],gap_sm[ai],psm[ai]]
            if len(features)!=93 or not np.isfinite(features).all():raise ValueError('Feature schema/nonfinite')
            xs.append(features);points.append(point)
    if len(xs)!=6:raise ValueError('Expected six points')
    return np.asarray(xs,dtype=np.float64),points

def assemble(observations, family, manifests=None):
    features=[];points=[];targets=[]
    for obs in observations:
        x,p=extract(obs,family);features.append(x);points.append(p)
        if manifests is not None:
            row=manifests[obs['sample_id']]
            if row['image_sha256']!=obs['image_sha256'] or row['action']!=obs['action']:raise ValueError('Identity mismatch')
            if digest(row['gt_path'])!=row['gt_sha256']:raise ValueError('GT integrity')
            gt=cv2.imread(row['gt_path'],cv2.IMREAD_GRAYSCALE)
            if gt is None or gt.shape!=(obs['height'],obs['width']) or gt.max()<=0:raise ValueError('Invalid native GT')
            targets.append([ev.bilinear_value(gt,q)/float(gt.max()) for q in p])
    return np.asarray(features),points,np.asarray(targets) if manifests is not None else None

def prediction_rows(observations,family,points,scores,seeds):
    return [dict(sample_id=o['sample_id'],image_sha256=o['image_sha256'],action=o['action'],family=family,
                 points_xy=points[i],scores={k:v[i].tolist() for k,v in scores.items()},
                 seed_scores={k:v[:,i].tolist() for k,v in seeds.items()}) for i,o in enumerate(observations)]

def train(args):
    start=time.perf_counter();cpu=time.process_time();out=args.root/'training'
    out.mkdir(parents=True,exist_ok=False)
    plan=Path(__file__).with_name('LEARNED_RANK_PLAN.md')
    config=dict(plan_sha256=digest(plan),script_sha256=digest(__file__),features=FEATURES,seeds=SEEDS,steps=600,
                batch_images=32,adamw_lr=.001,weight_decay=.001,pair_weight=.25,fit_only_standardization=True,
                ridge_alpha=1,hgb=dict(max_iter=200,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1.,min_samples_leaf=20,early_stopping=False,random_state=20261015),
                old_predictions_sha256=digest(args.old_predictions),fit_manifest_sha256=digest(args.root/'data/fit_samples.jsonl'),
                cal_manifest_sha256=digest(args.root/'data/calibration_samples.jsonl'),fresh_GT_read=False)
    write_json(out/'configuration.json',config)
    (out/'learn_candidate_quality_at_training.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'feature_schema.json',dict(names=FEATURES,dimension=93,grid_transform='ROI pixel center coordinate *14/ROI size, inherited bilinear samples with -0.5 center offset',degenerate_entropy=0))
    old_obs={r['sample_id']:r for r in read_rows(args.old_predictions)}
    fit={r['sample_id']:r for r in read_rows(args.root/'data/fit_samples.jsonl')}
    cal={r['sample_id']:r for r in read_rows(args.root/'data/calibration_samples.jsonl')}
    assert not set(fit)&set(cal)
    fit_obs=[old_obs[k] for k in sorted(fit)];cal_obs=[old_obs[k] for k in sorted(cal)]
    predictions=[];gates={};receipts={}
    for family in ('candidates','fixed'):
        X,points,y=assemble(fit_obs,family,fit)
        C,cp,cy=assemble(cal_obs,family,cal)
        np.savez(out/f'{family}_fit_features_targets.npz',X=X,y=y,sample_ids=np.array([o['sample_id'] for o in fit_obs]),points=np.asarray(points))
        np.savez(out/f'{family}_cal_features_targets.npz',X=C,y=cy,sample_ids=np.array([o['sample_id'] for o in cal_obs]),points=np.asarray(cp))
        receipts[family]=train_family(X,y,out/family,cpu)
        tick=time.perf_counter();pcpu=time.process_time();scores,seeds=predict_family(C,out/family)
        receipts[family].append(dict(method='cal_all_learners_predict',wall_s=time.perf_counter()-tick,cpu_s=time.process_time()-pcpu))
        predictions.extend(prediction_rows(cal_obs,family,cp,scores,seeds))
        gates[family]={}
        for name,values in scores.items():
            selected=values.argmax(1);conf=values[np.arange(len(values)),selected]
            gates[family][name]=ev.calibrate_coverage(conf.tolist(),len(conf))
        for name,values in seeds.items():
            for i,seed in enumerate(SEEDS):
                gates[family][f'{name}_seed{seed}']=ev.calibrate_coverage(values[i].max(1).tolist(),len(cal_obs))
    write_rows(out/'calibration_predictions.jsonl',predictions)
    write_json(out/'sealed_gates.json',dict(settings=gates,cal_images=len(cal),fit_images=len(fit),fresh_GT_read=False))
    files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'training_seal.json',dict(status='SUCCESS',files=files,fit_images=len(fit),cal_images=len(cal),wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,backend='CPU',threads=2,receipts=receipts,fresh_GT_read=False))
    print(json.dumps(dict(status='SUCCESS',fit=len(fit),cal=len(cal),cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start)),flush=True)

def predict(args):
    start=time.perf_counter();cpu=time.process_time();training=args.root/'training'
    seal=json.loads((training/'training_seal.json').read_text())
    for name,expected in seal['files'].items():
        if digest(training/name)!=expected:raise ValueError('Training seal mismatch: '+name)
    out=args.root/'learned_predictions';out.mkdir(parents=True,exist_ok=False)
    observations=read_rows(args.predictions)
    if len({r['sample_id'] for r in observations})!=len(observations):raise ValueError('Duplicate IDs')
    records=[];receipts={}
    for family in ('candidates','fixed'):
        tick=time.perf_counter();pcpu=time.process_time()
        X,points,_=assemble(observations,family)
        extraction_wall=time.perf_counter()-tick;extraction_cpu=time.process_time()-pcpu
        np.savez(out/f'{family}_features.npz',X=X,points=np.asarray(points),sample_ids=np.array([o['sample_id'] for o in observations]))
        tick=time.perf_counter();pcpu=time.process_time();scores,seeds=predict_family(X,training/family)
        receipts[family]=dict(feature_wall_s=extraction_wall,feature_cpu_s=extraction_cpu,predict_wall_s=time.perf_counter()-tick,predict_cpu_s=time.process_time()-pcpu)
        records.extend(prediction_rows(observations,family,points,scores,seeds))
    write_rows(out/'fresh_predictions.jsonl',records)
    write_json(out/'prediction_seal.json',dict(status='SUCCESS',images=len(observations),records=len(records),fresh_GT_read=False,
        observation_predictions_sha256=digest(args.predictions),training_seal_sha256=digest(training/'training_seal.json'),
        script_sha256=digest(__file__),prediction_sha256=digest(out/'fresh_predictions.jsonl'),
        wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,receipts=receipts))
    print(json.dumps(dict(status='SUCCESS',images=len(observations),wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)),flush=True)

def selftest():
    z=torch.tensor([[1.,2.,3.,4.,5.,6.]])
    assert pair_loss(z,torch.zeros_like(z)).item()==0
    y=torch.arange(6).float()[None,:]/5
    assert pair_loss(z,y)<pair_loss(-z,y)
    a=np.arange(196,dtype=float).reshape(14,14)
    assert ev.bilinear_value(a,[.5,.5])==0 and ev.bilinear_value(a,[13.5,13.5])==195
    assert len(FEATURES)==93 and len(set(FEATURES))==93
    print('SELFTEST PASS: zero gaps, rank direction, pixel centers, schema')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('train','predict','selftest'))
    p.add_argument('--root',type=Path);p.add_argument('--old-predictions',type=Path);p.add_argument('--predictions',type=Path)
    args=p.parse_args()
    if args.mode=='selftest':selftest()
    elif args.mode=='train':train(args)
    else:predict(args)

if __name__=='__main__': main()
