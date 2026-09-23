"""One new dense-slice fit; historical control and all evaluation inputs frozen."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import run_contact_boundary as legacy
from contact_boundary_model import ContactBoundaryModel
from contact_boundary_data import queries,counts,choose_threshold
from boundary_supervision_data import schedule,training_labels,decision
from query_occupancy_data import read,write,sha,new_stage_directory

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
ART=REPO/'artifacts.local/evidence'
OLD=ART/'ba-contact-boundary-20260923'
PREPARED=ART/'ba-query-occupancy-20260922-prepared'
OUT=ART/'ba-boundary-supervision-20260923-v2'
SEED=202609231


def spec():
    inputs=[dict(alias='saved',path=str(OLD),role='evaluator',purpose='consumed-public-feature-and-label-development'),
            dict(alias='identities',path=str(PREPARED/'observations/identities.json'),role='observation',purpose='fixed-layout-partitions')]
    s=dict(schema='blindassist-asset-run-v1',id='boundary-supervision-20260923-v2',route='ue-query-occupancy',
        question='Does fixed-budget dense slice supervision improve metric boundaries with the same representation?',
        evaluator='research/active/dtr-r0/nearfield/run_boundary_supervision.py',
        evidence_boundary='Consumed simulation Development; one new geometry fit, historical frozen control',
        reuse=dict(mode='development',query='contact boundary cached public features initial weights dense train query labels'),
        inputs=inputs,outputs=[dict(alias='result',path=str(OUT/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=OUT.with_name(OUT.name+'-run.json');write(path,s);print(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def fit(initial,features,indices,targets,qepoch,permutations,dev_target,device,weight):
    model=ContactBoundaryModel(mode='geometry').to(device);model.load_state_dict(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[indices['train']],device=device)
    dx=torch.tensor(features[indices['dev']],device=device)
    dy=torch.tensor(dev_target,device=device,dtype=torch.float32)
    devq=torch.tensor(queries()['seen'],device=device);pw=torch.tensor(weight,device=device)
    best=float('inf');best_epoch=None;history=[];dev_logits_history=[];started=time.perf_counter()
    write(OUT/'fit-started.json',dict(seed=SEED,epochs=100,updates=2700,device=device,
          stop='One fit; interrupted training is not automatically restarted'))
    for epoch in range(100):
        q=torch.tensor(qepoch[epoch],device=device)
        y=torch.tensor(targets[epoch],device=device,dtype=torch.float32)
        total=0.;model.train()
        for start in range(0,len(x),32):
            ix=torch.tensor(permutations[epoch,start:start+32],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(x[ix],q),y[ix],pos_weight=pw)
            assert torch.isfinite(loss),'Preserve nonfinite fit; no automatic restart'
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
        if (epoch+1)%10==0:
            model.eval()
            with torch.inference_mode():
                logits=torch.cat([model(dx[i:i+32],devq) for i in range(0,len(dx),32)])
                dev_loss=float(F.binary_cross_entropy_with_logits(logits,dy))
                dev_logits_history.append(logits.cpu().numpy())
            history.append(dict(epoch=epoch+1,train_balanced_BCE=total/len(x),dev_BCE=dev_loss))
            print('FIT',epoch+1,'dev_BCE',dev_loss,flush=True)
            if dev_loss<best:
                best=dev_loss;best_epoch=epoch+1
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},OUT/'dense.pt')
    duration=time.perf_counter()-started
    np.savez_compressed(OUT/'dev-checkpoint-logits.npz',epochs=np.arange(10,101,10),
                        logits=np.stack(dev_logits_history))
    model.load_state_dict(torch.load(OUT/'dense.pt',map_location=device,weights_only=True))
    score=legacy.predict(model,features[indices['dev']],dict(seen=queries()['seen']),device)['seen']
    threshold=choose_threshold(score,dev_target)
    selection=dict(epoch=best_epoch,dev_BCE=best,threshold=threshold,dev=counts(score,dev_target,threshold),
        history=history,training_seconds=duration,checkpoint_sha256=sha(OUT/'dense.pt'),
        parameters=model.parameter_counts(),updates=2700,old_positive_weight=weight)
    write(OUT/'selection.json',selection)
    return model,selection


def run():
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state']=='running','Use tools/ba.ps1 run research-ue'
    new_stage_directory(OUT);started=time.perf_counter();legacy.configure();legacy.DEST=OUT
    files=[OLD/n for n in ('features.npy','feature-receipt.json','normalization.npz','initial.pt',
        'batch-schedule.npy','cohort.json','queries.json','source-seal.json','prediction-seal.json',
        'training-targets.npz','dev-selection-targets.npz','evaluation-targets.npz',
        'geometry-selection.json','geometry.pt','geometry-train-predictions.npz',
        'geometry-dev-predictions.npz','geometry-evaluation-predictions.npz','result.json')]
    files.append(PREPARED/'observations/identities.json')
    inputs={str(p):sha(p) for p in files}
    code={n:sha(HERE/n) for n in ('run_boundary_supervision.py','boundary_supervision_data.py',
        'BOUNDARY_SUPERVISION_PROTOCOL_20260923.md','contact_boundary_model.py','contact_boundary_data.py',
        'run_contact_boundary.py','query_occupancy_data.py')}
    prior=read(OLD/'source-seal.json')
    for n in ('contact_boundary_model.py','contact_boundary_data.py','run_contact_boundary.py'):
        assert code[n]==prior['code'][n],n
    for n,h in read(OLD/'prediction-seal.json')['files'].items():assert sha(OLD/n)==h
    assert sha(OLD/'geometry.pt')==read(OLD/'prediction-seal.json')['checkpoint_hashes']['geometry']
    assert sha(OLD/'geometry-selection.json')==read(OLD/'prediction-seal.json')['selection_hashes']['geometry']
    assert sha(OLD/'features.npy')==read(OLD/'feature-receipt.json')['features_sha256']
    write(OUT/'input-seal.json',dict(files=inputs,code=code))
    qepoch,mapping=schedule();np.save(OUT/'query-schedule.npy',qepoch)
    write(OUT/'query-mapping.json',mapping)
    cohort=read(OLD/'cohort.json');indices={k:np.array(v,np.int64) for k,v in cohort['indices'].items()}
    write(OUT/'cohort.json',cohort)
    meta=read(PREPARED/'observations/identities.json')
    assert [len(indices[s]) for s in ('train','dev','evaluation')]==[864,288,576]
    groups={s:{meta[int(i)]['base_group_id'] for i in idx} for s,idx in indices.items()}
    assert all(not groups[a]&groups[b] for a,b in [('train','dev'),('train','evaluation'),('dev','evaluation')])
    features=np.load(OLD/'features.npy');normal=load_npz(OLD/'normalization.npz')
    np.testing.assert_array_equal(normal['mean'],features[indices['train']].mean(0))
    np.testing.assert_array_equal(normal['std'],np.maximum(features[indices['train']].std(0),.01))
    features=((features-normal['mean'])/normal['std']).astype(np.float32)
    train=load_npz(OLD/'training-targets.npz');dev=load_npz(OLD/'dev-selection-targets.npz')['seen']
    target=training_labels(train,mapping)
    assert target.shape==(100,864,72)
    assert not target[np.isclose(qepoch[:,None,:,1],.3).repeat(864,axis=1)].any(),'Structural endpoint label violation'
    np.save(OUT/'training-targets.npy',target)
    old_positive=int(train['seen'].sum());old_negative=int(train['seen'].size-old_positive)
    weight=old_negative/old_positive
    initial=torch.load(OLD/'initial.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(SEED);fresh=ContactBoundaryModel(mode='geometry')
    assert all(torch.equal(initial[k],v) for k,v in fresh.state_dict().items()),'Original initialization parity'
    permutations=np.load(OLD/'batch-schedule.npy');rng=np.random.default_rng(SEED)
    np.testing.assert_array_equal(permutations,np.array([rng.permutation(864) for _ in range(100)]))
    write(OUT/'feasibility.json',dict(train_rows=864,dev_rows=288,evaluation_rows=576,
        total_training_exposures=int(target.size),unique_queries_per_image=258,
        old_positive_fraction=old_positive/train['seen'].size,new_positive_fraction=float(target.mean()),
        old_positive_weight=weight,initialization_exact=True,batch_schedule_exact=True,
        schedule_geometry_independent=True,evaluation_targets_loaded=False))
    device=legacy.choose_training_backend(fresh,features[indices['train']],target[0],qepoch[0],weight)
    model,selection=fit(initial,features,indices,target,qepoch,permutations,dev,device,weight)
    predicted={};hashes={}
    for split,idx in indices.items():
        p=legacy.predict(model,features[idx],queries(),device);predicted[split]=p
        path=OUT/f'{split}-predictions.npz';np.savez_compressed(path,**p);hashes[path.name]=sha(path)
    del model
    write(OUT/'prediction-seal.json',dict(files=hashes,checkpoint=sha(OUT/'dense.pt'),selection=sha(OUT/'selection.json'),
        dev_checkpoint_logits=sha(OUT/'dev-checkpoint-logits.npz'),
        all_predictions_before_evaluation_target_access=True,evaluation_targets_loaded=False))
    # Only now consume old evaluation labels and old outcome summaries.
    evaluation=load_npz(OLD/'evaluation-targets.npz');old_result=read(OLD/'result.json')['arms']['geometry']
    new={};old=dict(train=old_result['training_layouts'],evaluation={k:v for k,v in old_result.items() if k!='training_layouts'})
    for split,truth in [('train',train),('evaluation',evaluation)]:
        boundaries={k:truth[k+'_boundary'] for k in ('width','horizon')}
        new[split]=legacy.summary(predicted[split],truth,boundaries,[meta[int(i)] for i in indices[split]],selection['threshold'])
    verdict=decision(old,new)
    # Fixed saved working points; do not select a new equal-realized-FPR cutoff.
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    for n,h in hashes.items():assert sha(OUT/n)==h,n
    result=dict(status='PASS',**verdict,old=old,new=new,selection=selection,
        device=device,elapsed_s=time.perf_counter()-started,old_inputs_unchanged=True,
        budget=dict(new_fits=1,epochs=100,updates=2700,query_exposures=6220800),
        evidence_boundary='Consumed same-generator layout-disjoint Development; dense query slices trained; no hardware or safety claim')
    write(OUT/'result.json',result);print('COMPLETE',verdict,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['spec','run']);args=p.parse_args()
    spec() if args.command=='spec' else run()
