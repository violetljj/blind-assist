"""One governed continuous-query Development comparison; no old recipe mutation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from contact_boundary_data import (LAYERS, queries, camera_boxes, contact_labels,
    critical_values, counts, choose_threshold, boundary_metrics)
from query_occupancy_data import read, write, sha, new_stage_directory
from query_occupancy_learning import tensor_batch
from query_occupancy_model import canonicalize_tof

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ART = REPO/'artifacts.local'
SOURCE = ART/'evidence/ba-query-occupancy-20260922'
PREPARED = SOURCE.with_name(SOURCE.name+'-prepared')
CAPTURE = SOURCE.with_name(SOURCE.name+'-capture')
DEST = ART/'evidence/ba-contact-boundary-20260923'
SEED, EPOCHS, BATCH = 202609231, 100, 32
PROTOCOL = HERE/'CONTACT_BOUNDARY_PROTOCOL_20260923.md'


def make_spec():
    entries = [
        ('observations', PREPARED/'observations', 'observation'),
        ('visibility', PREPARED/'labels/visibility-audit.json', 'evaluator'),
        ('manifest', PREPARED/'materialization.json', 'configuration'),
        ('geometry', CAPTURE/'evaluator/geometry.json', 'evaluator'),
        ('plan', SOURCE/'plan', 'configuration')]
    spec = dict(schema='blindassist-asset-run-v1', id='contact-boundary-20260923',
        route='ue-query-occupancy', question='Do direct or monotone geometric contact readouts generalize to unseen body widths and horizons?',
        evaluator='research/active/dtr-r0/nearfield/run_contact_boundary.py',
        evidence_boundary='Consumed grouped simulation Development; query geometry is evaluator-only',
        reuse=dict(mode='development', query='Existing query occupancy RGB regional ToF rendered AABB lateral counterfactual source'),
        inputs=[dict(alias=a, path=str(p), role=r, purpose='scoped-contact-boundary-development') for a,p,r in entries],
        outputs=[dict(alias='result', path=str(DEST/'result.json'), role='result', required=True)],
        result_output='result', command=[sys.executable, '-B', str(Path(__file__).resolve()), 'run'])
    path = DEST.with_name(DEST.name+'-run.json')
    write(path, spec)
    print(path, flush=True)


def configure():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def extract_features(rgb, tof, weights):
    sys.path.insert(0, str(REPO))
    from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation
    backbone = mobilenet_v3_small(weights=None)
    backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    encoder = backbone.features[:7].eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    candidates = {}
    for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        model = copy.deepcopy(encoder).to(dev)
        x, _ = tensor_batch(rgb, tof, np.arange(BATCH), dev)
        def probe(m=model, xx=x):
            with torch.inference_mode():
                return F.adaptive_avg_pool2d(m(xx), (8,14)).flatten(1)
        candidates[dev] = BackendCandidate(dev, dev, probe,
            lambda output, m=model: torch_observation(model=m),
            torch.cuda.synchronize if dev == 'cuda' else lambda: None)
    chosen = select_backend(Workload.MODEL_INFERENCE, cpu=candidates['cpu'], gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=DEST/'encoder-backend.json', warmups=1, repeats=3)
    device = chosen['selected_device_type']
    encoder.to(device)
    features = []
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(rgb), BATCH):
            indices = np.arange(first, min(first+BATCH, len(rgb)))
            x,t = tensor_batch(rgb, tof, indices, device)
            visual = F.adaptive_avg_pool2d(encoder(x), (8,14)).flatten(1)
            sensor = canonicalize_tof(t).flatten(1)
            features.append(torch.cat((visual,sensor), dim=1).cpu().numpy())
    values = np.concatenate(features)
    assert values.shape == (1728,4864) and np.isfinite(values).all()
    np.save(DEST/'features.npy', values)
    write(DEST/'feature-receipt.json', dict(device=device, elapsed_s=time.perf_counter()-started,
        features_sha256=sha(DEST/'features.npy'), frozen_encoder=True,
        input_fields=['normalized RGB','regional ToF tokens'], all_public_rows_encoded=True))
    return values


def labels_for(indices, geometry, qdict, audit=False):
    labels = {name: [] for name in qdict}
    boundaries = {k: [] for k in ('width','horizon')}
    audited = 0
    if audit:
        from audit_query_occupancy import world_box_labels
    for i in indices:
        geo = geometry[int(i)]
        boxes = camera_boxes(geo)
        for name, q in qdict.items():
            hit, _ = contact_labels(boxes, q)
            labels[name].append(hit)
            if audit and name in ('seen','both'):
                for j,(width,horizon,layer) in enumerate(q):
                    yl,yh = LAYERS[int(layer)]
                    oldq = np.array([[-float(width)/2,float(width)/2,yl,yh]],np.float64)
                    cls,_ = world_box_labels(geo['objects'],geo['declared_camera'],oldq,
                        np.array([.3,float(horizon)],np.float64))
                    assert bool(cls[0] == 0) == bool(hit[j]), (int(i),name,j)
                    audited += 1
        for kind in boundaries:
            boundaries[kind].append(critical_values(boxes,kind))
    return ({k:np.asarray(v,bool) for k,v in labels.items()},
            {k:np.asarray(v,np.float64) for k,v in boundaries.items()},audited)


def choose_training_backend(model, features, target, q, weight):
    from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation
    candidates={}
    for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        m=copy.deepcopy(model).to(dev).train()
        x=torch.tensor(features[:BATCH],device=dev)
        y=torch.tensor(target[:BATCH],device=dev,dtype=torch.float32)
        query=torch.tensor(q,device=dev)
        pos=torch.tensor(weight,device=dev)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        def probe(mm=m,xx=x,yy=y,qq=query,ww=pos,oo=opt):
            oo.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(mm(xx,qq),yy,pos_weight=ww)
            loss.backward(); oo.step()
            return loss.detach()
        candidates[dev]=BackendCandidate(dev,dev,probe,lambda output,mm=m:torch_observation(model=mm),
            torch.cuda.synchronize if dev=='cuda' else lambda:None)
    selected=select_backend(Workload.BATCH_TENSOR,cpu=candidates['cpu'],gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=DEST/'training-backend.json',warmups=1,repeats=3)
    return selected['selected_device_type']


def predict(model,features,qdict,device):
    model.eval()
    output={}
    with torch.inference_mode():
        for name,q in qdict.items():
            qq=torch.tensor(q,device=device)
            rows=[]
            for i in range(0,len(features),BATCH):
                xx=torch.tensor(features[i:i+BATCH],device=device)
                rows.append(model(xx,qq).sigmoid().cpu().numpy())
            output[name]=np.concatenate(rows)
    return output


def fit(mode,initial,features,train_idx,dev_idx,train_y,dev_y,q,schedule,device,pos_weight):
    from contact_boundary_model import ContactBoundaryModel
    model=ContactBoundaryModel(mode=mode).to(device)
    model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[train_idx],device=device)
    y=torch.tensor(train_y,device=device,dtype=torch.float32)
    dx=torch.tensor(features[dev_idx],device=device)
    dy=torch.tensor(dev_y,device=device,dtype=torch.float32)
    qq=torch.tensor(q,device=device)
    pw=torch.tensor(pos_weight,device=device)
    best=float('inf'); best_epoch=None; history=[]
    start=time.perf_counter()
    for epoch,permutation in enumerate(schedule,1):
        model.train();total=0.
        for i in range(0,len(permutation),BATCH):
            rows=torch.tensor(permutation[i:i+BATCH],device=device)
            opt.zero_grad(set_to_none=True)
            logits=model(x[rows],qq)
            loss=F.binary_cross_entropy_with_logits(logits,y[rows],pos_weight=pw)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite fit loss; preserve failed receipt, no new fit')
            loss.backward();opt.step();total+=float(loss.detach())*len(rows)
        if epoch%10==0:
            model.eval()
            with torch.inference_mode():
                dev_logits=torch.cat([model(dx[i:i+BATCH],qq) for i in range(0,len(dx),BATCH)])
                dev_loss=float(F.binary_cross_entropy_with_logits(dev_logits,dy))
            history.append(dict(epoch=epoch,train_balanced_BCE=total/len(x),dev_BCE=dev_loss))
            print('FIT',mode,epoch,'dev_BCE',round(dev_loss,6),flush=True)
            if dev_loss<best:
                best,best_epoch=dev_loss,epoch
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},DEST/(mode+'.pt'))
    duration=time.perf_counter()-start
    model.load_state_dict(torch.load(DEST/(mode+'.pt'),map_location=device,weights_only=True))
    dev_score=predict(model,features[dev_idx],dict(seen=q),device)['seen']
    threshold=choose_threshold(dev_score,dev_y)
    selection=dict(epoch=best_epoch,dev_BCE=best,threshold=threshold,dev=counts(dev_score,dev_y,threshold),
        history=history,training_seconds=duration,parameters=model.parameter_counts(),
        checkpoints_sha256=sha(DEST/(mode+'.pt')),updates=EPOCHS*int(np.ceil(len(train_idx)/BATCH)))
    write(DEST/(mode+'-selection.json'),selection)
    return model,selection


def lateral_pairs(score,truth,metadata,threshold):
    score=np.asarray(score,np.float64)
    groups={}
    for i,m in enumerate(metadata):
        groups.setdefault((m['base_group_id'],m['frame_in_clip']),{})[m['layout_relation']]=i
    margins=[];joint=[]
    for rows in groups.values():
        for a,b in [('INSIDE','BOUNDARY'),('BOUNDARY','OUTSIDE'),('INSIDE','OUTSIDE')]:
            ia,ib=rows[a],rows[b]
            changed=truth[ia] != truth[ib]
            sign=np.where(truth[ia],1.,-1.)
            margins.extend(((score[ia]-score[ib])*sign)[changed].tolist())
            joint.extend((((score[ia]>=threshold)==truth[ia]) & ((score[ib]>=threshold)==truth[ib]))[changed].tolist())
    m=np.asarray(margins)
    return dict(changed_pairs=len(m),strict_order_correct=int((m>0).sum()),ties=int((m==0).sum()),
        order_accuracy=float(((m>0)+.5*(m==0)).mean()) if len(m) else None,
        both_decisions_correct=int(np.sum(joint)),joint_accuracy=float(np.mean(joint)) if joint else None)


def summary(pred,truth,boundary,metadata,threshold):
    result={name:counts(pred[name],truth[name],threshold) for name in ('seen','width','horizon','both','extrapolation')}
    result['boundaries']={kind:boundary_metrics(pred[kind+'_curve'],threshold,boundary[kind],kind)
                          for kind in ('width','horizon')}
    for field in ('type_id','layout_relation'):
        result[field]={}
        for value in sorted({m[field] for m in metadata}):
            rows=np.array([m[field]==value for m in metadata])
            result[field][value]=counts(pred['both'][rows],truth['both'][rows],threshold)
    result['lateral_pairs']=lateral_pairs(pred['both'],truth['both'],metadata,threshold)
    result['unknown_sensor_frames']=sum(bool(m['baseline']['unknown']) for m in metadata)
    c=result['both'];b=result['boundaries']
    result['component_gate']=bool(c['recall']>=.8 and c['FPR']<=.05 and
        all((b[k]['joint_within_5cm'] or 0)>=.8 for k in b))
    return result


def paired_interval(preds,truth,metadata,selections):
    group_names=sorted({m['base_group_id'] for m in metadata})
    contributions={}
    for mode in preds:
        rows=[]
        for group in group_names:
            idx=np.array([m['base_group_id']==group for m in metadata])
            c=counts(preds[mode]['both'][idx],truth['both'][idx],selections[mode]['threshold'])
            rows.append([c['TP'],c['FN'],c['FP'],c['TN']])
        contributions[mode]=np.array(rows)
    rng=np.random.default_rng(SEED+1)
    differences=[]
    for _ in range(1000):
        sample=rng.integers(len(group_names),size=len(group_names))
        rates=[]
        for mode in ('direct','geometry'):
            tp,fn,fp,tn=contributions[mode][sample].sum(0)
            rates.append([tp/max(1,tp+fn),fp/max(1,fp+tn)])
        differences.append(np.array(rates[1])-rates[0])
    interval=np.quantile(differences,[.025,.975],axis=0)
    return dict(groups=len(group_names),resamples=1000,direction='geometry-minus-direct',
        recall_difference_95pct=interval[:,0].tolist(),FPR_difference_95pct=interval[:,1].tolist())


def run():
    from contact_boundary_model import ContactBoundaryModel
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state']=='running','Use tools/ba.ps1 run research-ue'
    new_stage_directory(DEST)
    started=time.perf_counter();configure()
    code_files=['run_contact_boundary.py','contact_boundary_data.py','contact_boundary_model.py',
                'query_occupancy_data.py','query_occupancy_learning.py','query_occupancy_model.py',
                'audit_query_occupancy.py','CONTACT_BOUNDARY_PROTOCOL_20260923.md']
    code={name:sha(HERE/name) for name in code_files}
    snapshot=DEST/'source-snapshot';snapshot.mkdir()
    for name in code_files:
        (snapshot/name).write_bytes((HERE/name).read_bytes())
    materialization=read(PREPARED/'materialization.json')
    inputs={str(PREPARED/name):materialization['hashes'][name] for name in
        ('observations/rgb.npy','observations/tof.npy','observations/identities.json','labels/visibility-audit.json')}
    weights=SOURCE/'plan/mobilenet_v3_small.pth'
    for path in (weights,PREPARED/'materialization.json',CAPTURE/'evaluator/geometry.json',SOURCE/'plan/spec.json'):
        inputs[str(path)]=sha(path)
    for path,digest in inputs.items():
        assert sha(path)==digest,path
    write(DEST/'source-seal.json',dict(code=code,input_hashes=inputs,protocol_sha256=sha(PROTOCOL),seed=SEED))
    identities=read(PREPARED/'observations/identities.json')
    visibility=read(PREPARED/'labels/visibility-audit.json')
    assert len(identities)==1728 and all(all(v['query_label_valid']) for v in visibility)
    indices={s:np.array([m['index'] for m in identities if m['split']==s],np.int64)
             for s in ('train','dev','evaluation')}
    assert [len(v) for v in indices.values()]==[864,288,576]
    groups={s:{identities[int(i)]['base_group_id'] for i in idx} for s,idx in indices.items()}
    assert all(not groups[a]&groups[b] for a,b in [('train','dev'),('train','evaluation'),('dev','evaluation')])
    qdict=queries();write(DEST/'queries.json',{k:v.tolist() for k,v in qdict.items()})
    write(DEST/'cohort.json',dict(indices={s:v.tolist() for s,v in indices.items()},
        groups={s:sorted(v) for s,v in groups.items()},independence='Consumed same-generator whole-layout Development'))
    geometry=read(CAPTURE/'evaluator/geometry.json')
    assert all(g['id']==m['id'] and g['sample_index']==m['index'] for g,m in zip(geometry,identities))
    train_label,train_bound,n_train=labels_for(indices['train'],geometry,qdict,audit=True)
    dev_label,_,n_dev=labels_for(indices['dev'],geometry,dict(seen=qdict['seen']),audit=True)
    np.savez_compressed(DEST/'training-targets.npz',**train_label,**{k+'_boundary':v for k,v in train_bound.items()})
    np.savez_compressed(DEST/'dev-selection-targets.npz',**dev_label)
    positive=int(train_label['seen'].sum());negative=int(train_label['seen'].size-positive)
    assert positive and negative
    write(DEST/'feasibility.json',dict(train_positive=positive,train_negative=negative,
        world_space_query_checks=n_train+n_dev,visible_domain_subset=True,
        evaluation_labels_generated=False,geometry_backend='TASK_NOT_GPU_SUITABLE'))
    print('FEASIBILITY',positive,negative,'independent checks',n_train+n_dev,flush=True)
    rgb=np.load(PREPARED/'observations/rgb.npy',mmap_mode='r')
    tof=np.load(PREPARED/'observations/tof.npy',mmap_mode='r')
    features=extract_features(rgb,tof,weights)
    mean=features[indices['train']].mean(0);std=np.maximum(features[indices['train']].std(0),.01)
    np.savez(DEST/'normalization.npz',mean=mean,std=std)
    features=((features-mean)/std).astype(np.float32)
    torch.manual_seed(SEED)
    carrier=ContactBoundaryModel(mode='direct')
    initial={k:v.detach().cpu().clone() for k,v in carrier.state_dict().items()}
    torch.save(initial,DEST/'initial.pt')
    rng=np.random.default_rng(SEED)
    schedule=np.array([rng.permutation(len(indices['train'])) for _ in range(EPOCHS)])
    np.save(DEST/'batch-schedule.npy',schedule)
    device=choose_training_backend(carrier,features[indices['train']],train_label['seen'],qdict['seen'],negative/positive)
    selections={};prediction_files={}
    for mode in ('direct','geometry'):
        model,selection=fit(mode,initial,features,indices['train'],indices['dev'],
            train_label['seen'],dev_label['seen'],qdict['seen'],schedule,device,negative/positive)
        selections[mode]=selection
        # All three splits have public-input predictions sealed before evaluation truth is computed.
        for split,idx in indices.items():
            pred=predict(model,features[idx],qdict,device)
            path=DEST/(mode+'-'+split+'-predictions.npz')
            np.savez_compressed(path,**pred)
            prediction_files[path.name]=sha(path)
        del model
    write(DEST/'prediction-seal.json',dict(files=prediction_files,
        checkpoint_hashes={m:sha(DEST/(m+'.pt')) for m in selections},
        selection_hashes={m:sha(DEST/(m+'-selection.json')) for m in selections},
        evaluation_labels_generated=False,all_predicted_before_evaluator_join=True))
    evaluation_label,evaluation_bound,n_eval=labels_for(indices['evaluation'],geometry,qdict,audit=True)
    np.savez_compressed(DEST/'evaluation-targets.npz',**evaluation_label,
        **{k+'_boundary':v for k,v in evaluation_bound.items()})
    reports={};eval_preds={}
    for mode in ('direct','geometry'):
        pred=dict(np.load(DEST/(mode+'-evaluation-predictions.npz')));eval_preds[mode]=pred
        meta=[identities[int(i)] for i in indices['evaluation']]
        result=summary(pred,evaluation_label,evaluation_bound,meta,selections[mode]['threshold'])
        train_pred=dict(np.load(DEST/(mode+'-train-predictions.npz')))
        train_meta=[identities[int(i)] for i in indices['train']]
        result['training_layouts']=summary(train_pred,train_label,train_bound,train_meta,selections[mode]['threshold'])
        reports[mode]=result
    d,g=(reports[k]['both'] for k in ('direct','geometry'))
    relative=(g['recall']-d['recall']>=.05 and g['FPR']-d['FPR']<=.02) or (
        g['FPR']<=.7*d['FPR'] and d['FPR']>0 and g['recall']-d['recall']>=-.03)
    relative=relative and all(reports['geometry']['type_id'][f]['recall']-
        reports['direct']['type_id'][f]['recall']>=-.05 for f in reports['direct']['type_id'])
    for path,digest in inputs.items():
        assert sha(path)==digest,path
    for name,digest in code.items():
        assert sha(HERE/name)==digest,name
    for name,digest in prediction_files.items():
        assert sha(DEST/name)==digest,name
    decision=('GEOMETRY_COMPONENT' if reports['geometry']['component_gate'] else
              'DIRECT_COMPONENT' if reports['direct']['component_gate'] else 'NEITHER_CONTACT_PACKAGE_PASSES')
    result=dict(status='PASS',decision=decision,arms=reports,selection=selections,
        geometry_relative_advantage=bool(relative),
        paired_layout_bootstrap=paired_interval(eval_preds,evaluation_label,meta,selections),
        source_hashes_unchanged=True,world_space_query_checks=n_train+n_dev+n_eval,
        actual_device=device,epochs_per_arm=EPOCHS,elapsed_s=time.perf_counter()-started,
        evidence_boundary='Consumed whole-layout simulation Development; fixed visual features; no real-body, alert, hardware or safety claim',
        task_processes_released='In-process tensor work only; no child process or paid allocation')
    write(DEST/'result.json',result)
    print('COMPLETE',decision,'geometry relative advantage',relative,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['spec','run'])
    args=parser.parse_args()
    make_spec() if args.command=='spec' else run()
