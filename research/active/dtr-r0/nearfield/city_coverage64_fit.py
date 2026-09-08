"""One full-parameter coverage64 fit; final-only DEV, fixed new EVAL assessment."""
import argparse
from collections import Counter
from pathlib import Path
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from city_data import CitySupervisedDataset,pixel_support_bce
from city_full_fit import evaluator,fixed_losses
from city_dev_baseline import sha,read,write,tensor_digest,operating_metrics
from city_dev_selection import select_threshold
from city_finetune_pilot import predict
from city_pilot_metrics import evaluate
from city_score_separation import curve
from decoupled_model import DecoupledModel


def pair_ranking(prob,truth,groups):
    """All unordered label-discordant within-group pairs; exact ties separate."""
    result={}
    for j,head in enumerate(('BODY','HEAD')):
        rows=[]
        for group in dict.fromkeys(groups):
            ids=np.flatnonzero(np.array(groups)==group)
            positive=ids[truth[ids,j]==1];negative=ids[truth[ids,j]==0]
            delta=(prob[positive,j,None]-prob[negative,j][None,:]).reshape(-1)
            rows.append(dict(group_id=group,positive=len(positive),negative=len(negative),pairs=len(delta),correct=int((delta>0).sum()),ties=int((delta==0).sum()),reversed=int((delta<0).sum())))
        counts={k:sum(r[k] for r in rows) for k in ('pairs','correct','ties','reversed')}
        result[head]=dict(**counts,strict_accuracy=counts['correct']/counts['pairs'] if counts['pairs'] else None,groups=rows)
    return result


def score(prob,maps,truth,support,groups,thresholds):
    return dict(fixed05=evaluate(prob,maps,truth,support,groups),DEV_thresholds=operating_metrics(prob,maps,truth,support,thresholds,groups),ranking={h:curve(prob[:,j].astype(float).tolist(),truth[:,j].astype(int).tolist()) for j,h in enumerate(('BODY','HEAD'))},within_group=pair_ranking(prob,truth,groups) if groups is not None else None)



def metadata_ref(cache,record):
    path=(cache/record['path']).resolve(strict=True)
    if not path.is_relative_to(cache.resolve()) or sha(path)!=record['sha256']:
        raise ValueError('Cache metadata reference path/hash mismatch')
    return path,read(path)


def coverage_admission(train_cache,eval_cache):
    """Only manifest/source/capture metadata; never EVAL supervision arrays."""
    records={};identities=[]
    for role,cache,split in [('TRAIN_ONLY',train_cache,'train'),('EVAL_ONLY',eval_cache,'eval')]:
        manifest=read(cache/'manifest.json')
        if manifest.get('schema')!='city-training-cache-v1' or manifest.get('status')!='PASS' or manifest.get('role')!=role:
            raise ValueError('Completed role-specific coverage cache required')
        admission=manifest['admission']
        if admission['status']!='PASS':raise ValueError('Source admission is not PASS')
        source_path,spec=metadata_ref(cache,manifest['source_spec'])
        capture_path,capture=metadata_ref(cache,manifest['capture_admission'])
        if capture.get('status')!='PASS' or capture.get('source_spec_sha256')!=sha(source_path):
            raise ValueError('Capture admission/source binding mismatch')
        cases=[(i,c) for i,c in enumerate(spec['cases']) if c.get('source_role',spec.get('source_role'))==role]
        if len(cases)!=64:raise ValueError('Role must contain64 source cases')
        groups={}
        for i,c in cases:
            groups.setdefault(c['group_id'],[]).append(c)
            if not c.get('source_site_id'):raise ValueError('Source site identity absent')
        if len(groups)!=16 or any(len(v)!=4 for v in groups.values()):
            raise ValueError('Require16 complete role-specific quartets')
        for group,rows in groups.items():
            if len({tuple(sorted(c['camera'].items())) for c in rows})!=1 or len({c['source_site_id'] for c in rows})!=1:
                raise ValueError('Quartet camera/site mismatch')
            if len({c['variant_id'] for c in rows})!=4:raise ValueError('Quartet variant identities repeated')
        ids=manifest['partitions'][split]['sample_indices']
        if ids!=[i for i,c in cases]:raise ValueError('Cache order does not match captured role source order')
        cameras=[c['camera'] for rows in groups.values() for c in rows[:1]]
        records[role]=dict(groups=set(groups),sites={c['source_site_id'] for i,c in cases},cameras=cameras,admission=admission)
        identities.extend([dict(path=str(cache/'manifest.json'),sha256=sha(cache/'manifest.json')),dict(path=str(source_path),sha256=sha(source_path)),dict(path=str(capture_path),sha256=sha(capture_path))])
    train,ev=records['TRAIN_ONLY'],records['EVAL_ONLY']
    if train['groups']&ev['groups'] or train['sites']&ev['sites']:
        raise ValueError('TRAIN/EVAL group or site identity overlap')
    tx=np.array([[c['x'],c['y']] for c in train['cameras']]);ex=np.array([[c['x'],c['y']] for c in ev['cameras']])
    distance=float(np.linalg.norm(tx[:,None]-ex[None,:],axis=2).min())
    if distance<30:raise ValueError('TRAIN/EVAL camera distance below30m')
    for item in records.values():
        if item['admission']['minimum_train_eval_camera_distance_m']<30:
            raise ValueError('Missing TRAIN/EVAL separation admission')
    prior=ev['admission']['minimum_eval_prior_train_dev_camera_distance_m']
    if prior<30:raise ValueError('EVAL too near prior TRAIN/DEV')
    return dict(status='PASS',minimum_train_eval_camera_distance_m=distance,minimum_eval_prior_train_dev_camera_distance_m=prior,group_ids_disjoint=True,site_ids_disjoint=True,source_identities=identities,scope='Metadata only; source positions may be irregular; same mixed spec allowed')


def run(a):
    started=time.perf_counter()
    baseline=read(a.baseline_run/'result.json');base_receipt=read(a.baseline_run/'receipt.json');base_protocol=read(a.baseline_run/'protocol.json')
    checkpointA=a.baseline_run/'F-step2000.pt'
    assert sha(a.baseline_run/'result.json')==base_receipt['result_sha256']
    assert sha(checkpointA)==baseline['fit']['checkpoint_sha256']
    assert sha(a.baseline_run/'selection.json')==base_receipt['selection_sha256']
    assert torch.cuda.is_available() and torch.__version__==base_receipt['torch_version']=='2.9.1+cu128'
    pilot=read(a.pilot/'protocol.json');assert sha(a.initial)==pilot['checkpoint_sha256']
    for name in ('city_data.py','decoupled_model.py','representation_model.py','city_pilot_metrics.py'):
        assert sha(Path(__file__).with_name(name))==pilot['source_sha256'][name],name
    inherited=read(a.baseline_run.parent/'runtime/source-hashes.json')
    assert sha(Path(__file__).with_name('city_full_fit.py'))==inherited['city_full_fit.py']
    for name in ('city_dev_baseline.py','city_dev_selection.py','city_score_separation.py'):
        assert sha(Path(__file__).with_name(name))==base_protocol['source_sha256'][name],name
    for cache in (a.old_cache,a.relational_cache,a.dev_cache):assert sha(cache/'manifest.json') in base_protocol['input_sha256'].values()
    admission=coverage_admission(a.train64,a.eval64)
    trainmeta=read(a.train64/'manifest.json')
    assert trainmeta['role']=='TRAIN_ONLY' and trainmeta['admission']['status']=='PASS'
    trainrec=read(a.train64/'supervision/train.json');counts=Counter(trainrec['group_ids'])
    assert len(counts)==16 and set(counts.values())=={4}
    # EVAL manifest identity alone is pinned here; no EVAL arrays/labels before fit.
    immutable_paths=[a.initial,checkpointA,a.baseline_run/'result.json',a.baseline_run/'selection.json']+[c/'manifest.json' for c in (a.old_cache,a.relational_cache,a.dev_cache,a.train64,a.eval64)]
    immutable={str(p):sha(p) for p in immutable_paths}
    immutable.update({r['path']:r['sha256'] for r in admission['source_identities']})
    old=CitySupervisedDataset(a.old_cache);relational=CitySupervisedDataset(a.relational_cache);added=CitySupervisedDataset(a.train64)
    assert (len(old),len(relational),len(added))==(750,384,64)
    train=ConcatDataset([old,relational,added]);assert len(train)==1198
    out=a.output.resolve();artifacts=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert not out.exists() and out.is_relative_to(artifacts) and out!=artifacts
    out.mkdir(parents=True)
    schedule=np.random.default_rng(17).integers(0,1198,size=(2000,32));np.save(out/'training_indices.npy',schedule,allow_pickle=False)
    draws=np.bincount(schedule.reshape(-1),minlength=1198)[1134:]
    groupdraw={g:int(sum(draws[i] for i,x in enumerate(trainrec['group_ids']) if x==g)) for g in counts}
    exposure=dict(total_draws=int(draws.sum()),per_sample=draws.tolist(),sample_indices=trainrec['sample_indices'],sample_min=int(draws.min()),sample_max=int(draws.max()),per_group=groupdraw,group_min=min(groupdraw.values()),group_max=max(groupdraw.values()))
    write(out/'protocol.json',dict(input_sha256=immutable,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).with_name('CITY_COVERAGE64_20260908.md')),seed=17,steps=2000,batch=32,frames=1198,sampling='Uniform replacement over1198; regenerated sequence, not identical1134 draws',schedule_sha256=sha(out/'training_indices.npy'),source_admission=admission,new64_exposure=exposure,learning_rate=1e-5,weight_decay=1e-4,support_weight=.25,all_parameters_trainable=True,BN_running_buffers_frozen=True,scope='A reused F2000; B one new fit; final DEV calibration; EVAL assessment only'))
    step=0
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        model=DecoupledModel(a.pretrained).cuda();model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=True),strict=True)
        for parameter in model.parameters():parameter.requires_grad_(True)
        buffers=tensor_digest(model,buffers_only=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
        model.train()
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
        fit_start=time.perf_counter();history=[]
        for step,indices in enumerate(schedule,1):
            rows=[train[int(i)] for i in indices]
            rgb=torch.stack([r['rgb'] for r in rows]).cuda();near=torch.stack([r['near'] for r in rows]).cuda();support=torch.stack([r['support'] for r in rows]).cuda()
            n,m=model(rgb);nl=F.binary_cross_entropy_with_logits(n,near);sl=pixel_support_bce(m,support);loss=nl+.25*sl
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            optimizer.step()
            if step%100==0:
                history.append(dict(step=step,phase='PRE_UPDATE',near_BCE=float(nl.detach()),support_BCE=float(sl.detach()),loss=float(loss.detach())))
                write(out/'progress.json',history[-1])
        torch.cuda.synchronize();assert tensor_digest(model,buffers_only=True)==buffers
        checkpoint=out/'B-step2000.pt';torch.save(model.state_dict(),checkpoint)
        torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),completed_steps=step),out/'optimizer-resume-state.pt')
        fit=dict(steps=step,seconds=time.perf_counter()-fit_start,checkpoint_sha256=sha(checkpoint),BN_unchanged=True,history=history)
        write(out/'fit-complete.json',fit)
        del optimizer,rgb,near,support,n,m,nl,sl,loss,rows
        dev,dy,ds,dg=evaluator(a.dev_cache,'dev');dn,dm=predict(model,dev)
        heads={h:select_threshold(dn[:,j],dy[:,j],min_count=48) for j,h in enumerate(('BODY','HEAD'))}
        thresholds=[heads[h]['threshold'] for h in ('BODY','HEAD')]
        selection=dict(heads=heads,thresholds=thresholds,checkpoint_sha256=sha(checkpoint),dev_manifest_sha256=sha(a.dev_cache/'manifest.json'))
        write(out/'selection.json',selection);selection_sha=sha(out/'selection.json')
        np.savez_compressed(out/'B-dev-predictions.npz',near=dn,support=dm,sample_indices=np.array(dev.ids))
        result=dict(status='PASS',fit=fit,selection=selection,exposure=exposure,DEV={'A':baseline['DEV_selected'],'B':score(dn,dm,dy,ds,dg,thresholds)},TRAIN={},EVAL={})
        for tag,cache,data in [('old750',a.old_cache,old),('relational384',a.relational_cache,relational),('added64',a.train64,added)]:
            data_eval,y,s,g=evaluator(cache,'train');pn,pm=predict(model,data_eval)
            np.savez_compressed(out/f'B-{tag}-predictions.npz',near=pn,support=pm,sample_indices=np.array(data_eval.ids))
            result['TRAIN'][tag]=dict(metrics=score(pn,pm,y,s,g,thresholds),losses=fixed_losses(model,data))
        # First EVAL truth access occurs only after final checkpoint and DEV selection.
        em=read(a.eval64/'manifest.json');assert em['role']=='EVAL_ONLY' and em['admission']['status']=='PASS'
        ev,ey,es,eg=evaluator(a.eval64,'eval');assert len(ev)==64
        ec=Counter(eg);assert len(ec)==16 and set(ec.values())=={4} and not set(ec)&set(counts)
        for arm,path,ts in [('B',checkpoint,thresholds),('A',checkpointA,baseline['selection']['thresholds'])]:
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            pn,pm=predict(model,ev)
            np.savez_compressed(out/f'{arm}-eval-predictions.npz',near=pn,support=pm,sample_indices=np.array(ev.ids))
            result['EVAL'][arm]=score(pn,pm,ey,es,eg,ts)
        assert sha(out/'selection.json')==selection_sha
        for p,digest in immutable.items():assert sha(p)==digest
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',new_fits=1,optimizer_steps=2000,backend='CUDA',device=torch.cuda.get_device_name(),torch_version=torch.__version__,seconds=time.perf_counter()-started,result_sha256=sha(out/'result.json'),selection_sha256=selection_sha,scope='No EVAL threshold/checkpoint selection; no rescue fit'))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',completed_steps=step,error=str(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','baseline-run','initial','pretrained','old-cache','relational-cache','train64','eval64','dev-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
