"""One fixed support-Dice fit on1198; final DEV calibration and consumed EVAL."""
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


from city_coverage64_fit import coverage_admission,score
from support_dice import masked_soft_dice

def run(a):
    started=time.perf_counter()
    baseline=read(a.baseline_run/'result.json');base_receipt=read(a.baseline_run/'receipt.json');base_protocol=read(a.baseline_run/'protocol.json')
    checkpointA=a.baseline_run/'B-step2000.pt'
    assert sha(a.baseline_run/'B-eval-predictions.npz')=='31c2a8332682ab45d80e5c05e29d095ce915f560c621fa4363ce479259dfaede'
    assert sha(a.baseline_run/'result.json')==base_receipt['result_sha256']
    assert sha(checkpointA)==baseline['fit']['checkpoint_sha256']
    assert sha(a.baseline_run/'selection.json')==base_receipt['selection_sha256']
    assert torch.cuda.is_available() and torch.__version__==base_receipt['torch_version']=='2.9.1+cu128'
    pilot=read(a.pilot/'protocol.json');assert sha(a.initial)==pilot['checkpoint_sha256']
    for name in ('city_data.py','decoupled_model.py','representation_model.py','city_pilot_metrics.py'):
        assert sha(Path(__file__).with_name(name))==pilot['source_sha256'][name],name
    for name in ('city_coverage64_fit.py','city_full_fit.py'):
        assert sha(Path(__file__).with_name(name))==base_protocol['source_sha256'][name],name
    for name in ('city_dev_baseline.py','city_dev_selection.py','city_score_separation.py'):
        assert sha(Path(__file__).with_name(name))==base_protocol['source_sha256'][name],name
    for cache in (a.old_cache,a.relational_cache,a.dev_cache,a.train64,a.eval64):assert sha(cache/'manifest.json') in base_protocol['input_sha256'].values()
    admission=coverage_admission(a.train64,a.eval64)
    trainmeta=read(a.train64/'manifest.json')
    assert trainmeta['role']=='TRAIN_ONLY' and trainmeta['admission']['status']=='PASS'
    trainrec=read(a.train64/'supervision/train.json');counts=Counter(trainrec['group_ids'])
    assert len(counts)==16 and set(counts.values())=={4}
    # EVAL manifest identity alone is pinned here; no EVAL arrays/labels before fit.
    immutable_paths=[a.initial,checkpointA,a.baseline_run/'result.json',a.baseline_run/'selection.json',a.baseline_run/'training_indices.npy',a.baseline_run/'B-eval-predictions.npz']+[c/'manifest.json' for c in (a.old_cache,a.relational_cache,a.dev_cache,a.train64,a.eval64)]
    immutable={str(p):sha(p) for p in immutable_paths}
    immutable.update({r['path']:r['sha256'] for r in admission['source_identities']})
    old=CitySupervisedDataset(a.old_cache);relational=CitySupervisedDataset(a.relational_cache);added=CitySupervisedDataset(a.train64)
    assert (len(old),len(relational),len(added))==(750,384,64)
    train=ConcatDataset([old,relational,added]);assert len(train)==1198
    out=a.output.resolve();artifacts=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert not out.exists() and out.is_relative_to(artifacts) and out!=artifacts
    out.mkdir(parents=True)
    schedule=np.load(a.baseline_run/'training_indices.npy',allow_pickle=False)
    assert sha(a.baseline_run/'training_indices.npy')==base_protocol['schedule_sha256']
    assert np.array_equal(schedule,np.random.default_rng(17).integers(0,1198,size=(2000,32)))
    np.save(out/'training_indices.npy',schedule,allow_pickle=False)
    draws=np.bincount(schedule.reshape(-1),minlength=1198)[1134:]
    groupdraw={g:int(sum(draws[i] for i,x in enumerate(trainrec['group_ids']) if x==g)) for g in counts}
    exposure=dict(total_draws=int(draws.sum()),per_sample=draws.tolist(),sample_indices=trainrec['sample_indices'],sample_min=int(draws.min()),sample_max=int(draws.max()),per_group=groupdraw,group_min=min(groupdraw.values()),group_max=max(groupdraw.values()))
    write(out/'protocol.json',dict(input_sha256=immutable,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).with_name('CITY_SUPPORT_DICE_20260908.md')),seed=17,steps=2000,batch=32,frames=1198,sampling='Exact immutable coverage64 training_indices2000x32',schedule_sha256=sha(out/'training_indices.npy'),source_admission=admission,new64_exposure=exposure,learning_rate=1e-5,weight_decay=1e-4,support_weight=.25,dice_weight_inside_support=1.0,all_parameters_trainable=True,BN_running_buffers_frozen=True,scope='A reused coverage64; B Dice-only loss change; final DEV; consumed EVAL'))
    step=0
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        model=DecoupledModel(a.pretrained).cuda();model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=True),strict=True)
        if a.resume_fit is None:
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
                n,m=model(rgb);nl=F.binary_cross_entropy_with_logits(n,near);sl=pixel_support_bce(m,support);dl=masked_soft_dice(m,support);loss=nl+.25*(sl+dl)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                assert torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                optimizer.step()
                if step%100==0:
                    history.append(dict(step=step,phase='PRE_UPDATE',near_BCE=float(nl.detach()),support_BCE=float(sl.detach()),masked_soft_dice=float(dl.detach()),loss=float(loss.detach())))
                    write(out/'progress.json',history[-1])
            torch.cuda.synchronize();assert tensor_digest(model,buffers_only=True)==buffers
            checkpoint=out/'B-step2000.pt';torch.save(model.state_dict(),checkpoint)
            torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),completed_steps=step),out/'optimizer-resume-state.pt')
            fit=dict(steps=step,seconds=time.perf_counter()-fit_start,checkpoint_sha256=sha(checkpoint),BN_unchanged=True,history=history)
            write(out/'fit-complete.json',fit)
            del optimizer,rgb,near,support,n,m,nl,sl,dl,loss,rows
        else:
            prior=read(a.resume_fit/'fit-complete.json');prior_protocol=read(a.resume_fit/'protocol.json')
            source_checkpoint=a.resume_fit/'B-step2000.pt'
            assert prior['steps']==2000 and sha(source_checkpoint)==prior['checkpoint_sha256']
            assert prior_protocol['input_sha256']==immutable
            assert prior_protocol['source_sha256']['city_support_dice_fit.py']==sha(Path(__file__))
            assert prior_protocol['source_sha256']['support_dice.py']==sha(Path(__file__).with_name('support_dice.py'))
            assert sha(a.resume_fit/'training_indices.npy')==sha(out/'training_indices.npy')
            model.load_state_dict(torch.load(source_checkpoint,map_location='cpu',weights_only=True),strict=True)
            checkpoint=out/'B-step2000.pt';checkpoint.write_bytes(source_checkpoint.read_bytes())
            fit=prior;step=2000
            write(out/'fit-complete.json',dict(fit,recovered_from=str(a.resume_fit)))
        dev,dy,ds,dg=evaluator(a.dev_cache,'dev');dn,dm=predict(model,dev)
        heads={h:select_threshold(dn[:,j],dy[:,j],min_count=48) for j,h in enumerate(('BODY','HEAD'))}
        thresholds=[heads[h]['threshold'] for h in ('BODY','HEAD')]
        selection=dict(heads=heads,thresholds=thresholds,checkpoint_sha256=sha(checkpoint),dev_manifest_sha256=sha(a.dev_cache/'manifest.json'))
        write(out/'selection.json',selection);selection_sha=sha(out/'selection.json')
        np.savez_compressed(out/'B-dev-predictions.npz',near=dn,support=dm,sample_indices=np.array(dev.ids))
        result=dict(status='PASS',fit=fit,selection=selection,exposure=exposure,DEV={'A':baseline['DEV']['B']['DEV_thresholds'],'B':score(dn,dm,dy,ds,dg,thresholds)},TRAIN={},EVAL={})
        for tag,cache,data in [('old750',a.old_cache,old),('relational384',a.relational_cache,relational),('added64',a.train64,added)]:
            data_eval,y,s,g=evaluator(cache,'train');pn,pm=predict(model,data_eval)
            np.savez_compressed(out/f'B-{tag}-predictions.npz',near=pn,support=pm,sample_indices=np.array(data_eval.ids))
            result['TRAIN'][tag]=dict(metrics=score(pn,pm,y,s,g,thresholds),losses=fixed_losses(model,data))
            result['TRAIN'][tag]['losses']['scope']+='; excludes Dice, total is legacy BCE objective diagnostic only'
        # First EVAL truth access occurs only after final checkpoint and DEV selection.
        em=read(a.eval64/'manifest.json');assert em['role']=='EVAL_ONLY' and em['admission']['status']=='PASS'
        ev,ey,es,eg=evaluator(a.eval64,'eval');assert len(ev)==64
        ec=Counter(eg);assert len(ec)==16 and set(ec.values())=={4} and not set(ec)&set(counts)
        for arm,path,ts in [('B',checkpoint,thresholds),('A',checkpointA,baseline['selection']['thresholds'])]:
            if arm=='A':
                with np.load(a.baseline_run/'B-eval-predictions.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['sample_indices'],ev.ids)
                    pn,pm=z['near'].copy(),z['support'].copy()
            else:
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                pn,pm=predict(model,ev)
            np.savez_compressed(out/f'{arm}-eval-predictions.npz',near=pn,support=pm,sample_indices=np.array(ev.ids))
            result['EVAL'][arm]=score(pn,pm,ey,es,eg,ts)
        assert sha(out/'selection.json')==selection_sha
        for p,digest in immutable.items():assert sha(p)==digest
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',new_fits=1 if a.resume_fit is None else 0,optimizer_steps=2000 if a.resume_fit is None else 0,backend='CUDA',device=torch.cuda.get_device_name(),torch_version=torch.__version__,seconds=time.perf_counter()-started,result_sha256=sha(out/'result.json'),selection_sha256=selection_sha,scope='No EVAL threshold/checkpoint selection; no rescue fit'))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',completed_steps=step,error=str(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','baseline-run','initial','pretrained','old-cache','relational-cache','train64','eval64','dev-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-fit',type=Path,help='Evaluate completed2000-step fit in fresh output; never resumes optimization')
    run(parser.parse_args())
