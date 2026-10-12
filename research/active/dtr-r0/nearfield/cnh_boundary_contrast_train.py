"""Matched-exposure boundary contrast on retained native-CNH Development.

Both branches receive original frame BCE and the same extra boundary-pair BCE.
CCON alone adds a ranking margin. Pair truth is train-only, never runtime input.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_query_mass_train as QM
import cnh_boundary_contrast_data as D

RT, RE = QM.RT, QM.RE
OUT = RT.ST.SS.WORK/'cnh-boundary-contrast-20261003'
RUN = 'CNH_BOUNDARY_CONTRAST_20261003'
ARMS = ('CBASE','CCON')
SPLITS, FRAMES = QM.SPLITS, QM.FRAMES
EPOCHS, BATCH, PAIR_BATCH, LR, WD = 10, 64, 8, .001, .0001
PAIR_WEIGHT, MARGIN, CONTRAST_WEIGHT = .5, 1., 1.
sha, read, save = QM.sha, QM.read, QM.save


def plan_sha():
    plan=read(OUT/'PLAN.json')
    expected=dict(run=RUN,training_units=SPLITS['train'],calibration_units=SPLITS['calib'],
        evaluation_units=SPLITS['evaluation'],decision_frames=FRAMES.tolist(),arms=['M3',*ARMS],
        train_arms=list(ARMS),seeds=[0],epochs=EPOCHS,batch_size=BATCH,pair_batch=PAIR_BATCH,
        learning_rate=LR,weight_decay=WD,pair_bce_weight=PAIR_WEIGHT,contrast_margin=MARGIN,
        contrast_weight=CONTRAST_WEIGHT)
    if any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Boundary contrast recipe differs from PLAN')
    if RUN not in (Path(__file__).parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Missing scoped run record')
    return sha(OUT/'PLAN.json')


def rank_loss(positive,negative):
    return F.relu(MARGIN-positive+negative).mean()


def load_pairs(rows):
    receipt=read(OUT/'pairs_receipt.json')
    if receipt.get('status')!='COMPLETE' or receipt['plan_sha256']!=plan_sha():
        raise ValueError('Boundary pairs are not complete for this PLAN')
    if receipt['output_sha256']['pairs.npz']!=sha(OUT/'pairs.npz'):
        raise ValueError('Boundary pair bytes changed')
    row_sha=hashlib.sha256(b''.join(rows[k].tobytes() for k in ('unit','config','frame'))).hexdigest()
    if row_sha!=receipt['row_identity_sha256']:
        raise ValueError('Boundary pair/training row identity mismatch')
    with np.load(OUT/'pairs.npz',allow_pickle=False) as z:
        pairs={k:z[k].astype(np.int64) for k in ('positive','negative','query')}
    n=len(pairs['positive'])
    if not n or any(v.shape!=(n,) for v in pairs.values()) or any(np.any((pairs[k]<0)|(pairs[k]>=len(rows['unit']))) for k in ('positive','negative')) or np.any((pairs['query']<0)|(pairs['query']>1)):
        raise ValueError('Invalid boundary pair axes/ranges')
    return pairs,receipt


def train():
    digest,runtime=plan_sha(),RT.ST.cuda_runtime()
    data=RT.NativeRayInputs('train')
    net=opt=scheduler=batch=None
    try:
        rows,labels=RT.ST.training_rows()
        pairs,pair_receipt=load_pairs(rows)
        if any(pair_receipt['input_sha256'].get(p)!=v for p,v in data.input_sha256.items()):
            raise ValueError('Pair metadata and training native observations differ')
        q,p,n=(pairs[k] for k in ('query','positive','negative'))
        if not np.all(labels[p,q]==1) or not np.all(labels[n,q]==0):
            raise ValueError('Pair membership disagrees with original frame targets')
        base,baseline_hashes=RT.retained_baseline(rows)
        sources={str(Path(p).resolve()):sha(p) for p in (__file__,D.__file__,QM.__file__,RT.__file__,RT.ST.__file__,RT.CP.__file__)}
        request=dict(plan_sha256=digest,source_sha256=sources,native_sha256=data.input_sha256,
            baseline_sha256=baseline_hashes,pairs_receipt_sha256=sha(OUT/'pairs_receipt.json'),
            pairs_sha256=sha(OUT/'pairs.npz'),original_M3_labels_sha256=sha(RT.ML.OUT/'train_labels.npz'),runtime=runtime,
            recipe=dict(epochs=EPOCHS,batch_size=BATCH,pair_batch=PAIR_BATCH,learning_rate=LR,weight_decay=WD,
                main='original M3 frame BCE',shared_boundary='0.5 x selected pair-query BCE for BOTH arms',
                CCON_only='mean relu(1-positive raw logit+negative raw logit), weight1',
                initialization='random same seed0 QueryMass architecture, zero alarm residual; maps are latent, not physical support',
                pair_rng_seed=2026100304,selection='finalepoch10',surface_mass_regression=False))
        req=OUT/'training_request.json'
        if req.exists():
            if read(req)!=request:
                raise ValueError('Recorded training request differs')
        else:
            save(req,request)
        rd=sha(req); done=OUT/'training_receipt.json'
        if done.exists():
            previous=read(done)
            if previous['request_sha256']!=rd or any(sha(OUT/p)!=v for p,v in previous['models_sha256'].items()):
                raise ValueError('Completed boundary training differs')
            return previous
        histories,models,initial={},{},{}
        for arm in ARMS:
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            rng=np.random.default_rng(0); pair_rng=np.random.default_rng(2026100304)
            net=QM.QueryMass().cuda()
            initial[arm]=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
            opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
            folder=OUT/'checkpoints'/arm/'seed0'; folder.mkdir(parents=True,exist_ok=True)
            last=RT.ST.load_epoch(folder,rd); history,start=[],0
            if last is not None:
                if last['arm']!=arm:
                    raise ValueError('Checkpoint branch mismatch')
                net.load_state_dict(last['model']); opt.load_state_dict(last['optimizer']); scheduler.load_state_dict(last['scheduler'])
                rng.bit_generator.state=last['numpy_rng']; pair_rng.bit_generator.state=last['pair_rng']
                torch.set_rng_state(last['torch_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in last['cuda_rng']])
                history,start=last['history'],last['epoch']
            for epoch in range(start,EPOCHS):
                began=time.monotonic(); order=rng.permutation(len(labels)); losses=np.zeros(4)
                pair_draws=[]; net.train()
                for offset in range(0,len(order),BATCH):
                    ids=order[offset:offset+BATCH]; selected=pair_rng.integers(len(p),size=PAIR_BATCH,dtype=np.int64)
                    pair_draws.append(selected)
                    pair_ids=np.concatenate((p[selected],n[selected])); query=np.tile(q[selected],2)
                    all_ids=np.concatenate((ids,pair_ids))
                    batch=[torch.as_tensor(v,device='cuda') for v in data.batch(*(rows[k][all_ids] for k in ('unit','config','frame')))]
                    baseline=torch.as_tensor(base[all_ids],device='cuda')
                    opt.zero_grad(set_to_none=True)
                    logits,_=net(*batch,baseline)
                    main=F.binary_cross_entropy_with_logits(logits[:len(ids)],torch.as_tensor(labels[ids],device='cuda'))
                    pair_logits=logits[len(ids):].gather(1,torch.as_tensor(query[:,None],device='cuda')).squeeze(1)
                    pair_targets=torch.cat((torch.ones(PAIR_BATCH,device='cuda'),torch.zeros(PAIR_BATCH,device='cuda')))
                    pair_bce=F.binary_cross_entropy_with_logits(pair_logits,pair_targets)
                    contrast=rank_loss(pair_logits[:PAIR_BATCH],pair_logits[PAIR_BATCH:]) if arm=='CCON' else main.new_zeros(())
                    loss=main+PAIR_WEIGHT*pair_bce+CONTRAST_WEIGHT*contrast
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite boundary contrast loss')
                    loss.backward(); opt.step()
                    losses+=np.array([loss.item(),main.item(),pair_bce.item(),contrast.item()])*len(ids)
                scheduler.step()
                entry=dict(epoch=epoch+1,loss=float(losses[0]/len(labels)),main_loss=float(losses[1]/len(labels)),
                    pair_bce=float(losses[2]/len(labels)),contrast_loss=float(losses[3]/len(labels)),elapsed_s=time.monotonic()-began,
                    permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                    pair_draw_sha256=hashlib.sha256(np.concatenate(pair_draws).tobytes()).hexdigest())
                history.append(entry)
                checkpoint=dict(epoch=epoch+1,arm=arm,request_sha256=rd,model=net.state_dict(),optimizer=opt.state_dict(),
                    scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,pair_rng=pair_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
                path=folder/f'epoch{epoch+1}.pt'
                with path.open('xb') as stream:
                    torch.save(checkpoint,stream)
                save(path.with_suffix('.json'),dict(status='COMPLETE',request_sha256=rd,epoch=epoch+1,sha256=sha(path)))
                print(arm,entry,flush=True)
            final=OUT/'models'/arm/'model_seed0.pt'; final.parent.mkdir(parents=True,exist_ok=True)
            state={k:v.detach().cpu() for k,v in net.state_dict().items()}
            if any(not bool(torch.isfinite(v).all()) for v in state.values()):
                raise ValueError('Nonfinite boundary model')
            if final.exists():
                old=torch.load(final,weights_only=True)
                if old.keys()!=state.keys() or any(not torch.equal(old[k],v) for k,v in state.items()):
                    raise ValueError('Existing final boundary model differs')
            else:
                with final.open('xb') as stream:
                    torch.save(state,stream)
            models[str(final.relative_to(OUT)).replace('\\','/')]=sha(final); histories[arm]=history
            net=opt=scheduler=batch=logits=loss=main=pair_bce=contrast=None
            gc.collect(); torch.cuda.empty_cache()
        paired=lambda field: [x[field] for x in histories['CBASE']]==[x[field] for x in histories['CCON']]
        if initial['CBASE']!=initial['CCON'] or not paired('permutation_sha256') or not paired('pair_draw_sha256'):
            raise ValueError('Paired training exposure mismatch')
        if plan_sha()!=digest or any(sha(p)!=v for p,v in sources.items()):
            raise ValueError('Training source changed')
        receipt=dict(status='COMPLETE',plan_sha256=digest,request_sha256=rd,source_sha256=sources,
            models_sha256=models,baseline_sha256=baseline_hashes,pairs_receipt_sha256=sha(OUT/'pairs_receipt.json'),
            runtime=runtime,recipe=request['recipe'],history=histories,seeds=[0],paired_initialization_exact=True,
            paired_shuffles_exact=True,paired_extra_examples_exact=True)
        save(done,receipt); return receipt
    finally:
        data.close(); net=opt=scheduler=batch=logits=loss=main=pair_bce=contrast=None
        gc.collect(); torch.cuda.empty_cache()


def inherited_infer():
    """Synchronous label-free inference engine reuse, restoring prior settings."""
    changes=dict(OUT=OUT,RUN=RUN,ARMS=ARMS,plan_sha=plan_sha)
    previous={k:getattr(QM,k) for k in changes}
    try:
        for k,v in changes.items():
            setattr(QM,k,v)
        return QM.infer()
    finally:
        for k,v in previous.items():
            setattr(QM,k,v)


def pair_fit():
    """Train-only discrimination of frozen models; no evaluation geometry."""
    digest,runtime=plan_sha(),RT.ST.cuda_runtime()
    data=RT.NativeRayInputs('train'); models={}
    try:
        rows,labels=RT.ST.training_rows(); pairs,_=load_pairs(rows); base,_=RT.retained_baseline(rows)
        logits={arm:np.empty_like(base) for arm in ARMS}
        for arm in ARMS:
            net=QM.QueryMass().cuda()
            net.load_state_dict(torch.load(OUT/'models'/arm/'model_seed0.pt',map_location='cpu',weights_only=True))
            models[arm]=net.eval()
        with torch.no_grad():
            for begin in range(0,len(labels),BATCH):
                ii=np.arange(begin,min(begin+BATCH,len(labels)))
                batch=[torch.as_tensor(v,device='cuda') for v in data.batch(*(rows[k][ii] for k in ('unit','config','frame')))]
                baseline=torch.as_tensor(base[ii],device='cuda')
                for arm,net in models.items():
                    values,_=net(*batch,baseline); logits[arm][ii]=values.cpu().numpy()
        logits['M3']=base; p,n,q=(pairs[k] for k in ('positive','negative','query'))
        metrics={}
        for arm,values in logits.items():
            delta=values[p,q]-values[n,q]
            metrics[arm]=dict(pairs=len(p),correct_order=int((delta>0).sum()),margin_ge_one=int((delta>=MARGIN).sum()),
                mean_margin=float(delta.mean()),mean_hinge=float(np.maximum(MARGIN-delta,0).mean()))
        result=dict(status='COMPLETE',plan_sha256=digest,runtime=runtime,metrics=metrics,
            scope='Consumed train-only fixed matched curriculum; repeated negative examples and different physical objects; no causal or generalization claim')
        save(OUT/'pair_fit.json',result); print(result,flush=True); return result
    finally:
        data.close(); models.clear(); net=batch=baseline=values=None
        gc.collect(); torch.cuda.empty_cache()


def check():
    torch.set_num_threads(2); torch.manual_seed(0)
    positive=torch.tensor([0.,2.],requires_grad=True); negative=torch.tensor([0.,0.],requires_grad=True)
    loss=rank_loss(positive,negative); assert loss.item()==.5
    loss.backward(); assert positive.grad[0]<0 and negative.grad[0]>0
    assert positive.grad[1]==0 and negative.grad[1]==0
    net=QM.QueryMass(); base=torch.randn(2,2)
    logits,_=net(torch.randn(2,8,8,8,16),torch.zeros(2,104),torch.eye(4)[None].repeat(2,1,1),base)
    assert torch.equal(logits,base)
    rng0=np.random.default_rng(2026100304); rng1=np.random.default_rng(2026100304)
    assert np.array_equal(rng0.integers(7,size=32),rng1.integers(7,size=32))
    print('PASS contrast gradient direction/inactive margin, initial exact M3 and paired draws; synthetic only')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',required=True,choices=['check','train','infer','fit'])
    args=parser.parse_args(); {'check':check,'train':train,'infer':inherited_infer,'fit':pair_fit}[args.stage]()
