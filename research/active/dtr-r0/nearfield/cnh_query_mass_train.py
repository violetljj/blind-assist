"""Direct public-query visible support: paired native-CNH Development pilot.

QMASS alone reads training surface targets. Inference uses native observations,
causal pose, public transforms/query corners and retained M3 logits only.
"""
import argparse
import gc
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cnh_ray_surface_train as RT
import cnh_ray_surface_data as RD
import cnh_ray_surface_evaluate as RE

OUT = RT.ST.SS.WORK / 'cnh-query-mass-20261003'
RUN = 'CNH_QUERY_MASS_20261003'
ARMS = ('QBCE', 'QMASS')
SPLITS, FRAMES = RT.SPLITS, RT.FRAMES
EPOCHS, BATCH, LR, WD = 10, 64, .001, .0001
MASS_SCALE, BETA = .01, .05
LOG_SCALE = math.log1p(1/MASS_SCALE)
sha, read, save = RT.sha, RT.read, RT.create_json


def plan_sha():
    plan = read(OUT/'PLAN.json')
    expected = dict(run=RUN, training_units=SPLITS['train'], calibration_units=SPLITS['calib'],
        evaluation_units=SPLITS['evaluation'], decision_frames=FRAMES.tolist(), arms=['M3', *ARMS],
        train_arms=list(ARMS), seeds=[0], epochs=EPOCHS, batch_size=BATCH, learning_rate=LR,
        weight_decay=WD, mass_scale=MASS_SCALE, auxiliary_beta=BETA)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Query mass recipe/PLAN mismatch')
    if RUN not in (Path(__file__).parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Missing run record')
    return sha(OUT/'PLAN.json')


def encode_mass(mass):
    return torch.log1p(mass/MASS_SCALE)/LOG_SCALE


def decode_mass(encoded):
    return torch.expm1(encoded*LOG_SCALE)*MASS_SCALE


def support_loss(prediction, target):
    """Equal means over supported and zero-support cells, never task labels."""
    element = F.smooth_l1_loss(prediction, encode_mass(target), beta=BETA, reduction='none')
    supported = target > 0
    groups = [element[mask].mean() for mask in (supported, ~supported) if bool(mask.any())]
    return torch.stack(groups).mean()


class QueryMass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(8, 32, 3, padding=1)
        self.pose = nn.Linear(120, 32)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)
        corners = np.concatenate((RD.QUERY_LOW, RD.QUERY_HIGH), axis=-1)/RT.RADIUS
        self.register_buffer('corners', torch.as_tensor(corners.copy(), dtype=torch.float32))
        self.query_head = nn.Sequential(nn.Conv2d(518, 64, 1), nn.GELU(), nn.Conv2d(64, 4, 1))
        self.alarm = nn.Sequential(nn.Linear(263, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.alarm[-1].weight)
        nn.init.zeros_(self.alarm[-1].bias)

    def predict_support(self, native, pose104, transform):
        public = torch.cat((pose104, transform.flatten(1)), -1)
        h = F.gelu(self.conv1(native)+self.pose(public)[:, :, None, None, None])
        h = F.gelu(self.conv2(h))
        h = h.permute(0, 1, 4, 2, 3).reshape(len(h), 512, 8, 8)
        h = h[:, None].expand(-1, 2, -1, -1, -1)
        q = self.corners[None, :, :, None, None].expand(len(h), -1, -1, 8, 8)
        joint = torch.cat((h, q), 2).reshape(len(h)*2, 518, 8, 8)
        return F.pixel_shuffle(self.query_head(joint), 2).reshape(len(h), 2, 16, 16).sigmoid()

    def forward(self, native, pose104, transform, base):
        support = self.predict_support(native, pose104, transform)
        corners = self.corners[None].expand(len(native), -1, -1)
        features = torch.cat((support.flatten(2), corners, base.detach()[..., None]), -1)
        return base.detach()+self.alarm(features).squeeze(-1), support


def teacher_mass(depth, valid, transform, rays, weights):
    points = rays[None]*(depth*RT.RADIUS)[..., None]
    points = torch.einsum('bij,byxj->byxi', transform[:, :3, :3], points)+transform[:, None, None, :3, 3]
    return torch.stack([RT.micro_integral(RT.compact_membership(points,
        torch.as_tensor(lo.copy(), dtype=depth.dtype, device=depth.device), torch.as_tensor(hi.copy(), dtype=depth.dtype, device=depth.device))
        *valid, weights) for lo, hi in zip(RD.QUERY_LOW, RD.QUERY_HIGH)], 1)


def prepare_targets():
    """Project inherited train-only first-visible labels, not new scene truth."""
    digest = plan_sha()
    runtime = RT.ST.cuda_runtime()
    done = OUT/'targets_receipt.json'
    if done.exists():
        receipt = read(done)
        if receipt['plan_sha256'] != digest or sha(OUT/'train_mass.npy') != receipt['output_sha256']['train_mass.npy']:
            raise ValueError('Existing target cache changed')
        return receipt
    data = RT.NativeRayInputs('train')
    labels = None
    started = time.monotonic()
    try:
        rows, _ = RT.ST.training_rows()
        labels = RT.RayLabels(RT.plan_sha(), data.input_sha256)
        rays, weights = [torch.as_tensor(x.copy(), dtype=torch.float32, device='cuda') for x in RD.public_rays()]
        mass = np.empty((len(rows['unit']), 2, 16, 16), np.float32)
        with torch.no_grad():
            for unit in SPLITS['train']:
                ids = np.flatnonzero(rows['unit'] == unit)
                for begin in range(0, len(ids), BATCH):
                    ii = ids[begin:begin+BATCH]
                    u, c, f = (rows[k][ii] for k in ('unit', 'config', 'frame'))
                    # Metadata transform only; no label-generated inference inputs.
                    trans = np.stack([data.transforms[int(v)%3][int(t)-3] for v, t in zip(u, f)]).astype(np.float32)
                    td, tv = labels.batch(u, c, f)
                    mass[ii] = teacher_mass(torch.as_tensor(td, device='cuda'), torch.as_tensor(tv, device='cuda'),
                        torch.as_tensor(trans, device='cuda'), rays, weights).cpu().numpy()
                print('targets', unit, flush=True)
        if not np.isfinite(mass).all() or np.any((mass < 0) | (mass > 1+1e-6)):
            raise ValueError('Invalid query support targets')
        with (OUT/'train_mass.npy').open('xb') as stream:
            np.save(stream, mass)
        receipt = dict(status='COMPLETE', plan_sha256=digest, shape=list(mass.shape), dtype='float32',
            output_sha256={'train_mass.npy': sha(OUT/'train_mass.npy')}, native_sha256=data.input_sha256,
            inherited_labels_receipt_sha256=sha(RT.OUT/'labels_receipt.json'),
            source_sha256={str(Path(p).resolve()): sha(p) for p in (__file__, RT.__file__, RD.__file__)},
            row_identity_sha256=hashlib.sha256(b''.join(rows[k].tobytes() for k in ('unit','config','frame'))).hexdigest(),
            supported_cells=int((mass>0).sum()), cells=int(mass.size), queries_with_support=int((mass.sum((-1,-2))>0).sum()),
            mean_mass=float(mass.mean()), max_mass=float(mass.max()), elapsed_s=time.monotonic()-started, runtime=runtime,
            semantics='First-visible solid-angle weighted support per microcell with inherited +-1cm soft query membership; not collision probability, physical area, or clear-space evidence')
        save(done, receipt)
        return receipt
    finally:
        data.close()
        if labels is not None:
            labels.close()
        rays = weights = mass = None
        gc.collect(); torch.cuda.empty_cache()


def train():
    digest, runtime = plan_sha(), RT.ST.cuda_runtime()
    target_receipt = read(OUT/'targets_receipt.json')
    if target_receipt['plan_sha256'] != digest or sha(OUT/'train_mass.npy') != target_receipt['output_sha256']['train_mass.npy']:
        raise ValueError('Training targets differ')
    data = RT.NativeRayInputs('train')
    target_mass = net = opt = scheduler = None
    try:
        rows, labels = RT.ST.training_rows()
        if hashlib.sha256(b''.join(rows[k].tobytes() for k in ('unit','config','frame'))).hexdigest() != target_receipt['row_identity_sha256']:
            raise ValueError('Training target row identity differs')
        base, base_hashes = RT.retained_baseline(rows)
        if data.input_sha256 != target_receipt['native_sha256']:
            raise ValueError('Training observation/target source mismatch')
        sources = {str(Path(p).resolve()): sha(p) for p in (__file__, RT.__file__, RD.__file__, RT.ST.__file__, RT.CP.__file__)}
        request = dict(plan_sha256=digest, source_sha256=sources, native_sha256=data.input_sha256,
            baseline_sha256=base_hashes, targets_receipt_sha256=sha(OUT/'targets_receipt.json'),
            original_M3_labels_sha256=sha(RT.ML.OUT/'train_labels.npz'), runtime=runtime,
            recipe=dict(epochs=EPOCHS, batch_size=BATCH, learning_rate=LR, weight_decay=WD, seed=0,
                main='original M3 frame BCE', auxiliary='equal supported/zero cell means of log1p support SmoothL1',
                auxiliary_weight=1., beta=BETA, mass_scale=MASS_SCALE, QBCE_surface_targets_read=False,
                public_conditioning='causal104 pose + current16 rigid transform + fixed6 query corners',
                head='263->64 GELU->1 zero residual final layer'))
        req = OUT/'training_request.json'
        if req.exists():
            if read(req) != request:
                raise ValueError('Training request changed')
        else:
            save(req, request)
        rd = sha(req)
        done = OUT/'training_receipt.json'
        if done.exists():
            previous = read(done)
            if previous['request_sha256'] != rd or any(sha(OUT/p) != v for p,v in previous['models_sha256'].items()):
                raise ValueError('Completed training changed')
            return previous
        histories, models, initial = {}, {}, {}
        for arm in ARMS:
            if arm == 'QMASS':
                target_mass = np.load(OUT/'train_mass.npy', mmap_mode='r')
                if target_mass.shape != (len(labels),2,16,16):
                    raise ValueError('Training support shape differs')
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            rng = np.random.default_rng(0)
            net = QueryMass().cuda()
            initial[arm] = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            folder = OUT/'checkpoints'/arm/'seed0'
            folder.mkdir(parents=True, exist_ok=True)
            last = RT.ST.load_epoch(folder, rd)
            history, start = [], 0
            if last is not None:
                net.load_state_dict(last['model']); opt.load_state_dict(last['optimizer']); scheduler.load_state_dict(last['scheduler'])
                rng.bit_generator.state = last['numpy_rng']; torch.set_rng_state(last['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([v.cpu() for v in last['cuda_rng']])
                history, start = last['history'], last['epoch']
            for epoch in range(start,EPOCHS):
                began, order, losses = time.monotonic(), rng.permutation(len(labels)), np.zeros(3)
                net.train()
                for offset in range(0,len(order),BATCH):
                    ids = order[offset:offset+BATCH]
                    u,c,f = (rows[k][ids] for k in ('unit','config','frame'))
                    batch = [torch.as_tensor(x,device='cuda') for x in data.batch(u,c,f)]
                    baseline = torch.as_tensor(base[ids], device='cuda')
                    target = torch.as_tensor(labels[ids], device='cuda')
                    opt.zero_grad(set_to_none=True)
                    logits, support = net(*batch, baseline)
                    main = F.binary_cross_entropy_with_logits(logits,target)
                    aux = support_loss(support,torch.as_tensor(np.array(target_mass[ids]),device='cuda')) if arm=='QMASS' else main.new_zeros(())
                    loss = main+aux
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward(); opt.step()
                    losses += np.array([loss.item(),main.item(),aux.item()])*len(ids)
                scheduler.step()
                entry = dict(epoch=epoch+1, loss=float(losses[0]/len(labels)), main_loss=float(losses[1]/len(labels)),
                    support_loss=float(losses[2]/len(labels)), elapsed_s=time.monotonic()-began,
                    permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest())
                history.append(entry)
                checkpoint = dict(epoch=epoch+1, arm=arm, request_sha256=rd, model=net.state_dict(), optimizer=opt.state_dict(),
                    scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                path=folder/f'epoch{epoch+1}.pt'
                with path.open('xb') as stream:
                    torch.save(checkpoint,stream)
                save(path.with_suffix('.json'),dict(status='COMPLETE',request_sha256=rd,epoch=epoch+1,sha256=sha(path)))
                print(arm,entry,flush=True)
            final=OUT/'models'/arm/'model_seed0.pt'
            final.parent.mkdir(parents=True,exist_ok=True)
            state={k:v.detach().cpu() for k,v in net.state_dict().items()}
            if any(not bool(torch.isfinite(v).all()) for v in state.values()):
                raise ValueError('Nonfinite trained model')
            if final.exists():
                old=torch.load(final,weights_only=True)
                if any(not torch.equal(old[k],v) for k,v in state.items()):
                    raise ValueError('Existing model differs')
            else:
                with final.open('xb') as stream:
                    torch.save(state,stream)
            models[str(final.relative_to(OUT)).replace('\\','/')]=sha(final)
            histories[arm]=history
            net=opt=scheduler=None
            gc.collect(); torch.cuda.empty_cache()
        if initial['QBCE']!=initial['QMASS'] or [v['permutation_sha256'] for v in histories['QBCE']] != [v['permutation_sha256'] for v in histories['QMASS']]:
            raise ValueError('Paired initialization/shuffles mismatch')
        if plan_sha()!=digest or any(sha(p)!=v for p,v in sources.items()):
            raise ValueError('Source changed during training')
        receipt=dict(status='COMPLETE',plan_sha256=digest,request_sha256=rd,source_sha256=sources,
            models_sha256=models,baseline_sha256=base_hashes,targets_receipt_sha256=sha(OUT/'targets_receipt.json'),
            runtime=runtime,recipe=request['recipe'],history=histories,paired_initialization_exact=True,paired_shuffles_exact=True,seeds=[0])
        save(done,receipt)
        return receipt
    finally:
        data.close()
        if target_mass is not None:
            target_mass._mmap.close()
        target_mass=net=opt=scheduler=batch=logits=support=loss=main=aux=None
        gc.collect(); torch.cuda.empty_cache()


def infer():
    """Seal complete raw logits before evaluation loads all-object truth."""
    digest,runtime=plan_sha(),RT.ST.cuda_runtime()
    training=read(OUT/'training_receipt.json')
    if training['plan_sha256']!=digest or sha(OUT/'training_request.json')!=training['request_sha256']:
        raise ValueError('Inference training identity mismatch')
    sources=dict(training['source_sha256'])
    sources.update({str(OUT/p):v for p,v in training['models_sha256'].items()})
    sources.update({str(OUT/p):sha(OUT/p) for p in ('PLAN.json','training_request.json','training_receipt.json')})
    RE.YE.verify_hashes(sources)
    path=OUT/'scores_receipt.json'
    if path.exists():
        receipt=read(path); RE.verify_inputs(receipt)
        if receipt['plan_sha256']!=digest or any(sha(OUT/p)!=v for p,v in receipt['output_sha256'].items()):
            raise ValueError('Completed inference differs')
        return receipt
    base,base_hashes,base_receipt=RE.retained_m3()
    sources.update(base_hashes)
    units=SPLITS['calib']+SPLITS['evaluation']
    config=np.repeat(np.arange(40),13); frames=np.tile(FRAMES,40)
    values={arm:np.empty((len(units),40,13,2),np.float32) for arm in ARMS}
    stores,models,bulk,stat={},{},{},{}
    began=time.monotonic()
    try:
        for split in ('calib','evaluation'):
            stores[split]=RT.NativeRayInputs(split)
            bulk.update(stores[split].input_sha256)
        stat={p:RE.file_snapshot(p) for p in bulk}
        for arm in ARMS:
            net=QueryMass().cuda()
            net.load_state_dict(torch.load(OUT/'models'/arm/'model_seed0.pt',weights_only=True,map_location='cpu'))
            models[arm]=net.eval()
        with torch.no_grad():
            for ui,unit in enumerate(units):
                split='calib' if unit in SPLITS['calib'] else 'evaluation'
                for begin in range(0,520,BATCH):
                    end=min(begin+BATCH,520); cc,ff=config[begin:end],frames[begin:end]
                    batch=[torch.as_tensor(v,device='cuda') for v in stores[split].batch(np.full(end-begin,unit),cc,ff)]
                    baseline=torch.as_tensor(base[ui,cc,ff-3],device='cuda')
                    for arm,net in models.items():
                        logits,_=net(*batch,baseline)
                        if logits.shape!=(end-begin,2) or not bool(torch.isfinite(logits).all()):
                            raise ValueError('Inference score axes/finite mismatch')
                        values[arm][ui,cc,ff-3]=logits.cpu().numpy()
                if (ui+1)%12==0:
                    print('inferred',ui+1,'/',len(units),'elapsed_s',round(time.monotonic()-began,2),flush=True)
        identity=dict(input_sha256=sources,bulk_sha256=bulk,bulk_stat=stat)
        RE.verify_inputs(identity)
        outputs={}
        for arm in ARMS:
            file=OUT/f'frame_scores_{arm}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream,**{str(u):values[arm][i] for i,u in enumerate(units)})
            outputs[file.name]=sha(file)
    finally:
        for store in stores.values():
            store.close()
        stores.clear(); models.clear()
        net=batch=logits=baseline=None
        gc.collect(); torch.cuda.empty_cache()
    receipt=dict(status='COMPLETE',**identity,output_sha256=outputs,plan_sha256=digest,
        training_receipt_sha256=sha(OUT/'training_receipt.json'),units=units,frames=FRAMES.tolist(),
        shape_per_unit=[40,13,2],arms=list(ARMS),retained_M3=base_receipt,elapsed_s=time.monotonic()-began,runtime=runtime,
        operations=dict(surface_labels=False,geometry_truth=False,native_observations=True,public_query=True,rendering=False),
        partial_resume=False,limit='Inference is short; interrupted incomplete inference is preserved and requires inspection before retry')
    save(path,receipt)
    print('COMPLETE sealed raw query-mass scores',flush=True)
    return receipt


def fit_diagnostic():
    """Training fit of both frozen models; no calibration/evaluation targets."""
    digest,runtime=plan_sha(),RT.ST.cuda_runtime()
    data=RT.NativeRayInputs('train')
    mass=np.load(OUT/'train_mass.npy',mmap_mode='r')
    rows,labels=RT.ST.training_rows(); base,_=RT.retained_baseline(rows)
    sums={arm:np.zeros(6) for arm in ARMS}
    models={}
    try:
        for arm in ARMS:
            model=QueryMass().cuda()
            model.load_state_dict(torch.load(OUT/'models'/arm/'model_seed0.pt',weights_only=True,map_location='cpu'))
            models[arm]=model.eval()
        with torch.no_grad():
            for begin in range(0,len(labels),BATCH):
                ii=np.arange(begin,min(begin+BATCH,len(labels)))
                batch=[torch.as_tensor(x,device='cuda') for x in data.batch(*(rows[k][ii] for k in ('unit','config','frame')))]
                target=torch.as_tensor(np.array(mass[ii]),device='cuda')
                baseline=torch.as_tensor(base[ii],device='cuda')
                mask=target>0
                for arm,model in models.items():
                    _,encoded=model(*batch,baseline)
                    error=(decode_mass(encoded)-target).abs()
                    values=(error[mask].sum().item(),mask.sum().item(),error[~mask].sum().item(),(~mask).sum().item(),
                        error.sum().item(),error.numel())
                    sums[arm]+=values
        result=dict(status='COMPLETE',plan_sha256=digest,runtime=runtime,
            scope='Training-only fit, no independent generalization or alert conclusion',
            n_rows=len(labels),metrics={arm:dict(supported_cell_mae=float(v[0]/v[1]),supported_cells=int(v[1]),
                zero_cell_mae=float(v[2]/v[3]),zero_cells=int(v[3]),all_cell_mae=float(v[4]/v[5])) for arm,v in sums.items()})
        save(OUT/'train_fit.json',result)
        print(result,flush=True)
        return result
    finally:
        data.close(); mass._mmap.close(); models.clear()
        model=batch=target=baseline=encoded=error=None
        gc.collect(); torch.cuda.empty_cache()


def check():
    torch.set_num_threads(2); torch.manual_seed(0)
    net=QueryMass()
    native=torch.randn(2,8,8,8,16); pose=torch.zeros(2,104); transform=torch.eye(4)[None].repeat(2,1,1); base=torch.randn(2,2)
    logits,support=net(native,pose,transform,base)
    assert support.shape==(2,2,16,16) and torch.equal(logits,base)
    mass=torch.linspace(0,1,17)
    torch.testing.assert_close(decode_mass(encode_mass(mass)),mass)
    target=torch.zeros_like(support); target[:,0,0,0]=.03
    loss=support_loss(support,target)
    loss.backward()
    assert net.query_head[-1].weight.grad.abs().sum()>0
    for value in (torch.zeros_like(target),torch.ones_like(target)):
        assert bool(torch.isfinite(support_loss(support.detach(),value)))
    # Solid-angle tile conservation and direct query projection independently.
    rays,weights=[torch.as_tensor(x.copy(),dtype=torch.float32) for x in RD.public_rays()]
    torch.testing.assert_close(RT.micro_integral(torch.ones(1,128,128),weights),torch.ones(1,16,16))
    depth=torch.ones(1,128,128)/RT.RADIUS; valid=torch.ones_like(depth,dtype=torch.bool)
    result=teacher_mass(depth,valid,torch.eye(4)[None],rays,weights)
    points=rays[None]*depth[...,None]*RT.RADIUS
    expected=[]
    for lo,hi in zip(RD.QUERY_LOW,RD.QUERY_HIGH):
        margin=torch.minimum(points-torch.as_tensor(lo),torch.as_tensor(hi)-points).amin(-1)
        t=(.5+margin/.02).clamp(0,1)
        field=t*t*(3-2*t)*weights
        # Alternative explicit tile axes, not micro_integral implementation.
        expected.append(field.reshape(1,16,8,16,8).sum((2,4)))
    torch.testing.assert_close(result,torch.stack(expected,1).float())
    assert not teacher_mass(depth,torch.zeros_like(valid),torch.eye(4)[None],rays,weights).any()
    print('PASS initial M3 identity, transformed support roundtrip, sparse/empty supervision gradients, query target projection and tile normalization')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['check','targets','train','infer','fit'],required=True)
    args=parser.parse_args()
    {'check':check,'targets':prepare_targets,'train':train,'infer':infer,'fit':fit_diagnostic}[args.stage]()
