"""Matched fixed bounded-residual training and public-only native predictions.

Stages preserve cumulative budgets, source snapshots, and calibration-before-eval.
Train targets use the original supervision bank. Cal/eval truth is never opened.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

START = time.perf_counter()
import cv2
import numpy as np
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays
from rgb_body_query_residual_target import PUBLIC, public_mapping
from rgb_body_query_3rscan import sample_prediction

SEED = 20261010
ARMS = ('trained_depth_ray', 'trained_context')
B = 3.1847211408780134
A = .3713312368564329
INTERCEPT = .2635894826641125


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def raw_features(depth, intrinsic):
    """All context uses public native DP, independently of supervision validity."""
    depth = np.asarray(depth, np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    log = np.zeros_like(depth)
    log[valid] = np.log(depth[valid])
    rx, ry = rays(np.asarray(intrinsic), depth.shape)
    pad = np.pad(log, 1, mode='edge')
    vp = np.pad(valid, 1, mode='edge')
    left = np.where(vp[1:-1, :-2], pad[1:-1, :-2], log)
    right = np.where(vp[1:-1, 2:], pad[1:-1, 2:], log)
    top = np.where(vp[:-2, 1:-1], pad[:-2, 1:-1], log)
    bottom = np.where(vp[2:, 1:-1], pad[2:, 1:-1], log)
    features = [log, rx, ry, (right-left)/2, (bottom-top)/2]
    for size in (3, 9, 25):
        def total(x):
            return cv2.boxFilter(x, -1, (size, size), normalize=False,
                                 borderType=cv2.BORDER_REPLICATE)
        count = total(valid.astype(np.float64))
        denominator = np.maximum(count, 1)
        mean = total(log)/denominator
        variance = np.maximum(total(log*log)/denominator-mean*mean, 0)
        features.extend([mean-log, np.sqrt(variance)])
    result = np.stack(features, axis=-1)
    result[~valid] = 0
    return result, valid


def check(remaining, reserve=3):
    if time.perf_counter()-START >= remaining-reserve:
        raise TimeoutError('Cumulative stage allocation exhausted')


def prepare(repo, root, receipt, remaining):
    previous = repo/'artifacts.local/work/rgb-body-query-residual-target-dev-20261010/diagnostic'
    rosterpath = previous/'public_roster.json'
    bankpath = previous/'train_bank.npz'
    rows = load(rosterpath)['rows']
    assert len(rows) == 360 and [r['role'] for r in rows[:56]] == ['train']*56
    assert all(r['role'] == 'cal' for r in rows[56:])
    bank = np.load(bankpath)
    frameids = bank['frame_index']
    flatids = bank['flat_index']
    assert len(flatids) == 1591791 and set(np.unique(frameids)) == set(range(56))
    rawbank = np.empty((len(flatids), 11), np.float64)
    caches = []
    cachefolder = root/'features'; cachefolder.mkdir()
    offsets = []
    for i, row in enumerate(rows):
        check(remaining)
        assert sha(row['public_native_path']) == row['public_native_sha256']
        with np.load(row['public_native_path']) as native:
            depth = native['depth']
            raw, valid = raw_features(depth, row['depth_K'])
        if i < 56:
            indices = np.flatnonzero(frameids == i)
            rawbank[indices] = raw.reshape(-1, 11)[flatids[indices]]
            assert np.all(valid.ravel()[flatids[indices]])
            assert np.allclose(rawbank[indices, :3], bank['features_raw'][indices], atol=1e-12, rtol=0)
            offsets.append([int(indices[0]), int(indices[-1])+1])
        caches.append(raw)
        row['sampled_depth_path'] = row['public_native_path']
        row['sampled_depth_sha256'] = row['public_native_sha256']
        if i % 60 == 0:
            print(json.dumps(dict(stage='public_context', frames=i+1)), flush=True)
    mean = rawbank.mean(axis=0)
    std = np.maximum(rawbank.std(axis=0), 1e-6)
    np.savez(root/'normalization.npz', mean=mean, std=std)
    np.save(root/'train_features.npy', ((rawbank-mean)/std).astype(np.float32))
    np.save(root/'train_targets.npy', bank['residual'].astype(np.float32))
    np.savez(root/'train_identity.npz', frame_index=frameids, flat_index=flatids,
             environment_index=bank['environment_index'], residual_float64=bank['residual'])
    np.savez(root/'audit_train_frame0.npz', raw=caches[0], eligible_flat=flatids[frameids==0],
             supervised_raw=rawbank[frameids==0])
    for i, (row, raw) in enumerate(zip(rows, caches)):
        check(remaining)
        path = cachefolder/f'{i:03d}.npy'
        np.save(path, ((raw-mean)/std).astype(np.float32))
        row.update(feature_path=str(path.resolve()), feature_sha256=sha(path))
    rng = np.random.default_rng(SEED)
    schedule = np.empty((600, 56*512), np.int32)
    for step in range(600):
        for frame, (begin, end) in enumerate(offsets):
            schedule[step, frame*512:(frame+1)*512] = rng.integers(begin, end, 512)
    np.save(root/'batch_indices.npy', schedule)
    write(root/'public_roster.json', dict(rows=rows, evaluation_cohort_reads=False,
                                         cal_reference_reads=False))
    info = dict(affine=dict(a=A, b=INTERCEPT), B=B, bound_B=B,
        seed=SEED, original_train_frames=56, train_points=1591791, cal_frames=304,
        steps=600, batch_points=28672, batch_sampling='numpy.default_rng uniform replacement per frame',
        offsets=offsets, original_bank_path=str(bankpath.resolve()), original_bank_sha256=sha(bankpath),
        source_public_roster_sha256=sha(rosterpath), train_reference_target_reads=True,
        cal_reference_reads=False, evaluation_cohort_reads=False,
        normalization_mean=mean.tolist(), normalization_std=std.tolist(),
        feature_names=load(root/'plan.json')['residual']['features'],
        files={name:dict(path=str((root/name).resolve()), sha256=sha(root/name)) for name in
          ('normalization.npz','train_features.npy','train_targets.npy','train_identity.npz','batch_indices.npy','public_roster.json')})
    write(root/'training_inputs.json', info)
    receipt.update(train_points=len(flatids), public_frames=len(rows), status='COMPLETE')


def model_factory(torch):
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(torch.nn.Linear(11,64),torch.nn.ReLU(),
        torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def predict(torch, root, rows, models, receipt, remaining, cohort):
    folder = root/'predictions'/cohort; folder.mkdir(parents=True)
    outputs = []
    for index, row in enumerate(rows):
        check(remaining, 5)
        features = torch.from_numpy(np.load(row['feature_path']).reshape(-1,11)).cuda()
        with np.load(row['sampled_depth_path']) as f:
            depth = np.asarray(f['depth'], np.float64)
        valid = np.isfinite(depth)&(depth>0)
        affine = np.full(depth.shape, np.nan, np.float64)
        affine[valid] = A*np.log(depth[valid])+INTERCEPT
        output = {k:v for k,v in row.items() if k in PUBLIC or k in
            ('cohort','sampled_depth_path','sampled_depth_sha256','prediction_path','prediction_sha256')}
        output['distributions'] = {}
        for arm, model in models.items():
            with torch.no_grad():
                x = features.clone() if arm == ARMS[0] else features
                if arm == ARMS[0]:
                    x[:,3:] = 0
                residual = (B*torch.tanh(model(x).squeeze(-1)/B)).cpu().numpy().reshape(depth.shape)
            assert np.all(np.isfinite(residual[valid])) and np.max(np.abs(residual[valid])) <= B+1e-6
            mu = affine+residual.astype(np.float64)
            residual[~valid] = np.nan
            path = folder/f'{index:03d}_{arm}.npz'
            np.savez(path, mu=mu, residual=residual, valid=valid, affine_mu=affine)
            output['distributions'][arm] = dict(path=str(path.resolve()), sha256=sha(path))
        outputs.append(output)
        receipt['predicted_frames'] = receipt.get('predicted_frames',0)+1
        if index % 80 == 0:
            print(json.dumps(dict(stage='predict', cohort=cohort, frames=index+1)), flush=True)
    write(folder/'predictions.json', dict(status='COMPLETE',rows=outputs,B=B,
        arms=list(ARMS),evaluation_reference_reads=False,cal_reference_reads=False))


def gpu_stage(repo, root, receipt, remaining, stage, freeze):
    import torch
    torch.set_num_threads(4)
    receipt.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(0), matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(), training_dtype='float32')
    models = {}
    if stage == 'train':
        inputs = load(root/'training_inputs.json')
        features_cpu = torch.from_numpy(np.load(root/'train_features.npy'))
        targets_cpu = torch.from_numpy(np.load(root/'train_targets.npy'))
        batches = np.load(root/'batch_indices.npy')
        idx = torch.from_numpy(batches[0].astype(np.int64))
        benchmark = []
        benchstart = time.perf_counter()
        for device in ('cuda','cpu'):
            m = model_factory(torch).to(device)
            x = features_cpu[idx].to(device); y = targets_cpu[idx].to(device)
            elapsed = []
            for _ in range(3):
                t = time.perf_counter(); m.zero_grad(set_to_none=True)
                loss = torch.nn.functional.smooth_l1_loss(B*torch.tanh(m(x).squeeze(-1)/B), y, beta=.2)
                loss.backward()
                if device=='cuda': torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-t)
            benchmark.append(dict(device=device,seconds=elapsed,median_s=float(np.median(elapsed))))
            del m,x,y,loss
        assert time.perf_counter()-benchstart <= 30
        # Benchmark chooses placement only. Fresh model and predeclared batches.
        device = min(benchmark,key=lambda row:row['median_s'])['device']
        write(root/'benchmark.json',dict(rows=benchmark,selected_device=device,
            wall_s=time.perf_counter()-benchstart,optimizer_steps=0,batch_rng_consumed=False))
        features = features_cpu.to(device); targets = targets_cpu.to(device)
        schedule = torch.from_numpy(batches.astype(np.int64)).to(device)
        del features_cpu,targets_cpu
        initialhash = None
        for arm in ARMS:
            model = model_factory(torch).to(device)
            currenthash = state_sha(model)
            if initialhash is None: initialhash=currenthash
            assert currenthash == initialhash
            assert sum(p.numel() for p in model.parameters()) == 4993
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            progress = []; armstart=time.perf_counter()
            for step in range(600):
                check(remaining, 10)
                x = features[schedule[step]]
                if arm==ARMS[0]: x[:,3:]=0
                y = targets[schedule[step]]
                optimizer.zero_grad(set_to_none=True)
                residual = B*torch.tanh(model(x).squeeze(-1)/B)
                if step==0: assert torch.count_nonzero(residual).item()==0
                loss=torch.nn.functional.smooth_l1_loss(residual,y,beta=.2)
                loss.backward(); optimizer.step()
                receipt.setdefault('completed_steps',{})[arm]=step+1
                if step==0 or (step+1)%100==0:
                    if device=='cuda':torch.cuda.synchronize()
                    progress.append(dict(step=step+1,loss=float(loss.item()),arm_wall_s=time.perf_counter()-armstart))
                    checkpoint=dict(state_dict=model.cpu().state_dict(),arm=arm,step=step+1,B=B,
                        affine=dict(a=A,b=INTERCEPT),initial_state_sha256=initialhash,
                        batch_schedule_sha256=inputs['files']['batch_indices.npy']['sha256'],
                        parent_plan_sha256=receipt['parent_plan_sha256'],source_sha256=receipt['source_sha256'],
                        training_inputs_sha256=sha(root/'training_inputs.json'))
                    torch.save(checkpoint,root/f'{arm}_step{step+1:03d}.pt')
                    model.to(device)
                    write(root/f'{arm}_progress.json',dict(rows=progress,initial_state_sha256=initialhash))
                    print(json.dumps(dict(stage='train',arm=arm,**progress[-1])),flush=True)
            models[arm]=model.cuda().eval()
            del optimizer,x,y,residual,loss
        receipt.update(initial_state_sha256=initialhash,selected_device=device)
        del features,targets,schedule,batches,idx
        rows=load(root/'public_roster.json')['rows'][56:]
        assert len(rows)==304
        predict(torch,root,rows,models,receipt,remaining,'cal')
    else:
        frozen=load(freeze)
        assert frozen['status']=='FROZEN_BEFORE_EVALUATION' and frozen['cal_frames']==304
        assert frozen['evaluation_truth_read'] is False and set(frozen['arms'])=={'affine',*ARMS}
        receipt['calibration_freeze_sha256']=sha(freeze)
        norm=np.load(root/'normalization.npz');mean=norm['mean'];std=norm['std']
        for arm in ARMS:
            ckpt=torch.load(root/f'{arm}_step600.pt',map_location='cpu',weights_only=False)
            assert ckpt['step']==600 and ckpt['B']==B
            model=model_factory(torch);model.load_state_dict(ckpt['state_dict'])
            models[arm]=model.cuda().eval()
        base=repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009'
        specs=[(name,base/'fixed-grid-sensor'/name,count) for name,count in
            [('original_validation24',24),('new_3rscan64',64),('arkit16',16)]]
        specs += [(f'arkit_{name}',base/'additional-arkit-sensor'/name,16) for name in ('40777060','40777065')]
        evalfeatures=root/'eval-features';evalfeatures.mkdir()
        for cohort, folder, count in specs:
            observations=load(folder/'observations.json')['rows']
            predictions=load(folder/'depthpro/predictions.json')['rows']
            lookup={(r['scan'],r['frame']):r for r in predictions}
            assert len(observations)==count
            rows=[]
            for i, observation in enumerate(observations):
                check(remaining,5)
                row={k:v for k,v in observation.items() if k in PUBLIC}
                dp=lookup[(row['scan'],row['frame'])]
                assert sha(dp['path'])==dp['sha256']
                with np.load(dp['path']) as f:
                    native=sample_prediction(f['depth'],*public_mapping(row))
                raw,valid=raw_features(native,row['depth_K'])
                nativepath=evalfeatures/f'{cohort}_{i:03d}_depth.npz';np.savez(nativepath,depth=native)
                featurepath=evalfeatures/f'{cohort}_{i:03d}.npy';np.save(featurepath,((raw-mean)/std).astype(np.float32))
                row.update(cohort=cohort,sampled_depth_path=str(nativepath.resolve()),sampled_depth_sha256=sha(nativepath),
                    feature_path=str(featurepath.resolve()),feature_sha256=sha(featurepath),
                    prediction_path=dp['path'],prediction_sha256=dp['sha256'])
                rows.append(row)
            predict(torch,root,rows,models,receipt,remaining,cohort)
    receipt.update(status='COMPLETE',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    models.clear();gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','predict-eval'))
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--calibration-freeze',type=Path)
    args=parser.parse_args()
    root=args.output or args.repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010'
    plan=load(root/'plan.json')
    assert plan['residual']['bound_B']==B and plan['residual']['steps']==600
    assert plan['affine']==dict(a=A,b=INTERCEPT) and plan['residual']['arms']==list(ARMS)
    terminal=root/f'{args.stage}_terminal.json'
    if terminal.exists():raise FileExistsError('Stage already has terminal: preserve lineage; no implicit overwrite/restart')
    writer=root/'producer_writer.json'
    with writer.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),stage=args.stage),f)
    budgetkind='cpu' if args.stage=='prepare' else 'gpu'
    ledgerpath=root/f'{budgetkind}_ledger.json'
    ledger=load(ledgerpath) if ledgerpath.exists() else dict(attempts=[],used_wall_s=0)
    budget=180 if budgetkind=='cpu' else 900
    remaining=budget-ledger['used_wall_s']
    snapshot=root/f'{args.stage}_executed.py';shutil.copyfile(__file__,snapshot)
    receipt=dict(status='STARTING',stage=args.stage,pid=os.getpid(),source_sha256=sha(__file__),
        source_snapshot_path=str(snapshot.resolve()),parent_plan_sha256=sha(root/'plan.json'),
        budget_kind=budgetkind,cumulative_budget_s=budget,previous_used_s=ledger['used_wall_s'],
        evaluation_reference_reads=False,cal_reference_reads=False,new_depthpro_inference_calls=0,download_bytes=0)
    try:
        if args.stage=='prepare':prepare(args.repo,root,receipt,remaining)
        else:gpu_stage(args.repo,root,receipt,remaining,args.stage,args.calibration_freeze)
    except BaseException as error:
        receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        print(receipt['traceback'],flush=True)
        raise
    finally:
        if args.stage!='prepare':
            import torch
            gc.collect();torch.cuda.empty_cache()
            receipt['cuda_allocated_after_cleanup_bytes']=torch.cuda.memory_allocated()
        elapsed=time.perf_counter()-START
        receipt.update(stage_wall_s=elapsed,cumulative_wall_s=ledger['used_wall_s']+elapsed,
            within_budget=ledger['used_wall_s']+elapsed<=budget)
        if not receipt['within_budget']:receipt['status']='BUDGET_VIOLATION'
        ledger['attempts'].append(receipt);ledger['used_wall_s']+=elapsed
        write(ledgerpath,ledger);write(terminal,receipt)
        writer.unlink()
        print(json.dumps(dict(terminal=receipt)),flush=True)


if __name__=='__main__':main()
