"""Train-only bounded learner diagnosis; existing six Development layouts only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from cnh_rgb_alley_v2 import AzimuthMaskedFrustumFusion, train_pos_weight, COLLECTION_SHA256, PARTITION_SHA256
from cnh_rgb_dev_comparison import read_inputs, checked, sha, SEEDS
from cnh_rgb_visible_depth_ceiling import depth_batch, native_support
from cnh_rgb_visible_depth_audit import load_scene_depth, visible_witness
from cnh_rgb_visible_depth_metrics import rank_metrics


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stats(a):
    a = np.asarray(a)
    finite = a[np.isfinite(a)]
    return dict(min=float(finite.min()) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None,
                mean=float(finite.mean()) if len(finite) else None,
                zero_fraction=float(np.mean(a == 0)), finite_fraction=float(np.isfinite(a).mean()))


def select32(data):
    # Label-blind fixed spread over all three train layouts, not dev selection.
    result = []
    names = sorted({r['layout_id'] for i, r in enumerate(data['rows']) if data['train'][i]})
    for name, count in zip(names, (11, 11, 10)):
        ids = np.array([i for i, r in enumerate(data['rows']) if r['layout_id'] == name and data['train'][i]])
        result.extend(ids[np.linspace(0, len(ids)-1, count).round().astype(int)])
    return np.array(result)


def overfit(data, depth, support, ids, arm, output, steps=1200, seconds=1200):
    seed = 20260924
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    model = AzimuthMaskedFrustumFusion().to(device)
    weight, _ = train_pos_weight(data['labels'], data['train'])
    labels = torch.tensor(np.maximum(data['labels'][ids], 0), dtype=torch.float32, device=device)
    known = torch.tensor(data['labels'][ids] >= 0, device=device)
    weight = torch.tensor(weight, device=device)
    if arm == 'visible_depth_full':
        rgb = depth_batch(depth, ids, device)
        hist = torch.zeros((32,64,16), device=device)
        ambient = scalar = age = torch.zeros((32,64), device=device)
        valid = torch.zeros((32,64), dtype=torch.bool, device=device)
    else:
        rgb = torch.zeros((32,3,72,128), device=device)
        hist = torch.tensor(data['histogram'][ids], device=device)
        ambient = torch.tensor(data['ambient'][ids], device=device)
        scalar = torch.tensor(data['scalar'][ids], device=device)
        valid = torch.tensor(data['valid'][ids], device=device)
        age = torch.zeros_like(ambient)
    support = torch.tensor(support, device=device)
    enc = list(model.rgb.parameters())
    enc_ids = {id(p) for p in enc}
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.parameters() if id(p) not in enc_ids], 'lr':3e-4},
        {'params':enc, 'lr':3e-5}], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    gradients = {}
    begin = time.monotonic()
    stop = 'STEP_LIMIT'
    for step in range(1, steps+1):
        indices = torch.randperm(32, generator=generator)[:16].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb[indices], hist[indices], ambient[indices], scalar[indices], valid[indices], age[indices], support)['occupancy_logits']
        loss = F.binary_cross_entropy_with_logits(logits, labels[indices], pos_weight=weight, reduction='none')[known[indices]].mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite overfit loss')
        loss.backward()
        if step == 1:
            gradients = {name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        if step == 1 or step % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(rgb[i:i+16], hist[i:i+16], ambient[i:i+16], scalar[i:i+16], valid[i:i+16], age[i:i+16], support)['occupancy_logits'] for i in (0,16)]).cpu().numpy()
            point = dict(step=step, loss=float(loss), **rank_metrics(data['labels'][ids], pred), wall_s=time.monotonic()-begin)
            history.append(point)
            save(output/f'{arm}-overfit-progress.json', dict(status='RUNNING', history=history))
            print(json.dumps(dict(arm=arm, **point)), flush=True)
            if point['auprc'] >= .99:
                stop = 'TRAIN_AP_AT_LEAST_0.99'
                break
        if time.monotonic()-begin >= seconds:
            stop = 'WALL_LIMIT'
            break
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(rgb[i:i+16], hist[i:i+16], ambient[i:i+16], scalar[i:i+16], valid[i:i+16], age[i:i+16], support)['occupancy_logits'] for i in (0,16)]).cpu().numpy()
    np.savez_compressed(output/f'{arm}-overfit.npz', indices=ids, labels=data['labels'][ids], logits=pred)
    torch.save(model.state_dict(), output/f'{arm}-overfit.pt')
    result = dict(arm=arm, seed=seed, frames=32, steps=step, stop=stop,
                  final=rank_metrics(data['labels'][ids],pred), history=history,
                  first_step_gradient_norms=gradients, wall_s=time.monotonic()-begin,
                  optimization='Same V2 AdamW learning rates/weight decay and train class weights; constant LR, no cosine; 16 of 32 frames each step; no dev evaluation or tuning')
    save(output/f'{arm}-overfit-result.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    collection=Path('artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json')
    partition=Path('artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json')
    prepared=Path('artifacts.local/evidence/cnh-visible-depth-audit-20260925-v1')
    assert sha(collection)==COLLECTION_SHA256 and sha(partition)==PARTITION_SHA256
    data=read_inputs(collection,partition)
    receipt=json.loads((prepared/'result.json').read_text())
    assert all(sha(prepared/name)==digest for name,digest in receipt['files'].items())
    depth=np.load(prepared/'visible-depth-f32.npy',mmap_mode='r')
    frames=[]
    for item in data['collection']['layouts']:
        frames.extend(json.loads(checked({'path':item['overlay'],'sha256':item['overlay_sha256']}).read_text(encoding='utf-8-sig'))['frames'])
    assert [f['frame_key'] for f in frames]==[r['frame_key'] for r in data['rows']]
    audit=[]
    with np.load(prepared/'perfect-tof-h3.npz') as f:
        witnesses=f['visible_witness'].copy()
    for i,frame in enumerate(frames):
        source,camera=load_scene_depth(frame)
        tensor=depth_batch(depth,np.array([i]),torch.device('cpu')).numpy()[0]
        match=np.array_equal(source,depth[i],equal_nan=True)
        assert match and np.array_equal(visible_witness(source,camera),witnesses[i])
        expected=np.log1p(np.where(np.isfinite(source),source,0))/np.log(101)
        assert np.allclose(tensor[0],expected,atol=1e-7,rtol=1e-6)
        audit.append(dict(frame_key=frame['frame_key'], split='train' if data['train'][i] else 'dev',
                          raw_m=stats(source), model_depth=stats(tensor[0]), model_valid=stats(tensor[1]),
                          third_channel=stats(tensor[2]), source_exact=match))
        if (i+1)%160==0:
            print(json.dumps(dict(depth_audited=i+1,total=960)),flush=True)
    save(out/'depth-per-frame.json',audit)
    original=[]
    for arm,folder in [('tof_only','cnh-rgb-alley-dev-v2-20260925-v1'),('visible_depth_full','cnh-visible-depth-learned-ceilings-20260925-v1')]:
        for seed in SEEDS:
            path=Path('artifacts.local/evidence')/folder/f'{arm}-seed-{seed}-predictions.npz'
            with np.load(path) as f:
                assert list(f['frame_key'].astype(str))==[r['frame_key'] for r in data['rows']]
                logits=f['logits']
            original.append(dict(arm=arm,seed=seed,train=rank_metrics(data['labels'][data['train']],logits[data['train']]),
                                 dev=rank_metrics(data['labels'][data['dev']],logits[data['dev']]),path=str(path),sha256=sha(path)))
    save(out/'original-train-dev.json',original)
    ids=select32(data)
    save(out/'overfit-selection.json',dict(indices=ids.tolist(), frame_keys=[data['rows'][i]['frame_key'] for i in ids],
                                         labels=data['labels'][ids].tolist(),rule='11/11/10 equally spaced frames from alphabetically sorted train layouts; no label/dev selection'))
    support,_=native_support(data)
    results=[]
    for arm,s in [('tof_only',data['support']),('visible_depth_full',support)]:
        results.append(overfit(data,depth,s,ids,arm,out))
    save(out/'result.json',dict(status='COMPLETE',depth_exact_matches=len(audit),original=original,overfit=results,
                               code_sha256=sha(__file__),device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                               claim_limit='Reused Development, no test/City/capture. Failed bounded overfit does not prove impossibility.'))


if __name__=='__main__':
    main()
