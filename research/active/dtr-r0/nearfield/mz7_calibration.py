"""One DEV-fitted positive-scale/threshold comparator; no sequence fitting."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble, load_npz, read, sha, write


def metrics(z, truth):
    p=z>=0
    return dict(exact=int((p==truth).all(1).sum()), tp=(p&truth).sum(0).tolist(),
                fp=(p&~truth).sum(0).tolist(), positives=truth.sum(0).tolist())


def main(root, paired, output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cache=root/'artifacts.local/work/mz1-tiny-fusion-20260910/cache-v1/features.npz'
    checkpoint=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)==read(paired/'receipt.json')['inputs'][str(checkpoint)]
    d=load_npz(cache); dev=d['role']=='DEV_ONLY'
    assert dev.sum()==1000
    model=CompactEnsemble.from_checkpoint(checkpoint)
    with torch.inference_mode():
        r,t=model.branches(torch.from_numpy(d['visual'][dev]),torch.from_numpy(d['tof'][dev]))
    x=torch.stack((r,t)).double(); truth=d['truth'][dev].astype(bool)
    y=torch.from_numpy(truth).double()[None].expand(2,-1,-1)
    log_scale=torch.zeros((2,1,4),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([log_scale],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(x*log_scale.clamp(-6,6).exp(),y)
        loss.backward();return loss
    opt.step(closure)
    scales=log_scale.detach().clamp(-6,6).exp().numpy()[:,0]
    base=(r.numpy()+t.numpy())*.5
    calibrated=(x.numpy()*scales[:,None]).mean(0)
    budget=((base>=0)&~truth).sum(0)
    thresholds=[]
    for q in range(4):
        candidates=np.r_[np.unique(calibrated[:,q]),np.nextafter(calibrated[:,q].max(),np.inf)]
        scored=[]
        for threshold in candidates:
            p=calibrated[:,q]>=threshold
            tp=int((p&truth[:,q]).sum()); fp=int((p&~truth[:,q]).sum())
            if fp<=budget[q]:scored.append((tp,-fp,float(threshold)))
        thresholds.append(max(scored)[2])
    thresholds=np.array(thresholds)
    params=dict(scale=scales.tolist(),threshold=thresholds.tolist(),fit_role='DEV_ONLY',
        fit_frames=1000,cache_sha256=sha(cache),checkpoint_sha256=sha(checkpoint),
        baseline_dev=metrics(base,truth),calibrated_dev=metrics(calibrated-thresholds,truth))
    write(output/'parameters.json',params)
    # All fitting is finished before accessing the consumed MZ6 labels here.
    data=load_npz(paired/'branch-logits.npz'); results={}; predictions={}
    for condition in ['clean','stress']:
        z=(data[condition+'/RGB']*scales[0]+data[condition+'/ToF']*scales[1])*.5-thresholds
        predictions[condition]=z
        results[condition]={}
        for clip in ['ALL',*dict.fromkeys(data['clip'].tolist())]:
            mask=np.ones(len(z),dtype=bool) if clip=='ALL' else data['clip']==clip
            results[condition][clip]=dict(baseline=metrics(data[condition+'/CURRENT'][mask],data['truth'][mask]),
                                         calibrated=metrics(z[mask],data['truth'][mask]))
    np.savez_compressed(output/'predictions.npz',**predictions)
    write(output/'result.json',results)
    write(output/'receipt.json',dict(status='PASS',script_sha256=sha(Path(__file__)),
        paired_logits_sha256=sha(paired/'branch-logits.npz'),
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}, training_backbone=False))
    print(params)
    print(results['clean'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['root','paired','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();main(args.root,args.paired,args.output)
