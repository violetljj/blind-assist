"""Independent packet/count audit and unchanged-weight compact ensemble export."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble, read, write, sha, load_npz
from mz43_restricted_training import conditions, packet_features, CONDITIONS


def scalar_metrics(logits, truth, known):
    counts = {name: [0]*4 for name in ('tp', 'fp', 'fn', 'tn')}
    exact = 0
    for row in range(len(truth)):
        correct = True
        for q in range(4):
            if not known[row, q]:
                correct = False
                continue
            positive, target = bool(logits[row, q] >= 0), bool(truth[row, q])
            name = ('tp' if target else 'fp') if positive else ('fn' if target else 'tn')
            counts[name][q] += 1
            correct &= positive == target
        exact += int(correct)
    return dict(attempted_frames=len(truth), known=known.sum(0).tolist(),
                unknown=(~known).sum(0).tolist(), **counts,
                complete_frames=int(known.all(1).sum()), exact_frames=exact)


def run(root, run_dir, output):
    root=root.resolve(); output=output.resolve()
    assert output.is_relative_to((root/'artifacts.local').resolve()) and not output.exists()
    receipt=read(run_dir/'receipt.json'); assert receipt['status']=='PASS'
    for name, digest in receipt['outputs'].items(): assert sha(run_dir/name)==digest, name
    for path, digest in receipt['inputs'].items(): assert sha(path)==digest, path
    output.mkdir(parents=True); started=time.perf_counter()
    result=read(run_dir/'result.json'); saved=load_npz(run_dir/'predictions.npz')
    old=root/'artifacts.local/work/mz1-tiny-fusion-20260910'
    data=load_npz(old/'cache-v1/features.npz')
    np.testing.assert_array_equal(saved['original_alerts'], data['original_alerts'])
    schedule=np.load(run_dir/'schedule.npy'); choices=np.load(run_dir/'condition-schedule.npy')
    assert (data['role'][schedule]=='TRAIN_ONLY').all()
    np.testing.assert_array_equal(choices.ravel(), np.tile([0,1,2],12800))
    all_tof=conditions(data['tof']); scalar_packet_slots=0
    # Independent scalar reconstruction on every TRAIN packet, both restrictions.
    for i in np.flatnonzero(data['role']=='TRAIN_ONLY'):
        for ci, mode in enumerate(CONDITIONS[1:], 1):
            expect=data['tof'][i].copy()
            for zone in range(64):
                a,b=2*zone,2*zone+1
                r0,r1=(float(data['tof'][i,j]*np.float32(4)) for j in (a,b))
                valid0,valid1=(bool(data['tof'][i,128+j]) for j in (a,b))
                if valid0 and valid1 and r1-r0 < .600:
                    if mode=='MERGE_CLOSE':
                        expect[a]=np.float32((r0+r1)*.5)/np.float32(4)
                        expect[b]=0; expect[128+b]=0
                    else:
                        expect[[a,b,128+a,128+b]]=0
                scalar_packet_slots+=2
            np.testing.assert_array_equal(expect, all_tof[ci,i])
    score_bits=0
    for condition, roles in result['old'].items():
        for role, methods in roles.items():
            ix=data['role']==role
            for name, expected in methods.items():
                actual=scalar_metrics(saved['old/'+condition+'/'+name][ix],data['truth'][ix],saved['old_known'][ix])
                assert actual==expected, (condition,role,name)
                score_bits+=int(ix.sum())*4
    for condition, block in result['MZ36'].items():
        for name, expected in block['metrics'].items():
            actual=scalar_metrics(saved['MZ36/'+condition+'/'+name],saved['mz36_truth'],saved['mz36_known'])
            assert actual==expected,(condition,name)
            score_bits+=1600
    assert (~saved['mz36_known']).sum()==80
    rgb=torch.load(old/'run-v1/RGB_ONLY.pt',map_location='cpu',weights_only=True)
    tof=torch.load(run_dir/'MIXED-TOF_ONLY.pt',map_location='cpu',weights_only=True)
    compact=CompactEnsemble.from_original(rgb,tof)
    torch.save(compact.state_dict(),output/'compact.pt')
    compact=CompactEnsemble.from_checkpoint(output/'compact.pt').cuda().eval()
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    compact_error={}; compared=0; inference_seconds=0.
    try:
        def compare(name,visual,features,expected):
            nonlocal compared,inference_seconds
            values=[]
            with torch.inference_mode():
                for begin in range(0,len(visual),256):
                    v=torch.from_numpy(visual[begin:begin+256]).cuda()
                    t=torch.from_numpy(features[begin:begin+256]).cuda()
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    pred=compact(v,t)
                    torch.cuda.synchronize(); inference_seconds+=time.perf_counter()-tick
                    values.append(pred.cpu().numpy())
            logits=np.concatenate(values)
            np.testing.assert_array_equal(logits>=0,expected>=0)
            error=float(np.max(np.abs(logits-expected)))
            assert error<1e-5, (name,error)
            compact_error[name]=error; compared+=len(visual)
        for ci,condition in enumerate(CONDITIONS):
            compare('old/'+condition,data['visual'],all_tof[ci],saved['old/'+condition+'/MIXED/ENSEMBLE'])
        for condition in CONDITIONS:
            folder=root/'artifacts.local/work/mz36-new-source-20260910/inference-v1' if condition=='IDEAL' else root/'artifacts.local/work/mz40-l8cx-constrained-20260910'/condition
            raw=load_npz(folder/'predictions.npz')
            compare('MZ36/'+condition,raw['visual'],packet_features(raw['ranges'],raw['valid']),saved['admitted/'+condition+'/MIXED/ENSEMBLE'])
        audit=dict(status='PASS',run_receipt_sha256=sha(run_dir/'receipt.json'),
            source_sha256=sha(__file__),train_only_exposures=38400,mixed_exposures_per_condition=[12800]*3,
            scalar_packet_slots=scalar_packet_slots,independently_scored_bits=score_bits,
            original_alert_parity=5000,mz36_unknown_bits=80,
            compact=dict(path='compact.pt',sha256=sha(output/'compact.pt'),bytes=(output/'compact.pt').stat().st_size,
                parameters=sum(p.numel() for p in compact.parameters()),frames_compared=compared,
                all_decisions_identical=True,max_logit_error=compact_error,
                timing_scope='Cached head GPU compute over validation batches; excludes backbone, I/O and device pipeline',
                accumulated_cuda_seconds=inference_seconds,device=torch.cuda.get_device_name()),
            seconds=time.perf_counter()-started,
            change='Remove always-zero first-layer columns only; frozen RGB plus matched restricted-training ToF. No refit/threshold changes.')
        write(output/'receipt.json',audit)
        print('AUDIT',audit,flush=True)
    finally:
        del compact
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.root,args.run,args.output)
