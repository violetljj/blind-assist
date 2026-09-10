"""Reconstruct simulator return contributors, evaluator-only labels."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz9_contributors import reconstruct


def main(cache,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt=read(cache/'receipt.json');assert receipt['status']=='PASS'
    data=load_npz(cache/'observations.npz');assert sha(cache/'observations.npz')==receipt['files']['observations.npz']
    records=receipt['inputs'];assert len(records)==3700
    assert torch.cuda.is_available();torch.set_num_threads(1)
    source=[];queries=[];known=[];maxerr=0
    for begin in range(0,3700,8):
        depths=[]
        for row in records[begin:begin+8]:
            path=Path(row['native']);assert sha(path)==row['native_sha'];depths.append(np.load(path,allow_pickle=False))
        result=reconstruct(torch.from_numpy(np.stack(depths)).cuda())
        valid=result['valid'].cpu().numpy();ranges=torch.nan_to_num(result['range_m'],nan=0).cpu().numpy()
        np.testing.assert_array_equal(valid,data['valid'][begin:begin+len(valid)])
        err=float(np.abs(ranges-data['ranges'][begin:begin+len(valid)]).max());maxerr=max(maxerr,err);assert err<1e-6
        for value in [result['source_counts'],result['query_counts'],result['cell_known_counts']]:assert int(value.max())<65536
        source.append(result['source_counts'].cpu().numpy().astype(np.uint16))
        queries.append(result['query_counts'].cpu().numpy().astype(np.uint16))
        known.append(result['cell_known_counts'].cpu().numpy().astype(np.uint16))
        if begin%400==0:print('CONTRIBUTORS',begin+len(valid),'/3700',flush=True)
    s,q,k=np.concatenate(source),np.concatenate(queries),np.concatenate(known)
    assert (q<=s[...,None]).all()
    np.savez_compressed(output/'evaluator.npz',source_counts=s,query_counts=q,cell_known_counts=k)
    write(output/'receipt.json',dict(status='PASS',frames=3700,packet_max_error=maxerr,
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        cache_receipt_sha256=sha(cache/'receipt.json'),labels_sha256=sha(output/'evaluator.npz'),
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz9_contributors.py')]},
        interpretation='Actual simulated contributing pixels, evaluator/training only; not model input or hardware guarantee'))
    print('PASS',maxerr,time.perf_counter()-started,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.cache,a.output)
