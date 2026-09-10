"""Quantify a documented target-separation mismatch; never modify frozen packets."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(root,output):
    root=root.resolve();output=output.resolve()
    assert output.is_relative_to((root/'artifacts.local').resolve()) and not output.exists()
    prior=root/'artifacts.local/work/mz36-new-source-20260910/inference-v1'
    source=prior/'predictions.npz';receipt_path=prior/'receipt.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['status']=='PASS'
    assert sha(source)==receipt['outputs']['predictions.npz']
    here=Path(__file__).resolve().parent
    files=[source,receipt_path,here/'multizone64_observation.py',
           here/'MZ39_L8CX_READOUT_AUDIT_PROTOCOL_20260910.md',Path(__file__)]
    inputs={str(p):sha(p) for p in files}
    output.mkdir(parents=True);started=time.perf_counter()
    with np.load(source,allow_pickle=False) as archive:
        ranges=archive['ranges'];valid=archive['valid'];ids=archive['frame_ids']
    assert ranges.shape==valid.shape==(380,64,2) and valid.dtype==np.bool_
    assert len(ids)==380 and len(set(ids.tolist()))==380
    assert np.isfinite(ranges[valid]).all() and (ranges[valid]>0).all() and (ranges[valid]<=4).all()
    assert not (valid[:,:,1]&~valid[:,:,0]).any()
    dual=valid.all(2);gap=ranges[:,:,1].astype(float)-ranges[:,:,0].astype(float)
    assert (gap[dual]>0).all()
    bad=dual&(gap<.600);equal=dual&(gap==.600)
    bins=[int((valid.sum(2)==n).sum()) for n in range(3)]
    # Independent scalar recount of every zone, with no NumPy masks reused.
    counts=[0,0,0];scalar_bad=[];scalar_equal=0;scalar_dual=0;scalar_small=0
    rr=ranges.tolist();vv=valid.tolist()
    for i,frame in enumerate(rr):
        for z,pair in enumerate(frame):
            flags=vv[i][z];n=sum(bool(v) for v in flags);counts[n]+=1
            if n==2:
                scalar_dual+=1;delta=pair[1]-pair[0]
                if delta<.600:scalar_bad.append((i,z))
                if delta==.600:scalar_equal+=1
                if delta<.100:scalar_small+=1
    vector_bad=[tuple(map(int,row)) for row in np.argwhere(bad)]
    assert counts==bins and scalar_bad==vector_bad
    assert scalar_dual==int(dual.sum()) and scalar_equal==int(equal.sum())
    assert scalar_small==int((dual&(gap<.100)).sum()) and sum(bins)==380*64
    violations=[dict(frame_id=str(ids[i]),zone=z,nearer_m=float(ranges[i,z,0]),
                     farther_m=float(ranges[i,z,1]),separation_m=float(gap[i,z])) for i,z in vector_bad]
    result=dict(status='PASS',frames=380,zones_per_frame=64,total_zones=380*64,
                valid_return_count_histogram=bins,dual_zones=int(dual.sum()),
                dual_frames=int(dual.any(1).sum()),sub600mm_zones=int(bad.sum()),
                sub600mm_frames=int(bad.any(1).sum()),equal600mm_zones=int(equal.sum()),
                sub100mm_zones=scalar_small,
                separation_quantiles_m={str(q):float(np.quantile(gap[dual],q)) for q in (0,.25,.5,.75,1)},
                decision='EXERCISED_IDEAL_READOUT_MISMATCH' if bad.any() else 'SEPARATION_CONSTRAINT_NOT_EXERCISED',
                limits='Readout diagnostic only; separated pairs are not certified valid; no degraded-model task result')
    write(output/'violations.json',violations);write(output/'result.json',result)
    assert all(sha(p)==h for p,h in inputs.items())
    write(output/'audit.json',dict(status='PASS',checks=['Independent scalar recount of24320zones',
          'Exact violating identities and strict600mm predicate','Source shape, ordering, validity and unchanged hashes'],
          result_sha256=sha(output/'result.json'),violations_sha256=sha(output/'violations.json')))
    write(output/'receipt.json',dict(status='PASS',backend='CPU NumPy/std library; TASK_NOT_GPU_SUITABLE',
          seconds=time.perf_counter()-started,training_steps=0,model_inference_frames=0,inputs=inputs,
          outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();run(a.root,a.output)
