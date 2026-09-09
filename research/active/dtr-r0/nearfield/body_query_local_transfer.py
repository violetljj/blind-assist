"""Frozen LOCAL challenger on consumed transfer cohorts, no fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from body_query_context_evidence import ContextEvidence
from body_query_collection_labels import read,write,sha
from body_query_background_analysis import score,ranges
from body_query_fresh_size_eval import FROZEN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    work=Path('artifacts.local/work');baseline=work/'body-query-10000-b-20260909/run-v1';decoder=work/'body-query-context-decoder-20260909/run-v1'
    for n,h in FROZEN.items():assert sha((baseline if n=='NEW-step2000.pt' else decoder)/n)==h
    local_sha=sha(decoder/'LOCAL.pt');assert local_sha=='15d0fc5acd0a26de1a6b0fe32a908a307e7be77ce9e0c37e7c73610693995ca8'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model=ContextEvidence(baseline,decoder,work/'body-query-v1-20260908/model-inputs/pretrained',arm='LOCAL').cuda().eval()
    inputs=[('original_fixture',work/'body-query-background-only-20260909','admission-visible-head-v2'),('simplified_fixture',work/'body-query-fresh-size-20260909','admission-v1')]
    summary={};start=time.perf_counter();total=0;improvements=[]
    for name,root,adname in inputs:
        ad=root/'returned-full-capture-v1'/adname;receipt=read(ad/'admission.json');reference=read(root/'inference-v1/result.json')
        assert sha(ad/'admission.json')==reference['admission_sha256']
        assert sha(ad/'frame_metadata.json')==receipt['frame_metadata_sha256']
        assert sha(ad/'evaluator_truth.npz')==receipt['truth_sha256']
        assert sha(root/'inference-v1/predictions.npz')==reference['predictions_sha256']
        rows=read(ad/'frame_metadata.json');saved=np.load(root/'inference-v1/predictions.npz',allow_pickle=False)
        arrays={k:[] for k in ('LOCAL_counts','LOCAL_ranges','retained_alerts','BASE_counts')}
        with torch.inference_mode():
            for begin in range(0,len(rows),16):
                rgb=[]
                for row in rows[begin:begin+16]:
                    path=root/'returned-full-capture-v1/rgb'/row['region_id']/f"{row['capture_index']:04d}.png";assert sha(path)==row['rgb_sha256']
                    with Image.open(path) as image:rgb.append(np.array(image.convert('RGB').resize((256,144),Image.Resampling.BOX)))
                result=model(torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).cuda().float()/255)
                for key,value in [('LOCAL_counts',result['geometry_count_logits'].softmax(-1)),('LOCAL_ranges',result['range_probabilities']),('retained_alerts',result['alerts']),('BASE_counts',result['legacy_count_logits'].softmax(-1))]:arrays[key].append(value.cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(a.output/(name+'.npz'),**arrays)
        parity=(arrays['retained_alerts']==saved['BASE_alerts']).all(1);base_error=float(abs(arrays['BASE_counts']-saved['BASE_counts']).max());assert base_error<2e-6 and parity.all()
        assert np.array_equal(ranges(arrays['LOCAL_counts'])>=.5,arrays['LOCAL_ranges']>=.5)
        truth=np.load(ad/'evaluator_truth.npz',allow_pickle=False)['counts'];metrics={}
        for arm in sorted({r['arm'] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r['admitted'] and r['arm']==arm]
            local=score(arrays['LOCAL_counts'],truth,rows,ids,dict(numerator=int(parity[ids].sum()),denominator=len(ids)))
            joint=score(saved['JOINT_counts'],truth,rows,ids,reference['metrics'][arm]['JOINT']['original_alert_parity'])
            assert joint==reference['metrics'][arm]['JOINT']
            change=(local['pair_correct']['numerator']-joint['pair_correct']['numerator'])/local['pair_correct']['denominator'];improvements.append(change)
            metrics[arm]=dict(LOCAL=local,JOINT=joint,strict_pair_delta=change)
        summary[name]=dict(metrics=metrics,all_frame_parity=int(parity.sum()),frames=len(rows),baseline_probability_error=base_error,source_primary_status=receipt.get('original_primary_status',receipt['status']),admission_sha256=sha(ad/'admission.json'),predictions_sha256=sha(a.output/(name+'.npz')))
        total+=len(rows)
    assert total==600
    decision='LOCAL_ROBUSTNESS_CHALLENGER_SUPPORTED' if all(x>=.1 for x in improvements) else 'LOCAL_READOUT_ALONE_NOT_SUFFICIENT_FOR_TRANSFER'
    write(a.output/'result.json',dict(status='PASS',decision=decision,cohorts=summary,frames=total,training_steps=0,local_sha256=local_sha,frozen_hashes=FROZEN,source_sha256=sha(__file__),device=torch.cuda.get_device_name(0),seconds=time.perf_counter()-start,promotion=False))
    print(decision);print(summary)


if __name__=='__main__':main()
