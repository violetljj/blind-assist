"""Exact-alert and numeric geometry parity for the two-output research interface."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time
from pathlib import Path
import numpy as np
import torch
from body_query_context_evidence import ContextEvidence
from body_query_data import QueryRGB,read,write,sha,fresh_output


def run(a):
    out=fresh_output(a.output);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model=ContextEvidence(a.baseline,a.decoder,a.pretrained).cuda().eval()
    data=QueryRGB(a.cache,'eval');base=np.load(a.decoder/'BASE-eval.npz',allow_pickle=False);joint=np.load(a.decoder/'JOINT-eval.npz',allow_pickle=False)
    thresholds=np.array(read(a.decoder/'selection.json')['BASE']['thresholds']);errors=dict(alert=0.,support=0.,geometry_count=0.)
    mismatches=0;range_codes=[];disagreements=[];start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(data.ids),32):
            ids=np.arange(begin,min(begin+32,len(data.ids)));r=model(data.tensor(ids,'cuda'))
            for key,pred,expected in [('alert',r['alert_logits'].sigmoid(),base['near'][ids]),('support',r['support_logits'].sigmoid(),base['support'][ids]),('geometry_count',r['geometry_count_logits'].softmax(-1),joint['counts'][ids])]:
                errors[key]=max(errors[key],float(np.abs(pred.cpu().numpy()-expected).max()))
            mismatches+=int((r['alerts'].cpu().numpy()!=(base['near'][ids]>=thresholds)).sum())
            range_codes.append(r['range_code'].cpu().numpy());disagreements.append(r['alert_range_disagreement'].cpu().numpy())
        x=data.tensor(np.array([0]),'cuda')
        def bench(call):
            for _ in range(5):call(x)
            times=[]
            for _ in range(30):
                torch.cuda.synchronize();t=time.perf_counter();call(x);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t))
            return dict(p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))
        timing=dict(baseline=bench(model.base),two_output=bench(model),scope='Warm batch1 CUDA forward; excludes preprocessing/IO;30 samples each')
    assert max(errors.values())<2e-6,errors
    assert mismatches==0,mismatches
    code=np.concatenate(range_codes);disagree=np.concatenate(disagreements)
    result=dict(status='PASS',frames=3000,alert_decision_mismatches=mismatches,probability_max_abs_error=errors,
           range_code_counts={name:[int((code[:,h]==v).sum()) for v in range(4)] for h,name in enumerate(['BODY','HEAD'])},
           code_meaning=['UNKNOWN_NO_ASSERTED_RANGE','NEAR','FAR','BOTH'],alert_range_disagreement=disagree.sum(0).tolist(),
           runtime=timing,seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),
           interface_sha256=sha(Path(__file__).with_name('body_query_context_evidence.py')),
           normalization_sha256=sha(a.decoder/'normalization.npz'),baseline_sha256=sha(a.baseline/'NEW-step2000.pt'),
           decoder_sha256=sha(a.decoder/'JOINT.pt'),selection_sha256=sha(a.decoder/'selection.json'),
           scope='Research interface engineering parity; unchanged alerts plus separate range evidence; no App promotion')
    write(out/'validation.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['baseline','decoder','pretrained','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
