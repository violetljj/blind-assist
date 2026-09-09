"""Frozen RGB plus clean metric observation ladder; no learned fusion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from body_query_collection_labels import read,write,sha


def metric(pred,true):
    count=lambda n,d:dict(numerator=int(n),denominator=int(d))
    p=pred.reshape(-1,2,2);t=true.reshape(-1,2,2);n=len(p)
    nearonly=t[:,:,0]&~t[:,:,1]
    bodyonly=t[:,0].any(-1)&~t[:,1].any(-1);headonly=t[:,1].any(-1)&~t[:,0].any(-1)
    return dict(spatial_exact=count((pred==true).all(-1).sum(),n),
      body_head_accuracy=count((p.any(-1)==t.any(-1)).all(-1).sum(),n),
      near_far_accuracy=count((p.any(1)==t.any(1)).all(-1).sum(),n),
      wrong_far=count((p[:,:,1]&nearonly).sum(),nearonly.sum()),
      body_to_head=count(p[bodyonly,1].any(-1).sum(),bodyonly.sum()),
      head_to_body=count(p[headonly,0].any(-1).sum(),headonly.sum()),
      event_confusion={name:dict(tp=int((pred[:,k]&true[:,k]).sum()),fp=int((pred[:,k]&~true[:,k]).sum()),
        fn=int((~pred[:,k]&true[:,k]).sum()),tn=int((~pred[:,k]&~true[:,k]).sum())) for k,name in enumerate(('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR'))})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['rgb','depth','score'],required=True)
    a=p.parse_args();work=Path('artifacts.local/work');dataset=work/'body-query-5000-20260909/dataset-v1'
    manifest=read(dataset/'manifest.json');assert manifest['status']=='PASS' and sha(dataset/'index.json')==manifest['index_sha256']
    rows=[r for r in read(dataset/'index.json')['frames'] if r['source_role']=='EVAL_ONLY'];assert len(rows)==1500
    assert all(r['status']=='PASS' for r in rows)
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if a.phase!='score':
        assert torch.cuda.is_available();torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    start=time.perf_counter()
    if a.phase=='rgb':
        from body_query_context_evidence import ContextEvidence
        from body_query_fresh_size_eval import FROZEN
        assert not (a.output/'rgb.npz').exists()
        base=work/'body-query-10000-b-20260909/run-v1';decoder=work/'body-query-context-decoder-20260909/run-v1'
        for name,digest in FROZEN.items():assert sha((base if name=='NEW-step2000.pt' else decoder)/name)==digest
        model=ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
        rgb_events=[];alerts=[];baseline_alerts=[];maxerr=0.
        with torch.inference_mode():
            for begin in range(0,len(rows),16):
                images=[]
                for row in rows[begin:begin+16]:
                    path=Path(row['rgb_file']);assert sha(path)==row['rgb_sha256']
                    with Image.open(path) as image:images.append(np.array(image.convert('RGB').resize((256,144),Image.Resampling.BOX)))
                x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
                out=model(x);original=model.base(x)
                maxerr=max(maxerr,float((out['legacy_count_logits']-original[2]).abs().max()))
                rgb_events.extend((out['range_probabilities']>=.5).cpu().numpy().reshape(-1,4))
                alerts.extend(out['alerts'].cpu().numpy());baseline_alerts.extend((original[0].sigmoid()>=model.alert_thresholds).cpu().numpy())
        assert np.array_equal(alerts,baseline_alerts) and maxerr<2e-6
        np.savez_compressed(a.output/'rgb.npz',events=rgb_events,alerts=alerts,baseline_alerts=baseline_alerts)
        write(a.output/'rgb-receipt.json',dict(status='PASS',frames=1500,source_index_sha256=sha(dataset/'index.json'),frozen_hashes=FROZEN,
          rgb_sha256=sha(a.output/'rgb.npz'),baseline_logit_max_error=maxerr,original_alert_parity=1500,training_steps=0,
          device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,operator_sha256=sha(__file__)))
        print('RGB PASS',time.perf_counter()-start)
    elif a.phase=='depth':
        from multizone64_observation import observe,center_events,native_events
        assert not (a.output/'depth.npz').exists()
        methods=[(size,readout) for size in (1,8) for readout in ('nearest','median','multi_surface')]
        arrays={};append=lambda key,value:arrays.setdefault(key,[]).append(value.cpu().numpy())
        for begin in range(0,len(rows),16):
            frames=[]
            for row in rows[begin:begin+16]:
                path=Path(row['native_file']);assert sha(path)==row['native_sha256']
                frames.append(np.load(path,allow_pickle=False))
            depth=torch.from_numpy(np.stack(frames)).cuda()
            for size,readout in methods:
                key=f'z{size}_{readout}';packet=observe(depth,zones_per_axis=size,readout=readout)
                decoded=center_events(packet)
                append(key+'_events',decoded['events']);append(key+'_range',packet['range_m']);append(key+'_valid',packet['valid'])
                append(key+'_observation_valid',decoded['observation_valid'])
            for crop,name in ((True,'D_crop'),(False,'D_full')):
                ob=native_events(depth,crop=crop);append(name+'_events',ob['events']);append(name+'_counts',ob['counts'])
        arrays={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(a.output/'depth.npz',**arrays)
        write(a.output/'depth-receipt.json',dict(status='PASS',frames=1500,source_index_sha256=sha(dataset/'index.json'),depth_sha256=sha(a.output/'depth.npz'),
          device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,operator_sha256=sha(__file__),
          observation_operator_sha256=sha(Path(__file__).with_name('multizone64_observation.py'))))
        print('DEPTH PASS',time.perf_counter()-start)
    else:
        assert not (a.output/'result.json').exists()
        rr=read(a.output/'rgb-receipt.json');dr=read(a.output/'depth-receipt.json')
        assert sha(a.output/'rgb.npz')==rr['rgb_sha256'] and sha(a.output/'depth.npz')==dr['depth_sha256']
        assert rr['source_index_sha256']==dr['source_index_sha256']==sha(dataset/'index.json')
        with np.load(a.output/'rgb.npz',allow_pickle=False) as z:rgb={k:z[k] for k in z.files}
        with np.load(a.output/'depth.npz',allow_pickle=False) as z:depth={k:z[k] for k in z.files}
        true=[]
        for row in rows:
            path=Path(row['label_file']);assert sha(path)==row['labels']['sha256']
            with np.load(path,allow_pickle=False) as z:counts=z['counts']
            assert np.array_equal(counts,row['counts'])
            true.append(counts.reshape(2,2,3).sum(-1).reshape(4)>=3)
        true=np.array(true);gate=np.repeat(rgb['alerts'],2,axis=1)
        pred={'A_RGB':rgb['events']};depth_only={}
        for key,v in depth.items():
            if key.endswith('_events'):
                name=key[:-7];pred[name]=v&gate;depth_only[name]=v
        scores={name:metric(v,true) for name,v in pred.items()}
        rawscores={name:metric(v,true) for name,v in depth_only.items()};baseline=scores['A_RGB'];gates={}
        for key in ('z8_nearest','z8_median','z8_multi_surface'):
            s=scores[key];cross=lambda m:m['body_to_head']['numerator']+m['head_to_body']['numerator']
            gates[key]=dict(exact_gain_pp=100*(s['spatial_exact']['numerator']-baseline['spatial_exact']['numerator'])/1500,
                wrong_far_halved=s['wrong_far']['numerator']<=baseline['wrong_far']['numerator']/2,
                cross_body_halved=cross(s)<=cross(baseline)/2,original_alert_parity=1500)
        np.savez_compressed(a.output/'scored.npz',truth=true,original_alerts=rgb['alerts'],**pred)
        write(a.output/'result.json',dict(status='PASS',scores=scores,depth_only_scores=rawscores,illustrative_gates=gates,
          full_native_vs_truth_mismatch=int((depth_only['D_full']!=true).any(1).sum()),
          conditions={c:sum(r['condition']==c for r in rows) for c in sorted({r['condition'] for r in rows})},
          original_alert_parity=1500,training_steps=0,frames=1500,source_index_sha256=sha(dataset/'index.json'),
          rgb_receipt_sha256=sha(a.output/'rgb-receipt.json'),depth_receipt_sha256=sha(a.output/'depth-receipt.json'),
          scored_sha256=sha(a.output/'scored.npz'),operator_sha256=sha(__file__),
          scope='Consumed clean simulated metric input; simple center readout, not information-theoretic ceiling or hardware result; no temporal scores'))
        print({k:v['spatial_exact'] for k,v in scores.items()});print(gates)


if __name__=='__main__':main()
