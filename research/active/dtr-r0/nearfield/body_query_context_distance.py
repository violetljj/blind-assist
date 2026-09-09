"""Frozen decoder transfer to admitted rigid near/far pairs; zero fits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,time
from pathlib import Path
import numpy as np
import torch
from body_query_context_decoder import CountDecoder,BASE_SHA
from body_query_model import BodyQueryModel,near_from_counts
from body_query_data import read,write,sha,fresh_output
from body_query_train import digest
from body_query_10000_distance_analysis import _load_rgb
from body_query_10000_readout_analysis import range_event_from_counts,confusion


def run(a):
    out=fresh_output(a.output);start=time.perf_counter();stage='preflight'
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        old_receipt=read(a.distance/'receipt.json');fit_receipt=read(a.decoder/'receipt.json')
        assert old_receipt['status']==fit_receipt['status']=='PASS'
        assert sha(a.baseline/'NEW-step2000.pt')==BASE_SHA
        for name,key in [('frame_metadata.json','frame_metadata_sha256'),('evaluator_truth.npz','evaluator_truth_sha256'),('predictions.npz','predictions_sha256')]:
            assert sha(a.distance/name)==old_receipt[key]
        rows=read(a.distance/'frame_metadata.json');assert len(rows)==5000
        fits=read(a.decoder/'fits.json');selection=read(a.decoder/'selection.json')
        selection_sha=sha(a.decoder/'selection.json');assert selection_sha==fit_receipt['selection_sha256']
        for arm in ['LOCAL','JOINT']:assert sha(a.decoder/f'{arm}.pt')==fits[arm]['sha256']
        normalization=np.load(a.decoder/'normalization.npz',allow_pickle=False)
        write(out/'protocol.json',dict(fits=0,frames=5000,baseline_sha256=BASE_SHA,
              decoder_hashes={arm:fits[arm]['sha256'] for arm in ['LOCAL','JOINT']},
              normalization_sha256=sha(a.decoder/'normalization.npz'),selection_sha256=selection_sha,
              distance_receipt_sha256=sha(a.distance/'receipt.json'),source_sha256=sha(Path(__file__)),
              scope='Consumed paired-source Development; no threshold fitting; BASE alert retained separately'))
        stage='RGB load';rgb,metadata=_load_rgb(rows);write(out/'frame_metadata.json',metadata)
        model=BodyQueryModel(a.pretrained,'B').cuda().eval();model.load_state_dict(torch.load(a.baseline/'NEW-step2000.pt',map_location='cuda',weights_only=True));before=digest(model.state_dict())
        xyz=(model.query_xyz*model.query_valid[:,:,None]).sum(1)/model.query_valid.sum(1)[:,None].clamp_min(1)
        decoders={}
        for arm in ['LOCAL','JOINT']:
            decoders[arm]=CountDecoder(arm,xyz).cuda().eval();decoders[arm].load_state_dict(torch.load(a.decoder/f'{arm}.pt',map_location='cuda',weights_only=True))
        mean=torch.as_tensor(normalization['mean'],device='cuda');std=torch.as_tensor(normalization['std'],device='cuda')
        capture={};hook=model.query_point.register_forward_pre_hook(lambda m,args:capture.update(raw=args[0]))
        arrays={key:[] for key in ['BASE_near','BASE_counts','LOCAL_near','LOCAL_counts','JOINT_near','JOINT_counts']}
        saved=np.load(a.distance/'predictions.npz',allow_pickle=False);parity=0.;stage='frozen RGB inference'
        try:
            with torch.inference_mode():
                for begin in range(0,len(rgb),32):
                    x=torch.from_numpy(rgb[begin:begin+32].copy()).cuda().permute(0,3,1,2).float()/255.
                    n,s,c=model(x);mask=model.query_valid[None,:,:,None]
                    raw=(capture['raw'][...,:64]*mask).sum(2)/mask.sum(2).clamp_min(1);raw=(raw-mean)/std
                    cp=c.softmax(-1).cpu().numpy();npred=n.sigmoid().cpu().numpy()
                    parity=max(parity,float(np.abs(cp-saved['NEW_counts'][begin:begin+32]).max()),float(np.abs(npred-saved['NEW_near'][begin:begin+32]).max()))
                    arrays['BASE_near'].append(npred);arrays['BASE_counts'].append(cp)
                    for arm,decoder in decoders.items():
                        logits=decoder(raw);arrays[arm+'_counts'].append(logits.softmax(-1).cpu().numpy());arrays[arm+'_near'].append(near_from_counts(logits).sigmoid().cpu().numpy())
                    if begin%512==0:write(out/'progress.json',dict(stage=stage,completed=min(begin+32,5000),total=5000))
        finally:hook.remove()
        assert parity<2e-6,parity
        assert digest(model.state_dict())==before
        arrays={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(out/'predictions.npz',**arrays)
        # Evaluator arrays are opened after all model outputs are fixed.
        gt=np.load(a.distance/'evaluator_truth.npz',allow_pickle=False);counts=gt['native_counts'].reshape(-1,12);events=gt['native_events']
        pairs={}
        for i,row in enumerate(rows):pairs.setdefault(row['pair_id'],{})[row['endpoint']]=i
        assert len(pairs)==2500 and all(set(v)=={'near','far'} for v in pairs.values())
        result={}
        scopes={'all':np.arange(5000),'eval':np.array([i for i,r in enumerate(rows) if r['source_partition']=='eval'])}
        scopes.update({'region:'+region:np.array([i for i,r in enumerate(rows) if r['region_id']==region]) for region in sorted({r['region_id'] for r in rows})})
        for scope,ids in scopes.items():
            selected=set(ids.tolist());ps=[p for p in pairs.values() if p['near'] in selected and p['far'] in selected];result[scope]={}
            for arm in ['BASE','LOCAL','JOINT']:
                c=arrays[arm+'_counts'];r=range_event_from_counts(c);n=arrays[arm+'_near'];thresholds=np.array(selection[arm]['thresholds'])
                delta=np.array([r[p['far'],1,1]-r[p['near'],1,1] for p in ps]);truth=events[ids,1]
                result[scope][arm]=dict(frames=len(ids),pairs=len(ps),far_higher=int((delta>1e-6).sum()),ties=int((np.abs(delta)<=1e-6).sum()),mean_delta=float(delta.mean()),
                  HEAD_near_query=confusion(1-c[ids,6:9,0],counts[ids,6:9]>0,.5),
                  HEAD_near_event=confusion(r[ids,1,0],truth[:,0],.5),HEAD_far_event=confusion(r[ids,1,1],truth[:,1],.5),
                  exact_HEAD_range=int(((r[ids,1]>=.5)==truth).all(1).sum()),
                  count_derived_HEAD_hits=int((n[ids,1]>=thresholds[1]).sum()),count_derived_BODY_FP=int((n[ids,0]>=thresholds[0]).sum()),
                  retained_BASE_HEAD_hits=int((arrays['BASE_near'][ids,1]>=selection['BASE']['thresholds'][1]).sum()),
                  retained_BASE_BODY_FP=int((arrays['BASE_near'][ids,0]>=selection['BASE']['thresholds'][0]).sum()))
        assert len(scopes['eval'])==1500 and sha(a.decoder/'selection.json')==selection_sha
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',fits=0,training_steps=0,seconds=time.perf_counter()-start,
              baseline_prediction_parity=parity,baseline_state_unchanged=True,backend='CUDA',device=torch.cuda.get_device_name(),
              result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz')))
        print(result['eval'],flush=True)
    except Exception as exc:write(out/'failure.json',dict(stage=stage,error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['baseline','decoder','distance','pretrained','output']:parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
