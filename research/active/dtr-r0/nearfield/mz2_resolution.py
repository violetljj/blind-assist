"""Frozen MZ2 two-resolution matched fits and three separate frozen stresses."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from mz1_tiny_fusion import TinyFusion,state_sha
from mz0_clean import metric
from multizone64_observation import observe
from mz2_coarse_observation import observe4,encode
from body_query_collection_labels import read,write,sha


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stress_features(features):
    r=features[:,:128].reshape(-1,8,8,2).astype(np.float64)*4
    v=features[:,128:].reshape(-1,8,8,2).astype(bool)
    result={}
    noise=np.random.default_rng(61).normal(0,.05,r.shape)
    noisy=np.where(v,r+noise,0);nv=v&(noisy>0)&(noisy<=4)
    dropped=np.random.default_rng(67).random(v.shape[:-1])<.2
    shifted=np.zeros_like(r);sv=np.zeros_like(v)
    shifted[:,:,1:]=r[:,:,:-1];sv[:,:,1:]=v[:,:,:-1]
    for name,values,valid in (('gaussian05',noisy,nv),('dropout20',r,v&~dropped[...,None]),('right_shift1',shifted,sv)):
        result[name]=np.concatenate((np.where(valid,values,0).reshape(-1,128).astype(np.float32)/4,
                                     valid.reshape(-1,128).astype(np.float32)),1)
    return result


def run(taskroot):
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    work=Path('artifacts.local/work');mz1=work/'mz1-tiny-fusion-20260910'
    cache,out=taskroot/'cache-v1',taskroot/'run-v1'
    if cache.exists() or out.exists():raise FileExistsError('Fresh MZ2 output/cache required')
    fr=read(mz1/'cache-v1/features-receipt.json');mr=read(mz1/'run-v1/receipt.json')
    assert sha(mz1/'cache-v1/features.npz')==fr['feature_sha256']==mr['feature_sha256']
    for name,key in (('initial.pt','initial_file_sha256'),('schedule.npy','schedule_sha256'),('predictions.npz','predictions_sha256')):
        assert sha(mz1/'run-v1'/name)==mr[key]
    for arm in ('FUSION','RGB_ONLY'):
        assert sha(mz1/'run-v1'/f'{arm}.pt')==mr['arms'][arm]['checkpoint_sha256']
    data=load(mz1/'cache-v1/features.npz');old=load(mz1/'run-v1/predictions.npz')
    index=work/'body-query-5000-20260909/dataset-v1/index.json'
    assert sha(index)==fr['source_index_sha256']
    rows=read(index)['frames'];assert len(rows)==5000
    np.testing.assert_array_equal([r['source_role'] for r in rows],data['role'])
    schedule=np.load(mz1/'run-v1/schedule.npy',allow_pickle=False)
    assert schedule.shape==(300,128) and np.all(data['role'][schedule]=='TRAIN_ONLY')
    initial=torch.load(mz1/'run-v1/initial.pt',map_location='cpu',weights_only=True)
    assert state_sha(initial)==mr['initial_state_sha256']
    source_hashes={name:sha(Path(__file__).with_name(name)) for name in
      ('mz2_resolution.py','mz2_coarse_observation.py','multizone64_observation.py','mz1_tiny_fusion.py','mz0_clean.py')}
    cache.mkdir(parents=True);out.mkdir(parents=True);start=time.perf_counter()
    try:
        chunks={1:[],4:[]}
        for begin in range(0,5000,16):
            frames=[]
            for row in rows[begin:begin+16]:
                path=Path(row['native_file']);assert sha(path)==row['native_sha256']
                frames.append(np.load(path,allow_pickle=False))
            native=torch.from_numpy(np.stack(frames)).cuda()
            chunks[1].append(encode(observe(native,1,'multi_surface'),1).cpu().numpy())
            chunks[4].append(encode(observe4(native),4).cpu().numpy())
            if begin%400==0:print('native',begin,flush=True)
        coarse={f'z{k}':np.concatenate(v) for k,v in chunks.items()}
        ev=data['role']=='EVAL_ONLY'
        mz0=load(work/'mz0-clean-20260910/run-v1/depth.npz')
        expected=np.concatenate((np.nan_to_num(mz0['z1_multi_surface_range'],nan=0).astype(np.float32).repeat(64,axis=1).reshape(1500,128)/4,
                                 mz0['z1_multi_surface_valid'].repeat(64,axis=1).reshape(1500,128).astype(np.float32)),1)
        np.testing.assert_array_equal(coarse['z1'][ev],expected)
        np.savez_compressed(cache/'coarse.npz',**coarse)
        write(cache/'receipt.json',dict(status='PASS',frames=5000,coarse_sha256=sha(cache/'coarse.npz'),
          source_index_sha256=sha(index),mz1_features_sha256=fr['feature_sha256'],source_hashes=source_hashes,
          eval_MZ0_singlezone_parity=1500,native_passes=1,seconds=time.perf_counter()-start))
        visual=torch.as_tensor(data['visual'],device='cuda')
        train=np.flatnonzero(data['role']=='TRAIN_ONLY');targets=torch.as_tensor(data['truth'][train],device='cuda',dtype=torch.float32)
        mapping=np.full(5000,-1);mapping[train]=np.arange(len(train))
        batches=torch.as_tensor(schedule,device='cuda');target_batches=torch.as_tensor(mapping[schedule],device='cuda')
        predictions=dict(role=data['role'],truth=data['truth'],original_alerts=data['original_alerts'].copy())
        scores={};fit_receipts={}
        def forward(model,vis,features):
            pieces=[]
            with torch.inference_mode():
                for begin in range(0,len(features),256):pieces.append(model(vis[begin:begin+256],features[begin:begin+256]).cpu().numpy())
            return np.concatenate(pieces)
        for key in ('z1','z4'):
            model=TinyFusion('FUSION');model.load_state_dict(initial);assert state_sha(model.state_dict())==mr['initial_state_sha256']
            model=model.cuda().train();tof=torch.as_tensor(coarse[key],device='cuda')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);losses=[]
            for step in range(300):
                ids=batches[step];optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(visual[ids],tof[ids]),targets[target_batches[step]])
                assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss.detach()))
            model.eval();logits=forward(model,visual,tof);flags=(torch.from_numpy(logits).sigmoid()>=.5).numpy()
            predictions[key+'_logits']=logits;predictions[key+'_flags']=flags
            scores[key]={role:metric(flags[data['role']==role],data['truth'][data['role']==role]) for role in ('TRAIN_ONLY','DEV_ONLY','EVAL_ONLY')}
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/(key+'.pt'))
            fit_receipts[key]=dict(steps=300,initial_state_sha256=mr['initial_state_sha256'],schedule_sha256=mr['schedule_sha256'],
                                   checkpoint_sha256=sha(out/(key+'.pt')),loss_by_step=losses)
            print('fit',key,scores[key]['EVAL_ONLY']['spatial_exact'],flush=True)
            del model,optimizer,tof
        corruptions=stress_features(data['tof'][ev]);np.savez_compressed(out/'stress-features.npz',**corruptions)
        model=TinyFusion('FUSION').cuda().eval()
        model.load_state_dict(torch.load(mz1/'run-v1/FUSION.pt',map_location='cpu',weights_only=True))
        stress={}
        for name,values in corruptions.items():
            logits=forward(model,visual[ev],torch.as_tensor(values,device='cuda'))
            flags=(torch.from_numpy(logits).sigmoid()>=.5).numpy()
            predictions[name+'_logits']=logits;predictions[name+'_flags']=flags
            stress[name]=metric(flags,data['truth'][ev]);print('stress',name,stress[name]['spatial_exact'],flush=True)
        assert np.array_equal(predictions['original_alerts'],data['original_alerts'])
        np.savez_compressed(out/'predictions.npz',**predictions)
        baselines={arm:{role:metric(old[arm+'_flags'][data['role']==role],data['truth'][data['role']==role])
                       for role in ('TRAIN_ONLY','DEV_ONLY','EVAL_ONLY')} for arm in ('FUSION','RGB_ONLY')}
        write(out/'result.json',dict(status='PASS',resolution_scores=scores,stress_scores=stress,completed_MZ1_baselines=baselines,
          original_alert_parity=5000,scope='Consumed controlled Development; matched resolution retraining and frozen corruption stress are distinct comparisons; no promotion or hardware claim'))
        torch.cuda.synchronize()
        write(out/'receipt.json',dict(status='PASS',source_hashes=source_hashes,mz1_receipt_sha256=sha(mz1/'run-v1/receipt.json'),
          features_receipt_sha256=sha(mz1/'cache-v1/features-receipt.json'),cache_receipt_sha256=sha(cache/'receipt.json'),
          initial_sha256=mr['initial_file_sha256'],schedule_sha256=mr['schedule_sha256'],fit_receipts=fit_receipts,
          frozen_fusion_sha256=mr['arms']['FUSION']['checkpoint_sha256'],predictions_sha256=sha(out/'predictions.npz'),
          stress_features_sha256=sha(out/'stress-features.npz'),result_sha256=sha(out/'result.json'),
          seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),new_fit_steps=600,
          original_alert_parity=5000,eval_stress_frames_per_arm=1500))
    except Exception as exc:
        write(out/'failure.json',dict(status='FAILED',error=repr(exc),source_hashes=source_hashes,seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--taskroot',type=Path,required=True)
    run(p.parse_args().taskroot)
