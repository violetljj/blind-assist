"""Frozen dense RGB map cache plus evaluator-only sampled depth supervision."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, math, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN


def sampling_grid():
    fraction=(np.arange(7)+.5)/7
    vv,uu=np.meshgrid(fraction,fraction,indexing='ij')
    az=np.array([-22.5+(z%8+uu)*5.625 for z in range(64)])
    el=np.array([22.5-(z//8+vv)*5.625 for z in range(64)])
    rays=np.stack([np.ones_like(az),np.tan(np.deg2rad(az)),np.tan(np.deg2rad(el))],-1).reshape(64,49,3)
    grid=np.stack([rays[...,1]/math.tan(math.radians(50)), -rays[...,2]/math.tan(math.radians(50))/(360/640)],-1)
    return grid,rays


def main(root,output,reuse_dense=None):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    work=root/'artifacts.local/work';old=work/'mz1-tiny-fusion-20260910/cache-v1'
    old_receipt=read(old/'features-receipt.json');assert sha(old/'features.npz')==old_receipt['feature_sha256']
    dataset=work/'body-query-5000-20260909/dataset-v1/index.json'
    assert sha(dataset)==old_receipt['source_index_sha256']
    old_data=load_npz(old/'features.npz'); rows=read(dataset)['frames']
    ids=np.flatnonzero(old_data['role']!='EVAL_ONLY');assert len(ids)==3500
    capture=work/'mz6-short-sequence-20260910/capture-v3'; seq=work/'mz6-short-sequence-20260910/evaluation-v1'
    receipt=read(seq/'prepare-receipt.json')
    input_hashes={str(Path(k).resolve()):v for k,v in receipt['input_sha256'].items()}
    for name in ['observations','evaluator']:assert sha(seq/f'{name}.npz')==receipt[f'{name}_sha256']
    assert read(capture/'source-admission.json')['status']=='PASS'
    sd=load_npz(seq/'observations.npz');ev=load_npz(seq/'evaluator.npz')
    records=[dict(rgb=r['rgb_file'],native=r['native_file'],rgb_sha=r['rgb_sha256'],native_sha=r['native_sha256']) for r in (rows[i] for i in ids)]
    for i in range(200):
        rgb=capture/f'model/sample/{i:04d}.png';native=capture/f'evaluator/native/{i:04d}.npy'
        records.append(dict(rgb=str(rgb),native=str(native),rgb_sha=input_hashes[str(rgb.resolve())],native_sha=input_hashes[str(native.resolve())]))
    expected=np.concatenate([old_data['visual'][ids],sd['visual']])
    ranges=np.concatenate([old_data['tof'][ids,:128].reshape(-1,64,2)*4,sd['ranges']]).astype(np.float32)
    valid=np.concatenate([old_data['tof'][ids,128:].reshape(-1,64,2)>0,sd['valid']])
    truth=np.concatenate([old_data['truth'][ids],ev['truth']]);roles=np.r_[old_data['role'][ids],np.full(200,'MZ6_TRAIN')]
    base=work/'body-query-10000-b-20260909/run-v1';decoder=work/'body-query-context-decoder-20260909/run-v1'
    for name,digest in FROZEN.items():assert sha((base if name=='NEW-step2000.pt' else decoder)/name)==digest
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    model=ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval();m=model.base
    maps=np.lib.format.open_memmap(output/'dense.npy',mode='w+',dtype=np.float32,shape=(3700,64,18,32))
    grid,rays=sampling_grid();px=np.rint((grid[...,0]+1)*320-.5).astype(int);py=np.rint((grid[...,1]+1)*180-.5).astype(int)
    f=320/math.tan(math.radians(50));radial_factor=np.sqrt(1+((px-319.5)/f)**2+((py-179.5)/f)**2)
    sampled_depth=[];maxerr=0
    reused=np.load(reuse_dense,mmap_mode='r') if reuse_dense else None
    cuts=np.r_[0,np.flatnonzero(ids[1:]//16!=ids[:-1]//16)+1,3500]
    batches=list(zip(cuts[:-1],cuts[1:]))+[(i,min(i+16,3700)) for i in range(3500,3700,16)]
    for begin,end in batches:
        images=[]
        for row in records[begin:end]:
            rgb,native=Path(row['rgb']),Path(row['native']);assert sha(rgb)==row['rgb_sha'] and sha(native)==row['native_sha']
            with Image.open(rgb) as im:images.append(np.array(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
            depth=np.load(native,allow_pickle=False)[py,px]
            sampled_depth.append(np.where(np.isfinite(depth)&(depth>0)&(depth<100),depth*radial_factor,np.nan))
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode():
            if reused is not None and end<=3488:
                dense=torch.from_numpy(np.array(reused[begin:end])).cuda()
            else:
                # Old cache used16 even at our TRAIN/DEV cut. Pad with repeated
                # old RGB, never access the following EVAL rows for this purpose.
                original_batch_size=len(x) if begin>=3500 else min(16,5000-(int(ids[begin])//16)*16)
                infer_x=x if len(x)==original_batch_size else torch.cat([x,x[-1:].expand(original_batch_size-len(x),-1,-1,-1)])
                deep,shallow=m.extract((infer_x-m.image_mean)/m.image_std)
                deep=F.interpolate(m.deep_projection(deep),size=(18,32),mode='bilinear',align_corners=False)
                dense=torch.cat([deep,m.detail(shallow)],1)[:len(x)]
            sampled=(dense.flatten(2)@m.query_projection.T).transpose(1,2).reshape(len(x),12,27,64)
            mask=m.query_valid[None,:,:,None]
            pooled=(sampled*mask).sum(2)/mask.sum(2).clamp_min(1)
            normalized=((pooled-model.feature_mean)/model.feature_std).flatten(1)
            err=float((normalized-torch.from_numpy(expected[begin:begin+len(x),:768]).cuda()).abs().max())
            maxerr=max(err,maxerr);assert err<1e-4, (begin,end,err)
            maps[begin:begin+len(x)]=dense.cpu().numpy()
        if begin%400==0:print('CACHE',begin+len(x),'/3700',flush=True)
    maps.flush();del maps
    np.savez_compressed(output/'observations.npz',ranges=ranges,valid=valid,role=roles,old_index=ids,
        visual=expected,stress_ranges=sd['stress_ranges'],stress_valid=sd['stress_valid'],clip=sd['clip'])
    np.savez_compressed(output/'evaluator.npz',truth=truth,sampled_radial=np.stack(sampled_depth).astype(np.float32))
    np.savez(output/'grid.npz',grid=grid,rays=rays)
    write(output/'receipt.json',dict(status='PASS',frames=3700,old_train=2500,old_dev=1000,mz6_consumed_training=200,
        source_sha256=sha(Path(__file__)),frozen_hashes=FROZEN,pooled_parity_max_error=maxerr,
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        reused_prefix_frames=3488 if reused is not None else 0,
        reused_prefix_source_sha256=sha(reuse_dense) if reused is not None else None,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},inputs=records))
    print('PASS',maxerr,time.perf_counter()-started,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-dense',type=Path)
    a=p.parse_args();main(a.root,a.output,a.reuse_dense)
