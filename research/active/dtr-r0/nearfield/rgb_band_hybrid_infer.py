"""Two resident official RGB depth models; PUBLIC inputs only, no evaluator truth.

Both original FP32 models stay resident, with inherited CUDA FP16 autocast
inside each official recipe. Each frame executes DAV2 then UniDepth, with
synchronized individual and combined latency. No precision fallback occurs.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
START = time.perf_counter()

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def dav_infer_factory(model, torch, cv2, np):
    def infer(rgb, K):
        x,size=model.image2tensor(cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR),518)
        with torch.autocast('cuda',dtype=torch.float16): d=model.forward(x)
        return torch.nn.functional.interpolate(d[:,None].float(),size,mode='bilinear',align_corners=True)[0,0].cpu().numpy()
    return infer

def run(a):
    root=a.runroot
    for p in [root/'dual_resident_terminal.json',root/'dav2/terminal.json',root/'unidepth/terminal.json',
              root/'dav2/predictions.json',root/'unidepth/predictions.json']:
        if p.exists(): raise FileExistsError('Preserve prior dual-resident failure/output; no implicit rerun: '+str(p))
    for name in ['dav2','unidepth']: (root/name).mkdir(exist_ok=True)
    receipt=dict(status='STARTING',pid=os.getpid(),training_calls=0,evaluator_read=False,old496_rerun=False,
                 dual_resident_requested=True,dual_resident_loaded=False,inference_order=['dav2','unidepth'],
                 budget_gpu_process_wall_s=a.budget_s,frames=0,source_sha256=sha(__file__))
    models={};infers={};meta={};rows={'dav2':[],'unidepth':[]};latencies={'dav2':[],'unidepth':[]}
    combined=[];torch=None;model=None;infer=None
    try:
        import torch as torch_module
        torch=torch_module
        import numpy as np
        import cv2
        from PIL import Image
        from rgb_body_query_residual_target import PUBLIC,public_mapping
        from rgb_body_query_3rscan import sample_prediction
        from rgb_depth_backbone_adapters import load_model
        torch.set_num_threads(4);cv2.setNumThreads(2)
        assert torch.cuda.is_available()
        roster_path=root/'new_public_roster.json';roster=load(roster_path)['rows']
        assert roster and all(r['role']=='eval' for r in roster)
        assert all(set(r)<=PUBLIC|{'role','cohort'} for r in roster),'Public roster contains extra fields'
        receipt.update(roster_sha256=sha(roster_path),total_frames=len(roster),torch=torch.__version__,
                       device=torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()
        old=a.repo/'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010'
        sys.path.insert(0,str(old/'upstream/Depth-Anything-V2/metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        weight=a.repo/'artifacts.local/models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth'
        model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=20)
        model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval().requires_grad_(False).cuda()
        models['dav2']=model;infers['dav2']=dav_infer_factory(model,torch,cv2,np)
        meta['dav2']=dict(official_recipe='Inherited DAV2 Large518/maxdepth20/FP32 weights + CUDA AMP FP16',
                         weight_path=str(weight),source_receipt=str(old/'dav2_provenance.json'))
        model,infer,info=load_model('unidepth',old/'upstream/UniDepth',a.repo/'artifacts.local/models/lpiccinelli--unidepth-v2-vitl14')
        model.cuda();models['unidepth']=model;infers['unidepth']=infer;meta['unidepth']=info
        model=infer=None
        torch.cuda.synchronize()
        parameters={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}
        receipt.update(dual_resident_loaded=True,parameters=parameters,total_parameters=sum(parameters.values()),
            resident_allocated_bytes=torch.cuda.memory_allocated(),resident_reserved_bytes=torch.cuda.memory_reserved(),
            model_metadata=meta)
        for name in models:
            save(root/name/'model_identity.json',dict(model=name,parameters=parameters[name],meta=meta[name],
                roster_sha256=receipt['roster_sha256'],dual_resident=True,pid=os.getpid()))
        for i,r in enumerate(roster):
            if time.perf_counter()-START+max(5,max(combined[-8:],default=5)*1.2)>=a.budget_s:
                raise TimeoutError('Owned GPU process-wall cap before next dual-model frame')
            assert sha(r['rgb_path'])==r['rgb_sha256']
            with Image.open(r['rgb_path']) as im:rgb=im.convert('RGB')
            assert [rgb.height,rgb.width]==r['color_shape']
            K=np.asarray(r['color_K'],np.float32)
            depths={}
            torch.cuda.synchronize();frame_start=time.perf_counter()
            for name in ['dav2','unidepth']:
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode(): depths[name]=np.asarray(infers[name](rgb,K),np.float32)
                torch.cuda.synchronize();latencies[name].append(time.perf_counter()-t)
                assert list(depths[name].shape)==r['color_shape']
            torch.cuda.synchronize();combined.append(time.perf_counter()-frame_start)
            mx,my=public_mapping(r)
            for name in ['dav2','unidepth']:
                native=sample_prediction(depths[name],mx,my)
                p=root/name/f'{i:03d}_{r["scan"]}_{r["frame"]:06d}.npz'
                np.savez(p,depth=native)
                rows[name].append(dict(r,path=str(p.resolve()),sha256=sha(p),
                    gpu_frame_latency_s=latencies[name][-1],combined_gpu_frame_latency_s=combined[-1]))
            receipt['frames']=i+1
            if (i+1)%8==0:print(json.dumps(dict(frames=i+1,total=len(roster),elapsed_s=time.perf_counter()-START)),flush=True)
        receipt['status']='COMPLETE'
    except Exception as exc:
        receipt.update(status='BUDGET_STOP' if isinstance(exc,TimeoutError) else 'FAILED',error=repr(exc),
                       traceback=traceback.format_exc())
        print(receipt['traceback'],flush=True)
    finally:
        if torch is not None and torch.cuda.is_available():
            receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        # Inference closures themselves retain models, so release them first.
        infers.clear();models.clear();model=infer=None;gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache();torch.cuda.synchronize()
            receipt.update(released_allocated_bytes=torch.cuda.memory_allocated(),
                           released_reserved_bytes=torch.cuda.memory_reserved())
        receipt.update(gpu_process_wall_s=time.perf_counter()-START,resources_released=True,
                       combined_latencies_s=combined,model_latencies_s=latencies)
        for name in ['dav2','unidepth']:
            item=dict(receipt,model=name,frames=len(rows[name]),parameters=receipt.get('parameters',{}).get(name),
                      latencies_s=latencies[name],meta=meta.get(name))
            save(root/name/'predictions.json',dict(status=receipt['status'],rows=rows[name],evaluator_read=False,
                 roster_sha256=receipt.get('roster_sha256')))
            save(root/name/'terminal.json',item)
        save(root/'dual_resident_terminal.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['traceback','combined_latencies_s','model_latencies_s','model_metadata']}),flush=True)
    return receipt['status']=='COMPLETE'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--runroot',type=Path,required=True);p.add_argument('--budget-s',type=float,default=180)
    sys.exit(0 if run(p.parse_args()) else 1)
