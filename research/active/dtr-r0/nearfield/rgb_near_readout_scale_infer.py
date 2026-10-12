"""New eval-only public RGB inference; existing496 caches are never opened or rerun."""
from __future__ import annotations
import argparse, dataclasses, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
START = time.perf_counter()

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def run(a):
    out = a.runroot / a.model
    out.mkdir(exist_ok=True)
    if (out/'terminal.json').exists() or (out/'predictions.json').exists():
        raise FileExistsError('No implicit rerun or overwrite')
    receipt = dict(model=a.model,pid=os.getpid(),status='STARTING',training_calls=0,
                   evaluator_read=False,old496_rerun=False,budget_gpu_process_wall_s=a.budget_s,frames=0)
    model = infer = torch = None
    rows = []; latencies = []
    try:
        import torch as torch_module
        torch = torch_module
        import numpy as np
        import cv2
        from PIL import Image
        from rgb_body_query_residual_target import PUBLIC, public_mapping
        from rgb_body_query_3rscan import sample_prediction
        torch.set_num_threads(4); cv2.setNumThreads(2)
        assert torch.cuda.is_available()
        roster = load(a.runroot/'new_public_roster.json')['rows']
        assert roster and all(r['role']=='eval' for r in roster)
        assert all(set(r) <= PUBLIC | {'role','cohort'} for r in roster), 'Public roster contains extra fields'
        receipt.update(roster_sha256=sha(a.runroot/'new_public_roster.json'),source_sha256=sha(__file__),
                       total_frames=len(roster),torch=torch.__version__,device=torch.cuda.get_device_name(0))
        old = a.repo/'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010'
        if a.model=='dav2':
            sys.path.insert(0,str(old/'upstream/Depth-Anything-V2/metric_depth'))
            from depth_anything_v2.dpt import DepthAnythingV2
            weight = a.repo/'artifacts.local/models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth'
            model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=20)
            model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)); model.eval().cuda()
            def infer(rgb,K):
                x,size=model.image2tensor(cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR),518)
                with torch.autocast('cuda',dtype=torch.float16): d=model.forward(x)
                return torch.nn.functional.interpolate(d[:,None].float(),size,mode='bilinear',align_corners=True)[0,0].cpu().numpy()
            receipt['official_recipe']='Inherited DAV2 Large518/maxdepth20/AMP FP16'
        elif a.model=='depthpro':
            base=a.repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
            weight=base/'depth_pro.pt'
            assert sha(weight)=='3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
            sys.path.insert(0,str(base/'upstream/src'))
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            cfg=dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(weight))
            model,transform=depth_pro.create_model_and_transforms(cfg,device=torch.device('cuda'),precision=torch.float16)
            model.eval().requires_grad_(False)
            def infer(rgb,K):
                pred=model.infer(transform(rgb),f_px=torch.tensor(K[0][0],device='cuda'))
                return pred['depth'].float().cpu().numpy()
            receipt['official_recipe']='Inherited raw RGB FP16 supplied fx; no centered remap'
        else:
            from rgb_depth_backbone_adapters import load_model
            src,wd = {'unidepth':('UniDepth','lpiccinelli--unidepth-v2-vitl14'),
                      'metric3d':('Metric3D','JUGGHM--Metric3D')}[a.model]
            model,infer,meta=load_model(a.model,old/'upstream'/src,a.repo/'artifacts.local/models'/wd)
            model.cuda(); receipt['meta']=meta
        receipt['parameters']=sum(p.numel() for p in model.parameters())
        save(out/'model_identity.json',receipt)
        for i,r in enumerate(roster):
            if time.perf_counter()-START+max(5,max(latencies[-8:],default=5)*1.2)>=a.budget_s:
                raise TimeoutError('Owned GPU process-wall cap before next frame')
            assert sha(r['rgb_path'])==r['rgb_sha256']
            with Image.open(r['rgb_path']) as im: rgb=im.convert('RGB')
            assert [rgb.height,rgb.width]==r['color_shape']
            torch.cuda.synchronize(); t=time.perf_counter()
            with torch.inference_mode(): depth=np.asarray(infer(rgb,np.asarray(r['color_K'],np.float32)),np.float32)
            torch.cuda.synchronize(); latencies.append(time.perf_counter()-t)
            assert list(depth.shape)==r['color_shape']
            mx,my=public_mapping(r); native=sample_prediction(depth,mx,my)
            path=out/f'{i:03d}_{r["scan"]}_{r["frame"]:06d}.npz'
            np.savez(path,depth=native)
            rows.append(dict(r,path=str(path.resolve()),sha256=sha(path),gpu_frame_latency_s=latencies[-1]))
            receipt['frames']=i+1
            if (i+1)%8==0: print(json.dumps(dict(model=a.model,frames=i+1,total=len(roster),elapsed_s=time.perf_counter()-START)),flush=True)
        receipt['status']='COMPLETE'
    except Exception as e:
        receipt.update(status='BUDGET_STOP' if isinstance(e,TimeoutError) else 'FAILED',error=repr(e),traceback=traceback.format_exc())
        print(receipt['traceback'],flush=True)
    finally:
        model=infer=None; gc.collect()
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()
        receipt.update(gpu_process_wall_s=time.perf_counter()-START,resources_released=True,latencies_s=latencies)
        save(out/'predictions.json',dict(status=receipt['status'],rows=rows,evaluator_read=False,roster_sha256=receipt.get('roster_sha256')))
        save(out/'terminal.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('latencies_s','traceback')}),flush=True)
    return receipt['status']=='COMPLETE'

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--runroot',type=Path,required=True)
    p.add_argument('--model',choices=['dav2','unidepth','metric3d','depthpro'],required=True); p.add_argument('--budget-s',type=float,required=True)
    sys.exit(0 if run(p.parse_args()) else 1)
