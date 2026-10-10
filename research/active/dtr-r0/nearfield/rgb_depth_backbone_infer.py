"""Public-only official metric backbone inference with cumulative process-wall cap."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
START=time.perf_counter()

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(repo,root):
    from rgb_body_query_residual_target import PUBLIC
    work=repo/'artifacts.local/work';source=work/'rgb-body-query-bounded-residual-dev-20261010/public_roster.json'
    rows=[dict({k:r[k] for k in PUBLIC},cohort=r['cohort'],role=r['role']) for r in load(source)['rows']]
    refs=work/'rgb-body-query-query-level-dev-20261009'
    obs=[(n,refs/'fixed-grid-sensor'/n/'observations.json') for n in ('original_validation24','new_3rscan64','arkit16')]
    obs += [('arkit_'+n,refs/'additional-arkit-sensor'/n/'observations.json') for n in ('40777060','40777065')]
    for cohort,path in obs:
        rows += [dict({k:r[k] for k in PUBLIC},cohort=cohort,role='eval') for r in load(path)['rows']]
    assert len(rows)==len({(r['scan'],r['frame']) for r in rows})==496
    for r in rows:assert sha(r['rgb_path'])==r['rgb_sha256']
    save(root/'public_roster.json',{'rows':rows,'evaluator_read':False,'source_sha256':sha(source),'frames':496})
    print('PUBLIC_ROSTER496_FROZEN',flush=True)

def run(repo,root,name,source,weight,budget):
    out=root/name;out.mkdir(exist_ok=True)
    terminal=out/'terminal.json'
    if terminal.exists():raise FileExistsError('No implicit terminal reset')
    receipt={'model':name,'status':'STARTING','pid':os.getpid(),'budget_gpu_process_wall_s':budget,'evaluator_read':False,'training_calls':0,'frames':0}
    model=None;infer=None;torch=None;latencies=[];rows=[]
    try:
        import torch as torch_module
        torch=torch_module
        import numpy as np
        import cv2
        from PIL import Image
        from rgb_body_query_residual_target import public_mapping
        from rgb_body_query_3rscan import sample_prediction
        torch.set_num_threads(4);cv2.setNumThreads(2)
        assert torch.cuda.is_available()
        roster=load(root/'public_roster.json')['rows']
        receipt.update(device=torch.cuda.get_device_name(0),torch=torch.__version__,source_sha256=sha(__file__),roster_sha256=sha(root/'public_roster.json'))
        if name=='dav2':
            sys.path.insert(0,str(source/'metric_depth'))
            from depth_anything_v2.dpt import DepthAnythingV2
            model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=20)
            model.load_state_dict(torch.load(weight/'depth_anything_v2_metric_hypersim_vitl.pth',map_location='cpu',weights_only=True));model.eval().cuda()
            def infer(rgb,K):
                x,size=model.image2tensor(cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR),518)
                with torch.autocast('cuda',dtype=torch.float16):d=model.forward(x)
                return torch.nn.functional.interpolate(d[:,None].float(),size,mode='bilinear',align_corners=True)[0,0].cpu().numpy()
            meta={'backbone':'DepthAnythingV2 Metric Hypersim Large','input_size':518,'max_depth':20,'supplied_K':False,'precision':'FP32 weights, autocastFP16 inference; output interpolationFP32'}
        else:
            from rgb_depth_backbone_adapters import load_model
            adapter=Path(__file__).with_name('rgb_depth_backbone_adapters.py')
            shutil.copyfile(adapter,out/'adapters_executed.py')
            receipt['adapter_sha256']=sha(adapter)
            model,infer,meta=load_model(name,source,weight)
            model.cuda()
        receipt.update(meta=meta,parameters=sum(p.numel() for p in model.parameters()))
        save(out/'model_identity.json',receipt)
        shutil.copyfile(__file__,out/'runner_executed.py')
        for i,r in enumerate(roster):
            elapsed=time.perf_counter()-START
            reserve=max(5,max(latencies[-10:],default=5)*1.2)
            if elapsed+reserve>=budget:raise TimeoutError('GPU process-wall cap before next frame')
            assert sha(r['rgb_path'])==r['rgb_sha256']
            with Image.open(r['rgb_path']) as im:rgb=im.convert('RGB')
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():depth=np.asarray(infer(rgb,np.asarray(r['color_K'],np.float32)),np.float32)
            torch.cuda.synchronize();latencies.append(time.perf_counter()-t)
            assert list(depth.shape)==r['color_shape']
            mx,my=public_mapping(r);native=sample_prediction(depth,mx,my)
            path=out/f'{i:03d}_{r["scan"]}_{r["frame"]:06d}.npz'
            np.savez(path,depth=native)
            rows.append(dict(r,path=str(path.resolve()),sha256=sha(path),gpu_frame_latency_s=latencies[-1],finite_positive_pixels=int((np.isfinite(native)&(native>0)).sum())))
            receipt['frames']=i+1
            if i%25==0:
                save(out/'progress.json',dict(receipt,elapsed_s=time.perf_counter()-START,total=496));print(json.dumps({'model':name,'frames':i+1,'elapsed_s':round(time.perf_counter()-START,2),'latency_s':round(latencies[-1],3)}),flush=True)
        receipt['status']='COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED' if not isinstance(exc,TimeoutError) else 'BUDGET_STOP',error=repr(exc),traceback=traceback.format_exc())
        print(receipt['traceback'],flush=True)
    finally:
        if torch is not None:
            model=None;infer=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        receipt.update(gpu_process_wall_s=time.perf_counter()-START,resources_released=True,latencies_s=latencies)
        save(out/'predictions.json',dict(status=receipt['status'],rows=rows,evaluator_read=False,roster_sha256=sha(root/'public_roster.json')))
        save(terminal,receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('latencies_s','traceback')}),flush=True)
    return receipt['status']=='COMPLETE'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['prepare','run'],required=True);p.add_argument('--model');p.add_argument('--source',type=Path);p.add_argument('--weight',type=Path);p.add_argument('--budget-s',type=float,default=700)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.repo,a.root)
    else:sys.exit(0 if run(a.repo,a.root,a.model,a.source,a.weight,a.budget_s) else 1)
