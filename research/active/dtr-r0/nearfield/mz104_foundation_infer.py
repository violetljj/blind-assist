"""RGB-only fixed official TAO FoundationStereo small dynamic v2 ONNX producer.

This is the official adapted TAO checkpoint, not original NVlabs 11-33-40.
The ONNX graph fixes refinement iterations and FP32 inputs; no training or GT.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

MODEL_SHA='a001a7bc0512a0bc3b3218194e924784e58b20656c6f1ea2c151024e555cfd64'
MODEL_URL='https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationstereo/versions/deployable_foundation_stereo_s_dynamic_v2.0/files/deployable_foundation_stereo_s_dynamic_v2.0.onnx'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    for key in ('inputs','model','runtime-deps','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--limit',type=int,default=None,help='Engineering-only first N frames; omit for full replay')
    args=p.parse_args();root=Path(__file__).resolve().parents[4];art=(root/'artifacts.local').resolve()
    out=args.output.resolve()
    if not out.is_relative_to(art) or out.exists():raise ValueError('Fresh canonical output required')
    if sha(args.model)!=MODEL_SHA:raise ValueError('Official model digest mismatch')
    inputs=json.loads(args.inputs.read_text());rig=inputs['rig'];rows=inputs['frames']
    if args.limit is not None:
        if args.limit<1:raise ValueError('Positive canary limit required')
        rows=rows[:args.limit]
    elif len(rows)!=576:raise ValueError('Frozen full replay requires 576 RGB pairs')
    if rig['width']!=640 or rig['height']!=360 or rig['baseline_m']!=.1:raise ValueError('Frozen rig mismatch')
    required={'panel','id','left','right','left_sha256','right_sha256'}
    for r in rows:
        if set(r)!=required:raise ValueError('RGB-only row contract violated')
        for k in ('panel','id'):
            if not r[k] or any(x in r[k] for x in ('/','\\','..')):raise ValueError('Unsafe output identity')
        for k in ('left','right'):
            if Path(r[k]).suffix.lower()!='.png' or sha(r[k])!=r[k+'_sha256']:raise ValueError('RGB input integrity mismatch')
    out.mkdir(parents=True)
    sys.path.insert(0,str(args.runtime_deps.resolve()))
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import onnxruntime as ort
    # Resolve already-installed CUDA runtime DLLs without mutating system PATH.
    dll_handles=[];site=Path(torch.__file__).parent.parent
    for d in (Path(torch.__file__).parent/'lib',site/'nvidia/cublas/bin',site/'nvidia/cuda_runtime/bin'):
        if d.is_dir():dll_handles.append(os.add_dll_directory(str(d)))
    if hasattr(ort,'preload_dlls'):ort.preload_dlls()
    if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA ORT unavailable; no CPU-only fallback')
    before=torch.cuda.mem_get_info();memory=[before[1]-before[0]];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            free,total=torch.cuda.mem_get_info();memory.append(total-free)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(out/'ort-profile')
    options.intra_op_num_threads=4;options.inter_op_num_threads=1
    provider_options={'cudnn_conv_use_max_workspace':'0','gpu_mem_limit':str(6*1024**3),'arena_extend_strategy':'kSameAsRequested'}
    started=time.perf_counter();hashes={};times=[];pair_times=[];profile_summary=None;numeric_checks=[]
    receipt=dict(status='RUNNING',input_manifest_sha256=sha(args.inputs),model_sha256=MODEL_SHA,producer_sha256=sha(__file__),frames=0)
    try:
        session=ort.InferenceSession(str(args.model),sess_options=options,providers=[('CUDAExecutionProvider',provider_options),'CPUExecutionProvider'])
        if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA provider initialization failed')
        load_seconds=time.perf_counter()-started
        means=np.array([.485,.456,.406],np.float32)[None,:,None,None];std=np.array([.229,.224,.225],np.float32)[None,:,None,None]
        fx=rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))
        for i,row in enumerate(rows):
            pair_begin=time.perf_counter();tensors={}
            for key in ('left','right'):
                rgb=np.array(Image.open(row[key]).convert('RGB'),dtype=np.float32)
                if rgb.shape!=(360,640,3):raise ValueError('Unexpected RGB dimensions')
                rgb=cv2.resize(rgb,(512,288),interpolation=cv2.INTER_AREA)
                x=(rgb.transpose(2,0,1)[None]/255.-means)/std
                tensors[key+'_image']=x
            begin=time.perf_counter();disparity=session.run(['disparity'],tensors)[0];times.append(time.perf_counter()-begin)
            if disparity.shape!=(1,1,288,512):raise ValueError('Unexpected dynamic disparity output shape '+str(disparity.shape))
            disp=cv2.resize(disparity[0,0].astype(np.float32),(640,360),interpolation=cv2.INTER_LINEAR)/.8
            positive=np.isfinite(disp)&(disp>0)
            if not positive.any():raise RuntimeError('No finite positive disparity values')
            numeric_checks.append(dict(finite_positive_disparity_count=int(positive.sum()),disparity_quantiles=np.quantile(disp[positive],[.05,.5,.95]).tolist()))
            depth=np.full((360,640),np.nan,dtype=np.float32)
            valid=np.isfinite(disp)&(disp>0)
            depth[valid]=fx*rig['baseline_m']/disp[valid]
            # Official demo masks observations projecting left of the right image.
            xx=np.arange(640)[None,:]
            valid &= (xx-disp>=0)&(depth>=.5)&(depth<=4.)
            depth[~valid]=np.nan
            numeric_checks[-1]['valid_depth_count']=int(valid.sum())
            path=out/row['panel']/'depth'/(row['id']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,depth,allow_pickle=False);hashes[path.relative_to(out).as_posix()]=sha(path)
            pair_times.append(time.perf_counter()-pair_begin)
            receipt['frames']=i+1
            if i==0:
                profile=Path(session.end_profiling());events=json.loads(profile.read_text())
                counts=Counter((e.get('args',{}).get('provider'),e.get('args',{}).get('op_name')) for e in events if e.get('cat')=='Node')
                cuda_conv=sum(n for (pr,op),n in counts.items() if pr=='CUDAExecutionProvider' and op and 'Conv' in op)
                if cuda_conv==0:raise RuntimeError('No actual CUDA convolution evidence in ORT profile')
                profile_summary=dict(cuda_convolution_nodes=cuda_conv,op_provider_counts=[dict(provider=pr,op=op,count=n) for (pr,op),n in counts.items()],profile_path=profile.name)
                hashes[profile.name]=sha(profile)
            write(out/'progress.json',dict(frames=i+1,total=len(rows),last_inference_seconds=times[-1]))
        manifest=dict(model='NVIDIA TAO FoundationStereo small dynamic v2.0',model_url=MODEL_URL,model_sha256=MODEL_SHA,
            license='NVIDIA Open Model License',input_manifest_sha256=sha(args.inputs),rig=rig,
            preprocess='RGB INTER_AREA uniform0.8 resize640x360 to512x288; /255 ImageNet mean .485,.456,.406 std .229,.224,.225; no padding',
            output='INTER_LINEAR disparity upsample to640x360 then divide0.8; axial metres fx*.1/disparity; finite positive disparity; 0.5..4m; right-image left-boundary visibility; no LR cross-inference or spatial support filter',
            iterations='BAKED_IN_OFFICIAL_ONNX_NOT_EXTERNALLY_SET',precision='OFFICIAL_ONNX_FLOAT32_INPUT_AND_GRAPH',
            providers=session.get_providers(),provider_options=session.get_provider_options(),onnxruntime_version=ort.__version__,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
            cuda_profile=profile_summary,load_seconds=load_seconds,inference_seconds=times,pair_total_seconds=pair_times,opencv_version=cv2.__version__,numeric_checks=numeric_checks,
            memory_scope='DEVICE_TOTAL_USED_SAMPLED_AT100MS_NOT_PROCESS_EXCLUSIVE',device_used_before_bytes=memory[0],device_used_peak_bytes=max(memory),
            frames=[dict(panel=r['panel'],id=r['id'],left_sha256=r['left_sha256'],right_sha256=r['right_sha256']) for r in rows])
        write(out/'manifest.json',manifest);hashes['manifest.json']=sha(out/'manifest.json')
        receipt.update(status='PASS',hashes=hashes,wall_seconds=time.perf_counter()-started)
    except Exception as error:
        receipt.update(status='FAIL',error=repr(error),hashes=hashes,wall_seconds=time.perf_counter()-started)
        raise
    finally:
        stop.set();monitor_thread.join();write(out/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
