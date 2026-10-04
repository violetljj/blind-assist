"""Frozen five-seed M3 inference for declared yaw-return experiments; no training."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-coverage-policy-20261004'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def check_deadline():
    if time.time() >= read(OUT/'budget.json')['deadline_unix_s']:
        raise TimeoutError('Authorized one-hour coverage budget reached')

def run(limit=None):
    import torch
    import cnh_displacement_ceiling as D
    import cnh_margin_confirm as MC
    import cnh_cvr_pilot as CP
    from cnh_temporal_readout_data import normalized_z
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    import cnh_structure_space as SS
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=read(OUT/'PLAN.json'); units=p['units'][:limit] if limit else p['units']
    paths=MC.model_paths('M3'); nets=[]; projector=None
    started=time.monotonic(); projection_s=network_s=0.
    outputs=[]; identities={}; prefix_max=0.
    try:
        for path in paths:
            net=CP.CVR().cuda().eval()
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            nets.append(net)
        projector=BatchedProjector(); masks=torch.as_tensor(query_masks(), device='cuda')
        bias=torch.as_tensor(np.load(D.BIAS), dtype=torch.float32, device='cuda')
        for unit in units:
            check_deadline(); obs=OUT/'observations'/f'unit{unit}.npz'
            with np.load(obs, allow_pickle=False) as d:
                hist=d['hist'].copy(); ambient=d['ambient'].copy()
                noisy=d['noisy'].copy(); query=d['query'].copy(); rc=d['rc_m'].copy()
                z=torch.as_tensor(normalized_z(hist, ambient[:,None,None]), dtype=torch.float32, device='cuda')
            raw=np.empty((len(rc),7,4,13,2), np.float32)
            for arm in range(len(rc)):
                for k in range(4):
                    check_deadline(); vox=[]; torch.cuda.synchronize(); tick=time.monotonic()
                    for frame in range(3,16):
                        matrices=query[arm,frame]@np.linalg.inv(noisy[arm,k,frame])@noisy[arm,k,max(0,frame-7):frame+1]
                        vox.append(D.project_many(projector, z[arm,:,k,max(0,frame-7):frame+1], matrices).half())
                    torch.cuda.synchronize(); projection_s+=time.monotonic()-tick
                    x=torch.stack(vox,1).reshape(7*13,3,24,17,33)
                    torch.cuda.synchronize(); tick=time.monotonic(); logits=[]
                    with torch.inference_mode():
                        for begin in range(0,len(x),64):
                            xb=x[begin:begin+64].float().clone()
                            xb[:,0]=xb[:,0].sign()*xb[:,0].abs().log1p()
                            xb[:,2]=xb[:,2].sign()*xb[:,2].abs().log1p(); xb[:,1]/=8
                            xb=torch.cat((xb,masks[None].expand(len(xb),-1,-1,-1,-1)),1)
                            logits.append(torch.stack([n(xb) for n in nets]).mean(0).cpu().numpy())
                        raw[arm,:,k]=np.concatenate(logits).reshape(7,13,2)
                    torch.cuda.synchronize(); network_s+=time.monotonic()-tick
            if not np.isfinite(raw).all(): raise ValueError('Nonfinite M3 score')
            path=OUT/'scores'/f'unit{unit}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,raw=raw,rc_m=rc,frames=np.arange(3,16),unit=unit)
            identities[str(path.relative_to(OUT))]=SS.sha(path)
            save(path.with_suffix('.json'),dict(status='COMPLETE',observation_sha256=SS.sha(obs),score_sha256=SS.sha(path),plan_sha256=SS.sha(OUT/'PLAN.json')))
            outputs.append(raw)
            print('M3',unit,'seconds',round(time.monotonic()-started,2),flush=True)
        path=OUT/'predictions/return_scores.npz'; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,units=units,rc=rc,frames=np.arange(3,16),raw_logits=np.stack(outputs))
        result=dict(status='COMPLETE' if not limit else 'ENGINEERING_SUBSET', units=units,elapsed_s=time.monotonic()-started,
            projection_s=projection_s,network_s=network_s,models_sha256={str(v):SS.sha(v) for v in paths},
            bias_sha256=SS.sha(D.BIAS),source_sha256=SS.sha(__file__),plan_sha256=SS.sha(OUT/'PLAN.json'),outputs_sha256=identities,
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),tf32=False),
            input_contract='signed histogram, calibrated ambient and bias, estimated relative poses, declared current head-to-query rotation; no evaluator target/truth input')
        save(OUT/'inference_runtime.json',result)
        return result
    finally:
        nets.clear(); net=projector=bias=masks=hist=z=vox=x=xb=None
        gc.collect(); torch.cuda.empty_cache()
        save(OUT/'inference_release.json',dict(elapsed_s=time.monotonic()-started,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.limit)
