"""Frozen M3 five-model mean logits on pilot2 observation cache only."""
import gc
import time
import numpy as np
import torch
import cnh_readout_pilot2 as C
import cnh_readout_pilot2_train as TR
import cnh_temporal_readout_model as M
import cnh_margin_confirm as MC


def infer():
    C.setup();C.check_budget();TR.TR.configure('cuda');began=time.monotonic()
    target=C.OUT/'predictions/M3_fresh_evaluation.npz'
    if target.exists():raise FileExistsError('Frozen M3 scores already exist')
    data=TR.Inputs('fresh_evaluation','M3');before=data.hashes();cache=data.enable_gpu_cache('cuda')
    models=MC.model_paths('M3');hashes={str(p):C.sha(p) for p in models};nets=[]
    try:
        for path in models:
            net=M.CVR().cuda().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));nets.append(net)
        raw=np.empty((data.n,2),np.float32)
        with torch.inference_mode():
            for offset in range(0,data.n,64):
                ids=np.arange(offset,min(offset+64,data.n));batch=data.batch(ids,'cuda');x=M.prepare_voxels(batch['voxels'])
                raw[ids]=torch.stack([net(x) for net in nets]).mean(0).cpu().numpy()
        if not np.isfinite(raw).all() or data.hashes()!=before or any(C.sha(p)!=v for p,v in hashes.items()):raise ValueError('Frozen M3 integrity')
        target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,raw=raw)
        receipt=dict(status='COMPLETE',rows=data.n,seconds=time.monotonic()-began,input_sha256=before,models_sha256=hashes,score_sha256=C.sha(target),plan_sha256=C.plan_sha(),label_access=False,gpu_input_cache=cache)
        C.save(target.with_suffix('.json'),receipt);print(receipt,flush=True)
    finally:data.close();nets.clear();gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':infer()
