"""Label-blind five-seed frozen M3 inference on the fresh pilot cache."""
import gc
from pathlib import Path
import time

import numpy as np
import torch

import cnh_temporal_readout as C
import cnh_temporal_readout_train as TR
import cnh_temporal_readout_model as M
import cnh_margin_confirm as MC


def infer():
    C.check_budget();TR.configure('cuda');started=time.monotonic()
    target=C.OUT/'predictions/M3_fresh_evaluation.npz'
    if target.exists():raise FileExistsError('Frozen M3 output already exists')
    data=TR.Inputs(C.OUT,'fresh_evaluation','V',training=False)
    before=data.hashes();cache=data.enable_gpu_cache('cuda');paths=MC.model_paths('M3')
    models={str(path):C.sha(path) for path in paths}
    source={str(Path(module.__file__)):C.sha(module.__file__) for module in (M,TR)}
    values=np.empty((data.n,2),np.float32);nets=[]
    try:
        for path in paths:
            net=M.CVR().cuda().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));nets.append(net)
        with torch.inference_mode():
            for offset in range(0,data.n,64):
                C.check_budget();ids=np.arange(offset,min(offset+64,data.n))
                batch=data.batch(ids,'cuda');x=M.prepare_voxels(batch['voxels'])
                values[ids]=torch.stack([net(x) for net in nets]).mean(0).cpu().numpy()
        if not np.isfinite(values).all() or data.hashes()!=before:
            raise ValueError('M3 input changed or nonfinite logits')
        if any(C.sha(path)!=digest for path,digest in models.items()):
            raise ValueError('Frozen M3 checkpoints changed')
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target,raw=values)
        C.save(target.with_suffix('.json'),dict(status='COMPLETE',rows=data.n,models_sha256=models,
             input_sha256=before,seconds=time.monotonic()-started,score_sha256=C.sha(target),
             source_sha256=source|{str(Path(__file__)):C.sha(__file__)},plan_sha256=C.plan_sha(),
             label_access=False,gpu_input_cache=cache,raw_ensemble='Five original M3 models mean logits; smoothing remains evaluator-only'))
        print('FROZEN M3 fresh inference',data.n,round(time.monotonic()-started,2),flush=True)
    finally:
        data.close();nets.clear();gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':infer()
