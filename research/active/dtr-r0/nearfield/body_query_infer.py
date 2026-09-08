"""Independent single-RGB inference. No evaluator, pose, depth or history API."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from body_query_model import BodyQueryModel,CALIBRATION


def infer(image,checkpoint,pretrained,selection,arm,device):
    # Match the evaluator's convolution policy for reproducible saved-map display.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(1)
    chosen=json.loads(Path(selection).read_text(encoding='utf-8-sig'))[arm]
    from body_query_data import sha
    if chosen.get('checkpoint_sha256') and sha(checkpoint)!=chosen['checkpoint_sha256']:
        raise ValueError('Checkpoint and DEV selection do not match')
    with Image.open(image) as im:
        if im.width*360!=im.height*640:
            raise ValueError('Expected original fixed-camera 16:9 image; arbitrary crops are not calibrated')
        pixels=np.asarray(im.convert('RGB').resize((256,144),Image.Resampling.BOX)).copy()
    model=BodyQueryModel(pretrained,arm).to(device)
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device));model.eval()
    # Keep the evaluator's NHWC-to-NCHW stride layout, including the batch axis.
    rgb=torch.from_numpy(pixels[None].copy()).to(device).permute(0,3,1,2).float()/255.
    with torch.inference_mode():n,s,c=model(rgb)
    scores=n.sigmoid()[0].cpu().numpy();thresholds=np.array(chosen['thresholds'],dtype=float)
    clipped=np.clip(scores.astype(float),1e-7,1-1e-7);t=np.clip(thresholds,1e-7,1-1e-7)
    unknown=np.abs(np.log(clipped/(1-clipped))-np.log(t/(1-t)))<.5
    result=dict(arm=arm,image=str(Path(image).resolve()),calibration=CALIBRATION,
        calibration_notice='Caller must supply fixed camera height1.70m, pitch0, HFOV100 image; dimensions alone do not verify calibration',
        heads={name:dict(score=float(scores[h]),threshold=float(thresholds[h]),
            binary_positive=bool(scores[h]>=thresholds[h]),state='UNKNOWN' if unknown[h] else 'EVIDENCE' if scores[h]>=thresholds[h] else 'NO_DETECTED_EVIDENCE') for h,name in enumerate(('BODY','HEAD'))},
        support=s.sigmoid()[0].cpu().tolist(),query_count_probabilities=c.softmax(-1)[0].cpu().tolist(),
        unknown_semantics='Heuristic half-logit decision margin; not physical visibility estimation',
        scope='Single RGB, synthetic Development model; no temporal state or certified free-space output',
        device=str(device))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('image','checkpoint','pretrained','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();result=infer(a.image,a.checkpoint,a.pretrained,a.selection,a.arm,a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('support','query_count_probabilities')}))


if __name__=='__main__':main()
