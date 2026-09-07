"""Frozen MDE + supported raw/ground geometry adapted to NF-G7 contact boxes.

Reads RGB/calibration/current known wearer motion only; never evaluator targets,
native depth, object boxes or train/val labels. The TTC/corridor adapter is new,
so this is not a verbatim historical nine-cell G4 score.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
from near_field import Camera, surface_support
from ground_anchor import fit_ground
from rgb_replay import _metric_class
from sparse_structure import pose_matrix

from research_backend import torch_observation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def contact_scores(depth, frame, hfov):
    h,w = depth.shape
    camera_pose, body_pose = frame['camera_transform'], frame['wearer_transform']
    cam = Camera(w,h,hfov,camera_pose['z']-body_pose['z'],camera_pose['pitch'])
    fit = fit_ground(depth,cam)
    d = torch.as_tensor(depth,device='cuda:0')
    yy,xx = torch.meshgrid(torch.arange(h,device=d.device),torch.arange(w,device=d.device),indexing='ij')
    f = w/(2*np.tan(np.deg2rad(hfov/2)))
    ray = torch.stack(((xx-(w-1)/2)/f,(yy-(h-1)/2)/f,torch.ones_like(d)),-1)
    rotation, origin = pose_matrix(camera_pose)
    points = (ray*d[...,None]) @ torch.tensor(rotation.T,device=d.device,dtype=d.dtype)
    points += torch.tensor(origin-np.array([body_pose[k] for k in ('x','y','z')]),device=d.device,dtype=d.dtype)
    # This bounded source has a body heading of zero. Head yaw is independent.
    assert body_pose['yaw'] == 0 and body_pose['pitch'] == 0 and body_pose['roll'] == 0
    heights = [points[...,2]]
    if fit.status == 'ACCEPTED':
        a,b,intercept = fit.plane
        pitch = np.deg2rad(camera_pose['pitch'])
        rayx = np.cos(pitch)+np.sin(pitch)*ray[...,1]
        rayz = np.sin(pitch)-np.cos(pitch)*ray[...,1]
        heights.append(d*rayz-a*d*rayx-b*d*ray[...,0]-intercept)
    valid = torch.isfinite(d)&(d>.08)&(d<12.)
    scores,states,nearest,weak_counts = [],[],[],[]
    for low,high,halfwidth,front in ((.65,1.4,.28,.18),(1.4,1.85,.18,.13)):
        best = float('inf')
        weak = 0
        for height in heights:
            eligible = valid&(height>=low)&(height<high)
            support = surface_support(d,height,eligible)
            corridor = (points[...,1].abs()<=halfwidth)&(points[...,0]>=-front)
            relevant = eligible&corridor
            weak += int((relevant&~support).sum())
            selected = support&corridor
            if selected.any():
                best = min(best,float(points[...,0][selected].min())-front)
        contact = max(0.,best)/frame['speed_m_s']
        predicted = [int(contact<=horizon) for horizon in (1.,2.,3.)]
        uncertain = weak>0 or float(valid.float().mean())<.5 or fit.status!='ACCEPTED'
        state = ['OBSTACLE' if s else 'UNKNOWN' if uncertain else 'NO_NEAR_OBSERVED' for s in predicted]
        scores.append(predicted);states.append(state);nearest.append(None if not np.isfinite(contact) else contact)
        weak_counts.append(weak)
    return scores,states,dict(contact_s=nearest,weak_counts=weak_counts,ground=fit.summary())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--metric-source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();capture=args.capture.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    assert torch.cuda.is_available()
    out.mkdir(parents=True);(out/'depth').mkdir()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data_path=capture/'model/dataset.json';data=json.loads(data_path.read_text())
    frames={f['sample_index']:f for f in data['frames']}
    cls,identity=_metric_class(args.metric_source.resolve())
    assert sha(args.weights)=='b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545'
    model=cls(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20.)
    model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
    model.to('cuda:0').eval()
    rows=[];details=[];times=[];started=time.perf_counter()
    write=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    try:
        with torch.inference_mode():
            for n,sample in enumerate(data['samples']):
                frame=frames[sample['frame_indices'][-1]]
                path=capture/'model'/frame['rgb_path']
                image=cv2.imread(str(path));assert image.shape==(360,640,3)
                torch.cuda.synchronize();tick=time.perf_counter()
                depth=np.asarray(model.infer_image(image,518),dtype=np.float32)
                torch.cuda.synchronize();infer_ms=(time.perf_counter()-tick)*1000
                tick=time.perf_counter()
                scores,states,detail=contact_scores(depth,frame,data['calibration']['horizontal_fov_degrees'])
                torch.cuda.synchronize();post_ms=(time.perf_counter()-tick)*1000
                np.save(out/'depth'/f"{frame['sample_index']:04d}.npy",depth,allow_pickle=False)
                rows.append(dict(arm='mde_contact',seed=0,sample_id=sample['sample_id'],split=sample['split'],scores=scores,states=states))
                details.append(dict(sample_id=sample['sample_id'],rgb_sha256=sha(path),
                                    depth_sha256=sha(out/'depth'/f"{frame['sample_index']:04d}.npy"),**detail))
                times.append(dict(infer_ms=infer_ms,post_ms=post_ms))
                if (n+1)%24==0:
                    progress=dict(stage='mde',completed=n+1,total=len(data['samples']),elapsed_s=time.perf_counter()-started)
                    write('progress.json',progress);print(json.dumps(progress),flush=True)
        write('predictions.json',dict(rows=rows,target_order=['BODY','HEAD'],horizons_s=[1,2,3],complete=True))
        write('receipt.json',dict(status='COMPLETE',details=details,times=times,input_dataset_sha256=sha(data_path),
              model=dict(weights_sha256=sha(args.weights),source=identity,backend=asdict(torch_observation(model=model))),
              inference_calls=len(rows),test_labels_read=False,native_depth_read=False,code_sha256=sha(Path(__file__)),
              scope='Fixed MDE and raw/ground support adapted to body corridor and constant speed; binary operating point',
              timing={k:dict(p50=float(np.median([t[k] for t in times])),p95=float(np.quantile([t[k] for t in times],.95))) for k in times[0]}))
        print(json.dumps(dict(status='COMPLETE',samples=len(rows))))
    except BaseException as e:
        write('failure.json',dict(error=repr(e),completed=len(rows)))
        raise
    finally:
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
