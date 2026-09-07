"""G14 all-visible-scene native-depth truth; no object/AABB/name filtering."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from contact_retina_spec import BODY_BOXES


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_spec(source_path,evaluator_path,expected_sha256):
    if sha(source_path)!=expected_sha256:
        raise ValueError('Frozen source spec byte identity mismatch')
    source=read(source_path); evaluator=read(evaluator_path)
    if source.get('schema')!='nf-g14-worlds-spec-v1' or source!=evaluator:
        raise ValueError('Frozen evaluator spec semantic identity mismatch')
    return evaluator


def camera_points(native,camera,hfov=100.):
    """Axial camera-forward depth, not Euclidean ray length; UE right/down rays."""
    if abs(float(camera.get('roll',0.)))>1e-8: raise ValueError('Camera roll must be zero for this projection contract')
    h,w=native.shape; device=native.device; dtype=native.dtype
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    focal=w/(2*np.tan(np.radians(hfov/2)))
    rx,ry=(xx-(w-1)/2)/focal,(yy-(h-1)/2)/focal
    p,y=np.radians([float(camera['pitch']),float(camera['yaw'])])
    forward=torch.tensor([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)],device=device,dtype=dtype)
    right=torch.tensor([-np.sin(y),np.cos(y),0.],device=device,dtype=dtype)
    up=torch.tensor([-np.sin(p)*np.cos(y),-np.sin(p)*np.sin(y),np.cos(p)],device=device,dtype=dtype)
    rays=forward+rx[...,None]*right-ry[...,None]*up
    origin=torch.tensor([camera[k] for k in ('x','y','z')],device=device,dtype=dtype)
    return origin+native[...,None]*rays


def corridor_masks(points,valid,wearer,query_range=3.):
    if any(abs(float(wearer.get(k,0.)))>1e-8 for k in ('yaw','pitch','roll')):
        raise ValueError('Frozen wearer query axes require zero yaw/pitch/roll')
    masks=[]
    for low,high in BODY_BOXES:
        near=(points[...,0]>=wearer['x']+high[0])&(points[...,0]<=wearer['x']+high[0]+query_range)
        near&=(points[...,1]>=wearer['y']+low[1])&(points[...,1]<=wearer['y']+high[1])
        near&=(points[...,2]>=wearer['z']+low[2])&(points[...,2]<=wearer['z']+high[2])
        masks.append(near&valid)
    return torch.stack(masks)


def visible_support(native,camera,wearer,hfov=100.,query_range=3.):
    valid=torch.isfinite(native)&(native>0)&(native<100.)
    points=camera_points(native,camera,hfov)
    masks=corridor_masks(points,valid,wearer,query_range)
    target=(masks.sum((-2,-1))>=3).to(torch.int64)
    support=F.max_pool2d(masks[None].float(),kernel_size=(20,20),stride=(20,20))[0].byte()
    return target,support,masks,valid


@torch.inference_mode()
def verify(root):
    root=root.resolve()
    if (root/'verification.json').exists(): raise FileExistsError('Refuse overwrite verification')
    receipt=read(root/'receipt.json')
    if receipt.get('status')!='PASS' or not receipt.get('source_unchanged'): raise ValueError('Capture did not preserve source or complete')
    if not read(root/'process-release.json').get('released'): raise ValueError('Capture process not released')
    spec_path=root/'evaluator/spec.json'; dataset_path=root/'model/dataset.json'
    spec=verified_spec(root/'source/spec.json',spec_path,receipt['spec_sha256']); dataset=read(dataset_path)
    if dataset['samples']!=spec['samples'] or len(dataset['frames'])!=len(spec['cases']): raise ValueError('Capture coverage does not match specification')
    calibration=dataset['calibration']
    if (calibration['width'],calibration['height'],float(calibration['horizontal_fov_degrees']))!=(640,360,100.): raise ValueError('Frozen optical calibration mismatch')
    if float(spec.get('query_range_m',3.))!=3.: raise ValueError('Frozen query range mismatch')
    samples=dataset['samples']; indices={}; ids=set()
    for sample in samples:
        if set(sample)!={'sample_id','clip_id','group_id','split','frame_indices'} or sample['split']!='test' or len(sample['frame_indices'])!=1: raise ValueError('Sanitized single-frame TEST samples required')
        index=int(sample['frame_indices'][0])
        if sample['sample_id'] in ids or index in indices: raise ValueError('Duplicate sample/frame identity')
        ids.add(sample['sample_id']); indices[index]=sample
    if set(indices)!=set(range(len(spec['cases']))): raise ValueError('Samples do not cover every captured case exactly')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for dense native truth verification')
    torch.set_num_threads(1); started=time.perf_counter(); targets={}; supports={}; rows=[]
    for index,(frame,case) in enumerate(zip(dataset['frames'],spec['cases'])):
        if frame!=dict(sample_index=index,clip_id=case['clip_id'],frame_in_clip=case['frame_in_clip'],rgb_path=f'sample/{index:04d}.png',time_s=case['time_s']): raise ValueError('Frame/spec identity mismatch')
        sample=indices[index]
        if sample['clip_id']!=case['clip_id'] or sample['group_id']!=case['group_id']: raise ValueError('Case/sample group identity mismatch')
        if not all(k in case for k in ('world_id','layout_id','group_id','relation','camera','wearer')): raise ValueError('Missing world/geometry metadata')
        for pose in (case['camera'],case['wearer']):
            if not all(np.isfinite(float(pose[k])) for k in ('x','y','z','pitch','yaw','roll')): raise ValueError('Nonfinite pose')
        rgb_path=root/'model'/frame['rgb_path']; native_path=root/f'evaluator/native/{index:04d}.npy'
        with Image.open(rgb_path) as image:
            if image.size!=(640,360): raise ValueError('RGB dimensions mismatch')
            image.convert('RGB').load()
        native=np.load(native_path,allow_pickle=False)
        if native.shape!=(360,640) or native.dtype!=np.float32 or not np.isfinite(native).all() or (native<0).any() or (native>=100).any(): raise ValueError('Invalid native axial-depth array')
        label,support,masks,valid=visible_support(torch.from_numpy(native).cuda(),case['camera'],case['wearer'])
        valid_count=int(valid.sum())
        if valid_count==0: raise ValueError('Native depth has no visible scene surfaces')
        sid=sample['sample_id']; targets[sid]=label.cpu().tolist()+[-1,-1];supports[sid]=support.cpu().numpy()
        rows.append(dict(sample_index=index,sample_id=sid,world_id=case['world_id'],layout_id=case['layout_id'],group_id=case['group_id'],relation=case['relation'],
            valid_native_pixels=valid_count,invalid_or_no_return_pixels=int(valid.numel())-valid_count,visible_near_pixels=masks.sum((-2,-1)).cpu().tolist(),
            near_targets=targets[sid][:2],rgb_sha256=sha(rgb_path),native_sha256=sha(native_path)))
    labels_path=root/'evaluator/labels.json'; support_path=root/'evaluator/support.npz'
    labels_path.write_text(json.dumps(dict(targets=targets),indent=2)+'\n',encoding='utf-8');np.savez_compressed(support_path,**supports)
    torch.cuda.synchronize()
    result=dict(status='PASS',schema='nf-g14-worlds-verification-v1',frames=len(rows),samples=len(targets),rows=rows,backend='CUDA',device=torch.cuda.get_device_name(),elapsed_s=time.perf_counter()-started,
        dataset_sha256=sha(dataset_path),source_spec_sha256=sha(root/'source/spec.json'),spec_sha256=sha(spec_path),labels_sha256=sha(labels_path),support_sha256=sha(support_path),code_sha256=sha(Path(__file__)),
        assumptions=dict(depth='linear axial camera-forward metres, valid0<depth<100; zero is no return, not free space',resolution=[640,360],horizontal_fov_degrees=100,principal_point=[319.5,179.5],camera_roll_degrees=0,
            wearer_angles_degrees=[0,0,0],body_boxes=BODY_BOXES,query_forward='wearer.x+head high[0] through +3m inclusive',near_positive_native_pixels=3,support='20x20 maxpool to18x32; all visible scene surfaces, no asset/name/AABB filtering'),
        scope='Geometry-derived visible near-surface presence only. Negative means no visible in-query surface above pixel minimum; invalid/occluded/unseen space is not certified free. Native surfaces and fixed projection assumptions remain simulator evidence.')
    (root/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True,type=Path);verify(p.parse_args().capture)
