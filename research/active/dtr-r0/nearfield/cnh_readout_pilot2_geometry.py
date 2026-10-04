"""Exact retained SUB3 query-overlap operator and shared pose-table cache.

This module opens transforms/valid lengths only. Unit-pulse engineering checks
use constructed poses and an independent numpy voxel integration reference.
Complete cache generation requires a frozen pilot2 plan and explicit CLI call.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cnh_cvr_projection import LOW,STEP,SHAPE,WIDTH,EDGE,SUB,grid,cell_volumes,query_masks

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OLD=ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(path)


class QueryOverlap:
    def __init__(self,device='cuda',batch_poses=8):
        self.device=device;self.batch_poses=batch_poses
        centers,points=grid();masks=query_masks().reshape(2,-1)
        select=np.flatnonzero(masks.any(0))
        self.points=torch.as_tensor(points[select].reshape(-1,3),device=device,dtype=torch.float64)
        self.point_masks=torch.as_tensor(np.repeat(masks[:,select],SUB**3,axis=1),device=device,dtype=torch.float64)
        self.volumes=torch.as_tensor(cell_volumes().reshape(-1),device=device,dtype=torch.float64)
        self.voxel_volume=float(np.prod(STEP))

    @torch.no_grad()
    def weights(self,transforms):
        """[P,4,4] -> [P,2,8,8,16] float32 query mass fractions."""
        ts=torch.as_tensor(transforms,device=self.device,dtype=torch.float64)
        if ts.ndim!=3 or ts.shape[1:]!=(4,4):raise ValueError('Rigid transforms[P,4,4] required')
        output=[]
        for start in range(0,len(ts),self.batch_poses):
            t=ts[start:start+self.batch_poses]
            points=torch.bmm(self.points[None]-t[:,None,:3,3],t[:,:3,:3])
            radius=torch.linalg.vector_norm(points,dim=-1)
            xy=points[...,:2]/points[...,2:3].clamp_min(1e-30)
            ij=torch.floor((xy+EDGE)/(2*EDGE)*8).long()
            bins=torch.floor(radius/WIDTH).long()
            valid=(points[...,2]>0)&(ij>=0).all(-1)&(ij<8).all(-1)&(bins>=0)&(bins<16)
            index=(ij[...,1].clamp(0,7)*8+ij[...,0].clamp(0,7))*16+bins.clamp(0,15)
            base=valid*self.voxel_volume/(SUB**3)/self.volumes[index]
            values=base[:,None]*self.point_masks[None]
            result=torch.zeros((len(t),2,1024),device=self.device,dtype=torch.float64)
            result.scatter_add_(2,index[:,None].expand(-1,2,-1),values)
            output.append(result.float().reshape(len(t),2,8,8,16))
        return torch.cat(output,0)


def independent_field(transform,zone_y,zone_x,radial_bin):
    """Independent SUB3 integration, retaining the same voxel-center field."""
    indices=np.stack(np.meshgrid(*(np.arange(s) for s in SHAPE),indexing='ij'),-1)
    centers=LOW+(indices+.5)*STEP
    offsets=np.stack(np.meshgrid(*([(np.arange(SUB)+.5)/SUB-.5]*3),indexing='ij'),-1).reshape(-1,3)
    points=(centers.reshape(-1,1,3)+offsets[None]*STEP).reshape(-1,3)
    p=(points-transform[:3,3])@transform[:3,:3]
    radius=np.linalg.norm(p,axis=-1)
    # Nonforward points are excluded independently; avoid meaningless int64
    # overflow while classifying their tangent slopes in this CPU reference.
    xy=p[:,:2]/np.where(p[:,2:3]>0,p[:,2:3],1.)
    angular=np.floor((xy+EDGE)/(2*EDGE)*8).astype(np.int64)
    radial=np.floor(radius/WIDTH).astype(np.int64)
    selected=(p[:,2]>0)&(angular[:,0]==zone_x)&(angular[:,1]==zone_y)&(radial==radial_bin)
    vol=float(cell_volumes()[zone_y,zone_x,radial_bin])
    field=(selected.astype(np.float64)*float(np.prod(STEP))/(SUB**3)/vol).reshape(-1,SUB**3).sum(-1).reshape(SHAPE).astype(np.float32)
    return centers,field


def rotation(yaw=0.,pitch=0.):
    a,b=np.deg2rad([yaw,pitch]);ca,sa,cb,sb=np.cos(a),np.sin(a),np.cos(b),np.sin(b)
    ry=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    rx=np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    return ry@rx


def check(device='cuda',out=OUT,receipt_name='geometry_pulse_check_v2.json'):
    """Small engineering impulses; no world generation or learned model runs."""
    from cnh_cvr_v2_materialize import BatchedProjector
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    began=time.monotonic();projector=BatchedProjector(device=device);overlap=QueryOverlap(device)
    identity=np.eye(4);moved=np.eye(4);moved[:3,:3]=rotation(23,-11);moved[:3,3]=[.17,.08,.13]
    cropped=np.eye(4);cropped[:3,:3]=rotation(-17,8);cropped[:3,3]=[-.28,.12,-.45]
    cases=[(identity,3,3,0),(identity,3,4,1),(identity,3,3,5),
           (identity,4,0,7),(identity,4,7,7),(moved,1,2,4),
           (moved,5,6,8),(cropped,2,1,11),(cropped,6,7,15),
           (identity,7,3,6),(identity,4,0,5),(identity,4,7,5),
           (moved,3,0,2),(cropped,6,7,11),(cropped,6,7,12)]
    rows=[];masks=query_masks().astype(np.float64)
    for t,y,x,r in cases:
        pulse=np.zeros((1,8,8,16),np.float32);pulse[0,y,x,r]=1.
        observed=projector.sequence(pulse,t[None])[0].cpu().numpy()
        centers,expected=independent_field(t,y,x,r)
        field_delta=float(np.max(np.abs(observed-expected)))
        mass=float(observed.sum(dtype=np.float64));mass_ref=float(expected.sum(dtype=np.float64))
        q=overlap.weights(t[None])[0,:,y,x,r].cpu().numpy().astype(np.float64)
        qref=(expected[None].astype(np.float64)*masks).sum((1,2,3))
        qdelta=float(np.max(np.abs(q-qref)))
        centroid_delta=None
        if mass_ref>1e-10 and mass>1e-10:
            c=(centers*observed[...,None]).sum((0,1,2))/mass
            cr=(centers*expected[...,None]).sum((0,1,2))/mass_ref
            centroid_delta=float(np.linalg.norm(c-cr))
        if field_delta>1e-6 or abs(mass-mass_ref)>1e-6 or qdelta>1e-6:
            raise ValueError(('SUB3 impulse mismatch',y,x,r,field_delta,mass,mass_ref,qdelta))
        if centroid_delta is not None and centroid_delta>=.025:raise ValueError('Matched centroid exceeds2.5cm')
        rows.append(dict(zone_y=y,zone_x=x,bin=r,transform=t.tolist(),field_max_abs=field_delta,
                         mass=mass,mass_reference=mass_ref,query_overlap=q.tolist(),query_overlap_reference=qref.tolist(),
                         query_max_abs=qdelta,centroid_distance_m=centroid_delta,
                         centroid_status='MATCHED_DISCRETE_REFERENCE' if centroid_delta is not None else 'NO_SUPPORT'))
    # Axis/radial direction is checked analytically without pretending the
    # point midpoint is the shell's volume centroid.
    from cnh_readout_pilot2_model import volume_bin_centers,parameter_count
    c=volume_bin_centers()
    assert c[3,0,5,0]<0<c[3,7,5,0] and c[0,3,5,1]<0<c[7,3,5,1]
    assert np.all(np.diff(np.linalg.norm(c[3,3],axis=-1))>0)
    assert parameter_count()==38161 and parameter_count()<=230485
    assert any(row['query_overlap'][1]>0 for row in rows),'BODY overlap coverage missing'
    assert any(1e-5<row['mass']<.9 for row in rows),'Nonzero partial clipping coverage missing'
    receipt=dict(status='PASS',device=device,seconds=time.monotonic()-began,cases=rows,
                 coordinate_axes='Xright,Ydown; zone[y,x]; radial shells',parameters=parameter_count(),
                 reference='Independent numpy SUB3 integration; voxel-center centroid, original query masks, cropped mass retained',
                 source_sha256={str(Path(p).resolve()):sha(p) for p in (__file__,)})
    path=Path(out)/receipt_name
    if path.exists():raise FileExistsError('Pulse engineering evidence exists')
    save(path,receipt)
    del projector,overlap
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return receipt


def cache(split,device='cuda',batch_poses=8,out=OUT):
    import cnh_readout_pilot2 as C
    p=C.load_plan();C.check_budget()
    root=OLD if split in ('train','calibration','evaluation') else Path(out)
    inputs=root/'inputs'/split
    ts=np.load(inputs/'transforms.npy',mmap_mode='r');length=np.load(inputs/'length.npy',mmap_mode='r')
    if ts.shape!=(len(length),8,4,4):raise ValueError('Pose/cache axes mismatch')
    valid=np.arange(8)[None]>=8-np.asarray(length)[:,None]
    began=time.monotonic()
    unique,inverse=np.unique(np.asarray(ts)[valid].reshape(-1,16),axis=0,return_inverse=True)
    folder=Path(out)/'geometry'/split;folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'receipt.json'
    if receipt_path.exists():raise FileExistsError('Geometry cache already sealed')
    table=np.lib.format.open_memmap(folder/'table.npy',mode='w+',dtype=np.float32,shape=(len(unique)+1,2,8,8,16))
    table[0]=0
    index=np.zeros((len(length),8),np.int32);index[valid]=inverse.astype(np.int32)+1
    np.save(folder/'index.npy',index);np.save(folder/'unique_transforms.npy',unique.reshape(-1,4,4))
    operator=QueryOverlap(device,batch_poses)
    for start in range(0,len(unique),batch_poses):
        C.check_budget();chunk=unique[start:start+batch_poses].reshape(-1,4,4)
        table[start+1:start+len(chunk)+1]=operator.weights(chunk).cpu().numpy()
        if start%(batch_poses*200)==0:print('geometry',split,start,'/',len(unique),round(time.monotonic()-began,1),flush=True)
    table.flush();del table,operator
    sources=[inputs/'transforms.npy',inputs/'length.npy']
    outputs=[folder/name for name in ('table.npy','index.npy','unique_transforms.npy')]
    receipt=dict(status='COMPLETE',split=split,rows=len(length),valid_exposures=int(valid.sum()),unique_poses=len(unique),
                 seconds=time.monotonic()-began,shape=[len(unique)+1,2,8,8,16],dtype='float32',padding_index=0,
                 input_sha256={str(path):sha(path) for path in sources},output_sha256={str(path):sha(path) for path in outputs},
                 source_sha256=sha(__file__),operator='sum_v original query_masks[v] * original SUB3 voxel-volume/cell-volume; no cropped-mass renormalization')
    save(receipt_path,receipt)
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return receipt


def benchmark(device='cuda',out=OUT):
    """One engineering timing comparison on32 retained, label-free poses."""
    ts=np.load(OLD/'inputs/train/transforms.npy',mmap_mode='r')
    lengths=np.load(OLD/'inputs/train/length.npy',mmap_mode='r')
    valid=np.arange(8)[None]>=8-np.asarray(lengths[:64])[:,None]
    sample=np.unique(np.asarray(ts[:64])[valid].reshape(-1,16),axis=0)[:32].reshape(-1,4,4)
    results={};outputs={}
    for batch in (8,16):
        operator=QueryOverlap(device,batch)
        operator.weights(sample[:batch])
        if device.startswith('cuda'):torch.cuda.synchronize()
        start=time.monotonic();result=operator.weights(sample)
        if device.startswith('cuda'):torch.cuda.synchronize()
        seconds=time.monotonic()-start
        outputs[batch]=result.cpu().numpy()
        results[str(batch)]=dict(seconds=seconds,poses_per_second=len(sample)/seconds)
        del result,operator
    delta=float(np.max(np.abs(outputs[8]-outputs[16])))
    if delta>1e-6:raise ValueError('Engineering pose batching changed overlap')
    selected=min((8,16),key=lambda x:results[str(x)]['seconds'])
    receipt=dict(status='PASS',sample_poses=len(sample),batches=results,selected_batch=selected,
                 cross_batch_max_abs=delta,source_sha256=sha(__file__),
                 pose_bytes_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
                 role='one engineering throughput check; retained transform inputs, no scores/labels')
    path=Path(out)/'geometry_throughput_check.json'
    if path.exists():raise FileExistsError('Engineering timing already exists')
    save(path,receipt)
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('check','cache','benchmark'))
    parser.add_argument('--split',default='train');parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-poses',type=int,default=8);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--receipt-name',default='geometry_pulse_check_v2.json')
    args=parser.parse_args()
    result=check(args.device,args.out,args.receipt_name) if args.stage=='check' else benchmark(args.device,args.out) if args.stage=='benchmark' else cache(args.split,args.device,args.batch_poses,args.out)
    print(json.dumps(result,indent=2),flush=True)
