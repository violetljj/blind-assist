"""Bounded local inverse-depth surface recovery; no scene/evaluator inputs.

One fixed mechanism: complete 3x3 support, original axial span<=5cm,
least-squares inverse-depth plane and max metric residual<=1cm. Missing pixels,
isolated/thin evidence and failed fits retain original vertices. No extrapolation.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

WIDTH, HEIGHT = 640, 360
FOCAL = WIDTH / (2 * math.tan(math.radians(35.)))
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def fit_numpy(depth):
    z = np.asarray(depth, dtype=np.float64)
    patches = np.stack([z[1+dy:HEIGHT-1+dy,1+dx:WIDTH-1+dx] for dy,dx in OFFSETS])
    valid = np.isfinite(patches) & (patches >= .5) & (patches <= 4.)
    safe = np.where(valid,patches,1.)
    inverse = 1. / safe
    c = inverse.mean(0)
    a = sum(dx*inverse[k] for k,(_,dx) in enumerate(OFFSETS)) / 6.
    b = sum(dy*inverse[k] for k,(dy,_) in enumerate(OFFSETS)) / 6.
    fitted_inverse = np.stack([c+dx*a+dy*b for dy,dx in OFFSETS])
    positive = (fitted_inverse > 0).all(0)
    predicted = 1. / np.maximum(fitted_inverse,np.finfo(np.float64).tiny)
    residual = np.max(np.abs(predicted-safe),axis=0)
    accepted = valid.all(0) & ((safe.max(0)-safe.min(0)) <= .05) & positive & (residual <= .01)
    out=z.copy();out[1:-1,1:-1]=np.where(accepted,1./c,z[1:-1,1:-1])
    mask=np.zeros(z.shape,bool);mask[1:-1,1:-1]=accepted
    return out,mask


def fit_cuda(depth):
    import torch
    z=torch.as_tensor(np.asarray(depth),dtype=torch.float64,device='cuda')
    assert z.is_cuda
    patches=torch.stack([z[1+dy:HEIGHT-1+dy,1+dx:WIDTH-1+dx] for dy,dx in OFFSETS])
    valid=torch.isfinite(patches)&(patches>=.5)&(patches<=4.)
    safe=torch.where(valid,patches,1.)
    inverse=1./safe;c=inverse.mean(0)
    a=sum(dx*inverse[k] for k,(_,dx) in enumerate(OFFSETS))/6.
    b=sum(dy*inverse[k] for k,(dy,_) in enumerate(OFFSETS))/6.
    fitted_inverse=torch.stack([c+dx*a+dy*b for dy,dx in OFFSETS])
    residual=(1./fitted_inverse.clamp_min(torch.finfo(torch.float64).tiny)-safe).abs().amax(0)
    accepted=valid.all(0)&((safe.amax(0)-safe.amin(0))<=.05)&(fitted_inverse>0).all(0)&(residual<=.01)
    out=z.clone();out[1:-1,1:-1]=torch.where(accepted,1./c,z[1:-1,1:-1])
    mask=torch.zeros_like(z,dtype=torch.bool);mask[1:-1,1:-1]=accepted
    # Include transfer back to CPU topology/contact in measured work.
    return out.cpu().numpy(),mask.cpu().numpy()


def depth_to_geometry(depth, *, backend='numpy'):
    z=np.asarray(depth,dtype=np.float64)
    if z.shape!=(HEIGHT,WIDTH):raise ValueError('Expected640x360 axial depth')
    if backend not in ('numpy','cuda'):raise ValueError('Select measured numpy or CUDA backend')
    fitted,accepted=(fit_cuda if backend=='cuda' else fit_numpy)(z)
    yy,xx=np.indices(z.shape)
    right=(xx-WIDTH/2)/FOCAL;up=-(yy-HEIGHT/2)/FOCAL
    fov=(np.abs(np.degrees(np.arctan(right)))<=22.5)&(np.abs(np.degrees(np.arctan(up)))<=20.)
    keep=np.isfinite(z)&(z>=.5)&(z<=4.)&fov
    index=np.full(z.shape,-1,np.int32);index[keep]=np.arange(int(keep.sum()),dtype=np.int32)
    rays=np.stack([np.ones_like(right),right,up],axis=-1)[keep]
    raw=rays*z[keep,None];vertices=rays*fitted[keep,None]
    horizontal=keep[:,:-1]&keep[:,1:]&(np.abs(z[:,:-1]-z[:,1:])<=.05)
    vertical=keep[:-1,:]&keep[1:,:]&(np.abs(z[:-1,:]-z[1:,:])<=.05)
    edges=[np.stack([index[:,:-1][horizontal],index[:,1:][horizontal]],axis=1),
           np.stack([index[:-1,:][vertical],index[1:,:][vertical]],axis=1)]
    a,b,c,d=index[:-1,:-1],index[:-1,1:],index[1:,:-1],index[1:,1:]
    complete=(a>=0)&(b>=0)&(c>=0)&(d>=0)
    za,zb,zc,zd=z[:-1,:-1],z[:-1,1:],z[1:,:-1],z[1:,1:]
    diagonal=np.abs(zb-zc)<=.05
    first=complete&diagonal&(np.abs(za-zb)<=.05)&(np.abs(za-zc)<=.05)
    second=complete&diagonal&(np.abs(zd-zb)<=.05)&(np.abs(zd-zc)<=.05)
    triangles=np.concatenate([np.stack([a[first],b[first],c[first]],axis=1),
                              np.stack([b[second],d[second],c[second]],axis=1)])
    diag=first|second
    edges.append(np.stack([b[diag],c[diag]],axis=1))
    return dict(raw_points=raw,fitted_points=vertices,edges=np.concatenate(edges),triangles=triangles,
        metadata=dict(mechanism='complete3x3_inverse_depth_plane',stencil=3,min_depth=.5,max_depth=4.,
            original_span_m=.05,max_metric_residual_m=.01,complete_cell_required=True,
            common_fov_az_deg=22.5,common_fov_el_deg=20.,backend=backend,
            valid_vertices=int(keep.sum()),fitted_vertices=int((accepted&keep).sum()),
            moved_vertices=int((keep&(fitted!=z)).sum()),
            missing_pixels_filled=0,vertices_deleted=0,
            contact_backend='numba_cpu',contact_cpu_reason='GPU_BACKEND_UNAVAILABLE',
            contact_reason='Variable-length convex clipping implemented as fixed CPU branching kernel; no CUDA clipping implementation'))


@njit
def _objective(vertices,count,axis):
    if axis==0:
        best=np.inf
        for i in range(count):best=min(best,vertices[i,0])
        return best
    low=np.inf;high=-np.inf
    for i in range(count):low=min(low,vertices[i,1]);high=max(high,vertices[i,1])
    if low<=0.<=high:return 0.
    return min(abs(low),abs(high))


@njit
def _contact(points,edges,triangles,lo,hi,axis,with_surface):
    best=np.inf
    for k in range(len(points)):
        p=points[k]
        if np.all(p>=lo) and np.all(p<=hi):
            value=p[0] if axis==0 else abs(p[1]);best=min(best,value)
    if not with_surface:return best
    segment=np.empty((2,3),np.float64)
    for edge in edges:
        a=points[edge[0]];b=points[edge[1]];lower=0.;upper=1.;ok=True
        for j in range(3):
            delta=b[j]-a[j]
            if delta==0.:
                if a[j]<lo[j] or a[j]>hi[j]:ok=False;break
            else:
                t0=(lo[j]-a[j])/delta;t1=(hi[j]-a[j])/delta
                lower=max(lower,min(t0,t1));upper=min(upper,max(t0,t1))
                if lower>upper:ok=False;break
        if ok:
            segment[0]=a+lower*(b-a);segment[1]=a+upper*(b-a)
            best=min(best,_objective(segment,2,axis))
    polygon=np.empty((16,3),np.float64);scratch=np.empty((16,3),np.float64)
    for triangle in triangles:
        reject=False
        for j in range(3):
            mn=min(points[triangle[0],j],points[triangle[1],j],points[triangle[2],j])
            mx=max(points[triangle[0],j],points[triangle[1],j],points[triangle[2],j])
            if mx<lo[j] or mn>hi[j]:reject=True;break
        if reject:continue
        for i in range(3):polygon[i]=points[triangle[i]]
        count=3
        for plane in range(6):
            j=plane//2;is_lower=plane%2==0;bound=lo[j] if is_lower else hi[j]
            next_count=0
            if count==0:break
            for i in range(count):
                previous=polygon[(i+count-1)%count];current=polygon[i]
                old_inside=previous[j]>=bound if is_lower else previous[j]<=bound
                new_inside=current[j]>=bound if is_lower else current[j]<=bound
                if old_inside!=new_inside:
                    t=(bound-previous[j])/(current[j]-previous[j])
                    scratch[next_count]=previous+t*(current-previous);next_count+=1
                if new_inside:scratch[next_count]=current;next_count+=1
            count=next_count
            for i in range(count):polygon[i]=scratch[i]
        if count:best=min(best,_objective(polygon,count,axis))
    return best


def contact(geometry,lo,hi,axis='x',representation='surface'):
    """Minimum forward x or absolute lateral y of clipped geometry; None=UNKNOWN.

    Caller supplies camera-coordinate box. This function intersects axial domain
    [.5,4] and never treats absence of supported contact as free space.
    """
    if axis not in ('x','abs_y'):raise ValueError('axis must be x or abs_y')
    if representation not in ('raw','fitted','surface'):raise ValueError('Invalid representation')
    lo=np.asarray(lo,dtype=np.float64).copy();hi=np.asarray(hi,dtype=np.float64).copy()
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(np.r_[lo,hi]).all():raise ValueError('Finite3D box required')
    lo[0]=max(lo[0],.5);hi[0]=min(hi[0],4.)
    if np.any(lo>hi):return None
    points=geometry['raw_points'] if representation=='raw' else geometry['fitted_points']
    if representation!='surface':
        selected=points[((points>=lo)&(points<=hi)).all(axis=1)]
        if not len(selected):return None
        return float(selected[:,0].min() if axis=='x' else np.abs(selected[:,1]).min())
    value=_contact(np.asarray(points,dtype=np.float64),geometry['edges'],geometry['triangles'],lo,hi,
                   0 if axis=='x' else 1,representation=='surface')
    return float(value) if np.isfinite(value) else None
