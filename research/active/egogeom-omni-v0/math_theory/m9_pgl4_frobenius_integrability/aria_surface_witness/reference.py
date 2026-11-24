"""Evaluator-only FoundationStereo geometry of the frozen Aria pack."""
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

cv2.setNumThreads(1)
V,U=np.indices((160,160))
SV,SU=np.indices((72,96))
NATIVE_U=np.floor((4*SU+.5)/.75).astype(int).ravel()
NATIVE_V=np.floor((4*SV+48+.5)/.75).astype(int).ravel()
BASE=1000000
OFFSET=100000


def keys3(xyz):
    grid=np.floor(xyz/.05).astype(np.int64)+OFFSET
    if np.any((grid<0)|(grid>=BASE)):
        raise ValueError("voxel outside integer encoding")
    return (grid[:,0]*BASE+grid[:,1])*BASE+grid[:,2]


def centres(keys):
    return (np.stack((keys//(BASE*BASE),keys//BASE%BASE,keys%BASE),axis=-1)-OFFSET+.5)*.05


@dataclass
class Native:
    points: np.ndarray
    source_range: np.ndarray
    source_xyz: np.ndarray
    digest: str
    pose_error: float


def read_native(base,meta,packed_r,packed_t):
    idx=meta["index"]
    data=(base/"depth"/f"depth_{idx:08d}.png").read_bytes()
    mm=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
    if mm.shape!=(512,512) or mm.dtype!=np.uint16:
        raise ValueError("Unexpected native depth")
    fu,fv,x0,y0=meta["camera"]["Parameters"]
    r=Rotation.from_quat(meta["T_world_camera"]["QuaternionXYZW"]).as_matrix()
    t=np.asarray(meta["T_world_camera"]["Translation"])
    error=max(float(np.max(abs(r-packed_r))),float(np.max(abs(t-packed_t))))
    if error>1.e-5:
        raise ValueError("Source pose mismatch")
    ds=cv2.resize(mm,(160,160),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000.
    ok=(ds>.3)&(ds<12.)
    z=ds[ok]
    scale=160/512
    pc=np.stack(((U[ok]-x0*scale)/(fu*scale)*z,(V[ok]-y0*scale)/(fv*scale)*z,z),axis=-1)
    points=((r@pc.T).T+t).astype(np.float32)
    z=mm[NATIVE_V,NATIVE_U].astype(float)/1000.
    pc=np.stack(((NATIVE_U-x0)*z/fu,(NATIVE_V-y0)*z/fv,z),axis=-1)
    return Native(points,z,pc@r.T+t,hashlib.sha256(data).hexdigest(),error)


class Ground:
    def __init__(self,pw,position):
        below=pw[pw[:,2]<position[2]-.8]
        if len(below)<200:
            raise ValueError("Insufficient ground depth on scored row")
        cells=np.floor(below[:,:2]/.25).astype(np.int64)
        codes=cells[:,0]*BASE+cells[:,1]+OFFSET
        if np.any((cells[:,1]+OFFSET<0)|(cells[:,1]+OFFSET>=BASE)):
            raise ValueError("Cell key overflow")
        order=np.lexsort((below[:,2],codes))
        unique,first,count=np.unique(codes[order],return_index=True,return_counts=True)
        self.codes=unique
        keys=np.column_stack((unique//BASE,unique%BASE-OFFSET))
        self.tree=cKDTree((keys+.5)*.25)
        q=.1*(count-1)
        lo=np.floor(q).astype(int)
        hi=np.ceil(q).astype(int)
        z=below[order,2]
        # NumPy percentile uses the input precision for b-a before interpolation.
        a,b=z[first+lo],z[first+hi]
        f=q-lo
        self.height=np.where(f<.5,a+(b-a)*f,b-(b-a)*(1.-f))

    def above(self,pw):
        ds,idx=self.tree.query(pw[:,:2],distance_upper_bound=.8,workers=1)
        inside=np.isfinite(ds)
        h=np.full(len(pw),np.nan)
        h[inside]=pw[inside,2]-self.height[idx[inside]]
        return h

    def coverage(self,position,fw,dist):
        loc=position[:2]+np.arange(.30,min(dist,4.)+1.e-6,.25)[:,None]*fw
        cells=np.floor(loc/.25).astype(np.int64)
        return float(np.isin(cells[:,0]*BASE+cells[:,1]+OFFSET,self.codes).mean()) if len(loc) else 1.


def corridor_reference(frames,position,fw):
    """Return local ground & occupied corridor voxels. fw is horizontal unit length."""
    side=np.array([fw[1],-fw[0]])
    pw=np.concatenate([f.points for f in frames])
    ground=Ground(pw,position)
    # Voxel centres move <=0.036 m horizontally from input points, so this cut is conservative.
    rel=pw[:,:2]-position[:2]
    front,lat=rel@fw,rel@side
    pw=pw[(np.abs(lat)<=.35)&(front>=.25)&(front<=4.05)]
    h=ground.above(pw)
    vox=np.unique(keys3(pw[(h>=.15)&(h<=1.9)]))
    vc=centres(vox)
    offset=vc[:,:2]-position[:2]
    f=offset@fw
    inside=(abs(offset@side)<=.30)&(f>=.30)&(f<=4.0)
    vox,f=vox[inside],f[inside]
    z=np.sort(f)
    jj=np.flatnonzero(np.searchsorted(z,z+.30,side="right")-np.arange(len(z))>=8)
    dist=float(z[jj[0]]) if len(jj) else np.inf
    target=vox[(f>=dist-1.e-12)&(f<=dist+.30+1.e-12)] if np.isfinite(dist) else vox[:0]
    return ground,vox,target,dist,ground.coverage(position,fw,dist)
