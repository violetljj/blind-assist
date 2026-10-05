"""Causal public query volume coverage; no objects, truth or labels accepted.

The travel tangent is estimated from the last three noisy translation increments
(0.6 s), with declared gravity vertical and the current estimated sensor origin.
Coverage is exact convex-polyhedron volume inside the rectangular 45 degree FOV.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull

FRAMES=np.arange(3,16)
HEIGHTS=((-0.2,0.42),(0.42,0.9)) # HEAD, BODY in original M3 query coordinates
EDGE=float(np.tan(np.pi/8))
LOOKBACK=3

def extrinsic(angle):
    from cnh_dual_sensor_alarm import extrinsic as e
    return e(float(angle))

def estimated_query_poses(noisy,angles=(-15,15),frames=FRAMES):
    """[sensor,frame,4,4], based only on poses available by each frame.

    Horizontal displacement shorter than 1 micrometre gives NaN, mapped to
    zero public coverage (UNKNOWN), never silently treated as observed clear.
    """
    noisy=np.asarray(noisy,float);frames=np.asarray(frames,int)
    if noisy.ndim!=3 or noisy.shape[1:]!=(4,4) or not np.isfinite(noisy).all():
        raise ValueError('Expected finite estimated pose sequence')
    if (frames<LOOKBACK).any() or (frames>=len(noisy)).any():raise ValueError('Three causal displacement intervals required')
    result=np.full((len(angles),len(frames),4,4),np.nan)
    for ti,f in enumerate(frames):
        d=noisy[f,:3,3]-noisy[f-LOOKBACK,:3,3]
        if np.hypot(d[0],d[2])<1e-6:continue
        yaw=np.arctan2(d[0],d[2]);c,s=np.cos(yaw),np.sin(yaw)
        world_to_query=np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for bi,a in enumerate(angles):
            q=np.eye(4);q[:3,:3]=world_to_query@noisy[f,:3,:3]@extrinsic(a)[:3,:3]
            result[bi,ti]=q
    return result

def _polytope_volume(a,b):
    """Volume of bounded {x:a@x<=b}, by plane-triplet vertices."""
    triples=np.array(list(combinations(range(len(b)),3)))
    aa=a[triples];bb=b[triples];ok=np.abs(np.linalg.det(aa))>1e-12
    if not ok.any():return 0.
    vertices=np.linalg.solve(aa[ok],bb[ok,...,None])[...,0]
    vertices=vertices[(vertices@a.T<=b[None]+1e-10).all(1)]
    vertices=np.unique(np.round(vertices,12),axis=0)
    if len(vertices)<4 or np.linalg.matrix_rank(vertices-vertices[0],tol=1e-10)<3:return 0.
    return float(ConvexHull(vertices).volume)

@lru_cache(maxsize=65536)
def _coverage_one(key,query):
    q=np.asarray(key).reshape(4,4)
    inv=np.linalg.inv(q);r=inv[:3,:3];t=inv[:3,3]
    lo=np.array([-.29,HEIGHTS[query][0],.9]);hi=np.array([.29,HEIGHTS[query][1],2.5])
    # FOV sensor points: |x/z|<=tan22.5, |y/z|<=tan22.5, z>=0.
    ns=np.array([[1,0,-EDGE],[-1,0,-EDGE],[0,1,-EDGE],[0,-1,-EDGE],[0,0,-1.]])
    a=np.concatenate((np.eye(3),-np.eye(3),ns@r))
    b=np.concatenate((hi,-lo,-ns@t))
    return float(np.clip(_polytope_volume(a,b)/np.prod(hi-lo),0.,1.))

def query_coverage(corridor_from_sensor,query=None):
    """[...,4,4] -> [...,2] HEAD/BODY volume fractions, or [...] query.

    NaN pose means unavailable direction and coverage zero. A rectangular
    sensor FOV is used; no target, reflectance, return or evaluator truth enters.
    """
    q=np.asarray(corridor_from_sensor,float)
    if q.shape[-2:]!=(4,4):raise ValueError('Rigid-transform shape required')
    queries=(0,1) if query is None else (int(query),)
    if not set(queries).issubset({0,1}):raise ValueError('Query must be HEAD0/BODY1')
    values=np.array([[0. if not np.isfinite(p).all() else _coverage_one(tuple(np.round(p.ravel(),12)),i) for i in queries] for p in q.reshape(-1,4,4)])
    shaped=values.reshape(q.shape[:-2]+(len(queries),))
    return shaped if query is None else shaped[...,0]

def natural_coverage(unit,config,frames=FRAMES,angles=(-15,15)):
    from cnh_cvr_pilot import motion_metadata
    noisy=motion_metadata(int(unit),int(config))[2]
    return query_coverage(estimated_query_poses(noisy,angles,frames))

def fuse(scores,coverage,rule):
    """Scores/coverage [...,sensor,frame] -> [...,frame]; after smoothing.

    No eligible branch returns -inf: this is an absent alarm, not clear proof.
    G2 ties within 1e-12 use max; G3 averages already smoothed logits.
    """
    scores=np.asarray(scores,float);coverage=np.broadcast_to(np.asarray(coverage,float),scores.shape)
    if scores.shape[-2]!=2 or not np.isfinite(scores).all():raise ValueError('Two finite branch logits required')
    if not np.isfinite(coverage).all() or (coverage<0).any() or (coverage>1).any():raise ValueError('Finite [0,1] coverage required')
    if rule.startswith('G1_'):
        return np.where(coverage>=float(rule.split('_')[1]),scores,-np.inf).max(-2)
    if rule=='G2':
        eligible=(coverage>0)&(coverage>=coverage.max(-2,keepdims=True)-1e-12)
        return np.where(eligible,scores,-np.inf).max(-2)
    if rule=='G3':
        den=coverage.sum(-2)
        return np.divide((scores*coverage).sum(-2),den,out=np.full_like(den,-np.inf),where=den>0)
    raise ValueError('Frozen candidate not recognized: '+rule)
