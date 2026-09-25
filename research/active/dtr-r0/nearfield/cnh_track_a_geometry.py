"""Exact triangle geometry for the frozen Track A procedural pilot.

Coordinates and transforms are supplied by the caller; no camera pose or labels
are inferred here. Distances are metric (directions are normalized internally).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def box_mesh(low, high):
    low, high = np.asarray(low, float), np.asarray(high, float)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
        raise ValueError('box bounds must have positive extents')
    v = np.array([[low[j] if not (i >> j) & 1 else high[j] for j in range(3)] for i in range(8)])
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return np.array([v[list(t)] for a,b,c,d in faces for t in [(a,b,c),(a,c,d)]])


def cylinder_mesh(center, radius, height, axis=1, sides=64):
    center = np.asarray(center, float)
    if center.shape != (3,) or radius <= 0 or height <= 0 or axis not in (0,1,2) or sides < 3:
        raise ValueError('invalid cylinder')
    other = [i for i in range(3) if i != axis]
    ring = np.tile(center, (sides,1))
    angle = np.arange(sides) * (2*np.pi/sides)
    ring[:,other[0]] += radius*np.cos(angle)
    ring[:,other[1]] += radius*np.sin(angle)
    lower, upper = ring.copy(), ring.copy()
    lower[:,axis] -= height/2; upper[:,axis] += height/2
    lc, uc = center.copy(), center.copy()
    lc[axis] -= height/2; uc[axis] += height/2
    triangles = []
    for i in range(sides):
        j = (i+1)%sides
        triangles.extend([[lower[i],lower[j],upper[j]], [lower[i],upper[j],upper[i]],
                          [lc,lower[j],lower[i]], [uc,upper[i],upper[j]]])
    return np.asarray(triangles)


def raycast(origin, directions, triangles, object_ids, rho):
    """Two-sided Moller-Trumbore first hit; no-hit distance inf, ID -1.

    object_ids and rho are per triangle. Incidence cosine uses the absolute
    normal dot product so mesh winding does not change opaque response.
    """
    origin = np.asarray(origin, float)
    directions = np.asarray(directions, float)
    shape = directions.shape[:-1]
    d = directions.reshape(-1,3)
    norm = np.linalg.norm(d, axis=1)
    if origin.shape != (3,) or np.any(norm == 0):
        raise ValueError('invalid origin/direction')
    d = d/norm[:,None]
    tri = np.asarray(triangles,float).reshape(-1,3,3)
    ids, reflectance = np.broadcast_to(object_ids,(len(tri),)), np.broadcast_to(rho,(len(tri),))
    dist = np.full(len(d),np.inf); index = np.full(len(d),-1,int)
    cosine = np.zeros(len(d))
    for k, (a,b,c) in enumerate(tri):
        e1,e2 = b-a,c-a
        n = np.cross(e1,e2); nn = np.linalg.norm(n)
        if nn < 1e-15: continue
        p = np.cross(d,e2); det = p@e1
        ok = np.abs(det)>1e-12
        inv = np.divide(1.,det,out=np.zeros_like(det),where=ok)
        s = origin-a; u = (p@s)*inv
        q = np.cross(s,e1); v = (d@q)*inv; t = np.dot(e2,q)*inv
        hit = ok & (u>=-1e-10) & (v>=-1e-10) & (u+v<=1+1e-10) & (t>=1e-9) & (t<dist)
        dist[hit]=t[hit]; index[hit]=k; cosine[hit]=np.abs(d[hit]@n)/nn
    valid=index>=0
    output_rho=np.zeros(len(d)); output_ids=np.full(len(d),-1,dtype=np.int64)
    output_rho[valid]=reflectance[index[valid]]; output_ids[valid]=ids[index[valid]]
    return {key:value.reshape(shape) for key,value in dict(distance=dist,rho=output_rho,
            cos=cosine,object_id=output_ids,triangle_id=index,valid=valid).items()}


def signed_margin(triangles, low, high, return_witness=False):
    """max_{p on union of triangle surfaces} min six signed box slacks.

    Each triangle is represented by barycentric coordinates; maximizing the
    minimum slack is a linear program, including separated and touching cases.
    Optional witness is the maximizing surface point, including zero-area contact;
    an empty triangle union returns (-inf, None) in witness mode.
    """
    low,high=np.asarray(low,float),np.asarray(high,float)
    if np.any(high < low): raise ValueError('invalid box')
    best=-np.inf
    witness=None
    for tri in np.asarray(triangles,float).reshape(-1,3,3):
        upper=float(np.minimum(tri.max(axis=0)-low,high-tri.min(axis=0)).min())
        if upper <= best + 1e-12: continue
        # lambda*coordinate - t >= low; lambda*coordinate + t <= high.
        a=np.concatenate([np.column_stack((-tri.T,np.ones(3))),
                          np.column_stack((tri.T,np.ones(3)))])
        result=linprog([0,0,0,-1], A_ub=a,b_ub=np.concatenate([-low,high]),
                       A_eq=[[1,1,1,0]],b_eq=[1],bounds=[(0,None)]*3+[(None,None)],
                       method='highs',options={'primal_feasibility_tolerance':1e-9,
                                               'dual_feasibility_tolerance':1e-9})
        if not result.success: raise RuntimeError('margin LP failed: '+result.message)
        value=float(result.x[3])
        if value > best:
            best=value
            witness=result.x[:3] @ tri
    return (best,witness) if return_witness else best


def clip_triangles(triangles, low, high):
    """Sutherland-Hodgman clipping to six planes, followed by fan triangulation."""
    output=[]
    for tri in np.asarray(triangles,float).reshape(-1,3,3):
        polygon=list(tri)
        for axis in range(3):
            for bound,sign in [(low[axis],1),(high[axis],-1)]:
                if not polygon: break
                new=[]
                for i,b in enumerate(polygon):
                    a=polygon[i-1]; sa=sign*(a[axis]-bound); sb=sign*(b[axis]-bound)
                    if (sa>=0)!=(sb>=0): new.append(a+(b-a)*(sa/(sa-sb)))
                    if sb>=0: new.append(b)
                polygon=new
        for i in range(1,len(polygon)-1):
            t=np.array([polygon[0],polygon[i],polygon[i+1]])
            if np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))>1e-15: output.append(t)
    return np.asarray(output,float).reshape(-1,3,3)


def triangle_areas(triangles):
    t=np.asarray(triangles,float).reshape(-1,3,3)
    return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)


def surface_quadrature(triangles, max_edge=.01):
    """Deterministic longest-edge subdivision, area-weighted centroids.

    Does not deduplicate overlapping surfaces; caller must establish union
    ownership before combining objects. No visibility/oracle input is produced.
    """
    if max_edge<=0: raise ValueError('max_edge must be positive')
    pending=list(np.asarray(triangles,float).reshape(-1,3,3)); leaves=[]
    while pending:
        t=pending.pop()
        lens=np.array([np.linalg.norm(t[(i+1)%3]-t[i]) for i in range(3)])
        edge=int(np.argmax(lens))
        if lens[edge]<=max_edge*(1+1e-12): leaves.append(t); continue
        a,b,c=t[edge],t[(edge+1)%3],t[(edge+2)%3]; mid=(a+b)/2
        pending.extend([np.array([a,mid,c]),np.array([mid,b,c])])
    leaves=np.asarray(leaves).reshape(-1,3,3)
    return leaves.mean(axis=1),triangle_areas(leaves)
