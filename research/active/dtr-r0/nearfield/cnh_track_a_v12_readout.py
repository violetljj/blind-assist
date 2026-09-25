"""H3-only v1.2 transport/scans, with exact shared-source noise propagation.

One call is one trajectory, in chronological exposure order. Residual and
variance inputs are [N,8,8,16] float arrays; poses are world_from_tof [N,4,4].
No labels, scene geometry, raw128, or fitted thresholds enter these functions.
"""
import numpy as np
from scipy import sparse
from cnh_track_a_readout import (SHAPE, EDGE, BOXES, cell_points, _hist, _poses,
                                _transform, _indices, query_weights)
from cnh_scan_development import scan, WINDOWS


def transport_matrix(current_from_past, *, range_compensation=True):
    """Return CSR A[destcell,sourcecell], W[8,8,16], and diagnostic counts.

    Each source cell contributes 16 correlated subpoints. COO duplicates are
    summed before any variance operation, including points in the same cell.
    W retains the old inverse-FOV/window coverage and never includes gain.
    """
    t = _poses(np.asarray(current_from_past)[None], 1)[0]
    original = cell_points()
    points = _transform(original, t)
    dest, valid = _indices(points)
    radius = np.linalg.norm(points, axis=-1)
    zero = radius <= 0
    valid &= ~zero
    gain = np.ones(radius.shape)
    if range_compensation:
        np.divide(np.linalg.norm(original, axis=-1), radius, out=gain, where=~zero)
        gain *= gain
    source = np.broadcast_to(np.arange(1024)[:, None], valid.shape)
    a = sparse.coo_matrix((gain[valid]/16, (dest[valid], source[valid])),
                          shape=(1024, 1024)).tocsr()
    a.sum_duplicates()
    _, covered = _indices(_transform(original, np.linalg.inv(t)))
    return a, covered.mean(-1).reshape(SHAPE), dict(zero_radius_points=int(zero.sum()),
        retained_points=int(valid.sum()), total_points=int(valid.size))


def _inputs(residual, variance, poses):
    r = _hist(residual).astype(float, copy=False)
    v = _hist(variance).astype(float, copy=False)
    if r.shape != v.shape or np.any(v < 0):
        raise ValueError('Equal residual/variance shapes and nonnegative variance required')
    return r, v, _poses(poses, len(r))


def accumulate(residual, variance, poses, k=4, *, range_compensation=True):
    """Return dict sum, coverage, mean, mean_variance, sources, diagnostics.

    sources[i] holds (past_index, CSR A) for independent original exposures.
    mean_variance is diagonal only; scans MUST use sources for covariance.
    Current exposure always has identity gain and coverage 1.
    """
    r, v, p = _inputs(residual, variance, poses)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('Positive integer history length required')
    total, var, coverage = r.copy(), v.copy(), np.ones_like(r)
    identity = sparse.eye(1024, format='csr')
    sources, diagnostics = [], []
    for i in range(len(r)):
        current = [(i, identity)]
        for j in range(max(0, i-k+1), i):
            a, w, d = transport_matrix(np.linalg.inv(p[i]) @ p[j],
                                       range_compensation=range_compensation)
            total[i] += (a @ r[j].ravel()).reshape(SHAPE)
            var[i] += (a.multiply(a) @ v[j].ravel()).reshape(SHAPE)
            coverage[i] += w
            current.append((j, a))
            diagnostics.append(dict(current=i, past=j, **d))
        sources.append(current)
    return dict(sum=total, coverage=coverage, mean=total/coverage,
                mean_variance=var/coverage**2, sources=sources, diagnostics=diagnostics)


def supported_windows(weights, tau):
    """Return six CSR window selectors [admissible_windows,1024]."""
    weights = np.asarray(weights)
    if weights.shape not in ((6,64,16), (6,8,8,16)) or not np.isfinite(weights).all():
        raise ValueError('Finite six-query H3 weights required')
    if tau not in (.25, .5, .75):
        raise ValueError('External frozen tau must be .25, .5, or .75')
    support = weights.reshape(6,8,8,16) >= tau
    ids = np.arange(1024).reshape(SHAPE)
    matrices = []
    for q in range(6):
        rows, cols, count = [], [], 0
        for h,w,d in WINDOWS:
            shape = (h,w,d)
            admissible = np.lib.stride_tricks.sliding_window_view(support[q],shape).all((-3,-2,-1))
            cells = np.lib.stride_tricks.sliding_window_view(ids,shape)[admissible].reshape(-1,h*w*d)
            rows.append(np.repeat(np.arange(count,count+len(cells)),h*w*d))
            cols.append(cells.ravel())
            count += len(cells)
        row, col = np.concatenate(rows), np.concatenate(cols)
        matrices.append(sparse.csr_matrix((np.ones(len(col)), (row,col)), shape=(count,1024)))
    return matrices


def scan_transported(residual, variance, sources, weights, tau):
    """One output frame [6]: sum_source coeff*r / sqrt(sum_source coeff²*v).

    A window's coefficients are U@A, combining source-cell contributions
    across destination cells before squaring. No independence of splats.
    """
    r, v = _hist(residual), _hist(variance)
    if r.shape != v.shape or np.any(v < 0):
        raise ValueError('Nonnegative variance matching residual required')
    output = np.full(6, -50.)
    for q, u in enumerate(supported_windows(weights, tau)):
        if not u.shape[0]:
            continue
        total, var = np.zeros(u.shape[0]), np.zeros(u.shape[0])
        for j, a in sources:
            c = u @ a
            c.sum_duplicates()
            total += c @ r[j].ravel()
            var += c.multiply(c) @ v[j].ravel()
        output[q] = max(-50., float(np.max(total/np.sqrt(np.maximum(var,1e-9)))))
    return output


def s1(residual, variance, weights, tau):
    """K1 EXACT existing Part0 scan, external tau, output [N,6]."""
    r, v = _hist(residual), _hist(variance)
    return scan(r.reshape(-1,64,16), v.reshape(-1,64,16),
                np.asarray(weights).reshape(6,64,16), tau)


def memory_scan(residual, variance, poses, current_index, T_Q_tof, k=8):
    """Outside-current-angular-FOV memory [6], fixed Q-origin .2m voxels.

    For each original exposure, each voxel's source coefficients combine all
    16 subpoints before variance. Past radial gain uses current ToF radius;
    voxel placement uses current Q. No age decay or object/visibility input.
    """
    r, v, p = _inputs(residual, variance, poses)
    i = int(current_index)
    if i != current_index or not 0 <= i < len(r) or k != 8:
        raise ValueError('Valid current index and frozen k=8 required')
    tq = _poses(np.asarray(T_Q_tof)[None], 1)[0]
    original = cell_points()
    old_radius = np.linalg.norm(original, axis=-1)
    source = np.broadcast_to(np.arange(1024)[:,None], (1024,16))
    # One dictionary per query: voxel -> [total residual, independent-frame variance].
    voxel_values = [dict() for _ in range(6)]
    for j in range(max(0, i-7), i+1):
        current = _transform(original, np.linalg.inv(p[i]) @ p[j])
        radius = np.linalg.norm(current, axis=-1)
        front = current[...,2] > 0
        tangent = np.divide(current[...,:2],current[...,2,None],
                            out=np.full((1024,16,2),np.inf),where=front[...,None])
        outside = ~(front & (np.abs(tangent) < EDGE).all(-1))
        point_q = _transform(current, tq)
        gain = np.ones_like(radius)
        if j != i:
            np.divide(old_radius,radius,out=gain,where=radius>0)
            gain *= gain
        for q,(lo,hi) in enumerate(BOXES):
            keep = outside & (radius>0) & (point_q>=lo).all(-1) & (point_q<=hi).all(-1)
            if not keep.any():
                continue
            voxels = np.floor(point_q[keep]/.2).astype(np.int64)
            unique, inverse = np.unique(voxels,axis=0,return_inverse=True)
            a = sparse.coo_matrix((gain[keep]/16,(inverse,source[keep])),
                                   shape=(len(unique),1024)).tocsr()
            a.sum_duplicates()
            total = a @ r[j].ravel()
            var = a.multiply(a) @ v[j].ravel()
            for voxel,rv,vv in zip(unique,total,var):
                record = voxel_values[q].setdefault(tuple(voxel),[0.,0.])
                record[0] += float(rv);record[1] += float(vv)
    result = np.full(6,-50.)
    for q,values in enumerate(voxel_values):
        if values:
            result[q] = max(-50.,max(rv/np.sqrt(max(vv,1e-9)) for rv,vv in values.values()))
    return result


def sequence_readouts(residual, variance, poses, T_Q_tof, *, tau_s1, tau_s2):
    """One trajectory -> B1_R/S1/S2/S3 [N,6], W and diagnostics.

    T_Q_tof accepts [4,4] shared or [N,4,4]; taus are selected externally on
    calib only. New trajectory means a new call (no hidden persistent state).
    """
    r,v,p = _inputs(residual,variance,poses)
    tq = np.asarray(T_Q_tof,float)
    if tq.shape == (4,4): tq = np.broadcast_to(tq,(len(r),4,4))
    tq = _poses(tq,len(r))
    acc = accumulate(r,v,p,4)
    out = {key:np.empty((len(r),6)) for key in ('B1_R','S1','S2','S3')}
    for i in range(len(r)):
        weights = query_weights(tq[i])
        out['B1_R'][i] = np.einsum('zb,qzb->q',acc['mean'][i].reshape(64,16),weights)
        out['S1'][i] = s1(r[i:i+1],v[i:i+1],weights,tau_s1)[0]
        # Direct K1 call makes exact scanner equality at a fresh trajectory.
        out['S2'][i] = (s1(r[i:i+1],v[i:i+1],weights,tau_s2)[0] if i==0 else
                        scan_transported(r,v,acc['sources'][i],weights,tau_s2))
        out['S3'][i] = np.maximum(out['S2'][i],memory_scan(r,v,p,i,tq[i]))
    return dict(**out, coverage=acc['coverage'],mean_variance=acc['mean_variance'],
                diagnostics=acc['diagnostics'])
