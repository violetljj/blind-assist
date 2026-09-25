"""GPU (torch, float32) port of the Track A H3 readouts for development iteration.

Same definitions as the CPU implementation (cnh_track_a_readout, cnh_track_a_v12_readout,
cnh_scan_development.scan, cnh_track_a_scale_fast): B0, S1, S1cell, B1-R, S2, S2r4,
memory (S3 = max(S2, memory)). Float32 arithmetic is not bitwise equal to the CPU
float64 path; use only after a tolerance check against the CPU scorer.
"""
import numpy as np
import torch
import torch.nn.functional as F
from cnh_route_sensor import angular_rays
from cnh_track_a_readout import START, WIDTH, END, EDGE, BOXES, cell_points
from cnh_scan_development import WINDOWS

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float32
TAUS = (.25, .5, .75)

D64 = torch.float64
_orig = torch.tensor(cell_points(), dtype=D64, device=DEV)           # [1024,16,3] geometry kept in float64
_orig_r = _orig.norm(dim=-1)
_rays, _area = angular_rays(16)
_area = _area.reshape(64, 256)
_rays = torch.tensor(_rays.reshape(64, 256, 3), dtype=D64, device=DEV)
_area = torch.tensor(_area/_area.sum(-1, keepdims=True), dtype=D64, device=DEV)
_edges = torch.tensor(START+np.arange(17)*WIDTH, dtype=D64, device=DEV)
_boxes = torch.tensor(BOXES, dtype=DT, device=DEV)                     # [6,2,3]
_src = torch.arange(1024, device=DEV)[:, None].expand(1024, 16)


def T(x):
    return torch.as_tensor(np.asarray(x), dtype=DT, device=DEV)


def transform(points, t):
    """points [...,3] with t [4,4] or batched [B,4,4] -> [B,...,3]."""
    if t.dim() == 2:
        return points @ t[:3, :3].T + t[:3, 3]
    return torch.einsum('...k,bjk->b...j', points, t[:, :3, :3]) + t[:, None, None, :3, 3]


def indices(points):
    r = points.norm(dim=-1)
    z = points[..., 2]
    xy = torch.where(z[..., None] > 0, points[..., :2]/z[..., None].clamp_min(1e-12), torch.full_like(points[..., :2], float('inf')))
    valid = (z > 0) & (xy.abs() < EDGE).all(-1) & (r >= START) & (r < END)
    safe = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
    colrow = torch.clamp(torch.floor((safe+EDGE)/(2*EDGE)*8), 0, 7).long()
    bins = torch.clamp(torch.floor((r-START)/WIDTH), 0, 15).long()
    return (colrow[..., 1]*8+colrow[..., 0])*16+bins, valid


def query_weights(tq):
    """[B,4,4] T_Q_tof -> [B,6,64,16] float64; port of cnh_track_a_readout.query_weights."""
    D = torch.float64
    tq = tq.to(D)
    rays = torch.einsum('zrk,bjk->bzrj', _rays, tq[:, :3, :3])
    origin = tq[:, :3, 3]
    area, edges = _area, _edges
    out = []
    for box in torch.as_tensor(BOXES, dtype=D, device=DEV):
        lo, hi = box
        near = torch.zeros(rays.shape[:3], dtype=D, device=DEV)
        far = torch.full(rays.shape[:3], float('inf'), dtype=D, device=DEV)
        for ax in range(3):
            d = rays[..., ax]
            par = d.abs() < 1e-12
            o = origin[:, ax][:, None, None]
            safe = torch.where(par, torch.ones_like(d), d)
            a = torch.where(par, torch.full_like(d, -float('inf')), (lo[ax]-o)/safe)
            b = torch.where(par, torch.full_like(d, float('inf')), (hi[ax]-o)/safe)
            near = torch.maximum(near, torch.minimum(a, b))
            far = torch.minimum(far, torch.maximum(a, b))
            outside = ~((lo[ax] <= o) & (o <= hi[ax]))
            far = torch.where(par & outside, torch.full_like(far, -float('inf')), far)
        ov = (torch.minimum(far[..., None], edges[1:]) - torch.maximum(near[..., None], edges[:-1])).clamp_min(0)/WIDTH
        out.append(torch.einsum('zr,bzrk->bzk', area, ov))
    return torch.stack(out, 1).clamp(0, 1)


def boxsum(x, shape):
    """x [...,8,8,16] -> window sums [...,8-h+1,8-w+1,16-d+1]."""
    h, w, d = shape
    lead = x.shape[:-3]
    y = F.avg_pool3d(x.reshape(-1, 1, 8, 8, 16), shape, stride=1)*(h*w*d)
    return y.reshape(*lead, *y.shape[-3:])


def scan(total, var, support):
    """total,var [B,8,8,16]; support [B,6,8,8,16] bool -> [B,6] max windowed z (-50 empty)."""
    out = torch.full(support.shape[:2], -50., dtype=DT, device=DEV)
    for shape in WINDOWS:
        n = shape[0]*shape[1]*shape[2]
        z = boxsum(total, shape)/boxsum(var, shape).clamp_min(1e-9).sqrt()
        adm = boxsum(support.to(DT), shape) > n-.5
        zq = torch.where(adm, z[:, None].expand_as(adm), torch.full_like(adm, -float('inf'), dtype=DT))
        out = torch.maximum(out, zq.flatten(2).max(-1).values)
    return out.clamp_min(-50.)


def transport(cur_from_past, power):
    """Batched [B,4,4] -> dense A [B,1024(dest),1024(src)] with gain (r_past/r_now)^(2*power)."""
    pts = transform(_orig, cur_from_past)                      # [B,1024,16,3]
    dest, valid = indices(pts)
    radius = pts.norm(dim=-1)
    valid = valid & (radius > 0)
    gain = torch.where(radius > 0, _orig_r/radius.clamp_min(1e-12), torch.ones_like(radius))**(2*power)
    B = pts.shape[0]
    A = torch.zeros(B, 1024*1024, dtype=D64, device=DEV)
    flat = dest*1024+_src
    A.scatter_add_(1, torch.where(valid, flat, torch.zeros_like(flat)).reshape(B, -1),
                   torch.where(valid, gain/16, torch.zeros_like(gain)).reshape(B, -1))
    return A.reshape(B, 1024, 1024)


def coverage(cur_from_past):
    _, covered = indices(transform(_orig, torch.linalg.inv(cur_from_past)))
    return covered.to(D64).mean(-1)                              # [B,1024]


def sequence_readouts(hist, ambient, bias, poses, tq, noisy, with_r4=True):
    """One trajectory (numpy in) -> dict of [N,6] numpy arrays per arm and tau."""
    r = T(hist)-T(bias)
    v = 16*T(ambient)[..., None]+T(bias).clamp_min(0)
    n = len(r)
    tq = torch.as_tensor(np.asarray(tq), dtype=D64, device=DEV)
    w64 = query_weights(tq)                                      # [n,6,64,16] float64
    w = w64.to(DT)
    out = {'B0': torch.einsum('nzb,nqzb->nq', T(hist).double().reshape(n, 64, 16), w64)}
    r64 = (torch.as_tensor(np.asarray(hist), dtype=torch.float64, device=DEV)-torch.as_tensor(np.asarray(bias), dtype=torch.float64, device=DEV)).reshape(n, 1024)
    sup = {tau: (w64 >= tau).reshape(n, 6, 8, 8, 16) for tau in TAUS}
    rr, vv = r.reshape(n, 8, 8, 16), v.reshape(n, 8, 8, 16)
    zc = (r/v.clamp_min(1e-9).sqrt()).reshape(n, 1, 64, 16)
    for tau in TAUS:
        out[f'S1|{tau}'] = scan(rr, vv, sup[tau])
        out[f'S1cell|{tau}'] = torch.where(w64 >= tau, zc.expand(n, 6, 64, 16), torch.full_like(w, -float('inf'))).flatten(2).max(-1).values.clamp_min(-50.)
    pairs = [(i, j) for i in range(1, n) for j in range(max(0, i-3), i)]
    for motion, p in (('GT', poses), ('noisy', noisy)):
        p = torch.as_tensor(np.asarray(p), dtype=D64, device=DEV)
        rel = torch.stack([torch.linalg.inv(p[i]) @ p[j] for i, j in pairs])
        variants = [(motion, 1)]+([('r4', -2)] if with_r4 and motion == 'noisy' else [])
        for name, power in variants:
            A = transport(rel, power)                           # [P,1024,1024]
            total = r.reshape(n, 1024).clone()
            win_var = {shape: boxsum(vv, shape).clone() for shape in WINDOWS}
            cov = torch.ones(n, 1024, dtype=torch.float64, device=DEV)
            total64 = r64.clone()
            if power == 1:
                covs = coverage(rel)
            for k, (i, j) in enumerate(pairs):
                Ak = A[k].to(DT)
                total[i] += Ak @ r[j].reshape(1024)
                if power == 1:
                    total64[i] += A[k] @ r64[j]
                    cov[i] += covs[k]
                At = Ak.T.reshape(1024, 8, 8, 16)            # per source: dest image
                for shape in WINDOWS:
                    c = boxsum(At, shape)                       # [1024(src),...]
                    win_var[shape][i] += torch.einsum('s,sxyz->xyz', v[j].reshape(1024), c*c)
            tt = total.reshape(n, 8, 8, 16)
            key = f'S2/{motion}' if name == motion else 'S2r4/noisy'
            for tau in TAUS:
                res = torch.full((n, 6), -50., dtype=DT, device=DEV)
                for shape in WINDOWS:
                    nn = shape[0]*shape[1]*shape[2]
                    z = boxsum(tt, shape)/win_var[shape].clamp_min(1e-9).sqrt()
                    adm = boxsum(sup[tau].to(DT), shape) > nn-.5
                    zq = torch.where(adm, z[:, None].expand_as(adm), torch.full_like(adm, -float('inf'), dtype=DT))
                    res = torch.maximum(res, zq.flatten(2).max(-1).values)
                res[0] = out[f'S1|{tau}'][0]
                out[f'{key}|{tau}'] = res.clamp_min(-50.)
            if power == 1:
                mean = total64/cov
                out[f'B1-R/{motion}'] = torch.einsum('nc,nqc->nq', mean, w64.reshape(n, 6, 1024))
        out[f"memory/{motion}"] = memory_batched(r, v, p, tq)
    return {k: val.double().cpu().numpy() for k, val in out.items()}


def memory(r, v, p, tq):
    """Port of memory_scan (k=8): outside-FOV in-box past evidence, 0.2 m voxels."""
    n = len(r)
    res = torch.full((n, 6), -50., dtype=DT, device=DEV)
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    for i in range(n):
        js = list(range(max(0, i-7), i+1))
        rel = torch.stack([torch.linalg.inv(p[i]) @ p[j] for j in js])
        cur = transform(_orig, rel)                                   # [J,1024,16,3]
        radius = cur.norm(dim=-1)
        z = cur[..., 2]
        tan = cur[..., :2]/z[..., None].clamp_min(1e-12)
        inside_fov = (z > 0) & (tan.abs() < EDGE).all(-1)
        pq = transform(cur, tq[i])
        gain = (torch.where(radius > 0, _orig_r/radius.clamp_min(1e-12), torch.ones_like(radius))**2)
        gain[torch.tensor(js, device=DEV) == i] = 1.
        vox = torch.floor(pq/.2).long()
        jidx = torch.arange(len(js), device=DEV)[:, None, None].expand(len(js), 1024, 16)
        src = _src[None].expand(len(js), 1024, 16)
        for q, (lo, hi) in enumerate(_boxes):
            keep = ~inside_fov & (radius > 0) & (pq >= lo).all(-1) & (pq <= hi).all(-1)
            if not keep.any():
                continue
            key_v = vox[keep]
            jj, ss, gg = jidx[keep], src[keep], (gain[keep]/16).to(DT)
            # coefficient per (j, voxel, source) = sum of subpoint gains, then per (j, voxel) sums
            vk, vinv = torch.unique(key_v, dim=0, return_inverse=True)
            comp = (jj*len(vk)+vinv)*1024+ss
            ck, cinv = torch.unique(comp, return_inverse=True)
            coef = torch.zeros(len(ck), dtype=DT, device=DEV).scatter_add_(0, cinv, gg)
            cj = ck//(len(vk)*1024)
            cv = (ck//1024) % len(vk)
            cs = ck % 1024
            js_t = torch.tensor(js, device=DEV)[cj]
            tot = torch.zeros(len(vk), dtype=DT, device=DEV).scatter_add_(0, cv, coef*rf[js_t, cs])
            var = torch.zeros(len(vk), dtype=DT, device=DEV).scatter_add_(0, cv, coef*coef*vf[js_t, cs])
            res[i, q] = torch.maximum(res[i, q], (tot/var.clamp_min(1e-9).sqrt()).max())
    return res


def memory_batched(r, v, p, tq):
    """Vectorized memory scan for a whole trajectory (same definition as memory())."""
    n = len(r)
    pairs = [(i, j) for i in range(n) for j in range(max(0, i-7), i+1)]
    I = torch.tensor([a for a, _ in pairs], device=DEV)
    J = torch.tensor([b for _, b in pairs], device=DEV)
    rel = torch.linalg.inv(p[I]) @ p[J]                                  # [K,4,4]
    cur = transform(_orig, rel)                                          # [K,1024,16,3]
    radius = cur.norm(dim=-1)
    z = cur[..., 2]
    tan = cur[..., :2]/z[..., None].clamp_min(1e-12)
    outside = ~((z > 0) & (tan.abs() < EDGE).all(-1))
    tqi = tq[I]
    pq = torch.einsum('kstj,kij->ksti', cur, tqi[:, :3, :3]) + tqi[:, None, None, :3, 3]
    gain = torch.where(radius > 0, _orig_r/radius.clamp_min(1e-12), torch.ones_like(radius))**2
    gain = torch.where((I == J)[:, None, None], torch.ones_like(gain), gain)
    vox = torch.floor(pq/.2).long()
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    rows = []
    for q, (lo, hi) in enumerate(_boxes.to(D64)):
        keep = outside & (radius > 0) & (pq >= lo).all(-1) & (pq <= hi).all(-1)
        k, s, _ = keep.nonzero(as_tuple=True)
        if len(k):
            vx = vox[keep]
            rows.append((torch.stack([I[k], torch.full_like(k, q), vx[:, 0], vx[:, 1], vx[:, 2], J[k], s], 1),
                         (gain[keep]/16).to(DT)))
    res = torch.full((n, 6), -50., dtype=DT, device=DEV)
    if not rows:
        return res
    keys = torch.cat([a for a, _ in rows])
    gains = torch.cat([b for _, b in rows])
    k1, inv1 = torch.unique(keys, dim=0, return_inverse=True)            # (i,q,voxel,j,src)
    coef = torch.zeros(len(k1), dtype=DT, device=DEV).scatter_add_(0, inv1, gains)
    contrib_t = coef*rf[k1[:, 5], k1[:, 6]]
    contrib_v = coef*coef*vf[k1[:, 5], k1[:, 6]]
    k2, inv2 = torch.unique(k1[:, :5], dim=0, return_inverse=True)     # (i,q,voxel)
    tot = torch.zeros(len(k2), dtype=DT, device=DEV).scatter_add_(0, inv2, contrib_t)
    var = torch.zeros(len(k2), dtype=DT, device=DEV).scatter_add_(0, inv2, contrib_v)
    zv = tot/var.clamp_min(1e-9).sqrt()
    flat = k2[:, 0]*6+k2[:, 1]
    best = torch.full((n*6,), -float('inf'), dtype=DT, device=DEV).scatter_reduce_(0, flat, zv, 'amax')
    return torch.maximum(res, best.reshape(n, 6))
