"""Batched variant of cnh_track_a_gpu_readout.sequence_readouts (same definitions).

All history pairs of a trajectory are processed at once: transported residual sums
via batched matmul + index_add, and the S2 window variance via one pooling call per
window shape over [pairs*1024] per-source destination images. Summation order differs
from the per-pair loop (float32 rounding only); validate against the CPU scorer.
"""
import numpy as np
import torch
import torch.nn.functional as F
from cnh_scan_development import WINDOWS
import cnh_track_a_gpu_readout as g
from cnh_track_a_gpu_readout import DEV, DT, D64, TAUS, query_weights, boxsum, scan, transport, coverage, memory_batched, T


def sequence_readouts(hist, ambient, bias, poses, tq, noisy, with_r4=True):
    r = T(hist)-T(bias)
    v = 16*T(ambient)[..., None]+T(bias).clamp_min(0)
    n = len(r)
    tq = torch.as_tensor(np.asarray(tq), dtype=D64, device=DEV)
    w64 = query_weights(tq)
    w = w64.to(DT)
    out = {'B0': torch.einsum('nzb,nqzb->nq', T(hist).double().reshape(n, 64, 16), w64)}
    r64 = (torch.as_tensor(np.asarray(hist), dtype=D64, device=DEV)-torch.as_tensor(np.asarray(bias), dtype=D64, device=DEV)).reshape(n, 1024)
    sup = {tau: (w64 >= tau).reshape(n, 6, 8, 8, 16) for tau in TAUS}
    rr, vv = r.reshape(n, 8, 8, 16), v.reshape(n, 8, 8, 16)
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    zc = (r/v.clamp_min(1e-9).sqrt()).reshape(n, 1, 64, 16)
    for tau in TAUS:
        out[f'S1|{tau}'] = scan(rr, vv, sup[tau])
        out[f'S1cell|{tau}'] = torch.where(w64 >= tau, zc.expand(n, 6, 64, 16), torch.full_like(w, -float('inf'))).flatten(2).max(-1).values.clamp_min(-50.)
    pairs = [(i, j) for i in range(1, n) for j in range(max(0, i-3), i)]
    I = torch.tensor([a for a, _ in pairs], device=DEV)
    J = torch.tensor([b for _, b in pairs], device=DEV)
    P = len(pairs)
    for motion, p in (('GT', poses), ('noisy', noisy)):
        p = torch.as_tensor(np.asarray(p), dtype=D64, device=DEV)
        rel = torch.linalg.inv(p[I]) @ p[J]
        variants = [(motion, 1)]+([('r4', -2)] if with_r4 and motion == 'noisy' else [])
        for name, power in variants:
            A = transport(rel, power)                                     # [P,1024,1024] float64
            Af = A.to(DT)
            total = rf.clone().index_add_(0, I, torch.bmm(Af, rf[J].unsqueeze(-1)).squeeze(-1))
            if power == 1:
                total64 = r64.clone().index_add_(0, I, torch.bmm(A, r64[J].unsqueeze(-1)).squeeze(-1))
                cov = torch.ones(n, 1024, dtype=D64, device=DEV).index_add_(0, I, coverage(rel))
            At = Af.transpose(1, 2).reshape(P*1024, 1, 8, 8, 16)            # per (pair, source): dest image
            vJ = vf[J]                                                       # [P,1024]
            tt = total.reshape(n, 8, 8, 16)
            key = f'S2/{motion}' if name == motion else 'S2r4/noisy'
            win_var = {}
            for shape in WINDOWS:
                c = F.avg_pool3d(At, shape, stride=1)*(shape[0]*shape[1]*shape[2])
                c = c.reshape(P, 1024, *c.shape[-3:])
                contrib = torch.einsum('ps,psxyz->pxyz', vJ, c*c)
                win_var[shape] = boxsum(vv, shape).clone().index_add_(0, I, contrib)
                del c
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
                out[f'B1-R/{motion}'] = torch.einsum('nc,nqc->nq', total64/cov, w64.reshape(n, 6, 1024))
            del A, Af, At
        out[f'memory/{motion}'] = memory_batched(r, v, p, tq)
    return {k: val.double().cpu().numpy() for k, val in out.items()}
