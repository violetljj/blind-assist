"""Heading uncertainty in the CNH three-state output (EXPLORE, consumed simulation Development).

Inject per-sequence head-to-travel query errors e_t = sigma*b + (sigma/2)*w_t (b, w_t ~ N(0,1),
common random numbers across sigma) into the stored passive observations of batches
98000/99000, and score the frozen M3 at the erroneous centre query and at +/-k rotated
queries (k = sigma). Mitigations are evaluated offline from these scores:
  none        alarm/clear from the centre query
  union-alarm alarm and clear from max over {centre, +k, -k}
  union-clear alarm from the centre; CLEAR only if all three are <= tau (else UNKNOWN)
Observations, noisy history, M3 (5 seeds), smoothing, alarm threshold, r3 gate and event
definitions are frozen. Numerics: fused FP32 CUDA projection + FP16 autocast M3; the exact
condition is recomputed through the same path (parity vs stored FP64 recorded).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_active_scan_dev as A  # noqa: E402

OUT = A.ROOT/'artifacts.local/work/cnh-heading-uncertainty-dev-20261007'
SIGMAS = (2., 4., 8.)
SEED = 2026100781


def conditions():
    out = [('exact', 0., 0.)]
    for s in SIGMAS:
        for tag, k in (('c', 0.), ('p', s), ('m', -s)):
            out.append((f's{int(s)}{tag}', s, k))
    return out


def errors(unit, config):
    rng = np.random.default_rng([SEED, unit, config])
    return rng.standard_normal(), rng.standard_normal(16)


def check_deadline():
    if time.time() > A.read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('heading-uncertainty wall budget reached')


class Runner:
    def __init__(self):
        from cnh_fused_projection import FusedProjector
        self.eng = A.Engine(); self.F = FusedProjector(self.eng.projector)

    def raw(self, z, nn, qq):
        """z[C,16,...], nn/qq[C,16,4,4] -> raw logits [C,13,2] (fused projection, AMP M3)."""
        from cnh_temporal_readout_model import prepare_voxels
        torch = self.eng.torch; C = len(z); vox = []
        for f in range(3, 16):
            ix = np.arange(max(0, f-7), f+1)
            m = (qq[:, f]@np.linalg.inv(nn[:, f]))[:, None]@nn[:, ix]
            vox.append(self.F(z[:, ix], m).half())
        v = torch.stack(vox, 1).reshape(-1, 3, 24, 17, 33); out = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for b in range(0, len(v), 130):
                x = prepare_voxels(v[b:b+130], self.eng.masks)
                out.append(torch.stack([n(x).float() for n in self.eng.nets]).mean(0))
        return torch.cat(out).float().cpu().numpy().reshape(C, 13, 2)


def run_unit(rn, unit):
    import cnh_cvr_pilot as CP
    dest = OUT/'units'/f'unit{unit}.npz'
    if dest.exists():
        return
    tick = time.monotonic(); obs, sc = A.stored(unit); C = len(obs['noisy_center'])
    pq = obs['public_query']; draws = [errors(unit, c) for c in range(C)]
    res = dict(unit=unit, mode=unit % 3)
    for name, sigma, k in conditions():
        check_deadline()
        q = np.repeat(pq[None], C, 0).copy(); err = np.zeros((C, 16), np.float32)
        if sigma:
            for c, (b, w) in enumerate(draws):
                e = sigma*b+(sigma/2)*w; err[c] = e
                for f in range(16):
                    q[c, f, :3, :3] = CP.rotation(float(e[f]+k), 'y')@pq[f, :3, :3]
        raw = np.empty((3, C, 13, 2), np.float32)
        for s, a in enumerate(A.ANGLES):
            ex = rn.eng.N.extrinsic(a)
            raw[s] = rn.raw(obs['z1'][s], obs['noisy_center']@ex, q@ex)
        res[f'{name}_raw'] = raw
        if k == 0. and sigma:
            res[f's{int(sigma)}_err'] = err
    res['exact_vs_stored_max_abs'] = float(np.abs(res['exact_raw']-sc['reference']).max())
    res['seconds'] = time.monotonic()-tick
    dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **res); os.replace(tmp, dest)
    print('unit', unit, round(res['seconds'], 1), 's exact-vs-stored', round(res['exact_vs_stored_max_abs'], 4), flush=True)


def freeze(hours):
    now = time.time()
    A.save(OUT/'PLAN.json', dict(task='CNH_HEADING_UNCERTAINTY_DEV_20261007', lane='EXPLORE consumed simulation Development',
        authorization='User "做" (2026-10-07) after Claude proposed heading uncertainty in three-state',
        question='How do realistic-magnitude head-to-travel errors degrade frozen M3 three-state results, and can a widened query (union-alarm) or uncertainty-aware CLEAR (union-clear) recover timely alarms, at what false-alarm / unknown cost?',
        units='98000-98047 + 99000-99095 stored passive S/L/R observations; 369 events, 1255 controls',
        error_model='per sequence e_t = sigma*b + (sigma/2)*w_t, b,w_t ~ N(0,1) (common draws across sigma, shared by S/L/R); sigma in 2/4/8 deg; applied as yaw rotation of the exact query',
        mitigations='none / union-alarm / union-clear with k = sigma; scores per query smoothed then max',
        numerics='fused FP32 CUDA projection (parity: max |dlogit| .019, 1 flip in 98280 vs stored) + FP16 autocast M3; exact recomputed via same path',
        frozen='observations, noisy history, M3 5 seeds, smoothing, alarm 0.8557642486787612, r3 gate, 20 tau, events/controls',
        budget_wall_hours=hours, started_unix=now, deadline_unix=now+hours*3600, source_sha256=A.sha(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['freeze', 'run'])
    ap.add_argument('--hours', type=float, default=2.); a = ap.parse_args()
    if a.stage == 'freeze':
        return freeze(a.hours)
    rn = Runner()
    for u in A.UNITS:
        run_unit(rn, u)


if __name__ == '__main__':
    main()
