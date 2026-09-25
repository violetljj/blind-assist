"""Coarse SNR-tier anchor: real VL53L5CX zone validity (ZJUL5) vs simulated validity per tier.

Real side: ZJUL5 frames with a DELTAR mask; zones whose RealSense GT inside the zone
rectangle covers >80% of pixels with a 10-90% depth spread < 0.3 m; validity rate and
|L5 mean - GT median| by GT median distance. Simulated side: full-zone surface at a
uniform random range within each bin, rho ~ U[0.1, 0.9], validity = derive_readout
proxy (peak >= 5 sigma), 200 draws per bin and tier. Data are not redistributed.
"""
import glob
from dataclasses import replace
import h5py
import numpy as np
from cnh_route_sensor import SensorParameters, angular_rays, synthesize_response, derive_readout, H3

ROOT = 'E:/linnan/linnan/artifacts.local/datasets/zjul5/ZJUL5'
BINS = [(.3, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4)]
GAINS = {'SNR3': 250.18050517621333, 'SNR6': 606.1126027316143, 'SNR12': 1708.5668826655542, 'old-alley-4000': 4000.}


def real_curve():
    rows = []
    for path in glob.glob(f'{ROOT}/*/*.h5'):
        f = h5py.File(path, 'r')
        if 'mask' not in f:
            continue
        d, fr, m, h = f['depth'][:], f['fr'][:], f['mask'][:], f['hist_data'][:]
        for z in range(64):
            a, b, c, e = fr[z]
            y0, y1 = sorted((a, c))
            x0, x1 = sorted((b, e))
            patch = d[y0:y1, x0:x1]
            v = patch[(patch > .05) & (patch < 60)]
            if patch.size == 0 or v.size < 20:
                continue
            rows.append((np.median(v), np.percentile(v, 90)-np.percentile(v, 10), v.size/patch.size, m[z], h[z, 0]))
    R = np.array(rows, float)
    homog = (R[:, 1] < .3) & (R[:, 2] > .8)
    out = []
    for lo, hi in BINS:
        s = homog & (R[:, 0] >= lo) & (R[:, 0] < hi)
        ok = s & (R[:, 3] > 0)
        out.append(dict(n=int(s.sum()), valid=float(R[s, 3].mean()), median_abs_err=float(np.median(np.abs(R[ok, 4]-R[ok, 0])))))
    return out, int(homog.sum()), len(R)


def sim_curve(gain, rng):
    _, w = angular_rays(16)
    p = replace(SensorParameters(), signal_counts=gain)
    out = []
    for lo, hi in BINS:
        ok = []
        for _ in range(200):
            resp = synthesize_response(np.full((8, 8, 256), rng.uniform(lo, hi)), rng.uniform(.1, .9), 1., w,
                                       params=p, seed=int(rng.integers(1 << 31)))
            ok.append(derive_readout(resp, H3)['valid'][3, 3])
        out.append(float(np.mean(ok)))
    return out


if __name__ == '__main__':
    real, homog, total = real_curve()
    print(f'zones {total}, homogeneous {homog}')
    print('real', [round(r['valid'], 3) for r in real], [r['n'] for r in real], [round(r['median_abs_err'], 3) for r in real])
    rng = np.random.default_rng(0)
    for name, g in GAINS.items():
        sim = sim_curve(g, rng)
        rms = float(np.sqrt(np.mean((np.array(sim)-[r['valid'] for r in real])**2)))
        print(name, [round(x, 2) for x in sim], f'RMS {rms:.3f}')
