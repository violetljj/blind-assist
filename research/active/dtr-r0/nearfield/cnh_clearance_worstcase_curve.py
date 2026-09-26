"""Worst-placement certifiable distance vs reference size (Development, fast lane, analytic).

For each SNR tier and reference class (square side in metres, or 25% of a zone), the
worst-placement reference z per H3 bin is z = counts/sqrt(v) with the median public
noise v of the calib units; the certifiable distance is the far edge of the last bin
reachable from 0.3 m with every bin z >= kappa_cap. Sensor model known gain; no truth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_clearance as cc
from cnh_track_a_readout import START, WIDTH
import cnh_track_a_scale_evaluate as se


def certifiable_m(z, kappa):
    ok = z[1:] >= kappa
    n = len(ok) if ok.all() else int(ok.argmin())
    return round(START+(1+n)*WIDTH, 2) if n else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--calib', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kappa-cap', type=float, default=6.)
    a = p.parse_args()
    sizes = [None, .03, .05, .075, .10, .15, .20, .30]
    out = dict(kappa_cap=a.kappa_cap, bins_m=[round(START+(b+1)*WIDTH, 2) for b in range(10)], tiers={})
    unit_z = {(s, pool): cc.reference_worst(1., .5, side_m=s, pool=pool)[0] for s in sizes for pool in (True, False)}
    centred = cc.reference_counts(1., .5, .25)
    for snr in (3, 6, 12):
        si = (3, 6, 12).index(snr)
        gain = json.loads((a.sensor/f'unit{a.calib[0]:02d}-mount-10.json').read_text(encoding='utf-8'))['parameters'][str(snr)]['signal_counts']
        recs = [r for u in a.calib for r in se.unit_records(a.geometry, a.sensor, u, -10, si)[1]]
        hist = np.concatenate([r['hist'] for r in recs])
        amb = np.concatenate([r['ambient'] for r in recs])
        v = float(np.median(16*amb[..., None]+np.maximum(np.median(hist, 0), 0)))
        tier = dict(gain=gain, median_v=v, centred_fill25=dict(
            z=np.round(gain*centred[0]/np.sqrt(centred[1]*v), 2)[:10].tolist(),
            certifiable_m=certifiable_m(gain*centred[0]/np.sqrt(centred[1]*v), a.kappa_cap)))
        for (s, pool), c in unit_z.items():
            z = gain*c/np.sqrt(v)
            tier[f"{'fill25' if s is None else f'side{s:.3f}'}_{'pool2x2' if pool else '1x1'}"] = dict(
                z=np.round(z, 2)[:10].tolist(), certifiable_m=certifiable_m(z, a.kappa_cap))
        out['tiers'][str(snr)] = tier
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding='utf-8')
    for snr, t in out['tiers'].items():
        print(f'SNR{snr} v={t["median_v"]:.1f}  centred fill25: {t["centred_fill25"]["certifiable_m"]}')
        for k, r in t.items():
            if isinstance(r, dict) and k.endswith('pool2x2'):
                print(f'  {k:22s} certifiable {r["certifiable_m"]}  z/bin {r["z"][1:7]}')


if __name__ == '__main__':
    main()
