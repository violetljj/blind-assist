"""Summarize cnh_clearance_subregion outputs (Development, descriptive only)."""
import json
import sys
from pathlib import Path
import numpy as np
from cnh_clearance_subregion import clearance_from_fractions, SLABS, BANDS

GROUPS = ('HEAD', 'BODY')                      # query q = 2*lateral + group


def load(root):
    units = sorted(root.glob('unit*.npz'), key=lambda p: int(p.stem[4:]))
    return [dict(np.load(p)) | {'unit': int(p.stem[4:])} for p in units]


def per_unit(u):
    m = u['main'].astype(bool)
    frac, z, ref, size = u['frac'][m], u['truth'][m], u['ref'][m], u['size'][m]
    d = clearance_from_fractions(frac)
    occ = np.isfinite(z)
    fc = occ & (z < d)
    return dict(d=d, z=z, occ=occ, ref=ref & occ, fc=fc, size=size, frac=frac, frames=int(m.sum()))


def boot_upper(num, den, b=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (b, len(num)))
    rates = num[idx].sum(1)/np.maximum(den[idx].sum(1), 1)
    return float(np.quantile(rates, .95))


def summarize(root):
    rows = [per_unit(u) for u in load(root)]
    cat = {k: np.concatenate([r[k] for r in rows]) for k in ('d', 'z', 'occ', 'ref', 'fc', 'size', 'frac')}
    out = dict(units=len(rows), frames=sum(r['frames'] for r in rows))
    ref_pos, ref_fc = np.array([r['ref'].sum() for r in rows]), np.array([(r['ref'] & r['fc']).sum() for r in rows])
    out['ref'] = dict(positives=int(ref_pos.sum()), false_clear=int(ref_fc.sum()),
                      FCR=float(ref_fc.sum()/max(1, ref_pos.sum())), FCR_unit_boot_upper95=boot_upper(ref_fc, ref_pos))
    out['all_obstacles'] = {s: dict(positives=int((cat['occ'] & (cat['size'] == s)).sum()),
                                    false_clear=int((cat['fc'] & (cat['size'] == s)).sum()))
                            for s in ('tiny', 'realistic', 'wide', 'background')}
    empty = ~cat['occ']
    table = {}
    for g, name in enumerate(GROUPS):
        for b, band in enumerate(BANDS):
            e = empty[:, g::2, b].ravel()
            d = cat['d'][:, g::2, b].ravel()[e]
            f = cat['frac'][:, g::2, b].reshape(-1, len(SLABS))[e]
            table[f'{name}/{band}'] = dict(empty=int(e.sum()), unknown=float(np.mean(d <= .3)),
                                           median_D=float(np.median(d)), p75_D=float(np.quantile(d, .75)),
                                           D_ge_1m=float(np.mean(d >= 1.)),
                                           certified_at=dict((f'{z:.1f}', float(f[:, i].mean())) for i, z in
                                                             ((0, .3), (2, .5), (7, 1.), (17, 2.), (26, 2.9))))
    out['empty_bands'] = table
    # Whole-box single number: min over bands vs per-band reporting, on fully empty boxes.
    for g, name in enumerate(GROUPS):
        e = empty[:, g::2].all(-1).ravel()
        dmin = cat['d'][:, g::2].min(-1).ravel()[e]
        dany = (cat['d'][:, g::2] > .3).any(-1).ravel()[e]
        out[f'{name}_whole_box'] = dict(empty_boxes=int(e.sum()), median_min_band_D=float(np.median(dmin)),
                                        whole_box_unknown=float(np.mean(dmin <= .3)),
                                        some_band_known=float(np.mean(dany)))
    occ_ref = cat['ref']
    out['ref_slack_median_m'] = float(np.median((cat['z']-cat['d'])[occ_ref])) if occ_ref.any() else None
    out['ref_known_fraction'] = float(np.mean(cat['d'][occ_ref] > .3)) if occ_ref.any() else None
    return out


if __name__ == '__main__':
    for root in sys.argv[1:]:
        print(root)
        print(json.dumps(summarize(Path(root)), indent=1, ensure_ascii=False))
