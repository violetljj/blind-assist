"""Summarize the clearance rule-fix runs (Development, fast lane, descriptive only).

Rules on the saved per-band-slab outputs of cnh_clearance_subregion:
  eps      original: slab passes if >= 90% certified
  eps+occ  also stop at any occupied voxel in the slab
  window   stop at any occupied voxel or any hideable k x k uncertified block
           (k = 1 for the 25%-zone and 5 cm classes, 2 for 10 cm, 4 for 20 cm)
Each run is scored against its own reference class. Errors of the original rule are
attributed with the saved rule flags, the matching worst-placement run and the
sensor oracle (per-ray object id and incidence cosine, evaluator-only).
"""
import json
import sys
from pathlib import Path
import numpy as np
from cnh_clearance_subregion import SLABS, BANDS

GROUPS = ('HEAD', 'BODY')
CLASS_OF = {'fill25': ('ref', 1), 's05': ('ref_s05', 1), 's10': ('ref_s10', 2), 's20': ('ref_s20', 4)}


def load(run):
    units = sorted(run.glob('unit*.npz'), key=lambda p: int(p.stem[4:]))
    return [dict(np.load(p)) | {'unit': int(p.stem[4:])} for p in units]


def ref_class(run):
    name = run.name
    return next((c for c in CLASS_OF if f'-{c}-' in name), 'fill25')


def passes(u, rule, k):
    if rule == 'eps':
        return u['frac'] >= .9
    if rule == 'eps+occ':
        return (u['frac'] >= .9) & ~u['occ']
    return ~u['occ'] & ~u[f'hole{k}']


def clearance(ok):
    first = np.where(~ok.all(-1), (~ok).argmax(-1), ok.shape[-1])
    return .3+first*(SLABS[1]-SLABS[0])


def summarize(run, rule):
    cls = ref_class(run)
    ref_key, k = CLASS_OF[cls]
    rows, fc_units, pos_units, encounters = [], [], [], set()
    for u in load(run):
        m = u['main'].astype(bool)
        d = clearance(passes(u, rule, k))[m]
        z, ref, size = u['truth'][m], u[ref_key][m] & np.isfinite(u['truth'][m]), u['size'][m]
        fc = np.isfinite(z) & (z < d)
        who, cfg = u['who'][m], u['config'][m]
        for i, q, b in zip(*np.nonzero(ref)):
            encounters.add((u['unit'], int(cfg[i]), int(who[i, q, b])))
        fc_units.append(int((fc & ref).sum()))
        pos_units.append(int(ref.sum()))
        empty_slab = (SLABS[None, None, None]+.1 <= np.nan_to_num(z, nan=9.)[..., None])
        rows.append(dict(d=d, z=z, fc=fc, ref=ref, size=size, anyok=passes(u, rule, k)[m].any(-1),
                         false_occ=(u['occ'][m] & empty_slab)[..., SLABS < 1.5].sum(), empty_slabs=empty_slab[..., SLABS < 1.5].sum()))
    cat = {key: np.concatenate([r[key] for r in rows]) for key in ('d', 'z', 'fc', 'ref', 'size', 'anyok')}
    n_fc = int(sum(fc_units))
    out = dict(run=run.name, rule=rule, ref_class=cls, hole_k=k, units=len(rows), frames=int(len(cat['d'])),
               ref_positives=int(sum(pos_units)), ref_false_clear=n_fc, ref_encounters=len(encounters),
               ref_fc_per_unit=fc_units,
               zero_event_note=(f'0 observed; rule-of-three on {len(encounters)} object encounters gives <= '
                                f'{3/max(1, len(encounters)):.4f}, encounters are not independent') if n_fc == 0 else None,
               small_false_clear={s: [int((cat['fc'] & (cat['size'] == s)).sum()), int((np.isfinite(cat['z']) & (cat['size'] == s)).sum())]
                                  for s in ('tiny', 'realistic', 'wide', 'background')},
               false_occupied_slabs_below_1p5m=[int(sum(r['false_occ'] for r in rows)), int(sum(r['empty_slabs'] for r in rows))])
    empty = ~np.isfinite(cat['z'])
    bands = {}
    for g, name in enumerate(GROUPS):
        for b, band in enumerate(BANDS):
            e = empty[:, g::2, b].ravel()
            d = cat['d'][:, g::2, b].ravel()[e]
            bands[f'{name}/{band}'] = dict(empty=int(e.sum()), unknown=round(float(np.mean(d <= .3)), 3),
                                           D_ge_06=round(float(np.mean(d >= .6)), 3), D_ge_1=round(float(np.mean(d >= 1.)), 3),
                                           some_slab_supported=round(float(cat['anyok'][:, g::2, b].ravel()[e].mean()), 3))
    out['empty_bands'] = bands
    return out


def oracle_geometry(sensor, unit, obs_index, obj):
    """Object footprint over the 8-frame memory window: largest single-zone fill, the
    best zone's share of its rays in the most visible frame, mean incidence cosine
    (evaluator-only)."""
    with np.load(sensor/f'unit{unit}-mount-10-oracle.npz') as f:
        ids, cos = f['object_id'][max(0, obs_index-7):obs_index+1], f['raycos'][max(0, obs_index-7):obs_index+1]
    hit = ids == obj
    per_zone = hit.sum(-1)                                     # [T, 8, 8]
    total = per_zone.sum((1, 2))
    seen = total > 0
    if not seen.any():
        return dict(seen_frames=0)
    t = int(total.argmax())
    return dict(seen_frames=int(seen.sum()), max_zone_fraction=round(float(per_zone.max()/256), 3),
                best_zone_share_at_most_visible=round(float(per_zone[t].max()/total[t]), 2),
                mean_cos=round(float(cos[hit].mean()), 2), zones_touched=int((per_zone.max(0) > 0).sum()))


def attribute(base, worst, sensor, kind):
    """Original-rule errors in `base`, flagged by which fix removes them."""
    out = []
    wmap = {u['unit']: u for u in load(worst)}
    for u in load(base):
        in_main = u['main'].astype(bool)
        z = u['truth']
        err = np.isfinite(z) & (z < clearance(passes(u, 'eps', 1)))
        err &= u['ref'].astype(bool) if kind == 'ref' else (u['size'] == 'tiny')
        err &= in_main[:, None, None]
        for i, q, b in zip(*np.nonzero(err)):
            w = wmap[u['unit']]
            d_eps = clearance(passes(u, 'eps', 1))[i, q, b]
            before = SLABS < d_eps
            item = dict(unit=u['unit'], config=int(u['config'][i]), frame=int(i), box=int(q),
                        group=GROUPS[q % 2], band=BANDS[b], object=int(u['who'][i, q, b]),
                        size_class=str(u['size'][i, q, b]), truth_z=round(float(z[i, q, b]), 2), D=round(float(d_eps), 2),
                        occupied_voxel_passed=bool(u['occ'][i, q, b][before].any()),
                        fixed_by_occ_stop=bool(z[i, q, b] >= clearance(passes(u, 'eps+occ', 1))[i, q, b]),
                        fixed_by_strict=bool(z[i, q, b] >= clearance(passes(u, 'window', 1))[i, q, b]),
                        fixed_by_worst_placement=bool(z[i, q, b] >= clearance(passes(w, 'eps', 1))[i, q, b]),
                        fixed_by_all=bool(z[i, q, b] >= clearance(passes(w, 'window', 1))[i, q, b]))
            item |= oracle_geometry(sensor, u['unit'], int(u['obs_index'][i]), item['object'])
            out.append(item)
    return out


def main():
    root, sensor = Path(sys.argv[1]), Path(sys.argv[2])
    runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('r'))
    report = dict(summaries=[summarize(r, rule) for r in runs for rule in ('eps', 'eps+occ', 'window')])
    report['attribution'] = {
        'snr12_noisy_ref': attribute(root/'r0-centred-snr12-noisy', root/'r1-worst-fill25-snr12-noisy', sensor, 'ref'),
        'snr6_GT_ref': attribute(root/'r0-centred-snr6-GT', root/'r1-worst-fill25-snr6-GT', sensor, 'ref'),
        'snr6_noisy_tiny': attribute(root/'r0-centred-snr6-noisy', root/'r1-worst-fill25-snr6-noisy', sensor, 'tiny')}
    (root/'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    for s in report['summaries']:
        bands = ' '.join(f"{v['D_ge_1']:.2f}" for v in s['empty_bands'].values())
        print(f"{s['run']:30s} {s['rule']:8s} ref FC {s['ref_false_clear']}/{s['ref_positives']} (enc {s['ref_encounters']})"
              f"  tiny {s['small_false_clear']['tiny']}  D>=1m {bands}")
    for k, items in report['attribution'].items():
        print(k, len(items))
        for it in items:
            print('  ', it)


if __name__ == '__main__':
    main()
