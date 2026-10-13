"""Where does the RGB branch lose thin poles? Stage-wise diagnostic on SANPO human pole masks (EXPLORE).

Six human-annotated pole instances x 5 keyframes (round4 sanpo_provenance index). Reference = SANPO CRES
stereo depth (an estimate, not independent truth). Frozen v1/v2/v3 are untouched; nothing is adopted.
Stages, each judged on the human mask:
  0 reference : pole solid angle (mask px / fx^2) vs the RGB alert threshold 150 Atom px = 150/425^2 sr;
                reference contrast = pole vs surrounding ring (median CRES depth).
  1 resolution: DAV2-S input at (a) 518 short side on the SANPO frame and (b) the Atom angular resolution
                (model-input px/rad equal to Atom fx * 518/480).
  2 model     : with an ORACLE affine fit (all valid CRES pixels), fraction of mask pixels within 20 % of
                CRES depth, and whether the pole median is nearer than the ring by >= half the reference gap.
  3 anchor    : the v3 anchor chain on an 8x8 zone emulation (zone median CRES depth -> fit_affine ->
                robust replacement); same pole metrics.
  4 threshold : correct-depth pole solid angle (pixels within 20 %) vs the 150-px-equivalent threshold.
Usage: sanpo_thin_loss_dev_20261013.py <out_dir>
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import ROOT
import real_rgb_small_model_dev_20261012 as S
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

INDEX = ROOT / 'artifacts.local/work/data-potential-dev-20261013/round4/sanpo_provenance/summary.json'
DATA = ROOT / 'artifacts.local/datasets/sanpo-real-coverage-20260927'
FX_ATOM, MIN_PX = 425.5, 150
OMEGA_THR = MIN_PX / FX_ATOM ** 2


def depth(path):
    raw = gzip.decompress(path.read_bytes())
    h, w = map(int, struct.unpack('<ee', raw[:4]))
    z = np.frombuffer(raw[4:], '<f2').reshape(h, w).astype(np.float32)
    return np.where(np.isfinite(z) & (z > 0), z, np.nan)


def pole_stats(zfit, zref, mask, ring, px_sr):
    ok = mask & np.isfinite(zref) & np.isfinite(zfit)
    if ok.sum() < 20:
        return None
    err = np.abs(zfit[ok] - zref[ok]) / zref[ok]
    good = err <= .20
    zp, zb = np.nanmedian(zfit[ok]), np.nanmedian(np.where(ring, zfit, np.nan))
    rp, rb = np.nanmedian(zref[ok]), np.nanmedian(np.where(ring, zref, np.nan))
    gap_ref = 1 / rp - 1 / rb
    separated = bool(np.isfinite(zb) and gap_ref > 0 and (1 / zp - 1 / zb) >= .5 * gap_ref)
    return dict(within20=float(good.mean()), median_rel_err=float(np.median(err)), pole_z=float(zp), separated=separated,
                omega_correct=float(good.sum() * px_sr), above_threshold=bool(good.sum() * px_sr >= OMEGA_THR))


def fit_oracle(disp, zref):
    m = np.isfinite(zref) & (zref > .3) & (zref < 30)
    x, y = 1 / zref[m], disp[m]
    a, b = np.polyfit(x, y, 1)
    return (float(a), float(b)) if a > 0 else None


def fit_zones(disp, zref):
    h, w = disp.shape
    dz, inv = np.full(64, np.nan), np.full(64, np.nan)
    for r in range(8):
        for c in range(8):
            sl = (slice(r * h // 8, (r + 1) * h // 8), slice(c * w // 8, (c + 1) * w // 8))
            zr = zref[sl]
            if np.isfinite(zr).mean() < .5:
                continue
            d = np.nanmedian(zr)
            if not (.3 < d < 4.):        # ToF range used by the frozen anchor
                continue
            dz[r * 8 + c] = np.median(disp[sl]); inv[r * 8 + c] = 1 / d
    fit = S.fit_affine(dz, inv); src = 'tof' if fit else None
    if not fit_quality(dz, inv, fit)['accepted']:
        rb = robust_affine(dz, inv)['fit']
        if rb is not None:
            fit, src = rb, 'robust'
    return fit, src, int(np.isfinite(inv).sum())


def to_depth(disp, fit):
    a, b = fit
    inv = (disp - b) / a
    return np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.nan)


def main(out):
    import torch
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    groups = json.loads(INDEX.read_text(encoding='utf8'))['five_anchor_human_pole_mask_groups']
    model, _ = S.load_dav2_small(); rows = []; start = time.perf_counter()
    for gi, g in enumerate(groups):
        base = DATA / g['session'] / g['camera'] / 'left'
        cam = json.loads((DATA / g['session'] / 'description.json').read_text(encoding='utf8'))['session_camera_details'][0]['left_camera_params']
        for f in g['frames']:
            zref = depth(base / 'depth_maps' / f'{f:06d}.float16.gz'); h, w = zref.shape
            fx = cam['fx'] * w / cam['image_width']; px_sr = 1 / fx ** 2
            m = np.asarray(Image.open(base / 'segmentation_masks' / f'{f:06d}.png'))
            inst = (m[..., 1].astype(int) << 8) | m[..., 2]
            full = (m[..., 0] == g['semantic_class']) & (inst == g['instance'])
            mask = np.asarray(Image.fromarray(full).resize((w, h), Image.Resampling.NEAREST), bool)
            ring = ndimage.binary_dilation(mask, iterations=max(3, int(.02 * w))) & ~ndimage.binary_dilation(mask, iterations=2)
            ring &= ~np.isin(np.asarray(Image.fromarray(m[..., 0]).resize((w, h), Image.Resampling.NEAREST)), [g['semantic_class']])
            rgb = np.asarray(Image.open(base / 'video_frames' / f'{f:06d}.png').convert('RGB'))
            zr_pole = np.nanmedian(np.where(mask, zref, np.nan)); zr_ring = np.nanmedian(np.where(ring, zref, np.nan))
            row = dict(group=gi, session=g['session'][:8], instance=g['instance'], frame=f, mask_px=int(mask.sum()),
                       ref_valid_frac=float(np.isfinite(zref[mask]).mean()), ref_pole_z=float(zr_pole), ref_ring_z=float(zr_ring),
                       omega_mask=float(mask.sum() * px_sr), omega_thr=OMEGA_THR)
            atom_side = int(round(rgb.shape[0] * (FX_ATOM * 518 / 480) / cam['fx']))
            for res, side in (('sanpo518', 518), ('atomres', atom_side)):
                with torch.inference_mode():
                    disp = model.infer_image(np.ascontiguousarray(rgb[:, :, ::-1]), side)
                disp = np.asarray(Image.fromarray(disp.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))
                fo = fit_oracle(disp, zref)
                row[f'{res}_oracle'] = pole_stats(to_depth(disp, fo), zref, mask, ring, px_sr) if fo else None
                fz, src, nz = fit_zones(disp, zref)
                row[f'{res}_zone'] = dict(pole_stats(to_depth(disp, fz), zref, mask, ring, px_sr) or {}, src=src, zones=nz) if fz else dict(src=None, zones=nz)
                row[f'{res}_input_side'] = side
            rows.append(row)
            print(gi, f, f"ref pole {zr_pole:.2f} ring {zr_ring:.2f} omega {row['omega_mask']/OMEGA_THR:.2f}x thr | "
                  + ' | '.join(f"{k}: w20 {row[k]['within20']:.2f} sep {row[k]['separated']} thr {row[k]['above_threshold']}" if row.get(k) and 'within20' in row[k] else f"{k}: -"
                               for k in ('sanpo518_oracle', 'atomres_oracle', 'atomres_zone')), f'{time.perf_counter()-start:.0f}s', flush=True)
    (out / 'stages.json').write_text(json.dumps(dict(evidence='consumed SANPO train human masks; CRES stereo reference', rows=rows), indent=1), encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1])
