"""Calibrate the RGB box readout on labelled real ARKitScenes frames (EXPLORE).

Data: sync-rgb-tof-dataset-v1 (12 visits, 384 frames; train/cal/eval roles kept). RGB 256x192 with
known K; reference depth + observed mask as truth; the phone LiDAR native depth emulates an 8x8 ToF
(45 deg square FoV on the optical axis; zone median optical depth, 0.3-4 m, >=50% valid pixels,
2% multiplicative noise) that anchors DAV2-Small relative depth exactly as on the Atom
(real_rgb_small_model_dev_20261012.fit_affine; frames with < 6 zones are reported unanchored).
Boxes are camera-frame (no gravity: truth and prediction share the geometry):
HEAD y -0.20..0.42, BODY 0.42..0.90 (down), x +/-0.30, z 0.30..2.00 m.
Score = predicted solid angle inside a box (pixels / (fx fy)); truth POS if reference solid angle
>= POS_SR, FREE if no reference point inside and the box's projected image region is >= 90% observed.
The Atom readout's 150 px at fx,fy = 427,424 equals 8.28e-4 sr. Thin structures are absent at this
resolution, so this calibrates general obstacle/false-alarm behaviour only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import ROOT
import real_rgb_small_model_dev_20261012 as S

MANIFEST = ROOT / 'artifacts.local/work/sync-rgb-tof-dataset-v1-dev-20261011/dataset_manifest_v1.json'
OUT = ROOT / 'artifacts.local/work/rgb-readout-calibration-20261012'
BOXES = {'HEAD': (-.20, .42), 'BODY': (.42, .90)}
ATOM_SR = 150 / (427.2 * 423.8)
POS_SR = 2e-4
EDGE = np.tan(np.pi / 8)


def emulate_tof(native_m, K, rng):
    h, w = native_m.shape
    vv, uu = np.mgrid[0:h, 0:w]
    tx, ty = (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]
    col = np.floor((tx + EDGE) / (2 * EDGE) * 8).astype(int); row = np.floor((ty + EDGE) / (2 * EDGE) * 8).astype(int)
    inside = (col >= 0) & (col < 8) & (row >= 0) & (row < 8)
    zones = []
    for r in range(8):
        for c in range(8):
            m = inside & (row == r) & (col == c)
            if m.sum() < 4:
                zones.append((None, None)); continue
            v = native_m[m]; ok = v > 0
            if ok.mean() < .5:
                zones.append((None, m)); continue
            z = float(np.median(v[ok]))
            zones.append((z * rng.normal(1, .02) if .3 < z < 4 else None, m))
    return zones


def main():
    import torch
    frames = json.loads(MANIFEST.read_text(encoding='utf8'))['frames']
    model, _ = S.load_dav2_small()
    rng = np.random.default_rng(20261012)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in frames:
        rgb = np.array(Image.open(f['rgb']['path']).convert('RGB'))
        K = np.array(f['color_K'])
        native = np.array(Image.open(f['native_depth']['path'])).astype(np.float32) / 1000
        ref = np.load(f['reference']['path']); gt = ref['depth'].astype(np.float32); observed = ref['observed'].astype(bool)
        disp = model.infer_image(np.ascontiguousarray(rgb[:, :, ::-1]), 518)
        zones = emulate_tof(native, K, rng)
        dz = np.array([np.median(disp[m]) if m is not None and z is not None else np.nan for z, m in zones])
        inv_z = np.array([1 / z if z is not None else np.nan for z, _ in zones])
        fit = S.fit_affine(dz, inv_z)
        h, w = disp.shape
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
        sr = 1 / (K[0, 0] * K[1, 1])
        row = dict(source_id=f['source_id'], visit=f['visit_id'], role=f['role'], anchored=fit is not None, zones=int(np.isfinite(inv_z).sum()))
        gt_ok = observed & (gt > 0)
        P_gt = dirs * gt[..., None]
        if fit is not None:
            a, b = fit; inv = (disp - b) / a
            zp = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf); P = dirs * zp[..., None]
        for name, (lo, hi) in BOXES.items():
            def inbox(p):
                return (np.abs(p[..., 0]) <= .30) & (p[..., 1] >= lo) & (p[..., 1] <= hi) & (p[..., 2] >= .30) & (p[..., 2] <= 2.0)
            g = inbox(P_gt) & gt_ok
            # image region the box can project onto: pixels whose ray crosses the box within z range
            region = np.zeros((h, w), bool)
            for zz in np.linspace(.3, 2.0, 18):
                region |= inbox(dirs * zz)
            label = 'POS' if g.sum() * sr >= POS_SR else ('FREE' if g.sum() == 0 and region.any() and gt_ok[region].mean() >= .9 else 'UNKNOWN')
            score = float(inbox(P).sum() * sr) if fit is not None else None
            row[name] = dict(label=label, gt_sr=float(g.sum() * sr), pred_sr=score)
        rows.append(row)
    (OUT / 'readout_rows.json').write_text(json.dumps(rows, indent=1), encoding='utf8')
    # ROC on cal and eval (anchored frames only), per box and pooled
    report = dict(atom_threshold_sr=ATOM_SR, pos_sr=POS_SR, frames=len(rows), anchored=int(sum(r['anchored'] for r in rows)))
    for role in ('cal', 'eval', 'train'):
        sub = [r for r in rows if r['role'] == role and r['anchored']]
        pos = np.array([r[q]['pred_sr'] for r in sub for q in BOXES if r[q]['label'] == 'POS'])
        free = np.array([r[q]['pred_sr'] for r in sub for q in BOXES if r[q]['label'] == 'FREE'])
        if not len(pos) or not len(free):
            continue
        thr_grid = np.r_[0, np.unique(np.r_[pos, free]), np.inf]
        roc = [(float(t), float((pos >= t).mean()), float((free >= t).mean())) for t in thr_grid]
        at = lambda t: dict(threshold_sr=t, tpr=float((pos >= t).mean()), fpr=float((free >= t).mean()))
        five = min((t for t, tp, fp in roc if fp <= .05), default=None)
        report[role] = dict(pos=int(len(pos)), free=int(len(free)), at_atom_threshold=at(ATOM_SR), at_fpr5=at(five) if five is not None else None,
                            auc=float(np.trapezoid([tp for _, tp, _ in roc][::-1], [fp for _, _, fp in roc][::-1])))
        print(role, json.dumps(report[role]), flush=True)
    (OUT / 'readout_calibration.json').write_text(json.dumps(report, indent=1), encoding='utf8')
    print('frames', report['frames'], 'anchored', report['anchored'])


if __name__ == '__main__':
    main()
