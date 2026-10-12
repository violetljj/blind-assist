"""Histogram-consistent scale anchoring and cross-modal thin-obstacle confirmation (EXPLORE).

1) Anchor: relative RGB disparity d (DAV2-Small) is turned into metric depth by d = a / z + b.
   Instead of fitting (a, b) to firmware zone distances, each hypothesis renders the 8x8x16 CNH it
   implies with a minimal forward model (each camera pixel -> its ToF zone via the fitted
   camera<-ToF rotation, range bin of 0.3003 m, energy ~ 1 / r^2, 3-tap pulse blur) and is scored by
   per-zone cosine similarity with the measured normalised evidence (bins 1-15, per-zone baseline
   removed, near bins 1-2 reduced by a 3 sigma crosstalk-tail allowance, positive part) over zones
   with a measured return, minus the share of predicted energy within ~3 m that falls in zones
   where the sensor measured nothing (negative evidence). Hypotheses: metric
   depth of the 95th and 40th disparity percentiles on a log grid (24 x 20).
2) Confirm: RGB pixels inside a query box are checked against the measured evidence of their ToF
   zone at the predicted range bin (+/-1); confirmed when the evidence there exceeds CONFIRM_Z.
"""
from __future__ import annotations

import math

import numpy as np

WIDTH = .3002784
EDGE = math.tan(math.pi / 8)
CONFIRM_Z = 2.0
TAIL_Z = 3.0
ACTIVE_Z = 2.5
NEG_BINS = 10          # predicted surfaces within ~3 m in a zone with no measured return are penalised
NEG_WEIGHT = 1.0
PULSE = np.array([.2, .6, .2])


class PixelZoneMap:
    """Per (downsampled) camera pixel: ToF zone index (model order, -1 outside) and ray-norm factor."""

    def __init__(self, K, Rct, h, w, step=4):
        self.step = step
        vv, uu = np.mgrid[0:h:step, 0:w:step]
        self.shape = vv.shape
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
        t = d @ Rct                                   # ToF-frame direction (Rct maps ToF -> camera)
        ok = t[:, 2] > 0
        tx, ty = t[:, 0] / np.where(ok, t[:, 2], 1), t[:, 1] / np.where(ok, t[:, 2], 1)
        col = np.floor((tx + EDGE) / (2 * EDGE) * 8).astype(int); row = np.floor((ty + EDGE) / (2 * EDGE) * 8).astype(int)
        inside = ok & (col >= 0) & (col < 8) & (row >= 0) & (row < 8)
        self.zone = np.where(inside, row * 8 + col, -1)
        self.norm = np.linalg.norm(d, axis=1)         # range = optical z * norm (rotation keeps length)
        self.sel = np.flatnonzero(inside)


def measured_evidence(z_model):
    """[8,8,16] normalised CNH (model order) -> [64,16] positive evidence with bin0 and baseline removed."""
    z = np.asarray(z_model, np.float64).reshape(64, 16).copy()
    z[:, 0] = 0
    z = z - np.median(z[:, 3:], axis=1, keepdims=True)
    z[:, 1:3] -= TAIL_Z                    # near bins carry a ~1-3 sigma crosstalk tail on the real sensor
    return np.clip(z, 0, None)


def render(zmap, inv_z, pzm):
    """Predicted [H, 64, 16] energy for H hypotheses of per-pixel inverse optical depth [H, N]."""
    H = inv_z.shape[0]
    zone = pzm.zone[pzm.sel]; norm = pzm.norm[pzm.sel]
    r = np.where(inv_z > 1e-3, norm[None] / np.maximum(inv_z, 1e-3), np.inf)
    b = np.floor(r / WIDTH)
    ok = np.isfinite(r) & (b >= 1) & (b < 16)
    e = np.where(ok, 1 / np.maximum(r, .1) ** 2, 0.)
    idx = (np.arange(H)[:, None] * 1024 + zone[None] * 16 + np.where(ok, b, 0).astype(int))
    hist = np.bincount(idx.ravel(), weights=e.ravel(), minlength=H * 1024).reshape(H, 64, 16)
    blurred = hist * PULSE[1]
    blurred[..., 1:] += hist[..., :-1] * PULSE[0]
    blurred[..., :-1] += hist[..., 1:] * PULSE[2]
    return blurred


def fit_histogram(disp, z_model, pzm, min_zones=4):
    """Return (a, b, info) or (None, None, info)."""
    d = disp[::pzm.step, ::pzm.step].reshape(-1)[pzm.sel].astype(np.float64)
    meas = measured_evidence(z_model)
    active_meas = meas.max(1) > ACTIVE_Z
    if active_meas.sum() < min_zones:
        return None, None, dict(reason='few_measured_zones', measured_zones=int(active_meas.sum()))
    dn, df = np.percentile(d, 95), np.percentile(d, 40)
    if dn - df < 1e-6:
        return None, None, dict(reason='flat_disparity')
    zn = np.exp(np.linspace(np.log(.3), np.log(4.), 24))
    hyps = [(a_zn, a_zn * k) for a_zn in zn for k in np.exp(np.linspace(np.log(1.15), np.log(10.), 20))]
    A = np.array([(dn - df) / (1 / z1 - 1 / z2) for z1, z2 in hyps]); B = np.array([dn - a / z1 for a, (z1, _) in zip(A, hyps)])
    inv = (d[None] - B[:, None]) / A[:, None]
    pred = render(None, inv, pzm)                                  # [H,64,16]
    pn = np.linalg.norm(pred, axis=2); mn = np.linalg.norm(meas, axis=1)
    cos = (pred * meas[None]).sum(2) / np.maximum(pn * mn[None], 1e-12)
    # positive evidence: shape agreement in zones with a measured return
    pos = np.where(active_meas[None], cos, 0).sum(1) / max(active_meas.sum(), 1)
    # negative evidence: predicted near energy in zones where the sensor saw nothing
    near = pred[..., 1:NEG_BINS].sum(2)
    neg = np.where(~active_meas[None], near, 0).sum(1) / np.maximum(pred.sum((1, 2)), 1e-12)
    score = pos - NEG_WEIGHT * neg
    k = int(np.argmax(score))
    info = dict(score=float(score[k]), median_score=float(np.median(score)), measured_zones=int(active_meas.sum()),
                z95=float(hyps[k][0]), z40=float(hyps[k][1]))
    if score[k] - np.median(score) < .05:
        return None, None, dict(info, reason='flat_score')
    return float(A[k]), float(B[k]), info


def confirm_pixels(box_mask_full, z_full, pzm, z_model):
    """Fraction of in-box RGB pixels (downsampled) whose zone shows evidence at the predicted range bin +/-1."""
    meas = measured_evidence(z_model)
    m = box_mask_full[::pzm.step, ::pzm.step].reshape(-1)
    zz = z_full[::pzm.step, ::pzm.step].reshape(-1)
    idx = np.flatnonzero(m & (pzm.zone >= 0) & np.isfinite(zz))
    if not len(idx):
        return 0., 0
    r = zz[idx] * pzm.norm[idx]; b = np.clip(np.floor(r / WIDTH).astype(int), 1, 15); zone = pzm.zone[idx]
    ev = np.max(np.stack([meas[zone, np.clip(b + o, 1, 15)] for o in (-1, 0, 1)]), 0)
    return float((ev > CONFIRM_Z).mean()), int(len(idx))
