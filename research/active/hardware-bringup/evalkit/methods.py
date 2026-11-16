"""Depth methods for the minimal evaluation kit. A method maps a Frame to a predicted optical-z map (HxW, metres) or None.
Methods must not read f.ref_depth, except those explicitly named as references/oracles (`reference`, `oracle_dav2s`)
and the ToF EMULATION, which samples the reference depth only inside its own zones (as a ToF sensor would measure).

  reference     the reference depth itself (sanity: zero error, perfect detection)
  tof_dav2s     DAV2-S relative disparity, affine-anchored to an emulated coaxial 45 deg 8x8 ToF (project chain:
                fit -> robust fit -> last fit per sequence)
  unidepth_s    UniDepth-V2-S metric depth with the known K, no anchor
  oracle_dav2s  DAV2-S affine-fitted to the full reference depth per frame (shape ceiling)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parents[1] / 'host'
sys.path.insert(0, str(HOST))


class Methods:
    def __init__(self):
        import sanpo_scale_without_tof_dev_20261117 as P
        from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
        self.P, self.fit_quality, self.robust_affine = P, fit_quality, robust_affine
        self.rel, _, _, self.ud_infer = P.load_models()
        self.S, self.zone_rays = P.V1.S, P.V1.X.zone_rays()
        self.last_fit = {}
        self._cache = None

    def _disp(self, f):
        import torch
        key = (f.dataset, f.seq, f.index)
        if self._cache is None or self._cache[0] != key:
            with torch.inference_mode():
                self._cache = (key, self.rel.infer_image(f.image, 518).astype(np.float32).reshape(-1))
        return self._cache[1]

    def reference(self, f):
        return f.ref_depth.copy()

    def unidepth_s(self, f):
        return self._unidepth_cached(f)

    def _unidepth_raw(self, f):
        from PIL import Image
        return self.ud_infer(Image.fromarray(np.ascontiguousarray(f.image[:, :, ::-1])), f.K).astype(np.float32)

    def oracle_dav2s(self, f):
        disp = self._disp(f); z = f.ref_depth.reshape(-1); m = np.isfinite(z) & (z > .3) & np.isfinite(disp)
        a, b = np.polyfit(1 / z[m], disp[m], 1)
        out = self.P.to_depth(disp, (a, b) if a > 0 else None)
        return None if out is None else out.reshape(f.ref_depth.shape)

    OUTDOOR_MAX_M, OUTDOOR_NOISE = 2.85, .03     # VL53L8CX ranging under ambient light (ST flyer), per-zone noise

    def _zones(self, f, outdoor=False):
        """Emulated coaxial 45 deg 8x8 ToF: per-zone range from the reference (median of the zone's sample rays), plus the
        zone sample pixels. outdoor=True keeps zones <= 2.85 m and multiplies by (1 + N(0, 3%)), seeded by the frame."""
        H, W = f.ref_depth.shape; K = f.K; rc = self.zone_rays
        zu = np.round(K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]).astype(int); zv = np.round(K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < W) & (zv >= 0) & (zv < H); zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
        dist = np.full(len(rc), np.nan)
        for zi in range(len(rc)):
            s = zin[zi]
            if s.sum() < 13:
                continue
            zz = f.ref_depth[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
            if ok.sum() >= 13:
                r = float(np.median(zz[ok] / zcos[zi][s][ok]))
                if .3 < r < 4.:
                    dist[zi] = r
        if outdoor:
            import hashlib
            rng = np.random.default_rng(int(hashlib.sha256(f'{f.dataset}:{f.seq}:{f.index}'.encode()).hexdigest()[:8], 16))
            dist = np.where(dist <= self.OUTDOOR_MAX_M, dist * (1 + rng.normal(0, self.OUTDOOR_NOISE, len(dist))), np.nan)
        return dist, zu, zv, zin, zcos

    def _unidepth_cached(self, f):
        key = ('ud', f.dataset, f.seq, f.index)
        if getattr(self, '_ud_cache', (None,))[0] != key:
            self._ud_cache = (key, self._unidepth_raw(f))
        return self._ud_cache[1]

    def tof_dav2s(self, f, outdoor=False):
        disp = self._disp(f); H, W = f.ref_depth.shape; K = f.K; rc = self.zone_rays
        dist = self._zones(f, outdoor)[0]
        dz, zfac = self.S.zone_disparity(disp.reshape(H, W), K, np.eye(3), rc)
        inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
        fit = self.S.fit_affine(dz, inv)
        if not self.fit_quality(dz, inv, fit)['accepted']:
            rb = self.robust_affine(dz, inv)['fit']
            if rb is not None:
                fit = rb
        key = ('dav2s', outdoor, f.dataset, f.seq)
        fit = fit if fit is not None else self.last_fit.get(key)
        if fit is not None:
            self.last_fit[key] = fit
        out = self.P.to_depth(disp, fit)
        return None if out is None else out.reshape(H, W)

    def tof_unidepth(self, f, outdoor=False):
        """UniDepth-V2-S times one scale s = median over valid zones of (ToF range / UniDepth range on the zone's rays)."""
        z = self._unidepth_cached(f); dist, zu, zv, zin, zcos = self._zones(f, outdoor)
        ratios = []
        for zi in np.flatnonzero(np.isfinite(dist)):
            s = zin[zi]; zz = z[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
            if ok.sum() >= 13:
                ratios.append(dist[zi] / float(np.median(zz[ok] / zcos[zi][s][ok])))
        key = ('ud', outdoor, f.dataset, f.seq)
        sc = float(np.median(ratios)) if len(ratios) >= 5 else self.last_fit.get(key)
        if sc is not None:
            self.last_fit[key] = sc
        return z if sc is None else z * sc

    def tof_dav2s_out(self, f):
        return self.tof_dav2s(f, outdoor=True)

    def tof_unidepth_out(self, f):
        return self.tof_unidepth(f, outdoor=True)


METHOD_NAMES = ('reference', 'tof_dav2s', 'unidepth_s', 'oracle_dav2s')
A1_METHODS = ('tof_dav2s', 'tof_unidepth', 'tof_dav2s_out', 'tof_unidepth_out', 'unidepth_s', 'oracle_dav2s')
