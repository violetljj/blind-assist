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
        self.rel, _, self.ud, self.ud_infer = P.load_models()
        self.S, self.zone_rays = P.V1.S, P.V1.X.zone_rays()
        self.last_fit = {}
        self._cache = None
        self.source = {}             # per-call scale source of the scaled UniDepth arms (logged by runners)

    def _disp(self, f):
        import torch
        key = (f.dataset, f.seq, f.index)
        if self._cache is None or self._cache[0] != key:
            with torch.inference_mode():
                self._cache = (key, self.rel.infer_image(f.image, 518).astype(np.float32).reshape(-1))
        return self._cache[1]

    def set_unidepth_pixels(self, n):
        """Fix UniDepth's input pixel budget (both bounds = n; aspect kept, sides rounded up to multiples of 14), to emulate
        a phone resolution (A3: 203056 = 392x518, 315560 = 490x644). Default bounds are 200k-600k."""
        self.ud.shape_constraints = dict(self.ud.shape_constraints, pixels_min=n, pixels_max=n)
        self._ud_cache = (None,)

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

    def _tof_scale(self, f, outdoor):
        """One scale = median over valid zones (>= 5) of (ToF range / UniDepth range on the zone's rays); None otherwise."""
        z = self._unidepth_cached(f); dist, zu, zv, zin, zcos = self._zones(f, outdoor)
        ratios = []
        for zi in np.flatnonzero(np.isfinite(dist)):
            s = zin[zi]; zz = z[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
            if ok.sum() >= 13:
                ratios.append(dist[zi] / float(np.median(zz[ok] / zcos[zi][s][ok])))
        return float(np.median(ratios)) if len(ratios) >= 5 else None

    AFFINE_MIN_ZONES, AFFINE_MIN_SPAN, AFFINE_A, AFFINE_B = 12, .75, (.5, 2.), .5

    def _tof_affine(self, f, outdoor):
        """Scale + offset in range from the ToF zones (DIRECTION_A_INDOOR_ADT_DEV_20261118 follow-up): ToF range ~ a * UniDepth
        range + b, least squares, then refit without zones whose residual exceeds 3 x MAD (>= 0.02 m). Used only with >= 12
        zones spanning >= 0.75 m of UniDepth range and a in [0.5, 2], |b| <= 0.5 m; None otherwise."""
        z = self._unidepth_cached(f); dist, zu, zv, zin, zcos = self._zones(f, outdoor)
        tr, ur = [], []
        for zi in np.flatnonzero(np.isfinite(dist)):
            s = zin[zi]; zz = z[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
            if ok.sum() >= 13:
                tr.append(dist[zi]); ur.append(float(np.median(zz[ok] / zcos[zi][s][ok])))
        tr, ur = np.array(tr), np.array(ur)
        if len(tr) < self.AFFINE_MIN_ZONES or np.ptp(ur) < self.AFFINE_MIN_SPAN:
            return None
        a, b = np.polyfit(ur, tr, 1); res = np.abs(tr - (a * ur + b)); keep = res <= max(.02, 3 * np.median(res) / .6745)
        if keep.sum() >= self.AFFINE_MIN_ZONES and np.ptp(ur[keep]) >= self.AFFINE_MIN_SPAN:
            a, b = np.polyfit(ur[keep], tr[keep], 1)
        if not (self.AFFINE_A[0] <= a <= self.AFFINE_A[1] and abs(b) <= self.AFFINE_B):
            return None
        return float(a), float(b)

    def hybrid_aff(self, f, outdoor, name):
        """ToF scale + offset (if accepted) -> one ToF scale (>= 5 zones) -> own ground x eye height -> last scale.
        Optical z per pixel: z' = a z + b / |ray| (range r' = a r + b)."""
        aff = self._tof_affine(f, outdoor)
        if aff is None:
            out = self.hybrid_out(f, 0., name, outdoor=outdoor)
            return out
        z = self._unidepth_cached(f); H, W = z.shape; a, b = aff
        nrm = np.linalg.norm(__import__('core').rays(f.K, H, W), axis=1).reshape(H, W)
        self.source[name] = 'tof_affine'
        self.last_fit[(name, f.dataset, f.seq)] = a          # a later frame without ToF falls back to this scale
        return a * z + b / nrm

    def hybrid_in_aff(self, f):
        return self.hybrid_aff(f, False, 'hybrid_in_aff')

    def hybrid_out_aff(self, f):
        return self.hybrid_aff(f, True, 'hybrid_out_aff')

    def tof_unidepth(self, f, outdoor=False):
        """UniDepth-V2-S times one scale s = median over valid zones of (ToF range / UniDepth range on the zone's rays)."""
        z = self._unidepth_cached(f)
        key = ('ud', outdoor, f.dataset, f.seq)
        sc = self._tof_scale(f, outdoor)
        self.source['tof_unidepth_out' if outdoor else 'tof_unidepth'] = 'tof' if sc is not None else ('last' if self.last_fit.get(key) is not None else 'none')
        sc = sc if sc is not None else self.last_fit.get(key)
        if sc is not None:
            self.last_fit[key] = sc
        return z if sc is None else z * sc

    FLOOR_SCALE_RANGE = (.5, 2.)

    def _floor_scale(self, f, dh=0.):
        """Scale from the model's own ground: RANSAC plane below the camera in the UniDepth point cloud (normal within 15 deg
        of device gravity, or 35 deg of the camera y axis without it); s = (known eye height + dh) / predicted plane distance.
        Uses only the prediction, the device gravity and the sequence's known eye height. Cached per frame (dh applied after)."""
        import hashlib
        import core
        key = ('floor', f.dataset, f.seq, f.index)
        if getattr(self, '_floor_cache', (None,))[0] != key:
            z = self._unidepth_cached(f).reshape(-1); H, W = f.ref_depth.shape
            ok = np.isfinite(z) & (z > .3) & (z < 10)
            pts = core.rays(f.K, H, W)[ok] * z[ok, None]
            rng = np.random.default_rng(int(hashlib.sha256(f'floor:{f.dataset}:{f.seq}:{f.index}'.encode()).hexdigest()[:8], 16))
            gp = (core.ground_plane(pts, f.down, rng) if f.down is not None
                  else core.ground_plane(pts, np.array([0, 1., 0]), rng, max_deg=35.))
            self._floor_cache = (key, None if gp is None else gp[1])
        d = self._floor_cache[1]
        if d is None or f.eye_height is None:
            return None
        sc = (f.eye_height + dh) / d
        return sc if self.FLOOR_SCALE_RANGE[0] <= sc <= self.FLOOR_SCALE_RANGE[1] else None

    def floor_unidepth(self, f):
        z = self._unidepth_cached(f); key = ('floor', f.dataset, f.seq)
        sc = self._floor_scale(f)
        self.source['floor_unidepth'] = 'floor' if sc is not None else ('last' if self.last_fit.get(key) is not None else 'none')
        sc = sc if sc is not None else self.last_fit.get(key)
        if sc is not None:
            self.last_fit[key] = sc
        return z if sc is None else z * sc

    def hybrid_out(self, f, dh=0., name='hybrid_out', outdoor=True):
        """Outdoor-limited ToF scale if >= 5 zones, else the model's own ground scale, else the sequence's last scale.
        outdoor=False (hybrid_in, DIRECTION_A_INDOOR_ADT_DEV_20261118): the same chain with the ideal ToF emulation."""
        z = self._unidepth_cached(f); key = (name, f.dataset, f.seq)
        sc, src = self._tof_scale(f, outdoor), 'tof'
        if sc is None:
            sc, src = self._floor_scale(f, dh), 'floor'
        if sc is None:
            sc, src = self.last_fit.get(key), 'last'
        if sc is None:
            src = 'none'
        self.source[name] = src
        if sc is not None:
            self.last_fit[key] = sc
        return z if sc is None else z * sc

    def hybrid_out_hm10(self, f):
        return self.hybrid_out(f, -.10, 'hybrid_out_hm10')

    def hybrid_out_hm5(self, f):
        return self.hybrid_out(f, -.05, 'hybrid_out_hm5')

    def hybrid_out_hp5(self, f):
        return self.hybrid_out(f, .05, 'hybrid_out_hp5')

    def hybrid_out_hp10(self, f):
        return self.hybrid_out(f, .10, 'hybrid_out_hp10')

    def hybrid_in(self, f):
        return self.hybrid_out(f, 0., 'hybrid_in', outdoor=False)

    def hybrid_in_hm5(self, f):
        return self.hybrid_out(f, -.05, 'hybrid_in_hm5', outdoor=False)

    def hybrid_in_hp5(self, f):
        return self.hybrid_out(f, .05, 'hybrid_in_hp5', outdoor=False)

    def tof_dav2s_out(self, f):
        return self.tof_dav2s(f, outdoor=True)

    def tof_unidepth_out(self, f):
        return self.tof_unidepth(f, outdoor=True)


METHOD_NAMES = ('reference', 'tof_dav2s', 'unidepth_s', 'oracle_dav2s')
A1_METHODS = ('tof_dav2s', 'tof_unidepth', 'tof_dav2s_out', 'tof_unidepth_out', 'unidepth_s', 'oracle_dav2s')
A2_METHODS = ('tof_dav2s_out', 'unidepth_s', 'tof_unidepth_out', 'floor_unidepth', 'hybrid_out',
              'hybrid_out_hm10', 'hybrid_out_hm5', 'hybrid_out_hp5', 'hybrid_out_hp10')
