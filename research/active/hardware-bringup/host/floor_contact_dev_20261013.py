"""Floor-contact geometry readout (EXPLORE): verify RGB in-box blobs by where they meet the floor.

The floor geometry (height + gravity) is the most reliable metric cue we have; DAV2's relative depth shape is not.
For each 8-connected in-box blob that passes the v6 rule (>= 10 px, median z <= 1.8 m), scan image columns below the
blob for the first floor pixel (fitted point within FLOOR_TOL of the floor plane), requiring the pixels in between to stay
at the blob's depth (the blob stands on that floor spot). If >= GROUND_FRAC of the sampled columns find such a contact,
the blob is grounded: its contact pixel is intersected with the floor plane (ray-plane, independent of DAV2 scale) and
the blob counts only if that contact point lies inside the box footprint (|x| <= 0.30, 0.30 <= z <= 2.0). Ungrounded
blobs (overhangs, hidden bases, unreadable glossy floors) keep the v6 decision.
"""
from __future__ import annotations

import numpy as np

FLOOR_TOL, GAP_TOL, MAX_SCAN, GROUND_FRAC, MAX_COLS = .15, .35, 120, .3, 24


def contact_readout(cc_mask_fn, pts, ok, rays_corr, height, h, w, boxes, min_px=150, blob_z=1.8, min_blob=10):
    """pts: [h*w,3] corridor points (NaN where invalid); rays_corr: [h*w,3] corridor ray directions; returns
    per query (v6_alert, contact_alert, n_grounded, n_vetoed)."""
    import cv2
    Y = pts[:, 1].reshape(h, w); Z = pts[:, 2].reshape(h, w)
    floor = (np.abs(Y - height) < FLOOR_TOL).reshape(h, w) & ok.reshape(h, w)
    out = []
    for lo, hi in boxes:
        ins = ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= 2.)
        if not ins.any():
            out.append((False, False, 0, 0)); continue
        n, cc = cv2.connectedComponents(ins.reshape(h, w).astype(np.uint8), connectivity=8)
        v6 = 0; kept = 0; grounded = 0; vetoed = 0
        for c in range(1, n):
            m = cc == c; npx = int(m.sum())
            if npx < min_blob or np.median(Z[m]) > blob_z:
                continue
            v6 += npx
            ys, xs = np.nonzero(m); cols = np.unique(xs)
            if len(cols) > MAX_COLS:
                cols = cols[np.linspace(0, len(cols) - 1, MAX_COLS).astype(int)]
            zb = float(np.median(Z[m])); hits = []
            for u in cols:
                vb = ys[xs == u].max()
                for v in range(vb + 1, min(h, vb + 1 + MAX_SCAN)):
                    if floor[v, u]:
                        hits.append((v, u)); break
                    if not np.isfinite(Z[v, u]) or abs(Z[v, u] - zb) > GAP_TOL:
                        break
            if len(hits) >= max(1, GROUND_FRAC * len(cols)):
                grounded += 1
                v, u = np.median(np.array(hits), axis=0).astype(int)
                r = rays_corr[v * w + u]
                if r[1] > .02:
                    p = r * (height / r[1])                       # ray-plane intersection with y = height (down)
                    if abs(p[0]) <= .30 and .30 <= p[2] <= 2.:
                        kept += npx
                    else:
                        vetoed += 1
                    continue
            kept += npx                                           # ungrounded: keep the v6 decision
        out.append((v6 >= min_px, kept >= min_px, grounded, vetoed))
    return out
