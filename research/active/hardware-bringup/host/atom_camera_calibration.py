"""Atom RGB intrinsic calibration from a screen checkerboard (9x6 inner corners), raw JPEG orientation.

Fits OpenCV pinhole+rational and fisheye models on a spread subset of detections; reports RMS and
held-out reprojection error. Output: calibration.json beside the raw session. Square size is
arbitrary (screen pixels), so only intrinsics/distortion are meaningful, not board pose scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SESSION = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/raw/capture-20261012T101018599Z-9fbf8074'
OUT = SESSION.parent.parent / 'calibration.json'
PATTERN = (9, 6)


def detect(step=2):
    files = sorted((SESSION / 'rgb').glob('*.jpg'))[::step]
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    found = []
    size = None
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        size = img.shape[::-1]
        ok, corners = cv2.findChessboardCornersSB(img, PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            found.append((f.name, corners.astype(np.float32)))
    return obj, found, size, len(files)


def spread(found, size, k=60):
    """Greedy farthest-point selection on (centroid x, y, apparent size) for pose diversity."""
    feats = np.array([[c[:, 0, 0].mean() / size[0], c[:, 0, 1].mean() / size[1], np.ptp(c[:, 0, 0]) / size[0]] for _, c in found])
    chosen = [int(np.argmax(feats[:, 2]))]
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < min(k, len(found)):
        i = int(np.argmax(d)); chosen.append(i); d = np.minimum(d, np.linalg.norm(feats - feats[i], axis=1))
    return sorted(chosen)


def main():
    obj, found, size, tried = detect()
    print('detected', len(found), 'of', tried, 'frames; image', size, flush=True)
    if len(found) < 15:
        sys.exit('Too few checkerboard detections')
    sel = spread(found, size, 60)
    rest = [i for i in range(len(found)) if i not in sel][::3][:40]
    objp = [obj] * len(sel); imgp = [found[i][1] for i in sel]
    result = dict(session=SESSION.name, image_size=list(size), orientation='raw JPEG (before 90 deg CW + mirror display)',
                  detections=len(found), frames_tried=tried, fit_frames=len(sel), heldout_frames=len(rest))
    # Pinhole + rational distortion
    flags = cv2.CALIB_RATIONAL_MODEL
    rms, K, dist, rv, tv = cv2.calibrateCamera(objp, imgp, size, None, None, flags=flags)
    errs = []
    for i in rest:
        ok, r, t = cv2.solvePnP(obj, found[i][1], K, dist)
        proj, _ = cv2.projectPoints(obj, r, t, K, dist); errs.append(float(np.sqrt(((proj - found[i][1]) ** 2).sum(-1).mean())))
    result['pinhole_rational'] = dict(rms=float(rms), heldout_rms_median=float(np.median(errs)) if errs else None, K=K.tolist(), dist=dist.ravel().tolist(),
                                      hfov_deg=float(2 * np.degrees(np.arctan(size[0] / 2 / K[0, 0]))), vfov_deg=float(2 * np.degrees(np.arctan(size[1] / 2 / K[1, 1]))))
    # Fisheye
    try:
        Kf = np.eye(3); Df = np.zeros((4, 1))
        rmsf, Kf, Df, _, _ = cv2.fisheye.calibrate([o[:, None, :].astype(np.float64) for o in objp], [p.astype(np.float64) for p in imgp], size, Kf, Df,
                                                   flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
                                                   criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9))
        result['fisheye'] = dict(rms=float(rmsf), K=Kf.tolist(), D=Df.ravel().tolist())
    except cv2.error as e:
        result['fisheye'] = dict(error=str(e))
    OUT.write_text(json.dumps(result, indent=1), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items()}, indent=1))


if __name__ == '__main__':
    main()
