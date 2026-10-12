"""Atom RGB camera model: fisheye undistortion and the upright (display) pinhole intrinsics.

Raw JPEGs are 640x480 landscape; the display transform (90 deg CW + horizontal mirror, verified
against the ToF left/right layout) equals a transpose: displayed (u, v) = raw (v, u). Calibration:
atom_camera_calibration.py (screen checkerboard, fisheye RMS 0.37 px, 521/527 detections).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CALIBRATION = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/calibration.json'


class AtomCamera:
    def __init__(self, path=CALIBRATION, balance=0.0):
        cal = json.loads(Path(path).read_text(encoding='utf8'))
        self.size = tuple(cal['image_size'])                       # raw (w, h)
        self.K = np.array(cal['fisheye']['K']); self.D = np.array(cal['fisheye']['D']).reshape(4, 1)
        self.Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, self.size, np.eye(3), balance=balance)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.Knew, self.size, cv2.CV_16SC2)
        k = self.Knew
        # Upright image = transpose of the raw image: swap axes in K.
        self.K_upright = np.array([[k[1, 1], 0, k[1, 2]], [0, k[0, 0], k[0, 2]], [0, 0, 1.]])
        self.upright_size = (self.size[1], self.size[0])

    def undistort_upright(self, raw_rgb):
        """raw HxWx3 (480x640) -> undistorted upright (640x480 rows x cols) pinhole image with K_upright."""
        und = cv2.remap(raw_rgb, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return np.ascontiguousarray(und.transpose(1, 0, 2))

    def fov_deg(self):
        k = self.K_upright; w, h = self.upright_size
        return 2 * np.degrees(np.arctan(w / 2 / k[0, 0])), 2 * np.degrees(np.arctan(h / 2 / k[1, 1]))
