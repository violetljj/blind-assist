"""NumPy ASE Fisheye624 adapter; mathematical checks, not verified SDK parity.

Calibration source:
https://raw.githubusercontent.com/facebookresearch/projectaria_tools/main/projects/AriaSyntheticEnvironment/AseCalibrationProvider.cpp
Formula adapted from:
https://raw.githubusercontent.com/facebookresearch/projectaria_tools/main/core/calibration/camera_projections/FisheyeRadTanThinPrism.h

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Python adaptation adds conservative <=1 rad visibility and NaN invalid outputs.
Unprojection uses a bounded radial bisection rather than the SDK Newton solver.
Camera axes are +X right, +Y down, +Z optical forward. No image rotation is applied.
"""
from __future__ import annotations
import numpy as np


class AseCalibration:
    def __init__(self, size=704):
        if size not in (704, 1408):
            raise ValueError('ASE supports square images of size 704 or 1408')
        self.size = int(size)
        self.params = np.array([
            297.6375381033778, 357.6599197217746, 349.1922497127481,
            .3650890375644368, -.1738082418112771, -.7534945484033189,
            2.434788882752295, -2.57786220300886, .8788483538598834,
            .0008005198595407136, -.000294237814554143, 0., 0., 0., 0.])
        scale = size / 704
        self.params[0] *= scale
        self.params[1:3] = scale * (self.params[1:3] + .5) - .5
        self.valid_radius = 353.75 * scale
        self.max_angle = 1.
        q = np.array([.9441858687689326, .326409343828490850,
                      .029274992008313648, .033361059956531547])
        w, x, y, z = q / np.linalg.norm(q)
        self._T_device_camera = np.eye(4)
        self._T_device_camera[:3, :3] = [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        # CameraCalibration constructor expects T_Device_Camera, despite the
        # provider's misleading local variable name T_Camera_Device.
        self._T_device_camera[:3, 3] = [-.007530096566173914,
                                       -.010908549841580260, -.003598063315542823]

    @property
    def T_device_camera(self):
        return self._T_device_camera.copy()

    @property
    def principal_point(self):
        return self.params[1:3].copy()

    def _pixel_valid(self, uv):
        return (np.isfinite(uv).all(axis=-1)
                & (uv >= 0).all(axis=-1) & (uv < self.size).all(axis=-1)
                & (np.linalg.norm(uv-self.params[1:3], axis=-1) <= self.valid_radius))

    def _radial(self, theta):
        square = theta * theta
        polynomial = np.zeros_like(theta) + self.params[8]
        for k in self.params[3:8][::-1]:
            polynomial = polynomial * square + k
        return theta * (1 + square * polynomial)

    def _distort(self, xy):
        radius2 = (xy * xy).sum(axis=-1)
        p = self.params[9:11]
        output = xy + 2 * (xy*p).sum(axis=-1)[..., None] * xy + radius2[..., None] * p
        output[..., 0] += self.params[11]*radius2 + self.params[12]*radius2**2
        output[..., 1] += self.params[13]*radius2 + self.params[14]*radius2**2
        return output

    def project(self, points):
        """Project ...x3 optical points to ...x2 pixels; invalids are NaN."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError('Expected ...x3 optical points')
        good = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
        safe = np.where(good[..., None], points, [0., 0., 1.])
        radius = np.linalg.norm(safe[..., :2], axis=-1)
        theta = np.arctan2(radius, safe[..., 2])
        scale = np.divide(self._radial(theta), radius, out=np.zeros_like(radius), where=radius > 0)
        xy = safe[..., :2] * scale[..., None]
        uv = self.params[0]*self._distort(xy) + self.params[1:3]
        good &= (theta <= self.max_angle) & self._pixel_valid(uv)
        return np.where(good[..., None], uv, np.nan)

    def unproject(self, uv):
        """Unproject ...x2 pixels to ...x3 unit rays; never axial-depth rays."""
        uv = np.asarray(uv, dtype=np.float64)
        if uv.shape[-1:] != (2,):
            raise ValueError('Expected ...x2 pixels')
        good = self._pixel_valid(uv)
        target = (np.where(good[..., None], uv, self.params[1:3])-self.params[1:3])/self.params[0]
        xy = target.copy()
        p0, p1 = self.params[9:11]
        # ASE has zero thin-prism coefficients. Full terms remain in project;
        # inversion below is specialized to this fixed official calibration.
        for _ in range(6):
            x, y = xy[..., 0], xy[..., 1]
            j00 = 1 + 6*x*p0 + 2*y*p1
            j11 = 1 + 6*y*p1 + 2*x*p0
            j01 = 2*(x*p1+y*p0)
            error = self._distort(xy)-target
            determinant = j00*j11-j01*j01
            dx = (j11*error[..., 0]-j01*error[..., 1])/determinant
            dy = (j00*error[..., 1]-j01*error[..., 0])/determinant
            xy -= np.stack((dx, dy), axis=-1)
        radial = np.linalg.norm(xy, axis=-1)
        low = np.zeros_like(radial); high = np.ones_like(radial)*self.max_angle
        for _ in range(38):
            mid = (low+high)*.5
            below = self._radial(mid) < radial
            low = np.where(below, mid, low); high = np.where(below, high, mid)
        theta = np.where(radial == 0, 0., (low+high)*.5)
        good &= radial <= self._radial(np.asarray(self.max_angle))
        good &= np.linalg.norm(self._distort(xy)-target, axis=-1) < 1e-10
        scale = np.divide(np.sin(theta), radial, out=np.zeros_like(radial), where=radial > 0)
        rays = np.concatenate((xy*scale[..., None], np.cos(theta)[..., None]), axis=-1)
        return np.where(good[..., None], rays, np.nan)
