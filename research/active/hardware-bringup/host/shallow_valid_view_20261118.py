"""Image support of the three camera inputs in the paired shallow-obstacle replay."""

from functools import lru_cache

import cv2
import numpy as np

from spires_insert_dev_20261117 import CX, CY, D0, H, K0, R_CV, W


@lru_cache(maxsize=1)
def _oxford_view():
    # The wider rectification used for these recordings has focal length 280 px.
    # Map each virtual ray back to cam0 using the calibration of pinhole_rgb().
    v, u = np.mgrid[:H, :W]
    rays = np.stack(((u - CX) / 280., (v - CY) / 280., np.ones_like(u)), axis=-1)
    rays = rays @ R_CV.T
    k = np.array([[K0[0], 0., K0[2]], [0., K0[1], K0[3]], [0., 0., 1.]])
    normalized = (rays[..., :2] / rays[..., 2:]).reshape(-1, 1, 2)
    pixels = cv2.fisheye.distortPoints(normalized, k, np.array(D0))
    x, y = pixels.reshape(H, W, 2).astype(np.float32).transpose(2, 0, 1)
    theta = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    # The original rectifier assigns zero outside the raw 1440x1080 sensor or
    # its 80-degree angle limit, even if the projected coordinate is in bounds.
    mask = (x >= 0) & (x <= 1439) & (y >= 0) & (y <= 1079) & (theta < np.deg2rad(80))
    mask.setflags(write=False)
    return mask


def valid_view(source_alias, source_row, shape):
    """Return the calibrated RGB footprint on the input depth grid (height, width)."""
    if source_alias in ("egowalk-union", "lin-yellow"):
        # Directly decoded ZED left RGB and resized native ARKit RGB. No virtual
        # camera raster or constant border was introduced in either source path.
        return np.ones(shape, dtype=bool)
    if source_alias != "oxford":
        raise ValueError(f"No camera support rule for {source_alias!r}")
    expected_k = np.array([[280., 0., CX], [0., 280., CY], [0., 0., 1.]])
    if tuple(shape) != (H, W) or not np.allclose(source_row["K"], expected_k):
        raise ValueError("Oxford replay requires its 480x640, 280 px virtual camera")
    return _oxford_view()
