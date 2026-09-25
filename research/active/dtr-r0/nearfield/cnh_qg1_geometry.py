"""Public query geometry for QG1; no scene depth or evaluator inputs.

Current boxes and cameras share the same frame. Callers must not reuse this
identity-transform helper for a changed mounting pose without transforming
the boxes first. Image bounds use pixel-centre coordinates: image edges are
-0.5 and width/height-0.5. ROI grids use align_corners=False.
"""
from __future__ import annotations

import itertools
import numpy as np

from cnh_street_e2e_materialize import BOXES
from cnh_h3_geometry_diagnostic import query_weights


def _intrinsics(K):
    k = np.asarray(K, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1]) or
            abs(k[0, 1]) > 1e-12 or abs(k[1, 0]) > 1e-12):
        raise ValueError('Finite zero-skew pinhole intrinsics required')
    return k


def _boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or
            not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= boxes[:, 0])):
        raise ValueError('Nondegenerate [query,low/high,xyz] boxes required')
    return boxes


def project_boxes(K, image_size=(640, 360), boxes=BOXES):
    """Conservative clipped [Q,4] xyxy projection, independent of scene depth.

    Camera-aligned AABBs crossing the camera plane are clipped at z=1e-6m
    before projection. Entirely behind-camera/outside boxes yield zero-area
    bounds. These remain explicit missing ROIs, never clamped edge content.
    """
    k, boxes = _intrinsics(K), _boxes(boxes)
    width, height = image_size
    if int(width) != width or int(height) != height or min(width, height) <= 0:
        raise ValueError('Positive integer image dimensions required')
    result = np.zeros((len(boxes), 4), dtype=np.float32)
    for q, (low, high) in enumerate(boxes):
        if high[2] <= 1e-6:
            continue
        low = low.copy()
        low[2] = max(low[2], 1e-6)
        corners = np.array(list(itertools.product(*zip(low, high))))
        u = k[0, 0] * corners[:, 0] / corners[:, 2] + k[0, 2]
        v = k[1, 1] * corners[:, 1] / corners[:, 2] + k[1, 2]
        xmin, xmax = np.clip([u.min(), u.max()], -.5, width-.5)
        ymin, ymax = np.clip([v.min(), v.max()], -.5, height-.5)
        if xmax > xmin and ymax > ymin:
            result[q] = [xmin, ymin, xmax, ymax]
    return result


def roi_valid(bounds):
    bounds = np.asarray(bounds)
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not np.isfinite(bounds).all():
        raise ValueError('Finite [Q,4] bounds required')
    return (bounds[:, 2] > bounds[:, 0]) & (bounds[:, 3] > bounds[:, 1])


def roi_grids(bounds, H, W, size=4):
    """Bin-centre sampling grids [Q,size,size,2], x then y.

    Pass actual grid_sample input dimensions H,W. Bounds must be in that
    input's pixel-centre coordinates. Invalid ROIs receive grid value 2
    (outside image); additionally multiply outputs by roi_valid(bounds).
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    valid = roi_valid(bounds)
    if any(int(v) != v or v <= 0 for v in (H, W, size)):
        raise ValueError('Positive integer sampling dimensions required')
    t = (np.arange(size, dtype=np.float64) + .5) / size
    x = bounds[:, 0, None] + (bounds[:, 2]-bounds[:, 0])[:, None] * t
    y = bounds[:, 1, None] + (bounds[:, 3]-bounds[:, 1])[:, None] * t
    result = np.empty((len(bounds), size, size, 2), dtype=np.float32)
    result[..., 0] = (2 * (x + .5) / W - 1)[:, None, :]
    result[..., 1] = (2 * (y + .5) / H - 1)[:, :, None]
    result[~valid] = 2
    return result


def query_descriptors(boxes=BOXES):
    """[Q,10]: centre xyz, size xyz, ymin/ymax/zmin/zmax divided by 3m.

    Fixed 3m length scale preserves sign and relative geometry, and is not
    fitted to data. Descriptors are public query specification only.
    """
    boxes = _boxes(boxes)
    low, high = boxes[:, 0], boxes[:, 1]
    return (np.concatenate(((low+high)/2, high-low, low[:, 1:2],
                            high[:, 1:2], low[:, 2:3], high[:, 2:3]), axis=1)/3).astype(np.float32)


def ray_coordinates(K, H, W):
    """Public pinhole x/z,y/z at integer pixel centres, [2,H,W]."""
    k = _intrinsics(K)
    if any(int(v) != v or v <= 0 for v in (H, W)):
        raise ValueError('Positive integer image dimensions required')
    y, x = np.indices((H, W), dtype=np.float64)
    return np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1])).astype(np.float32)


def tof_query_weights(boxes=BOXES):
    """Full [Q,64,16] geometric weights; no left/centre/right column mask."""
    return query_weights(_boxes(boxes)).astype(np.float32)
