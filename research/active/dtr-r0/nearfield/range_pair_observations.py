"""Public range readouts and separate evaluator-only return attribution.

Footprint overlap is compatibility, not ownership. No truth or target mask is
accepted by the public readout; attribution never changes a selected zone.
"""
import numpy as np

from inherit_spatial_model import canonical_tof, FOCAL
from ba_camera_corridor import sample_native, rays
from query_occupancy_data import camera_bounds


def public_readouts(tof):
    """Read nominal range without imposing the 3m outcome boundary."""
    z, valid, boxes, _, _ = canonical_tof(tof)
    xl = (boxes[:, 1]*320-160)/FOCAL*z
    xh = (boxes[:, 3]*320-160)/FOCAL*z
    yl = (boxes[:, 0]*180-90)/FOCAL*z
    yh = (boxes[:, 2]*180-90)/FOCAL*z
    compatible = valid & (xh >= -.3) & (xl <= .3) & (yh >= -.2) & (yl <= .9)

    def read(mask, count):
        ids = np.flatnonzero(mask)
        if len(ids) < count:
            return dict(estimate_m=None, selected_zone_ids=[])
        chosen = ids[np.argsort(z[ids], kind='stable')[:count]]
        return dict(estimate_m=float(np.median(z[chosen])),
                    selected_zone_ids=chosen.tolist())

    return dict(global_min=read(valid, 1), corridor_min=read(compatible, 1),
                corridor_median3=read(compatible, 3), valid_count=int(valid.sum()),
                corridor_count=int(compatible.sum()))


def evaluator_target_mask(depth, geo):
    """Evaluator-only sampled native XYZ membership in actual target cubes.

The current source has zero camera rotation; reject unsupported poses rather
than silently treating camera-forward Z as world X. Cubes crossing camera Z=0
remain valid bounds, while nonpositive/invalid native observations are excluded.
"""
    depth = np.asarray(depth)
    if depth.shape != (360, 640):
        raise ValueError('Native optical Z must be 360x640')
    camera = dict(geo['declared_camera'])
    if any(abs(float(camera.get(k, 0))) > 1e-12 for k in ('pitch', 'yaw', 'roll')):
        raise ValueError('Only zero-rotation source cameras are supported')
    if 'actual_camera_location_m' in geo:
        camera.update(zip(('x', 'y', 'z'), geo['actual_camera_location_m']))
    targets = [obj for obj in geo['objects'] if obj['name'] in ('target_a', 'target_b')]
    if len(targets) != 2 or len({obj['name'] for obj in targets}) != 2:
        raise ValueError('Require exactly target_a and target_b')
    z = sample_native(depth).astype(np.float64)
    ax, ay = rays()
    with np.errstate(invalid='ignore'):
        x, y = ax*z, ay*z
    valid = np.isfinite(z) & (z > 0)
    mask = np.zeros(z.shape, bool)
    for lo, hi in camera_bounds(targets, camera):
        mask |= (valid & (x >= lo[0]-.002) & (x <= hi[0]+.002)
                 & (y >= lo[1]-.002) & (y <= hi[1]+.002)
                 & (z >= lo[2]-.002) & (z <= hi[2]+.002))
    return mask


def summarize_trace_support(traces, selected_zone_ids, mask):
    """Count actual winning-bin contributor pixels, never unobserved bins."""
    mask = np.asarray(mask)
    if mask.shape != (192, 256) or mask.dtype != np.bool_:
        raise ValueError('Require sampled boolean target mask')
    selected = list(selected_zone_ids)
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate selected zones')
    by_zone = {int(t['zone_id']): t for t in traces}
    target, total = 0, 0
    for zone in selected:
        trace = by_zone[zone]
        indices = np.asarray(trace['pixel_indices'], dtype=np.int64)
        if not trace['observed'] or indices.ndim != 1 or not len(indices):
            raise ValueError('Selected public return lacks observed lineage')
        if (indices < 0).any() or (indices >= mask.size).any():
            raise ValueError('Contributor outside sampled lattice')
        total += len(indices)
        target += int(mask.ravel()[indices].sum())
    return dict(target_pixels=target, total_pixels=total, target_witnessed=target > 0)
