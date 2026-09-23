"""Privileged training-only first-contact positions for fixed width/layer queries."""
import numpy as np

from contact_boundary_data import contact_labels


def metric_targets(boxes, queries):
    """Return Q axial Z values in [.3,3], or inf for right censoring at 3 m.

    The supplied horizons are validated but do not constrain the metric target.
    Boxes must use the existing camera-frame AABB convention.
    """
    query = np.asarray(queries, dtype=np.float64)
    # Validate the supplied query before replacing its horizon.
    contact_labels([], query)
    full_sweep = query.copy()
    full_sweep[:, 1] = 3.
    _, distance = contact_labels(boxes, full_sweep)
    return distance + .3
