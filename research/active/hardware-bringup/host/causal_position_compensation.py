"""Position lead from two consecutive, observable half-second displacements."""

import numpy as np


HISTORY_S = (1.0, 0.5, 0.0)
POSE_MAX_AGE_S = 0.25
HALF_MAX_GAP_S = 0.75
MIN_SPEED_M_S = 0.30
MAX_SPEED_M_S = 1.8
MAX_DIRECTION_CHANGE_DEG = 25.0


def compensate(time_s, xy, lead_s):
    """Return gated and ungated corrections indexed by native observation time."""
    observations = time_s[:, None] - np.array(HISTORY_S)
    indices = np.searchsorted(time_s, observations + 1e-9, side="right") - 1
    safe_indices = np.maximum(indices, 0)
    observed = time_s[safe_indices]
    intervals = np.diff(observed, axis=1)
    fresh = np.all((indices >= 0) & (observations - observed <= POSE_MAX_AGE_S + 1e-9), axis=1)
    supported = fresh & np.all((intervals > 0) & (intervals <= HALF_MAX_GAP_S + 1e-9), axis=1)

    delta = np.diff(xy[safe_indices], axis=1)
    lengths = np.linalg.norm(delta, axis=2)
    speeds = np.divide(lengths, intervals, out=np.zeros_like(lengths), where=intervals > 0)
    direction = np.divide(delta, lengths[:, :, None], out=np.zeros_like(delta), where=lengths[:, :, None] > 0)
    normalized_dot = np.sum(direction[:, 0] * direction[:, 1], axis=1)
    moving = np.all(speeds >= MIN_SPEED_M_S, axis=1)
    consistent = normalized_dot >= np.cos(np.radians(MAX_DIRECTION_CHANGE_DEG))
    lead = direction[:, 1] * (
        lead_s * np.minimum(np.min(speeds, axis=1), MAX_SPEED_M_S)
    )[:, None]
    unguarded = np.where(supported[:, None], lead, 0.0)
    guarded = np.where((supported & moving & consistent)[:, None], lead, 0.0)
    return guarded, unguarded


def compare_heading_lead(time_s, yaw_deg, gated_correction, lead_s):
    """Compare heading integration with the same recent-heading consistency check."""
    if lead_s == 0.0:
        return gated_correction.copy(), gated_correction.copy()
    yaw_rad = np.radians(yaw_deg)
    unit = np.column_stack((np.cos(yaw_rad), np.sin(yaw_rad)))
    native_steps = np.diff(time_s)
    # The orientation at each native observation applies until the next observation.
    integral = np.vstack((np.zeros(2), np.cumsum(unit[:-1] * native_steps[:, None], axis=0)))
    cut = time_s - lead_s
    previous = np.maximum(np.searchsorted(time_s, cut, side="right") - 1, 0)
    average = (integral - integral[previous] - (cut - time_s[previous])[:, None] * unit[previous]) / lead_s
    stale_gaps = np.flatnonzero(native_steps > POSE_MAX_AGE_S + 1e-9)
    through_sample = np.searchsorted(stale_gaps, np.arange(len(time_s)))
    before_start = through_sample[previous]
    enough = (cut >= time_s[0]) & (through_sample == before_start)
    distance = np.linalg.norm(gated_correction, axis=1)
    aligned = np.einsum("ij,ij->i", gated_correction, unit) >= distance * np.cos(
        np.radians(MAX_DIRECTION_CHANGE_DEG)
    )
    usable = enough & aligned & np.isfinite(average).all(axis=1)
    motion_control = np.where(usable[:, None], gated_correction, 0.0)
    heading_lead = np.where(usable[:, None], distance[:, None] * average, 0.0)
    return motion_control, heading_lead
