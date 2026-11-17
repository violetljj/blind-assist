"""Read out measured outgoing distance after an observed change of movement direction."""

import numpy as np

from causal_position_compensation import MAX_DIRECTION_CHANGE_DEG, MIN_SPEED_M_S, POSE_MAX_AGE_S


SAMPLE_PERIOD_S = 0.5
INCOMING_CONTEXT_S = 5.0
NEAR_CORNER_M = 4.0
MAX_LATERAL_M = 1.5


def route_distance_corrections(time_s, xy, vertices, start_s, stop_s):
    """Choose an anchor from observations and replace only the longitudinal coordinate."""
    prev, corner, goal = vertices
    incoming = corner - prev
    outgoing = goal - corner
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    normal = np.array([-outgoing[1], outgoing[0]])
    targets = np.arange(start_s - INCOMING_CONTEXT_S, stop_s + 1e-6, SAMPLE_PERIOD_S)
    indices = np.searchsorted(time_s, targets + 1e-9, side="right") - 1
    safe = np.maximum(indices, 0)
    stale = (indices < 0) | (targets - time_s[safe] > POSE_MAX_AGE_S + 1e-9)
    threshold = np.cos(np.radians(MAX_DIRECTION_CHANGE_DEG))
    near = np.linalg.norm(xy[safe] - corner, axis=1) <= NEAR_CORNER_M

    incoming_seen = False
    outgoing_streak = 0
    anchor_index = None
    corrected = []
    for i in range(1, len(targets)):
        right = safe[i]
        left = safe[i - 1]
        if stale[i] or stale[i - 1] or right <= left:
            outgoing_streak = 0
            continue
        delta = xy[right] - xy[left]
        distance = np.linalg.norm(delta)
        moving = distance >= MIN_SPEED_M_S * (time_s[right] - time_s[left])
        in_along = float(delta @ incoming)
        out_along = float(delta @ outgoing)
        supports_in = moving and in_along > out_along and in_along >= threshold * distance
        supports_out = moving and out_along > in_along and out_along >= threshold * distance
        if supports_in and near[i]:
            incoming_seen = True
        outgoing_streak = outgoing_streak + 1 if supports_out else 0
        if anchor_index is None and incoming_seen and outgoing_streak == 2 and near[i - 2] and near[i]:
            anchor_index = safe[i - 2]
        cross = float((xy[right] - corner) @ normal)
        if targets[i] >= start_s and anchor_index is not None and outgoing_streak >= 2 and abs(cross) <= MAX_LATERAL_M:
            # Moving the absolute position by this constant keeps the
            # observed displacement and the perpendicular coordinate intact.
            offset = -float((xy[anchor_index] - corner) @ outgoing) * outgoing
            corrected.append((int(right), offset))
    return corrected, (float(time_s[anchor_index]) if anchor_index is not None else None)
