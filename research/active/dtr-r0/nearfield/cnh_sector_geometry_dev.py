"""Evaluator-only contact-object localization on the cached physical-yaw replay.

Box 0 defines the inherited 0.9 m reference, not the contact object's identity.
Every box with original HEAD/BODY contact surface at that reference is retained.
LEFT/CENTER/RIGHT mean sensor-local x negative/central/positive with +/-10 degree
boundaries, z forward. Horizontal head yaw and true head position are used;
fixed sensor pitch is removed, matching the head-horizontal candidate sectors. A finite
box can occupy several sectors; this is set-valued localization, not proof that
any one reported sector uniquely identifies the object. Behind-head volume is
unsupported. No observations, queries, scores, or candidate outputs are read.
"""
from __future__ import annotations

import numpy as np

SECTORS = ('LEFT', 'CENTER', 'RIGHT')
OUTPUT_FRAMES = np.arange(3, 16)
BOUNDARY_DEG = 10.


def _hull(points):
    """Convex hull of the finite box projected into sensor x/z coordinates."""
    p = sorted(set(map(tuple, np.asarray(points, float))))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    low, high = [], []
    for q in p:
        while len(low) >= 2 and cross(low[-2], low[-1], q) <= 0:
            low.pop()
        low.append(q)
    for q in reversed(p):
        while len(high) >= 2 and cross(high[-2], high[-1], q) <= 0:
            high.pop()
        high.append(q)
    return np.asarray(low[:-1]+high[:-1], float).reshape(-1, 2)


def _clip(polygon, normal):
    """Clip a convex x/z polygon to normal dot point >= 0."""
    if not len(polygon):
        return polygon
    result = []
    previous = polygon[-1]
    a = float(previous @ normal)
    for current in polygon:
        b = float(current @ normal)
        if (a >= 0) != (b >= 0):
            result.append(previous + (a/(a-b))*(current-previous))
        if b >= 0:
            result.append(current)
        previous, a = current, b
    return np.asarray(result, float).reshape(-1, 2)


def box_sector_mask(box, sensor_pose):
    """Sectors with positive projected area of this true finite box in front.

    An edge-only tangent does not add a second sector. Near/through-head boxes
    retain all intersected sectors rather than inventing a center direction.
    """
    import cnh_sequence_observed_geometry as G
    sensor_pose = np.asarray(sensor_pose, float)
    yaw = np.degrees(np.arctan2(sensor_pose[0, 2], sensor_pose[2, 2]))
    horizontal = sensor_pose.copy()
    horizontal[:3, :3] = G.S.ry(yaw)
    local = G.transform(G.corners(box), horizontal)
    polygon = _hull(local[:, (0, 2)])
    k = np.tan(np.radians(BOUNDARY_DEG))
    planes = (((0., 1.), (-1., -k)),
              ((0., 1.), (1., k), (-1., k)),
              ((0., 1.), (1., -k)))
    mask = np.zeros(3, bool)
    for i, normals in enumerate(planes):
        clipped = polygon
        for normal in normals:
            clipped = _clip(clipped, np.asarray(normal))
        if len(clipped) >= 3:
            twice_area = abs(np.sum(clipped[:, 0]*np.roll(clipped[:, 1], -1)
                                    - clipped[:, 1]*np.roll(clipped[:, 0], -1)))
            mask[i] = twice_area > 1e-12
    return mask


def localization_truth(boxes, travel, sensor, deadline, contact, *, return_details=False):
    """Return a (13,3) legal-sector mask for this inherited contact event.

    Inputs are one scene's boxes, original true travel[16,4,4], condition-specific
    true sensor[16,4,4], inherited output deadline, and inherited event boolean.
    With return_details=True return a dict with mask and per-object masks.
    Noncontact/censored scenes return an empty mask and explicit support status.
    Contact metadata inconsistent with geometry raises rather than choosing box0.
    All output times are evaluated, while the caller applies timely/deadline masks.
    """
    import cnh_sequence_observed_geometry as G
    travel, sensor = np.asarray(travel, float), np.asarray(sensor, float)
    if travel.shape != (16, 4, 4) or sensor.shape != (16, 4, 4):
        raise ValueError('Expected true travel/sensor poses at 16 input frames')
    if not np.isfinite(travel).all() or not np.isfinite(sensor).all() or not boxes:
        raise ValueError('Finite poses and nonempty physical boxes required')
    mask = np.zeros((13, 3), bool)
    details = dict(mask=mask, status='NOT_CONTACT', target_ids=[], target_count=0,
                   reference_fraction=None, reference_categories=[],
                   output_frames=OUTPUT_FRAMES.tolist())
    if not contact:
        return details if return_details else mask
    _, reference = G.deadline_reference(G.corners(boxes[0]), travel[OUTPUT_FRAMES])
    if not reference['covered']:
        raise ValueError('Inherited contact event has no covered geometry reference')
    fraction = float(reference['reference_fraction'])
    inherited = int(np.searchsorted(OUTPUT_FRAMES, fraction+1e-8, side='right')-1)
    if int(deadline) != inherited:
        raise ValueError(f'Inherited deadline {deadline} differs from geometry {inherited}')
    indices, categories = [], []
    for i, box in enumerate(boxes):
        triangles = G.S.box_mesh(box['lo'], box['hi'])
        local = G.transform(triangles, reference['reference_pose'])
        cats = [G.surface_category(local, q) for q in (0, 1)]
        if any(c.startswith('contact') for c in cats):
            indices.append(i)
            categories.append(cats)
    if not indices:
        raise ValueError('Inherited contact event has no original contact surface object')
    per_object = np.asarray([[box_sector_mask(boxes[i], sensor[f])
                              for f in OUTPUT_FRAMES] for i in indices])
    mask = per_object.any(0)
    details.update(mask=mask, status='SUPPORTED', target_ids=indices, target_count=len(indices),
                   reference_fraction=fraction, reference_categories=categories,
                   object_sector_mask=per_object, supported_frames=mask.any(-1),
                   ambiguous_frames=mask.sum(-1) > 1,
                   unsupported_frames=~mask.any(-1),
                   noncontact_object_count=len(boxes)-len(indices))
    return details if return_details else mask
