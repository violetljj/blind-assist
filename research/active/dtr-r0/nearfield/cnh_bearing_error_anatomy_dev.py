"""Evaluator-only, nonexclusive anatomy of a selected bearing label.

The contact target and its legal-sector mask remain evaluation inputs. These
descriptions never replace that target, modify clean/unique, or establish that a
notice supported by a competing object was useful or reasonable.

Distances are the shortest Euclidean distances from true head position to each
axis-aligned *world* box footprint in XZ. They ignore height, occlusion, and the
chosen sector; closer geometry is not causal attribution of a model response.
"""
from __future__ import annotations

import numpy as np
from cnh_sector_geometry_dev import box_sector_mask

NOT_EVALUABLE = 'NOT_EVALUABLE'


def _distance_xz(box, head_xz):
    lower = np.asarray(box['lo'], float)[[0, 2]]
    upper = np.asarray(box['hi'], float)[[0, 2]]
    nearest = np.clip(head_xz, lower, upper)
    return float(np.linalg.norm(head_xz-nearest))


def describe(boxes, target_box, sensor_pose, selected_label, target_legal):
    """Return eight nonexclusive descriptions; no label or metric is changed.

    selected_label is LEFT/CENTER/RIGHT = 0/1/2. Adjacency describes an
    unsupported selected sector immediately next to a legal target sector; a
    supported choice is not counted as an adjacent error.

    competitor_count counts all boxes except the known target. Its selected
    counterpart counts only those with finite footprint in the selected sector.
    competitor_closer compares those selected competitors to the actual target.

    A missing/out-of-range target index makes target-dependent and competitor
    descriptions NOT_EVALUABLE. A missing legal mask makes only its three
    descriptions NOT_EVALUABLE. Invalid geometry/label shapes raise instead of
    being interpreted as empty geometry. None is permitted for target metadata.
    """
    if isinstance(selected_label, (bool, np.bool_)) or selected_label not in (0, 1, 2):
        raise ValueError('Selected label must be 0, 1 or 2')
    label = int(selected_label)
    pose = np.asarray(sensor_pose, float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Finite true sensor pose[4,4] required')
    # box_sector_mask validates each physical box and uses the same horizontal
    # true-head convention as the unchanged localization evaluator.
    masks = [box_sector_mask(box, pose) for box in boxes]
    result = dict(selected_supported_by_target=NOT_EVALUABLE,
                  target_cross_boundary=NOT_EVALUABLE,
                  adjacent_to_target_sector=NOT_EVALUABLE,
                  selected_has_competitor=NOT_EVALUABLE,
                  selected_geometry_empty=not any(mask[label] for mask in masks),
                  competitor_closer=NOT_EVALUABLE,
                  competitor_count=NOT_EVALUABLE,
                  selected_competitor_count=NOT_EVALUABLE)
    known_target = (isinstance(target_box, (int, np.integer)) and
                    not isinstance(target_box, (bool, np.bool_)) and
                    0 <= int(target_box) < len(boxes))
    if not known_target:
        return result
    target = int(target_box)
    if target_legal is not None:
        legal = np.asarray(target_legal)
        if legal.shape != (3,) or legal.dtype != np.dtype(bool):
            raise ValueError('Target legal mask must be bool[3] or None')
        supported = bool(legal[label])
        result.update(selected_supported_by_target=supported,
                      target_cross_boundary=int(legal.sum()) > 1,
                      adjacent_to_target_sector=(not supported and any(
                          abs(label-int(other)) == 1 for other in np.flatnonzero(legal))))
    selected_competitors = [i for i, mask in enumerate(masks)
                            if i != target and mask[label]]
    head_xz = pose[[0, 2], 3]
    target_distance = _distance_xz(boxes[target], head_xz)
    result.update(selected_has_competitor=bool(selected_competitors),
                  competitor_count=len(boxes)-1,
                  selected_competitor_count=len(selected_competitors),
                  competitor_closer=any(_distance_xz(boxes[i], head_xz) < target_distance
                                        for i in selected_competitors))
    return result
