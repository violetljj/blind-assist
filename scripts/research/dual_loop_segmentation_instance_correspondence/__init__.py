"""Development-only candidate instance correspondence for Failure Atlas."""

from .correspondence import (
    CorrespondenceThresholds,
    EvidenceWeights,
    annotate_frame,
    assign_one_to_one,
    bbox_iou,
    class_compatibility,
    depth_consistency,
    mask_box_metrics,
    warp_box,
    warp_mask,
)

__all__ = [
    "CorrespondenceThresholds",
    "EvidenceWeights",
    "annotate_frame",
    "assign_one_to_one",
    "bbox_iou",
    "class_compatibility",
    "depth_consistency",
    "mask_box_metrics",
    "warp_box",
    "warp_mask",
]
