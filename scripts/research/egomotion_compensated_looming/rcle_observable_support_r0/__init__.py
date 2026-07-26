"""Observable-only three-frame support management for RCLE-Minimal."""

from .support_manager import (
    CURRENT_LEG_SURVIVOR,
    GEOMETRIC_FIELD_EXIT,
    OBSERVABLE_OCCLUSION,
    ORDINARY_NEW_TRACK_FAILURE,
    ObservableTrackDiagnostics,
    activated_cell_indices,
    classify_new_track_failures,
    classify_prior_survivors,
    merge_path_correspondences,
    median_centered_patch_errors,
    observable_occlusion_centers,
    select_spatial_supplements,
    track_observable_points,
)

__all__ = [
    "CURRENT_LEG_SURVIVOR",
    "GEOMETRIC_FIELD_EXIT",
    "OBSERVABLE_OCCLUSION",
    "ORDINARY_NEW_TRACK_FAILURE",
    "ObservableTrackDiagnostics",
    "activated_cell_indices",
    "classify_new_track_failures",
    "classify_prior_survivors",
    "merge_path_correspondences",
    "median_centered_patch_errors",
    "observable_occlusion_centers",
    "select_spatial_supplements",
    "track_observable_points",
]
