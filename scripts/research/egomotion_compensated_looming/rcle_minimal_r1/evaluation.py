from __future__ import annotations

from threading import RLock
from typing import Any

from scripts.research.egomotion_compensated_looming.rcle_minimal import (
    evaluation as r0_evaluation,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
)

from .local_expansion import fit_fixed_grid_local_affine
from .sparse_flow import detect_fixed_grid_features, track_features


IMPLEMENTATION_REVISION = "RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1"
_PATCH_LOCK = RLock()


def run_trial(
    spec: TrialSpec,
    protocol: dict[str, Any],
    include_cell_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the frozen R0 trial logic with only the versioned R1 seams replaced."""
    with _PATCH_LOCK:
        original_detect = r0_evaluation.detect_fixed_grid_features
        original_track = r0_evaluation.track_features
        original_fit = r0_evaluation.fit_fixed_grid_local_affine
        try:
            r0_evaluation.detect_fixed_grid_features = (
                detect_fixed_grid_features
            )
            r0_evaluation.track_features = track_features
            r0_evaluation.fit_fixed_grid_local_affine = (
                fit_fixed_grid_local_affine
            )
            result, runtime = r0_evaluation.run_trial(
                spec,
                protocol,
                include_cell_details=include_cell_details,
            )
        finally:
            r0_evaluation.detect_fixed_grid_features = original_detect
            r0_evaluation.track_features = original_track
            r0_evaluation.fit_fixed_grid_local_affine = original_fit
    result["implementation_revision"] = IMPLEMENTATION_REVISION
    runtime["implementation_revision"] = IMPLEMENTATION_REVISION
    return result, runtime


summarize_and_decide = r0_evaluation.summarize_and_decide
wilson_interval = r0_evaluation.wilson_interval

__all__ = [
    "IMPLEMENTATION_REVISION",
    "run_trial",
    "summarize_and_decide",
    "wilson_interval",
]
