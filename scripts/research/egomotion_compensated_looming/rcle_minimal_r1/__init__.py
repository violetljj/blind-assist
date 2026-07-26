"""Versioned RCLE-Minimal Phase A implementation-only coverage revision R1."""

from .evaluation import run_trial, summarize_and_decide
from .local_expansion import fit_fixed_grid_local_affine
from .sparse_flow import (
    SparseTrackResult,
    detect_fixed_grid_features,
    track_features,
)

__all__ = [
    "SparseTrackResult",
    "detect_fixed_grid_features",
    "fit_fixed_grid_local_affine",
    "run_trial",
    "summarize_and_decide",
    "track_features",
]
