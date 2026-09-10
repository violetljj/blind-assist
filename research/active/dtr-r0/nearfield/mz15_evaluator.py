"""Evaluator-only contributor alignment for the MZ6 stress packet transform."""
import numpy as np


def align_stress_labels(source, query, clean_ranges, clean_valid, stress_ranges, stress_valid):
    source, query = source.copy(), query.copy()
    moved = (clean_valid.all(2) & stress_valid[:, :, 0] & ~stress_valid[:, :, 1]
             & np.isclose(stress_ranges[:, :, 0], clean_ranges[:, :, 1], atol=1e-6, rtol=0))
    source[moved, 0] = source[moved, 1]
    source[moved, 1] = False
    query[moved, 0] = query[moved, 1]
    query[moved, 1] = False
    source &= stress_valid[..., None]
    query &= stress_valid[..., None, None]
    return source, query, moved
