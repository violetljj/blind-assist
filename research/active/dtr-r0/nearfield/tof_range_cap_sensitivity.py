"""Datasheet-endpoint range-survival sensitivity for VL53L8CX-like packets.

This is not a calibrated sensor model.  DS14161 Rev 12 Table 20 reports only
typical/minimum maximum-ranging endpoints, not a per-distance probability curve.
We therefore apply deterministic caps and preserve every removed return as
invalid/UNKNOWN rather than inventing detection probabilities.
"""
from __future__ import annotations

import numpy as np

DATASHEET_URL = 'https://www.st.com/resource/en/datasheet/vl53l8cx.pdf'
DATASHEET_REVISION = 12
DATASHEET_TABLE = 20

# Continuous 8x8 at 15 Hz. Values are (inner_m, corner_m).
PROFILES = {
    'IDEAL_4M': (4.00, 4.00),
    'DARK_WHITE88_TYP': (4.00, 3.95),
    'DARK_LIGHTGRAY54_TYP': (3.30, 3.10),
    'DARK_GRAY17_TYP': (2.45, 1.95),
    '5KLUX_WHITE88_TYP': (1.55, 1.40),
    '5KLUX_LIGHTGRAY54_TYP': (1.40, 1.25),
    '5KLUX_GRAY17_TYP': (1.15, 0.95),
    '5KLUX_GRAY17_MIN': (0.90, 0.70),
}


def zone_caps(inner_m: float, corner_m: float) -> np.ndarray:
    """Interpolate center-four and corner endpoints across an 8x8 grid.

    The radial interpolation is an explicit sensitivity assumption; DS14161
    specifies endpoints, not the intervening per-zone surface.
    """
    if not (0 < corner_m <= inner_m <= 4.0):
        raise ValueError('expected 0 < corner <= inner <= 4 m')
    row, col = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    radius = np.sqrt((row - 3.5) ** 2 + (col - 3.5) ** 2)
    inner_radius = np.sqrt(0.5)
    corner_radius = np.sqrt(24.5)
    alpha = np.clip((radius - inner_radius) / (corner_radius - inner_radius), 0.0, 1.0)
    return (inner_m + alpha * (corner_m - inner_m)).reshape(64)


def apply_range_cap(ranges: np.ndarray, valid: np.ndarray,
                    inner_m: float, corner_m: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(ranges)
    valid = np.asarray(valid, dtype=bool)
    if ranges.ndim != 3 or ranges.shape[1] != 64 or valid.shape != ranges.shape:
        raise ValueError('ranges/valid must have matching [N,64,K] shapes')
    usable = valid & np.isfinite(ranges) & (ranges > 0.0) & (ranges <= 4.0)
    caps = zone_caps(inner_m, corner_m)[None, :, None]
    kept = usable & (ranges <= caps)
    removed = usable & ~kept
    return np.where(kept, ranges, 0.0), kept, removed
