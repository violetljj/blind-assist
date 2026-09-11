"""Opt-in unresolved-return sensitivity inputs, not a hardware simulator.

Both endpoints are tested separately: neither assumes that a small near target
is detectable or that the farther target has the strongest measured signal.
Inputs are the legacy geometric first/last arrays, never native labels or RGB.
"""
import numpy as np

from vl53l8cx_measurement_packet import MeasurementPacket

MODES = ('CLOSEST_REPORTED_PROXY', 'FARTHEST_REPORTED_PROXY')


def constrain(ranges, valid, mode, *, separation_m=.600):
    """Mask one existing close return; do not invent an intermediate distance.

    Keep the surviving return in its original slot. The affected-zone mask is
    diagnostic metadata and must not be supplied as a predictor feature. The
    threshold is a declared sensitivity setting, not a detection probability.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown sensitivity mode: {mode}')
    if not np.isfinite(separation_m) or separation_m <= 0:
        raise ValueError('separation_m must be finite and positive')
    ranges, valid = np.asarray(ranges), np.asarray(valid)
    if (ranges.dtype != np.float32 or valid.dtype != np.bool_
            or ranges.shape != valid.shape or ranges.ndim != 3
            or ranges.shape[1:] != (64, 2)):
        raise ValueError('Expected legacy float32/bool arrays [batch,64,2]')
    if not np.isfinite(ranges[valid]).all() or (ranges[valid] <= 0).any():
        raise ValueError('Usable distances must be positive and finite')
    dual = valid.all(axis=2)
    gap = ranges[:, :, 1].astype(np.float64) - ranges[:, :, 0].astype(np.float64)
    if (gap[dual] < 0).any():
        raise ValueError('Legacy first/last slots must be ordered; no implicit sorting')
    affected = dual & (gap < separation_m)
    result_ranges, result_valid = ranges.copy(), valid.copy()
    dropped_slot = 1 if mode == MODES[0] else 0
    result_ranges[:, :, dropped_slot][affected] = 0
    result_valid[:, :, dropped_slot][affected] = False
    return dict(ranges=result_ranges, valid=result_valid), affected


def as_measurement_packet(ranges, valid, mode, *, measurement_id, separation_m=.600):
    """Wrap one legacy case without fabricated status, quality or timestamps."""
    result, _ = constrain(np.asarray(ranges)[None], np.asarray(valid)[None], mode,
                          separation_m=separation_m)
    return MeasurementPacket(measurement_id, result['ranges'][0], result['valid'][0],
                             'SIMULATION_PROXY', 'legacy_first_last')
