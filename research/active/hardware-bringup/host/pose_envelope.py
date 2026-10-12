"""Measured-pose validity gate for the ToF alert stack.

The current joint stack (cnh_pose_floor_dr_dev_20261012 --noise) was trained with true pitch
-25..+5 deg, public pitch error +/-5 deg and no roll randomization. Outside that envelope its
alerts are extrapolation: the 2026-10-12 phone captures looking at a ceiling (+12..+70 deg)
produced near-continuous light alerts. Frames whose measured pose leaves the envelope are
reported as POSE_OUT_OF_ENVELOPE and carry no alert (UNKNOWN, not clear).

Looking down is deliberately not gated: down to about -50 deg the new captures stayed quiet
on an empty floor and still alarmed at a near desk, and walking while looking down is normal
use. Unmeasured pose (no IMU) is not gated here; callers keep their nominal-pose policy.
"""
from __future__ import annotations

import numpy as np

PITCH_MAX_DEG = 10.0   # trained true pitch <= +5 deg plus 5 deg public error
ROLL_MAX_DEG = 25.0    # training had no roll; observed head roll is mostly within +/-10 deg
GATE_ID = 'pose-envelope-v1-pitch<=10-absroll<=25'


def in_envelope(pitch_deg, roll_deg):
    """Boolean array: True where a measured pose is inside the alert envelope (NaN -> True)."""
    p = np.asarray(pitch_deg, float); r = np.asarray(roll_deg, float)
    bad = (p > PITCH_MAX_DEG) | (np.abs(r) > ROLL_MAX_DEG)
    return ~np.where(np.isfinite(p) & np.isfinite(r), bad, False)


def apply(grade, pitch_deg, roll_deg):
    """Zero alert grades [N, queries] on frames outside the envelope; return (gated, valid)."""
    valid = in_envelope(pitch_deg, roll_deg)
    g = np.array(grade, copy=True)
    g[~valid] = 0
    return g, valid
