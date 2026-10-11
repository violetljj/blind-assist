"""Real CNH input contract v5: orientation, per-frame crosstalk tail, DC baseline.

Observation-only transforms applied to real decoded CNH before the frozen chain.
They correct real-only artefacts the simulator never produced:

* orientation: raw zone order is rotated 180 degrees against the model grid
  (cnh_zone_orientation).
* crosstalk tail: bin0 (0-0.30 m) is internal crosstalk (~57-63 units, ambient
  independent, +10% after reboot). Its tail reaches bins 1-3 at ~2% / 0.6-0.9% /
  0.3-0.6% of bin0 indoors. The v2/v4 template came mostly from the balcony
  (ambient ~474) where bin1 is only ~0.5% of bin0, so indoor tails survived and,
  at 0.3-1.2 m, project only into the HEAD query. v5 predicts the tail per frame
  as ratio * bin0 with per-zone ratios estimated on donor recordings.
* DC baseline: target-free bins carry an ambient-dependent offset (+0.14..+0.23
  indoors, -2.3 under strong daylight). v5 removes a per-frame, per-zone robust
  baseline (median of normalized bins 3..15). The simulator's Skellam residual
  has zero median, so this is inert on simulated inputs.

No labels, scores or thresholds enter these functions.
"""
from __future__ import annotations

import numpy as np

from cnh_zone_orientation import to_model
from local_signed_bias_adapter import real_adapter

TAIL_BINS = (1, 2, 3)
BASELINE_BINS = slice(3, 16)


def tail_ratios(histograms):
    """Per-zone median ratio of bins 1..3 to bin0 over donor frames, raw order.

    histograms: iterable of [N, 64, 16] decoded raw-order arrays.
    """
    stacked = np.concatenate([np.asarray(h, np.float64) for h in histograms])
    if stacked.ndim != 3 or stacked.shape[1:] != (64, 16):
        raise ValueError('Donor histograms must be [N,64,16]')
    b0 = stacked[..., :1]
    if np.any(b0 <= 0):
        raise ValueError('Donor bin0 must be positive crosstalk')
    return np.median(stacked[..., 1:4] / b0, axis=0)


def subtract_tail(histogram, ratios):
    h = np.array(histogram, np.float64, copy=True)
    h[..., 1:4] -= ratios * h[..., :1]
    return h


def adapter_v5(hist, ambient, signed_bias, noise, ratios=None, orient=True, baseline=True):
    """Return model-ordered normalized evidence [N,8,8,16] (float32)."""
    h = np.asarray(hist, np.float64)
    bias = np.array(signed_bias, np.float64, copy=True)
    if ratios is not None:
        h = subtract_tail(h, ratios)
        bias[:, 1:4] = 0      # tail handled per frame; avoid double subtraction
    z = real_adapter(h, ambient, bias, noise).reshape(-1, 8, 8, 16)
    if orient:
        z = to_model(z)
    if baseline:
        z = z - np.median(z[..., BASELINE_BINS], axis=-1, keepdims=True)
    return np.ascontiguousarray(z, dtype=np.float32)
