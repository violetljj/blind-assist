"""Opt-in observed measurement packets; no native depth, truth or quality synthesis.

ULD inputs must already use host units (mm, kcps/SPAD, percent), not raw firmware
fixed point. Timing uses one explicitly shared clock and acquisition interval.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TARGET_QUALITY = ('signal_per_spad', 'range_sigma_mm', 'reflectance')
ZONE_QUALITY = ('ambient_per_spad', 'nb_spads_enabled')


def _array(value, shape, name, kind=None):
    out = np.asarray(value)
    if out.shape != shape or out.dtype.kind not in 'buif':
        raise ValueError(f'{name} must be numeric with shape {shape}')
    if kind and out.dtype.kind not in kind:
        raise ValueError(f'{name} must have dtype kind {kind}')
    return out.copy()


@dataclass(frozen=True)
class MeasurementPacket:
    measurement_id: str
    range_m: np.ndarray
    valid: np.ndarray
    origin: str
    target_order: str
    quality: dict = field(default_factory=dict)
    target_status: Optional[np.ndarray] = None
    nb_target_detected: Optional[np.ndarray] = None
    sequence_id: Optional[str] = None
    clock_id: Optional[str] = None
    acquisition_start_ns: Optional[int] = None
    acquisition_end_ns: Optional[int] = None
    received_ns: Optional[int] = None
    configured_hz: Optional[float] = None
    accepted_statuses: Optional[tuple] = None

    def __post_init__(self):
        if not self.measurement_id or self.origin not in ('DEVICE_ULD_HOST', 'SIMULATION_PROXY', 'LEGACY_GEOMETRIC_PROXY'):
            raise ValueError('Explicit measurement identity and provenance required')
        shape = np.shape(self.range_m)
        if len(shape) != 2 or shape[0] not in (16, 64) or shape[1] not in (1, 2, 3, 4):
            raise ValueError('Expected [16 or64 zones, 1..4 configured target slots]')
        ranges = _array(self.range_m, shape, 'range_m', 'f')
        valid = _array(self.valid, shape, 'valid', 'b')
        if not np.isfinite(ranges[valid]).all() or (ranges[valid] <= 0).any():
            raise ValueError('Usable ranges must be finite and positive')
        if self.target_order not in ('strongest', 'closest', 'legacy_first_last'):
            raise ValueError('Target ordering must remain explicit')
        if self.origin == 'DEVICE_ULD_HOST' and self.target_order == 'legacy_first_last':
            raise ValueError('Legacy geometric slots are not hardware target order')
        object.__setattr__(self, 'range_m', ranges)
        object.__setattr__(self, 'valid', valid)
        if self.target_status is not None:
            status = _array(self.target_status, shape, 'target_status', 'ui')
            if ((status < 0) | (status > 255)).any():
                raise ValueError('Target status must fit one ULD status byte')
            object.__setattr__(self, 'target_status', status)
        if self.nb_target_detected is not None:
            counts = _array(self.nb_target_detected, (shape[0],), 'nb_target_detected', 'ui')
            if ((counts < 0) | (counts > shape[1])).any():
                raise ValueError('Target count exceeds the configured slots')
            if (valid & (np.arange(shape[1])[None, :] >= counts[:, None])).any():
                raise ValueError('A slot beyond the reported target count cannot be usable')
            object.__setattr__(self, 'nb_target_detected', counts)
        if self.origin == 'DEVICE_ULD_HOST':
            if self.target_status is None or self.nb_target_detected is None or not self.accepted_statuses:
                raise ValueError('Device usability requires raw status, target count and declared status policy')
            if not set(self.accepted_statuses) <= {5, 6, 9}:
                raise ValueError('Device status policy must remain within statuses5/6/9')
            expected = np.isin(self.target_status, self.accepted_statuses)
            expected &= np.arange(shape[1])[None, :] < self.nb_target_detected[:, None]
            expected &= np.isfinite(ranges) & (ranges > 0)
            if not np.array_equal(valid, expected):
                raise ValueError('Usability differs from the recorded status/count policy')
        if set(self.quality) - set(TARGET_QUALITY + ZONE_QUALITY):
            raise ValueError('Only declared observed quality fields are accepted')
        quality = {}
        for name, value in self.quality.items():
            dims = shape if name in TARGET_QUALITY else (shape[0],)
            a = _array(value, dims, name)
            if np.isinf(a).any() or (a[np.isfinite(a)] < 0).any():
                raise ValueError('Quality is nonnegative, with NaN only for unavailable entries')
            quality[name] = a
        object.__setattr__(self, 'quality', quality)
        times = (self.acquisition_start_ns, self.acquisition_end_ns, self.received_ns)
        if any(t is not None for t in times):
            if not self.sequence_id or not self.clock_id or any(type(t) is not int or t < 0 for t in times):
                raise ValueError('Timing requires sequence, shared clock and all three integer timestamps')
            if not times[0] <= times[1] <= times[2]:
                raise ValueError('Acquisition and receipt timestamps must be causal')
        if self.configured_hz is not None:
            maximum = 15 if shape[0] == 64 else 60
            if not np.isfinite(self.configured_hz) or not 1 <= self.configured_hz <= maximum:
                raise ValueError('Configured rate exceeds UM3109 resolution limits')

    def observed_field(self, name):
        """Separate availability from value: absent is never a confident zero."""
        if name in TARGET_QUALITY + ZONE_QUALITY:
            value = self.quality.get(name)
            shape = self.range_m.shape if name in TARGET_QUALITY else (len(self.range_m),)
        elif name == 'target_status':
            value, shape = self.target_status, self.range_m.shape
        elif name == 'nb_target_detected':
            value, shape = self.nb_target_detected, (len(self.range_m),)
        else:
            raise ValueError('Unknown observation field')
        if value is None:
            return np.zeros(shape, np.float32), np.zeros(shape, bool)
        available = np.isfinite(value)
        return np.where(available, value, 0).astype(np.float32), available.copy()


def from_legacy(range_m, valid, *, measurement_id):
    """Wrap a static old packet without inventing time, status, counts or signal."""
    return MeasurementPacket(measurement_id, range_m, valid,
                             'LEGACY_GEOMETRIC_PROXY', 'legacy_first_last')


def from_uld(distance_mm, target_status, nb_target_detected, *, measurement_id,
             accepted_statuses=(5,), target_order='strongest', quality=None, **timing):
    """Preserve raw status/quality; status5-only usability is a declared policy.

    An explicit (5,6,9) policy may include the manual's lower-confidence returns.
    These policy choices are not calibrated confidence probabilities.
    """
    distance = np.asarray(distance_mm)
    if distance.ndim != 2 or distance.dtype.kind not in 'uif':
        raise ValueError('Reshape ULD zone-major results into [zones, configured slots]')
    status = _array(target_status, distance.shape, 'target_status', 'ui')
    counts = _array(nb_target_detected, (len(distance),), 'nb_target_detected', 'ui')
    if not accepted_statuses or not set(accepted_statuses) <= {5, 6, 9}:
        raise ValueError('Declare a nonempty policy within statuses5/6/9')
    ranges = distance.astype(np.float32) / 1000
    valid = np.isin(status, accepted_statuses) & (np.arange(distance.shape[1])[None, :] < counts[:, None])
    valid &= np.isfinite(ranges) & (ranges > 0)
    return MeasurementPacket(measurement_id, ranges, valid, 'DEVICE_ULD_HOST',
                             target_order, quality or {}, status, counts,
                             accepted_statuses=tuple(accepted_statuses), **timing)


def align_to_rgb(packets, *, sequence_id, clock_id, rgb_timestamp_ns,
                 decision_timestamp_ns, max_age_ns, previous_measurement_id=None):
    """Causal last-completed sample hold; no interpolation or invented new ToF."""
    if not sequence_id or not clock_id or any(type(t) is not int or t < 0 for t in (rgb_timestamp_ns, decision_timestamp_ns, max_age_ns)):
        raise ValueError('Explicit sequence, clock and nonnegative integer times required')
    if decision_timestamp_ns < rgb_timestamp_ns:
        raise ValueError('Decision cannot precede the RGB reference timestamp')
    eligible = [p for p in packets if p.sequence_id == sequence_id and p.clock_id == clock_id
                and p.acquisition_end_ns is not None and p.acquisition_end_ns <= rgb_timestamp_ns
                and p.received_ns <= decision_timestamp_ns]
    if not eligible:
        return dict(state='NO_ALIGNED_TOF', packet=None, age_ns=None, reused=False)
    latest = max(eligible, key=lambda p: (p.acquisition_end_ns, p.received_ns))
    age = rgb_timestamp_ns - latest.acquisition_end_ns
    return dict(state='STALE' if age > max_age_ns else 'AVAILABLE', packet=latest,
                age_ns=age, reused=latest.measurement_id == previous_measurement_id)
