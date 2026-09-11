"""Explicit observed-packet adapter; no physics simulation or fitted confidence.

Raw target order remains in MeasurementPacket. Only this model view orders usable
targets by distance, then unusable targets by original index. Quality follows the
same permutation, including observed quality on unusable returns.
"""
from dataclasses import dataclass
import numpy as np
from vl53l8cx_measurement_packet import TARGET_QUALITY, ZONE_QUALITY, MeasurementPacket


def _canonical_arrays(ranges, valid):
    ranges, valid = np.asarray(ranges), np.asarray(valid)
    if (ranges.ndim != 3 or ranges.shape != valid.shape or ranges.dtype.kind != 'f'
            or valid.dtype != np.bool_ or ranges.shape[1] not in (16, 64)
            or ranges.shape[2] not in (1, 2, 3, 4)):
        raise ValueError('Expected floating/bool [batch,16 or64,1..4] arrays')
    if not np.isfinite(ranges[valid]).all() or (ranges[valid] <= 0).any():
        raise ValueError('Usable ranges must be finite and positive')
    indices = np.argsort(np.where(valid, ranges, np.inf), axis=2, kind='stable')
    usable = np.take_along_axis(valid, indices, axis=2)
    values = np.take_along_axis(ranges, indices, axis=2)
    return np.where(usable, values, 0), usable, indices


@dataclass(frozen=True)
class ModelObservation:
    range_m: np.ndarray
    valid: np.ndarray
    original_slot: np.ndarray
    target_fields: dict
    zone_fields: dict
    measurement_id: str
    origin: str
    raw_target_order: str


def adapt_packet(packet: MeasurementPacket):
    """Independent copies; retain every configured slot, quality value and mask.

Field tuples are (float32 value, bool observed). Unavailable differs from zero.
original_slot is trace metadata, not a model feature; it restores raw ordering.
Do not interpret ordering or validity as measured confidence probabilities.
"""
    rr, vv, ii = _canonical_arrays(packet.range_m[None], packet.valid[None])
    indices = ii[0]
    target = {}
    for name in TARGET_QUALITY + ('target_status',):
        value, observed = packet.observed_field(name)
        target[name] = (np.take_along_axis(value, indices, axis=1),
                        np.take_along_axis(observed, indices, axis=1))
    zones = {name: packet.observed_field(name) for name in ZONE_QUALITY + ('nb_target_detected',)}
    return ModelObservation(rr[0], vv[0], indices, target, zones,
                            packet.measurement_id, packet.origin, packet.target_order)


def legacy_batch_inputs(ranges, valid):
    """Opt-in MZ70-compatible range/valid view; no silent resize or truncation.

Accept64 zones with one or two configured slots and usable distances <=4m.
One-slot packets receive an unavailable zero second slot. This is the frozen
model's input domain, not a hardware range guarantee. Four-target/16-zone modes
require their own model adapter; callers must not treat rejection as clearance.
"""
    ranges = np.asarray(ranges)
    if ranges.ndim != 3 or ranges.shape[1] != 64 or ranges.shape[2] not in (1, 2):
        raise ValueError('Frozen model requires64 zones and at most2 configured slots')
    rr, vv, indices = _canonical_arrays(ranges, valid)
    if (rr[vv] > 4).any():
        raise ValueError('Usable range exceeds the frozen4m model domain')
    rr = rr.astype(np.float32)
    if (rr[vv] <= 0).any():
        raise ValueError('Usable range cannot be represented in model float32')
    if ranges.shape[2] == 1:
        rr = np.pad(rr, ((0, 0), (0, 0), (0, 1)))
        vv = np.pad(vv, ((0, 0), (0, 0), (0, 1)))
        indices = np.pad(indices, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
    return dict(ranges=rr, valid=vv), indices


def legacy_packet_inputs(packet: MeasurementPacket):
    """Convenience bridge; raw packet must remain available for quality/timing.

The frozen models still consume range/valid only. No quality-aware neural gain
follows from this bridge. Perform causal RGB alignment before choosing a packet.
"""
    values, indices = legacy_batch_inputs(packet.range_m[None], packet.valid[None])
    return {key: value[0] for key, value in values.items()}, indices[0]
