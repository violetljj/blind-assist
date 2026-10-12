"""Causal corridor-neighbourhood candidates on consumed Development recordings.

No object identity or pose-compensated tracking is claimed. Occupancy is already
in the query corridor (x bins 6, z bins 8). Unknown RGB is not FREE: use the
separate coverage masks when calculating negative/quiet denominators. This file
does not modify the frozen v1/v2 scorer or its 150-pixel readout threshold.
"""
from __future__ import annotations

from collections import deque
import numpy as np

MAX_GAP_NS = 500_000_000
PIXELS = 150.0
SOFT_MIN_PIXELS = 75.0


def coverage_masks(frames):
    """All masks are bool[N,2]; incomplete fusion coverage is not a negative.

    ToF can still independently support positive output without RGB coverage.
    These masks describe current inputs, not correctness of any output.
    """
    pose = np.array([bool(f['pose_ok']) for f in frames], bool)
    rgb = pose & np.array([bool(f['available']) for f in frames], bool)
    return {k: np.repeat(v[:, None], 2, axis=1) for k, v in
            {'pose_valid': pose, 'rgb_available': rgb,
             'fusion_complete': rgb}.items()}


def _discontinuous(previous, current):
    if previous is None:
        return True
    host_delta = int(current['t_ns']) - int(previous['t_ns'])
    device_delta = int(current['device_ms']) - int(previous['device_ms'])
    seq_delta = int(current['seq']) - int(previous['seq'])
    return (not 0 < host_delta <= MAX_GAP_NS or
            not 0 < device_delta <= MAX_GAP_NS / 1_000_000 or
            seq_delta != 1)


def _dilate(occupied):
    """One x/z bin in each direction, with no wraparound at corridor edges."""
    padded = np.pad(occupied, 1)
    return np.logical_or.reduce([
        padded[x:x + 6, z:z + 8] for x in range(3) for z in range(3)])


def _pair_support(a, b):
    """Pixel mass in mutually nearby occupied cells; no identity association."""
    return (float(a[_dilate(b > 0)].sum()),
            float(b[_dilate(a > 0)].sum()))


def run_candidates(frames):
    """Return candidate names -> bool[N,2] HEAD/BODY alerts.

    baseline_2of3 exactly preserves no-reset v2 confirmation on gated raw OR.
    reset_2of3 resets at >500ms gaps, clock reversal/duplication, seq jumps,
    and rejected pose. RGB dropout clears spatial RGB history only.
    spatial_hard requires >=150 nearby pixels in the current frame and one
    previous frame in the last three, in addition to both raw RGB flags.
    spatial_soft also permits three consecutive RGB observations: each has
    >=75 mutually-nearby pixels and their total >=300. Thus stable 100px can
    be explored as a new readout, explicitly separate from the frozen 150px.
    spatial_history permits any two hard-supported frames within the last
    three observations, preserving one-frame RGB holes without persisting
    through missing RGB, rejected pose, or discontinuity.
    spatial_bounded_history instead gives unavailable RGB a None time slot:
    two supported observations within three ready slots may still alert,
    with current missing-input coverage remaining explicitly unavailable.
    All spatial arms OR independent ToF strong 2/3 confirmation; they never
    require ToF/RGB corroboration. No spatial evidence is carried across gaps.
    """
    out = {k: np.zeros((len(frames), 2), bool) for k in
           ('baseline_2of3', 'reset_2of3', 'spatial_hard', 'spatial_soft',
            'spatial_history', 'spatial_bounded_history')}
    legacy, fused, tof, rgb = (deque(maxlen=3) for _ in range(4))
    bounded_rgb = deque(maxlen=3)
    previous = None
    for i, frame in enumerate(frames):
        pose = bool(frame['pose_ok'])
        available = pose and bool(frame['available'])
        strong = np.asarray(frame['tof_strong'], bool) & pose
        raw_rgb = np.asarray(frame['rgb'], bool) & available
        raw = strong | raw_rgb
        legacy.append(raw)
        out['baseline_2of3'][i] = np.sum(legacy, axis=0) >= 2
        if _discontinuous(previous, frame) or not pose:
            fused.clear(); tof.clear(); rgb.clear(); bounded_rgb.clear()
        previous = frame
        if not pose:
            continue
        fused.append(raw)
        tof.append(strong)
        out['reset_2of3'][i] = np.sum(fused, axis=0) >= 2
        strong_confirmed = np.sum(tof, axis=0) >= 2
        hard = np.zeros(2, bool)
        soft = np.zeros(2, bool)
        history = np.zeros(2, bool)
        bounded_history = np.zeros(2, bool)
        if not available:
            rgb.clear()
            bounded_rgb.append(None)
        else:
            occ = np.asarray(frame['occupancy'], float)
            if occ.shape != (2, 6, 8) or not np.isfinite(occ).all() or (occ < 0).any():
                raise ValueError('occupancy must be finite nonnegative [2,6,8]')
            rgb.append((occ, raw_rgb))
            bounded_rgb.append((occ, raw_rgb))
            for q in range(2):
                current = rgb[-1][0][q]
                entries = list(rgb)
                for j, (left, left_raw) in enumerate(entries):
                    for right, right_raw in entries[j + 1:]:
                        left_mass, right_mass = _pair_support(left[q], right[q])
                        history[q] |= bool(left_raw[q] and right_raw[q] and
                                           min(left_mass, right_mass) >= PIXELS)
                for old, old_raw in list(rgb)[:-1]:
                    now_mass, old_mass = _pair_support(current, old[q])
                    hard[q] |= bool(raw_rgb[q] and old_raw[q] and
                                    min(now_mass, old_mass) >= PIXELS)
                if len(rgb) == 3:
                    first, middle, last = [r[0][q] for r in rgb]
                    # Middle support must belong to cells near both neighbours.
                    supported_middle = middle * _dilate(first > 0) * _dilate(last > 0)
                    masses = [float(first[_dilate(supported_middle > 0)].sum()),
                              float(supported_middle.sum()),
                              float(last[_dilate(supported_middle > 0)].sum())]
                    soft[q] = min(masses) >= SOFT_MIN_PIXELS and sum(masses) >= 2 * PIXELS
        bounded_entries = [entry for entry in bounded_rgb if entry is not None]
        for q in range(2):
            for j, (left, left_raw) in enumerate(bounded_entries):
                for right, right_raw in bounded_entries[j + 1:]:
                    left_mass, right_mass = _pair_support(left[q], right[q])
                    bounded_history[q] |= bool(left_raw[q] and right_raw[q] and
                                               min(left_mass, right_mass) >= PIXELS)
        out['spatial_hard'][i] = strong_confirmed | hard
        out['spatial_soft'][i] = strong_confirmed | hard | soft
        out['spatial_history'][i] = strong_confirmed | history
        out['spatial_bounded_history'][i] = strong_confirmed | bounded_history
    return out


def numerical_checks():
    """Synthetic mechanism checks, not real detection evidence."""
    def frame(i, pixels=200, cell=(2, 3), **kw):
        occ = np.zeros((2, 6, 8)); occ[0, cell[0], cell[1]] = pixels
        row = dict(t_ns=i * 100_000_000, device_ms=i * 100, seq=i,
                   pose_ok=True, available=True, tof_strong=[False, False],
                   rgb=[pixels >= PIXELS, False], occupancy=occ, px=[pixels, 0])
        row.update(kw)
        return row
    results = {}
    frames = [frame(1), frame(2), frame(10)]
    a = run_candidates(frames)
    assert a['baseline_2of3'][:, 0].tolist() == [False, True, True]
    assert a['reset_2of3'][:, 0].tolist() == [False, True, False]
    results['gap_reset'] = 'baseline 2 alerts; reset 1 alert'
    a = run_candidates([frame(1), frame(2), frame(3, pose_ok=False), frame(4)])
    assert not a['reset_2of3'][2:].any()
    results['pose_reset'] = '0 alerts on rejected pose and first recovered frame'
    a = run_candidates([frame(1, cell=(0, 0)), frame(2, cell=(5, 7))])
    assert a['reset_2of3'][1, 0] and not a['spatial_hard'].any()
    assert not a['spatial_history'].any()
    assert not a['spatial_bounded_history'].any()
    results['spatial_jump'] = 'OR confirmation 1 alert; spatial 0 alerts'
    a = run_candidates([frame(i, pixels=100) for i in (1, 2, 3)])
    assert not a['spatial_hard'].any() and a['spatial_soft'][:, 0].tolist() == [False, False, True]
    results['stable_weak'] = '100+100+100px: soft 1 alert; hard 0 alerts'
    frames = [frame(i, available=False, tof_strong=[True, False]) for i in (1, 2)]
    a = run_candidates(frames)
    assert a['spatial_soft'][1, 0] and not coverage_masks(frames)['fusion_complete'].any()
    results['tof_independent_missing_rgb'] = 'ToF 1 alert; complete RGB coverage 0/2 frames'
    a = run_candidates([frame(1), frame(2), frame(3, pixels=0)])
    assert a['spatial_history'][:, 0].tolist() == [False, True, True]
    assert a['spatial_hard'][:, 0].tolist() == [False, True, False]
    results['history_one_frame_hole'] = 'strong,strong,weak: history 2 alerts; hard 1 alert'
    a = run_candidates([frame(1), frame(2), frame(3, available=False)])
    assert a['baseline_2of3'][2, 0] and not a['spatial_history'][2, 0]
    results['history_dropout'] = 'missing RGB clears history; weak but available RGB does not'
    frames = [frame(1), frame(2, available=False), frame(3),
              frame(4, available=False), frame(5, available=False)]
    a = run_candidates(frames)
    assert a['spatial_bounded_history'][:, 0].tolist() == [False, False, True, False, False]
    assert not a['spatial_history'].any()
    frames = [frame(1), frame(2), frame(3, available=False)]
    a = run_candidates(frames)
    assert a['spatial_bounded_history'][2, 0] and not coverage_masks(frames)['fusion_complete'][2].any()
    results['bounded_missing_slot'] = 'one hole tolerated; two holes expire; missing current coverage stays unknown'
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(numerical_checks(), indent=2))
