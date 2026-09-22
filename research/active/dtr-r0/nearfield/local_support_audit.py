"""Independent saved-output/native audit of the LOCAL support diagnostic.

No diagnostic, model, feature extractor, sensor simulator or metric implementation
is imported. Native depth is evaluator-only. Band cross-counts are audited on a
declared deterministic subset; contributor counts and task metrics cover all rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


QUERIES = np.asarray([[x-.3, x+.3, lo, hi]
    for lo, hi in ((.42, .9), (-.2, .42)) for x in (-.3, 0., .3)], np.float64)
ARMS = ('A_current', 'local', 'privileged_winner', 'privileged_either')
PARTITIONS = ('WINNER_OBSERVED', 'OTHER_CENTRE_ONLY_OBSERVED',
              'NATIVE_VISIBLE_ONLY', 'NO_CENTRE_OBSERVED_OR_VISIBLE')
DT = .2
NATIVE_SHAPE = (360, 640)
FOCAL = 320 / math.tan(math.radians(50))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Checks:
    def __init__(self):
        self.counts = Counter()

    def check(self, condition, category, message):
        require(bool(condition), category+': '+message)
        self.counts[category] += 1

    def same(self, actual, expected, category, message, tolerance=1e-9):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), category, message+' mapping')
            for key, value in expected.items():
                self.check(key in actual, category, message+' missing '+key)
                self.same(actual[key], value, category, message+'/'+key, tolerance)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                       category, message+' sequence length')
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.same(a, e, category, message+'/'+str(i), tolerance)
        elif isinstance(expected, float):
            self.check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance),
                category, message+f': {actual!r} != {expected!r}')
        else:
            self.check(actual == expected, category, message+f': {actual!r} != {expected!r}')


def active_journal(repo, result):
    artifact = (repo/'artifacts.local').resolve()
    name = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    require(bool(name), 'Run this audit through tools/ba.ps1 run research-ue -RunSpec')
    path = Path(name).resolve()
    require(path.is_relative_to(artifact) and path.is_file(), 'Invalid governed journal path')
    journal = read(path)
    require(journal.get('schema') == 'blindassist-asset-run-journal-v1'
            and journal.get('state') == 'running', 'Governed journal must be active')
    require(journal.get('reuse_preflight', {}).get('status') == 'PASS', 'UE admission did not pass')
    outputs = []
    for record in journal.get('outputs', []):
        out = Path(record['path'])
        outputs.append((out if out.is_absolute() else artifact/out).resolve())
    require(result.resolve() in outputs, 'Audit result is not owned by the active governed run')
    command = journal.get('command', [])
    require(command.count('--result') == 1 and command.index('--result')+1 < len(command),
            'Governed command must bind one result')
    require(Path(command[command.index('--result')+1]).resolve() == result.resolve(),
            'Governed command binds another result')
    return dict(path=str(path), run_id=journal['id'], sha256_at_start=digest(path))


def native_sample_map():
    """Integer arithmetic for floor((sample+.5)*native_size/sample_size)."""
    rows = ((2*np.arange(192, dtype=np.int64)+1)*360)//384
    cols = ((2*np.arange(256, dtype=np.int64)+1)*640)//512
    return (rows[:, None]*640+cols[None, :]).ravel()


def native_xyz(native):
    require(native.shape == NATIVE_SHAPE, 'Expected native optical Z at 360x640')
    z = np.asarray(native, np.float64)
    u = np.arange(640, dtype=np.float64)+.5
    v = np.arange(360, dtype=np.float64)+.5
    x = z*((u-320)/FOCAL)[None, :]
    y = z*((v-180)/FOCAL)[:, None]
    valid = np.isfinite(z) & (z > 0)
    return x, y, z, valid


def in_query(x, y, z, valid, query):
    left, right, top, bottom = query
    return (valid & (z >= .3) & (z <= 3.) & (x >= left) & (x <= right)
            & (y >= top) & (y <= bottom))


def compatibility_masks(x, y, z, valid, objects, camera):
    """Point/AABB compatibility only; overlaps are retained, never assigned."""
    world = (z+float(camera['x']), x+float(camera['y']), float(camera['z'])-y)
    masks = {}
    for obj in objects:
        name = obj['name']
        require(name not in masks, 'Duplicate declared object name')
        center = obj['render_bounds_center_m']
        extent = obj['render_bounds_extent_m']
        inside = valid.copy()
        for values, mid, half in zip(world, center, extent):
            inside &= (values >= mid-half-.02) & (values <= mid+half+.02)
        masks[name] = inside
    require('target' in masks, 'Declared target missing')
    total = np.sum(np.stack(list(masks.values())), axis=0)
    background = np.zeros(valid.shape, bool)
    for name, mask in masks.items():
        if name != 'target':
            background |= mask
    return dict(objects=masks, target=masks['target'], background=background,
                ambiguous=total > 1, unassigned=valid & (total == 0),
                target_only=masks['target'] & (total == 1),
                background_only=background & (total == 1))


def public_bands(tof):
    """Independent pixel-centre raster and interval/axis-aligned-prism clipping."""
    t = np.asarray(tof, np.float64)
    require(t.shape == (64, 6), 'Expected 64x6 public ToF')
    require(np.isin(t[:, 1], [0, 1]).all(), 'Invalid public ToF validity')
    distance = t[:, 0]*8
    valid = (t[:, 1] == 1) & np.isfinite(distance) & (distance >= .1) & (distance < 8)
    radius = .1+3*(.01+.02*distance)
    lower = np.where(valid, np.maximum(.1, distance-radius), 0.)
    upper = np.where(valid, distance+radius, 0.)
    py = np.arange(180, dtype=np.float64)+.5
    px = np.arange(320, dtype=np.float64)+.5
    zones = np.full((180, 320), -1, np.int16)
    for zone, box in enumerate(t[:, 2:]):
        y0, x0, y1, x1 = box*np.array([180, 320, 180, 320])
        mask = ((py[:, None] >= y0) & (py[:, None] < y1)
                & (px[None, :] >= x0) & (px[None, :] < x1))
        require(not (zones[mask] >= 0).any(), 'Overlapping public footprints')
        zones[mask] = zone
    codes = np.full((6, 180, 320), -2, np.int8)
    covered = zones >= 0
    codes[:, covered] = -1
    usable = covered & valid[np.maximum(zones, 0)]
    a, b = np.meshgrid((px-160)/(FOCAL/2), (py-90)/(FOCAL/2))
    for qi, query in enumerate(QUERIES):
        entry = np.full((180, 320), .3)
        leave = np.full((180, 320), 3.)
        for slope, low, high in ((a, query[0], query[1]), (b, query[2], query[3])):
            nonzero = slope != 0
            start = np.zeros_like(slope)
            stop = np.zeros_like(slope)
            np.divide(low, slope, out=start, where=nonzero)
            np.divide(high, slope, out=stop, where=nonzero)
            near, far = np.minimum(start, stop), np.maximum(start, stop)
            near[~nonzero] = -np.inf if low <= 0 <= high else np.inf
            far[~nonzero] = np.inf if low <= 0 <= high else -np.inf
            entry = np.maximum(entry, near)
            leave = np.minimum(leave, far)
        lo, hi = lower[np.maximum(zones, 0)], upper[np.maximum(zones, 0)]
        possible = np.maximum(entry, lo) <= np.minimum(leave, hi)+1e-12
        definite = possible & (lo >= entry-1e-12) & (hi <= leave+1e-12)
        codes[qi, usable & ~possible] = 2
        codes[qi, usable & possible & ~definite] = 1
        codes[qi, usable & definite] = 0
    return codes, zones, lower, upper


def contiguous_runs(flags):
    values = np.asarray(flags, bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(),
                    np.flatnonzero(changes == -1).tolist()))


def frame_counts(rows, arm):
    known = [r for r in rows if r['truth'] is not None]
    truth = np.asarray([r['truth'] for r in known], bool)
    alert = np.asarray([r['predictions'][arm]['alert'] for r in known], bool)
    unknown = np.asarray([r['predictions'][arm]['unknown'] for r in known], bool)
    ambiguous = np.asarray([r['predictions'][arm]['ambiguous'] for r in known], bool)
    masks = dict(TP=truth & alert, FP=~truth & alert, FN=truth & ~alert,
        TN=~truth & ~alert & ~unknown, prediction_unknown=unknown,
        prediction_unknown_positive=truth & unknown, prediction_unknown_negative=~truth & unknown,
        abstained_positive=truth & unknown & ~alert, abstained_negative=~truth & unknown & ~alert,
        ambiguous=ambiguous, ambiguous_alerts=ambiguous & alert)
    out = dict(frames=len(known), **{key: int(value.sum()) for key, value in masks.items()})
    for name, num, den in (('recall', out['TP'], int(truth.sum())),
                           ('precision', out['TP'], int(alert.sum())),
                           ('false_alert_rate_known_negative', out['FP'], int((~truth).sum()))):
        out[name] = num/den if den else None
    return out


def event_records(rows, arm):
    clips = defaultdict(list)
    for row in rows:
        clips[row['clip_id']].append(row)
    events, segments, interior_segments, gaps = [], [], [], []
    for clip_id, clip in sorted(clips.items()):
        clip.sort(key=lambda r: r['frame_in_clip'])
        truth = [r['truth'] for r in clip]
        alert = [r['predictions'][arm]['alert'] for r in clip]
        boundary = [r['boundary'] for r in clip]
        for start, stop in contiguous_runs([v is True for v in truth]):
            frame_ids = list(range(start, stop))
            hits = [i for i in frame_ids if alert[i]]
            first = hits[0] if hits else None
            event = dict(clip_id=clip_id, start_frame=start, end_frame=stop-1,
                entry_time_s=clip[start]['time_s'], sampled_duration_s=(stop-start)*DT,
                detected=bool(hits), first_alert_time_s=clip[first]['time_s'] if hits else None,
                first_alert_relative_to_entry_s=clip[first]['time_s']-clip[start]['time_s'] if hits else None,
                left_censored=start == 0, preceding_truth_unknown=start > 0 and truth[start-1] is None,
                preexisting_alert_at_entry=(alert[start-1] and alert[start]) if start else None,
                abstained_frames=sum(clip[i]['predictions'][arm]['unknown'] and not alert[i] for i in frame_ids),
                interior_frames=sum(not boundary[i] for i in frame_ids),
                interior_detected=any(not boundary[i] and alert[i] for i in frame_ids),
                boundary_frames=sum(boundary[i] for i in frame_ids))
            events.append(event)
            gaps.append(dict(clip_id=clip_id, start_frame=start, end_frame=stop-1,
                positive_frames=stop-start, alert_frames=len(hits), silent_frames=stop-start-len(hits),
                silent_runs=len(contiguous_runs([not alert[i] for i in frame_ids])),
                internal_silent_runs=len(contiguous_runs([not alert[i] for i in range(hits[0], hits[-1]+1)])) if hits else 0,
                first_alert_frame=first, last_alert_frame=hits[-1] if hits else None))
        for interior, destination in ((False, segments), (True, interior_segments)):
            flags = [y is False and a and (not interior or not b) for y, a, b in zip(truth, alert, boundary)]
            for start, stop in contiguous_runs(flags):
                item = dict(clip_id=clip_id, start_frame=start, end_frame=stop-1,
                            sampled_duration_s=(stop-start)*DT)
                if not interior:
                    item.update(start_time_s=clip[start]['time_s'], end_time_s=clip[stop-1]['time_s'], left_censored=start == 0)
                destination.append(item)
    return events, segments, interior_segments, gaps


def metric_recount(rows, arms=ARMS):
    known = [r for r in rows if r['truth'] is not None]
    out = dict(frames=len(rows), clips=len({r['clip_id'] for r in rows}),
               known_truth_frames=len(known), unknown_truth_frames=len(rows)-len(known),
               truth_coverage=len(known)/len(rows) if rows else None,
               boundary_frames=sum(r['boundary'] for r in rows), arms={})
    for arm in arms:
        events, segments, interior, _ = event_records(rows, arm)
        internal_events = [e for e in events if e['interior_frames']]
        counts = dict(frames={name: frame_counts(sub, arm) for name, sub in (
            ('all_known', known), ('interior', [r for r in known if not r['boundary']]),
            ('boundary', [r for r in known if r['boundary']]))},
            prediction_unknown_frames=sum(r['predictions'][arm]['unknown'] for r in rows),
            alerts_on_unknown_truth=sum(r['truth'] is None and r['predictions'][arm]['alert'] for r in rows),
            events=events, false_alert_segments=segments, interior_false_alert_segments=interior,
            event_count=len(events), detected_events=sum(e['detected'] for e in events),
            event_recall=sum(e['detected'] for e in events)/len(events) if events else None,
            left_censored_events=sum(e['left_censored'] for e in events),
            interior_event_count=len(internal_events),
            interior_detected_events=sum(e['interior_detected'] for e in internal_events),
            interior_event_recall=sum(e['interior_detected'] for e in internal_events)/len(internal_events) if internal_events else None,
            false_alert_segment_count=len(segments),
            false_alert_sampled_duration_s=sum(s['end_frame']-s['start_frame']+1 for s in segments)*DT,
            interior_false_alert_segment_count=len(interior),
            interior_false_alert_sampled_duration_s=sum(s['end_frame']-s['start_frame']+1 for s in interior)*DT)
        out['arms'][arm] = counts
    return out


def representative_indices(rows):
    """All new alert frames and counterparts, plus fixed start/middle/end poses."""
    by_pair = defaultdict(list)
    selected, new_alerts = set(), set()
    for i, row in enumerate(rows):
        by_pair[row['appearance_pair_id']].append(i)
        if row['predictions']['local']['alert'] and not row['predictions']['A_current']['alert']:
            new_alerts.add(i)
        if row['frame_in_clip'] in (0, 5, 11):
            selected.add(i)
    selected |= new_alerts
    for indices in by_pair.values():
        if selected.intersection(indices):
            selected.update(indices)
    return sorted(selected), sorted(new_alerts)


def inspect_lineage(native, lineage, tof, checks, identity):
    """Verify stored observed bins without replaying RNG/noise or sensor outputs."""
    offsets = np.asarray(lineage['zone_offsets'])
    sampled = np.asarray(lineage['sampled_indices'])
    native_indices = np.asarray(lineage['native_indices'])
    saved_weights = np.asarray(lineage['weights'])
    checks.check(offsets.shape == (65,) and offsets.dtype.kind in 'iu', 'lineage', identity+' offsets')
    checks.check(offsets[0] == 0 and offsets[-1] == len(sampled)
                 and (np.diff(offsets) >= 0).all(), 'lineage', identity+' offset bounds')
    checks.check(sampled.ndim == 1 and sampled.dtype.kind in 'iu'
                 and native_indices.shape == sampled.shape and saved_weights.shape == sampled.shape,
                 'lineage', identity+' vector shapes')
    checks.check(((sampled >= 0) & (sampled < 192*256)).all(), 'lineage', identity+' sampled index range')
    mapping = native_sample_map()
    checks.check(np.array_equal(native_indices, mapping[sampled]), 'lineage', identity+' exact native centres')
    checks.check(len(np.unique(sampled)) == len(sampled), 'lineage', identity+' unique contributors')
    values = np.asarray(native).ravel()[mapping]
    # Public box coordinates were normalized on the unchanged 192x256 lattice.
    scaled = np.asarray(tof[:, 2:], np.float64)*[192, 256, 192, 256]
    boxes = np.rint(scaled).astype(np.int64)
    checks.check(np.allclose(scaled, boxes, rtol=0, atol=2e-5), 'lineage', identity+' integral source boxes')
    observed_bins = []
    for zone, (y0, x0, y1, x1) in enumerate(boxes):
        sl = slice(int(offsets[zone]), int(offsets[zone+1]))
        actual = sampled[sl]
        observed = bool(tof[zone, 1] == 1)
        checks.check(observed == bool(len(actual)), 'lineage', identity+f' observed zone {zone}')
        if not observed:
            observed_bins.append(None)
            continue
        checks.check(0 <= y0 < y1 <= 192 and 0 <= x0 < x1 <= 256,
                     'lineage', identity+f' source box {zone}')
        candidates = np.asarray([y*256+x for y in range(y0, y1) for x in range(x0, x1)], np.int64)
        depth = values[candidates]
        eligible = np.isfinite(depth) & (depth >= .1) & (depth < 8.)
        candidates, depth = candidates[eligible], depth[eligible]
        checks.check(len(depth) >= 4, 'lineage', identity+f' observed eligible count {zone}')
        # Keep the original float32 bin/weight arithmetic, but independently sum
        # each bin and select its lexicographically first maximum.
        bins = np.minimum(np.floor(depth/np.asarray(.1, dtype=depth.dtype)).astype(np.int64), 79)
        weights = (1/np.maximum(depth, np.asarray(.3, dtype=depth.dtype))**2).astype(np.float64)
        energy = np.zeros(80, np.float64)
        for b, w in zip(bins, weights):
            energy[b] += w
        winner = int(max(range(80), key=lambda b: (energy[b], -b)))
        expected = candidates[bins == winner]
        expected_weights = weights[bins == winner]
        checks.check(np.array_equal(actual, expected), 'lineage', identity+f' full winning-bin membership {zone}')
        checks.check(np.allclose(saved_weights[sl], expected_weights, rtol=1e-6, atol=1e-12),
                     'lineage', identity+f' inverse-square weights {zone}')
        observed_bins.append(dict(zone_id=zone, winner_bin=winner, contributors=len(expected),
                                  weight_sum=float(expected_weights.sum())))
    return native_indices, observed_bins


def compatibility_counts(masks, selection, valid):
    # Saved categories distinguish target from the union of background boxes.
    # They do not distinguish multiple compatible background objects.
    return dict(target_only=int((selection & masks['target'] & ~masks['background']).sum()),
        background_only=int((selection & masks['background'] & ~masks['target']).sum()),
        both=int((selection & masks['target'] & masks['background']).sum()),
        unassigned=int((selection & masks['unassigned']).sum()))


def independent_band_counts(native, tof, objects, camera):
    codes, zones, lower, upper = public_bands(tof)
    x, y, z, valid = native_xyz(native)
    masks = compatibility_masks(x, y, z, valid, objects, camera)
    expanded_zones = np.repeat(np.repeat(zones, 2, axis=0), 2, axis=1)
    ranges = (valid & (expanded_zones >= 0)
              & (z >= lower[np.maximum(expanded_zones, 0)])
              & (z <= upper[np.maximum(expanded_zones, 0)]))
    denominator = max(1, int((zones >= 0).sum()))
    queries = []
    for qi, query in enumerate(QUERIES):
        inside = in_query(x, y, z, valid, query)
        bands = []
        for band in range(3):
            pixel_selection = codes[qi] == band
            selected = np.repeat(np.repeat(pixel_selection, 2, axis=0), 2, axis=1)
            pixel_count = int(pixel_selection.sum())
            bands.append(dict(band_index=band, rgb_pixels=pixel_count,
                subpixels_total=4*pixel_count, valid_native=int((selected & valid).sum()),
                invalid_native=int((selected & ~valid).sum()),
                native_query_inside=int((selected & inside).sum()),
                native_query_outside_observed=int((selected & valid & ~inside).sum()),
                regional_range_agreement=int((selected & ranges).sum()),
                regional_range_disagreement=int((selected & valid & ~ranges).sum()),
                query_inside_and_range_agreement=int((selected & inside & ranges).sum()),
                reconstructed_pixel_fraction=pixel_count/denominator,
                compatibility_counts=compatibility_counts(masks, selected, valid),
                native_query_inside_compatibility=compatibility_counts(masks, selected & inside, valid),
                regional_range_agreement_compatibility=compatibility_counts(masks, selected & ranges, valid)))
        queries.append(bands)
    return codes, queries


def independently_compare(rows, baseline, candidate):
    added_tp, lost_tp, added_fp, removed_fp = [], [], [], []
    for row in rows:
        old, new = row['predictions'][baseline]['alert'], row['predictions'][candidate]['alert']
        if row['truth'] is True:
            if new and not old:
                added_tp.append(row['id'])
            elif old and not new:
                lost_tp.append(row['id'])
        elif row['truth'] is False:
            if new and not old:
                added_fp.append(row['id'])
            elif old and not new:
                removed_fp.append(row['id'])
    old_events = event_records(rows, baseline)[0]
    new_events = event_records(rows, candidate)[0]
    onset = []
    for old, new in zip(old_events, new_events):
        require((old['clip_id'], old['start_frame'], old['end_frame']) ==
                (new['clip_id'], new['start_frame'], new['end_frame']), 'Event identity disagreement')
        old_time, new_time = old['first_alert_time_s'], new['first_alert_time_s']
        onset.append(dict(clip_id=old['clip_id'], start_frame=old['start_frame'], end_frame=old['end_frame'],
            baseline_first_s=old_time, candidate_first_s=new_time,
            delta_s=new_time-old_time if old_time is not None and new_time is not None else None,
            event_lost=old_time is not None and new_time is None,
            event_gained=old_time is None and new_time is not None))
    return dict(added_tp=added_tp, lost_tp=lost_tp, added_fp=added_fp, removed_fp=removed_fp,
        onset=onset, earlier=sum(r['delta_s'] is not None and r['delta_s'] < -1e-9 for r in onset),
        delayed=sum(r['delta_s'] is not None and r['delta_s'] > 1e-9 for r in onset),
        events_lost=sum(r['event_lost'] for r in onset), events_gained=sum(r['event_gained'] for r in onset))


def synthetic_checks():
    """Pure analytic fixtures only. Never reads scientific assets."""
    checks = Checks()
    mapping = native_sample_map().reshape(192, 256)
    checks.check(mapping[0, 0] == 1 and mapping[-1, -1] == 359*640+638,
                 'synthetic', 'native sample corner locations')
    checks.check(len(np.unique(mapping)) == 192*256, 'synthetic', 'sample map bijection to selected rays')
    native = np.full(NATIVE_SHAPE, np.nan, np.float32)
    native[0:2, 0:2] = [[1., 2.], [3., np.nan]]
    x, y, z, valid = native_xyz(native)
    checks.check(valid.sum() == 3 and z[0, 1] == 2 and z[1, 0] == 3,
                 'synthetic', 'four subpixels retain distinct optical depths')
    q = np.array([-.3, .3, -.2, .9])
    selected = in_query(np.array([.3, .30001]), np.zeros(2), np.ones(2), np.ones(2, bool), q)
    checks.check(selected.tolist() == [True, False], 'synthetic', 'closed prism boundary')
    tof = np.array([[.25, 1., y/8, x/8, (y+1)/8, (x+1)/8]
                    for y in range(8) for x in range(8)], np.float32)
    codes, zones, low, high = public_bands(tof)
    checks.check(np.isin(codes, [0, 1, 2]).all() and (zones >= 0).all(),
                 'synthetic', 'disjoint bands exhaust valid public footprint')
    checks.check(codes[4, 90, 160] == 0 and codes[1, 90, 160] == 2,
                 'synthetic', 'centre ray is definite HEAD and outside BODY')
    checks.check(low[0] == 1.75 and high[0] == 2.25, 'synthetic', 'unchanged regional interval')
    tof[0, 1] = 0.
    invalid_codes = public_bands(tof)[0]
    checks.check((invalid_codes[:, zones == 0] == -1).all(), 'synthetic', 'invalid zone is not outside band')
    objects = [dict(name=name, render_bounds_center_m=[1, 0, 0], render_bounds_extent_m=[.1, .1, .1])
               for name in ('target', 'background')]
    masks = compatibility_masks(np.array([0., 2.]), np.zeros(2), np.ones(2), np.ones(2, bool),
                                objects, dict(x=0, y=0, z=0))
    checks.check(masks['ambiguous'].tolist() == [True, False]
                 and masks['unassigned'].tolist() == [False, True]
                 and not masks['target_only'].any(), 'synthetic', 'overlap is ambiguous, not first-owner')
    checks.check(contiguous_runs([True, True, False, True]) == [(0, 2), (3, 4)],
                 'synthetic', 'inclusive sampled run endpoints')
    rows = []
    for i, (truth, old, new) in enumerate(zip([False, True, True, True, False],
                                            [False, True, False, True, True],
                                            [False, False, False, True, False])):
        preds = {arm: dict(alert=new if arm.startswith('privileged') else old, unknown=True,
                          ambiguous=new if arm.startswith('privileged') else old) for arm in ARMS}
        rows.append(dict(id=str(i), clip_id='synthetic', frame_in_clip=i, time_s=i*DT,
                         truth=truth, boundary=False, predictions=preds))
    metric = metric_recount(rows)
    checks.same(metric['arms']['A_current']['frames']['all_known']['TN'], 0, 'synthetic', 'UNKNOWN is not TN')
    checks.same(metric['arms']['A_current']['false_alert_sampled_duration_s'], .2, 'synthetic', 'false sampled duration')
    checks.same(event_records(rows, 'A_current')[3][0]['internal_silent_runs'], 1, 'synthetic', 'within-event gap')
    comparison = independently_compare(rows, 'local', 'privileged_winner')
    checks.same(comparison['lost_tp'], ['1'], 'synthetic', 'lost onset frame identity')
    checks.same(comparison['delayed'], 1, 'synthetic', 'onset cost despite event retention')
    return dict(status='PASS', checks=dict(checks.counts), science_inputs_read=False)


def comparison_for_storage(rows, reference, candidate, metrics):
    direct = independently_compare(rows, reference, candidate)
    before, after = (metrics['arms'][a] for a in (reference, candidate))
    bc, ac = before['frames']['all_known'], after['frames']['all_known']
    groups = {}
    for group in sorted({r['base_group_id'] for r in rows}):
        groups[group] = sum(int(r['predictions'][candidate]['alert'])-int(r['predictions'][reference]['alert'])
            for r in rows if r['base_group_id'] == group and r['truth'] is True)
    return dict(TP_delta=ac['TP']-bc['TP'], FP_delta=ac['FP']-bc['FP'],
        recall_delta=ac['recall']-bc['recall'] if ac['recall'] is not None else None,
        false_segment_delta=after['false_alert_segment_count']-before['false_alert_segment_count'],
        lost_true_frames=direct['lost_tp'], gained_true_frames=direct['added_tp'],
        added_false_frames=direct['added_fp'], removed_false_frames=direct['removed_fp'],
        group_TP_delta=groups, improved_groups=sum(v > 0 for v in groups.values()),
        event_differences=[dict(clip_id=e['clip_id'], start_frame=e['start_frame'],
            reference_first_s=e['baseline_first_s'], candidate_first_s=e['candidate_first_s'],
            delay_s=e['delta_s'], lost=e['event_lost'], gained=e['event_gained']) for e in direct['onset']])


def gap_recount(rows, arm):
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row['clip_id']].append(row)
    records = []
    for clip_id, clip in sorted(by_clip.items()):
        clip.sort(key=lambda r: r['frame_in_clip'])
        for start, stop in contiguous_runs([r['truth'] is True for r in clip]):
            part = clip[start:stop]
            alert = np.asarray([r['predictions'][arm]['alert'] for r in part], bool)
            hits = np.flatnonzero(alert)
            intervals = []
            if len(hits):
                inside = ~alert.copy()
                inside[:hits[0]+1] = False
                inside[hits[-1]:] = False
                for left, right in contiguous_runs(inside):
                    intervals.append(dict(start_frame=start+left, end_frame=start+right-1,
                                          frames=right-left, sampled_duration_s=(right-left)*DT))
            records.append(dict(clip_id=clip_id, start_frame=start, end_frame=stop-1,
                truth_frames=len(part), alerted_true_frames=len(hits), alert_coverage=len(hits)/len(part),
                sampled_alert_coverage_s=len(hits)*DT,
                unalerted_true_ids=[r['id'] for r in part if not r['predictions'][arm]['alert']],
                leading_silence_frames=int(hits[0]) if len(hits) else len(part),
                trailing_silence_frames=len(part)-int(hits[-1])-1 if len(hits) else 0,
                internal_gap_segments=len(intervals), internal_gap_frames=sum(g['frames'] for g in intervals),
                max_internal_gap_frames=max([g['frames'] for g in intervals], default=0), internal_gaps=intervals))
    return dict(events=records, total_alerted_true_frames=sum(r['alerted_true_frames'] for r in records),
        total_internal_gap_frames=sum(r['internal_gap_frames'] for r in records),
        total_internal_gap_segments=sum(r['internal_gap_segments'] for r in records))


def summary_recount(details):
    compat = ('target_only', 'background_only', 'both', 'unassigned')
    numeric_band_keys = ('rgb_pixels', 'subpixels_total', 'valid_native', 'invalid_native',
        'native_query_inside', 'native_query_outside_observed', 'regional_range_agreement',
        'regional_range_disagreement', 'query_inside_and_range_agreement')
    queries = []
    for qi in range(6):
        values = [d['queries'][qi] for d in details]
        bands = []
        for bi, name in enumerate(('definite', 'possible_only', 'outside')):
            items = [v['bands'][bi] for v in values]
            band = dict(band_index=bi, name=name,
                        **{key: sum(b[key] for b in items) for key in numeric_band_keys})
            for key in ('compatibility_counts', 'native_query_inside_compatibility', 'regional_range_agreement_compatibility'):
                band[key] = {c: sum(b[key][c] for b in items) for c in compat}
            bands.append(band)
        queries.append(dict(query_index=qi, frame_query_denominator=len(details),
            label_invalid=sum(not v['query_label_valid'] for v in values),
            observed_supported_frames=sum(v['observed_contributor_points'] > 0 for v in values),
            observed_unsupported_frames=sum(v['observed_contributor_points'] == 0 for v in values),
            visible_frames=sum(v['native_visible_points'] > 0 for v in values),
            observed_contributor_points=sum(v['observed_contributor_points'] for v in values),
            native_visible_points=sum(v['native_visible_points'] for v in values),
            full_extent_positive_without_observed_support=sum(v['query_label_valid'] and v['query_truth']
                and v['observed_contributor_points'] == 0 for v in values),
            full_extent_negative_with_observed_support=sum(v['query_label_valid'] and not v['query_truth']
                and v['observed_contributor_points'] > 0 for v in values),
            visible_without_observed_support=sum(v['native_visible_points'] > 0
                and v['observed_contributor_points'] == 0 for v in values),
            observed_source_compatibility={c: sum(v['observed_source_compatibility'][c] for v in values) for c in compat},
            bands=bands))
    return dict(frames=len(details), frame_ids=[d['id'] for d in details],
        support_partition_counts={part: sum(d['support_partition'] == part for d in details) for part in PARTITIONS},
        queries=queries, public_tof_mismatch_frames=sum(not d['public_tof_exact_parity'] for d in details),
        band_fraction_mismatches=sum(d['band_fraction_mismatches'] for d in details),
        contributor_visible_mismatches=sum(d['contributor_visible_mismatches'] for d in details))


def _audit(repo, diagnostic, result, checks, governance, started):
    source_at_start = digest(Path(__file__))
    primary_result_at_start = digest(diagnostic/'result.json')
    seal = read(diagnostic/'input-seal.json')
    out_seal = read(diagnostic/'output-seal.json')
    primary = read(diagnostic/'result.json')
    checks.same(seal['schema'], 'local-support-input-seal-v1', 'schema', 'input seal')
    checks.same(primary['stage_status'], 'PASS', 'schema', 'completed diagnostic stage')
    checks.same(out_seal['status'], 'PASS', 'schema', 'output seal status')
    for name in ('metrics', 'frame_diagnostics', 'readout_rows', 'input_seal', 'output_seal'):
        file = diagnostic/(name.replace('_', '-')+'.json')
        checks.same(digest(file), primary[name+'_sha256'], 'seals', name)
    checks.same(digest(diagnostic/'input-seal.json'), out_seal['input_seal_sha256'], 'seals', 'sealed inputs')
    for relative, expected in out_seal['files'].items():
        path = (diagnostic/relative).resolve()
        checks.check(path.is_relative_to(diagnostic), 'seals', 'output stays in diagnostic tree')
        checks.same(digest(path), expected, 'seals', relative)
    hashed_inputs = {}
    for relative, expected in seal['input_file_hashes'].items():
        path = (repo/relative).resolve()
        actual = digest(path)
        checks.same(actual, expected, 'input_hashes', relative)
        hashed_inputs[str(path)] = actual
    for relative, expected in seal['source_hashes'].items():
        checks.same(digest(repo/relative), expected, 'source_hashes', relative)
        snapshot = diagnostic/'source-snapshot'/relative
        checks.same(digest(snapshot), expected, 'source_hashes', 'snapshot '+relative)
    paths = {key: Path(value).resolve() for key, value in seal['input_paths'].items()}
    checks.check(all(str(p) in hashed_inputs for p in paths.values()), 'input_hashes', 'every input path is sealed')
    spec, geometry = read(paths['spec']), read(paths['geometry'])
    original = read(paths['saved_rows'])
    details, rows, stored = [read(diagnostic/name) for name in
                           ('frame-diagnostics.json', 'readout-rows.json', 'metrics.json')]
    labels = dict(np.load(paths['labels'], allow_pickle=False))
    with np.load(paths['probability'], allow_pickle=False) as package:
        probabilities = package['local']
    with np.load(paths['feature'], allow_pickle=False) as package:
        features = package['local']
    tof = np.load(paths['tof'], mmap_mode='r', allow_pickle=False)
    freeze, baseline = read(paths['prediction_freeze']), read(paths['baseline'])
    checks.check(len(spec['cases']) == len(geometry) == len(original) == len(details) == len(rows) == 576,
                 'denominators', 'full 576-frame alignment')
    checks.check(tof.shape == (576, 64, 6) and features.shape == (576, 6, 961)
                 and probabilities.shape == (576, 6), 'denominators', 'original tensor shapes')
    checks.check(np.array_equal(labels['indices'], np.arange(576)) and labels['valid'].all(),
                 'denominators', 'all 3456 query labels valid')
    checks.same(seal['query_boxes'], QUERIES.tolist(), 'schema', 'fixed query boxes')
    checks.same(seal['bound_compatibility_tolerance_m'], .02, 'schema', 'descriptive AABB tolerance')
    subset, added_alerts = representative_indices(original)
    checks.same(len(added_alerts), 38, 'denominators', 'all 36 rescues and two added FP')
    subset_set = set(subset)
    clips, pairs = defaultdict(list), defaultdict(list)
    native_checks, band_checks = [], []
    category_counts = Counter()
    for i, (case, geo, old, detail, row) in enumerate(zip(spec['cases'], geometry, original, details, rows)):
        identity = old['id']
        checks.check(case['name'] == geo['id'] == detail['id'] == row['id'] == identity,
                     'identity', f'frame {i}')
        checks.check(geo['sample_index'] == old['index'] == detail['index'] == row['index'] == i,
                     'identity', identity+' index')
        for key in ('clip_id', 'frame_in_clip', 'time_s', 'appearance_pair_id', 'appearance',
                    'layer', 'layout_relation', 'base_group_id', 'type_id'):
            checks.same(detail[key], old[key], 'identity', identity+'/'+key)
            checks.same(row[key], old[key], 'identity', identity+'/readout '+key)
        clips[row['clip_id']].append(row)
        pairs[row['appearance_pair_id']].append(i)
        checks.same(case['camera'], geo['declared_camera'], 'identity', identity+' camera')
        checks.check(all(case['camera'][key] == 0 for key in ('pitch', 'yaw', 'roll')),
                     'identity', identity+' frozen camera axes')
        checks.same(case['target_name'], 'target', 'identity', identity+' target compatibility label')
        checks.same(old['query_truth'], (labels['classes'][i] < 6).tolist(), 'labels', identity+' query truth')
        checks.same(old['query_valid'], labels['valid'][i].astype(bool).tolist(), 'labels', identity+' query valid')
        truth = bool((labels['classes'][i, [1, 4]] < 6).any())
        checks.same(old['truth'], truth, 'labels', identity+' original frame truth')
        checks.same(row['truth'], truth, 'labels', identity+' readout frame truth')
        checks.same(detail['truth'], truth, 'labels', identity+' diagnostic truth')
        checks.same(row['boundary'], row['layout_relation'] == 'BOUNDARY', 'labels', identity+' boundary')
        checks.same(old['query_probabilities']['local'], probabilities[i].tolist(), 'readout', identity+' sealed scores')
        winner = 1 if probabilities[i, 1] >= probabilities[i, 4] else 4
        local_positive = bool(probabilities[i, winner] >= freeze['thresholds']['local'])
        a = bool(baseline[i]['alert'])
        unknown = bool(baseline[i]['unknown'])
        checks.same(old['predictions']['A_current']['alert'], a, 'readout', identity+' original A')
        checks.same(old['predictions']['local_standalone']['alert'], local_positive, 'readout', identity+' original cutoff')
        checks.same(old['predictions']['local']['alert'], a or local_positive, 'readout', identity+' original union')
        checks.same(detail['winner_query'], winner, 'query_selection', identity+' score-selected query with BODY tie')
        checks.same(detail['local_standalone_alert'], local_positive, 'readout', identity+' standalone flag')
        checks.same(detail['local_centre_scores'], probabilities[i, [1, 4]].tolist(), 'query_selection', identity+' centre scores')
        for key, value in old.items():
            if key != 'predictions':
                checks.same(row[key], value, 'preserved_rows', identity+'/'+key)
        for arm, value in old['predictions'].items():
            checks.same(row['predictions'][arm], value, 'preserved_rows', identity+'/'+arm)
        tags = (['local_rescue'] if truth and (a or local_positive) and not a else
                ['local_added_fp'] if not truth and (a or local_positive) and not a else
                ['local_miss'] if truth and not (a or local_positive) else [])
        checks.same(detail['cohort_tags'], tags, 'cohorts', identity)
        category_counts.update(tags)
        native_path = paths[f'native/{i:04}']
        checks.same(hashed_inputs[str(native_path)], geo['native_sha256'], 'input_hashes', identity+' native geometry seal')
        checks.same(detail['native_input_sha256'], geo['native_sha256'], 'input_hashes', identity+' detail native seal')
        lineage_path = (diagnostic/detail['lineage_path']).resolve()
        checks.check(lineage_path.is_relative_to(diagnostic/'lineage'), 'lineage', identity+' canonical path')
        checks.same(digest(lineage_path), detail['lineage_sha256'], 'lineage', identity+' lineage seal')
        with np.load(lineage_path, allow_pickle=False) as package:
            lineage = {name: package[name] for name in package.files}
        native = np.load(native_path, allow_pickle=False)
        native_indices, bins = inspect_lineage(native, lineage, tof[i], checks, identity)
        x, y, z, valid = native_xyz(native)
        masks = compatibility_masks(x, y, z, valid, geo['objects'], case['camera'])
        observed_mask = np.zeros(NATIVE_SHAPE, bool)
        observed_mask.ravel()[native_indices] = True
        checks.same(detail['native_valid_pixels'], int(valid.sum()), 'native', identity+' valid pixels')
        checks.same(detail['native_invalid_pixels'], int((~valid).sum()), 'native', identity+' invalid pixels')
        checks.same(detail['observed_contributor_points_total'], len(native_indices), 'native', identity+' observed count')
        checks.same(detail['all_observed_source_compatibility'], compatibility_counts(masks, observed_mask, valid),
                    'native_compatibility', identity+' all observed compatibility')
        checks.check(len(detail['zone_observations']) == 64, 'lineage', identity+' all zones')
        for zi, (saved_zone, independent_bin) in enumerate(zip(detail['zone_observations'], bins)):
            checks.same(saved_zone['zone_id'], zi, 'lineage', identity+' zone index')
            checks.same(saved_zone['observed'], independent_bin is not None, 'lineage', identity+' observed flag')
            checks.same(saved_zone['contributor_count'], int(lineage['zone_offsets'][zi+1]-lineage['zone_offsets'][zi]),
                        'lineage', identity+' zone contributor count')
            checks.same(saved_zone['winner_bin'], independent_bin['winner_bin'] if independent_bin else None,
                        'lineage', identity+' bin identity')
            checks.same(saved_zone['distance_m'], float(tof[i, zi, 0]*8) if independent_bin else None,
                        'lineage', identity+' stored observed distance')
        codes = lineage['band_codes']
        checks.check(codes.shape == (6, 180, 320) and np.isin(codes, [-2, -1, 0, 1, 2]).all(),
                     'band_saved_integrity', identity+' complete band code domain')
        checks.check(np.array_equal(np.isfinite(lineage['simulated_values']), tof[i, :, 1].astype(bool)),
                     'parity_receipt', identity+' saved simulated validity')
        public_valid = tof[i, :, 1].astype(bool)
        checks.check(np.array_equal(lineage['simulated_values'][public_valid], tof[i, public_valid, 0]*8),
                     'parity_receipt', identity+' saved simulated public values; RNG not rerun')
        checks.same(detail['public_tof_exact_parity'], True, 'parity_receipt', identity+' primary exact parity claim')
        checks.same(detail['band_fraction_mismatches'], 0, 'band_saved_integrity', identity+' fraction mismatches')
        checks.same(detail['contributor_visible_mismatches'], 0, 'native', identity+' visible consistency')
        footprint = int((codes[0] >= -1).sum())
        checks.same(detail['footprint_rgb_pixels'], footprint, 'band_saved_integrity', identity+' footprint')
        checks.same(detail['invalid_zone_rgb_pixels'], int((codes[0] == -1).sum()), 'band_saved_integrity', identity+' invalid footprint')
        observed_counts, visible_counts = [], []
        observed_zones = np.repeat(np.arange(64), np.diff(lineage['zone_offsets']))
        for qi, query in enumerate(QUERIES):
            saved = detail['queries'][qi]
            inside = in_query(x, y, z, valid, query)
            support = inside & observed_mask
            observed_count, visible_count = int(support.sum()), int(inside.sum())
            observed_counts.append(observed_count)
            visible_counts.append(visible_count)
            is_observed_inside = inside.ravel()[native_indices]
            checks.same(saved['query_index'], qi, 'native_query', identity+' query index')
            checks.same(saved['query_bounds'], query.tolist(), 'native_query', identity+' query bounds')
            checks.same(saved['query_truth'], old['query_truth'][qi], 'labels', identity+' diagnostic query truth')
            checks.same(saved['query_label_valid'], old['query_valid'][qi], 'labels', identity+' diagnostic query validity')
            checks.same(saved['observed_contributor_points'], observed_count, 'native_query', identity+f' observed query {qi}')
            checks.same(saved['native_visible_points'], visible_count, 'native_query', identity+f' visible query {qi}')
            checks.same(saved['observed_contributor_weight'], float(lineage['weights'][is_observed_inside].sum()),
                        'native_query', identity+f' query weight {qi}')
            checks.same(saved['observed_contributing_zones'], sorted(set(observed_zones[is_observed_inside].tolist())),
                        'native_query', identity+f' contributing zones {qi}')
            checks.same(saved['observed_source_compatibility'], compatibility_counts(masks, support, valid),
                        'native_compatibility', identity+f' observed query {qi}')
            checks.same(saved['visible_source_compatibility'], compatibility_counts(masks, inside, valid),
                        'native_compatibility', identity+f' visible query {qi}')
            for bi, band in enumerate(saved['bands']):
                fraction = int((codes[qi] == bi).sum())/max(1, footprint)
                checks.same(band['rgb_pixels'], int((codes[qi] == bi).sum()), 'band_saved_integrity', identity+' band size')
                checks.same(band['subpixels_total'], band['rgb_pixels']*4, 'band_saved_integrity', identity+' four native subpixels')
                checks.same(band['saved_pixel_fraction'], float(features[i, qi, 910+15*bi+14]),
                            'band_saved_integrity', identity+' frozen feature fraction')
                checks.same(band['reconstructed_pixel_fraction'], fraction, 'band_saved_integrity', identity+' stored-code fraction')
                checks.check(abs(float(features[i, qi, 910+15*bi+14])-fraction) <= 1e-6,
                             'band_saved_integrity', identity+' all-frame saved fraction parity')
        other = 4 if winner == 1 else 1
        partition = (PARTITIONS[0] if observed_counts[winner] else PARTITIONS[1] if observed_counts[other]
                     else PARTITIONS[2] if visible_counts[1] or visible_counts[4] else PARTITIONS[3])
        checks.same(detail['support_partition'], partition, 'partitions', identity)
        for arm, supported in (('privileged_winner', observed_counts[winner] > 0),
                               ('privileged_either', observed_counts[1]+observed_counts[4] > 0)):
            flag = a or (local_positive and supported)
            checks.same(row['predictions'][arm], dict(alert=flag, unknown=unknown, ambiguous=flag and unknown),
                        'readout', identity+'/'+arm)
        native_checks.append(dict(index=i, id=identity, observed_contributor_points=observed_counts,
            native_visible_points=visible_counts, observed_bins=sum(v is not None for v in bins),
            actual_observed_points=len(native_indices), support_partition=partition,
            multi_object_compatible_observed_points=int((observed_mask & masks['ambiguous']).sum())))
        if i in subset_set:
            rebuilt_codes, independent_bands = independent_band_counts(native, tof[i], geo['objects'], case['camera'])
            checks.check(np.array_equal(codes, rebuilt_codes), 'independent_bands', identity+' full pixel membership')
            for qi in range(6):
                for bi in range(3):
                    checks.same(detail['queries'][qi]['bands'][bi], independent_bands[qi][bi],
                                'independent_bands', identity+f'/query{qi}/band{bi}')
            band_checks.append(dict(index=i, id=identity, queries=6, bands=18, status='PASS'))
        if (i+1) % 96 == 0:
            print(json.dumps(dict(stage='independent_local_support_audit', frames=i+1,
                                 band_subset_checked=len(band_checks), elapsed_s=time.perf_counter()-started)), flush=True)
    checks.same(len(clips), 48, 'denominators', 'complete clips')
    checks.same(len(pairs), 288, 'denominators', 'appearance pairs')
    checks.same(dict(category_counts), dict(local_rescue=36, local_added_fp=2, local_miss=28), 'cohorts', 'fixed cohort sizes')
    checks.same(sum(r['truth'] is True for r in rows), 256, 'denominators', 'positive frames')
    checks.same(sum(r['truth'] is False for r in rows), 320, 'denominators', 'negative frames')
    for clip_id, clip in clips.items():
        clip.sort(key=lambda r: r['frame_in_clip'])
        checks.same([r['frame_in_clip'] for r in clip], list(range(12)), 'identity', clip_id+' complete frame order')
        checks.same([r['time_s'] for r in clip], [i*DT for i in range(12)], 'identity', clip_id+' sampled clock')
    for pair, indices in pairs.items():
        checks.same(sorted(rows[i]['appearance'] for i in indices), ['base', 'changed'], 'pairs', pair)
        checks.check(not subset_set.intersection(indices) or set(indices).issubset(subset_set), 'pairs', pair+' subset paired')
        left, right = indices
        checks.check(np.array_equal(tof[left], tof[right]), 'pairs', pair+' public ToF equality')
        checks.same(native_checks[left]['observed_contributor_points'], native_checks[right]['observed_contributor_points'], 'pairs', pair+' support equality')
        checks.same(native_checks[left]['native_visible_points'], native_checks[right]['native_visible_points'], 'pairs', pair+' visible equality')
    metrics = metric_recount(rows)
    checks.same(stored['metrics'], metrics, 'metrics', 'full denominator')
    axes = ('appearance', 'layer', 'layout_relation', 'base_group_id')
    for axis in axes:
        values = sorted({r[axis] for r in rows})
        checks.same(sorted(stored['strata'][axis]), values, 'metrics', axis+' full groups')
        for value in values:
            checks.same(stored['strata'][axis][value], metric_recount([r for r in rows if r[axis] == value]),
                        'metrics', axis+'/'+value)
            checks.same(stored['diagnostic_strata'][axis][value], summary_recount([d for d in details if d[axis] == value]),
                        'summary_arithmetic', axis+'/'+value)
    comparisons = {}
    for candidate in ('privileged_winner', 'privileged_either'):
        for reference in ('A_current', 'local'):
            name = candidate+'_vs_'+reference
            comparisons[name] = comparison_for_storage(rows, reference, candidate, metrics)
            checks.same(stored['comparisons'][name], comparisons[name], 'comparisons', name)
    for arm in ARMS:
        checks.same(stored['event_gaps'][arm], gap_recount(rows, arm), 'event_gaps', arm)
    cohorts = dict(all=details, local_rescues=[d for d in details if 'local_rescue' in d['cohort_tags']],
        local_added_fp=[d for d in details if 'local_added_fp' in d['cohort_tags']],
        local_misses=[d for d in details if 'local_miss' in d['cohort_tags']])
    for name, group in cohorts.items():
        checks.same(stored['cohort_summaries'][name], summary_recount(group), 'summary_arithmetic', name)
    opportunity = {}
    by_id = {r['id']: r for r in rows}
    rescued = [d['id'] for d in cohorts['local_rescues']]
    added = [d['id'] for d in cohorts['local_added_fp']]
    for arm in ('privileged_winner', 'privileged_either'):
        kept = [name for name in rescued if by_id[name]['predictions'][arm]['alert']]
        lost = [name for name in rescued if not by_id[name]['predictions'][arm]['alert']]
        removed = [name for name in added if not by_id[name]['predictions'][arm]['alert']]
        versus_a = independently_compare(rows, 'A_current', arm)
        preserved = not versus_a['lost_tp'] and not versus_a['events_lost'] and not versus_a['delayed']
        opportunity[arm] = dict(status='PASS' if len(kept) >= 32 and len(removed) == 2 and preserved else 'FAIL',
            label='PRIVILEGED_OPPORTUNITY_NOT_DEPLOYABLE', rescues_before=36, rescues_retained=len(kept),
            rescue_ids_retained=kept, rescue_ids_lost=lost, added_FP_before=2, added_FP_removed=len(removed),
            added_FP_ids_removed=removed, A_preserved=preserved, A_preservation_by_OR_construction=True,
            role='PRIMARY_INFORMATION_OPPORTUNITY' if arm == 'privileged_winner' else 'DESCRIPTIVE_OTHER_CENTRE_CONTROL')
    checks.same(stored['opportunity'], opportunity, 'opportunity', 'independent frozen costs')
    checks.same(primary['opportunity'], opportunity, 'opportunity', 'terminal costs')
    checks.same(stored['hypothesis_status'], opportunity['privileged_winner']['status'], 'opportunity', 'primary hypothesis status')
    checks.same(primary['hypothesis_status'], opportunity['privileged_winner']['status'], 'opportunity', 'terminal hypothesis status')
    checks.same(primary['public_tof_exact_parity_frames'], 576, 'parity_receipt', 'reported parity denominator')
    checks.same(primary['band_fraction_checks'], 10368, 'band_saved_integrity', 'reported fraction denominator')
    checks.same(primary['band_fraction_mismatches'], 0, 'band_saved_integrity', 'reported fraction mismatches')
    checks.same(primary['denominators'], dict(frames=576, queries=3456, bands=10368), 'denominators', 'reported denominators')
    checks.same(digest(Path(__file__)), source_at_start, 'source_hashes', 'audit source unchanged')
    for relative, expected in out_seal['files'].items():
        checks.same(digest(diagnostic/relative), expected, 'end_seals', relative)
    checks.same(digest(diagnostic/'result.json'), primary_result_at_start, 'end_seals', 'terminal unchanged')
    limits = [
        'No model, simulator/RNG, primary diagnostic or original metric implementation is imported or executed.',
        'All 576 frames, 48 clips, 3456 native query counts and every stored observed bin are independently recounted.',
        'Original RNG/noisy distance replay is not independently repeated; its primary exact-parity receipt and saved values are sealed and checked.',
        'Public band membership and native subpixel cross-counts are independently reconstructed only for the declared deterministic paired subset.',
        'All-frame band fractions additionally match stored membership codes and frozen feature columns; this is not full independent all-frame band reconstruction.',
        'Target/background categories are geometric compatibility, not renderer identity; multiple background boxes remain one background category.',
        'Native optical depth authenticity is authenticated by hashes, not validated against an independent depth device or captured segmentation.',
        'Privileged query support and false-positive removal do not establish RGB learnability, free space, physical ownership or deployment value.']
    return dict(status='PASS', stage_status='PASS', hypothesis_status=opportunity['privileged_winner']['status'],
        frames=576, clips=48, queries=3456, checks=dict(checks.counts), checks_total=sum(checks.counts.values()),
        opportunity=opportunity, full_metrics=metrics, comparisons=comparisons,
        native_checks=native_checks, band_subset=dict(rule='All 38 added alert frames plus their appearance counterparts; frame indices 0,5,11 of every clip, with both appearances',
            frames=len(subset), fraction=len(subset)/576, indices=subset,
            ids=[rows[i]['id'] for i in subset], added_alert_indices=added_alerts, all_added_alerts_included=set(added_alerts).issubset(subset_set),
            checked_bands=len(subset)*18, results=band_checks),
        source_hashes={str(Path(__file__).relative_to(repo)): source_at_start,
                       **seal['source_hashes']}, diagnostic_result_sha256=digest(diagnostic/'result.json'),
        input_seal_sha256=digest(diagnostic/'input-seal.json'), output_seal_sha256=digest(diagnostic/'output-seal.json'),
        input_files_checked=len(hashed_inputs), governance=governance, limits=limits,
        backend=dict(device='CPU', framework='NumPy '+np.__version__, reason='TASK_NOT_GPU_SUITABLE',
                     executable=sys.executable, host=platform.processor()),
        elapsed_s=time.perf_counter()-started, scientific_predictions=0, model_fits=0, simulator_calls=0)


def audit(repo: Path, diagnostic: Path, result: Path):
    """Governed entry point for one immutable independent audit result."""
    repo, diagnostic, result = Path(repo).resolve(), Path(diagnostic).resolve(), Path(result).resolve()
    require(diagnostic.is_relative_to((repo/'artifacts.local').resolve()), 'Diagnostic must stay in canonical artifacts')
    require(not result.is_relative_to(diagnostic), 'Audit output must not modify the diagnostic input tree')
    require(not result.exists(), 'Preserve completed or failed audit output')
    governance = active_journal(repo, result)
    started, checks = time.perf_counter(), Checks()
    result.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(repo))
        from tools.research_backend import BackendCandidate, DeviceObservation, select_backend
        select_backend('scalar-scoring', cpu=BackendCandidate('independent-local-support-audit-cpu', 'cpu',
            lambda: len(QUERIES),
            lambda _: DeviceObservation('cpu', platform.processor() or 'host CPU', 'NumPy '+np.__version__)),
            cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=result.parent/'backend.json',
            capabilities=dict(workload='Independent saved-output/native membership and scalar recount',
                              model_inference=False, simulator_calls=0, fitting=False))
        answer = _audit(repo, diagnostic, result, checks, governance, started)
        answer['backend_record_sha256'] = digest(result.parent/'backend.json')
    except Exception as exc:
        answer = dict(status='FAIL', stage_status='FAIL', error=str(exc),
            exception_type=type(exc).__name__, traceback=traceback.format_exc(),
            checks=dict(checks.counts), checks_total=sum(checks.counts.values()),
            governance=governance, elapsed_s=time.perf_counter()-started,
            source_sha256=digest(Path(__file__)))
        with result.open('x', encoding='utf-8') as stream:
            json.dump(answer, stream, indent=2, allow_nan=False)
        raise
    with result.open('x', encoding='utf-8') as stream:
        json.dump(answer, stream, indent=2, allow_nan=False)
    print(json.dumps({key: answer[key] for key in ('status', 'hypothesis_status', 'frames', 'checks_total', 'elapsed_s')}), flush=True)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo, args.diagnostic, args.result)
