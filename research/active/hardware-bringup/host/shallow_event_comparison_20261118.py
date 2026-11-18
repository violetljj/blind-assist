"""Score the registered shallow-obstacle readouts using frozen events and empty walks."""

import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[3] / 'artifacts.local/work/shallow-alerts-20261118'
ROSTER = HERE.parent / 'PAIRED_LOW_SITES_20261118.json'
METHODS = ('literal', 'global', 'local')
MIN_AREA = 0.005


def check_areas(row, target=False):
    for method in METHODS:
        area = row['total'][method]
        if not math.isfinite(area) or area < 0:
            raise ValueError(f'Invalid scene area on {row["i"]}: {area}')
        if target:
            overlap = row['target'][method]
            if not math.isfinite(overlap) or not 0 <= overlap <= area + 1e-10:
                raise ValueError(f'Invalid target area on {row["i"]}: {overlap}/{area}')


def validate(data, roster, registered=None):
    known = {e['event']: e for e in roster['events'] if e['history'] == 'new_facility'}
    events, windows = data['events'], data['windows']
    historical_events = {e['id']: e for e in registered['events']} if registered else {}
    historical_windows = {(w['site'], w['id']): w for w in registered['windows']} if registered else {}
    if len(events) != 8 or {e['id'] for e in events} != set(known):
        raise ValueError('Expected exactly the eight registered approaches')
    expected_windows = {}
    for event in events:
        locked = known[event['id']]
        original_hit = event['registered_old_hit'] if registered else event['old_hit']
        if (event['site'], event['category'], event['band'], set(event['objects']), original_hit) != (
            locked['site'], locked['category'], locked['height_band'], set(locked['entities']), locked['any_target_hit']['geom']
        ):
            raise ValueError(f'Roster mismatch: {event["id"]}')
        if type(event['old_hit']) is not bool or not 0 <= event['pass_old_alert_count'] <= event['pass_samples'] or event['pass_samples'] == 0:
            raise ValueError(f'Missing original approach coverage: {event["id"]}')
        obs = event['observations']
        if registered:
            prior = historical_events[event['id']]
            if (original_hit != prior['old_hit'] or event['registered_pass_old_alert_count'] != prior['pass_old_alert_count']
                    or event['pass_samples'] != prior['pass_samples'] or event['pass_old_alert_count'] > prior['pass_old_alert_count']):
                raise ValueError(f'Camera mask changed the registered population or created an old alert: {event["id"]}')
            previous = {(r['i'],r['object_id'],r['t']): r for r in prior['observations']}
            if {(r['i'],r['object_id'],r['t']) for r in obs} != set(previous):
                raise ValueError(f'Camera mask changed the near observation list: {event["id"]}')
            for r in obs:
                before = previous[r['i'],r['object_id'],r['t']]
                if (r['old_target'] and not before['old_target']) or (r['old_any'] and not before['old_any']):
                    raise ValueError('Aperture restriction created old positive support')
        if not obs or {r['object_id'] for r in obs} != set(event['objects']):
            raise ValueError(f'Missing object observations for {event["id"]}')
        if len({(r['i'],r['object_id']) for r in obs}) != len(obs):
            raise ValueError(f'Duplicate object observation in {event["id"]}')
        if any(r['old_target'] for r in obs) != event['old_hit']:
            raise ValueError(f'Original object matches disagree in {event["id"]}')
        frame_totals = {}
        for row in obs:
            if type(row['old_target']) is not bool or type(row['old_any']) is not bool or (row['old_target'] and not row['old_any']):
                raise ValueError('Invalid original alarm')
            if row['old_any'] and event['pass_old_alert_count'] == 0:
                raise ValueError('Registered silent pass contains an old alert')
            check_areas(row, target=True)
            evidence = (row['t'], row['old_any'], row['total'])
            if frame_totals.setdefault(row['i'], evidence) != evidence:
                raise ValueError('Two targets at the same time have conflicting scene measurements')
        for window in locked['windows']:
            if window['accepted']:
                expected_windows[(event['site'], window['id'])] = window
    pass_count_key = 'registered_pass_old_alert_count' if registered else 'pass_old_alert_count'
    if sum(e[pass_count_key] == 0 for e in events) != 5:
        raise ValueError('The five original silent approaches do not reproduce')
    if len(windows) != 8 or {(w['site'], w['id']) for w in windows} != set(expected_windows):
        raise ValueError('Expected the same eight distinct empty windows')
    for window in windows:
        locked = expected_windows[(window['site'], window['id'])]
        rows = window['rows']
        if len(rows) != locked['observed_samples'] or sorted(r['i'] for r in rows) != sorted(locked['frames']):
            raise ValueError(f'Control observations changed: {window["id"]}')
        if registered:
            prior_rows = {r['i']: r for r in historical_windows[window['site'], window['id']]['rows']}
            for row in rows:
                before = prior_rows[row['i']]
                if (row['t'] != before['t'] or row.get('registered_old', before['old']) != before['old']
                        or (row['old'] and not before['old'])):
                    raise ValueError(f'Control ID or masked old alarm inconsistent with source: {window["id"]}')
        elif any(r['old'] for r in rows) != locked['any_alarm']['geom']:
            raise ValueError(f'Original control alarm changed: {window["id"]}')
        for row in rows:
            if type(row['old']) is not bool:
                raise ValueError('Unknown original control alarm')
            check_areas(row)
    if sum(len(w['rows']) for w in windows) != 55 or (not registered and sum(r['old'] for w in windows for r in w['rows']) != 9):
        raise ValueError('Control samples or original burden changed')


def match_threshold(windows, method):
    """The lowest supported float that excludes every old-negative calibration timestamp."""
    maximum = max(r['total'][method] for w in windows for r in w['rows'] if not r['old'])
    return max(MIN_AREA, math.nextafter(maximum, math.inf))


def score_event(event, method, cutoff):
    targets, necessary, old_targets, alert_times = set(), set(), set(), set()
    first_strict = None
    for row in sorted(event['observations'], key=lambda r: (r['t'], r['object_id'])):
        if row['old_target']:
            old_targets.add(row['object_id'])
        any_alert = row['old_any']
        if method != 'old':
            total, target = row['total'][method], row['target'][method]
            extension = total >= cutoff
            any_alert |= extension
            if extension and target >= cutoff:
                targets.add(row['object_id'])
                if first_strict is None:
                    first_strict = {'i': row['i'], 't': row['t'], 'object_id': row['object_id']}
            if not row['old_any'] and extension and target >= MIN_AREA and total - target < cutoff:
                necessary.add(row['object_id'])
        if any_alert:
            alert_times.add(row['i'])
    silent = event.get('registered_pass_old_alert_count', event['pass_old_alert_count']) == 0
    return {
        'id': event['id'], 'site': event['site'], 'category': event['category'], 'band': event['band'],
        'objects': sorted(event['objects']), 'silent_before': silent, 'old_target': event.get('registered_old_hit', event['old_hit']),
        **({'camera_valid_old_target': event['old_hit'], 'camera_valid_pass_old_alert_count': event['pass_old_alert_count']}
           if 'registered_old_hit' in event else {}),
        'objects_hit': sorted(old_targets | targets), 'hit': bool(old_targets | targets),
        'recovered': silent and bool(targets), 'necessary_recovery': silent and bool(necessary),
        'necessary_or_sufficient_recovery': silent and bool(necessary | targets),
        'necessary_objects': sorted(necessary), 'first_strict_new_support': first_strict,
        'near_frames': len({r['i'] for r in event['observations']}), 'alerting_near_frames': len(alert_times),
        'silent_pass_samples': event['pass_samples'] if silent else None,
    }


def score_window(window, method, cutoff):
    alarm = [r['i'] for r in window['rows'] if r['old'] or (method != 'old' and r['total'][method] >= cutoff)]
    additional = [r['i'] for r in window['rows'] if not r['old'] and r['i'] in alarm]
    return {'id': window['id'], 'site': window['site'], 'samples': len(window['rows']),
            'alarm_samples': len(alarm), 'alarm_i': alarm, 'additional_i': additional}


def event_counts(events):
    silent = [e for e in events if e['silent_before']]
    return {
        'events': len(events), 'hits': sum(e['hit'] for e in events),
        'objects': len({(e['site'], o) for e in events for o in e['objects']}),
        'objects_hit': len({(e['site'], o) for e in events for o in e['objects_hit']}),
        'originally_detected': sum(e['old_target'] for e in events),
        'retained': sum(e['hit'] and e['old_target'] for e in events),
        'silent': len(silent), 'first_detections': sum(e['recovered'] for e in silent),
        'necessary_recoveries': sum(e['necessary_recovery'] for e in silent),
        'necessary_or_sufficient_recoveries': sum(e['necessary_or_sufficient_recovery'] for e in silent),
        'silent_passes_with_any_near_alarm': sum(e['alerting_near_frames'] > 0 for e in silent),
    }


def window_counts(windows):
    return {'windows': len(windows), 'alerting_windows': sum(w['alarm_samples'] > 0 for w in windows),
            'samples': sum(w['samples'] for w in windows), 'alerting_samples': sum(w['alarm_samples'] for w in windows)}


def score(data, method, thresholds):
    sites = sorted({e['site'] for e in data['events']})
    events = [score_event(e, method, thresholds[e['site']]) for e in data['events']]
    windows = [score_window(w, method, thresholds[w['site']]) for w in data['windows']]
    shallow = [e for e in events if e['band'] == 'shallow']
    shallow_sites = {e['site'] for e in shallow}
    return {
        'area_threshold_by_site_m2': thresholds, 'all': event_counts(events), 'shallow': event_counts(shallow),
        'shallow_by_category': {key: event_counts([e for e in shallow if e['category'] == key])
                                for key in sorted({e['category'] for e in shallow})},
        'controls': window_counts(windows), 'shallow_controls': window_counts([w for w in windows if w['site'] in shallow_sites]),
        'by_site': {site: {'events': event_counts([e for e in events if e['site'] == site]),
                           'controls': window_counts([w for w in windows if w['site'] == site])} for site in sites},
        'events': events, 'windows': windows,
    }


def fit_burden_cap(windows, method, cap):
    boundaries = sorted({r['total'][method] for w in windows for r in w['rows']
                         if not r['old'] and r['total'][method] >= MIN_AREA})
    for cutoff in [MIN_AREA] + [math.nextafter(x, math.inf) for x in boundaries]:
        burden = window_counts([score_window(w, method, cutoff) for w in windows])
        if burden['alerting_windows'] <= cap['alerting_windows'] and burden['alerting_samples'] <= cap['alerting_samples']:
            return cutoff
    raise ValueError('Original alarm burden alone exceeds the anchor')


def compare_alarm_locations(anchor, other):
    def identities(windows):
        return ({(w['site'], w['id']) for w in windows if w['alarm_i']},
                {(w['site'], w['id'], i) for w in windows for i in w['alarm_i']})

    result = {}
    for label, a, b in zip(('windows', 'timestamps'), identities(anchor), identities(other), strict=True):
        result[label] = {'shared': sorted(a & b), 'literal_only': sorted(a - b), 'method_only': sorted(b - a)}
    return result


def compare(data, roster, registered=None):
    if registered:
        validate(registered, roster)
    validate(data, roster, registered)
    windows = data['windows']
    sites = sorted({w['site'] for w in windows})
    old = score(data, 'old', dict.fromkeys(sites))
    result = {'old_only': old, 'raw': {}, 'matched': {}, 'leave_site_out': {}}
    for method in METHODS:
        matched = match_threshold(windows, method)
        other = {s: match_threshold([w for w in windows if w['site'] != s], method) for s in sites}
        for name, cuts in [('raw', dict.fromkeys(sites, MIN_AREA)), ('matched', dict.fromkeys(sites, matched)), ('leave_site_out', other)]:
            scored = score(data, method, cuts)
            if name == 'matched' and any(w['additional_i'] for w in scored['windows']):
                raise ValueError('Matched control traces differ from the original')
            result[name][method] = scored
    anchor = result['raw']['literal']
    posthoc = {'anchor': 'literal_at_0.005_m2', 'burden_cap': anchor['controls'], 'methods': {}}
    for method in METHODS:
        cutoff = fit_burden_cap(windows, method, anchor['controls'])
        scored = score(data, method, dict.fromkeys(sites, cutoff))
        scored['both_burden_counts_equal'] = scored['controls'] == anchor['controls']
        scored['alarm_locations_vs_literal'] = compare_alarm_locations(anchor['windows'], scored['windows'])
        posthoc['methods'][method] = scored
    result['posthoc_raw_literal_burden'] = posthoc
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera-valid', action='store_true', help='Compare aperture-restricted readouts against the original registered population')
    parser.add_argument('--registered-input', type=Path, default=RUN / 'areas.json')
    args = parser.parse_args()
    args.input = args.input or RUN / ('areas-camera-valid.json' if args.camera_valid else 'areas.json')
    args.output = args.output or RUN / ('comparison-camera-valid.json' if args.camera_valid else 'comparison.json')
    if args.camera_valid and args.output.resolve() == (RUN / 'comparison.json').resolve():
        parser.error('Write the aperture-corrected comparison separately from the preserved original')
    registered_bytes = args.registered_input.read_bytes() if args.camera_valid else None
    payload = args.input.read_bytes()
    result = compare(json.loads(payload), json.loads(ROSTER.read_bytes()), json.loads(registered_bytes) if registered_bytes else None)
    if args.camera_valid:
        result['analysis'] = 'Common camera-valid aperture; eligibility and retention anchored to the historical replay'
        result['registered_input_sha256'] = hashlib.sha256(registered_bytes).hexdigest()
    result['input_sha256'] = hashlib.sha256(payload).hexdigest()
    result['roster_sha256'] = hashlib.sha256(ROSTER.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)
