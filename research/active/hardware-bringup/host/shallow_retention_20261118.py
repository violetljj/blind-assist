"""Join independently checked far-target alarms to frozen near-target and empty-route readouts."""

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ('direct', 'edge')
METHOD = '3f2449b480cdcbb459c8d5756220daf69db026f06b0771ea36892085d0bc4a62'
WORK = Path(__file__).resolve().parents[4] / 'artifacts.local/work'


def read(path):
    return json.loads(path.read_bytes())


def duration(start, end):
    return round(end['t'] - start['t'], 4)


def near_readout(trace, arm):
    """Describe sampled near evidence. A missing reference is a gap, never a zero."""
    misses = [f for f in trace if f[arm] is False]
    hits = [f for f in trace if f[arm] is True]
    if not trace or len(hits) + len(misses) != len(trace):
        return dict(status='unknown', all_hit=None, any_hit=bool(hits) or None)
    first = next((i for i, f in enumerate(trace) if not f[arm]), None)
    recovery = next((f for f in trace[first+1:] if f[arm]), None) if first is not None else None
    # Split at every gap in the 5 Hz source; do not infer how long a lost alarm continued through a gap.
    runs = []
    for i, f in enumerate(trace):
        if not f[arm]:
            if not runs or i == 0 or trace[i-1][arm] or f['i'] != trace[i-1]['i']+1:
                runs.append([])
            runs[-1].append(f)
    intervals = []
    by_i = {f['i']:f for f in trace}
    for run in runs:
        after = by_i.get(run[-1]['i']+1)
        intervals.append(dict(first_i=run[0]['i'], last_i=run[-1]['i'],
                              sampled_span_s=duration(run[0], run[-1]),
                              to_next_hit_s=duration(run[0], after) if after and after[arm] else None))
    return dict(status='always_hit' if not misses else 'always_missed' if not hits else 'intermittent',
                any_hit=bool(hits), all_hit=not misses, first_miss_i=trace[first]['i'] if first is not None else None,
                first_recovery_i=recovery['i'] if recovery else None, last_eligible_hit=trace[-1][arm],
                missed_intervals=intervals)


def control(w):
    if 'trace' in w:
        trace = w['trace']
        complete = w['complete']
    else:
        trace = [dict(i=i, **{a:i in w['alarms'][a] for a in ARMS}) for i in w['frames']]
        complete = True
    if len({f['i'] for f in trace}) != len(trace) or any(a not in f for f in trace for a in ARMS):
        raise ValueError('Invalid empty-window readout')
    return dict(id=w['id'], site=w['site'], frames=[f['i'] for f in trace],
                complete=complete and all(type(f[a]) is bool for f in trace for a in ARMS),
                alarms={a:[f['i'] for f in trace if f[a]] for a in ARMS},
                runs={a:sum(bool(f[a]) and (i==0 or not trace[i-1][a]) for i,f in enumerate(trace)) for a in ARMS})


def empty_cost(windows):
    return dict(windows=len(windows), samples=sum(len(w['frames']) for w in windows),
                traces_equal=all(w['complete'] and w['alarms']['direct']==w['alarms']['edge'] for w in windows)
                             if windows else None,
                **{a:dict(windows=sum(bool(w['alarms'][a]) for w in windows),
                          frames=sum(len(w['alarms'][a]) for w in windows), runs=sum(w['runs'][a] for w in windows))
                   for a in ARMS})


def analyze(name, sources, far_rows):
    far = {(r['event'],r['object_id']):r for r in far_rows}
    if len(far) != len(far_rows):
        raise ValueError(f'Duplicate far reference in {name}')
    windows = {}
    objects = []
    for source in sources:
        for w in source['controls']:
            k = (w['site'],w['id'])
            if k in windows:
                raise ValueError(f'Duplicate control {k}')
            windows[k] = control(w)
        for e in source['events']:
            if e['category']!='wheel_stop':
                continue
            for oid in e['objects']:
                proof = far.pop((e['event'],oid))
                trace = [dict(i=f['i'],t=f['t'],forward_m=t['forward_median_m'],direct=t['direct'],edge=t['edge'])
                         for f in e['target_observations'] for t in f['targets'] if t['object_id']==oid]
                for f in trace:
                    if not .29999<=f['forward_m']<=1.50001 or (f['direct'] is True and f['edge'] is False):
                        raise ValueError('Invalid frozen near readout')
                for a in ARMS:
                    p = proof[a]
                    if p['status'] not in ('hit','miss','unknown'):
                        raise ValueError(f'Unexpected far status for {oid}')
                    if p['status']=='hit' and not (trace and p['frame']<trace[0]['i'] and p['median_m']>1.5):
                        raise ValueError(f'Far observation did not precede near entry for {oid}')
                objects.append(dict(site=e['site'],event=e['event'],object_id=oid,sequence=e['sequence'],
                                    control_ids=e['control_ids'],far=proof,near=trace,
                                    eligible=all(proof[a]['status']=='hit' for a in ARMS),
                                    **{a:near_readout(trace,a) for a in ARMS}))
    if far:
        raise ValueError(f'Unmatched far rows in {name}: {list(far)}')
    def controls_for(rows):
        ids = sorted({(o['site'],wid) for o in rows for wid in o['control_ids']})
        return [windows[k] for k in ids]
    for o in objects:
        o['cost_matched'] = empty_cost(controls_for([o]))['traces_equal']
    return dict(cohort=name, objects=objects, windows=controls_for(objects),
                eligible_summary=summarize(objects,controls_for),
                all_wheel_stop_control_cost=empty_cost(controls_for(objects)))


def summarize(objects, controls_for):
    eligible = [o for o in objects if o['eligible']]
    evaluable = [o for o in eligible if all(o[a]['all_hit'] is not None for a in ARMS)]
    gains = [o for o in evaluable if o['edge']['all_hit'] and not o['direct']['all_hit']]
    losses = [o for o in evaluable if o['direct']['all_hit'] and not o['edge']['all_hit']]
    return dict(objects=len(eligible), evaluable=len(evaluable), walks=len({(o['sequence'],o['event']) for o in eligible}),
                full_near_gained=len(gains), full_near_lost=len(losses),
                walks_with_a_gained_object=sorted({o['event'] for o in gains}),
                arms={a:dict(full_near=sum(o[a]['all_hit'] for o in evaluable),
                             any_near=sum(o[a]['any_hit'] for o in evaluable),
                             at_last_sample=sum(o[a]['last_eligible_hit'] for o in evaluable),
                             hit_frames=sum(f[a] for o in evaluable for f in o['near'])) for a in ARMS},
                eligible_object_frames=sum(len(o['near']) for o in evaluable),
                paired_controls=empty_cost(controls_for(eligible)))


def run(work):
    old = work/'shallow-lead-20261118'
    new = work/'shallow-retention-20261118'
    paths = [old/'august-first-alerts.json',old/'previous/first-alerts.json',old/'previous/transfer-first-alerts.json']
    aug, prior, transfer = map(read, paths)
    if not all(j['method_sha256']==METHOD and j['threshold_m2']==.005 for j in (aug,prior,transfer)):
        raise ValueError('Unexpected prior method or threshold')
    groups = dict(august=aug['sources'], development=[prior['development']], transfer=transfer['sources'])
    cohorts = []
    for name,sources in groups.items():
        path = new/name/'far-status.json'
        paths.append(path)
        cohorts.append(analyze(name,sources,read(path)))
    all_objects = [o for c in cohorts for o in c['objects']]
    all_windows = [w for c in cohorts for w in c['windows']]
    index = {(w['site'],w['id']):w for w in all_windows}
    if len(index)!=len(all_windows):
        raise ValueError('Empty window counted in multiple cohorts')
    def controls_for(rows):
        keys = sorted({(o['site'],wid) for o in rows for wid in o['control_ids']})
        return [index[k] for k in keys]
    pure_far = [o for o in all_objects if all(o['far'][a].get('pure_far') for a in ARMS)]
    multi = [o for o in all_objects if any(y['i']==x['i']+1 for x,y in zip(o['near'],o['near'][1:]))]
    for evidence in ('august/retention-frames.json', 'development/replay.json', 'transfer/drop-traces.json',
                     'transfer/outbound-white-adjacent.json', 'aria/source-check.json'):
        paths.append(new/evidence)
    return dict(method_sha256=METHOD, threshold_m2=.005, cohorts=cohorts,
                inventory=dict(object_approaches=len(all_objects),
                               walks=len({(o['sequence'],o['event']) for o in all_objects})),
                pooled=summarize(all_objects,controls_for),
                two_consecutive_near_samples=summarize(multi,controls_for),
                far_p10_over_1p5_m=summarize(pure_far,controls_for),
                all_original_wheel_stop_control_cost=empty_cost(all_windows),
                input_sha256={p.relative_to(work).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(run(args.work),indent=1,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
