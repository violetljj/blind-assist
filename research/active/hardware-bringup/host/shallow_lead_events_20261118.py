"""Read frozen results without inference. Pass --source COMPARISON MANIFEST LABELS for each recording."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ARMS = ('direct', 'edge')
METHOD = '3f2449b480cdcbb459c8d5756220daf69db026f06b0771ea36892085d0bc4a62'


def frame_key(row):
    return row['seq'], row['i']


def indexed(rows, key=frame_key):
    result = {key(r):r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate observation keys')
    return result


def first_hits(frames, object_ids, restorable_timeline):
    """Separate first *observed* hit from first on an evaluable registered timeline."""
    origin = frames[0]['t']
    wanted = set(object_ids)
    targets = [[t for t in r['targets'] if t['object_id'] in wanted] for r in frames]
    referenced = {t['object_id'] for items in targets for t in items}
    eligible = [r['i'] for r,items in zip(frames,targets) if items]
    arms = {}
    for arm in ARMS:
        seen, missing, alarms, absent_mask, no_overlap = [], [], [], [], []
        for r, items in zip(frames,targets):
            any_alarm = None if r['alert'] is None else r['alert'].get(arm)
            if any_alarm:
                alarms.append(r)
            hits = [t for t in items if t[arm] is True]
            if any(t[arm] is None for t in items):
                missing.append(r)
            if hits:
                seen.append((r,hits))
            elif any_alarm and not items:
                absent_mask.append(r['i'])
            elif any_alarm and all(t[arm] is False for t in items):
                no_overlap.append(r['i'])
        first = None
        if seen:
            r, hits = seen[0]
            first = dict(frame=r['i'], source_t=r['t'],from_start_s=round(r['t']-origin,6),
                         at_first_eligible_frame=r['i']==eligible[0],
                         at_pass_start=r['i']==frames[0]['i'],
                         objects=[dict(object_id=t['object_id'],forward_median_m=t['forward_median_m']) for t in hits])
        fully_observed = restorable_timeline and referenced==wanted
        first_known = (fully_observed and first is not None and not any(r['t']<first['source_t'] for r in missing))
        complete_miss = first is None and fully_observed and not missing
        scene_complete = restorable_timeline and all(r['alert'] is not None and type(r['alert'].get(arm)) is bool for r in frames)
        arms[arm] = dict(first_target=first, first_target_time_known=first_known,
                         status='hit' if first_known else 'miss' if complete_miss else 'unknown',
                         target_miss_on_all_eligible=(False if first else True if complete_miss else None),
                         first_any_frame=alarms[0]['i'] if alarms else None,
                         first_any_observed_t=alarms[0]['t'] if alarms else None,
                         full_pass_no_scene_alarm=(False if alarms else True if scene_complete else None),
                         any_frames=[r['i'] for r in alarms], target_frames=[r['i'] for r,_ in seen],
                         missing_target_frames=[r['i'] for r in missing],
                         alarm_without_eligible_mask=absent_mask, alarm_without_qualifying_target_overlap=no_overlap)
    direct, edge = [arms[a] for a in ARMS]
    states = (direct['status'], edge['status'])
    pair = {'hit':'both_hit', 'miss':'both_missed'}.get(states[0]) if states[0]==states[1] else None
    if 'unknown' in states:
        pair = 'unknown'
    elif pair is None:
        pair = 'direct_only' if direct['status']=='hit' else 'edge_only'
    lead_s, lead_m = None, {}
    if pair=='both_hit':
        d,e = direct['first_target'], edge['first_target']
        common={x['object_id'] for x in d['objects']} & {x['object_id'] for x in e['objects']}
        if common:
            lead_s = round(d['source_t']-e['source_t'],6)
            for obj in common:
                distances=[next(t['forward_median_m'] for t in f['objects'] if t['object_id']==obj) for f in (d,e)]
                lead_m[obj]=None if None in distances else distances[1]-distances[0]
        else:
            pair='different_first_objects'
    return dict(eligible_frames=eligible, missing_reference_objects=sorted(wanted-referenced), arms=arms,
                outcome=pair, edge_lead_s=lead_s, edge_lead_m_by_common_object=lead_m)


def summarize(event, frames):
    times=[r['t'] for r in frames]
    if not frames or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Empty or nonmonotonic registered event')
    gap=max((b-a for a,b in zip(times,times[1:])),default=None)
    timeline_ok=gap is not None and gap<=.50002
    result=dict(event=event['id'],site=event['site'],category=event['category'],objects=event['objects'],
                control_ids=event['window_ids'],sequence=frames[0]['seq'],pass_start_i=frames[0]['i'],
                pass_start_t=times[0],pass_end_i=frames[-1]['i'],pass_end_t=times[-1],
                sampled_frames=len(frames),max_gap_s=round(gap,6) if gap is not None else None,
                scene_missing_frames=[r['i'] for r in frames if r['alert'] is None],
                target_observations=[dict(i=r['i'],t=r['t'],targets=r['targets']) for r in frames if r['targets']])
    result.update(first_hits(frames,event['objects'],timeline_ok))
    result['by_object']={oid:first_hits(frames,[oid],timeline_ok) for oid in event['objects']}
    return result


def matched_lead(endpoint, controls_match):
    """Return None when a comparison lacks matched costs, distances or a common first target."""
    if controls_match is not True or endpoint['edge_lead_s'] is None:
        return None
    if not any(d is not None for d in endpoint['edge_lead_m_by_common_object'].values()):
        return None
    return endpoint['edge_lead_s']>0


def load_source(comparison_path, manifest_path, labels_path):
    def load(path):
        data=path.read_bytes()
        return json.loads(data),hashlib.sha256(data).hexdigest()
    receipt,receipt_sha=load(comparison_path)
    manifest,manifest_sha=load(manifest_path)
    labels,labels_sha=load(labels_path)
    if (receipt['method_sha256'],receipt['variant'],receipt['threshold_m2'])!=(METHOD,'rgb_relief',.005):
        raise ValueError('Unexpected method identity or threshold')
    if labels['manifest_sha256']!=manifest_sha or (receipt['manifest_sha256'],receipt['labels_sha256'])!=(manifest_sha,labels_sha):
        raise ValueError('Source or labels differ from the scored inputs')
    if 'frames' in manifest:
        clock={k:r['t'] for k,r in indexed(manifest['frames']).items()}
    else:
        clock={(s,i):i/manifest.get('fps',15) for s,indices in manifest['sessions'].items() for i in indices}
    scored=indexed(receipt['frames'])
    if any(k not in clock or abs(r['t']-clock[k])>1e-6 for k,r in scored.items()):
        raise ValueError('Score timestamp differs from original recording')
    def frame(k):
        return scored.get(k,dict(seq=k[0],i=k[1],t=clock[k],alert=None,targets=[]))
    grouped=defaultdict(list)
    for c in labels['cases']:
        grouped[c['site_id'],c['encounter_id']].append(c)
    scored_events=indexed(receipt['events'],lambda e:(e['site'],e['id']))
    if grouped.keys()!=scored_events.keys():
        raise ValueError('Scored event identities differ from locked labels')
    results=[]
    for key,cases in grouped.items():
        e=scored_events[key]
        keys=[tuple(k) for k in e['frames']]
        expected_controls={w for c in cases for w in c['window_ids']}
        if (set(e['objects'])!={c['object_id'] for c in cases} or set(e['window_ids'])!=expected_controls or
                any(set(keys)!={frame_key(f) for f in c['pass_frames']} for c in cases)):
            raise ValueError('Pass membership differs from locked labels')
        if not keys or len(set(keys))!=len(keys) or len({k[0] for k in keys})!=1:
            raise ValueError('Pass must use unique frames from one source sequence')
        ordered=sorted((k for k,t in clock.items() if k[0]==keys[0][0] and clock[keys[0]]<=t<=clock[keys[-1]]),key=clock.get)
        if ordered!=keys:
            raise ValueError('Pass omitted available source samples')
        expected=defaultdict(dict)
        for c in cases:
            for o in c['observations']:
                if o.get('near',True):
                    k=frame_key(o)
                    if k not in keys or c['object_id'] in expected[k]:
                        raise ValueError('Invalid or duplicate target observation')
                    expected[k][c['object_id']]=o
        rows=[]
        for k in keys:
            r=frame(k)
            by_object=indexed([t for t in r['targets'] if t['event']==e['id']],lambda t:t['object_id'])
            if by_object.keys()-expected[k].keys():
                raise ValueError('Unexpected unregistered target score')
            targets=[]
            for oid,o in expected[k].items():
                scores=by_object.get(oid,{}) if r['alert'] is not None else {}
                dist=o.get('forward_median_m')
                if dist is not None and not (.3-1e-5<=dist<=1.5+1e-5 and math.isfinite(dist)):
                    raise ValueError('Invalid near reference distance')
                targets.append(dict(object_id=oid,forward_median_m=dist,**{a:scores.get(a) for a in ARMS}))
            rows.append(dict(r,targets=targets))
        results.append(summarize(e,rows))
    controls=[]
    locked_windows=indexed(labels['windows'],lambda w:(w['site_id'],w['id']))
    scored_windows=indexed(receipt['windows'],lambda w:(w['site'],w['id']))
    if locked_windows.keys()!=scored_windows.keys():
        raise ValueError('Control identities differ from locked labels')
    for key,w in locked_windows.items():
        keys=[frame_key(f) for f in w['frames'] if f.get('eligible',True)]
        if keys!=[tuple(k) for k in scored_windows[key]['frames']]:
            raise ValueError('Control frames differ from locked labels')
        trace=[]
        for k in keys:
            a=frame(k)['alert'] or {}
            trace.append(dict(seq=k[0],i=k[1],t=clock[k],**{arm:a.get(arm) for arm in ARMS}))
        gaps=[b['t']-a['t'] for a,b in zip(trace,trace[1:])]
        coverage=(len(trace)>=3 and trace[-1]['t']-trace[0]['t']>=1.-1e-5 and
                  all(0<gap<=min(w.get('max_gap_s',.50001),.50001)+1e-5 for gap in gaps))
        complete=coverage and all(type(f[a]) is bool for f in trace for a in ARMS)
        added=[f['i'] for f in trace if f['direct'] is False and f['edge'] is True]
        removed=[f['i'] for f in trace if f['direct'] is True and f['edge'] is False]
        false_windows={a:True if any(f[a] for f in trace) else
                       False if coverage and all(f[a] is False for f in trace) else None for a in ARMS}
        controls.append(dict(site=key[0],id=key[1],trace=trace,added_frames=added,removed_frames=removed,
                             false_window=false_windows, complete=complete,
                             window_change=None if None in false_windows.values() else int(false_windows['edge'])-int(false_windows['direct']),
                             alarm_trace_identical=(False if added or removed else True if complete else None)))
    control_index={(w['site'],w['id']):w for w in controls}
    for e in results:
        paired=[control_index[e['site'],w] for w in e['control_ids']]
        if any({f['seq'] for f in w['trace']}!={e['sequence']} for w in paired):
            raise ValueError('Paired control is from a different recording')
        checks=[w['alarm_trace_identical'] for w in paired]
        e['paired_control_trace_identical']=False if False in checks else True if checks and all(v is True for v in checks) else None
        for endpoint in [e,*e['by_object'].values()]:
            endpoint['strictly_earlier_with_meters_and_matched_controls']=matched_lead(endpoint,e['paired_control_trace_identical'])
    return dict(comparison_sha256=receipt_sha,manifest_sha256=manifest_sha,labels_sha256=labels_sha,
                events=results,controls=controls)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source',nargs=3,type=Path,action='append',
                        metavar=('COMPARISON','MANIFEST','LABELS'))
    inputs.add_argument('--previous-transfer',action='store_true',
                        help='Read the four original transfer receipts, including controls-only recordings.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.previous_transfer:
        root=Path(__file__).resolve().parents[4]/'artifacts.local/work/shallow-edge-20261118'
        frozen=[('25b1ff5c4291-ee947a179afe','egowalk/locked-outdoor'),
                ('be3c89b72215-e3263bb3d415','egowalk/locked-garage'),
                ('eb7d712d49e7-f8f8665d57cc','native/sanpo-urban-control'),
                ('92973c1c0bbf-d2812d49ee6a','controls/lamar')]
        bundles=[(root/'results'/digest/'comparison.json',root/folder/'manifest.json',root/folder/'labels.json')
                 for digest,folder in frozen]
    else:
        bundles=args.source
    sources=[load_source(*paths) for paths in bundles]
    if len({s['comparison_sha256'] for s in sources})!=len(sources):
        raise ValueError('Duplicate recording receipt')
    events=[e for s in sources for e in s['events']]
    indexed(events,lambda e:(e['site'],e['event']))
    doc=dict(method_sha256=METHOD, threshold_m2=.005, units='source seconds and forward meters',
             scope='First observed alert on registered eligible target footprints. Missing references, scores, and timelines are explicit. Count as a lead only where the same object triggered both methods and its distances are known.',
             event_counts=dict(Counter(e['outcome'] for e in events)), sources=sources)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,indent=1,ensure_ascii=False,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
