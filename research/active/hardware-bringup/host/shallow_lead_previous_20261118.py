"""Recover first target-attributed alert times from previously used shallow-obstacle events."""

from collections import defaultdict
import cv2
import numpy as np

import shallow_paired_areas_20261118 as prior
import shallow_edge_holdout_20261118 as method
from shallow_valid_view_20261118 import valid_view
from shallow_lead_events_20261118 import summarize

replay = prior.replay
OUTPUT = method.ROOT.parent / 'shallow-lead-20261118/previous/first-alerts.json'
ARMS = ('direct', 'edge')


def windows(report):
    return [dict(id=w['id'], site=w['site'], frames=[i for _,i in w['frames']],
                 alarms={arm:w['alarms'][arm] for arm in ARMS}) for w in report['windows']]


def transfer():
    events, controls = [], []
    for path in sorted((method.ROOT/'results').glob('*/comparison.json')):
        r = replay.read(path)
        assert r['threshold_m2']==method.AREA and r['variant']==method.VARIANT and r['method_sha256']==method.METHOD
        controls.extend(windows(r))
        if not r['events']:
            continue
        folder = 'locked-outdoor' if r['events'][0]['id']=='outbound-brown' else 'locked-garage'
        label = method.ROOT/'egowalk'/folder/'labels.json'
        assert r['labels_sha256']==replay.sha(label)
        entries = {(c['encounter_id'],o['i'],c['object_id']):o
                   for c in replay.read(label)['cases'] for o in c['observations'] if o.get('near', True)}
        rows = prior.by_frame(r['frames'])
        for event in r['events']:
            frames = []
            for seq,i in event['frames']:
                row=rows[seq,i]
                targets=[dict(object_id=t['object_id'], direct=t['direct'], edge=t['edge'],
                              forward_median_m=entries[event['id'],i,t['object_id']]['forward_median_m'])
                         for t in row['targets'] if t['event']==event['id']]
                frames.append(dict(row,targets=targets))
            events.append(summarize(event, frames))
    assert len(events)==4 and len(controls)==9
    return dict(events=events, controls=controls)


def forward_samples(frame, ground, mask):
    down=-np.asarray(ground[0])
    forward=np.array([0.,0.,1.])-down*down[2]
    forward /= np.linalg.norm(forward)
    rays=replay.core.rays(frame.K,*frame.ref_depth.shape)
    along=((rays @ forward)*frame.ref_depth.ravel()).reshape(mask.shape)
    return along[mask & np.isfinite(along) & (along>=.30) & (along<=1.50)]


def brown_boundary_check(transfer_result):
    event=next(e for e in transfer_result['events'] if e['event']=='outbound-brown')
    first=event['arms']['edge']['first_target']
    folder=method.ROOT/'egowalk'
    manifest=folder/'locked-outdoor/manifest.json'
    dataset, rows=replay.inputs(manifest)
    late=event['arms']['direct']['first_any_frame']
    row=next(r for r in rows if r['i']==late and r['seq']==event['sequence'])
    frame=replay.load_frame(dataset,row,folder)
    old=replay.read(method.ROOT/'diagnostic/brown-late-localization.json')
    label=replay.read(folder/'diagnostic'/f'brown-{late}.json')
    down=-np.asarray(row['ground'][0])
    forward=np.array([0.,0.,1.])-down*down[2]
    forward/=np.linalg.norm(forward)
    z=frame.ref_depth.ravel()
    cloud=replay.core.rays(frame.K,*frame.ref_depth.shape)*np.nan_to_num(z)[:,None]
    above,along,corridor=replay.core._corridor(cloud,np.isfinite(z)&(z>.30)&(z<10),
                                                (down,forward,np.cross(down,forward)),row['ground'][1])
    shallow=corridor&(along<=1.50)&(above>=.05)&(above<.15)
    results={}
    early_footprint=next(t for c in replay.read(manifest.with_name('labels.json'))['cases']
                         if c['encounter_id']==event['event'] for t in c['observations'] if t['i']==first['frame'])
    early_mask=cv2.imread(str(manifest.parent/early_footprint['reference_footprint']),0)>0
    for boundary,record in old['results'].items():
        maskpath=folder/'diagnostic'/label['masks'][boundary]
        assert replay.sha(maskpath)==record['mask_sha256']
        mask=cv2.imread(str(maskpath),0)>0
        near=replay.core._open(shallow & mask.ravel(),*mask.shape)
        assert abs(float(np.percentile(along[near],10))-label['metrics'][boundary]['forward_p10'])<1e-5
        early_outline=cv2.imread(str(folder/'diagnostic'/f"brown-{first['frame']}-{boundary}.png"),0)>0
        includes_strict_first=not np.any(early_mask & ~early_outline)
        results[boundary]=dict(direct_overlap_m2=record['direct_area'],direct_emitted=record['direct'],
                              late_reference_forward_median_m=float(np.median(along[near])),
                              includes_first_edge_strict_footprint=bool(includes_strict_first),
                              edge_lead_s=round(row['t']-first['source_t'],6) if record['direct'] and includes_strict_first else None)
    return dict(frame=late, source_t=row['t'], labels='Post hoc visible outline, eight-pixel erosion or ten-pixel dilation. Outside primary endpoint.',
                first_strict_edge_frame=first['frame'], first_strict_edge_t=first['source_t'],
                first_strict_edge_m=first['objects'][0]['forward_median_m'], contours=results)


def development():
    pairs={e['encounter']: e for e in replay.read(prior.PREVIOUS/'clips/pairs.json')}
    events,controls=[],[]
    for name in ('egowalk-union','lin-yellow','oxford'):
        r=replay.read(method.ROOT/'dev'/(name+'.json'))
        for w in r['windows']:
            direct=[f['i'] for f in w['rows'] if f['old'] or f['total']['literal']>=method.AREA]
            edge=[f['i'] for f in w['rows'] if f['i'] in direct or f['edge'][method.VARIANT]>=method.AREA]
            controls.append(dict(id=w['id'],site=w['site'],frames=[f['i'] for f in w['rows']],
                                 alarms=dict(direct=direct,edge=edge)))
        retained=[e for e in r['events'] if e['band']=='shallow' and e['pass_old_alert_count']==0]
        if not retained:
            continue
        manifest=prior.PREVIOUS/prior.SOURCES[name]/'manifest.json'
        dataset, rows=replay.inputs(manifest)
        source=prior.by_frame(rows)
        digest=replay.sha(manifest)
        directory=prior.PREVIOUS/'replay'/digest[:16]
        cache=prior.OUT/'cache'/digest[:16]
        cache_rows=prior.by_frame(replay.read(cache/'index.json')['rows'])
        references=prior.by_frame(replay.read(directory/'reference.json')['rows'])
        label=prior.PREVIOUS/'scoring/labels'/(name+'.json')
        assert replay.sha(label)==r['sources'][0]['labels_sha256'] and digest==r['sources'][0]['manifest_sha256']
        entries={(c['encounter_id'],o['i'],c['object_id']):o
                 for c in replay.read(label)['cases'] for o in c['observations'] if o.get('near',True)}
        for event in retained:
            pair=pairs[event['id']]
            event['window_ids']=[w['id'] for w in pair['windows']]
            seq=pair['source_sequence']
            observations=defaultdict(list)
            for observation in event['observations']:
                observations[observation['i']].append(observation)
            frames=[]
            for i,t in zip(pair['target_frames'],pair['target_times_s'],strict=True):
                native=source[seq,i]
                ground=references[seq,i]['ground']
                assert native['t']==t
                with np.load(cache/cache_rows[seq,i]['file'],allow_pickle=False) as packet:
                    z=packet['z']
                    bgr=replay.image(native,tuple(reversed(z.shape)))
                    reader=method.component_scores(z,packet['K'],packet['probability'],bgr,
                                                   valid_view(name,native,z.shape),ground)
                    alert=reader()
                targets=[]
                for record in observations[i]:
                    oid=record['object_id']
                    entry=entries[event['id'],i,oid]
                    maskpath=label.parent/entry['reference_footprint']
                    assert replay.sha(maskpath)==entry['reference_mask_sha256']
                    mask=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)
                    f=replay.load_frame(dataset,native,directory)
                    if name=='lin-yellow':
                        original=next(o for o in replay.read(manifest.with_name('labels.json'))['observations'] if o['i']==i)
                        metric_ground=original['ground']
                    else:
                        metric_ground=ground
                    distances=forward_samples(f,metric_ground,mask>0)
                    median=float(np.median(distances))
                    if name=='lin-yellow':
                        assert abs(median-original['hist']['short_1.5']['distance'][2])<.001
                    else:
                        assert abs(np.percentile(distances,10)-entry['shallow_near']['d10'])<.002
                    if mask.shape!=z.shape:
                        mask=cv2.resize(mask,tuple(reversed(z.shape)),interpolation=cv2.INTER_NEAREST)
                    scores=reader(mask>0)
                    direct=record['old_target'] or record['target']['literal']>=method.AREA
                    edge=direct or record['edge_target'][method.VARIANT]>=method.AREA
                    assert (scores['direct'],scores['edge'])==(direct,edge)
                    direct_any=record['old_any'] or record['total']['literal']>=method.AREA
                    assert (alert['direct'],alert['edge'])==(direct_any,direct_any or record['edge'][method.VARIANT]>=method.AREA)
                    targets.append(dict(object_id=oid,direct=direct,edge=edge,forward_median_m=median))
                frames.append(dict(seq=seq,i=i,t=t,alert=alert,targets=targets))
            assert len(frames)==event['pass_samples']
            events.append(summarize(event,frames))
    assert len(events)==5 and len(controls)==8
    return dict(events=events,controls=controls)


if __name__=='__main__':
    cv2.setNumThreads(1)
    report=dict(method_sha256=method.METHOD,variant=method.VARIANT,threshold_m2=method.AREA,
                timing='Original recording seconds, relative to first sampled pass frame. Positive difference means earlier edge alert.',
                attribution='Frozen near target footprint only. Earlier scene alerts without an eligible target mask are unclassified. Distance is median forward floor-axis distance in the native near reference footprint.',
                provenance='Previously used routes. These event times are a retrospective control and cannot count as new sites.',
                prior_transfer=transfer(), development=development())
    report['brown_late_sensitivity']=brown_boundary_check(report['prior_transfer'])
    replay.save(OUTPUT,report)
    print(OUTPUT)
