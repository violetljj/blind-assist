"""Post-seal stability, onset and false-alert accounting; no model invocation."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from query_occupancy_data import read,write,sha,stage_path,new_stage_directory
from ba_camera_corridor_metrics import evaluate_rows
from local_transfer_metrics import comparison

ARMS = ('A_current','raw','local','raw_standalone','local_standalone')
AXES = ('trajectory','size_level','shape','layer','layout_relation','base_group_id')


def make_rows(ids,baseline,probabilities,labels,cutoffs):
    assert len(ids) == len(baseline) == 576
    assert np.array_equal(labels['indices'],np.arange(576))
    assert labels['classes'].shape == labels['valid'].shape == (576,6)
    assert np.isin(labels['classes'],np.arange(7)).all()
    rows = []
    for i, (meta,a) in enumerate(zip(ids,baseline)):
        scores = {arm:float(probabilities[arm][i,[1,4]].max()) for arm in ('raw','local')}
        flags = dict(A_current=bool(a['alert']))
        for arm in ('raw','local'):
            flags[arm+'_standalone'] = scores[arm] >= cutoffs[arm]
            flags[arm] = bool(a['alert'] or flags[arm+'_standalone'])
        rows.append(dict(**{k:meta[k] for k in ('id','clip_id','frame_in_clip','time_s','index','type_id',*AXES)},
            truth=bool((labels['classes'][i,[1,4]] < 6).any()) if labels['valid'][i,[1,4]].all() else None,
            boundary=meta['layout_relation']=='BOUNDARY',query_truth=(labels['classes'][i]<6).tolist(),
            query_valid=labels['valid'][i].tolist(),frame_scores=scores,
            query_probabilities={arm:probabilities[arm][i].tolist() for arm in ('raw','local')},
            predictions={arm:dict(alert=bool(flag),unknown=bool(a['unknown']),ambiguous=bool(flag and a['unknown']))
                         for arm,flag in flags.items()}))
    return rows


def event_benefits(rows,metrics):
    metadata = {r['clip_id']:r for r in rows}
    reference = {(e['clip_id'],e['start_frame']):e for e in metrics['arms']['A_current']['events']}
    records = []
    for event in metrics['arms']['local']['events']:
        old = reference[event['clip_id'],event['start_frame']]
        t0,t1 = old['first_alert_time_s'],event['first_alert_time_s']
        entry = next(r['time_s'] for r in rows if r['clip_id']==event['clip_id'] and r['frame_in_clip']==event['start_frame'])
        opportunity = t0 is None or t0 >= entry+.2-1e-9
        gain = t1 is not None and (t0 is None or t1 <= t0-.2+1e-9)
        records.append(dict(clip_id=event['clip_id'],start_frame=event['start_frame'],entry_s=entry,
            trajectory=metadata[event['clip_id']]['trajectory'],size_level=metadata[event['clip_id']]['size_level'],
            base_group_id=metadata[event['clip_id']]['base_group_id'],A_first_s=t0,LOCAL_first_s=t1,
            advance_s=None if t0 is None or t1 is None else t0-t1,
            opportunity=opportunity,benefit=gain,recovered=t0 is None and t1 is not None))
    return dict(events=records,opportunity_events=sum(r['opportunity'] for r in records),
        benefit_events=sum(r['benefit'] for r in records),recovered_events=sum(r['recovered'] for r in records),
        by_trajectory={t:dict(events=sum(r['trajectory']==t for r in records),
            opportunities=sum(r['trajectory']==t and r['opportunity'] for r in records),
            benefits=sum(r['trajectory']==t and r['benefit'] for r in records)) for t in sorted({r['trajectory'] for r in rows})})


def coverage(rows,metrics):
    clips = defaultdict(list)
    for r in rows:clips[r['clip_id']].append(r)
    answer = {}
    for arm in ARMS:
        records = []
        for e in metrics['arms'][arm]['events']:
            seq = [r for r in clips[e['clip_id']] if e['start_frame'] <= r['frame_in_clip'] <= e['end_frame']]
            flags = np.array([r['predictions'][arm]['alert'] for r in seq],bool)
            indices = np.flatnonzero(flags)
            interior = ~flags[indices[0]:indices[-1]+1] if len(indices) else np.array([],bool)
            segments = int((interior & ~np.r_[False,interior[:-1]]).sum()) if len(interior) else 0
            records.append(dict(clip_id=e['clip_id'],start_frame=e['start_frame'],positive_frames=len(seq),
                alert_frames=int(flags.sum()),fraction=float(flags.mean()),
                leading_misses=int(indices[0]) if len(indices) else len(seq),
                trailing_misses=int(len(seq)-1-indices[-1]) if len(indices) else 0,
                internal_gap_frames=int(interior.sum()),internal_gap_segments=segments))
        answer[arm] = dict(events=records,minimum_event_coverage=min((r['fraction'] for r in records),default=None),
            mean_event_coverage=float(np.mean([r['fraction'] for r in records])) if records else None,
            internal_gap_frames=sum(r['internal_gap_frames'] for r in records),
            internal_gap_segments=sum(r['internal_gap_segments'] for r in records))
    return answer


def stability(rows,metrics,strata,admissible=True):
    timing = event_benefits(rows,metrics)
    vs_a = comparison(rows,metrics,'local','A_current')
    vs_raw = comparison(rows,metrics,'local','raw')
    a_preserved = not vs_a['lost_true_frames'] and not any(e['lost'] or
        (e['delay_s'] is not None and e['delay_s']>1e-9) for e in vs_a['event_differences'])
    trajectory_costs = {t:comparison([r for r in rows if r['trajectory']==t],m,'local','A_current')
                        for t,m in strata['trajectory'].items()}
    timing_ok = timing['benefit_events'] >= 8 and all(v['benefits']>=3 for v in timing['by_trajectory'].values()) and a_preserved
    cost_ok = vs_a['FP_delta']<=2 and vs_a['false_segment_delta']<=1 and all(
        v['FP_delta']<=2 and v['false_segment_delta']<=1 for v in trajectory_costs.values())
    layer_ok = all(m['arms']['local']['frames']['all_known']['recall'] >=
        m['arms']['raw']['frames']['all_known']['recall']-.05-1e-9 for m in strata['layer'].values())
    raw_ok = (vs_raw['recall_delta'] is not None and vs_raw['recall_delta']>=.05-1e-9 and
              vs_raw['FP_delta']<=1 and vs_raw['false_segment_delta']<=1 and
              vs_raw['improved_groups']>=4 and layer_ok)
    timing_status = 'NOT_EVALUABLE' if timing['opportunity_events']<8 else 'PASS' if timing_ok else 'FAIL'
    status = ('NOT_EVALUABLE' if not admissible else 'FAIL' if not cost_ok or not raw_ok or timing_status=='FAIL'
              else 'NOT_EVALUABLE' if timing_status=='NOT_EVALUABLE' else 'PASS')
    return dict(status=status,source_admissible=admissible,timing_status=timing_status,
        timing=timing,A_preserved=a_preserved,costs_pass=cost_ok,RAW_comparison_pass=raw_ok,
        layer_noninferior_to_RAW=layer_ok,versus_A=vs_a,versus_RAW=vs_raw,
        trajectory_costs={t:{k:v[k] for k in ('FP_delta','false_segment_delta')} for t,v in trajectory_costs.items()})


def evaluate(root,result):
    start = time.perf_counter();pred = stage_path(root,'predictions')
    out = result.parent;new_stage_directory(out)
    seal = read(pred/'prediction-seal.json')
    assert seal['status']=='PASS' and seal['fits']==seal['cutoff_selections']==0 and seal['evaluation_labels_opened'] is False
    for name,digest in seal['hashes'].items():assert sha(pred/name)==digest,name
    freeze = read(pred/'freeze.json')
    label_path = stage_path(root,'prepared')/'labels/evaluation.npz'
    assert sha(label_path)==freeze['expected_evaluation_labels_sha256']
    assert sha(root/'plan/stability-protocol.md')==freeze['protocol_sha256']
    labels = dict(np.load(label_path,allow_pickle=False))
    probabilities = dict(np.load(pred/'probabilities.npz',allow_pickle=False))
    assert np.array_equal(probabilities.pop('indices'),np.arange(576))
    rows = make_rows(read(pred/'identities.json'),read(pred/'baseline.json'),probabilities,labels,seal['thresholds'])
    admission_path = stage_path(root,'prepared')/'labels/source-admission.json'
    materialization = read(stage_path(root,'prepared')/'materialization.json')
    assert sha(stage_path(root,'prepared')/'materialization.json')==freeze['materialization_sha256']
    assert sha(admission_path)==materialization['hashes']['labels/source-admission.json']
    admission = read(admission_path)
    metrics = evaluate_rows(rows,arms=ARMS)
    strata = {key:{value:evaluate_rows([r for r in rows if r[key]==value],arms=ARMS)
                   for value in sorted({r[key] for r in rows})} for key in AXES}
    gate = stability(rows,metrics,strata,admission['admissible'])
    write(out/'frame-results.json',rows)
    write(out/'metrics.json',dict(metrics=metrics,strata=strata,stability=gate,coverage=coverage(rows,metrics),
        comparisons={a+'_vs_'+b:comparison(rows,metrics,a,b) for a,b in [('local','A_current'),('local','raw'),('raw','A_current')]}))
    decision = {'PASS':'LOCAL_COMPOSITE_TRAJECTORY_STABILITY_COMPONENT','FAIL':'LOCAL_COMPOSITE_TRAJECTORY_STABILITY_NEGATIVE',
                'NOT_EVALUABLE':'LOCAL_STABILITY_OPPORTUNITY_OR_SOURCE_NOT_EVALUABLE'}[gate['status']]
    write(result,dict(status='PASS',decision=decision,stability_status=gate['status'],source_admission=admission,
        metrics={a:dict(**metrics['arms'][a]['frames']['all_known'],
                       false_segments=metrics['arms'][a]['false_alert_segment_count'],
                       events_detected=metrics['arms'][a]['detected_events']) for a in ARMS},
        timing_summary={k:v for k,v in gate['timing'].items() if k!='events'},
        elapsed_s=time.perf_counter()-start,backend='CPU NumPy TASK_NOT_GPU_SUITABLE',
        prediction_seal_sha256=sha(pred/'prediction-seal.json'),label_sha256=sha(label_path),
        fits=0,source_scope='NEW_SAME_GENERATOR_COMPOSITE_POSED_DEVELOPMENT'))
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(decision,flush=True)
