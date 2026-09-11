"""Scalar-only MZ77 sequence metrics (TASK_NOT_GPU_SUITABLE).

Predictions are signed margins, positive at >=0. Native scene support and
intended-target opportunities are distinct: target opportunity hits do NOT
establish that an alert was caused by the target. No latency/clearance claim.
"""
import numpy as np

QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR', 'BODY_ANY', 'HEAD_ANY')
GAPS = ((20, 22), (30, 32))


def _six(x):
    return np.column_stack((x, x[:, :2].any(1), x[:, 2:].any(1)))


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(s), int(e)) for s,e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)]


def _record(i, index, time, distance):
    if i is None:
        return None
    return dict(index=int(index[i]), time_s=float(time[i]),
                front_distance_m=float(distance[i]) if np.isfinite(distance[i]) else None)


def _metric(y, k, a, index, time, distance, target=False):
    positive = y & k
    episodes = []
    for start, end in _runs(positive):
        hits = np.flatnonzero(a[start:end + 1])
        first = int(start + hits[0]) if len(hits) else None
        episodes.append(dict(first_eligible=_record(start, index, time, distance),
            last_eligible=_record(end, index, time, distance),
            first_true_hit=_record(first, index, time, distance),
            nominal_delay_s=float(time[first] - time[start]) if first is not None else None,
            missed=first is None, positive_samples=int(end-start+1), true_hit_samples=int(len(hits)),
            left_censored=start == 0 or not bool(k[start - 1]),
            right_censored=end == len(y)-1 or not bool(k[end + 1]),
            preexisting_alert_before_onset=bool(a[start-1]) if start else None,
            onset_status='LEFT_CENSORED' if start == 0 or not k[start-1] else
                         ('PREEXISTING_ALERT' if a[start-1] else 'OBSERVED_ONSET')))
    silent = _runs(positive & ~a)
    out = dict(status='EVALUABLE' if positive.any() else 'NO_POSITIVE_EPISODE',
        known_samples=int(k.sum()), unknown_samples=int((~k).sum()),
        unknown_alert_samples=int((a & ~k).sum()), positive_samples=int(positive.sum()),
        positive_retention=float((positive & a).sum()/positive.sum()) if positive.any() else None,
        missed_episodes=sum(e['missed'] for e in episodes), episodes=episodes,
        alert_on_transitions=int((~a[:-1] & a[1:]).sum()),
        alert_off_transitions=int((a[:-1] & ~a[1:]).sum()),
        longest_silence_positive_samples=max((b-s+1 for s,b in silent), default=0),
        longest_silence_positive_nominal_s=max((b-s+1 for s,b in silent), default=0)/10,
        alert_indices=index[a].astype(int).tolist(),
        positive_silent_indices=index[positive & ~a].astype(int).tolist())
    first_positive = np.flatnonzero(positive)
    first_hit = np.flatnonzero(positive & a)
    out['first_eligible'] = _record(int(first_positive[0]) if len(first_positive) else None, index,time,distance)
    out['first_true_hit'] = _record(int(first_hit[0]) if len(first_hit) else None,index,time,distance)
    # Episode-local delays above avoid charging silent intervening negative spans.
    if target:
        out.update(opportunity_hits=int((positive & a).sum()), opportunity_misses=int((positive & ~a).sum()),
                   activations_without_target_support=int((~y & k & a).sum()),
                   attribution='Alert overlap with visible intended-target support; not target attribution or target FP.')
    else:
        out['confusion'] = dict(TP=int((y & k & a).sum()), FN=int((y & k & ~a).sum()),
                              FP=int((~y & k & a).sum()), TN=int((~y & k & ~a).sum()))
        assert sum(out['confusion'].values()) == int(k.sum())
    return out


def _gap(y, k, clean, gap, index):
    windows = []
    for start, end in GAPS:
        w = (index >= start) & (index <= end)
        before = np.flatnonzero(index == start-1)
        by_query = {}
        for q, name in enumerate(QUERIES):
            known = k[:, q]; pos = y[:, q] & known
            lost = clean[:, q] & ~gap[:, q] & w
            gained = ~clean[:, q] & gap[:, q] & w
            active_before = bool(before.size and gap[before[0], q])
            dropped = np.flatnonzero(w & pos & ~gap[:, q]) if active_before else np.array([], dtype=int)
            recovery = dict(status='NOT_EVALUABLE_NO_PRIOR_WARNING' if not active_before else
                           'NOT_EVALUABLE_NO_POSITIVE_DROPOUT', first_recovered_index=None,
                           samples_after_packet_return=None, right_censored=False)
            if len(dropped):
                # Recovery belongs to the interrupted positive episode, never a later episode.
                first_drop = int(dropped[0]); eligible = []
                boundary = None
                for j in range(first_drop, len(index)):
                    if not pos[j]:
                        boundary = j
                        break
                    if index[j] > end:
                        eligible.append(j)
                hits = [j for j in eligible if gap[j, q]]
                if hits:
                    recovery.update(status='RECOVERED', first_recovered_index=int(index[hits[0]]),
                                    samples_after_packet_return=int(index[hits[0]]-(end+1)))
                else:
                    recovery.update(status='RIGHT_CENSORED' if boundary is None else
                                    ('UNKNOWN_CENSORED' if not known[boundary] else 'POSITIVE_EPISODE_ENDED'),
                                    right_censored=boundary is None)
            by_query[name] = dict(lost_true_indices=index[lost & pos].astype(int).tolist(),
                gained_true_indices=index[gained & pos].astype(int).tolist(),
                lost_false_indices=index[lost & known & ~y[:, q]].astype(int).tolist(),
                gained_false_indices=index[gained & known & ~y[:, q]].astype(int).tolist(),
                lost_unknown_indices=index[lost & ~known].astype(int).tolist(),
                gained_unknown_indices=index[gained & ~known].astype(int).tolist(),
                prior_warning_active=active_before,
                previously_active_alert_drop_indices=index[w & ~gap[:, q]].astype(int).tolist() if active_before else [],
                positive_miss_indices=index[w & pos & ~gap[:, q]].astype(int).tolist(), recovery=recovery)
            for kind in ('lost_true','gained_true','lost_false','gained_false','lost_unknown','gained_unknown'):
                by_query[name][kind + '_bits'] = len(by_query[name][kind + '_indices'])
        windows.append(dict(start_index=start, end_index=end, queries=by_query,
                            head_near_miss_indices=by_query['HEAD_NEAR']['positive_miss_indices']))
    return windows


def analyze(arrays, predictions):
    """Return JSON-safe metrics. See module header; no fitting or I/O occurs.

    arrays: clip/index/time_s[N], truth/known/target_truth[N,4], distance[N]
    named front_distance_m. Optional packets={condition:{ranges,valid[N,64,2]}}.
    predictions={condition:{method: margins[N,4]}}. Exactly 40 samples per clip.
    """
    clip = np.asarray(arrays['clip']).astype(str)
    n = len(clip)
    truth, known, target = [np.asarray(arrays[key], dtype=bool) for key in ('truth','known','target_truth')]
    if any(x.shape != (n,4) for x in (truth,known,target)):
        raise ValueError('truth/known/target_truth must be [N,4]')
    if np.any(truth & ~known) or np.any(target & ~truth):
        raise ValueError('positive truth requires knownness; target support must be subset of scene support')
    alert = {}
    for condition, methods in predictions.items():
        alert[condition] = {}
        for method, values in methods.items():
            values = np.asarray(values)
            if values.shape != (n,4) or not np.isfinite(values).all():
                raise ValueError('margins must be finite [N,4]')
            alert[condition][method] = _six(values >= 0)
    result = dict(schema='MZ77_SEQUENCE_METRICS_V1', backend='CPU',
        cpu_reason='TASK_NOT_GPU_SUITABLE', alert_rule='margin >= 0',
        nominal_sample_period_s=0.1,
        limits='Simulated sampled positions, stops before contact; no measured latency, clearance, or target attribution.', clips={})
    for name in dict.fromkeys(clip.tolist()):
        rows = np.flatnonzero(clip == name)
        rows = rows[np.argsort(np.asarray(arrays['index'])[rows])]
        index = np.asarray(arrays['index'])[rows]
        time = np.asarray(arrays['time_s'], dtype=float)[rows]
        distance = np.asarray(arrays['front_distance_m'], dtype=float)[rows]
        if not np.array_equal(index, np.arange(40)) or not np.allclose(np.diff(time), .1, atol=1e-6):
            raise ValueError('each clip must contain indices 0..39 at nominal 10Hz')
        y, t = _six(truth[rows]), _six(target[rows])
        k4 = known[rows]
        k = np.column_stack((k4, y[:,4] | k4[:,:2].all(1), y[:,5] | k4[:,2:].all(1)))
        c = dict(frame_count=40, conditions={}, gap_comparisons={})
        for condition, methods in alert.items():
            c['conditions'][condition] = dict(methods={})
            for method, values in methods.items():
                a = values[rows]
                c['conditions'][condition]['methods'][method] = dict(
                    full_scene={q:_metric(y[:,j], k[:,j], a[:,j], index,time,distance) for j,q in enumerate(QUERIES)},
                    intended_target={q:_metric(t[:,j], k[:,j], a[:,j], index,time,distance,True) for j,q in enumerate(QUERIES)})
        if 'CLEAN' in alert and 'CENTER_GAP' in alert:
            for method in alert['CLEAN'].keys() & alert['CENTER_GAP'].keys():
                c['gap_comparisons'][method] = _gap(y,k,alert['CLEAN'][method][rows],alert['CENTER_GAP'][method][rows],index)
        packets = arrays.get('packets')
        if packets:
            valid = {cond:np.asarray(p['valid'],dtype=bool)[rows] for cond,p in packets.items()}
            c['packet_availability'] = {cond:dict(valid_slots=int(v.sum()), zero_valid_indices=index[~v.any(axis=(1,2))].astype(int).tolist()) for cond,v in valid.items()}
            if 'CLEAN' in valid:
                c['naturally_zero_valid_indices'] = index[~valid['CLEAN'].any(axis=(1,2))].astype(int).tolist()
            if 'CLEAN' in valid and 'CENTER_GAP' in valid:
                removed = valid['CLEAN'] & ~valid['CENTER_GAP']
                c['gap_pressure'] = [dict(start_index=s,end_index=e,deleted_valid_slots=int(removed[s:e+1].sum()),
                    status='EVALUABLE' if removed[s:e+1].any() else 'NOT_EVALUABLE_NO_VALID_RETURN_REMOVED') for s,e in GAPS]
        result['clips'][name] = c
    return result
