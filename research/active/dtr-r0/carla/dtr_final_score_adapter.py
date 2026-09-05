"""Pure shared R1 scoring adapter; caller must enforce sealed outcome access.

No file reads, inference, training or source selection. The full captured truth
tail supports contact timing even when scoring stops before physical contact.
"""
import math
import dtr_final_reckoning_event_metrics as metrics


def aligned_rows(truth, predictions, annex, arm):
    if len(truth) != len(predictions):
        raise ValueError('prediction_truth_count')
    start, end = annex['score_start_s'], annex['score_end_s']
    rows=[]
    for source, prediction in zip(truth, predictions):
        if source['sample_index'] != prediction['sample_index'] or abs(source['time_s']-prediction['time_s'])>1e-8:
            raise ValueError('prediction_truth_identity')
        now=float(source['time_s'])
        if not start-1e-8 <= now <= end+1e-8:
            continue
        native=source['truth']
        risk=native['future_contact_within_horizon']
        if type(risk) is not bool:
            raise ValueError('native_future_not_boolean')
        # Missing future support is UNKNOWN, never a negative label.
        known_risk=risk if risk or truth[-1]['time_s']-now>=3.-1e-8 else None
        delay=native['realized_time_to_contact_seconds']
        if risk and (type(delay) not in (int,float) or not math.isfinite(delay) or not 0<=delay<=3.+1e-8):
            raise ValueError('missing_or_invalid_contact_time')
        active=prediction['arms'][arm]['route_risk']
        if active is not None and type(active) is not bool:
            raise ValueError('prediction_not_tristate')
        rows.append({'time_s':now, 'truth_risk':known_risk, 'current_contact':native['current_contact'],
                     'truth_contact_time_s':now+delay if risk else None, 'predicted_active':active})
    if not rows:
        raise ValueError('empty_scored_rows')
    return rows


def score_final(roster, truth_groups, prediction_groups, annex_groups):
    expected_groups={'FINAL_A','FINAL_B'}
    if set(truth_groups)!=expected_groups or set(prediction_groups)!=expected_groups or set(annex_groups)!=expected_groups:
        raise ValueError('exact_final_groups_required_no_fit_or_auxiliary')
    arms=[row['arm_id'] for row in roster['arms']]
    strata=[row['stratum_id'] for row in roster['source_design']['strata']]
    by_arm={arm:{} for arm in arms}
    by_stratum={arm:{s:[] for s in strata} for arm in arms}
    for group in sorted(expected_groups):
        expected={row['episode_id'] for row in annex_groups[group]['strata'].values()}
        if len(expected)!=10 or set(truth_groups[group])!=expected or set(prediction_groups[group])!=expected:
            raise ValueError('exact_ten_main_episodes_required')
        for stratum in strata:
            annex=annex_groups[group]['strata'][stratum]
            ep=annex['episode_id']
            prediction=prediction_groups[group][ep]
            if any(set(row['arms'])!=set(arms) for row in prediction):
                raise ValueError('exact_eleven_arms_required')
            for arm in arms:
                value=metrics.evaluate_episode(aligned_rows(truth_groups[group][ep],prediction,annex,arm))
                by_arm[arm][group+':'+ep]=value
                by_stratum[arm][stratum].append(value)
    pairs=[('KALMAN_CV_ROUTE_TUBE','X94_EVIDENCE_MODEL'),
           ('X94_EVIDENCE_MODEL','X94_EVIDENCE_PLUS_SIMPLE_HYSTERESIS_0P60S'),
           ('KALMAN_CV_ROUTE_TUBE_HYSTERESIS_0P60S','X94_EVIDENCE_PLUS_SIMPLE_HYSTERESIS_0P60S'),
           ('X94_EVIDENCE_PLUS_SIMPLE_HYSTERESIS_0P60S','X95_EVENT_CHALLENGER')]
    uncertainty=roster['evaluator_contract']['uncertainty']
    if uncertainty['paired_bootstrap_replicates']!=10000 or uncertainty['bootstrap_seed']!=517999:
        raise ValueError('frozen_bootstrap_drift')
    return {'schema':'dtr-final-shared-score-v1', 'claim':'SYNTHETIC_R1_ONLY',
            'arm_scores':{a:metrics.aggregate(list(v.values())) for a,v in by_arm.items()},
            'per_stratum':{a:{s:metrics.aggregate(v) for s,v in scores.items()} for a,scores in by_stratum.items()},
            'per_episode':by_arm,
            'decomposition':[{ 'reference':a, 'candidate':b,
                              **metrics.paired_episode_bootstrap(by_arm[a],by_arm[b])} for a,b in pairs],
            'metric_contract':metrics.contract(), 'final_episode_count':20,
            'fit_episodes_scored':0, 'auxiliary_episodes_scored':0}
