"""Fixed complete detector-frame removals; no truth or native object identity.

The raw ledger is retained. Eligibility uses the frozen X24 measured, confirmed
route-risk output on adjacent raw frames, solely as a source admission check.
It never changes removal indices or enters an arm's input. All candidates on a
selected frame are removed, modeling controlled detector-output disappearance.
"""
import copy
import hashlib
import json
import dtr_carla_x24_plan_adherent_predictor as x24


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest().upper()


def measured_credential(frame):
    arm=frame['arms'][x24.ARM_X24]
    measured={str(t['track_id']) for t in frame['tracks'] if t['disposition']=='MEASURED'}
    confirmed={str(t) for t in arm['confirmed_risk_track_ids']}
    return arm['route_risk'] is True and bool(measured & confirmed)


def intervene_episode(candidate_values, raw_x24_episode, stratum_id, windows):
    for value in candidate_values:x24.adapter.assert_sanitized_model_value(value)
    frames=raw_x24_episode['frames']
    if len(frames)!=len(candidate_values):raise ValueError('raw_credential_ledger_count')
    lengths=sorted(map(len,windows))
    expected=[1] if stratum_id.startswith('S02') else [2,3,6] if stratum_id.startswith('S03') else []
    if lengths!=expected:raise ValueError('frozen_removal_lengths')
    used=set();receipts=[]
    for window in windows:
        if window!=list(range(window[0],window[-1]+1)) or window[0]<=0 or window[-1]+1>=len(frames) or used.intersection(window):
            raise ValueError('invalid_fixed_removal_indices')
        for i in [window[0]-1,window[-1]+1]:
            if frames[i]['sample_index']!=i or not candidate_values[i]['candidates'] or not measured_credential(frames[i]):
                raise ValueError('ADMISSION_FAILED_MEASURED_RAW_COLLISION_CREDENTIAL_NO_INDEX_RESCUE')
        if any(not candidate_values[i]['candidates'] for i in window):
            raise ValueError('ADMISSION_FAILED_REMOVAL_HAS_NO_RAW_OBSERVATION')
        used.update(window)
        receipts.append({'indices':window,'before_index':window[0]-1,'after_index':window[-1]+1,
                         'raw_credential_arm':x24.ARM_X24})
    transformed=copy.deepcopy(candidate_values)
    for i in sorted(used):
        transformed[i]['candidates']=[]
        if 'candidate_count' in transformed[i]:transformed[i]['candidate_count']=0
        if 'candidate_counts_by_class' in transformed[i]:transformed[i]['candidate_counts_by_class']={}
    return transformed, {'schema':'dtr-final-fixed-detector-intervention-v1',
        'scope':'ALL_DETECTOR_CANDIDATES_ON_FIXED_FRAME_NOT_NATURAL_PREVALENCE',
        'stratum_id':stratum_id,'windows':receipts,'removed_frames':sorted(used),
        'removed_candidates':sum(len(candidate_values[i]['candidates']) for i in used),
        'raw_sha256':digest(candidate_values),'intervened_sha256':digest(transformed),
        'truth_or_native_ids_used':False,'all_arms_share_transformed_ledger':True}
