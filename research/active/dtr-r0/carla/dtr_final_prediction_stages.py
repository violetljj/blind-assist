"""Pure R1 stage adapters; external runner owns immutable seals and access order.

No file I/O. In particular, preparing nonlearned predictions never accepts truth.
Model fitting requires an explicitly named FIT_ONLY group and exact episode set.
"""
import copy
import numpy as np
import dtr_carla_raw_kalman_baseline as raw
import dtr_carla_classic_motion_baselines as classic
import dtr_carla_bounded_event_emitter as emitter
import dtr_carla_x94_one_frame_full_dropout_continuity as x94
import dtr_carla_x95_credentialed_hazard_state_model as x95

RAW_NAMES={raw.ARM_RADIAL:'RADIAL_TTC',raw.ARM_ROUTE:'KALMAN_CV_ROUTE_TUBE',
           raw.ARM_HYSTERESIS:'KALMAN_CV_ROUTE_TUBE_HYSTERESIS_0P60S'}


def align(reference, rows):
    if len(reference)!=len(rows):raise ValueError('stage_frame_count')
    for a,b in zip(reference,rows):
        if a['sample_index']!=b['sample_index'] or abs(a['time_s']-b['time_s'])>1e-8:
            raise ValueError('stage_frame_identity')


def merge_nonlearned(raw_episode, classic_episode, structural):
    if len({raw_episode['episode_id'],classic_episode['episode_id'],structural['episode_id']})!=1:
        raise ValueError('stage_episode_identity')
    frames=raw_episode['frames']
    align(frames,classic_episode['frames'])
    core=structural['core_episodes']['X94_EVIDENCE_MODEL']
    align(frames,core['frames'])
    held=emitter.apply_episode(core,evidence_arm=x94.ARM_X94)
    for rows in structural['arms'].values():align(frames,rows)
    result=[]
    for i,frame in enumerate(frames):
        arms={new:copy.deepcopy(frame['arms'][old]) for old,new in RAW_NAMES.items()}
        arms.update(copy.deepcopy(classic_episode['frames'][i]['arms']))
        for name,rows in structural['arms'].items():
            arms[name]={k:copy.deepcopy(v) for k,v in rows[i].items() if k not in ('sample_index','time_s')}
        arms.update(held['frames'][i]['arms'])
        if len(arms)!=9:raise ValueError('nine_nonlearned_arms_required')
        result.append({'sample_index':frame['sample_index'],'time_s':frame['time_s'],
                       'arms':arms, 'tiny_features':classic_episode['frames'][i]['tiny_features']})
    return {'episode_id':raw_episode['episode_id'],'frames':result,
            'x94_core':core,'status':'NONLEARNED_OUTPUT_PENDING_EXTERNAL_SEAL'}


def fit_models(group_id, episodes, truths, annex):
    if group_id!='FIT_ONLY':raise ValueError('fit_only_required')
    expected={v['episode_id'] for v in annex['strata'].values()}
    if len(expected)!=10 or set(episodes)!=expected or set(truths)!=expected:
        raise ValueError('exact_ten_fit_episodes_required')
    settings={v['episode_id']:v for v in annex['strata'].values()}
    tiny_vectors=[];hazard_vectors=[];labels=[]
    for ep in sorted(expected):
        frames=episodes[ep]['frames']; core=episodes[ep]['x94_core']['frames']
        align(frames,core);align(frames,truths[ep])
        parents=set()
        for frame,native,state in zip(frames,truths[ep],core):
            parents.update(str(p) for p in state['arms'][x94.ARM_X94].get('x75_collision_credential_birth_parent_ids',[]))
            obs=x95.observation(state,parents)
            if not settings[ep]['score_start_s']-1e-8<=frame['time_s']<=settings[ep]['score_end_s']+1e-8:continue
            label=native['truth']['future_contact_within_horizon']
            if type(label) is not bool:raise ValueError('unknown_fit_label')
            if not label and truths[ep][-1]['time_s']-frame['time_s']<3.-1e-8:continue
            tiny_vectors.append(frame['tiny_features']);hazard_vectors.append(obs['vector']);labels.append(label)
    tiny=classic.fit_tiny_logistic(tiny_vectors,labels)
    hazard=x95.fit_logistic(hazard_vectors,labels)
    return {'fit_group':'FIT_ONLY','fit_rows':len(labels),'positive_rows':sum(labels),
            'tiny':tiny.to_json(),'x95':hazard.to_json()}


def apply_learned(episode, models):
    if models['fit_group']!='FIT_ONLY':raise ValueError('fit_provenance')
    tiny=classic.TinyLogisticModel(*(np.asarray(models['tiny'][k],dtype=float) for k in ('mean','scale','weights')),bias=float(models['tiny']['bias']))
    hazard=x95.LogisticEmission(*(np.asarray(models['x95'][k],dtype=float) for k in ('mean','scale','weights')))
    decoded,diagnostics=x95.decode_episode(episode['x94_core']['frames'],hazard)
    align(episode['frames'],decoded)
    frames=copy.deepcopy(episode['frames'])
    for frame,state in zip(frames,decoded):
        probability=tiny.probability(frame['tiny_features'])
        frame['arms']['TINY_LEARNED_PREDICTOR']={'route_risk':tiny.route_risk(frame['tiny_features']),
            'probability':probability,'minimum_entry_s':None}
        frame['arms']['X95_EVENT_CHALLENGER']=state
    return {'frames':frames,'x95_diagnostics':diagnostics,'status':'ALL_ARMS_PENDING_EXTERNAL_PREDICTION_SEAL'}
