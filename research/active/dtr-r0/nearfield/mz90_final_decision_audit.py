"""Read-only trace of sealed matched_hold errors, evidence age and source coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import mz90_observable_contract as contract
import mz90_radar_fallback_audit as provenance


def replay_source(root, regime):
    text=provenance.instrument((root/'mz90_observation_source.py').read_text())
    replacements=[
        ('    return raw, evaluator, origins','    return raw, evaluator, origins, tof_origins'),
        ('    origins = np.full((n, 4), "NONE", dtype="U24")',
         '    tof_origins = np.full((n, 8), "NONE", dtype="U24")\n    origins = np.full((n, 4), "NONE", dtype="U24")'),
        ('for r, a, _, reflectivity in sorted(returns):',
         'for origin_index, (r, a, _, reflectivity) in sorted(enumerate(returns), key=lambda pair: pair[1]):'),
        ('                    raw["tof_status"][j, b] = 5',
         '                    raw["tof_status"][j, b] = 5\n                    tof_origins[j, b] = f"real:{origin_index}"')]
    for before,after in replacements:
        assert text.count(before)==1,before
        text=text.replace(before,after)
    namespace={'__name__':'final_decision_provenance'}
    exec(compile(text,'<sealed-source-labels>','exec'),namespace)
    source=json.loads((root/'source.json').read_text())
    raw,evaluator,radar_origins,tof_origins=namespace['materialize'](source,regime)
    provenance.parity(raw,dict(np.load(root/regime/'raw-observations.npz')))
    provenance.parity(evaluator,dict(np.load(root/regime/'evaluator.npz')))
    hazard_radar=np.zeros_like(raw['radar_valid']);hazard_tof=np.zeros_like(raw['tof_status'],bool)
    for i in range(len(raw['episode_id'])):
        scene=source[int(evaluator['scene'][i])];t=raw['time_s'][i]
        hazardous=set()
        for k,obj in enumerate(scene['objects']):
            if namespace['geometric_hazard'](obj['x']+obj['vx']*t,obj['z']+(obj['vz']-.7)*t,obj['vx'],obj['vz']-.7):
                hazardous.add(f'real:{k}')
        hazard_radar[i]=[label in hazardous for label in radar_origins[i]]
        hazard_tof[i]=[label in hazardous for label in tof_origins[i]]
    return raw,evaluator,hazard_radar,hazard_tof


def trace(raw):
    """Causal trace only; no evaluator/provenance input."""
    obs=contract.adapt(raw);n=len(obs['episode_id']);ids=obs['episode_id']
    no_radar={**obs,'radar_valid':np.zeros_like(obs['radar_valid'])}
    tof_controls=contract.simple_controls(no_radar)
    direct,tof=tof_controls['current_geometry'],tof_controls['matched_hard']
    temporal=np.zeros(n,bool);temporal_previous=[None]*n
    yaw,_=contract.old.integrate_with_uncertainty(obs['delta_yaw'],obs['imu_valid'],ids)
    pitch,_=contract.old.integrate_with_uncertainty(obs['delta_pitch'],obs['imu_valid'],ids)
    history=[];history_i=None
    for i in range(n):
        if i==0 or ids[i]!=ids[i-1]:history=[];history_i=None
        if not obs['tof_known'][i]:continue
        points=[(r,contract.previous.rotate_ray(obs['tof_theta_deg'][i,k],yaw[i],pitch[i])[0],'UNKNOWN',True)
                for k,r in enumerate(obs['tof_range_m'][i]) if np.isfinite(r)]
        dt=None if history_i is None else (i-history_i)*contract.previous.DT
        pairs=contract.previous.matches(history,points) if dt is not None and dt<=.3+1e-9 else {}
        for k,(r,angle,_,_) in enumerate(points):
            if k in pairs and r<3.6:
                temporal[i]|=contract.old.segment_intersects_corridor(angle,angle+(angle-history[pairs[k]][1])/dt)
        temporal_previous[i]=history_i if temporal[i] else None
        history,history_i=points,i
    np.testing.assert_array_equal(direct|temporal,tof)
    q=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
    raw_support=q.any(axis=1)
    radar=contract.old.radar_expert(obs['radar_range_m'],obs['radar_velocity'],obs['radar_angle'],obs['radar_valid'],ids)
    hard=np.where(obs['tof_known'],tof,radar)
    final=contract.simple_controls(obs)['matched_hold']
    rows=[];activation=None;seeds=[];latest=None;last_tof=None;segment=None
    for i in range(n):
        reset=i==0 or ids[i]!=ids[i-1]
        if reset:activation=None;seeds=[];latest=None;last_tof=None;segment=None
        if raw_support[i]:latest=i
        if radar[i] and (reset or not radar[i-1]):
            activation=i
            seeds=[j for j in range(max(0,i-2),i+1) if ids[j]==ids[i] and raw_support[j]]
            assert len(seeds)>=2
        if not radar[i]:activation=None;seeds=[]
        held=bool(final[i] and not hard[i])
        if held:assert not reset and hard[i-1] and not obs['tof_known'][i]
        if final[i] and (reset or not final[i-1]):segment=i
        if not final[i]:segment=None
        sensor='NONE';mode='NONE';witness=[];origin=None
        if hard[i]:
            sensor='ToF' if obs['tof_known'][i] else 'Radar';origin=i
            if sensor=='ToF':
                mode='current_direct' if direct[i] else 'current_plus_temporal_geometry'
                witness=[i] if direct[i] else [last_tof,i]
                assert None not in witness
            else:
                mode='current_return_with_hysteresis' if raw_support[i] else 'hysteresis_without_current_return'
                witness=sorted(set(seeds+[latest]))
        elif held:
            previous=rows[i-1];sensor=previous['origin_sensor'];origin=i-1
            mode='outer_one_frame_hold';witness=previous['witness_frames']
        row=dict(frame=i,episode=str(ids[i]),time_s=float(raw['time_s'][i]),
            selected_sensor='ToF' if obs['tof_known'][i] else 'Radar',origin_sensor=sensor,evidence_mode=mode,
            tof_known=bool(obs['tof_known'][i]),tof_packet_received=bool(raw['tof_packet_received'][i]),
            tof_branch=bool(tof[i]),tof_direct=bool(direct[i]),radar_raw=bool(raw_support[i]),radar_branch=bool(radar[i]),
            tof_temporal=bool(temporal[i]),tof_temporal_witness_frames=[] if not temporal[i] else [temporal_previous[i],i],
            radar_valid_returns=int(obs['radar_valid'][i].sum()),hard=bool(hard[i]),final=bool(final[i]),
            outer_hold=held,origin_hard_frame=origin,witness_frames=witness,
            radar_activation_frame=activation,radar_activation_seed_frames=seeds.copy(),
            radar_latest_raw_support_frame=latest,final_segment_start=segment)
        if held:
            row['inherited_evidence_mode']=rows[i-1]['evidence_mode']
            row['inherited_radar_activation_frame']=rows[i-1]['radar_activation_frame'] if sensor=='Radar' else None
        rows.append(row)
        if obs['tof_known'][i]:last_tof=i
    return obs,q,rows


def summarize(rows):
    paths={};fn=Counter();coverage=Counter();selection=Counter()
    for r in rows:
        outcome=r['outcome']
        if r['final']:
            key=r['origin_sensor']+' / '+r['evidence_mode']
            paths.setdefault(key,Counter())[outcome]+=1
        if outcome=='FN':fn[r['fn_path']]+=1
        if outcome in ('FN','TP'):
            key=('ToF+Radar' if r['hazard_tof_return'] and r['hazard_radar_return'] else
                 'ToF_only' if r['hazard_tof_return'] else 'Radar_only' if r['hazard_radar_return'] else 'no_hazardous_return')
            coverage[f'{outcome} / {key}']+=1
        if r['tof_known'] and not r['tof_branch'] and r['radar_branch']:
            selection[outcome]+=1
    gate_cohorts={}
    predicates=dict(raw_Radar_waiting=lambda r:not r['tof_known'] and r['radar_raw'] and not r['radar_branch'],
        valid_Radar_all_fail_gate=lambda r:not r['tof_known'] and r['radar_valid_returns']>0 and not r['radar_raw'],
        no_valid_current_returns=lambda r:not r['tof_known'] and r['radar_valid_returns']==0)
    for name,predicate in predicates.items():gate_cohorts[name]=dict(Counter(r['outcome'] for r in rows if predicate(r)))
    gate_failures=Counter()
    for r in rows:
        if r.get('fn_path')=='Radar_returns_fail_raw_gate':
            gate_failures['hazardous_radar_return_present' if r['hazard_radar_return'] else 'no_hazardous_radar_return']+=1
            for reason in r['hazard_radar_all_fail_criteria']:gate_failures[reason]+=1
    return dict(outcomes=dict(Counter(r['outcome'] for r in rows)),alert_paths=paths,fn_paths=fn,
                gate_cohorts=gate_cohorts,fn_Radar_gate_details=gate_failures,
                current_hazardous_source_return_coverage=coverage,tof_nonalert_over_radar_alert=selection)


def run(root,out):
    if out.exists():raise ValueError('Audit output must be new')
    receipt=json.loads((root/'receipt.json').read_text())
    for name,digest in receipt['hashes'].items():assert contract.old.sha(root/name)==digest,name
    summaries={};all_rows={}
    for regime in ('ideal','sensor_proxy'):
        raw,evaluator,hazard_radar,hazard_tof=replay_source(root,regime)
        obs,q,rows=trace(raw);truth=evaluator['truth']
        sealed=np.load(root/regime/'predictions.npz')
        np.testing.assert_array_equal([r['hard'] for r in rows],sealed['matched_hard'])
        np.testing.assert_array_equal([r['final'] for r in rows],sealed['matched_hold'])
        for i,r in enumerate(rows):
            r['truth']=bool(truth[i]);r['outcome']=('TP' if truth[i] else 'FP') if r['final'] else ('FN' if truth[i] else 'TN')
            r['hazard_tof_return']=bool(hazard_tof[i].any());r['hazard_radar_return']=bool(hazard_radar[i].any())
            r['hazard_radar_qualifies']=bool((hazard_radar[i]&q[i]).any())
            criteria=dict(range=obs['radar_range_m'][i]<3.18,closing_speed=obs['radar_velocity'][i]<=-.35,
                          bearing=np.abs(obs['radar_angle'][i])<=20)
            r['hazard_radar_all_fail_criteria']=[name for name,mask in criteria.items()
                if hazard_radar[i].any() and not (hazard_radar[i]&mask).any()]
            r['witness_frame_truth']=[bool(truth[j]) for j in r['witness_frames']]
            r['final_segment_initial_truth']=None if r['final_segment_start'] is None else bool(truth[r['final_segment_start']])
            if r['outcome']=='FN':
                if r['tof_known']:
                    r['fn_path']='selected_ToF_nonalert_over_Radar_alert' if r['radar_branch'] else 'selected_ToF_nonalert_no_Radar_alert'
                elif r['radar_raw']:r['fn_path']='Radar_raw_support_not_activated'
                elif r['radar_valid_returns']:r['fn_path']='Radar_returns_fail_raw_gate'
                else:r['fn_path']='no_valid_current_ToF_or_Radar_return'
        summaries[regime]=summarize(rows);all_rows[regime]=rows
    assert [r['truth'] for r in all_rows['ideal']]==[r['truth'] for r in all_rows['sensor_proxy']]
    assert [(r['episode'],r['time_s']) for r in all_rows['ideal']]==[(r['episode'],r['time_s']) for r in all_rows['sensor_proxy']]
    transitions=Counter();paired=[]
    for a,b in zip(all_rows['ideal'],all_rows['sensor_proxy']):
        transitions[a['outcome']+' -> '+b['outcome']]+=1
        if a['outcome']!=b['outcome']:
            paired.append(dict(frame=a['frame'],transition=a['outcome']+' -> '+b['outcome'],
                ideal_selected=a['selected_sensor'],proxy_selected=b['selected_sensor'],
                proxy_path=b.get('fn_path',b['origin_sensor']+' / '+b['evidence_mode']),
                proxy_hazard_tof=b['hazard_tof_return'],proxy_hazard_radar=b['hazard_radar_return']))
    result=dict(status='PASS',analysis='POSTHOC_FROZEN_FINAL_DECISION_AUDIT',regimes=summaries,
        paired_transitions=transitions,sealed_input_hashes_verified=len(receipt['hashes']),
        exact_raw_evaluator_parity=True,exact_hard_and_hold_parity=True,policy_counterfactuals=0)
    out.mkdir(parents=True)
    for regime,rows in all_rows.items():contract.old.write(out/(regime+'-rows.json'),rows)
    contract.old.write(out/'paired-transitions.json',paired)
    contract.old.write(out/'result.json',result)
    contract.old.write(out/'receipt.json',dict(hashes={p.name:contract.old.sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source_run,args.output)
