"""Exclusive R1 prediction/fit/final stages over already sealed detector ledgers.

This command does not capture, detect, choose removals, resume, tune or repair.
A crash preserves seals and access markers. An existing execution directory is
never resumed; a consumed FIT/final marker cannot be cleared by this command.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
GROUPS=('FIT_ONLY','FINAL_A','FINAL_B')
REQUIRED_CODE=('run_dtr_final_prediction_stages.py','dtr_final_prediction_stages.py',
 'dtr_final_structural_prediction_adapter.py','dtr_final_classic_prediction_adapter.py',
 'dtr_final_score_adapter.py','dtr_final_reckoning_event_metrics.py',
 'dtr_carla_raw_kalman_baseline.py','dtr_carla_classic_motion_baselines.py',
 'dtr_carla_bounded_event_emitter.py','dtr_carla_x95_credentialed_hazard_state_model.py',
 'dtr_final_detector_intervention.py')


def require(condition,reason):
    if not condition:raise ValueError(reason)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest().upper()


def path_for(root,reference):
    path=Path(reference['path'])
    return (path if path.is_absolute() else root/path).resolve(strict=True)


def verified_json(root,reference):
    path=path_for(root,reference)
    require(sha(path)==reference['sha256'].upper(),'hash_mismatch:'+str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def seal(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False,separators=(',',':'))
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return {'path':str(path.resolve()),'sha256':sha(path)}


def _resolve_directory(root,value):
    path=Path(value)
    return (path if path.is_absolute() else root/path).resolve(strict=True)


def prepare(root,manifest):
    """Verify all groups and locks before invoking any prediction function.

    Truth paths may be resolved/stat'ed here, but their contents are not read or
    hashed. Their hashes are checked only behind the later access markers.
    """
    require(manifest['schema']=='dtr-final-inference-manifest-v1','inference_manifest_schema')
    require(set(manifest['groups'])==set(GROUPS),'exact_three_groups_required')
    if manifest.get('source_authority'):
        authority=verified_json(root,manifest['source_authority'])
        admission=verified_json(root,manifest['source_admission'])
        require(authority['claim']==manifest['source_claim'] and admission['claim']==authority['claim'],'composite_claim_binding')
        require(authority['status']=='NEW_DEVELOPMENT_COMPOSITE_SOURCE_SEALED' and authority['claim']=='DEVELOPMENT_COMPOSITE_REUSED_SOURCE_NOT_FRESH_CONFIRMATION','composite_development_only')
        require(admission['status']=='DEVELOPMENT_COMPOSITE_SOURCE_ADMITTED' and admission['execution_authority_sha256']==manifest['source_authority']['sha256'],'composite_source_admission')
        composite_root=path_for(root,manifest['source_authority']).parent
        require(path_for(root,manifest['source_admission'])==composite_root/'source-admission.json','composite_admission_path')
        for group in GROUPS:
            source_root=_resolve_directory(root,manifest['groups'][group]['source_root'])
            require(source_root==(composite_root/'raw'/group).resolve(strict=True),'composite_source_root_binding')
            require(sha(source_root/'r1-joined-result.json')==admission['joined_results'][group],'composite_join_admission_binding')
    roster=verified_json(root,manifest['roster'])
    require(roster['roster_id']=='DTR_FINAL_RECKONING_ROSTER_R1','R1_roster_required')
    for lock in roster['implementation_locks']:
        require(sha((REPO/lock['path']).resolve(strict=True))==lock['sha256'].upper(),'frozen_roster_implementation_drift')
    code={}
    for ref in manifest['code_files']:
        path=path_for(root,ref)
        require(path not in code,'duplicate_code_lock')
        require(sha(path)==ref['sha256'].upper(),'code_lock_drift:'+path.name)
        code[path]=ref
    require(all((HERE/name).resolve() in code for name in REQUIRED_CODE),'incomplete_stage_code_locks')
    dependency_lock=verified_json(root,manifest['structural_dependency_lock'])
    # Imports have no prediction/fit effects; all code hashes are checked first.
    import dtr_final_structural_prediction_adapter as structural
    require(dependency_lock==structural.dependency_manifest(),'structural_dependency_drift')
    annexes={};settings={};gates={};seen_truth_paths=set()
    strata={s['stratum_id'] for s in roster['source_design']['strata']}
    for group in GROUPS:
        entry=manifest['groups'][group]
        annex=verified_json(root,entry['annex'])
        require(annex['group']==next(g for g in roster['source_design']['seed_groups'] if g['group_id']==group) and set(annex['strata'])==strata,'annex_group_or_strata')
        ids={r['episode_id'] for r in annex['strata'].values()}
        require(len(ids)==10 and set(entry['truth_episodes'])==ids,'exact_ten_main_truth_references')
        source_root=_resolve_directory(root,entry['source_root'])
        for ep,ref in entry['truth_episodes'].items():
            truth_path=path_for(root,ref)
            require(truth_path==(source_root/'evaluator/episodes'/ep/'frames.jsonl').resolve(strict=True),'truth_reference_outside_native_main_episode')
            require(truth_path not in seen_truth_paths,'cross_group_truth_reuse')
            seen_truth_paths.add(truth_path)
        source_gate=verified_json(root,entry['source_gate'])
        require(source_gate['status']=='SOURCE_GATE_MET' and source_gate['source_semantics']['status']=='SOURCE_GATE_MET','all_source_gates_required:'+group)
        provenance=source_gate['provenance']
        require(not provenance.get('historical_diagnostic',False),'historical_probe_cannot_admit_formal_source')
        require(Path(provenance['source_root']).resolve()==source_root,'source_gate_root_binding')
        require(provenance['annex_sha256'].upper()==entry['annex']['sha256'].upper() and provenance['roster_sha256'].upper()==manifest['roster']['sha256'].upper(),'source_gate_protocol_binding')
        require(set(provenance['verified_sensors'])=={'instance','witness'},'native_source_modalities_not_verified')
        gate_rows=source_gate['source_semantics']['strata']
        require(len(gate_rows)==10 and {r['stratum_id'] for r in gate_rows}==strata and all(r['status']=='SOURCE_GATE_MET' for r in gate_rows),'source_stratum_gate_incomplete')
        detector_gate=verified_json(root,entry['detector_intervention_gate'])
        require(detector_gate['status']=='PASS','measured_detector_credential_gate_required')
        require(detector_gate['intervened_sha256'].upper()==entry['intervened_values_path']['sha256'].upper(),'intervention_gate_ledger_binding')
        require(set(detector_gate['episodes'])==ids,'intervention_gate_episode_set')
        annexes[group]=annex;settings[group]=entry;gates[group]=detector_gate
    # No predictor is called until all three groups pass the above checks.
    import dtr_carla_x24_plan_adherent_predictor as x24
    prepared={}
    for group in GROUPS:
        entry=settings[group]
        freeze=verified_json(root,entry['raw_x24_freeze'])
        freeze_path=path_for(root,entry['raw_x24_freeze'])
        require(freeze_path.name=='freeze-x24.json','retained_raw_freeze_filename')
        checked,contract,raw=x24.require_freeze(freeze_path.parent)
        require(checked==freeze,'raw_freeze_changed_during_preflight')
        require(contract.model_root.resolve()==_resolve_directory(root,entry['model_root']),'model_root_freeze_binding')
        require(gates[group]['raw_candidate_aggregate_sha256'].upper()==freeze['candidates']['aggregate_sha256'].upper(),'intervention_gate_raw_binding')
        intervened=verified_json(root,entry['intervened_values_path'])['episodes']
        annex=annexes[group]; ids={r['episode_id'] for r in annex['strata'].values()}
        require({ep.episode_id for ep in contract.episodes}==ids and len(contract.episodes)==10 and set(intervened)==ids,'model_or_intervention_main_episode_set')
        per_id={r['episode_id']:r for r in annex['strata'].values()}
        cursor=0
        for episode in contract.episodes:
            values=raw[cursor:cursor+len(episode.observations)];cursor+=len(values)
            expected=copy.deepcopy(values)
            removal_indices={i for window in per_id[episode.episode_id].get('removal_windows',[]) for i in window}
            by_sample={o.sample_index:i for i,o in enumerate(episode.observations)}
            for index in removal_indices:
                require(index in by_sample,'removal_index_missing')
                row=expected[by_sample[index]]
                require(bool(row['candidates']),'removed_frame_has_no_raw_candidates')
                row.update(candidates=[],candidate_count=0,candidate_counts_by_class={})
            require(expected==intervened[episode.episode_id],'nonfrozen_intervention_difference:'+episode.episode_id)
        require(cursor==len(raw),'raw_candidate_tail')
        prepared[group]={'contract':contract,'values':intervened,'annex':annex,'entry':entry}
    return roster,dependency_lock,prepared


def open_truth(root,entry):
    """Call only after a durable exclusive group access marker exists."""
    result={}
    for ep,ref in entry['truth_episodes'].items():
        path=path_for(root,ref)
        require(sha(path)==ref['sha256'].upper(),'truth_hash_mismatch_after_access:'+ep)
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        require(rows and all(r['episode_id']==ep for r in rows),'truth_episode_identity')
        result[ep]=rows
    return result


def execute_phases(root,destination,manifest,roster,dependency_lock,prepared):
    import dtr_final_structural_prediction_adapter as structural
    import dtr_final_classic_prediction_adapter as classic
    import dtr_carla_raw_kalman_baseline as raw
    import dtr_final_prediction_stages as stages
    import dtr_final_score_adapter as score
    nonlearned={};seals={};phase=0
    def checkpoint(name,payload):
        nonlocal phase
        phase+=1
        return seal(destination/f'phase-{phase:02d}-{name}.json',payload)
    for group in GROUPS:
        setup=prepared[group];contract=setup['contract'];episodes={}
        for episode in contract.episodes:
            values=setup['values'][episode.episode_id]
            raw_result=raw.predict_episode(episode,values,contract.calibration)
            classic_result=classic.predict_episode(episode,values,contract.calibration)
            structural_result=structural.predict_episode(episode,values,contract.calibration,dependency_lock=dependency_lock)
            episodes[episode.episode_id]=stages.merge_nonlearned(raw_result,classic_result,structural_result)
        nonlearned[group]=episodes
        seals[group]=seal(destination/f'{group}-nonlearned.json',{'group':group,'episodes':episodes})
        checkpoint('nonlearned-'+group,{'status':'SEALED','prediction':seals[group]})
    # Verify all nine-arm seals exist, across all groups, before any fit access.
    for ref in seals.values():verified_json(root,ref)
    checkpoint('all-nonlearned-sealed',{'predictions':seals})
    # A failure after this marker is consumed; no resume implementation exists.
    seal(destination/'FIT_ONLY-access.json',{'status':'CONSUMED_ON_ACCESS','input_predictions':seals,
         'truth_references':manifest['groups']['FIT_ONLY']['truth_episodes']})
    fit_truth=open_truth(root,prepared['FIT_ONLY']['entry'])
    models=stages.fit_models('FIT_ONLY',nonlearned['FIT_ONLY'],fit_truth,prepared['FIT_ONLY']['annex'])
    model_ref=seal(destination/'fitted-models.json',models)
    checkpoint('fit-models-sealed',{'models':model_ref})
    del fit_truth
    verified_json(root,model_ref)
    final_predictions={};final_seals={}
    for group in ('FINAL_A','FINAL_B'):
        episodes={ep:stages.apply_learned(value,models) for ep,value in nonlearned[group].items()}
        final_predictions[group]={ep:value['frames'] for ep,value in episodes.items()}
        final_seals[group]=seal(destination/f'{group}-all-arms.json',{'group':group,'episodes':episodes,'models':model_ref})
        checkpoint('all-arms-'+group,{'prediction':final_seals[group]})
    for ref in final_seals.values():verified_json(root,ref)
    checkpoint('all-final-predictions-sealed',{'predictions':final_seals})
    seal(destination/'FINAL-access.json',{'status':'CONSUMED_ON_ACCESS','predictions':final_seals,
         'truth_references':{g:manifest['groups'][g]['truth_episodes'] for g in ('FINAL_A','FINAL_B')}})
    truth={g:open_truth(root,prepared[g]['entry']) for g in ('FINAL_A','FINAL_B')}
    scored=score.score_final(roster,truth,final_predictions,{g:prepared[g]['annex'] for g in truth})
    if 'source_claim' in manifest:scored['claim']=manifest['source_claim']
    if manifest.get('source_authority'):
        scored['source_authority']=manifest['source_authority']
        scored['fresh_confirmation']=False
    score_ref=seal(destination/'final-score.json',scored)
    checkpoint('final-score-sealed',{'score':score_ref,'terminal':'NO_X97_NO_RESCUE_ACCEPT_OBSERVED_RESULT'})
    return score_ref


def run(root):
    root=root.resolve(strict=True)
    manifest_path=root/'inference-manifest.json'
    # Exclusive output directory also blocks any restart after a partial marker.
    destination=root/'prediction-stages'
    destination.mkdir()
    try:
        frozen_bytes=manifest_path.read_bytes()
        with (destination/'inference-manifest.json').open('xb') as stream:
            stream.write(frozen_bytes);stream.flush();os.fsync(stream.fileno())
        manifest=json.loads(frozen_bytes.decode('utf-8-sig'))
        freeze=seal(destination/'manifest-freeze.json',{'sha256':hashlib.sha256(frozen_bytes).hexdigest().upper(),
             'status':'FROZEN_BEFORE_PREDICTIONS','resume_allowed':False})
        roster,lock,prepared=prepare(root,manifest)
        seal(destination/'preflight.json',{'status':'PASS','manifest_freeze':freeze,'all_three_groups_verified':True})
        return execute_phases(root,destination,manifest,roster,lock,prepared)
    except Exception as error:
        seal(destination/'failure.json',{'status':'STOPPED_PRESERVE_ALL_SEALS_AND_ACCESS_MARKERS',
             'error_type':type(error).__name__,'reason':str(error),'resume_allowed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.run_root)
    print(json.dumps({'status':'FINAL_SCORE_SEALED','score':result}))


if __name__=='__main__':main()
