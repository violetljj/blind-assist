"""Prepare once-only shared R1 ledgers after all three complete source gates.

No fitting or final evaluator rows are read. Truth hashes come only from the
producer's sealed file inventory. Output is exclusive and failure is retained.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_final_structural_prediction_adapter as structural
import dtr_final_detector_intervention as intervention
import materialize_dtr_final_main_model_view as main_view

HERE=Path(__file__).resolve().parent
GROUPS=('FIT_ONLY','FINAL_A','FINAL_B')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):x24.write_json_exclusive(path,value)
def ref(path):return {'path':str(path.resolve(strict=True)),'sha256':x24.sha256_file(path)}


def prepare(source_seal, output, model):
    source_seal=source_seal.resolve(strict=True);model=model.resolve(strict=True)
    authority=None
    if (source_seal/'execution-authority.json').exists():
        authority=read(source_seal/'execution-authority.json')
        admission=read(source_seal/'source-admission.json')
        if authority['status']!='NEW_DEVELOPMENT_COMPOSITE_SOURCE_SEALED' or admission['status']!='DEVELOPMENT_COMPOSITE_SOURCE_ADMITTED':
            raise ValueError('composite_source_not_admitted')
        if authority['claim']!='DEVELOPMENT_COMPOSITE_REUSED_SOURCE_NOT_FRESH_CONFIRMATION' or admission['claim']!=authority['claim']:
            raise ValueError('composite_claim_binding')
        if admission['execution_authority_sha256']!=x24.sha256_file(source_seal/'execution-authority.json'):
            raise ValueError('composite_authority_binding')
        if admission['fast_png_validation_sha256']!=x24.sha256_file(source_seal/'fast-png-validation.json'):
            raise ValueError('composite_pixel_validation_binding')
        if read(source_seal/'fast-png-validation.json')['status']!='PASS':
            raise ValueError('composite_pixel_validation_failed')
    inputs={}
    for group in GROUPS:
        root=source_seal/'raw'/group
        gate=root/'roster-source-gate.json'
        if read(gate)['status']!='SOURCE_GATE_MET':raise ValueError('three_source_groups_required')
        result=read(root/'r1-joined-result.json')
        if authority is not None and admission['joined_results'][group]!=x24.sha256_file(root/'r1-joined-result.json'):
            raise ValueError('composite_join_admission_binding')
        if result['status']!='DTR_R1_FOUR_SENSOR_SOURCE_COMPLETE':raise ValueError('four_sensor_join_required')
        if result['legacy_result_sha256']!=x24.sha256_file(root/'result.json') or result['source_gate_sha256']!=x24.sha256_file(gate):
            raise ValueError('R1_join_lineage_binding')
        inventory=root/'sealed_evidence_manifest.json'
        if x24.sha256_file(inventory)!=result['sealed_evidence_manifest_sha256']:raise ValueError('producer_inventory_hash')
        by_path={row['path']:row for row in read(inventory)}
        truths={}
        for ep in main_view.MAIN_IDS:
            relative=f'evaluator/episodes/{ep}/frames.jsonl'
            truths[ep]={'path':str((root/relative).resolve()),'sha256':by_path[relative]['sha256']}
        inputs[group]={'source_root':root,'source_gate':ref(gate),'truth_episodes':truths,
                       'annex':ref(source_seal/group/'annex.json')}
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_FOR_FROZEN_DETECTOR_PREPARATION')
    output.mkdir(parents=True,exist_ok=False)
    lock=structural.dependency_manifest()
    write(output/'structural-dependency-lock.json',lock)
    names=['prepare_dtr_final_detector_ledgers.py','run_dtr_final_prediction_stages.py',
           'dtr_final_prediction_stages.py','dtr_final_classic_prediction_adapter.py',
           'dtr_final_structural_prediction_adapter.py','dtr_final_score_adapter.py',
           'dtr_final_reckoning_event_metrics.py','dtr_final_detector_intervention.py',
           'materialize_dtr_final_main_model_view.py','finalize_dtr_final_roster_join.py','join_dtr_final_roster_source.py','dtr_carla_bounded_event_emitter.py',
           'dtr_carla_raw_kalman_baseline.py','dtr_carla_classic_motion_baselines.py',
           'dtr_carla_x95_credentialed_hazard_state_model.py','dtr_carla_yolo_metric_candidates.py']
    files={item['path']:item for item in lock['files'].values()}
    for name in names:files[str((HERE/name).resolve())]=ref(HERE/name)
    code=list(files.values())
    freeze={'schema':'dtr-final-detector-preparation-v1','status':'PRE_DETECTOR_FROZEN',
            'source_claim':authority['claim'] if authority else 'SYNTHETIC_R1_ONLY',
            'model':ref(model),'code_files':code,'device':'cuda:0','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0),'python':ref(Path(sys.executable)),
            'removal_eligibility':'FIXED_X24_CURRENT_MEASURED_CONFIRMED_RISK_ON_ADJACENT_RAW_FRAMES_NO_INDEX_RESCUE',
            'removal_scope':'ALL_CANDIDATES_ON_FIXED_FRAME',
            'truth_rows_opened':False}
    write(output/'preparation-freeze.json',freeze)
    started=time.perf_counter();groups={}
    try:
        for group in GROUPS:
            if x24.sha256_file(model)!=freeze['model']['sha256']:raise ValueError('detector_weight_drift')
            for file in code:
                if x24.sha256_file(Path(file['path']))!=file['sha256']:raise ValueError('preparation_code_drift')
            print('PREPARE_SHARED_DETECTOR '+group,flush=True)
            directory=output/group;directory.mkdir()
            model_root=directory/'model'
            main_view.materialize(inputs[group]['source_root']/'model',model_root,hardlink=True)
            raw=directory/'raw-ledger';raw.mkdir()
            args=SimpleNamespace(run_root=raw,model_root=model_root,model_manifest_sha256=x24.sha256_file(model_root/'manifest.json'))
            x24.build_index(args)
            temporary=directory/'temporary';temporary.mkdir()
            env=dict(os.environ,TEMP=str(temporary),TMP=str(temporary))
            command=[sys.executable,str(HERE/'dtr_carla_yolo_metric_candidates.py'),
                     '--image-index',str(raw/'x24-rgb-index.jsonl'),'--model',str(model),
                     '--output-dir',str(raw/'candidates'),'--device','0']
            with (directory/'detector.stdout.log').open('x') as stdout,(directory/'detector.stderr.log').open('x') as stderr:
                completed=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,shell=False)
            if completed.returncode:raise RuntimeError('detector_failed_preserve_partial_ledger')
            x24.freeze(args)
            frozen,contract,values=x24.require_freeze(raw)
            if frozen['candidates']['detector_model']['sha256']!=freeze['model']['sha256']:
                raise ValueError('detector_result_model_drift')
            annex=read(Path(inputs[group]['annex']['path']))
            specs={v['episode_id']:(s,v) for s,v in annex['strata'].items()}
            cursor=0;transformed={};receipts={};credentials={}
            for episode in contract.episodes:
                count=len(episode.observations);candidates=values[cursor:cursor+count];cursor+=count
                # Raw X24 output serves admission only, never a scored extra arm.
                credential=x24.predict_episode(episode,candidates,contract.calibration)
                credentials[episode.episode_id]=credential
                write(directory/f'raw-credential-{episode.episode_id}.json',credential)
                stratum,spec=specs[episode.episode_id]
                transformed[episode.episode_id],receipts[episode.episode_id]=intervention.intervene_episode(
                    candidates,credential,stratum,spec.get('removal_windows',[]))
            write(directory/'raw-x24-credential-check.json',{'episodes':credentials,'scope':'ADMISSION_ONLY_NO_TRUTH_OR_SCORE'})
            write(directory/'intervened-values.json',{'episodes':transformed})
            gate={'status':'PASS','intervened_sha256':x24.sha256_file(directory/'intervened-values.json'),
                  'raw_candidate_aggregate_sha256':frozen['candidates']['aggregate_sha256'],'episodes':receipts}
            write(directory/'detector-intervention-gate.json',gate)
            groups[group]={'model_root':str(model_root.resolve()),'raw_x24_freeze':ref(raw/'freeze-x24.json'),
                           'intervened_values_path':ref(directory/'intervened-values.json'),
                           'annex':inputs[group]['annex'],'source_gate':inputs[group]['source_gate'],
                           'detector_intervention_gate':ref(directory/'detector-intervention-gate.json'),
                           'source_root':str(inputs[group]['source_root'].resolve()),'truth_episodes':inputs[group]['truth_episodes']}
        manifest={'schema':'dtr-final-inference-manifest-v1','roster':ref(HERE/'dtr_final_reckoning_roster_protocol.json'),
                  'source_claim':freeze['source_claim'],
                  'source_authority':ref(source_seal/'execution-authority.json') if authority else None,
                  'source_admission':ref(source_seal/'source-admission.json') if authority else None,
                  'structural_dependency_lock':ref(output/'structural-dependency-lock.json'),
                  'code_files':code,'groups':groups}
        write(output/'inference-manifest.json',manifest)
        write(output/'preparation-result.json',{'status':'SHARED_RAW_AND_INTERVENED_LEDGERS_SEALED',
              'elapsed_s':time.perf_counter()-started,'manifest':ref(output/'inference-manifest.json'),
              'device':'cuda:0','truth_rows_opened':False})
    except Exception as error:
        write(output/'preparation-result.json',{'status':'NOT_EVALUABLE_PREPARATION_FAILED_NO_FIT_OR_FINAL_ACCESS',
              'reason':str(error),'elapsed_s':time.perf_counter()-started,'truth_rows_opened':False})
        raise
    finally:
        for directory in output.glob('*/temporary'):
            if directory.is_dir() and not any(directory.iterdir()):directory.rmdir()
    return ref(output/'inference-manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-seal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source_seal,args.output,args.model)))
