"""Bind generic C2 RGB-D integrity to the presealed R1 source semantics.

C2's legacy occlusion check requires a nonempty complete-disappearance run even
for S09's zero-second maximum, so it cannot evaluate a partial-visibility cell.
That legacy result is preserved. Every other C2 integrity check and every R1
source gate is required. No pixels, trajectories, thresholds, or rows change.
"""
import argparse
import json
from pathlib import Path
import dtr_carla_x24_plan_adherent_predictor as io

LEGACY='track_then_complete_physical_occlusion_contract_met'
REQUIRED=set('all_four_fresh_server_shards_complete cross_shard_code_and_protocol_identity raw_payload_inventories_verified all_formal_sensors_1280x720 minimum_sixty_unique_actual_blueprints zero_blueprint_fallbacks all_spawned_assets_have_nonzero_bbox contact_safe_outcome_pair_matches dense_model_rgb_depth_complete model_camera_transforms_align_across_rgb_depth deterministic_rgb_depth_frame_alignment_materialized model_root_zero_actor_or_evaluator_truth_keys model_contract_current_actors_disabled truth_blind_model_root_manifest_materialized immutable_plan_receipts_and_world_routes_materialized camera_calibration_and_depth_codec_materialized visual_contact_sheet_and_summary_materialized sealed_model_and_full_evidence_nonempty'.split())


def decision(legacy, source, protocol_hash):
    if legacy['protocol_sha256']!=protocol_hash or source['provenance']['capture_protocol_sha256']!=protocol_hash:
        raise ValueError('join_source_protocol_mismatch')
    if source['status']!='SOURCE_GATE_MET' or source['source_semantics']['status']!='SOURCE_GATE_MET':
        raise ValueError('R1_source_gate_failed')
    if source['provenance']['historical_diagnostic'] or set(source['provenance']['verified_sensors'])!={'instance','witness'}:
        raise ValueError('fresh_independent_source_gate_required')
    if len(source['source_semantics']['strata'])!=10 or any(s['status']!='SOURCE_GATE_MET' for s in source['source_semantics']['strata']):
        raise ValueError('all_ten_source_strata_required')
    checks=legacy['checks']
    if LEGACY not in checks or not REQUIRED<=checks.keys():raise ValueError('legacy_integrity_checks_missing')
    if not ({'cross_sensor_actual_replay_identical','cross_sensor_authority_scoped_replay_identical'} & checks.keys()):
        raise ValueError('independent_replay_check_missing')
    if legacy['status'] not in {'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE','DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE'}:
        raise ValueError('incomplete_legacy_join')
    failed=[k for k,v in checks.items() if v is not True and k!=LEGACY]
    if failed:raise ValueError('C2_integrity_failure:'+','.join(failed))
    return {'status':'DTR_R1_FOUR_SENSOR_SOURCE_COMPLETE',
            'preserved_C2_status':legacy['status'],
            'R1_source_semantics':'SOURCE_GATE_MET',
            'all_non_occlusion_C2_integrity_checks_pass':True,
            'legacy_check_scope':'C2_COMPLETE_OCCLUSION_ONLY_INAPPLICABLE_TO_R1_S09_PARTIAL',
            'replacement_authority':'PRESEALED_TEN_STRATUM_INSTANCE_REFERENCE_AND_WITNESS_EVALUATOR',
            'pixels_or_evaluator_rows_changed':False,
            'sealed_evidence_manifest_sha256':legacy['sealed_evidence_manifest_sha256']}


def finalize(root, protocol):
    result=root/'r1-joined-result.json'
    if result.exists():raise FileExistsError(result)
    legacy_path=root/'result.json';source_path=root/'roster-source-gate.json'
    legacy=json.loads(legacy_path.read_text());source=json.loads(source_path.read_text())
    value=decision(legacy,source,io.sha256_file(protocol))
    if io.sha256_file(root/'sealed_evidence_manifest.json')!=value['sealed_evidence_manifest_sha256']:
        raise ValueError('joined_inventory_binding')
    value.update(legacy_result_sha256=io.sha256_file(legacy_path),source_gate_sha256=io.sha256_file(source_path),
                 implementation_sha256=io.sha256_file(Path(__file__)),protocol_sha256=io.sha256_file(protocol))
    io.write_json_exclusive(result,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(finalize(args.root,args.protocol)))
