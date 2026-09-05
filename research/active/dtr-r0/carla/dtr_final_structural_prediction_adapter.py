"""Exact frozen X24/X73/X94 composition over already-sanitized causal RGB-D.

No historical runner, WORK bootstrap, source truth, detector invocation, fit or
scoring is imported. Call dependency_manifest() before execution and freeze its
return alongside the shared detector ledger; prediction requires that lock.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
CLOSURE=HERE/'dtr_carla_c44_x96_dropout_survival_protocol.json'
CLOSURE_SHA256='7C940C55362585D553BF86E198BEA42B641F29AA0AD730B99E4E7479EF881555'
MODULES={'x24': 'dtr_carla_x24_plan_adherent_predictor', 'x25': 'dtr_carla_x25_rigid_footprint_predictor', 'x54': 'dtr_carla_x54_metric_bootstrap_dropout_continuation', 'x65': 'dtr_carla_x65_ancestry_synchronized_conflict_handback', 'x67': 'dtr_carla_x67_measurement_horizon_receding_release', 'x68': 'dtr_carla_x68_object_local_lateral_dequantization', 'x69': 'dtr_carla_x69_mature_cross_route_rigid_contradiction', 'x70': 'dtr_carla_x70_triple_credential_surface_dropout_handback', 'x71': 'dtr_carla_x71_entry_cotransport_occupancy_birth', 'x72': 'dtr_carla_x72_credentialed_surface_boundary_completion', 'x73': 'dtr_carla_x73_credentialed_parent_hull_reconstruction', 'x74': 'dtr_carla_x74_metric_handback_class_contradiction', 'x75': 'dtr_carla_x75_collision_credentialed_object_permanence', 'x76': 'dtr_carla_x76_zero_shift_parent_hull_motion_rejection', 'x77': 'dtr_carla_x77_receding_metric_temporal_handoff_rejection', 'x78': 'dtr_carla_x78_nonclosing_zero_shift_permanence_release', 'x79': 'dtr_carla_x79_collision_credentialed_lateral_only_release', 'x80': 'dtr_carla_x80_cross_route_footprint_credential_release', 'x81': 'dtr_carla_x81_zero_shift_cross_route_shape_release', 'x82': 'dtr_carla_x82_held_proxy_consensus_release', 'x83': 'dtr_carla_x83_rigid_risk_reference_projection', 'x84': 'dtr_carla_x84_branch_overloaded_closing_continuation_release', 'x85': 'dtr_carla_x85_dequantization_completion_precedence_release', 'x86': 'dtr_carla_x86_receding_handback_horizon_release', 'x87': 'dtr_carla_x87_solo_completion_horizon_release', 'x88': 'dtr_carla_x88_motion_epoch_contradiction_release', 'x89': 'dtr_carla_x89_branch_overloaded_receding_release', 'x90': 'dtr_carla_x90_collision_credentialed_lateral_dominant_release', 'x91': 'dtr_carla_x91_held_risk_birth_horizon_release', 'x92': 'dtr_carla_x92_held_risk_birth_horizon_latch', 'x93': 'dtr_carla_x93_conflicted_nonclosing_future_release', 'x94': 'dtr_carla_x94_one_frame_full_dropout_continuity'}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def dependency_manifest():
    """Read-only recursive local implementation inventory; no prediction imports."""
    if _sha(CLOSURE)!=CLOSURE_SHA256:raise ValueError('C44 closure protocol drift')
    frozen=json.loads(CLOSURE.read_text(encoding='utf-8-sig'))['c44_x96_preregistration']['frozen_component_sha256']
    pending=[name+'.py' for name in MODULES.values()];files={}
    while pending:
        name=pending.pop()
        if name in files:continue
        path=HERE/name
        digest=_sha(path)
        if name in frozen and digest!=frozen[name].upper():raise ValueError('Frozen component drift: '+name)
        files[name]={'path':str(path.resolve()),'sha256':digest,'c44_direct_lock':name in frozen}
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            imports=([a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else
                     [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in imports:
                if (HERE/(module+'.py')).is_file():pending.append(module+'.py')
    return {'schema':'dtr-final-structural-dependencies-v1','closure_sha256':_sha(CLOSURE),
            'adapter_sha256':_sha(Path(__file__)), 'files':dict(sorted(files.items())),
            'historical_WORK_modules_required':False}


def _load_locked(lock):
    if lock!=dependency_manifest():raise ValueError('Structural dependency lock mismatch')
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    for alias,name in MODULES.items():
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=Path(lock['files'][name+'.py']['path']):
            raise ValueError('Import shadowing: '+name)
        globals()[alias]=module
    for name,item in lock['files'].items():
        module=sys.modules.get(name[:-3])
        if module is not None and Path(module.__file__).resolve()!=Path(item['path']):
            raise ValueError('Transitive import shadowing: '+name)


def _core_x93_and_x73(
    episode: Any,
    candidates: list[dict[str, Any]],
    calibration: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metric = x24.predict_episode(episode, candidates, calibration)
    rigid = x25.predict_episode(episode, candidates, calibration)
    core54 = x54.predict_episode(episode, candidates, calibration)
    core65 = x65.apply_ancestry_handback_episode(core54, metric)
    core67 = x67.apply_measurement_horizon_receding_release_episode(core65)
    core68 = x68.apply_object_local_lateral_dequantization_episode(core67, metric, episode)
    core69 = x69.apply_mature_cross_route_rigid_contradiction_episode(core68, rigid)
    core70 = x70.apply_triple_credential_surface_dropout_handback_episode(core69, rigid, metric)
    core71 = x71.apply_entry_cotransport_occupancy_birth_episode(core70, rigid, metric)
    core72 = x72.apply_credentialed_surface_boundary_completion_episode(core71, rigid)
    core73 = x73.apply_credentialed_parent_hull_reconstruction_episode(core72, rigid, episode)
    core74 = x74.apply_metric_handback_class_contradiction_episode(core73, rigid)
    core75 = x75.apply_collision_credentialed_object_permanence_episode(core74, rigid, metric)
    core76 = x76.apply_zero_shift_parent_hull_motion_rejection_episode(core75)
    core77 = x77.apply_receding_metric_temporal_handoff_rejection_episode(core76)
    core78 = x78.apply_nonclosing_zero_shift_permanence_release_episode(core77)
    core79 = x79.apply_collision_credentialed_lateral_only_release_episode(core78)
    core80 = x80.apply_cross_route_footprint_credential_release_episode(core79)
    core81 = x81.apply_zero_shift_cross_route_shape_release_episode(core80)
    core82 = x82.apply_held_proxy_consensus_release_episode(core81)
    core83 = x83.apply_rigid_risk_reference_projection_episode(core82)
    core84 = x84.apply_branch_overloaded_closing_continuation_release_episode(core83)
    core85 = x85.apply_dequantization_completion_precedence_release_episode(core84)
    core86 = x86.apply_receding_handback_horizon_release_episode(core85)
    core87 = x87.apply_solo_completion_horizon_release_episode(core86)
    core88 = x88.apply_motion_epoch_contradiction_release_episode(core87)
    core89 = x89.apply_branch_overloaded_receding_release_episode(core88)
    core90 = x90.apply_collision_credentialed_lateral_dominant_release_episode(core89)
    core91 = x91.apply_held_risk_birth_horizon_release_episode(core90)
    core92 = x92.apply_held_risk_birth_horizon_latch_episode(core91)
    core93 = x93.apply_conflicted_nonclosing_future_release_episode(core92)
    return core93, metric, rigid, core73


def predict_episode(episode, candidate_values, calibration, *, dependency_lock):
    """Return frozen per-frame evidence for three roster arms.

    episode and calibration must come from retained load_model_contract(). Shared
    ledger sealing/model+weight hashes are the caller's responsibility. This
    function never creates interventions or fits thresholds. Whole-episode APIs
    retain the original causal frame loop; observations must be time ordered.
    """
    _load_locked(dependency_lock)
    adapter=x24.adapter
    if type(episode) is not adapter.Episode or type(calibration) is not adapter.CameraCalibration:
        raise TypeError('Require retained sanitized Episode and CameraCalibration')
    if len(episode.observations)<2 or len(candidate_values)!=len(episode.observations):
        raise ValueError('Candidate/observation count mismatch or short episode')
    for i,(observation,value) in enumerate(zip(episode.observations,candidate_values)):
        if observation.episode_id!=episode.episode_id or (i and observation.time_s<=episode.observations[i-1].time_s):
            raise ValueError('Causal observation identity/time mismatch')
        adapter.assert_sanitized_model_value(value)
        if value.get('sample_index',observation.sample_index)!=observation.sample_index:
            raise ValueError('Candidate sample mismatch')
        if value.get('episode_id',episode.episode_id)!=episode.episode_id:
            raise ValueError('Candidate episode mismatch')
        for j,candidate in enumerate(value['candidates']):x24.validate_candidate(candidate,i,j)
    core93,metric,rigid,core73=_core_x93_and_x73(episode,copy.deepcopy(candidate_values),calibration)
    core94=x94.apply_one_frame_full_dropout_continuity_episode(core93)
    names={'X24_CORE':(metric,x24.ARM_X24),'X73_STRUCTURAL_GEOMETRY':(core73,x73.ARM_X73),
           'X94_EVIDENCE_MODEL':(core94,x94.ARM_X94)}
    outputs={}
    for name,(core,arm_id) in names.items():
        outputs[name]=[{'sample_index':f['sample_index'],'time_s':f['time_s'],
                       **copy.deepcopy(f['arms'][arm_id])} for f in core['frames']]
    return {'schema':'dtr-final-structural-predictions-v1','episode_id':episode.episode_id,
            'arms':outputs,'core_episodes':{name:copy.deepcopy(core) for name,(core,_) in names.items()},'dependency_lock':copy.deepcopy(dependency_lock),
            'diagnostics':{'X24':metric.get('diagnostics',{}),'X73':core73.get('diagnostics',{}),
                           'X94':core94.get('diagnostics',{})},'evaluator_opened':False}
