"""Fail-closed contracts for route-target evidence closure R1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "blindassist_ustrf_route_target_evidence_closure_prereg_r1"
ROLE_TRUTH_SCHEMA = "blindassist_ustrf_route_role_truth_r1"
ROLES = (
    "route_intersecting",
    "approaching_route",
    "adjacent_safe",
    "receding",
    "cleared",
)
ORACLE_IDS = (
    "O1_ORACLE_PERSON_CURRENT_ROUTE_EVENT",
    "O2_CURRENT_DETECTOR_ORACLE_ROUTE_TARGET_RELATION",
    "O3_CURRENT_EVIDENCE_ORACLE_LIFECYCLE",
)
CANDIDATE_IDS = (
    "C1_CAUSAL_ROUTE_RELATION_FSM",
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
    "C3_DUAL_KEY_CLEARANCE_FSM",
)
FROZEN_TRUE_FLAGS = (
    "association_optimization_closed",
    "detector_optimization_closed",
    "depth_closed",
    "ttc_closed",
    "route_risk_flip_closed",
    "android_shadow_closed_until_holdout_pass",
)


class ContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def resolve_bound(repo: Path, binding: Any, where: str) -> Path:
    row = require_object(binding, where)
    requested = Path(str(row.get("path", "")))
    path = requested if requested.is_absolute() else repo / requested
    if not path.is_file():
        raise ContractError(f"{where} path is missing: {path}")
    expected = row.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ContractError(f"{where} hash mismatch")
    return path


def validate_prereg(config: Any, *, repo: Path) -> Mapping[str, Any]:
    row = require_object(config, "preregistration")
    if row.get("schema") != SCHEMA:
        raise ContractError("preregistration schema mismatch")
    if row.get("phase") != "preregistered_inputs_sealed_not_materialized":
        raise ContractError("R1 must remain sealed until route-role truth and holdout are materialized")

    bindings = require_object(row.get("parent_bindings"), "parent_bindings")
    paths = {name: resolve_bound(repo, binding, f"parent_bindings.{name}") for name, binding in bindings.items()}
    required_bindings = {"result_doc", "target_attribution", "association_only", "frozen_person_truth"}
    if set(paths) != required_bindings:
        raise ContractError("parent bindings must be exact")
    seen_inputs = require_object(row.get("seen_inputs"), "seen_inputs")
    resolve_bound(repo, seen_inputs.get("windows"), "seen_inputs.windows")
    seen_sources = require_object(seen_inputs.get("sources"), "seen_inputs.sources")
    expected_seen_sources = {"lilocbench_dynamics_0_front", "lilocbench_lt_changes_dynamics_0_front"}
    if set(seen_sources) != expected_seen_sources:
        raise ContractError("seen input source inventory drifted")
    for source_id, source in seen_sources.items():
        source_row = require_object(source, f"seen_inputs.sources.{source_id}")
        if set(source_row) != {"route", "frames", "bundle"}:
            raise ContractError(f"seen input bindings for {source_id} must be exact")
        resolve_bound(repo, source_row["route"], f"seen_inputs.sources.{source_id}.route")
        resolve_bound(repo, source_row["frames"], f"seen_inputs.sources.{source_id}.frames")
        resolve_bound(repo, source_row["bundle"], f"seen_inputs.sources.{source_id}.bundle")

    proposal = require_object(row.get("seen_truth_proposal_protocol"), "seen_truth_proposal_protocol")
    review_bundle_path = resolve_bound(repo, proposal.get("review_bundle"), "seen_truth_proposal_protocol.review_bundle")
    review_bundle = require_object(load_json(review_bundle_path), "seen route-role review bundle")
    if review_bundle.get("candidate_alerts_exposed") is not False or review_bundle.get("detector_outputs_exposed") is not False:
        raise ContractError("seen review bundle exposed candidate or detector output")
    if review_bundle.get("window_labels_exposed") is not False or review_bundle.get("future_truth_anchors_exposed") is not False:
        raise ContractError("seen review bundle exposed scoring labels or future anchors")
    for source in review_bundle.get("sources", []):
        for window in source.get("windows", []):
            if set(window) != {"blind_window_id", "frames"}:
                raise ContractError("reviewer-facing window contains non-blind metadata")
            for frame in window["frames"]:
                for seed in frame.get("person_seed_boxes", []):
                    bbox = seed.get("bbox_xyxy")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ContractError("review bundle contains malformed person seed bbox")
    scorer_path = resolve_bound(repo, proposal.get("isolated_scorer_binding"), "seen_truth_proposal_protocol.isolated_scorer_binding")
    scorer = require_object(load_json(scorer_path), "isolated scorer binding")
    if scorer.get("review_bundle_sha256") != proposal["review_bundle"]["sha256"]:
        raise ContractError("isolated scorer binding points to another review bundle")
    pass_a_path = resolve_bound(repo, proposal.get("pass_a"), "seen_truth_proposal_protocol.pass_a")
    pass_a = require_object(load_json(pass_a_path), "proposal pass A")
    if pass_a.get("baseline_detector_outputs_accessed") is not False or proposal["pass_a"].get("candidate_or_baseline_detector_output") is not False:
        raise ContractError("proposal pass A is not independent from the baseline detector")
    if pass_a.get("annotation_model_sha256") != proposal["pass_a"].get("model_sha256"):
        raise ContractError("proposal pass A model binding mismatch")
    pass_b = require_object(proposal.get("pass_b"), "seen_truth_proposal_protocol.pass_b")
    model_path = Path(str(pass_b.get("model_path", "")))
    if not model_path.is_file() or sha256_file(model_path) != pass_b.get("model_sha256"):
        raise ContractError("proposal pass B model binding mismatch")
    if (pass_b.get("imgsz"), pass_b.get("confidence"), pass_b.get("class_id")) != (960, 0.01, 0):
        raise ContractError("proposal pass B inference contract drifted")
    if pass_b.get("candidate_or_baseline_detector_output") is not False:
        raise ContractError("proposal pass B cannot be a candidate or baseline detector output")
    pass_b_output = resolve_bound(repo, {"path": pass_b.get("output_path"), "sha256": pass_b.get("output_sha256")}, "seen_truth_proposal_protocol.pass_b.output")
    pass_b_payload = require_object(load_json(pass_b_output), "proposal pass B output")
    if pass_b_payload.get("model_sha256") != pass_b.get("model_sha256") or pass_b_payload.get("frame_count") != 4594:
        raise ContractError("proposal pass B output binding mismatch")
    if pass_b_payload.get("candidate_alerts_exposed") is not False or pass_b_payload.get("baseline_app_detector_outputs_exposed") is not False:
        raise ContractError("proposal pass B output exposed forbidden evidence")
    negative_audit_path = resolve_bound(repo, proposal.get("negative_additional_audit"), "seen_truth_proposal_protocol.negative_additional_audit")
    negative_audit = require_object(load_json(negative_audit_path), "negative additional audit")
    if negative_audit.get("model_sha256") != proposal["negative_additional_audit"].get("model_sha256"):
        raise ContractError("negative additional audit model binding mismatch")
    fusion = require_object(proposal.get("fusion"), "seen_truth_proposal_protocol.fusion")
    expected_proposal_tracking = {
        "dual_proposal_match_iou_min": 0.30,
        "single_pass_non_seed": "quarantine_until_temporal_or_visual_adjudication",
    }
    if any(fusion.get(key) != expected for key, expected in expected_proposal_tracking.items()):
        raise ContractError("person proposal fusion contract drifted")
    tracker = require_object(fusion.get("proposal_identity_tracker"), "seen_truth_proposal_protocol.fusion.proposal_identity_tracker")
    expected_tracker = {
        "role": "annotation_only_ultralytics_bytetrack_defaults_not_candidate_tracker",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "reset_on_unobserved_frame_gap": True,
    }
    if dict(tracker) != expected_tracker:
        raise ContractError("annotation ByteTrack contract drifted")
    if fusion.get("candidate_alerts_and_app_detector_hidden") is not True:
        raise ContractError("person truth creation must hide candidate and App detector outputs")
    fusion_output = resolve_bound(
        repo,
        fusion.get("materialized_output"),
        "seen_truth_proposal_protocol.fusion.materialized_output",
    )
    fusion_payload = require_object(load_json(fusion_output), "person proposal fusion output")
    materialized = require_object(fusion.get("materialized_output"), "fusion materialized output")
    if fusion_payload.get("config_sha256") != materialized.get("generated_from_config_sha256"):
        raise ContractError("person proposal fusion config binding mismatch")
    for key in ("frame_count", "tracklet_count", "consensus_tracklet_count", "adjudication_tracklet_count"):
        if fusion_payload.get(key) != materialized.get(key):
            raise ContractError(f"person proposal fusion {key} binding mismatch")
    if fusion_payload.get("candidate_alerts_exposed") is not False or fusion_payload.get("baseline_app_detector_outputs_exposed") is not False:
        raise ContractError("person proposal fusion exposed forbidden evidence")
    adjudicator = require_object(fusion.get("third_model_adjudicator"), "fusion third model adjudicator")
    adjudicator_model = Path(str(adjudicator.get("model_path", "")))
    if not adjudicator_model.is_file() or sha256_file(adjudicator_model) != adjudicator.get("model_sha256"):
        raise ContractError("third model adjudicator binding mismatch")
    expected_adjudicator = {
        "role": "disagreement_only_closed_vocabulary_person_adjudicator_not_candidate_detector",
        "imgsz": 960,
        "confidence": 0.01,
        "class_id": 0,
        "scope": "only_frames_referenced_by_third_model_adjudication_required_tracklets",
        "candidate_or_baseline_detector_output": False,
        "unresolved_after_third_model": "quarantine_person_episode",
    }
    if any(adjudicator.get(key) != value for key, value in expected_adjudicator.items()):
        raise ContractError("third model adjudicator contract drifted")
    adjudication_bundle = repo / str(adjudicator.get("bundle_path", ""))
    if not adjudication_bundle.is_file() or sha256_file(adjudication_bundle) != adjudicator.get("bundle_sha256"):
        raise ContractError("third model adjudication bundle binding mismatch")
    adjudication_output = resolve_bound(
        repo,
        {"path": adjudicator.get("output_path"), "sha256": adjudicator.get("output_sha256")},
        "third model adjudicator output",
    )
    adjudication_payload = require_object(load_json(adjudication_output), "third model adjudicator output")
    if adjudication_payload.get("config_sha256") != adjudicator.get("generated_from_config_sha256"):
        raise ContractError("third model adjudicator config binding mismatch")
    if adjudication_payload.get("frame_count") != adjudicator.get("frame_count") or adjudication_payload.get("frame_count") != 1855:
        raise ContractError("third model adjudicator frame count mismatch")
    if adjudication_payload.get("model_sha256") != adjudicator.get("model_sha256"):
        raise ContractError("third model adjudicator output model mismatch")
    expected_resolution = {
        "node_match_iou_min": 0.30,
        "single_model_node_requires_third_match": True,
        "ambiguous_node_requires_third_match": True,
        "third_identity_crossing_fusion_tracklets": "quarantine_both",
        "third_identity_mapping_multiple_same_model_lineages": "quarantine_episode",
        "frozen_seed_identity_conflict": "quarantine_episode",
        "single_frozen_seed_identity_has_precedence_over_extension_disagreement": True,
    }
    if dict(require_object(adjudicator.get("resolution_contract"), "third model resolution contract")) != expected_resolution:
        raise ContractError("third model resolution contract drifted")
    identity_output_path = resolve_bound(
        repo,
        fusion.get("identity_adjudication_output"),
        "person identity adjudication output",
    )
    identity_payload = require_object(load_json(identity_output_path), "person identity adjudication output")
    identity_binding = require_object(fusion.get("identity_adjudication_output"), "person identity adjudication binding")
    for key in ("tracklet_count", "accepted_tracklet_count", "quarantined_tracklet_count"):
        if identity_payload.get(key) != identity_binding.get(key):
            raise ContractError(f"person identity adjudication {key} binding mismatch")

    attribution = require_object(load_json(paths["target_attribution"]), "target attribution")
    if attribution.get("decision") != "STOP_DETECTOR_CHANGES_AND_REOPEN_T0_T3":
        raise ContractError("parent detector hard gate is not the frozen pass state")
    association = require_object(load_json(paths["association_only"]), "association result")
    if association.get("shadow_entry_gate_passed") is not False or association.get("decision") != "STOP_BEFORE_ANDROID_SHADOW":
        raise ContractError("parent association result must fail shadow entry")
    if association.get("detector_fixed") is not True or association.get("route_and_event_kernel_fixed") is not True:
        raise ContractError("parent fixed axes are not proven")
    if association.get("ttc_or_depth_used") is not False:
        raise ContractError("parent unexpectedly used TTC or depth")

    frozen = require_object(row.get("frozen_axes"), "frozen_axes")
    for name in FROZEN_TRUE_FLAGS:
        if frozen.get(name) is not True:
            raise ContractError(f"frozen_axes.{name} must be true")
    if frozen.get("person_confidence_threshold") != 0.35 or frozen.get("classwise_nms_iou_threshold") != 0.45:
        raise ContractError("detector thresholds drifted")
    expected_kernel = {
        "route_point_margin_fraction": 0.08,
        "min_alert_frames": 2,
        "min_clear_frames": 3,
        "target_match_iou": 0.30,
        "frame_rate_hz": 15.0,
    }
    for name, expected in expected_kernel.items():
        if frozen.get(name) != expected:
            raise ContractError(f"frozen_axes.{name} drifted")

    truth = require_object(row.get("route_role_truth"), "route_role_truth")
    if tuple(truth.get("roles", ())) != ROLES or truth.get("route_invalid_behavior") != "abstain_not_a_role":
        raise ContractError("route-role vocabulary must be the frozen five-state contract")
    transitions = require_object(truth.get("allowed_transitions"), "allowed_transitions")
    if set(transitions) != set(ROLES) or any(
        not isinstance(values, list) or not values or any(value not in ROLES for value in values)
        for values in transitions.values()
    ):
        raise ContractError("route-role transition grammar is invalid")
    creation = require_object(truth.get("truth_creation"), "truth_creation")
    for name in ("candidate_alert_hidden", "detector_output_hidden", "two_independent_model_passes", "third_model_adjudication_on_disagreement"):
        if creation.get(name) is not True:
            raise ContractError(f"truth_creation.{name} must be true")
    geometry = require_object(truth.get("model_proxy_geometry_contract"), "route role model proxy geometry contract")
    if geometry.get("route_evidence") != "causal_route_predictions_only_route_truth_and_future_frame_forbidden":
        raise ContractError("route role truth must use causal route predictions only")
    if geometry.get("registered_rgbd_depth") != "annotation_truth_support_only_H2_candidate_remains_closed":
        raise ContractError("route role depth authority drifted")
    width = require_object(geometry.get("corridor_width_source"), "route role corridor width source")
    width_path = resolve_bound(repo, width, "route role corridor width source")
    width_payload = require_object(load_json(width_path), "route role corridor width source")
    if width_payload.get("assumed_geometry", {}).get("route_width_m") != 0.9 or width.get("route_width_m") != 0.9 or width.get("half_width_m") != 0.45:
        raise ContractError("route role corridor width contract drifted")
    expected_person_support = {
        "bbox_bottom_fraction": 0.20,
        "bbox_center_width_fraction": 0.50,
        "minimum_valid_depth_pixels": 5,
        "aggregation": "median_registered_depth_backprojected_at_bbox_bottom_center",
    }
    if dict(require_object(geometry.get("person_ground_support"), "person ground support")) != expected_person_support:
        raise ContractError("person ground support contract drifted")
    expected_route_support = {
        "route_uv_radius_px": 2,
        "minimum_valid_depth_pixels": 5,
        "aggregation": "median_registered_depth_backprojected_at_causal_route_uv",
    }
    if dict(require_object(geometry.get("route_ground_support"), "route ground support")) != expected_route_support:
        raise ContractError("route ground support contract drifted")
    expected_trend = {
        "causal_observation_count": 3,
        "approaching_delta_m_min": 0.10,
        "receding_delta_m_min": 0.10,
        "uses_future_frame": False,
    }
    if dict(require_object(geometry.get("trend_contract"), "route role trend contract")) != expected_trend:
        raise ContractError("route role trend contract drifted")
    if geometry.get("missing_occluded_or_invalid_depth_can_clear") is not False:
        raise ContractError("invalid evidence cannot clear a person episode")
    role_output_path = resolve_bound(repo, geometry.get("materialized_output"), "route role model proxy truth output")
    role_payload = validate_role_truth(load_json(role_output_path), config=row)
    role_binding = require_object(geometry.get("materialized_output"), "route role model proxy truth binding")
    if role_payload.get("config_sha256") != role_binding.get("generated_from_config_sha256"):
        raise ContractError("route role model proxy config binding mismatch")
    episode_count = sum(len(source["person_episodes"]) for source in role_payload["sources"])
    if episode_count != role_binding.get("person_episode_count"):
        raise ContractError("route role model proxy episode count mismatch")
    if role_payload.get("quarantined_identity_episode_count") != role_binding.get("quarantined_identity_episode_count"):
        raise ContractError("route role model proxy quarantine count mismatch")
    if role_payload.get("positive_target_coverage") != {
        "window_count": 15,
        "accepted_target_tracklet_count": 15,
        "quarantined_target_tracklet_count": 0,
    }:
        raise ContractError("route role model proxy target coverage is incomplete")

    seen = require_object(row.get("seen_diagnostic"), "seen_diagnostic")
    if seen.get("candidate_selection_authority") is not False or seen.get("scalar_tuning_authority") is not False:
        raise ContractError("seen diagnostic windows cannot select or tune candidates")
    if (seen.get("positive_windows"), seen.get("negative_windows")) != (15, 15):
        raise ContractError("seen diagnostic inventory drifted")
    oracle_output_path = resolve_bound(repo, seen.get("oracle_materialized_output"), "seen oracle attribution output")
    oracle_payload = require_object(load_json(oracle_output_path), "seen oracle attribution output")
    oracle_binding = require_object(seen.get("oracle_materialized_output"), "seen oracle attribution binding")
    if oracle_payload.get("config_sha256") != oracle_binding.get("generated_from_config_sha256"):
        raise ContractError("seen oracle attribution config binding mismatch")
    expected_oracle_results = {
        "T0_CURRENT", "O1_ORACLE_PERSON", "O2_ORACLE_ROUTE_RELATION", "O3_ORACLE_LIFECYCLE"
    }
    if set(oracle_payload.get("results", {})) != expected_oracle_results:
        raise ContractError("seen oracle attribution arm inventory drifted")
    if oracle_payload["results"]["O1_ORACLE_PERSON"].get("status") != "not_evaluable_all_person_truth_incomplete":
        raise ContractError("O1 must fail closed while person truth is quarantined")

    oracle_ids = tuple(item.get("id") for item in row.get("oracle_arms", ()) if isinstance(item, dict))
    if oracle_ids != ORACLE_IDS:
        raise ContractError("oracle arms must be the frozen three-arm attribution")
    oracle_by_id = {item["id"]: item for item in row["oracle_arms"]}
    if oracle_by_id[ORACLE_IDS[0]].get("negative_false_alert_authority_requires_all_person_stable_ids") is not True:
        raise ContractError("oracle-person must preserve all cooccurring persons")
    if "without_cross_frame_truth_id" not in oracle_by_id[ORACLE_IDS[1]].get("route_target_relation", ""):
        raise ContractError("oracle-relation cannot import cross-frame truth identity")
    if oracle_by_id[ORACLE_IDS[2]].get("truth_cannot_create_alert_without_current_evidence") is not True:
        raise ContractError("oracle-lifecycle cannot create alerts from truth")
    candidates = row.get("candidate_roster")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ContractError("candidate roster must contain one to three candidates")
    if tuple(item.get("id") for item in candidates if isinstance(item, dict)) != CANDIDATE_IDS:
        raise ContractError("candidate roster or order drifted")
    if any("scalar_grid_search" not in item.get("forbidden", ()) for item in candidates):
        raise ContractError("every candidate must forbid scalar grid search")
    candidate_binding = require_object(row.get("candidate_implementation_binding"), "candidate implementation binding")
    resolve_bound(repo, candidate_binding, "candidate implementation binding")
    if candidate_binding.get("domain") != "ustrf-route-target-evidence-closure" or candidate_binding.get("tool") != "candidates.py":
        raise ContractError("candidate stable adapter route drifted")
    if sha256_file(Path(__file__).with_name("candidates.py")) != candidate_binding.get("implementation_sha256"):
        raise ContractError("candidate implementation hash drifted behind stable adapter")
    if candidate_binding.get("frozen_before_holdout_inventory_decode") is not True or candidate_binding.get("detector_tracker_depth_ttc_or_route_risk_flip_changed") is not False:
        raise ContractError("candidate implementation authority drifted")

    holdout = require_object(row.get("sealed_holdout"), "sealed_holdout")
    expected_holdout = {
        "state": "inventory_unopened_not_materialized",
        "source_count": 2,
        "fresh_from_seen_diagnostic": True,
        "source_group_disjoint": True,
        "parent_sequence_disjoint": True,
        "person_route_trace_disjoint": True,
        "near_duplicate_frames_forbidden": True,
        "candidate_roster_hash_frozen_before_discovery_or_decode": True,
        "selection_runs_per_candidate": 1,
        "no_candidate_specific_replay": True,
        "pooled_result_cannot_override_source_failure": True,
    }
    for key, expected in expected_holdout.items():
        if holdout.get(key) != expected:
            raise ContractError(f"sealed_holdout.{key} must equal {expected!r}")
    inventory_path = resolve_bound(repo, holdout.get("inventory_receipt"), "sealed holdout inventory receipt")
    inventory = require_object(load_json(inventory_path), "sealed holdout inventory receipt")
    if inventory.get("candidate_implementation_sha256") != candidate_binding.get("implementation_sha256"):
        raise ContractError("holdout inventory predates another candidate implementation")
    if inventory.get("sequence_content_decoded") is not False or inventory.get("admitted_source_count") != 0:
        raise ContractError("holdout inventory must remain unopened until two valid sources are admitted")
    preadmission_binding = holdout.get("source_preadmission_receipt")
    if preadmission_binding is not None:
        preadmission_path = resolve_bound(repo, preadmission_binding, "sealed holdout source preadmission receipt")
        preadmission = require_object(load_json(preadmission_path), "sealed holdout source preadmission receipt")
        if preadmission.get("candidate_implementation_sha256") != candidate_binding.get("implementation_sha256"):
            raise ContractError("holdout source preadmission predates another candidate implementation")
        if preadmission.get("candidate_outputs_executed_on_sources") is not False:
            raise ContractError("holdout source preadmission cannot inspect candidate outputs")
        if preadmission.get("sequence_content_decoded") is not False:
            raise ContractError("holdout source preadmission must precede sequence content decode")
        if preadmission.get("metadata_preadmitted_source_count") != 2 or preadmission.get("admitted_source_count") != 0:
            raise ContractError("holdout source preadmission must name two metadata-only sources")
    qualification_binding = holdout.get("content_qualification_receipt")
    if qualification_binding is not None:
        qualification_path = resolve_bound(repo, qualification_binding, "sealed holdout content qualification receipt")
        qualification = require_object(load_json(qualification_path), "sealed holdout content qualification receipt")
        if qualification.get("candidate_implementation_sha256") != candidate_binding.get("implementation_sha256"):
            raise ContractError("holdout content qualification predates another candidate implementation")
        if qualification.get("candidate_outputs_executed_on_any_screened_source") is not False:
            raise ContractError("holdout content qualification cannot inspect candidate outputs")
        if qualification.get("rgb_or_depth_content_decoded") is not False:
            raise ContractError("holdout capacity qualification must precede RGB-D decode")
        if qualification.get("capacity_qualified_source_count") != 2 or qualification.get("admitted_source_count") != 0:
            raise ContractError("holdout capacity qualification must name two not-yet-admitted sources")
        if len(qualification.get("selected_source_ids", ())) != 2:
            raise ContractError("holdout capacity qualification selected source count drifted")
    modality_binding = holdout.get("modality_probe_receipt")
    if modality_binding is not None:
        modality_path = resolve_bound(repo, modality_binding, "sealed holdout modality probe receipt")
        modality = require_object(load_json(modality_path), "sealed holdout modality probe receipt")
        if modality.get("candidate_outputs_executed") is not False:
            raise ContractError("holdout modality probe cannot inspect candidate outputs")
        if modality.get("rgb_contract", {}).get("suitable_for_frozen_app_detector_input") is not True:
            raise ContractError("holdout modality probe lacks frozen App detector RGB")
        if modality.get("aligned_depth_contract", {}).get("candidate_H2_authority") is not False:
            raise ContractError("holdout modality probe cannot open candidate H2")
        if modality.get("privacy_and_use_boundary", {}).get("production_authority") is not False:
            raise ContractError("holdout modality probe cannot grant production authority")
    route_input = require_object(holdout.get("causal_route_input_contract"), "sealed holdout causal route input")
    inherited_route_path = resolve_bound(
        repo,
        route_input.get("inherited_from"),
        "sealed holdout causal route input inherited contract",
    )
    inherited_route = require_object(load_json(inherited_route_path), "inherited sensor replay preregistration").get("route", {})
    expected_route_input = {
        "policy": "past_pose_prefix_only_no_future_ground_truth_v1",
        "pose_input": "published_synchronized_qolo_pose_current_and_past_samples_only",
        "camera_extrinsic_input": "raw_bag_tf_graph_bound_per_sequence",
        "rgb_pose_join": "latest_pose_timestamp_not_after_rgb_timestamp",
        "maximum_pose_age_ms": 200.0,
        "causal_history_frames": 12,
        "projection_horizon_frames": 24,
        "nominal_frame_rate_hz": 15.0,
        "minimum_forward_displacement_m": 0.03,
        "future_pose_access_for_candidate": False,
        "actual_future_polyline_authority": "annotation_route_truth_and_capacity_only_never_candidate_input",
        "actual_future_route_truth_representation": "all_front_projectable_pose_samples_as_uv_polyline_not_endpoint_only",
        "missing_stale_or_unprojectable_route": "unknown_abstain_and_cannot_clear",
        "candidate_output_used_to_choose_or_modify_route": False,
        "new_scalar_tuned": False,
    }
    if {key: route_input.get(key) for key in expected_route_input} != expected_route_input:
        raise ContractError("sealed holdout causal route input contract drifted")
    inherited_expected = {
        "causal_history_frames": inherited_route.get("causal_history_frames"),
        "projection_horizon_frames": inherited_route.get("truth_horizon_frames"),
        "minimum_forward_displacement_m": inherited_route.get("minimum_forward_displacement_m"),
    }
    if any(route_input.get(key) != value for key, value in inherited_expected.items()):
        raise ContractError("sealed holdout causal route inputs do not inherit frozen R3 values")
    truth_freeze = require_object(holdout.get("holdout_truth_freeze_contract"), "sealed holdout truth freeze")
    if truth_freeze.get("authority") != "model_proxy_benchmark_truth_not_human_truth_not_production":
        raise ContractError("holdout truth authority drifted")
    if truth_freeze.get("candidate_and_app_detector_event_outputs_hidden_until_truth_and_windows_hash_frozen") is not True:
        raise ContractError("holdout truth must remain blind to candidate and App detector/event outputs")
    presence = require_object(truth_freeze.get("all_person_presence"), "holdout all-person presence")
    visual_passes = presence.get("visual_passes")
    if not isinstance(visual_passes, list) or [item.get("id") for item in visual_passes] != [
        "HOLDOUT_PERSON_PASS_B_YOLOV8N",
        "HOLDOUT_PERSON_PASS_C_YOLO11X",
    ]:
        raise ContractError("holdout visual person pass roster drifted")
    for visual_pass in visual_passes:
        model_path = Path(str(visual_pass.get("model_path", "")))
        if not model_path.is_file() or sha256_file(model_path) != visual_pass.get("model_sha256"):
            raise ContractError(f"holdout visual person model binding mismatch: {visual_pass.get('id')}")
        if (
            visual_pass.get("imgsz"),
            visual_pass.get("confidence"),
            visual_pass.get("iou"),
            visual_pass.get("class_id"),
        ) != (960, 0.01, 0.7, 0):
            raise ContractError("holdout visual person inference contract drifted")
    if presence.get("visual_consensus_iou_min") != row["frozen_axes"]["target_match_iou"]:
        raise ContractError("holdout visual consensus threshold does not inherit frozen IoU")
    if presence.get("proposal_acceptance_confidence") != row["frozen_axes"]["person_confidence_threshold"]:
        raise ContractError("holdout visual proposal acceptance does not inherit frozen person confidence")
    if presence.get("full_frame_discovery_required") is not True or presence.get("published_lidar_tracks_can_declare_visual_person_absent") is not False:
        raise ContractError("holdout person truth can hide untracked visible people")
    if presence.get("accepted_presence_paths") != "two_visual_pass_consensus_or_one_visual_pass_plus_role_consistent_projected_track_group_or_one_visual_pass_plus_two_consecutive_annotation_bytetrack_frames":
        raise ContractError("holdout person truth lacks two-signal presence support")
    if presence.get("visual_identity_tracker") != "inherit_seen_truth_proposal_protocol.fusion.proposal_identity_tracker_annotation_only":
        raise ContractError("holdout visual identity tracker does not inherit annotation-only tracker")
    if presence.get("minimum_temporal_presence_support_frames") != row["frozen_axes"]["min_alert_frames"]:
        raise ContractError("holdout temporal presence support does not inherit min-alert frames")
    identity = require_object(truth_freeze.get("stable_identity_and_metric_role"), "holdout identity and role")
    if identity.get("rgb_track_join") != "latest_track_timestamp_not_after_rgb_timestamp_age_at_most_200ms":
        raise ContractError("holdout track join is not causal or age-bounded")
    if identity.get("future_robot_pose_visibility_to_visual_proposals") is not False:
        raise ContractError("holdout visual proposals can access future route truth")
    role_truth = require_object(truth_freeze.get("role_truth"), "holdout role truth")
    if role_truth.get("visual_only_route_intersecting") != "stable_visual_person_box_intersects_margin_expanded_actual_future_uv_polyline_not_endpoint_only":
        raise ContractError("holdout visual route truth regressed to endpoint-only geometry")
    if role_truth.get("approaching_route") != "outside_now_three_consecutive_valid_distance_reduction_at_least_0.10m_and_actual_future_track_intersects_actual_future_route_within_1.6s_annotation_only":
        raise ContractError("holdout approaching-route truth lost future route-entry evidence")
    if role_truth.get("receding") != "after_prior_route_intersection_three_consecutive_valid_observations_distance_increase_at_least_0.10m":
        raise ContractError("holdout receding truth is not anchored to a prior intersection")
    if role_truth.get("missing_occluded_stale_or_unknown_can_clear") is not False:
        raise ContractError("holdout missing evidence cannot clear")
    event_truth = require_object(truth_freeze.get("event_truth"), "holdout event truth")
    if event_truth.get("single_event_per_person_contiguous_active_episode") is not True or event_truth.get("new_event_requires_prior_terminal_clear") is not True:
        raise ContractError("holdout lifecycle truth can regenerate before terminal clear")
    window_freeze = require_object(truth_freeze.get("window_freeze"), "holdout window freeze")
    if window_freeze.get("minimum_matched_negative_windows_each_source") != holdout.get("minimum_matched_negative_windows_each_source"):
        raise ContractError("holdout negative window minimum drifted")
    if window_freeze.get("candidate_specific_window_replay") is not False:
        raise ContractError("holdout windows permit candidate-specific replay")
    if truth_freeze.get("candidate_output_used_to_resolve_truth_or_quarantine") is not False:
        raise ContractError("holdout candidate output can influence truth")
    if holdout.get("minimum_positive_events_each_source", 0) < 10 or holdout.get("minimum_matched_negative_windows_each_source", 0) < 10:
        raise ContractError("sealed holdout is below the preregistered per-source minimum")
    if holdout.get("minimum_critical_events_each_source", 0) < 2 or holdout.get("minimum_scorable_negative_exposure_minutes_each_source", 0.0) < 10.0:
        raise ContractError("sealed holdout lacks critical-event or false-alert exposure power")
    if holdout.get("cooccurring_person_required_each_source") is not True:
        raise ContractError("each holdout source must exercise person cooccurrence")

    gate = require_object(row.get("selection_gate_each_source"), "selection_gate_each_source")
    expected_gate = {
        "event_recall_min": 0.9,
        "critical_miss_max": 0,
        "false_alerts_per_minute_max": 0.5,
        "clearance_rate_min": 0.9,
        "clearance_p95_ms_max": 1500.0,
        "repeat_alert_count_max": 0,
        "event_regeneration_count_max": 0,
        "evidence_age_p95_ms_max": 200.0,
        "active_alert_on_unknown_route_frames_max": 0,
    }
    if dict(gate) != expected_gate:
        raise ContractError("per-source selection gate drifted")
    execution = require_object(row.get("candidate_execution_contract"), "candidate execution contract")
    if execution.get("truth_and_windows_hash_frozen_before_app_detector_or_candidate_run") is not True:
        raise ContractError("candidate execution can observe outputs before truth freeze")
    if execution.get("full_sequence_single_pass_no_window_reset") is not True:
        raise ContractError("candidate execution is not a one-shot full-sequence replay")
    if execution.get("detector_input_authority") != "exact_android_canvas_canonical_raw_export_not_host_pil_reconstruction":
        raise ContractError("candidate execution detector input is not exact Android Canvas evidence")
    device_exporter = require_object(execution.get("device_exporter"), "candidate execution device exporter")
    exporter_path = repo / str(device_exporter.get("path", ""))
    if not exporter_path.is_file() or sha256_file(exporter_path) != device_exporter.get("sha256"):
        raise ContractError("candidate execution Android exporter binding mismatch")
    if (
        device_exporter.get("runtime") != "android_imagepreprocessor_canvas_tflite_cpu_4_threads"
        or device_exporter.get("expected_frame_count_argument") != "ustrfDetectorTaxonomyExpectedFrameCount"
    ):
        raise ContractError("candidate execution Android exporter contract drifted")
    detector = require_object(execution.get("app_detector"), "candidate execution app detector")
    detector_bindings = (("model_path", "model_sha256"), ("labels_path", "labels_sha256"))
    for path_key, hash_key in detector_bindings:
        path = repo / str(detector.get(path_key, ""))
        if not path.is_file() or sha256_file(path) != detector.get(hash_key):
            raise ContractError(f"candidate execution detector binding mismatch: {path_key}")
    if (
        detector.get("input_shape"),
        detector.get("output_shape"),
        detector.get("person_class_index"),
        detector.get("confidence_threshold"),
        detector.get("nms_iou_threshold"),
    ) != ([1, 320, 320, 3], [1, 84, 2100], 0, frozen["person_confidence_threshold"], frozen["classwise_nms_iou_threshold"]):
        raise ContractError("candidate execution App detector axes drifted")
    association = require_object(execution.get("association"), "candidate execution association")
    association_path = repo / str(association.get("config_path", ""))
    if association.get("arm") != "T0" or association.get("optimization_closed") is not True:
        raise ContractError("candidate execution association is not frozen T0")
    if not association_path.is_file() or sha256_file(association_path) != association.get("config_sha256"):
        raise ContractError("candidate execution T0 binding mismatch")
    if execution.get("candidate_outputs_executed_before_truth_freeze") is not False:
        raise ContractError("candidate outputs were exposed before truth freeze")
    if row.get("winner_rule", {}).get("on_no_eligible_candidate") != "STOP_NO_ANDROID_SHADOW_KEEP_H2_CLOSED":
        raise ContractError("failure must keep Android shadow and H2 closed")
    return row


def validate_role_truth(payload: Any, *, config: Mapping[str, Any]) -> Mapping[str, Any]:
    row = require_object(payload, "route-role truth")
    if row.get("schema") != ROLE_TRUTH_SCHEMA or row.get("split") != "seen_diagnostic_only":
        raise ContractError("route-role truth schema or split mismatch")
    if tuple(row.get("roles", ())) != ROLES:
        raise ContractError("route-role truth vocabulary drifted")
    sources = row.get("sources")
    if not isinstance(sources, list) or {source.get("source_id") for source in sources if isinstance(source, dict)} != set(config["seen_diagnostic"]["source_ids"]):
        raise ContractError("route-role truth must cover both seen diagnostic sources")
    allowed = config["route_role_truth"]["allowed_transitions"]
    for source in sources:
        persons = source.get("person_episodes")
        if not isinstance(persons, list):
            raise ContractError("source person_episodes must be an array")
        person_ids: set[str] = set()
        for person in persons:
            person_id = person.get("person_id")
            if not isinstance(person_id, str) or not person_id or person_id in person_ids:
                raise ContractError("person_id must be stable and unique within source")
            person_ids.add(person_id)
            if not isinstance(person.get("risk_event_id"), str) or not person["risk_event_id"]:
                raise ContractError("person episode requires a person-bound risk_event_id")
            event_truth = person.get("event_truth")
            if not isinstance(event_truth, dict) or any(key not in event_truth for key in (
                "first_visible_frame", "alertable_start_frame", "clear_frame", "should_alert", "critical"
            )):
                raise ContractError("person episode lacks lifecycle truth")
            frames = person.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ContractError("person episode must contain frames")
            previous = None
            for frame in frames:
                if not isinstance(frame.get("frame_id"), str) or not frame["frame_id"]:
                    raise ContractError("role frame lacks frame_id")
                if frame.get("visibility") not in ("visible", "occluded", "not_visible_cleared"):
                    raise ContractError("role frame visibility is invalid")
                if frame.get("visibility") == "visible":
                    bbox = frame.get("bbox_xyxy")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ContractError("visible role frame requires bbox_xyxy")
                if not isinstance(frame.get("source_capture_timestamp_ns"), int):
                    raise ContractError("role frame lacks source capture timestamp")
                if frame.get("route_status") != "known":
                    if frame.get("role") is not None:
                        raise ContractError("unknown route must abstain instead of assigning a role")
                    previous = None
                    continue
                if not isinstance(frame.get("route_receipt_id"), str) or not frame["route_receipt_id"]:
                    raise ContractError("known route frame lacks route receipt identity")
                if not isinstance(frame.get("route_evidence_age_ms"), (int, float)) or frame["route_evidence_age_ms"] < 0:
                    raise ContractError("known route frame lacks non-negative evidence age")
                if frame.get("geometry_status", "known") != "known":
                    if frame.get("role") is not None:
                        raise ContractError("unknown geometry must abstain instead of assigning a role")
                    previous = None
                    continue
                role = frame.get("role")
                if role not in ROLES:
                    raise ContractError("known route frame has invalid role")
                if previous is not None and role not in allowed[previous]:
                    raise ContractError(f"illegal role transition: {previous} -> {role}")
                previous = role
    return row
