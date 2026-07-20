#!/usr/bin/env python3
"""Build a deterministic JSON + HTML USTRF-SC research benchmark report.

This report consumes already-produced receipts.  A passing analytic or source-native data gate
never authorizes device or user-facing safety behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(name: str, passed: bool, detail: str, authority: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, "authority": authority}


def build(
    geometry_audit: Path,
    dynamic_audit: Path,
    temporal_audits: list[Path],
    vkitti_track_audit: Path | None = None,
    argoverse_ttc_audit: Path | None = None,
    carla_rgbd_audit: Path | None = None,
    corridor_safety_audit: Path | None = None,
    corridor_safety_replay: Path | None = None,
    bonn_rgbd_audit: Path | None = None,
    bonn_reprojection_audit: Path | None = None,
    revel_rgb_labels_audit: Path | None = None,
    revel_vicon_trajectory_audit: Path | None = None,
    revel_rgb_vicon_reprojection_audit: Path | None = None,
    revel_detector_benchmark: Path | None = None,
    revel_detector_guard: Path | None = None,
    revel_detector_vicon_alignment: Path | None = None,
) -> dict[str, Any]:
    geometry = _read(geometry_audit)
    dynamic = _read(dynamic_audit)
    temporal = [_read(path) for path in temporal_audits]
    geometry_passed = (
        geometry.get("ok") is True
        and geometry.get("static_reprojection_rmse_meters") == 0.0
        and geometry.get("visibility_gap_false_drop_count") == 0
        and geometry.get("drop_expected_sample_count") == geometry.get("drop_detected_sample_count")
    )
    dynamic_passed = (
        dynamic.get("max_velocity_error_mps") == 0.0
        and dynamic.get("max_ttc_error_ms", float("inf")) <= 1.0
        and dynamic.get("collision_label_accuracy") == 1.0
        and dynamic.get("rejected_sequence_count") == 2
    )
    temporal_passed = all(
        report.get("aggregate", {}).get("median_valid_projection_fraction", 0.0) >= .95
        and report.get("aggregate", {}).get("median_pair_median_abs_depth_residual_m", float("inf")) <= .05
        and report.get("ustrf_geometry_input_admitted") is False
        for report in temporal
    )
    vkitti = _read(vkitti_track_audit) if vkitti_track_audit else None
    argoverse = _read(argoverse_ttc_audit) if argoverse_ttc_audit else None
    carla = _read(carla_rgbd_audit) if carla_rgbd_audit else None
    corridor = _read(corridor_safety_audit) if corridor_safety_audit else None
    corridor_replay = _read(corridor_safety_replay) if corridor_safety_replay else None
    bonn = _read(bonn_rgbd_audit) if bonn_rgbd_audit else None
    bonn_reprojection = _read(bonn_reprojection_audit) if bonn_reprojection_audit else None
    revel = _read(revel_rgb_labels_audit) if revel_rgb_labels_audit else None
    revel_vicon = _read(revel_vicon_trajectory_audit) if revel_vicon_trajectory_audit else None
    revel_reprojection = _read(revel_rgb_vicon_reprojection_audit) if revel_rgb_vicon_reprojection_audit else None
    revel_detector = _read(revel_detector_benchmark) if revel_detector_benchmark else None
    detector_guard = _read(revel_detector_guard) if revel_detector_guard else None
    detector_vicon = _read(revel_detector_vicon_alignment) if revel_detector_vicon_alignment else None
    gates = [
        _gate(
            "analytic_metric_geometry",
            geometry_passed,
            f"static RMSE={geometry.get('static_reprojection_rmse_meters')}m; false DROP={geometry.get('visibility_gap_false_drop_count')}; detected DROP={geometry.get('drop_detected_sample_count')}/{geometry.get('drop_expected_sample_count')}",
            "offline-theory-only",
        ),
        _gate(
            "analytic_dynamic_ttc",
            dynamic_passed,
            f"velocity error={dynamic.get('max_velocity_error_mps')}m/s; TTC error={dynamic.get('max_ttc_error_ms')}ms; collision accuracy={dynamic.get('collision_label_accuracy')}",
            "offline-theory-only",
        ),
        _gate(
            "public_source_native_temporal_consistency",
            temporal_passed,
            "; ".join(
                f"{report.get('trajectory')}: valid={report.get('aggregate', {}).get('median_valid_projection_fraction')}, residual={report.get('aggregate', {}).get('median_pair_median_abs_depth_residual_m')}m"
                for report in temporal
            ),
            "source-data-screening-only",
        ),
    ]
    if vkitti is not None:
        source_dynamic_passed = (
            vkitti.get("source_moving_pair_count", 0) > 0
            and vkitti.get("classification", {}).get("precision", 0.0) >= .99
            and vkitti.get("classification", {}).get("recall", 0.0) >= .99
            and vkitti.get("physical_ttc_seconds_admitted") is False
            and vkitti.get("ustrf_motion_input_admitted") is False
        )
        gates.append(_gate(
            "public_dynamic_source_native_tracks",
            source_dynamic_passed,
            f"pairs={vkitti.get('consecutive_track_pair_count')}; source-moving={vkitti.get('source_moving_pair_count')}; precision={vkitti.get('classification', {}).get('precision')}; recall={vkitti.get('classification', {}).get('recall')}; no timestamp/body receipt",
            "source-data-screening-only",
        ))
    if argoverse is not None:
        timestamped_ttc_passed = (
            argoverse.get("source_ttc_seconds_available") is True
            and argoverse.get("front_facing_track_pair_count", 0) > 0
            and argoverse.get("timestamp", {}).get("median_period_seconds", 0.0) > 0.0
            and argoverse.get("kinematics", {}).get("ttc_collision_candidate_count_within_horizon", 0) > 0
            and argoverse.get("ustrf_motion_input_admitted") is False
        )
        gates.append(_gate(
            "public_timestamped_source_native_ttc",
            timestamped_ttc_passed,
            f"pairs={argoverse.get('front_facing_track_pair_count')}; period={argoverse.get('timestamp', {}).get('median_period_seconds')}s; approaching={argoverse.get('kinematics', {}).get('approaching_track_pair_count')}; candidates={argoverse.get('kinematics', {}).get('ttc_collision_candidate_count_within_horizon')}; no RGB-D/body receipt",
            "source-data-screening-only",
        ))
    if carla is not None:
        rgbd_pose_passed = (
            carla.get("ok") is True
            and carla.get("source_rgbd_pose_sequence_admitted") is True
            and carla.get("frame_count", 0) >= 2
            and carla.get("timestamp", {}).get("median_interval_seconds", 0.0) > 0.0
            and carla.get("depth", {}).get("minimum_m", 0.0) > 0.0
            and carla.get("ustrf_metric_geometry_input_admitted") is False
        )
        gates.append(_gate(
            "public_timestamped_rgbd_pose_sequence",
            rgbd_pose_passed,
            f"frames={carla.get('frame_count')}; period={carla.get('timestamp', {}).get('median_interval_seconds')}s; depth={carla.get('depth', {}).get('minimum_m')}..{carla.get('depth', {}).get('maximum_m')}m; handedness={carla.get('camera_coordinate_handedness')}; no body/ground/event receipt",
            "source-data-screening-only",
        ))
    if corridor is not None or corridor_replay is not None:
        if corridor is None or corridor_replay is None:
            corridor_passed = False
            detail = "blocked: both CUDA truth audit and Kotlin replay receipt are required"
        else:
            corridor_passed = (
                corridor.get("scene_count") == 256
                and corridor.get("body_frame_ground_truth") is True
                and corridor.get("local_ground_truth") is True
                and corridor.get("dynamic_event_truth") is True
                and corridor.get("expected_clear_stop_count") == 0
                and corridor_replay.get("scene_count") == corridor.get("scene_count")
                and corridor_replay.get("action_match_count") == corridor_replay.get("scene_count")
                and corridor_replay.get("expected_stop_count") == corridor_replay.get("actual_stop_count")
                and corridor_replay.get("clear_stop_count") == 0
                and corridor_replay.get("eligible_corridor_selection_count") == corridor_replay.get("matching_corridor_selection_count")
                and corridor_replay.get("fault_scene_count") == corridor_replay.get("fault_stop_count")
            )
            detail = f"scenes={corridor.get('scene_count')}; expected/actual STOP={corridor_replay.get('expected_stop_count')}/{corridor_replay.get('actual_stop_count')}; clear STOP={corridor_replay.get('clear_stop_count')}; corridor selections={corridor_replay.get('matching_corridor_selection_count')}/{corridor_replay.get('eligible_corridor_selection_count')}"
        gates.append(_gate("analytic_body_capsule_corridor_safety", corridor_passed, detail, "offline-theory-only"))
    if bonn is not None or bonn_reprojection is not None:
        if bonn is None or bonn_reprojection is None:
            bonn_passed = False
            detail = "blocked: both source integrity/synchronization and CUDA temporal-reprojection receipts are required"
        else:
            bonn_passed = (
                bonn.get("audit_passed") is True
                and bonn.get("frame_count", 0) >= 100
                and bonn.get("rgb_depth_association", {}).get("within_20ms_fraction", 0.0) == 1.0
                and bonn.get("rgb_pose_association", {}).get("within_20ms_fraction", 0.0) >= .995
                and bonn.get("depth", {}).get("valid_fraction", 0.0) >= .50
                and bonn.get("ustrf_metric_geometry_input_admitted") is False
                and bonn_reprojection.get("aggregate", {}).get("median_valid_projection_fraction", 0.0) >= .80
                and bonn_reprojection.get("aggregate", {}).get("median_pair_median_abs_depth_residual_m", float("inf")) <= .05
                and bonn_reprojection.get("ustrf_geometry_input_admitted") is False
            )
            detail = f"frames={bonn.get('frame_count')}; RGB-depth 20ms={bonn.get('rgb_depth_association', {}).get('within_20ms_fraction')}; RGB-pose 20ms={bonn.get('rgb_pose_association', {}).get('within_20ms_fraction')}; reprojection valid={bonn_reprojection.get('aggregate', {}).get('median_valid_projection_fraction')}; median residual={bonn_reprojection.get('aggregate', {}).get('median_pair_median_abs_depth_residual_m')}m; no body/event receipt"
        gates.append(_gate("public_dynamic_rgbd_pose_sequence", bonn_passed, detail, "source-data-screening-only"))
    if revel is not None:
        revel_passed = (
            revel.get("pairing", {}).get("paired_frames", 0) >= 1_000
            and revel.get("pairing", {}).get("image_without_label") == 0
            and revel.get("pairing", {}).get("label_without_image") == 0
            and revel.get("labels", {}).get("annotated_boxes", 0) >= 1_000
            and revel.get("labels", {}).get("valid_normalized_box_fraction") == 1.0
            and revel.get("temporal", {}).get("median_frame_rate_hz", 0.0) > 0.0
            and revel.get("admission", {}).get("external_2d_dynamic_object_truth_admitted") is True
            and "physical TTC" in revel.get("admission", {}).get("not_admitted_for", [])
        )
        detail = f"paired frames={revel.get('pairing', {}).get('paired_frames')}; boxes={revel.get('labels', {}).get('annotated_boxes')}; valid boxes={revel.get('labels', {}).get('valid_normalized_box_fraction')}; rate={revel.get('temporal', {}).get('median_frame_rate_hz')}Hz; 2D-only, no metric/TTC/body receipt"
        gates.append(_gate("public_dynamic_2d_person_labels", revel_passed, detail, "source-data-screening-only"))
    if revel_detector is not None or detector_guard is not None:
        if revel_detector is None or detector_guard is None:
            detector_passed = False
            detail = "blocked: both bounded detector benchmark and guarded-run receipt are required"
        else:
            detector_dataset = revel_detector.get("dataset", {})
            detector_model = revel_detector.get("model", {})
            detector_metrics = revel_detector.get("fixed_score_metrics", {})
            detector_strata = revel_detector.get("recall_by_normalized_box_area", {})
            details_receipt = revel_detector.get("details_receipt") or {}
            details_path = Path(details_receipt.get("path", ""))
            details_valid = (
                details_path.is_file()
                and details_receipt.get("frame_records") == detector_dataset.get("evaluated_frames")
                and details_receipt.get("sha256") == _sha256(details_path)
                and sum(1 for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()) == detector_dataset.get("evaluated_frames")
            )
            guard_limits = detector_guard.get("limits", {})
            guard_observed = detector_guard.get("observed", {})
            detector_passed = (
                revel_detector.get("format") == "blindassist_revel_yolo11n_person_benchmark_v2"
                and detector_dataset.get("total_frames") == 8_580
                and detector_dataset.get("evaluated_frames", 0) >= 512
                and detector_dataset.get("selection") == "uniform"
                and detector_dataset.get("selected_first_index") == 0
                and detector_dataset.get("selected_last_index") == 8_579
                and detector_dataset.get("person_ground_truth_boxes", 0) >= 700
                and details_valid
                and detector_model.get("batch") == 1
                and detector_model.get("half") is False
                and 0.0 <= revel_detector.get("ap50_over_score_floor", -1.0) <= 1.0
                and all(0.0 <= detector_metrics.get(metric, -1.0) <= 1.0 for metric in ("precision", "recall", "f1"))
                and all(detector_strata.get(name, {}).get("ground_truth", 0) > 0 for name in ("small", "medium", "large"))
                and revel_detector.get("admission", {}).get("offline_rgb_person_detection_baseline_admitted") is True
                and revel_detector.get("production_authority") is False
                and detector_guard.get("format") == "blindassist_guarded_gpu_run_v1"
                and detector_guard.get("exit_code") == 0
                and detector_guard.get("stop_reason") is None
                and detector_guard.get("monitor_samples", 0) > 0
                and guard_limits.get("max_frames") == detector_dataset.get("evaluated_frames")
                and guard_limits.get("batch") == 1
                and guard_observed.get("relevant_system_events") == 0
                and guard_observed.get("max_temperature_c", float("inf")) < guard_limits.get("max_temperature_c", 0)
            )
            detail = (
                f"bounded frames={detector_dataset.get('evaluated_frames')}/{detector_dataset.get('total_frames')}; detail receipt valid={details_valid}; "
                f"boxes={detector_dataset.get('person_ground_truth_boxes')}; AP50={revel_detector.get('ap50_over_score_floor')}; "
                f"precision/recall/F1={detector_metrics.get('precision')}/{detector_metrics.get('recall')}/{detector_metrics.get('f1')}; "
                f"small/medium/large recall={detector_strata.get('small', {}).get('recall')}/{detector_strata.get('medium', {}).get('recall')}/{detector_strata.get('large', {}).get('recall')}; "
                f"max temperature={guard_observed.get('max_temperature_c')}C; max power={guard_observed.get('max_power_draw_w')}W; "
                f"system events={guard_observed.get('relevant_system_events')}; bounded 2D baseline only, no distance/TTC/body/event authority"
            )
        gates.append(_gate("public_dynamic_2d_person_detector_measurement", detector_passed, detail, "bounded-public-rgb-baseline-only"))
    if detector_vicon is not None:
        detector_details = (revel_detector or {}).get("details_receipt") or {}
        alignment_summary = detector_vicon.get("summary", {})
        document_range = alignment_summary.get("document_range_summary", {})
        within = document_range.get("within_0_5m", {})
        beyond = document_range.get("beyond_5m", {})
        alignment_passed = (
            revel_detector is not None
            and detector_vicon.get("format") in {"blindassist_revel_detector_vicon_failure_alignment_v1", "blindassist_revel_detector_vicon_failure_alignment_v2"}
            and detector_vicon.get("source", {}).get("details_sha256") == detector_details.get("sha256")
            and alignment_summary.get("box_count") == revel_detector.get("dataset", {}).get("person_ground_truth_boxes")
            and alignment_summary.get("vicon_aligned_box_count", 0) >= 500
            and within.get("ground_truth", 0) >= 400
            and beyond.get("ground_truth", 0) >= 50
            and 0.0 <= within.get("recall", -1.0) <= 1.0
            and 0.0 <= beyond.get("recall", -1.0) <= 1.0
            and len(within.get("recall_wilson95") or []) == 2
            and len(beyond.get("recall_wilson95") or []) == 2
            and detector_vicon.get("admission", {}).get("source_detector_range_stratification_admitted") is True
            and detector_vicon.get("production_authority") is False
        )
        detail = (
            f"boxes={alignment_summary.get('box_count')}; Vicon-aligned={alignment_summary.get('vicon_aligned_box_count')}; "
            f"0-5m recall={within.get('recall')} (Wilson95={within.get('recall_wilson95')}); "
            f">5m recall={beyond.get('recall')} (Wilson95={beyond.get('recall_wilson95')}); "
            "source helmet-to-sensor range only, no user-body distance/TTC/event authority"
        )
        gates.append(_gate("public_detector_source_vicon_range_stratification", alignment_passed, detail, "source-range-stratification-only"))
        if detector_vicon.get("format") == "blindassist_revel_detector_vicon_failure_alignment_v2":
            motion_count = alignment_summary.get("source_motion_aligned_box_count", 0)
            by_motion = alignment_summary.get("recall_by_source_radial_motion", {})
            by_ttc = alignment_summary.get("recall_by_source_ttc_proxy", {})
            motion_contract = detector_vicon.get("source", {}).get("motion_contract", {})
            record_receipt = detector_vicon.get("source", {}).get("box_records_receipt", {})
            record_path = Path(record_receipt.get("path", ""))
            record_receipt_valid = (
                record_path.is_file()
                and record_receipt.get("records") == alignment_summary.get("box_count")
                and record_receipt.get("sha256") == _sha256(record_path)
                and sum(1 for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()) == record_receipt.get("records")
            )
            motion_ground_truth = sum((by_motion.get(name, {}).get("ground_truth") or 0) for name in ("approaching", "quasi_static", "receding"))
            ttc_ground_truth = sum((item.get("ground_truth") or 0) for item in by_ttc.values())
            approaching = by_motion.get("approaching", {})
            quasi_static = by_motion.get("quasi_static", {})
            receding = by_motion.get("receding", {})
            radial_passed = (
                revel_detector is not None
                and detector_vicon.get("source", {}).get("details_sha256") == detector_details.get("sha256")
                and detector_vicon.get("source", {}).get("bag_sha256_verified") == detector_vicon.get("source", {}).get("bag_sha256_from_audit")
                and alignment_summary.get("box_count") == revel_detector.get("dataset", {}).get("person_ground_truth_boxes")
                and motion_count >= 400
                and motion_ground_truth == motion_count
                and all((by_motion.get(name, {}).get("ground_truth") or 0) > 0 for name in ("approaching", "quasi_static", "receding"))
                and all(0.0 <= by_motion.get(name, {}).get("recall", -1.0) <= 1.0 for name in ("approaching", "quasi_static", "receding"))
                and ttc_ground_truth == approaching.get("ground_truth")
                and motion_contract.get("timestamp_basis") == "rosbag record time"
                and motion_contract.get("minimum_continuous_interval_s") == .005
                and motion_contract.get("maximum_continuous_interval_s") == .05
                and motion_contract.get("maximum_single_track_world_speed_mps") == 5.0
                and motion_contract.get("approach_recede_deadband_mps") == .10
                and motion_contract.get("offline_noncausal") is True
                and record_receipt_valid
                and detector_vicon.get("admission", {}).get("source_detector_radial_motion_stratification_admitted") is True
                and "physical assistive TTC" in detector_vicon.get("admission", {}).get("not_admitted_for", [])
                and detector_vicon.get("production_authority") is False
            )
            within_3s = alignment_summary.get("document_motion_summary", {}).get("ttc_proxy_within_3s", {})
            radial_detail = (
                f"motion-aligned={motion_count}/{alignment_summary.get('box_count')}; records valid={record_receipt_valid}; "
                f"approaching/quasi-static/receding recall={approaching.get('recall')}/{quasi_static.get('recall')}/{receding.get('recall')}; "
                f"TTC-proxy<3s recall={within_3s.get('recall')} over {within_3s.get('ground_truth')}; "
                "offline noncausal source marker-range proxy only, no physical TTC/body/event/device authority"
            )
            gates.append(_gate("public_detector_source_vicon_radial_motion_stratification", radial_passed, radial_detail, "source-motion-stratification-only"))
    if revel_vicon is not None:
        people = revel_vicon.get("helmet_people", {})
        vicon_passed = (
            revel_vicon.get("admission", {}).get("external_metric_person_sensor_trajectory_truth_admitted") is True
            and revel_vicon.get("source", {}).get("world_frame") == ["/vicon/world"]
            and len(people) == 2
            and all(
                person.get("valid_nonorigin_pose_count", 0) >= 20_000
                and person.get("relative_to_sensor", {}).get("synchronized_valid_pose_fraction", 0.0) >= .90
                and person.get("relative_to_sensor", {}).get("continuity_filtered_relative_pair_count", 0) >= 20_000
                and person.get("relative_to_sensor", {}).get("sensor_local_range_m", {}).get("min", 0.0) > 0.0
                for person in people.values()
            )
            and "physical assistive TTC" in revel_vicon.get("admission", {}).get("not_admitted_for", [])
        )
        detail = "; ".join(
            f"{name}: poses={person.get('valid_nonorigin_pose_count')}; sync={person.get('relative_to_sensor', {}).get('synchronized_valid_pose_fraction')}; continuous relative={person.get('relative_to_sensor', {}).get('continuity_filtered_relative_pair_count')}; range median={person.get('relative_to_sensor', {}).get('sensor_local_range_m', {}).get('median')}m"
            for name, person in sorted(people.items())
        ) + "; Vicon person/sensor source truth only, no wearable body/event receipt"
        gates.append(_gate("public_dynamic_metric_person_sensor_trajectories", vicon_passed, detail, "source-data-screening-only"))
    if revel_reprojection is not None:
        classes = revel_reprojection.get("reprojection", {})
        reprojection_passed = (
            revel_reprojection.get("admission", {}).get("source_cross_modal_2d_3d_alignment_admitted") is True
            and revel_reprojection.get("frame_alignment", {}).get("archive_image_count") == revel_reprojection.get("frame_alignment", {}).get("bag_image_count")
            and len(classes) == 2
            and all(
                item.get("labelled_frame_count", 0) >= 6_000
                and item.get("valid_sensor_and_person_sync_count", 0) >= 4_000
                and item.get("inside_any_matching_box_fraction_of_usable", 0.0) >= .89
                and item.get("closest_matching_box_outside_distance_px", {}).get("p95", float("inf")) <= 3.0
                and item.get("ambiguous_same_class_frame_count", 0) <= 1
                for item in classes.values()
            )
            and "physical assistive TTC" in revel_reprojection.get("admission", {}).get("not_admitted_for", [])
        )
        detail = "; ".join(
            f"{name}: sync={item.get('valid_sensor_and_person_sync_count')}; inside={item.get('inside_any_matching_box_fraction_of_usable')}; p95 outside={item.get('closest_matching_box_outside_distance_px', {}).get('p95')}px; ambiguous={item.get('ambiguous_same_class_frame_count')}"
            for name, item in sorted(classes.items())
        ) + "; calibrated source RGB/Vicon consistency only"
        gates.append(_gate("public_dynamic_rgb_vicon_cross_modal_alignment", reprojection_passed, detail, "source-data-screening-only"))
    gates.append(_gate(
        "device_metric_geometry_admission",
        False,
        "blocked: no independently verified device depth registration, full camera-body extrinsics, body-local ground truth, dynamic event truth, or target-device latency/thermal receipt",
        "not-authorized",
    ))
    return {
        "format": "blindassist_ustrf_sc_research_benchmark_v3",
        "decision": "CONDITIONAL_RESEARCH_GO",
        "decision_summary": "Analytic geometry/TTC contracts and public source-native temporal checks are reproducible; bounded REveL detector range and radial-motion strata remain source-only; no device or user-facing safety authorization exists.",
        "gates": gates,
        "input_receipts": {
            "geometry": str(geometry_audit), "dynamic": str(dynamic_audit), "source_native_temporal": [str(path) for path in temporal_audits], "source_native_dynamic_tracks": str(vkitti_track_audit) if vkitti_track_audit else None, "source_native_timestamped_ttc": str(argoverse_ttc_audit) if argoverse_ttc_audit else None, "source_native_rgbd_pose": str(carla_rgbd_audit) if carla_rgbd_audit else None, "analytic_body_capsule_corridor": str(corridor_safety_audit) if corridor_safety_audit else None, "analytic_body_capsule_corridor_replay": str(corridor_safety_replay) if corridor_safety_replay else None, "public_dynamic_rgbd_pose": str(bonn_rgbd_audit) if bonn_rgbd_audit else None, "public_dynamic_rgbd_reprojection": str(bonn_reprojection_audit) if bonn_reprojection_audit else None, "public_dynamic_2d_person_labels": str(revel_rgb_labels_audit) if revel_rgb_labels_audit else None, "public_dynamic_2d_person_detector": str(revel_detector_benchmark) if revel_detector_benchmark else None, "public_dynamic_2d_person_detector_guard": str(revel_detector_guard) if revel_detector_guard else None, "public_detector_source_vicon_range_stratification": str(revel_detector_vicon_alignment) if revel_detector_vicon_alignment else None, "public_detector_source_vicon_radial_motion_stratification": str(revel_detector_vicon_alignment) if detector_vicon and detector_vicon.get("format") == "blindassist_revel_detector_vicon_failure_alignment_v2" else None, "public_dynamic_metric_person_sensor_trajectories": str(revel_vicon_trajectory_audit) if revel_vicon_trajectory_audit else None, "public_dynamic_rgb_vicon_cross_modal_alignment": str(revel_rgb_vicon_reprojection_audit) if revel_rgb_vicon_reprojection_audit else None,
        },
        "next_required_evidence": [
            "tracked continuous RGB-D/VIO sequences with dynamic trajectory truth stratified by TTC",
            "controlled metric depth registration, camera-body extrinsics, and local ground-plane truth",
            "target-device bounded-queue latency, memory, power, and thermal p50/p95/p99 receipts",
            "independently fixed larger/full detector evaluation with small/distant-target and TTC-stratified recall",
        ],
        "production_authority": False,
    }


def write(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "research_benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(gate['name'])}</td><td>{'PASS' if gate['passed'] else 'BLOCKED'}</td>"
        f"<td>{html.escape(gate['authority'])}</td><td>{html.escape(gate['detail'])}</td>"
        "</tr>"
        for gate in report["gates"]
    )
    required = "".join(f"<li>{html.escape(item)}</li>" for item in report["next_required_evidence"])
    document = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>USTRF-SC Research Benchmark</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:.55rem;text-align:left}}th{{background:#eef3f8}}.blocked{{color:#9b1c1c}}</style>
<h1>USTRF-SC 研究基准</h1><p><strong>{html.escape(report['decision'])}</strong> — {html.escape(report['decision_summary'])}</p>
<table><thead><tr><th>Gate</th><th>状态</th><th>授权边界</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table>
<h2>后续必须证据</h2><ul>{required}</ul><p>本报告不构成设备、用户或生产安全授权。</p></html>"""
    (output / "research_benchmark_report.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, action="append", required=True)
    parser.add_argument("--vkitti-track-audit", type=Path)
    parser.add_argument("--argoverse-ttc-audit", type=Path)
    parser.add_argument("--carla-rgbd-audit", type=Path)
    parser.add_argument("--corridor-safety-audit", type=Path)
    parser.add_argument("--corridor-safety-replay", type=Path)
    parser.add_argument("--bonn-rgbd-audit", type=Path)
    parser.add_argument("--bonn-reprojection-audit", type=Path)
    parser.add_argument("--revel-rgb-labels-audit", type=Path)
    parser.add_argument("--revel-vicon-trajectory-audit", type=Path)
    parser.add_argument("--revel-rgb-vicon-reprojection-audit", type=Path)
    parser.add_argument("--revel-detector-benchmark", type=Path)
    parser.add_argument("--revel-detector-guard", type=Path)
    parser.add_argument("--revel-detector-vicon-alignment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.geometry_audit, args.dynamic_audit, args.temporal_audit, args.vkitti_track_audit, args.argoverse_ttc_audit, args.carla_rgbd_audit, args.corridor_safety_audit, args.corridor_safety_replay, args.bonn_rgbd_audit, args.bonn_reprojection_audit, args.revel_rgb_labels_audit, args.revel_vicon_trajectory_audit, args.revel_rgb_vicon_reprojection_audit, args.revel_detector_benchmark, args.revel_detector_guard, args.revel_detector_vicon_alignment)
    write(report, args.output)
    print(json.dumps({"decision": report["decision"], "gate_count": len(report["gates"]), "passing_gate_count": sum(gate["passed"] for gate in report["gates"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
