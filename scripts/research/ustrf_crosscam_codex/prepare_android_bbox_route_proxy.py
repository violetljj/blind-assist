#!/usr/bin/env python3
"""Build non-authoritative inputs for the shipped Android USTRF bbox-route adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2

try:
    from .contract import BUNDLE_SCHEMA, CONTRACT_ID, load_json, sha256_file, write_json
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import BUNDLE_SCHEMA, CONTRACT_ID, load_json, sha256_file, write_json


U0_CONTRACT_ID = "ustrf_sc_u0_teacher_upper_bound_v1"
ARM_ID = "detector_bbox_explicit_route"
ADAPTER_ID = "detector_bbox_explicit_route_adapter_v1"


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"input must stay inside inference root: {path}") from error


def route_waypoints(polygon: list[list[float]]) -> list[tuple[float, float]]:
    near_x = (float(polygon[0][0]) + float(polygon[1][0])) / 2.0
    near_y = (float(polygon[0][1]) + float(polygon[1][1])) / 2.0
    far_x = (float(polygon[2][0]) + float(polygon[3][0])) / 2.0
    far_y = (float(polygon[2][1]) + float(polygon[3][1])) / 2.0
    return [
        (near_x + (far_x - near_x) * ratio, near_y + (far_y - near_y) * ratio)
        for ratio in (0.25, 0.55, 0.85)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    bundle = load_json(args.bundle_manifest)
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("contract_id") != CONTRACT_ID:
        raise ValueError("bundle schema/contract mismatch")
    if bundle.get("source_projection", {}).get("mode") != "rectilinear_identity_v1":
        raise ValueError("Android candidate must receive the same rectilinear pixels as the Codex bundle")
    video = Path(bundle["video_path"])
    if sha256_file(video) != bundle.get("video_sha256"):
        raise ValueError("bundle video SHA drift")
    threshold = load_json(args.threshold_config)
    teacher_contract = load_json(args.teacher_contract)
    shared_kernel_sha = teacher_contract["prediction_evidence_contract"]["shared_decision_kernel_implementation_sha256"]
    rows = bundle["review_artifacts"]["causal_codex_baseline"]["frames"]
    if not rows or any(int(row["relative_ms"]) % 500 for row in rows):
        raise ValueError("bundle causal inventory is not the frozen 500ms grid")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("cannot open bundle video")
    frames: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows):
            timestamp_ms = int(row["source_timestamp_ms"])
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode source timestamp {timestamp_ms}ms")
            frames.append({
                "frame_id": row["frame_id"],
                "frame_index": index,
                "capture_timestamp_ns": timestamp_ms * 1_000_000,
                "video_pts_ms": timestamp_ms,
                "frame_payload_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            })
    finally:
        capture.release()
    args.output_dir.mkdir(parents=True)
    episode_id = f"crosscam-{bundle['source']['source_id']}-{bundle['window']['start_ms']}-{bundle['window']['duration_ms']}"
    ledger_path = args.output_dir / "capture-frame-ledger.json"
    write_json(ledger_path, {
        "schema": "blindassist_capture_frame_ledger_v1",
        "episode_id": episode_id,
        "source_video_sha256": bundle["video_sha256"],
        "frame_payload_contract": "opencv_bgr8_row_major_proxy_v1",
        "authority": "non_authoritative_crosscam_proxy_only",
        "frames": frames,
    })
    polygon = bundle["assumed_geometry"]["route_polygon_xy_norm"]
    points = route_waypoints(polygon)
    geometry = bundle["assumed_geometry"]
    route_source = geometry.get("route_source")
    route_source_authority = geometry.get("route_source_authority")
    route_projection_confidence = geometry.get("route_projection_confidence", 0.5)
    route_projection_confidence_origin = (
        "bundle_assumed_geometry" if "route_projection_confidence" in geometry else "proxy_default_0p5_v1"
    )
    if not isinstance(route_source, str) or not route_source or not isinstance(route_source_authority, str):
        raise ValueError("Android proxy route requires disclosed route_source and route_source_authority")
    if not isinstance(route_projection_confidence, (int, float)) or not 0.0 <= float(route_projection_confidence) <= 1.0:
        raise ValueError("Android proxy route requires bounded route_projection_confidence")
    projection_receipt_path = args.output_dir / "route-projection-receipt.json"
    write_json(projection_receipt_path, {
        "schema": "blindassist_ustrf_crosscam_route_projection_receipt_v1",
        "contract_id": CONTRACT_ID,
        "episode_id": episode_id,
        "source_projection_mode": bundle["source_projection"]["mode"],
        "forward_axis_authority": bundle["source_projection"].get(
            "forward_axis_authority", "source_rectilinear_optical_axis_proxy_v1"
        ),
        "route_source": route_source,
        "route_source_authority": route_source_authority,
        "route_projection_confidence": float(route_projection_confidence),
        "route_projection_confidence_origin": route_projection_confidence_origin,
        "route_polygon_xy_norm": polygon,
        "dynamic_projection_present": False,
        "world_route_present": False,
        "camera_pose_receipt_present": False,
        "authority": "manual_current_frame_proxy_only",
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    })
    route_path = args.output_dir / "explicit-route-intent.json"
    write_json(route_path, {
        "schema": "blindassist_explicit_route_intent_episode_v1",
        "episode_id": episode_id,
        "parent_source_id": bundle["source"]["source_id"],
        "provider": {
            "type": "explicit_user_choice",
            "provider_id": route_source,
            "inferred_by_risk_model": False,
            "input_space": "current_camera_frame",
        },
        "coordinate_contract": {
            "space": "normalized_current_camera_frame_xy",
            "projection_receipt_id": f"{episode_id}-assumed-projection",
            "projection_receipt_path": relative_path(args.inference_root, projection_receipt_path),
            "projection_receipt_sha256": sha256_file(projection_receipt_path),
            "projection_receipt_required": True,
            "device_to_world_alignment_receipt_id": None,
            "device_to_world_alignment_receipt_required_for_world_waypoints": True,
        },
        "samples": [{
            "timestamp_ms": frame["video_pts_ms"],
            "valid_until_timestamp_ms": frame["video_pts_ms"] + 500,
            "confidence": 1.0,
            "route_valid": True,
            "horizon_waypoints": [
                {"horizon_ms": horizon, "xy_norm": [point[0], point[1]]}
                for horizon, point in zip((1000, 2000, 3000), points)
            ],
        } for frame in frames],
        "fallback": {
            "missing_stale_or_low_confidence_route": "context_attention_only",
            "directional_instruction_allowed": False,
            "intervention_upgrade_allowed": False,
        },
        "training_isolation": {
            "future_video_teacher_allowed": "train_only_oracle_proxy",
            "future_video_teacher_allowed_in_eval_or_runtime": False,
            "production_authorized": False,
            "risk_event_truth": False,
            "same_parent_source_split_required": True,
        },
        "authorization": {"role": "non_authoritative_crosscam_proxy", "runtime_authorized": False},
        "corridor_extension": {
            "schema": "blindassist_explicit_route_corridor_polygon_r1",
            "space": "normalized_current_camera_frame_xy",
            "polygon_xy_norm": polygon,
            "object_contact_proxy": "bbox_bottom_center_v1",
            "boundary_uncertainty_required": True,
            "v1_android_adapter_consumes_extension": False,
        },
    })
    artifact_path = args.output_dir / "fixed-no-fit-artifact.json"
    write_json(artifact_path, {
        "schema": "blindassist_ustrf_sc_u0_fixed_no_fit_artifact_v1",
        "arm_id": ARM_ID,
        "fit_executed": False,
        "authority": "non_authoritative_crosscam_proxy_only",
    })
    no_fit_receipt_path = args.output_dir / "fixed-no-fit-receipt.json"
    write_json(no_fit_receipt_path, {
        "schema": "blindassist_ustrf_crosscam_fixed_no_fit_receipt_v1",
        "bundle_manifest_sha256": sha256_file(args.bundle_manifest),
        "fit_executed": False,
        "training_authorized": False,
    })
    route_sha = sha256_file(route_path)
    ledger_sha = sha256_file(ledger_path)
    artifact_sha = sha256_file(artifact_path)
    cadence = {
        "schema": "blindassist_ustrf_sc_u0_decision_cadence_v1",
        "canonical_step_ms": 500,
        "frame_selection_policy": "ledger_exact_grid_from_zero_v1",
        "missing_grid_frame_policy": "fail_episode_v1",
        "interpolation_policy": "forbidden",
        "episode_reset_policy": "reset_before_first_frame_v1",
        "route_sample_policy": "latest_valid_generated_at_or_before_frame_v1",
    }
    manifest_path = args.output_dir / "sanitized-inference-manifest.json"
    manifest = {
        "schema": "blindassist_ustrf_sc_u0_sanitized_inference_manifest_v1",
        "contract_id": U0_CONTRACT_ID,
        "arm_id": ARM_ID,
        "episode_id": episode_id,
        "route_input_policy": "episode_explicit_causal_route_v1",
        "adapter_route_input_path": relative_path(args.inference_root, route_path),
        "adapter_route_input_sha256": route_sha,
        "adapter_route_source_episode_id": episode_id,
        "truth_route_intent_sha256": route_sha,
        "capture_frame_ledger_path": relative_path(args.inference_root, ledger_path),
        "capture_frame_ledger_sha256": ledger_sha,
        "input_video_path": relative_path(args.inference_root, video),
        "input_video_sha256": bundle["video_sha256"],
        "decision_cadence": cadence,
        "frames": frames,
        "blind_accessed": False,
        "future_inputs_used": False,
        "review_fields_present": False,
        "adjudication_fields_present": False,
        "event_label_fields_present": False,
    }
    write_json(manifest_path, manifest)
    request_path = args.output_dir / "adapter-request.json"
    write_json(request_path, {
        "schema": "blindassist_ustrf_sc_u0_candidate_adapter_request_v1",
        "contract_id": U0_CONTRACT_ID,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "adapter_runtime_id": "python_subprocess_v1",
        "episode_id": episode_id,
        "fold_held_out_session_id": bundle["source"]["source_id"],
        "fold_training_input_manifest_sha256": sha256_file(args.bundle_manifest),
        "fold_training_receipt_sha256": sha256_file(no_fit_receipt_path),
        "artifact_inventory_sha256": artifact_sha,
        "fold_artifact_sha256": artifact_sha,
        "threshold_config_sha256": sha256_file(args.threshold_config),
        "implementation_sha256": threshold["host_adapter_implementation_sha256"],
        "shared_decision_kernel_contract_id": "blindassist_shared_decision_kernel_v1",
        "shared_decision_kernel_implementation_sha256": shared_kernel_sha,
        "kernel_execution_backend_id": "android_kotlin_assist_decision_kernel_v1",
        "decision_profile_id": "STANDARD",
        "fit_policy": "fixed_no_fit_v1",
        "event_identity_policy": "kernel_native_optional_v1",
        "feedback_adapter_id": "offline_u0_feedback_receipt_v1",
        "kernel_trace_order": ["analyzer", "temporal", "stabilizer", "event", "confirmation", "feedback"],
        "route_input_policy": "episode_explicit_causal_route_v1",
        "adapter_route_input_sha256": route_sha,
        "adapter_route_source_episode_id": episode_id,
        "truth_route_intent_sha256": route_sha,
        "sanitized_inference_manifest_sha256": sha256_file(manifest_path),
        "input_video_sha256": bundle["video_sha256"],
        "source_capture_frame_ledger_sha256": ledger_sha,
        "decision_cadence": cadence,
        "frames": frames,
        "synthetic_fixture": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
    })
    receipt = {
        "bundle_manifest": str(args.bundle_manifest.resolve()),
        "request": str(request_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "artifact": str(artifact_path.resolve()),
        "threshold_config": str(args.threshold_config.resolve()),
        "inference_root": str(args.inference_root.resolve()),
        "frame_count": len(frames),
        "authority": "non_authoritative_crosscam_proxy_only",
    }
    write_json(args.output_dir / "input_receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = run(parse_args(argv))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "frame_count": receipt["frame_count"], "request": receipt["request"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
