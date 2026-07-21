#!/usr/bin/env python3
"""Execute preregistered USTRF U0 adapters and materialize an evidence bundle.

The runner owns orchestration and provenance, not model semantics.  Every arm
is launched through a fixed subprocess protocol.  The adapter must return the
per-frame result of the preregistered shared-decision-kernel backend; this
runner verifies identities, truth-frame alignment, authority declarations and
process completion before it writes an admissible prediction bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import validate_ustrf_sc_u0_prediction_bundle as admission


REGISTRY_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_registry_v1"
REQUEST_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_request_v1"
OUTPUT_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_output_v1"
RUNTIME_ID = "python_subprocess_v1"


class RunnerError(ValueError):
    """The requested execution cannot produce admissible U0 evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unchanged(path: Path, expected_sha256: str, *, where: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RunnerError(f"{where} changed after evidence materialization")


def canonical_json_sha256(path: Path) -> str:
    value = load_json(path, where=str(path))
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{where} must contain a JSON object")
    return value


def require_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError(f"{where} must be a non-empty string")
    return value


def require_bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise RunnerError(f"{where} must be boolean")
    return value


def local_file(root: Path, relative: Any, *, where: str) -> Path:
    text = require_text(relative, where=where)
    path = (root / text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RunnerError(f"{where} escapes its declared root") from error
    if not path.is_file():
        raise RunnerError(f"{where} is not a local file")
    return path


def bound_file(root: Path, binding: Mapping[str, Any], *, where: str) -> Path:
    if not isinstance(binding, Mapping):
        raise RunnerError(f"{where} binding must be an object")
    path = local_file(root, binding.get("path"), where=f"{where}.path")
    expected = binding.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise RunnerError(f"{where} SHA-256 mismatch")
    return path


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def copy_bound(source: Path, destination: Path) -> tuple[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256_file(destination)
    if digest != sha256_file(source):
        raise RunnerError(f"copied evidence changed: {destination}")
    return destination.as_posix(), digest


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "blindassist_ustrf_sc_u0_teacher_upper_bound_contract_v1":
        raise RunnerError("unexpected U0 contract schema")
    evidence = contract.get("prediction_evidence_contract")
    if not isinstance(evidence, dict):
        raise RunnerError("U0 contract lacks prediction_evidence_contract")
    if evidence.get("schema") != "blindassist_ustrf_sc_u0_prediction_evidence_contract_v2":
        raise RunnerError("unexpected U0 prediction evidence contract schema")
    if contract.get("prediction_schema") != "blindassist_ustrf_sc_u0_six_arm_predictions_v2":
        raise RunnerError("unexpected U0 prediction bundle schema")
    expected = {
        "adapter_registry_schema": REGISTRY_SCHEMA,
        "adapter_request_schema": REQUEST_SCHEMA,
        "adapter_output_schema": OUTPUT_SCHEMA,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise RunnerError(f"U0 contract {key} mismatch")
    allowed = evidence.get("allowed_adapter_runtime_ids")
    if not isinstance(allowed, list) or RUNTIME_ID not in allowed:
        raise RunnerError(f"U0 contract does not allow {RUNTIME_ID}")
    timeout_seconds = evidence.get("maximum_episode_execution_seconds")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise RunnerError("U0 contract maximum_episode_execution_seconds must be positive integer")
    if evidence.get("adapter_input_policy") != "sanitized_inference_manifest_only_v1":
        raise RunnerError("U0 adapter input policy must remain sanitized")
    try:
        admission._validate_decision_cadence(
            [{"video_pts_ms": 0, "capture_timestamp_ns": 0}],
            evidence,
            where="contract-smoke",
        )
    except admission.ContractError as error:
        raise RunnerError(str(error)) from error
    return evidence


def validate_registry(
    registry: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    registry_root: Path,
    truth_manifest: Mapping[str, Any],
    truth_manifest_sha256: str,
    synthetic_fixture: bool,
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Path], dict[str, Any]]:
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("contract_id") != contract.get("contract_id"):
        raise RunnerError("adapter registry identity mismatch")
    if registry.get("synthetic_fixture") is not synthetic_fixture:
        raise RunnerError("adapter registry synthetic_fixture mismatch")
    for key in ("blind_accessed", "future_inputs_used", "production_model_replacement_authorized"):
        if registry.get(key) is not False:
            raise RunnerError(f"adapter registry must declare {key}=false")
    kernel_path = bound_file(
        registry_root,
        registry.get("shared_decision_kernel_implementation"),
        where="shared_decision_kernel_implementation",
    )
    if normalized_text_sha256(kernel_path) != evidence.get("shared_decision_kernel_implementation_sha256"):
        raise RunnerError("shared decision kernel implementation differs from the frozen contract")
    expected_dependencies = evidence.get("shared_decision_kernel_dependency_sha256")
    dependency_rows = registry.get("shared_decision_kernel_dependencies")
    if not isinstance(expected_dependencies, dict) or not isinstance(dependency_rows, list):
        raise RunnerError("shared decision kernel dependency inventory is missing")
    if (
        any(not isinstance(row, dict) for row in dependency_rows)
        or len({row.get("name") for row in dependency_rows}) != len(dependency_rows)
        or {row.get("name") for row in dependency_rows} != set(expected_dependencies)
    ):
        raise RunnerError("shared decision kernel dependency inventory mismatch")
    kernel_dependencies: dict[str, Path] = {}
    for row in dependency_rows:
        name = row["name"]
        dependency = bound_file(registry_root, row, where=f"shared_decision_kernel_dependencies/{name}")
        if normalized_text_sha256(dependency) != expected_dependencies[name]:
            raise RunnerError(f"shared decision kernel dependency differs: {name}")
        kernel_dependencies[name] = dependency
    if registry.get("kernel_execution_backend_id") != evidence.get("kernel_execution_backend_id"):
        raise RunnerError("adapter registry kernel execution backend mismatch")

    required = {row.get("arm_id"): row for row in contract.get("required_arms", []) if isinstance(row, dict)}
    rows = registry.get("arms")
    if (
        not isinstance(rows, list)
        or len(rows) != len(required)
        or any(not isinstance(row, dict) for row in rows)
        or len({row.get("arm_id") for row in rows}) != len(rows)
        or {row.get("arm_id") for row in rows} != set(required)
    ):
        raise RunnerError("adapter registry must contain every preregistered arm exactly once")
    sessions = sorted({row["session_id"] for row in truth_manifest["episodes"]})
    episodes_by_session = {
        session_id: sorted(row["episode_id"] for row in truth_manifest["episodes"] if row["session_id"] == session_id)
        for session_id in sessions
    }
    result: dict[str, dict[str, Any]] = {}
    snapshot_rows: list[dict[str, Any]] = []
    for row in rows:
        arm_id = row["arm_id"]
        arm_contract = required[arm_id]
        for key in ("candidate_adapter_id", "fit_policy", "event_identity_policy", "route_input_policy"):
            if row.get(key) != arm_contract.get(key):
                raise RunnerError(f"{arm_id}.{key} differs from preregistration")
        if row.get("runtime_id") != RUNTIME_ID:
            raise RunnerError(f"{arm_id} runtime must be {RUNTIME_ID}")
        implementation = bound_file(registry_root, row.get("implementation"), where=f"{arm_id}.implementation")
        threshold = bound_file(registry_root, row.get("threshold_config"), where=f"{arm_id}.threshold_config")
        folds = row.get("folds")
        if (
            not isinstance(folds, list)
            or len(folds) != len(sessions)
            or any(not isinstance(fold, dict) for fold in folds)
            or len({fold.get("held_out_session_id") for fold in folds}) != len(folds)
            or {fold.get("held_out_session_id") for fold in folds} != set(sessions)
        ):
            raise RunnerError(f"{arm_id} registry folds must cover every truth session exactly once")
        fold_result: dict[str, dict[str, Any]] = {}
        fold_snapshot: list[dict[str, Any]] = []
        for fold in folds:
            held_out = fold["held_out_session_id"]
            artifact = bound_file(registry_root, fold.get("artifact"), where=f"{arm_id}/{held_out}.artifact")
            training_manifest_path = bound_file(
                registry_root,
                fold.get("training_input_manifest"),
                where=f"{arm_id}/{held_out}.training_input_manifest",
            )
            training_receipt_path = bound_file(
                registry_root,
                fold.get("training_receipt"),
                where=f"{arm_id}/{held_out}.training_receipt",
            )
            fit_policy = arm_contract["fit_policy"]
            training_sessions = [] if fit_policy == "fixed_no_fit_v1" else [value for value in sessions if value != held_out]
            training_episodes = [] if fit_policy == "fixed_no_fit_v1" else sorted(
                episode_id for session_id in training_sessions for episode_id in episodes_by_session[session_id]
            )
            expected_manifest = {
                "schema": evidence["fold_training_input_manifest_schema"],
                "contract_id": contract["contract_id"],
                "arm_id": arm_id,
                "candidate_adapter_id": arm_contract["candidate_adapter_id"],
                "fit_policy": fit_policy,
                "held_out_session_id": held_out,
                "truth_manifest_sha256": truth_manifest_sha256,
                "training_session_ids": training_sessions,
                "training_episode_ids": training_episodes,
                "held_out_inputs_used": False,
                "blind_accessed": False,
                "future_inputs_used": False,
            }
            if load_json(training_manifest_path, where=f"{arm_id}/{held_out} training manifest") != expected_manifest:
                raise RunnerError(f"{arm_id}/{held_out} training manifest is not the exact LOSO inventory")
            expected_receipt = {
                "schema": evidence["fold_training_receipt_schema"],
                "contract_id": contract["contract_id"],
                "arm_id": arm_id,
                "candidate_adapter_id": arm_contract["candidate_adapter_id"],
                "fit_policy": fit_policy,
                "held_out_session_id": held_out,
                "training_input_manifest_sha256": sha256_file(training_manifest_path),
                "artifact_sha256": sha256_file(artifact),
                "fit_executed": fit_policy == "leave_one_session_out_fit_v1",
                "held_out_inputs_used": False,
                "blind_accessed": False,
                "future_inputs_used": False,
                "provenance_completed": True,
                "failure_count": 0,
            }
            if load_json(training_receipt_path, where=f"{arm_id}/{held_out} training receipt") != expected_receipt:
                raise RunnerError(f"{arm_id}/{held_out} training receipt mismatch")
            fold_result[held_out] = {
                "artifact": artifact,
                "training_input_manifest": training_manifest_path,
                "training_receipt": training_receipt_path,
            }
            fold_snapshot.append({
                "held_out_session_id": held_out,
                "artifact_sha256": sha256_file(artifact),
                "training_input_manifest_sha256": sha256_file(training_manifest_path),
                "training_receipt_sha256": sha256_file(training_receipt_path),
            })
        result[arm_id] = {
            "row": row,
            "implementation": implementation,
            "threshold_config": threshold,
            "folds": fold_result,
        }
        snapshot_rows.append({
            "arm_id": arm_id,
            "candidate_adapter_id": arm_contract["candidate_adapter_id"],
            "runtime_id": RUNTIME_ID,
            "fit_policy": arm_contract["fit_policy"],
            "event_identity_policy": arm_contract["event_identity_policy"],
            "route_input_policy": arm_contract["route_input_policy"],
            "implementation_sha256": sha256_file(implementation),
            "threshold_config_sha256": sha256_file(threshold),
            "folds": sorted(fold_snapshot, key=lambda value: value["held_out_session_id"]),
        })
    snapshot = {
        "schema": evidence["adapter_registry_schema"],
        "contract_id": contract["contract_id"],
        "synthetic_fixture": synthetic_fixture,
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
        "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
        "shared_decision_kernel_dependency_sha256": expected_dependencies,
        "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
        "arms": sorted(snapshot_rows, key=lambda value: value["arm_id"]),
    }
    return result, kernel_path, kernel_dependencies, snapshot


def validate_truth_file(truth_root: Path, episode: Mapping[str, Any], stem: str, *, where: str) -> Path:
    path_key = f"{stem}_path"
    sha_key = f"{stem}_sha256"
    path = local_file(truth_root, episode.get(path_key), where=f"{where}.{path_key}")
    if sha256_file(path) != episode.get(sha_key):
        raise RunnerError(f"{where}.{stem} SHA-256 mismatch")
    return path


def expected_request(
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    arm_id: str,
    adapter_id: str,
    hashes: Mapping[str, str],
    fold: Mapping[str, Any],
    episode: Mapping[str, Any],
    inference_manifest: Mapping[str, Any],
    inference_manifest_sha256: str,
    frames: list[dict[str, Any]],
    synthetic_fixture: bool,
) -> dict[str, Any]:
    return {
        "schema": evidence["adapter_request_schema"],
        "contract_id": contract["contract_id"],
        "arm_id": arm_id,
        "candidate_adapter_id": adapter_id,
        "adapter_runtime_id": RUNTIME_ID,
        "shared_decision_kernel_contract_id": evidence["shared_decision_kernel_contract_id"],
        "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
        "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
        "decision_profile_id": evidence["decision_profile_id"],
        "feedback_adapter_id": evidence["feedback_adapter_id"],
        "kernel_trace_order": evidence["kernel_trace_order"],
        "decision_cadence": evidence["decision_cadence"],
        "fit_policy": fold["fit_policy"],
        "event_identity_policy": fold["event_identity_policy"],
        "route_input_policy": fold["route_input_policy"],
        "implementation_sha256": hashes["implementation"],
        "artifact_inventory_sha256": hashes["artifact"],
        "threshold_config_sha256": hashes["threshold_config"],
        "fold_held_out_session_id": episode["session_id"],
        "fold_artifact_sha256": fold["artifact_sha256"],
        "fold_training_input_manifest_sha256": fold["training_input_manifest_sha256"],
        "fold_training_receipt_sha256": fold["training_receipt_sha256"],
        "episode_id": episode["episode_id"],
        "sanitized_inference_manifest_sha256": inference_manifest_sha256,
        "input_video_sha256": episode["video_sha256"],
        "truth_route_intent_sha256": episode["route_intent_sha256"],
        "adapter_route_input_sha256": inference_manifest["adapter_route_input_sha256"],
        "adapter_route_source_episode_id": inference_manifest["adapter_route_source_episode_id"],
        "source_capture_frame_ledger_sha256": episode["capture_frame_ledger_sha256"],
        "synthetic_fixture": synthetic_fixture,
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
        "frames": frames,
    }


def validate_adapter_output(
    output: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_sha256: str,
    threshold_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[int], bool]:
    copied = {
        key: request[key]
        for key in (
            "contract_id", "arm_id", "candidate_adapter_id", "adapter_runtime_id",
            "shared_decision_kernel_contract_id", "shared_decision_kernel_implementation_sha256",
            "kernel_execution_backend_id", "decision_profile_id", "feedback_adapter_id",
            "kernel_trace_order", "decision_cadence", "fit_policy", "event_identity_policy",
            "route_input_policy", "implementation_sha256", "artifact_inventory_sha256",
            "threshold_config_sha256", "fold_held_out_session_id", "fold_artifact_sha256",
            "fold_training_input_manifest_sha256", "fold_training_receipt_sha256", "episode_id",
            "sanitized_inference_manifest_sha256", "input_video_sha256", "truth_route_intent_sha256",
            "adapter_route_input_sha256", "adapter_route_source_episode_id",
            "source_capture_frame_ledger_sha256",
            "synthetic_fixture", "blind_accessed", "future_inputs_used",
            "production_model_replacement_authorized",
        )
    }
    expected = {
        "schema": evidence["adapter_output_schema"],
        **copied,
        "execution_completed": True,
        "failure_count": 0,
    }
    for key, value in expected.items():
        if output.get(key) != value:
            raise RunnerError(f"adapter output {key} mismatch")
    try:
        admission._validate_android_backend_receipt(
            output,
            request=request,
            request_sha256=request_sha256,
            threshold_config=threshold_config,
            evidence=evidence,
            where="adapter output",
        )
        admission._validate_bbox_route_conditioning_receipt(
            output,
            request=request,
            threshold_config=threshold_config,
            evidence=evidence,
            where="adapter output",
        )
        admission._validate_dense_risk_evidence_receipt(
            output,
            request=request,
            threshold_config=threshold_config,
            evidence=evidence,
            where="adapter output",
        )
    except admission.ContractError as error:
        raise RunnerError(str(error)) from error
    abstained = require_bool(output.get("abstained"), where="adapter output abstained")
    frames = output.get("frames")
    expected_frames = request["frames"]
    if not isinstance(frames, list) or len(frames) != len(expected_frames):
        raise RunnerError("adapter output must contain every requested truth frame exactly once")
    alerts: list[int] = []
    normalized: list[dict[str, Any]] = []
    for index, (actual, expected_frame) in enumerate(zip(frames, expected_frames)):
        where = f"adapter output frames[{index}]"
        if not isinstance(actual, dict):
            raise RunnerError(f"{where} must be an object")
        for key, expected_value in expected_frame.items():
            if actual.get(key) != expected_value:
                raise RunnerError(f"{where}.{key} differs from requested truth ledger")
        try:
            delivered = admission._validate_decision(
                actual.get("decision"),
                contract=evidence,
                adapter_id=str(request["candidate_adapter_id"]),
                event_identity_policy=str(request["event_identity_policy"]),
                where=where,
            )
        except admission.ContractError as error:
            raise RunnerError(str(error)) from error
        if delivered:
            alerts.append(expected_frame["video_pts_ms"])
        normalized.append(actual)
    if abstained and alerts:
        raise RunnerError("abstained adapter output cannot deliver feedback")
    return normalized, alerts, abstained


def shuffled_route_sources(truth_manifest: Mapping[str, Any]) -> dict[str, str]:
    by_session: dict[str, list[str]] = {}
    for row in truth_manifest["episodes"]:
        by_session.setdefault(row["session_id"], []).append(row["episode_id"])
    result: dict[str, str] = {}
    for session_id, values in by_session.items():
        ordered = sorted(values)
        if len(ordered) < 2:
            raise RunnerError(f"shuffled control requires at least two episodes in session {session_id}")
        for index, episode_id in enumerate(ordered):
            result[episode_id] = ordered[(index + 1) % len(ordered)]
    return result


def copy_evidence(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise RunnerError(f"existing inference evidence differs: {destination}")
        return destination
    shutil.copyfile(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise RunnerError(f"inference evidence copy changed: {destination}")
    return destination


def build_inference_manifest(
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    arm_contract: Mapping[str, Any],
    episode: Mapping[str, Any],
    truth_by_id: Mapping[str, Mapping[str, Any]],
    frames: list[dict[str, Any]],
    shuffled_sources: Mapping[str, str],
    truth_root: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    episode_id = episode["episode_id"]
    common_root = output_dir / "inference" / "common" / episode_id
    video_source = local_file(truth_root, episode["video_path"], where=f"{episode_id}.video_path")
    ledger_source = local_file(truth_root, episode["capture_frame_ledger_path"], where=f"{episode_id}.ledger_path")
    video_copy = copy_evidence(video_source, common_root / ("video" + video_source.suffix))
    ledger_copy = copy_evidence(ledger_source, common_root / "capture-frame-ledger.json")
    policy = arm_contract["route_input_policy"]
    route_path: Path | None = None
    route_sha: str | None = None
    route_source_id: str | None = None
    if policy == "episode_explicit_causal_route_v1":
        route_source = local_file(truth_root, episode["route_intent_path"], where=f"{episode_id}.route_intent_path")
        route_path = copy_evidence(route_source, common_root / "explicit-route-intent.json")
        route_sha = sha256_file(route_path)
        route_source_id = episode_id
    elif policy == "within_heldout_session_sorted_episode_cyclic_shift_one_v1":
        route_source_id = shuffled_sources[episode_id]
        source_episode = truth_by_id[route_source_id]
        route_source = local_file(
            truth_root,
            source_episode["route_intent_path"],
            where=f"{episode_id}.shuffled_route_intent_path",
        )
        route_path = copy_evidence(
            route_source,
            output_dir / "inference" / "routes" / arm_contract["arm_id"] / episode_id / "shuffled-route-intent.json",
        )
        route_sha = sha256_file(route_path)
    elif policy == "uniform_full_frame_equal_weight_v1":
        route_path = output_dir / "inference" / "routes" / arm_contract["arm_id"] / episode_id / "uniform-route.json"
        write_json(route_path, {
            "schema": "blindassist_ustrf_sc_u0_uniform_route_control_v1",
            "contract_id": contract["contract_id"],
            "episode_id": episode_id,
            "field_definition": "full_frame_equal_weight",
            "constant_weight": 1.0,
            "uses_episode_route": False,
            "uses_labels": False,
            "future_inputs_used": False,
        })
        route_sha = sha256_file(route_path)
    elif policy != "no_route_input_v1":
        raise RunnerError(f"unknown route input policy: {policy}")
    manifest = {
        "schema": evidence["sanitized_inference_manifest_schema"],
        "contract_id": contract["contract_id"],
        "arm_id": arm_contract["arm_id"],
        "episode_id": episode_id,
        "input_video_path": relative_to(video_copy, output_dir),
        "input_video_sha256": episode["video_sha256"],
        "capture_frame_ledger_path": relative_to(ledger_copy, output_dir),
        "capture_frame_ledger_sha256": episode["capture_frame_ledger_sha256"],
        "truth_route_intent_sha256": episode["route_intent_sha256"],
        "route_input_policy": policy,
        "adapter_route_input_path": relative_to(route_path, output_dir) if route_path else None,
        "adapter_route_input_sha256": route_sha,
        "adapter_route_source_episode_id": route_source_id,
        "decision_cadence": evidence["decision_cadence"],
        "frames": frames,
        "blind_accessed": False,
        "future_inputs_used": False,
        "review_fields_present": False,
        "adjudication_fields_present": False,
        "event_label_fields_present": False,
    }
    manifest_path = output_dir / "inference" / "manifests" / arm_contract["arm_id"] / f"{episode_id}.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def run_bundle(
    *,
    contract_path: Path,
    truth_config_path: Path,
    truth_manifest_path: Path,
    registry_path: Path,
    output_dir: Path,
    synthetic_fixture: bool,
) -> Path:
    contract_path = contract_path.resolve()
    truth_config_path = truth_config_path.resolve()
    truth_manifest_path = truth_manifest_path.resolve()
    registry_path = registry_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RunnerError(f"output directory already exists: {output_dir}")
    contract = load_json(contract_path, where="U0 contract")
    evidence = validate_contract(contract)
    runner_sha = normalized_text_sha256(Path(__file__).resolve())
    if runner_sha != evidence.get("runner_implementation_sha256"):
        raise RunnerError("runner implementation differs from the frozen U0 contract")
    truth_config = load_json(truth_config_path, where="truth config")
    truth_manifest = load_json(truth_manifest_path, where="truth manifest")
    if truth_manifest.get("contract_id") != truth_config.get("contract_id"):
        raise RunnerError("truth config/manifest contract identity mismatch")
    truth_root = truth_manifest_path.parent.resolve()
    truth_manifest_sha = canonical_json_sha256(truth_manifest_path)
    registry = load_json(registry_path, where="adapter registry")
    registry_rows, kernel_source, kernel_dependencies, registry_snapshot = validate_registry(
        registry,
        contract=contract,
        evidence=evidence,
        registry_root=registry_path.parent.resolve(),
        truth_manifest=truth_manifest,
        truth_manifest_sha256=truth_manifest_sha,
        synthetic_fixture=synthetic_fixture,
    )
    try:
        ordered_ledger_sha, truth_frames = admission._truth_ledgers(truth_manifest, truth_root=truth_root)
    except admission.ContractError as error:
        raise RunnerError(str(error)) from error
    for episode_id, frames in truth_frames.items():
        try:
            admission._validate_decision_cadence(frames, evidence, where=f"truth/{episode_id}")
        except admission.ContractError as error:
            raise RunnerError(str(error)) from error
    episodes = truth_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RunnerError("truth manifest must contain at least one episode")
    truth_by_id: dict[str, dict[str, Any]] = {}
    for raw_episode in episodes:
        if not isinstance(raw_episode, dict):
            raise RunnerError("truth manifest episode must be an object")
        episode_id = require_text(raw_episode.get("episode_id"), where="truth episode_id")
        if episode_id in truth_by_id:
            raise RunnerError(f"duplicate truth episode: {episode_id}")
        validate_truth_file(truth_root, raw_episode, "video", where=episode_id)
        validate_truth_file(truth_root, raw_episode, "route_intent", where=episode_id)
        truth_by_id[episode_id] = raw_episode

    shuffled_sources = shuffled_route_sources(truth_manifest)

    output_dir.mkdir(parents=True, exist_ok=False)
    runner_copy = output_dir / "runner" / Path(__file__).name
    runner_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), runner_copy)
    registry_copy = output_dir / "runner" / "adapter-registry.json"
    write_json(registry_copy, registry_snapshot)
    kernel_copy = output_dir / "runner" / kernel_source.name
    shutil.copyfile(kernel_source, kernel_copy)
    kernel_dependency_rows: list[dict[str, str]] = []
    for name, source in sorted(kernel_dependencies.items()):
        destination = output_dir / "runner" / "kernel-dependencies" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        expected_sha = evidence["shared_decision_kernel_dependency_sha256"][name]
        if normalized_text_sha256(destination) != expected_sha:
            raise RunnerError(f"kernel dependency evidence copy changed: {name}")
        kernel_dependency_rows.append({
            "name": name,
            "path": relative_to(destination, output_dir),
            "file_sha256": sha256_file(destination),
            "normalized_text_sha256": expected_sha,
        })
    registry_sha = sha256_file(registry_copy)
    if normalized_text_sha256(runner_copy) != runner_sha:
        raise RunnerError("runner evidence copy changed")
    if normalized_text_sha256(kernel_copy) != evidence["shared_decision_kernel_implementation_sha256"]:
        raise RunnerError("kernel evidence copy changed")

    predictions: dict[str, Any] = {
        "schema": contract["prediction_schema"],
        "contract_id": contract["contract_id"],
        "truth_config_sha256": canonical_json_sha256(truth_config_path),
        "truth_manifest_sha256": truth_manifest_sha,
        "runner_implementation_path": relative_to(runner_copy, output_dir),
        "runner_implementation_file_sha256": sha256_file(runner_copy),
        "runner_implementation_sha256": runner_sha,
        "adapter_registry_path": relative_to(registry_copy, output_dir),
        "adapter_registry_sha256": registry_sha,
        "shared_decision_kernel_implementation_path": relative_to(kernel_copy, output_dir),
        "shared_decision_kernel_implementation_file_sha256": sha256_file(kernel_copy),
        "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
        "shared_decision_kernel_dependencies": kernel_dependency_rows,
        "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
        "synthetic_fixture": synthetic_fixture,
        "arms": [],
    }

    required_arms = {row["arm_id"]: row for row in contract["required_arms"]}
    for arm_id in required_arms:
        registration = registry_rows[arm_id]
        arm_contract = required_arms[arm_id]
        adapter_id = arm_contract["candidate_adapter_id"]
        arm_root = output_dir / "arms" / arm_id
        copied: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for stem in ("implementation", "threshold_config"):
            source = registration[stem]
            destination = arm_root / "inputs" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise RunnerError(f"{arm_id} {stem} evidence copy changed")
            copied[stem] = destination
            hashes[stem] = sha256_file(destination)
        fold_inventory_rows: list[dict[str, Any]] = []
        materialized_folds: dict[str, dict[str, Any]] = {}
        for held_out, sources in sorted(registration["folds"].items()):
            fold_root = arm_root / "folds" / held_out
            fold_material: dict[str, Any] = {
                "fit_policy": arm_contract["fit_policy"],
                "event_identity_policy": arm_contract["event_identity_policy"],
                "route_input_policy": arm_contract["route_input_policy"],
            }
            inventory_row: dict[str, Any] = {"held_out_session_id": held_out}
            for stem in ("artifact", "training_input_manifest", "training_receipt"):
                source = sources[stem]
                destination = fold_root / source.name
                copy_evidence(source, destination)
                fold_material[stem] = destination
                fold_material[f"{stem}_sha256"] = sha256_file(destination)
                inventory_row[f"{stem}_path"] = relative_to(destination, output_dir)
                inventory_row[f"{stem}_sha256"] = sha256_file(destination)
            materialized_folds[held_out] = fold_material
            fold_inventory_rows.append(inventory_row)
        artifact_inventory = {
            "schema": evidence["fold_artifact_inventory_schema"],
            "contract_id": contract["contract_id"],
            "arm_id": arm_id,
            "candidate_adapter_id": adapter_id,
            "fit_policy": arm_contract["fit_policy"],
            "truth_manifest_sha256": truth_manifest_sha,
            "folds": fold_inventory_rows,
        }
        artifact_inventory_path = arm_root / "fold-artifact-inventory.json"
        write_json(artifact_inventory_path, artifact_inventory)
        copied["artifact"] = artifact_inventory_path
        hashes["artifact"] = sha256_file(artifact_inventory_path)
        arm: dict[str, Any] = {
            **arm_contract,
            "implementation_path": relative_to(copied["implementation"], output_dir),
            "implementation_sha256": hashes["implementation"],
            "artifact_path": relative_to(copied["artifact"], output_dir),
            "artifact_sha256": hashes["artifact"],
            "threshold_config_path": relative_to(copied["threshold_config"], output_dir),
            "threshold_config_sha256": hashes["threshold_config"],
            "ordered_frame_ledger_sha256": ordered_ledger_sha,
            "shared_decision_kernel_contract_id": evidence["shared_decision_kernel_contract_id"],
            "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
            "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
            "adapter_runtime_id": RUNTIME_ID,
            "episodes": [],
        }
        trace_hashes: dict[str, str] = {}
        request_hashes: dict[str, str] = {}
        output_hashes: dict[str, str] = {}
        inference_hashes: dict[str, str] = {}
        fold_artifact_hashes: dict[str, str] = {}
        fold_training_receipt_hashes: dict[str, str] = {}
        exit_codes: dict[str, int] = {}
        durations_ms: dict[str, int] = {}
        for episode_id in sorted(truth_by_id):
            episode = truth_by_id[episode_id]
            fold = materialized_folds[episode["session_id"]]
            episode_root = arm_root / "episodes" / episode_id
            inference_manifest_path, inference_manifest = build_inference_manifest(
                contract=contract,
                evidence=evidence,
                arm_contract=arm_contract,
                episode=episode,
                truth_by_id=truth_by_id,
                frames=truth_frames[episode_id],
                shuffled_sources=shuffled_sources,
                truth_root=truth_root,
                output_dir=output_dir,
            )
            inference_manifest_sha = sha256_file(inference_manifest_path)
            request = expected_request(
                contract=contract,
                evidence=evidence,
                arm_id=arm_id,
                adapter_id=adapter_id,
                hashes=hashes,
                fold=fold,
                episode=episode,
                inference_manifest=inference_manifest,
                inference_manifest_sha256=inference_manifest_sha,
                frames=truth_frames[episode_id],
                synthetic_fixture=synthetic_fixture,
            )
            request_path = episode_root / "adapter-request.json"
            output_path = episode_root / "adapter-output.json"
            stdout_path = episode_root / "adapter-stdout.txt"
            stderr_path = episode_root / "adapter-stderr.txt"
            write_json(request_path, request)
            request_sha = sha256_file(request_path)
            require_unchanged(
                copied["implementation"], hashes["implementation"], where=f"{arm_id} copied implementation"
            )
            require_unchanged(
                copied["threshold_config"], hashes["threshold_config"], where=f"{arm_id} copied threshold config"
            )
            started = time.monotonic_ns()
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(copied["implementation"]),
                        "--request", str(request_path),
                        "--inference-manifest", str(inference_manifest_path),
                        "--inference-root", str(output_dir),
                        "--artifact", str(fold["artifact"]),
                        "--threshold-config", str(copied["threshold_config"]),
                        "--output", str(output_path),
                    ],
                    cwd=episode_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=int(evidence["maximum_episode_execution_seconds"]),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_text(error.stdout or "", encoding="utf-8", newline="\n")
                stderr_path.write_text(error.stderr or "", encoding="utf-8", newline="\n")
                raise RunnerError(f"{arm_id}/{episode_id} adapter timed out") from error
            duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
            stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
            stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
            if completed.returncode != 0:
                raise RunnerError(f"{arm_id}/{episode_id} adapter exited {completed.returncode}")
            if not output_path.is_file():
                raise RunnerError(f"{arm_id}/{episode_id} adapter produced no output")
            require_unchanged(
                copied["implementation"], hashes["implementation"], where=f"{arm_id} copied implementation"
            )
            require_unchanged(
                copied["threshold_config"], hashes["threshold_config"], where=f"{arm_id} copied threshold config"
            )
            raw_output = load_json(output_path, where=f"{arm_id}/{episode_id} adapter output")
            threshold_config = load_json(copied["threshold_config"], where=f"{arm_id} threshold config")
            frames, alerts, abstained = validate_adapter_output(
                raw_output,
                request=request,
                request_sha256=request_sha,
                threshold_config=threshold_config,
                evidence=evidence,
            )
            output_sha = sha256_file(output_path)
            trace = {
                "schema": evidence["episode_trace_schema"],
                "contract_id": contract["contract_id"],
                "arm_id": arm_id,
                "episode_id": episode_id,
                "candidate_adapter_id": adapter_id,
                "adapter_runtime_id": RUNTIME_ID,
                "runner_implementation_sha256": runner_sha,
                "adapter_registry_sha256": registry_sha,
                "adapter_request_sha256": request_sha,
                "adapter_output_sha256": output_sha,
                "sanitized_inference_manifest_sha256": inference_manifest_sha,
                "shared_decision_kernel_contract_id": evidence["shared_decision_kernel_contract_id"],
                "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
                "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
                "decision_profile_id": evidence["decision_profile_id"],
                "feedback_adapter_id": evidence["feedback_adapter_id"],
                "decision_cadence": evidence["decision_cadence"],
                "fit_policy": arm_contract["fit_policy"],
                "event_identity_policy": arm_contract["event_identity_policy"],
                "route_input_policy": arm_contract["route_input_policy"],
                "fold_held_out_session_id": episode["session_id"],
                "fold_artifact_sha256": fold["artifact_sha256"],
                "fold_training_receipt_sha256": fold["training_receipt_sha256"],
                "input_video_sha256": episode["video_sha256"],
                "truth_route_intent_sha256": episode["route_intent_sha256"],
                "adapter_route_input_sha256": inference_manifest["adapter_route_input_sha256"],
                "adapter_route_source_episode_id": inference_manifest["adapter_route_source_episode_id"],
                "source_capture_frame_ledger_sha256": episode["capture_frame_ledger_sha256"],
                "abstained": abstained,
                "kernel_trace_order": evidence["kernel_trace_order"],
                "frames": frames,
            }
            trace_path = episode_root / "decision-trace.json"
            write_json(trace_path, trace)
            trace_sha = sha256_file(trace_path)
            request_hashes[episode_id] = request_sha
            output_hashes[episode_id] = output_sha
            inference_hashes[episode_id] = inference_manifest_sha
            fold_artifact_hashes[episode_id] = fold["artifact_sha256"]
            fold_training_receipt_hashes[episode_id] = fold["training_receipt_sha256"]
            trace_hashes[episode_id] = trace_sha
            exit_codes[episode_id] = completed.returncode
            durations_ms[episode_id] = int(duration_ms)
            arm["episodes"].append({
                "episode_id": episode_id,
                "fold_held_out_session_id": episode["session_id"],
                "input_video_sha256": episode["video_sha256"],
                "source_route_intent_sha256": episode["route_intent_sha256"],
                "source_capture_frame_ledger_sha256": episode["capture_frame_ledger_sha256"],
                "frame_ids_sha256": admission._canonical_sha256([frame["frame_id"] for frame in truth_frames[episode_id]]),
                "fold_artifact_sha256": fold["artifact_sha256"],
                "fold_training_receipt_sha256": fold["training_receipt_sha256"],
                "sanitized_inference_manifest_path": relative_to(inference_manifest_path, output_dir),
                "sanitized_inference_manifest_sha256": inference_manifest_sha,
                "adapter_request_path": relative_to(request_path, output_dir),
                "adapter_request_sha256": request_sha,
                "adapter_output_path": relative_to(output_path, output_dir),
                "adapter_output_sha256": output_sha,
                "adapter_stdout_path": relative_to(stdout_path, output_dir),
                "adapter_stdout_sha256": sha256_file(stdout_path),
                "adapter_stderr_path": relative_to(stderr_path, output_dir),
                "adapter_stderr_sha256": sha256_file(stderr_path),
                "adapter_exit_code": completed.returncode,
                "adapter_duration_ms": int(duration_ms),
                "prediction_trace_path": relative_to(trace_path, output_dir),
                "prediction_trace_sha256": trace_sha,
                "abstained": abstained,
                "alert_timestamps_ms": alerts,
            })
        receipt = {
            "schema": evidence["execution_receipt_schema"],
            "contract_id": contract["contract_id"],
            "arm_id": arm_id,
            "candidate_adapter_id": adapter_id,
            "adapter_runtime_id": RUNTIME_ID,
            "runner_implementation_sha256": runner_sha,
            "adapter_registry_sha256": registry_sha,
            "shared_decision_kernel_contract_id": evidence["shared_decision_kernel_contract_id"],
            "shared_decision_kernel_implementation_sha256": evidence["shared_decision_kernel_implementation_sha256"],
            "kernel_execution_backend_id": evidence["kernel_execution_backend_id"],
            "decision_profile_id": evidence["decision_profile_id"],
            "feedback_adapter_id": evidence["feedback_adapter_id"],
            "decision_cadence": evidence["decision_cadence"],
            "fit_policy": arm_contract["fit_policy"],
            "event_identity_policy": arm_contract["event_identity_policy"],
            "route_input_policy": arm_contract["route_input_policy"],
            "implementation_sha256": hashes["implementation"],
            "artifact_sha256": hashes["artifact"],
            "threshold_config_sha256": hashes["threshold_config"],
            "ordered_frame_ledger_sha256": ordered_ledger_sha,
            "synthetic_fixture": synthetic_fixture,
            "blind_accessed": False,
            "future_inputs_used": False,
            "execution_completed": True,
            "failure_count": 0,
            "adapter_request_sha256_by_episode": request_hashes,
            "adapter_output_sha256_by_episode": output_hashes,
            "sanitized_inference_manifest_sha256_by_episode": inference_hashes,
            "fold_artifact_sha256_by_episode": fold_artifact_hashes,
            "fold_training_receipt_sha256_by_episode": fold_training_receipt_hashes,
            "adapter_exit_code_by_episode": exit_codes,
            "adapter_duration_ms_by_episode": durations_ms,
            "prediction_trace_sha256_by_episode": trace_hashes,
        }
        receipt_path = arm_root / "execution-receipt.json"
        write_json(receipt_path, receipt)
        arm["execution_receipt_path"] = relative_to(receipt_path, output_dir)
        arm["execution_receipt_sha256"] = sha256_file(receipt_path)
        predictions["arms"].append(arm)

    predictions_path = output_dir / "predictions.json"
    write_json(predictions_path, predictions)
    try:
        admission.validate_bundle(
            contract,
            truth_manifest,
            predictions,
            truth_root=truth_root,
            prediction_root=output_dir,
        )
    except admission.ContractError as error:
        predictions_path.unlink(missing_ok=True)
        raise RunnerError(f"generated bundle failed admission: {error}") from error
    return predictions_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--truth-config", required=True, type=Path)
    parser.add_argument("--truth-manifest", required=True, type=Path)
    parser.add_argument("--adapter-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    try:
        predictions_path = run_bundle(
            contract_path=args.contract,
            truth_config_path=args.truth_config,
            truth_manifest_path=args.truth_manifest,
            registry_path=args.adapter_registry,
            output_dir=args.output_dir,
            synthetic_fixture=args.synthetic_fixture,
        )
    except RunnerError as error:
        parser.error(str(error))
    print(json.dumps({
        "status": "complete",
        "predictions_path": str(predictions_path),
        "u0_authority_granted": False,
        "training_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
