from __future__ import annotations

import json
import gzip
import hashlib
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import exploratory_profiles_r2_l1 as r1


PREREG_SCHEMA = "blindassist_ustrf_route_target_r2_l1x_l2p_prereg_r1"
OVERALL_SCHEMA = "blindassist_ustrf_route_target_r2_l1x_l2p_terminal_receipt_r1"
MECHANISM_AUDIT_SCHEMA = (
    "blindassist_ustrf_route_target_r2_l1x_l2p_mechanism_gap_audit_r1"
)
PROGRESS_SCHEMA = "blindassist_ustrf_route_target_r2_l1x_l2p_progress_receipt_r1"
LEGAL_TERMINAL_STATES = {
    "L2_FRESH_SELECTION_PREREG_READY",
    "FAIL_CLOSED_INPUT_BLOCKED",
    "FAIL_CLOSED_EXECUTION_ABORTED",
}
READY_STATE = "L2_FRESH_SELECTION_PREREG_READY"
FORBIDDEN_AUTHORITY_TRUE = {
    "new_replay_data",
    "candidate_selection",
    "candidate_ranking",
    "candidate_recommendation",
    "provisional_selection_output",
    "android_shadow",
    "l3_execution",
    "h2",
    "human_outcome",
    "independent_walking_safety",
    "production",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bound_file(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = (repo / binding["path"]).resolve()
    if not path.is_file():
        raise r1.InputBlocked(f"{label}_missing:{binding['path']}")
    actual = r1.sha256_file(path)
    if actual != binding["sha256"]:
        raise r1.InputBlocked(
            f"{label}_sha256_mismatch:expected={binding['sha256']}:actual={actual}"
        )
    return path


def verify_baseline_commit(repo: Path, expected: str) -> None:
    process = subprocess.run(
        ["git", "cat-file", "-e", f"{expected}^{{commit}}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise r1.InputBlocked(f"baseline_commit_missing:{expected}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, head],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise r1.InputBlocked(
            f"baseline_commit_not_ancestor:baseline={expected}:head={head}"
        )


def verify_parent_r1_preservation(
    repo: Path, prereg: dict[str, Any]
) -> dict[str, Any]:
    parents = prereg["immutable_r1_parent"]
    verified: dict[str, Any] = {}
    for name, binding in parents["bindings"].items():
        path = verify_bound_file(repo, binding, f"immutable_r1_{name}")
        verified[name] = {
            "path": str(path.relative_to(repo)).replace("\\", "/"),
            "sha256": r1.sha256_file(path),
        }
    terminal = load_json(repo / parents["bindings"]["terminal_receipt"]["path"])
    if terminal.get("terminal_state") != "FAIL_CLOSED_EXECUTION_ABORTED":
        raise r1.InputBlocked("r1_terminal_state_rewritten")
    execution = terminal.get("candidate_execution", {})
    if execution.get("started") is not False:
        raise r1.InputBlocked("r1_candidate_execution_history_rewritten")
    if execution.get("authoritative_trace_count") != 0:
        raise r1.InputBlocked("r1_trace_history_rewritten")
    if terminal.get("profiles") != []:
        raise r1.InputBlocked("r1_profile_history_rewritten")
    validation = load_json(repo / parents["bindings"]["validation_receipt"]["path"])
    if validation.get("status") != "VALID":
        raise r1.InputBlocked("r1_validation_receipt_not_valid")
    guard = load_json(repo / parents["bindings"]["resource_guard_attempts"]["path"])
    attempts = guard.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 3
        or guard.get("retry_limit_exhausted") is not True
        or guard.get("automatic_retry_allowed_after_receipt") is not False
    ):
        raise r1.InputBlocked("r1_three_attempt_history_not_preserved")
    if any(row.get("device_attempt_created") for row in attempts):
        raise r1.InputBlocked("r1_guard_history_claim_drift")
    return {
        "terminal_state": terminal["terminal_state"],
        "candidate_execution_started": False,
        "authoritative_trace_count": 0,
        "profile_count": 0,
        "guard_attempt_count": 3,
        "old_attempts_count_toward_r2": False,
        "verified_bindings": verified,
    }


def load_and_verify_prereg(
    repo: Path, prereg_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_path = prereg_path.resolve()
    prereg = load_json(prereg_path)
    if prereg.get("schema") != PREREG_SCHEMA:
        raise r1.InputBlocked("unexpected_r2_l1x_l2p_prereg_schema")
    if prereg.get("stage") != "R2-L1X-L2P":
        raise r1.InputBlocked("unexpected_r2_l1x_l2p_stage")
    if set(prereg.get("terminal_states", [])) != LEGAL_TERMINAL_STATES:
        raise r1.InputBlocked("r2_l1x_l2p_terminal_state_contract_drift")
    authority = prereg.get("authority", {})
    if any(authority.get(key) is not False for key in FORBIDDEN_AUTHORITY_TRUE):
        raise r1.InputBlocked("r2_l1x_l2p_authority_opened")
    if prereg.get("freeze_order", {}).get(
        "l2_and_l3_contracts_validated_before_any_new_candidate_output"
    ) is not True:
        raise r1.InputBlocked("l2_l3_not_frozen_before_candidate_output")
    recovery = prereg["execution_recovery"]
    if recovery.get("stage") != "R2-L1E-R2":
        raise r1.InputBlocked("recovery_stage_drift")
    if recovery.get("parent_r1_attempts_count_toward_this_stage") is not False:
        raise r1.InputBlocked("old_r1_attempts_reused")
    if recovery.get("attempt_namespace") != "r2-l1e-r2":
        raise r1.InputBlocked("recovery_attempt_namespace_drift")
    if recovery.get("initial_attempts") != 1 or recovery.get("bounded_retries") != 2:
        raise r1.InputBlocked("recovery_attempt_budget_drift")
    guards = recovery["resource_guards"]
    if guards.get("minimum_system_available_physical_memory_bytes") != 6442450944:
        raise r1.InputBlocked("six_gib_memory_guard_drift")
    if guards.get("host_maximum_rss_bytes") != 8589934592:
        raise r1.InputBlocked("host_rss_guard_drift")
    if guards.get("reserve_bytes") != 5368709120:
        raise r1.InputBlocked("disk_reserve_guard_drift")
    if prereg["l1_profile"]["candidate_order"] != [
        "C1_CAUSAL_ROUTE_RELATION_FSM",
        "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
        "C3_DUAL_KEY_CLEARANCE_FSM",
    ]:
        raise r1.InputBlocked("candidate_order_drift")
    if prereg["l1_profile"].get("runs_per_candidate") != 1:
        raise r1.InputBlocked("candidate_single_run_contract_drift")
    verify_baseline_commit(repo, prereg["baseline_commit"])
    parent_summary = verify_parent_r1_preservation(repo, prereg)
    base_config_path = verify_bound_file(
        repo, prereg["immutable_r1_parent"]["bindings"]["config"], "r1_config"
    )
    base_config, parent_bindings = r1.load_and_verify_config(repo, base_config_path)
    base_guards = {
        key: value
        for key, value in base_config["resource_guards"].items()
        if key != "output_root"
    }
    recovery_guards = {
        key: value for key, value in guards.items() if key != "output_root"
    }
    if base_guards != recovery_guards:
        raise r1.InputBlocked("recovery_resource_guards_not_identical_to_r1")
    if guards.get("output_root") != recovery["output_root"]:
        raise r1.InputBlocked("recovery_guard_output_namespace_drift")
    for name in ("l2_prereg", "l3_lockbox_template", "l2_l3_validator"):
        verify_bound_file(repo, prereg["preoutput_frozen_contracts"][name], name)
    bindings: dict[str, Any] = {
        **parent_bindings,
        "parent_r1_config_sha256": parent_bindings["config_sha256"],
        "r2_l1x_l2p_prereg_sha256": r1.sha256_file(prereg_path),
    }
    bindings["config_sha256"] = bindings["r2_l1x_l2p_prereg_sha256"]
    for name, binding in prereg["implementation_bindings"].items():
        path = verify_bound_file(repo, binding, name)
        bindings[f"{name}_sha256"] = r1.sha256_file(path)
    return prereg, {
        "base_config": base_config,
        "bindings": bindings,
        "parent_summary": parent_summary,
    }


def validate_preoutput_freeze(repo: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    from validate_l2_l3_prereg_r1 import validate_contracts

    result = validate_contracts(
        repo,
        load_json(repo / prereg["preoutput_frozen_contracts"]["l2_prereg"]["path"]),
        load_json(
            repo
            / prereg["preoutput_frozen_contracts"]["l3_lockbox_template"]["path"]
        ),
    )
    if result.get("decision") != "VALID_L2_L3_PREREG_R1":
        raise r1.ExecutionAborted("l2_l3_preoutput_freeze_not_valid")
    return result


def build_context(
    repo: Path, prereg_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg, verified = load_and_verify_prereg(repo, prereg_path)
    freeze = validate_preoutput_freeze(repo, prereg)
    config = verified["base_config"]
    mask = r1.load_json(repo / config["parent_bindings"]["eligibility_mask"]["path"])
    groups, resets = r1.validate_mask_contract(config, mask)
    route_map = r1.load_route_map(config, repo)
    if len(groups) != 41 or sum(len(rows) for _, rows in groups) != 62229:
        raise r1.InputBlocked("parent_mask_coverage_drift")
    if len(resets) != 15:
        raise r1.InputBlocked("parent_discontinuity_reset_count_drift")
    return prereg, {
        **verified,
        "preoutput_freeze": freeze,
        "mask": mask,
        "groups": groups,
        "resets": resets,
        "route_map": route_map,
    }


def output_root(repo: Path, prereg: dict[str, Any]) -> Path:
    root = (repo / prereg["execution_recovery"]["output_root"]).resolve()
    try:
        root.relative_to((repo / "artifacts.local").resolve())
    except ValueError as error:
        raise r1.ExecutionAborted("recovery_output_outside_artifacts_local") from error
    return root


def verified_scope(
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    root: Path,
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    gaps = r1.gap_matrix(groups, root, route_map)
    ledgers = sum(not row["missing_fields"] for row in gaps)
    frames = sum(
        row["expected_frame_count"] for row in gaps if not row["missing_fields"]
    )
    return gaps, ledgers, frames


def record_resource_guard_failure_r2(
    path: Path,
    prereg: dict[str, Any],
    bindings: dict[str, Any],
    observed_available_bytes: int,
    verified_ledgers: int,
    verified_frames: int,
) -> dict[str, Any]:
    maximum_attempts = int(prereg["execution_recovery"]["initial_attempts"]) + int(
        prereg["execution_recovery"]["bounded_retries"]
    )
    required = int(
        prereg["execution_recovery"]["resource_guards"][
            "minimum_system_available_physical_memory_bytes"
        ]
    )
    if path.exists():
        receipt = load_json(path)
        if receipt.get("automatic_retry_allowed_after_receipt") is False:
            raise r1.ExecutionAborted("r2_resource_guard_retry_limit_already_exhausted")
        if receipt.get("prereg_sha256") != bindings["r2_l1x_l2p_prereg_sha256"]:
            raise r1.ExecutionAborted("r2_resource_guard_prereg_binding_drift")
    else:
        receipt = {
            "schema": "blindassist_ustrf_route_target_l1e_r2_resource_guard_attempts_r1",
            "stage": "R2-L1E-R2",
            "attempt_namespace": prereg["execution_recovery"]["attempt_namespace"],
            "prereg_sha256": bindings["r2_l1x_l2p_prereg_sha256"],
            "parent_r1_attempts_count_toward_this_stage": False,
            "parent_r1_resource_guard_receipt_sha256": prereg[
                "immutable_r1_parent"
            ]["bindings"]["resource_guard_attempts"]["sha256"],
            "guard": "system_available_physical_memory_bytes",
            "required_minimum_bytes": required,
            "maximum_attempts": maximum_attempts,
            "attempts": [],
            "device_attempt_created": False,
            "canonical_raw_shard_created": False,
            "candidate_execution_started": False,
            "candidate_trace_created": False,
            "profile_authority": False,
            "automatic_retry_allowed_after_receipt": True,
            "retry_limit_exhausted": False,
        }
    if len(receipt["attempts"]) >= maximum_attempts:
        raise r1.ExecutionAborted("r2_resource_guard_retry_limit_already_exhausted")
    number = len(receipt["attempts"]) + 1
    receipt["attempts"].append(
        {
            "attempt_number": number,
            "attempt_id": (
                f"r2-l1e-r2-host-memory-pre-device-{number:03d}-"
                f"{uuid.uuid4().hex}"
            ),
            "observation_time_utc": datetime.now(timezone.utc).isoformat(),
            "observed_available_bytes": observed_available_bytes,
            "required_available_bytes": required,
            "device_attempt_created": False,
            "last_safe_checkpoint": {
                "verified_sequence_ledgers": verified_ledgers,
                "verified_frames": verified_frames,
                "candidate_execution_started": False,
            },
            "outcome": "STOPPED_BEFORE_DEVICE_ATTEMPT",
        }
    )
    exhausted = len(receipt["attempts"]) == maximum_attempts
    receipt["automatic_retry_allowed_after_receipt"] = not exhausted
    receipt["retry_limit_exhausted"] = exhausted
    r1.atomic_write_json(path, receipt)
    return receipt


def materialize_one_crowdbot_r2(
    prereg: dict[str, Any],
    config: dict[str, Any],
    bindings: dict[str, Any],
    repo: Path,
    descriptor: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    ledger_path, successor_path = r1.compact_paths(
        root, descriptor["source_id"], descriptor["sequence_id"]
    )
    if r1.validate_compact_ledger(
        ledger_path, successor_path, descriptor, mask_rows
    ):
        return
    guards = config["resource_guards"]
    projected_bytes = len(mask_rows) * r1.RAW_BYTES_PER_FRAME
    required_free = 2 * projected_bytes + int(guards["reserve_bytes"])
    observed_free = r1.disk_free_bytes(root)
    if observed_free < required_free:
        raise r1.ExecutionAborted(
            f"free_space_guard:observed={observed_free}:required={required_free}"
        )
    observed_memory_preflight = r1.available_memory_bytes()
    required_memory = int(guards["minimum_system_available_physical_memory_bytes"])
    if observed_memory_preflight < required_memory:
        gaps, verified_ledgers, verified_frames = verified_scope(
            __import__("exploratory_profiles_r2_l1").grouped_mask(
                r1.load_json(
                    repo
                    / config["parent_bindings"]["eligibility_mask"]["path"]
                )
            ),
            root,
            r1.load_route_map(config, repo),
        )
        del gaps
        record_resource_guard_failure_r2(
            root / "resource-guard-attempts-r2.json",
            prereg,
            bindings,
            observed_memory_preflight,
            verified_ledgers,
            verified_frames,
        )
        raise r1.ExecutionAborted(
            "available_memory_guard:"
            f"observed={observed_memory_preflight}:required={required_memory}"
        )
    bundle_path, image_rows = r1.load_crowdbot_images(
        repo, config, descriptor, mask_rows
    )
    observed_memory_after_load = r1.available_memory_bytes()
    if observed_memory_after_load < required_memory:
        groups = r1.grouped_mask(
            r1.load_json(repo / config["parent_bindings"]["eligibility_mask"]["path"])
        )
        _, verified_ledgers, verified_frames = verified_scope(
            groups, root, r1.load_route_map(config, repo)
        )
        record_resource_guard_failure_r2(
            root / "resource-guard-attempts-r2.json",
            prereg,
            bindings,
            observed_memory_after_load,
            verified_ledgers,
            verified_frames,
        )
        raise r1.ExecutionAborted(
            "available_memory_guard_after_bundle_load:"
            f"observed={observed_memory_after_load}:required={required_memory}"
        )
    slug = r1.stable_slug(descriptor["source_id"], descriptor["sequence_id"])
    attempt_root = root / "attempts" / prereg["execution_recovery"][
        "attempt_namespace"
    ] / slug
    existing_attempts = (
        sorted(attempt_root.glob("attempt-*")) if attempt_root.exists() else []
    )
    maximum_attempts = int(prereg["execution_recovery"]["initial_attempts"]) + int(
        prereg["execution_recovery"]["bounded_retries"]
    )
    if len(existing_attempts) >= maximum_attempts:
        raise r1.ExecutionAborted(f"retry_limit_exhausted:{slug}")
    attempt_number = len(existing_attempts) + 1
    attempt_dir = attempt_root / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = attempt_dir / "device-manifest.json"
    raw_path = attempt_dir / "canonical-raw.gz"
    device_receipt_path = attempt_dir / "device-raw-receipt.json"
    manifest = {
        "schema": r1.DEVICE_MANIFEST_SCHEMA,
        "stage": "R2-L1E-R2",
        "parent_compatible_stage": "R2-L1E",
        "attempt_namespace": prereg["execution_recovery"]["attempt_namespace"],
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "frame_count": len(mask_rows),
        "input_shape": config["input_contract"]["detector"]["input_shape"],
        "output_shape": config["input_contract"]["detector"]["output_shape"],
        "model_sha256": config["input_contract"]["detector"]["model_sha256"],
        "labels_sha256": config["input_contract"]["detector"]["labels_sha256"],
        "person_class_index": 0,
        "confidence_threshold": 0.35,
        "nms_iou_threshold": 0.45,
        "bundle_sha256": r1.sha256_file(bundle_path),
        "projected_raw_bytes": projected_bytes,
        "free_space_guard": {
            "observed_bytes": observed_free,
            "required_bytes": required_free,
            "formula": guards["minimum_free_space_formula"],
        },
        "memory_guard": {
            "observed_available_bytes_preflight": observed_memory_preflight,
            "observed_available_bytes_after_bundle_load": observed_memory_after_load,
            "required_available_bytes": required_memory,
            "maximum_host_rss_bytes": guards["host_maximum_rss_bytes"],
        },
        "frames": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in image_rows
        ],
        "authority": {
            "candidate_input_only": True,
            "selection": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "production": False,
        },
    }
    r1.atomic_write_json(manifest_path, manifest)
    adb = r1.locate_adb(repo)
    remote_relative = f"r2l1e-r2/{slug}/attempt-{attempt_number:03d}"
    remote_absolute = (
        "/sdcard/Android/data/com.linnan.blindassist/files/" + remote_relative
    )
    r1.bounded_remote_cleanup(
        adb, remote_relative, int(guards["cleanup_retry_count"])
    )
    r1.run_command(
        [str(adb), "shell", "mkdir", "-p", f"{remote_absolute}/images"],
        timeout=120,
    )
    r1.run_command(
        [str(adb), "push", str(manifest_path), f"{remote_absolute}/manifest.json"],
        timeout=300,
    )
    image_directory = Path(image_rows[0]["_host_image_path"]).parent
    r1.run_command(
        [
            str(adb),
            "push",
            f"{image_directory}{os.sep}.",
            f"{remote_absolute}/images/",
        ],
        timeout=1800,
    )
    instrumentation = [
        str(adb),
        "shell",
        "am",
        "instrument",
        "-w",
        "-r",
        "-e",
        "class",
        "com.linnan.blindassist.benchmark.UstrfR2L1ExploratoryProfileR2DeviceTest",
        "-e",
        "ustrfR2L1eR2Required",
        "true",
        "-e",
        "ustrfR2L1eR2Input",
        f"{remote_relative}/manifest.json",
        "-e",
        "ustrfR2L1eR2RawOutput",
        f"{remote_relative}/canonical-raw.gz",
        "-e",
        "ustrfR2L1eR2ReceiptOutput",
        f"{remote_relative}/device-raw-receipt.json",
        "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner",
    ]
    started = time.monotonic()
    process = r1.run_command(instrumentation, timeout=3600)
    (attempt_dir / "instrumentation.stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    r1.run_command(
        [
            str(adb),
            "pull",
            f"{remote_absolute}/device-raw-receipt.json",
            str(device_receipt_path),
        ],
        timeout=300,
    )
    r1.run_command(
        [
            str(adb),
            "pull",
            f"{remote_absolute}/canonical-raw.gz",
            str(raw_path),
        ],
        timeout=1800,
    )
    receipt = r1.load_json(device_receipt_path)
    if receipt.get("stage") != "R2-L1E-R2":
        raise r1.ExecutionAborted("unexpected_r2_device_receipt_stage")
    r1.validate_device_receipt(
        receipt, manifest_path, raw_path, descriptor, mask_rows
    )
    labels = r1.detector_labels(config, repo)
    frames = []
    raw_digest = hashlib.sha256()
    with gzip.open(raw_path, "rb") as stream:
        for expected, device_row in zip(mask_rows, receipt["frames"], strict=True):
            raw_bytes = r1.read_exact(
                stream,
                r1.RAW_BYTES_PER_FRAME,
                f"{descriptor['sequence_id']}/{expected['frame_id']}",
            )
            raw_digest.update(raw_bytes)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            if raw_sha != device_row["android_raw_output_sha256"]:
                raise r1.ExecutionAborted("device_raw_record_sha256_mismatch")
            detector_latency = int(device_row["detector_processing_latency_ns"])
            decode_started = time.perf_counter_ns()
            person_detections = r1.decode_raw_record(
                raw_bytes, [640, 480], labels, config
            )
            decode_latency = time.perf_counter_ns() - decode_started
            r1.enforce_host_rss_guard(config)
            frames.append(
                {
                    **expected,
                    "source_size": [640, 480],
                    "android_raw_output_sha256": raw_sha,
                    "detector_processing_latency_ns": detector_latency,
                    "host_decode_latency_ns": decode_latency,
                    "person_detections": person_detections,
                }
            )
        if stream.read(1):
            raise r1.ExecutionAborted("device_raw_stream_has_trailing_records")
    if raw_digest.hexdigest() != receipt["canonical_raw_stream"][
        "uncompressed_sha256"
    ]:
        raise r1.ExecutionAborted("device_raw_uncompressed_sha256_mismatch")
    ledger = {
        "schema": r1.COMPACT_SCHEMA,
        "stage": "R2-L1E",
        "recovery_stage": "R2-L1E-R2",
        "attempt_namespace": prereg["execution_recovery"]["attempt_namespace"],
        "authority": (
            "candidate_input_only_no_selection_android_h2_human_or_production_authority"
        ),
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "frame_count": len(frames),
        "canonical_raw_source": (
            "R2_L1E_R2_sequence_sharded_android_canvas_stream"
        ),
        "canonical_raw_sha256": r1.sha256_file(raw_path),
        "device_receipt_sha256": r1.sha256_file(device_receipt_path),
        "device_manifest_sha256": r1.sha256_file(manifest_path),
        "frames": frames,
    }
    r1.atomic_write_json(ledger_path, ledger)
    successor = {
        "schema": r1.SUCCESSOR_SCHEMA,
        "stage": "R2-L1E",
        "recovery_stage": "R2-L1E-R2",
        "attempt_namespace": prereg["execution_recovery"]["attempt_namespace"],
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "raw_sha256": r1.sha256_file(raw_path),
        "raw_retained_as_parent_evidence": False,
        "device_receipt_sha256": r1.sha256_file(device_receipt_path),
        "device_manifest_sha256": r1.sha256_file(manifest_path),
        "compact_ledger_sha256": r1.sha256_file(ledger_path),
        "frame_count": len(frames),
        "wall_time_seconds": time.monotonic() - started,
        "projected_raw_bytes": projected_bytes,
        "validated": True,
    }
    r1.atomic_write_json(successor_path, successor)
    if not r1.validate_compact_ledger(
        ledger_path, successor_path, descriptor, mask_rows
    ):
        raise r1.ExecutionAborted("compact_successor_validation_failed")
    cleanup_retries = int(guards["cleanup_retry_count"])
    r1.bounded_local_cleanup(raw_path, cleanup_retries)
    r1.bounded_remote_cleanup(adb, remote_relative, cleanup_retries)


def write_progress(
    repo: Path,
    root: Path,
    prereg: dict[str, Any],
    context: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    gaps, ledgers, frames = verified_scope(
        context["groups"], root, context["route_map"]
    )
    receipt = {
        "schema": PROGRESS_SCHEMA,
        "stage": "R2-L1X-L2P",
        "recovery_stage": "R2-L1E-R2",
        "phase": phase,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": context["bindings"]["r2_l1x_l2p_prereg_sha256"],
        "attempt_namespace": prereg["execution_recovery"]["attempt_namespace"],
        "parent_r1_attempts_count_toward_this_stage": False,
        "expected_sequence_ledgers": 41,
        "verified_sequence_ledgers": ledgers,
        "expected_frames": 62229,
        "verified_frames": frames,
        "discontinuity_reset_count": len(context["resets"]),
        "candidate_execution_started": (root / "candidate-traces").exists(),
        "gap_ledger_count": sum(bool(row["missing_fields"]) for row in gaps),
        "last_safe_checkpoint": {
            "verified_sequence_ledgers": ledgers,
            "verified_frames": frames,
        },
    }
    r1.atomic_write_json(root / "progress-receipt-r2.json", receipt)
    return receipt


def _metric_signal(metric_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    if metric_name == "critical_miss":
        return {"violates_point_gate": metric["numerator"] > 0}
    if metric_name == "clearance":
        denominator = metric["denominator"]
        rate = metric["numerator"] / denominator if denominator else None
        return {
            "value": rate,
            "violates_point_gate": rate is None or rate < 0.9,
        }
    if metric_name == "unknown_or_stale_alert":
        return {"violates_point_gate": metric["numerator"] > 0}
    if metric_name == "repeat":
        return {
            "violates_point_gate": metric["numerator"] > 0,
            "denominator_insufficient": metric["denominator"]
            < metric["minimum_denominator"],
        }
    if metric_name == "evidence_age":
        p95_ns = metric["p95_ns"]
        return {
            "violates_point_gate": p95_ns is None or p95_ns > 200_000_000,
            "denominator_incomplete": metric["timestamp_frame_count"]
            != metric["required_timestamp_frame_count"],
        }
    if metric_name in {"event_recall", "regeneration"}:
        return {
            "l0_diagnostic_only": True,
            "denominator_insufficient": metric.get("denominator", 0) == 0,
        }
    if metric_name == "false_alerts_per_minute":
        return {
            "l0_diagnostic_only": True,
            "denominator_insufficient": metric["negative_exposure_ns"]
            < metric["minimum_l1_exposure_ns"],
        }
    raise r1.ExecutionAborted(f"unknown_profile_metric:{metric_name}")


def build_mechanism_gap_audit(
    profiles: list[dict[str, Any]],
    prereg: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    expected = prereg["l1_profile"]["candidate_order"]
    if [row["candidate_id"] for row in profiles] != expected:
        raise r1.ExecutionAborted("mechanism_audit_candidate_order_drift")
    metric_names = list(profiles[0]["metrics"])
    matrix = []
    for metric_name in metric_names:
        by_candidate = []
        for profile in profiles:
            metric = profile["metrics"][metric_name]
            by_candidate.append(
                {
                    "candidate_id": profile["candidate_id"],
                    "level": metric["level"],
                    "eligibility_status": metric["eligibility_status"],
                    "numerator": metric.get("numerator"),
                    "denominator": metric.get("denominator"),
                    "signal": _metric_signal(metric_name, metric),
                }
            )
        violations = [
            bool(row["signal"].get("violates_point_gate")) for row in by_candidate
        ]
        insufficient = [
            bool(
                row["signal"].get("denominator_insufficient")
                or row["signal"].get("denominator_incomplete")
            )
            for row in by_candidate
        ]
        categories = []
        if all(violations):
            categories.append("common_algorithm_performance_signal")
        elif any(violations):
            categories.append("candidate_specific_algorithm_performance_signal")
        if any(insufficient):
            categories.append("statistical_power_or_observability_gap")
        if metric_name in {"clearance", "repeat", "regeneration"}:
            categories.append("event_lifecycle_mechanism")
        if metric_name == "unknown_or_stale_alert":
            categories.append("input_identity_route_validity_mechanism")
        source_support = [
            {
                "candidate_id": profile["candidate_id"],
                "source_contributions": profile["metrics"][metric_name].get(
                    "source_contributions"
                ),
                "provenance_family_contributions": profile["metrics"][
                    metric_name
                ].get("provenance_family_contributions"),
            }
            for profile in profiles
        ]
        matrix.append(
            {
                "metric": metric_name,
                "candidate_signals": by_candidate,
                "common_failure": all(violations),
                "candidate_specific_failure_present": any(violations)
                and not all(violations),
                "denominator_or_power_gap_present": any(insufficient),
                "mechanism_categories": categories,
                "source_family_support": source_support,
                "source_outcome_claim_boundary": (
                    "denominator_support_only_when_profile_lacks_source_specific_numerator;"
                    "pooled_result_cannot_override_evaluable_source_failure"
                ),
            }
        )
    vetoes = []
    for row in matrix:
        if row["metric"] in {"critical_miss", "unknown_or_stale_alert"}:
            for candidate in row["candidate_signals"]:
                if candidate["signal"].get("violates_point_gate"):
                    vetoes.append(
                        {
                            "metric": row["metric"],
                            "candidate_id": candidate["candidate_id"],
                            "hard_veto_observed": True,
                        }
                    )
    return {
        "schema": MECHANISM_AUDIT_SCHEMA,
        "stage": "R2-L1X-L2P",
        "authority": "mechanism_gap_audit_only_no_winner_rank_selection_or_promotion",
        "profile_bindings": [
            {
                "candidate_id": row["candidate_id"],
                "profile_sha256": row["profile_sha256"],
            }
            for row in profiles
        ],
        "metric_gap_matrix": matrix,
        "hard_veto_observations": vetoes,
        "separation": {
            "algorithm_performance": "candidate output against frozen per-metric truth",
            "input_identity_route_truth": "unknown/stale/identity/route validity and source-attribution limits",
            "lifecycle_mechanism": "delivery repeat clear and regeneration closure",
            "statistical_power": "empty incomplete or below-floor denominators remain not_evaluable_or_underpowered",
        },
        "l2_justification": {
            "execution_ready_preregistration_is_warranted": True,
            "current_l1_data_role": "seen_exploratory_only",
            "current_l1_outputs_may_not_select_l2_candidate": True,
            "future_data_role_required": "fresh_selection",
            "no_candidate_selected": True,
            "reason": (
                "L1 may localize shared and candidate-specific mechanism signals, "
                "but only fresh candidate-blind L2 data may exercise the frozen selection contract."
            ),
        },
        "claim_boundary": {
            "winner": False,
            "rank": False,
            "best_candidate": False,
            "promotion": False,
            "pooled_overrides_source_failure": False,
            "not_evaluable_recoded_as_zero": False,
        },
    }


def build_terminal(
    terminal_state: str,
    prereg: dict[str, Any],
    context: dict[str, Any],
    root: Path,
    first_blocker: str | None,
) -> dict[str, Any]:
    if terminal_state not in LEGAL_TERMINAL_STATES:
        raise ValueError(terminal_state)
    gaps, ledgers, frames = verified_scope(
        context["groups"], root, context["route_map"]
    )
    return {
        "schema": OVERALL_SCHEMA,
        "stage": "R2-L1X-L2P",
        "terminal_state": terminal_state,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": context["bindings"],
        "immutable_r1_parent_summary": context["parent_summary"],
        "preoutput_freeze_validation": context["preoutput_freeze"],
        "verified_scope": {
            "expected_sequence_ledgers": 41,
            "verified_sequence_ledgers": ledgers,
            "expected_frames": 62229,
            "verified_frames": frames,
            "expected_discontinuity_resets": 15,
            "verified_discontinuity_resets": len(context["resets"]),
            "first_blocker": first_blocker,
        },
        "gap_matrix": gaps,
        "claim_boundary": {
            "new_replay_data_added": False,
            "candidate_selected": False,
            "candidate_ranked": False,
            "candidate_recommended": False,
            "provisional_selection_produced": False,
            "android_shadow_executed": False,
            "l3_executed": False,
            "h2_opened": False,
            "human_outcome_claimed": False,
            "independent_walking_safety_claimed": False,
            "production_authorized": False,
        },
    }
