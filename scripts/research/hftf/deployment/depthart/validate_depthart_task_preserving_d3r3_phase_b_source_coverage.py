#!/usr/bin/env python3
"""Independently replay the D3R3 Phase-B member-coverage census."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.hftf.deployment.depthart.census_depthart_task_preserving_d3r3_phase_b_source_coverage import (  # noqa: E402
    ASSETS,
    ACTIVATION_FALSE_AUTHORITY,
    ACTIVATION_EXECUTION_POLICY,
    ACTIVATION_FORBIDDEN,
    ACTIVATION_NEXT_ACTION,
    ACTIVATION_TOP_LEVEL_FIELDS,
    ACTIVATION_TRUE_AUTHORITY,
    SOURCE_SCOPE_SCHEMA,
    SOURCE_SCOPE_AUTHORITY_FIELDS,
    TRANSIENT_TRANSPORT_ERROR_TYPES,
    head_lookup,
    load_json,
    require,
    request_plan,
    selection_rows,
    same_path,
    sha256_file,
    verify_file_entry,
    write_json_exclusive,
)


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_census_protocol_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_census_activation_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_asset_checkpoint_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_manifest_v1"
VALIDATION_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_validation_v1"
PASS_TERMINAL = "D3R3_PHASE_B_EXACT64_MEMBER_COVERAGE_CENSUS_COMPLETE_NO_MEMBER_PAYLOAD_OR_TRUTH_READ"
PASS_NEXT_GATE = "EXPLICIT_D3R3_PHASE_B_MISSING_SOURCE_POLICY_REGISTRATION"
PROTOCOL_PATH = REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_PROTOCOL_2026-08-13.json"


def selected_stem_sha256(stems: list[str]) -> str:
    return hashlib.sha256(("\n".join(stems) + "\n").encode("ascii")).hexdigest().upper()


def load_sealed(path: Path, schema: str) -> dict[str, Any]:
    require(path.is_file(), f"sealed JSON missing: {path}")
    payload = path.read_bytes()
    seal_path = path.with_suffix(".sha256.json")
    require(seal_path.is_file(), f"seal missing: {seal_path}")
    seal = load_json(seal_path)
    require(seal == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }, f"seal mismatch: {path}")
    value = json.loads(payload.decode("utf-8"))
    require(value.get("schema") == schema, f"schema drift: {path}")
    return value


def independent_archive_coverage(path: Path, selected_stems: list[str]) -> dict[str, Any]:
    """Replay directory path safety, uniqueness and coverage without producer code."""

    mapping: dict[str, str] = {}
    seen_names: set[str] = set()
    file_count = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename.replace("\\", "/"))
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {info.filename}")
            require(info.filename not in seen_names, f"duplicate ZIP member name: {info.filename}")
            seen_names.add(info.filename)
            if info.is_dir():
                continue
            file_count += 1
            if pure.suffix.lower() != ".png":
                continue
            require(pure.stem not in mapping, f"duplicate ZIP frame stem: {pure.stem}")
            mapping[pure.stem] = info.filename
    require(mapping, f"no PNG members: {path}")
    selected_set = set(selected_stems)
    present = [stem for stem in selected_stems if stem in mapping]
    missing = [stem for stem in selected_stems if stem not in mapping]
    return {
        "archive_file_count": file_count,
        "png_frame_member_count": len(mapping),
        "selected_present_count": len(present),
        "selected_missing_count": len(missing),
        "selected_missing_stems": missing,
        "selected_extra_member_count": len(set(mapping) - selected_set),
        "selected_present_stems_sha256": selected_stem_sha256(present),
        "archive_member_payload_bytes_read": 0,
        "zip_crc_verified": False,
    }


def independent_identity_coverage(identity: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    stems = [str(value) for value in identity["selected_frame_stems"]]
    require(len(stems) == 300 and len(set(stems)) == 300, "selected frame plan drift")
    depth = independent_archive_coverage(source_dir / "lowres_depth.zip", stems)
    confidence = independent_archive_coverage(source_dir / "confidence.zip", stems)
    depth_missing = set(depth["selected_missing_stems"])
    confidence_missing = set(confidence["selected_missing_stems"])
    paired_present = [stem for stem in stems if stem not in depth_missing and stem not in confidence_missing]
    paired_missing = [stem for stem in stems if stem not in paired_present]
    return {
        "selected_frame_count": 300,
        "selected_frame_plan_sha256": selected_stem_sha256(stems),
        "lowres_depth": depth,
        "confidence": confidence,
        "paired_exact_present_count": len(paired_present),
        "paired_exact_missing_count": len(paired_missing),
        "paired_exact_missing_stems": paired_missing,
        "paired_exact_present_stems_sha256": selected_stem_sha256(paired_present),
        "neighbor_substitution_used": False,
        "source_truth_derived": False,
    }


def validate_activation(activation: dict[str, Any], protocol_path: Path) -> None:
    require(set(activation) == set(ACTIVATION_TOP_LEVEL_FIELDS), "activation top-level field set drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(activation.get("status") == "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_CENSUS_ACTIVATED", "activation status drift")
    require(isinstance(activation.get("authorization_verbatim"), str) and bool(activation["authorization_verbatim"].strip()), "activation authorization missing")
    for key in ("activation_id", "activated_at", "authorization_context"):
        require(isinstance(activation.get(key), str) and bool(activation[key].strip()), f"activation {key} missing")
    require(activation.get("activated_by") == "user", "activation author drift")
    require(activation.get("protocol_sha256") == sha256_file(protocol_path), "activation protocol SHA drift")
    protocol_entry = activation["protocol"]
    require(set(protocol_entry) == {"path", "bytes", "sha256"}, "activation protocol binding field set drift")
    bound_protocol_path = Path(str(protocol_entry["path"]))
    if not bound_protocol_path.is_absolute():
        bound_protocol_path = REPO_ROOT / bound_protocol_path
    require(same_path(bound_protocol_path, protocol_path), "activation protocol path drift")
    require(int(protocol_entry["bytes"]) == protocol_path.stat().st_size, "activation protocol byte drift")
    require(protocol_entry["sha256"] == sha256_file(protocol_path), "activation protocol binding SHA drift")
    require(activation["execution_policy"] == ACTIVATION_EXECUTION_POLICY, "activation execution policy drift")
    require(activation["forbidden"] == list(ACTIVATION_FORBIDDEN), "activation forbidden list drift")
    require(activation["next_action"] == ACTIVATION_NEXT_ACTION, "activation next action drift")
    authority = activation["authority"]
    require(set(authority) == set(ACTIVATION_TRUE_AUTHORITY + ACTIVATION_FALSE_AUTHORITY + ("r2_access",)), "activation authority field set drift")
    for key in ACTIVATION_TRUE_AUTHORITY:
        require(authority.get(key) is True, f"activation missing authority: {key}")
    for key in ACTIVATION_FALSE_AUTHORITY:
        require(authority.get(key) is False, f"activation authority widened: {key}")
    require(authority.get("r2_access") == "NONE", "activation R2 authority widened")


def validate_protocol_activation_contract(protocol: dict[str, Any]) -> None:
    contract = protocol["activation_contract"]
    require(contract["current_activation_exists"] is False, "protocol activation state drift")
    require(contract["required_activation_schema"] == ACTIVATION_SCHEMA, "protocol activation schema drift")
    require(contract["required_status"] == "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_CENSUS_ACTIVATED", "protocol activation status drift")
    require(contract["required_top_level_fields"] == list(ACTIVATION_TOP_LEVEL_FIELDS), "protocol activation field contract drift")
    require(contract["required_execution_policy"] == ACTIVATION_EXECUTION_POLICY, "protocol execution policy drift")
    require(contract["must_authorize_true"] == list(ACTIVATION_TRUE_AUTHORITY), "protocol true-authority contract drift")
    require(contract["must_keep_false"] == list(ACTIVATION_FALSE_AUTHORITY), "protocol false-authority contract drift")
    require(contract["r2_access"] == "NONE", "protocol R2 activation contract drift")
    require(contract["required_forbidden"] == list(ACTIVATION_FORBIDDEN), "protocol forbidden contract drift")
    require(contract["required_next_action"] == ACTIVATION_NEXT_ACTION, "protocol activation next-action drift")


def expected_attempt_receipt(
    protocol: dict[str, Any], protocol_path: Path, activation_path: Path,
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen = protocol["frozen_files"]
    return {
        "schema": ATTEMPT_SCHEMA,
        "bindings": {
            "protocol_sha256": sha256_file(protocol_path),
            "activation_sha256": sha256_file(activation_path),
            "source_scope_sha256": frozen["source_scope"]["sha256"],
            "phase_a_result_sha256": frozen["phase_a_result"]["sha256"],
            "phase_a_manifest_sha256": frozen["phase_a_manifest"]["sha256"],
            "head_validation_sha256": frozen["head_validation"]["sha256"],
            "head_machine_result_sha256": frozen["head_machine_result"]["sha256"],
            "producer_sha256": frozen["producer"]["sha256"],
        },
        "output_root": str((REPO_ROOT / protocol["output_root"]).resolve()),
        "identity_plan": [
            {key: row[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")}
            | {"selected_frame_plan_sha256": selected_stem_sha256(row["selected_frame_stems"])}
            for row in selected
        ],
        "asset_plan": [
            {key: row[key] for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset")}
            for row in request_plan(selected)
        ],
        "policy": {
            "assets": list(ASSETS), "identity_count": 32, "frames_per_identity": 300,
            "total_frame_count": 9600, "all_32_processed": True, "pixel_decode": False,
            "source_truth": False, "selection": False, "neighbor_substitution": False,
            "max_attempts": 3, "range_get": False, "redirect_following": False,
            "source_archives_retained": True,
        },
        "declared_download_bytes": int(protocol["declared_download_bytes"]),
        "scientific_terminal": None,
        "selection_evaluated": False,
        "authority": "D3R3_PHASE_B_SOURCE_TRANSPORT_CONTAINER_AND_MEMBER_NAME_COVERAGE_ONLY",
        "rgb_access": False,
        "model_output_access": False,
        "role_assignment": False,
        "training": False,
        "development_outcome": False,
        "r2_access": "NONE",
    }


def validate_transport_receipt(receipt: dict[str, Any], planned: dict[str, Any], head: dict[str, Any]) -> None:
    expected_keys = {
        "asset", "url", "bytes", "sha256", "path", "attempts", "attempt_history",
        "response_http_status", "response_final_url", "response_content_length_bytes",
        "response_etag", "response_last_modified", "frozen_head_content_length_bytes",
        "frozen_head_etag", "frozen_head_last_modified", "range_request_used", "redirect_followed",
        "transport_response_body_bytes_read_total",
    }
    require(set(receipt) == expected_keys, "source receipt field set drift")
    require(receipt["asset"] == planned["asset"] and receipt["url"] == head["url"], "source receipt plan drift")
    attempts = int(receipt["attempts"])
    history = receipt["attempt_history"]
    require(1 <= attempts <= 3 and isinstance(history, list) and len(history) == attempts, "source retry count drift")
    for index, event in enumerate(history, start=1):
        require(set(event) == {
            "attempt", "method", "http_status", "error", "error_type", "retry_class",
            "expected_body_bytes", "response_body_bytes_read",
        }, "source attempt field set drift")
        require(event["attempt"] == index and event["method"] == "GET", "source attempt order/method drift")
        require(event["expected_body_bytes"] == int(head["content_length_bytes"]), "attempt expected bytes drift")
        if index < attempts:
            require(event["error"] and event["error_type"], "successful attempt cannot be retried")
            require(event["retry_class"] in {"TRANSIENT_HTTP", "TRANSIENT_TRANSPORT", "TRANSIENT_BODY_SHORT_READ"}, "terminal attempt was retried")
            status = event["http_status"]
            if event["retry_class"] == "TRANSIENT_HTTP":
                require(status in {408, 429} or (isinstance(status, int) and 500 <= status <= 599), "HTTP retry class drift")
                require(event["error_type"] == "HTTPError", "HTTP retry error type drift")
            elif event["retry_class"] == "TRANSIENT_TRANSPORT":
                require(status in {None, 200}, "transport retry status drift")
                require(event["error_type"] in TRANSIENT_TRANSPORT_ERROR_TYPES, "transport retry error type drift")
            else:
                require(status == 200 and event["error_type"] == "BodyShortRead", "short-read evidence drift")
                require(event["response_body_bytes_read"] < event["expected_body_bytes"], "short-read byte drift")
    final = history[-1]
    require(final["http_status"] == 200 and final["error"] is None and final["error_type"] is None and final["retry_class"] is None, "source final attempt is not successful")
    require(final["response_body_bytes_read"] == int(head["content_length_bytes"]), "final body byte drift")
    require(receipt["response_http_status"] == 200, "source response status drift")
    require(receipt["response_final_url"] == receipt["url"] == head["url"], "source final URL drift")
    require(int(receipt["bytes"]) == int(receipt["response_content_length_bytes"]) == int(head["content_length_bytes"]), "source response length drift")
    require(receipt["response_etag"] == receipt["frozen_head_etag"] == head["etag"], "source response ETag drift")
    require(receipt["response_last_modified"] == receipt["frozen_head_last_modified"] == head["last_modified"], "source response Last-Modified drift")
    require(int(receipt["frozen_head_content_length_bytes"]) == int(head["content_length_bytes"]), "source frozen HEAD length drift")
    require(receipt["range_request_used"] is False and receipt["redirect_followed"] is False, "source transport boundary drift")
    require(receipt["transport_response_body_bytes_read_total"] == sum(int(event["response_body_bytes_read"]) for event in history), "transport total byte drift")


def comparable_asset_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        key: checkpoint[key]
        for key in (
            "archive_file_count",
            "png_frame_member_count",
            "selected_present_count",
            "selected_missing_count",
            "selected_missing_stems",
            "selected_extra_member_count",
            "selected_present_stems_sha256",
            "archive_member_payload_bytes_read",
            "zip_crc_verified",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-validation", type=Path, required=True)
    parser.add_argument("--head-machine-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(same_path(args.protocol, PROTOCOL_PATH), "protocol path drift")
    protocol = load_json(args.protocol)
    activation = load_json(args.activation)
    source_scope = load_json(args.source_scope)
    phase_a_result = load_json(args.phase_a_result)
    phase_a_manifest = load_json(args.phase_a_manifest)
    head_machine = load_json(args.head_machine_result)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    validate_protocol_activation_contract(protocol)
    require(source_scope.get("schema") == SOURCE_SCOPE_SCHEMA, "source scope schema drift")
    current_authority = source_scope["current_authority"]
    require(set(current_authority) == set(SOURCE_SCOPE_AUTHORITY_FIELDS), "source scope authority field set drift")
    require(
        current_authority["source_scope_registration"] is True
        and current_authority["protocol_design"] is True
        and current_authority["synthetic_tests"] is True,
        "source scope design authority drift",
    )
    for key in set(SOURCE_SCOPE_AUTHORITY_FIELDS) - {"source_scope_registration", "protocol_design", "synthetic_tests", "r2_access"}:
        require(current_authority[key] is False, f"source scope authority widened: {key}")
    require(current_authority["r2_access"] == "NONE", "source scope R2 authority widened")
    for label, entry in protocol["frozen_files"].items():
        verify_file_entry(entry, label)
    verify_file_entry(protocol["runtime"]["launcher"], "runtime launcher")
    python_path = verify_file_entry(protocol["runtime"]["python_executable"], "Python executable")
    require(same_path(Path(sys.executable), python_path), "Python executable drift")
    require(sys.version.split()[0] == protocol["runtime"]["python_version"], "Python version drift")
    for label, path in {
        "source_scope": args.source_scope,
        "phase_a_result": args.phase_a_result,
        "phase_a_manifest": args.phase_a_manifest,
        "head_validation": args.head_validation,
        "head_machine_result": args.head_machine_result,
    }.items():
        require(same_path(path, verify_file_entry(protocol["frozen_files"][label], label)), f"passed {label} path drift")
    validate_activation(activation, args.protocol)
    activation_labels = (
        "source_scope", "phase_a_result", "phase_a_manifest", "head_validation",
        "head_machine_result", "d3r2_execution_stop",
    )
    require(set(activation["bindings"]) == set(activation_labels), "activation binding field set drift")
    for label in activation_labels:
        require(activation["bindings"][label] == protocol["frozen_files"][label], f"activation binding drift: {label}")
    require(activation["request_scope"] == {
        "identity_count": 32, "asset_count": 64, "assets": list(ASSETS),
        "selected_frame_count": 9600,
    }, "activation request scope drift")
    root = REPO_ROOT / protocol["output_root"]
    require(same_path(args.manifest, root / "manifest.json"), "manifest path drift")
    require(same_path(args.output, root / "validation.json"), "validation output path drift")
    require(not args.output.exists(), "validation overwrite forbidden")
    manifest = load_sealed(args.manifest, MANIFEST_SCHEMA)
    attempt = load_json(root / "attempt.json")
    require(attempt.get("schema") == ATTEMPT_SCHEMA, "attempt schema drift")
    require(manifest.get("terminal") == PASS_TERMINAL, "manifest terminal drift")
    selected = selection_rows(source_scope, phase_a_result, phase_a_manifest)
    heads = head_lookup(head_machine, selected)
    require(len(selected) == 32 and len(heads) == 64, "frozen plan drift")
    require(
        attempt == expected_attempt_receipt(protocol, args.protocol, args.activation, selected),
        "attempt receipt drift",
    )
    attempt_sha = hashlib.sha256((json.dumps(attempt, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest().upper()

    allowed_root = {"attempt.json", "receipts", "source", "manifest.json", "manifest.sha256.json"}
    require({path.name for path in root.iterdir()} == allowed_root, "completed root inventory drift")
    source_parent = root / "source"
    require({path.name for path in source_parent.iterdir()} == {"Training"}, "source fold inventory drift")
    source_root = source_parent / "Training"
    require({path.name for path in source_root.iterdir()} == {str(row["video_id"]) for row in selected}, "source identity inventory drift")
    receipts_root = root / "receipts"
    expected_receipt_names: set[str] = set()
    asset_replays: dict[tuple[str, str], dict[str, Any]] = {}
    downloaded_bytes = 0
    transport_body_bytes = 0
    for planned in request_plan(selected):
        asset_token = str(planned["asset"]).replace(".zip", "")
        name = f"{int(planned['request_order']):03d}-{planned['video_id']}-{asset_token}.json"
        expected_receipt_names |= {name, name.replace(".json", ".sha256.json")}
        checkpoint = load_sealed(receipts_root / name, CHECKPOINT_SCHEMA)
        require(set(checkpoint) == {
            "schema", "attempt_sha256", "request_order", "selection_order", "pool_order",
            "visit_id", "video_id", "fold", "asset", "selected_frame_plan_sha256",
            "source_asset", "archive_file_count", "png_frame_member_count",
            "selected_present_count", "selected_missing_count", "selected_missing_stems",
            "selected_extra_member_count", "selected_present_stems_sha256",
            "archive_member_payload_bytes_read", "zip_crc_verified",
            "transport_response_body_bytes_read", "scientific_terminal", "pixel_decode",
            "source_truth", "selection_evaluated", "range_get_used", "redirect_followed",
            "role_assigned", "training", "development_outcome_read", "r2_access",
        }, "checkpoint field set drift")
        require(checkpoint["attempt_sha256"] == attempt_sha, "checkpoint attempt binding drift")
        require(checkpoint["selected_frame_plan_sha256"] == selected_stem_sha256(planned["selected_frame_stems"]), "checkpoint frame-plan binding drift")
        for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset"):
            require(checkpoint[key] == planned[key], f"checkpoint plan drift: {key}")
        receipt = checkpoint["source_asset"]
        head = heads[(str(planned["video_id"]), str(planned["asset"]))]
        validate_transport_receipt(receipt, planned, head)
        source_dir = source_root / str(planned["video_id"])
        require({path.name for path in source_dir.iterdir()} == set(ASSETS), "source archive inventory drift")
        path = source_dir / str(planned["asset"])
        require(same_path(Path(receipt["path"]), path), "source receipt path drift")
        require(path.stat().st_size == int(receipt["bytes"]) == int(head["content_length_bytes"]), "source length drift")
        require(sha256_file(path) == receipt["sha256"], "source SHA drift")
        require(receipt["response_etag"] == head["etag"], "source ETag drift")
        require(receipt["response_last_modified"] == head["last_modified"], "source Last-Modified drift")
        downloaded_bytes += int(receipt["bytes"])
        transport_body_bytes += int(receipt["transport_response_body_bytes_read_total"])
        replay = independent_archive_coverage(path, planned["selected_frame_stems"])
        require(comparable_asset_checkpoint(checkpoint) == replay, f"independent coverage mismatch: {planned['video_id']}/{planned['asset']}")
        require(
            checkpoint["transport_response_body_bytes_read"]
            == int(receipt["transport_response_body_bytes_read_total"]),
            "checkpoint transport-body count drift",
        )
        require(checkpoint["scientific_terminal"] is None, "checkpoint scientific terminal drift")
        for key in ("pixel_decode", "source_truth", "selection_evaluated", "range_get_used", "redirect_followed", "role_assigned", "training", "development_outcome_read"):
            require(checkpoint[key] is False, f"checkpoint authority widened: {key}")
        require(checkpoint["r2_access"] == "NONE", "checkpoint R2 authority widened")
        key = str(planned["video_id"]), str(planned["asset"])
        require(key not in asset_replays, "duplicate asset replay")
        asset_replays[key] = replay
    require({path.name for path in receipts_root.iterdir()} == expected_receipt_names, "receipt inventory drift")

    replayed = [independent_identity_coverage(identity, source_root / str(identity["video_id"])) for identity in selected]
    require(len(asset_replays) == 64, "asset replay count drift")
    paired_present = sum(int(row["paired_exact_present_count"]) for row in replayed)
    paired_missing = sum(int(row["paired_exact_missing_count"]) for row in replayed)
    identities_missing = sum(int(row["paired_exact_missing_count"] > 0) for row in replayed)
    require(paired_present + paired_missing == 9600, "paired coverage denominator drift")
    require(set(manifest) == {
        "schema", "terminal", "scientific_terminal", "bindings", "attempt_sha256",
        "declared_download_bytes", "downloaded_body_bytes", "transport_response_body_bytes_read",
        "archive_container_bytes_sha256_hashed", "asset_checkpoint_count", "identity_count",
        "processed_identity_count", "selected_frame_count", "paired_exact_present_frame_count",
        "paired_exact_missing_frame_count", "identities_with_any_paired_missing", "processed",
        "source_archives_retained_for_offline_validation", "archive_member_payload_bytes_read",
        "zip_directory_parsed_count", "zip_crc_verified", "pixel_decode",
        "source_truth_derived", "truth_support_gate_evaluated", "selection_evaluated",
        "phase_b_selection_locked", "selected_phase_b", "rgb_read", "model_output_read",
        "role_assignment_made", "training", "development_outcome_read", "r2_access",
        "performance_claim", "android_default_authority", "production_authority",
        "safety_authority", "next_gate",
    }, "manifest field set drift")
    require(manifest["asset_checkpoint_count"] == 64, "manifest asset-checkpoint count drift")
    require(manifest["bindings"] == attempt["bindings"] and manifest["attempt_sha256"] == attempt_sha, "manifest binding drift")
    require(manifest["identity_count"] == 32, "manifest identity count drift")
    require(manifest["processed_identity_count"] == 32 and manifest["selected_frame_count"] == 9600, "manifest count drift")
    require(int(manifest["declared_download_bytes"]) == int(protocol["declared_download_bytes"]), "manifest declared-byte drift")
    require(manifest["downloaded_body_bytes"] == downloaded_bytes == int(protocol["declared_download_bytes"]), "manifest body-byte drift")
    require(
        manifest["transport_response_body_bytes_read"] == transport_body_bytes,
        "manifest transport-body count drift",
    )
    require(manifest["archive_container_bytes_sha256_hashed"] == downloaded_bytes, "manifest container-hash count drift")
    require(manifest["paired_exact_present_frame_count"] == paired_present, "manifest paired-present drift")
    require(manifest["paired_exact_missing_frame_count"] == paired_missing, "manifest paired-missing drift")
    require(manifest["identities_with_any_paired_missing"] == identities_missing, "manifest missing-identity drift")
    require(manifest["processed"] == [
        {key: identity[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")} | replay
        for identity, replay in zip(selected, replayed, strict=True)
    ], "manifest processed coverage drift")
    require(manifest["zip_directory_parsed_count"] == 64, "manifest ZIP-directory count drift")
    require(manifest["source_archives_retained_for_offline_validation"] is True, "manifest source-retention drift")
    require(manifest["archive_member_payload_bytes_read"] == 0 and manifest["zip_crc_verified"] is False, "manifest member-payload/CRC boundary drift")
    require(manifest["truth_support_gate_evaluated"] is False and manifest["phase_b_selection_locked"] is False, "manifest science boundary drift")
    require(manifest["scientific_terminal"] is None and manifest["selection_evaluated"] is False, "manifest scientific boundary drift")
    require(manifest["selected_phase_b"] is None and manifest["next_gate"] == PASS_NEXT_GATE, "manifest successor drift")
    for key in ("pixel_decode", "source_truth_derived", "rgb_read", "model_output_read", "role_assignment_made", "training", "development_outcome_read", "performance_claim", "android_default_authority", "production_authority", "safety_authority"):
        require(manifest[key] is False, f"manifest authority widened: {key}")
    require(manifest["r2_access"] == "NONE", "manifest R2 authority widened")

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_VALIDATION_PASS",
        "protocol_sha256": sha256_file(args.protocol),
        "activation_sha256": sha256_file(args.activation),
        "manifest_sha256": sha256_file(args.manifest),
        "processed_identity_count": 32,
        "source_asset_count": 64,
        "transport_response_body_bytes_read": transport_body_bytes,
        "archive_container_bytes_sha256_hashed": downloaded_bytes,
        "selected_frame_count": 9600,
        "paired_exact_present_frame_count": paired_present,
        "paired_exact_missing_frame_count": paired_missing,
        "identities_with_any_paired_missing": identities_missing,
        "all_zip_directories_replayed": True,
        "zip_crc_verified": False,
        "archive_member_payload_bytes_read": 0,
        "all_member_names_replayed": True,
        "pixel_decode": False,
        "source_truth_derived": False,
        "truth_support_gate_evaluated": False,
        "selection_evaluated": False,
        "scientific_terminal": None,
        "selected_phase_b": None,
        "rgb_read": False,
        "model_output_read": False,
        "role_assignment_made": False,
        "training": False,
        "development_outcome_read": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "android_default_authority": False,
        "production_authority": False,
        "safety_authority": False,
        "next_gate": PASS_NEXT_GATE,
        "authority": "OFFLINE_SOURCE_CONTAINER_AND_MEMBER_COVERAGE_VALIDATION_ONLY",
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
