#!/usr/bin/env python3
"""Validate candidate-blind camera-native source-prescreen policy and roster."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_bound(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"{label} hash mismatch")
    return path


def validate_policy(repo: Path, policy: dict[str, Any]) -> dict[str, Any]:
    require(
        policy.get("schema") == "blindassist_ustrf_route_target_camera_source_prescreen_policy_r1",
        "camera source prescreen policy schema mismatch",
    )
    base_path = resolve_bound(repo, policy["base_preregistration"], "base preregistration")
    base = load_json(base_path)
    candidate_path = resolve_bound(repo, policy["candidate_implementation"], "candidate implementation")
    require(candidate_path.name == "candidates.py", "candidate implementation path drifted")
    require(policy["candidate_implementation"]["candidate_count"] == 3, "candidate count drifted")
    resolve_bound(repo, policy["truth_implementation"]["fusion"], "truth fusion implementation")
    resolve_bound(repo, policy["truth_implementation"]["window_freeze"], "truth window implementation")
    require(
        policy["truth_implementation"][
            "must_use_same_identity_role_clear_and_negative_window_semantics_as_final_holdout"
        ]
        is True,
        "canary truth semantics are not bound to final holdout",
    )

    boundary = policy["information_boundary"]
    require(boundary["candidate_outputs_executed"] is False, "candidate outputs exposed before prescreen")
    require(
        boundary["app_detector_or_event_outputs_exposed"] is False,
        "app detector or event outputs exposed before prescreen",
    )
    required_forbidden = {
        "rgb_or_depth_decode",
        "visual_person_proposals",
        "candidate_outputs",
        "app_detector_or_event_outputs",
        "manual_camera_content_review",
    }
    require(
        required_forbidden.issubset(set(boundary["forbidden_before_source_specific_canary_roster_freeze"])),
        "prescreen information boundary is incomplete",
    )

    roster = policy["canary_roster"]
    require(roster["canary_count_each_source"] == 2, "canary count must remain two")
    require(
        roster["selected_canaries_permanently_excluded_from_any_future_lockbox"] is True,
        "canaries are not permanently excluded",
    )
    require(
        roster["canary_pass_can_only_authorize_noncanary_source_materialization"] is True,
        "canary pass gained holdout authority",
    )
    require(policy["decisions"]["no_replay_after_failure"] is True, "failed canary replay opened")
    require(
        policy["decisions"]["no_threshold_or_truth_relaxation_after_failure"] is True,
        "post-failure threshold or truth relaxation opened",
    )

    final = policy["final_holdout_gates_unchanged"]
    base_holdout = base["sealed_holdout"]
    for key in (
        "minimum_positive_events_each_source",
        "minimum_critical_events_each_source",
        "minimum_matched_negative_windows_each_source",
        "minimum_scorable_negative_exposure_minutes_each_source",
    ):
        require(final[key] == base_holdout[key], f"final holdout gate drifted: {key}")
    require(final["two_independent_sources_required"] is True, "two-source requirement drifted")
    require(final["per_source_and_worst_source_required"] is True, "worst-source requirement drifted")
    require(policy["android_shadow"] == "closed", "Android shadow opened")
    require(policy["h2_depth_ttc_route_risk_flip"] == "closed", "H2 opened")
    return base


def metric_by_sequence(capacity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(len(capacity["sources"]) == 1, "capacity proxy must contain exactly one source")
    source = capacity["sources"][0]
    rows = {row["sequence_id"]: row for row in source["sequence_metrics"]}
    require(len(rows) == source["sequence_count"], "capacity proxy sequence ids are not unique")
    return source, rows


def inventory_entries(
    repo: Path, bindings: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    entries: dict[str, dict[str, Any]] = {}
    inventory_path_by_sequence: dict[str, str] = {}
    for binding in bindings:
        path = resolve_bound(repo, binding, "raw filename inventory")
        payload = load_json(path)
        require(payload["filename_inventory_decoded"] is True, "raw filename inventory not decoded")
        require(payload["sequence_content_decoded"] is False, "raw sequence content decoded before roster freeze")
        require(payload["archive_payload_bytes_downloaded"] == 0, "raw archive payload downloaded before roster freeze")
        for entry in payload["entries"]:
            if entry["is_directory"] or not entry["name"].endswith(".bag"):
                continue
            sequence_id = Path(entry["name"]).stem
            require(sequence_id not in entries, f"duplicate raw inventory sequence: {sequence_id}")
            entries[sequence_id] = entry
            inventory_path_by_sequence[sequence_id] = binding["path"]
    return entries, inventory_path_by_sequence


def validate_roster(repo: Path, roster: dict[str, Any]) -> dict[str, Any]:
    require(
        roster.get("schema") == "blindassist_ustrf_route_target_camera_source_prescreen_roster_r1",
        "camera source prescreen roster schema mismatch",
    )
    policy_path = resolve_bound(repo, roster["policy"], "camera source prescreen policy")
    policy = load_json(policy_path)
    base = validate_policy(repo, policy)

    capacity_path = resolve_bound(repo, roster["capacity_proxy"], "capacity proxy")
    capacity = load_json(capacity_path)
    require(capacity["candidate_outputs_executed"] is False, "capacity proxy exposed candidate outputs")
    require(capacity["forward_camera_visibility_verified"] is False, "capacity proxy is not metadata-only")
    source, metrics = metric_by_sequence(capacity)
    require(source["source_id"] == roster["source_id"], "capacity proxy source mismatch")
    require(source["sequence_count"] == roster["metadata_sequence_count"], "metadata sequence count drifted")

    entries, inventory_paths = inventory_entries(repo, roster["raw_inventories"])
    inventory_gap = roster["inventory_gap"]
    require(inventory_gap["sequence_id"] in metrics, "declared inventory gap not present in capacity proxy")
    require(inventory_gap["sequence_id"] not in entries, "declared inventory gap has a raw entry")
    require(
        inventory_gap["excluded_from_capacity_available_for_materialization_until_official_inventory_is_found"]
        is True,
        "inventory gap was counted as available",
    )

    eligible = [row for sequence_id, row in metrics.items() if sequence_id in entries]
    require(len(eligible) >= 2, "fewer than two raw-inventory-backed sequences")
    expected_event = max(
        eligible,
        key=lambda row: (
            row["positive_event_capacity_proxy"],
            row["critical_event_capacity_proxy"],
            row["active_route_frame_count"],
            tuple(-ord(char) for char in row["sequence_id"]),
        ),
    )
    remaining = [row for row in eligible if row["sequence_id"] != expected_event["sequence_id"]]
    expected_negative = max(
        remaining,
        key=lambda row: (
            row["negative_route_seconds"] / entries[row["sequence_id"]]["compressed_size"],
            tuple(-ord(char) for char in row["sequence_id"]),
        ),
    )

    selected = roster["selected_canaries"]
    require(len(selected) == 2, "source roster must contain exactly two canaries")
    by_purpose = {row["purpose"]: row for row in selected}
    require(set(by_purpose) == {"event_capacity", "negative_information_per_byte"}, "canary purposes drifted")
    expected_by_purpose = {
        "event_capacity": expected_event,
        "negative_information_per_byte": expected_negative,
    }
    for purpose, metric in expected_by_purpose.items():
        selected_row = by_purpose[purpose]
        sequence_id = metric["sequence_id"]
        require(selected_row["sequence_id"] == sequence_id, f"{purpose} canary selection drifted")
        require(
            selected_row["source_sequence_key"] == f"{roster['source_id']}::{sequence_id}",
            f"{purpose} source sequence key drifted",
        )
        for key in (
            "positive_event_capacity_proxy",
            "critical_event_capacity_proxy",
            "active_route_frame_count",
            "negative_route_seconds",
        ):
            require(selected_row[key] == metric[key], f"{purpose} metric binding drifted: {key}")
        entry = entries[sequence_id]
        require(
            selected_row["archive_inventory_path"] == inventory_paths[sequence_id],
            f"{purpose} archive inventory path drifted",
        )
        for key in ("crc32", "compressed_size", "uncompressed_size"):
            require(selected_row[key] == entry[key], f"{purpose} raw entry binding drifted: {key}")
        require(selected_row["entry_name"] == entry["name"], f"{purpose} entry name drifted")

    canary_count = policy["canary_roster"]["canary_count_each_source"]
    final = base["sealed_holdout"]
    scaled = roster["scaled_reject_only_gates"]
    require(scaled["canary_count"] == canary_count, "scaled canary count drifted")
    expected_scaled = {
        "minimum_accepted_positive_events": math.ceil(
            final["minimum_positive_events_each_source"] * canary_count / source["sequence_count"]
        ),
        "minimum_accepted_critical_events": math.ceil(
            final["minimum_critical_events_each_source"] * canary_count / source["sequence_count"]
        ),
        "minimum_matched_negative_windows": math.ceil(
            final["minimum_matched_negative_windows_each_source"] * canary_count / source["sequence_count"]
        ),
        "minimum_scorable_negative_exposure_minutes": (
            final["minimum_scorable_negative_exposure_minutes_each_source"]
            * canary_count
            / source["sequence_count"]
        ),
    }
    for key, expected in expected_scaled.items():
        require(math.isclose(scaled[key], expected), f"scaled reject-only gate drifted: {key}")

    selected_ids = {row["sequence_id"] for row in selected}
    residual_rows = [
        row for row in eligible if row["sequence_id"] not in selected_ids
    ]
    residual = roster["residual_metadata_proxy_after_canary_exclusion"]
    require(residual["sequence_count_with_raw_inventory"] == len(residual_rows), "residual sequence count drifted")
    require(
        residual["positive_event_capacity_proxy"]
        == sum(row["positive_event_capacity_proxy"] for row in residual_rows),
        "residual positive capacity drifted",
    )
    require(
        residual["critical_event_capacity_proxy"]
        == sum(row["critical_event_capacity_proxy"] for row in residual_rows),
        "residual critical capacity drifted",
    )
    expected_negative_minutes = sum(row["negative_route_seconds"] for row in residual_rows) / 60.0
    require(
        math.isclose(residual["negative_route_minutes_proxy"], expected_negative_minutes),
        "residual negative capacity drifted",
    )

    expected_exclusions = sorted(row["source_sequence_key"] for row in selected)
    require(
        sorted(roster["permanent_lockbox_exclusions"]) == expected_exclusions,
        "permanent lockbox exclusions drifted",
    )
    transport = roster["transport_and_storage"]
    resolve_bound(repo, transport["canary_runner"], "camera source prescreen canary runner")
    resolve_bound(repo, transport["streamer"], "range streamer")
    require(transport["range_workers"] == 8, "prescreen transport must use frozen eight-way concurrency")
    require(transport["range_parts"] == 64, "prescreen transport part count drifted")
    require(transport["request_timeout_seconds"] == 45, "prescreen transport timeout drifted")
    require(
        transport["logical_root"].startswith("artifacts.local/"),
        "prescreen logical storage is outside artifacts.local",
    )
    require(
        transport["physical_backing_root"].startswith("D:/linnan-artifacts/"),
        "prescreen physical backing root drifted",
    )
    require(
        transport["candidate_outputs_written_to_storage"] is False,
        "candidate outputs written to prescreen storage",
    )
    require(roster["candidate_outputs_executed"] is False, "candidate outputs exposed")
    require(
        roster["app_detector_or_event_outputs_exposed"] is False,
        "app detector or event outputs exposed",
    )
    require(
        roster["camera_content_decoded_before_roster_freeze"] is False,
        "camera content decoded before roster freeze",
    )
    require(roster["canary_result"] == "not_run", "source roster is not a pre-run freeze")
    require(roster["android_shadow"] == "closed", "Android shadow opened")
    require(roster["h2_depth_ttc_route_risk_flip"] == "closed", "H2 opened")

    return {
        "schema": "blindassist_ustrf_route_target_camera_source_prescreen_validation_r1",
        "source_id": roster["source_id"],
        "policy_sha256": sha256_file(policy_path),
        "selected_canaries": expected_exclusions,
        "compressed_bytes_planned": sum(row["compressed_size"] for row in selected),
        "scaled_reject_only_gates": expected_scaled,
        "candidate_outputs_executed": False,
        "decision": "VALID_PREDECODE_REJECT_ONLY_CANARY_ROSTER",
        "android_shadow": "closed",
        "h2_depth_ttc_route_risk_flip": "closed",
    }


def validate_execution(
    repo: Path,
    execution_path: Path,
    freeze_receipt_path: Path,
) -> dict[str, Any]:
    execution = load_json(execution_path)
    require(
        execution.get("schema") == "blindassist_ustrf_route_target_camera_source_prescreen_execution_r1",
        "camera source prescreen execution schema mismatch",
    )
    resolve_bound(repo, execution["base_preregistration"], "execution base preregistration")
    resolve_bound(repo, execution["prescreen_policy"], "execution prescreen policy")
    roster_path = resolve_bound(repo, execution["prescreen_roster"], "execution prescreen roster")
    roster = load_json(roster_path)
    validate_roster(repo, roster)
    require(len(execution["replacement_sources"]) == 1, "prescreen execution must contain one source")
    source = execution["replacement_sources"][0]
    require(source["source_id"] == roster["source_id"], "prescreen execution source drifted")
    require(
        set(source["sequence_ids"]) == {row["sequence_id"] for row in roster["selected_canaries"]},
        "prescreen execution canary coverage drifted",
    )
    require(
        source["permanently_excluded_from_future_lockbox"] is True,
        "prescreen execution canaries gained lockbox eligibility",
    )
    expected_scaled = {
        key: value
        for key, value in roster["scaled_reject_only_gates"].items()
        if key != "canary_count"
    }
    require(
        execution["scaled_reject_only_gates"] == expected_scaled,
        "prescreen execution scaled gates drifted",
    )
    require(
        execution["planned_outputs"]["dataset_root"]
        == f"{roster['transport_and_storage']['logical_root']}/dataset",
        "prescreen execution dataset root drifted",
    )
    for path in execution["planned_outputs"].values():
        require(path.startswith(f"{roster['transport_and_storage']['logical_root']}/"), "planned output escaped prescreen root")
    for key, expected in {
        "camera_content_decoded_before_execution_freeze": False,
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "candidate_or_truth_protocol_change": False,
        "candidate_threshold_nms_tracker_or_scalar_change": False,
        "canary_pass_grants_holdout_admission": False,
        "canary_pass_grants_candidate_execution": False,
    }.items():
        require(execution["execution_boundaries"][key] is expected, f"execution boundary drifted: {key}")
    require(execution["android_shadow"] == "closed", "execution opened Android shadow")
    require(execution["h2_depth_ttc_route_risk_flip"] == "closed", "execution opened H2")

    receipt = load_json(freeze_receipt_path)
    require(
        receipt.get("schema")
        == "blindassist_ustrf_route_target_camera_source_prescreen_execution_freeze_receipt_r1",
        "execution freeze receipt schema mismatch",
    )
    require(
        receipt["execution_config"]["path"]
        == execution_path.resolve().relative_to(repo).as_posix(),
        "execution freeze receipt path drifted",
    )
    require(
        receipt["execution_config"]["sha256"] == sha256_file(execution_path),
        "execution freeze receipt hash drifted",
    )
    transport_state = receipt["transport_state_at_freeze"]
    require(transport_state["complete_bag_count"] == 0, "bag completed before execution freeze")
    require(transport_state["partial_bag_count"] == 0, "bag decode started before execution freeze")
    require(transport_state["bundle_count"] == 0, "camera bundle existed before execution freeze")
    require(
        transport_state["camera_content_decoded_before_freeze"] is False,
        "camera content decoded before execution freeze",
    )
    require(receipt["candidate_outputs_executed"] is False, "candidate outputs exposed before execution freeze")
    require(
        receipt["app_detector_or_event_outputs_exposed"] is False,
        "app outputs exposed before execution freeze",
    )
    return {
        "execution_sha256": sha256_file(execution_path),
        "execution_freeze_receipt_sha256": sha256_file(freeze_receipt_path),
        "decision": "VALID_PREDECODE_CANARY_TRUTH_EXECUTION_FREEZE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--freeze-receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    roster_path = args.roster.resolve()
    roster = load_json(roster_path)
    result = validate_roster(repo, roster)
    result["roster_sha256"] = sha256_file(roster_path)
    if args.execution or args.freeze_receipt:
        require(
            args.execution is not None and args.freeze_receipt is not None,
            "--execution and --freeze-receipt must be provided together",
        )
        result["truth_execution_freeze"] = validate_execution(
            repo,
            args.execution.resolve(),
            args.freeze_receipt.resolve(),
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
