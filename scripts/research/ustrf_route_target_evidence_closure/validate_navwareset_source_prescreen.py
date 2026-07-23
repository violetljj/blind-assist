#!/usr/bin/env python3
"""Validate the predecode, two-stage NavWareSet reject-only prescreen."""

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


def validate(
    repo: Path,
    config_path: Path,
    freeze_receipt_path: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    require(
        config.get("schema") == "blindassist_ustrf_route_target_navwareset_source_prescreen_r1",
        "NavWareSet prescreen schema mismatch",
    )
    policy_path = resolve_bound(repo, config["base_policy"], "camera source prescreen policy")
    policy = load_json(policy_path)
    audit_path = resolve_bound(repo, config["source_audit"], "NavWareSet source audit")
    audit = load_json(audit_path)
    require(audit["source_id"] == config["source_id"], "NavWareSet audit source mismatch")
    require(
        audit["decision"]
        == "BEST_CURRENT_SECOND_SOURCE_CANDIDATE_REQUIRE_TWO_STAGE_REJECT_ONLY_CANARY",
        "NavWareSet audit did not authorize reject-only canary",
    )
    require(config["source_independent_from_crowdbot"] is True, "NavWareSet is not independent")
    resolve_bound(repo, config["stage_a_runner"], "NavWareSet stage A runner")
    amendment_path = resolve_bound(
        repo,
        config["stage_a_transport_binding_amendment"],
        "NavWareSet stage A transport binding amendment",
    )
    amendment = load_json(amendment_path)
    require(
        amendment["authority"]
        == "predecode_transport_integrity_binding_correction_not_source_admission",
        "transport amendment gained source-admission authority",
    )
    original_snapshot = resolve_bound(
        repo,
        amendment["original_prescreen_config_snapshot"],
        "original NavWareSet prescreen snapshot",
    )
    require(
        amendment["corrected_runner"]["sha256"] == config["stage_a_runner"]["sha256"],
        "transport amendment does not bind the corrected runner",
    )
    require(
        amendment["robot_or_grs_bag_content_decoded_before_amendment"] is False
        and amendment["annotation_content_decoded_before_amendment"] is False,
        "NavWareSet payload was decoded before transport binding correction",
    )
    planned_outputs = config["planned_outputs"]
    require(
        planned_outputs["stage_a_root"]
        == "artifacts.local/camera-source-prescreen-r1/navwareset/stage-a",
        "stage A output root drifted",
    )
    require(
        planned_outputs["stage_a_receipt"]
        == (
            "artifacts.local/camera-source-prescreen-r1/navwareset/evidence/"
            "stage-a-micro-canary-receipt-r1.json"
        ),
        "stage A receipt path drifted",
    )

    commit = config["official_tutorial_commit"]
    require(len(commit) == 40 and all(char in "0123456789abcdef" for char in commit), "tutorial commit drifted")
    stage_a = config["stage_a_micro_canary"]
    require(stage_a["scene_id"] == 13, "stage A scene drifted")
    require(
        stage_a["parent_scene_permanently_excluded_from_future_lockbox"] is True,
        "stage A parent scene gained lockbox eligibility",
    )
    files = stage_a["files"]
    require(len(files) == 6, "stage A file roster drifted")
    require(len({row["path"] for row in files}) == len(files), "duplicate stage A file path")
    amendment_probes = {row["path"]: row for row in amendment["transport_binding_probes"]}
    require(
        set(amendment_probes) == {row["path"] for row in files},
        "transport amendment file roster drifted",
    )
    for row in files:
        require(f"/{commit}/" in row["url"], f"stage A URL is not commit-bound: {row['path']}")
        require(row["bytes"] > 0, f"stage A byte count invalid: {row['path']}")
        etag = row["http_etag_sha256"]
        require(len(etag) == 64 and all(char in "0123456789abcdef" for char in etag), f"stage A ETag drifted: {row['path']}")
        content_sha = row["content_sha256"]
        require(
            len(content_sha) == 64 and all(char in "0123456789abcdef" for char in content_sha),
            f"stage A content SHA drifted: {row['path']}",
        )
        require(
            amendment_probes[row["path"]]["bytes"] == row["bytes"]
            and amendment_probes[row["path"]]["content_sha256"] == content_sha,
            f"stage A content binding differs from predecode probe: {row['path']}",
        )
    bag_bytes = sum(row["bytes"] for row in files if row["path"].endswith(".bag"))
    require(bag_bytes == audit["packaging"]["micro_canary_payload_bytes"], "stage A bag byte total drifted")
    require(f"/{commit}?" in stage_a["official_tree_api_url"], "stage A tree API is not commit-bound")
    require(
        stage_a["maximum_matching_annotation_json_files"] == 20,
        "stage A annotation fetch cap drifted",
    )
    for key in (
        "robot_rgb_topic_and_camera_info_decode",
        "robot_tf_chain_or_pose_available_at_rgb_timestamp",
        "robot_and_grs_timestamp_intervals_overlap",
        "pose_csv_interval_overlaps_bag_interval",
        "annotation_uuid_set_stable_for_at_least_two_consecutive_overlapping_frames",
        "grs_to_robot_offset_and_metadata_parse",
    ):
        require(stage_a["gates"][key] is True, f"stage A gate drifted: {key}")
    require(stage_a["gates"]["candidate_outputs_executed"] is False, "stage A exposed candidate outputs")
    require(stage_a["pass_authority"] == "authorize_stage_b_only", "stage A gained admission authority")

    stage_b = config["stage_b_full_lifecycle_canary"]
    require(stage_b["scene_id"] == 26, "stage B scene drifted")
    require(
        stage_b["parent_scene_permanently_excluded_from_future_lockbox"] is True,
        "stage B parent scene gained lockbox eligibility",
    )
    downloads = {row["name"]: row for row in stage_b["downloads_only_after_stage_a_pass"]}
    require(set(downloads) == {"26_robot.zip", "26_annotated.zip", "26_poses.zip"}, "stage B download roster drifted")
    for row in downloads.values():
        require(row["range_supported"] is True, f"stage B Range support missing: {row['name']}")
        require(row["bytes"] > 0, f"stage B byte count invalid: {row['name']}")
    require(
        downloads["26_robot.zip"]["bytes"] == audit["packaging"]["full_scene_26_robot_zip_bytes"],
        "stage B robot archive bytes drifted",
    )
    require(
        downloads["26_annotated.zip"]["bytes"] == audit["packaging"]["full_scene_26_annotated_zip_bytes"],
        "stage B annotation archive bytes drifted",
    )
    require(
        downloads["26_poses.zip"]["bytes"] == audit["packaging"]["full_scene_26_poses_zip_bytes"],
        "stage B pose archive bytes drifted",
    )
    final = policy["final_holdout_gates_unchanged"]
    expected_scaled = {
        "minimum_accepted_positive_events": math.ceil(final["minimum_positive_events_each_source"] / 48),
        "minimum_accepted_critical_events": math.ceil(final["minimum_critical_events_each_source"] / 48),
        "minimum_matched_negative_windows": math.ceil(final["minimum_matched_negative_windows_each_source"] / 48),
        "minimum_scorable_negative_exposure_minutes": (
            final["minimum_scorable_negative_exposure_minutes_each_source"] / 48
        ),
    }
    for key, expected in expected_scaled.items():
        require(
            math.isclose(stage_b["scaled_reject_only_gates"][key], expected),
            f"stage B scaled reject-only gate drifted: {key}",
        )
    require(stage_b["structural_gates"]["identity_swap_count"] == 0, "stage B identity swaps allowed")
    require(
        stage_b["structural_gates"]["out_of_frame_or_detection_loss_cannot_clear"] is True,
        "stage B disappearance can clear",
    )
    require(
        stage_b["structural_gates"]["causal_route_uses_current_and_past_robot_pose_only"] is True,
        "stage B route is not causal",
    )
    require(
        stage_b["structural_gates"]["actual_future_robot_path_annotation_only"] is True,
        "stage B future path leaked to candidate input",
    )
    require(stage_b["structural_gates"]["candidate_outputs_executed"] is False, "stage B exposed candidates")
    require(
        sorted(config["permanent_lockbox_exclusions"])
        == ["navwareset::scene_13", "navwareset::scene_26"],
        "NavWareSet canary exclusions drifted",
    )
    require(config["candidate_outputs_executed"] is False, "NavWareSet prescreen exposed candidates")
    require(
        config["app_detector_or_event_outputs_exposed"] is False,
        "NavWareSet prescreen exposed app outputs",
    )
    require(config["android_shadow"] == "closed", "NavWareSet prescreen opened Android shadow")
    require(config["h2_depth_ttc_route_risk_flip"] == "closed", "NavWareSet prescreen opened H2")

    receipt = load_json(freeze_receipt_path)
    require(
        receipt.get("schema")
        == "blindassist_ustrf_route_target_navwareset_source_prescreen_freeze_receipt_r1",
        "NavWareSet freeze receipt schema mismatch",
    )
    require(
        receipt["prescreen_config"]["path"] == config_path.resolve().relative_to(repo).as_posix(),
        "NavWareSet freeze receipt path drifted",
    )
    require(
        receipt["prescreen_config"]["sha256"] == sha256_file(original_snapshot),
        "NavWareSet original freeze receipt hash drifted",
    )
    require(receipt["payload_bytes_downloaded_before_freeze"] == 0, "NavWareSet payload downloaded before freeze")
    require(
        receipt["robot_or_grs_bag_content_decoded_before_freeze"] is False
        and receipt["annotation_content_decoded_before_freeze"] is False,
        "NavWareSet content decoded before freeze",
    )
    require(receipt["candidate_outputs_executed"] is False, "candidate outputs exposed before NavWareSet freeze")
    return {
        "schema": "blindassist_ustrf_route_target_navwareset_source_prescreen_validation_r1",
        "source_id": config["source_id"],
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(freeze_receipt_path),
        "stage_a_payload_bytes": sum(row["bytes"] for row in files),
        "stage_b_payload_bytes": sum(row["bytes"] for row in downloads.values()),
        "decision": "VALID_PREDECODE_TWO_STAGE_REJECT_ONLY_NAVWARESET_PRESCREEN",
        "candidate_outputs_executed": False,
        "android_shadow": "closed",
        "h2_depth_ttc_route_risk_flip": "closed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = validate(repo, args.config.resolve(), args.freeze_receipt.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
