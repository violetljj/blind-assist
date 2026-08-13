from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENE_PREFIX = "INSTRUMENTATION_STATUS: taro_arcore_yolo_positive_evidence_scene="
SCENE_SCHEMA = "blindassist_taro_arcore_yolo_positive_evidence_scene_v1"
PROTOCOL_SCHEMA = "blindassist.taro.rgb_pair_frozen_visual_evidence_backend_preflight.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ingest(test_log: Path, output: Path) -> dict[str, Any]:
    candidates = [
        line[len(SCENE_PREFIX):]
        for line in test_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(SCENE_PREFIX)
    ]
    require(bool(candidates), "scene instrumentation receipt not found")
    payload = json.loads(candidates[-1])
    require(payload["schema"] == SCENE_SCHEMA, "scene schema mismatch")
    require(payload["privacy"]["raw_images_persisted"] is False, "raw image persistence is forbidden")
    require(payload["privacy"]["detections_or_boxes_persisted"] is False, "box persistence is forbidden")
    write_json(output, payload)
    return payload


def aggregate(protocol_path: Path, scene_paths: list[Path], output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["schema"] == PROTOCOL_SCHEMA, "protocol schema mismatch")
    lock = protocol["frozen_shadow_protocol"]
    selected_backend = protocol["backend_audit"]["selected"]
    result_protocol_id = protocol["unique_successor"]
    scenes = [json.loads(path.read_text(encoding="utf-8")) for path in scene_paths]
    require(all(row["schema"] == SCENE_SCHEMA for row in scenes), "scene schema mismatch")
    scene_ids = [row["scene_id"] for row in scenes]
    require(len(scene_ids) == len(set(scene_ids)), "scene ids must be unique")
    required_scenes = lock["required_distinct_scene_parents"]
    minimum_total = lock["minimum_evaluable_references_total"]
    minimum_per_scene = lock["minimum_evaluable_references_per_scene"]
    maximum_per_scene = lock["maximum_evaluable_references_per_scene"]
    minimum_positive_scenes = lock["minimum_scene_parents_with_positive_support"]
    minimum_opportunity_scenes = lock["minimum_opportunity_scene_parents"]
    minimum_strict_win_scenes = lock["minimum_pose_strict_win_scene_parents"]
    maximum_latency_p95 = lock["gates"]["maximum_unique_inference_total_latency_p95_ms"]
    require(len(scenes) == required_scenes, "formal aggregate requires exactly the locked scene count")
    for row in scenes:
        require(row["protocol_id"] == result_protocol_id, "scene protocol id mismatch")
        require(row["model_sha256"] == selected_backend["model_sha256"], "scene model hash mismatch")
        require(row["labels_sha256"] == selected_backend["labels_sha256"], "scene labels hash mismatch")
        require(
            row["execution_backend"].lower() == selected_backend["execution_backend"].lower(),
            "scene execution backend mismatch",
        )
        require(row["availability"] == "SUPPORTED_INSTALLED", "scene ARCore availability mismatch")
        require(row["detector_ready_at_start"] is True, "scene detector was not ready")
        require(
            row["target_evaluable_references"] == row["evaluable_reference_count"],
            "scene did not reach its frozen reference target",
        )
        require(
            row["exact_passive_payload_lookup_count"] == row["evaluable_reference_count"],
            "scene passive arm is not exact-complete",
        )
        require(
            row["exact_pose_payload_lookup_count"] == row["evaluable_reference_count"],
            "scene pose arm is not exact-complete",
        )
        require(row["privacy"]["raw_images_persisted"] is False, "raw image persistence is forbidden")
        require(
            row["privacy"]["detections_or_boxes_persisted"] is False,
            "detection or box persistence is forbidden",
        )
        require(row["authorization"]["benchmark_only"] is True, "scene escaped benchmark-only scope")
        require(
            row["authorization"]["screen_space_positive_evidence_only"] is True,
            "scene escaped positive-only scope",
        )
        require(row["authorization"]["absence_is_safe"] is False, "absence was promoted to safe")
        require(row["authorization"]["production_authorized"] is False, "production scope is forbidden")

    total_references = sum(row["evaluable_reference_count"] for row in scenes)
    positive_scene_count = sum(row["positive_support_reference_count"] > 0 for row in scenes)
    opportunity_scene_count = sum(row["opportunity_reference_count"] > 0 for row in scenes)
    strict_win_scene_count = sum(
        row["pose_new_focused_token_mean"] > row["passive_new_focused_token_mean"]
        for row in scenes
        if row["evaluable_reference_count"] > 0
    )
    tie_scene_count = sum(
        row["pose_new_focused_token_mean"] == row["passive_new_focused_token_mean"]
        for row in scenes
        if row["evaluable_reference_count"] > 0
    )
    loss_scene_count = sum(
        row["pose_new_focused_token_mean"] < row["passive_new_focused_token_mean"]
        for row in scenes
        if row["evaluable_reference_count"] > 0
    )
    passive_parent_macro = (
        sum(row["passive_new_focused_token_mean"] for row in scenes) / len(scenes)
        if scenes else None
    )
    pose_parent_macro = (
        sum(row["pose_new_focused_token_mean"] for row in scenes) / len(scenes)
        if scenes else None
    )
    maximum_scene_detector_p95 = max(
        (row["detector_total_latency_ms"]["p95"] for row in scenes),
        default=None,
    )
    denominator_checks = {
        "required_distinct_scene_parents": len(scenes) == required_scenes,
        "minimum_evaluable_references_total": total_references >= minimum_total,
        "per_scene_reference_bounds": all(
            minimum_per_scene <= row["evaluable_reference_count"] <= maximum_per_scene
            for row in scenes
        ),
        "minimum_scene_parents_with_positive_support": positive_scene_count >= minimum_positive_scenes,
        "minimum_opportunity_scene_parents": opportunity_scene_count >= minimum_opportunity_scenes,
    }
    runtime_checks = {
        "all_scene_structural_gates": all(row["structural_gate_pass"] for row in scenes),
        "maximum_scene_detector_p95_within_lock": (
            maximum_scene_detector_p95 is not None and maximum_scene_detector_p95 <= maximum_latency_p95
        ),
        "zero_source_identity_mismatches": sum(row["source_identity_mismatch_count"] for row in scenes) == 0,
        "zero_selected_payload_lookup_misses": sum(row["selected_payload_lookup_miss_count"] for row in scenes) == 0,
        "zero_model_failures": sum(sum(row["model_failure_counts"].values()) for row in scenes) == 0,
        "zero_resource_errors": sum(sum(row["resource_error_counts"].values()) for row in scenes) == 0,
    }
    decision_checks = {
        "pose_parent_macro_must_exceed_passive": (
            pose_parent_macro is not None and passive_parent_macro is not None and
            pose_parent_macro > passive_parent_macro
        ),
        "minimum_pose_strict_win_scene_parents": strict_win_scene_count >= minimum_strict_win_scenes,
    }
    if not all(denominator_checks.values()):
        terminal = lock["terminal_if_denominator_fails"]
    elif not all(runtime_checks.values()):
        terminal = lock["terminal_if_runtime_gate_fails"]
    elif not all(decision_checks.values()):
        terminal = lock["terminal_if_pose_does_not_beat_passive"]
    else:
        terminal = "POSE_DIVERSE_POSITIVE_VISUAL_EVIDENCE_PASS"
    result = {
        "schema": "blindassist.taro.rgb_pair_yolo_positive_evidence_shadow_result.v1",
        "protocol_id": result_protocol_id,
        "terminal": terminal,
        "scene_ids": sorted(scene_ids),
        "scene_count": len(scenes),
        "evaluable_reference_count": total_references,
        "positive_support_scene_count": positive_scene_count,
        "opportunity_scene_count": opportunity_scene_count,
        "pose_strict_win_scene_count": strict_win_scene_count,
        "tie_scene_count": tie_scene_count,
        "pose_loss_scene_count": loss_scene_count,
        "passive_parent_macro_new_focused_tokens": passive_parent_macro,
        "pose_parent_macro_new_focused_tokens": pose_parent_macro,
        "maximum_scene_detector_total_latency_p95_ms": maximum_scene_detector_p95,
        "denominator_checks": denominator_checks,
        "runtime_checks": runtime_checks,
        "decision_checks": decision_checks,
        "scenes": scenes,
        "claim_ceiling": protocol["claim_ceiling"],
    }
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--test-log", required=True, type=Path)
    ingest_parser.add_argument("--output", required=True, type=Path)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--protocol", required=True, type=Path)
    aggregate_parser.add_argument("--scene", required=True, action="append", type=Path)
    aggregate_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (
        ingest(args.test_log, args.output)
        if args.command == "ingest"
        else aggregate(args.protocol, args.scene, args.output)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
