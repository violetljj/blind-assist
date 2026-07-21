#!/usr/bin/env python3
"""Validate frozen R1.2a/R1.3 contracts and stage seen continuous replay input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import load_json, sha256_file, write_json
from diagnostic_contract import require


PROTOCOL_SCHEMA = "blindassist_ustrf_crosscam_continuous_event_protocol_v1"
PREREG_SCHEMA = "blindassist_ustrf_crosscam_continuous_event_preregistration_v1"
OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_continuous_android_input_v1"
REMOTE_ROOT = "ustrf-crosscam-r12a"


def validate_protocol(protocol: dict) -> None:
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "R1.2a protocol schema mismatch")
    require(protocol.get("dataset_role") == "seen_diagnostic_not_held_out", "R1.2a must be seen diagnostic")
    detector = protocol["frozen_detector"]
    require(detector["confidence_threshold"] == 0.05, "frozen .05 confidence changed")
    require(detector["target_anchor_iou_threshold"] == 0.30, "frozen .30 anchor IoU changed")
    for key in ("prompt_or_label_inventory_change_allowed", "threshold_change_allowed", "bbox_or_polygon_refit_allowed"):
        require(detector[key] is False, f"{key} must remain false")
    replay = protocol["replay_contract"]
    require(replay["sample_period_ms"] == 500, "sample period drift")
    events = protocol["events"]
    require(len(events) == 12, "R1.2a must contain the 12 already opened R1.1/R1.2 sources")
    require(len({row["event_id"] for row in events}) == 12, "duplicate event id")
    require(sum(row["expected_class"] == "positive" for row in events) == 6, "expected six positives")
    require(sum(row["expected_class"] == "negative" for row in events) == 6, "expected six negatives")
    for row in events:
        start, end = row["clip_window_ms"]
        require(replay["minimum_clip_duration_ms"] <= end - start <= replay["maximum_clip_duration_ms"],
                f"{row['event_id']}: clip must be 5-15 seconds")
    vancouver = next(row for row in events if row["event_id"] == "vancouver_right_delineator_lateral_clear")
    require(vancouver.get("diagnostic_role") == "miss_lead_only" and vancouver.get("gate_eligible") is False,
            "Vancouver must remain a non-gating miss lead")
    authority = protocol["authority"]
    require(authority["new_held_out_read"] is False and authority["benchmark_only"] is True,
            "R1.2a authority boundary drift")
    require(not authority["training_authorized"] and not authority["production_model_replacement_authorized"],
            "R1.2a cannot authorize training or production")


def validate_r13(prereg: dict, protocol_path: Path) -> None:
    require(prereg.get("schema") == PREREG_SCHEMA, "R1.3 prereg schema mismatch")
    require(prereg.get("dataset_role") == "preregistered_unopened_inventory_slots", "R1.3 must remain unopened")
    require(prereg["prerequisite"]["r12a_protocol_sha256"] == sha256_file(protocol_path), "R1.3/R1.2a hash mismatch")
    inventory = prereg["inventory"]
    slots = inventory["source_slots"]
    require(len(slots) == 12 and len({row["slot_id"] for row in slots}) == 12, "R1.3 requires 12 unique slots")
    require(sum(row["expected_class"] == "positive" for row in slots) == 6, "R1.3 requires six positives")
    require(sum(row["expected_class"] == "negative" for row in slots) == 6, "R1.3 requires six negatives")
    access = prereg["novelty_and_access"]
    require(access["source_discovery_authorized_in_r12a_round"] is False, "R1.3 discovery opened early")
    require(access["download_decode_or_detector_inference_authorized_in_r12a_round"] is False, "R1.3 data access opened early")
    require(access["result_access_authorized"] is False, "R1.3 results opened early")
    review = prereg["event_truth_review"]
    require(review["truth_authority"] == "dual_vlm_reviewed_provisional_event_truth_not_human_truth",
            "R1.3 VLM truth authority overstated")
    require(review["detector_outputs_hidden_from_reviewers"] is True, "detector leakage into VLM review")


def event_map(document: dict) -> dict[str, dict]:
    return {row["event_id"]: row for row in document["events"]}


def select_route_polygon(projection_event: dict, target_anchor_ms: int) -> tuple[list, int]:
    frames = [row for row in projection_event["frames"] if row.get("status") == "admitted"]
    require(frames, "projection event has no admitted frame")
    selected = min(frames, key=lambda row: abs(int(row["timestamp_ms"]) - target_anchor_ms))
    return selected["route_polygon_xy_norm"], int(selected["timestamp_ms"])


def source_video(staging: dict, event_id: str, repo: Path) -> dict:
    marker = f"/videos/{event_id}"
    matches = [row for row in staging["files"] if marker in row["device_relative_path"].replace("\\", "/")]
    require(len(matches) == 1, f"{event_id}: expected one staged source video")
    row = matches[0]
    host = Path(row["host_path"]).resolve()
    require(host.is_file(), f"{event_id}: source video missing")
    require(sha256_file(host) == row["sha256"], f"{event_id}: source video SHA mismatch")
    require(host.is_relative_to(repo), f"{event_id}: source video must remain under repo artifacts.local")
    return {"host_path": host, "sha256": row["sha256"], "suffix": host.suffix.lower()}


def materialize(repo: Path, protocol_path: Path, prereg_path: Path, output_dir: Path) -> dict:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    validate_r13(load_json(prereg_path), protocol_path)
    inputs = protocol["evidence_inputs"]
    rounds = {}
    for name in ("r11", "r12"):
        ledger_path = (repo / inputs[f"{name}_target_ledger"]).resolve()
        projection_path = (repo / inputs[f"{name}_projection_receipt"]).resolve()
        staging_path = (repo / inputs[f"{name}_staging_receipt"]).resolve()
        require(sha256_file(ledger_path) == inputs[f"{name}_target_ledger_sha256"], f"{name} ledger SHA mismatch")
        require(sha256_file(projection_path) == inputs[f"{name}_projection_receipt_sha256"], f"{name} projection SHA mismatch")
        rounds[name] = {
            "ledger": event_map(load_json(ledger_path)),
            "projection": event_map(load_json(projection_path)),
            "staging": load_json(staging_path),
        }
    sources, staged_files = [], []
    for spec in protocol["events"]:
        source_round = rounds[spec["round"]]
        ledger_event = source_round["ledger"].get(spec["event_id"])
        projection_event = source_round["projection"].get(spec["event_id"])
        require(ledger_event is not None and projection_event is not None, f"{spec['event_id']}: frozen evidence missing")
        target = ledger_event["target_instance"]
        visible_anchors = [row for row in target["frames"] if row["visibility"] == "visible"]
        require(visible_anchors, f"{spec['event_id']}: visible target anchor missing")
        primary_anchor = min(visible_anchors, key=lambda row: abs(int(row["timestamp_ms"]) - sum(spec["clip_window_ms"]) / 2))
        polygon, polygon_anchor_ms = select_route_polygon(projection_event, int(primary_anchor["timestamp_ms"]))
        video = source_video(source_round["staging"], spec["event_id"], repo)
        remote_video = f"{REMOTE_ROOT}/videos/{spec['event_id']}{video['suffix']}"
        source = {
            "event_id": spec["event_id"], "source_id": ledger_event["source_id"],
            "parent_round": spec["round"], "dataset_role": "seen_diagnostic_not_held_out",
            "expected_class": spec["expected_class"], "clip_window_ms": spec["clip_window_ms"],
            "alertable_start_ms": spec.get("alertable_start_ms"),
            "known_not_visible_from_ms": spec.get("known_not_visible_from_ms"),
            "known_not_visible_until_ms": spec.get("known_not_visible_until_ms"),
            "gate_eligible": spec.get("gate_eligible", True),
            "diagnostic_role": spec.get("diagnostic_role", "gate_diagnostic"),
            "video_path": remote_video, "video_sha256": video["sha256"],
            "target_instance_id": target["target_instance_id"],
            "detector_label_allowlist": target["detector_label_allowlist"],
            "expected_route_relation": target["expected_route_relation"],
            "target_anchors": target["frames"],
            "primary_anchor_timestamp_ms": int(primary_anchor["timestamp_ms"]),
            "route_polygon_xy_norm": polygon,
            "route_polygon_anchor_timestamp_ms": polygon_anchor_ms,
            "route_proxy_is_geometry_truth": False,
        }
        sources.append(source)
        staged_files.append({"host_path": str(video["host_path"]), "device_relative_path": remote_video, "sha256": video["sha256"]})
    output = {
        "schema": OUTPUT_SCHEMA,
        "contract_id": protocol["contract_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_role": "seen_diagnostic_not_held_out",
        "frozen_detector": protocol["frozen_detector"],
        "replay_contract": protocol["replay_contract"],
        "device_gate": protocol["device_gate"],
        "event_gate": protocol["event_gate"],
        "sources": sources,
        "authority": protocol["authority"],
    }
    output_path = output_dir / "android_r12a_continuous_input.json"
    write_json(output_path, output)
    staged_files.append({"host_path": str(output_path.resolve()), "device_relative_path": f"{REMOTE_ROOT}/input.json",
                         "sha256": sha256_file(output_path)})
    receipt = {
        "schema": "blindassist_ustrf_crosscam_continuous_android_staging_v1",
        "dataset_role": "seen_diagnostic_not_held_out",
        "input_sha256": sha256_file(output_path),
        "new_held_out_read": False,
        "files": staged_files,
    }
    write_json(output_dir / "host_staging_receipt.json", receipt)
    return {"input": str(output_path), "input_sha256": receipt["input_sha256"], "source_count": len(sources)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--r13-prereg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.repo_root.resolve(), args.protocol.resolve(), args.r13_prereg.resolve(), args.output_dir.resolve())
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
