from __future__ import annotations

import argparse
import json
from pathlib import Path

from materialize_device_bundle import validate_host_ledgers
from run_host_coverage import read_json, sha256


CONSENSUS = {
    "dynamics_0": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/dynamics_0-review-consensus-v2.json"),
    "lt_changes_dynamics_0": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/lt_changes_dynamics_0-review-v1/review-consensus-v2.json"),
}


def index_unique_frames(rows: list[dict], label: str) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["source_name"], row["frame_id"])
        if key in indexed:
            raise ValueError(f"duplicate {label} frame identity: {key}")
        indexed[key] = row
    return indexed


def validate_device_binding(config: dict, manifest: dict, manifest_sha: str, device: dict) -> None:
    if manifest.get("schema") != "blindassist_ustrf_detector_taxonomy_device_input_v1":
        raise ValueError("unexpected device input manifest schema")
    if device.get("schema") != "blindassist_ustrf_detector_taxonomy_device_output_v1":
        raise ValueError("unexpected device receipt schema")
    if device.get("input_manifest_sha256") != manifest_sha:
        raise ValueError("device receipt/input manifest hash mismatch")
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("device input manifest config/window binding mismatch")
    if "windows_sha256" in manifest and manifest["windows_sha256"] != config["parent"]["windows_sha256"]:
        raise ValueError("device input manifest config/window binding mismatch")
    if manifest.get("model_sha256") != config["detector"]["model_sha256"] or manifest.get("labels_sha256") != config["detector"]["labels_sha256"]:
        raise ValueError("device input manifest model/labels binding mismatch")
    if manifest.get("input_shape") != config["detector"]["input_shape"] or manifest.get("output_shape") != config["detector"]["output_shape"]:
        raise ValueError("device input manifest tensor contract mismatch")
    expected_count = config["parent"]["frame_count"]
    if manifest.get("frame_count") != expected_count or device.get("frame_count") != expected_count or device.get("failure_count") != 0:
        raise ValueError("incomplete device manifest/receipt")
    manifest_rows = index_unique_frames(manifest["frames"], "manifest")
    device_rows = index_unique_frames(device["frames"], "device")
    if len(manifest_rows) != expected_count or set(manifest_rows) != set(device_rows):
        raise ValueError("device receipt frame inventory mismatch")
    actual_source_counts: dict[str, int] = {}
    for key, expected in manifest_rows.items():
        actual = device_rows[key]
        actual_source_counts[expected["source_id"]] = actual_source_counts.get(expected["source_id"], 0) + 1
        bindings = {
            "image_sha256": "image_sha256",
            "host_input_tensor_sha256": "host_input_tensor_sha256",
            "host_raw_output_sha256": "host_raw_output_sha256",
        }
        for expected_field, actual_field in bindings.items():
            if actual.get(actual_field) != expected.get(expected_field):
                raise ValueError(f"device receipt per-frame binding mismatch: {key}/{actual_field}")
    if actual_source_counts != config["parent"]["source_frame_counts"]:
        raise ValueError("device manifest per-source inventory mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-input-manifest", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--host-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    config = read_json(args.config)
    config["_config_sha256"] = sha256(args.config)
    manifest = read_json(args.device_input_manifest)
    device = read_json(args.device)
    validate_device_binding(config, manifest, sha256(args.device_input_manifest), device)
    threshold = float(config["detector"]["confidence_threshold"])
    device_by_source: dict[str, dict[str, dict]] = {}
    for row in device["frames"]:
        device_by_source.setdefault(row["source_name"], {})[row["frame_id"]] = row
    source_summaries = []
    total_disagreements = 0
    event_rows = []
    _, host_ledgers = validate_host_ledgers(args.config, args.host_ledger)
    manifest_rows = index_unique_frames(manifest["frames"], "manifest")
    for host in host_ledgers:
        source_name = host["source_name"]
        device_rows = device_by_source[source_name]
        both_person = both_no_person = host_only = device_only = 0
        max_score_difference = 0.0
        for row in host["frames"]:
            other = device_rows[row["frame_id"]]
            expected = manifest_rows[(source_name, row["frame_id"])]
            if expected["host_input_tensor_sha256"] != row["input_tensor_sha256"] or expected["host_raw_output_sha256"] != row["raw_output_sha256"] or expected["image_sha256"] != row["image_sha256"]:
                raise ValueError(f"manifest/host ledger per-frame binding mismatch: {source_name}/{row['frame_id']}")
            host_person = float(row["raw_person_max_confidence"]) >= threshold
            device_person = float(other["android_raw_person_max_confidence"]) >= threshold
            if host_person and device_person:
                both_person += 1
            elif not host_person and not device_person:
                both_no_person += 1
            elif host_person:
                host_only += 1
            else:
                device_only += 1
            max_score_difference = max(
                max_score_difference,
                abs(float(row["raw_person_max_confidence"]) - float(other["android_raw_person_max_confidence"])),
            )
        total_disagreements += host_only + device_only
        consensus_path = CONSENSUS[source_name]
        expected_consensus_hash = config["parent"]
        parent_config = read_json(Path(expected_consensus_hash["config_path"]))
        input_key = "dynamics_0" if source_name == "dynamics_0" else "lt_changes_dynamics_0"
        if sha256(consensus_path) != parent_config["inputs"][input_key]["event_consensus_sha256"]:
            raise ValueError(f"event consensus hash mismatch: {source_name}")
        events = read_json(consensus_path)["sources"][0]["events"]
        covered_events = 0
        for event in events:
            frame_ids = range(int(event["alertable_frame"]), int(event["passed_or_cleared_frame"]) + 1)
            values = [
                float(device_rows[f"{frame_id:06d}"]["android_raw_person_max_confidence"])
                for frame_id in frame_ids if f"{frame_id:06d}" in device_rows
            ]
            hits = sum(value >= threshold for value in values)
            covered = hits > 0
            covered_events += int(covered)
            event_rows.append({
                "source_name": source_name,
                "event_id": event["event_id"],
                "scored_frame_count": len(values),
                "person_proposal_frame_count": hits,
                "max_person_confidence": max(values, default=None),
                "raw_person_proposal_covered": covered,
                "target_instance_attribution": "not_evaluable_without_person_bbox_truth",
            })
        source_summaries.append({
            "source_name": source_name,
            "frame_count": host["frame_count"],
            "host_person_frame_count": host["person_frame_count"],
            "device_person_frame_count": sum(
                float(row["android_raw_person_max_confidence"]) >= threshold for row in device_rows.values()
            ),
            "both_person": both_person,
            "both_no_person": both_no_person,
            "host_only_person": host_only,
            "device_only_person": device_only,
            "max_abs_person_score_difference": max_score_difference,
            "raw_person_proposal_event_coverage": f"{covered_events}/{len(events)}",
        })
    canary = device_by_source["dynamics_0"][config["gates"]["G2_controlled_person_canary"]["frame_id"]]
    payload = {
        "schema": "blindassist_ustrf_detector_taxonomy_parity_analysis_v1",
        "config_sha256": sha256(args.config),
        "device_receipt_sha256": sha256(args.device),
        "frame_count": device["frame_count"],
        "G0_manifest": "pass",
        "G1_android_host_exact_parity": "fail",
        "input_tensor_exact_match_count": device["input_tensor_exact_match_count"],
        "raw_output_exact_match_count": device["raw_output_exact_match_count"],
        "threshold_person_presence_disagreement_frames": total_disagreements,
        "G2_controlled_person_canary": {
            "status": "partial_pass_raw_tensor_index_labels_person_score; source_box_and_target_match_not_evaluable",
            "frame_id": canary["frame_id"],
            "android_person_score": canary["android_raw_person_max_confidence"],
            "host_person_score": canary["host_raw_person_max_confidence"],
            "frozen_threshold": threshold,
        },
        "G3_taxonomy_attribution": "closed_by_G1_and_missing_target_bbox_truth",
        "G4_baseline_target_coverage": "not_evaluable",
        "G5_candidate_comparison": "closed",
        "T0_T3": "closed",
        "H2": "closed",
        "source_summaries": source_summaries,
        "events": event_rows,
        "interpretation": "historical zero-person result was caused by host output-layout decoding; Android has person proposals, but PIL/Canvas exact parity and target-instance truth are not closed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "G1": payload["G1_android_host_exact_parity"],
        "threshold_disagreements": total_disagreements,
        "event_count": len(event_rows),
        "raw_person_proposal_events_covered": sum(row["raw_person_proposal_covered"] for row in event_rows),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
