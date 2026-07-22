from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_host_coverage import read_json, sha256


CONSENSUS = {
    "dynamics_0": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/dynamics_0-review-consensus-v2.json"),
    "lt_changes_dynamics_0": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/lt_changes_dynamics_0-review-v1/review-consensus-v2.json"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--host-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    config = read_json(args.config)
    device = read_json(args.device)
    if device["schema"] != "blindassist_ustrf_detector_taxonomy_device_output_v1":
        raise ValueError("unexpected device receipt schema")
    if device["frame_count"] != config["parent"]["frame_count"] or device["failure_count"] != 0:
        raise ValueError("incomplete device receipt")
    threshold = float(config["detector"]["confidence_threshold"])
    device_by_source: dict[str, dict[str, dict]] = {}
    for row in device["frames"]:
        device_by_source.setdefault(row["source_name"], {})[row["frame_id"]] = row
    source_summaries = []
    total_disagreements = 0
    event_rows = []
    for host_path in args.host_ledger:
        host = read_json(host_path)
        source_name = host["source_name"]
        device_rows = device_by_source[source_name]
        both_person = both_no_person = host_only = device_only = 0
        max_score_difference = 0.0
        for row in host["frames"]:
            other = device_rows[row["frame_id"]]
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
