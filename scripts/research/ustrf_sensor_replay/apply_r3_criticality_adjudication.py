from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contract import read_json, sha256, write_json


INPUT_SCHEMA = "blindassist_ustrf_sensor_replay_r3_criticality_adjudication_inputs_v1"
OUTPUT_SCHEMA = "blindassist_ustrf_sensor_replay_r3_criticality_adjudication_v1"


def anchors(event: dict[str, Any]) -> tuple[int, int, int, int]:
    values = tuple(event.get(key) for key in ("onset_frame", "alertable_frame", "passed_or_cleared_frame", "end_frame"))
    if any(not isinstance(value, int) or value < 0 for value in values) or not values[0] <= values[1] <= values[2] <= values[3]:
        raise ValueError("invalid frozen criticality event anchors")
    return values


def validate(repo: Path, inputs_path: Path, adjudication_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = read_json(inputs_path)
    adjudication = read_json(adjudication_path)
    if inputs.get("schema") != INPUT_SCHEMA or inputs.get("candidate_alerts_visible") is not False or inputs.get("evaluator_output_visible") is not False:
        raise ValueError("invalid criticality adjudication inputs")
    prompt_path = inputs_path.with_name("criticality_adjudication_prompt_v2.txt")
    if sha256(prompt_path) != inputs.get("prompt_sha256"):
        raise ValueError("criticality prompt hash mismatch")
    review_hashes = []
    for row in inputs.get("input_reviews", []):
        path = (repo / row["path"]).resolve()
        if sha256(path) != row["sha256"]:
            raise ValueError("criticality input-review hash mismatch")
        review_hashes.append(row["sha256"])
    if len(review_hashes) < 3:
        raise ValueError("criticality adjudication lacks review evidence")
    manifest_path = (repo / inputs["manifest_path"]).resolve()
    if sha256(manifest_path) != inputs["manifest_sha256"]:
        raise ValueError("criticality manifest hash mismatch")
    consensus_rows = [row for row in inputs["input_reviews"] if row.get("role") == "frozen_consensus_v1"]
    if len(consensus_rows) != 1:
        raise ValueError("criticality input must bind one frozen consensus")
    consensus = read_json((repo / consensus_rows[0]["path"]).resolve())
    source_rows = consensus.get("sources", [])
    if len(source_rows) != 1 or source_rows[0].get("source_id") != inputs.get("source_id") or source_rows[0].get("route_event_admitted") is not True:
        raise ValueError("criticality source is not an admitted frozen consensus")
    if source_rows[0].get("events") != [{**event, "critical": False} for event in inputs.get("events", [])]:
        raise ValueError("criticality event anchors do not match frozen consensus")

    if (
        adjudication.get("schema") != OUTPUT_SCHEMA
        or adjudication.get("reviewer_type") != "ai_model"
        or adjudication.get("reviewer_role") != "criticality_adjudicator"
        or adjudication.get("workflow_id") != "ustrf_event_criticality_adjudication_v1"
        or adjudication.get("prompt_sha256") != inputs["prompt_sha256"]
        or adjudication.get("input_sha256") != sha256(inputs_path)
        or adjudication.get("input_review_sha256s") != review_hashes
        or adjudication.get("isolated_context") is not True
        or adjudication.get("candidate_alerts_viewed") is not False
        or adjudication.get("candidate_output_visible") is not False
        or adjudication.get("evaluator_output_visible") is not False
        or adjudication.get("other_context_viewed") is not False
        or adjudication.get("abstained") is not False
        or adjudication.get("source_id") != inputs["source_id"]
        or adjudication.get("manifest_sha256") != inputs["manifest_sha256"]
    ):
        raise ValueError("criticality adjudication identity, isolation, or hash binding mismatch")
    confidence = adjudication.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.65 <= float(confidence) <= 1.0:
        raise ValueError("criticality adjudication confidence invalid")
    adjudicated_events = adjudication.get("events", [])
    if len(adjudicated_events) != len(inputs["events"]):
        raise ValueError("criticality adjudication changed event count")
    for frozen, resolved in zip(inputs["events"], adjudicated_events):
        if resolved.get("event_id") != frozen["event_id"] or anchors(resolved) != anchors(frozen):
            raise ValueError("criticality adjudication changed event identity or anchors")
        if not isinstance(resolved.get("critical"), bool):
            raise ValueError("criticality adjudication label is not boolean")
        event_confidence = resolved.get("confidence")
        if not isinstance(event_confidence, (int, float)) or isinstance(event_confidence, bool) or not 0.65 <= float(event_confidence) <= 1.0:
            raise ValueError("criticality event confidence invalid")
    return inputs, adjudication, consensus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        inputs, adjudication, consensus = validate(args.repo.resolve(), args.inputs.resolve(), args.adjudication.resolve())
        labels = {row["event_id"]: row["critical"] for row in adjudication["events"]}
        source = consensus["sources"][0]
        source["events"] = [{**event, "critical": labels[event["event_id"]]} for event in source["events"]]
        consensus["criticality_adjudication"] = {
            "workflow_id": adjudication["workflow_id"],
            "inputs_sha256": sha256(args.inputs.resolve()),
            "adjudication_sha256": sha256(args.adjudication.resolve()),
            "critical_event_count": sum(bool(event["critical"]) for event in source["events"]),
            "candidate_output_visible": False,
            "evaluator_output_visible": False,
        }
        write_json(args.output.resolve(), consensus)
        print(json.dumps({"source_id": source["source_id"], "events": len(source["events"]), "critical_events": consensus["criticality_adjudication"]["critical_event_count"]}))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
