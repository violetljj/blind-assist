from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import read_json, sha256, write_json


def validate_review(value: dict, role: str, expected: dict[str, str]) -> dict[str, dict]:
    if value.get("schema") != "blindassist_ustrf_sensor_replay_r3_review_v1":
        raise ValueError(f"reviewer {role} schema mismatch")
    if not isinstance(value.get("reviewer_role"), str) or not value["reviewer_role"] or value.get("independent_review") is not True:
        raise ValueError(f"reviewer {role} identity/isolation mismatch")
    if value.get("other_reviewer_outputs_viewed") is not False or value.get("candidate_alerts_viewed") is not False:
        raise ValueError(f"reviewer {role} was not isolated")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError(f"reviewer {role} sources missing")
    by_id = {row["source_id"]: row for row in rows}
    if set(by_id) != set(expected):
        raise ValueError(f"reviewer {role} source identity mismatch")
    for source_id, row in by_id.items():
        if row.get("manifest_sha256") != expected[source_id]:
            raise ValueError(f"reviewer {role} manifest binding mismatch: {source_id}")
        if row.get("route_valid") not in (True, False, "abstain"):
            raise ValueError(f"reviewer {role} invalid route disposition: {source_id}")
        if not isinstance(row.get("events"), list):
            raise ValueError(f"reviewer {role} events missing: {source_id}")
    return by_id


def anchor(event: dict, key: str) -> int:
    value = event.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {key}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-tolerance-frames", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    inputs = read_json(args.root / "review_inputs.json")
    expected = {row["source_id"]: row["manifest_sha256"] for row in inputs["sources"]}
    review_a_path = args.root / "reviewer_a_raw.json"
    review_b_path = args.root / "reviewer_b_raw.json"
    raw_a = read_json(review_a_path)
    raw_b = read_json(review_b_path)
    if raw_a.get("reviewer_role") == raw_b.get("reviewer_role"):
        raise ValueError("reviewer role labels must be distinct")
    a = validate_review(raw_a, "a", expected)
    b = validate_review(raw_b, "b", expected)
    sources = []
    for source_id in expected:
        first, second = a[source_id], b[source_id]
        admitted = first["route_valid"] is True and second["route_valid"] is True
        consensus_events = []
        reason = "both reviewers admitted route/event sequence"
        if admitted:
            if len(first["events"]) != len(second["events"]) or not first["events"]:
                admitted = False
                reason = "event count disagreement or no events"
            else:
                for event_index, (left, right) in enumerate(zip(first["events"], second["events"])):
                    values = {}
                    for key in ("onset_frame", "alertable_frame", "passed_or_cleared_frame", "end_frame"):
                        pair = [anchor(left, key), anchor(right, key)]
                        if abs(pair[0] - pair[1]) > args.anchor_tolerance_frames:
                            admitted = False
                            reason = f"anchor disagreement beyond tolerance: event {event_index} {key}"
                            break
                        values[key] = int(round(sum(pair) / 2))
                    if not admitted:
                        break
                    if not (values["onset_frame"] <= values["alertable_frame"] <= values["passed_or_cleared_frame"] <= values["end_frame"]):
                        admitted = False
                        reason = f"invalid lifecycle order: event {event_index}"
                        break
                    values.update({
                        "event_id": f"{source_id}_event_{event_index:02d}",
                        "critical": bool(left.get("critical")) and bool(right.get("critical")),
                    })
                    consensus_events.append(values)
        else:
            reason = f"route rejected or abstained: a={first['route_valid']} b={second['route_valid']}"
        if not admitted:
            consensus_events = []
        sources.append({"source_id": source_id, "route_event_admitted": admitted, "events": consensus_events, "reason": reason})
    all_admitted = len(sources) >= 3 and all(row["route_event_admitted"] for row in sources)
    report = {
        "schema": "blindassist_ustrf_sensor_replay_r3_review_consensus_v1",
        "workflow_id": "ustrf_r3_complete_sequence_two_model_review_v1",
        "review_inputs_sha256": sha256(args.root / "review_inputs.json"),
        "reviewer_a_sha256": sha256(review_a_path),
        "reviewer_b_sha256": sha256(review_b_path),
        "candidate_alerts_visible_to_reviewers": False,
        "sources": sources,
        "all_sources_admitted": all_admitted,
        "event_truth_authority": all_admitted,
        "third_model_adjudication_required": any("disagreement" in row["reason"] for row in sources),
        "production_authority": False,
    }
    write_json(args.output, report)
    print(json.dumps({"all_sources_admitted": all_admitted, "sources": [{"source_id": row["source_id"], "admitted": row["route_event_admitted"], "events": len(row["events"])} for row in sources]}))
    return 0 if all_admitted else 3


if __name__ == "__main__":
    raise SystemExit(main())
