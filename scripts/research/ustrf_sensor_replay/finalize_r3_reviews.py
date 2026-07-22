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


def build_consensus_events(first: dict, second: dict, tolerance: int, frame_count: int) -> tuple[list[dict], str | None]:
    """Return direct two-review consensus, or a disagreement reason."""
    if len(first["events"]) != len(second["events"]):
        return [], "event count disagreement"
    if not first["events"]:
        return [], None
    result = []
    for event_index, (left, right) in enumerate(zip(first["events"], second["events"])):
        values = {}
        for key in ("onset_frame", "alertable_frame", "passed_or_cleared_frame", "end_frame"):
            pair = [anchor(left, key), anchor(right, key)]
            if abs(pair[0] - pair[1]) > tolerance:
                return [], f"anchor disagreement beyond tolerance: event {event_index} {key}"
            values[key] = int(round(sum(pair) / 2))
        if not (values["onset_frame"] <= values["alertable_frame"] <= values["passed_or_cleared_frame"] <= values["end_frame"] < frame_count):
            raise ValueError(f"invalid lifecycle order or range: event {event_index}")
        values.update({
            "event_id": f"consensus_event_{event_index:02d}",
            "critical": bool(left.get("critical")) and bool(right.get("critical")),
        })
        result.append(values)
    return result, None


def validate_adjudication(
    value: dict,
    root: Path,
    inputs_sha256: str,
    review_hashes: list[str],
    expected: dict[str, str],
    frame_counts: dict[str, int],
) -> dict[str, dict]:
    if value.get("schema") != "blindassist_ustrf_sensor_replay_r3_adjudication_v1" or value.get("method") != "independent_ai_adjudicator":
        raise ValueError("R3 adjudication schema/method mismatch")
    if value.get("reviewer_type") != "ai_model" or value.get("reviewer_role") not in {"gpt_adjudicator", "codex_adjudicator"}:
        raise ValueError("R3 adjudicator identity mismatch")
    if value.get("isolated_context") is not True or value.get("candidate_output_visible") is not False:
        raise ValueError("R3 adjudicator was not isolated from candidate output")
    if value.get("workflow_id") != "ustrf_event_review_v1" or value.get("input_sha256") != inputs_sha256:
        raise ValueError("R3 adjudication input binding mismatch")
    if value.get("prompt_sha256") != sha256(root / "reviewer_adjudication_prompt.txt"):
        raise ValueError("R3 adjudication prompt binding mismatch")
    if value.get("input_review_sha256s") != review_hashes:
        raise ValueError("R3 adjudication review binding mismatch")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.65 <= float(confidence) <= 1.0:
        raise ValueError("R3 adjudication confidence invalid")
    if value.get("abstained") is not False or value.get("candidate_alerts_viewed") is not False:
        raise ValueError("R3 adjudication abstained or viewed candidate")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError("R3 adjudication sources missing")
    by_id = {row["source_id"]: row for row in rows}
    if set(by_id) != set(expected):
        raise ValueError("R3 adjudication source identity mismatch")
    for source_id, row in by_id.items():
        if row.get("manifest_sha256") != expected[source_id]:
            raise ValueError(f"R3 adjudication manifest binding mismatch: {source_id}")
        if not isinstance(row.get("events"), list) or not isinstance(row.get("route_event_admitted"), bool):
            raise ValueError(f"R3 adjudication disposition invalid: {source_id}")
        for event_index, event in enumerate(row["events"]):
            values = [anchor(event, key) for key in ("onset_frame", "alertable_frame", "passed_or_cleared_frame", "end_frame")]
            if not (values[0] <= values[1] <= values[2] <= values[3] < frame_counts[source_id]):
                raise ValueError(f"R3 adjudication lifecycle invalid: {source_id} event {event_index}")
    return by_id


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
    frame_counts = {row["source_id"]: int(row["frame_count"]) for row in inputs["sources"]}
    review_a_path = args.root / "reviewer_a_raw.json"
    review_b_path = args.root / "reviewer_b_raw.json"
    raw_a = read_json(review_a_path)
    raw_b = read_json(review_b_path)
    inputs_sha256 = sha256(args.root / "review_inputs.json")
    for role, raw, prompt_path in (
        ("a", raw_a, args.root / "reviewer_a_prompt.txt"),
        ("b", raw_b, args.root / "reviewer_b_prompt.txt"),
    ):
        if raw.get("input_sha256") != inputs_sha256 or raw.get("prompt_sha256") != sha256(prompt_path):
            raise ValueError(f"reviewer {role} governance hash binding mismatch")
        if raw.get("reviewer_type") != "ai_model" or raw.get("workflow_id") != "ustrf_event_review_v1":
            raise ValueError(f"reviewer {role} governance identity mismatch")
        if raw.get("isolated_context") is not True or raw.get("candidate_output_visible") is not False:
            raise ValueError(f"reviewer {role} governance isolation mismatch")
    if raw_a.get("reviewer_role") == raw_b.get("reviewer_role"):
        raise ValueError("reviewer role labels must be distinct")
    a = validate_review(raw_a, "a", expected)
    b = validate_review(raw_b, "b", expected)
    review_hashes = [sha256(review_a_path), sha256(review_b_path)]
    adjudication_path = args.root / "reviewer_adjudication_raw.json"
    adjudication = None
    if adjudication_path.is_file():
        adjudication = validate_adjudication(
            read_json(adjudication_path),
            args.root,
            inputs_sha256,
            review_hashes,
            expected,
            frame_counts,
        )
    sources = []
    adjudication_required = False
    adjudication_used = False
    for source_id in expected:
        first, second = a[source_id], b[source_id]
        admitted = first["route_valid"] is True and second["route_valid"] is True
        consensus_events = []
        reason = "both reviewers admitted route/event sequence"
        if admitted:
            consensus_events, disagreement = build_consensus_events(first, second, args.anchor_tolerance_frames, frame_counts[source_id])
            if disagreement is not None:
                adjudication_required = True
                if adjudication is None:
                    admitted = False
                    reason = f"{disagreement}; independent adjudication missing"
                else:
                    resolved = adjudication[source_id]
                    admitted = resolved["route_event_admitted"] is True and resolved.get("route_valid") is True and bool(resolved["events"])
                    consensus_events = resolved["events"] if admitted else []
                    reason = resolved.get("reason", f"independent adjudicator resolved {disagreement}")
                    adjudication_used = True
            elif not consensus_events:
                admitted = False
                reason = "both reviewers found no consensus event"
        else:
            reason = f"route rejected or abstained: a={first['route_valid']} b={second['route_valid']}"
        if not admitted:
            consensus_events = []
        sources.append({"source_id": source_id, "route_event_admitted": admitted, "source_count_credit": 1 if admitted else 0, "events": consensus_events, "reason": reason})
    admitted_source_count = sum(row["source_count_credit"] for row in sources)
    all_admitted = admitted_source_count >= 3 and all(row["route_event_admitted"] for row in sources)
    report = {
        "schema": "blindassist_ustrf_sensor_replay_r3_review_consensus_v1",
        "workflow_id": "ustrf_r3_complete_sequence_two_model_review_v1",
        "review_inputs_sha256": sha256(args.root / "review_inputs.json"),
        "reviewer_a_sha256": sha256(review_a_path),
        "reviewer_b_sha256": sha256(review_b_path),
        "adjudication_sha256": sha256(adjudication_path) if adjudication_used else None,
        "candidate_alerts_visible_to_reviewers": False,
        "sources": sources,
        "admitted_source_count": admitted_source_count,
        "minimum_admitted_sources_met": admitted_source_count >= 3,
        "all_sources_admitted": all_admitted,
        "event_truth_authority": all_admitted,
        "third_model_adjudication_required": adjudication_required,
        "third_model_adjudication_used": adjudication_used,
        "production_authority": False,
    }
    write_json(args.output, report)
    print(json.dumps({"all_sources_admitted": all_admitted, "sources": [{"source_id": row["source_id"], "admitted": row["route_event_admitted"], "events": len(row["events"])} for row in sources]}))
    return 0 if all_admitted else 3


if __name__ == "__main__":
    raise SystemExit(main())
