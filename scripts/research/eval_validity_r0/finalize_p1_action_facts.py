from __future__ import annotations

"""Fail-close two fresh P1 full-event action-fact reviews before trace access.

The receipt produced here is deliberately not yet a semantic scene-fact
cohort: it records only the agreed RGB-derived alertable and passed-clear
intervals.  A later source-mask-only scene-fact finalization may merge those
facts, but no model, oracle or feedback trace is read here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .common import KNOWNNESS, P0_ANCHOR_AGREEMENT_SCHEMA, P1_ACTION_FACTS_SCHEMA, P1_ACTION_REVIEW_SCHEMA, PROTOCOL_ID, read_json, sha256_file, sha256_json
from .finalize_p0_anchor_agreement import P0_PASSED_STATUS, _screening_index, _validate_admission
from .prepare_p1_review_packets import PACKET_SCHEMA, PRIVATE_MAP_SCHEMA


P1_PASSED_STATUS = "P1_ACTION_FACTS_FROZEN_AFTER_P0_CONSISTENCY"
P1_STOP_STATUS = "STOP_FULL_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED"


class P1AgreementError(ValueError):
    """Raised for malformed, substituted, or disclosure-unsafe P1 inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P1AgreementError(message)


def _validate_p0(p0: dict[str, Any], *, cohort_sha: str, admission_sha: str) -> str:
    _require(p0.get("schema_version") == P0_ANCHOR_AGREEMENT_SCHEMA and p0.get("protocol_id") == PROTOCOL_ID, "P0 receipt schema/protocol mismatch")
    _require(p0.get("status") == P0_PASSED_STATUS, "P0 did not pass")
    _require(p0.get("screening_cohort_sha256") == cohort_sha and p0.get("admission_receipt_sha256") == admission_sha, "P0 cohort/admission binding mismatch")
    _require(p0.get("candidate_outputs_opened") is False, "P0 records forbidden output access")
    agreement = p0.get("anchor_agreement")
    _require(isinstance(agreement, dict) and agreement.get("passed") is True, "P0 agreement is not passed")
    return sha256_json(p0)


def _p0_anchor_consensus(p0: dict[str, Any], cohort: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[str, str]]]:
    items = p0.get("consensus_items")
    _require(isinstance(items, list) and len(items) == len(cohort), "P0 consensus item coverage mismatch")
    result: dict[str, dict[int, dict[str, str]]] = {}
    for item in items:
        _require(isinstance(item, dict), "P0 consensus item malformed")
        event_id, anchors = item.get("screening_event_id"), item.get("anchors")
        _require(event_id in cohort and event_id not in result and isinstance(anchors, list), "P0 consensus event identity mismatch")
        per_anchor: dict[int, dict[str, str]] = {}
        for anchor in anchors:
            _require(isinstance(anchor, dict), "P0 consensus anchor malformed")
            frame, value = anchor.get("anchor_frame_index"), anchor.get("consensus")
            _require(frame in cohort[event_id]["source_window"]["p0_anchor_offsets"] and frame not in per_anchor, "P0 consensus anchor identity mismatch")
            _require(anchor.get("resolved") is True and isinstance(value, dict), "P0 passed receipt contains unresolved anchor")
            _require(value.get("knownness") == "KNOWN" and value.get("reminder_now") in {"YES", "NO"} and value.get("cleared") in {"YES", "NO"}, "P0 passed receipt contains invalid anchor fact")
            per_anchor[frame] = {key: value[key] for key in ("reminder_now", "cleared", "knownness")}
        _require(set(per_anchor) == set(cohort[event_id]["source_window"]["p0_anchor_offsets"]), "P0 consensus anchor coverage mismatch")
        result[event_id] = per_anchor
    _require(set(result) == set(cohort), "P0 consensus event coverage mismatch")
    return result


def _matches_p0_anchors(fact: dict[str, Any], anchors: dict[int, dict[str, str]]) -> bool:
    reminder, cleared = fact["reminder_now_interval"], fact["cleared_interval"]
    for frame, anchor in anchors.items():
        reminder_contains = reminder is not None and reminder[0] <= frame <= reminder[1]
        cleared_contains = cleared is not None and cleared[0] <= frame <= cleared[1]
        if (anchor["reminder_now"] == "YES") != reminder_contains:
            return False
        if (anchor["cleared"] == "YES") != cleared_contains:
            return False
    return True


def _validate_private(
    private: dict[str, Any], *, cohort: dict[str, dict[str, Any]], cohort_sha: str, admission_sha: str,
    p0_sha: str, packet_a_sha: str, packet_b_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(private.get("schema_version") == PRIVATE_MAP_SCHEMA and private.get("protocol_id") == PROTOCOL_ID, "P1 private map schema/protocol mismatch")
    _require(private.get("status") == "P1_PRIVATE_REVIEW_MAP_FROZEN_BEFORE_SUBMISSIONS", "P1 private map state mismatch")
    _require(private.get("screening_cohort_sha256") == cohort_sha and private.get("admission_receipt_sha256") == admission_sha and private.get("p0_anchor_agreement_sha256") == p0_sha, "P1 private map lineage binding mismatch")
    _require(private.get("packet_a_sha256") == packet_a_sha and private.get("packet_b_sha256") == packet_b_sha, "P1 private map packet binding mismatch")
    maps: list[dict[str, dict[str, Any]]] = []
    for field, prefix in (("reviewer_a_map", "p1a-"), ("reviewer_b_map", "p1b-")):
        value = private.get(field)
        _require(isinstance(value, dict) and len(value) == 48, f"P1 {field} coverage mismatch")
        _require(all(isinstance(key, str) and key.startswith(prefix) and isinstance(item, dict) for key, item in value.items()), f"P1 {field} opaque identity mismatch")
        events: set[str] = set()
        for mapping in value.values():
            event_id, count = mapping.get("screening_event_id"), mapping.get("frame_count")
            _require(event_id in cohort and event_id not in events and isinstance(count, int), f"P1 {field} event binding mismatch")
            _require(mapping.get("source_session_id") == cohort[event_id]["source_session_id"], f"P1 {field} session binding mismatch")
            _require(count == cohort[event_id]["source_window"]["frame_count"], f"P1 {field} frame count mismatch")
            events.add(event_id)
        _require(events == set(cohort), f"P1 {field} event coverage mismatch")
        maps.append(value)
    return maps[0], maps[1]


def _validate_packet(packet: dict[str, Any], *, role: str, expected_map: dict[str, dict[str, Any]]) -> None:
    _require(packet.get("schema_version") == PACKET_SCHEMA and packet.get("protocol_id") == PROTOCOL_ID, f"{role}: packet schema/protocol mismatch")
    _require(packet.get("reviewer_role") == role and packet.get("status") == "P1_FULL_EVENT_CAUSAL_RGB_REVIEW_PENDING", f"{role}: packet role/state mismatch")
    disclosures = packet.get("disclosures")
    expected_disclosures = {
        "model_or_oracle_output_visible": False,
        "source_mask_visible": False,
        "source_session_or_event_identity_visible": False,
        "screening_stratum_or_bucket_visible": False,
        "p0_review_or_consensus_visible": False,
        "other_reviewer_visible": False,
        "all_event_frames_visible_in_causal_order": True,
    }
    _require(isinstance(disclosures, dict) and all(disclosures.get(key) is value for key, value in expected_disclosures.items()), f"{role}: unsafe packet disclosure")
    items = packet.get("items")
    _require(isinstance(items, list) and len(items) == 48, f"{role}: packet item count mismatch")
    seen: set[str] = set()
    for item in items:
        _require(isinstance(item, dict), f"{role}: malformed packet item")
        opaque_id, count, assets = item.get("review_item_id"), item.get("frame_count"), item.get("causal_rgb_frames")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in seen, f"{role}: packet opaque identity mismatch")
        _require(count == expected_map[opaque_id]["frame_count"] and isinstance(assets, list) and len(assets) == count and all(isinstance(value, str) for value in assets), f"{role}: packet frame sequence mismatch")
        seen.add(opaque_id)
    _require(seen == set(expected_map), f"{role}: packet/private map coverage mismatch")


def _interval(value: Any, *, frame_count: int, where: str) -> list[int] | None:
    if value is None:
        return None
    _require(isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value), f"{where}: interval must be [first,last] or null")
    _require(0 <= value[0] <= value[1] < frame_count, f"{where}: interval outside event")
    return value


def _index_submission(
    review: dict[str, Any], *, role: str, cohort_sha: str, p0_sha: str, expected_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    required = {
        "schema_version", "protocol_id", "reviewer_role", "screening_cohort_sha256", "p0_anchor_agreement_sha256",
        "isolated_context", "reviewer_is_not_a_p0_reviewer", "other_review_visible_before_submission", "model_or_oracle_output_visible", "items",
    }
    _require(set(review) == required, f"{role}: submission has unsupported or missing fields")
    _require(review.get("schema_version") == P1_ACTION_REVIEW_SCHEMA and review.get("protocol_id") == PROTOCOL_ID, f"{role}: submission schema/protocol mismatch")
    _require(review.get("reviewer_role") == role and review.get("screening_cohort_sha256") == cohort_sha and review.get("p0_anchor_agreement_sha256") == p0_sha, f"{role}: submission lineage binding mismatch")
    _require(review.get("isolated_context") is True and review.get("reviewer_is_not_a_p0_reviewer") is True and review.get("other_review_visible_before_submission") is False and review.get("model_or_oracle_output_visible") is False, f"{role}: reviewer isolation/output disclosure failure")
    items = review.get("items")
    _require(isinstance(items, list) and len(items) == len(expected_map), f"{role}: submission item count mismatch")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        _require(isinstance(item, dict) and set(item) == {"review_item_id", "event_fact"}, f"{role}: malformed submission item")
        opaque_id, fact = item.get("review_item_id"), item.get("event_fact")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in result, f"{role}: opaque submission identity mismatch")
        _require(isinstance(fact, dict) and set(fact) == {"knownness", "reminder_now_interval", "cleared_interval"}, f"{role}: malformed event fact")
        knownness = fact.get("knownness")
        _require(knownness in KNOWNNESS, f"{role}: invalid knownness")
        frame_count = expected_map[opaque_id]["frame_count"]
        reminder = _interval(fact.get("reminder_now_interval"), frame_count=frame_count, where=f"{role}: reminder")
        cleared = _interval(fact.get("cleared_interval"), frame_count=frame_count, where=f"{role}: clearance")
        if knownness == "UNKNOWN":
            _require(reminder is None and cleared is None, f"{role}: UNKNOWN may not carry resolved intervals")
        elif reminder is None:
            _require(cleared is None, f"{role}: no-reminder event may not carry a clearance interval")
        else:
            _require(cleared is not None and cleared[0] > reminder[1], f"{role}: reminder must be followed by a passed-clear interval")
        result[opaque_id] = {"knownness": knownness, "reminder_now_interval": reminder, "cleared_interval": cleared}
    _require(set(result) == set(expected_map), f"{role}: submission coverage mismatch")
    return result


def finalize_p1(
    *, screening_cohort: dict[str, Any], admission_receipt: dict[str, Any], p0_agreement: dict[str, Any], private_map: dict[str, Any],
    packet_a: dict[str, Any], packet_b: dict[str, Any], review_a: dict[str, Any], review_b: dict[str, Any],
    packet_a_sha256: str, packet_b_sha256: str, review_a_sha256: str, review_b_sha256: str,
) -> dict[str, Any]:
    try:
        cohort, cohort_sha = _screening_index(screening_cohort)
    except ValueError as error:
        raise P1AgreementError(str(error)) from error
    try:
        admission_sha = _validate_admission(admission_receipt, cohort_sha)
    except ValueError as error:
        raise P1AgreementError(str(error)) from error
    p0_sha = _validate_p0(p0_agreement, cohort_sha=cohort_sha, admission_sha=admission_sha)
    p0_anchors = _p0_anchor_consensus(p0_agreement, cohort)
    map_a, map_b = _validate_private(
        private_map, cohort=cohort, cohort_sha=cohort_sha, admission_sha=admission_sha, p0_sha=p0_sha,
        packet_a_sha=packet_a_sha256, packet_b_sha=packet_b_sha256,
    )
    _validate_packet(packet_a, role="P1_FULL_EVENT_REVIEW_A", expected_map=map_a)
    _validate_packet(packet_b, role="P1_FULL_EVENT_REVIEW_B", expected_map=map_b)
    left = _index_submission(review_a, role="P1_FULL_EVENT_REVIEW_A", cohort_sha=cohort_sha, p0_sha=p0_sha, expected_map=map_a)
    right = _index_submission(review_b, role="P1_FULL_EVENT_REVIEW_B", cohort_sha=cohort_sha, p0_sha=p0_sha, expected_map=map_b)
    facts_a = {mapping["screening_event_id"]: left[opaque_id] for opaque_id, mapping in map_a.items()}
    facts_b = {mapping["screening_event_id"]: right[opaque_id] for opaque_id, mapping in map_b.items()}
    unresolved = 0
    items: list[dict[str, Any]] = []
    for event_id in sorted(cohort):
        equal = facts_a[event_id] == facts_b[event_id]
        known = facts_a[event_id]["knownness"] == "KNOWN" if equal else False
        p0_compatible = _matches_p0_anchors(facts_a[event_id], p0_anchors[event_id]) if equal and known else False
        resolved = equal and known and p0_compatible
        unresolved += int(not resolved)
        items.append({
            "screening_event_id": event_id,
            "resolved": resolved,
            "p0_anchor_compatible": p0_compatible,
            "alertable_interval_frames": facts_a[event_id]["reminder_now_interval"] if resolved else None,
            "passed_interval_frames": facts_a[event_id]["cleared_interval"] if resolved else None,
        })
    passed = unresolved == 0
    return {
        "schema_version": P1_ACTION_FACTS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": P1_PASSED_STATUS if passed else P1_STOP_STATUS,
        "screening_cohort_sha256": cohort_sha,
        "admission_receipt_sha256": admission_sha,
        "p0_anchor_agreement_sha256": p0_sha,
        "candidate_outputs_opened": False,
        "independent_full_review_evidence": {
            "packet_a_sha256": packet_a_sha256, "packet_b_sha256": packet_b_sha256,
            "review_a_sha256": review_a_sha256, "review_b_sha256": review_b_sha256,
            "reviewers_isolated": True, "reviewers_are_fresh_after_p0": True,
            "model_or_oracle_output_visible": False, "source_mask_visible": False,
            "agreement_passed": passed, "unknown_or_disagreement_event_count": unresolved,
        },
        "items": items,
        "next_required_gate": (
            "Freeze deterministic source-mask-only scene facts and the 12-12-12-12 semantic cohort before any trace materialization."
            if passed else "HOLD. Do not freeze scene/event facts, alter intervals, replace events, or materialize any trace."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--p0-agreement", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--packet-a", type=Path, required=True)
    parser.add_argument("--packet-b", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = finalize_p1(
        screening_cohort=read_json(args.screening_cohort), admission_receipt=read_json(args.admission_receipt), p0_agreement=read_json(args.p0_agreement),
        private_map=read_json(args.private_map), packet_a=read_json(args.packet_a), packet_b=read_json(args.packet_b),
        review_a=read_json(args.review_a), review_b=read_json(args.review_b),
        packet_a_sha256=sha256_file(args.packet_a), packet_b_sha256=sha256_file(args.packet_b),
        review_a_sha256=sha256_file(args.review_a), review_b_sha256=sha256_file(args.review_b),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} unresolved={result['independent_full_review_evidence']['unknown_or_disagreement_event_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
