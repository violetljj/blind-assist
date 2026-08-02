from __future__ import annotations

"""Fail-close the two independent P0 causal-anchor reviews.

This is deliberately a JSON-only gate: it never opens an RGB, mask, scene
fact, model output, oracle output, or feedback trace.  Its only authority is
to verify that two independently packaged blind submissions cover exactly the
same fixed anchors and to record whether the actionability construct was
consistent enough to authorize P1 review-packet generation.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    ACTION_REVIEW_SCHEMA,
    ADMISSION_PASSED_STATUSES,
    ADMISSION_RECONCILIATION_SCHEMA,
    KNOWNNESS,
    P0_ANCHOR_AGREEMENT_SCHEMA,
    PROTOCOL_ID,
    THREE_STATE,
    read_json,
    sha256_file,
    sha256_json,
)
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA
from .prepare_p0_review_packets import PACKET_SCHEMA, PRIVATE_MAP_SCHEMA


P0_PASSED_STATUS = "P0_ANCHOR_CONSISTENCY_PASSED"
P0_STOP_STATUS = "STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED"


class P0AgreementError(ValueError):
    """Raised for malformed, substituted, or disclosure-unsafe inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P0AgreementError(message)


def _screening_index(cohort: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    _require(cohort.get("schema_version") == SCREENING_COHORT_SCHEMA, "screening cohort schema mismatch")
    _require(cohort.get("protocol_id") == PROTOCOL_ID, "screening cohort protocol mismatch")
    _require(cohort.get("status") == "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", "screening cohort is not output-blind and window-frozen")
    _require(cohort.get("candidate_outputs_opened") is False, "screening cohort records forbidden output access")
    _require(cohort.get("final_event_facts_frozen") is False, "screening cohort has an invalid event-fact state")
    items = cohort.get("items")
    _require(isinstance(items, list) and len(items) == 48, "screening cohort must have exactly 48 items")
    indexed: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    for item in items:
        _require(isinstance(item, dict), "screening cohort item must be an object")
        event_id, session_id = item.get("screening_event_id"), item.get("source_session_id")
        window = item.get("source_window")
        _require(isinstance(event_id, str) and event_id and event_id not in indexed, "screening cohort event identity is invalid")
        _require(isinstance(session_id, str) and session_id and session_id not in sessions, "screening cohort is not session-disjoint")
        _require(isinstance(window, dict), f"{event_id}: missing source window")
        anchors = window.get("p0_anchor_offsets")
        count = window.get("frame_count")
        _require(isinstance(count, int) and count >= 20, f"{event_id}: invalid frame count")
        _require(isinstance(anchors, list) and len(anchors) == 4 and anchors == sorted(set(anchors)), f"{event_id}: invalid P0 anchors")
        _require(all(isinstance(anchor, int) and 0 <= anchor < count for anchor in anchors), f"{event_id}: P0 anchor outside window")
        indexed[event_id] = item
        sessions.add(session_id)
    return indexed, sha256_json(cohort)


def _validate_packet(
    packet: dict[str, Any], *, role: str, expected_map: dict[str, dict[str, Any]],
    cohort: dict[str, dict[str, Any]],
) -> None:
    _require(packet.get("schema_version") == PACKET_SCHEMA and packet.get("protocol_id") == PROTOCOL_ID, f"{role}: packet schema/protocol mismatch")
    _require(packet.get("reviewer_role") == role and packet.get("status") == "P0_CAUSAL_RGB_REVIEW_PENDING", f"{role}: packet role/state mismatch")
    disclosures = packet.get("disclosures")
    _require(isinstance(disclosures, dict), f"{role}: missing packet disclosures")
    required_disclosures = {
        "model_or_oracle_output_visible": False,
        "source_mask_visible": False,
        "source_session_or_event_identity_visible": False,
        "screening_stratum_or_bucket_visible": False,
        "other_reviewer_visible": False,
        "one_item_per_anchor": True,
        "future_frames_visible_in_item": False,
    }
    _require(all(disclosures.get(key) is value for key, value in required_disclosures.items()), f"{role}: unsafe packet disclosure")
    items = packet.get("items")
    _require(isinstance(items, list) and len(items) == 192, f"{role}: packet item count mismatch")
    observed: set[str] = set()
    for item in items:
        _require(isinstance(item, dict), f"{role}: malformed packet item")
        opaque_id, anchor = item.get("review_item_id"), item.get("current_frame_ordinal")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in observed, f"{role}: packet opaque identity mismatch")
        mapping = expected_map[opaque_id]
        event_id = mapping.get("screening_event_id")
        _require(event_id in cohort and mapping.get("source_session_id") == cohort[event_id]["source_session_id"], f"{role}: private map event/session mismatch")
        _require(anchor == mapping.get("anchor_frame_index"), f"{role}: packet anchor does not match its private binding")
        _require(anchor in cohort[event_id]["source_window"]["p0_anchor_offsets"], f"{role}: packet anchor not frozen in cohort")
        causal = item.get("causal_rgb_frames")
        _require(isinstance(causal, list) and len(causal) == anchor + 1 and all(isinstance(value, str) for value in causal), f"{role}: non-causal packet frames")
        observed.add(opaque_id)
    _require(observed == set(expected_map), f"{role}: private map/packet coverage mismatch")


def _validate_private_map(
    private: dict[str, Any], *, screening_sha: str, admission_sha: str, packet_a_sha: str, packet_b_sha: str,
    cohort: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(private.get("schema_version") == PRIVATE_MAP_SCHEMA and private.get("protocol_id") == PROTOCOL_ID, "private map schema/protocol mismatch")
    _require(private.get("status") == "PRIVATE_REVIEW_MAP_FROZEN_BEFORE_SUBMISSIONS", "private map state mismatch")
    _require(private.get("screening_cohort_sha256") == screening_sha, "private map cohort binding mismatch")
    _require(private.get("admission_receipt_sha256") == admission_sha, "private map admission binding mismatch")
    _require(private.get("packet_a_sha256") == packet_a_sha and private.get("packet_b_sha256") == packet_b_sha, "private map packet binding mismatch")
    maps: list[dict[str, dict[str, Any]]] = []
    for field, prefix in (("reviewer_a_map", "a-"), ("reviewer_b_map", "b-")):
        value = private.get(field)
        _require(isinstance(value, dict) and len(value) == 192, f"private map {field} coverage mismatch")
        _require(all(isinstance(key, str) and key.startswith(prefix) and isinstance(item, dict) for key, item in value.items()), f"private map {field} identity mismatch")
        seen_pairs: set[tuple[str, int]] = set()
        for opaque_id, mapping in value.items():
            event_id, anchor = mapping.get("screening_event_id"), mapping.get("anchor_frame_index")
            _require(event_id in cohort and isinstance(anchor, int), f"private map {field} has invalid event/anchor")
            _require(mapping.get("source_session_id") == cohort[event_id]["source_session_id"], f"private map {field} source-session mismatch")
            _require(anchor in cohort[event_id]["source_window"]["p0_anchor_offsets"], f"private map {field} anchor is not frozen")
            _require((event_id, anchor) not in seen_pairs, f"private map {field} duplicates an anchor")
            seen_pairs.add((event_id, anchor))
        expected_pairs = {(event_id, anchor) for event_id, item in cohort.items() for anchor in item["source_window"]["p0_anchor_offsets"]}
        _require(seen_pairs == expected_pairs, f"private map {field} does not cover all frozen anchors")
        maps.append(value)
    return maps[0], maps[1]


def _validate_admission(admission: dict[str, Any], screening_sha: str) -> str:
    _require(admission.get("protocol_id") == PROTOCOL_ID, "data admission protocol mismatch")
    status = admission.get("status")
    _require(status in ADMISSION_PASSED_STATUSES, "data admission did not pass")
    _require(admission.get("screening_cohort_sha256") == screening_sha, "data admission cohort binding mismatch")
    _require(admission.get("candidate_outputs_opened") is False, "data admission records forbidden output access")
    if status == "EVAL_VALIDITY_DATA_ADMISSION_PASSED":
        _require(admission.get("schema_version") == "blindassist.eval_validity_r0.data_admission_receipt.v1", "direct data admission schema mismatch")
    else:
        checks, evidence = admission.get("checks"), admission.get("evidence")
        _require(admission.get("schema_version") == ADMISSION_RECONCILIATION_SCHEMA, "manual pHash admission schema mismatch")
        _require(isinstance(checks, dict) and checks.get("p_hash_manual_all_cases_distinct") is True, "manual pHash admission checks mismatch")
        _require(isinstance(evidence, dict) and isinstance(evidence.get("held_admission_receipt_sha256"), str) and isinstance(evidence.get("p_hash_manual_resolution_sha256"), str), "manual pHash admission lineage mismatch")
    return sha256_json(admission)


def _index_submission(
    review: dict[str, Any], *, role: str, screening_sha: str, expected_map: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], dict[str, str]]:
    required = {
        "schema_version", "protocol_id", "reviewer_role", "screening_cohort_sha256", "isolated_context",
        "other_review_visible_before_submission", "model_or_oracle_output_visible", "items",
    }
    _require(set(review) == required, f"{role}: submission has unsupported or missing fields")
    _require(review.get("schema_version") == ACTION_REVIEW_SCHEMA and review.get("protocol_id") == PROTOCOL_ID, f"{role}: submission schema/protocol mismatch")
    _require(review.get("reviewer_role") == role and review.get("screening_cohort_sha256") == screening_sha, f"{role}: submission role/cohort mismatch")
    _require(review.get("isolated_context") is True and review.get("other_review_visible_before_submission") is False and review.get("model_or_oracle_output_visible") is False, f"{role}: reviewer isolation/output disclosure failure")
    items = review.get("items")
    _require(isinstance(items, list) and len(items) == len(expected_map), f"{role}: submission item coverage mismatch")
    indexed: dict[tuple[str, int], dict[str, str]] = {}
    seen_ids: set[str] = set()
    for item in items:
        _require(isinstance(item, dict) and set(item) == {"review_item_id", "anchor"}, f"{role}: malformed submission item")
        opaque_id, anchor = item.get("review_item_id"), item.get("anchor")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in seen_ids, f"{role}: submission opaque identity mismatch")
        _require(isinstance(anchor, dict) and set(anchor) == {"frame_index", "reminder_now", "cleared", "knownness"}, f"{role}: malformed anchor response")
        binding = expected_map[opaque_id]
        _require(anchor.get("frame_index") == binding["anchor_frame_index"], f"{role}: anchor response identity mismatch")
        _require(anchor.get("reminder_now") in THREE_STATE and anchor.get("cleared") in THREE_STATE and anchor.get("knownness") in KNOWNNESS, f"{role}: response value outside ontology")
        pair = (str(binding["screening_event_id"]), int(binding["anchor_frame_index"]))
        _require(pair not in indexed, f"{role}: duplicate resolved anchor")
        indexed[pair] = {key: str(anchor[key]) for key in ("reminder_now", "cleared", "knownness")}
        seen_ids.add(opaque_id)
    _require(seen_ids == set(expected_map), f"{role}: submission does not cover each opaque item")
    return indexed


def finalize_p0(
    *, screening_cohort: dict[str, Any], admission_receipt: dict[str, Any], private_map: dict[str, Any], packet_a: dict[str, Any], packet_b: dict[str, Any],
    review_a: dict[str, Any], review_b: dict[str, Any], packet_a_sha256: str, packet_b_sha256: str,
    review_a_sha256: str, review_b_sha256: str,
) -> dict[str, Any]:
    cohort, screening_sha = _screening_index(screening_cohort)
    admission_sha = _validate_admission(admission_receipt, screening_sha)
    map_a, map_b = _validate_private_map(
        private_map, screening_sha=screening_sha, admission_sha=admission_sha,
        packet_a_sha=packet_a_sha256, packet_b_sha=packet_b_sha256, cohort=cohort,
    )
    _validate_packet(packet_a, role="ACTION_REVIEW_A", expected_map=map_a, cohort=cohort)
    _validate_packet(packet_b, role="ACTION_REVIEW_B", expected_map=map_b, cohort=cohort)
    indexed_a = _index_submission(review_a, role="ACTION_REVIEW_A", screening_sha=screening_sha, expected_map=map_a)
    indexed_b = _index_submission(review_b, role="ACTION_REVIEW_B", screening_sha=screening_sha, expected_map=map_b)

    total = 0
    matches: Counter[str] = Counter()
    unknown = 0
    sequence_matches = 0
    consensus_items: list[dict[str, Any]] = []
    for event_id, item in sorted(cohort.items()):
        same_sequence = True
        anchors: list[dict[str, Any]] = []
        for frame in item["source_window"]["p0_anchor_offsets"]:
            left, right = indexed_a[(event_id, frame)], indexed_b[(event_id, frame)]
            total += 1
            exact = all(left[key] == right[key] for key in ("reminder_now", "cleared", "knownness"))
            for key in ("reminder_now", "cleared", "knownness"):
                matches[key] += int(left[key] == right[key])
            same_sequence = same_sequence and exact
            unresolved = (not exact) or "UNKNOWN" in (left["reminder_now"], left["cleared"], left["knownness"])
            unknown += int(unresolved)
            anchors.append({
                "anchor_frame_index": frame,
                "consensus": left if not unresolved else {"reminder_now": "UNKNOWN", "cleared": "UNKNOWN", "knownness": "UNKNOWN"},
                "resolved": not unresolved,
            })
        sequence_matches += int(same_sequence)
        consensus_items.append({"screening_event_id": event_id, "anchors": anchors})
    metrics = {
        "reminder_now_exact_agreement": matches["reminder_now"] / total,
        "cleared_exact_agreement": matches["cleared"] / total,
        "knownness_exact_agreement": matches["knownness"] / total,
        "parent_event_sequence_exact_agreement": sequence_matches / len(cohort),
        "unknown_anchor_burden": unknown / total,
        "anchor_count": total,
        "unresolved_anchor_count": unknown,
    }
    passed = all((
        metrics["reminder_now_exact_agreement"] == 1.0,
        metrics["cleared_exact_agreement"] == 1.0,
        metrics["knownness_exact_agreement"] == 1.0,
        metrics["parent_event_sequence_exact_agreement"] == 1.0,
        metrics["unknown_anchor_burden"] == 0.0,
    ))
    return {
        "schema_version": P0_ANCHOR_AGREEMENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": P0_PASSED_STATUS if passed else P0_STOP_STATUS,
        "screening_cohort_sha256": screening_sha,
        "admission_receipt_sha256": admission_sha,
        "candidate_outputs_opened": False,
        "review_evidence": {
            "packet_a_sha256": packet_a_sha256, "packet_b_sha256": packet_b_sha256,
            "review_a_sha256": review_a_sha256, "review_b_sha256": review_b_sha256,
            "reviewers_isolated": True, "model_or_oracle_output_visible": False,
            "source_mask_visible": False, "other_review_visible_before_submission": False,
        },
        "anchor_agreement": {"metrics": metrics, "passed": passed},
        "consensus_items": consensus_items,
        "next_required_gate": (
            "Prepare two new, isolated P1 full-event causal RGB packets. Do not materialize any model, truth or oracle trace."
            if passed else "HOLD. Do not prepare P1, inspect outputs, alter labels, replace anchors, or materialize any trace."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--packet-a", type=Path, required=True)
    parser.add_argument("--packet-b", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = finalize_p0(
        screening_cohort=read_json(args.screening_cohort), admission_receipt=read_json(args.admission_receipt), private_map=read_json(args.private_map),
        packet_a=read_json(args.packet_a), packet_b=read_json(args.packet_b),
        review_a=read_json(args.review_a), review_b=read_json(args.review_b),
        packet_a_sha256=sha256_file(args.packet_a), packet_b_sha256=sha256_file(args.packet_b),
        review_a_sha256=sha256_file(args.review_a), review_b_sha256=sha256_file(args.review_b),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} anchors={result['anchor_agreement']['metrics']['anchor_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
