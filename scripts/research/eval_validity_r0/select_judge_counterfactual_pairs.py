"""Select frozen counterfactual pairs from a complete label-blind candidate ledger.

The candidate ledger is built only from selection metadata available after the
review bundle is sealed: YOLO box similarity, geometry/scale/position/visibility
similarity and a fixed sampling slot.  This command reads the review seal only
for its canonical bundle hash and never parses primitive or derived labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_json
from .judge_audit import PAIR_SCHEMA, _contract
from .prepare_judge_burned_pilot import FREEZE_SCHEMA
from .seal_judge_review_bundle import SEAL_SCHEMA


CANDIDATE_SCHEMA = "blindassist.eval_validity_r0.judge_pair_candidate_ledger.v1"
SELECTION_FIELDS = ["yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot"]
FORBIDDEN_ITEM_TOKENS = ("phase", "motion", "actionability", "physical_condition", "path_relation", "primitive", "derived", "label")


class PairSelectionError(ValueError):
    """Raised when a pair universe or review boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairSelectionError(message)


def _validate_freeze(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require(freeze.get("schema_version") == FREEZE_SCHEMA, "pilot freeze schema mismatch")
    _require(freeze.get("protocol_id") == PROTOCOL_ID, "pilot freeze protocol mismatch")
    _require(freeze.get("formal_denominator_inclusion") is False, "formal denominator is open for pilot freeze")
    items = freeze.get("items")
    _require(isinstance(items, list) and 8 <= len(items) <= 12, "pilot freeze event count is outside 8-12")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        event_id = item.get("pilot_event_id") if isinstance(item, dict) else None
        _require(isinstance(event_id, str) and event_id not in result, "pilot freeze event identity is invalid")
        frames = item.get("frame_indices")
        _require(isinstance(frames, list) and frames == list(range(len(frames))), f"{event_id}: pilot frame indices must be local contiguous ordinals")
        result[event_id] = item
    return result


def _validate_seal(seal: dict[str, Any], freeze: dict[str, Any]) -> str:
    _require(seal.get("schema_version") == SEAL_SCHEMA, "review seal schema mismatch")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "review seal protocol mismatch")
    _require(seal.get("status") == "PRIMITIVE_REVIEWS_SEALED_BEFORE_PAIR_SELECTION", "review bundle is not sealed")
    _require(seal.get("pilot_freeze_sha256") == sha256_json(freeze), "review seal is bound to a different pilot freeze")
    bundle_hash = seal.get("review_bundle_sha256")
    _require(isinstance(bundle_hash, str) and len(bundle_hash) == 64, "review bundle hash is invalid")
    for field in ("primitive_labels_opened_to_pair_builder", "derived_labels_opened_to_pair_builder", "reviewed_event_phase_opened_to_pair_builder", "reviewed_motion_relation_opened_to_pair_builder"):
        _require(seal.get(field) is False, f"review seal exposes forbidden label field: {field}")
    _require(seal.get("pair_selection_access") == "HASH_ONLY", "pair builder is not restricted to review hash only")
    return bundle_hash


def _validate_candidates(value: dict[str, Any], events: dict[str, dict[str, Any]], freeze: dict[str, Any], review_bundle_hash: str) -> list[dict[str, Any]]:
    _require(value.get("schema_version") == CANDIDATE_SCHEMA, "candidate pair ledger schema mismatch")
    _require(value.get("protocol_id") == PROTOCOL_ID, "candidate pair ledger protocol mismatch")
    _require(value.get("pilot_freeze_sha256") == sha256_json(freeze), "candidate pair ledger is bound to a different pilot freeze")
    _require(value.get("review_bundle_sha256") == review_bundle_hash, "candidate pair ledger is bound to a different sealed review bundle")
    _require(value.get("yolo_role") == "SELECTION_ONLY", "candidate pair ledger YOLO role is not selection-only")
    for field in ("yolo_visible_to_reviewers", "yolo_used_for_truth", "primitive_labels_visible_to_pair_builder", "derived_labels_visible_to_pair_builder", "reviewed_event_phase_visible_to_pair_builder", "reviewed_motion_relation_visible_to_pair_builder"):
        _require(value.get(field) is False, f"candidate pair ledger exposes forbidden field: {field}")
    _require(value.get("enumeration_complete") is True, "candidate pair ledger does not assert complete eligible-pair enumeration")
    _require(value.get("selection_time_slot_source") == "fixed_sampling_slot", "candidate pair ledger time-slot source is not frozen")
    _require(value.get("selection_fields") == SELECTION_FIELDS, "candidate pair ledger selection fields are not frozen")
    items = value.get("items")
    _require(isinstance(items, list), "candidate pair ledger items are missing")
    required = {"candidate_id", "event_a_id", "event_b_id", "yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot", "comparison_frame_index_a", "comparison_frame_index_b"}
    seen_candidates: set[str] = set()
    seen_pairs: set[tuple[str, str, int]] = set()
    for index, item in enumerate(items):
        where = f"candidate pair {index}"
        _require(isinstance(item, dict) and set(item) == required, f"{where}: incomplete or label-bearing candidate fields")
        _require(not any(any(token in key.lower() for token in FORBIDDEN_ITEM_TOKENS) for key in item), f"{where}: candidate item contains reviewer-derived terminology")
        candidate_id = item["candidate_id"]
        a_id, b_id, slot = item["event_a_id"], item["event_b_id"], item["selection_time_slot"]
        _require(isinstance(candidate_id, str) and candidate_id and candidate_id not in seen_candidates, f"{where}: invalid/duplicate candidate_id")
        _require(a_id in events and b_id in events and a_id != b_id and a_id < b_id, f"{where}: event pair is invalid or not canonicalized")
        _require(isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0, f"{where}: invalid selection time slot")
        _require(slot < len(events[a_id]["frame_indices"]) and slot < len(events[b_id]["frame_indices"]), f"{where}: selection time slot outside event")
        _require(item["comparison_frame_index_a"] == events[a_id]["frame_indices"][slot] and item["comparison_frame_index_b"] == events[b_id]["frame_indices"][slot], f"{where}: comparison frames are not derived from selection slot")
        for field in ("yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity"):
            value_number = item[field]
            _require(isinstance(value_number, (int, float)) and not isinstance(value_number, bool) and 0 <= value_number <= 1, f"{where}: {field} outside [0,1]")
        pair_key = (a_id, b_id, slot)
        _require(pair_key not in seen_pairs, f"{where}: duplicate event/slot candidate")
        seen_candidates.add(candidate_id)
        seen_pairs.add(pair_key)
    return items


def _ordering_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (-float(item["yolo_box_similarity"]), -float(item["distance_scale_similarity"]), -float(item["position_similarity"]), -float(item["visibility_similarity"]), item["event_a_id"], item["event_b_id"], item["selection_time_slot"])


def select_pairs(*, contract: dict[str, Any], freeze: dict[str, Any], seal: dict[str, Any], candidates: dict[str, Any], output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite pair manifest: {output}")
    contract = _contract(contract)
    events = _validate_freeze(freeze)
    review_bundle_hash = _validate_seal(seal, freeze)
    candidate_items = _validate_candidates(candidates, events, freeze, review_bundle_hash)
    candidate_universe_hash = sha256_json({"schema_version": candidates["schema_version"], "protocol_id": candidates["protocol_id"], "items": candidate_items})
    eligible = [
        item for item in candidate_items
        if item["yolo_box_similarity"] >= contract["yolo_box_similarity_threshold"]
        and item["distance_scale_similarity"] >= contract["minimum_pair_distance_scale_similarity"]
        and item["position_similarity"] >= contract["minimum_pair_position_similarity"]
        and item["visibility_similarity"] >= contract["minimum_pair_visibility_similarity"]
    ]
    eligible.sort(key=_ordering_key)
    minimum, maximum = contract["minimum_counterfactual_pairs"], contract["maximum_counterfactual_pairs"]
    selected = eligible[:maximum]
    builder = {
        "stage": contract["counterfactual_pair_policy"]["selection_stage"],
        "review_bundle_sealed": True,
        "review_bundle_sha256": review_bundle_hash,
        "yolo_role": contract["counterfactual_pair_policy"]["yolo_role"],
        "yolo_visible_to_reviewers": False,
        "yolo_used_for_truth": False,
        "primitive_labels_visible_to_pair_builder": False,
        "derived_labels_visible_to_pair_builder": False,
        "reviewed_event_phase_visible_to_pair_builder": False,
        "reviewed_motion_relation_visible_to_pair_builder": False,
        "pair_freeze_rule_version": contract["counterfactual_pair_policy"]["version"],
        "selection_fields": contract["counterfactual_pair_policy"]["selection_fields"],
        "selection_time_slot_source": contract["counterfactual_pair_policy"]["selection_time_slot_source"],
        "ordering_rule": contract["counterfactual_pair_policy"]["ordering_rule"],
        "candidate_pair_universe_sha256": candidate_universe_hash,
        "candidate_pair_universe_count": len(candidate_items),
        "eligible_pair_count_before_label_access": len(eligible),
        "pair_count_frozen": True,
        "below_minimum_terminal": contract["counterfactual_pair_policy"]["below_minimum_terminal"],
    }
    items = [{
        "pair_id": f"pair-{rank:03d}",
        "pair_rank": rank,
        "event_a_id": item["event_a_id"],
        "event_b_id": item["event_b_id"],
        "yolo_box_similarity": item["yolo_box_similarity"],
        "distance_scale_similarity": item["distance_scale_similarity"],
        "position_similarity": item["position_similarity"],
        "visibility_similarity": item["visibility_similarity"],
        "selection_time_slot": item["selection_time_slot"],
        "comparison_frame_index_a": item["comparison_frame_index_a"],
        "comparison_frame_index_b": item["comparison_frame_index_b"],
    } for rank, item in enumerate(selected, start=1)]
    result = {"schema_version": PAIR_SCHEMA, "protocol_id": PROTOCOL_ID, "pair_builder": builder, "items": items}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "PAIR_MANIFEST_FROZEN" if len(eligible) >= minimum else "NOT_EVALUABLE", "eligible_pair_count": len(eligible), "selected_pair_count": len(items), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pilot-freeze", type=Path, required=True)
    parser.add_argument("--review-seal", type=Path, required=True)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_pairs(
        contract=read_json(args.contract),
        freeze=read_json(args.pilot_freeze),
        seal=read_json(args.review_seal),
        candidates=read_json(args.candidate_ledger),
        output=args.output,
    )
    print(f"status={result['status']} eligible_pair_count={result['eligible_pair_count']} selected_pair_count={result['selected_pair_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
