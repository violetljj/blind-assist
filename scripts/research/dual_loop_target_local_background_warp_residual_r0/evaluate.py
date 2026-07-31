"""Truth-late evaluator and deterministic B ring selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from .common import (
    ABSTENTION_REASONS,
    DEADBAND_PER_S,
    IMPLEMENTATION_ID,
    MAX_COVERAGE_LOSS,
    MAX_SINGLE_CONTRIBUTION,
    MIN_CONTRIBUTING_EVENTS,
    MIN_EVENT_FINITE_PAIRS,
    MIN_EVENT_PAIR_COVERAGE,
    PROTOCOL_ID,
    RING_CONFIGS,
    contract_sha256,
    read_jsonl,
    sha256_file,
)


TRUTH_SCHEMA = "blindassist.target_local_warp_residual_truth.v1"
EVALUATION_SCHEMA = "blindassist.target_local_warp_residual_evaluation.v1"
PAIR_KEY_FIELDS = (
    "source_id",
    "session_id",
    "sequence_id",
    "previous_source_frame_id",
    "current_source_frame_id",
    "target_id",
    "track_epoch",
)


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in PAIR_KEY_FIELDS)


def _state(score: float | None) -> str | None:
    if score is None or not math.isfinite(score):
        return None
    if score > DEADBAND_PER_S:
        return "approach"
    if score < -DEADBAND_PER_S:
        return "receding"
    return "quasi-static"


def _wrong_signed(truth_state: str, predicted: str | None) -> bool:
    if predicted is None:
        return False
    if truth_state == "quasi-static":
        return predicted != "quasi-static"
    return predicted in {"approach", "receding"} and predicted != truth_state


def _event_score(values: list[float], denominator: int) -> tuple[float | None, bool, float]:
    coverage = len(values) / denominator if denominator else 0.0
    if len(values) < MIN_EVENT_FINITE_PAIRS or coverage < MIN_EVENT_PAIR_COVERAGE:
        return None, False, coverage
    values.sort()
    middle = len(values) // 2
    score = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    return float(score), True, coverage


def _truth_index(truth_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in truth_rows:
        if row.get("schema") not in {None, TRUTH_SCHEMA}:
            raise ValueError("unsupported truth schema")
        key = _pair_key(row)
        if key in index:
            raise ValueError(f"duplicate truth pair identity: {key}")
        if not row.get("truth_eligible", False):
            continue
        if row.get("truth_state") not in {"approach", "quasi-static", "receding"}:
            raise ValueError("truth state must be canonical")
        if row.get("parent_event_id") is None:
            raise ValueError("truth pair lacks parent_event_id")
        index[key] = row
    return index


def _summarize_events(rows: list[dict[str, Any]], truth: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        match = truth.get(_pair_key(row))
        if match is None:
            continue
        grouped[(str(row.get("session_id")), str(match["parent_event_id"]), str(row.get("target_id")))].append((row, match))
    event_rows: list[dict[str, Any]] = []
    for (session_id, event_id, target_id), pairs in sorted(grouped.items()):
        states = {str(truth_row["truth_state"]) for _, truth_row in pairs}
        if len(states) != 1:
            raise ValueError(f"truth state changed within event {event_id}")
        truth_state = next(iter(states))
        residual_values = [float(row["residual_rate_per_s"]) for row, _ in pairs if row.get("residual_rate_per_s") is not None and math.isfinite(float(row["residual_rate_per_s"]))]
        raw_values = [float(row["raw_rate_per_s"]) for row, _ in pairs if row.get("raw_rate_per_s") is not None and math.isfinite(float(row["raw_rate_per_s"]))]
        residual_score, residual_evaluable, residual_coverage = _event_score(residual_values, len(pairs))
        raw_score, raw_evaluable, raw_coverage = _event_score(raw_values, len(pairs))
        residual_state = _state(residual_score)
        raw_state = _state(raw_score)
        event_rows.append(
            {
                "session_id": session_id,
                "parent_event_id": event_id,
                "target_id": target_id,
                "truth_state": truth_state,
                "truth_eligible_pair_count": len(pairs),
                "residual_finite_pair_count": len(residual_values),
                "raw_finite_pair_count": len(raw_values),
                "residual_coverage": residual_coverage,
                "raw_coverage": raw_coverage,
                "residual_score_per_s": residual_score,
                "raw_score_per_s": raw_score,
                "residual_state": residual_state,
                "raw_state": raw_state,
                "residual_evaluable": residual_evaluable,
                "raw_evaluable": raw_evaluable,
                "residual_correct": bool(residual_evaluable and residual_state == truth_state),
                "raw_correct": bool(raw_evaluable and raw_state == truth_state),
                "residual_wrong_signed": bool(residual_evaluable and _wrong_signed(truth_state, residual_state)),
                "raw_wrong_signed": bool(raw_evaluable and _wrong_signed(truth_state, raw_state)),
            }
        )
    return event_rows


def _metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(events)
    residual_correct = sum(bool(row["residual_correct"]) for row in events)
    raw_correct = sum(bool(row["raw_correct"]) for row in events)
    residual_wrong = sum(bool(row["residual_wrong_signed"]) for row in events)
    raw_wrong = sum(bool(row["raw_wrong_signed"]) for row in events)
    residual_evaluable = sum(bool(row["residual_evaluable"]) for row in events)
    raw_evaluable = sum(bool(row["raw_evaluable"]) for row in events)
    gains = [row for row in events if row["residual_correct"] and not row["raw_correct"]]
    return {
        "truth_eligible_event_count": denominator,
        "residual_evaluable_event_count": residual_evaluable,
        "raw_evaluable_event_count": raw_evaluable,
        "residual_coverage": residual_evaluable / denominator if denominator else 0.0,
        "raw_coverage": raw_evaluable / denominator if denominator else 0.0,
        "coverage_loss": (raw_evaluable - residual_evaluable) / denominator if denominator else 0.0,
        "residual_correct_count": residual_correct,
        "raw_correct_count": raw_correct,
        "paired_event_gain_count": residual_correct - raw_correct,
        "paired_event_gain": (residual_correct - raw_correct) / denominator if denominator else 0.0,
        "residual_wrong_signed_count": residual_wrong,
        "raw_wrong_signed_count": raw_wrong,
        "positive_contribution_event_count": len(gains),
        "positive_contribution_fraction": len(gains) / denominator if denominator else 0.0,
    }


def _concentration_ok(events: list[dict[str, Any]], *, by_target: bool) -> bool:
    gains = [row for row in events if row["residual_correct"] and not row["raw_correct"]]
    if len(gains) < MIN_CONTRIBUTING_EVENTS:
        return False
    key = "target_id" if by_target else "parent_event_id"
    grouped: dict[str, int] = defaultdict(int)
    for row in gains:
        grouped[str(row[key])] += 1
    total = len(gains)
    return max(grouped.values(), default=0) / total <= MAX_SINGLE_CONTRIBUTION


def _ring_area(rows: list[dict[str, Any]]) -> float:
    values = sorted(float(row["ring_area_px"]) for row in rows if row.get("ring_area_px") is not None)
    if not values:
        return math.inf
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def _median_session_coverage(by_session: dict[str, dict[str, Any]]) -> float:
    values = sorted(float(item["residual_coverage"]) for item in by_session.values())
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def _select_ring(per_ring: dict[str, Any], session_ids: list[str]) -> tuple[str | None, str]:
    candidates: list[str] = []
    for ring_id, ring_result in per_ring.items():
        by_session = ring_result["by_session"]
        if all(session in by_session and by_session[session]["residual_evaluable_event_count"] > 0 for session in session_ids):
            candidates.append(ring_id)
    if not candidates:
        return None, "SIMILARITY_CANARY_NOT_SUPPORTED"
    ranked = sorted(
        candidates,
        key=lambda ring_id: (
            -min(per_ring[ring_id]["by_session"][session]["paired_event_gain_count"] for session in session_ids),
            -per_ring[ring_id]["metrics"]["paired_event_gain_count"],
            -_median_session_coverage(per_ring[ring_id]["by_session"]),
            per_ring[ring_id]["median_ring_area_px"],
        ),
    )
    first_key = (
        -min(per_ring[ranked[0]]["by_session"][session]["paired_event_gain_count"] for session in session_ids),
        -per_ring[ranked[0]]["metrics"]["paired_event_gain_count"],
        -_median_session_coverage(per_ring[ranked[0]]["by_session"]),
        per_ring[ranked[0]]["median_ring_area_px"],
    )
    tied = [ring_id for ring_id in ranked if (
        -min(per_ring[ring_id]["by_session"][session]["paired_event_gain_count"] for session in session_ids),
        -per_ring[ring_id]["metrics"]["paired_event_gain_count"],
        -_median_session_coverage(per_ring[ring_id]["by_session"]),
        per_ring[ring_id]["median_ring_area_px"],
    ) == first_key]
    if len(tied) != 1:
        return None, "B_SELECTION_NOT_UNIQUE"
    return tied[0], "UNIQUE_SELECTION"


def evaluate(producer_output_path: Path, producer_receipt_path: Path, truth_path: Path, output_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    receipt = json.loads(producer_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE" or receipt.get("truth_read") is not False:
        raise ValueError("producer receipt fails truth firewall")
    if receipt.get("contract_sha256") != contract_sha256(repo_root):
        raise ValueError("producer receipt contract hash mismatch")
    if receipt.get("output_sha256") != sha256_file(producer_output_path):
        raise ValueError("producer output hash mismatch")
    producer_rows = read_jsonl(producer_output_path)
    if len(producer_rows) != int(receipt.get("output_row_count", -1)):
        raise ValueError("producer row count does not match receipt")
    expected_rows = int(receipt.get("input_row_count", -1)) * len(RING_CONFIGS)
    if expected_rows >= 0 and len(producer_rows) != expected_rows:
        raise ValueError("producer output does not contain exactly one row per input/ring")
    truth_rows = read_jsonl(truth_path)
    truth = _truth_index(truth_rows)
    if not truth:
        raise ValueError("truth manifest has no eligible pairs")
    by_ring: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, tuple[Any, ...]]] = set()
    for row in producer_rows:
        if row.get("schema") != "blindassist.target_local_warp_residual_pair.v1" or row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("producer schema/protocol identity mismatch")
        if row.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError("producer implementation identity mismatch")
        ring_id = str(row.get("ring_config_id"))
        if ring_id not in RING_CONFIGS:
            raise ValueError("unknown ring configuration")
        pair_identity = (ring_id, _pair_key(row))
        if pair_identity in seen_pairs:
            raise ValueError("duplicate producer pair identity")
        seen_pairs.add(pair_identity)
        if not row.get("parameter_set_id") or not row.get("input_manifest_sha256") or not row.get("detection_manifest_sha256"):
            raise ValueError("producer provenance fields are incomplete")
        if row.get("quality") not in {"PASS", "ABSTAIN"}:
            raise ValueError("invalid producer quality")
        if row.get("quality") == "PASS":
            if row.get("residual_rate_per_s") is None or not math.isfinite(float(row["residual_rate_per_s"])):
                raise ValueError("PASS row residual is not finite")
            if row.get("abstention_reason") is not None:
                raise ValueError("PASS row has abstention reason")
        elif row.get("residual_rate_per_s") is not None:
            raise ValueError("ABSTAIN row has residual")
        if row.get("abstention_reason") not in ({None} if row.get("quality") == "PASS" else set(ABSTENTION_REASONS)):
            raise ValueError("invalid abstention reason")
        by_ring[ring_id].append(row)
    session_ids = sorted({str(item["session_id"]) for item in truth.values()})
    per_ring: dict[str, Any] = {}
    for ring_id in sorted(RING_CONFIGS):
        rows = by_ring.get(ring_id, [])
        events = _summarize_events(rows, truth)
        grouped_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped_sessions[event["session_id"]].append(event)
        by_session = {session: _metrics(grouped_sessions.get(session, [])) for session in session_ids}
        per_ring[ring_id] = {
            "metrics": _metrics(events),
            "by_session": by_session,
            "median_session_residual_coverage": _median_session_coverage(by_session),
            "median_ring_area_px": _ring_area(rows),
            "event_rows": events,
        }
    selected_ring, selection_status = _select_ring(per_ring, session_ids)
    selected = per_ring[selected_ring] if selected_ring else None
    development_passed = False
    if selected is not None:
        development_passed = (
            all(selected["by_session"][session]["paired_event_gain_count"] > 0 for session in session_ids)
            and all(selected["by_session"][session]["paired_event_gain_count"] >= 0 for session in session_ids)
            and selected["metrics"]["residual_wrong_signed_count"] <= selected["metrics"]["raw_wrong_signed_count"]
            and selected["metrics"]["coverage_loss"] <= MAX_COVERAGE_LOSS
            and _concentration_ok(selected["event_rows"], by_target=False)
        )
    if selected_ring is None and selection_status == "SIMILARITY_CANARY_NOT_SUPPORTED":
        terminal = "SIMILARITY_CANARY_NOT_SUPPORTED"
    elif not development_passed:
        terminal = "NO_DEVELOPMENT_INCREMENT / CLOSE_CANDIDATE"
    else:
        terminal = "B_DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY"
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "VALID",
        "stage": "DEVELOPMENT",
        "claim_ceiling": "DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "contract_sha256": contract_sha256(repo_root),
        "producer_output_sha256": sha256_file(producer_output_path),
        "producer_receipt_sha256": sha256_file(producer_receipt_path),
        "truth_sha256": sha256_file(truth_path),
        "truth_read_by_producer": False,
        "truth_read_by_evaluator": True,
        "session_ids": session_ids,
        "ring_selection_status": selection_status,
        "selected_ring_config_id": selected_ring,
        "per_ring": per_ring,
        "development_gate_passed": development_passed,
        "terminal": terminal,
        "limitations": [
            "B Development only; no unseen Confirmation or C1/C2 authority",
            "event and pair observations are not independent samples",
            "no Android, product, safety or active-policy conclusion",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-output", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.producer_output, args.producer_receipt, args.truth, args.output, args.repo_root)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"], "selected_ring_config_id": result["selected_ring_config_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
