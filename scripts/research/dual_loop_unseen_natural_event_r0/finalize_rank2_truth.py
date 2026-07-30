from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.validate_ai_review_receipt import validate_model_pass


PROTOCOL_ID = "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ"
SOURCE_ID = "commons_iran_shiraz_city_tour_2021_5"
CAPTURE_SESSION_ID = "commons_page_143041813_capture_2021-11-20"
DEPENDENCY_GROUP = "4KTourtv_iran_2021"
CANDIDATE_COMMIT = "039757b2da41c051373f8ee3189c4b06028f5295"
INPUT_SHA256 = "303e592812b62ce404f4bb2dce7c21ac3cb55a26431c2175c4a9c5b1d8fd94a8"
REVIEW_PROMPT_SHA256 = (
    "974087fe50e1a7817c7dddad9d457fdad267d371eb34985d5f858d839ee6287e"
)
ADJUDICATION_PROMPT_SHA256 = (
    "2d00c0f5d681ca057cc42e2dd907e208862d73c1e4efd2947d2ef777e8c96204"
)
MINIMUM_POSITIVES = 3
MINIMUM_NEGATIVES = 6
MAX_CAPTURE_TIMESTAMP_NS = 489_000_000_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review(path: Path, allowed_role: str) -> dict:
    review = json.loads(path.read_text(encoding="utf-8"))
    validate_model_pass(
        review,
        policy={
            "candidate_output_hidden_from_reviewers": True,
            "minimum_confidence": 0.65,
        },
        where=path.name,
        allowed_roles={allowed_role},
    )
    if (
        review["workflow_id"] != "dual_loop_unseen_natural_event_truth_v1"
        or review["prompt_sha256"] != REVIEW_PROMPT_SHA256
        or review["input_sha256"] != INPUT_SHA256
        or review.get("verdict") != "TRUTH_ADEQUATE"
    ):
        raise ValueError(f"{path.name}: review binding or verdict mismatch")
    result = review.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: missing result")
    if (
        result.get("rgb_only") is not True
        or result.get("outputs_inspected") is not False
        or result.get("contact_sheets_reviewed") != 25
        or result.get("summary", {}).get("positive_count", 0) < MINIMUM_POSITIVES
    ):
        raise ValueError(f"{path.name}: incomplete or truth-inadequate RGB pass")
    return review


def load_adjudication(
    path: Path,
    review_a_sha256: str,
    review_b_sha256: str,
) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required_text = {
        "schema": "blindassist.dual_loop_unseen_truth_adjudication.v1",
        "reviewer_id": "rank2_truth_adjudicator",
        "reviewer_type": "ai_model",
        "reviewer_role": "gpt_adjudicator",
        "workflow_id": "dual_loop_unseen_natural_event_truth_v1",
        "prompt_sha256": ADJUDICATION_PROMPT_SHA256,
        "input_sha256": INPUT_SHA256,
        "verdict": "TRUTH_ADEQUATE",
    }
    for key, expected in required_text.items():
        if value.get(key) != expected:
            raise ValueError(f"{path.name}: {key} mismatch")
    if (
        value.get("isolated_context") is not True
        or value.get("other_review_visible_before_submission") is not True
        or value.get("candidate_output_visible") is not False
        or value.get("abstained") is not False
    ):
        raise ValueError(f"{path.name}: adjudicator visibility/isolation mismatch")
    if value.get("input_review_sha256s") != [review_a_sha256, review_b_sha256]:
        raise ValueError(f"{path.name}: ordered review hash binding mismatch")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.65 <= float(confidence) <= 1:
        raise ValueError(f"{path.name}: invalid confidence")
    result = value.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: missing result")
    summary = result.get("summary", {})
    if (
        summary.get("truth_adequacy") != "TRUTH_ADEQUATE"
        or summary.get("admitted_positive_event_count", 0) < MINIMUM_POSITIVES
        or summary.get("final_negative_window_count", 0) < MINIMUM_NEGATIVES
    ):
        raise ValueError(f"{path.name}: adjudicated truth is inadequate")
    return value


def canonical_negative_category(value: str) -> str:
    if value.startswith("static_obstacle_off_corridor"):
        return "STATIC_OBSTACLE_OFF_CORRIDOR"
    if value == "camera_turn_or_pan":
        return "TURN_OR_NEAR_IN_PLACE_ROTATION"
    raise ValueError(f"unexpected adjudicated negative category: {value}")


def validate_emitted_ledger(path: Path) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers: set[str] = set()
    metric_intervals: list[tuple[int, int, str]] = []
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in identifiers:
            raise ValueError(f"invalid or duplicate truth item id: {item_id!r}")
        identifiers.add(item_id)
        confidence = row.get("truth_confidence")
        if (
            not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"{item_id}: invalid truth confidence")
        start_ns = row.get("start_ns")
        end_ns = row.get("end_ns")
        if (
            not isinstance(start_ns, int)
            or not isinstance(end_ns, int)
            or start_ns < 0
            or end_ns > MAX_CAPTURE_TIMESTAMP_NS
            or end_ns < start_ns
        ):
            raise ValueError(f"{item_id}: invalid truth interval")
        if not isinstance(row.get("region"), str) or not row["region"].strip():
            raise ValueError(f"{item_id}: empty region")
        if row.get("should_alert") is True:
            alertable_start_ns = row.get("alertable_start_ns")
            if (
                not isinstance(alertable_start_ns, int)
                or not start_ns <= alertable_start_ns <= end_ns
            ):
                raise ValueError(f"{item_id}: invalid alertable interval")
            metric_start_ns = alertable_start_ns
        elif row.get("should_alert") is False:
            if row.get("alertable_start_ns") is not None:
                raise ValueError(f"{item_id}: negative item has alertable start")
            metric_start_ns = start_ns
        else:
            raise ValueError(f"{item_id}: invalid should_alert")
        metric_intervals.append((metric_start_ns, end_ns, item_id))
    metric_intervals.sort()
    for previous, current in zip(metric_intervals, metric_intervals[1:]):
        if previous[1] >= current[0]:
            raise ValueError(
                "truth metric intervals overlap at closed endpoints: "
                f"{previous[2]} and {current[2]}"
            )


def finalize(
    review_a_path: Path,
    review_b_path: Path,
    adjudication_path: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    review_a = load_review(review_a_path, "gpt_multimodal_reviewer")
    review_b = load_review(review_b_path, "codex_evidence_reviewer")
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ValueError("independent reviewer identities must differ")
    review_a_sha = sha256_file(review_a_path)
    review_b_sha = sha256_file(review_b_path)
    adjudication = load_adjudication(
        adjudication_path,
        review_a_sha,
        review_b_sha,
    )
    result = adjudication["result"]

    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)

    ledger_path = temporary / "truth_ledger.jsonl"
    positive_count = 0
    category_counts: Counter[str] = Counter()
    with ledger_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in result["admitted_positive_events"]:
            row = {
                "schema_version": "blindassist.dual_loop_event_truth_item.v1",
                "item_id": f"R2_{item['item_id']}",
                "item_kind": "positive_event",
                "should_alert": True,
                "category": "APPROACH_HAZARD",
                "source_id": SOURCE_ID,
                "capture_session_id": CAPTURE_SESSION_ID,
                "dependency_group": DEPENDENCY_GROUP,
                "start_ns": int(round(float(item["start_s"]) * 1_000_000_000)),
                "alertable_start_ns": int(
                    round(float(item["alertable_start_s"]) * 1_000_000_000)
                ),
                "end_ns": int(round(float(item["end_s"]) * 1_000_000_000)),
                "region": item["region"],
                "truth_confidence": float(item["confidence"]),
                "truth_notes": item["notes"],
                "truth_provenance": "two_independent_ai_rgb_reviews_plus_fresh_adjudication",
                "outcome_access_state": "BASELINE_AND_CANDIDATE_UNOPENED",
            }
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            positive_count += 1
        for item in result["final_negative_windows"]:
            category = canonical_negative_category(item["category"])
            row = {
                "schema_version": "blindassist.dual_loop_event_truth_item.v1",
                "item_id": f"R2_{item['item_id']}",
                "item_kind": "negative_window",
                "should_alert": False,
                "category": category,
                "source_id": SOURCE_ID,
                "capture_session_id": CAPTURE_SESSION_ID,
                "dependency_group": DEPENDENCY_GROUP,
                "start_ns": int(round(float(item["start_s"]) * 1_000_000_000)),
                "alertable_start_ns": None,
                "end_ns": int(round(float(item["end_s"]) * 1_000_000_000)),
                "region": item["region"],
                "truth_confidence": float(item["confidence"]),
                "truth_notes": item["notes"],
                "truth_provenance": "two_independent_ai_rgb_reviews_plus_fresh_adjudication",
                "outcome_access_state": "BASELINE_AND_CANDIDATE_UNOPENED",
            }
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            category_counts[category] += 1

    negative_count = sum(category_counts.values())
    validate_emitted_ledger(ledger_path)
    truth_adequacy = (
        positive_count >= MINIMUM_POSITIVES
        and negative_count >= MINIMUM_NEGATIVES
        and len(category_counts) >= 2
        and min(category_counts.values()) >= 2
    )
    if not truth_adequacy:
        raise ValueError("adjudicated ledger does not meet the pre-frozen truth floor")
    receipt = {
        "schema_version": "blindassist.dual_loop_truth_freeze_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "TRUTH_FROZEN_ADEQUATE",
        "source_id": SOURCE_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "input_sha256": INPUT_SHA256,
        "review_prompt_sha256": REVIEW_PROMPT_SHA256,
        "adjudication_prompt_sha256": ADJUDICATION_PROMPT_SHA256,
        "review_method": "independent_ai_adjudicator",
        "review_a_sha256": review_a_sha,
        "review_b_sha256": review_b_sha,
        "adjudication_sha256": sha256_file(adjudication_path),
        "positive_event_count": positive_count,
        "minimum_positive_events": MINIMUM_POSITIVES,
        "negative_window_count": negative_count,
        "negative_category_counts": dict(sorted(category_counts.items())),
        "truth_adequacy": True,
        "baseline_output_opened": False,
        "candidate_output_opened": False,
        "truth_ledger_sha256": sha256_file(ledger_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "next_allowed_action": "run baseline-only adequacy on the fixed 10 Hz input",
    }
    (temporary / "truth_freeze_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.review_a,
                args.review_b,
                args.adjudication,
                args.output,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
