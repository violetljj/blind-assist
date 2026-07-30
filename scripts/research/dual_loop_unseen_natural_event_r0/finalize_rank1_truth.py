from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.validate_ai_review_receipt import validate_model_pass


PROTOCOL_ID = "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0"
SOURCE_ID = "commons_shanghai_shopping_street_night_2024"
CAPTURE_SESSION_ID = "commons_page_153983964_capture_2024-10-10"
DEPENDENCY_GROUP = "AmbienceX_shanghai_night_2024"
CANDIDATE_COMMIT = "039757b2da41c051373f8ee3189c4b06028f5295"
PROMPT_SHA256 = "ed24afa88aa3bd455a00bf0f789c10137059f549ed976a0a9ad22537b1f1564c"
INPUT_SHA256 = "97204f6cb914ab5080db8c6c8dbf1888b7a6c426d04dc44222677ec438ddbd0a"
MINIMUM_POSITIVES = 3

NEGATIVE_WINDOWS = [
    ("R1_NEG_001", "NORMAL_WALKING_SHAKE", 74, 87, "center sidewalk"),
    ("R1_NEG_002", "NORMAL_WALKING_SHAKE", 427, 451, "building-side sidewalk"),
    ("R1_NEG_003", "TURN_OR_NEAR_IN_PLACE_ROTATION", 124, 139, "full frame"),
    ("R1_NEG_004", "TURN_OR_NEAR_IN_PLACE_ROTATION", 465, 469, "storefront display"),
    ("R1_NEG_005", "SAFE_OPPOSING_FLOW_WITH_CLEAR_SPLIT", 40, 42, "left sidewalk"),
    ("R1_NEG_006", "SAFE_OPPOSING_FLOW_WITH_CLEAR_SPLIT", 350, 354, "right plaza lane"),
]


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
    if review["workflow_id"] != "dual_loop_unseen_natural_event_truth_v1":
        raise ValueError(f"{path.name}: workflow mismatch")
    if review["prompt_sha256"] != PROMPT_SHA256 or review["input_sha256"] != INPUT_SHA256:
        raise ValueError(f"{path.name}: prompt/input binding mismatch")
    if review.get("verdict") != "TRUTH_INADEQUATE":
        raise ValueError(f"{path.name}: unexpected verdict")
    result = review.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: missing result")
    if (
        result.get("rgb_only") is not True
        or result.get("outputs_inspected") is not False
        or result.get("contact_sheets_reviewed") != 29
        or result.get("positive_events") != []
        or result.get("summary", {}).get("positive_count") != 0
    ):
        raise ValueError(f"{path.name}: result is not a complete zero-positive RGB pass")
    return review


def finalize(review_a_path: Path, review_b_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    review_a = load_review(review_a_path, "gpt_multimodal_reviewer")
    review_b = load_review(review_b_path, "codex_evidence_reviewer")
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ValueError("independent reviewer identities must differ")

    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)

    ledger = temporary / "truth_ledger.jsonl"
    with ledger.open("w", encoding="utf-8", newline="\n") as stream:
        for item_id, category, start_s, end_s, region in NEGATIVE_WINDOWS:
            row = {
                "schema_version": "blindassist.dual_loop_event_truth_item.v1",
                "item_id": item_id,
                "item_kind": "negative_window",
                "should_alert": False,
                "category": category,
                "source_id": SOURCE_ID,
                "capture_session_id": CAPTURE_SESSION_ID,
                "dependency_group": DEPENDENCY_GROUP,
                "start_ns": start_s * 1_000_000_000,
                "alertable_start_ns": None,
                "end_ns": end_s * 1_000_000_000,
                "region": region,
                "truth_provenance": "two_independent_ai_rgb_reviews_model_consensus",
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

    reviews = [
        {
            "reviewer_id": review_a["reviewer_id"],
            "reviewer_role": review_a["reviewer_role"],
            "review_sha256": sha256_file(review_a_path),
            "verdict": review_a["verdict"],
        },
        {
            "reviewer_id": review_b["reviewer_id"],
            "reviewer_role": review_b["reviewer_role"],
            "review_sha256": sha256_file(review_b_path),
            "verdict": review_b["verdict"],
        },
    ]
    truth_receipt = {
        "schema_version": "blindassist.dual_loop_truth_freeze_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "TRUTH_FROZEN_INADEQUATE",
        "source_id": SOURCE_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "input_sha256": INPUT_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "review_method": "model_consensus",
        "reviews": reviews,
        "positive_event_count": 0,
        "minimum_positive_events": MINIMUM_POSITIVES,
        "negative_window_count": len(NEGATIVE_WINDOWS),
        "negative_category_counts": {
            "NORMAL_WALKING_SHAKE": 2,
            "TURN_OR_NEAR_IN_PLACE_ROTATION": 2,
            "SAFE_OPPOSING_FLOW_WITH_CLEAR_SPLIT": 2,
        },
        "truth_adequacy": False,
        "baseline_output_opened": False,
        "candidate_output_opened": False,
        "truth_ledger_sha256": sha256_file(ledger),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    (temporary / "truth_freeze_receipt.json").write_text(
        json.dumps(truth_receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    terminal = {
        "schema_version": "blindassist.dual_loop_source_terminal.v1",
        "protocol_id": PROTOCOL_ID,
        "source_id": SOURCE_ID,
        "terminal": "FIRST_UNSEEN_SOURCE_NOT_EVALUABLE",
        "reason": "zero model-consensus positive events; pre-frozen minimum is three",
        "execution_valid": True,
        "algorithm_failure_claimed": False,
        "baseline_output_opened": False,
        "candidate_output_opened": False,
        "candidate_retuned": False,
        "negative_windows_reuse_role": "source_characterization_and_regression_only",
        "next_allowed_action": "disclose terminal, then start rank-2 as a new evidence instance",
        "truth_freeze_receipt_sha256": sha256_file(
            temporary / "truth_freeze_receipt.json"
        ),
    }
    (temporary / "terminal.json").write_text(
        json.dumps(terminal, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(args.review_a, args.review_b, args.output),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
