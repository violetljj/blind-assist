"""Correctly adjudicate the consumed PB1 result against abstention collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjudicate(result: dict[str, Any]) -> dict[str, Any]:
    baseline_name = str(result["development_selected_baseline"])
    baseline = result["metrics"]["test"][baseline_name]
    head = result["metrics"]["test"]["portal_binding_head"]
    no_top3_loss = head["top3_truth_retention"] >= baseline["top3_truth_retention"]
    top1_effect = head["top1_truth_retention"] >= baseline["top1_truth_retention"] + 1
    baseline_wrong = int(baseline["wrong_portal_commits"])
    commit_effect = baseline_wrong > 0 and head["wrong_portal_commits"] <= 0.5 * baseline_wrong
    positive_admission_not_lower = (
        head["commit_or_set_valued_truth_coverage"]
        >= baseline["commit_or_set_valued_truth_coverage"]
    )
    non_degenerate = head["commit_or_set_valued_truth_coverage"] > 0
    passed = (
        no_top3_loss
        and positive_admission_not_lower
        and non_degenerate
        and (top1_effect or commit_effect)
    )
    return {
        "schema": "l10-named-poi-portal-binding-adjudication-v1",
        "original_mechanical_gate": result["promotion_gate"],
        "adjudication_invariant": "A portal binder cannot be promoted by reducing false commits through universal NONE. Correct COMMIT/SET_VALUED truth coverage must be nonzero and no lower than the selected baseline.",
        "development_selected_baseline": baseline_name,
        "baseline_test": baseline,
        "head_test": head,
        "checks": {
            "no_top3_loss": no_top3_loss,
            "top1_gain_at_least_one_building": top1_effect,
            "wrong_portal_commit_reduction_at_least_50_percent": commit_effect,
            "positive_admission_not_lower_than_baseline": positive_admission_not_lower,
            "non_degenerate_positive_admission": non_degenerate,
        },
        "passed": passed,
        "decision": (
            "L10_PB1_TARGET_CONDITIONED_PORTAL_BINDING_EFFECT"
            if passed
            else "L10_PB1_FRESH_BUILDING_GATE_NOT_MET_ALL_NONE_STOP_EMBEDDING_FUSION"
        ),
        "next_action": (
            "Proceed to L10-AV0"
            if passed
            else "Do not tune weights, thresholds, embeddings, backbones, or fusion on the consumed cohort. A successor must change the identity information source/representation and use a new building-disjoint confirmation cohort. L10-AV0 remains blocked."
        ),
        "claim_boundary": "Post-result evaluator-integrity correction only; it does not alter model outputs, rankings, labels, or denominators.",
    }


def self_test() -> dict[str, Any]:
    result = {
        "development_selected_baseline": "baseline",
        "promotion_gate": {"passed": True},
        "metrics": {
            "test": {
                "baseline": {
                    "top1_truth_retention": 4,
                    "top3_truth_retention": 5,
                    "wrong_portal_commits": 2,
                    "commit_or_set_valued_truth_coverage": 4,
                },
                "portal_binding_head": {
                    "top1_truth_retention": 4,
                    "top3_truth_retention": 6,
                    "wrong_portal_commits": 0,
                    "commit_or_set_valued_truth_coverage": 0,
                },
            }
        },
    }
    decision = adjudicate(result)
    if decision["passed"] or decision["checks"]["non_degenerate_positive_admission"]:
        raise AssertionError("ALL_NONE_COLLAPSE_WAS_PROMOTED")
    return {"schema": "l10-pb1-adjudication-self-test-v1", "status": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-portal-binding-v1/result.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result_path = args.result.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else result_path.with_name("adjudication.json")
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = adjudicate(result)
    payload["result_path"] = str(result_path)
    payload["result_sha256"] = _sha256(result_path)
    payload["self_test"] = self_test()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
