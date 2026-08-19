"""Preregister the small matched L10M-B1 searchability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.research.l10m_b0.closure import PROTOCOL_ID as B0_CLOSURE_PROTOCOL_ID
from scripts.research.l10m_b0.closure import VERDICT as B0_CLOSURE_VERDICT
from scripts.research.l10m_b0.closure import build_closure_manifest

from .policy_space import (
    FALLBACK_ACTIONS,
    INITIAL_SPEC,
    QUALITY_FLOORS,
    RECOVERY_ACTIONS,
    STUCK_RESPONSES,
    TURN_THRESHOLDS,
    all_specs,
    canonical_spec,
    parse_raw,
    parse_structured,
    render_raw,
    render_structured,
)


PROTOCOL_ID = "L10M-B1-STRUCTURED-SEARCHABILITY-MATCHED-V1"
STATUS = "B1_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED"
EXPECTED_SHA256 = {
    "scripts/research/l10m_b1/hidden_cohort_v1.json": "960172ce5dd5cc227eddf778ff9c586956f784ac681916b1cbb45a0e11013e7e",
    "scripts/research/l10m_b1/evaluator.py": "5ca40163ead786127a53afbd994862f31b5bb87a9d83ffcc806406cdc0525a05",
    "scripts/research/l10m_b1/policy_space.py": "923ea40af80866fb6f98725d2031e2c450972f138e3ce31e8df6de4270779c43",
}
PAIRED_SEEDS = (17, 29, 43)
GENERATIONS_PER_ARM_PER_SEED = 8
EVALUATIONS_PER_ARM_PER_SEED = 8
MIN_DISCOVERY_IMPROVEMENT = 0.02
FINAL_SCORE_EQUIVALENCE_MARGIN = 0.01


COMMON_SEARCH_INFORMATION = """You are improving one policy candidate in a frozen synthetic policy-mechanics evaluator.
You may change only five admitted fields. Allowed values are:
- action_selection_turn_threshold: 0.10, 0.20, 0.30
- fallback_min_quality: 0.35, 0.50, 0.65
- fallback_action: STOP, LEFT, RIGHT
- stuck_response: ENTER_RECOVERY, STOP
- recovery_transition_action: RECOVER, LEFT, RIGHT
The progress three-state contract, stuck update, recovery enter/exit semantics, confirmed-arrival terminal invariant,
UNKNOWN behavior, evaluator, truth, unsafe definition, hard safety shield, cohort, score, budget, and selection rule are immutable.
Return exactly one candidate in the interface format you are shown. No explanation.
"""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_interface_equivalence() -> dict[str, object]:
    specs = all_specs()
    canonical = {canonical_spec(spec) for spec in specs}
    raw_roundtrip = {canonical_spec(parse_raw(render_raw(spec))) for spec in specs}
    structured_roundtrip = {
        canonical_spec(parse_structured(render_structured(spec))) for spec in specs
    }
    if canonical != raw_roundtrip or canonical != structured_roundtrip:
        raise RuntimeError("raw and structured interfaces do not expose the same candidate space")
    if canonical_spec(parse_raw(render_raw(INITIAL_SPEC))) != canonical_spec(INITIAL_SPEC):
        raise RuntimeError("raw initial candidate changed")
    if canonical_spec(parse_structured(render_structured(INITIAL_SPEC))) != canonical_spec(INITIAL_SPEC):
        raise RuntimeError("structured initial candidate changed")
    return {
        "candidate_count": len(canonical),
        "raw_candidate_count": len(raw_roundtrip),
        "structured_candidate_count": len(structured_roundtrip),
        "same_initial_candidate": True,
        "same_canonical_space": True,
    }


def build_protocol_manifest() -> dict[str, object]:
    closure = build_closure_manifest()
    if closure["protocol_id"] != B0_CLOSURE_PROTOCOL_ID or closure["verdict"] != B0_CLOSURE_VERDICT:
        raise RuntimeError("B1 cannot start without the frozen B0 closure")

    root = _repository_root()
    actual_hashes = {path: _sha256(root / path) for path in EXPECTED_SHA256}
    if actual_hashes != EXPECTED_SHA256:
        raise RuntimeError("B1 evaluator, policy space, or hidden cohort identity changed")

    return {
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "research_question": "Under frozen B0 state semantics, does structured exposure of the same mutable policy space make behaviorally better candidates easier to find?",
        "claim_ceiling": "matched search-interface value on a hidden synthetic mechanics cohort; not state-machine revalidation, large-scale Structured Search, end-to-end, device, user, or safety-effect evidence",
        "parent_freeze": {
            "protocol_id": closure["protocol_id"],
            "verdict": closure["verdict"],
            "b0_semantics_mutable": False,
            "b0_e_allowed": False,
            "frozen_source_sha256": closure["frozen_source_sha256"],
            "frozen_result_sha256": closure["frozen_result_sha256"],
        },
        "frozen_b1_sha256": actual_hashes,
        "arms": {
            "control": {
                "name": "RAW_SEARCH",
                "interface": "literal source-level policy assignments",
                "parser": "non-executing restricted Python AST",
            },
            "treatment": {
                "name": "STRUCTURED_SEARCH",
                "interface": "component-grouped JSON",
                "components": [
                    "progress_contract_read_only",
                    "stuck_response",
                    "recovery_transition",
                    "action_selection",
                    "fallback",
                ],
            },
        },
        "matched_factors": {
            "common_search_information_sha256": hashlib.sha256(COMMON_SEARCH_INFORMATION.encode()).hexdigest(),
            "model": "bind once in the execution manifest and reuse identically for every pair",
            "provider": "bind executable path, CLI version, and executable SHA-256 once before any run artifact",
            "initial_candidate": canonical_spec(INITIAL_SPEC),
            "seeds": list(PAIRED_SEEDS),
            "generations_per_arm_per_seed": GENERATIONS_PER_ARM_PER_SEED,
            "evaluations_per_arm_per_seed": EVALUATIONS_PER_ARM_PER_SEED,
            "feedback": "same behavioral vector and validity fields after each admitted evaluation",
            "selection_rule": "best semantic-valid and safe candidate by frozen behavioral_score; earliest evaluation breaks ties",
            "pair_order": "alternate first arm by seed; preserve paired generation and evaluation counts",
        },
        "candidate_space": {
            "action_selection_turn_threshold": list(TURN_THRESHOLDS),
            "fallback_min_quality": list(QUALITY_FLOORS),
            "fallback_action": list(FALLBACK_ACTIONS),
            "stuck_response": list(STUCK_RESPONSES),
            "recovery_transition_action": list(RECOVERY_ACTIONS),
            "equivalence_receipt": _verify_interface_equivalence(),
        },
        "blind_isolation": {
            "searcher_receives": [
                "common search information",
                "arm-specific representation of the same initial/current candidate",
                "matched evaluator feedback after each admitted candidate",
            ],
            "searcher_must_not_receive": [
                "repository access",
                "hidden cohort contents or path",
                "episode truth or accepted actions",
                "other arm candidates, feedback, or outcomes",
                "evaluator source or hashes",
            ],
            "evaluation_process_only": [
                "hidden cohort",
                "truth",
                "unsafe definition",
                "hard safety shield",
                "behavioral score",
            ],
        },
        "estimands": {
            "primary": "paired median of (best improvement from initial)_structured - (best improvement from initial)_raw",
            "discovery_success": f"best valid safe candidate improves frozen behavioral_score by at least {MIN_DISCOVERY_IMPROVEMENT:.2f} from the paired initial candidate",
            "secondary": [
                "discovery success rate",
                "best-of-budget behavioral score and full behavioral vector",
                "generation and evaluation index of first valid improvement",
                "unsafe candidate rate",
                "semantic-invalid candidate rate",
                "changed component ledger",
                "paired seed stability",
            ],
        },
        "verdict_rules": {
            "STRUCTURED_SEARCHABILITY_VALUE_ESTABLISHED": "positive paired median primary estimand, structured wins at least 2 of 3 seeds, and no unsafe-rate regression",
            "STRUCTURED_SEARCH_EFFICIENCY_VALUE_ESTABLISHED": f"best scores are within {FINAL_SCORE_EQUIVALENCE_MARGIN:.2f}, while structured reaches a valid improvement earlier or with fewer invalid candidates in at least 2 of 3 seeds and has no unsafe-rate regression",
            "REPRESENTATION_NOT_BOTTLENECK_SHARED_CAUSAL_COMPONENT": "best scores are equivalent and both arms independently discover the same changed component in at least 2 of 3 paired seeds",
            "SEARCH_OPERATOR_OR_GENERATION_BOTTLENECK_NOT_RESOLVED": "neither arm discovers a valid safe improvement in at least 2 of 3 paired seeds; this does not establish that structured representation lacks value",
            "B1_INCONCLUSIVE": "all remaining patterns, including isolated final-score differences without hit-rate, latency, validity, or seed-stability support",
        },
        "execution_boundary": {
            "formal_search_started": False,
            "model_calls_made": 0,
            "hidden_outcomes_exposed_to_searcher": False,
            "large_scale_structured_search_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_protocol_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
