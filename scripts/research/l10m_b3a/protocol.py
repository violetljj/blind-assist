"""Frozen protocol manifest for the B3-A paired causal test."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, canonical_spec
from scripts.research.l10m_b1.protocol import COMMON_SEARCH_INFORMATION


PROTOCOL_ID = "L10M-B3-A-BALANCED-EXPLORATION-CAUSAL-TEST-V1"
ARMS = ("structured_control", "structured_balanced")
GENERATIONS_PER_ARM_PER_SEED = 8
EVALUATIONS_PER_ARM_PER_SEED = 8
MIN_DISCOVERY_IMPROVEMENT = 0.02
INITIAL_SCORE = 0.9517241379310345
MODEL = "gpt-5.6-sol"
TRANSPORT_QUALIFICATION_SHA256 = "2af462f351814045dafcf488781fb8e914bd5847abd851a61ac7d962d13a0e1b"
TRANSPORT_QUALIFICATION_RUN_ID = "b1-i0-proxy-20260820T025833-4e438512"
SOURCE_FILES = (
    "scripts/research/l10m_b1/evaluator.py",
    "scripts/research/l10m_b1/hidden_cohort_v1.json",
    "scripts/research/l10m_b1/policy_space.py",
    "scripts/research/l10m_b1/protocol.py",
    "scripts/research/l10m_b1/provider_transport.py",
    "scripts/research/l10m_b1/run_search.py",
    "scripts/research/l10m_b3a/exploration.py",
    "scripts/research/l10m_b3a/protocol.py",
    "scripts/research/l10m_b3a/run_experiment.py",
    "scripts/research/l10m_b3a/analyze_result.py",
)
B3_I0_RESULT = "artifacts.local/evidence/l10m_b3/b3_i0_lineage_autopsy/result.json"
B3_I0_RESULT_SHA256 = "c839e053f55691d74a6341c2c68ab147e9302cda060559a968e8ed9a7010720f"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def derive_fresh_seeds() -> tuple[int, int, int]:
    seeds = []
    for index in range(3):
        digest = hashlib.sha256(f"{PROTOCOL_ID}|fresh-seed|{index}".encode("utf-8")).digest()
        seeds.append(int.from_bytes(digest[:4], "big") % 9000 + 1000)
    return tuple(seeds)  # type: ignore[return-value]


PAIRED_SEEDS = derive_fresh_seeds()


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_protocol_manifest(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    source_hashes = {relative: _sha256(repo_root / relative) for relative in SOURCE_FILES}
    if _sha256(repo_root / B3_I0_RESULT) != B3_I0_RESULT_SHA256:
        raise RuntimeError("B3-I0 source receipt hash mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "B3_A_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED",
        "research_question": "Does outcome-blind canonical move coverage repair the observed Structured semantic exploration collapse and add paired search value?",
        "causal_arms": {
            "structured_control": "frozen B1 Structured proposal, feedback, incumbent, and strict selection",
            "structured_balanced": "same search plus one outcome-blind canonical move coverage and anti-repeat operator",
        },
        "freshness": {
            "seed_derivation": "1000 + uint32_be(sha256(protocol_id + '|fresh-seed|' + index)[0:4]) mod 9000 for index 0..2",
            "paired_seeds": list(PAIRED_SEEDS),
            "excluded_consumed_b1_seeds": [17, 29, 43, 53, 71, 89],
            "seed89_diagnostic_only": True,
            "provider_sampling_seed_control": False,
            "seed_scope": "fresh outcome-blind paired prompt and session identities, matching B1 semantics; not a provider-level deterministic sampling seed",
        },
        "balanced_operator": {
            "unit": "parameter + incumbent from-value + adjacent UP/DOWN or categorical destination + to-value",
            "ledger_scope": "one arm and paired seed only",
            "rule": "admit a model-proposed untried legal move when present; otherwise project to one untried legal move by frozen outcome-blind hash rank; use model proposal unchanged only after local coverage exhaustion",
            "projection_rank": "sha256('L10M-B3-A|seed|generation|source|move_token')",
            "score_or_target_used_by_operator": False,
            "target_value_or_hash_in_intervention": False,
            "one_step_numeric_moves": True,
            "selection_after_evaluation": "same strict score improvement as control",
        },
        "matched_factors": {
            "interface": "component-grouped Structured JSON",
            "initial_candidate": canonical_spec(INITIAL_SPEC),
            "initial_score": INITIAL_SCORE,
            "model": MODEL,
            "temperature": "provider default frozen identically; no arm-specific override",
            "generation_budget": GENERATIONS_PER_ARM_PER_SEED,
            "evaluation_budget": EVALUATIONS_PER_ARM_PER_SEED,
            "evaluator": "frozen B1 evaluator and hidden cohort",
            "feedback": "frozen B1 same-arm best-incumbent feedback; treatment additionally receives its attempted-move ledger",
            "selection": "replace incumbent only with semantic-valid, safe, strictly higher behavioral score",
            "pair_order": "alternate first arm by seed index",
            "provider_failure": "seal whole cohort NOT_EVALUABLE immediately; no retry, replacement, or resume",
        },
        "estimands": {
            "discovery": f"valid safe score improvement >= {MIN_DISCOVERY_IMPROVEMENT} from initial",
            "primary": "paired discovery reach and best-final-score wins/losses",
            "secondary": [
                "first discovery generation",
                "exact B2 target reach as diagnostic only",
                "unique canonical admitted proposals and moves",
                "non-improving exploration fraction",
                "unsafe and semantic-invalid rates",
            ],
        },
        "verdict_rules": {
            "B3A_BALANCED_EXPLORATION_ADMITTED": "balanced discovery rate strictly exceeds control, balanced best score wins >=1 paired seed and loses none, unsafe rate does not increase, and operator integrity passes",
            "B3A_BALANCED_EXPLORATION_NOT_ADMITTED": "complete evaluable run without the full admission pattern; diversity alone is not search value",
            "B3A_NOT_EVALUABLE_RUNTIME": "any provider, transport, isolation, ledger, or execution-integrity failure",
        },
        "claim_ceiling": "causal search-value evidence for this balanced exploration operator in the frozen finite synthetic Structured interface only; not general model superiority, device, user, safety-effect, or production evidence",
        "execution": {
            "new_search_authorized": True,
            "only_experiment_authorized": "this two-arm B3-A protocol",
            "planned_model_calls": len(PAIRED_SEEDS) * len(ARMS) * GENERATIONS_PER_ARM_PER_SEED,
            "algorithm_admission_requires_fresh_result": True,
        },
        "source": {
            "git_commit": _git_commit(repo_root),
            "source_sha256": source_hashes,
            "common_search_information_sha256": _sha256_bytes(COMMON_SEARCH_INFORMATION.encode("utf-8")),
            "b3_i0_result_sha256": B3_I0_RESULT_SHA256,
        },
        "transport": {
            "route": "proxy",
            "qualification_run_id": TRANSPORT_QUALIFICATION_RUN_ID,
            "qualification_sha256": TRANSPORT_QUALIFICATION_SHA256,
            "qualification_scope": "same Docker image, Codex version, Responses HTTPS streaming mode, auth, proxy route, and response shape used by completed B1 V2",
        },
    }


def canonical_manifest_sha256(manifest: dict[str, object]) -> str:
    return _sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
