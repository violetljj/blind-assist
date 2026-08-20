"""Frozen protocol manifest for the B4-A harder-cohort paired comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, canonical_spec


PROTOCOL_ID = "L10M-B4-A-HARDER-COHORT-PAIRED-SEARCH-V1"
ARMS = ("structured_control", "structured_balanced")
INSTANCE_IDS = ("amber", "cobalt", "jade")
REPLICATES_PER_INSTANCE = 3
GENERATIONS_PER_TRAJECTORY = 8
MODEL = "gpt-5.6-sol"
HARD_BENCHMARK_CERTIFICATE = "artifacts.local/evidence/l10m_b4/hard_benchmark_v1/certificate.json"
HARD_BENCHMARK_CERTIFICATE_SHA256 = "7f2cf3a1fb4db8534e5af3839c264dc377be48db63538d7c85c023aabf3c2696"
B3A_TERMINAL_RESULT = "artifacts.local/evidence/l10m_b3a/runs/b3a-20260820T124003-69a8df8a/result.json"
B3A_TERMINAL_RESULT_SHA256 = "bfa265c677e2ff733456ec4c873ba9573ee6b425ffd24001d670a3b785fbeb1b"
TRANSPORT_QUALIFICATION_SHA256 = "2af462f351814045dafcf488781fb8e914bd5847abd851a61ac7d962d13a0e1b"
TRANSPORT_QUALIFICATION_RUN_ID = "b1-i0-proxy-20260820T025833-4e438512"
CONSUMED_IDENTITIES = {17, 29, 43, 53, 71, 89, 1768, 7368, 1872}
SOURCE_FILES = (
    "scripts/research/l10m_b0/evaluation.py",
    "scripts/research/l10m_b0/b0c_precedence.py",
    "scripts/research/l10m_b1/evaluator.py",
    "scripts/research/l10m_b1/policy_space.py",
    "scripts/research/l10m_b1/protocol.py",
    "scripts/research/l10m_b1/provider_transport.py",
    "scripts/research/l10m_b1/run_search.py",
    "scripts/research/l10m_b3a/exploration.py",
    "scripts/research/l10m_b4/hard_benchmark_v1.json",
    "scripts/research/l10m_b4/hard_benchmark.py",
    "scripts/research/l10m_b4/protocol_b4a.py",
    "scripts/research/l10m_b4/run_b4a.py",
    "scripts/research/l10m_b4/analyze_b4a.py",
    "scripts/research/l10m_b4/summarize_b4a.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_fresh_identities() -> tuple[dict[str, int | str], ...]:
    rows: list[dict[str, int | str]] = []
    seen = set(CONSUMED_IDENTITIES)
    for instance_id in INSTANCE_IDS:
        for replicate in range(REPLICATES_PER_INSTANCE):
            nonce = 0
            while True:
                payload = f"{PROTOCOL_ID}|{instance_id}|{replicate}|{nonce}".encode()
                identity = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 900000 + 100000
                if identity not in seen:
                    break
                nonce += 1
            seen.add(identity)
            rows.append(
                {
                    "instance_id": instance_id,
                    "replicate": replicate,
                    "paired_identity": identity,
                }
            )
    return tuple(rows)


PAIRED_IDENTITIES = derive_fresh_identities()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_protocol_manifest(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    certificate_path = repo_root / HARD_BENCHMARK_CERTIFICATE
    if _sha256(certificate_path) != HARD_BENCHMARK_CERTIFICATE_SHA256:
        raise RuntimeError("harder-cohort certificate identity mismatch")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("terminal") != "B4_HARD_BENCHMARK_QUALIFIED":
        raise RuntimeError("harder cohort is not qualified")
    if tuple(row["instance_id"] for row in certificate["instances"]) != INSTANCE_IDS:
        raise RuntimeError("harder-cohort instance identities changed")
    b3a_result_path = repo_root / B3A_TERMINAL_RESULT
    if _sha256(b3a_result_path) != B3A_TERMINAL_RESULT_SHA256:
        raise RuntimeError("B3-A terminal transport evidence identity mismatch")
    b3a_result = json.loads(b3a_result_path.read_text(encoding="utf-8"))
    if b3a_result.get("terminal") != "B3A_EVALUABLE_COMPLETE" or b3a_result.get("model_calls") != 48:
        raise RuntimeError("B3-A terminal transport evidence is not complete")
    source_hashes = {relative: _sha256(repo_root / relative) for relative in SOURCE_FILES}
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "B4_A_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED",
        "research_question": "Does the frozen Balanced Exploration operator add final search value over the unmodified Structured Control on a fresh qualified higher-pressure L10M cohort?",
        "causal_arms": {
            "structured_control": "frozen B3-A Structured Control proposal, feedback, incumbent, and strict selection",
            "structured_balanced": "the exact frozen B3-A outcome-blind canonical move coverage operator",
        },
        "harder_cohort": {
            "benchmark_id": certificate["benchmark_id"],
            "certificate_path": HARD_BENCHMARK_CERTIFICATE,
            "certificate_sha256": HARD_BENCHMARK_CERTIFICATE_SHA256,
            "instances": list(INSTANCE_IDS),
            "qualification_terminal": certificate["terminal"],
            "model_calls_used_to_construct_or_qualify": certificate["model_call_count"],
            "landscape_outcomes_exposed_to_searcher": False,
        },
        "freshness": {
            "derivation": "100000 + uint32_be(sha256(protocol_id|instance_id|replicate|nonce)[0:4]) mod 900000, increment nonce on collision",
            "paired_identities": list(PAIRED_IDENTITIES),
            "excluded_consumed_identities": sorted(CONSUMED_IDENTITIES),
            "provider_sampling_seed_control": False,
            "identity_scope": "fresh outcome-blind paired prompt/session labels, not deterministic provider RNG seeds",
        },
        "matched_factors": {
            "interface": "component-grouped Structured JSON",
            "initial_candidate": canonical_spec(INITIAL_SPEC),
            "model": MODEL,
            "temperature": "provider default, identical across arms",
            "generation_budget": GENERATIONS_PER_TRAJECTORY,
            "evaluation_budget": GENERATIONS_PER_TRAJECTORY,
            "feedback": "same-arm incumbent score and behavioral vector only; treatment additionally receives its outcome-blind attempted-move ledger",
            "selection": "replace incumbent only with a semantic-valid, safe, strictly higher instance score",
            "pair_order": "alternate first arm across the nine frozen instance/identity pairs",
            "provider_failure": "seal the entire cohort NOT_EVALUABLE; no retry, replacement, or resume",
        },
        "balanced_operator": {
            "implementation": "exact scripts/research/l10m_b3a/exploration.py source bound below",
            "score_or_target_used_by_operator": False,
            "selection_after_evaluation": "identical strict improvement rule",
            "algorithm_change_from_b3a": False,
        },
        "estimands": {
            "primary": "paired final normalized progress, where (final-initial)/(qualified_global-initial) is computed within instance",
            "supporting": [
                "paired raw final-score wins/losses/ties",
                "global-optimum reach",
                "time to first strict improvement",
                "unique canonical move coverage",
                "unsafe and semantic-invalid counts",
            ],
        },
        "verdict_rules": {
            "B4A_BALANCED_SEARCH_VALUE_ESTABLISHED": "all nine pairs evaluable; median paired normalized-progress delta > 0; Balanced wins at least 6 pairs and loses 0; Balanced global-optimum reach is not lower; unsafe count does not increase; operator integrity passes",
            "B4A_BALANCED_SEARCH_VALUE_NOT_ESTABLISHED": "complete evaluable cohort without the full establishment pattern",
            "B4A_NOT_EVALUABLE_RUNTIME": "any provider, transport, isolation, ledger, evaluator, or execution-integrity failure",
        },
        "anti_post_hoc": {
            "generation_budget_reduced_to_two": False,
            "efficiency_claim_from_b3a_reused_as_primary": False,
            "future_time_or_token_efficiency_requires_separate_protocol": True,
        },
        "execution": {
            "authorized": True,
            "planned_model_calls": len(PAIRED_IDENTITIES) * len(ARMS) * GENERATIONS_PER_TRAJECTORY,
            "no_resume": True,
        },
        "transport": {
            "route": "proxy",
            "qualification_run_id": TRANSPORT_QUALIFICATION_RUN_ID,
            "qualification_sha256": TRANSPORT_QUALIFICATION_SHA256,
            "matching_formal_predecessor_result": B3A_TERMINAL_RESULT,
            "matching_formal_predecessor_result_sha256": B3A_TERMINAL_RESULT_SHA256,
            "matching_formal_predecessor_successful_model_calls": 48,
        },
        "claim_ceiling": "paired search-value evidence on the frozen finite synthetic B4 harder cohort only; not general model, end-to-end, device, user, safety-effect, or production evidence",
        "source": {
            "git_commit": _git_commit(repo_root),
            "source_sha256": source_hashes,
        },
    }


def canonical_manifest_sha256(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
