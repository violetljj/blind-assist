"""Frozen protocol manifest for B5-A fresh generalization replication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, canonical_spec


PROTOCOL_ID = "L10M-B5-A-FRESH-GENERALIZATION-REPLICATION-V1"
ARMS = ("structured_control", "structured_balanced")
INSTANCE_IDS = ("obsidian", "coral", "silver")
REPLICATES_PER_INSTANCE = 3
GENERATIONS_PER_TRAJECTORY = 8
MODEL = "gpt-5.6-sol"
FRESH_COHORT_CERTIFICATE = "artifacts.local/evidence/l10m_b5/fresh_harder_v1/certificate.json"
FRESH_COHORT_CERTIFICATE_SHA256 = "22be26089adaaa7d3302ea7f965b7373d642ac26033430046e89b8d828a9b446"
FRESH_COHORT_SOURCE_COMMIT = "52107056c420d92a61e2ff957dfcd56dd0a05205"
B4A_TERMINAL_RESULT = "artifacts.local/evidence/l10m_b4/b4a_v2/runs/b4av2-20260820T133016-815ed378/result.json"
B4A_TERMINAL_RESULT_SHA256 = "50102673579283c1ab4552c3827eb98d297e0e5b19c22dfdf28042b2280a1370"
TRANSPORT_QUALIFICATION_SHA256 = "2af462f351814045dafcf488781fb8e914bd5847abd851a61ac7d962d13a0e1b"
TRANSPORT_QUALIFICATION_RUN_ID = "b1-i0-proxy-20260820T025833-4e438512"
CONSUMED_IDENTITIES = {
    17, 29, 43, 53, 71, 89, 1768, 7368, 1872,
    519302, 862260, 549875, 858684, 452936, 717980, 206383, 545415, 636402,
    798051, 265768, 260706, 504836, 525186, 550482, 306391, 749048, 782903,
}
SOURCE_FILES = (
    "scripts/research/l10m_b0/evaluation.py",
    "scripts/research/l10m_b0/b0c_precedence.py",
    "scripts/research/l10m_b1/evaluator.py",
    "scripts/research/l10m_b1/policy_space.py",
    "scripts/research/l10m_b1/protocol.py",
    "scripts/research/l10m_b1/provider_transport.py",
    "scripts/research/l10m_b1/run_search.py",
    "scripts/research/l10m_b3a/exploration.py",
    "scripts/research/l10m_b4/hard_benchmark.py",
    "scripts/research/l10m_b4/certify_hard_benchmark.py",
    "scripts/research/l10m_b4/fresh_benchmark_v1.json",
    "scripts/research/l10m_b4/fresh_benchmark.py",
    "scripts/research/l10m_b4/certify_fresh_benchmark.py",
    "scripts/research/l10m_b4/run_b4a.py",
    "scripts/research/l10m_b4/protocol_b5a.py",
    "scripts/research/l10m_b4/run_b5a.py",
    "scripts/research/l10m_b4/analyze_b5a.py",
    "scripts/research/l10m_b4/summarize_b5a.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
                {"instance_id": instance_id, "replicate": replicate, "paired_identity": identity}
            )
    return tuple(rows)


PAIRED_IDENTITIES = derive_fresh_identities()


def build_protocol_manifest(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    certificate_path = repo_root / FRESH_COHORT_CERTIFICATE
    if _sha256(certificate_path) != FRESH_COHORT_CERTIFICATE_SHA256:
        raise RuntimeError("fresh-cohort certificate identity mismatch")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("terminal") != "B5_FRESH_HARDER_COHORT_QUALIFIED":
        raise RuntimeError("fresh harder cohort is not qualified")
    if certificate.get("source_commit") != FRESH_COHORT_SOURCE_COMMIT:
        raise RuntimeError("fresh-cohort construction commit mismatch")
    if tuple(row["instance_id"] for row in certificate["instances"]) != INSTANCE_IDS:
        raise RuntimeError("fresh-cohort instance identities changed")
    if any(row["shortest_strict_steps_to_global_optimum"] < 5 for row in certificate["instances"]):
        raise RuntimeError("fresh cohort does not retain five-step search pressure")
    b4a_path = repo_root / B4A_TERMINAL_RESULT
    if _sha256(b4a_path) != B4A_TERMINAL_RESULT_SHA256:
        raise RuntimeError("B4-A terminal evidence identity mismatch")
    b4a = json.loads(b4a_path.read_text(encoding="utf-8"))
    if (
        b4a.get("terminal") != "B4A_EVALUABLE_COMPLETE"
        or b4a.get("scientific_verdict") != "B4A_BALANCED_SEARCH_VALUE_ESTABLISHED"
        or b4a.get("model_calls") != 144
    ):
        raise RuntimeError("B4-A predecessor is not the admitted complete terminal")
    source_hashes = {relative: _sha256(repo_root / relative) for relative in SOURCE_FILES}
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "B5_A_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED",
        "research_question": "Does the exact B4-A Balanced Exploration operator replicate its relative final search value over Structured Control on a fresh qualified harder L10M cohort?",
        "mode": "FORMAL_CONFIRMATION",
        "causal_arms": {
            "structured_control": "exact B4-A V2 Structured Control proposal, feedback, incumbent, and strict selection",
            "structured_balanced": "exact B4-A V2 outcome-blind canonical move coverage operator",
        },
        "fresh_harder_cohort": {
            "benchmark_id": certificate["benchmark_id"],
            "certificate_path": FRESH_COHORT_CERTIFICATE,
            "certificate_sha256": FRESH_COHORT_CERTIFICATE_SHA256,
            "construction_source_commit": FRESH_COHORT_SOURCE_COMMIT,
            "instances": list(INSTANCE_IDS),
            "qualification_terminal": certificate["terminal"],
            "model_calls_used_to_construct_or_qualify": certificate["model_call_count"],
            "minimum_strict_steps_to_global_optimum": 5,
            "landscape_outcomes_exposed_to_searcher": False,
            "search_arm_outcomes_used_for_selection": False,
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
            "feedback": "same-arm incumbent score and behavioral vector only; Balanced additionally receives its outcome-blind attempted-move ledger",
            "selection": "replace incumbent only with a semantic-valid, safe, strictly higher instance score",
            "pair_order": "alternate first arm across the nine frozen instance/identity pairs",
            "provider_failure": "seal the entire cohort NOT_EVALUABLE; no retry, replacement, or resume",
        },
        "balanced_operator": {
            "implementation": "scripts/research/l10m_b3a/exploration.py bound byte-for-byte in source_sha256",
            "algorithm_change_from_b4a": False,
            "progress_conditioning": False,
            "search_state_memory_beyond_b4_attempted_move_ledger": False,
            "score_or_target_used_by_operator": False,
        },
        "estimands": {
            "primary": "paired final normalized progress within instance",
            "supporting": [
                "paired win/tie/loss",
                "global-optimum reach",
                "time to first strict improvement",
                "unique canonical move and candidate coverage",
                "unsafe and semantic-invalid counts",
                "matched model-call cost",
            ],
        },
        "verdict_rules": {
            "B5A_GENERALIZATION_REPLICATED_ADMITTED_L10M_SEARCH_OPERATOR": "all nine pairs evaluable; median paired normalized-progress delta > 0; Balanced wins at least 6 pairs and loses 0; Balanced global-optimum reach is not lower; unsafe and semantic-invalid counts do not increase; model-call cost remains matched; operator integrity passes",
            "B5A_GENERALIZATION_NOT_REPLICATED": "complete evaluable cohort without the full replication pattern",
            "B5A_NOT_EVALUABLE_RUNTIME": "any provider, transport, isolation, ledger, evaluator, or execution-integrity failure",
        },
        "global_optimum_boundary": "positive global-optimum reach is not required for B5-A replication; B5-B may address completion only after an evaluable B5-A pass",
        "execution": {
            "authorized": True,
            "planned_model_calls": len(PAIRED_IDENTITIES) * len(ARMS) * GENERATIONS_PER_TRAJECTORY,
            "per_arm_model_calls": len(PAIRED_IDENTITIES) * GENERATIONS_PER_TRAJECTORY,
            "no_retry": True,
            "no_resume": True,
        },
        "transport": {
            "route": "proxy",
            "qualification_run_id": TRANSPORT_QUALIFICATION_RUN_ID,
            "qualification_sha256": TRANSPORT_QUALIFICATION_SHA256,
            "matching_formal_predecessor_result": B4A_TERMINAL_RESULT,
            "matching_formal_predecessor_result_sha256": B4A_TERMINAL_RESULT_SHA256,
            "matching_formal_predecessor_successful_model_calls": 144,
        },
        "claim_ceiling": "fresh-cohort replication evidence for the unchanged Balanced operator within the qualified finite synthetic L10M benchmark family; not general algorithm or model superiority, complete search, device, user, safety-effect, or production evidence",
        "source": {"git_commit": _git_commit(repo_root), "source_sha256": source_hashes},
    }


def canonical_manifest_sha256(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
