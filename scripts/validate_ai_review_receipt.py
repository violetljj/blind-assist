#!/usr/bin/env python3
"""Validate hash-bound, independent GPT/Codex reviews and adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    pass


def _local(root: Path, relative: str, *, where: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ContractError(f"{where} must be a non-empty local path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes manifest root") from error
    if not path.is_file():
        raise ContractError(f"{where} is not a local file")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _load_bound(root: Path, relative: Any, expected: Any, *, where: str) -> tuple[dict[str, Any], str]:
    expected_sha = _sha_text(expected, where=f"{where}.sha256")
    path = _local(root, relative, where=f"{where}.path")
    if _sha(path) != expected_sha:
        raise ContractError(f"{where} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{where} must contain a JSON object")
    return value, expected_sha


def validate_model_pass(
    review: dict[str, Any],
    *,
    policy: dict[str, Any],
    where: str,
    allowed_roles: set[str],
) -> tuple[str, str, str]:
    reviewer_id = _text(review.get("reviewer_id"), where=f"{where}.reviewer_id")
    if review.get("reviewer_type") != "ai_model":
        raise ContractError(f"{where}.reviewer_type must be ai_model")
    role = _text(review.get("reviewer_role"), where=f"{where}.reviewer_role")
    if role not in allowed_roles:
        raise ContractError(f"{where}.reviewer_role is not allowed")
    for key in ("provider", "model", "model_version", "review_run_id", "workflow_id"):
        _text(review.get(key), where=f"{where}.{key}")
    prompt_sha = _sha_text(review.get("prompt_sha256"), where=f"{where}.prompt_sha256")
    input_sha = _sha_text(review.get("input_sha256"), where=f"{where}.input_sha256")
    if review.get("isolated_context") is not True or review.get("other_review_visible_before_submission") is not False:
        raise ContractError(f"{where} must be an isolated pass blind to other reviews")
    if policy.get("candidate_output_hidden_from_reviewers") is True and review.get("candidate_output_visible") is not False:
        raise ContractError(f"{where} must be blind to candidate output")
    confidence = review.get("confidence")
    minimum = policy.get("minimum_confidence", 0.65)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not float(minimum) <= float(confidence) <= 1.0:
        raise ContractError(f"{where}.confidence is below policy or outside [0,1]")
    if review.get("abstained") is not False or review.get("abstain_reasons") not in ([], None):
        raise ContractError(f"{where} abstained and cannot grant review authority")
    return reviewer_id, role, input_sha + prompt_sha


def validate_consensus_receipt(
    receipt: dict[str, Any],
    *,
    policy: dict[str, Any],
    subject_id: str,
    where: str,
) -> dict[str, Any]:
    """Validate an inline two-pass receipt used by P3, privacy, geometry, and release gates."""
    if receipt.get("schema") != policy.get("receipt_schema") or receipt.get("subject_id") != subject_id:
        raise ContractError(f"{where} schema/subject binding mismatch")
    reviews = receipt.get("reviews")
    required_roles = policy.get("required_reviewer_roles", ["gpt_multimodal_reviewer", "codex_evidence_reviewer"])
    if not isinstance(reviews, list) or len(reviews) != len(required_roles):
        raise ContractError(f"{where} must contain exactly {len(required_roles)} independent reviews")
    reviewer_ids: list[str] = []
    roles: list[str] = []
    input_hashes: list[str] = []
    verdicts: list[str] = []
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            raise ContractError(f"{where}.reviews[{index}] must be an object")
        reviewer_id, role, combined_hash = validate_model_pass(
            review,
            policy=policy,
            where=f"{where}.reviews[{index}]",
            allowed_roles=set(required_roles),
        )
        verdict = review.get("verdict")
        if verdict not in {"accept", "reject"}:
            raise ContractError(f"{where}.reviews[{index}].verdict must be accept or reject")
        reviewer_ids.append(reviewer_id)
        roles.append(role)
        input_hashes.append(combined_hash[:64])
        verdicts.append(verdict)
    if len(set(reviewer_ids)) != len(reviewer_ids) or set(roles) != set(required_roles):
        raise ContractError(f"{where} requires distinct reviewer runs satisfying every role")
    if len(set(input_hashes)) != 1 or receipt.get("input_sha256") != input_hashes[0]:
        raise ContractError(f"{where} reviews do not bind the declared common input")
    consensus = receipt.get("consensus")
    if not isinstance(consensus, dict):
        raise ContractError(f"{where}.consensus must be an object")
    method = consensus.get("method")
    disposition = consensus.get("disposition")
    if len(set(verdicts)) == 1:
        if method != "model_consensus" or disposition != verdicts[0]:
            raise ContractError(f"{where}.consensus does not match independent reviews")
    else:
        if method != "independent_ai_adjudicator":
            raise ContractError(f"{where} disagreement requires a fresh AI adjudicator")
        adjudicator = consensus.get("adjudicator")
        if not isinstance(adjudicator, dict):
            raise ContractError(f"{where}.consensus.adjudicator must be an object")
        adjudicator_id, _, adjudicator_hash = validate_model_pass(
            adjudicator,
            policy=policy,
            where=f"{where}.consensus.adjudicator",
            allowed_roles=set(policy.get("allowed_adjudicator_roles", ["gpt_adjudicator", "codex_adjudicator"])),
        )
        if adjudicator_id in reviewer_ids or adjudicator_hash[:64] != input_hashes[0]:
            raise ContractError(f"{where}.consensus.adjudicator is not an independent pass on the same input")
        if disposition != adjudicator.get("verdict") or disposition not in {"accept", "reject"}:
            raise ContractError(f"{where}.consensus disposition differs from adjudicator")
    if disposition != "accept":
        raise ContractError(f"{where} did not grant AI review authority")
    return {
        "input_sha256": input_hashes[0],
        "reviewer_ids": reviewer_ids,
        "reviewer_roles": sorted(roles),
        "method": method,
        "disposition": disposition,
    }


def validate_episode_review(row: dict[str, Any], *, root: Path, policy: dict[str, Any], where: str) -> dict[str, Any]:
    paths = row.get("independent_review_paths")
    hashes = row.get("independent_review_sha256s")
    required_count = policy.get("required_independent_review_count", 2)
    if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != required_count or len(hashes) != required_count:
        raise ContractError(f"{where} must contain exactly {required_count} independent review files and hashes")
    required_roles = policy.get("required_reviewer_roles", ["gpt_multimodal_reviewer", "codex_evidence_reviewer"])
    if not isinstance(required_roles, list) or len(required_roles) != required_count or len(set(required_roles)) != required_count:
        raise ContractError("independent AI review roles are invalid")
    reviews: list[dict[str, Any]] = []
    reviewer_ids: list[str] = []
    reviewer_roles: list[str] = []
    input_hashes: list[str] = []
    for index, (path, expected) in enumerate(zip(paths, hashes)):
        review, _ = _load_bound(root, path, expected, where=f"{where}.independent_reviews[{index}]")
        if review.get("schema") != policy.get("review_schema") or review.get("episode_id") != row.get("episode_id"):
            raise ContractError(f"{where}.independent review schema/episode binding mismatch")
        reviewer_id, reviewer_role, combined_hash = validate_model_pass(
            review,
            policy=policy,
            where=f"{where}.independent_reviews[{index}]",
            allowed_roles=set(required_roles),
        )
        for key in ("should_alert", "critical"):
            if not isinstance(review.get(key), bool):
                raise ContractError(f"{where}.independent review {key} must be boolean")
        if row.get("pair_role") == "positive":
            for key in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
                value = review.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ContractError(f"{where}.independent review {key} must be a non-negative integer")
        else:
            if any(review.get(key) is not None for key in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms")):
                raise ContractError(f"{where}.negative review must keep positive anchors null")
            _text(review.get("negative_reason"), where=f"{where}.independent review negative_reason")
        _text(review.get("route_relation"), where=f"{where}.independent review route_relation")
        _text(review.get("criticality_reason"), where=f"{where}.independent review criticality_reason")
        reviewer_ids.append(reviewer_id)
        reviewer_roles.append(reviewer_role)
        input_hashes.append(combined_hash[:64])
        reviews.append(review)
    expected_ids = row.get("annotation_reviewer_ids", [])
    if len(set(reviewer_ids)) != required_count or set(reviewer_ids) != set(expected_ids):
        raise ContractError(f"{where}.independent reviewer identities do not match the manifest")
    if set(reviewer_roles) != set(required_roles):
        raise ContractError(f"{where}.independent reviewer roles do not satisfy policy")
    if len(set(input_hashes)) != 1:
        raise ContractError(f"{where}.independent reviews do not bind the same input")
    if row.get("review_input_sha256") is not None and row.get("review_input_sha256") != input_hashes[0]:
        raise ContractError(f"{where}.review_input_sha256 differs from reviews")

    adjudication, _ = _load_bound(
        root,
        row.get("adjudication_evidence_path"),
        row.get("adjudication_evidence_sha256"),
        where=f"{where}.adjudication",
    )
    if adjudication.get("schema") != policy.get("adjudication_schema") or adjudication.get("episode_id") != row.get("episode_id"):
        raise ContractError(f"{where}.adjudication schema/episode binding mismatch")
    if adjudication.get("input_review_sha256s") != hashes:
        raise ContractError(f"{where}.adjudication does not bind the ordered independent review hashes")
    tolerance = policy.get("anchor_agreement_tolerance_ms")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ContractError("independent review anchor tolerance is invalid")
    labels_agree = all(review["should_alert"] == reviews[0]["should_alert"] and review["critical"] == reviews[0]["critical"] for review in reviews)
    anchors_agree = True
    if row.get("pair_role") == "positive":
        anchors_agree = all(
            max(review[key] for review in reviews) - min(review[key] for review in reviews) <= tolerance
            for key in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms")
        )
    method = adjudication.get("method")
    if labels_agree and anchors_agree:
        if method != "model_consensus":
            raise ContractError(f"{where}.adjudication method must be model_consensus")
    else:
        if method != "independent_ai_adjudicator":
            raise ContractError(f"{where}.review disagreement requires an independent AI adjudicator")
        adjudicator_id, _, _ = validate_model_pass(
            adjudication,
            policy=policy,
            where=f"{where}.adjudication",
            allowed_roles=set(policy.get("allowed_adjudicator_roles", ["gpt_adjudicator", "codex_adjudicator"])),
        )
        if adjudicator_id in reviewer_ids:
            raise ContractError(f"{where}.adjudicator must be a fresh model pass")
    for key in ("should_alert", "critical", "criticality_reason"):
        manifest_key = {"should_alert": "expected_should_alert", "critical": "expected_critical"}.get(key, key)
        if adjudication.get(key) != row.get(manifest_key):
            raise ContractError(f"{where}.manifest {manifest_key} differs from AI adjudication")
    if row.get("pair_role") == "positive":
        for key in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
            if adjudication.get(key) != row.get(key):
                raise ContractError(f"{where}.manifest {key} differs from AI adjudication")
    elif adjudication.get("negative_reason") != row.get("negative_reason"):
        raise ContractError(f"{where}.manifest negative_reason differs from AI adjudication")
    return {"reviewer_count": len(reviews), "reviewer_roles": sorted(reviewer_roles), "adjudication_method": method}
