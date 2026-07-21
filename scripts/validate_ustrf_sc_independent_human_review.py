#!/usr/bin/env python3
"""Validate two independent USTRF human reviews and hash-bound adjudication."""

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


def _load_bound(root: Path, relative: Any, expected: Any, *, where: str) -> tuple[dict[str, Any], str]:
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ContractError(f"{where}.sha256 must be a lowercase SHA-256")
    path = _local(root, relative, where=f"{where}.path")
    if _sha(path) != expected:
        raise ContractError(f"{where} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{where} must contain a JSON object")
    return value, expected


def validate_episode_review(row: dict[str, Any], *, root: Path, policy: dict[str, Any], where: str) -> dict[str, Any]:
    paths = row.get("independent_review_paths")
    hashes = row.get("independent_review_sha256s")
    required_count = policy.get("required_independent_review_count", 2)
    if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != required_count or len(hashes) != required_count:
        raise ContractError(f"{where} must contain exactly {required_count} independent review files and hashes")
    reviews: list[dict[str, Any]] = []
    reviewer_ids: list[str] = []
    for index, (path, expected) in enumerate(zip(paths, hashes)):
        review, _ = _load_bound(root, path, expected, where=f"{where}.independent_reviews[{index}]")
        if review.get("schema") != policy.get("review_schema") or review.get("episode_id") != row.get("episode_id"):
            raise ContractError(f"{where}.independent review schema/episode binding mismatch")
        reviewer_id = review.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip() or review.get("reviewer_type") != "human":
            raise ContractError(f"{where}.independent review requires a concrete human reviewer")
        if review.get("model_assistance_used") is not False or review.get("other_review_visible_before_submission") is not False:
            raise ContractError(f"{where}.independent review must be blind to models and the other review")
        if review.get("should_alert") is not row.get("expected_should_alert") or review.get("critical") is not row.get("expected_critical"):
            # Disagreement is allowed only through independent adjudication below.
            pass
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
            if not isinstance(review.get("negative_reason"), str) or not review["negative_reason"].strip():
                raise ContractError(f"{where}.negative review requires a reason")
        if not isinstance(review.get("route_relation"), str) or not review["route_relation"].strip():
            raise ContractError(f"{where}.independent review requires route_relation")
        if not isinstance(review.get("criticality_reason"), str) or not review["criticality_reason"].strip():
            raise ContractError(f"{where}.independent review requires criticality_reason")
        reviewer_ids.append(reviewer_id)
        reviews.append(review)
    if len(set(reviewer_ids)) != required_count or set(reviewer_ids) != set(row.get("annotation_reviewer_ids", [])):
        raise ContractError(f"{where}.independent reviewer identities do not match the manifest")

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
        if method not in {"reviewer_consensus", "independent_human_adjudicator"}:
            raise ContractError(f"{where}.adjudication method is invalid")
    elif method != "independent_human_adjudicator":
        raise ContractError(f"{where}.review disagreement requires an independent human adjudicator")
    if method == "independent_human_adjudicator":
        adjudicator_id = adjudication.get("adjudicator_id")
        if not isinstance(adjudicator_id, str) or not adjudicator_id.strip() or adjudicator_id in reviewer_ids:
            raise ContractError(f"{where}.adjudicator must be a third independent human")
    for key in ("should_alert", "critical", "criticality_reason"):
        manifest_key = {"should_alert": "expected_should_alert", "critical": "expected_critical"}.get(key, key)
        if adjudication.get(key) != row.get(manifest_key):
            raise ContractError(f"{where}.manifest {manifest_key} differs from independent adjudication")
    if row.get("pair_role") == "positive":
        for key in ("first_visible_ms", "alertable_start_ms", "passed_or_cleared_ms"):
            if adjudication.get(key) != row.get(key):
                raise ContractError(f"{where}.manifest {key} differs from independent adjudication")
    elif adjudication.get("negative_reason") != row.get("negative_reason"):
        raise ContractError(f"{where}.manifest negative_reason differs from independent adjudication")
    return {"reviewer_count": len(reviews), "adjudication_method": method}
