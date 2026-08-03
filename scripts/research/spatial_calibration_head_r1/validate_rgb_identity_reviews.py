#!/usr/bin/env python3
"""Validate two isolated capture-identity reviews and admit or hold the cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audit_rgb_identity import DEFAULT_AUDIT_PROTOCOL
from validate_protocol import sha256

ALLOWED = {"SAME_CAPTURE", "DISTINCT_CAPTURE", "UNKNOWN"}


def review_map(review: dict[str, Any], expected_candidate_sha256: str) -> tuple[str, dict[str, str]]:
    if review.get("schema") != "blindassist_spatial_calibration_head_r1_rgb_identity_review":
        raise ValueError("unexpected identity review schema")
    if review.get("candidate_sha256") != expected_candidate_sha256:
        raise ValueError("review candidate hash mismatch")
    reviewer = str(review.get("reviewer_id", "")).strip()
    if not reviewer:
        raise ValueError("reviewer_id required")
    labels = {}
    for row in review.get("edges", []):
        edge_id = str(row.get("edge_id", ""))
        label = str(row.get("label", ""))
        if not edge_id or edge_id in labels or label not in ALLOWED:
            raise ValueError("invalid or duplicate review edge")
        labels[edge_id] = label
    return reviewer, labels


def adjudicate(candidates: dict[str, Any], first: dict[str, Any], second: dict[str, Any], candidate_sha256: str) -> dict[str, Any]:
    expected = [row["edge_id"] for row in candidates.get("edges", [])]
    if len(expected) != len(set(expected)) or len(expected) != candidates.get("candidate_edge_count"):
        raise ValueError("candidate edge inventory mismatch")
    reviewer_a, labels_a = review_map(first, candidate_sha256)
    reviewer_b, labels_b = review_map(second, candidate_sha256)
    if reviewer_a == reviewer_b:
        raise ValueError("two distinct reviewers required")
    complete = set(labels_a) == set(expected) and set(labels_b) == set(expected)
    rows = []
    admitted = complete
    for edge_id in expected:
        left = labels_a.get(edge_id, "MISSING")
        right = labels_b.get(edge_id, "MISSING")
        passed = left == right == "DISTINCT_CAPTURE"
        admitted &= passed
        rows.append({"edge_id": edge_id, "reviewer_a_label": left, "reviewer_b_label": right, "passed": passed})
    return {
        "schema": "blindassist_spatial_calibration_head_r1_rgb_identity_review_validation",
        "candidate_sha256": candidate_sha256,
        "reviewer_ids": [reviewer_a, reviewer_b],
        "candidate_edge_count": len(expected),
        "complete": complete,
        "edges": rows,
        "terminal": (
            "SPATIAL_CALIBRATION_HEAD_R1_RGB_IDENTITY_REVIEW_ADMITTED"
            if admitted
            else "HOLD_COHORT_INDEPENDENCE"
        ),
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-protocol", type=Path, default=DEFAULT_AUDIT_PROTOCOL)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.audit_protocol.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if candidates.get("audit_protocol_sha256") != sha256(args.audit_protocol):
        raise ValueError("candidate audit protocol mismatch")
    candidate_hash = sha256(args.candidates)
    result = adjudicate(
        candidates,
        json.loads(args.review_a.read_text(encoding="utf-8")),
        json.loads(args.review_b.read_text(encoding="utf-8")),
        candidate_hash,
    )
    result.update({
        "audit_protocol_sha256": sha256(args.audit_protocol),
        "review_a_sha256": sha256(args.review_a),
        "review_b_sha256": sha256(args.review_b),
    })
    write_json_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "edges"}, indent=2))
    raise SystemExit(0 if result["terminal"].endswith("ADMITTED") else 1)


if __name__ == "__main__":
    main()
