#!/usr/bin/env python3
"""Fresh HierText test replay for query-span projection inside merged OCR lines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import l10_hiertext_test_multiscale_goal_verifier as v1


PROTOCOL_SCHEMA = "blindassist-l10-hiertext-test-span-carrier-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-hiertext-test-span-carrier-verifier-result-v1"
V1_SELECT_COHORT = v1.select_cohort
V1_SHA256 = v1.sha256


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if path.resolve() == Path(v1.__file__).resolve():
        return V1_SHA256(Path(__file__).resolve(), chunk_size)
    return V1_SHA256(path, chunk_size)


def select_cohort(
    annotations: list[dict[str, Any]],
    document_frequency: dict[str, int],
    documents: int,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix_contract = {
        key: value
        for key, value in contract.items()
        if key not in {"cohort_size", "skip_eligible_images"}
    }
    skip = int(contract["skip_eligible_images"])
    size = int(contract["cohort_size"])
    prefix_contract["cohort_size"] = skip + size
    prefix = V1_SELECT_COHORT(annotations, document_frequency, documents, prefix_contract)
    cohort = prefix[skip:]
    v1.require(len(cohort) == size, "FRESH_COHORT_SIZE_MISMATCH")
    for index, row in enumerate(cohort):
        selected = None
        for offset in range(1, len(cohort)):
            candidate = cohort[(index + offset) % len(cohort)]["token"]
            if all(candidate not in truth for truth in row["truth_tokens"]):
                selected = candidate
                break
        v1.require(selected is not None, f"NO_EXHAUSTIVE_ABSENT_QUERY:{row['image_id']}")
        row["absent_query"] = selected
    return cohort


def projected_span_box(box: list[list[float]], start: int, end: int, length: int) -> list[list[float]]:
    points = np.asarray(box, dtype=np.float32)
    v1.require(points.shape == (4, 2), "UNEXPECTED_OCR_QUADRILATERAL")
    start_fraction, end_fraction = start / length, end / length
    top_start = points[0] + start_fraction * (points[1] - points[0])
    top_end = points[0] + end_fraction * (points[1] - points[0])
    bottom_start = points[3] + start_fraction * (points[2] - points[3])
    bottom_end = points[3] + end_fraction * (points[2] - points[3])
    return np.asarray([top_start, top_end, bottom_end, bottom_start], dtype=float).tolist()


def best_match(
    query: str,
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    allow_edit_one: bool,
) -> dict[str, Any] | None:
    ranked = []
    for index, candidate in enumerate(candidates):
        token = candidate["token"]
        if candidate["confidence"] < contract["minimum_ocr_confidence"] or not token:
            continue
        match_kind = None
        match_box = candidate["box"]
        distance = v1.edit_distance(query, token)
        if token == query:
            match_kind = "EXACT_TOKEN"
            rank = 0
        elif (
            allow_edit_one
            and token.count(query) == 1
            and len(query) / len(token) >= contract["minimum_substring_query_fraction"]
        ):
            start = token.index(query)
            match_box = projected_span_box(candidate["box"], start, start + len(query), len(token))
            match_kind = "EXACT_QUERY_SPAN"
            rank = 1
            distance = 0
        elif (
            allow_edit_one
            and len(query) >= contract["minimum_edit_one_length"]
            and len(token) >= contract["minimum_edit_one_length"]
            and distance == 1
        ):
            match_kind = "EDIT_ONE_TOKEN"
            rank = 2
        else:
            continue
        ranked.append((rank, distance, -candidate["confidence"], candidate["source"], index, candidate, match_box, match_kind))
    if not ranked:
        return None
    rank, distance, _, _, _, candidate, match_box, match_kind = min(ranked)
    return {
        **candidate,
        "box": match_box,
        "original_ocr_box": candidate["box"],
        "match_kind": match_kind,
        "match_rank": rank,
        "edit_distance": distance,
    }


def main() -> None:
    v1.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    v1.RESULT_SCHEMA = RESULT_SCHEMA
    v1.select_cohort = select_cohort
    v1.best_match = best_match
    v1.sha256 = sha256
    v1.main()


if __name__ == "__main__":
    main()
