#!/usr/bin/env python3
"""Fresh HierText replay with unique-referent truth and spatial ambiguity abstention."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import l10_hiertext_test_multiscale_goal_verifier as v1
import l10_hiertext_test_span_carrier_verifier as v2


PROTOCOL_SCHEMA = "blindassist-l10-hiertext-test-unique-referent-span-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-hiertext-test-unique-referent-span-verifier-result-v1"
V1_SELECT_COHORT = v1.select_cohort
V1_SHA256 = v1.sha256


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if path.resolve() == Path(v1.__file__).resolve():
        return V1_SHA256(Path(__file__).resolve(), chunk_size)
    return V1_SHA256(path, chunk_size)


def complete_truth_tokens(image: dict[str, Any]) -> list[str]:
    tokens = []
    for paragraph in image.get("paragraphs", []):
        for line in paragraph.get("lines", []):
            for word in line.get("words", []):
                token = v1.normalize(str(word.get("text", "")))
                if token:
                    tokens.append(token)
    return tokens


def select_cohort(
    annotations: list[dict[str, Any]],
    document_frequency: dict[str, int],
    documents: int,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    old_contract = {
        key: value
        for key, value in contract.items()
        if key not in {"cohort_size", "consumed_eligible_images"}
    }
    old_contract["cohort_size"] = int(contract["consumed_eligible_images"])
    consumed = {
        row["image_id"]
        for row in V1_SELECT_COHORT(annotations, document_frequency, documents, old_contract)
    }
    cohort = []
    for image in annotations:
        if image["image_id"] in consumed:
            continue
        width, height = int(image["image_width"]), int(image["image_height"])
        printed_words = v1.all_words(image)
        complete_tokens = complete_truth_tokens(image)
        complete_counts = Counter(complete_tokens)
        eligible = []
        for row in printed_words:
            token = row["token"]
            x0, y0, x1, y1 = v1.bbox(row["vertices"])
            if not (
                contract["minimum_token_length"] <= len(token) <= contract["maximum_token_length"]
                and token.isalpha()
                and complete_counts[token] == 1
                and not any(token != other and token in other for other in complete_tokens)
                and (x1 - x0) / width >= contract["minimum_normalized_width"]
                and (y1 - y0) / height >= contract["minimum_normalized_height"]
                and v1.information_bits(token, document_frequency, documents) >= contract["minimum_background_information_bits"]
            ):
                continue
            eligible.append(
                {
                    "token": token,
                    "display_text": row["text"],
                    "truth_bbox": [x0, y0, x1, y1],
                    "normalized_area": ((x1 - x0) * (y1 - y0)) / (width * height),
                }
            )
        if eligible:
            target = max(eligible, key=lambda row: (row["normalized_area"], row["token"]))
            cohort.append(
                {
                    "image_id": image["image_id"],
                    "width": width,
                    "height": height,
                    "truth_tokens": sorted(set(complete_tokens)),
                    **target,
                }
            )
        if len(cohort) == contract["cohort_size"]:
            break
    v1.require(len(cohort) == contract["cohort_size"], "INSUFFICIENT_UNIQUE_REFERENT_COHORT")
    for index, row in enumerate(cohort):
        selected = None
        for offset in range(1, len(cohort)):
            candidate = cohort[(index + offset) % len(cohort)]["token"]
            if all(candidate not in truth for truth in row["truth_tokens"]):
                selected = candidate
                break
        v1.require(selected is not None, f"NO_COMPLETE_TRUTH_ABSENT_QUERY:{row['image_id']}")
        row["absent_query"] = selected
    return cohort


def box_iou(left: list[list[float]], right: list[list[float]]) -> float:
    left_points, right_points = np.asarray(left), np.asarray(right)
    lx0, ly0, lx1, ly1 = left_points[:, 0].min(), left_points[:, 1].min(), left_points[:, 0].max(), left_points[:, 1].max()
    rx0, ry0, rx1, ry1 = right_points[:, 0].min(), right_points[:, 1].min(), right_points[:, 0].max(), right_points[:, 1].max()
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    union = max(1.0, (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection)
    return float(intersection / union)


def best_match(
    query: str,
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    allow_edit_one: bool,
) -> dict[str, Any] | None:
    matches = []
    for index, candidate in enumerate(candidates):
        token = candidate["token"]
        if candidate["confidence"] < contract["minimum_ocr_confidence"] or not token:
            continue
        match_box = candidate["box"]
        distance = v1.edit_distance(query, token)
        if token == query:
            match_kind, rank = "EXACT_TOKEN", 0
        elif (
            allow_edit_one
            and token.count(query) == 1
            and len(query) / len(token) >= contract["minimum_substring_query_fraction"]
        ):
            start = token.index(query)
            match_box = v2.projected_span_box(candidate["box"], start, start + len(query), len(token))
            match_kind, rank, distance = "EXACT_QUERY_SPAN", 1, 0
        elif (
            allow_edit_one
            and len(query) >= contract["minimum_edit_one_length"]
            and len(token) >= contract["minimum_edit_one_length"]
            and distance == 1
        ):
            match_kind, rank = "EDIT_ONE_TOKEN", 2
        else:
            continue
        matches.append(
            {
                **candidate,
                "box": match_box,
                "original_ocr_box": candidate["box"],
                "match_kind": match_kind,
                "match_rank": rank,
                "edit_distance": distance,
                "candidate_index": index,
            }
        )
    if not matches:
        return None
    best_rank = min(match["match_rank"] for match in matches)
    best = [match for match in matches if match["match_rank"] == best_rank]
    components: list[list[dict[str, Any]]] = []
    for match in best:
        overlapping = [component for component in components if any(box_iou(match["box"], other["box"]) >= contract["same_carrier_minimum_iou"] for other in component)]
        if not overlapping:
            components.append([match])
        else:
            primary = overlapping[0]
            primary.append(match)
            for extra in overlapping[1:]:
                primary.extend(extra)
                components.remove(extra)
    if len(components) > 1:
        return None
    selected = max(components[0], key=lambda match: (match["confidence"], -match["candidate_index"]))
    selected.pop("candidate_index")
    return selected


def main() -> None:
    v1.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    v1.RESULT_SCHEMA = RESULT_SCHEMA
    v1.select_cohort = select_cohort
    v1.best_match = best_match
    v1.sha256 = sha256
    v1.main()


if __name__ == "__main__":
    main()
