"""Non-OCR functional portal-set proposals from repeated door-frame structure.

The representation groups near-vertical line segments that share upper and
lower boundaries. Adjacent door leaves become one portal lattice instead of
competing target instances. This module proposes geometry only; target identity
and approach safety remain separate authorities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PortalLatticeConfig:
    minimum_normalized_length: float = 0.06
    maximum_normalized_length: float = 0.45
    maximum_horizontal_fraction: float = 0.18
    endpoint_tolerance: float = 0.06
    duplicate_x_tolerance: float = 0.012
    minimum_distinct_posts: int = 3
    minimum_normalized_span: float = 0.12
    maximum_normalized_span: float = 0.92


@dataclass(frozen=True)
class PortalSetProposal:
    proposal_id: str
    family: str
    observed_post_box_xyxy: tuple[float, float, float, float]
    box_xyxy: tuple[float, float, float, float]
    post_count: int
    normalized_span: float
    normalized_height: float
    spacing_regularity: float
    horizontal_boundary_support: float
    proposal_score: float


@dataclass(frozen=True)
class _Post:
    x: float
    top: float
    bottom: float
    length: float


def _detect_posts(
    image: np.ndarray, config: PortalLatticeConfig
) -> tuple[list[_Post], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(equalized, 60, 160, apertureSize=3, L2gradient=True)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(equalized)[0]
    if detected is None:
        return [], edges
    height, width = gray.shape
    posts = []
    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in raw]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        normalized_length = length / height
        if not config.minimum_normalized_length <= normalized_length <= config.maximum_normalized_length:
            continue
        if abs(dx) / length > config.maximum_horizontal_fraction:
            continue
        posts.append(
            _Post(
                x=0.5 * (x1 + x2),
                top=min(y1, y2),
                bottom=max(y1, y2),
                length=length,
            )
        )
    return posts, edges


def _components(posts: Sequence[_Post], image_height: int, tolerance: float) -> list[list[_Post]]:
    parent = list(range(len(posts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    endpoint_limit = tolerance * image_height
    for left_index, left in enumerate(posts):
        for right_index in range(left_index + 1, len(posts)):
            right = posts[right_index]
            if abs(left.top - right.top) > endpoint_limit:
                continue
            if abs(left.bottom - right.bottom) > endpoint_limit:
                continue
            overlap = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
            if overlap >= 0.55 * min(left.length, right.length):
                union(left_index, right_index)
    grouped: dict[int, list[_Post]] = {}
    for index, post in enumerate(posts):
        grouped.setdefault(find(index), []).append(post)
    return list(grouped.values())


def _deduplicate_posts(
    posts: Sequence[_Post], image_width: int, tolerance: float
) -> list[_Post]:
    ordered = sorted(posts, key=lambda post: post.x)
    groups: list[list[_Post]] = []
    limit = tolerance * image_width
    for post in ordered:
        if not groups or abs(post.x - float(np.median([row.x for row in groups[-1]]))) > limit:
            groups.append([post])
        else:
            groups[-1].append(post)
    return [
        _Post(
            x=float(np.median([row.x for row in group])),
            top=float(np.median([row.top for row in group])),
            bottom=float(np.median([row.bottom for row in group])),
            length=float(np.median([row.length for row in group])),
        )
        for group in groups
    ]


def _boundary_support(
    edges: np.ndarray, left: float, right: float, top: float, bottom: float
) -> float:
    height, width = edges.shape
    x1, x2 = max(0, int(left)), min(width, int(right) + 1)
    if x2 <= x1:
        return 0.0
    radius = max(1, int(round(0.012 * height)))
    values = []
    for y in (top, bottom):
        y1, y2 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
        stripe = edges[y1:y2, x1:x2]
        if stripe.size:
            values.append(float(np.count_nonzero(stripe)) / stripe.size)
    return min(1.0, 8.0 * float(np.mean(values))) if values else 0.0


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def propose_portal_sets(
    image: np.ndarray, config: PortalLatticeConfig = PortalLatticeConfig()
) -> tuple[list[PortalSetProposal], dict[str, int]]:
    """Return repeated-frame portal lattices ranked by structural support."""
    if image is None or image.ndim != 3:
        raise ValueError("INVALID_BGR_IMAGE")
    height, width = image.shape[:2]
    posts, edges = _detect_posts(image, config)
    proposals = []
    for component in _components(posts, height, config.endpoint_tolerance):
        distinct = _deduplicate_posts(component, width, config.duplicate_x_tolerance)
        if len(distinct) < config.minimum_distinct_posts:
            continue
        xs = np.asarray([post.x for post in distinct], dtype=np.float32)
        span = float(xs[-1] - xs[0]) / width
        if not config.minimum_normalized_span <= span <= config.maximum_normalized_span:
            continue
        top = float(np.median([post.top for post in distinct]))
        bottom = float(np.median([post.bottom for post in distinct]))
        if bottom <= top:
            continue
        gaps = np.diff(xs)
        gap_median = float(np.median(gaps)) if gaps.size else 0.0
        regularity = (
            max(0.0, 1.0 - float(np.median(np.abs(gaps - gap_median))) / gap_median)
            if gap_median > 0.0
            else 0.0
        )
        boundary = _boundary_support(edges, float(xs[0]), float(xs[-1]), top, bottom)
        post_support = min(1.0, len(distinct) / 8.0)
        score = 0.45 * post_support + 0.35 * regularity + 0.20 * boundary
        pad_x = 0.5 * gap_median if gap_median > 0.0 else 0.015 * width
        observed_height = bottom - top
        inferred_height = max(observed_height, 2.0 * gap_median)
        center_y = 0.5 * (top + bottom)
        inferred_top = max(0.0, center_y - 0.5 * inferred_height)
        inferred_bottom = min(float(height), center_y + 0.5 * inferred_height)
        proposals.append(
            PortalSetProposal(
                proposal_id="",
                family="REPEATED_POST_LATTICE",
                observed_post_box_xyxy=(float(xs[0]), top, float(xs[-1]), bottom),
                box_xyxy=(
                    max(0.0, float(xs[0]) - pad_x),
                    inferred_top,
                    min(float(width), float(xs[-1]) + pad_x),
                    inferred_bottom,
                ),
                post_count=len(distinct),
                normalized_span=span,
                normalized_height=(bottom - top) / height,
                spacing_regularity=regularity,
                horizontal_boundary_support=boundary,
                proposal_score=score,
            )
        )
    handle_pair_count = 0
    for left_index, left in enumerate(posts):
        for right in posts[left_index + 1 :]:
            if right.x <= left.x:
                continue
            gap = right.x - left.x
            normalized_gap = gap / width
            if not 0.02 <= normalized_gap <= 0.18:
                continue
            length_similarity = min(left.length, right.length) / max(left.length, right.length)
            if length_similarity < 0.65:
                continue
            center_difference = abs(
                0.5 * (left.top + left.bottom) - 0.5 * (right.top + right.bottom)
            )
            if center_difference > 0.08 * height:
                continue
            mean_length = 0.5 * (left.length + right.length)
            length_to_gap = mean_length / gap
            if not 0.8 <= length_to_gap <= 5.0:
                continue
            endpoint_error = (
                abs(left.top - right.top) + abs(left.bottom - right.bottom)
            ) / (2.0 * 0.08 * height)
            alignment = max(0.0, 1.0 - endpoint_error)
            shape_support = min(1.0, length_to_gap / 1.5, 5.0 / length_to_gap)
            score = 0.40 * alignment + 0.35 * length_similarity + 0.25 * shape_support
            center_x = 0.5 * (left.x + right.x)
            center_y = 0.25 * (left.top + left.bottom + right.top + right.bottom)
            inferred_width = max(4.0 * gap, 2.0 * mean_length)
            inferred_height = max(5.0 * gap, 2.5 * mean_length)
            proposals.append(
                PortalSetProposal(
                    proposal_id="",
                    family="PAIRED_VERTICAL_HANDLE_APERTURE",
                    observed_post_box_xyxy=(
                        left.x,
                        min(left.top, right.top),
                        right.x,
                        max(left.bottom, right.bottom),
                    ),
                    box_xyxy=(
                        max(0.0, center_x - 0.5 * inferred_width),
                        max(0.0, center_y - 0.5 * inferred_height),
                        min(float(width), center_x + 0.5 * inferred_width),
                        min(float(height), center_y + 0.5 * inferred_height),
                    ),
                    post_count=2,
                    normalized_span=normalized_gap,
                    normalized_height=inferred_height / height,
                    spacing_regularity=length_similarity,
                    horizontal_boundary_support=alignment,
                    proposal_score=score,
                )
            )
            handle_pair_count += 1
    retained = []
    for proposal in sorted(proposals, key=lambda row: (-row.proposal_score, -row.post_count)):
        if any(_iou(proposal.box_xyxy, row.box_xyxy) >= 0.45 for row in retained):
            continue
        retained.append(proposal)
    ranked = [
        PortalSetProposal(
            proposal_id=f"portal-set-{index:03d}",
            family=proposal.family,
            observed_post_box_xyxy=proposal.observed_post_box_xyxy,
            box_xyxy=proposal.box_xyxy,
            post_count=proposal.post_count,
            normalized_span=proposal.normalized_span,
            normalized_height=proposal.normalized_height,
            spacing_regularity=proposal.spacing_regularity,
            horizontal_boundary_support=proposal.horizontal_boundary_support,
            proposal_score=proposal.proposal_score,
        )
        for index, proposal in enumerate(retained, start=1)
    ]
    return ranked, {
        "raw_vertical_segments": len(posts),
        "endpoint_components": len(_components(posts, height, config.endpoint_tolerance)),
        "portal_set_proposals": len(ranked),
        "raw_handle_pairs": handle_pair_count,
        "retained_lattice_proposals": sum(
            proposal.family == "REPEATED_POST_LATTICE" for proposal in ranked
        ),
        "retained_handle_pair_proposals": sum(
            proposal.family == "PAIRED_VERTICAL_HANDLE_APERTURE" for proposal in ranked
        ),
    }


def _center_inside(box: Sequence[float], region: Sequence[float]) -> bool:
    center_x = 0.5 * (box[0] + box[2])
    center_y = 0.5 * (box[1] + box[3])
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def _portal_set_member(box: Sequence[float], region: Sequence[float]) -> bool:
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    proposal_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    coverage = intersection / proposal_area if proposal_area > 0.0 else 0.0
    return _center_inside(box, region) and coverage >= 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    frame_by_index = {int(row["index"]): row for row in manifest["frames"]}
    truth_by_index = {
        int(index): row for index, row in audit["portal_set_truth"]["frames"].items()
    }
    rows = []
    admitted_frames = audit.get("admitted_frames", audit.get("admitted_ordered_frames"))
    if not admitted_frames:
        raise ValueError("NO_ADMITTED_PORTAL_SET_FRAMES")
    for frame_index in admitted_frames:
        source = frame_by_index[int(frame_index)]
        image = cv2.imread(source["local_path"], cv2.IMREAD_COLOR)
        proposals, diagnostics = propose_portal_sets(image)
        truth = truth_by_index[int(frame_index)]["portal_set_box_xyxy"]
        rows.append(
            {
                "frame_index": int(frame_index),
                "image_sha256": source["sha256"],
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "truth_box_xyxy": truth,
                "truth_retained_top1": bool(proposals)
                and _portal_set_member(proposals[0].box_xyxy, truth),
                "truth_retained_top3": any(
                    _portal_set_member(proposal.box_xyxy, truth) for proposal in proposals[:3]
                ),
                "diagnostics": diagnostics,
                "proposals": [asdict(proposal) for proposal in proposals[:10]],
            }
        )
    result = {
        "schema": "l10-functional-portal-lattice-mechanism-v1",
        "development_source": "CONSUMED_IFC_REVERSE_SIDE_TWO_FRAME_PROXY",
        "ocr_calls": 0,
        "configuration": asdict(PortalLatticeConfig()),
        "portal_set_membership_rule": "Candidate center is inside the frozen portal-set region and at least half of candidate area lies inside it.",
        "metrics": {
            "truth_retained_top1": sum(row["truth_retained_top1"] for row in rows),
            "truth_retained_top3": sum(row["truth_retained_top3"] for row in rows),
            "frames": len(rows),
        },
        "rows": rows,
        "claim_scope": "Consumed two-frame geometry mechanism only; no fresh generalization, target identity, access, traversability, active-view causality, arrival, user benefit, or safety evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
