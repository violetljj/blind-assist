"""Reference-conditioned commitment, independent of OCR/candidate ranking.

Inputs are deliberately pixels and a proposed mask only. Reference extraction
provenance and evaluator annotations belong to the caller, never this verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


CONTRACT = {
    "feature": "OpenCV SIFT defaults, native resolution",
    "matching": "Exact BF L2 two-nearest-neighbour Lowe ratio",
    "ratio": 0.7,
    "minimum_matches": 11,
    "minimum_inliers": 11,
    "homography": "RANSAC",
    "reprojection_pixels": 5.0,
    "minimum_inlier_fraction": 0.5,
    "minimum_projected_mask_iou": 0.5,
    "reference_rule": "At least one of two pre-supplied reference crops supports the unchanged proposal",
    "seed": 104729,
    "provenance": "https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html",
    "threshold_authority": "Preregistered Development defaults, not calibrated confidence or a probability",
}


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray | None
    shape: tuple[int, int]


def features(image: np.ndarray) -> Features:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    return Features(np.array([p.pt for p in keypoints], np.float32).reshape(-1, 2), descriptors, gray.shape)


def reference_support(reference: Features, query: Features, mask: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"supported": False, "reason": "INSUFFICIENT_FEATURES",
                           "matches": 0, "inliers": 0, "projected_mask_iou": None}
    if reference.descriptors is None or query.descriptors is None or len(query.descriptors) < 2:
        return row
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference.descriptors, query.descriptors, k=2)
    good = [a for a, b in pairs if a.distance < CONTRACT["ratio"] * b.distance]
    row["matches"] = len(good)
    row["reason"] = "INSUFFICIENT_DISTINCTIVE_MATCHES"
    if len(good) < CONTRACT["minimum_matches"]:
        return row
    source = reference.points[[m.queryIdx for m in good]]
    destination = query.points[[m.trainIdx for m in good]]
    cv2.setRNGSeed(CONTRACT["seed"])
    transform, inliers = cv2.findHomography(source, destination, cv2.RANSAC,
                                          CONTRACT["reprojection_pixels"])
    row["reason"] = "NO_HOMOGRAPHY"
    if transform is None or inliers is None:
        return row
    row["inliers"] = int(inliers.sum())
    row["inlier_fraction"] = row["inliers"] / len(good)
    row["reason"] = "INSUFFICIENT_GEOMETRIC_SUPPORT"
    if row["inliers"] < CONTRACT["minimum_inliers"] or row["inlier_fraction"] < CONTRACT["minimum_inlier_fraction"]:
        return row
    h, w = reference.shape
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    polygon = cv2.perspectiveTransform(corners[None], transform)[0]
    row["reason"] = "INVALID_PROJECTED_EXTENT"
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        return row
    # Clip before int conversion; projection can diverge for nearly singular fits.
    if np.max(np.abs(polygon)) > 10 * max(query.shape):
        return row
    projected = np.zeros(query.shape, np.uint8)
    cv2.fillConvexPoly(projected, np.rint(polygon).astype(np.int32), 1)
    candidate = mask.astype(bool)
    union = np.count_nonzero((projected != 0) | candidate)
    overlap = np.count_nonzero((projected != 0) & candidate)
    iou = overlap / union if union else 0.0
    row.update(projected_mask_iou=round(iou, 8), projected_polygon=polygon.round(4).tolist(),
               supported=iou >= CONTRACT["minimum_projected_mask_iou"])
    row["reason"] = "REFERENCE_GEOMETRY_SUPPORTS_PROPOSAL" if row["supported"] else "REFERENCE_GEOMETRY_DISAGREES_WITH_PROPOSAL"
    return row


class ReferenceCommitment:
    def __init__(self, reference_images: list[np.ndarray]):
        if len(reference_images) != 2:
            raise ValueError("Exactly two publicly supplied reference crops are required")
        self.references = [features(image) for image in reference_images]

    def verify(self, query_features: Features, proposed_mask: np.ndarray) -> dict[str, Any]:
        if proposed_mask.shape != query_features.shape:
            raise ValueError("Proposal mask and query must use the same pixel coordinates")
        witnesses = [reference_support(ref, query_features, proposed_mask) for ref in self.references]
        accepted = any(row["supported"] for row in witnesses)
        return {"accepted": accepted, "state": "REFERENCE_SUPPORTED_PROPOSAL" if accepted else "UNKNOWN",
                "references": witnesses, "reference_acquisition_cost": 2,
                "authority": "REFERENCE_CONDITIONED_ADDRESS_DOOR_SURROGATE_ONLY"}


def self_check() -> dict[str, str]:
    # A known translation gives independently specified correspondence truth.
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (180, 160), np.uint8)
    query = np.zeros((280, 340), np.uint8)
    query[50:230, 70:230] = reference
    verifier = ReferenceCommitment([reference, reference.copy()])
    observed = features(query)
    good = np.zeros_like(query); good[50:230, 70:230] = 1
    wrong = np.zeros_like(query); wrong[10:45, 250:320] = 1
    assert verifier.verify(observed, good)["accepted"]
    assert not verifier.verify(observed, wrong)["accepted"]
    assert not verifier.verify(features(np.zeros_like(query)), wrong)["accepted"]
    return {"translated_target": "PASS", "wrong_region": "PASS", "target_absent": "PASS"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
