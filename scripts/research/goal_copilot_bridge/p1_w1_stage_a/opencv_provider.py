"""Frozen, lightweight RGB evidence provider for P1-W1 Stage A.

ORB scene geometry is the spatial path.  A target-local HSV histogram is the
independent identity path.  No online template update, future frame, detector,
tracker, pose truth, depth, or global database is used.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from stage_a import FrameEvidence, SpatialEvidence


ORB_FEATURES = 1000
RATIO_TEST = 0.75
HOMOGRAPHY_RANSAC_PX = 3.0
MIN_SCENE_MATCHES = 12
MIN_H_INLIER_RATIO = 0.50
MIN_TARGET_MATCHES = 6
MIN_KEYPOINT_SPREAD = 0.10
HSV_IDENTITY_MIN = 0.70
FUNDAMENTAL_ADVANTAGE_OVER_H = 0.15
C0_SCALE_RANGE = (0.92, 1.08)
C0_MAX_SHEAR = 0.08
C0_MAX_PERSPECTIVE = 0.0015


def _crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2, y2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    return image[y1:y2, x1:x2]


def _histogram(image: np.ndarray) -> np.ndarray | None:
    if image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(value, value, alpha=1.0, norm_type=cv2.NORM_L1)
    return value


class FrozenRgbProvider:
    def __init__(self, keyframe: np.ndarray, source_region: tuple[float, float, float, float]):
        if keyframe.ndim != 3 or keyframe.shape[2] != 3:
            raise ValueError("BGR keyframe required")
        if _crop(keyframe, source_region).size == 0:
            raise ValueError("source region is empty")
        self._shape = keyframe.shape[:2]
        self._source_region = source_region
        self._orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=20)
        gray = cv2.cvtColor(keyframe, cv2.COLOR_BGR2GRAY)
        self._keypoints, self._descriptors = self._orb.detectAndCompute(gray, None)
        if self._descriptors is None or len(self._keypoints) < MIN_SCENE_MATCHES:
            raise ValueError("keyframe has insufficient ORB support")
        x1, y1, x2, y2 = source_region
        indices = [index for index, point in enumerate(self._keypoints) if x1 <= point.pt[0] <= x2 and y1 <= point.pt[1] <= y2]
        if len(indices) < MIN_TARGET_MATCHES:
            raise ValueError("source region has insufficient target-local ORB support")
        self._target_descriptors = self._descriptors[indices]
        self._identity_histogram = _histogram(_crop(keyframe, source_region))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _matches(self, query: np.ndarray, train: np.ndarray) -> list[cv2.DMatch]:
        return [pair[0] for pair in self._matcher.knnMatch(query, train, k=2) if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]

    def evidence(self, frame_id: str, frame: np.ndarray) -> FrameEvidence:
        if frame.shape[:2] != self._shape:
            raise ValueError("frame size drift")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points, current_descriptors = self._orb.detectAndCompute(gray, None)
        if current_descriptors is None:
            return self._failed(frame_id)
        scene_matches = self._matches(self._descriptors, current_descriptors)
        homography = None
        h_ratio = 0.0
        spread = 0.0
        f_ratio = 0.0
        if len(scene_matches) >= MIN_SCENE_MATCHES:
            source = np.float32([self._keypoints[item.queryIdx].pt for item in scene_matches])
            current = np.float32([current_points[item.trainIdx].pt for item in scene_matches])
            homography, h_mask = cv2.findHomography(source, current, cv2.RANSAC, HOMOGRAPHY_RANSAC_PX)
            if h_mask is not None:
                h_ratio = float(np.mean(h_mask))
                span = np.ptp(source[h_mask.ravel().astype(bool)], axis=0) if np.any(h_mask) else np.zeros(2)
                spread = float(min(span[0] / self._shape[1], span[1] / self._shape[0]))
            if len(scene_matches) >= 8:
                _, f_mask = cv2.findFundamentalMat(source, current, cv2.FM_RANSAC, HOMOGRAPHY_RANSAC_PX, 0.99)
                if f_mask is not None:
                    f_ratio = float(np.mean(f_mask))
        geometry_ok = homography is not None and h_ratio >= MIN_H_INLIER_RATIO and spread >= MIN_KEYPOINT_SPREAD
        parallax_overreach = f_ratio >= h_ratio + FUNDAMENTAL_ADVANTAGE_OVER_H

        candidate = None
        identity = "INSUFFICIENT"
        if geometry_ok:
            corners = np.float32([[self._source_region[0], self._source_region[1]], [self._source_region[2], self._source_region[3]]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            box = (float(min(warped[:, 0])), float(min(warped[:, 1])), float(max(warped[:, 0])), float(max(warped[:, 1])))
            target_matches = self._matches(self._target_descriptors, current_descriptors)
            if len(target_matches) >= MIN_TARGET_MATCHES and _crop(frame, box).size:
                candidate = box
                current_histogram = _histogram(_crop(frame, box))
                similarity = 1.0 - cv2.compareHist(self._identity_histogram, current_histogram, cv2.HISTCMP_BHATTACHARYYA)
                identity = "SUPPORTED" if similarity >= HSV_IDENTITY_MIN else "REJECTED"

        center = np.float32([[(self._source_region[0] + self._source_region[2]) / 2, (self._source_region[1] + self._source_region[3]) / 2]]).reshape(-1, 1, 2)
        projected_x = None if homography is None else float(cv2.perspectiveTransform(center, homography)[0, 0, 0])
        bearing = None if projected_x is None else (projected_x / self._shape[1] - 0.5) * 90.0
        compatibility = "SUPPORTED" if candidate is not None else "INSUFFICIENT"
        c0_overreach = True
        if homography is not None:
            normalized = homography / homography[2, 2]
            scale_x = float(np.linalg.norm(normalized[:2, 0]))
            scale_y = float(np.linalg.norm(normalized[:2, 1]))
            shear = abs(float(np.dot(normalized[:2, 0], normalized[:2, 1])))
            perspective = max(abs(float(normalized[2, 0])), abs(float(normalized[2, 1])))
            c0_overreach = not (
                C0_SCALE_RANGE[0] <= scale_x <= C0_SCALE_RANGE[1]
                and C0_SCALE_RANGE[0] <= scale_y <= C0_SCALE_RANGE[1]
                and shear <= C0_MAX_SHEAR
                and perspective <= C0_MAX_PERSPECTIVE
                and not parallax_overreach
            )
        common = {
            "geometry_supported": geometry_ok,
            "motion_observable": len(scene_matches) >= MIN_SCENE_MATCHES,
            "geometry_degenerate": not geometry_ok,
            "bearing_estimate_deg": bearing,
            "bearing_uncertainty_deg": None if bearing is None else HOMOGRAPHY_RANSAC_PX / self._shape[1] * 90.0,
            "compatibility": compatibility,
        }
        return FrameEvidence(
            frame_id=frame_id,
            observation_supported=candidate is not None,
            candidate_region_xyxy=candidate,
            independent_identity_confirmation=identity,
            observability_reason="IN_VIEW_CANDIDATE" if candidate is not None else "NO_OBSERVATION",
            c0_spatial=SpatialEvidence(reference_frame="CAMERA_RELATIVE", translation_overreach=c0_overreach, **common),
            t0_spatial=SpatialEvidence(reference_frame="KEYFRAME_RELATIVE", translation_overreach=parallax_overreach, **common),
        )

    def _failed(self, frame_id: str) -> FrameEvidence:
        def spatial(reference: str) -> SpatialEvidence:
            return SpatialEvidence(reference, False, False, False, True, None, None, "INSUFFICIENT")
        return FrameEvidence(frame_id, False, None, "INSUFFICIENT", "NO_OBSERVATION", spatial("CAMERA_RELATIVE"), spatial("KEYFRAME_RELATIVE"))
