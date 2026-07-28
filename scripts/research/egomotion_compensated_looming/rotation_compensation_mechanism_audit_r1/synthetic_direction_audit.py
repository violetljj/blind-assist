from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as discovery,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)


PROTOCOL_ID = "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1"


def axis_quaternion_wxyz(
    axis: str, angle_degrees: float
) -> np.ndarray:
    half = math.radians(angle_degrees) / 2.0
    scalar = math.cos(half)
    vector = math.sin(half)
    values = {
        "pitch_x": (scalar, vector, 0.0, 0.0),
        "yaw_y": (scalar, 0.0, vector, 0.0),
        "roll_z": (scalar, 0.0, 0.0, vector),
    }
    return np.asarray(values[axis], dtype=np.float64)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float64), np.ones(len(points), dtype=np.float64))
    )
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def run_audit() -> dict[str, Any]:
    height, width = 1280, 720
    rng = np.random.default_rng(20260728)
    texture = rng.integers(0, 256, (height, width), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    valid = np.full_like(texture, 255, dtype=np.uint8)
    previous_pose = (
        np.zeros(3, dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )
    landmarks = np.asarray(
        (
            (discovery.INTRINSIC[0, 2], discovery.INTRINSIC[1, 2]),
            (300.0, 500.0),
            (430.0, 700.0),
            (250.0, 780.0),
        ),
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for axis in ("pitch_x", "yaw_y", "roll_z"):
        for signed_degrees in (-10.0, 10.0):
            current_pose = (
                np.zeros(3, dtype=np.float64),
                axis_quaternion_wxyz(axis, signed_degrees),
            )
            homography, angular_speed, _ = discovery.pair_geometry(
                previous_pose,
                current_pose,
                1.0,
                quaternion_component_order="wxyz",
            )
            current = cv2.warpPerspective(
                texture,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            correct = compensate_current_to_previous(
                current, valid, valid, homography
            )
            reverse = compensate_current_to_previous(
                current, valid, valid, np.linalg.inv(homography)
            )
            region = np.zeros_like(valid, dtype=bool)
            region[250 : height - 250, 210 : width - 210] = True
            region &= correct.valid_mask > 0
            region &= reverse.valid_mask > 0
            if int(region.sum()) == 0:
                raise ValueError("SYNTHETIC_COMPARISON_REGION_EMPTY")
            raw_mae = float(
                np.mean(
                    np.abs(
                        current[region].astype(np.float64)
                        - texture[region].astype(np.float64)
                    )
                )
            )
            correct_mae = float(
                np.mean(
                    np.abs(
                        correct.image[region].astype(np.float64)
                        - texture[region].astype(np.float64)
                    )
                )
            )
            reverse_mae = float(
                np.mean(
                    np.abs(
                        reverse.image[region].astype(np.float64)
                        - texture[region].astype(np.float64)
                    )
                )
            )
            current_landmarks = transform_points(landmarks, homography)
            recovered = transform_points(
                current_landmarks, np.linalg.inv(homography)
            )
            landmark_error = float(
                np.max(np.linalg.norm(recovered - landmarks, axis=1))
            )
            explicit_inverse = cv2.warpPerspective(
                current,
                np.linalg.inv(homography),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            inverse_map = cv2.warpPerspective(
                current,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warp_semantics_max_abs = int(
                np.max(
                    np.abs(
                        explicit_inverse.astype(np.int16)
                        - inverse_map.astype(np.int16)
                    )
                )
            )
            passed = (
                abs(angular_speed - 10.0) < 1e-9
                and landmark_error < 1e-9
                and correct_mae < raw_mae
                and correct_mae < reverse_mae
                and warp_semantics_max_abs == 0
            )
            rows.append(
                {
                    "axis": axis,
                    "signed_degrees": signed_degrees,
                    "raw_mae": raw_mae,
                    "correct_mae": correct_mae,
                    "reverse_mae": reverse_mae,
                    "landmark_roundtrip_max_error_px": landmark_error,
                    "warp_semantics_max_abs_gray": warp_semantics_max_abs,
                    "pass": passed,
                }
            )
    return {
        "schema": "rcle.rotation_compensation.synthetic_direction_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "pose_convention": "camera_to_world_wxyz",
        "relative_rotation": "R_current.T @ R_previous",
        "arms": ["none_raw", "correct", "reverse"],
        "case_count": len(rows),
        "all_pass": all(row["pass"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    if not result["all_pass"]:
        raise ValueError("SYNTHETIC_DIRECTION_AUDIT_FAILED")
    discovery.write_json(args.output.resolve(), result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
