"""Validate that the frozen machine-readable protocol matches the geometry core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import core


EXPECTED_SCHEMA = "blindassist_known_camera_height_ground_scale_protocol_v1"
EXPECTED_PROTOCOL_ID = "KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0"


def validate(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected protocol schema")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("unexpected protocol id")
    operator = protocol["operator"]
    expected = {
        "lower_roi_start_fraction": core.LOWER_ROI_START_FRACTION,
        "ransac_seed": core.RANSAC_SEED,
        "ransac_iterations": core.RANSAC_ITERATIONS,
        "maximum_candidates": core.MAXIMUM_CANDIDATES,
        "minimum_candidates": core.MINIMUM_CANDIDATES,
        "minimum_inliers": core.MINIMUM_INLIERS,
        "minimum_inlier_fraction": core.MINIMUM_INLIER_FRACTION,
        "minimum_abs_normal_y": core.MINIMUM_ABS_NORMAL_Y,
        "maximum_normalized_plane_residual": core.MAXIMUM_NORMALIZED_PLANE_RESIDUAL,
        "camera_height_range_m": list(core.CAMERA_HEIGHT_RANGE_M),
        "maximum_camera_height_uncertainty_m": core.MAXIMUM_CAMERA_HEIGHT_UNCERTAINTY_M,
        "scale_range": list(core.SCALE_RANGE),
    }
    mismatches = {
        key: {"protocol": operator.get(key), "implementation": value}
        for key, value in expected.items()
        if operator.get(key) != value
    }
    if mismatches:
        raise ValueError(f"protocol/implementation mismatch: {mismatches}")
    if operator.get("offset_allowed") is not False:
        raise ValueError("offset must remain forbidden")
    if operator.get("per_band_scale_allowed") is not False:
        raise ValueError("per-band scale must remain forbidden")
    if operator.get("temporal_smoothing_allowed") is not False:
        raise ValueError("temporal smoothing must remain forbidden")
    return {
        "status": "VALID",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "implementation_constants": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
