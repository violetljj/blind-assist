#!/usr/bin/env python3
"""Apply Panoramax's official viewer-equivalent zero-pose projection gate.

The existing strict gate requires explicit zero ``pers:yaw/pitch/roll`` fields.
Panoramax web-viewer 5.2.0 instead resolves each missing field through its EXIF
fallbacks and then defaults the unresolved value to zero.  For a 360-degree
image it emits no sphere correction unless both effective pitch and roll are
non-zero.  This module opens only the exact effective-zero subset of that
official behavior; it does not infer or compensate a non-zero physical pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from l10_panolab_entrance_ray import (
    EXPECTED_PROTOCOL_SCHEMA,
    parse_number,
    projection_gate as strict_projection_gate,
)


SUCCESSOR_SCHEMA = "blindassist-l10-panolab-viewer-equivalent-projection-protocol-v1"
_POSE_FAILURES = {
    "YAW_NOT_EXPLICIT_ZERO",
    "PITCH_NOT_EXPLICIT_ZERO",
    "ROLL_NOT_EXPLICIT_ZERO",
}
_POSE_FIELDS = {
    "pers:yaw": (
        "Xmp.GPano.PoseHeadingDegrees",
        "Xmp.Camera.Yaw",
        "Exif.MpfInfo.MPFYawAngle",
    ),
    "pers:pitch": (
        "Xmp.GPano.PosePitchDegrees",
        "Xmp.Camera.Pitch",
        "Exif.MpfInfo.MPFPitchAngle",
    ),
    "pers:roll": (
        "Xmp.GPano.PoseRollDegrees",
        "Xmp.Camera.Roll",
        "Exif.MpfInfo.MPFRollAngle",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _effective_pose(properties: dict[str, Any], field: str) -> tuple[float, str]:
    value = properties.get(field)
    if value is not None:
        return float(value), field
    exif = properties.get("exif") or {}
    for name in _POSE_FIELDS[field]:
        if name not in exif:
            continue
        try:
            return parse_number(exif[name]), name
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return 0.0, "OFFICIAL_VIEWER_UNRESOLVED_DEFAULT_ZERO"


def projection_gate(
    item: dict[str, Any],
    strict_protocol: dict[str, Any],
    successor_protocol: dict[str, Any],
    *,
    downloaded_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return the strict gate plus the official-viewer effective-zero successor."""

    require(
        strict_protocol.get("schema") == EXPECTED_PROTOCOL_SCHEMA,
        "unexpected strict projection protocol schema",
    )
    require(
        successor_protocol.get("schema") == SUCCESSOR_SCHEMA,
        "unexpected viewer-equivalent projection protocol schema",
    )
    strict = strict_projection_gate(
        item,
        strict_protocol,
        downloaded_image_size=downloaded_image_size,
    )
    properties = item.get("properties") or {}
    pose = {
        field: {"degrees": value, "source": source}
        for field in _POSE_FIELDS
        for value, source in [_effective_pose(properties, field)]
    }
    effective_zero = all(abs(float(row["degrees"])) <= 1e-9 for row in pose.values())
    non_pose_failures = [failure for failure in strict["failures"] if failure not in _POSE_FAILURES]
    eligible = strict["eligible"] or (effective_zero and not non_pose_failures)
    if strict["eligible"]:
        mode = "STRICT_EXPLICIT_ZERO"
    elif eligible:
        mode = "OFFICIAL_VIEWER_EFFECTIVE_ZERO"
    else:
        mode = "INELIGIBLE"
    return {
        **strict,
        "eligible": eligible,
        "failures": non_pose_failures if effective_zero else strict["failures"],
        "strict_eligible": strict["eligible"],
        "strict_failures": strict["failures"],
        "projection_mode": mode,
        "effective_pose": pose,
        "effective_pose_is_zero": effective_zero,
        "sphere_correction": {},
        "sphere_correction_reason": (
            "OFFICIAL_360_VIEWER_EFFECTIVE_ZERO_NO_CORRECTION"
            if eligible
            else "NOT_AUTHORIZED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--strict-protocol", type=Path, required=True)
    parser.add_argument("--successor-protocol", type=Path, required=True)
    args = parser.parse_args()
    item = json.loads(args.item.read_text(encoding="utf-8"))
    strict_protocol = json.loads(args.strict_protocol.read_text(encoding="utf-8"))
    successor_protocol = json.loads(args.successor_protocol.read_text(encoding="utf-8"))
    print(json.dumps(projection_gate(item, strict_protocol, successor_protocol), indent=2))


if __name__ == "__main__":
    main()
