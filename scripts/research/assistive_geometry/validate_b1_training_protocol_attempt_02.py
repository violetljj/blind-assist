#!/usr/bin/env python3
"""Validate the dual-orientation B1 protocol overlay and its frozen base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.validate_b1_training_protocol import (  # noqa: E402
    sha256_file,
    validate as validate_base,
)


SCHEMA = "blindassist_assistive_geometry_b1_training_protocol_dual_orientation_overlay_v1"
EXPECTED_SHAPES = {
    "portrait": {"input_nchw": [1, 3, 608, 448], "target_hw": [608, 448], "orientation_indices": [1, 3]},
    "landscape": {"input_nchw": [1, 3, 448, 608], "target_hw": [448, 608], "orientation_indices": [0, 2]},
}


def validate(overlay: dict[str, Any], base: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(overlay.get("schema") == SCHEMA, "overlay schema drift")
    check(overlay.get("status") == "B1_PROTOCOL_ATTEMPT_02_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED", "overlay status drift")
    check(validate_base(base) == [], "base protocol is no longer valid")
    check(overlay.get("inherit_all_unmodified_base_fields") is True, "base inheritance drift")
    corrections = overlay.get("corrections", {})
    check(corrections.get("full_fov_tensor_shapes") == EXPECTED_SHAPES, "dual-orientation tensor shape drift")
    check(corrections.get("orientation_bucketed_batches") is True, "orientation buckets required")
    check(corrections.get("mixed_orientation_batch") == "FORBIDDEN", "mixed orientation batch must be forbidden")
    check(corrections.get("crop_pad_or_rotate_between_orientation_families") == "FORBIDDEN", "cross-orientation transform must be forbidden")
    check(corrections.get("k_transform") == "upright rotation then independent sx_sy full-FOV resize", "K transform drift")
    check(corrections.get("train_frame_count") == 4800, "TRAIN frame count drift")
    check(corrections.get("portrait_train_frame_count") == 2724, "portrait TRAIN count drift")
    check(corrections.get("landscape_train_frame_count") == 2076, "landscape TRAIN count drift")

    roles = corrections.get("development_roles", {})
    expected_calibration = [
        {"visit_id": "383240", "video_id": "41127065"},
        {"visit_id": "421012", "video_id": "42444793"},
        {"visit_id": "421260", "video_id": "42444891"},
        {"visit_id": "469828", "video_id": "47430934"},
    ]
    expected_selection = [
        {"visit_id": "426274", "video_id": "42898438"},
        {"visit_id": "435658", "video_id": "42899869"},
        {"visit_id": "464241", "video_id": "44358604"},
        {"visit_id": "472034", "video_id": "47332413"},
    ]
    check(roles.get("DEVELOPMENT_CALIBRATION") == expected_calibration, "orientation-balanced calibration identity drift")
    check(roles.get("DEVELOPMENT_SELECTION") == expected_selection, "orientation-balanced selection identity drift")
    identities = [(row["visit_id"], row["video_id"]) for rows in roles.values() if isinstance(rows, list) for row in rows]
    check(len(identities) == 8 and len(set(identities)) == 8, "Development overlay roles overlap")
    check(corrections.get("development_split_frozen_before_outcome") is True, "Development split must predate outcome")
    check(corrections.get("portrait_calibration_claim_ceiling") == "DEVELOPMENT_ONLY_SINGLE_PORTRAIT_DOMINANT_PARENT", "portrait calibration ceiling drift")
    check(corrections.get("orientation_reporting") == ["pooled", "portrait", "landscape", "parent_macro"], "orientation reporting drift")
    check(corrections.get("product_decision_uses") == "portrait stratum only", "product decision stratum drift")

    authority = overlay.get("authority", {})
    check(authority.get("train_target_materialization") is True, "TRAIN target materialization must remain authorized")
    check(authority.get("model_dual_shape_smoke") is True, "dual-shape model smoke must be authorized")
    check(authority.get("formal_student_training") is False, "formal training must remain closed")
    check(authority.get("development_outcome_access") is False, "Development outcome must remain closed")
    check(authority.get("confirmation_payload_or_outcome_access") is False, "Confirmation must remain sealed")
    check(overlay.get("next_successor") == "BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_TARGET_AND_MODEL_IMPLEMENTATION_LOCK", "successor drift")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    path = args.protocol.resolve()
    overlay = json.loads(path.read_text(encoding="utf-8"))
    base_binding = overlay["base_protocol"]
    base_path = (root / base_binding["path"]).resolve()
    errors: list[str] = []
    if not base_path.is_file() or sha256_file(base_path) != base_binding["sha256"]:
        errors.append("base protocol binding drift")
        base: dict[str, Any] = {}
    else:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    if base:
        errors.extend(validate(overlay, base))
    for binding in overlay.get("bindings", {}).values():
        bound = (root / binding["path"]).resolve()
        if not bound.is_file() or sha256_file(bound) != binding["sha256"]:
            errors.append(f"binding drift: {binding['path']}")
    if errors:
        print(json.dumps({"terminal": "B1_ATTEMPT_02_PROTOCOL_INVALID", "errors": errors}, indent=2))
        return 2
    print(json.dumps({
        "terminal": "B1_DUAL_ORIENTATION_PROTOCOL_LOCK_PASS",
        "protocol_sha256": sha256_file(path),
        "formal_student_training_authorized": False,
        "next_successor": overlay["next_successor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
