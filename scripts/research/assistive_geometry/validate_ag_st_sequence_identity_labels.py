#!/usr/bin/env python3
"""Validate sequence-identity multi-Teacher factor NPZ invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-identity-r1/result.json"
)
DEFAULT_OUTPUT = DEFAULT_RESULT.with_name("validation.json")


def validate(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(
        result.get("status") == "SEQUENCE_IDENTITY_MULTITEACHER_LABELS_MATERIALIZED",
        "sequence-identity result invalid",
    )
    receipts = result.get("frame_receipts")
    require(isinstance(receipts, list) and len(receipts) == 48, "sequence-identity receipt drift")
    failures: list[str] = []
    hashes: list[dict[str, Any]] = []
    eligible = 0
    unknown = 0
    total_bytes = 0
    for receipt in receipts:
        frame = str(receipt["frame_stem"])
        path = Path(str(receipt["output_path"]))
        require(path.is_file(), f"sequence-identity label missing: {path}")
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as arrays:
            identity_valid = bool(arrays["support_identity_valid"])
            plane_valid = bool(arrays["support_plane_valid"])
            support_valid = arrays["support_truth_valid_hw"].astype(np.bool_)
            evidence_valid = arrays["evidence_truth_valid_hw"].astype(np.bool_)
            normal_valid = arrays["normal_valid_hw"].astype(np.bool_)
            metric_valid = arrays["metric_depth_valid_hw"].astype(np.bool_)
            if receipt["support_identity_valid"]:
                eligible += 1
                if not identity_valid or not plane_valid:
                    failures.append(f"{frame}: eligible identity/plane invalid")
                camera_height = float(arrays["camera_height_m"])
                expected = (
                    float(arrays["camera_to_world_output"][2, 3])
                    - float(arrays["support_identity_world_height_m"])
                )
                if not (0.45 <= camera_height <= 2.20 and abs(camera_height - expected) <= 1e-5):
                    failures.append(f"{frame}: camera-height mismatch")
                rotation = arrays["camera_to_world_output"][:3, :3].astype(np.float64)
                gravity_camera = rotation.T @ np.asarray([0.0, 0.0, 1.0])
                gravity_camera /= np.linalg.norm(gravity_camera)
                normal = arrays["support_plane_normal_camera_xyz"].astype(np.float64)
                if float(np.dot(gravity_camera, normal)) < 0.999:
                    failures.append(f"{frame}: anti-gravity support normal")
                if np.any(support_valid & ~metric_valid) or np.any(evidence_valid & ~metric_valid):
                    failures.append(f"{frame}: derived validity escapes metric validity")
            else:
                unknown += 1
                if identity_valid or plane_valid:
                    failures.append(f"{frame}: UNKNOWN identity materialized")
                if np.any(support_valid) or np.any(evidence_valid) or np.any(normal_valid):
                    failures.append(f"{frame}: UNKNOWN factor pixels materialized")
        hashes.append(
            {
                "frame_stem": frame,
                "output_bytes": path.stat().st_size,
                "output_sha256": sha256_file(path),
            }
        )
    require(eligible == 31 and unknown == 17, "eligible/UNKNOWN count drift")
    passed = not failures
    return {
        "schema": "blindassist_ag_st_sequence_identity_factor_validation_v1",
        "status": (
            "SEQUENCE_IDENTITY_MULTITEACHER_LABEL_INVARIANTS_PASS"
            if passed
            else "SEQUENCE_IDENTITY_MULTITEACHER_LABEL_INVARIANTS_FAIL"
        ),
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "frame_count": len(receipts),
        "eligible_frame_count": eligible,
        "unknown_frame_count": unknown,
        "output_total_bytes": total_bytes,
        "invariant_failure_count": len(failures),
        "invariant_failures": failures,
        "output_receipts": hashes,
        "claim_boundary": "Mechanical factor-label validation only; not semantic or task truth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"sequence-identity validation exists: {args.output}")
    result = validate(args.result)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "frame_count", "eligible_frame_count", "unknown_frame_count", "invariant_failure_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
