#!/usr/bin/env python3
"""Profile metadata-only Assistive Geometry data capability from the master ledger."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_assistive_geometry_b0_data_capability_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _count(session: dict[str, Any], modality: str) -> int:
    value = session.get("counts", {}).get(f"{modality}_count")
    return int(value or 0)


def structurally_eligible_rgb_depth(session: dict[str, Any]) -> bool:
    return bool(
        _count(session, "rgb") > 0
        and _count(session, "depth") > 0
        and session.get("session_kind") == "media_session"
        and session.get("decodability", {}).get("status") == "all_profiled_readable"
        and not session.get("corrupt_frames")
        and not session.get("hash_errors")
        and session.get("rgb_mask_depth_pose_alignment", {}).get("status")
        == "aligned_by_frame_key"
    )


def audit(ledger: dict[str, Any], ledger_path: Path) -> dict[str, Any]:
    if ledger.get("schema_version") != "dataset-master-ledger-v1":
        raise ValueError("unexpected dataset master ledger schema")
    sessions = ledger.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("ledger sessions must be a list")
    rgb_depth = [
        session
        for session in sessions
        if _count(session, "rgb") > 0 and _count(session, "depth") > 0
    ]
    structural = [session for session in rgb_depth if structurally_eligible_rgb_depth(session)]
    datasets = sorted({session.get("dataset", "Unknown") for session in rgb_depth})
    dataset_profile: dict[str, Any] = {}
    for dataset in datasets:
        rows = [session for session in rgb_depth if session.get("dataset", "Unknown") == dataset]
        eligible = [session for session in rows if structurally_eligible_rgb_depth(session)]
        split_counts = collections.Counter(str(session.get("split", "unspecified")) for session in rows)
        dataset_profile[dataset] = {
            "rgb_depth_sessions": len(rows),
            "structurally_eligible_sessions": len(eligible),
            "structurally_eligible_rate": round(len(eligible) / len(rows), 6),
            "with_pose_sessions": sum(_count(session, "pose") > 0 for session in rows),
            "all_profiled_readable_sessions": sum(
                session.get("decodability", {}).get("status") == "all_profiled_readable"
                for session in rows
            ),
            "aligned_by_frame_key_sessions": sum(
                session.get("rgb_mask_depth_pose_alignment", {}).get("status")
                == "aligned_by_frame_key"
                for session in rows
            ),
            "split_counts": dict(sorted(split_counts.items())),
            "eligible_source_ids": sorted(str(session["source_id"]) for session in eligible),
        }
    missing_role_evidence = sum(
        all(session.get("role_flags", {}).get(key) is None for key in ("is_consumed", "is_burned", "is_fresh", "is_reserved"))
        for session in structural
    )
    return {
        "schema": SCHEMA,
        "status": "STRUCTURAL_CANDIDATES_FOUND_ROSTER_ADMISSION_NOT_AUTHORIZED",
        "ledger": {
            "path": str(ledger_path).replace("\\", "/"),
            "bytes": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path),
            "generated_at": ledger.get("generated_at"),
            "session_count": len(sessions),
        },
        "intended_grain": "source/session package; not frame truth and not research-role authority",
        "profile": {
            "rgb_depth_sessions": len(rgb_depth),
            "structurally_eligible_rgb_depth_sessions": len(structural),
            "structurally_eligible_rate": round(len(structural) / len(rgb_depth), 6) if rgb_depth else 0.0,
            "structural_candidates_with_all_role_flags_unknown": missing_role_evidence,
            "dataset_profile": dataset_profile,
        },
        "structural_eligibility_rule": {
            "requires": [
                "rgb_count > 0",
                "depth_count > 0",
                "session_kind == media_session",
                "decodability == all_profiled_readable",
                "no corrupt_frames",
                "no hash_errors",
                "rgb/depth aligned_by_frame_key",
            ],
            "does_not_prove": [
                "metric depth unit/scale",
                "camera intrinsics availability or correctness",
                "RGB-depth registration beyond filename/frame-key evidence",
                "ground/clearance truth support",
                "license scope for Assistive Geometry B0",
                "identity/ancestry or near-duplicate independence",
                "TRAIN/DEVELOPMENT/CONFIRMATION role eligibility",
            ],
        },
        "quality_findings": [
            {
                "severity": "critical",
                "finding": "ledger has no route-specific license, metric-unit, K, ground-reader, or clearance-truth admission fields",
                "impact": "no structural candidate can be promoted directly into a B0 roster",
            },
            {
                "severity": "high",
                "finding": "role flags are path/metadata evidence and UNKNOWN is not absence",
                "impact": "fresh/train/test path tokens cannot assign B0 roles",
            },
            {
                "severity": "high",
                "finding": "frame-key alignment is weaker than calibrated RGB-depth reprojection",
                "impact": "task truth remains UNKNOWN until source-specific K/registration readers pass",
            },
        ],
        "outcome_or_payload_opened": False,
        "authority": "Metadata structural capability only; no roster admission, training, confirmation, deployment, product, production, or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.ledger.open("r", encoding="utf-8-sig") as handle:
        ledger = json.load(handle)
    result = audit(ledger, args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
