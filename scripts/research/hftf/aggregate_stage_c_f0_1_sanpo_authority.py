#!/usr/bin/env python3
"""Aggregate the exact F0.1 SANPO source-authority cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LOCK_SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_source_lock"
LOCK_TERMINAL = "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"
ACQUISITION_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_sanpo_acquisition_audit"
)
ACQUISITION_TERMINAL = "F0_1_SANPO_ACQUISITION_AND_TRANSPORT_READY"
AUTHORITY_SCHEMA = "blindassist_hftf_sanpo_pose_geometry_authority_r0"
AUTHORITY_TERMINAL = "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED"
SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_authority_cohort"
READY = "F0_1_SANPO_SOURCE_AUTHORITY_COHORT_READY"
NOT_EVALUABLE = "F0_1_SANPO_SOURCE_AUTHORITY_COHORT_NOT_EVALUABLE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_authority(
    source: dict[str, Any],
    acquisition: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "role": source["role"],
            "official_split": source["official_split"],
            "session_id": source["session_id"],
            "authority_report_path": str(report_path.resolve()),
            "ok": False,
            "errors": ["authority_report_missing"],
        }
    report = _load_json(report_path)
    source_id = str(source["session_id"])
    input_hashes = report.get("input_hashes", {})
    official = report.get("official_loader_authority", {})
    pose = report.get("source_pose_authority", {})
    transform = report.get("transform_direction_canary", {})
    ground = report.get("ground_and_body_proxy_canary", {})
    capabilities = report.get("capability_decisions", {})
    if (
        report.get("schema") != AUTHORITY_SCHEMA
        or report.get("terminal") != AUTHORITY_TERMINAL
        or report.get("evaluation_mode") != "frozen_canonical_replication"
        or report.get("claim_ceiling")
        != "SOURCE_SPECIFIC_GEOMETRY_PROXY_ONLY"
        or report.get("source_session_ids") != [source_id]
        or report.get("manifest_frame_count") != 25
        or report.get("mainline_changed") is not False
        or report.get("default_app_changed") is not False
        or report.get("allowed_next_step") != "H0_2_COHORT_AGGREGATION"
    ):
        errors.append("top_level_authority_contract_mismatch")
    expected_hashes = {
        "dataset_spec_sha256": acquisition["dataset_spec_sha256"],
        "manifest_sha256": acquisition["manifest_sha256"],
        "camera_poses_sha256": acquisition["camera_poses_sha256"],
    }
    if not isinstance(input_hashes, dict) or any(
        input_hashes.get(key) != value
        for key, value in expected_hashes.items()
    ):
        errors.append("acquisition_hash_binding_mismatch")
    if (
        not isinstance(official, dict)
        or official.get("ok") is not True
        or official.get("expected_markers_present") is not True
        or official.get("clean_tracked_tree") is not True
    ):
        errors.append("official_loader_authority_failed")
    bindings = pose.get("bindings", []) if isinstance(pose, dict) else []
    if (
        not isinstance(pose, dict)
        or pose.get("ok") is not True
        or pose.get("gcs_description_authenticated") is not True
        or pose.get("gcs_camera_poses_authenticated") is not True
        or pose.get("binding_count") != 25
        or len(bindings) != 25
        or [item.get("source_frame_index") for item in bindings]
        != source["selected_source_frames"]
        or any(
            item.get("ok") is not True
            or item.get("tracking_state") != "TrackingState.READY"
            or item.get("raw_pose_row_index") != item.get("source_frame_index")
            for item in bindings
        )
    ):
        errors.append("source_pose_authority_failed")
    canonical = (
        transform.get("frozen_canonical_hypothesis", {})
        if isinstance(transform, dict)
        else {}
    )
    if (
        not isinstance(transform, dict)
        or transform.get("ok") is not True
        or transform.get("frozen_canonical_replication_admitted") is not True
        or transform.get("frozen_canonical_rank") != 1
        or transform.get("admitted_semantics")
        != "p_world = R_xyzw @ p_opencv_camera + camera_translation_m"
        or canonical.get("orientation_hypothesis") != "R"
        or canonical.get("camera_basis_rows")
        != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ):
        errors.append("frozen_transform_authority_failed")
    if (
        not isinstance(ground, dict)
        or ground.get("ok") is not True
        or ground.get("frame_count_with_ground") != 25
        or ground.get("local_ground_plane_frame_count") != 25
        or ground.get("vertical_axis") != "+Z"
        or ground.get("standard_body_proxy_frame_admitted_for_h1")
        is not True
        or ground.get("physical_camera_to_body_calibration_admitted")
        is not False
    ):
        errors.append("ground_proxy_authority_failed")
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("standard_body_proxy_for_h1_geometry_mechanics")
        != "ELIGIBLE"
        or capabilities.get("physical_camera_to_person_calibration")
        != "NOT_EVALUABLE"
        or capabilities.get("student_or_event_effect") != "NOT_EVALUABLE"
    ):
        errors.append("capability_ceiling_mismatch")
    best = transform.get("best", {}) if isinstance(transform, dict) else {}
    chosen = ground.get("chosen_axis", {}) if isinstance(ground, dict) else {}
    return {
        "role": source["role"],
        "official_split": source["official_split"],
        "session_id": source_id,
        "authority_report_path": str(report_path.resolve()),
        "authority_report_sha256": _sha256(report_path),
        "manifest_sha256": input_hashes.get("manifest_sha256"),
        "dataset_spec_sha256": input_hashes.get("dataset_spec_sha256"),
        "camera_poses_sha256": input_hashes.get("camera_poses_sha256"),
        "pose_binding_count": pose.get("binding_count"),
        "transform_coverage": best.get("coverage"),
        "transform_median_relative_depth_error": best.get(
            "median_relative_depth_error"
        ),
        "local_ground_plane_frame_count": ground.get(
            "local_ground_plane_frame_count"
        ),
        "median_ground_axis_alignment": chosen.get(
            "median_axis_alignment"
        ),
        "ok": not errors,
        "errors": errors,
    }


def aggregate(
    source_lock_path: Path,
    acquisition_path: Path,
    authority_root: Path,
) -> dict[str, Any]:
    source_lock = _load_json(source_lock_path)
    acquisition = _load_json(acquisition_path)
    if (
        source_lock.get("schema") != LOCK_SCHEMA
        or source_lock.get("terminal") != LOCK_TERMINAL
        or source_lock.get("teacher_label_or_corpus_authorized") is not False
        or source_lock.get("student_training_authorized") is not False
    ):
        raise ValueError("F0.1 source lock contract mismatch")
    if (
        acquisition.get("schema") != ACQUISITION_SCHEMA
        or acquisition.get("terminal") != ACQUISITION_TERMINAL
        or acquisition.get("all_sources_ok") is not True
        or acquisition.get("source_lock_sha256") != _sha256(source_lock_path)
        or acquisition.get("source_count") != 12
        or acquisition.get("frame_count") != 300
        or acquisition.get("authorization", {}).get(
            "teacher_geometry_outcome_authorized"
        )
        is not False
    ):
        raise ValueError("F0.1 acquisition audit contract mismatch")
    sources = source_lock.get("sources", [])
    acquisition_sources = acquisition.get("sources", [])
    if (
        len(sources) != 12
        or len(acquisition_sources) != 12
        or [item.get("session_id") for item in sources]
        != [item.get("session_id") for item in acquisition_sources]
    ):
        raise ValueError("F0.1 source/acquisition order mismatch")
    results = []
    for source, acquired in zip(sources, acquisition_sources, strict=True):
        report_path = (
            authority_root
            / str(source["session_id"])[:8]
            / "authority.json"
        )
        results.append(_validate_authority(source, acquired, report_path))
    ready = all(result["ok"] for result in results)
    return {
        "schema": SCHEMA,
        "terminal": READY if ready else NOT_EVALUABLE,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SOURCE_SPECIFIC_GEOMETRY_PROXY_ONLY",
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": _sha256(source_lock_path),
        "acquisition_audit_path": str(acquisition_path.resolve()),
        "acquisition_audit_sha256": _sha256(acquisition_path),
        "authority_root": str(authority_root.resolve()),
        "source_count": len(results),
        "role_counts": {
            role: sum(item["role"] == role for item in results)
            for role in ("train", "dev", "heldout")
        },
        "all_sources_authority_ready": ready,
        "sources": results,
        "allowed_next_step": (
            "F0_1_TEACHER_OPPORTUNITY_AUDIT" if ready else "NONE"
        ),
        "authorization": {
            "teacher_geometry_opportunity_audit_authorized": ready,
            "teacher_label_or_corpus_materialization_authorized": False,
            "student_training_authorized": False,
            "heldout_used_for_development_selection": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
        "prohibited_inferences": [
            "source proxy is physical participant calibration",
            "geometry proxy is human collision or safety truth",
            "teacher opportunity proves student effect",
            "research mainline promotion",
            "Android or production authorization",
        ],
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--acquisition-audit", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = aggregate(
            args.source_lock.resolve(),
            args.acquisition_audit.resolve(),
            args.authority_root.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "source_count": report["source_count"],
                    "allowed_next_step": report["allowed_next_step"],
                    "output": str(output),
                }
            )
        )
        return 0 if report["all_sources_authority_ready"] else 1
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
