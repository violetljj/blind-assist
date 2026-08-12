#!/usr/bin/env python3
"""Statically validate the frozen R2 F1 supervision source/label contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_"
    "CONTRACT_LOCK_2026-08-11.json"
)
SCHEMA = "blindassist_assistive_geometry_r2_f1_supervision_source_and_label_contract_v1"
STATUS = (
    "F1_SUPERVISION_CONTRACT_FROZEN_LABEL_MATERIALIZATION_AUTHORIZED_"
    "OPTIMIZER_NOT_AUTHORIZED"
)
TOKEN = "AG_R2_F1_SUPERVISION_R0_2026-08-11"
ROLE_COUNTS = {"FIT": 9, "CHECKPOINT_SELECTION": 2, "TRAIN_CANARY": 2}
ORIENTATIONS = {"LANDSCAPE_IDENTITY", "PORTRAIT_ROT90_CLOCKWISE"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def identity_digest(parent_id: str) -> str:
    return hashlib.sha256(f"{TOKEN}:{parent_id}".encode("utf-8")).hexdigest().upper()


def expected_role(order_zero_based: int) -> tuple[str, int]:
    if order_zero_based < 9:
        return "FIT", order_zero_based
    if order_zero_based < 11:
        return "CHECKPOINT_SELECTION", order_zero_based - 9
    return "TRAIN_CANARY", order_zero_based - 11


def validate_document(
    document: dict[str, Any],
    repo_root: Path = REPO_ROOT,
    *,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    gates: dict[str, bool] = {}

    gates["schema_and_status"] = (
        document.get("schema") == SCHEMA and document.get("status") == STATUS
    )
    require(gates["schema_and_status"], "contract schema/status drift")

    bindings = document.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == 5, "binding roster drift")
    binding_ids = [str(row.get("id")) for row in bindings]
    require(len(set(binding_ids)) == len(binding_ids), "duplicate binding id")
    bindings_exact = True
    if verify_bindings:
        for row in bindings:
            path = repo_root / str(row["path"])
            require(path.is_file(), f"binding missing: {path}")
            bindings_exact &= sha256_file(path) == str(row["sha256"])
    gates["bindings_exact"] = bindings_exact
    require(bindings_exact, "binding SHA drift")

    prelock = document.get("prelock_evidence", {})
    gates["prelock_identity_only"] = (
        prelock.get("source_payload_downloaded_or_opened") is False
        and prelock.get("model_or_task_outcome_read_for_roster_or_role_assignment") is False
        and prelock.get("exact_parent_identity_mentions_in_repository_before_lock") == 0
        and prelock.get("network_read_scope") == "OFFICIAL_DATASET_PAGE_AND_HTTP_HEAD_ONLY"
    )
    require(gates["prelock_identity_only"], "prelock evidence drift")

    source = document.get("source_contract", {})
    gates["source_format_exact"] = (
        source.get("license") == "CC BY 4.0"
        and source.get("native_resolution_hw") == [480, 640]
        and source.get("depth_encoding") == "uint16"
        and source.get("depth_scale_divisor") == 5000.0
        and source.get("maximum_rgb_depth_timestamp_delta_seconds") == 0.02
        and source.get("maximum_pose_bracketing_gap_seconds") == 0.1
        and source.get("accelerometer_window_seconds") == 0.03
        and source.get("imu_to_rgb_optical_rotation")
        == [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    require(gates["source_format_exact"], "source format/calibration drift")

    cohort = document.get("cohort_contract", {})
    parents = cohort.get("parents")
    require(isinstance(parents, list) and len(parents) == 13, "parent count drift")
    parent_ids = [str(row.get("parent_id")) for row in parents]
    require(len(set(parent_ids)) == 13, "duplicate parent identity")
    require(cohort.get("assignment_token") == TOKEN, "assignment token drift")
    require(cohort.get("minimum_joint_factor_parent_count") == 12, "joint minimum drift")
    require(cohort.get("parent_disjoint") is True, "parent disjointness disabled")

    expected_order = sorted(parent_ids, key=identity_digest)
    require(parent_ids == expected_order, "parent identity hash ordering drift")
    role_counts: Counter[str] = Counter()
    orientations_by_role: dict[str, set[str]] = {
        "FIT": set(),
        "CHECKPOINT_SELECTION": set(),
        "TRAIN_CANARY": set(),
    }
    total_bytes = 0
    for index, row in enumerate(parents):
        parent_id = str(row["parent_id"])
        role, role_index = expected_role(index)
        orientation = (
            "LANDSCAPE_IDENTITY"
            if role_index % 2 == 0
            else "PORTRAIT_ROT90_CLOCKWISE"
        )
        family = next(
            (value for value in ("freiburg1", "freiburg2", "freiburg3") if value in parent_id),
            None,
        )
        require(family is not None, f"parent family missing: {parent_id}")
        require(row.get("order") == index + 1, f"parent order drift: {parent_id}")
        require(row.get("assignment_sha256") == identity_digest(parent_id), f"assignment hash drift: {parent_id}")
        require(row.get("role") == role and row.get("role_index") == role_index, f"role drift: {parent_id}")
        require(row.get("orientation") == orientation, f"orientation drift: {parent_id}")
        require(row.get("family") == family, f"family drift: {parent_id}")
        require(
            row.get("source_url")
            == f"https://cvg.cit.tum.de/rgbd/dataset/{family}/{parent_id}.tgz",
            f"source URL drift: {parent_id}",
        )
        require(
            row.get("resolved_url")
            == f"https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{family}/{parent_id}.tgz",
            f"resolved URL drift: {parent_id}",
        )
        require(row.get("http_status") == 200, f"HEAD status drift: {parent_id}")
        require(int(row.get("content_length", 0)) > 0, f"content length invalid: {parent_id}")
        require(bool(row.get("last_modified")), f"last-modified missing: {parent_id}")
        role_counts[role] += 1
        orientations_by_role[role].add(orientation)
        total_bytes += int(row["content_length"])

    gates["role_roster_exact"] = dict(role_counts) == ROLE_COUNTS
    require(gates["role_roster_exact"], "role counts drift")
    require(cohort.get("role_parent_counts") == ROLE_COUNTS, "declared role counts drift")
    gates["held_orientations_complete"] = (
        orientations_by_role["CHECKPOINT_SELECTION"] == ORIENTATIONS
        and orientations_by_role["TRAIN_CANARY"] == ORIENTATIONS
    )
    require(gates["held_orientations_complete"], "held orientation coverage drift")
    gates["source_headers_complete"] = (
        total_bytes == 18_962_494_742
        and cohort.get("expected_download_bytes") == total_bytes
    )
    require(gates["source_headers_complete"], "download byte receipt drift")

    frames = document.get("frame_selection_contract", {})
    gates["frame_selection_frozen"] = (
        frames.get("frames_per_parent") == 3
        and frames.get("expected_frame_count") == 39
        and len(frames.get("selection_may_read", [])) == 4
        and "accelerometer.txt" not in frames.get("selection_may_read", [])
        and "depth pixel payload" in frames.get("selection_may_not_read", [])
        and "task outcome" in frames.get("selection_may_not_read", [])
    )
    require(gates["frame_selection_frozen"], "frame selection contract drift")

    orientation = document.get("orientation_contract", {})
    portrait = orientation.get("PORTRAIT_ROT90_CLOCKWISE", {})
    gates["orientation_geometry_frozen"] = (
        orientation.get("coordinate_space") == "DISPLAY_UPRIGHT_TENSOR_WITH_BOUND_K"
        and orientation.get("LANDSCAPE_IDENTITY", {}).get("output_shape_hw") == [480, 640]
        and portrait.get("output_shape_hw") == [640, 480]
        and portrait.get("camera_basis_new_from_old")
        == [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        and "fx_new=fy_source" in str(portrait.get("intrinsics_rule"))
        and len(orientation.get("required_invariants", [])) == 4
    )
    require(gates["orientation_geometry_frozen"], "orientation geometry drift")

    labels = document.get("continuous_label_contract", {})
    forbidden_inputs = set(labels.get("forbidden_derivation_inputs", []))
    gates["source_only_labels"] = (
        labels.get("admitted_inputs")
        == "Source-native depth, source K, source pose and source accelerometer only. No Teacher pixel is admitted to F1."
        and labels.get("depth_scale", {}).get("provenance") == "SENSOR_DERIVED_METRIC"
        and labels.get("support_surface", {}).get("provenance") == "SENSOR_DERIVED_SUPPORT"
        and labels.get("obstacle_boundary_evidence", {}).get("provenance")
        == "SENSOR_DERIVED_BOUNDARY"
        and {"teacher depth", "teacher confidence", "teacher consensus"}.issubset(forbidden_inputs)
        and {"clearance", "occupancy", "final task state"}.issubset(forbidden_inputs)
    )
    require(gates["source_only_labels"], "source-only label firewall drift")

    provenance = document.get("provenance_contract", {})
    allowed_provenance = set(provenance.get("allowed_by_field", {}).values())
    gates["provenance_closed"] = (
        provenance.get("teacher_provenance_admitted") is False
        and provenance.get("factor_specific_receipts_required") is True
        and allowed_provenance
        == {"SENSOR_DERIVED_METRIC", "SENSOR_DERIVED_SUPPORT", "SENSOR_DERIVED_BOUNDARY"}
        and len(provenance.get("required_receipt_fields", [])) == 14
    )
    require(gates["provenance_closed"], "provenance contract drift")

    uncertainty = document.get("uncertainty_contract", {})
    gates["uncertainty_residual_only"] = (
        uncertainty.get("direct_sigma_truth_required") is False
        and uncertainty.get("direct_uncertainty_proxy_regression_forbidden") is True
        and uncertainty.get("ag_st_r22_proxy_usage")
        == "DIAGNOSTIC_ONLY_NOT_TARGET_NOT_WEIGHT_NOT_SELECTOR"
        and uncertainty.get("fit_and_evaluation_parent_disjoint") is True
        and uncertainty.get("zero_or_constant_sigma_pseudo_truth_forbidden") is True
        and uncertainty.get("task_state_as_uncertainty_truth_forbidden") is True
        and "FIT residuals only" in str(uncertainty.get("homoscedastic_baseline"))
    )
    require(gates["uncertainty_residual_only"], "uncertainty contract drift")

    gate_ids = [str(row.get("id")) for row in document.get("frontdoor_gates", [])]
    gates["frontdoor_gate_set_exact"] = gate_ids == [f"F1_S{index:02d}_{suffix}" for index, suffix in enumerate(
        (
            "BINDINGS_EXACT",
            "SOURCE_ARCHIVES_EXACT",
            "ROLE_AND_ORIENTATION_EXACT",
            "FRAME_SELECTION_EXACT",
            "SCHEMA_AND_PROVENANCE_COMPLETE",
            "JOINT_PARENT_COVERAGE",
            "UNKNOWN_FAIL_CLOSED",
            "UNCERTAINTY_RESIDUAL_ONLY",
            "TASK_FIREWALL",
        ),
        start=1,
    )]
    require(gates["frontdoor_gate_set_exact"], "frontdoor gate roster drift")

    authority = document.get("execution_authority_after_this_lock", {})
    gates["authority_bounded"] = (
        authority.get("source_download") is True
        and authority.get("formal_label_materialization") is True
        and authority.get("frontdoor_validation") is True
        and authority.get("model_definition") is False
        and authority.get("trainer_creation") is False
        and authority.get("optimizer_step") is False
        and authority.get("f1_execution") is False
        and authority.get("deterministic_reducer_task_evaluation") is False
    )
    require(gates["authority_bounded"], "execution authority drift")
    successor = document.get("unique_successor", {})
    gates["successor_unique"] = successor.get("id") == (
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_"
        "MATERIALIZATION_AND_FRONTDOOR_RESULT"
    )
    require(gates["successor_unique"], "unique successor drift")

    return {
        "schema": "blindassist_assistive_geometry_r2_f1_supervision_contract_validation_v1",
        "status": "F1_SUPERVISION_CONTRACT_STATIC_VALIDATION_PASS",
        "passed": all(gates.values()),
        "gate_count": len(gates),
        "gates": gates,
        "parent_count": len(parents),
        "role_parent_counts": dict(role_counts),
        "orientation_by_role": {
            role: sorted(values) for role, values in orientations_by_role.items()
        },
        "expected_download_bytes": total_bytes,
        "next_successor": successor["id"],
    }


def validate_path(path: Path, *, verify_bindings: bool = True) -> dict[str, Any]:
    require(path.is_file(), f"contract missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), "contract root must be an object")
    return validate_document(document, verify_bindings=verify_bindings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = validate_path(args.contract.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "F1_SUPERVISION_CONTRACT_STATIC_VALIDATION_FAIL", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
