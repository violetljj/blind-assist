#!/usr/bin/env python3
"""Materialize the frozen G0-D1 current-only development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_f0_1_teacher_opportunity import (  # noqa: E402
    _pixel_lattices_disjoint,
    _probe_passes,
    _root_name,
)
from audit_swept_envelope_label_mechanics import (  # noqa: E402
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (  # noqa: E402
    _anchor_basis,
    _obstacle_points_world,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
    _theta_edges,
)
from run_stage_c_g0_signed_clearance_mechanics import (  # noqa: E402
    _signed_clearance_field,
)
from verify_sanpo_pose_geometry_authority import (  # noqa: E402
    _load_json,
    _load_jsonl,
)


DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_learnability_d1"
)
DESIGN_STATUS = (
    "FROZEN_SCIENTIFIC_DESIGN_AFTER_G0_D0_BEFORE_D1_"
    "IMPLEMENTATION_CORPUS_OR_STUDENT_OUTCOME"
)
EXECUTION_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "learnability_execution_contract_d1"
)
EXECUTION_CONTRACT_STATUS = (
    "FROZEN_BEFORE_D1_DEVELOPMENT_CORPUS_OR_STUDENT_OUTCOME"
)
TIMELINE_AMENDMENT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "learnability_d1_timeline_amendment"
)
TIMELINE_AMENDMENT_STATUS = EXECUTION_CONTRACT_STATUS
DATASET_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_development_corpus"
)
TEACHER_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_teacher_receipt"
)
READY = "G0_D1_CURRENT_CLEARANCE_DEVELOPMENT_CORPUS_READY"
NOT_EVALUABLE = (
    "G0_D1_CURRENT_CLEARANCE_DEVELOPMENT_CORPUS_NOT_EVALUABLE"
)
SOURCE_PLAN_SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
)
SOURCE_PLAN_READY = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_F0_1_STOP_BEFORE_G0_CLEARANCE_OR_SOURCE_SCAN_OUTCOME"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_mechanics_result_g0_d0"
)
RESULT_TERMINAL = (
    "G0_SIGNED_CLEARANCE_SOURCE_AND_MECHANICS_TERMINAL_VALIDATED"
)
MECHANICS_TERMINAL = (
    "G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY"
)
TRAINING_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_student_training_execution_contract_f0_1"
)
TRAINING_CONTRACT_STATUS = (
    "FROZEN_BEFORE_FIRST_F0_1_STUDENT_OPTIMIZATION_STEP"
)
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/"
    "materialize_stage_c_g0_d1_development_corpus.py"
)
HEIGHTS = ("body", "head")
EXPECTED_ROLE_COUNTS = {
    "development_reuse": 9,
    "one_shot_fresh_evaluation": 3,
    "reserved_fresh_heldout": 3,
}
STUDENT_TOP_LEVEL_KEYS = {
    "sample_id",
    "session_id",
    "role",
    "source_frame_index",
    "manifest_id",
    "current_rgb",
    "labels",
}
FORBIDDEN_STUDENT_FRAGMENTS = (
    "depth",
    "mask",
    "semantic",
    "pose",
    "teacher",
    "future",
    "truth",
    "clearance_raw",
    "support_count",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _resolve_parent(design_path: Path, receipt: dict[str, Any]) -> Path:
    raw = Path(str(receipt["path"]))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (_repository_root() / raw).resolve()
    return (design_path.parent / raw).resolve()


def _load_bound_parent(
    owner_path: Path,
    owner: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = owner.get("parents", {}).get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"Missing frozen parent receipt: {key}")
    path = _resolve_parent(owner_path, receipt)
    if not path.is_file() or _sha256(path) != str(receipt.get("sha256")):
        raise ValueError(f"Frozen parent hash mismatch: {key}")
    return path, _load_json(path)


def _implementation_receipt(contract: dict[str, Any]) -> None:
    receipt = contract.get("implementations", {}).get(
        "development_corpus_materializer"
    )
    if not isinstance(receipt, dict):
        raise ValueError(
            "D1 materializer implementation receipt is not frozen"
        )
    if (
        Path(str(receipt.get("path", ""))).as_posix()
        != IMPLEMENTATION_PATH
        or receipt.get("sha256") != _sha256(Path(__file__).resolve())
        or receipt.get("execution_authorized") is not True
    ):
        raise ValueError(
            "D1 materializer implementation receipt mismatch or unauthorized"
        )


def _nullable_labels(
    known: np.ndarray,
    support: np.ndarray,
    clipped_clearance: np.ndarray,
) -> dict[str, list[Any]]:
    shape = (6, 6, 2)
    if (
        known.shape != shape
        or support.shape != shape
        or clipped_clearance.shape != shape
        or known.dtype != np.bool_
        or not np.issubdtype(support.dtype, np.integer)
        or np.any(support < 0)
    ):
        raise ValueError("Frozen current target arrays must be 6x6x2")
    target_risk = support >= 2
    if np.any(~np.isfinite(clipped_clearance[known])):
        raise ValueError("Known clipped-clearance targets must be finite")
    if np.any(target_risk[known] != (clipped_clearance[known] < 0.0)):
        raise ValueError(
            "Known risk and signed-clearance target semantics disagree"
        )
    ordered_known = known.transpose(2, 0, 1)
    ordered_risk = target_risk.transpose(2, 0, 1)
    ordered_clearance = clipped_clearance.transpose(2, 0, 1)
    risk: list[list[list[int | None]]] = []
    clearance: list[list[list[float | None]]] = []
    for height_index in range(2):
        risk_rows: list[list[int | None]] = []
        clearance_rows: list[list[float | None]] = []
        for theta_index in range(6):
            risk_row: list[int | None] = []
            clearance_row: list[float | None] = []
            for distance_index in range(6):
                if ordered_known[
                    height_index, theta_index, distance_index
                ]:
                    risk_row.append(
                        int(
                            ordered_risk[
                                height_index,
                                theta_index,
                                distance_index,
                            ]
                        )
                    )
                    clearance_row.append(
                        float(
                            ordered_clearance[
                                height_index,
                                theta_index,
                                distance_index,
                            ]
                        )
                    )
                else:
                    risk_row.append(None)
                    clearance_row.append(None)
            risk_rows.append(risk_row)
            clearance_rows.append(clearance_row)
        risk.append(risk_rows)
        clearance.append(clearance_rows)
    return {
        "known_target": ordered_known.astype(np.uint8).tolist(),
        "risk_target_nullable": risk,
        "clearance_target_m_nullable": clearance,
    }


def _array_shape(value: Any) -> tuple[int, ...]:
    try:
        return np.asarray(value, dtype=object).shape
    except ValueError:
        return ()


def _student_record_firewall(record: dict[str, Any]) -> bool:
    if set(record) != STUDENT_TOP_LEVEL_KEYS:
        return False
    if (
        not isinstance(record.get("sample_id"), str)
        or not isinstance(record.get("session_id"), str)
        or record.get("role") not in ("train", "model_selection")
        or not isinstance(record.get("source_frame_index"), int)
        or not isinstance(record.get("manifest_id"), str)
        or set(record.get("current_rgb", {})) != {"path", "sha256"}
        or not isinstance(record["current_rgb"].get("path"), str)
        or not isinstance(record["current_rgb"].get("sha256"), str)
        or set(record.get("labels", {}))
        != {
            "known_target",
            "risk_target_nullable",
            "clearance_target_m_nullable",
        }
    ):
        return False
    labels = record["labels"]
    if any(
        _array_shape(labels[key]) != (2, 6, 6)
        for key in labels
    ):
        return False
    known = np.asarray(labels["known_target"], dtype=object)
    risk = np.asarray(labels["risk_target_nullable"], dtype=object)
    clearance = np.asarray(
        labels["clearance_target_m_nullable"], dtype=object
    )
    for index in np.ndindex((2, 6, 6)):
        is_known = known[index]
        if is_known not in (0, 1) or isinstance(is_known, bool):
            return False
        if is_known == 0:
            if risk[index] is not None or clearance[index] is not None:
                return False
        else:
            if (
                risk[index] not in (0, 1)
                or isinstance(risk[index], bool)
                or isinstance(clearance[index], bool)
                or not isinstance(
                    clearance[index], (int, float, np.integer, np.floating)
                )
                or not np.isfinite(float(clearance[index]))
                or (int(risk[index]) == 1)
                != (float(clearance[index]) < 0.0)
            ):
                return False

    def strings(value: Any) -> tuple[list[str], list[str]]:
        if isinstance(value, dict):
            keys: list[str] = []
            values: list[str] = []
            for key, item in value.items():
                keys.append(str(key).lower())
                nested_keys, nested_values = strings(item)
                keys.extend(nested_keys)
                values.extend(nested_values)
            return keys, values
        if isinstance(value, list):
            keys: list[str] = []
            values: list[str] = []
            for item in value:
                nested_keys, nested_values = strings(item)
                keys.extend(nested_keys)
                values.extend(nested_values)
            return keys, values
        return [], [value.lower()] if isinstance(value, str) else []

    keys, values = strings(record)
    if any(
        fragment in key
        for key in keys
        for fragment in FORBIDDEN_STUDENT_FRAGMENTS
    ):
        return False
    forbidden_paths = (
        "/source_depth/",
        "\\source_depth\\",
        "/source_masks/",
        "\\source_masks\\",
        "camera_poses.csv",
        "teacher_receipts.jsonl",
    )
    return not any(
        fragment in value
        for value in values
        for fragment in forbidden_paths
    )


def _select_development_sources(
    source_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        source_plan.get("schema") != SOURCE_PLAN_SCHEMA
        or source_plan.get("terminal") != SOURCE_PLAN_READY
        or source_plan.get("role_counts") != EXPECTED_ROLE_COUNTS
        or set(source_plan.get("roles", {})) != set(EXPECTED_ROLE_COUNTS)
    ):
        raise ValueError("G0 source plan identity or role counts drifted")
    roles = source_plan["roles"]
    all_ids = [
        str(item.get("session_id"))
        for group in roles.values()
        for item in group
    ]
    if len(all_ids) != 15 or len(set(all_ids)) != 15:
        raise ValueError("G0 source roles overlap or contain duplicate IDs")
    development = roles["development_reuse"]
    expected_prior_roles = ["train"] * 6 + ["dev"] * 3
    expected_g0_roles = (
        ["development_reuse_outcome_open_train"] * 6
        + ["development_reuse_outcome_open_model_selection"] * 3
    )
    if (
        len(development) != 9
        or [item.get("role") for item in development]
        != expected_prior_roles
        or [item.get("g0_source_role") for item in development]
        != expected_g0_roles
        or any(
            item.get("official_split") != "train"
            or item.get("fresh_evidence_credit") is not False
            or item.get("source_plan_origin")
            != "f0_metadata_plan_first_nine"
            or len(item.get("selected_source_frames", [])) != 25
            or float(item.get("source_fps", -1.0)) not in (5.0, 20.0)
            or float(item.get("target_fps", -1.0)) not in (5.0, 10.0)
            or item.get("selected_source_frames")
            != [
                frame
                * int(
                    round(
                        float(item["source_fps"])
                        / float(item["target_fps"])
                    )
                )
                for frame in range(25)
            ]
            for item in development
        )
    ):
        raise ValueError("Exact six-train/three-selection source set required")
    fresh = roles["one_shot_fresh_evaluation"]
    heldout = roles["reserved_fresh_heldout"]
    if any(
        item.get("g0_source_role")
        != "one_shot_fresh_evaluation_metadata_planned_only"
        or item.get("media_geometry_teacher_or_student_outcome_open")
        is not False
        or item.get("fresh_evidence_obtained") is not False
        for item in fresh
    ) or any(
        item.get("g0_source_role")
        != "metadata_only_future_heldout_reservation"
        or item.get("media_geometry_teacher_or_student_outcome_open")
        is not False
        or item.get("fresh_evidence_obtained") is not False
        for item in heldout
    ):
        raise ValueError("Fresh or reserved source firewall drifted")
    return [
        {
            **item,
            "materialized_role": (
                "train" if item["role"] == "train" else "model_selection"
            ),
        }
        for item in development
    ]


def _validate_design_contract(design: dict[str, Any]) -> None:
    corpus = design.get("development_corpus_contract", {})
    source = design.get("source_contract", {})
    authorization = design.get("ordered_authorization", {})
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("status") != DESIGN_STATUS
        or corpus.get("teacher_view") != "REFERENCE_STRIDE4_OFFSET2"
        or corpus.get("target_timeline")
        != "ALL_25_CURRENT_10FPS_FRAMES"
        or corpus.get("theta_distance_height_shape") != [6, 6, 2]
        or corpus.get("known_rule")
        != "AT_LEAST_5_OF_9_REFERENCE_PRISM_PROBES_PASS"
        or corpus.get("risk_rule") != "SUPPORT_COUNT_AT_LEAST_2"
        or corpus.get("clearance_rule")
        != "G0_SUPPORT_EQUIVALENT_SECOND_ORDER_PROXY_CLIPPED_TO_"
        "MINUS_0_5_PLUS_1_0_METERS"
        or corpus.get("unknown_risk_and_clearance_targets") is not None
        or corpus.get("train_record_count") != 150
        or corpus.get("model_selection_record_count") != 75
        or corpus.get("future_rgb_or_target_included") is not False
        or corpus.get("teacher_receipt_visible_to_student_loader")
        is not False
        or source.get("development_train", {}).get("count") != 6
        or source.get("development_model_selection", {}).get("count") != 3
        or authorization.get(
            "student_training_authorized_before_development_corpus_validation"
        )
        is not False
        or authorization.get("future_or_temporal_experiment_authorized")
        is not False
    ):
        raise ValueError("Frozen D1 development-corpus contract drifted")


def _load_context(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != EXECUTION_CONTRACT_SCHEMA
        or contract.get("status") != EXECUTION_CONTRACT_STATUS
    ):
        raise ValueError("D1 execution contract identity mismatch")
    _implementation_receipt(contract)
    design_path, design = _load_bound_parent(
        contract_path, contract, "d1_scientific_design"
    )
    amendment_path, amendment = _load_bound_parent(
        contract_path, contract, "d1_timeline_amendment"
    )
    _validate_design_contract(design)
    protocol_path, protocol = _load_bound_parent(
        design_path, design, "g0_protocol"
    )
    _, result = _load_bound_parent(
        design_path, design, "g0_d0_result"
    )
    source_plan_path, source_plan = _load_bound_parent(
        design_path, design, "g0_source_plan"
    )
    if (
        amendment.get("schema") != TIMELINE_AMENDMENT_SCHEMA
        or amendment.get("status") != TIMELINE_AMENDMENT_STATUS
        or amendment.get("corrected_contract")
        != "ALL_25_CURRENT_FRAMES_AT_EACH_SOURCE_PLAN_FROZEN_TARGET_FPS"
        or amendment.get("parents", {})
        .get("d1_scientific_design", {})
        .get("sha256")
        != _sha256(design_path)
        or amendment.get("parents", {})
        .get("g0_source_plan", {})
        .get("sha256")
        != _sha256(source_plan_path)
        or _sha256(amendment_path)
        != contract["parents"]["d1_timeline_amendment"]["sha256"]
    ):
        raise ValueError("D1 timeline amendment identity mismatch")
    _, training_contract = _load_bound_parent(
        design_path, design, "f0_1_student_training_contract"
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or result.get("schema") != RESULT_SCHEMA
        or result.get("terminal") != RESULT_TERMINAL
        or result.get("mechanics_terminal") != MECHANICS_TERMINAL
        or training_contract.get("schema") != TRAINING_CONTRACT_SCHEMA
        or training_contract.get("status") != TRAINING_CONTRACT_STATUS
        or source_plan.get("protocol_sha256") != _sha256(protocol_path)
        or result.get("parents", {}).get("g0_protocol", {}).get("sha256")
        != _sha256(protocol_path)
        or result.get("parents", {}).get("source_plan", {}).get("sha256")
        != _sha256(source_plan_path)
    ):
        raise ValueError("D1 parent identity or terminal mismatch")
    selected = _select_development_sources(source_plan)
    protocol_parents: dict[str, dict[str, Any]] = {}
    for key in protocol.get("parents", {}):
        _, protocol_parents[key] = _load_bound_parent(
            protocol_path, protocol, key
        )
    cohort = protocol_parents["f0_1_authority_cohort"]
    opportunity = protocol_parents["f0_1_teacher_opportunity"]
    source_lock = protocol_parents["f0_1_source_lock"]
    acquisition = protocol_parents["f0_1_acquisition_audit"]
    mechanics = protocol_parents["swept_envelope_mechanics"]
    if (
        cohort.get("terminal")
        != "F0_1_SANPO_SOURCE_AUTHORITY_COHORT_READY"
        or opportunity.get("terminal")
        != "F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS"
        or source_lock.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"
        or acquisition.get("terminal")
        != "F0_1_SANPO_ACQUISITION_AND_TRANSPORT_READY"
    ):
        raise ValueError("G0 source authority parent terminal mismatch")
    selected_ids = [str(item["session_id"]) for item in selected]
    parent_orders = (
        [str(item["session_id"]) for item in cohort.get("sources", [])],
        [
            str(item["session_id"])
            for item in opportunity.get("source_results", [])
        ],
        [str(item["session_id"]) for item in source_lock.get("sources", [])],
        [
            str(item["session_id"])
            for item in acquisition.get("sources", [])
        ],
    )
    if any(
        len(order) != 12 or order[:9] != selected_ids
        for order in parent_orders
    ):
        raise ValueError("Development source order differs from G0 parents")
    field = protocol.get("field_contract", {})
    view = field.get("teacher_view_for_every_role", {})
    clearance = protocol.get("signed_clearance_contract", {})
    if (
        view.get("point_sample_stride_xy") != 4
        or view.get("point_sample_offset_xy") != 2
        or view.get("name") != "reference"
        or field.get("current_only") is not True
        or clearance.get("order_statistic") != 2
        or clearance.get("raw_clearance_clip_m") != [-0.5, 1.0]
    ):
        raise ValueError("G0 current reference-view contract drifted")
    return (
        contract,
        design,
        protocol,
        mechanics,
        cohort,
        opportunity,
        selected,
    )


def _materialize_source(
    source: dict[str, Any],
    cohort_source: dict[str, Any],
    opportunity_source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    protocol: dict[str, Any],
    mechanics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_id = str(source["session_id"])
    role = str(source["materialized_role"])
    root = (datasets_root / _root_name(source)).resolve()
    authority_path = (
        authority_root / session_id[:8] / "authority.json"
    ).resolve()
    manifest_path = root / "manifest.replay.jsonl"
    spec_path = root / "dataset_spec.json"
    poses_path = root / "source_metadata/camera_poses.csv"
    rows = _load_jsonl(manifest_path)
    spec = _load_json(spec_path)
    authority = _load_json(authority_path)
    expected_frames = [
        int(value) for value in source["selected_source_frames"]
    ]
    if (
        len(rows) != 25
        or {str(row.get("session_id")) for row in rows} != {session_id}
        or [int(row["source_frame_index"]) for row in rows]
        != expected_frames
        or _sha256(manifest_path) != cohort_source["manifest_sha256"]
        or _sha256(spec_path) != cohort_source["dataset_spec_sha256"]
        or _sha256(poses_path) != cohort_source["camera_poses_sha256"]
        or _sha256(authority_path)
        != cohort_source["authority_report_sha256"]
        or cohort_source.get("role") != source["role"]
        or opportunity_source.get("role") != source["role"]
        or any(
            cohort_source.get(key) != opportunity_source.get(key)
            for key in (
                "manifest_sha256",
                "dataset_spec_sha256",
                "camera_poses_sha256",
                "authority_report_sha256",
            )
        )
    ):
        raise ValueError(f"{session_id}: source binding mismatch")
    bindings = {
        str(item["manifest_id"]): item
        for item in authority["source_pose_authority"]["bindings"]
    }
    planes = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    manifest_ids = {str(row["id"]) for row in rows}
    if set(bindings) != manifest_ids or set(planes) != manifest_ids:
        raise ValueError(f"{session_id}: authority coverage mismatch")
    camera = spec["camera"]
    if not _pixel_lattices_disjoint(
        int(camera["image_width"]), int(camera["image_height"])
    ):
        raise ValueError(f"{session_id}: teacher lattices overlap")
    field = protocol["field_contract"]
    clearance_contract = protocol["signed_clearance_contract"]
    theta_edges = _theta_edges(
        {
            "theta_bin_count": field["theta_bin_count"],
            "theta_range_degrees": field["theta_range_degrees"],
        }
    )
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][height])
        for height in HEIGHTS
    ]
    widths = np.asarray(
        [
            field["effective_lateral_half_width_m"][height]
            for height in HEIGHTS
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    view = field["teacher_view_for_every_role"]
    student_records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    for row in rows:
        manifest_id = str(row["id"])
        depth_path = _resolve_inside(
            root, str(row["source_depth_path"])
        )
        mask_path = _resolve_inside(root, str(row["source_mask_path"]))
        depth = _read_depth(
            depth_path, int(row["width"]), int(row["height"])
        )
        semantic = _read_semantic_class(
            mask_path, int(row["width"]), int(row["height"])
        )
        binding = bindings[manifest_id]
        basis = _anchor_basis(binding, planes[manifest_id])
        points, dynamic = _obstacle_points_world(
            root,
            row,
            binding,
            camera,
            stride=int(view["point_sample_stride_xy"]),
            offset=int(view["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
        probes = _swept_prism_probes_world(
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        passing = _probe_passes(
            probes,
            row,
            binding,
            camera,
            depth,
            semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        known = (passing.sum(axis=1) >= 5).reshape((6, 6, 2))
        support, _ = _swept_prism_counts(
            points,
            dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        _, clipped, inside_counts = _signed_clearance_field(
            points,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
            order_statistic=int(clearance_contract["order_statistic"]),
            final_edge_atol_m=float(
                clearance_contract["final_distance_edge_isclose"][
                    "atol_m"
                ]
            ),
            final_edge_rtol=float(
                clearance_contract["final_distance_edge_isclose"][
                    "rtol"
                ]
            ),
            clip_min_m=float(
                clearance_contract["raw_clearance_clip_m"][0]
            ),
            clip_max_m=float(
                clearance_contract["raw_clearance_clip_m"][1]
            ),
        )
        if not np.array_equal(support, inside_counts):
            raise ValueError(
                f"{session_id}:{manifest_id}: support/clearance mismatch"
            )
        labels = _nullable_labels(known, support, clipped)
        image_path = _resolve_inside(root, str(row["image_path"]))
        image_digest = _sha256(image_path)
        if (
            image_digest != str(row["image_sha256"])
            or _sha256(depth_path) != str(row["source_depth_sha256"])
            or _sha256(mask_path) != str(row["source_mask_sha256"])
        ):
            raise ValueError(
                f"{session_id}:{manifest_id}: current input hash mismatch"
            )
        source_frame_index = int(row["source_frame_index"])
        sample_id = (
            f"hftf_g0_d1_{role}_{session_id}_{source_frame_index:06d}"
        )
        student = {
            "sample_id": sample_id,
            "session_id": session_id,
            "role": role,
            "source_frame_index": source_frame_index,
            "manifest_id": manifest_id,
            "current_rgb": {
                "path": str(image_path),
                "sha256": image_digest,
            },
            "labels": labels,
        }
        if not _student_record_firewall(student):
            raise ValueError("Student sample violates teacher/future firewall")
        receipt = {
            "schema": TEACHER_RECEIPT_SCHEMA,
            "sample_id": sample_id,
            "session_id": session_id,
            "role": role,
            "source_frame_index": source_frame_index,
            "manifest_id": manifest_id,
            "teacher_view": {
                "name": "reference",
                "point_sample_stride_xy": 4,
                "point_sample_offset_xy": 2,
                "timeline": "current_only",
            },
            "teacher_inputs": {
                "source_depth": {
                    "path": str(depth_path),
                    "sha256": str(row["source_depth_sha256"]),
                },
                "source_mask": {
                    "path": str(mask_path),
                    "sha256": str(row["source_mask_sha256"]),
                },
                "camera_poses_sha256": cohort_source[
                    "camera_poses_sha256"
                ],
                "authority_report_sha256": cohort_source[
                    "authority_report_sha256"
                ],
            },
            "labels_sha256": _sha256_bytes(_canonical_bytes(labels)),
            "student_loader_authorized": False,
        }
        student_records.append(student)
        receipt_records.append(receipt)
    return student_records, receipt_records


def _scientific_payload(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    (
        contract,
        design,
        protocol,
        mechanics,
        cohort,
        opportunity,
        selected,
    ) = _load_context(contract_path)
    design_path = _resolve_parent(
        contract_path, contract["parents"]["d1_scientific_design"]
    )
    amendment_path = _resolve_parent(
        contract_path, contract["parents"]["d1_timeline_amendment"]
    )
    cohort_by_id = {
        str(item["session_id"]): item
        for item in cohort["sources"]
    }
    opportunity_by_id = {
        str(item["session_id"]): item
        for item in opportunity["source_results"]
    }
    students: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for source in selected:
        session_id = str(source["session_id"])
        source_students, source_receipts = _materialize_source(
            source,
            cohort_by_id[session_id],
            opportunity_by_id[session_id],
            datasets_root,
            authority_root,
            protocol,
            mechanics,
        )
        students.extend(source_students)
        receipts.extend(source_receipts)
    ids = [record["sample_id"] for record in students]
    if (
        len(students) != 225
        or len(receipts) != 225
        or len(set(ids)) != 225
        or ids != [record["sample_id"] for record in receipts]
        or sum(record["role"] == "train" for record in students) != 150
        or sum(
            record["role"] == "model_selection"
            for record in students
        )
        != 75
        or not all(_student_record_firewall(record) for record in students)
    ):
        raise ValueError("D1 corpus cardinality or firewall mismatch")
    student_bytes = _canonical_jsonl(students)
    receipt_bytes = _canonical_jsonl(receipts)
    spec = {
        "schema": DATASET_SCHEMA,
        "terminal": READY,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "CONSUMED_DEVELOPMENT_SOURCE_CORPUS_ONLY",
        "claim_ceiling": "SYNTHETIC_SIGNED_CLEARANCE_PROXY_ONLY",
        "parents": {
            "execution_contract": {
                "path": str(contract_path.resolve()),
                "sha256": _sha256(contract_path),
            },
            "d1_scientific_design": {
                "path": str(design_path),
                "sha256": _sha256(design_path),
            },
            "d1_timeline_amendment": {
                "path": str(amendment_path),
                "sha256": _sha256(amendment_path),
            }
        },
        "implementation": {
            "path": IMPLEMENTATION_PATH,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_count": 9,
        "student_record_count": 225,
        "teacher_receipt_count": 225,
        "role_source_counts": {"train": 6, "model_selection": 3},
        "role_record_counts": {"train": 150, "model_selection": 75},
        "source_session_ids": [
            str(source["session_id"]) for source in selected
        ],
        "source_target_fps": {
            str(source["session_id"]): float(source["target_fps"])
            for source in selected
        },
        "student_record_schema": {
            "top_level_keys": sorted(STUDENT_TOP_LEVEL_KEYS),
            "label_shape": [2, 6, 6],
            "current_rgb_only": True,
            "risk_and_clearance_unknown_values_are_null": True,
        },
        "teacher_contract": {
            "view": "REFERENCE_STRIDE4_OFFSET2",
            "current_frames_per_source": 25,
            "known_rule": "AT_LEAST_5_OF_9_REFERENCE_PRISM_PROBES_PASS",
            "risk_rule": "SUPPORT_COUNT_AT_LEAST_2",
            "clearance_rule": (
                "G0_SUPPORT_EQUIVALENT_SECOND_ORDER_PROXY_"
                "CLIPPED_TO_MINUS_0_5_PLUS_1_0_METERS"
            ),
        },
        "files": {
            "student_samples.jsonl": {
                "sha256": _sha256_bytes(student_bytes),
                "record_count": 225,
                "student_loader_authorized": True,
            },
            "teacher_receipts.jsonl": {
                "sha256": _sha256_bytes(receipt_bytes),
                "record_count": 225,
                "student_loader_authorized": False,
            },
        },
        "checks": {
            "exact_six_train_three_model_selection_sources": True,
            "all_25_current_frames_per_source": True,
            "reference_stride4_offset2_only": True,
            "student_teacher_ids_one_to_one": True,
            "current_rgb_hashes_match": True,
            "unknown_risk_and_clearance_targets_are_null": True,
            "student_records_exclude_teacher_truth_and_future": True,
            "fresh_and_reserved_source_media_unopened": True,
        },
        "authorization": {
            "development_student_samples_available_after_validation": True,
            "student_training_authorized_before_separate_validation": False,
            "fresh_source_opening_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }
    return student_bytes, receipt_bytes, spec


def _require_canonical_paths(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    repository = _repository_root()
    expected = (
        (
            contract_path.resolve(),
            (
                repository
                / "docs/research/hftf/"
                "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
                "EXECUTION_CONTRACT_D1_"
                "2026-08-01.json"
            ).resolve(),
            "contract",
        ),
        (
            datasets_root.resolve(),
            (repository / "artifacts.local/evidence/datasets").resolve(),
            "datasets",
        ),
        (
            authority_root.resolve(),
            (
                repository
                / "artifacts.local/evidence/hftf/"
                "stage-c-f0-1-sanpo-authority-20260801"
            ).resolve(),
            "authority",
        ),
        (
            output_root.resolve(),
            (
                repository
                / "artifacts.local/evidence/hftf/"
                "stage-c-g0-d1-development-corpus-20260801"
            ).resolve(),
            "output",
        ),
    )
    for actual, frozen, label in expected:
        if actual != frozen:
            raise ValueError(
                f"D1 materializer noncanonical {label} path: {actual}"
            )
    return tuple(item[0] for item in expected)  # type: ignore[return-value]


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_publish(
    output_root: Path,
    student_bytes: bytes,
    receipt_bytes: bytes,
    spec: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite corpus: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    try:
        _write_new(partial / "student_samples.jsonl", student_bytes)
        _write_new(partial / "teacher_receipts.jsonl", receipt_bytes)
        _write_new(
            partial / "dataset_spec.json",
            json.dumps(
                spec,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )
        partial.replace(output_root)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def materialize(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    (
        contract_path,
        datasets_root,
        authority_root,
        output_root,
    ) = _require_canonical_paths(
        contract_path, datasets_root, authority_root, output_root
    )
    student_bytes, receipt_bytes, spec = _scientific_payload(
        contract_path, datasets_root, authority_root
    )
    if not all(spec["checks"].values()):
        raise ValueError("D1 corpus validation checks did not all pass")
    _atomic_publish(output_root, student_bytes, receipt_bytes, spec)
    return spec


def main() -> int:
    repository = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
            "EXECUTION_CONTRACT_D1_"
            "2026-08-01.json"
        ),
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=repository / "artifacts.local/evidence/datasets",
    )
    parser.add_argument(
        "--authority-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-f0-1-sanpo-authority-20260801"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-20260801"
        ),
    )
    args = parser.parse_args()
    try:
        report = materialize(
            args.contract,
            args.datasets_root,
            args.authority_root,
            args.output_root,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": DATASET_SCHEMA,
                    "terminal": NOT_EVALUABLE,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "output_written": False,
                    "student_training_authorized": False,
                    "fresh_source_opening_authorized": False,
                    "future_or_temporal_experiment_authorized": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
