#!/usr/bin/env python3
"""Independently validate the frozen HFTF G0-D1 development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_stage_c_f0_1_teacher_opportunity import (
    _pixel_lattices_disjoint,
    _probe_passes,
    _root_name,
)
from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (
    _anchor_basis,
    _obstacle_points_world,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
    _theta_edges,
)
from run_stage_c_g0_signed_clearance_mechanics import (
    _signed_clearance_field,
)

CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "learnability_execution_contract_d1"
)
CONTRACT_STATUS = (
    "FROZEN_BEFORE_D1_DEVELOPMENT_CORPUS_OR_STUDENT_OUTCOME"
)
TIMELINE_AMENDMENT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "learnability_d1_timeline_amendment"
)
DATASET_SCHEMA = "blindassist_hftf_stage_c_g0_d1_development_corpus"
DATASET_READY = "G0_D1_CURRENT_CLEARANCE_DEVELOPMENT_CORPUS_READY"
RECEIPT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_teacher_receipt"
VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_development_corpus_validation"
)
VALIDATED = "G0_D1_DEVELOPMENT_CORPUS_VALIDATED"
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/"
    "validate_stage_c_g0_d1_development_corpus.py"
)
MATERIALIZER_PATH = (
    "scripts/research/hftf/"
    "materialize_stage_c_g0_d1_development_corpus.py"
)
STUDENT_KEYS = {
    "sample_id",
    "session_id",
    "role",
    "source_frame_index",
    "manifest_id",
    "current_rgb",
    "labels",
}
LABEL_KEYS = {
    "known_target",
    "risk_target_nullable",
    "clearance_target_m_nullable",
}
RECEIPT_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "role",
    "source_frame_index",
    "manifest_id",
    "teacher_view",
    "teacher_inputs",
    "labels_sha256",
    "student_loader_authorized",
}
ROLES = ("train", "model_selection")
HEIGHTS = ("body", "head")
VALIDATION_CHECKS = {
    "exact_file_set",
    "exact_six_train_three_model_selection_sources",
    "exact_25_frames_per_source",
    "student_teacher_receipts_one_to_one_and_ordered",
    "student_exact_schema_and_current_rgb_hashes",
    "risk_equals_clearance_strictly_below_zero",
    "unknown_targets_are_null_and_never_safe",
    "source_height_targets_nondegenerate",
    "fresh_and_reserved_sources_excluded",
    "teacher_receipts_not_student_authorized",
    "authoritative_manifest_teacher_and_label_rederived",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSONL object at {path}:{line_number}"
            )
        records.append(value)
    return records


def _resolve_parent(owner_path: Path, receipt: dict[str, Any]) -> Path:
    raw = Path(str(receipt.get("path", "")))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (_repository_root() / raw).resolve()
    return (owner_path.parent / raw).resolve()


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


def _load_source_roles(
    contract_path: Path,
    contract: dict[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, list[int]],
    dict[str, float],
    set[str],
]:
    design_path, design = _load_bound_parent(
        contract_path, contract, "d1_scientific_design"
    )
    _, source_plan = _load_bound_parent(
        design_path, design, "g0_source_plan"
    )
    roles = source_plan.get("roles", {})
    development = roles.get("development_reuse", [])
    fresh = roles.get("one_shot_fresh_evaluation", [])
    heldout = roles.get("reserved_fresh_heldout", [])
    if (
        not isinstance(development, list)
        or not isinstance(fresh, list)
        or not isinstance(heldout, list)
        or len(development) != 9
        or len(fresh) != 3
        or len(heldout) != 3
    ):
        raise ValueError("D1 source-plan role cardinality mismatch")
    expected: dict[str, str] = {}
    expected_frames: dict[str, list[int]] = {}
    target_fps: dict[str, float] = {}
    for index, item in enumerate(development):
        role = "train" if index < 6 else "model_selection"
        required_prior = "train" if index < 6 else "dev"
        required_g0 = (
            "development_reuse_outcome_open_train"
            if index < 6
            else "development_reuse_outcome_open_model_selection"
        )
        session_id = str(item.get("session_id", ""))
        selected_frames = item.get("selected_source_frames")
        source_fps = float(item.get("source_fps", -1.0))
        frozen_target_fps = float(item.get("target_fps", -1.0))
        if (
            not session_id
            or item.get("role") != required_prior
            or item.get("g0_source_role") != required_g0
            or item.get("fresh_evidence_credit") is not False
            or not isinstance(selected_frames, list)
            or len(selected_frames) != 25
            or len(set(selected_frames)) != 25
            or any(
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame < 0
                for frame in selected_frames
            )
            or source_fps not in (5.0, 20.0)
            or frozen_target_fps not in (5.0, 10.0)
            or selected_frames
            != [
                frame
                * int(round(source_fps / frozen_target_fps))
                for frame in range(25)
            ]
        ):
            raise ValueError("D1 development source role drifted")
        expected[session_id] = role
        expected_frames[session_id] = selected_frames
        target_fps[session_id] = frozen_target_fps
    forbidden = {
        str(item.get("session_id", "")) for item in [*fresh, *heldout]
    }
    if (
        len(expected) != 9
        or len(forbidden) != 6
        or set(expected) & forbidden
        or any(
            item.get("media_geometry_teacher_or_student_outcome_open")
            is not False
            for item in [*fresh, *heldout]
        )
    ):
        raise ValueError("Fresh/reserved source firewall drifted")
    return expected, expected_frames, target_fps, forbidden


def _strict_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected strict numeric target")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Target must be finite")
    return result


def _derived_labels(
    known: np.ndarray,
    support: np.ndarray,
    clipped: np.ndarray,
) -> dict[str, Any]:
    known_h = np.transpose(known, (2, 0, 1))
    support_h = np.transpose(support, (2, 0, 1))
    clipped_h = np.transpose(clipped, (2, 0, 1))
    result: dict[str, Any] = {
        "known_target": [],
        "risk_target_nullable": [],
        "clearance_target_m_nullable": [],
    }
    for height in range(2):
        known_rows: list[list[int]] = []
        risk_rows: list[list[int | None]] = []
        clearance_rows: list[list[float | None]] = []
        for theta in range(6):
            known_row: list[int] = []
            risk_row: list[int | None] = []
            clearance_row: list[float | None] = []
            for distance in range(6):
                is_known = bool(known_h[height, theta, distance])
                known_row.append(int(is_known))
                risk_row.append(
                    int(support_h[height, theta, distance] >= 2)
                    if is_known
                    else None
                )
                clearance_row.append(
                    float(clipped_h[height, theta, distance])
                    if is_known
                    else None
                )
            known_rows.append(known_row)
            risk_rows.append(risk_row)
            clearance_rows.append(clearance_row)
        result["known_target"].append(known_rows)
        result["risk_target_nullable"].append(risk_rows)
        result["clearance_target_m_nullable"].append(clearance_rows)
    return result


def _authoritative_bindings(
    contract_path: Path,
    contract: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    design_path, design = _load_bound_parent(
        contract_path, contract, "d1_scientific_design"
    )
    protocol_path, protocol = _load_bound_parent(
        design_path, design, "g0_protocol"
    )
    _, source_plan = _load_bound_parent(
        design_path, design, "g0_source_plan"
    )
    _, cohort = _load_bound_parent(
        protocol_path, protocol, "f0_1_authority_cohort"
    )
    _, mechanics = _load_bound_parent(
        protocol_path, protocol, "swept_envelope_mechanics"
    )
    sources = source_plan["roles"]["development_reuse"]
    cohort_by_id = {
        str(source["session_id"]): source for source in cohort["sources"]
    }
    datasets_root = (
        _repository_root() / "artifacts.local/evidence/datasets"
    ).resolve()
    authority_root = (
        _repository_root()
        / "artifacts.local/evidence/hftf/"
        "stage-c-f0-1-sanpo-authority-20260801"
    ).resolve()
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
    authoritative: dict[tuple[str, int], dict[str, Any]] = {}
    for source_index, source in enumerate(sources):
        session_id = str(source["session_id"])
        role = "train" if source_index < 6 else "model_selection"
        cohort_source = cohort_by_id[session_id]
        root = (datasets_root / _root_name(source)).resolve()
        manifest_path = root / "manifest.replay.jsonl"
        spec_path = root / "dataset_spec.json"
        poses_path = root / "source_metadata/camera_poses.csv"
        authority_path = (
            authority_root / session_id[:8] / "authority.json"
        )
        rows = _load_jsonl(manifest_path)
        spec = _load_json(spec_path)
        authority = _load_json(authority_path)
        if (
            _sha256(manifest_path) != cohort_source["manifest_sha256"]
            or _sha256(spec_path) != cohort_source["dataset_spec_sha256"]
            or _sha256(poses_path) != cohort_source["camera_poses_sha256"]
            or _sha256(authority_path)
            != cohort_source["authority_report_sha256"]
            or [int(row["source_frame_index"]) for row in rows]
            != source["selected_source_frames"]
        ):
            raise ValueError(f"{session_id}: authority source binding mismatch")
        pose_bindings = {
            str(item["manifest_id"]): item
            for item in authority["source_pose_authority"]["bindings"]
        }
        planes = {
            str(item["manifest_id"]): item["local_ground_plane"]
            for item in authority["ground_and_body_proxy_canary"]["per_frame"]
        }
        camera = spec["camera"]
        if not _pixel_lattices_disjoint(
            int(camera["image_width"]), int(camera["image_height"])
        ):
            raise ValueError(f"{session_id}: teacher lattices overlap")
        for row in rows:
            manifest_id = str(row["id"])
            binding = pose_bindings[manifest_id]
            basis = _anchor_basis(binding, planes[manifest_id])
            depth_path = _resolve_inside(
                root, str(row["source_depth_path"])
            )
            mask_path = _resolve_inside(
                root, str(row["source_mask_path"])
            )
            depth = _read_depth(
                depth_path, int(row["width"]), int(row["height"])
            )
            semantic = _read_semantic_class(
                mask_path, int(row["width"]), int(row["height"])
            )
            if (
                _sha256(depth_path) != str(row["source_depth_sha256"])
                or _sha256(mask_path) != str(row["source_mask_sha256"])
            ):
                raise ValueError(
                    f"{session_id}:{manifest_id}: teacher input hash drift"
                )
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
                order_statistic=int(
                    clearance_contract["order_statistic"]
                ),
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
                    f"{session_id}:{manifest_id}: support drift"
                )
            image_path = _resolve_inside(root, str(row["image_path"]))
            frame_index = int(row["source_frame_index"])
            authoritative[(session_id, frame_index)] = {
                "sample_id": (
                    f"hftf_g0_d1_{role}_{session_id}_{frame_index:06d}"
                ),
                "role": role,
                "manifest_id": manifest_id,
                "current_rgb": {
                    "path": str(image_path),
                    "sha256": str(row["image_sha256"]),
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
                "labels": _derived_labels(known, support, clipped),
            }
    if len(authoritative) != 225:
        raise ValueError("D1 authoritative binding count mismatch")
    return authoritative


def _validate_labels(
    labels: Any,
) -> tuple[list[int], list[int], list[float]]:
    if not isinstance(labels, dict) or set(labels) != LABEL_KEYS:
        raise ValueError("D1 label schema mismatch")
    flattened: tuple[list[Any], list[Any], list[Any]] = ([], [], [])
    for destination, key in zip(
        flattened,
        (
            "known_target",
            "risk_target_nullable",
            "clearance_target_m_nullable",
        ),
        strict=True,
    ):
        value = labels[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(height, list) or len(height) != 6 for height in value)
            or any(
                not isinstance(row, list) or len(row) != 6
                for height in value
                for row in height
            )
        ):
            raise ValueError("D1 target shape must be 2x6x6")
        destination.extend(
            cell for height in value for row in height for cell in row
        )
    known_raw, risk_raw, clearance_raw = flattened
    known: list[int] = []
    risk: list[int] = []
    clearance: list[float] = []
    for known_value, risk_value, clearance_value in zip(
        known_raw, risk_raw, clearance_raw, strict=True
    ):
        numeric_known = _strict_number(known_value)
        if numeric_known not in (0.0, 1.0):
            raise ValueError("Known target must be exact numeric 0 or 1")
        known.append(int(numeric_known))
        if numeric_known == 0.0:
            if risk_value is not None or clearance_value is not None:
                raise ValueError("UNKNOWN risk and clearance must be null")
            risk.append(-1)
            clearance.append(math.nan)
            continue
        numeric_risk = _strict_number(risk_value)
        numeric_clearance = _strict_number(clearance_value)
        if numeric_risk not in (0.0, 1.0):
            raise ValueError("Risk target must be exact numeric 0 or 1")
        if not -0.5 <= numeric_clearance <= 1.0:
            raise ValueError("Clearance target outside frozen clip")
        if bool(numeric_risk == 1.0) != bool(numeric_clearance < 0.0):
            raise ValueError("Risk and clearance sign disagree")
        risk.append(int(numeric_risk))
        clearance.append(numeric_clearance)
    return known, risk, clearance


def _validate_records(
    students: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    expected_roles: dict[str, str],
    forbidden_ids: set[str],
    expected_frames: dict[str, list[int]] | None = None,
    authoritative: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(students) != 225 or len(receipts) != 225:
        raise ValueError("D1 corpus must contain exactly 225 paired records")
    sample_ids: set[str] = set()
    rgb_paths: set[Path] = set()
    source_counts: Counter[tuple[str, str]] = Counter()
    frame_indices: dict[str, set[int]] = defaultdict(set)
    diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    for session_id in expected_roles:
        for height in HEIGHTS:
            diagnostics[(session_id, height)] = {
                "known": 0,
                "risk": 0,
                "safe": 0,
                "near_boundary": 0,
                "clearance_mm_bins": set(),
                "risk_non_clip_min": 0,
                "safe_non_clip_max": 0,
            }
    for index, (student, receipt) in enumerate(
        zip(students, receipts, strict=True)
    ):
        if set(student) != STUDENT_KEYS:
            raise ValueError("Student record exact top-level schema mismatch")
        if set(receipt) != RECEIPT_KEYS:
            raise ValueError("Teacher receipt exact schema mismatch")
        identity_keys = (
            "sample_id",
            "session_id",
            "role",
            "source_frame_index",
            "manifest_id",
        )
        if any(student[key] != receipt[key] for key in identity_keys):
            raise ValueError("Student/teacher receipt identity mismatch")
        sample_id = str(student["sample_id"])
        session_id = str(student["session_id"])
        role = str(student["role"])
        frame_index = student["source_frame_index"]
        if (
            not sample_id
            or sample_id in sample_ids
            or session_id in forbidden_ids
            or expected_roles.get(session_id) != role
            or role not in ROLES
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("D1 source, role, frame, or sample identity mismatch")
        sample_ids.add(sample_id)
        source_counts[(session_id, role)] += 1
        frame_indices[session_id].add(frame_index)
        rgb = student["current_rgb"]
        if not isinstance(rgb, dict) or set(rgb) != {"path", "sha256"}:
            raise ValueError("Current RGB schema mismatch")
        rgb_path = Path(str(rgb["path"]))
        if (
            not rgb_path.is_absolute()
            or not rgb_path.is_file()
            or _sha256(rgb_path) != str(rgb["sha256"])
        ):
            raise ValueError("Current RGB byte receipt mismatch")
        rgb_paths.add(rgb_path.resolve())
        known, risk, clearance = _validate_labels(student["labels"])
        if authoritative is not None:
            expected = authoritative.get((session_id, frame_index))
            if (
                expected is None
                or sample_id != expected["sample_id"]
                or role != expected["role"]
                or student["manifest_id"] != expected["manifest_id"]
                or student["current_rgb"] != expected["current_rgb"]
                or student["labels"] != expected["labels"]
                or receipt.get("teacher_inputs")
                != expected["teacher_inputs"]
            ):
                raise ValueError(
                    "D1 authoritative manifest, teacher, or label mismatch"
                )
        if receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("Teacher receipt identity mismatch")
        if (
            receipt.get("student_loader_authorized") is not False
            or receipt.get("labels_sha256")
            != _sha256_bytes(_canonical_bytes(student["labels"]))
            or receipt.get("teacher_view")
            != {
                "name": "reference",
                "point_sample_stride_xy": 4,
                "point_sample_offset_xy": 2,
                "timeline": "current_only",
            }
        ):
            raise ValueError("Teacher receipt firewall or label hash mismatch")
        for height_index, height in enumerate(HEIGHTS):
            scope = diagnostics[(session_id, height)]
            start = height_index * 36
            for offset in range(start, start + 36):
                if known[offset] == 0:
                    continue
                value = clearance[offset]
                scope["known"] += 1
                scope["risk"] += int(risk[offset] == 1)
                scope["safe"] += int(risk[offset] == 0)
                scope["near_boundary"] += int(abs(value) <= 0.2)
                scope["clearance_mm_bins"].add(round(value * 1000.0))
                scope["risk_non_clip_min"] += int(
                    risk[offset] == 1 and value > -0.5
                )
                scope["safe_non_clip_max"] += int(
                    risk[offset] == 0 and value < 1.0
                )
    frozen_frames = expected_frames or {
        session_id: list(range(25)) for session_id in expected_roles
    }
    if (
        set(frame_indices) != set(expected_roles)
        or set(frozen_frames) != set(expected_roles)
        or any(
            frame_indices[session_id] != set(frozen_frames[session_id])
            for session_id in expected_roles
        )
        or any(
            source_counts[(session_id, role)] != 25
            for session_id, role in expected_roles.items()
        )
        or len(rgb_paths) != 225
        or sum(role == "train" for role in expected_roles.values()) != 6
        or sum(role == "model_selection" for role in expected_roles.values())
        != 3
        or (
            authoritative is not None
            and {
                (
                    str(record["session_id"]),
                    int(record["source_frame_index"]),
                )
                for record in students
            }
            != set(authoritative)
        )
    ):
        raise ValueError("D1 exact 6/3 sources with 25 frames required")
    serializable_scopes: list[dict[str, Any]] = []
    for (session_id, height), scope in diagnostics.items():
        if (
            scope["known"] <= 0
            or scope["risk"] <= 0
            or scope["safe"] <= 0
            or scope["near_boundary"] < 5
            or len(scope["clearance_mm_bins"]) < 20
            or scope["risk_non_clip_min"] <= 0
            or scope["safe_non_clip_max"] <= 0
        ):
            raise ValueError(
                f"D1 nondegenerate target gate failed: {session_id}:{height}"
            )
        serializable_scopes.append(
            {
                "session_id": session_id,
                "height": height,
                "known_cell_count": scope["known"],
                "risk_cell_count": scope["risk"],
                "safe_cell_count": scope["safe"],
                "near_boundary_cell_count": scope["near_boundary"],
                "distinct_clearance_mm_bin_count": len(
                    scope["clearance_mm_bins"]
                ),
                "risk_non_clip_min_count": scope["risk_non_clip_min"],
                "safe_non_clip_max_count": scope["safe_non_clip_max"],
            }
        )
    return {
        "record_counts": {"train": 150, "model_selection": 75},
        "source_roles": expected_roles,
        "source_frame_indices": frozen_frames,
        "source_height_diagnostics": serializable_scopes,
        "unknown_to_safe_violation_count": 0,
    }


def _require_canonical_paths(
    contract_path: Path,
    corpus_root: Path,
    output_path: Path,
) -> None:
    repository = _repository_root()
    expected = {
        "contract": (
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ).resolve(),
        "corpus": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-20260801"
        ).resolve(),
        "output": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-validation-20260801/"
            "validation.json"
        ).resolve(),
    }
    actual = {
        "contract": contract_path.resolve(),
        "corpus": corpus_root.resolve(),
        "output": output_path.resolve(),
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"D1 corpus validator noncanonical {key} path")


def _atomic_write_new(output_path: Path, payload: bytes) -> None:
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite D1 corpus validation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.partial-",
        dir=output_path.parent,
    )
    partial_path = Path(partial_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if output_path.exists():
            raise FileExistsError(
                "D1 corpus validation output appeared during write"
            )
        partial_path.replace(output_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise


def validate(
    contract_path: Path,
    corpus_root: Path,
    output_path: Path,
    *,
    require_canonical_paths: bool = True,
) -> dict[str, Any]:
    if require_canonical_paths:
        _require_canonical_paths(contract_path, corpus_root, output_path)
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("D1 execution contract identity mismatch")
    amendment_path, amendment = _load_bound_parent(
        contract_path, contract, "d1_timeline_amendment"
    )
    if (
        amendment.get("schema") != TIMELINE_AMENDMENT_SCHEMA
        or amendment.get("status") != CONTRACT_STATUS
        or amendment.get("corrected_contract")
        != "ALL_25_CURRENT_FRAMES_AT_EACH_SOURCE_PLAN_FROZEN_TARGET_FPS"
    ):
        raise ValueError("D1 timeline amendment identity mismatch")
    implementations = contract.get("implementations", {})
    validator_receipt = implementations.get(
        "development_corpus_validator", {}
    )
    materializer_receipt = implementations.get(
        "development_corpus_materializer", {}
    )
    if (
        Path(str(validator_receipt.get("path", ""))).as_posix()
        != IMPLEMENTATION_PATH
        or validator_receipt.get("sha256")
        != _sha256(Path(__file__).resolve())
        or validator_receipt.get("execution_authorized") is not True
        or Path(str(materializer_receipt.get("path", ""))).as_posix()
        != MATERIALIZER_PATH
        or materializer_receipt.get("sha256")
        != _sha256(
            Path(__file__).resolve().parent
            / "materialize_stage_c_g0_d1_development_corpus.py"
        )
        or materializer_receipt.get("execution_authorized") is not True
    ):
        raise ValueError("D1 corpus implementation receipt mismatch")
    if (
        not corpus_root.is_dir()
        or {path.name for path in corpus_root.iterdir()}
        != {
            "student_samples.jsonl",
            "teacher_receipts.jsonl",
            "dataset_spec.json",
        }
    ):
        raise ValueError("D1 corpus exact file set mismatch")
    student_path = corpus_root / "student_samples.jsonl"
    receipt_path = corpus_root / "teacher_receipts.jsonl"
    spec_path = corpus_root / "dataset_spec.json"
    spec = _load_json(spec_path)
    if (
        spec.get("schema") != DATASET_SCHEMA
        or spec.get("terminal") != DATASET_READY
        or spec.get("parents", {}).get("execution_contract", {}).get("sha256")
        != _sha256(contract_path)
        or spec.get("parents", {})
        .get("d1_timeline_amendment", {})
        .get("sha256")
        != _sha256(amendment_path)
        or spec.get("implementation", {}).get("sha256")
        != materializer_receipt.get("sha256")
        or spec.get("files", {})
        .get("student_samples.jsonl", {})
        .get("sha256")
        != _sha256(student_path)
        or spec.get("files", {})
        .get("teacher_receipts.jsonl", {})
        .get("sha256")
        != _sha256(receipt_path)
        or spec.get("files", {})
        .get("teacher_receipts.jsonl", {})
        .get("student_loader_authorized")
        is not False
        or spec.get("authorization", {}).get("fresh_source_opening_authorized")
        is not False
        or spec.get("authorization", {}).get(
            "future_or_temporal_experiment_authorized"
        )
        is not False
    ):
        raise ValueError("D1 dataset specification receipt mismatch")
    (
        expected_roles,
        expected_frames,
        expected_target_fps,
        forbidden_ids,
    ) = _load_source_roles(contract_path, contract)
    if (
        spec.get("source_session_ids") != list(expected_roles)
        or spec.get("role_source_counts")
        != {"train": 6, "model_selection": 3}
        or spec.get("role_record_counts")
        != {"train": 150, "model_selection": 75}
        or spec.get("source_target_fps") != expected_target_fps
    ):
        raise ValueError("D1 dataset source summary mismatch")
    students = _load_jsonl(student_path)
    receipts = _load_jsonl(receipt_path)
    authoritative = _authoritative_bindings(contract_path, contract)
    diagnostics = _validate_records(
        students,
        receipts,
        expected_roles,
        forbidden_ids,
        expected_frames,
        authoritative,
    )
    result = {
        "schema": VALIDATION_SCHEMA,
        "terminal": VALIDATED,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SYNTHETIC_SIGNED_CLEARANCE_PROXY_ONLY",
        "implementation": {
            "path": IMPLEMENTATION_PATH,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "parents": {
            "execution_contract": {
                "path": str(contract_path.resolve()),
                "sha256": _sha256(contract_path),
            },
            "dataset_spec": {
                "path": str(spec_path.resolve()),
                "sha256": _sha256(spec_path),
            },
        },
        "student_samples_path": str(student_path.resolve()),
        "student_samples_sha256": _sha256(student_path),
        "teacher_receipts_path": str(receipt_path.resolve()),
        "teacher_receipts_sha256": _sha256(receipt_path),
        **diagnostics,
        "checks": {key: True for key in sorted(VALIDATION_CHECKS)},
        "authorization": {
            "development_training_authorized": True,
            "fresh_source_opening_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }
    payload = (
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_new(output_path, payload)
    return result


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
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-20260801"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-validation-20260801/"
            "validation.json"
        ),
    )
    arguments = parser.parse_args()
    result = validate(
        arguments.contract,
        arguments.corpus_root,
        arguments.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
