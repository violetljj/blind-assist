#!/usr/bin/env python3
"""Materialize the frozen F0.1 train/dev corpus with teacher quarantine."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_f0_1_teacher_opportunity import (
    EXECUTION_SHA256,
    F0_1_SHA256,
    F0_SHA256,
    MECHANICS_SHA256,
    READY as OPPORTUNITY_READY,
    SOURCE_LOCK_SHA256,
    _causal_future_basis,
    _canonical_bytes,
    _probe_passes,
    _root_name,
    _timeline_contract,
    _union_support,
)
from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (
    _obstacle_points_world,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
    _sha256,
    _theta_edges,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_c_f0_1_train_dev_corpus"
READY = "F0_1_SANPO_TRAIN_DEV_CORPUS_READY"
NOT_EVALUABLE = "F0_1_SANPO_TRAIN_DEV_CORPUS_NOT_EVALUABLE"
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_corpus_materialization_contract_f0_1"
)
CONTRACT_SHA256 = (
    "b2d5c2b1cadc05ba22731c974e79d159022af092709d5b9139d97837ba6a0376"
)
OPPORTUNITY_SHA256 = (
    "9db97892ae93267856e1388bccf808deb8947311e25cc5b39a1c362b4bb348b5"
)
HEIGHTS = ("body", "head")
HORIZONS = ("current", "future")
FORBIDDEN_STUDENT_KEY_FRAGMENTS = (
    "source_depth",
    "source_mask",
    "pose",
    "semantic",
    "future_rgb",
    "future_image",
    "teacher_receipt",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _nested_nullable_targets(
    known: np.ndarray, support: np.ndarray
) -> dict[str, list[Any]]:
    if known.shape != (6, 6, 2) or support.shape != (6, 6, 2):
        raise ValueError("Frozen body/head field must have shape 6x6x2")
    ordered_known = known.transpose(2, 0, 1)
    ordered_risk = (support >= 2).transpose(2, 0, 1)
    risk: list[list[list[int | None]]] = []
    for height_index in range(2):
        theta_rows: list[list[int | None]] = []
        for theta_index in range(6):
            theta_rows.append(
                [
                    (
                        int(ordered_risk[height_index, theta_index, distance])
                        if ordered_known[
                            height_index, theta_index, distance
                        ]
                        else None
                    )
                    for distance in range(6)
                ]
            )
        risk.append(theta_rows)
    return {
        "known_target": ordered_known.astype(np.uint8).tolist(),
        "risk_target_nullable": risk,
    }


def _student_record_firewall(record: dict[str, Any]) -> bool:
    def strings(value: Any) -> tuple[list[str], list[str]]:
        if isinstance(value, dict):
            nested_keys = [str(key).lower() for key in value]
            nested_values: list[str] = []
            for item in value.values():
                item_keys, item_values = strings(item)
                nested_keys.extend(item_keys)
                nested_values.extend(item_values)
            return nested_keys, nested_values
        if isinstance(value, list):
            nested_keys: list[str] = []
            nested_values: list[str] = []
            for item in value:
                item_keys, item_values = strings(item)
                nested_keys.extend(item_keys)
                nested_values.extend(item_values)
            return nested_keys, nested_values
        return ([], [value.lower()] if isinstance(value, str) else [])

    all_keys, all_values = strings(record)
    if any(
        fragment in key
        for key in all_keys
        for fragment in FORBIDDEN_STUDENT_KEY_FRAGMENTS
    ):
        return False
    forbidden_value_fragments = (
        "/source_depth/",
        "\\source_depth\\",
        "/source_masks/",
        "\\source_masks\\",
        "camera_poses.csv",
        "teacher_receipts.jsonl",
    )
    if any(
        fragment in value
        for value in all_values
        for fragment in forbidden_value_fragments
    ):
        return False
    history = record.get("history_rgb")
    if history is not None:
        if (
            not isinstance(history, list)
            or len(history) != 5
            or [item.get("relative_time_s") for item in history]
            != [-0.8, -0.6, -0.4, -0.2, 0.0]
            or any(
                set(item) != {
                    "relative_time_s",
                    "image_path",
                    "image_sha256",
                }
                or not isinstance(item.get("image_path"), str)
                or "images" not in Path(item["image_path"]).parts
                for item in history
            )
        ):
            return False
    return True


def _flatten_label(
    label: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    known = np.asarray(label["known_target"], dtype=np.uint8)
    risk_object = np.asarray(label["risk_target_nullable"], dtype=object)
    if known.shape != (2, 6, 6) or risk_object.shape != (2, 6, 6):
        raise ValueError("Serialized label shape mismatch")
    numeric_mask = np.vectorize(lambda value: value is not None)(
        risk_object
    )
    if not np.array_equal(numeric_mask, known.astype(bool)):
        raise ValueError("UNKNOWN risk must be null and KNOWN risk numeric")
    risk = np.zeros((2, 6, 6), dtype=np.uint8)
    for index in np.argwhere(numeric_mask):
        position = tuple(int(value) for value in index)
        value = risk_object[position]
        if value not in (0, 1):
            raise ValueError("Known risk target must be binary")
        risk[position] = int(value)
    return known.astype(bool), risk


def _empty_summary() -> dict[str, Any]:
    return {
        horizon: {
            height: {
                "known": 0,
                "positive_known": 0,
                "negative_known": 0,
                "unknown": 0,
                "unknown_to_safe_violations": 0,
            }
            for height in HEIGHTS
        }
        for horizon in HORIZONS
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        session_id = str(record["session_id"])
        source = by_source.setdefault(
            session_id,
            {
                "role": record["role"],
                "teacher_view": record["teacher_view"],
                "record_count": 0,
                "horizons": _empty_summary(),
            },
        )
        if (
            source["role"] != record["role"]
            or source["teacher_view"] != record["teacher_view"]
        ):
            raise ValueError("Source role/view changed within corpus")
        source["record_count"] += 1
        for horizon in HORIZONS:
            known, risk = _flatten_label(record["labels"][horizon])
            for height_index, height in enumerate(HEIGHTS):
                height_known = known[height_index]
                height_risk = risk[height_index]
                metrics = source["horizons"][horizon][height]
                metrics["known"] += int(height_known.sum())
                metrics["positive_known"] += int(
                    (height_known & (height_risk == 1)).sum()
                )
                metrics["negative_known"] += int(
                    (height_known & (height_risk == 0)).sum()
                )
                metrics["unknown"] += int((~height_known).sum())
    for source in by_source.values():
        denominator = source["record_count"] * 36
        source["denominator_per_height_per_horizon"] = denominator
        for horizon in HORIZONS:
            for height in HEIGHTS:
                metrics = source["horizons"][horizon][height]
                metrics["known_coverage"] = (
                    metrics["known"] / denominator
                )
    return by_source


def _validate_summary_against_opportunity(
    aggregate: dict[str, Any],
    opportunity_sources: list[dict[str, Any]],
) -> None:
    expected_ids = [
        str(source["session_id"])
        for source in opportunity_sources
        if source["role"] in ("train", "dev")
    ]
    if list(aggregate) != expected_ids:
        raise ValueError("Materialized source order differs from opportunity")
    for expected in opportunity_sources:
        if expected["role"] not in ("train", "dev"):
            continue
        session_id = str(expected["session_id"])
        actual = aggregate[session_id]
        view = str(expected["role_gate_view"])
        expected_horizons = expected["teacher_views"][view]["horizons"]
        if (
            actual["role"] != expected["role"]
            or actual["teacher_view"] != view
            or actual["record_count"] != expected["usable_anchor_count"]
            or actual["denominator_per_height_per_horizon"]
            != expected["frozen_denominator_per_height_per_horizon"]
        ):
            raise ValueError(f"{session_id}: corpus source contract mismatch")
        for horizon in HORIZONS:
            for height in HEIGHTS:
                expected_metrics = expected_horizons[horizon][height]
                actual_metrics = actual["horizons"][horizon][height]
                for key in (
                    "known",
                    "positive_known",
                    "negative_known",
                    "unknown",
                    "unknown_to_safe_violations",
                    "denominator",
                ):
                    expected_value = (
                        actual["denominator_per_height_per_horizon"]
                        if key == "denominator"
                        else actual_metrics.get(key)
                    )
                    if expected_metrics[key] != expected_value:
                        raise ValueError(
                            f"{session_id}:{horizon}:{height}:{key} "
                            "differs from opportunity"
                        )


def _validate_inputs(
    contract_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    execution_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    expected = (
        (contract_path, CONTRACT_SHA256),
        (f0_path, F0_SHA256),
        (f0_1_path, F0_1_SHA256),
        (execution_path, EXECUTION_SHA256),
        (mechanics_path, MECHANICS_SHA256),
        (source_lock_path, SOURCE_LOCK_SHA256),
        (opportunity_path, OPPORTUNITY_SHA256),
    )
    failures = [
        str(path)
        for path, digest in expected
        if _sha256(path) != digest
    ]
    if failures:
        raise ValueError(
            f"Frozen corpus input hash mismatch: {','.join(failures)}"
        )
    contract = _load_json(contract_path)
    f0 = _load_json(f0_path)
    mechanics = _load_json(mechanics_path)
    source_lock = _load_json(source_lock_path)
    opportunity = _load_json(opportunity_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_CELL_CORPUS_MATERIALIZATION"
        or opportunity.get("terminal") != OPPORTUNITY_READY
        or opportunity.get("determinism_check", {}).get(
            "canonical_payload_byte_exact"
        )
        is not True
        or opportunity.get("next_step_authorization", {}).get(
            "train_candidate_corpus_materialization_authorized"
        )
        is not True
        or opportunity.get("next_step_authorization", {}).get(
            "dev_reference_target_materialization_authorized"
        )
        is not True
        or opportunity.get("next_step_authorization", {}).get(
            "heldout_training_corpus_materialization_authorized"
        )
        is not False
        or opportunity.get("next_step_authorization", {}).get(
            "heldout_reference_target_materialization_authorized_before_frozen_checkpoint"
        )
        is not False
    ):
        raise ValueError("Frozen corpus authorization mismatch")
    return contract, f0, mechanics, source_lock, opportunity


def _materialize_source(
    source: dict[str, Any],
    opportunity_source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    f0: dict[str, Any],
    mechanics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role = str(source["role"])
    if role not in ("train", "dev"):
        raise ValueError("Only train/dev sources may be materialized")
    view = "candidate" if role == "train" else "reference"
    if opportunity_source.get("role_gate_view") != view:
        raise ValueError("Opportunity role view mismatch")
    root = (datasets_root / _root_name(source)).resolve()
    authority_path = (
        authority_root / str(source["session_id"])[:8] / "authority.json"
    ).resolve()
    rows = _load_jsonl(root / "manifest.replay.jsonl")
    spec = _load_json(root / "dataset_spec.json")
    authority = _load_json(authority_path)
    if (
        _sha256(root / "manifest.replay.jsonl")
        != opportunity_source["manifest_sha256"]
        or _sha256(root / "dataset_spec.json")
        != opportunity_source["dataset_spec_sha256"]
        or _sha256(root / "source_metadata/camera_poses.csv")
        != opportunity_source["camera_poses_sha256"]
        or _sha256(authority_path)
        != opportunity_source["authority_report_sha256"]
    ):
        raise ValueError("Corpus source differs from opportunity input")
    binding_by_id = {
        str(item["manifest_id"]): item
        for item in authority["source_pose_authority"]["bindings"]
    }
    plane_by_id = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    timeline = _timeline_contract(float(source["target_fps"]))
    if timeline != opportunity_source["timeline"]:
        raise ValueError("Corpus timeline differs from opportunity")
    camera = spec["camera"]
    field = f0["field_contract"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(field["distance_edges_m"], dtype=np.float64)
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][height])
        for height in HEIGHTS
    ]
    widths = np.asarray(
        [
            mechanics["standard_synthetic_envelope"][
                "effective_lateral_half_width_m"
            ][height]
            for height in HEIGHTS
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    stride, offset = (8, 4) if view == "candidate" else (4, 2)
    required_indices = sorted(
        {
            index
            for anchor in timeline["usable_anchor_indices"]
            for index in (anchor, anchor + timeline["future_offset"])
        }
    )

    @functools.lru_cache(maxsize=8)
    def observation(index: int) -> tuple[np.ndarray, np.ndarray]:
        row = rows[index]
        return (
            _read_depth(
                _resolve_inside(root, str(row["source_depth_path"])),
                int(row["width"]),
                int(row["height"]),
            ),
            _read_semantic_class(
                _resolve_inside(root, str(row["source_mask_path"])),
                int(row["width"]),
                int(row["height"]),
            ),
        )

    points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in required_indices:
        row = rows[index]
        depth, semantic = observation(index)
        points[index] = _obstacle_points_world(
            root,
            row,
            binding_by_id[str(row["id"])],
            camera,
            stride=stride,
            offset=offset,
            excluded_classes=set(obstacle["excluded_semantic_class_ids"]),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
    observation.cache_clear()
    image_hash_cache: dict[Path, str] = {}
    student_records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    for anchor_index in timeline["usable_anchor_indices"]:
        history_indices = [
            anchor_index + value for value in timeline["history_offsets"]
        ]
        history_velocity_index = (
            anchor_index + timeline["velocity_history_offset"]
        )
        future_index = anchor_index + timeline["future_offset"]
        anchor_row = rows[anchor_index]
        history_row = rows[history_velocity_index]
        future_row = rows[future_index]
        anchor_binding = binding_by_id[str(anchor_row["id"])]
        future_binding = binding_by_id[str(future_row["id"])]
        current_basis, future_basis, velocity = _causal_future_basis(
            binding_by_id[str(history_row["id"])],
            anchor_binding,
            plane_by_id[str(anchor_row["id"])],
        )
        current_probes = _swept_prism_probes_world(
            current_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        future_probes = _swept_prism_probes_world(
            future_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        anchor_depth, anchor_semantic = observation(anchor_index)
        future_depth, future_semantic = observation(future_index)
        current_passing = _probe_passes(
            current_probes,
            anchor_row,
            anchor_binding,
            camera,
            anchor_depth,
            anchor_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        anchor_future_passing = _probe_passes(
            future_probes,
            anchor_row,
            anchor_binding,
            camera,
            anchor_depth,
            anchor_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        observed_future_passing = _probe_passes(
            future_probes,
            future_row,
            future_binding,
            camera,
            future_depth,
            future_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        current_known = (current_passing.sum(axis=1) >= 5).reshape(
            (6, 6, 2)
        )
        anchor_points, anchor_dynamic = points[anchor_index]
        future_points, future_dynamic = points[future_index]
        current_support, _ = _swept_prism_counts(
            anchor_points,
            anchor_dynamic,
            current_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        anchor_future_support, _ = _swept_prism_counts(
            anchor_points,
            anchor_dynamic,
            future_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        observed_future_support, _ = _swept_prism_counts(
            future_points,
            future_dynamic,
            future_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        future_support, future_known = _union_support(
            anchor_future_support,
            observed_future_support,
            anchor_future_passing,
            observed_future_passing,
            (6, 6, 2),
        )
        history_rgb = []
        for relative_time, history_index in zip(
            (-0.8, -0.6, -0.4, -0.2, 0.0),
            history_indices,
            strict=True,
        ):
            row = rows[history_index]
            image_path = _resolve_inside(root, str(row["image_path"]))
            digest = image_hash_cache.get(image_path)
            if digest is None:
                digest = _sha256(image_path)
                image_hash_cache[image_path] = digest
            if digest != row["image_sha256"]:
                raise ValueError("History RGB hash mismatch")
            history_rgb.append(
                {
                    "relative_time_s": relative_time,
                    "image_path": str(image_path),
                    "image_sha256": digest,
                }
            )
        sample_id = (
            f"hftf_f0_1_{role}_{source['session_id']}_"
            f"{anchor_index:02d}"
        )
        student_record = {
            "schema": "blindassist_hftf_f0_1_student_sample",
            "sample_id": sample_id,
            "role": role,
            "teacher_view": view,
            "session_id": source["session_id"],
            "anchor_timeline_index": anchor_index,
            "anchor_source_frame_index": anchor_row["source_frame_index"],
            "target_fps": source["target_fps"],
            "history_rgb": history_rgb,
            "labels": {
                "current": _nested_nullable_targets(
                    current_known, current_support
                ),
                "future": _nested_nullable_targets(
                    future_known, future_support
                ),
            },
        }
        if not _student_record_firewall(student_record):
            raise ValueError("Student record contains teacher-only feature")
        receipt_record = {
            "schema": "blindassist_hftf_f0_1_teacher_receipt",
            "sample_id": sample_id,
            "role": role,
            "teacher_view": view,
            "session_id": source["session_id"],
            "anchor_manifest_id": anchor_row["id"],
            "future_manifest_id": future_row["id"],
            "anchor_source_frame_index": anchor_row["source_frame_index"],
            "future_source_frame_index": future_row["source_frame_index"],
            "anchor_depth_sha256": anchor_row["source_depth_sha256"],
            "anchor_mask_sha256": anchor_row["source_mask_sha256"],
            "future_depth_sha256": future_row["source_depth_sha256"],
            "future_mask_sha256": future_row["source_mask_sha256"],
            "camera_poses_sha256": opportunity_source[
                "camera_poses_sha256"
            ],
            "history_velocity_source_frame_indices": [
                rows[history_velocity_index]["source_frame_index"],
                anchor_row["source_frame_index"],
            ],
            "causal_future_origin_m": future_basis[0].tolist(),
            "anchor_forward": current_basis[1].tolist(),
            "anchor_right": current_basis[2].tolist(),
            "anchor_up": current_basis[3].tolist(),
            "history_only_tangent_velocity_mps": velocity.tolist(),
            "student_loader_authorized": False,
        }
        student_records.append(student_record)
        receipt_records.append(receipt_record)
    return student_records, receipt_records


def _scientific_payload(
    contract_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    execution_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    contract, f0, mechanics, source_lock, opportunity = _validate_inputs(
        contract_path,
        f0_path,
        f0_1_path,
        execution_path,
        mechanics_path,
        source_lock_path,
        opportunity_path,
    )
    source_by_id = {
        str(item["session_id"]): item
        for item in source_lock["sources"]
    }
    opportunity_sources = opportunity["source_results"]
    selected = [
        item
        for item in opportunity_sources
        if item["role"] in ("train", "dev")
    ]
    if (
        len(selected) != 9
        or [item["role"] for item in selected]
        != ["train"] * 6 + ["dev"] * 3
    ):
        raise ValueError("Exact train/dev source set required")
    student_records: list[dict[str, Any]] = []
    teacher_receipts: list[dict[str, Any]] = []
    for opportunity_source in selected:
        source = source_by_id[str(opportunity_source["session_id"])]
        records, receipts = _materialize_source(
            source,
            opportunity_source,
            datasets_root,
            authority_root,
            f0,
            mechanics,
        )
        student_records.extend(records)
        teacher_receipts.extend(receipts)
    if (
        len(student_records) != 129
        or len(teacher_receipts) != 129
        or sum(record["role"] == "train" for record in student_records)
        != 90
        or sum(record["role"] == "dev" for record in student_records)
        != 39
        or any(record["role"] == "heldout" for record in student_records)
        or [record["sample_id"] for record in student_records]
        != [record["sample_id"] for record in teacher_receipts]
        or not all(
            _student_record_firewall(record)
            for record in student_records
        )
    ):
        raise ValueError("Corpus record count/firewall mismatch")
    aggregate = _aggregate_records(student_records)
    _validate_summary_against_opportunity(aggregate, opportunity_sources)
    student_bytes = _canonical_jsonl(student_records)
    receipt_bytes = _canonical_jsonl(teacher_receipts)
    summary = {
        "schema": SCHEMA,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SYNTHETIC_BODY_HEAD_GEOMETRY_PROXY_ONLY",
        "contract_sha256": _sha256(contract_path),
        "f0_protocol_sha256": _sha256(f0_path),
        "f0_1_protocol_sha256": _sha256(f0_1_path),
        "teacher_execution_contract_sha256": _sha256(execution_path),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "source_lock_sha256": _sha256(source_lock_path),
        "teacher_opportunity_sha256": _sha256(opportunity_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "student_record_count": len(student_records),
        "teacher_receipt_count": len(teacher_receipts),
        "role_record_counts": {"train": 90, "dev": 39, "heldout": 0},
        "history_rgb_reference_count": sum(
            len(record["history_rgb"]) for record in student_records
        ),
        "source_aggregates": aggregate,
        "checks": {
            "exact_train_dev_sources_and_counts": True,
            "student_teacher_ids_one_to_one": True,
            "history_rgb_hashes_match": True,
            "labels_reaggregate_to_opportunity": True,
            "unknown_risk_targets_are_null": True,
            "student_records_exclude_teacher_only_features": True,
            "heldout_record_count_zero": True,
        },
        "firewall": {
            "teacher_receipts_quarantined_from_student_loader": True,
            "heldout_target_materialized": False,
            "heldout_student_output_read": False,
            "student_training_or_output_computed": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
    }
    return student_bytes, receipt_bytes, summary


def materialize(
    contract_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    execution_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
    datasets_root: Path,
    authority_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    args = (
        contract_path,
        f0_path,
        f0_1_path,
        execution_path,
        mechanics_path,
        source_lock_path,
        opportunity_path,
        datasets_root,
        authority_root,
    )
    first_student, first_receipts, first_summary = _scientific_payload(*args)
    second_student, second_receipts, second_summary = _scientific_payload(*args)
    deterministic = (
        first_student == second_student
        and first_receipts == second_receipts
        and _canonical_bytes(first_summary) == _canonical_bytes(second_summary)
    )
    ready = deterministic and all(first_summary["checks"].values())
    first_summary["student_samples_sha256"] = _sha256_bytes(first_student)
    first_summary["teacher_receipts_sha256"] = _sha256_bytes(first_receipts)
    first_summary["determinism_check"] = {
        "all_payload_files_byte_exact": deterministic
    }
    first_summary["terminal"] = READY if ready else NOT_EVALUABLE
    first_summary["authorization"] = {
        "train_samples_available_to_training_loader": ready,
        "dev_samples_available_to_evaluation_loader": ready,
        "student_training_authorized_before_separate_validation": False,
        "heldout_target_materialization_authorized": False,
        "heldout_student_output_authorized": False,
    }
    spec_bytes = (
        json.dumps(
            first_summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite corpus: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    (partial_root / "student_samples.jsonl").write_bytes(first_student)
    (partial_root / "teacher_receipts.jsonl").write_bytes(first_receipts)
    (partial_root / "dataset_spec.json").write_bytes(spec_bytes)
    partial_root.replace(output_root)
    return first_summary


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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--f0-protocol", type=Path, required=True)
    parser.add_argument("--f0-1-protocol", type=Path, required=True)
    parser.add_argument("--teacher-execution-contract", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--teacher-opportunity", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_root = _require_artifacts_output(args.output_root)
        report = materialize(
            args.contract.resolve(),
            args.f0_protocol.resolve(),
            args.f0_1_protocol.resolve(),
            args.teacher_execution_contract.resolve(),
            args.mechanics_protocol.resolve(),
            args.source_lock.resolve(),
            args.teacher_opportunity.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
            output_root,
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "student_record_count": report["student_record_count"],
                    "role_record_counts": report["role_record_counts"],
                    "student_samples_sha256": report[
                        "student_samples_sha256"
                    ],
                    "teacher_receipts_sha256": report[
                        "teacher_receipts_sha256"
                    ],
                    "determinism_check": report["determinism_check"],
                    "output_root": str(output_root),
                }
            )
        )
        return 0 if report["terminal"] == READY else 3
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
