#!/usr/bin/env python3
"""Materialize the one-shot G0-D1 current-only fresh evaluation package."""

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


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "fresh_execution_contract_d1"
)
CONTRACT_STATUS = (
    "FROZEN_BEFORE_D1_FRESH_SOURCE_OPENING_OR_PREDICTION"
)
SOURCE_PLAN_SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
)
SOURCE_PLAN_TERMINAL = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_F0_1_STOP_BEFORE_G0_CLEARANCE_OR_SOURCE_SCAN_OUTCOME"
)
PACKAGE_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_package"
INPUT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_input"
)
TRUTH_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth"
RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_teacher_receipt"
)
READY = "G0_D1_FRESH_PACKAGE_READY_FOR_ONE_SHOT_PREDICTION"
NOT_EVALUABLE = (
    "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
)
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/"
    "materialize_stage_c_g0_d1_fresh_package.py"
)
DEPENDENCY_RECEIPTS = {
    "teacher_opportunity_module": (
        "scripts/research/hftf/"
        "audit_stage_c_f0_1_teacher_opportunity.py"
    ),
    "swept_envelope_module": (
        "scripts/research/hftf/audit_swept_envelope_label_mechanics.py"
    ),
    "geometry_teacher_module": (
        "scripts/research/hftf/run_geometry_teacher_canary.py"
    ),
    "signed_clearance_module": (
        "scripts/research/hftf/"
        "run_stage_c_g0_signed_clearance_mechanics.py"
    ),
    "fresh_source_authority_verifier": (
        "scripts/research/hftf/"
        "verify_sanpo_pose_geometry_authority.py"
    ),
}
HEIGHTS = ("body", "head")
EXPECTED_SOURCE_IDS = (
    "15bc9dde1b9b54b6c109cb2ac4433f210fee71f800e1ae7bde9626913c3e02bf",
    "15d83b42fa73c1282c2d02fbaa486258216f59b66dea1eb490a1b1ea4cc9200f",
    "16401349eec7f73fecf3811da750a00821a899dcea6671fbe4fd3562a1f98de9",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_receipt(owner_path: Path, receipt: dict[str, Any]) -> Path:
    raw = Path(str(receipt["path"]))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (_repository_root() / raw).resolve()
    return (owner_path.parent / raw).resolve()


def _bound_parent(
    contract_path: Path,
    contract: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = contract.get("parents", {}).get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"Missing fresh parent receipt: {key}")
    path = _resolve_receipt(contract_path, receipt)
    if not path.is_file() or _sha256(path) != receipt.get("sha256"):
        raise ValueError(f"Fresh parent receipt mismatch: {key}")
    return path, _load_json(path)


def _implementation_receipt(contract: dict[str, Any]) -> None:
    receipt = contract.get("implementations", {}).get(
        "fresh_package_materializer"
    )
    if (
        not isinstance(receipt, dict)
        or Path(str(receipt.get("path", ""))).as_posix()
        != IMPLEMENTATION_PATH
        or receipt.get("sha256") != _sha256(Path(__file__).resolve())
        or receipt.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh materializer receipt mismatch or unauthorized")
    for key, relative in DEPENDENCY_RECEIPTS.items():
        dependency = contract.get("implementations", {}).get(key)
        path = (_repository_root() / relative).resolve()
        if (
            not isinstance(dependency, dict)
            or dependency.get("path") != relative
            or dependency.get("sha256") != _sha256(path)
            or dependency.get("execution_authorized") is not True
        ):
            raise ValueError(
                f"Fresh materializer dependency receipt mismatch: {key}"
            )


def _canonical_root(contract: dict[str, Any], output_root: Path) -> None:
    canonical = contract.get("canonical_artifacts", {})
    relative = canonical.get("fresh_package_root")
    expected_payloads = {
        "prediction_inputs": (
            "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/"
            "prediction_inputs.jsonl"
        ),
        "truth_labels": (
            "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl"
        ),
        "teacher_receipts": (
            "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/"
            "teacher_receipts.jsonl"
        ),
    }
    expected = (_repository_root() / str(relative)).resolve()
    if (
        relative
        != "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-20260801"
        or output_root.resolve() != expected
        or any(
            canonical.get(key) != payload
            or (output_root / Path(payload).name).resolve()
            != (_repository_root() / payload).resolve()
            for key, payload in expected_payloads.items()
        )
    ):
        raise ValueError("Fresh package output path is not canonical")


def _select_sources(
    source_plan: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    roles = source_plan.get("roles", {})
    sources = roles.get("one_shot_fresh_evaluation", [])
    contract_sources = contract.get("fresh_source_contract", {}).get(
        "sources", []
    )
    order = contract.get("fresh_source_contract", {}).get("source_order")
    if (
        source_plan.get("schema") != SOURCE_PLAN_SCHEMA
        or source_plan.get("terminal") != SOURCE_PLAN_TERMINAL
        or tuple(order or ()) != EXPECTED_SOURCE_IDS
        or len(sources) != 3
        or len(contract_sources) != 3
        or tuple(str(item.get("session_id")) for item in sources)
        != EXPECTED_SOURCE_IDS
        or tuple(str(item.get("session_id")) for item in contract_sources)
        != EXPECTED_SOURCE_IDS
    ):
        raise ValueError("Exact frozen one-shot fresh source order required")
    for planned, binding in zip(sources, contract_sources, strict=True):
        if (
            planned.get("role") != "fresh_evaluation"
            or planned.get("official_split") != "train"
            or planned.get("g0_source_role")
            != "one_shot_fresh_evaluation_metadata_planned_only"
            or planned.get("media_geometry_teacher_or_student_outcome_open")
            is not False
            or planned.get("fresh_evidence_obtained") is not False
            or planned.get("selected_source_frames")
            != list(range(0, 50, 2))
            or float(planned.get("source_fps", -1)) != 20.0
            or float(planned.get("target_fps", -1)) != 10.0
            or set(binding) != {"session_id"}
            or binding.get("session_id") != planned.get("session_id")
        ):
            raise ValueError("Frozen fresh role or source binding drifted")
    return [
        {
            **planned,
            "source_binding": binding,
            "authority_verifier_sha256": contract["implementations"][
                "fresh_source_authority_verifier"
            ]["sha256"],
        }
        for planned, binding in zip(sources, contract_sources, strict=True)
    ]


def _nullable_labels(
    known: np.ndarray,
    support: np.ndarray,
    clearance: np.ndarray,
) -> dict[str, Any]:
    if (
        known.shape != (6, 6, 2)
        or support.shape != (6, 6, 2)
        or clearance.shape != (6, 6, 2)
        or known.dtype != np.bool_
        or np.any(support < 0)
    ):
        raise ValueError("Fresh teacher arrays must be 6x6x2")
    risk = support >= 2
    if (
        np.any(~np.isfinite(clearance[known]))
        or np.any(risk[known] != (clearance[known] < 0.0))
    ):
        raise ValueError("Fresh risk and clearance semantics disagree")
    known_t = known.transpose(2, 0, 1)
    risk_t = risk.transpose(2, 0, 1)
    clearance_t = clearance.transpose(2, 0, 1)
    nullable_risk: list[Any] = []
    nullable_clearance: list[Any] = []
    for height in range(2):
        risk_rows: list[Any] = []
        clearance_rows: list[Any] = []
        for theta in range(6):
            risk_row: list[int | None] = []
            clearance_row: list[float | None] = []
            for distance in range(6):
                index = (height, theta, distance)
                if known_t[index]:
                    risk_row.append(int(risk_t[index]))
                    clearance_row.append(float(clearance_t[index]))
                else:
                    risk_row.append(None)
                    clearance_row.append(None)
            risk_rows.append(risk_row)
            clearance_rows.append(clearance_row)
        nullable_risk.append(risk_rows)
        nullable_clearance.append(clearance_rows)
    return {
        "known_target": known_t.astype(np.uint8).tolist(),
        "risk_target_nullable": nullable_risk,
        "clearance_target_m_nullable": nullable_clearance,
    }


def _opportunity(
    truths: list[dict[str, Any]],
    source_order: tuple[str, ...],
) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {}
    adequate = True
    for session_id in source_order:
        rows = [row for row in truths if row["session_id"] == session_id]
        if len(rows) != 25:
            raise ValueError(f"{session_id}: expected exactly 25 fresh frames")
        source: dict[str, Any] = {}
        for height_index, height in enumerate(HEIGHTS):
            known = 0
            positive = 0
            negative = 0
            unknown_safe = 0
            for row in rows:
                labels = row["labels"]
                known_array = np.asarray(
                    labels["known_target"], dtype=np.uint8
                )[height_index]
                risk = np.asarray(
                    labels["risk_target_nullable"], dtype=object
                )[height_index]
                known += int(known_array.sum())
                for index in np.ndindex((6, 6)):
                    if known_array[index] == 1:
                        positive += int(risk[index] == 1)
                        negative += int(risk[index] == 0)
                    elif risk[index] is not None:
                        unknown_safe += 1
            denominator = 25 * 6 * 6
            coverage = known / denominator
            passed = (
                coverage >= 0.1
                and positive >= 5
                and negative >= 20
                and unknown_safe == 0
            )
            adequate = adequate and passed
            source[height] = {
                "frame_count": 25,
                "denominator": denominator,
                "known": known,
                "known_coverage": coverage,
                "positive_known": positive,
                "negative_known": negative,
                "unknown": denominator - known,
                "unknown_to_safe_violations": unknown_safe,
                "gate_pass": passed,
            }
        result[session_id] = source
    return result, adequate


def _load_context(
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("Fresh execution contract identity mismatch")
    _implementation_receipt(contract)
    _, source_plan = _bound_parent(
        contract_path, contract, "g0_source_plan"
    )
    _, protocol = _bound_parent(contract_path, contract, "g0_protocol")
    _, training = _bound_parent(
        contract_path, contract, "d1_training_validation"
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or training.get("terminal") != "G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN"
    ):
        raise ValueError("Fresh scientific or checkpoint parent drifted")
    field = protocol.get("field_contract", {})
    clearance = protocol.get("signed_clearance_contract", {})
    view = field.get("teacher_view_for_every_role", {})
    if (
        field.get("current_only") is not True
        or view.get("name") != "reference"
        or view.get("point_sample_stride_xy") != 4
        or view.get("point_sample_offset_xy") != 2
        or clearance.get("order_statistic") != 2
        or clearance.get("raw_clearance_clip_m") != [-0.5, 1.0]
    ):
        raise ValueError("Fresh current teacher contract drifted")
    return contract, protocol, source_plan, _select_sources(
        source_plan, contract
    )


def _materialize_source(
    source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session_id = str(source["session_id"])
    root = (datasets_root / _root_name(source)).resolve()
    manifest_path = root / "manifest.replay.jsonl"
    spec_path = root / "dataset_spec.json"
    poses_path = root / "source_metadata/camera_poses.csv"
    authority_path = authority_root / session_id[:8] / "authority.json"
    rows = _load_jsonl(manifest_path)
    spec = _load_json(spec_path)
    authority = _load_json(authority_path)
    inventory = spec.get("source_inventory", {})
    sampling = spec.get("sampling", {})
    source_identity = spec.get("source", {})
    actual_camera_poses_sha256 = _sha256(poses_path)
    actual_authority_sha256 = _sha256(authority_path)
    authority_inputs = authority.get("input_hashes", {})
    if (
        len(rows) != 25
        or [int(row["source_frame_index"]) for row in rows]
        != source["selected_source_frames"]
        or {str(row.get("session_id")) for row in rows} != {session_id}
        or source_identity.get("session_id") != session_id
        or source_identity.get("official_split") != "train"
        or sampling.get("selected_source_frames")
        != source["selected_source_frames"]
        or float(sampling.get("source_fps", -1)) != 20.0
        or float(sampling.get("target_fps", -1)) != 10.0
        or _canonical_bytes(inventory.get("description"))
        != _canonical_bytes(source.get("description_object"))
        or _canonical_bytes(inventory.get("camera_poses"))
        != _canonical_bytes(source.get("camera_poses_object"))
        or authority.get("terminal")
        != "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED"
        or authority.get("evaluation_mode")
        != "frozen_canonical_replication"
        or authority_inputs.get("manifest_sha256")
        != _sha256(manifest_path)
        or authority_inputs.get("dataset_spec_sha256")
        != _sha256(spec_path)
        or authority_inputs.get("camera_poses_sha256")
        != actual_camera_poses_sha256
        or authority_inputs.get("verifier_sha256")
        != source["authority_verifier_sha256"]
    ):
        raise ValueError(f"{session_id}: fresh source binding mismatch")
    bindings = {
        str(item["manifest_id"]): item
        for item in authority["source_pose_authority"]["bindings"]
    }
    planes = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    if set(bindings) != {str(row["id"]) for row in rows} or set(planes) != set(
        bindings
    ):
        raise ValueError(f"{session_id}: fresh authority coverage mismatch")
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
    distance_edges = np.asarray(field["distance_edges_m"], dtype=np.float64)
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][height])
        for height in HEIGHTS
    ]
    widths = np.asarray(
        [field["effective_lateral_half_width_m"][height] for height in HEIGHTS],
        dtype=np.float64,
    )
    obstacle = protocol["parents"]
    mechanics_path = _resolve_receipt(
        Path(
            protocol.get("_owner_path", _repository_root())
        ),
        obstacle["swept_envelope_mechanics"],
    )
    if (
        not mechanics_path.is_file()
        or _sha256(mechanics_path)
        != obstacle["swept_envelope_mechanics"].get("sha256")
    ):
        raise ValueError("Fresh swept-envelope mechanics receipt mismatch")
    mechanics = _load_json(mechanics_path)
    obstacle_contract = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    inputs: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for row in rows:
        manifest_id = str(row["id"])
        depth_path = _resolve_inside(root, str(row["source_depth_path"]))
        mask_path = _resolve_inside(root, str(row["source_mask_path"]))
        image_path = _resolve_inside(root, str(row["image_path"]))
        depth = _read_depth(
            depth_path, int(row["width"]), int(row["height"])
        )
        semantic = _read_semantic_class(
            mask_path, int(row["width"]), int(row["height"])
        )
        basis = _anchor_basis(bindings[manifest_id], planes[manifest_id])
        view = field["teacher_view_for_every_role"]
        points, dynamic = _obstacle_points_world(
            root,
            row,
            bindings[manifest_id],
            camera,
            stride=int(view["point_sample_stride_xy"]),
            offset=int(view["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle_contract["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle_contract["dynamic_provenance_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
        probes = _swept_prism_probes_world(
            basis, theta_edges, distance_edges, height_bands, widths
        )
        passing = _probe_passes(
            probes,
            row,
            bindings[manifest_id],
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
        _, clipped, inside = _signed_clearance_field(
            points,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
            order_statistic=int(clearance_contract["order_statistic"]),
            final_edge_atol_m=float(
                clearance_contract["final_distance_edge_isclose"]["atol_m"]
            ),
            final_edge_rtol=float(
                clearance_contract["final_distance_edge_isclose"]["rtol"]
            ),
            clip_min_m=float(clearance_contract["raw_clearance_clip_m"][0]),
            clip_max_m=float(clearance_contract["raw_clearance_clip_m"][1]),
        )
        if not np.array_equal(support, inside):
            raise ValueError(f"{session_id}:{manifest_id}: support mismatch")
        labels = _nullable_labels(known, support, clipped)
        if (
            _sha256(image_path) != row["image_sha256"]
            or _sha256(depth_path) != row["source_depth_sha256"]
            or _sha256(mask_path) != row["source_mask_sha256"]
        ):
            raise ValueError(f"{session_id}:{manifest_id}: media hash mismatch")
        frame = int(row["source_frame_index"])
        sample_id = f"hftf_g0_d1_fresh_{session_id}_{frame:06d}"
        inputs.append(
            {
                "schema": INPUT_SCHEMA,
                "sample_id": sample_id,
                "session_id": session_id,
                "source_frame_index": frame,
                "manifest_id": manifest_id,
                "current_rgb": {
                    "path": str(image_path),
                    "sha256": row["image_sha256"],
                },
            }
        )
        truths.append(
            {
                "schema": TRUTH_SCHEMA,
                "sample_id": sample_id,
                "session_id": session_id,
                "source_frame_index": frame,
                "manifest_id": manifest_id,
                "labels": labels,
            }
        )
        receipts.append(
            {
                "schema": RECEIPT_SCHEMA,
                "sample_id": sample_id,
                "session_id": session_id,
                "source_frame_index": frame,
                "manifest_id": manifest_id,
                "teacher_view": "REFERENCE_STRIDE4_OFFSET2_CURRENT",
                "source_depth_sha256": row["source_depth_sha256"],
                "source_mask_sha256": row["source_mask_sha256"],
                "camera_poses_sha256": actual_camera_poses_sha256,
                "authority_report_sha256": actual_authority_sha256,
                "labels_sha256": _sha256_bytes(_canonical_bytes(labels)),
                "student_loader_authorized": False,
            }
        )
    return inputs, truths, receipts


def _scientific_payload(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    contract, protocol, _, sources = _load_context(contract_path)
    protocol["_owner_path"] = str(
        _resolve_receipt(
            contract_path, contract["parents"]["g0_protocol"]
        )
    )
    inputs: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for source in sources:
        source_inputs, source_truths, source_receipts = _materialize_source(
            source, datasets_root, authority_root, protocol
        )
        inputs.extend(source_inputs)
        truths.extend(source_truths)
        receipts.extend(source_receipts)
    sample_ids = [row["sample_id"] for row in inputs]
    if (
        len(inputs) != 75
        or len(truths) != 75
        or len(receipts) != 75
        or len(set(sample_ids)) != 75
        or sample_ids != [row["sample_id"] for row in truths]
        or sample_ids != [row["sample_id"] for row in receipts]
    ):
        raise ValueError("Fresh package identity/cardinality mismatch")
    source_order = tuple(str(source["session_id"]) for source in sources)
    opportunity, adequate = _opportunity(truths, source_order)
    input_bytes = _canonical_jsonl(inputs)
    truth_bytes = _canonical_jsonl(truths)
    receipt_bytes = _canonical_jsonl(receipts)
    package = {
        "schema": PACKAGE_SCHEMA,
        "terminal": READY if adequate else NOT_EVALUABLE,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "ONE_SHOT_FRESH_SYNTHETIC_PROXY_PACKAGE",
        "claim_ceiling": "SYNTHETIC_SIGNED_CLEARANCE_PROXY_ONLY",
        "contract_sha256": _sha256(contract_path),
        "materializer_sha256": _sha256(Path(__file__).resolve()),
        "source_order": list(source_order),
        "source_count": 3,
        "record_count": 75,
        "opportunity_gate": opportunity,
        "files": {
            "prediction_inputs.jsonl": {
                "sha256": _sha256_bytes(input_bytes),
                "record_count": 75,
                "student_loader_authorized": True,
            },
            "truth_labels.jsonl": {
                "sha256": _sha256_bytes(truth_bytes),
                "record_count": 75,
                "student_loader_authorized": False,
            },
            "teacher_receipts.jsonl": {
                "sha256": _sha256_bytes(receipt_bytes),
                "record_count": 75,
                "student_loader_authorized": False,
            },
        },
        "authorization": {
            "one_shot_prediction_authorized_after_independent_validation": (
                adequate
            ),
            "truth_join_before_predictions_frozen": False,
            "source_replacement_or_package_rematerialization": False,
            "reserved_heldout_opening": False,
            "mainline_or_safety_claim": False,
        },
    }
    return input_bytes, truth_bytes, receipt_bytes, package


def _atomic_publish(
    output_root: Path,
    input_bytes: bytes,
    truth_bytes: bytes,
    receipt_bytes: bytes,
    package: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError("Refusing to overwrite one-shot fresh package")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-", dir=output_root.parent
        )
    )
    try:
        payloads = {
            "prediction_inputs.jsonl": input_bytes,
            "truth_labels.jsonl": truth_bytes,
            "teacher_receipts.jsonl": receipt_bytes,
            "package.json": json.dumps(
                package, indent=2, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            + b"\n",
        }
        for name, payload in payloads.items():
            with (partial / name).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        partial.replace(output_root)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def _execution_root(contract: dict[str, Any]) -> Path:
    relative = contract.get("canonical_artifacts", {}).get(
        "fresh_package_execution_root"
    )
    expected_relative = (
        "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-execution-20260801"
    )
    if relative != expected_relative:
        raise ValueError("Fresh package execution root is not canonical")
    return (_repository_root() / relative).resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.partial-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _start_execution(
    contract_path: Path,
    contract: dict[str, Any],
) -> Path:
    execution_root = _execution_root(contract)
    if execution_root.exists():
        raise FileExistsError(
            "Fresh package materialization route is already consumed"
        )
    execution_root.mkdir(parents=True)
    try:
        _atomic_json(
            execution_root / "execution_receipt.json",
            {
                "schema": (
                    "blindassist_hftf_stage_c_g0_d1_"
                    "fresh_package_execution_receipt"
                ),
                "status": (
                    "STARTED_BEFORE_FIRST_FRESH_PACKAGE_SOURCE_OR_MEDIA_READ"
                ),
                "contract_sha256": _sha256(contract_path),
                "materializer_sha256": _sha256(
                    Path(__file__).resolve()
                ),
                "source_order": list(EXPECTED_SOURCE_IDS),
                "source_replacement_or_rematerialization_authorized": False,
                "reserved_heldout_opening_authorized": False,
            },
        )
    except BaseException:
        if execution_root.exists() and not any(execution_root.iterdir()):
            execution_root.rmdir()
        raise
    return execution_root


def materialize(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    _canonical_root(contract, output_root)
    _implementation_receipt(contract)
    execution_root = _start_execution(contract_path, contract)
    try:
        input_bytes, truth_bytes, receipt_bytes, package = (
            _scientific_payload(
                contract_path, datasets_root, authority_root
            )
        )
        package["materialization_execution_receipt_sha256"] = _sha256(
            execution_root / "execution_receipt.json"
        )
        _atomic_publish(
            output_root, input_bytes, truth_bytes, receipt_bytes, package
        )
        _atomic_json(
            execution_root / "completion.json",
            {
                "schema": (
                    "blindassist_hftf_stage_c_g0_d1_"
                    "fresh_package_execution_completion"
                ),
                "terminal": package["terminal"],
                "contract_sha256": _sha256(contract_path),
                "execution_receipt_sha256": _sha256(
                    execution_root / "execution_receipt.json"
                ),
                "package_manifest_sha256": _sha256(
                    output_root / "package.json"
                ),
                "rematerialization_authorized": False,
            },
        )
        return package
    except BaseException as error:
        try:
            _atomic_json(
                execution_root / "failure.json",
                {
                    "schema": (
                        "blindassist_hftf_stage_c_g0_d1_"
                        "fresh_package_execution_failure"
                    ),
                    "terminal": NOT_EVALUABLE,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fresh_route_consumed": True,
                    "rematerialization_authorized": False,
                    "source_replacement_authorized": False,
                },
            )
        finally:
            raise


def main() -> int:
    repository = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=repository / "artifacts.local/evidence/datasets",
    )
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801"
        ),
    )
    args = parser.parse_args()
    try:
        package = materialize(
            args.contract.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
            args.output_root.resolve(),
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": PACKAGE_SCHEMA,
                    "terminal": NOT_EVALUABLE,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "output_written": False,
                    "source_replacement_authorized": False,
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(package, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
