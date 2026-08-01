#!/usr/bin/env python3
"""Independently validate and rederive the G0-D1 one-shot fresh package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
PACKAGE_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_package"
INPUT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_input"
)
TRUTH_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth"
RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_teacher_receipt"
)
VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_package_validation"
)
PREDICTION_AUTHORIZATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_authorization"
)
PACKAGE_READY = "G0_D1_FRESH_PACKAGE_READY_FOR_ONE_SHOT_PREDICTION"
PACKAGE_NOT_EVALUABLE = (
    "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
)
READY = "G0_D1_FRESH_PACKAGE_VALIDATED_AND_OPPORTUNITY_ADEQUATE"
NOT_EVALUABLE = (
    "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
)
ERROR = NOT_EVALUABLE
PREDICTION_AUTHORIZATION_READY = (
    "G0_D1_FRESH_PREDICTION_AUTHORIZATION_READY"
)
PREDICTION_AUTHORIZATION_NOT_EVALUABLE = NOT_EVALUABLE
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/"
    "validate_stage_c_g0_d1_fresh_package.py"
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
EXPECTED_SOURCE_IDS = (
    "15bc9dde1b9b54b6c109cb2ac4433f210fee71f800e1ae7bde9626913c3e02bf",
    "15d83b42fa73c1282c2d02fbaa486258216f59b66dea1eb490a1b1ea4cc9200f",
    "16401349eec7f73fecf3811da750a00821a899dcea6671fbe4fd3562a1f98de9",
)
INPUT_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "current_rgb",
}
TRUTH_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "labels",
}
RECEIPT_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "teacher_view",
    "source_depth_sha256",
    "source_mask_sha256",
    "camera_poses_sha256",
    "authority_report_sha256",
    "labels_sha256",
    "student_loader_authorized",
}


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
        raise ValueError(f"Missing fresh validation parent: {key}")
    path = _resolve_receipt(contract_path, receipt)
    if not path.is_file() or _sha256(path) != receipt.get("sha256"):
        raise ValueError(f"Fresh validation parent mismatch: {key}")
    return path, _load_json(path)


def _implementation_receipt(contract: dict[str, Any]) -> None:
    receipt = contract.get("implementations", {}).get(
        "fresh_package_validator"
    )
    if (
        not isinstance(receipt, dict)
        or Path(str(receipt.get("path", ""))).as_posix()
        != IMPLEMENTATION_PATH
        or receipt.get("sha256") != _sha256(Path(__file__).resolve())
        or receipt.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh validator receipt mismatch or unauthorized")
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
                f"Fresh validator dependency receipt mismatch: {key}"
            )


def _canonical_paths(
    contract: dict[str, Any],
    package_root: Path,
    output_root: Path | None = None,
) -> None:
    canonical = contract.get("canonical_artifacts", {})
    package_relative = canonical.get("fresh_package_root")
    validation_relative = canonical.get("fresh_package_validation")
    prediction_authorization_relative = canonical.get(
        "fresh_prediction_authorization"
    )
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
    if (
        package_relative
        != "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-20260801"
        or package_root.resolve()
        != (_repository_root() / str(package_relative)).resolve()
        or any(
            canonical.get(key) != relative
            or (package_root / Path(relative).name).resolve()
            != (_repository_root() / relative).resolve()
            for key, relative in expected_payloads.items()
        )
    ):
        raise ValueError("Fresh package path is not canonical")
    if output_root is not None and (
        validation_relative
        != "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-validation-20260801/validation.json"
        or (output_root / "validation.json").resolve()
        != (_repository_root() / str(validation_relative)).resolve()
        or prediction_authorization_relative
        != "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-validation-20260801/"
        "prediction_authorization.json"
        or (output_root / "prediction_authorization.json").resolve()
        != (
            _repository_root()
            / str(prediction_authorization_relative)
        ).resolve()
    ):
        raise ValueError("Fresh package validation path is not canonical")


def _validate_materialization_execution(
    contract_path: Path,
    contract: dict[str, Any],
    package_root: Path,
    package: dict[str, Any],
) -> str:
    relative = contract.get("canonical_artifacts", {}).get(
        "fresh_package_execution_root"
    )
    expected_relative = (
        "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-execution-20260801"
    )
    execution_root = (_repository_root() / str(relative)).resolve()
    receipt_path = execution_root / "execution_receipt.json"
    completion_path = execution_root / "completion.json"
    if (
        relative != expected_relative
        or not execution_root.is_dir()
        or {path.name for path in execution_root.iterdir()}
        != {"execution_receipt.json", "completion.json"}
    ):
        raise ValueError("Fresh materialization execution set mismatch")
    receipt = _load_json(receipt_path)
    completion = _load_json(completion_path)
    receipt_sha256 = _sha256(receipt_path)
    if (
        receipt.get("schema")
        != (
            "blindassist_hftf_stage_c_g0_d1_"
            "fresh_package_execution_receipt"
        )
        or receipt.get("status")
        != "STARTED_BEFORE_FIRST_FRESH_PACKAGE_SOURCE_OR_MEDIA_READ"
        or receipt.get("contract_sha256") != _sha256(contract_path)
        or receipt.get("materializer_sha256")
        != contract["implementations"]["fresh_package_materializer"][
            "sha256"
        ]
        or receipt.get("source_order") != list(EXPECTED_SOURCE_IDS)
        or receipt.get(
            "source_replacement_or_rematerialization_authorized"
        )
        is not False
        or receipt.get("reserved_heldout_opening_authorized") is not False
        or completion.get("schema")
        != (
            "blindassist_hftf_stage_c_g0_d1_"
            "fresh_package_execution_completion"
        )
        or completion.get("terminal") != package.get("terminal")
        or completion.get("contract_sha256") != _sha256(contract_path)
        or completion.get("execution_receipt_sha256") != receipt_sha256
        or completion.get("package_manifest_sha256")
        != _sha256(package_root / "package.json")
        or completion.get("rematerialization_authorized") is not False
        or package.get("materialization_execution_receipt_sha256")
        != receipt_sha256
    ):
        raise ValueError("Fresh materialization execution receipt mismatch")
    return receipt_sha256


def _strict_labels(label: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(label, dict) or set(label) != {
        "known_target",
        "risk_target_nullable",
        "clearance_target_m_nullable",
    }:
        raise ValueError("Fresh truth label key set mismatch")
    known_object = np.asarray(label["known_target"], dtype=object)
    risk_object = np.asarray(label["risk_target_nullable"], dtype=object)
    clearance_object = np.asarray(
        label["clearance_target_m_nullable"], dtype=object
    )
    if (
        known_object.shape != (2, 6, 6)
        or risk_object.shape != (2, 6, 6)
        or clearance_object.shape != (2, 6, 6)
    ):
        raise ValueError("Fresh truth label shape mismatch")
    known = np.zeros((2, 6, 6), dtype=np.bool_)
    risk = np.zeros((2, 6, 6), dtype=np.uint8)
    clearance = np.full((2, 6, 6), np.nan, dtype=np.float64)
    for index in np.ndindex((2, 6, 6)):
        known_value = known_object[index]
        if type(known_value) is not int or known_value not in (0, 1):
            raise ValueError("Fresh known target must be exact JSON 0/1")
        known[index] = known_value == 1
        risk_value = risk_object[index]
        clearance_value = clearance_object[index]
        if not known[index]:
            if risk_value is not None or clearance_value is not None:
                raise ValueError("Fresh UNKNOWN targets must both be null")
            continue
        if (
            type(risk_value) is not int
            or risk_value not in (0, 1)
            or isinstance(clearance_value, bool)
            or not isinstance(clearance_value, (int, float))
            or not math.isfinite(float(clearance_value))
            or not -0.5 <= float(clearance_value) <= 1.0
            or (risk_value == 1) != (float(clearance_value) < 0.0)
        ):
            raise ValueError("Fresh known risk/clearance target invalid")
        risk[index] = risk_value
        clearance[index] = float(clearance_value)
    return known, risk, clearance


def _label_json(
    known: np.ndarray,
    support: np.ndarray,
    clearance: np.ndarray,
) -> dict[str, Any]:
    if (
        known.shape != (6, 6, 2)
        or support.shape != (6, 6, 2)
        or clearance.shape != (6, 6, 2)
        or np.any(support < 0)
    ):
        raise ValueError("Independently rederived teacher shape mismatch")
    risk = support >= 2
    if np.any(risk[known] != (clearance[known] < 0.0)):
        raise ValueError("Independently rederived target semantics disagree")
    known_t = known.transpose(2, 0, 1)
    risk_t = risk.transpose(2, 0, 1)
    clearance_t = clearance.transpose(2, 0, 1)
    risk_json: list[Any] = []
    clearance_json: list[Any] = []
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
        risk_json.append(risk_rows)
        clearance_json.append(clearance_rows)
    return {
        "known_target": known_t.astype(np.uint8).tolist(),
        "risk_target_nullable": risk_json,
        "clearance_target_m_nullable": clearance_json,
    }


def _validate_contract(
    contract_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("Fresh validation contract identity mismatch")
    _implementation_receipt(contract)
    protocol_path, protocol = _bound_parent(
        contract_path, contract, "g0_protocol"
    )
    _, source_plan = _bound_parent(
        contract_path, contract, "g0_source_plan"
    )
    _, training = _bound_parent(
        contract_path, contract, "d1_training_validation"
    )
    if (
        training.get("terminal") != "G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN"
        or source_plan.get("schema")
        != "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
        or source_plan.get("terminal")
        != "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
        or protocol.get("schema")
        != "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
    ):
        raise ValueError("Fresh validation scientific parent drifted")
    planned = source_plan.get("roles", {}).get(
        "one_shot_fresh_evaluation", []
    )
    bindings = contract.get("fresh_source_contract", {}).get("sources", [])
    order = contract.get("fresh_source_contract", {}).get("source_order")
    if (
        tuple(order or ()) != EXPECTED_SOURCE_IDS
        or tuple(str(row.get("session_id")) for row in planned)
        != EXPECTED_SOURCE_IDS
        or tuple(str(row.get("session_id")) for row in bindings)
        != EXPECTED_SOURCE_IDS
        or any(set(row) != {"session_id"} for row in bindings)
    ):
        raise ValueError("Fresh validation source order drifted")
    return (
        contract,
        protocol_path,
        protocol,
        [
            {
                **source,
                "source_binding": binding,
                "authority_verifier_sha256": contract[
                    "implementations"
                ]["fresh_source_authority_verifier"]["sha256"],
            }
            for source, binding in zip(planned, bindings, strict=True)
        ],
    )


def _derive_source(
    source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    session_id = str(source["session_id"])
    expected_frames = list(range(0, 50, 2))
    if (
        source.get("role") != "fresh_evaluation"
        or source.get("official_split") != "train"
        or source.get("selected_source_frames") != expected_frames
    ):
        raise ValueError(f"{session_id}: fresh role/timeline drifted")
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
        != expected_frames
        or {str(row.get("session_id")) for row in rows} != {session_id}
        or source_identity.get("session_id") != session_id
        or source_identity.get("official_split") != "train"
        or sampling.get("selected_source_frames") != expected_frames
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
        raise ValueError(f"{session_id}: manifest/authority binding mismatch")
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
        raise ValueError(f"{session_id}: authority coverage mismatch")
    camera = spec["camera"]
    if not _pixel_lattices_disjoint(
        int(camera["image_width"]), int(camera["image_height"])
    ):
        raise ValueError("Independent teacher lattices overlap")
    mechanics_receipt = protocol.get("parents", {}).get(
        "swept_envelope_mechanics"
    )
    if not isinstance(mechanics_receipt, dict):
        raise ValueError("Missing mechanics authority receipt")
    mechanics_path = _resolve_receipt(protocol_path, mechanics_receipt)
    if _sha256(mechanics_path) != mechanics_receipt.get("sha256"):
        raise ValueError("Mechanics authority receipt mismatch")
    mechanics = _load_json(mechanics_path)
    field = protocol["field_contract"]
    clearance_contract = protocol["signed_clearance_contract"]
    view = field["teacher_view_for_every_role"]
    if (
        view.get("name") != "reference"
        or view.get("point_sample_stride_xy") != 4
        or view.get("point_sample_offset_xy") != 2
        or field.get("current_only") is not True
        or clearance_contract.get("order_statistic") != 2
        or clearance_contract.get("raw_clearance_clip_m") != [-0.5, 1.0]
    ):
        raise ValueError("Independent fresh teacher contract drifted")
    theta_edges = _theta_edges(
        {
            "theta_bin_count": field["theta_bin_count"],
            "theta_range_degrees": field["theta_range_degrees"],
        }
    )
    distance_edges = np.asarray(field["distance_edges_m"], dtype=np.float64)
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][height])
        for height in ("body", "head")
    ]
    widths = np.asarray(
        [
            field["effective_lateral_half_width_m"][height]
            for height in ("body", "head")
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    expected: list[dict[str, Any]] = []
    for row in rows:
        manifest_id = str(row["id"])
        image_path = _resolve_inside(root, str(row["image_path"]))
        depth_path = _resolve_inside(root, str(row["source_depth_path"]))
        mask_path = _resolve_inside(root, str(row["source_mask_path"]))
        if (
            _sha256(image_path) != row["image_sha256"]
            or _sha256(depth_path) != row["source_depth_sha256"]
            or _sha256(mask_path) != row["source_mask_sha256"]
        ):
            raise ValueError(f"{session_id}:{manifest_id}: media hash mismatch")
        depth = _read_depth(
            depth_path, int(row["width"]), int(row["height"])
        )
        semantic = _read_semantic_class(
            mask_path, int(row["width"]), int(row["height"])
        )
        basis = _anchor_basis(bindings[manifest_id], planes[manifest_id])
        points, dynamic = _obstacle_points_world(
            root,
            row,
            bindings[manifest_id],
            camera,
            stride=4,
            offset=2,
            excluded_classes=set(obstacle["excluded_semantic_class_ids"]),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
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
        _, clearance, inside = _signed_clearance_field(
            points,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
            order_statistic=2,
            final_edge_atol_m=float(
                clearance_contract["final_distance_edge_isclose"]["atol_m"]
            ),
            final_edge_rtol=float(
                clearance_contract["final_distance_edge_isclose"]["rtol"]
            ),
            clip_min_m=-0.5,
            clip_max_m=1.0,
        )
        if not np.array_equal(support, inside):
            raise ValueError("Independent support/clearance mismatch")
        labels = _label_json(known, support, clearance)
        frame = int(row["source_frame_index"])
        expected.append(
            {
                "sample_id": (
                    f"hftf_g0_d1_fresh_{session_id}_{frame:06d}"
                ),
                "session_id": session_id,
                "source_frame_index": frame,
                "manifest_id": manifest_id,
                "image_path": str(image_path),
                "image_sha256": row["image_sha256"],
                "source_depth_sha256": row["source_depth_sha256"],
                "source_mask_sha256": row["source_mask_sha256"],
                "camera_poses_sha256": actual_camera_poses_sha256,
                "authority_report_sha256": actual_authority_sha256,
                "labels": labels,
            }
        )
    return expected


def _validate_rows(
    inputs: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    if not (
        len(inputs) == len(truths) == len(receipts) == len(expected) == 75
    ):
        raise ValueError("Fresh package record count mismatch")
    for input_row, truth, receipt, frozen in zip(
        inputs, truths, receipts, expected, strict=True
    ):
        if (
            set(input_row) != INPUT_KEYS
            or input_row.get("schema") != INPUT_SCHEMA
            or set(input_row.get("current_rgb", {})) != {"path", "sha256"}
            or input_row["current_rgb"].get("path") != frozen["image_path"]
            or input_row["current_rgb"].get("sha256")
            != frozen["image_sha256"]
            or _sha256(Path(input_row["current_rgb"]["path"]))
            != frozen["image_sha256"]
        ):
            raise ValueError("Fresh prediction input firewall/hash mismatch")
        if (
            set(truth) != TRUTH_KEYS
            or truth.get("schema") != TRUTH_SCHEMA
            or set(receipt) != RECEIPT_KEYS
            or receipt.get("schema") != RECEIPT_SCHEMA
        ):
            raise ValueError("Fresh truth or receipt schema mismatch")
        for row in (input_row, truth, receipt):
            if any(
                row.get(key) != frozen[key]
                for key in (
                    "sample_id",
                    "session_id",
                    "source_frame_index",
                    "manifest_id",
                )
            ):
                raise ValueError("Fresh package identity/order mismatch")
        _strict_labels(truth["labels"])
        if _canonical_bytes(truth["labels"]) != _canonical_bytes(
            frozen["labels"]
        ):
            raise ValueError("Fresh truth differs from independent rederivation")
        if (
            receipt.get("teacher_view")
            != "REFERENCE_STRIDE4_OFFSET2_CURRENT"
            or receipt.get("source_depth_sha256")
            != frozen["source_depth_sha256"]
            or receipt.get("source_mask_sha256")
            != frozen["source_mask_sha256"]
            or receipt.get("camera_poses_sha256")
            != frozen["camera_poses_sha256"]
            or receipt.get("authority_report_sha256")
            != frozen["authority_report_sha256"]
            or receipt.get("labels_sha256")
            != _sha256_bytes(_canonical_bytes(frozen["labels"]))
            or receipt.get("student_loader_authorized") is not False
        ):
            raise ValueError("Fresh teacher receipt mismatch")


def _opportunity(
    truths: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {}
    adequate = True
    for session_id in EXPECTED_SOURCE_IDS:
        rows = [row for row in truths if row["session_id"] == session_id]
        if len(rows) != 25:
            raise ValueError("Fresh opportunity requires 25 frames per source")
        source: dict[str, Any] = {}
        for height_index, height in enumerate(("body", "head")):
            known_count = 0
            positive = 0
            negative = 0
            unknown_safe = 0
            for row in rows:
                known, risk, _ = _strict_labels(row["labels"])
                mask = known[height_index]
                known_count += int(mask.sum())
                positive += int((mask & (risk[height_index] == 1)).sum())
                negative += int((mask & (risk[height_index] == 0)).sum())
                raw = np.asarray(
                    row["labels"]["risk_target_nullable"], dtype=object
                )[height_index]
                unknown_safe += sum(
                    raw[index] is not None
                    for index in np.ndindex((6, 6))
                    if not mask[index]
                )
            denominator = 900
            coverage = known_count / denominator
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
                "known": known_count,
                "known_coverage": coverage,
                "positive_known": positive,
                "negative_known": negative,
                "unknown": denominator - known_count,
                "unknown_to_safe_violations": unknown_safe,
                "gate_pass": passed,
            }
        result[session_id] = source
    return result, adequate


def _flat_validation_bindings(
    package_root: Path,
    opportunity: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = package_root / "prediction_inputs.jsonl"
    truth_path = package_root / "truth_labels.jsonl"
    return {
        "source_frame_indices": {
            session_id: list(range(0, 50, 2))
            for session_id in EXPECTED_SOURCE_IDS
        },
        "prediction_inputs_path": str(prediction_path.resolve()),
        "prediction_inputs_sha256": _sha256(prediction_path),
        "prediction_input_count": 75,
        "truth_labels_path": str(truth_path.resolve()),
        "truth_labels_sha256": _sha256(truth_path),
        "truth_label_count": 75,
        "unknown_to_safe_violation_count": sum(
            int(metrics["unknown_to_safe_violations"])
            for source in opportunity.values()
            for metrics in source.values()
        ),
    }


def validate(
    contract_path: Path,
    datasets_root: Path,
    authority_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    contract, protocol_path, protocol, sources = _validate_contract(
        contract_path
    )
    _canonical_paths(contract, package_root)
    expected_files = {
        "prediction_inputs.jsonl",
        "truth_labels.jsonl",
        "teacher_receipts.jsonl",
        "package.json",
    }
    if (
        not package_root.is_dir()
        or {path.name for path in package_root.iterdir()} != expected_files
    ):
        raise ValueError("Fresh package file set mismatch")
    package = _load_json(package_root / "package.json")
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("terminal")
        not in (PACKAGE_READY, PACKAGE_NOT_EVALUABLE)
        or package.get("contract_sha256") != _sha256(contract_path)
        or package.get("materializer_sha256")
        != contract["implementations"]["fresh_package_materializer"]["sha256"]
        or package.get("source_order") != list(EXPECTED_SOURCE_IDS)
        or package.get("record_count") != 75
    ):
        raise ValueError("Fresh package manifest mismatch")
    materialization_execution_receipt_sha256 = (
        _validate_materialization_execution(
            contract_path, contract, package_root, package
        )
    )
    for name in expected_files - {"package.json"}:
        if (
            _sha256(package_root / name)
            != package.get("files", {}).get(name, {}).get("sha256")
        ):
            raise ValueError(f"Fresh payload hash mismatch: {name}")
    inputs = _load_jsonl(package_root / "prediction_inputs.jsonl")
    truths = _load_jsonl(package_root / "truth_labels.jsonl")
    receipts = _load_jsonl(package_root / "teacher_receipts.jsonl")
    expected: list[dict[str, Any]] = []
    for source in sources:
        expected.extend(
            _derive_source(
                source,
                datasets_root,
                authority_root,
                protocol_path,
                protocol,
            )
        )
    _validate_rows(inputs, truths, receipts, expected)
    opportunity, adequate = _opportunity(truths)
    if (
        _canonical_bytes(opportunity)
        != _canonical_bytes(package.get("opportunity_gate"))
        or (package["terminal"] == PACKAGE_READY) != adequate
    ):
        raise ValueError("Fresh opportunity gate/terminal mismatch")
    prediction_path = package_root / "prediction_inputs.jsonl"
    truth_path = package_root / "truth_labels.jsonl"
    flat_bindings = _flat_validation_bindings(
        package_root, opportunity
    )
    return {
        "schema": VALIDATION_SCHEMA,
        "terminal": READY if adequate else NOT_EVALUABLE,
        "contract_sha256": _sha256(contract_path),
        "package_manifest_sha256": _sha256(package_root / "package.json"),
        "package_validator_sha256": _sha256(Path(__file__).resolve()),
        "materialization_execution_receipt_sha256": (
            materialization_execution_receipt_sha256
        ),
        "source_order": list(EXPECTED_SOURCE_IDS),
        **flat_bindings,
        "source_count": 3,
        "record_count": 75,
        "prediction_inputs": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
            "record_count": 75,
        },
        "truth_labels": {
            "path": str(truth_path.resolve()),
            "sha256": _sha256(truth_path),
            "record_count": 75,
            "prediction_loader_authorized": False,
        },
        "teacher_receipts_sha256": _sha256(
            package_root / "teacher_receipts.jsonl"
        ),
        "opportunity_gate": opportunity,
        "checks": {
            "exact_fixed_source_and_frame_order": True,
            "manifest_media_and_authority_hashes_rebound": True,
            "current_teacher_labels_independently_rederived": True,
            "prediction_input_contains_identity_and_current_rgb_only": True,
            "truth_and_teacher_receipts_prediction_loader_forbidden": True,
            "unknown_to_safe_violations_zero": True,
            "student_prediction_opened": False,
            "reserved_heldout_opened": False,
        },
        "authorization": {
            "fresh_prediction_authorized": adequate,
            "one_shot_prediction_authorized": adequate,
            "truth_join_authorized_before_predictions_frozen": False,
            "truth_join_before_all_predictions_frozen": False,
            "source_replacement_or_package_rematerialization": False,
            "reserved_heldout_opening": False,
            "mainline_or_safety_claim": False,
        },
    }


def _prediction_authorization(
    report: dict[str, Any],
) -> dict[str, Any]:
    adequate = report.get("terminal") == READY
    authorization = {
        "schema": PREDICTION_AUTHORIZATION_SCHEMA,
        "terminal": (
            PREDICTION_AUTHORIZATION_READY
            if adequate
            else PREDICTION_AUTHORIZATION_NOT_EVALUABLE
        ),
        "contract_sha256": report["contract_sha256"],
        "package_validator_sha256": report[
            "package_validator_sha256"
        ],
        "prediction_inputs_path": report["prediction_inputs_path"],
        "prediction_inputs_sha256": report["prediction_inputs_sha256"],
        "prediction_input_count": report["prediction_input_count"],
        "source_order": report["source_order"],
        "source_frame_indices": report["source_frame_indices"],
        "authorization": {
            "fresh_prediction_authorized": adequate,
            "truth_join_authorized_before_predictions_frozen": False,
            "source_replacement_or_package_rematerialization": False,
        },
    }
    forbidden_keys = {
        "truth_labels_path",
        "truth_labels_sha256",
        "truth_label_count",
        "teacher_receipts_sha256",
        "opportunity_gate",
        "unknown_to_safe_violation_count",
        "package_manifest_sha256",
    }
    if forbidden_keys & set(authorization):
        raise ValueError("Prediction authorization contains quarantined fields")
    return authorization


def _atomic_write(
    output_root: Path,
    report: dict[str, Any],
    prediction_authorization: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError("Refusing to overwrite fresh validation")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-", dir=output_root.parent
        )
    )
    try:
        payloads = {
            "validation.json": report,
            "prediction_authorization.json": prediction_authorization,
        }
        for name, payload in payloads.items():
            with (partial / name).open("xb") as stream:
                stream.write(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        partial.replace(output_root)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
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
        "--package-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801"
        ),
    )
    args = parser.parse_args()
    try:
        contract = _load_json(args.contract.resolve())
        _canonical_paths(
            contract,
            args.package_root.resolve(),
            args.output_root.resolve(),
        )
        report = validate(
            args.contract.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
            args.package_root.resolve(),
        )
        prediction_authorization = _prediction_authorization(report)
        _atomic_write(
            args.output_root.resolve(),
            report,
            prediction_authorization,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": VALIDATION_SCHEMA,
                    "terminal": ERROR,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "output_written": False,
                    "prediction_authorized": False,
                    "source_replacement_authorized": False,
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
