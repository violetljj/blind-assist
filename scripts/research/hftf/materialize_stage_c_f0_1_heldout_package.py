#!/usr/bin/env python3
"""Materialize the one-shot F0.1 heldout package with truth quarantine."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from audit_stage_c_f0_1_teacher_opportunity import (
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
from materialize_stage_c_f0_1_train_dev_corpus import (
    HEIGHTS,
    HORIZONS,
    _aggregate_records,
    _canonical_jsonl,
    _nested_nullable_targets,
    _student_record_firewall,
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


SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_package"
READY = "F0_1_SANPO_HELDOUT_PACKAGE_FROZEN"
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE"
)
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_heldout_execution_contract_f0_1"
)
EXPECTED_CONTRACT_STATUS = (
    "FROZEN_AFTER_NINE_CHECKPOINT_GATE_BEFORE_HELDOUT_TARGET_"
    "MATERIALIZATION_OR_STUDENT_OUTPUT"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_receipt(
    contract: dict[str, Any], key: str, path: Path
) -> None:
    receipt = contract.get("implementations", {}).get(key, {})
    if (
        Path(str(receipt.get("path", ""))).as_posix()
        != path.relative_to(path.parents[3]).as_posix()
        or receipt.get("sha256") != _sha256(path)
    ):
        raise ValueError(f"Frozen implementation receipt mismatch: {key}")


def _canonical_artifact_path(
    contract: dict[str, Any], key: str, path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    expected = (
        repository_root
        / str(contract["canonical_artifact_paths"][key])
    ).resolve()
    if path.resolve() != expected:
        raise ValueError(f"Noncanonical heldout artifact path: {key}")


def _validate_parent(
    contract: dict[str, Any], key: str, path: Path
) -> dict[str, Any]:
    receipt = contract["parents"][key]
    if _sha256(path) != receipt["sha256"]:
        raise ValueError(f"Frozen heldout parent hash mismatch: {key}")
    return _load_json(path)


def _validate_heldout_aggregate(
    aggregate: dict[str, Any],
    opportunity_sources: list[dict[str, Any]],
    expected_source_order: list[str],
) -> None:
    heldout = [
        source for source in opportunity_sources if source["role"] == "heldout"
    ]
    if list(aggregate) != expected_source_order or [
        str(source["session_id"]) for source in heldout
    ] != expected_source_order:
        raise ValueError("Heldout source order differs from frozen contract")
    for expected in heldout:
        session_id = str(expected["session_id"])
        actual = aggregate[session_id]
        expected_horizons = expected["teacher_views"]["reference"]["horizons"]
        if (
            actual["role"] != "heldout"
            or actual["teacher_view"] != "reference"
            or actual["record_count"] != 13
            or actual["denominator_per_height_per_horizon"] != 468
        ):
            raise ValueError(f"{session_id}: heldout aggregate identity mismatch")
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
                ):
                    if expected_metrics[key] != actual_metrics[key]:
                        raise ValueError(
                            f"{session_id}:{horizon}:{height}:{key} mismatch"
                        )
                if expected_metrics["denominator"] != 468:
                    raise ValueError("Frozen teacher denominator mismatch")


def _materialize_source(
    source: dict[str, Any],
    opportunity_source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    f0: dict[str, Any],
    mechanics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        source.get("role") != "heldout"
        or source.get("official_split") != "test"
        or opportunity_source.get("role") != "heldout"
        or opportunity_source.get("role_gate_view") != "reference"
    ):
        raise ValueError("Only frozen official-test heldout sources are allowed")
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
        raise ValueError("Heldout source differs from frozen opportunity input")
    binding_by_id = {
        str(item["manifest_id"]): item
        for item in authority["source_pose_authority"]["bindings"]
    }
    plane_by_id = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    timeline = _timeline_contract(float(source["target_fps"]))
    if (
        timeline != opportunity_source["timeline"]
        or timeline["usable_anchor_indices"] != list(range(8, 21))
    ):
        raise ValueError("Heldout timeline differs from frozen contract")
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
            stride=4,
            offset=2,
            excluded_classes=set(obstacle["excluded_semantic_class_ids"]),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
    observation.cache_clear()
    image_hash_cache: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
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
        history_rgb: list[dict[str, Any]] = []
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
                raise ValueError("Heldout history RGB hash mismatch")
            history_rgb.append(
                {
                    "relative_time_s": relative_time,
                    "image_path": str(image_path),
                    "image_sha256": digest,
                }
            )
        sample_id = (
            f"hftf_f0_1_heldout_{source['session_id']}_{anchor_index:02d}"
        )
        record = {
            "schema": "blindassist_hftf_f0_1_student_sample",
            "sample_id": sample_id,
            "role": "heldout",
            "teacher_view": "reference",
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
        if not _student_record_firewall(record):
            raise ValueError("Heldout record contains teacher-only feature")
        receipt = {
            "schema": "blindassist_hftf_f0_1_teacher_receipt",
            "sample_id": sample_id,
            "role": "heldout",
            "teacher_view": "reference",
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
        records.append(record)
        receipts.append(receipt)
    return records, receipts


def _split_record(
    record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    inference = {
        "schema": "blindassist_hftf_f0_1_heldout_inference_input",
        "sample_id": record["sample_id"],
        "session_id": record["session_id"],
        "role": record["role"],
        "target_fps": record["target_fps"],
        "anchor_timeline_index": record["anchor_timeline_index"],
        "anchor_source_frame_index": record["anchor_source_frame_index"],
        "history_rgb": record["history_rgb"],
    }
    truth = {
        "schema": "blindassist_hftf_f0_1_heldout_truth",
        "sample_id": record["sample_id"],
        "session_id": record["session_id"],
        "anchor_timeline_index": record["anchor_timeline_index"],
        "anchor_source_frame_index": record["anchor_source_frame_index"],
        "labels": record["labels"],
    }
    return inference, truth


def materialize(
    contract_path: Path,
    f0_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    authority_cohort_path: Path,
    opportunity_path: Path,
    training_validation_path: Path,
    datasets_root: Path,
    authority_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != EXPECTED_CONTRACT_STATUS
    ):
        raise ValueError("Frozen heldout contract identity mismatch")
    _implementation_receipt(
        contract, "heldout_materializer", Path(__file__).resolve()
    )
    _canonical_artifact_path(
        contract, "heldout_package_root", output_root
    )
    f0 = _validate_parent(contract, "f0_protocol", f0_path)
    mechanics = _validate_parent(
        contract, "swept_envelope_mechanics", mechanics_path
    )
    source_lock = _validate_parent(contract, "source_lock", source_lock_path)
    _validate_parent(
        contract, "authority_cohort", authority_cohort_path
    )
    opportunity = _validate_parent(
        contract, "teacher_opportunity_report", opportunity_path
    )
    training_validation = _validate_parent(
        contract, "nine_checkpoint_validation", training_validation_path
    )
    if (
        training_validation.get("terminal")
        != "F0_1_SANPO_NINE_CHECKPOINTS_FROZEN"
        or training_validation.get("authorization", {}).get(
            "heldout_execution_contract_may_be_frozen"
        )
        is not True
        or training_validation.get("authorization", {}).get(
            "heldout_target_materialization_authorized"
        )
        is not False
    ):
        raise ValueError("Nine-checkpoint authority mismatch")
    expected_source_order = contract["heldout_source_contract"][
        "source_order"
    ]
    sources = [
        source
        for source in source_lock["sources"]
        if source["role"] == "heldout"
    ]
    opportunity_sources = opportunity["source_results"]
    if [str(source["session_id"]) for source in sources] != expected_source_order:
        raise ValueError("Heldout source lock order mismatch")
    opportunity_by_id = {
        str(source["session_id"]): source for source in opportunity_sources
    }
    records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for source in sources:
        source_records, source_receipts = _materialize_source(
            source,
            opportunity_by_id[str(source["session_id"])],
            datasets_root,
            authority_root,
            f0,
            mechanics,
        )
        records.extend(source_records)
        receipts.extend(source_receipts)
    if (
        len(records) != 39
        or len(receipts) != 39
        or [record["sample_id"] for record in records]
        != [receipt["sample_id"] for receipt in receipts]
    ):
        raise ValueError("Heldout package record identity mismatch")
    aggregate = _aggregate_records(records)
    _validate_heldout_aggregate(
        aggregate, opportunity_sources, expected_source_order
    )
    split = [_split_record(record) for record in records]
    inference_records = [item[0] for item in split]
    truth_records = [item[1] for item in split]
    inference_bytes = _canonical_jsonl(inference_records)
    truth_bytes = _canonical_jsonl(truth_records)
    receipt_bytes = _canonical_jsonl(receipts)
    manifest = {
        "schema": SCHEMA,
        "terminal": READY,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "ONE_SHOT_SYNTHETIC_GEOMETRY_PROXY_CANARY",
        "contract_sha256": _sha256(contract_path),
        "materializer_sha256": _sha256(Path(__file__).resolve()),
        "source_count": 3,
        "record_count": 39,
        "source_order": expected_source_order,
        "sample_order": [record["sample_id"] for record in records],
        "source_aggregates": aggregate,
        "files": {
            "inference_inputs.jsonl": {
                "sha256": _sha256_bytes(inference_bytes),
                "record_count": 39,
            },
            "heldout_truth.jsonl": {
                "sha256": _sha256_bytes(truth_bytes),
                "record_count": 39,
            },
            "teacher_receipts.jsonl": {
                "sha256": _sha256_bytes(receipt_bytes),
                "record_count": 39,
            },
        },
        "firewall": {
            "inference_inputs_contain_truth_or_teacher_receipt": False,
            "truth_and_teacher_receipts_are_separate_files": True,
            "student_output_computed": False,
            "heldout_used_for_checkpoint_threshold_source_or_gate_change": False,
        },
        "authorization": {
            "package_validation_authorized": True,
            "student_prediction_authorized_before_package_validation": False,
            "truth_join_authorized": False,
        },
    }
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite heldout package: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    files = {
        "inference_inputs.jsonl": inference_bytes,
        "heldout_truth.jsonl": truth_bytes,
        "teacher_receipts.jsonl": receipt_bytes,
        "package.json": json.dumps(
            manifest, indent=2, ensure_ascii=False
        ).encode("utf-8")
        + b"\n",
    }
    for name, payload in files.items():
        with (partial_root / name).open("xb") as handle:
            handle.write(payload)
    partial_root.replace(output_root)
    return manifest


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("Output must stay under artifacts.local") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--authority-cohort", type=Path, required=True)
    parser.add_argument("--opportunity", type=Path, required=True)
    parser.add_argument("--training-validation", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize(
            args.contract.resolve(),
            args.f0.resolve(),
            args.mechanics.resolve(),
            args.source_lock.resolve(),
            args.authority_cohort.resolve(),
            args.opportunity.resolve(),
            args.training_validation.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
            _require_artifacts_output(args.output_root),
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "record_count": report["record_count"],
                    "files": report["files"],
                }
            )
        )
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
