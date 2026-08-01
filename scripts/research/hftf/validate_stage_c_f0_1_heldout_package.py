#!/usr/bin/env python3
"""Validate the one-shot F0.1 heldout package before any student forward."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from audit_stage_c_f0_1_teacher_opportunity import _root_name
from materialize_stage_c_f0_1_heldout_package import (
    CONTRACT_SCHEMA,
    EXPECTED_CONTRACT_STATUS,
    READY as PACKAGE_READY,
    SCHEMA as PACKAGE_SCHEMA,
    _canonical_artifact_path,
    _implementation_receipt,
)
from run_geometry_teacher_canary import _resolve_inside, _sha256
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_package_validation"
READY = "F0_1_SANPO_HELDOUT_PACKAGE_VALIDATED"
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE"
)
INPUT_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "role",
    "target_fps",
    "anchor_timeline_index",
    "anchor_source_frame_index",
    "history_rgb",
}
TRUTH_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "anchor_timeline_index",
    "anchor_source_frame_index",
    "labels",
}
RECEIPT_KEYS = {
    "schema",
    "sample_id",
    "role",
    "teacher_view",
    "session_id",
    "anchor_manifest_id",
    "future_manifest_id",
    "anchor_source_frame_index",
    "future_source_frame_index",
    "anchor_depth_sha256",
    "anchor_mask_sha256",
    "future_depth_sha256",
    "future_mask_sha256",
    "camera_poses_sha256",
    "history_velocity_source_frame_indices",
    "causal_future_origin_m",
    "anchor_forward",
    "anchor_right",
    "anchor_up",
    "history_only_tangent_velocity_mps",
    "student_loader_authorized",
}


def _strict_binary_matrix(value: Any, description: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2, 6, 6):
        raise ValueError(f"{description} shape mismatch")
    for item in raw.flat:
        if type(item) is not int or item not in (0, 1):
            raise ValueError(f"{description} must contain exact JSON integers 0/1")
    return np.asarray(raw, dtype=np.uint8)


def _validate_label(label: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if set(label) != {"known_target", "risk_target_nullable"}:
        raise ValueError("Heldout label key set mismatch")
    known = _strict_binary_matrix(
        label["known_target"], "Heldout known target"
    )
    risk_object = np.asarray(label["risk_target_nullable"], dtype=object)
    if (
        known.shape != (2, 6, 6)
        or risk_object.shape != (2, 6, 6)
    ):
        raise ValueError("Heldout label shape/value mismatch")
    numeric = np.vectorize(lambda value: value is not None)(risk_object)
    if not np.array_equal(numeric, known.astype(bool)):
        raise ValueError("UNKNOWN risk must be null iff known=0")
    risk = np.zeros((2, 6, 6), dtype=np.uint8)
    for index in np.argwhere(numeric):
        position = tuple(int(value) for value in index)
        value = risk_object[position]
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Known heldout risk must be binary")
        risk[position] = int(value)
    return known.astype(bool), risk


def _aggregate_truth(
    truth_records: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in truth_records:
        source = result.setdefault(
            str(record["session_id"]),
            {
                horizon: {
                    height: {
                        "known": 0,
                        "positive_known": 0,
                        "negative_known": 0,
                        "unknown": 0,
                    }
                    for height in ("body", "head")
                }
                for horizon in ("current", "future")
            },
        )
        for horizon in ("current", "future"):
            known, risk = _validate_label(record["labels"][horizon])
            for height_index, height in enumerate(("body", "head")):
                mask = known[height_index]
                target = risk[height_index]
                metrics = source[horizon][height]
                metrics["known"] += int(mask.sum())
                metrics["positive_known"] += int(
                    (mask & (target == 1)).sum()
                )
                metrics["negative_known"] += int(
                    (mask & (target == 0)).sum()
                )
                metrics["unknown"] += int((~mask).sum())
    return result


def _expected_sample_rows(
    contract: dict[str, Any],
    source_lock: dict[str, Any],
    opportunity: dict[str, Any],
    datasets_root: Path,
) -> list[dict[str, Any]]:
    sources = {
        str(source["session_id"]): source for source in source_lock["sources"]
    }
    opportunity_sources = {
        str(source["session_id"]): source
        for source in opportunity["source_results"]
    }
    expected: list[dict[str, Any]] = []
    for session_id in contract["heldout_source_contract"]["source_order"]:
        source = sources[session_id]
        opportunity_source = opportunity_sources[session_id]
        if source["role"] != "heldout" or source["official_split"] != "test":
            raise ValueError("Frozen heldout source role/split mismatch")
        root = (datasets_root / _root_name(source)).resolve()
        manifest_path = root / "manifest.replay.jsonl"
        if _sha256(manifest_path) != opportunity_source["manifest_sha256"]:
            raise ValueError("Heldout validator manifest hash mismatch")
        manifest = _load_jsonl(manifest_path)
        selected_frames = source["selected_source_frames"]
        if [row["source_frame_index"] for row in manifest] != selected_frames:
            raise ValueError("Heldout validator source-frame mapping mismatch")
        for anchor in range(8, 21):
            anchor_row = manifest[anchor]
            future_row = manifest[anchor + 4]
            history_rows = [
                manifest[index] for index in range(anchor - 8, anchor + 1, 2)
            ]
            expected.append(
                {
                    "sample_id": (
                        f"hftf_f0_1_heldout_{session_id}_{anchor:02d}"
                    ),
                    "session_id": session_id,
                    "anchor": anchor,
                    "anchor_source_frame": selected_frames[anchor],
                    "future_source_frame": selected_frames[anchor + 4],
                    "history_source_frames": [
                        row["source_frame_index"] for row in history_rows
                    ],
                    "history_rgb": [
                        {
                            "image_path": str(
                                _resolve_inside(root, str(row["image_path"]))
                            ),
                            "image_sha256": row["image_sha256"],
                        }
                        for row in history_rows
                    ],
                    "anchor_manifest_id": anchor_row["id"],
                    "future_manifest_id": future_row["id"],
                    "anchor_depth_sha256": anchor_row[
                        "source_depth_sha256"
                    ],
                    "anchor_mask_sha256": anchor_row[
                        "source_mask_sha256"
                    ],
                    "future_depth_sha256": future_row[
                        "source_depth_sha256"
                    ],
                    "future_mask_sha256": future_row[
                        "source_mask_sha256"
                    ],
                    "camera_poses_sha256": opportunity_source[
                        "camera_poses_sha256"
                    ],
                }
            )
    return expected


def _validate_probability_free_input(
    record: dict[str, Any], expected: dict[str, Any]
) -> None:
    if (
        set(record) != INPUT_KEYS
        or record.get("schema")
        != "blindassist_hftf_f0_1_heldout_inference_input"
        or record.get("sample_id") != expected["sample_id"]
        or record.get("session_id") != expected["session_id"]
        or record.get("role") != "heldout"
        or record.get("target_fps") != 10.0
        or record.get("anchor_timeline_index") != expected["anchor"]
        or record.get("anchor_source_frame_index")
        != expected["anchor_source_frame"]
    ):
        raise ValueError("Heldout inference input identity mismatch")
    history = record["history_rgb"]
    if (
        not isinstance(history, list)
        or len(history) != 5
        or [item.get("relative_time_s") for item in history]
        != [-0.8, -0.6, -0.4, -0.2, 0.0]
        or any(
            set(item)
            != {"relative_time_s", "image_path", "image_sha256"}
            for item in history
        )
    ):
        raise ValueError("Heldout history schema/time mismatch")
    for item, expected_item in zip(
        history, expected["history_rgb"], strict=True
    ):
        path = Path(str(item["image_path"]))
        if (
            str(path.resolve()) != expected_item["image_path"]
            or item["image_sha256"] != expected_item["image_sha256"]
            or _sha256(path) != item["image_sha256"]
        ):
            raise ValueError(f"Heldout history current file hash mismatch: {path}")


def _validate_truth(
    record: dict[str, Any], expected: dict[str, Any]
) -> None:
    if (
        set(record) != TRUTH_KEYS
        or record.get("schema") != "blindassist_hftf_f0_1_heldout_truth"
        or record.get("sample_id") != expected["sample_id"]
        or record.get("session_id") != expected["session_id"]
        or record.get("anchor_timeline_index") != expected["anchor"]
        or record.get("anchor_source_frame_index")
        != expected["anchor_source_frame"]
        or set(record.get("labels", {})) != {"current", "future"}
    ):
        raise ValueError("Heldout truth identity mismatch")
    _validate_label(record["labels"]["current"])
    _validate_label(record["labels"]["future"])


def _validate_receipt(
    record: dict[str, Any], expected: dict[str, Any]
) -> None:
    if (
        set(record) != RECEIPT_KEYS
        or record.get("schema")
        != "blindassist_hftf_f0_1_teacher_receipt"
        or record.get("sample_id") != expected["sample_id"]
        or record.get("session_id") != expected["session_id"]
        or record.get("role") != "heldout"
        or record.get("teacher_view") != "reference"
        or record.get("anchor_source_frame_index")
        != expected["anchor_source_frame"]
        or record.get("future_source_frame_index")
        != expected["future_source_frame"]
        or record.get("anchor_manifest_id")
        != expected["anchor_manifest_id"]
        or record.get("future_manifest_id")
        != expected["future_manifest_id"]
        or record.get("anchor_depth_sha256")
        != expected["anchor_depth_sha256"]
        or record.get("anchor_mask_sha256")
        != expected["anchor_mask_sha256"]
        or record.get("future_depth_sha256")
        != expected["future_depth_sha256"]
        or record.get("future_mask_sha256")
        != expected["future_mask_sha256"]
        or record.get("camera_poses_sha256")
        != expected["camera_poses_sha256"]
        or record.get("student_loader_authorized") is not False
        or record.get("history_velocity_source_frame_indices")
        != [
            expected["history_source_frames"][-3],
            expected["anchor_source_frame"],
        ]
    ):
        raise ValueError("Heldout teacher receipt identity mismatch")
    for key in (
        "anchor_depth_sha256",
        "anchor_mask_sha256",
        "future_depth_sha256",
        "future_mask_sha256",
        "camera_poses_sha256",
    ):
        digest = record[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Heldout receipt digest mismatch: {key}")
    for key in (
        "causal_future_origin_m",
        "anchor_forward",
        "anchor_right",
        "anchor_up",
        "history_only_tangent_velocity_mps",
    ):
        values = record[key]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError(f"Heldout receipt vector mismatch: {key}")


def validate(
    contract_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
    datasets_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != EXPECTED_CONTRACT_STATUS
    ):
        raise ValueError("Frozen heldout contract identity mismatch")
    _implementation_receipt(
        contract, "heldout_package_validator", Path(__file__).resolve()
    )
    _canonical_artifact_path(
        contract, "heldout_package_root", package_root
    )
    if (
        _sha256(source_lock_path)
        != contract["parents"]["source_lock"]["sha256"]
        or _sha256(opportunity_path)
        != contract["parents"]["teacher_opportunity_report"]["sha256"]
    ):
        raise ValueError("Heldout package validation parent mismatch")
    source_lock = _load_json(source_lock_path)
    opportunity = _load_json(opportunity_path)
    expected_files = {
        "inference_inputs.jsonl",
        "heldout_truth.jsonl",
        "teacher_receipts.jsonl",
        "package.json",
    }
    if (
        not package_root.is_dir()
        or {path.name for path in package_root.iterdir()} != expected_files
    ):
        raise ValueError("Heldout package file set mismatch")
    package = _load_json(package_root / "package.json")
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("terminal") != PACKAGE_READY
        or package.get("contract_sha256") != _sha256(contract_path)
        or package.get("materializer_sha256")
        != contract["implementations"]["heldout_materializer"]["sha256"]
        or package.get("record_count") != 39
    ):
        raise ValueError("Heldout package manifest mismatch")
    for name in expected_files - {"package.json"}:
        if _sha256(package_root / name) != package["files"][name]["sha256"]:
            raise ValueError(f"Heldout package payload hash mismatch: {name}")
    inputs = _load_jsonl(package_root / "inference_inputs.jsonl")
    truth = _load_jsonl(package_root / "heldout_truth.jsonl")
    receipts = _load_jsonl(package_root / "teacher_receipts.jsonl")
    expected_rows = _expected_sample_rows(
        contract, source_lock, opportunity, datasets_root
    )
    if (
        len(inputs) != 39
        or len(truth) != 39
        or len(receipts) != 39
        or len(expected_rows) != 39
    ):
        raise ValueError("Heldout package record count mismatch")
    for input_record, truth_record, receipt, expected in zip(
        inputs, truth, receipts, expected_rows, strict=True
    ):
        _validate_probability_free_input(input_record, expected)
        _validate_truth(truth_record, expected)
        _validate_receipt(receipt, expected)
    if (
        [record["sample_id"] for record in inputs]
        != [row["sample_id"] for row in expected_rows]
        or [record["sample_id"] for record in truth]
        != [row["sample_id"] for row in expected_rows]
        or [record["sample_id"] for record in receipts]
        != [row["sample_id"] for row in expected_rows]
    ):
        raise ValueError("Heldout package sample order mismatch")
    aggregate = _aggregate_truth(truth)
    opportunity_by_id = {
        str(source["session_id"]): source
        for source in opportunity["source_results"]
        if source["role"] == "heldout"
    }
    if list(aggregate) != contract["heldout_source_contract"]["source_order"]:
        raise ValueError("Heldout truth source order mismatch")
    for session_id, actual in aggregate.items():
        expected_horizons = opportunity_by_id[session_id]["teacher_views"][
            "reference"
        ]["horizons"]
        for horizon in ("current", "future"):
            for height in ("body", "head"):
                for key in (
                    "known",
                    "positive_known",
                    "negative_known",
                    "unknown",
                ):
                    if actual[horizon][height][key] != expected_horizons[
                        horizon
                    ][height][key]:
                        raise ValueError(
                            f"Heldout truth aggregate mismatch: "
                            f"{session_id}:{horizon}:{height}:{key}"
                        )
    return {
        "schema": SCHEMA,
        "terminal": READY,
        "contract_sha256": _sha256(contract_path),
        "package_validator_sha256": _sha256(Path(__file__).resolve()),
        "source_lock_sha256": _sha256(source_lock_path),
        "teacher_opportunity_sha256": _sha256(opportunity_path),
        "package_manifest_sha256": _sha256(package_root / "package.json"),
        "package_root": str(package_root.resolve()),
        "files": {
            name: {
                "sha256": _sha256(package_root / name),
                "record_count": 39,
            }
            for name in (
                "inference_inputs.jsonl",
                "heldout_truth.jsonl",
                "teacher_receipts.jsonl",
            )
        },
        "checks": {
            "exact_source_anchor_and_sample_order": True,
            "inference_schema_and_history_hashes_exact": True,
            "truth_schema_shape_null_mask_exact": True,
            "receipt_schema_causal_binding_exact": True,
            "truth_reaggregates_to_frozen_reference_opportunity": True,
            "student_output_computed": False,
        },
        "authorization": {
            "one_shot_prediction_authorized": True,
            "truth_join_authorized_before_predictions_frozen": False,
            "package_rematerialization_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--opportunity", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_root = args.output_root.resolve()
        contract = _load_json(args.contract.resolve())
        _canonical_artifact_path(
            contract, "heldout_package_validation_root", output_root
        )
        if output_root.exists():
            raise FileExistsError("Refusing to overwrite package validation")
        report = validate(
            args.contract.resolve(),
            args.source_lock.resolve(),
            args.opportunity.resolve(),
            args.datasets_root.resolve(),
            args.package_root.resolve(),
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(
            tempfile.mkdtemp(
                prefix=f"{output_root.name}.partial-",
                dir=output_root.parent,
            )
        )
        with (partial / "validation.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        partial.replace(output_root)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "validation_sha256": _sha256(
                        output_root / "validation.json"
                    ),
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
