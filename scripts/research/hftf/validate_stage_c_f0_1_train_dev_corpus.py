#!/usr/bin/env python3
"""Independently validate a materialized F0.1 train/dev corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_f0_1_teacher_opportunity import (
    SOURCE_LOCK_SHA256,
    _causal_future_basis,
    _root_name,
    _timeline_contract,
)
from materialize_stage_c_f0_1_train_dev_corpus import (
    CONTRACT_SHA256,
    OPPORTUNITY_SHA256,
    READY as MATERIALIZED_READY,
)
from run_geometry_teacher_canary import _sha256
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_c_f0_1_train_dev_corpus_validation"
READY = "F0_1_SANPO_TRAIN_DEV_CORPUS_VALIDATED"
NOT_EVALUABLE = "F0_1_SANPO_TRAIN_DEV_CORPUS_VALIDATION_FAILED"
STUDENT_SHA256 = (
    "c1bad3b69e769151179762c49586b1b4fa6775a7cfd36a3f0ee9bccc2fbc585f"
)
RECEIPT_SHA256 = (
    "d65b3bb99c301e62edce06efb92bcb3cfc6a9e22a12a60c2758cf6ec857183c3"
)
SPEC_SHA256 = (
    "3930ba34d764c585f5abee0c93bcd179eceb56df8f9d3412d5ed28da9ddf3239"
)
HEIGHTS = ("body", "head")
HORIZONS = ("current", "future")


def _load_jsonl_exact(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Object required at {path}:{line_number}")
        records.append(value)
    return records


def _validate_label(label: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(label, dict) or set(label) != {
        "known_target",
        "risk_target_nullable",
    }:
        raise ValueError("Exact nullable label fields required")
    known_raw = np.asarray(label["known_target"])
    risk_raw = np.asarray(label["risk_target_nullable"], dtype=object)
    if known_raw.shape != (2, 6, 6) or risk_raw.shape != (2, 6, 6):
        raise ValueError("Label shape must be [2,6,6]")
    if not np.isin(known_raw, [0, 1]).all():
        raise ValueError("Known target must be binary")
    known = known_raw.astype(bool)
    numeric = np.zeros((2, 6, 6), dtype=bool)
    risk = np.zeros((2, 6, 6), dtype=np.uint8)
    for index in np.ndindex(risk_raw.shape):
        value = risk_raw[index]
        if value is None:
            continue
        if value not in (0, 1):
            raise ValueError("Known risk target must be binary")
        numeric[index] = True
        risk[index] = int(value)
    if not np.array_equal(numeric, known):
        raise ValueError("Risk is numeric iff the cell is KNOWN")
    return known, risk


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


def _add_label(
    summary: dict[str, Any],
    horizon: str,
    known: np.ndarray,
    risk: np.ndarray,
) -> None:
    for height_index, height in enumerate(HEIGHTS):
        height_known = known[height_index]
        height_risk = risk[height_index]
        metrics = summary[horizon][height]
        metrics["known"] += int(height_known.sum())
        metrics["positive_known"] += int(
            (height_known & (height_risk == 1)).sum()
        )
        metrics["negative_known"] += int(
            (height_known & (height_risk == 0)).sum()
        )
        metrics["unknown"] += int((~height_known).sum())


def _expected_source_contexts(
    source_lock: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for source in source_lock["sources"]:
        if source["role"] == "heldout":
            continue
        session_id = str(source["session_id"])
        root = (datasets_root / _root_name(source)).resolve()
        rows = _load_jsonl(root / "manifest.replay.jsonl")
        authority_path = (
            authority_root / session_id[:8] / "authority.json"
        ).resolve()
        authority = _load_json(authority_path)
        contexts[session_id] = {
            "source": source,
            "root": root,
            "rows": rows,
            "timeline": _timeline_contract(float(source["target_fps"])),
            "bindings": {
                str(item["manifest_id"]): item
                for item in authority["source_pose_authority"]["bindings"]
            },
            "planes": {
                str(item["manifest_id"]): item["local_ground_plane"]
                for item in authority[
                    "ground_and_body_proxy_canary"
                ]["per_frame"]
            },
            "camera_poses_sha256": _sha256(
                root / "source_metadata/camera_poses.csv"
            ),
        }
    return contexts


def validate(
    contract_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
    corpus_root: Path,
    repeat_root: Path,
    datasets_root: Path,
    authority_root: Path,
) -> dict[str, Any]:
    if (
        _sha256(contract_path) != CONTRACT_SHA256
        or _sha256(source_lock_path) != SOURCE_LOCK_SHA256
        or _sha256(opportunity_path) != OPPORTUNITY_SHA256
    ):
        raise ValueError("Frozen corpus validator input hash mismatch")
    student_path = corpus_root / "student_samples.jsonl"
    receipts_path = corpus_root / "teacher_receipts.jsonl"
    spec_path = corpus_root / "dataset_spec.json"
    if (
        _sha256(student_path) != STUDENT_SHA256
        or _sha256(receipts_path) != RECEIPT_SHA256
        or _sha256(spec_path) != SPEC_SHA256
    ):
        raise ValueError("Primary corpus payload hash mismatch")
    repeat_hashes = {
        "student_samples": _sha256(repeat_root / "student_samples.jsonl"),
        "teacher_receipts": _sha256(
            repeat_root / "teacher_receipts.jsonl"
        ),
        "dataset_spec": _sha256(repeat_root / "dataset_spec.json"),
    }
    if repeat_hashes != {
        "student_samples": STUDENT_SHA256,
        "teacher_receipts": RECEIPT_SHA256,
        "dataset_spec": SPEC_SHA256,
    }:
        raise ValueError("Independent corpus repeat is not byte exact")
    contract = _load_json(contract_path)
    source_lock = _load_json(source_lock_path)
    opportunity = _load_json(opportunity_path)
    spec = _load_json(spec_path)
    students = _load_jsonl_exact(student_path)
    receipts = _load_jsonl_exact(receipts_path)
    if (
        contract.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_CELL_CORPUS_MATERIALIZATION"
        or spec.get("terminal") != MATERIALIZED_READY
        or spec.get("student_samples_sha256") != STUDENT_SHA256
        or spec.get("teacher_receipts_sha256") != RECEIPT_SHA256
        or spec.get("determinism_check", {}).get(
            "all_payload_files_byte_exact"
        )
        is not True
        or spec.get("role_record_counts")
        != {"train": 90, "dev": 39, "heldout": 0}
        or len(students) != 129
        or len(receipts) != 129
    ):
        raise ValueError("Materialized corpus contract mismatch")
    contexts = _expected_source_contexts(
        source_lock, datasets_root, authority_root
    )
    opportunity_by_id = {
        str(item["session_id"]): item
        for item in opportunity["source_results"]
    }
    receipt_by_id = {
        str(item["sample_id"]): item for item in receipts
    }
    if (
        len(receipt_by_id) != len(receipts)
        or {str(item["sample_id"]) for item in students}
        != set(receipt_by_id)
    ):
        raise ValueError("Student/teacher receipt IDs are not one-to-one")
    expected_order = [
        (session_id, anchor)
        for session_id, context in contexts.items()
        for anchor in context["timeline"]["usable_anchor_indices"]
    ]
    actual_order = [
        (str(record["session_id"]), int(record["anchor_timeline_index"]))
        for record in students
    ]
    if actual_order != expected_order:
        raise ValueError("Corpus source/anchor order mismatch")
    aggregates: dict[str, dict[str, Any]] = {}
    history_reference_count = 0
    for record in students:
        session_id = str(record["session_id"])
        context = contexts.get(session_id)
        if context is None:
            raise ValueError("Heldout or unknown source entered corpus")
        source = context["source"]
        role = str(source["role"])
        view = "candidate" if role == "train" else "reference"
        anchor = int(record["anchor_timeline_index"])
        timeline = context["timeline"]
        rows = context["rows"]
        if (
            record.get("schema") != "blindassist_hftf_f0_1_student_sample"
            or record.get("role") != role
            or record.get("teacher_view") != view
            or record.get("target_fps") != source["target_fps"]
            or anchor not in timeline["usable_anchor_indices"]
            or record.get("anchor_source_frame_index")
            != rows[anchor]["source_frame_index"]
            or record.get("sample_id")
            != f"hftf_f0_1_{role}_{session_id}_{anchor:02d}"
            or set(record) != {
                "schema",
                "sample_id",
                "role",
                "teacher_view",
                "session_id",
                "anchor_timeline_index",
                "anchor_source_frame_index",
                "target_fps",
                "history_rgb",
                "labels",
            }
        ):
            raise ValueError("Student record identity/firewall mismatch")
        expected_history_indices = [
            anchor + offset for offset in timeline["history_offsets"]
        ]
        history = record["history_rgb"]
        if (
            len(history) != 5
            or [item.get("relative_time_s") for item in history]
            != [-0.8, -0.6, -0.4, -0.2, 0.0]
        ):
            raise ValueError("Student history time contract mismatch")
        for item, history_index in zip(
            history, expected_history_indices, strict=True
        ):
            row = rows[history_index]
            expected_path = _resolve_path(
                context["root"], str(row["image_path"])
            )
            if (
                set(item) != {
                    "relative_time_s",
                    "image_path",
                    "image_sha256",
                }
                or Path(item["image_path"]).resolve() != expected_path
                or item["image_sha256"] != row["image_sha256"]
                or _sha256(expected_path) != row["image_sha256"]
            ):
                raise ValueError("Student history RGB mapping mismatch")
            history_reference_count += 1
        source_aggregate = aggregates.setdefault(
            session_id,
            {
                "role": role,
                "teacher_view": view,
                "record_count": 0,
                "horizons": _empty_summary(),
            },
        )
        source_aggregate["record_count"] += 1
        if set(record["labels"]) != set(HORIZONS):
            raise ValueError("Both current/future labels are required")
        for horizon in HORIZONS:
            known, risk = _validate_label(record["labels"][horizon])
            _add_label(
                source_aggregate["horizons"], horizon, known, risk
            )
        receipt = receipt_by_id[str(record["sample_id"])]
        future_index = anchor + timeline["future_offset"]
        history_index = anchor + timeline["velocity_history_offset"]
        anchor_row = rows[anchor]
        future_row = rows[future_index]
        current_basis, future_basis, velocity = _causal_future_basis(
            context["bindings"][str(rows[history_index]["id"])],
            context["bindings"][str(anchor_row["id"])],
            context["planes"][str(anchor_row["id"])],
        )
        if (
            receipt.get("student_loader_authorized") is not False
            or receipt.get("role") != role
            or receipt.get("teacher_view") != view
            or receipt.get("session_id") != session_id
            or receipt.get("anchor_manifest_id") != anchor_row["id"]
            or receipt.get("future_manifest_id") != future_row["id"]
            or receipt.get("anchor_depth_sha256")
            != anchor_row["source_depth_sha256"]
            or receipt.get("anchor_mask_sha256")
            != anchor_row["source_mask_sha256"]
            or receipt.get("future_depth_sha256")
            != future_row["source_depth_sha256"]
            or receipt.get("future_mask_sha256")
            != future_row["source_mask_sha256"]
            or receipt.get("camera_poses_sha256")
            != context["camera_poses_sha256"]
            or receipt.get("history_velocity_source_frame_indices")
            != [
                rows[history_index]["source_frame_index"],
                anchor_row["source_frame_index"],
            ]
            or not np.allclose(
                receipt.get("causal_future_origin_m"),
                future_basis[0],
                atol=1e-12,
                rtol=0.0,
            )
            or not np.allclose(
                receipt.get("anchor_forward"),
                current_basis[1],
                atol=1e-12,
                rtol=0.0,
            )
            or not np.allclose(
                receipt.get("anchor_right"),
                current_basis[2],
                atol=1e-12,
                rtol=0.0,
            )
            or not np.allclose(
                receipt.get("anchor_up"),
                current_basis[3],
                atol=1e-12,
                rtol=0.0,
            )
            or not np.allclose(
                receipt.get("history_only_tangent_velocity_mps"),
                velocity,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError("Teacher receipt geometry/firewall mismatch")
    if history_reference_count != 645:
        raise ValueError("Expected exactly 645 history RGB references")
    for session_id, aggregate in aggregates.items():
        expected = opportunity_by_id[session_id]
        denominator = aggregate["record_count"] * 36
        if (
            aggregate["role"] != expected["role"]
            or aggregate["teacher_view"] != expected["role_gate_view"]
            or aggregate["record_count"] != expected["usable_anchor_count"]
            or denominator
            != expected["frozen_denominator_per_height_per_horizon"]
        ):
            raise ValueError("Corpus aggregate identity mismatch")
        expected_horizons = expected["teacher_views"][
            expected["role_gate_view"]
        ]["horizons"]
        for horizon in HORIZONS:
            for height in HEIGHTS:
                actual = aggregate["horizons"][horizon][height]
                expected_metrics = expected_horizons[horizon][height]
                if any(
                    actual[key] != expected_metrics[key]
                    for key in (
                        "known",
                        "positive_known",
                        "negative_known",
                        "unknown",
                        "unknown_to_safe_violations",
                    )
                ):
                    raise ValueError(
                        "Serialized labels differ from opportunity counts"
                    )
    return {
        "schema": SCHEMA,
        "terminal": READY,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SYNTHETIC_BODY_HEAD_GEOMETRY_PROXY_ONLY",
        "contract_sha256": _sha256(contract_path),
        "source_lock_sha256": _sha256(source_lock_path),
        "teacher_opportunity_sha256": _sha256(opportunity_path),
        "corpus_payload_hashes": {
            "student_samples": STUDENT_SHA256,
            "teacher_receipts": RECEIPT_SHA256,
            "dataset_spec": SPEC_SHA256,
        },
        "independent_repeat_payload_hashes": repeat_hashes,
        "student_record_count": len(students),
        "teacher_receipt_count": len(receipts),
        "history_rgb_reference_count": history_reference_count,
        "role_record_counts": {"train": 90, "dev": 39, "heldout": 0},
        "source_aggregates": aggregates,
        "checks": {
            "payload_hashes_and_repeat_byte_exact": True,
            "source_role_anchor_order_exact": True,
            "history_rgb_timeline_paths_and_hashes_exact": True,
            "nullable_label_shapes_and_masks_valid": True,
            "serialized_counts_match_opportunity": True,
            "teacher_receipts_and_causal_geometry_exact": True,
            "heldout_records_zero": True,
        },
        "allowed_next_step": "FREEZE_F0_1_STUDENT_TRAINING_EXECUTION",
        "authorization": {
            "training_execution_contract_may_be_frozen": True,
            "student_training_before_contract_freeze": False,
            "heldout_target_materialization_authorized": False,
            "heldout_student_output_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
        "evidence_limit": (
            "Independent disk/schema/timeline/receipt/count validation; "
            "not an independent per-cell geometry oracle."
        ),
    }


def _resolve_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Corpus image path escapes replay root") from error
    if not candidate.is_file():
        raise ValueError("Corpus image path is missing")
    return candidate


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
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--teacher-opportunity", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = validate(
            args.contract.resolve(),
            args.source_lock.resolve(),
            args.teacher_opportunity.resolve(),
            args.corpus_root.resolve(),
            args.repeat_root.resolve(),
            args.datasets_root.resolve(),
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
                    "student_record_count": report["student_record_count"],
                    "history_rgb_reference_count": report[
                        "history_rgb_reference_count"
                    ],
                    "role_record_counts": report["role_record_counts"],
                    "allowed_next_step": report["allowed_next_step"],
                    "output": str(output),
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
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
