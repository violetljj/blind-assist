#!/usr/bin/env python3
"""Seal future-blind D2 persistence and causal-advected predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from stage_c_d2_mechanics_common import (
    ANCHORS,
    EXPECTED_ANCHOR_COUNT,
    HORIZONS,
    PREDICTION_RELATIVE_ROOT,
    PREPROCESSOR_COMPLETION_SCHEMA,
    PREPROCESSOR_FAILURE_RELATIVE_PATH,
    PREPROCESSOR_FAILURE_TERMINAL,
    PREPROCESSOR_TERMINAL,
    basis_receipt,
    compute_field,
    compute_known,
    compute_points,
    field_parameters,
    fit_current_plane,
    load_context,
    load_json,
    nullable_field,
    predicted_bases,
    resolve,
    sha256,
)
from verify_sanpo_pose_geometry_authority import (
    _read_depth,
    _read_semantic_class,
)


SCHEMA = "blindassist_hftf_stage_c_d2_future_blind_anchor_prediction"
IMPLEMENTATION_KEY = "future_blind_preprocessor"
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_future_blind_preprocessor_attempt"
)
ATTEMPT_STATUS = (
    "D2_FUTURE_BLIND_PREPROCESSOR_ATTEMPT_OPENED_"
    "BEFORE_FIRST_POSE_OR_MEDIA_READ"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d2_future_blind_preprocessor_failure"
)
FAILURE_TERMINAL = PREPROCESSOR_FAILURE_TERMINAL


def _frame_map(source: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = source.get("frames", [])
    indices = [int(item["normalized_index"]) for item in frames]
    if indices != list(range(13)):
        raise ValueError("D2 source requires ordered per-frame slices 0..12")
    forbidden = {
        "position_m",
        "quaternion_xyzw",
        "depth_values",
        "semantic_values",
    }
    if any(forbidden & set(item) for item in frames):
        raise ValueError("D2 source index embeds pose or media content")
    return {int(item["normalized_index"]): item for item in frames}


def _bound_file(
    repo_root: Path,
    index_root: Path,
    receipt: dict[str, Any],
) -> Path:
    path = resolve(repo_root, index_root, str(receipt["path"]))
    if sha256(path) != str(receipt["sha256"]):
        raise ValueError("D2 per-frame slice hash mismatch")
    return path


def _load_pose(
    repo_root: Path,
    index_root: Path,
    frame: dict[str, Any],
) -> dict[str, Any]:
    path = _bound_file(
        repo_root,
        index_root,
        frame["pose_slice"],
    )
    value = load_json(path)
    if (
        value.get("schema")
        != "blindassist_hftf_stage_c_d2_single_pose_slice"
        or int(value.get("normalized_index", -1))
        != int(frame["normalized_index"])
        or int(value.get("source_frame_index", -1))
        != int(frame["source_frame_index"])
        or value.get("tracking_state") != "TrackingState.READY"
        or len(str(value.get("source_pose_csv_sha256", ""))) != 64
        or not {"position_m", "quaternion_xyzw"}
        <= set(value.get("binding", {}))
    ):
        raise ValueError("D2 single-pose slice identity mismatch")
    return value["binding"]


def _load_current(
    repo_root: Path,
    index_root: Path,
    frame: dict[str, Any],
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    depth_path = _bound_file(
        repo_root,
        index_root,
        frame["depth"],
    )
    mask_path = _bound_file(
        repo_root,
        index_root,
        frame["mask"],
    )
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    return (
        _read_depth(depth_path, width, height),
        _read_semantic_class(mask_path, width, height),
    )


def preprocess_anchor(
    source: dict[str, Any],
    anchor_index: int,
    history_binding: dict[str, Any],
    current_binding: dict[str, Any],
    depth: np.ndarray,
    semantic: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    camera = source["camera"]
    frame = _frame_map(source)[anchor_index]
    row = {
        "id": frame["manifest_id"],
        "width": int(camera["image_width"]),
        "height": int(camera["image_height"]),
    }
    plane = fit_current_plane(
        depth,
        semantic,
        current_binding,
        camera,
        int(frame["source_frame_index"]),
    )
    current_basis, predicted, motion = predicted_bases(
        history_binding,
        current_binding,
        plane,
    )
    points = compute_points(
        depth,
        semantic,
        row,
        current_binding,
        camera,
        parameters,
    )
    current_clearance = compute_field(points, current_basis, parameters)
    current_counts, current_known = compute_known(
        depth,
        semantic,
        row,
        current_binding,
        camera,
        current_basis,
        parameters,
    )
    horizons: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        advected_clearance = compute_field(
            points,
            predicted[horizon],
            parameters,
        )
        advected_counts, advected_known = compute_known(
            depth,
            semantic,
            row,
            current_binding,
            camera,
            predicted[horizon],
            parameters,
        )
        horizons.append(
            {
                "horizon_s": horizon,
                "predicted_basis": basis_receipt(predicted[horizon]),
                "arms": {
                    "CURRENT_FIELD_PERSISTENCE": {
                        "probe_pass_counts": current_counts.tolist(),
                        "known": current_known.tolist(),
                        "clearance_m": nullable_field(
                            current_known,
                            current_clearance,
                        ),
                    },
                    "HISTORY_CAUSAL_ADVECTED_CURRENT_FIELD": {
                        "probe_pass_counts": advected_counts.tolist(),
                        "known": advected_known.tolist(),
                        "clearance_m": nullable_field(
                            advected_known,
                            advected_clearance,
                        ),
                    },
                },
            }
        )
    return (
        {
            "schema": SCHEMA,
            "session_id": source["session_id"],
            "anchor_normalized_index": anchor_index,
            "source_frame_index": int(frame["source_frame_index"]),
            "history_normalized_index": anchor_index - 2,
            "history_interval_s": 0.4,
            "current_basis": basis_receipt(current_basis),
            "motion_receipt": motion,
            "current_local_ground_plane": plane,
            "point_population_rule": (
                "ALL_FINITE_SEMANTICALLY_ADMITTED_STRIDE4_OFFSET2_"
                "POINTS_REMAIN_IN_EVERY_CELL_ORDER_STATISTIC"
            ),
            "horizons": horizons,
            "future_depth_mask_or_pose_read": False,
            "unknown_to_safe_violations": 0,
        },
        points,
    )


def _write_json_durable(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_points_durable(path: Path, points: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, points.astype("<f8", copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def run(contract_path: Path, output_root: Path) -> dict[str, Any]:
    context = load_context(
        contract_path,
        IMPLEMENTATION_KEY,
        Path(__file__),
    )
    authorization = context["contract"]["authorization"]
    if (
        authorization.get("future_blind_preprocessor_execution_authorized")
        is not True
        or authorization.get("future_truth_open_authorized_before_completion")
        is not False
    ):
        raise ValueError("D2 preprocessor authorization mismatch")
    expected_root = resolve(
        Path(__file__).resolve().parents[3],
        contract_path.parent,
        context["contract"]["canonical_artifacts"][
            "future_blind_prediction_root"
        ],
    )
    if output_root.resolve() != expected_root or output_root.exists():
        raise FileExistsError("D2 prediction root is noncanonical or exists")
    output_root.mkdir(parents=True, exist_ok=False)
    attempt_path = output_root / "attempt.json"
    _write_json_durable(
        attempt_path,
        {
            "schema": ATTEMPT_SCHEMA,
            "status": ATTEMPT_STATUS,
            "contract_sha256": sha256(contract_path),
            "expected_source_count": 6,
            "expected_anchor_prediction_count": EXPECTED_ANCHOR_COUNT,
            "expected_anchor_horizon_record_count": (
                EXPECTED_ANCHOR_COUNT * len(HORIZONS)
            ),
            "pose_or_media_opened_before_attempt": False,
            "future_depth_mask_or_pose_opened": False,
            "second_preprocessor_run_authorized": False,
            "truth_join_authorized_before_completion": False,
        },
    )
    parameters = field_parameters(context["g0"], context["mechanics"])
    repo_root = Path(__file__).resolve().parents[3]
    index_root = context["source_index_path"].parent
    receipts: list[dict[str, Any]] = []
    for source in context["sources"]:
        frames = _frame_map(source)
        source_root = output_root / str(source["session_id"])
        source_root.mkdir()
        for anchor in ANCHORS:
            history = _load_pose(
                repo_root,
                index_root,
                frames[anchor - 2],
            )
            current = _load_pose(
                repo_root,
                index_root,
                frames[anchor],
            )
            depth, semantic = _load_current(
                repo_root,
                index_root,
                frames[anchor],
                source["camera"],
            )
            prediction, points = preprocess_anchor(
                source,
                anchor,
                history,
                current,
                depth,
                semantic,
                parameters,
            )
            points_path = source_root / f"anchor-{anchor}.points.npy"
            record_path = source_root / f"anchor-{anchor}.json"
            _write_points_durable(points_path, points)
            prediction["points"] = {
                "path": str(points_path.resolve()),
                "sha256": sha256(points_path),
                "shape": list(points.shape),
                "dtype": "float64",
            }
            _write_json_durable(record_path, prediction)
            receipts.append(
                {
                    "session_id": source["session_id"],
                    "anchor_normalized_index": anchor,
                    "path": str(record_path.resolve()),
                    "sha256": sha256(record_path),
                    "points_sha256": sha256(points_path),
                }
            )
            del depth, semantic, points, prediction
    if len(receipts) != EXPECTED_ANCHOR_COUNT:
        raise ValueError("D2 prediction count is not 42")
    for receipt in receipts:
        if sha256(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("A later anchor mutated an earlier prediction")
    completion = {
        "schema": PREPROCESSOR_COMPLETION_SCHEMA,
        "terminal": PREPROCESSOR_TERMINAL,
        "contract_sha256": sha256(contract_path),
        "preprocessor_attempt_sha256": sha256(attempt_path),
        "prediction_record_count": len(receipts),
        "anchor_horizon_record_count": len(receipts) * 2,
        "records": receipts,
        "all_records_durable_before_truth_join": True,
        "future_depth_mask_or_pose_opened": False,
        "truth_join_authorized": True,
        "second_preprocessor_run_authorized": False,
    }
    _write_json_durable(output_root / "completion.json", completion)
    return completion


def seal_failure_if_attempted(
    output_root: Path,
    error: BaseException,
) -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_root = (repo_root / PREDICTION_RELATIVE_ROOT).resolve()
    failure_path = (
        repo_root / PREPROCESSOR_FAILURE_RELATIVE_PATH
    ).resolve()
    if output_root.resolve() != expected_root:
        return None
    attempt_path = expected_root / "attempt.json"
    completion_path = expected_root / "completion.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        return (
            FAILURE_TERMINAL
            if failure.get("terminal") == FAILURE_TERMINAL
            else None
        )
    if not attempt_path.is_file() or completion_path.exists():
        return None
    attempt = load_json(attempt_path)
    if (
        attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("status") != ATTEMPT_STATUS
    ):
        return None
    _write_json_durable(
        failure_path,
        {
            "schema": FAILURE_SCHEMA,
            "terminal": FAILURE_TERMINAL,
            "contract_sha256": attempt.get("contract_sha256"),
            "preprocessor_attempt_sha256": sha256(attempt_path),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_prediction_root_preserved": True,
            "future_depth_mask_or_pose_opened": False,
            "preprocessor_rerun_authorized": False,
            "source_replacement_authorized": False,
            "same_cohort_retuning_authorized": False,
            "truth_effect_execution_authorized": False,
            "rgb_student_execution_authorized": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    )
    return FAILURE_TERMINAL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.contract.resolve(), args.output_root.resolve())
        print(json.dumps({"terminal": result["terminal"]}))
        return 0
    except (
        OSError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        terminal = seal_failure_if_attempted(
            args.output_root.resolve(),
            error,
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "terminal": terminal,
                    "error": str(error),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
