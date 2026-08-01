#!/usr/bin/env python3
"""Freeze D3-Q0 future-blind predictions for the durable selected six.

This stage runs only after the sixth qualified selector has been durably
aggregated.  It reads current/history pose and current depth/mask inputs for
anchors 2..8, never sealed payloads and never future-only pose/media.  It
reuses the exact D2 future-blind ``preprocess_anchor`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from preprocess_stage_c_d2_future_blind import preprocess_anchor
from run_stage_c_d3_q0_next_slot import (
    CONTENT_INDEX_SCHEMA,
    CONTENT_READY,
    POSE_SLICE_SCHEMA,
)
from stage_c_d2_mechanics_common import (
    ANCHORS,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_HORIZON_RECORD_COUNT,
    field_parameters,
)
from stage_c_d3_q0_common import (
    SCREENING_ROOT_RELATIVE,
    aggregate_paths,
    load_json,
    preserve_temporary_artifact,
    sha256,
    slot_layout,
    validate_execution_contract,
    validate_selection,
    write_json_exclusive_fsync,
)
from verify_sanpo_pose_geometry_authority import (
    _read_depth,
    _read_semantic_class,
)


IMPLEMENTATION_KEY = "selected_future_blind_preprocessor"
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_future_blind_preprocessor_attempt"
)
ATTEMPT_STATUS = (
    "D3_Q0_FUTURE_BLIND_ATTEMPT_FSYNCED_BEFORE_CURRENT_HISTORY_READ"
)
PREDICTION_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_future_blind_anchor_prediction"
)
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_future_blind_predictions_frozen"
)
COMPLETION_TERMINAL = "D3_Q0_FUTURE_BLIND_PREDICTIONS_FROZEN"
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_future_blind_preprocessor_failure"
)
FAILURE_TERMINAL = (
    "D3_Q0_FUTURE_BLIND_PREPROCESSOR_FAILED_NO_RERUN_NO_REPLACEMENT"
)


class PreprocessorError(ValueError):
    """The selected-six future-blind precondition or content is invalid."""


def _prediction_paths(root: Path) -> dict[str, Path]:
    prediction_root = root / "formal" / "predictions"
    return {
        "root": prediction_root,
        "attempt": prediction_root / "attempt.json",
        "completion": prediction_root / "completion.json",
        "failure": prediction_root / "failure.json",
    }


def _freeze_preprocessor_failure(
    paths: dict[str, Path],
    context: dict[str, Any],
    error: BaseException,
    selection_sha256: str | None,
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["completion"].exists() or paths["failure"].exists():
        return
    preserve_temporary_artifact(paths["failure"])
    write_json_exclusive_fsync(
        paths["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": FAILURE_TERMINAL,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "contract_sha256": sha256(context["contract_path"]),
            "selection_sha256": selection_sha256,
            "preprocessor_attempt_sha256": (
                sha256(paths["attempt"])
                if paths["attempt"].is_file()
                else None
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_predictions_preserved": True,
            "sealed_payload_read": False,
            "future_pose_depth_or_mask_read": False,
            "preprocessor_rerun_authorized": False,
            "effect_authorized": False,
            "source_replacement_authorized": False,
        },
    )


def _token(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("ascii")).hexdigest()[:12]


def _bound_path(
    raw_path: str,
    expected_hash: str,
    canonical_root: Path,
) -> Path:
    path = Path(raw_path).resolve()
    try:
        path.relative_to(canonical_root.resolve())
    except ValueError as error:
        raise PreprocessorError(
            "D3 selected content escaped canonical root"
        ) from error
    if sha256(path) != expected_hash:
        raise PreprocessorError("D3 selected content hash mismatch")
    return path


def _frame_map(index: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = index.get("frames", [])
    indices = [int(item["normalized_index"]) for item in frames]
    if indices != list(range(13)):
        raise PreprocessorError("D3 content frames must be ordered 0..12")
    forbidden = {
        "position_m",
        "quaternion_xyzw",
        "depth_values",
        "semantic_values",
    }
    if any(forbidden & set(item) for item in frames):
        raise PreprocessorError("D3 index embeds pose or media content")
    return {int(item["normalized_index"]): item for item in frames}


def _load_pose(
    frame: dict[str, Any],
    canonical_root: Path,
) -> dict[str, Any]:
    path = _bound_path(
        str(frame["pose"]["path"]),
        str(frame["pose"]["sha256"]),
        canonical_root,
    )
    value = load_json(path)
    if (
        value.get("schema") != POSE_SLICE_SCHEMA
        or int(value.get("normalized_index", -1))
        != int(frame["normalized_index"])
        or int(value.get("source_frame_index", -1))
        != int(frame["source_frame_index"])
        or value.get("tracking_state") != "TrackingState.READY"
        or not {"position_m", "quaternion_xyzw"}
        <= set(value.get("binding", {}))
    ):
        raise PreprocessorError("D3 selected pose slice identity mismatch")
    return value["binding"]


def _load_current(
    frame: dict[str, Any],
    camera: dict[str, Any],
    canonical_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    depth_path = _bound_path(
        str(frame["depth"]["path"]),
        str(frame["depth"]["sha256"]),
        canonical_root,
    )
    mask_path = _bound_path(
        str(frame["mask"]["path"]),
        str(frame["mask"]["sha256"]),
        canonical_root,
    )
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    return (
        _read_depth(depth_path, width, height),
        _read_semantic_class(mask_path, width, height),
    )


def _load_selected_content(
    source: dict[str, Any],
    selector: dict[str, Any],
    layout: Any,
    root: Path,
) -> dict[str, Any]:
    index_path = Path(
        layout["content_index"]
        if isinstance(layout, dict)
        else layout.content_index
    ).resolve()
    if (
        sha256(index_path)
        != selector["source_authority_and_content_hashes"][
            "content_index_sha256"
        ]
    ):
        raise PreprocessorError("D3 selected content index hash mismatch")
    index = load_json(index_path)
    if (
        index.get("schema") != CONTENT_INDEX_SCHEMA
        or index.get("terminal") != CONTENT_READY
        or int(index.get("slot_index", -1))
        != int(source["d3_roster_slot_index"])
        or str(index.get("session_id")) != str(source["session_id"])
        or index.get("download_counts")
        != {"pose_csv": 1, "rgb": 0, "mask": 11, "depth": 11}
        or index.get("rgb_bytes_downloaded_or_read") is not False
    ):
        raise PreprocessorError("D3 selected content index identity mismatch")
    _bound_path(str(index_path), sha256(index_path), root)
    return index


def load_future_blind_inputs(
    index: dict[str, Any],
    canonical_root: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    frames = _frame_map(index)
    pose_cache = {
        index_value: _load_pose(frames[index_value], canonical_root)
        for index_value in range(0, 9)
    }
    media_cache = {
        index_value: _load_current(
            frames[index_value],
            index["camera"],
            canonical_root,
        )
        for index_value in ANCHORS
    }
    return pose_cache, media_cache


def _write_points(path: Path, points: np.ndarray) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError(f"D3 prediction points already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        np.save(
            handle,
            points.astype("<f8", copy=False),
            allow_pickle=False,
        )
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)


def run_preprocessor(contract_path: Path) -> dict[str, Any]:
    context = validate_execution_contract(
        contract_path,
        IMPLEMENTATION_KEY,
        Path(__file__),
        verify_git=True,
    )
    root = Path(context["root"]).resolve()
    if root != (
        Path(__file__).resolve().parents[3] / SCREENING_ROOT_RELATIVE
    ).resolve():
        raise PreprocessorError("D3 screening root is noncanonical")
    paths = _prediction_paths(root)
    if paths["completion"].exists() or paths["failure"].exists():
        raise FileExistsError(
            "D3 preprocessor terminal artifact already exists"
        )
    if paths["root"].exists() and any(paths["root"].iterdir()):
        error = PreprocessorError(
            "prior preprocessor attempt is incomplete; "
            "failure frozen without reopening selected inputs"
        )
        _freeze_preprocessor_failure(
            paths,
            context,
            error,
            selection_sha256=None,
        )
        raise error
    selection_path = Path(aggregate_paths(root)["selection"]).resolve()
    selection = validate_selection(
        selection_path,
        context["slots"],
        sha256(context["contract_path"]),
        context["roster_sha256"],
        context["carry_forward_authority"],
    )
    selected = selection["selected_sources"]
    if len(selected) != 6:
        raise PreprocessorError("D3 selection must contain exactly six sources")
    selection_sha256 = sha256(selection_path)
    write_json_exclusive_fsync(
        paths["attempt"],
        {
            "schema": ATTEMPT_SCHEMA,
            "status": ATTEMPT_STATUS,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "contract_sha256": sha256(context["contract_path"]),
            "roster_sha256": context["roster_sha256"],
            "selection_sha256": selection_sha256,
            "current_or_history_opened_before_attempt": False,
            "future_pose_depth_or_mask_authorized": False,
            "sealed_payload_open_authorized": False,
            "second_preprocessor_run_authorized": False,
        },
    )
    parameters = field_parameters(context["g0"], context["mechanics"])
    receipts: list[dict[str, Any]] = []
    try:
        for selected_row in selected:
            slot_index = int(selected_row["slot_index"])
            source = context["slots"][slot_index - 1]
            layout = slot_layout(root, source)
            selector_path = Path(
                layout["selector"]
                if isinstance(layout, dict)
                else layout.selector
            ).resolve()
            if sha256(selector_path) != str(
                selected_row["selector_sha256"]
            ):
                raise PreprocessorError("D3 selected selector hash mismatch")
            selector = load_json(selector_path)
            index = _load_selected_content(
                source,
                selector,
                layout,
                root,
            )
            pose_cache, media_cache = load_future_blind_inputs(
                index,
                root,
            )
            source_for_preprocessor = {
                **source,
                "frames": index["frames"],
            }
            source_root = (
                paths["root"] / "s" / _token(str(source["session_id"]))
            )
            for anchor in ANCHORS:
                depth, semantic = media_cache[anchor]
                record, points = preprocess_anchor(
                    source_for_preprocessor,
                    anchor,
                    pose_cache[anchor - 2],
                    pose_cache[anchor],
                    depth,
                    semantic,
                    parameters,
                )
                record = {
                    **record,
                    "schema": PREDICTION_SCHEMA,
                    "contract_sha256": sha256(context["contract_path"]),
                    "selection_sha256": sha256(selection_path),
                    "qualification_selector_sha256": sha256(
                        selector_path
                    ),
                    "sealed_payload_read": False,
                    "future_pose_depth_or_mask_read": False,
                }
                record_path = source_root / f"anchor-{anchor}.json"
                points_path = source_root / f"anchor-{anchor}.points.npy"
                _write_points(points_path, points)
                record["points"] = {
                    "path": str(points_path.resolve()),
                    "sha256": sha256(points_path),
                    "shape": list(points.shape),
                    "dtype": "float64",
                }
                write_json_exclusive_fsync(record_path, record)
                receipts.append(
                    {
                        "session_id": str(source["session_id"]),
                        "slot_index": slot_index,
                        "anchor_normalized_index": anchor,
                        "path": str(record_path.resolve()),
                        "sha256": sha256(record_path),
                        "points_sha256": sha256(points_path),
                    }
                )
            del pose_cache, media_cache
        if (
            len(receipts) != EXPECTED_ANCHOR_COUNT
            or sum(2 for _ in receipts)
            != EXPECTED_HORIZON_RECORD_COUNT
        ):
            raise PreprocessorError("D3 prediction record counts mismatch")
        completion = {
            "schema": COMPLETION_SCHEMA,
            "terminal": COMPLETION_TERMINAL,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "contract_sha256": sha256(context["contract_path"]),
            "roster_sha256": context["roster_sha256"],
            "selection_sha256": sha256(selection_path),
            "preprocessor_attempt_sha256": sha256(paths["attempt"]),
            "prediction_record_count": len(receipts),
            "anchor_horizon_record_count": (
                EXPECTED_HORIZON_RECORD_COUNT
            ),
            "records": receipts,
            "all_records_durable_before_sealed_truth_open": True,
            "sealed_payload_read": False,
            "future_pose_depth_or_mask_read": False,
            "effect_evaluator_authorized": True,
            "second_preprocessor_run_authorized": False,
        }
        write_json_exclusive_fsync(paths["completion"], completion)
        return completion
    except BaseException as error:
        _freeze_preprocessor_failure(
            paths,
            context,
            error,
            selection_sha256=selection_sha256,
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run_preprocessor(args.contract)
    print(json.dumps({"terminal": result["terminal"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
