#!/usr/bin/env python3
"""Consume exactly the next locked D3-Q0 screening slot.

The qualifier is reference-and-support only.  It downloads one pose CSV and
the normalized depth/mask frames 2..12 for the next roster slot, never RGB.
It computes the exact D2 persistence/advected *support* masks and future-truth
signed clearance, but it never computes either prediction arm's clearance.

The per-cell support/truth payload is written and fsynced before a closed-list
selector receipt.  A slot attempt is durable before the first content request;
after that point every failure consumes the slot and cannot be retried.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_sanpo_sequence_evalset import download, media_url  # noqa: E402
from stage_c_d2_mechanics_common import (  # noqa: E402
    ANCHORS,
    HEIGHTS,
    HORIZONS,
    basis_receipt,
    compute_field,
    compute_known,
    compute_points,
    field_parameters,
    fit_current_plane,
    nullable_field,
    predicted_bases,
)
from stage_c_d3_q0_common import (  # noqa: E402
    BUDGET_TERMINAL,
    CARRY_FORWARD_SCHEMA,
    CARRY_FORWARD_TERMINAL,
    FAILURE_SCHEMA,
    FIRST_ACTIVE_SLOT_INDEX,
    MAXIMUM_NEWLY_OPENED_SLOTS,
    QUALIFICATION_TERMINAL,
    SCREENING_ROOT_RELATIVE,
    SCREENING_ATTEMPT_SCHEMA,
    SCREENING_ATTEMPT_STATUS,
    SELECTOR_NOT_QUALIFIED,
    SELECTOR_QUALIFIED,
    SELECTOR_SCHEMA,
    SLOT_ATTEMPT_SCHEMA,
    SLOT_ATTEMPT_STATUS,
    aggregate_paths,
    canonical_json_sha256,
    durable_json_sha256,
    load_json,
    preserve_temporary_artifact,
    scan_screening_state,
    sha256,
    slot_layout,
    validate_execution_contract,
    validate_carry_forward,
    validate_screening_attempt,
    write_json_exclusive_fsync,
)
from verify_sanpo_pose_geometry_authority import (  # noqa: E402
    _read_depth,
    _read_semantic_class,
)


CONTENT_INDEX_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_slot_content_index"
)
CONTENT_READY = "D3_Q0_SLOT_POSE_DEPTH_MASK_CONTENT_READY"
POSE_SLICE_SCHEMA = "blindassist_hftf_stage_c_d3_q0_pose_slice"
SEALED_SCHEMA = "blindassist_hftf_stage_c_d3_q0_sealed_truth_payload"
SEALED_STATUS = (
    "D3_Q0_SEALED_TRUTH_AND_SUPPORT_PAYLOAD_FSYNCED_BEFORE_SELECTOR"
)
FAILURE_TERMINAL = (
    "D3_QUALIFICATION_SLOT_NOT_EVALUABLE_CONSUME_SLOT_CONTINUE_FROZEN_ORDER"
)
IMPLEMENTATION_KEY = "next_slot_runner"
ARM_NAMES = (
    "CURRENT_FIELD_PERSISTENCE",
    "HISTORY_CAUSAL_ADVECTED_CURRENT_FIELD",
)
PERSISTENCE, ADVECTED = ARM_NAMES
CELL_DENOMINATOR = 252
TOLERANCE = 1e-12


class SlotExecutionError(ValueError):
    """A frozen D3 slot cannot be executed or verified."""


def _layout_path(layout: Any, name: str) -> Path:
    value = (
        layout[name]
        if isinstance(layout, dict)
        else getattr(layout, name)
    )
    return Path(value).resolve()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - object receipt uses GCS MD5
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _object_receipt(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SlotExecutionError(f"{label} receipt is not an object")
    required = ("name", "generation", "size", "md5_base64")
    if any(not value.get(key) for key in required):
        raise SlotExecutionError(f"{label} receipt is incomplete")
    try:
        generation = str(int(value["generation"]))
        size = int(value["size"])
        base64.b64decode(str(value["md5_base64"]), validate=True)
    except (TypeError, ValueError) as error:
        raise SlotExecutionError(
            f"{label} generation, size, or MD5 is invalid"
        ) from error
    if int(generation) <= 0 or size <= 0:
        raise SlotExecutionError(f"{label} generation or size is nonpositive")
    return {
        "name": str(value["name"]),
        "generation": generation,
        "size": size,
        "md5_base64": str(value["md5_base64"]),
    }


def _verify_local(path: Path, receipt: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["size"])
        or _md5_base64(path) != str(receipt["md5_base64"])
    ):
        raise SlotExecutionError(f"{label} size or MD5 mismatch")


def _download_verified(
    receipt_value: Any,
    target: Path,
    retries: int,
    *,
    downloader: Callable[[str, Path, int], None] = download,
) -> dict[str, Any]:
    receipt = _object_receipt(receipt_value, str(target))
    if target.exists() or target.with_suffix(target.suffix + ".tmp").exists():
        raise SlotExecutionError(f"refusing to reuse content path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    downloader(
        media_url(receipt["name"], receipt["generation"]),
        target,
        retries,
    )
    _verify_local(target, receipt, str(target))
    return receipt


def _modality_receipt(
    source: dict[str, Any],
    modality: str,
    normalized_index: int,
) -> dict[str, Any]:
    selected = [int(value) for value in source["selected_source_frames"]]
    source_index = selected[normalized_index]
    values = source["media_object_listing_receipts"][modality][
        "required_frame_receipts"
    ]
    by_index = {int(value["frame_index"]): value for value in values}
    if source_index not in by_index:
        raise SlotExecutionError(
            f"{modality} selected frame receipt is absent"
        )
    return _object_receipt(
        by_index[source_index],
        f"{modality} normalized frame {normalized_index}",
    )


def planned_downloads(
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the exact content plan: one pose, 11 masks, 11 depths, no RGB."""
    selected = [int(value) for value in source["selected_source_frames"]]
    if len(selected) != 13:
        raise SlotExecutionError("D3 selected timeline must contain 13 frames")
    plan: list[dict[str, Any]] = [
        {
            "kind": "pose",
            "normalized_index": None,
            "source_frame_index": None,
            "relative_path": "content/pose.csv",
            "receipt": _object_receipt(
                source["camera_pose_object_receipt"],
                "camera pose CSV",
            ),
        }
    ]
    for normalized_index in range(2, 13):
        source_frame_index = selected[normalized_index]
        plan.extend(
            (
                {
                    "kind": "mask",
                    "normalized_index": normalized_index,
                    "source_frame_index": source_frame_index,
                    "relative_path": (
                        f"content/m/{normalized_index:02x}.png"
                    ),
                    "receipt": _modality_receipt(
                        source,
                        "mask",
                        normalized_index,
                    ),
                },
                {
                    "kind": "depth",
                    "normalized_index": normalized_index,
                    "source_frame_index": source_frame_index,
                    "relative_path": (
                        f"content/d/{normalized_index:02x}.f16.gz"
                    ),
                    "receipt": _modality_receipt(
                        source,
                        "depth",
                        normalized_index,
                    ),
                },
            )
        )
    if (
        sum(item["kind"] == "pose" for item in plan) != 1
        or sum(item["kind"] == "mask" for item in plan) != 11
        or sum(item["kind"] == "depth" for item in plan) != 11
        or any(item["kind"] == "rgb" for item in plan)
    ):
        raise SlotExecutionError("D3 content plan violated the frozen budget")
    return plan


def _parse_pose_slices(
    pose_path: Path,
    source: dict[str, Any],
    slot_root: Path,
) -> list[dict[str, Any]]:
    with pose_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pose_rows = list(csv.DictReader(handle))
    selected = [int(value) for value in source["selected_source_frames"]]
    pose_sha = sha256(pose_path)
    output: list[dict[str, Any]] = []
    for normalized_index, source_frame_index in enumerate(selected):
        if source_frame_index < 0 or source_frame_index >= len(pose_rows):
            raise SlotExecutionError("selected pose row index is invalid")
        row = pose_rows[source_frame_index]
        try:
            position = [
                float(row[key]) for key in ("pos_x", "pos_y", "pos_z")
            ]
            quaternion = [
                float(row[key]) for key in ("q_x", "q_y", "q_z", "q_w")
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise SlotExecutionError("selected pose values are invalid") from error
        norm = math.sqrt(sum(value * value for value in quaternion))
        if (
            row.get("tracking_state") != "TrackingState.READY"
            or not all(math.isfinite(value) for value in position)
            or not all(math.isfinite(value) for value in quaternion)
            or abs(norm - 1.0) > 1e-3
        ):
            raise SlotExecutionError(
                "selected pose row fails frozen authority checks"
            )
        path = slot_root / "content" / "p" / f"{normalized_index:02x}.json"
        value = {
            "schema": POSE_SLICE_SCHEMA,
            "normalized_index": normalized_index,
            "source_frame_index": source_frame_index,
            "tracking_state": "TrackingState.READY",
            "source_pose_csv_sha256": pose_sha,
            "binding": {
                "position_m": position,
                "quaternion_xyzw": quaternion,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive_fsync(path, value)
        output.append(
            {
                "normalized_index": normalized_index,
                "source_frame_index": source_frame_index,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    return output


def materialize_content(
    source: dict[str, Any],
    layout: Any,
    retries: int,
    *,
    downloader: Callable[[str, Path, int], None] = download,
) -> dict[str, Any]:
    """Download and index the one-slot pose/depth/mask package."""
    slot_root = _layout_path(layout, "slot_root")
    plan = planned_downloads(source)
    downloaded: list[dict[str, Any]] = []
    for item in plan:
        target = slot_root / str(item["relative_path"])
        receipt = _download_verified(
            item["receipt"],
            target,
            retries,
            downloader=downloader,
        )
        downloaded.append(
            {
                "kind": item["kind"],
                "normalized_index": item["normalized_index"],
                "source_frame_index": item["source_frame_index"],
                "path": str(target.resolve()),
                "sha256": sha256(target),
                "object_receipt": receipt,
            }
        )
    pose_download = next(
        item for item in downloaded if item["kind"] == "pose"
    )
    pose_slices = _parse_pose_slices(
        Path(pose_download["path"]),
        source,
        slot_root,
    )
    by_key = {
        (str(item["kind"]), item["normalized_index"]): item
        for item in downloaded
    }
    frames: list[dict[str, Any]] = []
    for normalized_index, pose in enumerate(pose_slices):
        row: dict[str, Any] = {
            "normalized_index": normalized_index,
            "source_frame_index": pose["source_frame_index"],
            "manifest_id": (
                f"d3q0_{source['session_id']}_{normalized_index:02d}"
            ),
            "pose": {
                "path": pose["path"],
                "sha256": pose["sha256"],
            },
        }
        if normalized_index >= 2:
            for modality in ("mask", "depth"):
                item = by_key[(modality, normalized_index)]
                row[modality] = {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "object_receipt": item["object_receipt"],
                }
        frames.append(row)
    index = {
        "schema": CONTENT_INDEX_SCHEMA,
        "terminal": CONTENT_READY,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "slot_index": int(source["d3_roster_slot_index"]),
        "session_id": str(source["session_id"]),
        "source_fps": float(source["source_fps"]),
        "selected_source_frames": source["selected_source_frames"],
        "camera": source["camera"],
        "pose_csv": {
            "path": pose_download["path"],
            "sha256": pose_download["sha256"],
            "object_receipt": pose_download["object_receipt"],
        },
        "frames": frames,
        "download_counts": {
            "pose_csv": 1,
            "rgb": 0,
            "mask": 11,
            "depth": 11,
        },
        "pose_rows_materialized": 13,
        "rgb_bytes_downloaded_or_read": False,
        "media_decoded_for_ranking": False,
        "candidate_arm_clearance_computed": False,
        "effect_metric_computed": False,
    }
    index_path = _layout_path(layout, "content_index")
    write_json_exclusive_fsync(index_path, index)
    return index


def _bound_json(path: Path, expected_hash: str) -> dict[str, Any]:
    if sha256(path) != expected_hash:
        raise SlotExecutionError(f"bound JSON hash mismatch: {path}")
    return load_json(path)


def load_slot_content(
    index: dict[str, Any],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    """Read each pose once and each required depth/mask frame once."""
    pose: dict[int, dict[str, Any]] = {}
    media: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    camera = index["camera"]
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    for frame in index["frames"]:
        normalized_index = int(frame["normalized_index"])
        pose_path = Path(str(frame["pose"]["path"])).resolve()
        pose_value = _bound_json(pose_path, str(frame["pose"]["sha256"]))
        if (
            pose_value.get("schema") != POSE_SLICE_SCHEMA
            or int(pose_value.get("normalized_index", -1))
            != normalized_index
            or pose_value.get("tracking_state")
            != "TrackingState.READY"
        ):
            raise SlotExecutionError("D3 pose slice identity mismatch")
        pose[normalized_index] = pose_value["binding"]
        if normalized_index >= 2:
            depth_path = Path(str(frame["depth"]["path"])).resolve()
            mask_path = Path(str(frame["mask"]["path"])).resolve()
            if (
                sha256(depth_path) != str(frame["depth"]["sha256"])
                or sha256(mask_path) != str(frame["mask"]["sha256"])
            ):
                raise SlotExecutionError("D3 depth/mask binding mismatch")
            media[normalized_index] = (
                _read_depth(depth_path, width, height),
                _read_semantic_class(mask_path, width, height),
            )
    if set(pose) != set(range(13)) or set(media) != set(range(2, 13)):
        raise SlotExecutionError("D3 slot content coverage mismatch")
    return pose, media


def _truth_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    truth = record["truth"]
    known = np.asarray(truth["known"], dtype=bool)
    nullable = np.asarray(truth["signed_clearance_m"], dtype=object)
    if known.shape != (2, 6, 6) or nullable.shape != (2, 6, 6):
        raise SlotExecutionError("D3 sealed truth shape mismatch")
    clearance = np.full((2, 6, 6), np.nan, dtype=np.float64)
    for index in np.ndindex((2, 6, 6)):
        if known[index]:
            if nullable[index] is None:
                raise SlotExecutionError("known truth cell is null")
            clearance[index] = float(nullable[index])
            if not math.isfinite(clearance[index]):
                raise SlotExecutionError("truth clearance is non-finite")
        elif nullable[index] is not None:
            raise SlotExecutionError("UNKNOWN truth cell became numeric")
    return known, clearance


def summarize_qualification(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute only frozen opportunity counts and booleans."""
    expected_keys = [
        (anchor, horizon)
        for anchor in ANCHORS
        for horizon in HORIZONS
    ]
    actual_keys = [
        (
            int(item["anchor_normalized_index"]),
            float(item["horizon_s"]),
        )
        for item in observations
    ]
    if actual_keys != expected_keys:
        raise SlotExecutionError("D3 sealed observation order mismatch")
    strata: list[dict[str, Any]] = []
    unknown_violations = sum(
        int(record.get("unknown_to_safe_violations", 0))
        for record in observations
    )
    for height_index, height in enumerate(HEIGHTS):
        for horizon in HORIZONS:
            common_count = 0
            risk_count = 0
            safe_count = 0
            for record in observations:
                if float(record["horizon_s"]) != horizon:
                    continue
                truth_known, truth_clearance = _truth_arrays(record)
                persistence = np.asarray(
                    record["support"][PERSISTENCE]["known"],
                    dtype=bool,
                )
                advected = np.asarray(
                    record["support"][ADVECTED]["known"],
                    dtype=bool,
                )
                if (
                    persistence.shape != (2, 6, 6)
                    or advected.shape != (2, 6, 6)
                ):
                    raise SlotExecutionError("D3 support shape mismatch")
                common = (
                    truth_known[height_index]
                    & persistence[height_index]
                    & advected[height_index]
                )
                target = truth_clearance[height_index][common]
                common_count += int(common.sum())
                risk_count += int(np.count_nonzero(target < 0.0))
                safe_count += int(np.count_nonzero(target >= 0.0))
            coverage = common_count / CELL_DENOMINATOR
            gates = {
                "common_known_coverage_at_least_0_10": (
                    coverage + TOLERANCE >= 0.1
                ),
                "known_risk_count_at_least_5": risk_count >= 5,
                "known_safe_count_at_least_20": safe_count >= 20,
            }
            strata.append(
                {
                    "height": height,
                    "horizon_s": horizon,
                    "denominator": CELL_DENOMINATOR,
                    "common_known_count": common_count,
                    "common_known_coverage": coverage,
                    "known_risk_count": risk_count,
                    "known_safe_count": safe_count,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
    qualified = (
        unknown_violations == 0
        and len(strata) == 4
        and all(item["passed"] for item in strata)
    )
    return {
        "strata": strata,
        "unknown_to_safe_violations": unknown_violations,
        "gates": {
            "all_four_strata_pass": (
                len(strata) == 4
                and all(item["passed"] for item in strata)
            ),
            "unknown_to_safe_violations_zero": (
                unknown_violations == 0
            ),
        },
        "qualified": qualified,
    }


def qualify_content(
    source: dict[str, Any],
    content_index: dict[str, Any],
    *,
    g0: dict[str, Any],
    mechanics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the sealed payload and closed selector content in memory."""
    pose, media = load_slot_content(content_index)
    parameters = field_parameters(g0, mechanics)
    camera = source["camera"]
    observations: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        depth, semantic = media[anchor]
        row = {
            "id": f"d3q0_{source['session_id']}_{anchor:02d}",
            "width": int(camera["image_width"]),
            "height": int(camera["image_height"]),
        }
        plane = fit_current_plane(
            depth,
            semantic,
            pose[anchor],
            camera,
            int(source["selected_source_frames"][anchor]),
        )
        current_basis, predicted, _motion = predicted_bases(
            pose[anchor - 2],
            pose[anchor],
            plane,
        )
        persistence_counts, persistence_known = compute_known(
            depth,
            semantic,
            row,
            pose[anchor],
            camera,
            current_basis,
            parameters,
        )
        for horizon in HORIZONS:
            advected_counts, advected_known = compute_known(
                depth,
                semantic,
                row,
                pose[anchor],
                camera,
                predicted[horizon],
                parameters,
            )
            future_index = anchor + (2 if horizon == 0.4 else 4)
            future_depth, future_semantic = media[future_index]
            future_row = {
                "id": (
                    f"d3q0_{source['session_id']}_{future_index:02d}"
                ),
                "width": int(camera["image_width"]),
                "height": int(camera["image_height"]),
            }
            truth_points = compute_points(
                future_depth,
                future_semantic,
                future_row,
                pose[future_index],
                camera,
                parameters,
            )
            truth_clearance = compute_field(
                truth_points,
                predicted[horizon],
                parameters,
            )
            truth_counts, truth_known = compute_known(
                future_depth,
                future_semantic,
                future_row,
                pose[future_index],
                camera,
                predicted[horizon],
                parameters,
            )
            observations.append(
                {
                    "anchor_normalized_index": anchor,
                    "horizon_s": horizon,
                    "future_normalized_index": future_index,
                    "predicted_basis": basis_receipt(predicted[horizon]),
                    "support": {
                        PERSISTENCE: {
                            "probe_pass_counts": (
                                persistence_counts.tolist()
                            ),
                            "known": persistence_known.tolist(),
                        },
                        ADVECTED: {
                            "probe_pass_counts": advected_counts.tolist(),
                            "known": advected_known.tolist(),
                        },
                    },
                    "truth": {
                        "probe_pass_counts": truth_counts.tolist(),
                        "known": truth_known.tolist(),
                        "signed_clearance_m": nullable_field(
                            truth_known,
                            truth_clearance,
                        ),
                    },
                    "unknown_to_safe_violations": 0,
                }
            )
            del truth_points
    summary = summarize_qualification(observations)
    payload = {
        "schema": SEALED_SCHEMA,
        "status": SEALED_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "slot_index": int(source["d3_roster_slot_index"]),
        "session_id": str(source["session_id"]),
        "content_index_sha256": canonical_json_sha256(content_index),
        "observation_count": len(observations),
        "observations": observations,
        "candidate_arm_clearance_computed_or_written": False,
        "mae_f1_confusion_delta_or_improvement_computed": False,
        "future_media_may_be_opened_again": False,
        "selector_or_aggregator_may_read_this_payload": False,
    }
    return payload, summary


def _selector(
    source: dict[str, Any],
    context: dict[str, Any],
    layout: Any,
    content_index: dict[str, Any],
    payload_sha256: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    content_path = _layout_path(layout, "content_index")
    authority_hashes = {
        "slot_attempt_sha256": sha256(
            _layout_path(layout, "attempt")
        ),
        "description_sha256": canonical_json_sha256(
            source["description_object"]
        ),
        "pose_sha256": canonical_json_sha256(
            source["camera_pose_object_receipt"]
        ),
        "rgb_receipts_sha256": canonical_json_sha256(
            source["media_object_listing_receipts"]["rgb"]
        ),
        "mask_receipts_sha256": canonical_json_sha256(
            source["media_object_listing_receipts"]["mask"]
        ),
        "depth_receipts_sha256": canonical_json_sha256(
            source["media_object_listing_receipts"]["depth"]
        ),
        "content_index_sha256": sha256(content_path),
        "sealed_payload_sha256": payload_sha256,
    }
    strata = []
    for row in summary["strata"]:
        gates = {
            "coverage": row["gates"][
                "common_known_coverage_at_least_0_10"
            ],
            "risk": row["gates"]["known_risk_count_at_least_5"],
            "safe": row["gates"]["known_safe_count_at_least_20"],
            "unknown_to_safe": (
                summary["unknown_to_safe_violations"] == 0
            ),
        }
        strata.append(
            {
                "height": row["height"],
                "horizon_s": row["horizon_s"],
                "denominator": row["denominator"],
                "common_known_count": row["common_known_count"],
                "common_known_coverage": row[
                    "common_known_coverage"
                ],
                "truth_risk_count": row["known_risk_count"],
                "truth_safe_count": row["known_safe_count"],
                "unknown_to_safe_violation_count": (
                    summary["unknown_to_safe_violations"]
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    return {
        "schema": SELECTOR_SCHEMA,
        "terminal": (
            SELECTOR_QUALIFIED
            if summary["qualified"]
            else SELECTOR_NOT_QUALIFIED
        ),
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": sha256(context["contract_path"]),
        "metadata_roster_sha256": context["roster_sha256"],
        "slot_index": int(source["d3_roster_slot_index"]),
        "session_id": str(source["session_id"]),
        "source_authority_and_content_hashes": authority_hashes,
        "strata": strata,
        "qualified": summary["qualified"],
    }


def publish_payload_then_selector(
    payload: dict[str, Any],
    selector_builder: Callable[[str], dict[str, Any]],
    layout: Any,
    *,
    writer: Callable[[Path, dict[str, Any]], None] = (
        write_json_exclusive_fsync
    ),
) -> dict[str, Any]:
    payload_path = _layout_path(layout, "sealed_payload")
    selector_path = _layout_path(layout, "selector")
    payload_sha256 = durable_json_sha256(payload)
    writer(payload_path, payload)
    selector = selector_builder(payload_sha256)
    writer(selector_path, selector)
    return selector


def _failure_receipt(
    source: dict[str, Any],
    context: dict[str, Any],
    layout: Any,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": FAILURE_SCHEMA,
        "terminal": FAILURE_TERMINAL,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": sha256(context["contract_path"]),
        "metadata_roster_sha256": context["roster_sha256"],
        "slot_attempt_sha256": sha256(
            _layout_path(layout, "attempt")
        ),
        "slot_index": int(source["d3_roster_slot_index"]),
        "session_id": str(source["session_id"]),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "slot_consumed": True,
        "rerun_authorized": False,
        "source_replacement_authorized": False,
    }


def seal_interrupted_slot(
    source: dict[str, Any],
    context: dict[str, Any],
    layout: Any,
) -> dict[str, Any]:
    failure = _failure_receipt(
        source,
        context,
        layout,
        SlotExecutionError(
            "prior durable slot attempt is incomplete; slot consumed"
        ),
    )
    failure_path = _layout_path(layout, "failure")
    preserve_temporary_artifact(failure_path)
    write_json_exclusive_fsync(failure_path, failure)
    return failure


def _open_global_and_slot_attempt(
    source: dict[str, Any],
    context: dict[str, Any],
    layout: Any,
) -> None:
    root = Path(context["root"]).resolve()
    paths = aggregate_paths(root)
    global_attempt = Path(paths["screening_attempt"]).resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    if not global_attempt.exists():
        write_json_exclusive_fsync(
            global_attempt,
            {
                "schema": SCREENING_ATTEMPT_SCHEMA,
                "status": SCREENING_ATTEMPT_STATUS,
                "workflow_profile": "THESIS_DEVELOPMENT",
                "contract_sha256": sha256(context["contract_path"]),
                "roster_sha256": context["roster_sha256"],
                "first_slot_index": FIRST_ACTIVE_SLOT_INDEX,
                "first_network_request_started": False,
                "slot_replacement_authorized": False,
                "budget_expansion_authorized": False,
            },
        )
    else:
        validate_screening_attempt(
            load_json(global_attempt),
            sha256(context["contract_path"]),
            context["roster_sha256"],
        )
    write_json_exclusive_fsync(
        _layout_path(layout, "attempt"),
        {
            "schema": SLOT_ATTEMPT_SCHEMA,
            "status": SLOT_ATTEMPT_STATUS,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "contract_sha256": sha256(context["contract_path"]),
            "roster_sha256": context["roster_sha256"],
            "slot_index": int(source["d3_roster_slot_index"]),
            "session_id": str(source["session_id"]),
            "internal_retries_per_request": context["retries"],
            "content_request_started": False,
            "slot_retry_authorized": False,
            "source_replacement_authorized": False,
            "candidate_arm_clearance_authorized": False,
            "effect_metric_authorized": False,
        },
    )


def _carry_forward_receipt(context: dict[str, Any]) -> dict[str, Any]:
    slot = context["slots"][0]
    authority = context["carry_forward_authority"]
    return {
        "schema": CARRY_FORWARD_SCHEMA,
        "terminal": CARRY_FORWARD_TERMINAL,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "q0_1_execution_contract_sha256": sha256(
            context["contract_path"]
        ),
        **authority,
        "original_slot_index": 1,
        "session_id": slot["session_id"],
        "burn_reason": "SCHEMA_INVALID_AFTER_MEDIA_SUPPORT_TRUTH_OPEN",
        "original_attempt_durable": True,
        "media_support_truth_opened": True,
        "selector_schema_valid": False,
        "selector_admitted": False,
        "permanently_burned": True,
        "counts_toward_original_40_budget": True,
        "counts_as_qualified": False,
        "counts_as_not_qualified": False,
        "counts_as_slot_failure": False,
        "reopen_authorized": False,
        "recompute_authorized": False,
        "rerun_authorized": False,
        "replacement_authorized": False,
        "first_remaining_original_slot": FIRST_ACTIVE_SLOT_INDEX,
        "maximum_remaining_original_slots": MAXIMUM_NEWLY_OPENED_SLOTS,
        "preserve_original_indices": True,
        "preserve_original_order": True,
        "sealed_payload_read": False,
        "invalid_selector_read": False,
        "outcome_fields_imported": False,
    }


def initialize_q0_1_control_plane(
    context: dict[str, Any],
) -> bool:
    root = Path(context["root"]).resolve()
    paths = aggregate_paths(root)
    screening_attempt = Path(paths["screening_attempt"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    created = False
    if not screening_attempt.exists():
        preserve_temporary_artifact(screening_attempt)
        write_json_exclusive_fsync(
            screening_attempt,
            {
                "schema": SCREENING_ATTEMPT_SCHEMA,
                "status": SCREENING_ATTEMPT_STATUS,
                "workflow_profile": "THESIS_DEVELOPMENT",
                "contract_sha256": sha256(context["contract_path"]),
                "roster_sha256": context["roster_sha256"],
                "first_slot_index": FIRST_ACTIVE_SLOT_INDEX,
                "first_network_request_started": False,
                "slot_replacement_authorized": False,
                "budget_expansion_authorized": False,
            },
        )
        created = True
    else:
        validate_screening_attempt(
            load_json(screening_attempt),
            sha256(context["contract_path"]),
            context["roster_sha256"],
        )
    slot = context["slots"][0]
    layout = slot_layout(root, slot)
    carry_path = _layout_path(layout, "carry_forward")
    if not carry_path.exists():
        if (
            layout["slot_root"].exists()
            and any(layout["slot_root"].iterdir())
        ):
            raise SlotExecutionError(
                "Q0.1 slot 1 contains non-carry-forward artifacts"
            )
        receipt = _carry_forward_receipt(context)
        write_json_exclusive_fsync(carry_path, receipt)
        created = True
    else:
        validate_carry_forward(
            load_json(carry_path),
            slot,
            sha256(context["contract_path"]),
            context["carry_forward_authority"],
        )
    return created


def _freeze_screening_invalid(
    context: dict[str, Any],
    error: BaseException,
) -> None:
    root = Path(context["root"]).resolve()
    invalid = aggregate_paths(root)["invalid"]
    root.mkdir(parents=True, exist_ok=True)
    if invalid.exists():
        return
    preserve_temporary_artifact(invalid)
    write_json_exclusive_fsync(
        invalid,
        {
            "schema": "blindassist_hftf_stage_c_d3_q0_1_screening_invalid",
            "terminal": "D3_QUALIFICATION_INVALID_STOP",
            "workflow_profile": "THESIS_DEVELOPMENT",
            "execution_contract_sha256": sha256(
                context["contract_path"]
            ),
            "metadata_roster_sha256": context["roster_sha256"],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_artifacts_preserved": True,
            "sealed_payload_read": False,
            "screening_rerun_authorized": False,
            "source_replacement_authorized": False,
            "budget_expansion_authorized": False,
        },
    )


def run_next_slot(
    contract_path: Path,
    *,
    retries: int,
) -> dict[str, Any]:
    context = validate_execution_contract(
        contract_path,
        IMPLEMENTATION_KEY,
        Path(__file__),
        verify_git=True,
    )
    if retries != int(context["retries"]):
        raise SlotExecutionError("retry count differs from frozen contract")
    root = Path(context["root"]).resolve()
    if root != (
        Path(__file__).resolve().parents[3] / SCREENING_ROOT_RELATIVE
    ).resolve():
        raise SlotExecutionError("D3 screening root is noncanonical")
    screening_paths = aggregate_paths(root)
    invalid = screening_paths["invalid"]
    if invalid.exists():
        raise SlotExecutionError(
            "D3 screening is terminal: D3_QUALIFICATION_INVALID_STOP"
        )
    try:
        initialized = initialize_q0_1_control_plane(context)
    except (KeyError, OSError, TypeError, ValueError) as error:
        _freeze_screening_invalid(context, error)
        raise
    if initialized:
        state = scan_screening_state(
            root,
            context["slots"],
            sha256(context["contract_path"]),
            context["roster_sha256"],
            context["carry_forward_authority"],
        )
        return {
            "control_terminal": CARRY_FORWARD_TERMINAL,
            "screening_initialized": True,
            "media_pose_support_truth_opened": False,
            "consumed_slot_count": state["consumed_count"],
            "newly_opened_slot_count": state["newly_opened_count"],
            "next_slot_index": state["next_slot"][
                "d3_roster_slot_index"
            ],
        }
    try:
        state = scan_screening_state(
            root,
            context["slots"],
            sha256(context["contract_path"]),
            context["roster_sha256"],
            context["carry_forward_authority"],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        _freeze_screening_invalid(context, error)
        raise
    if state.get("terminal") is not None:
        raise SlotExecutionError(
            f"D3 screening is terminal: {state['terminal']}"
        )
    interrupted = state.get("interrupted_slot")
    if interrupted is not None:
        source = interrupted
        layout = slot_layout(root, source)
        failure = seal_interrupted_slot(source, context, layout)
        after = scan_screening_state(
            root,
            context["slots"],
            sha256(context["contract_path"]),
            context["roster_sha256"],
            context["carry_forward_authority"],
        )
        return {
            "slot_terminal": failure["terminal"],
            "slot_index": failure["slot_index"],
            "qualified": False,
            "interrupted_attempt_consumed_without_reopen": True,
            "screening_terminal": after["terminal"],
            "aggregate_required": after["terminal"]
            in {QUALIFICATION_TERMINAL, BUDGET_TERMINAL},
        }
    next_slot = state.get("next_slot")
    if next_slot is None:
        raise SlotExecutionError("D3 state has no next slot")
    source = next_slot
    layout = slot_layout(root, source)
    _open_global_and_slot_attempt(source, context, layout)
    try:
        content = materialize_content(source, layout, retries)
        payload, summary = qualify_content(
            source,
            content,
            g0=context["g0"],
            mechanics=context["mechanics"],
        )
        payload.update(
            {
                "contract_sha256": sha256(context["contract_path"]),
                "roster_sha256": context["roster_sha256"],
                "slot_attempt_sha256": sha256(
                    _layout_path(layout, "attempt")
                ),
                "content_index_file_sha256": sha256(
                    _layout_path(layout, "content_index")
                ),
            }
        )
        selector = publish_payload_then_selector(
            payload,
            lambda payload_sha256: _selector(
                source,
                context,
                layout,
                content,
                payload_sha256,
                summary,
            ),
            layout,
        )
        after = scan_screening_state(
            root,
            context["slots"],
            sha256(context["contract_path"]),
            context["roster_sha256"],
            context["carry_forward_authority"],
        )
        return {
            "slot_terminal": selector["terminal"],
            "slot_index": selector["slot_index"],
            "qualified": selector["qualified"],
            "selector_sha256": sha256(_layout_path(layout, "selector")),
            "screening_terminal": after["terminal"],
            "aggregate_required": after["terminal"]
            in {QUALIFICATION_TERMINAL, BUDGET_TERMINAL},
        }
    except BaseException as error:
        selector_path = _layout_path(layout, "selector")
        failure_path = _layout_path(layout, "failure")
        if not selector_path.exists() and not failure_path.exists():
            failure = _failure_receipt(
                source,
                context,
                layout,
                error,
            )
            write_json_exclusive_fsync(failure_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--retries", type=int, required=True)
    args = parser.parse_args()
    result = run_next_slot(args.contract, retries=args.retries)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
