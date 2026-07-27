from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
from typing import Any, Sequence
import zipfile

import cv2
import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.rcle_minimal import (
    evaluation as r0_evaluation,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    SparseTrackResult,
    detect_fixed_grid_features,
    track_features,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0 import (
    OBSERVABLE_OCCLUSION,
    activated_cell_indices,
    classify_prior_survivors,
    merge_path_correspondences,
    observable_occlusion_centers,
    select_spatial_supplements,
    track_observable_points,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0 import (
    evaluation as observable_evaluation,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as cid_geometry,
)


PROTOCOL_ID = "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_CID_SIMS_FLOOR3_1"
SEQUENCE_ID = "floor3_1"
MAX_DT_SECONDS = Decimal("0.100")
WINDOW_SECONDS = Decimal("10.0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CONTRACT_OBJECT_REQUIRED")
    return value


def verify_contract(repo_root: Path, contract: dict[str, Any]) -> None:
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PROTOCOL_ID_MISMATCH")
    source = contract["source"]
    archive = repo_root / source["archive_path"]
    if not archive.is_file():
        raise ValueError("LOCAL_ARCHIVE_MISSING")
    if archive.stat().st_size != source["archive_bytes"]:
        raise ValueError("LOCAL_ARCHIVE_BYTES_MISMATCH")
    if sha256_file(archive) != source["archive_sha256"]:
        raise ValueError("LOCAL_ARCHIVE_SHA256_MISMATCH")
    for binding in contract["upstream_bindings"] + contract["algorithm_bindings"]:
        path = repo_root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"BOUND_FILE_MISMATCH:{binding['path']}")


def _shared_timestamps(names: Sequence[str]) -> list[Decimal]:
    color: set[Decimal] = set()
    depth: set[Decimal] = set()
    for raw in names:
        name = PurePosixPath(raw)
        if len(name.parts) != 3 or name.parts[0] != SEQUENCE_ID:
            continue
        if name.suffix.lower() != ".png":
            continue
        try:
            timestamp = Decimal(name.stem)
        except Exception:
            continue
        if name.parts[1] == "color":
            color.add(timestamp)
        elif name.parts[1] == "depth":
            depth.add(timestamp)
    return sorted(color & depth)


def _window_timestamps(
    shared: Sequence[Decimal], start: Decimal, end: Decimal
) -> list[Decimal]:
    rows = [timestamp for timestamp in shared if start <= timestamp < end]
    pairs = [
        (left, right)
        for left, right in zip(rows, rows[1:])
        if Decimal("0") < right - left <= MAX_DT_SECONDS
    ]
    if not pairs:
        raise ValueError("WINDOW_HAS_NO_PAIRS")
    if any(left != pairs[index - 1][1] for index, (left, _) in enumerate(pairs) if index):
        raise ValueError("WINDOW_PAIR_CONTINUITY_GAP")
    return [pairs[0][0], *(right for _, right in pairs)]


def _cache_manifest_payload(
    archive_sha256: str,
    control: dict[str, Any],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": "rcle.rgb_algorithm_cache.v1",
        "protocol_id": PROTOCOL_ID,
        "archive_sha256": archive_sha256,
        "control": control,
        "members": members,
        "member_count": len(members),
        "network_request_count": 0,
        "downloaded_bytes": 0,
    }
    payload["payload_sha256"] = canonical_sha(payload)
    return payload


def materialize_cache(
    archive_path: Path,
    cache_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_object(manifest_path)
        if manifest.get("payload_sha256") != canonical_sha(
            {key: value for key, value in manifest.items() if key != "payload_sha256"}
        ):
            raise ValueError("CACHE_MANIFEST_SELF_HASH_MISMATCH")
        for item in manifest["members"]:
            path = cache_dir / item["cache_relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != item["size_bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ValueError(f"CACHE_MEMBER_MISMATCH:{item['member_path']}")
        control = cache_dir / manifest["control"]["cache_relative_path"]
        if not control.is_file() or sha256_file(control) != manifest["control"]["sha256"]:
            raise ValueError("CACHE_CONTROL_MISMATCH")
        return manifest
    if cache_dir.exists():
        raise ValueError("PARTIAL_CACHE_DIRECTORY_EXISTS")
    cache_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        shared = _shared_timestamps(names)
        selected: dict[int, list[Decimal]] = {}
        for window in contract["windows"]:
            index = int(window["window_index"])
            timestamps = _window_timestamps(
                shared,
                Decimal(str(window["start_timestamp_s"])),
                Decimal(str(window["end_timestamp_s"])),
            )
            if len(timestamps) - 1 != int(window["candidate_pair_count"]):
                raise ValueError(f"WINDOW_PAIR_COUNT_MISMATCH:{index}")
            selected[index] = timestamps
        required = sorted({timestamp for rows in selected.values() for timestamp in rows})
        members: list[dict[str, Any]] = []
        for ordinal, timestamp in enumerate(required):
            member_path = f"{SEQUENCE_ID}/color/{timestamp}.png"
            raw = archive.read(member_path)
            relative = f"color/{timestamp}.png"
            target = cache_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                os.fspath(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            members.append(
                {
                    "archive_ordinal": ordinal,
                    "timestamp_s": str(timestamp),
                    "member_path": member_path,
                    "cache_relative_path": relative,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "window_indices": [
                        index for index, rows in selected.items() if timestamp in rows
                    ],
                }
            )
        control_member = contract["source"]["control_member"]
        control_raw = archive.read(control_member)
        control_relative = "groundtruth.txt"
        control_path = cache_dir / control_relative
        descriptor = os.open(
            os.fspath(control_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(control_raw)
            stream.flush()
            os.fsync(stream.fileno())
        control = {
            "member_path": control_member,
            "cache_relative_path": control_relative,
            "size_bytes": len(control_raw),
            "sha256": hashlib.sha256(control_raw).hexdigest(),
        }
    manifest = _cache_manifest_payload(
        contract["source"]["archive_sha256"], control, members
    )
    write_exclusive(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _decode_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"RGB_DECODE_INVALID:{path.name}") from error
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _intrinsic(contract: dict[str, Any]) -> np.ndarray:
    values = contract["source"]["intrinsics"]
    return np.asarray(
        (
            (float(values["fx"]), 0.0, float(values["cx"])),
            (0.0, float(values["fy"]), float(values["cy"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _homography(
    poses: Sequence[cid_geometry.PoseRow],
    previous_timestamp: Decimal,
    current_timestamp: Decimal,
    intrinsic: np.ndarray,
) -> np.ndarray:
    _, previous_quaternion = cid_geometry._interpolate_pose(
        poses, previous_timestamp
    )
    _, current_quaternion = cid_geometry._interpolate_pose(
        poses, current_timestamp
    )
    previous_rotation = cid_geometry._rotation(previous_quaternion)
    current_rotation = cid_geometry._rotation(current_quaternion)
    current_from_previous = current_rotation.T @ previous_rotation
    return intrinsic @ current_from_previous @ np.linalg.inv(intrinsic)


@dataclass
class PairState:
    survivors: SparseTrackResult | None = None
    dt_seconds: float | None = None


def _evaluate_pair(
    pair_index: int,
    previous: np.ndarray,
    current: np.ndarray,
    previous_timestamp: Decimal,
    current_timestamp: Decimal,
    homography: np.ndarray,
    protocol: dict[str, Any],
    state: PairState,
) -> dict[str, Any]:
    dt = float(current_timestamp - previous_timestamp)
    lk_parameters = protocol["sparse_lk"]
    affine_parameters = protocol["local_affine"]
    minimum_common = int(
        affine_parameters["minimum_common_evaluable_cells_per_pair"]
    )
    valid = np.full(previous.shape, 255, dtype=np.uint8)
    compensation = compensate_current_to_previous(
        current, valid, valid, homography
    )
    if compensation.overlap_fraction < 0.75:
        state.survivors = None
        state.dt_seconds = None
        return {
            "pair_index": pair_index,
            "previous_timestamp_s": float(previous_timestamp),
            "current_timestamp_s": float(current_timestamp),
            "dt_s": dt,
            "evaluable": False,
            "reason": "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75",
            "warp_overlap_fraction": compensation.overlap_fraction,
            "trigger": False,
        }

    initial_points = detect_fixed_grid_features(previous, valid, lk_parameters)
    raw_tracks = track_features(
        previous, current, initial_points, valid, lk_parameters
    )
    compensated_tracks = track_features(
        previous,
        compensation.image,
        initial_points,
        compensation.valid_mask,
        lk_parameters,
    )
    raw_cells = fit_fixed_grid_local_affine(
        raw_tracks, dt, previous.shape, affine_parameters
    )
    compensated_cells = fit_fixed_grid_local_affine(
        compensated_tracks, dt, previous.shape, affine_parameters
    )

    baseline_observable = track_observable_points(
        previous, current, initial_points, valid, valid, lk_parameters
    )
    activated: tuple[int, ...] = ()
    raw_carry = observable_evaluation._empty_diagnostics()
    compensated_carry = observable_evaluation._empty_diagnostics()
    raw_supplements = observable_evaluation._empty_diagnostics()
    compensated_supplements = observable_evaluation._empty_diagnostics()
    raw_classifications = np.empty((0,), dtype=object)
    compensated_classifications = np.empty((0,), dtype=object)
    supplement_points = np.empty((0, 2), dtype=np.float32)
    baseline_pool = observable_evaluation._baseline_source_union(
        initial_points, raw_tracks, compensated_tracks
    )

    if pair_index > 0:
        activated = activated_cell_indices(raw_cells, compensated_cells)
        if activated and state.survivors is not None and state.dt_seconds is not None:
            raw_carry_all = track_observable_points(
                previous,
                current,
                state.survivors.current_points,
                valid,
                valid,
                lk_parameters,
            )
            compensated_carry_all = track_observable_points(
                previous,
                compensation.image,
                state.survivors.current_points,
                valid,
                compensation.valid_mask,
                lk_parameters,
            )
            raw_classifications = classify_prior_survivors(
                state.survivors.current_points,
                state.survivors.current_points - state.survivors.previous_points,
                raw_carry_all,
                valid,
                prior_dt_seconds=state.dt_seconds,
                current_dt_seconds=dt,
            )
            compensated_classifications = classify_prior_survivors(
                state.survivors.current_points,
                state.survivors.current_points - state.survivors.previous_points,
                compensated_carry_all,
                compensation.valid_mask,
                prior_dt_seconds=state.dt_seconds,
                current_dt_seconds=dt,
            )
            carry_admitted = observable_evaluation._admit_shared_carried_points(
                raw_carry_all,
                compensated_carry_all,
                baseline_pool,
                raw_cells,
                activated,
            )
            raw_carry = observable_evaluation._select_diagnostics(
                raw_carry_all, carry_admitted
            )
            compensated_carry = observable_evaluation._select_diagnostics(
                compensated_carry_all, carry_admitted
            )
            exclusion_classes = np.where(
                (raw_classifications == OBSERVABLE_OCCLUSION)
                | (compensated_classifications == OBSERVABLE_OCCLUSION),
                OBSERVABLE_OCCLUSION,
                "",
            )
            exclusions = observable_occlusion_centers(
                state.survivors.current_points, exclusion_classes
            )
        else:
            exclusions = np.empty((0, 2), dtype=np.float32)
        if activated:
            existing = np.vstack((baseline_pool, raw_carry.initial_points))
            selected: list[np.ndarray] = []
            for cell_index in activated:
                points = select_spatial_supplements(
                    previous,
                    valid,
                    raw_cells[cell_index].region,
                    np.vstack((existing, *selected)) if selected else existing,
                    exclusions,
                )
                if points.size:
                    selected.append(points)
            if selected:
                supplement_points = np.ascontiguousarray(
                    np.vstack(selected).astype(np.float32)
                )
            raw_supplements = track_observable_points(
                previous,
                current,
                supplement_points,
                valid,
                valid,
                lk_parameters,
            )
            compensated_supplements = track_observable_points(
                previous,
                compensation.image,
                supplement_points,
                valid,
                compensation.valid_mask,
                lk_parameters,
            )
            managed_raw = merge_path_correspondences(
                raw_tracks, raw_carry, raw_supplements
            )
            managed_compensated = merge_path_correspondences(
                compensated_tracks, compensated_carry, compensated_supplements
            )
            managed_raw_cells = fit_fixed_grid_local_affine(
                managed_raw, dt, previous.shape, affine_parameters
            )
            managed_compensated_cells = fit_fixed_grid_local_affine(
                managed_compensated, dt, previous.shape, affine_parameters
            )
            raw_cells = observable_evaluation._splice_activated_cells(
                raw_cells, managed_raw_cells, activated
            )
            compensated_cells = observable_evaluation._splice_activated_cells(
                compensated_cells, managed_compensated_cells, activated
            )
            raw_tracks = managed_raw
            compensated_tracks = managed_compensated

    state.survivors = observable_evaluation._accepted_prior_tracks(
        (baseline_observable, raw_carry, raw_supplements)
    )
    state.dt_seconds = dt
    raw_values, compensated_values, common_indices = (
        r0_evaluation._common_cell_expansions(raw_cells, compensated_cells)
    )
    manager = {
        "baseline_only": pair_index == 0,
        "activated_cell_indices": list(activated),
        "spatial_supplement_count": int(supplement_points.shape[0]),
        "raw_observable_occlusion_count": int(
            np.count_nonzero(raw_classifications == OBSERVABLE_OCCLUSION)
        ),
        "compensated_observable_occlusion_count": int(
            np.count_nonzero(
                compensated_classifications == OBSERVABLE_OCCLUSION
            )
        ),
    }
    common = len(common_indices)
    base = {
        "pair_index": pair_index,
        "previous_timestamp_s": float(previous_timestamp),
        "current_timestamp_s": float(current_timestamp),
        "dt_s": dt,
        "warp_overlap_fraction": compensation.overlap_fraction,
        "raw_track_count": raw_tracks.valid_count,
        "compensated_track_count": compensated_tracks.valid_count,
        "common_cell_count": common,
        "support_manager": manager,
    }
    if common < minimum_common:
        reasons = Counter(
            cell.abstention_reason
            for cells in (raw_cells, compensated_cells)
            for cell in cells
            if not cell.evaluable and cell.abstention_reason
        )
        return {
            **base,
            "evaluable": False,
            "reason": "COMMON_GRID_SUPPORT_BELOW_5_OF_9",
            "cell_abstention_reasons": dict(sorted(reasons.items())),
            "trigger": False,
        }
    raw_median = float(np.median(raw_values))
    compensated_median = float(np.median(compensated_values))
    trigger_threshold = float(
        protocol["metrics"]["sign_accuracy_zero_band_per_s"]
    )
    return {
        **base,
        "evaluable": True,
        "reason": None,
        "common_cell_indices": common_indices,
        "raw_expansion_median_per_s": raw_median,
        "compensated_expansion_median_per_s": compensated_median,
        "raw_abs_expansion_median_per_s": float(np.median(np.abs(raw_values))),
        "compensated_abs_expansion_median_per_s": float(
            np.median(np.abs(compensated_values))
        ),
        "trigger_threshold_per_s": trigger_threshold,
        "trigger": bool(compensated_median > trigger_threshold),
    }


def _longest_trigger_run(
    rows: Sequence[dict[str, Any]],
) -> tuple[int, float]:
    longest_count = 0
    longest_duration = 0.0
    current_count = 0
    current_start: float | None = None
    for row in rows:
        if row["evaluable"] is True and row["trigger"] is True:
            if current_count == 0:
                current_start = float(row["previous_timestamp_s"])
            current_count += 1
            duration = float(row["current_timestamp_s"]) - float(current_start)
            if current_count > longest_count or (
                current_count == longest_count and duration > longest_duration
            ):
                longest_count = current_count
                longest_duration = duration
        else:
            current_count = 0
            current_start = None
    return longest_count, longest_duration


def evaluate_window(
    cache_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    window: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = Decimal(str(window["start_timestamp_s"]))
    end = Decimal(str(window["end_timestamp_s"]))
    members = [
        item
        for item in manifest["members"]
        if int(window["window_index"]) in item["window_indices"]
        and start <= Decimal(item["timestamp_s"]) < end
    ]
    members.sort(key=lambda item: Decimal(item["timestamp_s"]))
    timestamps = [Decimal(item["timestamp_s"]) for item in members]
    if len(timestamps) - 1 != int(window["candidate_pair_count"]):
        raise ValueError("CACHE_WINDOW_PAIR_COUNT_MISMATCH")
    frames = [
        _decode_gray(cache_dir / item["cache_relative_path"]) for item in members
    ]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("RGB_SHAPE_DRIFT")
    poses = cid_geometry._parse_poses(
        (cache_dir / manifest["control"]["cache_relative_path"]).read_bytes()
    )
    intrinsic = _intrinsic(contract)
    protocol = load_object(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "phase_a_synthetic_signal_audit_r0.json"
    )
    cv2.setRNGSeed(20260727 + int(window["window_index"]))
    cv2.setNumThreads(1)
    state = PairState()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (left, right) in enumerate(zip(timestamps, timestamps[1:])):
        if not (Decimal("0") < right - left <= MAX_DT_SECONDS):
            raise ValueError("CACHE_PAIR_DT_DRIFT")
        row = _evaluate_pair(
            index,
            frames[index],
            frames[index + 1],
            left,
            right,
            _homography(poses, left, right, intrinsic),
            protocol,
            state,
        )
        row["window_index"] = int(window["window_index"])
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(timestamps) - 1:
            elapsed = time.perf_counter() - started
            print(
                f"window={window['window_index']} completed={index + 1}/"
                f"{len(timestamps) - 1} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    evaluable = [row for row in rows if row["evaluable"] is True]
    triggered = [row for row in evaluable if row["trigger"] is True]
    abstentions = Counter(
        str(row["reason"]) for row in rows if row["evaluable"] is not True
    )
    longest_count, longest_duration = _longest_trigger_run(rows)
    first_delay = (
        float(triggered[0]["current_timestamp_s"]) - float(start)
        if triggered
        else None
    )
    summary = {
        "window_index": int(window["window_index"]),
        "role": window["role"],
        "window_start_s": float(start),
        "window_end_s": float(end),
        "geometry_median_signed_radial_expansion_per_s": window[
            "geometry_median_signed_radial_expansion_per_s"
        ],
        "candidate_pair_count": len(rows),
        "evaluable_pair_count": len(evaluable),
        "pair_coverage": len(evaluable) / len(rows),
        "abstention_count": len(rows) - len(evaluable),
        "abstention_reasons": dict(sorted(abstentions.items())),
        "median_compensated_expansion_per_s": (
            float(
                np.median(
                    [
                        row["compensated_expansion_median_per_s"]
                        for row in evaluable
                    ]
                )
            )
            if evaluable
            else None
        ),
        "trigger_threshold_per_s": 0.01,
        "trigger_count": len(triggered),
        "trigger_coverage_fixed_denominator": len(triggered) / len(rows),
        "trigger_coverage_evaluable": (
            len(triggered) / len(evaluable) if evaluable else None
        ),
        "first_trigger_delay_s": first_delay,
        "longest_consecutive_trigger_pair_count": longest_count,
        "longest_consecutive_trigger_duration_s": longest_duration,
        "runtime_s": time.perf_counter() - started,
    }
    return summary, rows


def run(
    repo_root: Path,
    contract_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_object(contract_path)
    verify_contract(repo_root, contract)
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_ALREADY_EXISTS")
    manifest = materialize_cache(
        repo_root / contract["source"]["archive_path"], cache_dir, contract
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for window in contract["windows"]:
        summary, rows = evaluate_window(cache_dir, manifest, contract, window)
        summaries.append(summary)
        all_rows.extend(rows)
    control, positive = summaries
    evaluable = all(
        summary["pair_coverage"]
        >= float(contract["algorithm"]["minimum_window_pair_coverage"])
        for summary in summaries
    )
    direction = bool(
        evaluable
        and positive["median_compensated_expansion_per_s"]
        > control["median_compensated_expansion_per_s"]
        and positive["trigger_coverage_fixed_denominator"]
        > control["trigger_coverage_fixed_denominator"]
    )
    terminal = (
        "DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID"
        if direction
        else (
            "DEVELOPMENT_SIGNAL_DIRECTION_NOT_SUPPORTED / VALID"
            if evaluable
            else "NOT_EVALUABLE / VALID"
        )
    )
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in all_rows
    )
    write_exclusive(output_dir / "pair_ledger.jsonl", ledger_text)
    result = {
        "schema_version": "rcle.rgb_algorithm_development_canary.result.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "authority": "REAL_RGB_ALGORITHM_DEVELOPMENT_CANARY_ONLY",
        "threshold_tuned": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "archive_sha256": contract["source"]["archive_sha256"],
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "contract_sha256": sha256_file(contract_path),
        "windows": summaries,
        "separation": {
            "positive_minus_control_median_compensated_expansion_per_s": (
                positive["median_compensated_expansion_per_s"]
                - control["median_compensated_expansion_per_s"]
            ),
            "positive_minus_control_trigger_coverage_fixed_denominator": (
                positive["trigger_coverage_fixed_denominator"]
                - control["trigger_coverage_fixed_denominator"]
            ),
            "direction_supported": direction,
            "performance_qualification": False,
            "independent_confirmation": False,
        },
        "pair_ledger_sha256": hashlib.sha256(ledger_text.encode("utf-8")).hexdigest(),
    }
    result["result_payload_sha256"] = canonical_sha(result)
    write_exclusive(
        output_dir / "result.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result

