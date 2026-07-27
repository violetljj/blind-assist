"""Formal two-phase disjoint-window holdout for CID-SIMS floor3_2."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import cv2
import numpy as np
import PIL

if __package__ in {None, ""}:
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
    )

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


PROTOCOL_ID = "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_CROSS_SEQUENCE_HOLDOUT_R0"
AUTHORITY = "CROSS_SEQUENCE_SAME_SOURCE_DEVELOPMENT_HOLDOUT_ONLY"
TOTAL_UNITS = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_exclusive_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    write_exclusive_bytes(path, payload)


def write_exclusive_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    write_exclusive_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def atomic_progress(
    path: Path,
    *,
    phase: str,
    completed_units: int,
    started: float,
    status: str,
    error: str | None = None,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    throughput = completed_units / elapsed
    remaining = max(TOTAL_UNITS - completed_units, 0)
    value: dict[str, Any] = {
        "schema_version": "rcle.disjoint_holdout.progress.v1",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "completed_units": completed_units,
        "total_units": TOTAL_UNITS,
        "throughput": throughput,
        "eta_seconds": remaining / throughput if throughput > 0 else None,
        "last_progress_at": _utc_now(),
        "status": status,
    }
    if error:
        value["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    os.replace(temporary, path)


def runtime_identity() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def verify_contract_and_lock(
    repo_root: Path, contract_path: Path, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_object(contract_path)
    lock = load_object(lock_path)
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PROTOCOL_ID_DRIFT")
    if contract.get("authority", {}).get("maximum_claim") != AUTHORITY:
        raise ValueError("AUTHORITY_DRIFT")
    expected_runtime = contract.get("runtime_identity")
    if runtime_identity() != expected_runtime:
        raise ValueError(
            f"RUNTIME_IDENTITY_DRIFT:{runtime_identity()}:{expected_runtime}"
        )
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("IMPLEMENTATION_LOCK_PROTOCOL")
    entries = lock.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("IMPLEMENTATION_LOCK_FILES")
    for entry in entries:
        path = repo_root / str(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    for binding in contract.get("immutable_algorithm_bindings", []):
        path = repo_root / str(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"ALGORITHM_BINDING_DRIFT:{binding['path']}")
    for binding_name in ("transport_lock", "transport_receipt"):
        path = repo_root / str(
            contract["source"][f"{binding_name}_path"]
        )
        if (
            not path.is_file()
            or sha256_file(path)
            != contract["source"][f"{binding_name}_sha256"]
        ):
            raise ValueError(f"{binding_name.upper()}_DRIFT")
    return contract, lock


def depth_timestamps(
    infos: Iterable[zipfile.ZipInfo], sequence_id: str
) -> list[Decimal]:
    prefix = f"{sequence_id}/depth/"
    values: set[Decimal] = set()
    seen: set[str] = set()
    for info in infos:
        name = info.filename
        if name in seen:
            raise ValueError("ZIP_MEMBER_NAME_DUPLICATE")
        seen.add(name)
        if not name.startswith(prefix) or not name.endswith(".png"):
            continue
        try:
            timestamp = Decimal(PurePosixPath(name).stem)
        except InvalidOperation as error:
            raise ValueError(f"DEPTH_TIMESTAMP_INVALID:{name}") from error
        if timestamp in values:
            raise ValueError(f"DEPTH_TIMESTAMP_DUPLICATE:{timestamp}")
        values.add(timestamp)
    return sorted(values)


def candidate_windows(
    timestamps: Sequence[Decimal], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rules = contract["geometry_only_selection"]
    if not timestamps:
        raise ValueError("NO_DEPTH_TIMESTAMPS")
    anchor = timestamps[0]
    duration = Decimal(rules["window_duration_s"])
    minimum_frames = int(rules["minimum_frames_per_window"])
    maximum_frames = int(rules["maximum_frames_per_window"])
    maximum_dt = Decimal(rules["maximum_pair_dt_s"])
    windows: list[dict[str, Any]] = []
    complete_window_count = int((timestamps[-1] - anchor) // duration)
    for index in range(complete_window_count):
        start = anchor + int(index) * duration
        end = start + duration
        rows = [timestamp for timestamp in timestamps if start <= timestamp < end]
        pairs = list(zip(rows, rows[1:]))
        identity_ok = minimum_frames <= len(rows) <= maximum_frames and all(
            Decimal("0") < right - left <= maximum_dt
            for left, right in pairs
        )
        windows.append(
            {
                "sequence_id": contract["source"]["sequence_id"],
                "window_index": int(index),
                "start_timestamp_s": str(start),
                "end_timestamp_s": str(end),
                "frame_timestamps_s": [str(value) for value in rows],
                "frame_count": len(rows),
                "pair_count": len(pairs),
                "identity_eligible": identity_ok,
                "identity_reason": (
                    None
                    if identity_ok
                    else "FRAME_COUNT_RANGE_OR_SOURCE_CONSECUTIVE_DT"
                ),
            }
        )
    return windows


def geometry_band(value: float) -> str:
    if value < 0.01:
        return "BELOW_TRIGGER_REFERENCE"
    if value < 0.05:
        return "WEAK_POSITIVE_RADIAL"
    return "POSITIVE_APPROACH_GEOMETRY"


def longest_band_run(
    rows: Sequence[dict[str, Any]], band: str
) -> tuple[int, float]:
    longest_count = 0
    longest_duration = Decimal("0")
    count = 0
    start: Decimal | None = None
    previous_pair: int | None = None
    for row in rows:
        contiguous = (
            previous_pair is not None
            and int(row["pair_index"]) == previous_pair + 1
        )
        if row.get("geometry_band") == band:
            if count == 0 or not contiguous:
                count = 0
                start = Decimal(str(row["previous_timestamp_s"]))
            count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - start
            if count > longest_count or (
                count == longest_count and duration > longest_duration
            ):
                longest_count = count
                longest_duration = duration
        else:
            count = 0
            start = None
        previous_pair = int(row["pair_index"])
    return longest_count, float(longest_duration)


def _normalize_geometry(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: record[key]
        for key in (
            "window_index",
            "pair_index",
            "previous_timestamp_s",
            "current_timestamp_s",
            "dt_s",
        )
    }
    if record.get("evaluable") is not True:
        result.update(
            geometry_evaluable=False,
            geometry_abstention_reason=str(
                record.get("reason", "NO_VALID_GEOMETRY_SAMPLES")
            ),
            geometry_band=None,
        )
        return result
    signed = float(record["median_signed_radial_expansion_per_s"])
    result.update(
        geometry_evaluable=True,
        geometry_abstention_reason=None,
        geometry_signed_radial_expansion_per_s=signed,
        geometry_radial_expansion_positive_fraction=float(
            record["radial_expansion_positive_fraction"]
        ),
        geometry_q90_time_normalized_parallax_rad_per_s=float(
            record["q90_time_normalized_parallax_rad_per_s"]
        ),
        geometry_band=geometry_band(signed),
    )
    return result


def compute_window_geometry(
    archive: zipfile.ZipFile,
    poses: Sequence[geometry.PoseRow],
    intrinsic: np.ndarray,
    window: Mapping[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    if window["identity_eligible"] is not True:
        return []
    timestamps = [Decimal(value) for value in window["frame_timestamps_s"]]
    tasks: list[tuple[Any, ...]] = []
    sequence_id = str(window["sequence_id"])
    for pair_index, (previous, current) in enumerate(
        zip(timestamps, timestamps[1:])
    ):
        base = {
            "window_index": int(window["window_index"]),
            "pair_index": pair_index,
            "previous_timestamp_s": float(previous),
            "current_timestamp_s": float(current),
            "dt_s": float(current - previous),
        }
        raw_depth = archive.read(f"{sequence_id}/depth/{previous}.png")
        tasks.append(
            (
                base,
                raw_depth,
                intrinsic,
                geometry._interpolate_pose(poses, previous),
                geometry._interpolate_pose(poses, current),
                float(current - previous),
            )
        )
    if workers == 1:
        evaluated = list(map(geometry._pair_worker, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(geometry._pair_worker, tasks))
    return [_normalize_geometry(record) for record, _ in evaluated]


def summarize_geometry_window(
    window: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    rules = contract["geometry_only_selection"]
    expected_pairs = int(window["pair_count"])
    if expected_pairs <= 0:
        expected_pairs = max(int(window["frame_count"]) - 1, 0)
    evaluable = [row for row in rows if row.get("geometry_evaluable") is True]
    counts = Counter(
        str(row["geometry_band"]) for row in evaluable
    )
    abstentions = Counter(
        str(row["geometry_abstention_reason"])
        for row in rows
        if row.get("geometry_evaluable") is not True
    )
    positive_count, positive_duration = longest_band_run(
        rows, "POSITIVE_APPROACH_GEOMETRY"
    )
    below_count, below_duration = longest_band_run(
        rows, "BELOW_TRIGGER_REFERENCE"
    )
    coverage = len(evaluable) / expected_pairs if expected_pairs else 0.0
    positive_fraction = (
        counts["POSITIVE_APPROACH_GEOMETRY"] / expected_pairs
        if expected_pairs
        else 0.0
    )
    below_fraction = (
        counts["BELOW_TRIGGER_REFERENCE"] / expected_pairs
        if expected_pairs
        else 0.0
    )
    positive_rule = rules["role_eligibility"]["positive_approach_window"]
    below_rule = rules["role_eligibility"]["below_trigger_reference_window"]
    coverage_ok = (
        window["identity_eligible"] is True
        and coverage >= float(rules["minimum_pair_geometry_coverage"])
    )
    positive_eligible = bool(
        coverage_ok
        and positive_fraction
        >= float(positive_rule["minimum_fixed_denominator_positive_fraction"])
        and positive_duration
        >= float(positive_rule["minimum_longest_positive_run_duration_s"])
    )
    below_eligible = bool(
        coverage_ok
        and below_fraction
        >= float(below_rule["minimum_fixed_denominator_below_fraction"])
        and below_duration
        >= float(below_rule["minimum_longest_below_run_duration_s"])
    )
    if positive_eligible and below_eligible:
        raise ValueError(f"GEOMETRY_ROLE_OVERLAP:{window['window_index']}")
    role = (
        "POSITIVE_APPROACH_WINDOW"
        if positive_eligible
        else (
            "BELOW_TRIGGER_REFERENCE_WINDOW"
            if below_eligible
            else "AMBIGUOUS_OR_INELIGIBLE"
        )
    )
    geometry_values = [
        float(row["geometry_signed_radial_expansion_per_s"])
        for row in evaluable
    ]
    first_positive = next(
        (
            float(row["current_timestamp_s"])
            for row in rows
            if row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY"
        ),
        None,
    )
    return {
        "window_index": int(window["window_index"]),
        "start_timestamp_s": window["start_timestamp_s"],
        "end_timestamp_s": window["end_timestamp_s"],
        "frame_count": int(window["frame_count"]),
        "candidate_pair_count": int(window["pair_count"]),
        "identity_eligible": bool(window["identity_eligible"]),
        "identity_reason": window["identity_reason"],
        "geometry_evaluable_pair_count": len(evaluable),
        "geometry_pair_coverage_fixed_denominator": coverage,
        "geometry_abstention_count": max(expected_pairs - len(evaluable), 0),
        "geometry_abstention_reasons": dict(sorted(abstentions.items())),
        "geometry_band_counts": dict(sorted(counts.items())),
        "positive_fraction_fixed_denominator": positive_fraction,
        "below_fraction_fixed_denominator": below_fraction,
        "longest_positive_run_pair_count": positive_count,
        "longest_positive_run_duration_s": positive_duration,
        "longest_below_run_pair_count": below_count,
        "longest_below_run_duration_s": below_duration,
        "median_signed_radial_expansion_per_s": (
            float(statistics.median(geometry_values))
            if geometry_values
            else None
        ),
        "first_positive_geometry_timestamp_s": first_positive,
        "role": role,
    }


def select_windows(
    summaries: Sequence[dict[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selection = contract["geometry_only_selection"]["selection"]
    required_positive = int(selection["required_positive_windows"])
    required_below = int(selection["required_below_reference_windows"])
    minimum_separation = Decimal(
        selection["minimum_selected_start_separation_s"]
    )
    positive = [
        item for item in summaries if item["role"] == "POSITIVE_APPROACH_WINDOW"
    ]
    below = [
        item
        for item in summaries
        if item["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"
    ]
    feasible: list[tuple[tuple[int, ...], list[dict[str, Any]]]] = []
    for positive_rows in itertools.combinations(positive, required_positive):
        for below_rows in itertools.combinations(below, required_below):
            combined = sorted(
                (*positive_rows, *below_rows),
                key=lambda item: int(item["window_index"]),
            )
            starts = [Decimal(item["start_timestamp_s"]) for item in combined]
            if all(
                right - left >= minimum_separation
                for left, right in zip(starts, starts[1:])
            ):
                indices = tuple(int(item["window_index"]) for item in combined)
                feasible.append((indices, combined))
    if not feasible:
        return []
    feasible.sort(key=lambda item: item[0])
    return feasible[0][1]


def selected_member_identity(
    archive: zipfile.ZipFile,
    selected: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    selection_sha256: str,
) -> dict[str, Any]:
    by_index = {int(item["window_index"]): item for item in windows}
    info_by_name = {info.filename: info for info in archive.infolist()}
    members: list[dict[str, Any]] = []
    for summary in selected:
        window = by_index[int(summary["window_index"])]
        for timestamp in window["frame_timestamps_s"]:
            name = f"floor3_2/color/{timestamp}.png"
            info = info_by_name.get(name)
            if info is None:
                raise ValueError(f"SELECTED_RGB_MEMBER_MISSING:{name}")
            members.append(
                {
                    "window_index": int(summary["window_index"]),
                    "timestamp_s": timestamp,
                    "archive_member": name,
                    "crc32": f"{info.CRC:08x}",
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
            )
    value = {
        "schema_version": "rcle.disjoint_holdout.rgb_identity.v1",
        "protocol_id": PROTOCOL_ID,
        "selection_sha256": selection_sha256,
        "selected_window_indices": [
            int(item["window_index"]) for item in selected
        ],
        "members": members,
    }
    value["identity_payload_sha256"] = canonical_sha(value)
    return value


def materialize_selected_rgb_cache(
    archive: zipfile.ZipFile,
    identity: Mapping[str, Any],
    cache_dir: Path,
    pose_member: str,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, Any]] = []
    for item in identity["members"]:
        raw = archive.read(item["archive_member"])
        if len(raw) != int(item["uncompressed_bytes"]):
            raise ValueError("SELECTED_RGB_SIZE_DRIFT")
        if f"{zipfile.crc32(raw) & 0xffffffff:08x}" != item["crc32"]:
            raise ValueError("SELECTED_RGB_CRC_DRIFT")
        relative = Path("color") / f"{item['timestamp_s']}.png"
        target = cache_dir / relative
        write_exclusive_bytes(target, raw)
        members.append(
            {
                **item,
                "cache_relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "window_indices": [int(item["window_index"])],
            }
        )
    pose_raw = archive.read(pose_member)
    write_exclusive_bytes(cache_dir / "groundtruth.txt", pose_raw)
    manifest = {
        "schema_version": "rcle.disjoint_holdout.cache.v1",
        "protocol_id": PROTOCOL_ID,
        "identity_payload_sha256": identity["identity_payload_sha256"],
        "members": members,
        "control": {
            "archive_member": pose_member,
            "cache_relative_path": "groundtruth.txt",
            "sha256": hashlib.sha256(pose_raw).hexdigest(),
        },
    }
    write_exclusive_json(cache_dir / "manifest.json", manifest)
    return manifest


def rgb_contract(
    contract: Mapping[str, Any],
    selected: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": {
            "archive_path": contract["source"]["archive_path"],
            "intrinsics": contract["source"]["intrinsics"],
        },
        "algorithm": {
            "minimum_window_pair_coverage": contract[
                "frozen_rgb_holdout"
            ]["minimum_window_pair_coverage"]
        },
        "windows": [
            {
                "window_index": int(item["window_index"]),
                "role": item["role"],
                "start_timestamp_s": item["start_timestamp_s"],
                "end_timestamp_s": item["end_timestamp_s"],
                "candidate_pair_count": int(item["candidate_pair_count"]),
                "geometry_median_signed_radial_expansion_per_s": item[
                    "median_signed_radial_expansion_per_s"
                ],
            }
            for item in selected
        ],
    }


def run_unchanged_rgb(
    cache_dir: Path,
    manifest: dict[str, Any],
    contract: Mapping[str, Any],
    selected: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0 import (
        producer as frozen_rgb,
    )

    adapted = rgb_contract(contract, selected)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for window in adapted["windows"]:
        summary, window_rows = frozen_rgb.evaluate_window(
            cache_dir, manifest, adapted, window
        )
        summaries.append(summary)
        rows.extend(window_rows)
    return summaries, rows


def aggregate_rgb(
    selected: Sequence[dict[str, Any]],
    rgb_summaries: Sequence[dict[str, Any]],
    geometry_rows: Sequence[dict[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    geometry_by_index = {
        int(item["window_index"]): item for item in selected
    }
    rgb_by_index = {
        int(item["window_index"]): dict(item) for item in rgb_summaries
    }
    combined: list[dict[str, Any]] = []
    for index in sorted(geometry_by_index):
        geometry_summary = geometry_by_index[index]
        rgb_summary = rgb_by_index[index]
        first_geometry = geometry_summary["first_positive_geometry_timestamp_s"]
        first_rgb = (
            float(geometry_summary["start_timestamp_s"])
            + float(rgb_summary["first_trigger_delay_s"])
            if rgb_summary["first_trigger_delay_s"] is not None
            else None
        )
        combined.append(
            {
                **rgb_summary,
                "role": geometry_summary["role"],
                "geometry_first_positive_timestamp_s": first_geometry,
                "rgb_first_trigger_timestamp_s": first_rgb,
                "rgb_minus_geometry_first_onset_s": (
                    first_rgb - first_geometry
                    if first_rgb is not None and first_geometry is not None
                    else None
                ),
            }
        )
    minimum_coverage = float(
        contract["frozen_rgb_holdout"]["minimum_window_pair_coverage"]
    )
    evaluable = all(
        item["pair_coverage"] >= minimum_coverage for item in combined
    )
    positive = [
        item for item in combined if item["role"] == "POSITIVE_APPROACH_WINDOW"
    ]
    below = [
        item
        for item in combined
        if item["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"
    ]
    comparisons = []
    for p_item in positive:
        for b_item in below:
            comparisons.append(
                {
                    "positive_window_index": p_item["window_index"],
                    "below_window_index": b_item["window_index"],
                    "median_expansion_strictly_higher": (
                        p_item["median_compensated_expansion_per_s"]
                        > b_item["median_compensated_expansion_per_s"]
                    ),
                    "trigger_coverage_strictly_higher": (
                        p_item["trigger_coverage_fixed_denominator"]
                        > b_item["trigger_coverage_fixed_denominator"]
                    ),
                }
            )
    direction = bool(
        evaluable
        and comparisons
        and all(
            item["median_expansion_strictly_higher"]
            and item["trigger_coverage_strictly_higher"]
            for item in comparisons
        )
    )
    terminal = (
        "RGB_HOLDOUT_NOT_EVALUABLE / VALID"
        if not evaluable
        else (
            "CROSS_SEQUENCE_DIRECTION_REPLICATED / VALID"
            if direction
            else "CROSS_SEQUENCE_DIRECTION_NOT_REPLICATED / VALID"
        )
    )
    return terminal, {
        "windows": combined,
        "all_positive_vs_all_below_comparisons": comparisons,
        "direction_replicated": direction,
        "rgb_algorithm_reexecuted": True,
        "algorithm_changed": False,
        "threshold_tuned": False,
        "geometry_ledger_row_count": len(geometry_rows),
    }


def run_formal(
    repo_root: Path,
    contract_path: Path,
    implementation_lock_path: Path,
    run_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers != int(load_object(contract_path)["execution"]["default_workers"]):
        raise ValueError("WORKER_COUNT_DRIFT")
    cv2.setNumThreads(1)
    progress_path = run_dir / "progress.json"
    success_path = run_dir / "SUCCESS.json"
    failure_path = run_dir / "FAILURE.json"
    claim_path = run_dir / "claim.json"
    if any(path.exists() for path in (success_path, failure_path, claim_path)):
        raise FileExistsError("FORMAL_OUTPUT_ALREADY_EXISTS")
    contract, _ = verify_contract_and_lock(
        repo_root, contract_path, implementation_lock_path
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    atomic_progress(
        progress_path,
        phase="PRECLAIM_LOCKS_VALID",
        completed_units=completed,
        started=started,
        status="RUNNING",
    )
    claim = {
        "schema_version": "rcle.disjoint_holdout.claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "contract_sha256": sha256_file(contract_path),
        "implementation_lock_sha256": sha256_file(implementation_lock_path),
        "one_shot": True,
    }
    write_exclusive_json(claim_path, claim)
    try:
        archive_path = repo_root / contract["source"]["archive_path"]
        if archive_path.stat().st_size != int(contract["source"]["archive_bytes"]):
            raise ValueError("ARCHIVE_SIZE_DRIFT")
        if sha256_file(archive_path) != contract["source"]["archive_sha256"]:
            raise ValueError("ARCHIVE_SHA256_DRIFT")
        if md5_file(archive_path) != contract["source"]["archive_md5"]:
            raise ValueError("ARCHIVE_MD5_DRIFT")
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            timestamps = depth_timestamps(
                infos, contract["source"]["sequence_id"]
            )
            windows = candidate_windows(timestamps, contract)
            poses = geometry._parse_poses(
                archive.read(contract["source"]["pose_member"])
            )
            intrinsic_values = contract["source"]["intrinsics"]
            intrinsic = np.asarray(
                (
                    (
                        float(intrinsic_values["fx"]),
                        0.0,
                        float(intrinsic_values["cx"]),
                    ),
                    (
                        0.0,
                        float(intrinsic_values["fy"]),
                        float(intrinsic_values["cy"]),
                    ),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
            geometry_rows: list[dict[str, Any]] = []
            geometry_summaries: list[dict[str, Any]] = []
            for number, window in enumerate(windows, start=1):
                rows = compute_window_geometry(
                    archive, poses, intrinsic, window, workers
                )
                geometry_rows.extend(rows)
                geometry_summaries.append(
                    summarize_geometry_window(window, rows, contract)
                )
                print(
                    f"geometry_window={number}/{len(windows)} "
                    f"index={window['window_index']} pairs={len(rows)}",
                    flush=True,
                )
            geometry_ledger_sha = write_exclusive_jsonl(
                run_dir / "geometry_pair_ledger.jsonl", geometry_rows
            )
            selected = select_windows(geometry_summaries, contract)
            selection_result: dict[str, Any] = {
                "schema_version": "rcle.disjoint_holdout.selection.v1",
                "protocol_id": PROTOCOL_ID,
                "contract_sha256": sha256_file(contract_path),
                "archive_sha256": contract["source"]["archive_sha256"],
                "rgb_member_bytes_read": 0,
                "candidate_windows": geometry_summaries,
                "selected_windows": selected,
                "geometry_pair_ledger_sha256": geometry_ledger_sha,
                "selection_evaluable": bool(selected),
                "terminal": (
                    "GEOMETRY_STRATIFIED_WINDOWS_SELECTED / VALID"
                    if selected
                    else "GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID"
                ),
            }
            selection_result["selection_payload_sha256"] = canonical_sha(
                selection_result
            )
            write_exclusive_json(
                run_dir / "geometry_selection.json", selection_result
            )
            completed = 2
            atomic_progress(
                progress_path,
                phase="GEOMETRY_SELECTION_FROZEN",
                completed_units=completed,
                started=started,
                status="RUNNING",
            )
            if not selected:
                result = {
                    "schema_version": "rcle.disjoint_holdout.result.v1",
                    "protocol_id": PROTOCOL_ID,
                    "terminal": (
                        "GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID"
                    ),
                    "authority": AUTHORITY,
                    "selection_sha256": sha256_file(
                        run_dir / "geometry_selection.json"
                    ),
                    "selected_rgb_identity_created": False,
                    "rgb_member_bytes_read": 0,
                    "rgb_algorithm_executed": False,
                    "threshold_tuned": False,
                    "cross_source_confirmation": False,
                    "performance_qualification": False,
                }
                result["result_payload_sha256"] = canonical_sha(result)
                write_exclusive_json(run_dir / "result.json", result)
            else:
                identity = selected_member_identity(
                    archive,
                    selected,
                    windows,
                    sha256_file(run_dir / "geometry_selection.json"),
                )
                write_exclusive_json(
                    run_dir / "selected_rgb_identity.json", identity
                )
                completed = 3
                atomic_progress(
                    progress_path,
                    phase="RGB_IDENTITY_FROZEN_BEFORE_MEMBER_READ",
                    completed_units=completed,
                    started=started,
                    status="RUNNING",
                )
                cache_dir = run_dir / "rgb_cache"
                manifest = materialize_selected_rgb_cache(
                    archive,
                    identity,
                    cache_dir,
                    contract["source"]["pose_member"],
                )
                rgb_summaries, rgb_rows = run_unchanged_rgb(
                    cache_dir, manifest, contract, selected
                )
                rgb_ledger_sha = write_exclusive_jsonl(
                    run_dir / "rgb_pair_ledger.jsonl", rgb_rows
                )
                terminal, aggregate = aggregate_rgb(
                    selected, rgb_summaries, geometry_rows, contract
                )
                result = {
                    "schema_version": "rcle.disjoint_holdout.result.v1",
                    "protocol_id": PROTOCOL_ID,
                    "terminal": terminal,
                    "authority": AUTHORITY,
                    "selection_sha256": sha256_file(
                        run_dir / "geometry_selection.json"
                    ),
                    "selected_rgb_identity_sha256": sha256_file(
                        run_dir / "selected_rgb_identity.json"
                    ),
                    "rgb_cache_manifest_sha256": sha256_file(
                        cache_dir / "manifest.json"
                    ),
                    "rgb_pair_ledger_sha256": rgb_ledger_sha,
                    "rgb_member_bytes_read": sum(
                        int(item["uncompressed_bytes"])
                        for item in identity["members"]
                    ),
                    "rgb_algorithm_executed": True,
                    "threshold_tuned": False,
                    "cross_source_confirmation": False,
                    "performance_qualification": False,
                    "aggregate": aggregate,
                }
                result["result_payload_sha256"] = canonical_sha(result)
                write_exclusive_json(run_dir / "result.json", result)
                completed = 4
                atomic_progress(
                    progress_path,
                    phase="RGB_HOLDOUT_COMPLETE",
                    completed_units=completed,
                    started=started,
                    status="RUNNING",
                )
        from scripts.research.egomotion_compensated_looming.rgb_algorithm_cid_sims_floor3_2_cross_sequence_holdout_r0 import (
            validator,
        )

        validation = validator.validate(
            repo_root,
            contract_path,
            implementation_lock_path,
            run_dir,
        )
        if validation["status"] != "VALID":
            raise ValueError(f"INDEPENDENT_VALIDATION:{validation['errors']}")
        write_exclusive_json(run_dir / "validation.json", validation)
        terminal_result = load_object(run_dir / "result.json")
        success = {
            "schema_version": "rcle.disjoint_holdout.success.v1",
            "protocol_id": PROTOCOL_ID,
            "terminal": terminal_result["terminal"],
            "authority": AUTHORITY,
            "result_sha256": sha256_file(run_dir / "result.json"),
            "validation_sha256": sha256_file(run_dir / "validation.json"),
            "claim_sha256": sha256_file(claim_path),
            "runtime_s": time.monotonic() - started,
        }
        write_exclusive_json(success_path, success)
        atomic_progress(
            progress_path,
            phase="TERMINAL_VALID",
            completed_units=TOTAL_UNITS,
            started=started,
            status="SUCCESS",
        )
        return success
    except BaseException as error:
        failure = {
            "schema_version": "rcle.disjoint_holdout.failure.v1",
            "protocol_id": PROTOCOL_ID,
            "terminal": "CROSS_SEQUENCE_HOLDOUT_INVALID / INVALID",
            "error_type": type(error).__name__,
            "error": str(error),
            "runtime_s": time.monotonic() - started,
        }
        if not failure_path.exists():
            write_exclusive_json(failure_path, failure)
        atomic_progress(
            progress_path,
            phase="TERMINAL_INVALID",
            completed_units=completed,
            started=started,
            status="FAILURE",
            error=str(error),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run_formal(
        args.repo_root.resolve(),
        args.contract.resolve(),
        args.implementation_lock.resolve(),
        args.run_dir.resolve(),
        args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
