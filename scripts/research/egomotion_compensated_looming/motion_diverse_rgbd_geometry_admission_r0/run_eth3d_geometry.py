"""Run the frozen geometry-only admission on ETH3D desk_3 depth and pose."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import statistics
from typing import Any, Sequence
import zlib

import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    DEFAULT_WORKERS,
    decimal_median,
    validate_execution_contract,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def write_exclusive(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    payload = b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def decode_depth(raw: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    with Image.open(BytesIO(raw)) as image:
        image.load()
        depth = np.asarray(image)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("ETH3D_DEPTH_PNG_FORMAT")
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:8, 0:width:8]
    sampled = depth[yy, xx].reshape(-1)
    valid = sampled > 0
    pixels = np.column_stack(
        (xx.reshape(-1)[valid], yy.reshape(-1)[valid])
    ).astype(np.float64)
    depth_m = sampled[valid].astype(np.float64) / 5000.0
    return pixels, depth_m, (width, height)


def pair_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        base,
        depth_path,
        intrinsic,
        previous_pose,
        current_pose,
        dt,
    ) = task
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limits = None
    else:
        limits = threadpool_limits(limits=1)
        limits.__enter__()
    try:
        try:
            pixels, depth_m, image_size = decode_depth(Path(depth_path).read_bytes())
        except (OSError, ValueError) as error:
            return {
                **base,
                "geometry_evaluable": False,
                "geometry_abstention_reason": str(error),
                "geometry_band": None,
            }
        rotation, translation = geometry._relative_pose(
            previous_pose, current_pose
        )
        summary = geometry.summarize_translation_induced_geometry(
            geometry.translation_induced_geometry(
                pixels,
                depth_m,
                intrinsic,
                rotation,
                translation,
                dt,
                image_size_wh=image_size,
                minimum_radius_px=8.0,
                zbuffer=True,
            )
        )
        if summary.get("evaluable") is not True:
            return {
                **base,
                "geometry_evaluable": False,
                "geometry_abstention_reason": "NO_VALID_GEOMETRY_SAMPLES",
                "geometry_band": None,
            }
        signed = float(summary["median_signed_radial_expansion_per_s"])
        band = (
            "BELOW_TRIGGER_REFERENCE"
            if signed < 0.01
            else (
                "WEAK_POSITIVE_RADIAL"
                if signed < 0.05
                else "POSITIVE_APPROACH_GEOMETRY"
            )
        )
        return {
            **base,
            "geometry_evaluable": True,
            "geometry_abstention_reason": None,
            "geometry_band": band,
            "geometry_signed_radial_expansion_per_s": signed,
            "geometry_radial_expansion_positive_fraction": float(
                summary["radial_expansion_positive_fraction"]
            ),
            "geometry_q90_time_normalized_parallax_rad_per_s": float(
                summary["q90_time_normalized_parallax_rad_per_s"]
            ),
        }
    finally:
        if limits is not None:
            limits.__exit__(None, None, None)


def longest(rows: Sequence[dict[str, Any]], band: str) -> tuple[int, Decimal]:
    best_count = 0
    best_duration = Decimal("0")
    count = 0
    start: Decimal | None = None
    previous_index: int | None = None
    for row in rows:
        contiguous = previous_index is not None and row["pair_index"] == previous_index + 1
        if row.get("geometry_band") == band:
            if not contiguous or count == 0:
                count = 0
                start = Decimal(str(row["previous_timestamp_s"]))
            count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - start
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count = 0
            start = None
        previous_index = int(row["pair_index"])
    return best_count, best_duration


def summarize(window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = int(window["pair_count"])
    evaluable = [row for row in rows if row["geometry_evaluable"] is True]
    counts = Counter(row["geometry_band"] for row in evaluable)
    abstentions = Counter(
        row["geometry_abstention_reason"]
        for row in rows
        if row["geometry_evaluable"] is not True
    )
    positive_count, positive_duration = longest(
        rows, "POSITIVE_APPROACH_GEOMETRY"
    )
    below_count, below_duration = longest(rows, "BELOW_TRIGGER_REFERENCE")
    coverage = Decimal(len(evaluable)) / Decimal(denominator)
    positive_fraction = Decimal(
        counts["POSITIVE_APPROACH_GEOMETRY"]
    ) / Decimal(denominator)
    below_fraction = Decimal(counts["BELOW_TRIGGER_REFERENCE"]) / Decimal(
        denominator
    )
    positive_ok = (
        coverage >= Decimal("0.8")
        and positive_fraction >= Decimal("0.8")
        and positive_duration >= Decimal("5")
    )
    below_ok = (
        coverage >= Decimal("0.8")
        and below_fraction >= Decimal("0.8")
        and below_duration >= Decimal("5")
    )
    if positive_ok and below_ok:
        raise ValueError(f"ROLE_OVERLAP:{window['window_index']}")
    role = (
        "POSITIVE_APPROACH_WINDOW"
        if positive_ok
        else (
            "BELOW_TRIGGER_REFERENCE_WINDOW"
            if below_ok
            else "AMBIGUOUS_OR_INELIGIBLE"
        )
    )
    values = [
        Decimal(str(row["geometry_signed_radial_expansion_per_s"]))
        for row in evaluable
    ]
    return {
        "window_index": int(window["window_index"]),
        "start_timestamp_s": window["start_timestamp_s"],
        "end_timestamp_s": window["end_timestamp_s"],
        "frame_count": int(window["frame_count"]),
        "candidate_pair_count": denominator,
        "geometry_evaluable_pair_count": len(evaluable),
        "geometry_pair_coverage_fixed_denominator": float(coverage),
        "geometry_abstention_count": denominator - len(evaluable),
        "geometry_abstention_reasons": dict(sorted(abstentions.items())),
        "geometry_band_counts": dict(sorted(counts.items())),
        "positive_fraction_fixed_denominator": float(positive_fraction),
        "below_fraction_fixed_denominator": float(below_fraction),
        "longest_positive_run_pair_count": positive_count,
        "longest_positive_run_duration_s": float(positive_duration),
        "longest_below_run_pair_count": below_count,
        "longest_below_run_duration_s": float(below_duration),
        "median_signed_radial_expansion_per_s": (
            float(decimal_median(values)) if values else None
        ),
        "role": role,
    }


def select(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in summaries if row["role"] == "POSITIVE_APPROACH_WINDOW"]
    below = [
        row
        for row in summaries
        if row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"
    ]
    feasible = []
    for positive_rows in itertools.combinations(positive, 2):
        for below_rows in itertools.combinations(below, 2):
            rows = sorted((*positive_rows, *below_rows), key=lambda row: row["window_index"])
            starts = [Decimal(row["start_timestamp_s"]) for row in rows]
            if all(right - left >= Decimal("20") for left, right in zip(starts, starts[1:])):
                feasible.append(rows)
    return min(
        feasible,
        key=lambda rows: tuple(row["window_index"] for row in rows),
        default=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--gate-completion", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--execution-claim", type=Path, required=True)
    parser.add_argument("--corrected-claim", type=Path, required=True)
    parser.add_argument("--window-freeze", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-inventory", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError("WORKERS_MUST_EQUAL_FROZEN_DEFAULT")
    contract = load(args.contract.resolve())
    validate_execution_contract(contract)
    implementation_lock = load(args.implementation_lock.resolve())
    execution_claim = load(args.execution_claim.resolve())
    repo = Path(__file__).resolve().parents[4]
    if implementation_lock.get("status") != "FROZEN_BEFORE_GEOMETRY_EXECUTION":
        raise ValueError("IMPLEMENTATION_LOCK_STATUS")
    for entry in implementation_lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    if (
        execution_claim.get("candidate_id") != "ETH3D_SLAM_DESK_3"
        or execution_claim.get("implementation_lock_sha256")
        != sha(args.implementation_lock.resolve())
    ):
        raise ValueError("EXECUTION_CLAIM_IDENTITY")
    gate = load(args.gate_completion.resolve())
    if gate.get("status") != "FROZEN_BEFORE_ANY_GEOMETRY_METRIC":
        raise ValueError("GATE_COMPLETION_STATUS")
    claim = load(args.corrected_claim.resolve())
    freeze_path = args.window_freeze.resolve()
    if claim["corrected_window_freeze_sha256"] != sha(freeze_path):
        raise ValueError("CORRECTED_FREEZE_IDENTITY")
    if (
        implementation_lock["bound_local_inputs"]["corrected_window_freeze_sha256"]
        != sha(freeze_path)
        or implementation_lock["bound_local_inputs"]["corrected_depth_claim_sha256"]
        != sha(args.corrected_claim.resolve())
    ):
        raise ValueError("IMPLEMENTATION_LOCK_LOCAL_INPUT")
    freeze = load(freeze_path)
    inventory = load(args.depth_inventory.resolve())
    by_name = {item["path"]: item for item in inventory["members"]}
    depth_root = args.depth_root.resolve()
    verification = []
    for window in freeze["windows"]:
        for member in window["depth_members"]:
            path = depth_root.joinpath(*Path(member).parts)
            info = by_name.get(member)
            if info is None or not path.is_file():
                raise ValueError(f"DEPTH_MEMBER_MISSING:{member}")
            data = path.read_bytes()
            if (
                len(data) != int(info["size"])
                or f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != info["crc32"]
            ):
                raise ValueError(f"DEPTH_MEMBER_IDENTITY:{member}")
            verification.append(
                {
                    "path": member,
                    "bytes": len(data),
                    "crc32": info["crc32"],
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    output = args.output_dir.resolve()
    write_exclusive(
        output / "depth_member_verification.json",
        {
            "candidate_id": "ETH3D_SLAM_DESK_3",
            "member_count": len(verification),
            "members": verification,
        },
    )
    poses = geometry._parse_poses(args.groundtruth.resolve().read_bytes())
    intrinsic = geometry._parse_intrinsic(args.calibration.resolve().read_bytes())
    all_rows: list[dict[str, Any]] = []
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for window in freeze["windows"]:
            tasks = []
            timestamps = [
                Decimal(Path(member).stem) for member in window["depth_members"]
            ]
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
                try:
                    previous_pose = geometry._interpolate_pose(poses, previous)
                    current_pose = geometry._interpolate_pose(poses, current)
                except ValueError as error:
                    all_rows.append(
                        {
                            **base,
                            "geometry_evaluable": False,
                            "geometry_abstention_reason": str(error),
                            "geometry_band": None,
                        }
                    )
                    continue
                member = window["depth_members"][pair_index]
                path = depth_root.joinpath(*Path(member).parts)
                tasks.append(
                    (
                        base,
                        str(path),
                        intrinsic,
                        previous_pose,
                        current_pose,
                        float(current - previous),
                    )
                )
            computed = list(executor.map(pair_worker, tasks))
            window_rows = sorted(
                [
                    row
                    for row in (*all_rows, *computed)
                    if row["window_index"] == window["window_index"]
                ],
                key=lambda row: row["pair_index"],
            )
            all_rows.extend(computed)
            summaries.append(summarize(window, window_rows))
    all_rows.sort(key=lambda row: (row["window_index"], row["pair_index"]))
    ledger_sha = write_jsonl_exclusive(output / "geometry_pair_ledger.jsonl", all_rows)
    selected = select(summaries)
    terminal = (
        "GEOMETRY_ADMITTED_FOUR_WINDOWS_FROZEN / VALID"
        if selected
        else "NOT_EVALUABLE_NO_RGB_NO_REPLACEMENT / VALID"
    )
    selection = {
        "candidate_id": "ETH3D_SLAM_DESK_3",
        "window_summaries": summaries,
        "selected_windows": selected,
        "selection_evaluable": bool(selected),
        "terminal": terminal,
        "workers": args.workers,
        "numeric_relative_tolerance": "1e-12",
        "numeric_absolute_tolerance": "1e-15",
        "rgb_bytes_accessed": 0,
        "candidate_replacement": False,
        "post_outcome_windows_added": 0,
        "geometry_pair_ledger_sha256": ledger_sha,
    }
    write_exclusive(output / "geometry_selection.json", selection)
    write_exclusive(
        output / "result.json",
        {
            "protocol_id": "RCLE_MOTION_DIVERSE_RGBD_GEOMETRY_ADMISSION_R0",
            "candidate_id": "ETH3D_SLAM_DESK_3",
            "terminal": terminal,
            "positive_window_count": sum(
                row["role"] == "POSITIVE_APPROACH_WINDOW" for row in summaries
            ),
            "below_reference_window_count": sum(
                row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW" for row in summaries
            ),
            "ambiguous_window_count": sum(
                row["role"] == "AMBIGUOUS_OR_INELIGIBLE" for row in summaries
            ),
            "selected_window_indices": [
                row["window_index"] for row in selected
            ],
            "rgb_bytes_accessed": 0,
            "algorithm_changed": False,
            "candidate_replacement": False,
        },
    )
    print(json.dumps(load(output / "result.json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
