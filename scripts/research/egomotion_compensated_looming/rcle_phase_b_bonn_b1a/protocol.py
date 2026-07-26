from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import PIL

from .geometry import (
    IndexRow,
    adjacent_pairs,
    assign_depth_rows,
    blank_truth_grids,
    classify_window,
    decode_depth_png,
    evaluate_truth,
    interpolate_pose,
    parse_index_text,
    parse_pose_text,
    relative_geometry,
    rotation_homography,
)


PROTOCOL_ID = "RCLE_PHASE_B_BONN_B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION"
DESIGN_LOCK_SHA256 = (
    "c53c9edaf7012df481b2ba286902af87f1716e3a5d4f57f27398303c4f74420e"
)
PREREGISTRATION_SHA256 = (
    "f3974b2c0096dae2334b1d6c8cd563d892b09288df4f2085604b8fee88d4cfd0"
)
B0_RECEIPT_SHA256 = (
    "dc0ffe9a890b539478ff4c035b4dfadea6c21347a11b36f164810a18eb811f86"
)
WINDOW_DENOMINATOR_SHA256 = (
    "f1e6f7f2e54da349d004af744573884e6273089f67bda86d5f0eb812234aa05b"
)
COHORT_IDENTITY_SHA256 = (
    "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
)
PASS_TERMINAL = (
    "B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION_VALID_"
    "B1B_BRANCH_SCOPE_MAY_BE_REVIEWED"
)
HOLD_TERMINAL = (
    "HOLD_B1_SOURCE_NATIVE_TRUTH_NOT_EVALUABLE_NO_WINDOW_REPLACEMENT"
)
INVALID_TERMINAL = "INVALID_EXECUTION_CLOSE_B1"
ROTATION_ROLE = "ROTATION_TRUTH_ELIGIBLE"
APPROACH_ROLE = "STATIC_APPROACH_TRUTH_ELIGIBLE"
WINDOWS = (
    (1, "rgbd_bonn_crowd2", 0, "1548339892.26121", "1548339902.26121"),
    (1, "rgbd_bonn_crowd2", 1, "1548339902.26121", "1548339912.26121"),
    (2, "rgbd_bonn_balloon_tracking", 0, "1548266633.5283", "1548266643.5283"),
    (3, "rgbd_bonn_balloon_tracking2", 0, "1548266677.42642", "1548266687.42642"),
    (4, "rgbd_bonn_moving_obstructing_box2", 0, "1548339380.76065", "1548339390.76065"),
    (4, "rgbd_bonn_moving_obstructing_box2", 1, "1548339390.76065", "1548339400.76065"),
    (5, "rgbd_bonn_balloon2", 0, "1548266530.56676", "1548266540.56676"),
    (6, "rgbd_bonn_moving_nonobstructing_box2", 0, "1548266187.16307", "1548266197.16307"),
    (6, "rgbd_bonn_moving_nonobstructing_box2", 1, "1548266197.16307", "1548266207.16307"),
    (6, "rgbd_bonn_moving_nonobstructing_box2", 2, "1548266207.16307", "1548266217.16307"),
)


def canonical_paths(repo_root: Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    package = (
        root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_phase_b_bonn_b1a"
    )
    return {
        "repo_root": root,
        "package": package,
        "preregistration": root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_B1_PREREGISTRATION_2026-07-26.md",
        "design_lock": root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_B1_DESIGN_LOCK_2026-07-26.json",
        "implementation_lock": package
        / "RCLE_PHASE_B_BONN_B1A_IMPLEMENTATION_LOCK.json",
        "bootstrap_runner": root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "run_phase_b_bonn_b1a.py",
        "b0_receipt": root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b0_r1"
        / "formal_entry_b0_r1"
        / "receipt.json",
        "bonn_official_page": root
        / "artifacts.local"
        / "datasets"
        / "egomotion_compensated_looming_r1"
        / "bonn_metadata_r0"
        / "official_page.html",
        "tum_file_formats": root
        / "artifacts.local"
        / "datasets"
        / "egomotion_compensated_looming_r1"
        / "bonn_b1_authority"
        / "tum_rgbd_file_formats.html",
        "output": root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b1"
        / "b1a_geometry_admission",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def float_hex(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("B1A_FLOAT_NONFINITE")
    return float(value).hex()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"B1A_JSON_OBJECT_REQUIRED:{path}")
    return value


def validate_implementation_lock(
    repo_root: Path, paths: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    resolved = dict(paths or canonical_paths(repo_root))
    if sha256_file(resolved["preregistration"]) != PREREGISTRATION_SHA256:
        raise ValueError("B1A_PREREGISTRATION_HASH_MISMATCH")
    if sha256_file(resolved["design_lock"]) != DESIGN_LOCK_SHA256:
        raise ValueError("B1A_DESIGN_LOCK_HASH_MISMATCH")
    lock_path = resolved["implementation_lock"]
    if not lock_path.is_file():
        raise ValueError("B1A_IMPLEMENTATION_LOCK_ABSENT")
    lock = _json(lock_path)
    if lock.get("design_lock_sha256") != DESIGN_LOCK_SHA256:
        raise ValueError("B1A_IMPLEMENTATION_LOCK_DESIGN_DRIFT")
    if lock.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("B1A_IMPLEMENTATION_LOCK_PREREGISTRATION_DRIFT")
    if lock.get("canonical_execution_authorized") is not True:
        raise ValueError("B1A_CANONICAL_EXECUTION_NOT_AUTHORIZED")
    source_files = lock.get("source_files", {})
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("B1A_IMPLEMENTATION_SOURCE_MANIFEST_ABSENT")
    for relative, expected in source_files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("B1A_IMPLEMENTATION_SOURCE_MANIFEST_TYPE")
        if sha256_file(Path(repo_root) / relative) != expected:
            raise ValueError(f"B1A_IMPLEMENTATION_SOURCE_HASH_MISMATCH:{relative}")
    return lock


def atomic_write_json(path: Path, payload: Any) -> None:
    target = Path(path)
    if target.name == "run_claim.json":
        raise ValueError("B1A_PROTOCOL_MUST_NOT_WRITE_RUN_CLAIM")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            move_file_ex = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).MoveFileExW
            move_file_ex.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                ctypes.c_uint32,
            ]
            move_file_ex.restype = ctypes.c_int
            movefile_replace_existing = 0x00000001
            movefile_write_through = 0x00000008
            if not move_file_ex(
                str(temporary),
                str(target),
                movefile_replace_existing | movefile_write_through,
            ):
                error = ctypes.get_last_error()
                raise OSError(
                    error,
                    "MoveFileExW atomic write-through replace failed",
                    str(target),
                )
        else:
            os.replace(temporary, target)
            directory = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_windows() -> list[dict[str, Any]]:
    return [
        {
            "sequence_rank": sequence_rank,
            "sequence_id": sequence_id,
            "window_rank": window_rank,
            "start": start,
            "end": end,
        }
        for sequence_rank, sequence_id, window_rank, start, end in WINDOWS
    ]


def _unique_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"B1A_CONTROL_MEMBER_CARDINALITY:{suffix}:{len(matches)}")
    return matches[0]


def _resolve_reference(root: str, reference: str, members: set[str]) -> str:
    candidate = f"{root}/{reference}" if root else reference
    if candidate not in members:
        raise ValueError(f"B1A_REFERENCED_MEMBER_ABSENT:{reference}")
    return candidate


def _blank_pair_grids(reason: str) -> list[dict[str, Any]]:
    return [
        {
            **grid,
            "c_truth_grid_hex": None,
        }
        for grid in blank_truth_grids(reason)
    ]


def _serialize_grids(grids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in grid.items()
            if key != "c_truth_grid"
        }
        | {"c_truth_grid_hex": float_hex(grid.get("c_truth_grid"))}
        for grid in grids
    ]


def _pair_identity(
    sequence_id: str,
    window_rank: int,
    previous: IndexRow,
    current: IndexRow,
) -> str:
    return (
        f"{sequence_id}:{window_rank}:"
        f"{previous.source_row_rank}:{current.source_row_rank}"
    )


def _window_result(
    base: dict[str, Any],
    adjacent_count: int,
    candidate_count: int,
    covered: list[dict[str, float]],
) -> dict[str, Any]:
    summary = classify_window(candidate_count, covered)
    return {
        **base,
        "all_adjacent_pair_count": adjacent_count,
        "candidate_pair_denominator_count": candidate_count,
        "truth_covered_pair_count": len(covered),
        "truth_coverage_hex": float_hex(float(summary["coverage"])),
        "window_truth_closing_hex": float_hex(summary.get("truth")),
        "window_angular_rate_rad_s_hex": float_hex(summary.get("angular")),
        "window_translation_speed_m_s_hex": float_hex(
            summary.get("translation")
        ),
        "window_absolute_truth_closing_hex": float_hex(
            summary.get("absolute_truth")
        ),
        "role": summary["role"],
    }


def _terminal_for_roles(
    windows: list[dict[str, Any]],
) -> tuple[dict[str, int], str]:
    counts = {
        role: len(
            {
                window["sequence_id"]
                for window in windows
                if window["role"] == role
            }
        )
        for role in (ROTATION_ROLE, APPROACH_ROLE)
    }
    terminal = (
        PASS_TERMINAL if any(count >= 2 for count in counts.values()) else HOLD_TERMINAL
    )
    return counts, terminal


def _claim_for_receipt(claim: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(claim)
    result.setdefault("application_data_operations_before_claim", 0)
    result.setdefault("exclusive_create", True)
    result.setdefault("maximum_claims", 1)
    result.setdefault("claim_permanently_retained", True)
    return result


def run_b1a(repo_root: Path, claim: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }
    if runtime != {
        "python": "3.11.9",
        "numpy": "2.1.3",
        "pillow": "12.2.0",
    }:
        raise ValueError(f"B1A_RUNTIME_DRIFT:{runtime}")
    paths = canonical_paths(repo_root)
    implementation_lock = validate_implementation_lock(repo_root, paths)
    implementation_lock_sha256 = sha256_file(paths["implementation_lock"])
    output = paths["output"].resolve()
    if Path(claim.get("canonical_output", "")).resolve() != output:
        raise ValueError("B1A_CLAIM_OUTPUT_DRIFT")
    if claim.get("design_lock_sha256") != DESIGN_LOCK_SHA256:
        raise ValueError("B1A_CLAIM_DESIGN_HASH_DRIFT")
    if claim.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("B1A_CLAIM_PREREGISTRATION_HASH_DRIFT")
    if claim.get("b0_receipt_sha256") != B0_RECEIPT_SHA256:
        raise ValueError("B1A_CLAIM_B0_RECEIPT_HASH_DRIFT")
    if claim.get("window_denominator_sha256") != WINDOW_DENOMINATOR_SHA256:
        raise ValueError("B1A_CLAIM_WINDOW_DENOMINATOR_HASH_DRIFT")
    if claim.get("implementation_lock_sha256") != implementation_lock_sha256:
        raise ValueError("B1A_CLAIM_IMPLEMENTATION_LOCK_HASH_DRIFT")
    if claim.get("bootstrap_runner_sha256") != sha256_file(
        paths["bootstrap_runner"]
    ):
        raise ValueError("B1A_CLAIM_BOOTSTRAP_RUNNER_HASH_DRIFT")
    if (
        claim.get("argv") != []
        or claim.get("canonical_run_claim")
        != str(output / "run_claim.json")
        or claim.get("application_data_operations_before_claim") != 0
        or claim.get("exclusive_create") is not True
        or claim.get("maximum_claims") != 1
        or claim.get("claim_permanently_retained") is not True
        or claim.get("delete_replace_or_rewrite_claim") != "FORBIDDEN"
        or claim.get(
            "success_failure_exception_or_interrupt_consumes_claim"
        )
        is not True
    ):
        raise ValueError("B1A_CLAIM_EXECUTION_CONTRACT_DRIFT")
    source_authority = claim.get("source_authority_sha256")
    expected_source_authority = {
        "bonn_official_page": (
            "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
        ),
        "tum_file_formats": (
            "721c8df093ade2b0078215c3154f6f1a3641a0c691b5123cd037e87b61b30107"
        ),
    }
    if source_authority != expected_source_authority:
        raise ValueError("B1A_CLAIM_SOURCE_AUTHORITY_DRIFT")
    for key, expected in expected_source_authority.items():
        if sha256_file(paths[key]) != expected:
            raise ValueError(f"B1A_SOURCE_AUTHORITY_HASH_MISMATCH:{key}")
    if sha256_file(paths["b0_receipt"]) != B0_RECEIPT_SHA256:
        raise ValueError("B1A_B0_RECEIPT_HASH_MISMATCH")
    b0 = _json(paths["b0_receipt"])
    if b0.get("window_denominator_sha256") != WINDOW_DENOMINATOR_SHA256:
        raise ValueError("B1A_WINDOW_DENOMINATOR_HASH_MISMATCH")
    if b0.get("cohort_identity_sha256") != COHORT_IDENTITY_SHA256:
        raise ValueError("B1A_COHORT_IDENTITY_MISMATCH")
    sequence_results = {
        row["sequence_id"]: row for row in b0.get("sequence_results", [])
    }
    expected_sequences = {row[1] for row in WINDOWS}
    if set(sequence_results) != expected_sequences:
        raise ValueError("B1A_B0_SEQUENCE_SET_MISMATCH")

    archive_hashes: dict[str, str] = {}
    pair_rows: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    depth_decode_operations = 0
    pose_numeric_rows_parsed = 0

    for sequence_rank, sequence_id in sorted(
        {(row[0], row[1]) for row in WINDOWS}
    ):
        source = sequence_results[sequence_id]
        archive = Path(source["archive_path"])
        expected_archive_hash = source["archive_sha256"]
        claim_hashes = claim.get("archive_sha256_by_sequence", {})
        if claim_hashes.get(sequence_id) != expected_archive_hash:
            raise ValueError(f"B1A_CLAIM_ARCHIVE_HASH_DRIFT:{sequence_id}")
        if sha256_file(archive) != expected_archive_hash:
            raise ValueError(f"B1A_ARCHIVE_HASH_MISMATCH:{sequence_id}")
        archive_hashes[sequence_id] = expected_archive_hash

        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError(f"B1A_DUPLICATE_ZIP_MEMBER:{sequence_id}")
            members = set(names)
            rgb_member = _unique_member(names, "rgb.txt")
            depth_member = _unique_member(names, "depth.txt")
            pose_member = _unique_member(names, "groundtruth.txt")
            roots = {
                name.rsplit("/", 1)[0] if "/" in name else ""
                for name in (rgb_member, depth_member, pose_member)
            }
            if len(roots) != 1:
                raise ValueError(f"B1A_CONTROL_ROOT_MISMATCH:{sequence_id}")
            root = roots.pop()
            rgb_rows = parse_index_text(bundle.read(rgb_member))
            depth_rows = parse_index_text(bundle.read(depth_member))
            pose_rows = parse_pose_text(bundle.read(pose_member))
            pose_numeric_rows_parsed += len(pose_rows)
            rgb_by_rank = {row.source_row_rank: row for row in rgb_rows}

            for row in WINDOWS:
                if row[1] != sequence_id:
                    continue
                _, _, window_rank, start_text, end_text = row
                start = Decimal(start_text)
                end = Decimal(end_text)
                base = {
                    "sequence_rank": sequence_rank,
                    "sequence_id": sequence_id,
                    "window_rank": window_rank,
                    "start": start_text,
                    "end": end_text,
                }
                pairs = adjacent_pairs(rgb_rows, start, end)
                depth_assignments = assign_depth_rows(
                    rgb_rows, depth_rows, start, end
                )
                covered: list[dict[str, float]] = []
                candidate_count = sum(bool(pair["candidate"]) for pair in pairs)

                for pair in pairs:
                    previous = pair["previous"]
                    current = pair["current"]
                    pair_row: dict[str, Any] = {
                        **base,
                        "pair_id": _pair_identity(
                            sequence_id, window_rank, previous, current
                        ),
                        "previous_rgb_source_row_rank": previous.source_row_rank,
                        "current_rgb_source_row_rank": current.source_row_rank,
                        "previous_rgb_timestamp": str(previous.timestamp),
                        "current_rgb_timestamp": str(current.timestamp),
                        "dt": str(pair["dt"]),
                        "candidate": bool(pair["candidate"]),
                        "truth_covered": False,
                        "abstention_reason": None,
                    }
                    if not pair["candidate"]:
                        pair_row["abstention_reason"] = "DT_OUTSIDE_CANDIDATE_RANGE"
                        pair_row["grids"] = _blank_pair_grids(
                            "DT_OUTSIDE_CANDIDATE_RANGE"
                        )
                        pair_rows.append(pair_row)
                        continue
                    try:
                        # B1A validates RGB references through central-directory
                        # metadata only; it never reads RGB member bytes.
                        for rgb in (previous, current):
                            rgb_path = _resolve_reference(root, rgb.path, members)
                            info = bundle.getinfo(rgb_path)
                            if info.file_size <= 0:
                                raise ValueError("B1A_RGB_DECLARED_SIZE_NONPOSITIVE")

                        previous_depth = depth_assignments.get(
                            previous.source_row_rank
                        )
                        current_depth = depth_assignments.get(
                            current.source_row_rank
                        )
                        if previous_depth is None or current_depth is None:
                            raise ValueError("B1A_DEPTH_JOIN_UNMATCHED")
                        previous_depth_member = _resolve_reference(
                            root, previous_depth.path, members
                        )
                        current_depth_member = _resolve_reference(
                            root, current_depth.path, members
                        )
                        previous_pose = interpolate_pose(
                            pose_rows, previous.timestamp
                        )
                        current_pose = interpolate_pose(
                            pose_rows, current.timestamp
                        )
                        geometry = relative_geometry(
                            previous_pose, current_pose, float(pair["dt"])
                        )
                        homography = rotation_homography(
                            geometry["R_current_from_previous"]
                        )
                        previous_depth_array = decode_depth_png(
                            bundle.read(previous_depth_member)
                        )
                        depth_decode_operations += 1
                        current_depth_array = decode_depth_png(
                            bundle.read(current_depth_member)
                        )
                        depth_decode_operations += 1
                        grids = evaluate_truth(
                            previous_depth_array,
                            current_depth_array,
                            geometry["R_current_from_previous"],
                            geometry["t_current_from_previous"],
                            float(pair["dt"]),
                        )
                        eligible = [
                            grid for grid in grids if grid["truth_eligible"]
                        ]
                        pair_row.update(
                            {
                                "previous_depth_source_row_rank": previous_depth.source_row_rank,
                                "current_depth_source_row_rank": current_depth.source_row_rank,
                                "previous_depth_delta": str(
                                    previous_depth.timestamp - previous.timestamp
                                ),
                                "current_depth_delta": str(
                                    current_depth.timestamp - current.timestamp
                                ),
                                "R_current_from_previous_hex": [
                                    [float_hex(float(value)) for value in matrix_row]
                                    for matrix_row in geometry[
                                        "R_current_from_previous"
                                    ]
                                ],
                                "t_current_from_previous_hex": [
                                    float_hex(float(value))
                                    for value in geometry[
                                        "t_current_from_previous"
                                    ]
                                ],
                                "H_previous_to_current_hex": [
                                    [float_hex(float(value)) for value in matrix_row]
                                    for matrix_row in homography
                                ],
                                "translation_speed_m_s_hex": float_hex(
                                    geometry["translation_speed_m_s"]
                                ),
                                "angular_rate_rad_s_hex": float_hex(
                                    geometry["angular_rate_rad_s"]
                                ),
                                "truth_covered": len(eligible) >= 5,
                                "grids": _serialize_grids(grids),
                            }
                        )
                        if len(eligible) >= 5:
                            truth_values = np.asarray(
                                [
                                    grid["c_truth_grid"]
                                    for grid in eligible
                                ],
                                dtype=np.float64,
                            )
                            pair_truth = float(np.median(truth_values))
                            pair_absolute = float(
                                np.median(np.abs(truth_values))
                            )
                            pair_row["pair_truth_closing_hex"] = float_hex(
                                pair_truth
                            )
                            pair_row[
                                "pair_absolute_truth_closing_hex"
                            ] = float_hex(pair_absolute)
                            covered.append(
                                {
                                    "truth": pair_truth,
                                    "absolute_truth": pair_absolute,
                                    "angular": geometry[
                                        "angular_rate_rad_s"
                                    ],
                                    "translation": geometry[
                                        "translation_speed_m_s"
                                    ],
                                }
                            )
                        else:
                            pair_row["abstention_reason"] = (
                                "TRUTH_ELIGIBLE_GRIDS_BELOW_5"
                            )
                    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
                        pair_row["abstention_reason"] = str(error)
                        pair_row["grids"] = _blank_pair_grids(str(error))
                    pair_rows.append(pair_row)

                window_results.append(
                    _window_result(
                        base,
                        len(pairs),
                        candidate_count,
                        covered,
                    )
                )

    canonical_windows = _canonical_windows()
    if [
        {
            key: row[key]
            for key in (
                "sequence_rank",
                "sequence_id",
                "window_rank",
                "start",
                "end",
            )
        }
        for row in window_results
    ] != canonical_windows:
        raise ValueError("B1A_WINDOW_RESULT_ORDER_DRIFT")
    branch_counts, terminal = _terminal_for_roles(window_results)
    gate = terminal == PASS_TERMINAL
    ledger = {
        "schema_version": "rcle.phase_b.bonn_b1a.ledger.v1",
        "windows": canonical_windows,
        "window_results": window_results,
        "pairs": pair_rows,
    }
    ledger_identity = canonical_sha256(ledger)
    claim_receipt = _claim_for_receipt(claim)
    run_claim_path = output / "run_claim.json"
    persisted_claim = _json(run_claim_path)
    if persisted_claim != claim_receipt:
        raise ValueError("B1A_PERSISTED_CLAIM_OBJECT_DRIFT")
    receipt = {
        "schema_version": "rcle.phase_b.bonn_b1a.receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(),
        "environment": runtime,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_lock_sha256": implementation_lock_sha256,
        "bootstrap_runner_sha256": claim_receipt[
            "bootstrap_runner_sha256"
        ],
        "b0_receipt_sha256": B0_RECEIPT_SHA256,
        "window_denominator_sha256": WINDOW_DENOMINATOR_SHA256,
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "source_authority_sha256_by_id": expected_source_authority,
        "archive_sha256_by_sequence": archive_hashes,
        "claim": claim_receipt,
        "run_claim_sha256": sha256_file(run_claim_path),
        "read_firewall": {
            "rgb_member_bytes_read": 0,
            "rgb_decode_operations": 0,
            "phase_b_metric_operations": 0,
            "static_map_reads": 0,
            "legacy_bonn_outcome_reads": 0,
            "network_operations": 0,
            "depth_decode_operations": depth_decode_operations,
            "pose_numeric_rows_parsed": pose_numeric_rows_parsed,
        },
        "ledger": ledger,
        "ledger_identity_sha256": ledger_identity,
        "branch_distinct_sequence_counts": branch_counts,
        "gate_pass": gate,
        "terminal_state": terminal,
        "b1b_branch_scope_may_be_reviewed": gate,
        "execution_authority_consumed": True,
        "b1b_implementation_authorized": False,
    }
    atomic_write_json(output / "ledger.json", ledger)
    atomic_write_json(output / "receipt.json", receipt)
    return {
        "receipt_path": str(output / "receipt.json"),
        "ledger_sha256": ledger_identity,
        "gate_pass": gate,
        "terminal_state": terminal,
    }


__all__ = [
    "APPROACH_ROLE",
    "B0_RECEIPT_SHA256",
    "COHORT_IDENTITY_SHA256",
    "DESIGN_LOCK_SHA256",
    "HOLD_TERMINAL",
    "INVALID_TERMINAL",
    "PASS_TERMINAL",
    "PREREGISTRATION_SHA256",
    "PROTOCOL_ID",
    "ROTATION_ROLE",
    "WINDOW_DENOMINATOR_SHA256",
    "WINDOWS",
    "atomic_write_json",
    "canonical_json",
    "canonical_paths",
    "canonical_sha256",
    "float_hex",
    "run_b1a",
    "sha256_file",
    "validate_implementation_lock",
]
