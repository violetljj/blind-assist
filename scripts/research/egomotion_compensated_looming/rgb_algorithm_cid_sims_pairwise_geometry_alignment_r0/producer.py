from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Callable, Iterable, Sequence
import zipfile

import numpy as np

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0"
)


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL_OBJECT_REQUIRED")
    return rows


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def verify_inputs(
    repo_root: Path, contract: dict[str, Any]
) -> tuple[dict[str, Path], str, str]:
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PROTOCOL_ID")
    expected_authority = {
        "maximum_claim": "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY",
        "algorithm_reexecution_authorized": False,
        "threshold_tuning_authorized": False,
        "outcome_blind_claim_authorized": False,
        "independent_confirmation_authorized": False,
        "performance_qualification_authorized": False,
        "product_or_safety_claim_authorized": False,
    }
    if contract.get("authority") != expected_authority:
        raise ValueError("AUTHORITY")
    expected_ids = {
        "rgb-pair-ledger",
        "rgb-result",
        "rgb-posthoc-validation-r1",
        "rgb-cache-manifest",
        "cid-development-descriptor",
        "cid-geometry-helper",
        "translation-geometry-helper",
    }
    bindings = contract.get("immutable_bindings", [])
    ids = [str(item.get("id", "")) for item in bindings]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("BINDING_INVENTORY")
    paths: dict[str, Path] = {}
    for binding in bindings:
        path = repo_root / binding["path"]
        paths[binding["id"]] = path
        if not path.is_file() or digest_file(path) != binding["sha256"]:
            raise ValueError(f"BINDING:{binding['id']}")
    source = contract["source"]
    archive = repo_root / source["archive_path"]
    if not archive.is_file() or archive.stat().st_size != source["archive_bytes"]:
        raise ValueError("ARCHIVE_SIZE")
    sha256, md5 = source_digests(archive)
    if sha256 != source["archive_sha256"] or md5 != source["archive_md5"]:
        raise ValueError("ARCHIVE_DIGEST")
    pair_geometry = contract["pair_geometry"]
    if Decimal(pair_geometry["maximum_pair_dt_s"]) != geometry.MAX_POSE_BRACKET_SECONDS:
        raise ValueError("GEOMETRY_MAXIMUM_BRACKET")
    if int(source["depth_sample_stride_px"]) != geometry.DEPTH_SAMPLE_STRIDE_PX:
        raise ValueError("GEOMETRY_DEPTH_STRIDE")
    if float(source["depth_units_per_meter"]) != geometry.DEPTH_UNITS_PER_METER:
        raise ValueError("GEOMETRY_DEPTH_UNITS")
    if float(source["minimum_radius_px"]) != geometry.MINIMUM_RADIUS_PX:
        raise ValueError("GEOMETRY_MINIMUM_RADIUS")
    if not np.array_equal(intrinsic_matrix(contract), geometry.INTRINSIC):
        raise ValueError("GEOMETRY_INTRINSIC")
    return paths, sha256, md5


def shared_timestamps(names: Iterable[str], sequence_id: str) -> list[Decimal]:
    color: set[Decimal] = set()
    depth: set[Decimal] = set()
    seen_names: set[str] = set()
    for raw in names:
        if raw in seen_names:
            raise ValueError("ZIP_MEMBER_NAME_DUPLICATE")
        seen_names.add(raw)
        name = PurePosixPath(raw)
        if len(name.parts) != 3 or name.parts[0] != sequence_id:
            continue
        if name.suffix.lower() != ".png":
            continue
        try:
            timestamp = Decimal(name.stem)
        except InvalidOperation:
            continue
        if name.parts[1] == "color":
            color.add(timestamp)
        elif name.parts[1] == "depth":
            depth.add(timestamp)
    return sorted(color & depth)


def expected_pairs(
    shared: Sequence[Decimal], window: dict[str, Any], maximum_dt: Decimal
) -> list[tuple[Decimal, Decimal]]:
    start = Decimal(window["start_timestamp_s"])
    end = Decimal(window["end_timestamp_s"])
    rows = [timestamp for timestamp in shared if start <= timestamp < end]
    if len(rows) != int(window["expected_frame_count"]):
        raise ValueError(f"WINDOW_FRAME_COUNT:{window['window_index']}")
    pairs = list(zip(rows, rows[1:]))
    if len(pairs) != int(window["expected_pair_count"]):
        raise ValueError(f"WINDOW_PAIR_COUNT:{window['window_index']}")
    if any(
        not (Decimal("0") < current - previous <= maximum_dt)
        for previous, current in pairs
    ):
        raise ValueError(f"WINDOW_DT:{window['window_index']}")
    return pairs


def intrinsic_matrix(contract: dict[str, Any]) -> np.ndarray:
    values = contract["source"]["intrinsics"]
    return np.asarray(
        (
            (float(values["fx"]), 0.0, float(values["cx"])),
            (0.0, float(values["fy"]), float(values["cy"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def geometry_band(value: float) -> str:
    if value < 0.01:
        return "BELOW_TRIGGER_REFERENCE"
    if value < 0.05:
        return "WEAK_POSITIVE_RADIAL"
    return "POSITIVE_APPROACH_GEOMETRY"


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_stats(
    rows: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    longest_count = 0
    longest_duration = Decimal("0")
    longest_start: Decimal | None = None
    longest_end: Decimal | None = None
    longest_start_pair: int | None = None
    longest_end_pair: int | None = None
    current_count = 0
    current_start: Decimal | None = None
    current_start_pair: int | None = None
    previous_window: int | None = None
    previous_pair: int | None = None
    for row in rows:
        window_index = int(row["window_index"])
        pair_index = int(row["pair_index"])
        contiguous = (
            previous_window == window_index
            and previous_pair is not None
            and pair_index == previous_pair + 1
        )
        if predicate(row):
            if current_count == 0 or not contiguous:
                current_start = Decimal(str(row["previous_timestamp_s"]))
                current_start_pair = pair_index
                current_count = 0
            current_count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - current_start
            if current_count > longest_count or (
                current_count == longest_count and duration > longest_duration
            ):
                longest_count = current_count
                longest_duration = duration
                longest_start = current_start
                longest_end = Decimal(str(row["current_timestamp_s"]))
                longest_start_pair = current_start_pair
                longest_end_pair = pair_index
        else:
            current_count = 0
            current_start = None
            current_start_pair = None
        previous_window = window_index
        previous_pair = pair_index
    return {
        "pair_count": longest_count,
        "duration_s": float(longest_duration),
        "duration_decimal_s": str(longest_duration),
        "start_timestamp_s": (
            float(longest_start) if longest_start is not None else None
        ),
        "end_timestamp_s": float(longest_end) if longest_end is not None else None,
        "start_pair_index": longest_start_pair,
        "end_pair_index": longest_end_pair,
    }


def summarize_window(
    window: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("geometry_evaluable") is True]
    abstentions = Counter(
        str(row.get("geometry_abstention_reason"))
        for row in rows
        if row.get("geometry_evaluable") is not True
    )
    bands: dict[str, dict[str, Any]] = {}
    band_ids = [
        "BELOW_TRIGGER_REFERENCE",
        "WEAK_POSITIVE_RADIAL",
        "POSITIVE_APPROACH_GEOMETRY",
    ]
    for band in band_ids:
        selected = [row for row in evaluable if row["geometry_band"] == band]
        triggered = [row for row in selected if row["rgb_trigger"] is True]
        rgb_values = [
            float(row["rgb_compensated_expansion_per_s"]) for row in selected
        ]
        bands[band] = {
            "pair_count": len(selected),
            "rgb_trigger_count": len(triggered),
            "rgb_trigger_fraction": (
                len(triggered) / len(selected) if selected else None
            ),
            "median_rgb_compensated_expansion_per_s": (
                float(statistics.median(rgb_values)) if rgb_values else None
            ),
        }
    geometry_values = [
        float(row["geometry_signed_radial_expansion_per_s"]) for row in evaluable
    ]
    rgb_values = [
        float(row["rgb_compensated_expansion_per_s"]) for row in evaluable
    ]
    spearman = correlation(
        average_ranks(geometry_values), average_ranks(rgb_values)
    )
    pearson = correlation(geometry_values, rgb_values)
    triggered_geometry = [
        float(row["geometry_signed_radial_expansion_per_s"])
        for row in evaluable
        if row["rgb_trigger"] is True
    ]
    nontriggered_geometry = [
        float(row["geometry_signed_radial_expansion_per_s"])
        for row in evaluable
        if row["rgb_trigger"] is not True
    ]
    positive = [
        row
        for row in evaluable
        if row["geometry_band"] == "POSITIVE_APPROACH_GEOMETRY"
    ]
    triggered = [row for row in evaluable if row["rgb_trigger"] is True]
    overlap = [
        row
        for row in evaluable
        if row["rgb_trigger"] is True
        and row["geometry_band"] == "POSITIVE_APPROACH_GEOMETRY"
    ]
    union = [
        row
        for row in evaluable
        if row["rgb_trigger"] is True
        or row["geometry_band"] == "POSITIVE_APPROACH_GEOMETRY"
    ]
    start = Decimal(window["start_timestamp_s"])
    first_geometry_delay = (
        Decimal(str(positive[0]["current_timestamp_s"])) - start
        if positive
        else None
    )
    first_trigger_delay = (
        Decimal(str(triggered[0]["current_timestamp_s"])) - start
        if triggered
        else None
    )
    onset_delta = (
        first_trigger_delay - first_geometry_delay
        if first_geometry_delay is not None and first_trigger_delay is not None
        else None
    )
    return {
        "window_index": int(window["window_index"]),
        "role": window["role"],
        "candidate_pair_count": len(rows),
        "geometry_evaluable_pair_count": len(evaluable),
        "geometry_coverage": len(evaluable) / len(rows) if rows else 0.0,
        "geometry_abstention_count": len(rows) - len(evaluable),
        "geometry_abstention_reasons": dict(sorted(abstentions.items())),
        "geometry_bands": bands,
        "pearson_geometry_vs_rgb": pearson,
        "spearman_geometry_vs_rgb": spearman,
        "median_geometry_when_rgb_triggered_per_s": (
            float(statistics.median(triggered_geometry))
            if triggered_geometry
            else None
        ),
        "median_geometry_when_rgb_not_triggered_per_s": (
            float(statistics.median(nontriggered_geometry))
            if nontriggered_geometry
            else None
        ),
        "first_positive_geometry_delay_s": (
            float(first_geometry_delay)
            if first_geometry_delay is not None
            else None
        ),
        "first_rgb_trigger_delay_s": (
            float(first_trigger_delay) if first_trigger_delay is not None else None
        ),
        "rgb_minus_geometry_onset_delta_s": (
            float(onset_delta) if onset_delta is not None else None
        ),
        "longest_positive_geometry_run": run_stats(
            rows,
            lambda row: row.get("geometry_evaluable") is True
            and row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY",
        ),
        "longest_rgb_trigger_run": run_stats(
            rows, lambda row: row.get("rgb_trigger") is True
        ),
        "longest_same_pair_overlap_run": run_stats(
            rows,
            lambda row: row.get("geometry_evaluable") is True
            and row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY"
            and row.get("rgb_trigger") is True,
        ),
        "positive_trigger_intersection_count": len(overlap),
        "positive_trigger_union_count": len(union),
        "positive_trigger_jaccard": len(overlap) / len(union) if union else None,
        "positive_geometry_trigger_coverage": (
            len(overlap) / len(positive) if positive else None
        ),
        "trigger_positive_geometry_fraction": (
            len(overlap) / len(triggered) if triggered else None
        ),
    }


def descriptive_interpretation(
    window_zero: dict[str, Any], combined: dict[str, Any]
) -> str:
    del window_zero, combined
    return "DESCRIPTIVE_METRICS_REPORTED_NO_EFFECT_SIZE_GATE"


def summarize_all(
    contract: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    windows = [
        summarize_window(
            window,
            [row for row in rows if row["window_index"] == window["window_index"]],
        )
        for window in contract["windows"]
    ]
    combined_window = {
        "window_index": -1,
        "role": "COMBINED_DIAGNOSTIC",
        "start_timestamp_s": contract["windows"][0]["start_timestamp_s"],
    }
    combined = summarize_window(combined_window, rows)
    interpretation = descriptive_interpretation(windows[0], combined)
    return windows, combined, interpretation


def compute_geometry_rows(
    contract: dict[str, Any],
    archive: zipfile.ZipFile,
    algorithm_rows: Sequence[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    names = [item.filename for item in archive.infolist()]
    shared = shared_timestamps(names, contract["source"]["sequence_id"])
    maximum_dt = Decimal(contract["pair_geometry"]["maximum_pair_dt_s"])
    poses = geometry._parse_poses(
        archive.read(contract["source"]["pose_member"])
    )
    intrinsic = intrinsic_matrix(contract)
    expected: list[tuple[int, int, Decimal, Decimal]] = []
    for window in contract["windows"]:
        for pair_index, (previous, current) in enumerate(
            expected_pairs(shared, window, maximum_dt)
        ):
            expected.append(
                (int(window["window_index"]), pair_index, previous, current)
            )
    if len(algorithm_rows) != len(expected):
        raise ValueError("ALGORITHM_LEDGER_COUNT")
    tasks: list[tuple[Any, ...]] = []
    for algorithm_row, identity in zip(algorithm_rows, expected, strict=True):
        window_index, pair_index, previous, current = identity
        if (
            algorithm_row.get("window_index") != window_index
            or algorithm_row.get("pair_index") != pair_index
            or Decimal(str(algorithm_row.get("previous_timestamp_s"))) != previous
            or Decimal(str(algorithm_row.get("current_timestamp_s"))) != current
            or Decimal(str(algorithm_row.get("dt_s"))) != current - previous
        ):
            raise ValueError(f"ALGORITHM_PAIR_IDENTITY:{window_index}:{pair_index}")
        if algorithm_row.get("evaluable") is not True:
            raise ValueError(f"RGB_ALGORITHM_ABSTENTION_UNEXPECTED:{window_index}:{pair_index}")
        rgb_value = float(algorithm_row["compensated_expansion_median_per_s"])
        expected_trigger = rgb_value > 0.01
        if algorithm_row.get("trigger") is not expected_trigger:
            raise ValueError(f"RGB_TRIGGER_DRIFT:{window_index}:{pair_index}")
        base = {
            "window_index": window_index,
            "pair_index": pair_index,
            "previous_timestamp_s": float(previous),
            "current_timestamp_s": float(current),
            "dt_s": float(current - previous),
            "rgb_compensated_expansion_per_s": rgb_value,
            "rgb_trigger": expected_trigger,
        }
        depth_member = (
            f"{contract['source']['sequence_id']}/depth/{previous}.png"
        )
        raw_depth = archive.read(depth_member)
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
    rows: list[dict[str, Any]] = []
    if workers == 1:
        iterator = map(geometry._pair_worker, tasks)
        for index, (record, _) in enumerate(iterator, start=1):
            rows.append(_normalize_geometry_record(record))
            if index % 50 == 0 or index == len(tasks):
                print(f"geometry={index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (record, _) in enumerate(
                executor.map(geometry._pair_worker, tasks), start=1
            ):
                rows.append(_normalize_geometry_record(record))
                if index % 50 == 0 or index == len(tasks):
                    print(f"geometry={index}/{len(tasks)}", flush=True)
    return rows


def _normalize_geometry_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: record[key]
        for key in (
            "window_index",
            "pair_index",
            "previous_timestamp_s",
            "current_timestamp_s",
            "dt_s",
            "rgb_compensated_expansion_per_s",
            "rgb_trigger",
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


def run(
    repo_root: Path,
    contract_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_ALREADY_EXISTS")
    if workers < 1:
        raise ValueError("WORKERS")
    contract = load_object(contract_path)
    bindings, archive_sha256, archive_md5 = verify_inputs(repo_root, contract)
    algorithm_rows = load_jsonl(bindings["rgb-pair-ledger"])
    archive_path = repo_root / contract["source"]["archive_path"]
    with zipfile.ZipFile(archive_path) as archive:
        rows = compute_geometry_rows(contract, archive, algorithm_rows, workers)
    windows, combined, interpretation = summarize_all(contract, rows)
    coverage_valid = all(
        window["geometry_coverage"]
        >= float(contract["pair_geometry"]["minimum_pair_geometry_coverage_per_window"])
        for window in windows
    )
    terminal = (
        "POSTHOC_PAIRWISE_ALIGNMENT_COMPUTED / VALID"
        if coverage_valid
        else "POSTHOC_PAIRWISE_ALIGNMENT_NOT_EVALUABLE / VALID"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    write_exclusive(output_dir / "alignment_ledger.jsonl", ledger_text)
    result = {
        "schema_version": "rcle.rgb_algorithm.pairwise_geometry_alignment.result.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "status": "VALID",
        "authority": "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY",
        "algorithm_reexecution_performed": False,
        "threshold_tuned": False,
        "outcome_blind": False,
        "independent_confirmation": False,
        "performance_qualification": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "archive_sha256": archive_sha256,
        "archive_md5": archive_md5,
        "contract_sha256": digest_file(contract_path),
        "rgb_pair_ledger_sha256": digest_file(bindings["rgb-pair-ledger"]),
        "alignment_ledger_sha256": hashlib.sha256(
            ledger_text.encode("utf-8")
        ).hexdigest(),
        "windows": windows,
        "combined": combined,
        "interpretation": interpretation,
        "r0_evidence_status": "INVALID_R0_EVIDENCE / INVALID",
    }
    result["result_payload_sha256"] = canonical_sha(result)
    write_exclusive(
        output_dir / "result.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result
