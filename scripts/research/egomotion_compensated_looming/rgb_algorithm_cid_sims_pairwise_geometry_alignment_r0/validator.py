from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Callable, Iterable, Sequence
import zipfile

import numpy as np

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as frozen_geometry,
)


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0"
)
LOCK_SCHEMA = (
    "rcle.rgb_algorithm.pairwise_geometry_alignment.implementation_lock.v1"
)
ACTIVATION_SCHEMA = "rcle.rgb_algorithm.pairwise_geometry_alignment.activation.v1"
VALIDATION_SCHEMA = "rcle.rgb_algorithm.pairwise_geometry_alignment.validation.v1"
MAXIMUM_AUTHORITY = "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY"
EXPECTED_LOCK_PATHS = {
    "docs/research/rcle/RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0_CONTRACT_2026-07-27.json",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/producer.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/validator.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/run.py",
    "scripts/research/egomotion_compensated_looming/tests_rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/tests_rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0/test_alignment.py",
}
BAND_IDS = (
    "BELOW_TRIGGER_REFERENCE",
    "WEAK_POSITIVE_RADIAL",
    "POSITIVE_APPROACH_GEOMETRY",
)
EXPECTED_AUTHORITY = {
    "maximum_claim": MAXIMUM_AUTHORITY,
    "algorithm_reexecution_authorized": False,
    "threshold_tuning_authorized": False,
    "outcome_blind_claim_authorized": False,
    "independent_confirmation_authorized": False,
    "performance_qualification_authorized": False,
    "product_or_safety_claim_authorized": False,
}
EXPECTED_BINDINGS = {
    "rgb-pair-ledger": (
        "artifacts.local/evidence/rcle_rgb_algorithm_development_canary_r0_cid_sims_floor3_1/run_r0/pair_ledger.jsonl",
        "9b381e6b3a5f387e54e315e59e81474fc781818bdacc0e4744e255d41f8a391c",
    ),
    "rgb-result": (
        "artifacts.local/evidence/rcle_rgb_algorithm_development_canary_r0_cid_sims_floor3_1/run_r0/result.json",
        "0264b14c901fd1e7460a8d915c6a9be43dc1c1e9bf329594994717ad7d082da0",
    ),
    "rgb-posthoc-validation-r1": (
        "artifacts.local/evidence/rcle_rgb_algorithm_development_canary_r0_posthoc_validator_r1/validation_r1.json",
        "c0885614f46305099b18c35f0be52c266df3cdc89b80cb9c5f8dd6d22fc6eacf",
    ),
    "rgb-cache-manifest": (
        "artifacts.local/caches/rcle_rgb_algorithm_development_canary_r0_cid_sims_floor3_1/manifest.json",
        "d31cd91859f9008722f399522db3ed74bfdcd09ec0681cd5f2750d7f860c91e5",
    ),
    "cid-development-descriptor": (
        "docs/research/rcle/RCLE_CID_SIMS_FLOOR3_1_DEVELOPMENT_CANARY_DESCRIPTOR_2026-07-27.json",
        "4f49489f5accd08ddb35a7e6725717a9023da581301e555a21bf47d5608a693a",
    ),
    "cid-geometry-helper": (
        "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/producer.py",
        "0ce6256e12dd4536f284c7047f1e63faf955fa7bcf87f28fcb93c3e5d9de1add",
    ),
    "translation-geometry-helper": (
        "scripts/research/egomotion_compensated_looming/pb_h1_role_proxy/geometry.py",
        "b399228e82e70dfa2e27ca1fe9b9831749f18c0aa87b31e6e60de32b62c12016",
    ),
}
ACTIVATION_FALSE_FIELDS = (
    "algorithm_reexecution_authorized",
    "threshold_tuning_authorized",
    "outcome_blind_claim_authorized",
    "independent_confirmation_authorized",
    "performance_qualification_authorized",
    "product_or_safety_claim_authorized",
    "network_access_authorized",
    "download_authorized",
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
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        value = json.loads(line, parse_float=Decimal)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{line_index}")
        rows.append(value)
    return rows


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _float_equal(actual: Any, expected: float) -> bool:
    return _finite_number(actual) and float(actual).hex() == expected.hex()


def _compare_exact(
    observed: Any, expected: Any, label: str, errors: list[str]
) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            errors.append(f"{label}:OBJECT")
            return
        observed_keys = set(observed)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            missing = ",".join(sorted(expected_keys - observed_keys))
            extra = ",".join(sorted(observed_keys - expected_keys))
            errors.append(f"{label}:KEYS:MISSING={missing}:EXTRA={extra}")
        for key in sorted(expected_keys & observed_keys):
            _compare_exact(
                observed[key], expected[key], f"{label}.{key}", errors
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list):
            errors.append(f"{label}:LIST")
            return
        if len(observed) != len(expected):
            errors.append(f"{label}:LENGTH:{len(observed)}:{len(expected)}")
        for index, (actual_item, expected_item) in enumerate(
            zip(observed, expected)
        ):
            _compare_exact(
                actual_item, expected_item, f"{label}[{index}]", errors
            )
        return
    if isinstance(expected, float):
        if not _float_equal(observed, expected):
            errors.append(f"{label}:FLOAT_HEX")
        return
    if isinstance(expected, bool):
        if type(observed) is not bool or observed is not expected:
            errors.append(f"{label}:BOOLEAN")
        return
    if isinstance(expected, int):
        if type(observed) is not int or observed != expected:
            errors.append(f"{label}:INTEGER")
        return
    if observed != expected:
        errors.append(f"{label}:VALUE")


def _geometry_band(value: float) -> str:
    if value < 0.01:
        return "BELOW_TRIGGER_REFERENCE"
    if value < 0.05:
        return "WEAK_POSITIVE_RADIAL"
    return "POSITIVE_APPROACH_GEOMETRY"


def _shared_timestamps(
    names: Iterable[str], sequence_id: str
) -> tuple[list[Decimal], list[str]]:
    errors: list[str] = []
    color: set[Decimal] = set()
    depth: set[Decimal] = set()
    seen_names: set[str] = set()
    for raw in names:
        if raw in seen_names:
            errors.append(f"ZIP_MEMBER_NAME_DUPLICATE:{raw}")
        seen_names.add(raw)
        name = PurePosixPath(raw)
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in raw
            or len(name.parts) != 3
            or name.parts[0] != sequence_id
            or name.suffix.lower() != ".png"
        ):
            continue
        try:
            timestamp = Decimal(name.stem)
        except InvalidOperation:
            continue
        target = color if name.parts[1] == "color" else (
            depth if name.parts[1] == "depth" else None
        )
        if target is None:
            continue
        if timestamp in target:
            errors.append(
                f"ZIP_{name.parts[1].upper()}_TIMESTAMP_DUPLICATE:{timestamp}"
            )
        target.add(timestamp)
    return sorted(color & depth), errors


def _window_pairs(
    shared: Sequence[Decimal],
    window: dict[str, Any],
    maximum_dt: Decimal,
) -> tuple[list[tuple[Decimal, Decimal]], list[str]]:
    errors: list[str] = []
    window_index = int(window["window_index"])
    start = Decimal(window["start_timestamp_s"])
    end = Decimal(window["end_timestamp_s"])
    timestamps = [item for item in shared if start <= item < end]
    if len(timestamps) != int(window["expected_frame_count"]):
        errors.append(
            f"ZIP_WINDOW_FRAME_COUNT:{window_index}:{len(timestamps)}"
        )
    pairs = list(zip(timestamps, timestamps[1:]))
    if len(pairs) != int(window["expected_pair_count"]):
        errors.append(f"ZIP_WINDOW_PAIR_COUNT:{window_index}:{len(pairs)}")
    for pair_index, (previous, current) in enumerate(pairs):
        if not (Decimal("0") < current - previous <= maximum_dt):
            errors.append(
                f"ZIP_PAIR_DT:{window_index}:{pair_index}:{current - previous}"
            )
    return pairs, errors


def _intrinsic_matrix(contract: dict[str, Any]) -> np.ndarray:
    intrinsic = contract["source"]["intrinsics"]
    return np.asarray(
        (
            (float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])),
            (0.0, float(intrinsic["fy"]), float(intrinsic["cy"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _verify_contract(
    repo_root: Path, contract_path: Path, contract: dict[str, Any]
) -> tuple[dict[str, Path], list[dict[str, str]], Path, list[str]]:
    errors: list[str] = []
    if contract.get("schema_version") != (
        "rcle.rgb_algorithm.pairwise_geometry_alignment.v1"
    ):
        errors.append("CONTRACT_SCHEMA")
    if contract.get("protocol_id") != PROTOCOL_ID:
        errors.append("CONTRACT_PROTOCOL")
    if contract.get("status") != (
        "OUTCOME_AWARE_POSTHOC_PREREGISTERED_BEFORE_FULL_PAIR_GEOMETRY_ACCESS"
    ):
        errors.append("CONTRACT_STATUS")
    if contract.get("authority") != EXPECTED_AUTHORITY:
        errors.append("CONTRACT_AUTHORITY_ESCALATION")
    if contract.get("rgb_trigger") != {
        "metric": "immutable ledger compensated_expansion_median_per_s",
        "operator": "GT",
        "threshold_per_s": "0.01",
        "threshold_tuning_forbidden": True,
    }:
        errors.append("CONTRACT_RGB_TRIGGER")
    expected_windows = [
        {
            "window_index": 0,
            "role": "WEAK_MOTION_ADJACENT_CONTROL",
            "start_timestamp_s": "1673419222.281298",
            "end_timestamp_s": "1673419232.281298",
            "expected_frame_count": 300,
            "expected_pair_count": 299,
        },
        {
            "window_index": 1,
            "role": "POSITIVE_APPROACH_DEVELOPMENT_CANARY",
            "start_timestamp_s": "1673419232.281298",
            "end_timestamp_s": "1673419242.281298",
            "expected_frame_count": 300,
            "expected_pair_count": 299,
        },
    ]
    if contract.get("windows") != expected_windows:
        errors.append("CONTRACT_WINDOWS")
    source = contract.get("source")
    if not isinstance(source, dict):
        return {}, [], repo_root / "__MISSING__", errors + ["CONTRACT_SOURCE"]
    if (
        source.get("archive_path")
        != "artifacts.local/datasets/cid_sims_v6/office_building/floor3/floor3_1.zip"
        or source.get("archive_bytes") != 2_211_008_069
        or source.get("archive_md5") != "585d38855ad7d04817991cdbbb72016b"
        or source.get("archive_sha256")
        != "b622be7918d0003c97f0e33cc30071c9995f49c59726240e7475f2cde8572984"
        or source.get("sequence_id") != "floor3_1"
        or source.get("pose_member") != "floor3_1/groundtruth.txt"
        or source.get("intrinsics")
        != {
            "fx": 386.52199190267083,
            "fy": 387.32300428823663,
            "cx": 326.5103569741365,
            "cy": 237.40293732598795,
        }
        or
        source.get("network_access_forbidden") is not True
        or source.get("download_forbidden") is not True
        or source.get("depth_sample_stride_px")
        != frozen_geometry.DEPTH_SAMPLE_STRIDE_PX
        or float(source.get("depth_units_per_meter", -1))
        != frozen_geometry.DEPTH_UNITS_PER_METER
        or float(source.get("minimum_radius_px", -1))
        != frozen_geometry.MINIMUM_RADIUS_PX
        or not np.array_equal(
            np.asarray(
                (
                    (
                        float(source.get("intrinsics", {}).get("fx", float("nan"))),
                        0.0,
                        float(source.get("intrinsics", {}).get("cx", float("nan"))),
                    ),
                    (
                        0.0,
                        float(source.get("intrinsics", {}).get("fy", float("nan"))),
                        float(source.get("intrinsics", {}).get("cy", float("nan"))),
                    ),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            ),
            frozen_geometry.INTRINSIC,
        )
    ):
        errors.append("CONTRACT_SOURCE_GEOMETRY_OR_ACCESS")
    pair_geometry = contract.get("pair_geometry")
    if (
        not isinstance(pair_geometry, dict)
        or pair_geometry.get("maximum_pair_dt_s") != "0.100"
        or float(
            pair_geometry.get(
                "minimum_pair_geometry_coverage_per_window", -1
            )
        )
        != 0.8
        or pair_geometry.get("abstention_imputation_forbidden") is not True
        or frozen_geometry.MAX_POSE_BRACKET_SECONDS != Decimal("0.100")
    ):
        errors.append("CONTRACT_PAIR_GEOMETRY")
    expected_bands = [
        ("BELOW_TRIGGER_REFERENCE", "signed_radial_expansion_per_s < 0.01"),
        (
            "WEAK_POSITIVE_RADIAL",
            "0.01 <= signed_radial_expansion_per_s < 0.05",
        ),
        (
            "POSITIVE_APPROACH_GEOMETRY",
            "signed_radial_expansion_per_s >= 0.05",
        ),
    ]
    observed_bands = contract.get("frozen_geometry_bands")
    if not isinstance(observed_bands, list) or [
        (item.get("id"), item.get("rule"))
        for item in observed_bands
        if isinstance(item, dict)
    ] != expected_bands:
        errors.append("CONTRACT_GEOMETRY_BANDS")
    bindings = contract.get("immutable_bindings")
    expected_ids = set(EXPECTED_BINDINGS)
    paths: dict[str, Path] = {}
    verified: list[dict[str, str]] = []
    if not isinstance(bindings, list):
        errors.append("IMMUTABLE_BINDINGS_NOT_LIST")
        bindings = []
    ids = [
        str(item.get("id", ""))
        for item in bindings
        if isinstance(item, dict)
    ]
    if (
        len(ids) != len(expected_ids)
        or len(ids) != len(set(ids))
        or set(ids) != expected_ids
    ):
        errors.append("IMMUTABLE_BINDING_INVENTORY")
    for item in bindings:
        if not isinstance(item, dict):
            errors.append("IMMUTABLE_BINDING_OBJECT")
            continue
        binding_id = str(item.get("id", ""))
        expected_binding = EXPECTED_BINDINGS.get(binding_id)
        if expected_binding is None or (
            item.get("path"),
            item.get("sha256"),
        ) != expected_binding:
            errors.append(f"IMMUTABLE_BINDING_DECLARATION:{binding_id}")
        path = repo_root / str(item.get("path", ""))
        paths[binding_id] = path
        actual = digest_file(path) if path.is_file() else "MISSING"
        expected = str(item.get("sha256", ""))
        verified.append(
            {
                "id": binding_id,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
        if actual != expected:
            errors.append(f"IMMUTABLE_BINDING:{binding_id}")
    archive_path = repo_root / str(source.get("archive_path", ""))
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != source.get("archive_bytes")
    ):
        errors.append("ARCHIVE_SIZE")
    else:
        archive_sha256, archive_md5 = source_digests(archive_path)
        if archive_sha256 != source.get("archive_sha256"):
            errors.append("ARCHIVE_SHA256")
        if archive_md5 != source.get("archive_md5"):
            errors.append("ARCHIVE_MD5")
    if digest_file(contract_path) != digest_file(
        repo_root
        / "docs/research/rcle/RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0_CONTRACT_2026-07-27.json"
    ):
        errors.append("CONTRACT_PATH_IDENTITY")
    return paths, verified, archive_path, errors


def _verify_locks(
    repo_root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    implementation_lock_path: Path,
    activation_path: Path,
    output_dir: Path,
) -> list[str]:
    errors: list[str] = []
    lock = load_object(implementation_lock_path)
    activation = load_object(activation_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        errors.append("IMPLEMENTATION_LOCK_SCHEMA")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("IMPLEMENTATION_LOCK_PROTOCOL")
    if lock.get("status") != "LOCKED_BEFORE_FULL_PAIR_GEOMETRY_ACCESS":
        errors.append("IMPLEMENTATION_LOCK_STATUS")
    files = lock.get("files")
    if not isinstance(files, list):
        errors.append("IMPLEMENTATION_LOCK_FILES")
        files = []
    paths = [
        str(item.get("path", ""))
        for item in files
        if isinstance(item, dict)
    ]
    if (
        len(paths) != len(EXPECTED_LOCK_PATHS)
        or len(paths) != len(set(paths))
        or set(paths) != EXPECTED_LOCK_PATHS
    ):
        errors.append("IMPLEMENTATION_LOCK_ALLOWLIST")
    for item in files:
        if not isinstance(item, dict):
            errors.append("IMPLEMENTATION_LOCK_FILE_OBJECT")
            continue
        relative = str(item.get("path", ""))
        path = repo_root / relative
        if not path.is_file() or digest_file(path) != item.get("sha256"):
            errors.append(f"IMPLEMENTATION_FILE:{relative}")
    if activation.get("schema_version") != ACTIVATION_SCHEMA:
        errors.append("ACTIVATION_SCHEMA")
    if activation.get("protocol_id") != PROTOCOL_ID:
        errors.append("ACTIVATION_PROTOCOL")
    if activation.get("status") != "AUTHORIZED_FOR_ONE_PAIRWISE_ALIGNMENT_RUN":
        errors.append("ACTIVATION_STATUS")
    if activation.get("maximum_authority") != MAXIMUM_AUTHORITY:
        errors.append("ACTIVATION_AUTHORITY")
    for field in ACTIVATION_FALSE_FIELDS:
        if activation.get(field) is not False:
            errors.append(f"ACTIVATION_FORBIDDEN:{field}")
    if activation.get("implementation_lock_sha256") != digest_file(
        implementation_lock_path
    ):
        errors.append("ACTIVATION_IMPLEMENTATION_LOCK_SHA")
    if activation.get("contract_sha256") != digest_file(contract_path):
        errors.append("ACTIVATION_CONTRACT_SHA")
    if activation.get("archive_sha256") != contract.get("source", {}).get(
        "archive_sha256"
    ):
        errors.append("ACTIVATION_ARCHIVE_SHA")
    try:
        expected_output = str(output_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        errors.append("OUTPUT_OUTSIDE_REPOSITORY")
    else:
        if activation.get("output_dir") != expected_output:
            errors.append("ACTIVATION_OUTPUT_DIR")
    return errors


def _algorithm_identity(
    row: dict[str, Any],
    window_index: int,
    pair_index: int,
    previous: Decimal,
    current: Decimal,
    errors: list[str],
) -> tuple[float, bool]:
    label = f"RGB_LEDGER:{window_index}:{pair_index}"
    if row.get("window_index") != window_index:
        errors.append(f"{label}:WINDOW_INDEX")
    if row.get("pair_index") != pair_index:
        errors.append(f"{label}:PAIR_INDEX")
    try:
        actual_previous = Decimal(str(row.get("previous_timestamp_s")))
        actual_current = Decimal(str(row.get("current_timestamp_s")))
        actual_dt = Decimal(str(row.get("dt_s")))
    except InvalidOperation:
        errors.append(f"{label}:TIMESTAMP_PARSE")
    else:
        if actual_previous != previous:
            errors.append(f"{label}:PREVIOUS_TIMESTAMP")
        if actual_current != current:
            errors.append(f"{label}:CURRENT_TIMESTAMP")
        if actual_dt != current - previous:
            errors.append(f"{label}:DT")
    if row.get("evaluable") is not True:
        errors.append(f"{label}:ABSTENTION")
    value_raw = row.get("compensated_expansion_median_per_s")
    if not _finite_number(value_raw):
        errors.append(f"{label}:RGB_NUMERIC")
        value = float("nan")
    else:
        value = float(value_raw)
    trigger = value > 0.01
    if row.get("trigger") is not trigger:
        errors.append(f"{label}:TRIGGER")
    if (
        not _finite_number(row.get("trigger_threshold_per_s"))
        or Decimal(str(row.get("trigger_threshold_per_s"))) != Decimal("0.01")
    ):
        errors.append(f"{label}:THRESHOLD")
    return value, trigger


def _normalize_geometry(record: dict[str, Any]) -> dict[str, Any]:
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
        geometry_band=_geometry_band(signed),
    )
    return result


def _recompute_rows(
    contract: dict[str, Any],
    archive: zipfile.ZipFile,
    algorithm_rows: Sequence[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[str], int]:
    errors: list[str] = []
    infos = archive.infolist()
    names = [item.filename for item in infos]
    shared, inventory_errors = _shared_timestamps(
        names, str(contract["source"]["sequence_id"])
    )
    errors.extend(inventory_errors)
    maximum_dt = Decimal(contract["pair_geometry"]["maximum_pair_dt_s"])
    expected: list[tuple[int, int, Decimal, Decimal]] = []
    for window in contract["windows"]:
        pairs, pair_errors = _window_pairs(shared, window, maximum_dt)
        errors.extend(pair_errors)
        expected.extend(
            (
                int(window["window_index"]),
                pair_index,
                previous,
                current,
            )
            for pair_index, (previous, current) in enumerate(pairs)
        )
    if len(expected) != 598:
        errors.append(f"EXPECTED_PAIR_TOTAL:{len(expected)}")
    if len(algorithm_rows) != len(expected):
        errors.append(f"RGB_LEDGER_COUNT:{len(algorithm_rows)}:{len(expected)}")
        return [], errors, len(infos)
    pose_member = str(contract["source"]["pose_member"])
    if pose_member not in set(names):
        errors.append("POSE_MEMBER_MISSING")
        return [], errors, len(infos)
    try:
        poses = frozen_geometry._parse_poses(archive.read(pose_member))
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        errors.append(f"POSE_MEMBER_INVALID:{type(error).__name__}:{error}")
        return [], errors, len(infos)
    intrinsic = _intrinsic_matrix(contract)
    tasks: list[tuple[Any, ...]] = []
    for algorithm_row, identity in zip(algorithm_rows, expected, strict=True):
        window_index, pair_index, previous, current = identity
        rgb_value, rgb_trigger = _algorithm_identity(
            algorithm_row,
            window_index,
            pair_index,
            previous,
            current,
            errors,
        )
        record = {
            "window_index": window_index,
            "pair_index": pair_index,
            "previous_timestamp_s": float(previous),
            "current_timestamp_s": float(current),
            "dt_s": float(current - previous),
            "rgb_compensated_expansion_per_s": rgb_value,
            "rgb_trigger": rgb_trigger,
        }
        depth_member = (
            f"{contract['source']['sequence_id']}/depth/{previous}.png"
        )
        if depth_member not in set(names):
            errors.append(f"DEPTH_MEMBER_MISSING:{window_index}:{pair_index}")
            continue
        try:
            raw_depth = archive.read(depth_member)
            previous_pose = frozen_geometry._interpolate_pose(poses, previous)
            current_pose = frozen_geometry._interpolate_pose(poses, current)
        except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            errors.append(
                f"PAIR_SOURCE_INVALID:{window_index}:{pair_index}:"
                f"{type(error).__name__}:{error}"
            )
            continue
        tasks.append(
            (
                record,
                raw_depth,
                intrinsic,
                previous_pose,
                current_pose,
                float(current - previous),
            )
        )
    if errors:
        return [], errors, len(infos)
    rows: list[dict[str, Any]] = []
    if workers == 1:
        evaluated = map(frozen_geometry._pair_worker, tasks)
        for index, (record, _) in enumerate(evaluated, start=1):
            rows.append(_normalize_geometry(record))
            if index % 50 == 0 or index == len(tasks):
                print(f"validator_geometry={index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (record, _) in enumerate(
                executor.map(frozen_geometry._pair_worker, tasks), start=1
            ):
                rows.append(_normalize_geometry(record))
                if index % 50 == 0 or index == len(tasks):
                    print(f"validator_geometry={index}/{len(tasks)}", flush=True)
    return rows, errors, len(infos)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
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


def _correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _run_stats(
    rows: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
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
            duration = (
                Decimal(str(row["current_timestamp_s"])) - current_start
            )
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
        "end_timestamp_s": (
            float(longest_end) if longest_end is not None else None
        ),
        "start_pair_index": longest_start_pair,
        "end_pair_index": longest_end_pair,
    }


def _summarize_window(
    window: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    evaluable = [
        row for row in rows if row.get("geometry_evaluable") is True
    ]
    abstentions = Counter(
        str(row.get("geometry_abstention_reason"))
        for row in rows
        if row.get("geometry_evaluable") is not True
    )
    bands: dict[str, dict[str, Any]] = {}
    for band in BAND_IDS:
        selected = [
            row for row in evaluable if row["geometry_band"] == band
        ]
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
        float(row["geometry_signed_radial_expansion_per_s"])
        for row in evaluable
    ]
    rgb_values = [
        float(row["rgb_compensated_expansion_per_s"]) for row in evaluable
    ]
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
    geometry_delay = (
        Decimal(str(positive[0]["current_timestamp_s"])) - start
        if positive
        else None
    )
    trigger_delay = (
        Decimal(str(triggered[0]["current_timestamp_s"])) - start
        if triggered
        else None
    )
    onset_delta = (
        trigger_delay - geometry_delay
        if geometry_delay is not None and trigger_delay is not None
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
        "pearson_geometry_vs_rgb": _correlation(
            geometry_values, rgb_values
        ),
        "spearman_geometry_vs_rgb": _correlation(
            _average_ranks(geometry_values), _average_ranks(rgb_values)
        ),
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
            float(geometry_delay) if geometry_delay is not None else None
        ),
        "first_rgb_trigger_delay_s": (
            float(trigger_delay) if trigger_delay is not None else None
        ),
        "rgb_minus_geometry_onset_delta_s": (
            float(onset_delta) if onset_delta is not None else None
        ),
        "longest_positive_geometry_run": _run_stats(
            rows,
            lambda row: row.get("geometry_evaluable") is True
            and row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY",
        ),
        "longest_rgb_trigger_run": _run_stats(
            rows, lambda row: row.get("rgb_trigger") is True
        ),
        "longest_same_pair_overlap_run": _run_stats(
            rows,
            lambda row: row.get("geometry_evaluable") is True
            and row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY"
            and row.get("rgb_trigger") is True,
        ),
        "positive_trigger_intersection_count": len(overlap),
        "positive_trigger_union_count": len(union),
        "positive_trigger_jaccard": (
            len(overlap) / len(union) if union else None
        ),
        "positive_geometry_trigger_coverage": (
            len(overlap) / len(positive) if positive else None
        ),
        "trigger_positive_geometry_fraction": (
            len(overlap) / len(triggered) if triggered else None
        ),
    }


def _descriptive_interpretation(
    window_zero: dict[str, Any], combined: dict[str, Any]
) -> str:
    del window_zero, combined
    return "DESCRIPTIVE_METRICS_REPORTED_NO_EFFECT_SIZE_GATE"


def _summarize_all(
    contract: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    windows = [
        _summarize_window(
            window,
            [
                row
                for row in rows
                if row["window_index"] == window["window_index"]
            ],
        )
        for window in contract["windows"]
    ]
    combined = _summarize_window(
        {
            "window_index": -1,
            "role": "COMBINED_DIAGNOSTIC",
            "start_timestamp_s": contract["windows"][0][
                "start_timestamp_s"
            ],
        },
        rows,
    )
    return windows, combined, _descriptive_interpretation(windows[0], combined)


def _base_payload(
    errors: Sequence[str], valid_terminal: str | None = None
) -> dict[str, Any]:
    valid = not errors
    return {
        "schema_version": VALIDATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "terminal": (
            (
                valid_terminal
                or "POSTHOC_PAIRWISE_ALIGNMENT_COMPUTED / VALID"
            )
            if valid
            else "POSTHOC_PAIRWISE_ALIGNMENT_INVALID / INVALID"
        ),
        "status": "VALID" if valid else "INVALID",
        "errors": sorted(set(errors)),
        "authority": MAXIMUM_AUTHORITY,
        "algorithm_reexecution_performed": False,
        "pair_geometry_recomputed": True,
        "alignment_aggregates_recomputed": True,
        "validator_imported_alignment_producer": False,
        "threshold_tuned": False,
        "outcome_blind": False,
        "independent_confirmation": False,
        "performance_qualification": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "r0_evidence_status": "INVALID_R0_EVIDENCE / INVALID",
    }


def validate(
    repo_root: Path,
    contract_path: Path,
    output_dir: Path,
    implementation_lock_path: Path,
    activation_path: Path,
    workers: int,
) -> dict[str, Any]:
    errors: list[str] = []
    verified_bindings: list[dict[str, str]] = []
    if workers < 1:
        errors.append("WORKERS")
        return _base_payload(errors)
    try:
        contract = load_object(contract_path)
        paths, verified_bindings, archive_path, contract_errors = (
            _verify_contract(repo_root, contract_path, contract)
        )
        errors.extend(contract_errors)
        errors.extend(
            _verify_locks(
                repo_root,
                contract_path,
                contract,
                implementation_lock_path,
                activation_path,
                output_dir,
            )
        )
        result_path = output_dir / "result.json"
        ledger_path = output_dir / "alignment_ledger.jsonl"
        if not result_path.is_file():
            errors.append("RESULT_MISSING")
        if not ledger_path.is_file():
            errors.append("ALIGNMENT_LEDGER_MISSING")
        if errors:
            payload = _base_payload(errors)
            payload["verified_bindings"] = verified_bindings
            return payload
        result = load_object(result_path)
        observed_rows = load_jsonl(ledger_path)
        algorithm_rows = load_jsonl(paths["rgb-pair-ledger"])
        with zipfile.ZipFile(archive_path) as archive:
            expected_rows, geometry_errors, zip_member_count = (
                _recompute_rows(contract, archive, algorithm_rows, workers)
            )
        errors.extend(geometry_errors)
        expected_identities = [
            (row["window_index"], row["pair_index"]) for row in expected_rows
        ]
        observed_identities = [
            (
                int(row.get("window_index", -1)),
                int(row.get("pair_index", -1)),
            )
            for row in observed_rows
        ]
        if observed_identities != expected_identities:
            errors.append("ALIGNMENT_LEDGER_GLOBAL_ORDER_OR_INVENTORY")
        if len(observed_rows) != len(expected_rows):
            errors.append(
                f"ALIGNMENT_LEDGER_COUNT:{len(observed_rows)}:"
                f"{len(expected_rows)}"
            )
        for index, (observed, expected) in enumerate(
            zip(observed_rows, expected_rows)
        ):
            _compare_exact(observed, expected, f"ALIGNMENT_ROW:{index}", errors)
        if not geometry_errors:
            windows, combined, interpretation = _summarize_all(
                contract, expected_rows
            )
            coverage_valid = all(
                window["geometry_coverage"]
                >= float(
                    contract["pair_geometry"][
                        "minimum_pair_geometry_coverage_per_window"
                    ]
                )
                for window in windows
            )
            expected_terminal = (
                "POSTHOC_PAIRWISE_ALIGNMENT_COMPUTED / VALID"
                if coverage_valid
                else "POSTHOC_PAIRWISE_ALIGNMENT_NOT_EVALUABLE / VALID"
            )
            expected_result: dict[str, Any] = {
                "schema_version": (
                    "rcle.rgb_algorithm.pairwise_geometry_alignment.result.v1"
                ),
                "protocol_id": PROTOCOL_ID,
                "terminal": expected_terminal,
                "status": "VALID",
                "authority": MAXIMUM_AUTHORITY,
                "algorithm_reexecution_performed": False,
                "threshold_tuned": False,
                "outcome_blind": False,
                "independent_confirmation": False,
                "performance_qualification": False,
                "network_request_count": 0,
                "downloaded_bytes": 0,
                "archive_sha256": contract["source"]["archive_sha256"],
                "archive_md5": contract["source"]["archive_md5"],
                "contract_sha256": digest_file(contract_path),
                "rgb_pair_ledger_sha256": digest_file(
                    paths["rgb-pair-ledger"]
                ),
                "alignment_ledger_sha256": digest_file(ledger_path),
                "windows": windows,
                "combined": combined,
                "interpretation": interpretation,
                "r0_evidence_status": "INVALID_R0_EVIDENCE / INVALID",
            }
            expected_result["result_payload_sha256"] = canonical_sha(
                expected_result
            )
            _compare_exact(result, expected_result, "RESULT", errors)
        else:
            windows = []
            combined = {}
            interpretation = "NOT_COMPUTED"
        payload = _base_payload(
            errors,
            expected_terminal if not geometry_errors else None,
        )
        payload["verified_bindings"] = verified_bindings
        payload["archive_identity"] = {
            "bytes": archive_path.stat().st_size,
            "sha256": contract["source"]["archive_sha256"],
            "md5": contract["source"]["archive_md5"],
            "zip_member_count": zip_member_count,
        }
        payload["alignment_ledger_sha256"] = digest_file(ledger_path)
        payload["result_sha256"] = digest_file(result_path)
        payload["recomputed_pair_count"] = len(expected_rows)
        payload["recomputed_windows"] = windows
        payload["recomputed_combined"] = combined
        payload["recomputed_interpretation"] = interpretation
        return payload
    except Exception as error:
        errors.append(
            f"VALIDATOR_EXCEPTION:{type(error).__name__}:{str(error)}"
        )
        payload = _base_payload(errors)
        payload["verified_bindings"] = verified_bindings
        return payload
