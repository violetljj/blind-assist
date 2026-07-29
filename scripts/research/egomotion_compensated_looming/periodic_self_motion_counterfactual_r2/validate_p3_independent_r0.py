"""Independent P3 validator.

This file intentionally does not import the P3 producer, transport adapter,
analysis implementation, R3 pair core, or runtime runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
P3_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0"
)
BLOCK = "ADVIO_14"
GIB = 1024**3
FRAME_COUNT = 602
PAIR_COUNT = 601
FORMAL_SEQUENCE_COUNT = 496
FORMAL_FACTORIAL_CLUSTERS = 80
FORMAL_GUARDRAIL_CLUSTERS = 8
SCHEDULER_NATIVE_THREADS = {4: 4, 8: 18}
RETRY_RESERVE_FRACTION = 0.10
WALL_CEILING_SECONDS = 12.0 * 3600.0
FACTORIAL_ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
GUARDRAIL_ARMS = (
    "MONOTONIC_APPROACH_ONLY__CLEAN",
    "MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF__CLEAN",
)
FAMILY = (
    "MOTION_CLEAN",
    "BLUR_STATIC",
    "LOW_TEXTURE_STATIC",
    "MOTION_X_BLUR",
    "MOTION_X_LOW_TEXTURE",
    "MOTION_BLUR_VS_STATIC_CLEAN",
    "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN",
    "BLUR_FAILURE_UNION_STATIC",
    "LOW_TEXTURE_FAILURE_UNION_STATIC",
)
FIXTURE_THETA = {
    "MOTION_CLEAN": 0.03,
    "BLUR_STATIC": 0.02,
    "LOW_TEXTURE_STATIC": 0.01,
    "MOTION_X_BLUR": 0.0,
    "MOTION_X_LOW_TEXTURE": 0.0,
    "MOTION_BLUR_VS_STATIC_CLEAN": 0.05,
    "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN": 0.04,
    "BLUR_FAILURE_UNION_STATIC": 0.12,
    "LOW_TEXTURE_FAILURE_UNION_STATIC": 0.11,
}


class InvalidP3(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidP3(f"OBJECT_REQUIRED:{path}")
    return value


def expected_identity_lock() -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for kind, arms in (
        ("FACTORIAL", FACTORIAL_ARMS),
        ("GUARDRAIL", GUARDRAIL_ARMS),
    ):
        token = f"{PROTOCOL_ID}|PREFLIGHT|{BLOCK}|{kind}|00"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        seed = int.from_bytes(bytes.fromhex(digest)[:8], "big")
        record = {
            "kind": kind,
            "token": token,
            "token_sha256": digest,
            "numeric_seed_uint64": seed,
        }
        seeds.append(record)
        for ordinal, arm in enumerate(arms):
            identities.append(
                {
                    "sequence_id": f"PREFLIGHT_{BLOCK}_{kind}_00__{arm}",
                    "cluster_kind": kind,
                    "cluster_token_sha256": digest,
                    "numeric_seed_uint64": seed,
                    "arm": arm,
                    "arm_ordinal": ordinal,
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
            )
    value = {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "p3_id": P3_ID,
        "role": "RUNTIME_PREFLIGHT_ONLY_NO_SCIENTIFIC_INTERPRETATION",
        "block": BLOCK,
        "seed_literal_case": "UPPERCASE_FACTORIAL_AND_GUARDRAIL",
        "seeds": seeds,
        "identities": identities,
        "identity_count": 8,
        "worker_profiles": [4, 8],
        "prohibited_worker_profiles": [12, 16],
        "formal_seed_access": False,
        "formal_execution_authorized": False,
        "p4_activated": False,
    }
    value["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return value


def validate_identity_lock(path: Path) -> dict[str, Any]:
    value = load(path)
    if canonical_bytes(value) != canonical_bytes(expected_identity_lock()):
        raise InvalidP3("IDENTITY_LOCK_DRIFT")
    return value


def validate_transport_lock(root: Path, path: Path) -> dict[str, Any]:
    value = load(path)
    if value.get("protocol_id") != PROTOCOL_ID:
        raise InvalidP3("TRANSPORT_PROTOCOL")
    if value.get("terminal") != (
        "TRANSPORT_EQUIVALENCE_PASS / VALID / PREFLIGHT_ONLY"
    ):
        raise InvalidP3("TRANSPORT_TERMINAL")
    fixture = value.get("fixture", {})
    required_true = (
        "nonzero_rotation_covered",
        "rgb_channel_sentinel",
        "partial_valid_mask",
        "continuous_pair_state",
        "rows_equal",
        "state_equal",
    )
    if fixture.get("pair_count") != 4 or any(
        fixture.get(key) is not True for key in required_true
    ):
        raise InvalidP3("TRANSPORT_EQUIVALENCE_EVIDENCE")
    if (
        fixture.get("pair_row_sha256")
        != fixture.get("reference_pair_row_sha256")
    ):
        raise InvalidP3("TRANSPORT_PAIR_HASH")
    for binding in value.get("bindings", []):
        target = root / binding["path"]
        if not target.is_file() or sha256_file(target) != binding["sha256"]:
            raise InvalidP3(f"TRANSPORT_BINDING:{binding['path']}")
    if (
        value.get("formal_execution_authorized") is not False
        or value.get("p4_activated") is not False
        or value.get("scientific_outcome_interpreted") is not False
    ):
        raise InvalidP3("TRANSPORT_AUTHORITY")
    return value


def validate_analysis_lock(root: Path, path: Path) -> dict[str, Any]:
    value = load(path)
    frozen = value.get("frozen_contract", {})
    expected = {
        "blocks": ["ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17"],
        "clusters_per_block": 20,
        "arms": [
            f"{motion}__{quality}"
            for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
            for quality in ("CLEAN", "BLUR", "LOW_TEXTURE")
        ],
        "pair_count": 601,
        "threshold_operator": "strict_greater_than",
        "threshold_per_s": 0.01,
        "required_consecutive_pairs": 3,
        "family": list(FAMILY),
        "bootstrap_seed": 20260728,
        "bootstrap_replicates": 20000,
        "bootstrap_sd_ddof": 1,
        "quantile_method": "linear_type_7",
        "overall_weighting": "equal_four_block_mean",
        "resampling": "one_shared_cluster_draw_matrix_for_all_nine",
    }
    if frozen != expected:
        raise InvalidP3("ANALYSIS_FROZEN_CONTRACT")
    theta = value.get("fixture", {}).get("estimand_theta", {})
    if set(theta) != set(FAMILY) or any(
        not math.isclose(theta[name], FIXTURE_THETA[name], abs_tol=1e-14)
        for name in FAMILY
    ):
        raise InvalidP3("ANALYSIS_FIXTURE_THETA")
    implementation = value.get("implementation", {})
    for path_key, hash_key in (
        ("path", "sha256"),
        ("mutation_test_path", "mutation_test_sha256"),
    ):
        target = root / implementation[path_key]
        if not target.is_file() or sha256_file(target) != implementation[hash_key]:
            raise InvalidP3(f"ANALYSIS_BINDING:{implementation[path_key]}")
    if (
        value.get("formal_input_read") is not False
        or value.get("formal_execution_authorized") is not False
        or value.get("scientific_outcome_interpreted") is not False
    ):
        raise InvalidP3("ANALYSIS_AUTHORITY")
    return value


def _validate_telemetry(directory: Path, workers: int) -> dict[str, Any]:
    value = load(directory / "telemetry.json")
    if value.get("worker_count") != workers:
        raise InvalidP3("TELEMETRY_WORKERS")
    if value.get("outcome_fields_present") is not False:
        raise InvalidP3("TELEMETRY_OUTCOME_FIREWALL")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise InvalidP3("TELEMETRY_SAMPLES")
    previous = None
    for index, sample in enumerate(samples):
        if sample.get("sample_index") != index:
            raise InvalidP3("TELEMETRY_ORDER")
        elapsed = sample.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
            raise InvalidP3("TELEMETRY_ELAPSED")
        if previous is not None and not 0.0 <= elapsed - previous <= 30.0:
            raise InvalidP3("HEARTBEAT_INTERVAL")
        previous = elapsed
        for key in (
            "available_ram_bytes",
            "resident_memory_bytes",
            "read_bytes",
            "write_bytes",
            "swap_in_total",
            "swap_out_total",
        ):
            if not isinstance(sample.get(key), int) or sample[key] < 0:
                raise InvalidP3(f"TELEMETRY_FIELD:{key}")
    return value


def validate_profile(
    directory: Path,
    workers: int,
    identity_lock_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    value = load(directory / "success.json")
    if value.get("workers") != workers or value.get("profile") != f"W{workers}":
        raise InvalidP3("PROFILE_WORKERS")
    native_threads = SCHEDULER_NATIVE_THREADS[workers]
    if (
        value.get("native_threads_per_worker") != native_threads
        or value.get("total_native_thread_budget") != workers * native_threads
    ):
        raise InvalidP3("PROFILE_SCHEDULER")
    if value.get("terminal") != "PROFILE_COMPLETE / PREFLIGHT_ONLY":
        raise InvalidP3("PROFILE_TERMINAL")
    if value.get("identity_lock_sha256") != sha256_file(identity_lock_path):
        raise InvalidP3("PROFILE_IDENTITY_LOCK")
    if value.get("identity_set_sha256") != identity["identity_set_sha256"]:
        raise InvalidP3("PROFILE_IDENTITY_SET")
    if (
        value.get("sequence_count") != 8
        or value.get("frame_count") != 8 * FRAME_COUNT
        or value.get("pair_count") != 8 * PAIR_COUNT
    ):
        raise InvalidP3("PROFILE_COUNTS")
    receipts = value.get("sequence_receipts")
    if not isinstance(receipts, list) or len(receipts) != 8:
        raise InvalidP3("PROFILE_RECEIPTS")
    expected_ids = [item["sequence_id"] for item in identity["identities"]]
    if [item.get("sequence_id") for item in receipts] != expected_ids:
        raise InvalidP3("PROFILE_SEQUENCE_ORDER")
    for expected, receipt in zip(identity["identities"], receipts):
        for key in ("sequence_id", "cluster_kind", "arm", "numeric_seed_uint64"):
            if receipt.get(key) != expected[key]:
                raise InvalidP3(f"PROFILE_SEQUENCE_IDENTITY:{key}")
        if (
            receipt.get("frame_count") != FRAME_COUNT
            or receipt.get("pair_count") != PAIR_COUNT
        ):
            raise InvalidP3("PROFILE_SEQUENCE_COUNTS")
        render_execution = receipt.get("render_execution", {})
        if expected["arm"].startswith("STATIC_CAMERA"):
            expected_render_execution = {
                "strategy": "STATIC_IDENTICAL_POSE_SINGLE_RENDER",
                "render_invocations": 1,
                "render_reuse_hits": FRAME_COUNT - 1,
            }
        else:
            expected_render_execution = {
                "strategy": "PER_FRAME_RENDER",
                "render_invocations": FRAME_COUNT,
                "render_reuse_hits": 0,
            }
        if render_execution != expected_render_execution:
            raise InvalidP3("PROFILE_RENDER_EXECUTION")
        thread_guard = receipt.get("thread_guard", {})
        if workers == 8:
            valid_thread_guard = (
                thread_guard.get("opencv_num_threads") == 1
                and thread_guard.get("openblas_observed_num_threads") == 18
                and thread_guard.get("applied_before_numeric_imports") is False
            )
        else:
            valid_thread_guard = thread_guard == {
                "omp_num_threads": "4",
                "openblas_num_threads": "4",
                "mkl_num_threads": "4",
                "numexpr_num_threads": "4",
                "veclib_maximum_threads": "4",
                "blis_num_threads": "4",
                "opencv_num_threads": 4,
                "openblas_observed_num_threads": 4,
                "applied_before_numeric_imports": True,
            }
        if not valid_thread_guard:
            raise InvalidP3("PROFILE_NATIVE_THREAD_GUARD")
        firewall = receipt.get("outcome_firewall", {})
        if any(
            firewall.get(key) is not False
            for key in (
                "response_values_emitted",
                "trigger_values_emitted",
                "scientific_interpretation",
            )
        ):
            raise InvalidP3("PROFILE_OUTCOME_FIREWALL")
    resource = value.get("resource", {})
    if resource.get("available_ram_at_launch_bytes", 0) < 8 * GIB:
        raise InvalidP3("PROFILE_LAUNCH_RAM")
    if resource.get("minimum_available_ram_bytes", 0) < 4 * GIB:
        raise InvalidP3("PROFILE_RUNTIME_RAM")
    if resource.get("sustained_paging") is not False:
        raise InvalidP3("PROFILE_PAGING")
    if resource.get("heartbeat_max_interval_seconds", 31.0) > 30.0:
        raise InvalidP3("PROFILE_HEARTBEAT")
    if value.get("residual_worker_pids") != []:
        raise InvalidP3("PROFILE_RESIDUAL_WORKER")
    if (
        value.get("formal_execution_authorized") is not False
        or value.get("p4_activated") is not False
        or value.get("scientific_outcome_interpreted") is not False
    ):
        raise InvalidP3("PROFILE_AUTHORITY")
    _validate_telemetry(directory, workers)
    progress = load(directory / "progress.json")
    if (
        progress.get("status") != "SUCCESS"
        or progress.get("terminal_state") != "SUCCESS"
        or progress.get("completed_units") != 8
        or progress.get("completed_pairs") != 8 * PAIR_COUNT
    ):
        raise InvalidP3("PROFILE_PROGRESS_TERMINAL")
    return value


def profile_projection(profile: dict[str, Any]) -> dict[str, Any]:
    workers = profile.get("workers")
    receipts = profile.get("sequence_receipts")
    if workers not in (4, 8) or not isinstance(receipts, list) or len(receipts) != 8:
        raise InvalidP3("PROFILE_PROJECTION_INPUT")
    components = {
        "render_seconds": "render_seconds",
        "r3_seconds": "r3_seconds",
        "validation_and_receipt_seconds": "validation_and_hash_seconds",
    }
    projected: dict[str, float] = {}
    for output_key, timing_key in components.items():
        total_work = 0.0
        for receipt in receipts:
            kind = receipt.get("cluster_kind")
            if kind == "FACTORIAL":
                multiplicity = FORMAL_FACTORIAL_CLUSTERS
            elif kind == "GUARDRAIL":
                multiplicity = FORMAL_GUARDRAIL_CLUSTERS
            else:
                raise InvalidP3("PROFILE_PROJECTION_CLUSTER_KIND")
            value = receipt.get("timing", {}).get(timing_key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise InvalidP3(f"PROFILE_PROJECTION_TIMING:{timing_key}")
            total_work += float(value) * multiplicity
        projected[output_key] = total_work / workers
    projected_core = sum(projected.values())
    if projected_core <= 0.0:
        raise InvalidP3("PROFILE_TIMING")
    retry = projected_core * RETRY_RESERVE_FRACTION
    total = projected_core + retry
    return {
        **projected,
        "retry_reserve_seconds": retry,
        "retry_reserve_fraction": RETRY_RESERVE_FRACTION,
        "total_seconds": total,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "formal_factorial_sequences": (
            FORMAL_FACTORIAL_CLUSTERS * len(FACTORIAL_ARMS)
        ),
        "formal_guardrail_sequences": (
            FORMAL_GUARDRAIL_CLUSTERS * len(GUARDRAIL_ARMS)
        ),
        "projection_method": (
            "MEASURED_PER_ARM_COMPONENT_WORK_BY_FORMAL_MULTIPLICITY"
            "_DIVIDED_BY_WORKERS"
        ),
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def validate_all(
    root: Path,
    identity_lock_path: Path,
    transport_lock_path: Path,
    analysis_lock_path: Path,
    w4_directory: Path,
    w8_directory: Path,
) -> dict[str, Any]:
    identity = validate_identity_lock(identity_lock_path)
    validate_transport_lock(root, transport_lock_path)
    validate_analysis_lock(root, analysis_lock_path)
    w4 = validate_profile(w4_directory, 4, identity_lock_path, identity)
    w8 = validate_profile(w8_directory, 8, identity_lock_path, identity)
    w4_receipts = {
        item["sequence_id"]: item for item in w4["sequence_receipts"]
    }
    w8_receipts = {
        item["sequence_id"]: item for item in w8["sequence_receipts"]
    }
    for sequence_id in w4_receipts:
        for field in (
            "scene_geometry_sha256",
            "frame_manifest_sha256",
            "ordered_pair_numeric_sha256",
            "transport_identity_sha256",
        ):
            if w4_receipts[sequence_id][field] != w8_receipts[sequence_id][field]:
                raise InvalidP3(f"PROFILE_EQUIVALENCE:{sequence_id}:{field}")
    projections = {
        "W4": profile_projection(w4),
        "W8": profile_projection(w8),
    }
    profiles = (w4, w8)
    fastest = min(
        profiles, key=lambda item: float(item["timing"]["wall_seconds"])
    )
    qualified_profiles = [
        item
        for item in profiles
        if projections[item["profile"]]["total_seconds"]
        <= WALL_CEILING_SECONDS
    ]
    selected = (
        min(
            qualified_profiles,
            key=lambda item: float(item["timing"]["wall_seconds"]),
        )
        if qualified_profiles
        else None
    )
    formal_paths = [
        root
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/p4_formal",
        root
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/formal_480_plus_16",
    ]
    if any(path.exists() for path in formal_paths):
        raise InvalidP3("FORMAL_PATH_PRESENT")
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "p3_id": P3_ID,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": (
            "PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED"
            if selected is not None
            else "PERFORMANCE_NOT_QUALIFIED / VALID / P4_NOT_ACTIVATED"
        ),
        "validated": True,
        "transport_equivalence": "VALID",
        "analysis_implementation_and_mutations": "VALID",
        "identity_count": 8,
        "profiles_completed": ["W4", "W8"],
        "profile_numeric_equivalence": "PASS",
        "fastest_measured_profile": fastest["profile"],
        "selected_profile": (
            selected["profile"] if selected is not None else None
        ),
        "selection_rule": "faster_measured_profile_after_all_guards",
        "scheduler_amendment": {
            "authority": "USER_AUTHORIZED_2026-07-29",
            "W4_native_threads_per_worker": 4,
            "W8_openblas_threads_per_worker": 18,
            "W8_opencv_threads_per_worker": 1,
            "W4_total_native_thread_budget": 16,
            "W8_openblas_thread_budget": 144,
            "r3_modified": False,
        },
        "performance_failure": (
            None
            if selected is not None
            else "BOTH_PROFILES_EXCEED_12_HOUR_496_SEQUENCE_PROJECTION"
        ),
        "projections": projections,
        "bindings": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in (
                identity_lock_path,
                transport_lock_path,
                analysis_lock_path,
                root
                / "scripts/research/egomotion_compensated_looming/"
                "periodic_self_motion_counterfactual_r2/"
                "p3_runtime_preflight_r0.py",
                root
                / "scripts/research/egomotion_compensated_looming/"
                "tests_periodic_self_motion_counterfactual_r2/"
                "test_p3_runtime_preflight.py",
                root
                / "scripts/research/egomotion_compensated_looming/"
                "tests_periodic_self_motion_counterfactual_r2/"
                "test_p3_transport_equivalence.py",
                w4_directory / "success.json",
                w4_directory / "telemetry.json",
                w8_directory / "success.json",
                w8_directory / "telemetry.json",
                Path(__file__).resolve(),
            )
        ],
        "formal_paths_absent": [
            path.relative_to(root).as_posix() for path in formal_paths
        ],
        "formal_seed_access": False,
        "formal_480_plus_16_run": False,
        "scientific_outcome_interpreted": False,
        "strength_retuned": False,
        "r3_threshold_or_three_pair_modified": False,
        "sequence16_android_realtime": False,
        "formal_execution_authorized": False,
        "p4_activated": False,
    }
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--transport-lock", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--w4-directory", type=Path, required=True)
    parser.add_argument("--w8-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    receipt = validate_all(
        root,
        (root / args.identity_lock).resolve(),
        (root / args.transport_lock).resolve(),
        (root / args.analysis_lock).resolve(),
        (root / args.w4_directory).resolve(),
        (root / args.w8_directory).resolve(),
    )
    write_exclusive((root / args.receipt).resolve(), receipt)
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "selected_profile": receipt["selected_profile"],
                "validated": receipt["validated"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
