"""Build and validate the R2 P4 scheduler, identity, and activation locks.

This module does not run the renderer, R3, analysis, or the formal bundle.
Issuing an activation is an explicit, exclusive-create operation kept separate
from the pure builders so review and tests cannot accidentally authorize P4.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import psutil


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
P4_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "FORMAL_IDENTITY_AND_ACTIVATION_R0"
)
SCHEDULER_AMENDMENT_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "P4_W8_OPENBLAS18_SCHEDULER_AMENDMENT_R0"
)
DEFAULT_USER_AUTHORIZATION_ID = "USER_DIRECT_P4_W8_OPENBLAS18_2026-07-29"
FRAME_COUNT = 602
PAIR_COUNT = 601
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MAIN_ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
P1_GUARD_TO_CANONICAL = {
    "MONOTONIC_APPROACH": "MONOTONIC_APPROACH_ONLY__CLEAN",
    "MONOTONIC_APPROACH_PLUS_PERIODIC": (
        "MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF__CLEAN"
    ),
}
FORMAL_RELATIVE_PATHS = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p4_formal",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "formal_480_plus_16",
)
P1_MANIFEST_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/all_seed_geometry_manifest.jsonl"
)
P1_RECEIPT_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/"
    "independent_geometry_validation_receipt.json"
)
P3_RECEIPT_RELATIVE = (
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0_"
    "INDEPENDENT_RECEIPT_R2_2026-07-29.json"
)
MODULE_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "p4_identity_activation_r0.py"
)
TEST_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "tests_periodic_self_motion_counterfactual_r2/"
    "test_p4_identity_activation_r0.py"
)
SCHEMA_RELATIVE = (
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "P4_IDENTITY_ACTIVATION_R0_SCHEMA_2026-07-29.json"
)
P4_IMPLEMENTATION_BINDINGS = (
    ("P4_IDENTITY_ACTIVATION_IMPLEMENTATION", MODULE_RELATIVE),
    ("P4_IDENTITY_ACTIVATION_TESTS", TEST_RELATIVE),
    ("P4_IDENTITY_ACTIVATION_SCHEMA", SCHEMA_RELATIVE),
    (
        "P4_MANIPULATION_IMPLEMENTATION",
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/p4_manipulation_r0.py",
    ),
    (
        "P4_MANIPULATION_INDEPENDENT_VALIDATOR",
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "validate_p4_manipulation_independent_r0.py",
    ),
    (
        "P4_MANIPULATION_TESTS",
        "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p4_manipulation.py",
    ),
    (
        "P4_FORMAL_RUNNER",
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/p4_formal_runner_r0.py",
    ),
    (
        "P4_FORMAL_RUNNER_TESTS",
        "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p4_formal_runner_r0.py",
    ),
    (
        "P4_FORMAL_ANALYSIS",
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/p4_formal_analysis_r0.py",
    ),
    (
        "P4_FORMAL_INDEPENDENT_VALIDATOR",
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "p4_formal_independent_validator_r0.py",
    ),
    (
        "P4_FORMAL_ANALYSIS_TESTS",
        "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p4_formal_analysis_r0.py",
    ),
    (
        "P4_FORMAL_INDEPENDENT_VALIDATOR_TESTS",
        "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p4_formal_independent_validator_r0.py",
    ),
)
PHASE_BINDINGS = (
    (
        "P0_CONTRACT",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json",
    ),
    (
        "P0_GEOMETRY_SPEC",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GEOMETRY_VALIDATION_R0_2026-07-28.json",
    ),
    (
        "P0_RUN_BUDGET",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "RUN_BUDGET_R0_2026-07-28.json",
    ),
    (
        "P1_IMPLEMENTATION_LOCK",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_"
        "KEYSET_REPAIR_R0_2026-07-29.json",
    ),
    ("P1_ALL_SEED_MANIFEST", P1_MANIFEST_RELATIVE),
    ("P1_INDEPENDENT_RECEIPT", P1_RECEIPT_RELATIVE),
    (
        "P2_STRENGTH_LOCK",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_"
        "GLOBAL_STRENGTH_LOCK_2026-07-29.json",
    ),
    (
        "P2_INDEPENDENT_RECEIPT",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_"
        "INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json",
    ),
    (
        "P3_TRANSPORT_LOCK",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json",
    ),
    (
        "P3_ANALYSIS_LOCK",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json",
    ),
    (
        "P3_PREFLIGHT_IDENTITY_LOCK",
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    ("P3_SCHEDULER_SUCCESSOR_RECEIPT", P3_RECEIPT_RELATIVE),
)
AUTHORIZATION_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{7,127}$")
WORKER_COMMAND_MARKERS = (
    "p3_runtime_preflight_r0.py",
    "p4_formal_runner_r0.py",
)


class InvalidP4IdentityActivation(ValueError):
    """Raised when any frozen identity or activation dependency drifts."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidP4IdentityActivation(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise InvalidP4IdentityActivation("P1_MANIFEST_OBJECT_ROWS")
    return rows


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(os.fspath(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def derive_seed(namespace: str, block: str, ordinal: int) -> int:
    if namespace not in {"MAIN", "GUARD"}:
        raise InvalidP4IdentityActivation("SEED_NAMESPACE")
    if block not in BLOCKS:
        raise InvalidP4IdentityActivation("SEED_BLOCK")
    maximum = 19 if namespace == "MAIN" else 1
    if not 0 <= ordinal <= maximum:
        raise InvalidP4IdentityActivation("SEED_ORDINAL")
    token = f"{PROTOCOL_ID}|{namespace}|{block}|{ordinal:02d}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")


def _binding(root: Path, label: str, relative: str) -> dict[str, str]:
    path = (root / relative).resolve()
    if not path.is_file():
        raise InvalidP4IdentityActivation(f"BINDING_MISSING:{label}")
    try:
        resolved_relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise InvalidP4IdentityActivation(f"BINDING_OUTSIDE_REPO:{label}") from error
    return {
        "phase": label,
        "path": resolved_relative,
        "sha256": sha256_file(path),
    }


def validate_phase_bindings(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    bindings = [_binding(root, label, relative) for label, relative in PHASE_BINDINGS]
    by_label = {item["phase"]: item for item in bindings}

    contract = load_json(root / by_label["P0_CONTRACT"]["path"])
    unchanged = contract.get("unchanged_algorithm_lock", {})
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("formal_execution_authorized") is not False
        or unchanged.get("implementation")
        != "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
        or unchanged.get("threshold_operator") != "strict_greater_than"
        or unchanged.get("threshold_per_s") != 0.01
        or unchanged.get("required_consecutive_evaluable_pairs") != 3
    ):
        raise InvalidP4IdentityActivation("P0_SCIENTIFIC_LOCK")

    run_budget = load_json(root / by_label["P0_RUN_BUDGET"]["path"])
    counts = run_budget.get("count_budget", {})
    if (
        counts.get("main_sequences") != 480
        or counts.get("guardrail_sequences") != 16
        or counts.get("formal_total_sequences") != 496
        or run_budget.get("formal_execution_authorized") is not False
    ):
        raise InvalidP4IdentityActivation("P0_RUN_BUDGET")

    p1_lock = load_json(root / by_label["P1_IMPLEMENTATION_LOCK"]["path"])
    if (
        p1_lock.get("protocol_id") != PROTOCOL_ID
        or p1_lock.get("formal_execution_authorized") is not False
        or p1_lock.get("identity_lock", {}).get("all_seed_manifest_sha256")
        != by_label["P1_ALL_SEED_MANIFEST"]["sha256"]
    ):
        raise InvalidP4IdentityActivation("P1_LOCK")
    p1_receipt = load_json(root / by_label["P1_INDEPENDENT_RECEIPT"]["path"])
    if (
        p1_receipt.get("terminal") != "GENERATOR_GEOMETRY_PASS"
        or p1_receipt.get("formal_execution_authorized") is not False
        or p1_receipt.get("failed_gates") != []
        or p1_receipt.get("errors") != []
    ):
        raise InvalidP4IdentityActivation("P1_RECEIPT")

    p2_receipt = load_json(root / by_label["P2_INDEPENDENT_RECEIPT"]["path"])
    if (
        p2_receipt.get("validated") is not True
        or p2_receipt.get("protocol_status") != "VALID"
        or p2_receipt.get("scientific_status") != "QUALITY_CALIBRATION_PASS"
        or p2_receipt.get("selected_global_strengths")
        != {"blur_sigma_px": 0.475, "low_texture_alpha": 0.15}
        or p2_receipt.get("formal_execution_authorized") is not False
    ):
        raise InvalidP4IdentityActivation("P2_RECEIPT")

    p3_receipt = load_json(root / by_label["P3_SCHEDULER_SUCCESSOR_RECEIPT"]["path"])
    successor = p3_receipt.get("profiles", {}).get("W8_SCHEDULER_SUCCESSOR", {})
    if (
        p3_receipt.get("validated") is not True
        or p3_receipt.get("terminal")
        != "PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED"
        or p3_receipt.get("selected_profile") != "W8"
        or successor.get("workers") != 8
        or successor.get("openblas_threads_per_worker") != 18
        or successor.get("opencv_threads_per_worker") != 1
        or p3_receipt.get("formal_execution_authorized") is not False
        or p3_receipt.get("p4_activated") is not False
    ):
        raise InvalidP4IdentityActivation("P3_SCHEDULER_RECEIPT")
    return bindings


def build_scheduler_amendment(
    root: Path,
    *,
    user_authorization_id: str = DEFAULT_USER_AUTHORIZATION_ID,
) -> dict[str, Any]:
    if not AUTHORIZATION_PATTERN.fullmatch(user_authorization_id):
        raise InvalidP4IdentityActivation("USER_AUTHORIZATION_ID")
    bindings = validate_phase_bindings(root)
    p3 = load_json(root / P3_RECEIPT_RELATIVE)
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_scheduler_amendment.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "amendment_id": SCHEDULER_AMENDMENT_ID,
        "scope": "SCHEDULER_ONLY_NO_SCIENTIFIC_LOCK_CHANGE",
        "user_authorization": {
            "authorization_id": user_authorization_id,
            "authorized_profile": "W8",
            "authorized_openblas_threads_per_worker": 18,
            "authority_scope": (
                "P4_FORMAL_SCHEDULER_ONLY_AFTER_IDENTITY_AND_ACTIVATION_LOCKS"
            ),
        },
        "scheduler": {
            "backend": "cpu_process_pool",
            "workers": 8,
            "opencv_threads_per_worker": 1,
            "openblas_threads_per_worker": 18,
            "maximum_openblas_thread_budget": 144,
            "selected_profile": "W8",
            "measured_preflight_wall_seconds": p3["measured_w8_wall_seconds"],
            "projected_formal_total_seconds": p3["projection"]["total_seconds"],
            "wall_ceiling_seconds": p3["projection"]["wall_ceiling_seconds"],
        },
        "scientific_lock": {
            "r3_implementation": "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3",
            "response_field": "compensated_expansion_median_per_s",
            "threshold_operator": "strict_greater_than",
            "threshold_per_s": 0.01,
            "required_consecutive_evaluable_pairs": 3,
            "algorithm_or_threshold_changed": False,
            "analysis_or_manipulation_changed": False,
            "seed_or_identity_changed": False,
        },
        "phase_bindings_sha256": hashlib.sha256(
            canonical_bytes(bindings)
        ).hexdigest(),
        "formal_execution_authorized": False,
        "p4_activated": False,
        "terminal": "SCHEDULER_AMENDMENT_VALID / ACTIVATION_REQUIRED",
    }


def validate_scheduler_amendment(
    root: Path,
    value: dict[str, Any],
) -> dict[str, Any]:
    authorization = value.get("user_authorization", {})
    expected = build_scheduler_amendment(
        root,
        user_authorization_id=authorization.get("authorization_id", ""),
    )
    if canonical_bytes(value) != canonical_bytes(expected):
        raise InvalidP4IdentityActivation("SCHEDULER_AMENDMENT_DRIFT")
    return value


def _formal_identity(
    record: dict[str, Any],
    arm: dict[str, Any],
    arm_ordinal: int,
) -> dict[str, Any]:
    record_type = record.get("record_type")
    if record_type == "main_cluster":
        role = "MAIN_FACTORIAL"
        namespace = "MAIN"
        source_arm = arm.get("arm_id")
        canonical_arm = source_arm
    elif record_type == "guardrail_cluster":
        role = "POSITIVE_GUARDRAIL"
        namespace = "GUARD"
        source_arm = arm.get("arm_id")
        canonical_arm = P1_GUARD_TO_CANONICAL.get(source_arm)
    else:
        raise InvalidP4IdentityActivation("P1_RECORD_TYPE")
    if not isinstance(canonical_arm, str):
        raise InvalidP4IdentityActivation("GUARD_ARM_MAPPING")
    block = record.get("block")
    ordinal = record.get("ordinal")
    if (
        block not in BLOCKS
        or not isinstance(ordinal, int)
        or record.get("numeric_seed_uint64")
        != derive_seed(namespace, block, ordinal)
    ):
        raise InvalidP4IdentityActivation("FORMAL_SEED_IDENTITY")
    cluster_id = f"{block}__{namespace}_{ordinal:02d}"
    if record.get("cluster_id") != cluster_id or arm.get("cluster_id") != cluster_id:
        raise InvalidP4IdentityActivation("FORMAL_CLUSTER_IDENTITY")
    if namespace == "MAIN" and (
        source_arm not in MAIN_ARMS or arm_ordinal != MAIN_ARMS.index(source_arm)
    ):
        raise InvalidP4IdentityActivation("MAIN_ARM_IDENTITY")
    if namespace == "GUARD" and (
        source_arm not in tuple(P1_GUARD_TO_CANONICAL)
        or arm_ordinal != tuple(P1_GUARD_TO_CANONICAL).index(source_arm)
        or arm.get("quality") != "CLEAN"
    ):
        raise InvalidP4IdentityActivation("GUARD_ARM_IDENTITY")
    for field in ("scene_geometry_sha256", "trajectory_sha256"):
        if not isinstance(arm.get(field), str) or len(arm[field]) != 64:
            raise InvalidP4IdentityActivation(f"FORMAL_ARM_HASH:{field}")
    return {
        "sequence_id": (
            f"FORMAL_{block}_{namespace}_{ordinal:02d}__{canonical_arm}"
        ),
        "cluster_id": cluster_id,
        "block": block,
        "ordinal": ordinal,
        "role": role,
        "source_arm_id": source_arm,
        "arm": canonical_arm,
        "arm_ordinal": arm_ordinal,
        "numeric_seed_uint64": record["numeric_seed_uint64"],
        "scene_geometry_sha256": arm["scene_geometry_sha256"],
        "trajectory_sha256": arm["trajectory_sha256"],
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
    }


def build_formal_identity_lock(
    root: Path,
    scheduler_amendment_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    scheduler = load_json(scheduler_amendment_path)
    validate_scheduler_amendment(root, scheduler)
    manifest_path = root / P1_MANIFEST_RELATIVE
    rows = load_jsonl(manifest_path)
    identities: list[dict[str, Any]] = []
    expected_records = [
        (record_type, block, ordinal)
        for record_type, count in (("main_cluster", 20), ("guardrail_cluster", 2))
        for block in BLOCKS
        for ordinal in range(count)
    ]
    observed_records = [
        (row.get("record_type"), row.get("block"), row.get("ordinal"))
        for row in rows
    ]
    if observed_records != expected_records:
        raise InvalidP4IdentityActivation("P1_MANIFEST_RECORD_ORDER")
    for record in rows:
        arms = record.get("arms")
        expected_arm_count = 6 if record["record_type"] == "main_cluster" else 2
        if not isinstance(arms, list) or len(arms) != expected_arm_count:
            raise InvalidP4IdentityActivation("P1_MANIFEST_ARMS")
        identities.extend(
            _formal_identity(record, arm, arm_ordinal)
            for arm_ordinal, arm in enumerate(arms)
        )
    if (
        len(identities) != 496
        or len({item["sequence_id"] for item in identities}) != 496
    ):
        raise InvalidP4IdentityActivation("FORMAL_IDENTITY_COUNT")
    payload = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_formal_identity_lock.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "p4_id": P4_ID,
        "phase": "P4_FORMAL_480_PLUS_16",
        "source_manifest": {
            "path": P1_MANIFEST_RELATIVE,
            "sha256": sha256_file(manifest_path),
        },
        "scheduler_amendment": {
            "path": scheduler_amendment_path.relative_to(root).as_posix(),
            "sha256": sha256_file(scheduler_amendment_path),
        },
        "guard_arm_name_map": dict(P1_GUARD_TO_CANONICAL),
        "counts": {
            "main_clusters": 80,
            "guardrail_clusters": 8,
            "main_sequences": 480,
            "guardrail_sequences": 16,
            "total_sequences": 496,
            "total_frames": 496 * FRAME_COUNT,
            "total_pairs": 496 * PAIR_COUNT,
        },
        "canonicalization": (
            "UTF8_JSON_SORT_KEYS_COMPACT_ENSURE_ASCII_FALSE_ALLOW_NAN_FALSE"
        ),
        "identities": identities,
        "identity_set_sha256": hashlib.sha256(
            canonical_bytes(identities)
        ).hexdigest(),
        "formal_execution_authorized": False,
        "terminal": "FORMAL_IDENTITY_LOCK_VALID / ACTIVATION_REQUIRED",
    }
    return payload


def validate_formal_identity_lock(
    root: Path,
    scheduler_amendment_path: Path,
    value: dict[str, Any],
) -> dict[str, Any]:
    expected = build_formal_identity_lock(root, scheduler_amendment_path)
    if canonical_bytes(value) != canonical_bytes(expected):
        raise InvalidP4IdentityActivation("FORMAL_IDENTITY_LOCK_DRIFT")
    return value


def live_research_worker_pids() -> list[int]:
    matches: list[int] = []
    own_pid = os.getpid()
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            pid = int(process.info["pid"])
            if pid == own_pid:
                continue
            process_name = str(process.info.get("name") or "").lower()
            if "python" not in process_name:
                continue
            command = " ".join(process.info.get("cmdline") or []).lower()
            if any(marker.lower() in command for marker in WORKER_COMMAND_MARKERS):
                matches.append(pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError, ValueError):
            continue
    return sorted(set(matches))


def ensure_activation_preconditions(root: Path) -> dict[str, Any]:
    present = [
        relative for relative in FORMAL_RELATIVE_PATHS if (root / relative).exists()
    ]
    if present:
        raise InvalidP4IdentityActivation(
            "FORMAL_PATH_PRESENT:" + ",".join(present)
        )
    worker_pids = live_research_worker_pids()
    if worker_pids:
        raise InvalidP4IdentityActivation(
            "RESIDUAL_WORKERS:" + ",".join(map(str, worker_pids))
        )
    return {
        "formal_paths_absent": list(FORMAL_RELATIVE_PATHS),
        "live_worker_scan_rule": list(WORKER_COMMAND_MARKERS),
        "live_worker_pids": [],
    }


def _verify_bindings(root: Path, bindings: Iterable[dict[str, str]]) -> None:
    for binding in bindings:
        path = (root / binding.get("path", "")).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise InvalidP4IdentityActivation("ACTIVATION_BINDING_OUTSIDE") from error
        if (
            not path.is_file()
            or sha256_file(path) != binding.get("sha256")
        ):
            raise InvalidP4IdentityActivation(
                f"ACTIVATION_BINDING_DRIFT:{binding.get('path')}"
            )


def build_activation_lock(
    root: Path,
    scheduler_amendment_path: Path,
    formal_identity_lock_path: Path,
    *,
    user_authorization_id: str = DEFAULT_USER_AUTHORIZATION_ID,
    issued_at_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    scheduler = load_json(scheduler_amendment_path)
    validate_scheduler_amendment(root, scheduler)
    identity = load_json(formal_identity_lock_path)
    validate_formal_identity_lock(root, scheduler_amendment_path, identity)
    if (
        scheduler["user_authorization"]["authorization_id"]
        != user_authorization_id
    ):
        raise InvalidP4IdentityActivation("ACTIVATION_AUTHORIZATION_MISMATCH")
    preconditions = ensure_activation_preconditions(root)
    phase_bindings = validate_phase_bindings(root)
    implementation_bindings = [
        *[
            _binding(root, label, relative)
            for label, relative in P4_IMPLEMENTATION_BINDINGS
        ],
        {
            "phase": "P4_SCHEDULER_AMENDMENT",
            "path": scheduler_amendment_path.relative_to(root).as_posix(),
            "sha256": sha256_file(scheduler_amendment_path),
        },
        {
            "phase": "P4_FORMAL_IDENTITY_LOCK",
            "path": formal_identity_lock_path.relative_to(root).as_posix(),
            "sha256": sha256_file(formal_identity_lock_path),
        },
    ]
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_activation_lock.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "p4_id": P4_ID,
        "activation_id": (
            "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_P4_ACTIVATION_R0"
        ),
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "user_authorization_id": user_authorization_id,
        "execution": {
            "one_shot": True,
            "execution_class": "formal",
            "formal_sequence_count": 496,
            "formal_frame_count": 496 * FRAME_COUNT,
            "formal_pair_count": 496 * PAIR_COUNT,
            "scheduler_profile": "W8",
            "workers": 8,
            "opencv_threads_per_worker": 1,
            "openblas_threads_per_worker": 18,
            "identity_set_sha256": identity["identity_set_sha256"],
        },
        "scientific_lock": scheduler["scientific_lock"],
        "preconditions": preconditions,
        "bindings": [*phase_bindings, *implementation_bindings],
        "authority_ceiling": {
            "sequence16": False,
            "android": False,
            "realtime_integration": False,
            "p4_result_interpretation_before_bundle_validation": False,
            "scientific_lock_change_authorized": False,
        },
        "formal_execution_authorized": True,
        "p4_activated": True,
        "terminal": "P4_FORMAL_EXECUTION_AUTHORIZED / ONE_SHOT",
    }


def validate_activation_lock(
    root: Path,
    activation: dict[str, Any],
) -> dict[str, Any]:
    if (
        activation.get("schema")
        != "rcle.periodic_self_motion_counterfactual.p4_activation_lock.v1"
        or activation.get("protocol_id") != PROTOCOL_ID
        or activation.get("p4_id") != P4_ID
        or activation.get("formal_execution_authorized") is not True
        or activation.get("p4_activated") is not True
        or activation.get("terminal")
        != "P4_FORMAL_EXECUTION_AUTHORIZED / ONE_SHOT"
    ):
        raise InvalidP4IdentityActivation("ACTIVATION_HEADER")
    bindings = activation.get("bindings")
    if not isinstance(bindings, list):
        raise InvalidP4IdentityActivation("ACTIVATION_BINDINGS")
    _verify_bindings(root, bindings)
    by_phase = {item.get("phase"): item for item in bindings}
    required = {label for label, _ in PHASE_BINDINGS} | {
        "P4_SCHEDULER_AMENDMENT",
        "P4_FORMAL_IDENTITY_LOCK",
    } | {label for label, _ in P4_IMPLEMENTATION_BINDINGS}
    if set(by_phase) != required:
        raise InvalidP4IdentityActivation("ACTIVATION_BINDING_SET")
    scheduler_path = root / by_phase["P4_SCHEDULER_AMENDMENT"]["path"]
    identity_path = root / by_phase["P4_FORMAL_IDENTITY_LOCK"]["path"]
    scheduler = load_json(scheduler_path)
    validate_scheduler_amendment(root, scheduler)
    identity = load_json(identity_path)
    validate_formal_identity_lock(root, scheduler_path, identity)
    expected = build_activation_lock(
        root,
        scheduler_path,
        identity_path,
        user_authorization_id=activation.get("user_authorization_id", ""),
        issued_at_utc=activation.get("issued_at_utc"),
    )
    if canonical_bytes(activation) != canonical_bytes(expected):
        raise InvalidP4IdentityActivation("ACTIVATION_DRIFT")
    return activation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    scheduler = subparsers.add_parser("write-scheduler-amendment")
    scheduler.add_argument("--repo-root", type=Path, required=True)
    scheduler.add_argument("--output", type=Path, required=True)
    scheduler.add_argument(
        "--user-authorization-id",
        default=DEFAULT_USER_AUTHORIZATION_ID,
    )

    identity = subparsers.add_parser("write-formal-identity-lock")
    identity.add_argument("--repo-root", type=Path, required=True)
    identity.add_argument("--scheduler-amendment", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)

    activation = subparsers.add_parser("issue-activation")
    activation.add_argument("--repo-root", type=Path, required=True)
    activation.add_argument("--scheduler-amendment", type=Path, required=True)
    activation.add_argument("--formal-identity-lock", type=Path, required=True)
    activation.add_argument("--output", type=Path, required=True)
    activation.add_argument(
        "--user-authorization-id",
        default=DEFAULT_USER_AUTHORIZATION_ID,
    )

    validate = subparsers.add_parser("validate-activation")
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--activation", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    if args.command == "write-scheduler-amendment":
        value = build_scheduler_amendment(
            root,
            user_authorization_id=args.user_authorization_id,
        )
        write_exclusive((root / args.output).resolve(), value)
    elif args.command == "write-formal-identity-lock":
        value = build_formal_identity_lock(
            root,
            (root / args.scheduler_amendment).resolve(),
        )
        write_exclusive((root / args.output).resolve(), value)
    elif args.command == "issue-activation":
        value = build_activation_lock(
            root,
            (root / args.scheduler_amendment).resolve(),
            (root / args.formal_identity_lock).resolve(),
            user_authorization_id=args.user_authorization_id,
        )
        write_exclusive((root / args.output).resolve(), value)
    else:
        value = validate_activation_lock(
            root,
            load_json((root / args.activation).resolve()),
        )
    print(
        json.dumps(
            {
                "schema": value["schema"],
                "terminal": value["terminal"],
                "formal_execution_authorized": value[
                    "formal_execution_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
