"""Recoverable W8 runner for the frozen RCLE periodic R2 formal identities.

This module intentionally does not import NumPy, OpenCV, the R3 pair core, or
the P3 transport adapter at module import time.  Every worker revalidates the
activation and response-blind manipulation receipts before those imports.

The runner retains ledgers, hashes, diagnostics, receipts, and resource
telemetry.  It never retains rendered RGB frames and never emits outcome
values through progress or telemetry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Callable, Iterable

import psutil

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# The selected P3 successor measured exactly this W8 native-thread behavior.
# Remove generic launcher's numeric caps before any numeric module is imported.
os.environ["RCLE_P3_THREAD_MODE"] = "LEGACY_UNRESTRICTED_OPENBLAS"
os.environ["RCLE_P3_NATIVE_THREADS"] = "18"
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.pop(_name, None)


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
RUNNER_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_P4_FORMAL_W8_R0"
WORKERS = 8
NATIVE_THREADS = 18
FRAME_COUNT = 602
PAIR_COUNT = 601
FORMAL_ARM_COUNT = 496
MAIN_ARM_COUNT = 480
GUARD_ARM_COUNT = 16
GIB = 1024**3
START_AND_REFILL_RAM_BYTES = 8 * GIB
DRAIN_RAM_BYTES = 6 * GIB
HARD_STOP_RAM_BYTES = 4 * GIB
MINIMUM_FREE_DISK_BYTES = 40 * GIB
HEARTBEAT_SECONDS = 20.0
DIAGNOSTIC_FIELDS = (
    "evaluable",
    "compensated_expansion_median_per_s",
    "detected_feature_count",
    "forward_backward_consistent_count",
    "forward_backward_consistent_fraction",
    "median_forward_backward_error_px",
    "occupied_3x3_cells",
)
ROLE_SET = {"MAIN_FACTORIAL", "POSITIVE_GUARDRAIL"}
BLOCK_SET = {"ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17"}


class InvalidFormalEvidence(ValueError):
    """A hash, identity, completed-arm, or authority invariant failed."""


class ResumableResourceStop(RuntimeError):
    """Host resources require an outcome-blind pause and exact resume."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidFormalEvidence(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(payload)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError as error:
        raise InvalidFormalEvidence(f"PATH_OUTSIDE_REPOSITORY:{path}") from error


def binding_map(value: dict[str, Any]) -> dict[str, str]:
    bindings = value.get("bindings")
    if not isinstance(bindings, list):
        raise InvalidFormalEvidence("ACTIVATION_BINDINGS")
    result: dict[str, str] = {}
    for item in bindings:
        if not isinstance(item, dict):
            raise InvalidFormalEvidence("ACTIVATION_BINDING_OBJECT")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or path in result
        ):
            raise InvalidFormalEvidence("ACTIVATION_BINDING_INVALID")
        result[path.replace("\\", "/")] = digest
    return result


def require_activation_binding(
    activation: dict[str, Any], path: Path
) -> str:
    relative = relative_to_repo(path)
    expected = binding_map(activation).get(relative)
    if expected is None:
        raise InvalidFormalEvidence(f"ACTIVATION_BINDING_MISSING:{relative}")
    actual = sha256_file(path)
    if actual != expected:
        raise InvalidFormalEvidence(f"ACTIVATION_BINDING_DRIFT:{relative}")
    return actual


def verify_all_activation_bindings(
    activation: dict[str, Any],
) -> dict[str, str]:
    """Re-hash every activation binding immediately before formal execution."""

    expected = binding_map(activation)
    root = repo_root().resolve()
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise InvalidFormalEvidence(
                f"ACTIVATION_BINDING_OUTSIDE:{relative}"
            ) from error
        if not path.is_file() or sha256_file(path) != digest:
            raise InvalidFormalEvidence(
                f"ACTIVATION_BINDING_DRIFT:{relative}"
            )
    return expected


def activation_phase_path(
    activation: dict[str, Any], phase: str
) -> Path:
    for item in activation.get("bindings", []):
        if isinstance(item, dict) and item.get("phase") == phase:
            path = item.get("path")
            if isinstance(path, str):
                return (repo_root() / path).resolve()
    raise InvalidFormalEvidence(f"ACTIVATION_PHASE_BINDING_MISSING:{phase}")


def validate_authority(
    activation_path: Path,
    manipulation_path: Path,
    identity_path: Path,
    trajectory_path: Path,
    run_id: str,
) -> dict[str, str]:
    activation = load_json(activation_path)
    if activation.get("protocol_id") != PROTOCOL_ID:
        raise InvalidFormalEvidence("ACTIVATION_PROTOCOL")
    if activation.get("formal_execution_authorized") is not True:
        raise InvalidFormalEvidence("FORMAL_EXECUTION_NOT_AUTHORIZED")
    if activation.get("p4_activated") is not True:
        raise InvalidFormalEvidence("P4_NOT_ACTIVATED")
    verify_all_activation_bindings(activation)
    selected = activation.get("selected_profile")
    if selected is None and isinstance(activation.get("scheduler"), dict):
        selected = activation["scheduler"].get("selected_profile")
    if selected is None and isinstance(activation.get("execution"), dict):
        selected = activation["execution"].get("scheduler_profile")
    if selected != "W8":
        raise InvalidFormalEvidence("ACTIVATION_PROFILE_NOT_W8")
    locked_run_id = activation.get(
        "run_id",
        activation.get("formal_run_id", activation.get("activation_id")),
    )
    if locked_run_id != run_id:
        raise InvalidFormalEvidence("ACTIVATION_RUN_ID")

    activation_sha = sha256_file(activation_path)
    identity_sha = require_activation_binding(activation, identity_path)
    runner_sha = require_activation_binding(
        activation, Path(__file__).resolve()
    )

    manipulation = load_json(manipulation_path)
    if manipulation.get("schema") != (
        "rcle.periodic_self_motion_counterfactual."
        "p4_manipulation_independent_validation.v1"
    ):
        raise InvalidFormalEvidence("MANIPULATION_INDEPENDENT_SCHEMA")
    if manipulation.get("validation") != "VALID":
        raise InvalidFormalEvidence("MANIPULATION_NOT_VALIDATED")
    if manipulation.get("terminal") != "PASS":
        raise InvalidFormalEvidence("FORMAL_MANIPULATION_NOT_PASS")
    producer_path = (
        manipulation_path.parent / "formal_manipulation_receipt.json"
    )
    if (
        not producer_path.is_file()
        or sha256_file(producer_path) != manipulation.get("receipt_sha256")
    ):
        raise InvalidFormalEvidence("MANIPULATION_PRODUCER_RECEIPT_DRIFT")
    producer = load_json(producer_path)
    if (
        producer.get("schema")
        != "rcle.periodic_self_motion_counterfactual.p4_manipulation.v1"
        or producer.get("protocol_id") != PROTOCOL_ID
        or producer.get("terminal") != "PASS"
    ):
        raise InvalidFormalEvidence("MANIPULATION_PRODUCER_NOT_PASS")
    if producer.get("r3_imported_or_executed") is not False:
        raise InvalidFormalEvidence("MANIPULATION_R3_FIREWALL")
    if producer.get("algorithm_output_read") is not False:
        raise InvalidFormalEvidence("MANIPULATION_OUTCOME_FIREWALL")

    p1_receipt_path = activation_phase_path(
        activation, "P1_INDEPENDENT_RECEIPT"
    )
    require_activation_binding(activation, p1_receipt_path)
    p1_receipt = load_json(p1_receipt_path)
    expected_trajectory_sha = p1_receipt.get("evidence_sha256", {}).get(
        "trajectory_manifest.json"
    )
    trajectory_sha = sha256_file(trajectory_path)
    if trajectory_sha != expected_trajectory_sha:
        raise InvalidFormalEvidence("TRAJECTORY_MANIFEST_DRIFT")
    return {
        "activation_sha256": activation_sha,
        "identity_manifest_sha256": identity_sha,
        "manipulation_receipt_sha256": sha256_file(manipulation_path),
        "manipulation_producer_receipt_sha256": sha256_file(
            producer_path
        ),
        "trajectory_manifest_sha256": trajectory_sha,
        "runner_sha256": runner_sha,
    }


def validate_identity_manifest(
    identity_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(identity_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise InvalidFormalEvidence("IDENTITY_PROTOCOL")
    identities = manifest.get("identities")
    if not isinstance(identities, list) or len(identities) != FORMAL_ARM_COUNT:
        raise InvalidFormalEvidence("FORMAL_IDENTITY_COUNT")
    declared_count = manifest.get(
        "identity_count", manifest.get("formal_arm_count")
    )
    if declared_count not in {None, FORMAL_ARM_COUNT}:
        raise InvalidFormalEvidence("DECLARED_IDENTITY_COUNT")
    seen: set[str] = set()
    main = guard = 0
    for item in identities:
        if not isinstance(item, dict):
            raise InvalidFormalEvidence("IDENTITY_OBJECT")
        sequence_id = item.get("sequence_id")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id.startswith("FORMAL_")
            or sequence_id in seen
        ):
            raise InvalidFormalEvidence("SEQUENCE_ID")
        seen.add(sequence_id)
        if item.get("block") not in BLOCK_SET:
            raise InvalidFormalEvidence(f"IDENTITY_BLOCK:{sequence_id}")
        role = item.get("role")
        if role not in ROLE_SET:
            raise InvalidFormalEvidence(f"IDENTITY_ROLE:{sequence_id}")
        if (
            not isinstance(item.get("cluster_id"), str)
            or not isinstance(item.get("ordinal"), int)
            or not isinstance(item.get("arm"), str)
            or not isinstance(item.get("source_arm_id"), str)
            or not isinstance(item.get("numeric_seed_uint64"), int)
            or item.get("frame_count") != FRAME_COUNT
            or item.get("pair_count") != PAIR_COUNT
        ):
            raise InvalidFormalEvidence(f"IDENTITY_FIELDS:{sequence_id}")
        if role == "MAIN_FACTORIAL":
            main += 1
            if item["ordinal"] not in range(20):
                raise InvalidFormalEvidence(f"MAIN_ORDINAL:{sequence_id}")
        else:
            guard += 1
            if item["ordinal"] not in range(2):
                raise InvalidFormalEvidence(f"GUARD_ORDINAL:{sequence_id}")
    if main != MAIN_ARM_COUNT or guard != GUARD_ARM_COUNT:
        raise InvalidFormalEvidence(f"ROLE_COUNTS:{main}:{guard}")

    source_binding = manifest.get("source_manifest")
    if not isinstance(source_binding, dict):
        raise InvalidFormalEvidence("SOURCE_MANIFEST_BINDING")
    source_path_text = source_binding.get("path")
    source_sha = source_binding.get("sha256")
    if (
        not isinstance(source_path_text, str)
        or not isinstance(source_sha, str)
        or len(source_sha) != 64
    ):
        raise InvalidFormalEvidence("SOURCE_MANIFEST_BINDING")
    source_path = (repo_root() / source_path_text).resolve()
    if sha256_file(source_path) != source_sha:
        raise InvalidFormalEvidence("SOURCE_MANIFEST_DRIFT")
    return manifest, identities


def resource_action(
    available_bytes: int,
    *,
    in_flight: int,
    paging_positive_streak: int,
    free_disk_bytes: int = MINIMUM_FREE_DISK_BYTES,
) -> str:
    if available_bytes < HARD_STOP_RAM_BYTES or paging_positive_streak >= 2:
        return "STOP_RESUMABLE"
    if free_disk_bytes < MINIMUM_FREE_DISK_BYTES:
        return "DRAIN" if in_flight else "PAUSE_RESUMABLE"
    if available_bytes < DRAIN_RAM_BYTES:
        return "DRAIN"
    if available_bytes < START_AND_REFILL_RAM_BYTES:
        return "WAIT_NO_REFILL" if in_flight else "PAUSE_RESUMABLE"
    return "REFILL"


def _load_source_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            cluster_id = item.get("cluster_id")
            if not isinstance(cluster_id, str) or cluster_id in records:
                raise InvalidFormalEvidence("SOURCE_CLUSTER_ID")
            records[cluster_id] = item
    if len(records) != 88:
        raise InvalidFormalEvidence(f"SOURCE_CLUSTER_COUNT:{len(records)}")
    return records


def _poses_for_identity(
    identity: dict[str, Any],
    source_record: dict[str, Any],
    trajectory_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    arm = identity["arm"]
    if identity["role"] == "POSITIVE_GUARDRAIL":
        source_arm = next(
            (
                item
                for item in source_record["arms"]
                if item["arm_id"] == identity["source_arm_id"]
            ),
            None,
        )
        if not isinstance(source_arm, dict):
            raise InvalidFormalEvidence("SOURCE_GUARD_ARM")
        poses = source_arm.get("trajectory")
    else:
        trajectory = trajectory_manifest.get(identity["block"])
        if not isinstance(trajectory, dict):
            raise InvalidFormalEvidence("TRAJECTORY_BLOCK")
        if arm.startswith("STATIC_CAMERA"):
            poses = [
                {
                    "frame_index": index,
                    "timestamp_s": pose["timestamp_s"],
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
                for index, pose in enumerate(trajectory["poses"])
            ]
        else:
            poses = trajectory.get("poses")
    if not isinstance(poses, list) or len(poses) != FRAME_COUNT:
        raise InvalidFormalEvidence("POSE_COUNT")
    return poses


def _worker_paths(
    output_dir: Path, sequence_id: str, attempt_id: str
) -> tuple[Path, Path]:
    staging = output_dir / "staging" / f"{sequence_id}.{attempt_id}.tmp"
    final = output_dir / "arms" / sequence_id
    return staging, final


def _evaluate_arm(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one complete arm and atomically publish its receipt."""

    activation_path = Path(task["activation_path"])
    manipulation_path = Path(task["manipulation_path"])
    identity_path = Path(task["identity_path"])
    trajectory_path = Path(task["trajectory_path"])
    output_dir = Path(task["output_dir"])
    run_id = task["run_id"]
    identity = task["identity"]
    authority = validate_authority(
        activation_path,
        manipulation_path,
        identity_path,
        trajectory_path,
        run_id,
    )
    manifest, _ = validate_identity_manifest(identity_path)
    if sha256_file(identity_path) != authority["identity_manifest_sha256"]:
        raise InvalidFormalEvidence("WORKER_IDENTITY_MANIFEST_DRIFT")

    # Only now may the worker import numeric/R3 code.
    import numpy as np

    from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0 import (
        producer as r3,
    )
    from scripts.research.egomotion_compensated_looming.temporal_structure_diagnostic_r1 import (
        extract as diagnostic,
    )
    from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
        p3_analysis_r0 as analysis,
        p3_runtime_preflight_r0 as runtime,
        p3_transport_r0 as transport,
    )

    runtime._initialize_worker()
    thread_guard = runtime._native_thread_guard()
    if not runtime._native_thread_guard_valid(thread_guard):
        raise InvalidFormalEvidence("NATIVE_THREAD_GUARD")

    source_path = (
        repo_root() / manifest["source_manifest"]["path"]
    ).resolve()
    records = _load_source_records(source_path)
    source_record = records.get(identity["cluster_id"])
    if not isinstance(source_record, dict):
        raise InvalidFormalEvidence("SOURCE_CLUSTER_MISSING")
    if (
        source_record.get("block") != identity["block"]
        or source_record.get("ordinal") != identity["ordinal"]
        or source_record.get("numeric_seed_uint64")
        != identity["numeric_seed_uint64"]
        or source_record.get("scene", {}).get("scene_geometry_sha256")
        != identity.get("scene_geometry_sha256")
    ):
        raise InvalidFormalEvidence("SOURCE_IDENTITY_DRIFT")
    source_arm = next(
        (
            item
            for item in source_record.get("arms", [])
            if item.get("arm_id") == identity["source_arm_id"]
        ),
        None,
    )
    if (
        not isinstance(source_arm, dict)
        or source_arm.get("trajectory_sha256")
        != identity.get("trajectory_sha256")
    ):
        raise InvalidFormalEvidence("SOURCE_ARM_TRAJECTORY_DRIFT")
    scene = source_record["scene"]
    trajectory_manifest = load_json(trajectory_path)
    poses = _poses_for_identity(identity, source_record, trajectory_manifest)
    protocol = load_json(repo_root() / transport.PROTOCOL_RELATIVE)

    attempt_id = task["attempt_id"]
    staging, final = _worker_paths(
        output_dir, identity["sequence_id"], attempt_id
    )
    if final.exists():
        raise InvalidFormalEvidence("FINAL_ARM_ALREADY_EXISTS")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = r3.PairState()
    previous_rgb = previous_mask = previous_pose = None
    static_frame = None
    frame_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    streak = 0
    for frame_index, pose in enumerate(poses):
        if identity["arm"].startswith("STATIC_CAMERA") and static_frame is not None:
            rgb, mask = static_frame
        else:
            rgb, mask = runtime._render_frame(scene, pose, identity["arm"])
            if identity["arm"].startswith("STATIC_CAMERA"):
                static_frame = (rgb, mask)
        frame_rows.append(
            {
                "frame_index": frame_index,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "valid_mask_sha256": hashlib.sha256(
                    mask.astype(np.uint8).tobytes()
                ).hexdigest(),
            }
        )
        if previous_rgb is not None:
            pair_index = frame_index - 1
            row = transport.evaluate_pair(
                pair_index=pair_index,
                previous_rgb=previous_rgb,
                current_rgb=rgb,
                previous_valid=previous_mask,
                current_valid=mask,
                previous_timestamp_s=previous_pose["timestamp_s"],
                current_timestamp_s=pose["timestamp_s"],
                previous_world_from_camera=np.asarray(
                    previous_pose["rotation_matrix"], dtype=np.float64
                ),
                current_world_from_camera=np.asarray(
                    pose["rotation_matrix"], dtype=np.float64
                ),
                intrinsic=runtime.geometry.K,
                protocol=protocol,
                state=state,
            )
            metrics = diagnostic.flow_direction_metrics(
                transport.rgb_to_gray(previous_rgb),
                transport.rgb_to_gray(rgb),
                transport.valid_mask(
                    previous_mask, previous_rgb.shape[:2]
                ),
            )
            row.update(metrics)
            row["occupied_3x3_cells"] = row["occupied_grid_cells"]
            response = row.get("compensated_expansion_median_per_s")
            if row.get("evaluable") is True:
                streak = streak + 1 if float(response) > 0.01 else 0
            else:
                streak = 0
                row.setdefault("compensated_expansion_median_per_s", None)
            row["compensated_three_pair_trigger"] = streak >= 3
            envelope = {
                "sequence_id": identity["sequence_id"],
                "cluster_id": identity["cluster_id"],
                "block": identity["block"],
                "ordinal": identity["ordinal"],
                "role": identity["role"],
                "arm": identity["arm"],
                "pair_index": pair_index,
            }
            envelope.update(row)
            for field in DIAGNOSTIC_FIELDS:
                if field not in envelope:
                    raise InvalidFormalEvidence(
                        f"DIAGNOSTIC_FIELD_MISSING:{field}"
                    )
            pair_rows.append(envelope)
        previous_rgb = rgb
        previous_mask = mask
        previous_pose = pose
    if len(frame_rows) != FRAME_COUNT or len(pair_rows) != PAIR_COUNT:
        raise InvalidFormalEvidence("ARM_LENGTH")

    frame_path = staging / "frame_manifest.jsonl"
    ledger_path = staging / "pair_ledger.jsonl"
    frame_path.write_bytes(b"".join(canonical_bytes(row) for row in frame_rows))
    ledger_path.write_bytes(b"".join(canonical_bytes(row) for row in pair_rows))
    reduced = analysis.reduce_pair_rows(pair_rows)
    reduced_path = staging / "reduced_metrics.json"
    write_json(reduced_path, reduced, exclusive=True)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_arm_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        **authority,
        "sequence_id": identity["sequence_id"],
        "cluster_id": identity["cluster_id"],
        "block": identity["block"],
        "ordinal": identity["ordinal"],
        "role": identity["role"],
        "arm": identity["arm"],
        "numeric_seed_uint64": identity["numeric_seed_uint64"],
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
        "pair_ledger_path": (
            f"arms/{identity['sequence_id']}/pair_ledger.jsonl"
        ),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "frame_manifest_path": (
            f"arms/{identity['sequence_id']}/frame_manifest.jsonl"
        ),
        "frame_manifest_sha256": sha256_file(frame_path),
        "reduced_metrics_path": (
            f"arms/{identity['sequence_id']}/reduced_metrics.json"
        ),
        "reduced_metrics_sha256": sha256_file(reduced_path),
        "thread_guard": thread_guard,
        "timing": {"wall_seconds": elapsed},
        "rgb_frames_retained": False,
        "sequence16_android_realtime": False,
        "terminal": "ARM_COMPLETE",
    }
    write_json(staging / "receipt.json", receipt, exclusive=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    return {
        "sequence_id": identity["sequence_id"],
        "receipt_path": f"arms/{identity['sequence_id']}/receipt.json",
        "receipt_sha256": sha256_file(final / "receipt.json"),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def validate_complete_arm(
    output_dir: Path,
    identity: dict[str, Any],
    expected: dict[str, str],
    run_id: str,
) -> dict[str, str]:
    arm_dir = output_dir / "arms" / identity["sequence_id"]
    receipt_path = arm_dir / "receipt.json"
    if not receipt_path.is_file():
        raise InvalidFormalEvidence(
            f"FINAL_ARM_WITHOUT_RECEIPT:{identity['sequence_id']}"
        )
    receipt = load_json(receipt_path)
    required = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_arm_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "activation_sha256": expected["activation_sha256"],
        "identity_manifest_sha256": expected["identity_manifest_sha256"],
        "manipulation_receipt_sha256": expected[
            "manipulation_receipt_sha256"
        ],
        "manipulation_producer_receipt_sha256": expected[
            "manipulation_producer_receipt_sha256"
        ],
        "trajectory_manifest_sha256": expected[
            "trajectory_manifest_sha256"
        ],
        "runner_sha256": expected["runner_sha256"],
        "sequence_id": identity["sequence_id"],
        "cluster_id": identity["cluster_id"],
        "block": identity["block"],
        "ordinal": identity["ordinal"],
        "role": identity["role"],
        "arm": identity["arm"],
        "numeric_seed_uint64": identity["numeric_seed_uint64"],
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
        "terminal": "ARM_COMPLETE",
        "rgb_frames_retained": False,
        "sequence16_android_realtime": False,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise InvalidFormalEvidence(
                f"ARM_RECEIPT_DRIFT:{identity['sequence_id']}:{key}"
            )
    for path_key, sha_key in (
        ("pair_ledger_path", "pair_ledger_sha256"),
        ("frame_manifest_path", "frame_manifest_sha256"),
        ("reduced_metrics_path", "reduced_metrics_sha256"),
    ):
        relative = receipt.get(path_key)
        if not isinstance(relative, str):
            raise InvalidFormalEvidence(f"ARM_PATH:{path_key}")
        path = (output_dir / relative).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as error:
            raise InvalidFormalEvidence("ARM_PATH_ESCAPE") from error
        if not path.is_file() or sha256_file(path) != receipt.get(sha_key):
            raise InvalidFormalEvidence(
                f"ARM_ARTIFACT_DRIFT:{identity['sequence_id']}:{path_key}"
            )
    rows = list(_iter_jsonl(output_dir / receipt["pair_ledger_path"]))
    if len(rows) != PAIR_COUNT:
        raise InvalidFormalEvidence("ARM_LEDGER_ROW_COUNT")
    for pair_index, row in enumerate(rows):
        for key in (
            "sequence_id",
            "cluster_id",
            "block",
            "ordinal",
            "role",
            "arm",
        ):
            if row.get(key) != identity[key]:
                raise InvalidFormalEvidence(
                    f"ARM_LEDGER_IDENTITY:{identity['sequence_id']}:{key}"
                )
        if row.get("pair_index") != pair_index:
            raise InvalidFormalEvidence("ARM_LEDGER_PAIR_ORDER")
        for field in DIAGNOSTIC_FIELDS:
            if field not in row:
                raise InvalidFormalEvidence(
                    f"ARM_LEDGER_DIAGNOSTIC:{field}"
                )
    return {
        "sequence_id": identity["sequence_id"],
        "receipt_path": f"arms/{identity['sequence_id']}/receipt.json",
        "receipt_sha256": sha256_file(receipt_path),
    }


def recover_state(
    output_dir: Path,
    identities: list[dict[str, Any]],
    authority: dict[str, str],
    run_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    by_id = {item["sequence_id"]: item for item in identities}
    completed: list[dict[str, str]] = []
    arms_root = output_dir / "arms"
    if arms_root.exists():
        for path in arms_root.iterdir():
            if not path.is_dir() or path.name not in by_id:
                raise InvalidFormalEvidence(f"UNKNOWN_FINAL_ARM:{path.name}")
            completed.append(
                validate_complete_arm(
                    output_dir, by_id[path.name], authority, run_id
                )
            )

    quarantine = output_dir / "quarantine"
    staging_root = output_dir / "staging"
    if staging_root.exists():
        for path in sorted(staging_root.iterdir()):
            if not path.is_dir() or not path.name.endswith(".tmp"):
                raise InvalidFormalEvidence(f"UNKNOWN_STAGING_ENTRY:{path.name}")
            sequence_id = next(
                (
                    candidate
                    for candidate in by_id
                    if path.name.startswith(candidate + ".")
                ),
                None,
            )
            if sequence_id is None:
                raise InvalidFormalEvidence(f"UNKNOWN_STAGING_ARM:{path.name}")
            quarantine.mkdir(parents=True, exist_ok=True)
            target = quarantine / f"{path.name}.incomplete"
            if target.exists():
                raise InvalidFormalEvidence("QUARANTINE_COLLISION")
            path.rename(target)
            write_json(
                target / "interruption.json",
                {
                    "schema": "p4_incomplete_arm_interruption.v1",
                    "protocol_id": PROTOCOL_ID,
                    "run_id": run_id,
                    "sequence_id": sequence_id,
                    "disposition": "RESTART_FROM_FRAME_ZERO",
                    "outcome_values_emitted": False,
                    "recorded_at_utc": utc_now(),
                },
                exclusive=True,
            )
    completed_ids = {item["sequence_id"] for item in completed}
    pending = [
        item for item in identities if item["sequence_id"] not in completed_ids
    ]
    return sorted(completed, key=lambda item: item["sequence_id"]), pending


def ensure_claim(
    output_dir: Path,
    run_id: str,
    authority: dict[str, str],
) -> dict[str, Any]:
    claim = {
        "schema": "p4_formal_claim.v1",
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "run_id": run_id,
        **authority,
        "workers": WORKERS,
        "native_threads_per_worker": NATIVE_THREADS,
        "resume_identity": (
            "EXACT_RUN_ID_ACTIVATION_IDENTITY_MANIPULATION_AND_RUNNER"
        ),
        "formal_arm_count": FORMAL_ARM_COUNT,
    }
    path = output_dir / "claim.json"
    if path.exists():
        if canonical_bytes(load_json(path)) != canonical_bytes(claim):
            raise InvalidFormalEvidence("FORMAL_CLAIM_DRIFT")
    else:
        write_json(path, claim, exclusive=True)
    return claim


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise InvalidFormalEvidence("OUTPUT_VOLUME_NOT_RESOLVABLE")
        candidate = candidate.parent
    return candidate


def _resource_sample(output_dir: Path) -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    root_process = psutil.Process()
    processes = [root_process]
    try:
        processes.extend(root_process.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    rss = read_bytes = write_bytes = 0
    for process in processes:
        try:
            rss += int(process.memory_info().rss)
            io = process.io_counters()
            read_bytes += int(io.read_bytes)
            write_bytes += int(io.write_bytes)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "available_ram_bytes": int(memory.available),
        "resident_memory_bytes": rss,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "swap_in_total": int(swap.sin),
        "swap_out_total": int(swap.sout),
        "free_disk_bytes": int(
            shutil.disk_usage(_existing_ancestor(output_dir)).free
        ),
    }


def _append_telemetry(
    output_dir: Path,
    *,
    resource: dict[str, int],
    completed_count: int,
    in_flight: int,
    action: str,
    started: float,
) -> None:
    record = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_resource_telemetry.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "sampled_at_utc": utc_now(),
        "elapsed_seconds": max(time.perf_counter() - started, 0.0),
        "completed_arms": completed_count,
        "total_arms": FORMAL_ARM_COUNT,
        "in_flight_arms": in_flight,
        "resource_action": action,
        **resource,
        "outcome_fields_present": False,
    }
    path = output_dir / "telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes(record))


def _write_progress(
    output_dir: Path,
    completed_count: int,
    in_flight: int,
    started: float,
    resource: dict[str, int],
    status: str,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    progress = {
        "schema": "p4_formal_progress.v1",
        "protocol_id": PROTOCOL_ID,
        "phase": "P4_FORMAL_W8",
        "completed_units": completed_count,
        "total_units": FORMAL_ARM_COUNT,
        "completed_arms": completed_count,
        "total_arms": FORMAL_ARM_COUNT,
        "completed_pairs": completed_count * PAIR_COUNT,
        "total_pairs": FORMAL_ARM_COUNT * PAIR_COUNT,
        "in_flight_arms": in_flight,
        "throughput": completed_count / elapsed,
        "throughput_pairs_per_s": completed_count * PAIR_COUNT / elapsed,
        "eta_seconds": (
            (FORMAL_ARM_COUNT - completed_count) * elapsed / completed_count
            if completed_count
            else None
        ),
        "last_progress_at": utc_now(),
        "last_heartbeat_utc": utc_now(),
        "status": status,
        "worker_count": WORKERS,
        **resource,
        "outcome_fields_present": False,
    }
    write_json(output_dir / "progress.json", progress)


def _terminate_executor(executor: ProcessPoolExecutor) -> None:
    processes = list(getattr(executor, "_processes", {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()


def run_formal(
    *,
    activation_path: Path,
    manipulation_path: Path,
    identity_path: Path,
    trajectory_path: Path,
    output_dir: Path,
    run_id: str,
    evaluator: Callable[[dict[str, Any]], dict[str, Any]] = _evaluate_arm,
) -> dict[str, Any]:
    authority = validate_authority(
        activation_path,
        manipulation_path,
        identity_path,
        trajectory_path,
        run_id,
    )
    _, identities = validate_identity_manifest(identity_path)
    initial = _resource_sample(output_dir)
    if initial["available_ram_bytes"] < START_AND_REFILL_RAM_BYTES:
        raise ResumableResourceStop("START_RAM_BELOW_8_GIB")
    if initial["free_disk_bytes"] < MINIMUM_FREE_DISK_BYTES:
        raise ResumableResourceStop("START_DISK_BELOW_40_GIB")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_claim(output_dir, run_id, authority)
    completed, pending = recover_state(
        output_dir, identities, authority, run_id
    )
    if not pending:
        return _finalize_success(
            output_dir, run_id, authority, completed
        )

    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + f"-pid-{os.getpid()}"
    )
    started = time.perf_counter()
    initial_swap_in = initial["swap_in_total"]
    initial_swap_out = initial["swap_out_total"]
    last_swap_in = initial_swap_in
    last_swap_out = initial_swap_out
    paging_streak = 0
    in_flight: dict[Any, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
    )
    stop_reason: str | None = None
    try:
        while pending or in_flight:
            resource = _resource_sample(output_dir)
            paging_delta = max(
                0,
                resource["swap_in_total"]
                - last_swap_in
                + resource["swap_out_total"]
                - last_swap_out,
            )
            paging_streak = paging_streak + 1 if paging_delta > 0 else 0
            last_swap_in = resource["swap_in_total"]
            last_swap_out = resource["swap_out_total"]
            action = resource_action(
                resource["available_ram_bytes"],
                in_flight=len(in_flight),
                paging_positive_streak=paging_streak,
                free_disk_bytes=resource["free_disk_bytes"],
            )
            if action == "STOP_RESUMABLE":
                stop_reason = (
                    "SUSTAINED_PAGING"
                    if paging_streak >= 2
                    else "RUN_RAM_BELOW_4_GIB"
                )
                raise ResumableResourceStop(stop_reason)
            if action == "PAUSE_RESUMABLE":
                stop_reason = (
                    "RUN_DISK_BELOW_40_GIB"
                    if resource["free_disk_bytes"]
                    < MINIMUM_FREE_DISK_BYTES
                    else "REFILL_RAM_BELOW_8_GIB"
                )
                break
            if action == "REFILL":
                while pending and len(in_flight) < WORKERS:
                    identity = pending.pop(0)
                    task = {
                        "activation_path": str(activation_path),
                        "manipulation_path": str(manipulation_path),
                        "identity_path": str(identity_path),
                        "trajectory_path": str(trajectory_path),
                        "output_dir": str(output_dir),
                        "run_id": run_id,
                        "identity": identity,
                        "attempt_id": attempt_id,
                    }
                    in_flight[executor.submit(evaluator, task)] = identity

            done, _ = wait(
                in_flight,
                timeout=HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                identity = in_flight.pop(future)
                result = future.result()
                if result.get("sequence_id") != identity["sequence_id"]:
                    raise InvalidFormalEvidence("WORKER_RESULT_IDENTITY")
                completed.append(result)
            status = (
                "DRAINING"
                if action in {"DRAIN", "WAIT_NO_REFILL"}
                else "RUNNING"
            )
            _write_progress(
                output_dir,
                len(completed),
                len(in_flight),
                started,
                resource,
                status,
            )
            _append_telemetry(
                output_dir,
                resource=resource,
                completed_count=len(completed),
                in_flight=len(in_flight),
                action=action,
                started=started,
            )
            if action == "DRAIN" and not in_flight:
                stop_reason = (
                    "RUN_DISK_BELOW_40_GIB"
                    if resource["free_disk_bytes"]
                    < MINIMUM_FREE_DISK_BYTES
                    else "RUN_RAM_BELOW_6_GIB"
                )
                break
        executor.shutdown(wait=True)
    except ResumableResourceStop:
        _terminate_executor(executor)
    except BaseException:
        _terminate_executor(executor)
        raise

    if stop_reason is not None:
        attempt = {
            "schema": "p4_formal_attempt.v1",
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": "PAUSED_RESUMABLE",
            "reason": stop_reason,
            "completed_arms": len(completed),
            "remaining_arms": FORMAL_ARM_COUNT - len(completed),
            "outcome_values_emitted": False,
            "recorded_at_utc": utc_now(),
        }
        write_json(
            output_dir / "attempts" / f"{attempt_id}.json",
            attempt,
            exclusive=True,
        )
        return attempt
    return _finalize_success(output_dir, run_id, authority, completed)


def _finalize_success(
    output_dir: Path,
    run_id: str,
    authority: dict[str, str],
    arms: list[dict[str, str]],
) -> dict[str, Any]:
    ordered = sorted(arms, key=lambda item: item["sequence_id"])
    if (
        len(ordered) != FORMAL_ARM_COUNT
        or len({item["sequence_id"] for item in ordered})
        != FORMAL_ARM_COUNT
    ):
        raise InvalidFormalEvidence("BUNDLE_ARM_SET")
    residual = [
        child.pid
        for child in psutil.Process().children(recursive=True)
        if child.is_running()
    ]
    if residual:
        raise InvalidFormalEvidence(
            "RESIDUAL_WORKERS:" + ",".join(map(str, residual))
        )
    terminal = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_bundle_success.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        **authority,
        "arms": ordered,
        "arm_count": FORMAL_ARM_COUNT,
        "frame_count": FORMAL_ARM_COUNT * FRAME_COUNT,
        "pair_count": FORMAL_ARM_COUNT * PAIR_COUNT,
        "prerequisite_gates": {
            "geometry_validation": "PASS",
            "quality_strength_lock": "PASS",
            "formal_main_manipulation": "PASS",
            "manipulation_receipt_sha256": authority[
                "manipulation_receipt_sha256"
            ],
        },
        "residual_worker_pids": residual,
        "rgb_frames_retained": False,
        "sequence16_android_realtime": False,
        "scientific_outcome_interpreted": False,
        "terminal": "BUNDLE_COMPLETE",
    }
    success_path = output_dir / "success.json"
    if success_path.exists():
        if canonical_bytes(load_json(success_path)) != canonical_bytes(terminal):
            raise InvalidFormalEvidence("BUNDLE_SUCCESS_DRIFT")
    else:
        write_json(success_path, terminal, exclusive=True)
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-lock", type=Path, required=True)
    parser.add_argument("--manipulation-receipt", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.workers != WORKERS:
        raise SystemExit("FORMAL_W8_ONLY")
    try:
        result = run_formal(
            activation_path=arguments.activation_lock.resolve(),
            manipulation_path=arguments.manipulation_receipt.resolve(),
            identity_path=arguments.identity_manifest.resolve(),
            trajectory_path=arguments.trajectory_manifest.resolve(),
            output_dir=arguments.output_dir.resolve(),
            run_id=arguments.run_id,
        )
    except ResumableResourceStop as error:
        print(
            json.dumps(
                {
                    "status": "PAUSED_RESUMABLE",
                    "reason": str(error),
                    "outcome_values_emitted": False,
                },
                sort_keys=True,
            )
        )
        return 3
    except (InvalidFormalEvidence, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "INVALID",
                    "error": str(error),
                    "outcome_values_emitted": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("terminal") == "BUNDLE_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
