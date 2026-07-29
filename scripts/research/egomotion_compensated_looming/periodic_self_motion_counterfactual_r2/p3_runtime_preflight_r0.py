"""Guarded-host P3 runtime preflight over exactly eight PREFLIGHT identities.

The runner emits transport hashes and resource telemetry only.  It never emits
or interprets arm-level RCLE response or trigger outcomes.
"""

from __future__ import annotations

import os

# Windows spawn imports this module in every worker.  These caps must therefore
# be installed before importing NumPy/OpenCV (and before either library can
# initialize a native thread pool).
THREAD_MODE = os.environ.get("RCLE_P3_THREAD_MODE", "CONTROLLED")
if THREAD_MODE not in {"CONTROLLED", "LEGACY_UNRESTRICTED_OPENBLAS"}:
    raise RuntimeError("RCLE_P3_THREAD_MODE_INVALID")
_native_threads_literal = os.environ.get(
    "RCLE_P3_NATIVE_THREADS",
    "18" if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS" else "1",
)
if _native_threads_literal not in {"1", "2", "4", "18"}:
    raise RuntimeError("RCLE_P3_NATIVE_THREADS_MUST_BE_1_2_4_OR_18")
NATIVE_THREADS = int(_native_threads_literal)
_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_variable in _NATIVE_THREAD_VARIABLES:
    if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS":
        os.environ.pop(_thread_variable, None)
    else:
        os.environ[_thread_variable] = _native_threads_literal

import argparse
import ctypes
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import time
from typing import Any
from unittest import mock

import cv2
import numpy as np
import psutil

from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from . import generator_geometry as geometry
from . import p3_transport_r0 as transport
from . import quality_interventions_r0 as quality


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
P3_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0"
)
BLOCK = "ADVIO_14"
FRAME_COUNT = 602
PAIR_COUNT = 601
WORKER_PROFILES = (4, 8)
SCHEDULER_NATIVE_THREADS = {4: 4, 8: 18}
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
BLUR_SIGMA_PX = 0.475
LOW_TEXTURE_ALPHA = 0.15
GIB = 1024**3


class InvalidPreflight(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return transport.canonical_bytes(value)


def sha256_file(path: Path) -> str:
    return transport.sha256_file(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_record(kind: str) -> dict[str, Any]:
    if kind not in {"FACTORIAL", "GUARDRAIL"}:
        raise InvalidPreflight("PREFLIGHT_KIND")
    token = f"{PROTOCOL_ID}|PREFLIGHT|{BLOCK}|{kind}|00"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "kind": kind,
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(
            bytes.fromhex(digest)[:8], "big"
        ),
    }


def identity_manifest() -> dict[str, Any]:
    seeds = [_seed_record("FACTORIAL"), _seed_record("GUARDRAIL")]
    identities: list[dict[str, Any]] = []
    for seed, arms in zip(seeds, (FACTORIAL_ARMS, GUARDRAIL_ARMS)):
        for ordinal, arm in enumerate(arms):
            identities.append(
                {
                    "sequence_id": (
                        f"PREFLIGHT_{BLOCK}_{seed['kind']}_00__{arm}"
                    ),
                    "cluster_kind": seed["kind"],
                    "cluster_token_sha256": seed["token_sha256"],
                    "numeric_seed_uint64": seed["numeric_seed_uint64"],
                    "arm": arm,
                    "arm_ordinal": ordinal,
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
            )
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "p3_id": P3_ID,
        "role": "RUNTIME_PREFLIGHT_ONLY_NO_SCIENTIFIC_INTERPRETATION",
        "block": BLOCK,
        "seed_literal_case": "UPPERCASE_FACTORIAL_AND_GUARDRAIL",
        "seeds": seeds,
        "identities": identities,
        "identity_count": 8,
        "worker_profiles": list(WORKER_PROFILES),
        "prohibited_worker_profiles": [12, 16],
        "formal_seed_access": False,
        "formal_execution_authorized": False,
        "p4_activated": False,
    }
    payload["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return payload


def validate_identity_manifest(value: dict[str, Any]) -> None:
    expected = identity_manifest()
    if canonical_bytes(value) != canonical_bytes(expected):
        raise InvalidPreflight("IDENTITY_MANIFEST_DRIFT")


def _build_scene(kind: str) -> dict[str, Any]:
    seed = _seed_record(kind)["numeric_seed_uint64"]
    namespace = "GUARD" if kind == "GUARDRAIL" else "PREFLIGHT"
    # Reuse the frozen P1 scene construction byte-for-byte while supplying the
    # distinct five-field PREFLIGHT seed required by the R2 contract.
    with mock.patch.object(geometry, "derive_seed", return_value=seed):
        scene = geometry.build_scene(BLOCK, 0, namespace)
    scene["namespace"] = "PREFLIGHT"
    scene["preflight_kind"] = kind
    scene["preflight_seed_token_sha256"] = _seed_record(kind)["token_sha256"]
    scene["scene_geometry_sha256"] = hashlib.sha256(
        geometry.canonical_bytes(
            {
                key: value
                for key, value in scene.items()
                if key != "scene_geometry_sha256"
            }
        )
    ).hexdigest()
    return scene


def _trajectory_manifest() -> dict[str, Any]:
    path = (
        transport.repo_root()
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    trajectory = value[BLOCK]
    if trajectory["frame_count"] != FRAME_COUNT or trajectory["pair_count"] != PAIR_COUNT:
        raise InvalidPreflight("TRAJECTORY_LENGTH")
    return trajectory


def _poses_for_arm(arm: str) -> list[dict[str, Any]]:
    trajectory = _trajectory_manifest()
    if arm.startswith("STATIC_CAMERA"):
        timestamps = [item["timestamp_s"] for item in trajectory["poses"]]
        return [
            {
                "frame_index": index,
                "timestamp_s": timestamp,
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_matrix": np.eye(3).tolist(),
            }
            for index, timestamp in enumerate(timestamps)
        ]
    if arm.startswith("PERIODIC_6DOF_SELF_MOTION"):
        return trajectory["poses"]
    if arm.startswith("MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF"):
        return geometry._guard_trajectory(trajectory, True)["poses"]
    if arm.startswith("MONOTONIC_APPROACH_ONLY"):
        return geometry._guard_trajectory(trajectory, False)["poses"]
    raise InvalidPreflight(f"ARM:{arm}")


def _low_texture_rgb(
    scene: dict[str, Any],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(geometry.WIDTH, dtype=np.float64),
        np.arange(geometry.HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    depth, object_id, world = geometry._raycast(
        scene, rotation, translation, uv
    )
    valid = np.isfinite(depth)
    linear = np.zeros((len(uv), 3), dtype=np.float64)
    by_id = {
        int(item["object_id"]): item for item in scene["world"]["objects"]
    }
    for identifier in np.unique(object_id[valid]):
        selected = object_id == identifier
        item = by_id[int(identifier)]
        base = np.asarray(item["linear_rgb"], dtype=np.float64)
        frequency = float(item["texture"]["cycles_per_m"])
        phase = float(item["texture"]["phase"])
        checker = (
            np.floor((world[selected, 0] * frequency + phase) % 2.0)
            + np.floor((world[selected, 1] * frequency + phase) % 2.0)
        ) % 2.0
        clean_modulation = 0.65 + 0.35 * checker
        modulation = 0.825 + LOW_TEXTURE_ALPHA * (
            clean_modulation - 0.825
        )
        linear[selected] = np.clip(
            base[None, :] * modulation[:, None], 0.0, 1.0
        )
    rgb = quality.linear_to_srgb_u8(
        linear.reshape(geometry.HEIGHT, geometry.WIDTH, 3)
    )
    return rgb, valid.reshape(geometry.HEIGHT, geometry.WIDTH)


def _render_frame(
    scene: dict[str, Any],
    pose: dict[str, Any],
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(pose["translation_m"], dtype=np.float64)
    if arm.endswith("__LOW_TEXTURE"):
        return _low_texture_rgb(scene, rotation, translation)
    clean = geometry.render(scene, rotation, translation)
    rgb = clean["rgb"]
    if arm.endswith("__BLUR"):
        rgb = quality.apply_blur(rgb, BLUR_SIGMA_PX)
    return rgb, np.isfinite(clean["depth"])


def _initialize_worker() -> None:
    cv2.setNumThreads(
        1 if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS" else NATIVE_THREADS
    )
    cv2.setRNGSeed(20260728)


def _openblas_thread_count() -> int:
    libraries = sorted(
        (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob(
            "*openblas*.dll"
        )
    )
    if len(libraries) != 1:
        raise InvalidPreflight("OPENBLAS_LIBRARY_IDENTITY")
    library = ctypes.CDLL(os.fspath(libraries[0]))
    for symbol in (
        "scipy_openblas_get_num_threads64_",
        "openblas_get_num_threads64_",
        "openblas_get_num_threads",
    ):
        function = getattr(library, symbol, None)
        if function is not None:
            function.restype = ctypes.c_int
            return int(function())
    raise InvalidPreflight("OPENBLAS_THREAD_SYMBOL")


def _native_thread_guard() -> dict[str, Any]:
    return {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "blis_num_threads": os.environ.get("BLIS_NUM_THREADS"),
        "opencv_num_threads": cv2.getNumThreads(),
        "openblas_observed_num_threads": _openblas_thread_count(),
        "applied_before_numeric_imports": (
            THREAD_MODE != "LEGACY_UNRESTRICTED_OPENBLAS"
        ),
    }


def _expected_native_thread_guard() -> dict[str, Any]:
    if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS":
        return {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
            "veclib_maximum_threads": os.environ.get(
                "VECLIB_MAXIMUM_THREADS"
            ),
            "blis_num_threads": os.environ.get("BLIS_NUM_THREADS"),
            "opencv_num_threads": 1,
            "openblas_observed_num_threads": NATIVE_THREADS,
            "applied_before_numeric_imports": False,
        }
    return {
        "omp_num_threads": _native_threads_literal,
        "openblas_num_threads": _native_threads_literal,
        "mkl_num_threads": _native_threads_literal,
        "numexpr_num_threads": _native_threads_literal,
        "veclib_maximum_threads": _native_threads_literal,
        "blis_num_threads": _native_threads_literal,
        "opencv_num_threads": NATIVE_THREADS,
        "openblas_observed_num_threads": NATIVE_THREADS,
        "applied_before_numeric_imports": True,
    }


def _native_thread_guard_valid(value: dict[str, Any]) -> bool:
    if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS":
        return (
            value.get("opencv_num_threads") == 1
            and value.get("openblas_observed_num_threads") == NATIVE_THREADS
            and value.get("applied_before_numeric_imports") is False
        )
    return value == _expected_native_thread_guard()


def _evaluate_identity(identity: dict[str, Any]) -> dict[str, Any]:
    _initialize_worker()
    thread_guard = _native_thread_guard()
    if not _native_thread_guard_valid(thread_guard):
        raise InvalidPreflight("NATIVE_THREAD_GUARD")
    started = time.perf_counter()
    arm = identity["arm"]
    scene = _build_scene(identity["cluster_kind"])
    if (
        scene["numeric_seed_uint64"] != identity["numeric_seed_uint64"]
        or scene["preflight_seed_token_sha256"]
        != identity["cluster_token_sha256"]
    ):
        raise InvalidPreflight("WORKER_IDENTITY_DRIFT")
    poses = _poses_for_arm(arm)
    protocol = json.loads(
        (transport.repo_root() / transport.PROTOCOL_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    state = r3.PairState()
    frame_hashes: list[dict[str, Any]] = []
    pair_digests: list[str] = []
    previous_rgb: np.ndarray | None = None
    previous_valid: np.ndarray | None = None
    previous_pose: dict[str, Any] | None = None
    render_seconds = 0.0
    r3_seconds = 0.0
    static_frame: tuple[np.ndarray, np.ndarray] | None = None
    render_invocations = 0
    render_reuse_hits = 0
    for index, pose in enumerate(poses):
        render_started = time.perf_counter()
        if arm.startswith("STATIC_CAMERA") and static_frame is not None:
            rgb, mask = static_frame
            render_reuse_hits += 1
        else:
            rgb, mask = _render_frame(scene, pose, arm)
            render_invocations += 1
            if arm.startswith("STATIC_CAMERA"):
                static_frame = (rgb, mask)
        render_seconds += time.perf_counter() - render_started
        frame_hashes.append(
            {
                "frame_index": index,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "valid_mask_sha256": hashlib.sha256(
                    mask.astype(np.uint8).tobytes()
                ).hexdigest(),
            }
        )
        if previous_rgb is not None:
            r3_started = time.perf_counter()
            row = transport.evaluate_pair(
                pair_index=index - 1,
                previous_rgb=previous_rgb,
                current_rgb=rgb,
                previous_valid=previous_valid,
                current_valid=mask,
                previous_timestamp_s=previous_pose["timestamp_s"],
                current_timestamp_s=pose["timestamp_s"],
                previous_world_from_camera=np.asarray(
                    previous_pose["rotation_matrix"], dtype=np.float64
                ),
                current_world_from_camera=np.asarray(
                    pose["rotation_matrix"], dtype=np.float64
                ),
                intrinsic=geometry.K,
                protocol=protocol,
                state=state,
            )
            r3_seconds += time.perf_counter() - r3_started
            pair_digests.append(
                hashlib.sha256(canonical_bytes(row)).hexdigest()
            )
        previous_rgb = rgb
        previous_valid = mask
        previous_pose = pose
    if len(frame_hashes) != FRAME_COUNT or len(pair_digests) != PAIR_COUNT:
        raise InvalidPreflight("WORKER_LENGTH")
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_sequence_receipt.v1",
        "sequence_id": identity["sequence_id"],
        "cluster_kind": identity["cluster_kind"],
        "arm": arm,
        "numeric_seed_uint64": identity["numeric_seed_uint64"],
        "scene_geometry_sha256": scene["scene_geometry_sha256"],
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
        "frame_manifest_sha256": hashlib.sha256(
            canonical_bytes(frame_hashes)
        ).hexdigest(),
        "ordered_pair_numeric_sha256": hashlib.sha256(
            canonical_bytes(pair_digests)
        ).hexdigest(),
        "transport_identity_sha256": hashlib.sha256(
            canonical_bytes(
                {
                    "identity": identity,
                    "scene_geometry_sha256": scene["scene_geometry_sha256"],
                    "frames": frame_hashes,
                    "pairs": pair_digests,
                }
            )
        ).hexdigest(),
        "timing": {
            "render_seconds": render_seconds,
            "r3_seconds": r3_seconds,
            "validation_and_hash_seconds": (
                time.perf_counter() - started - render_seconds - r3_seconds
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "render_execution": {
            "strategy": (
                "STATIC_IDENTICAL_POSE_SINGLE_RENDER"
                if arm.startswith("STATIC_CAMERA")
                else "PER_FRAME_RENDER"
            ),
            "render_invocations": render_invocations,
            "render_reuse_hits": render_reuse_hits,
        },
        "thread_guard": thread_guard,
        "outcome_firewall": {
            "response_values_emitted": False,
            "trigger_values_emitted": False,
            "scientific_interpretation": False,
        },
    }


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    if exclusive:
        transport.write_exclusive(path, value)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(canonical_bytes(value))
        os.replace(temporary, path)


def _process_tree_resources() -> dict[str, int]:
    processes = [psutil.Process()]
    try:
        processes.extend(processes[0].children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    resident = read_bytes = write_bytes = 0
    for process in processes:
        try:
            resident += int(process.memory_info().rss)
            io = process.io_counters()
            read_bytes += int(io.read_bytes)
            write_bytes += int(io.write_bytes)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "resident_memory_bytes": resident,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
    }


def run_profile(
    identity_lock: Path,
    output_dir: Path,
    workers: int,
    native_threads: int,
) -> dict[str, Any]:
    if workers not in WORKER_PROFILES:
        raise InvalidPreflight("WORKER_PROFILE")
    if (
        native_threads != SCHEDULER_NATIVE_THREADS[workers]
        or native_threads != NATIVE_THREADS
        or (
            workers == 8
            and THREAD_MODE != "LEGACY_UNRESTRICTED_OPENBLAS"
        )
        or (workers == 4 and THREAD_MODE != "CONTROLLED")
    ):
        raise InvalidPreflight("SCHEDULER_PROFILE")
    manifest = json.loads(identity_lock.read_text(encoding="utf-8"))
    validate_identity_manifest(manifest)
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    memory = psutil.virtual_memory()
    if memory.available < 8 * GIB:
        raise InvalidPreflight(
            f"LAUNCH_AVAILABLE_RAM_BELOW_8_GIB:{memory.available}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    started_wall = time.time()
    started = time.perf_counter()
    initial_swap = psutil.swap_memory()
    last_swap = initial_swap
    paging_positive_streak = 0
    min_available = memory.available
    max_used = 0
    completed: dict[str, dict[str, Any]] = {}
    telemetry_samples: list[dict[str, Any]] = []
    worker_pids: list[int] = []
    progress_path = output_dir / "progress.json"
    identities = manifest["identities"]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_evaluate_identity, identity): identity
            for identity in identities
        }
        worker_pids = sorted(int(pid) for pid in executor._processes)
        while futures:
            done, _ = wait(
                futures, timeout=20.0, return_when=FIRST_COMPLETED
            )
            current_memory = psutil.virtual_memory()
            current_swap = psutil.swap_memory()
            min_available = min(min_available, current_memory.available)
            max_used = max(max_used, current_memory.used)
            paging_delta = max(
                0,
                int(current_swap.sin - last_swap.sin)
                + int(current_swap.sout - last_swap.sout),
            )
            paging_positive_streak = (
                paging_positive_streak + 1 if paging_delta > 0 else 0
            )
            last_swap = current_swap
            if current_memory.available < 4 * GIB:
                for future in futures:
                    future.cancel()
                raise InvalidPreflight("RUN_AVAILABLE_RAM_BELOW_4_GIB")
            if paging_positive_streak >= 2:
                for future in futures:
                    future.cancel()
                raise InvalidPreflight("SUSTAINED_PAGING")
            for future in done:
                identity = futures.pop(future)
                receipt = future.result()
                completed[identity["sequence_id"]] = receipt
                _write_json(
                    output_dir
                    / "sequences"
                    / identity["sequence_id"]
                    / "receipt.json",
                    receipt,
                    exclusive=True,
                )
            elapsed = time.perf_counter() - started
            process_resources = _process_tree_resources()
            sample = {
                "sample_index": len(telemetry_samples),
                "elapsed_seconds": elapsed,
                "sampled_at_utc": _utc_now(),
                "available_ram_bytes": current_memory.available,
                "system_used_ram_bytes": current_memory.used,
                "resident_memory_bytes": process_resources[
                    "resident_memory_bytes"
                ],
                "read_bytes": process_resources["read_bytes"],
                "write_bytes": process_resources["write_bytes"],
                "swap_in_total": int(current_swap.sin),
                "swap_out_total": int(current_swap.sout),
                "completed_units": len(completed),
            }
            telemetry_samples.append(sample)
            progress = {
                "schema": "rcle.periodic_self_motion_counterfactual.p3_progress.v1",
                "protocol_id": PROTOCOL_ID,
                "phase": "P3_RUNTIME_PREFLIGHT",
                "completed_units": len(completed),
                "total_units": 8,
                "throughput": len(completed) / elapsed if elapsed else None,
                "eta_seconds": (
                    (8 - len(completed)) * elapsed / len(completed)
                    if completed
                    else None
                ),
                "last_progress_at": _utc_now(),
                "status": "SUCCESS" if not futures else "RUNNING",
                "completed_arms": len(completed),
                "total_arms": 8,
                "completed_pairs": len(completed) * PAIR_COUNT,
                "total_pairs": 8 * PAIR_COUNT,
                "throughput_pairs_per_s": (
                    len(completed) * PAIR_COUNT / elapsed if elapsed else None
                ),
                "eta_s": (
                    (8 - len(completed)) * elapsed / len(completed)
                    if completed
                    else None
                ),
                "worker_count": workers,
                "resident_memory_bytes": sample["resident_memory_bytes"],
                "read_bytes": sample["read_bytes"],
                "write_bytes": sample["write_bytes"],
                "last_heartbeat_utc": _utc_now(),
                "terminal_state": "SUCCESS" if not futures else "RUNNING",
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(progress_path, progress)
            _write_json(
                output_dir / "telemetry.json",
                {
                    "schema": (
                        "rcle.periodic_self_motion_counterfactual."
                        "p3_telemetry.v1"
                    ),
                    "protocol_id": PROTOCOL_ID,
                    "worker_count": workers,
                    "samples": telemetry_samples,
                    "outcome_fields_present": False,
                },
            )
    ordered = [completed[item["sequence_id"]] for item in identities]
    residual_worker_pids = [
        pid for pid in worker_pids if psutil.pid_exists(pid)
    ]
    if residual_worker_pids:
        raise InvalidPreflight(
            f"RESIDUAL_WORKERS:{','.join(map(str, residual_worker_pids))}"
        )
    elapsed = time.perf_counter() - started
    terminal = {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_profile.v1",
        "protocol_id": PROTOCOL_ID,
        "p3_id": P3_ID,
        "profile": f"W{workers}",
        "workers": workers,
        "native_threads_per_worker": native_threads,
        "total_native_thread_budget": workers * native_threads,
        "identity_lock_path": identity_lock.as_posix(),
        "identity_lock_sha256": sha256_file(identity_lock),
        "identity_set_sha256": manifest["identity_set_sha256"],
        "sequence_count": len(ordered),
        "frame_count": sum(item["frame_count"] for item in ordered),
        "pair_count": sum(item["pair_count"] for item in ordered),
        "sequence_receipts": ordered,
        "equivalence_manifest_sha256": hashlib.sha256(
            canonical_bytes(
                [
                    {
                        "sequence_id": item["sequence_id"],
                        "transport_identity_sha256": item[
                            "transport_identity_sha256"
                        ],
                    }
                    for item in ordered
                ]
            )
        ).hexdigest(),
        "resource": {
            "available_ram_at_launch_bytes": memory.available,
            "minimum_available_ram_bytes": min_available,
            "maximum_system_used_ram_bytes": max_used,
            "swap_in_delta": int(last_swap.sin - initial_swap.sin),
            "swap_out_delta": int(last_swap.sout - initial_swap.sout),
            "sustained_paging": False,
            "telemetry_sample_count": len(telemetry_samples),
            "heartbeat_max_interval_seconds": max(
                (
                    right["elapsed_seconds"] - left["elapsed_seconds"]
                    for left, right in zip(
                        telemetry_samples, telemetry_samples[1:]
                    )
                ),
                default=0.0,
            ),
        },
        "timing": {
            "started_unix_s": started_wall,
            "wall_seconds": elapsed,
            "render_seconds_sum": sum(
                item["timing"]["render_seconds"] for item in ordered
            ),
            "r3_seconds_sum": sum(
                item["timing"]["r3_seconds"] for item in ordered
            ),
            "validation_and_hash_seconds_sum": sum(
                item["timing"]["validation_and_hash_seconds"]
                for item in ordered
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "psutil": psutil.__version__,
            "platform": platform.platform(),
            "opencv_threads_per_process": (
                1
                if THREAD_MODE == "LEGACY_UNRESTRICTED_OPENBLAS"
                else native_threads
            ),
            "blas_threads_per_process": native_threads,
            "thread_caps_applied_before_numeric_imports": (
                THREAD_MODE != "LEGACY_UNRESTRICTED_OPENBLAS"
            ),
            "scheduler_amendment": (
                "USER_AUTHORIZED_2026-07-29_W8_"
                "LEGACY_UNRESTRICTED_OPENBLAS_PROFILE"
            ),
            "thread_mode": THREAD_MODE,
        },
        "terminal": "PROFILE_COMPLETE / PREFLIGHT_ONLY",
        "worker_pids": worker_pids,
        "residual_worker_pids": residual_worker_pids,
        "formal_execution_authorized": False,
        "p4_activated": False,
        "scientific_outcome_interpreted": False,
    }
    _write_json(output_dir / "success.json", terminal, exclusive=True)
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-identity-lock", type=Path)
    parser.add_argument("--identity-lock", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--native-threads", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_identity_lock is not None:
        if any(
            value is not None
            for value in (
                args.identity_lock,
                args.output_dir,
                args.workers,
                args.native_threads,
            )
        ):
            raise InvalidPreflight("IDENTITY_LOCK_MODE_EXCLUSIVE")
        transport.write_exclusive(
            args.write_identity_lock.resolve(), identity_manifest()
        )
        print(
            json.dumps(
                {
                    "identity_count": 8,
                    "identity_set_sha256": identity_manifest()[
                        "identity_set_sha256"
                    ],
                    "formal_execution_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if (
        args.identity_lock is None
        or args.output_dir is None
        or args.workers is None
        or args.native_threads is None
    ):
        raise InvalidPreflight("PROFILE_ARGUMENTS_REQUIRED")
    result = run_profile(
        args.identity_lock.resolve(),
        args.output_dir.resolve(),
        args.workers,
        args.native_threads,
    )
    print(
        json.dumps(
            {
                "profile": result["profile"],
                "terminal": result["terminal"],
                "wall_seconds": result["timing"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
