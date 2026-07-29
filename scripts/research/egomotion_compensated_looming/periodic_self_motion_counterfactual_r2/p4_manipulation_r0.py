"""Response-blind P4-A manipulation check for all 80 formal MAIN clusters.

This module deliberately imports only the frozen generator and response-blind
quality operators.  In particular, it must remain importable without importing
the R3 pair core, P3 transport, or any outcome-analysis module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from . import generator_geometry as geometry
from . import quality_interventions_r0 as quality


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
FRAME_POSITIONS = (
    0, 40, 80, 120, 160, 200, 240, 280,
    320, 360, 400, 440, 480, 520, 560, 601,
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
BLUR_SIGMA_PX = 0.475
LOW_TEXTURE_ALPHA = 0.15
BLUR_LAPLACIAN_RANGE = (0.35, 0.55)
BLUR_LOCAL_RMS_MINIMUM = 0.70
LOW_TEXTURE_GRADIENT_RANGE = (0.35, 0.55)
SUBGROUP_MINIMUM_PASS = 18
SUBGROUP_SEQUENCE_COUNT = 20


class InvalidManipulation(ValueError):
    """Raised when P4-A cannot produce a signable response-blind receipt."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
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


def _atomic_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise InvalidManipulation(f"EXCLUSIVE_PATH_EXISTS:{path.name}")
    handle, literal = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(literal)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise InvalidManipulation(f"EXCLUSIVE_PATH_EXISTS:{path.name}")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _activation_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("formal_execution_authorized") is not True:
        raise InvalidManipulation("ACTIVATION_NOT_AUTHORIZED")
    if value.get("p4_activated") is not True:
        raise InvalidManipulation("P4_NOT_ACTIVATED")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise InvalidManipulation("ACTIVATION_PROTOCOL")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise InvalidManipulation("ACTIVATION_BINDINGS")
    own_relative = Path(__file__).resolve().relative_to(repo_root.resolve()).as_posix()
    seen_paths: set[str] = set()
    own_bound = False
    for binding in bindings:
        if not isinstance(binding, dict):
            raise InvalidManipulation("ACTIVATION_BINDING_OBJECT")
        relative = binding.get("path")
        expected = binding.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or relative in seen_paths
        ):
            raise InvalidManipulation("ACTIVATION_BINDING_INVALID")
        seen_paths.add(relative)
        resolved = (repo_root / relative).resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError as error:
            raise InvalidManipulation("ACTIVATION_BINDING_OUTSIDE") from error
        if not resolved.is_file() or sha256_file(resolved) != expected:
            raise InvalidManipulation(f"ACTIVATION_BINDING_DRIFT:{relative}")
        own_bound = own_bound or relative.replace("\\", "/") == own_relative
    if not own_bound:
        raise InvalidManipulation("MANIPULATION_IMPLEMENTATION_NOT_BOUND")
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def freeze_main_scene_manifest(
    all_seed_manifest_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in all_seed_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    main = [row for row in rows if row.get("record_type") == "main_cluster"]
    if len(main) != 80:
        raise InvalidManipulation(f"MAIN_CLUSTER_COUNT:{len(main)}")
    identities = []
    for row in main:
        if row.get("block") not in BLOCKS or int(row.get("ordinal", -1)) not in range(20):
            raise InvalidManipulation("MAIN_CLUSTER_IDENTITY")
        identities.append(
            {
                "block": row["block"],
                "ordinal": int(row["ordinal"]),
                "cluster_id": row["cluster_id"],
                "numeric_seed_uint64": int(row["numeric_seed_uint64"]),
                "scene_geometry_sha256": row["scene"]["scene_geometry_sha256"],
                "scene": row["scene"],
            }
        )
    identities.sort(key=lambda item: (BLOCKS.index(item["block"]), item["ordinal"]))
    if len({item["cluster_id"] for item in identities}) != 80:
        raise InvalidManipulation("DUPLICATE_MAIN_CLUSTER")
    payload = b"".join(canonical_bytes(item) for item in identities)
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise InvalidManipulation("FROZEN_MAIN_SCENE_MANIFEST_DRIFT")
    else:
        _atomic_write(output_path, payload, exclusive=True)
    return identities


def _render_clean_and_low(
    scene: dict[str, Any],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Render clean and the frozen pre-quantization albedo contraction together."""

    u, v = np.meshgrid(
        np.arange(geometry.WIDTH, dtype=np.float64),
        np.arange(geometry.HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    depth, object_id, world = geometry._raycast(scene, rotation, translation, uv)
    valid = np.isfinite(depth)
    clean_linear = np.zeros((len(uv), 3), dtype=np.float64)
    low_linear = np.zeros((len(uv), 3), dtype=np.float64)
    by_id = {int(item["object_id"]): item for item in scene["world"]["objects"]}
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
        low_modulation = 0.825 + LOW_TEXTURE_ALPHA * (
            clean_modulation - 0.825
        )
        clean_linear[selected] = np.clip(
            base[None, :] * clean_modulation[:, None], 0.0, 1.0
        )
        low_linear[selected] = np.clip(
            base[None, :] * low_modulation[:, None], 0.0, 1.0
        )
    clean_rgb = quality.linear_to_srgb_u8(
        clean_linear.reshape(geometry.HEIGHT, geometry.WIDTH, 3)
    )
    low_rgb = quality.linear_to_srgb_u8(
        low_linear.reshape(geometry.HEIGHT, geometry.WIDTH, 3)
    )
    return (
        {
            "rgb": clean_rgb,
            "valid_mask": valid.reshape(geometry.HEIGHT, geometry.WIDTH),
        },
        low_rgb,
    )


def _pose_for(
    trajectory: dict[str, Any],
    motion: str,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if motion == "STATIC_CAMERA":
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    pose = trajectory["poses"][frame_index]
    if int(pose["frame_index"]) != frame_index:
        raise InvalidManipulation("TRAJECTORY_FRAME_ORDER")
    return (
        np.asarray(pose["rotation_matrix"], dtype=np.float64),
        np.asarray(pose["translation_m"], dtype=np.float64),
    )


def _median(values: list[float]) -> float:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise InvalidManipulation("SEQUENCE_METRIC_CARDINALITY_OR_FINITE")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def evaluate_sequence(
    identity: dict[str, Any],
    motion: str,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    if motion not in MOTIONS:
        raise InvalidManipulation("MOTION")
    frame_rows: list[dict[str, Any]] = []
    static_cache: tuple[dict[str, Any], np.ndarray] | None = None
    for frame_index in FRAME_POSITIONS:
        rotation, translation = _pose_for(trajectory, motion, frame_index)
        if motion == "STATIC_CAMERA" and static_cache is not None:
            clean, low_rgb = static_cache
        else:
            clean, low_rgb = _render_clean_and_low(
                identity["scene"], rotation, translation
            )
            if motion == "STATIC_CAMERA":
                static_cache = (clean, low_rgb)
        blur_rgb = quality.apply_blur(clean["rgb"], BLUR_SIGMA_PX)
        clean_y = quality.linear_luminance(clean["rgb"])
        blur_y = quality.linear_luminance(blur_rgb)
        low_y = quality.linear_luminance(low_rgb)
        valid = clean["valid_mask"]
        clean_lap = quality.variance_of_laplacian(clean_y, valid)
        clean_rms = quality.local_rms_contrast(clean_y, valid)
        blur_lap = quality.variance_of_laplacian(blur_y, valid)
        blur_rms = quality.local_rms_contrast(blur_y, valid)
        clean_gradient, low_gradient = quality.multiscale_gradient_density_pair(
            clean_y, low_y, valid
        )
        frame_rows.append(
            {
                "frame_index": frame_index,
                "clean_rgb_sha256": hashlib.sha256(clean["rgb"].tobytes()).hexdigest(),
                "blur_rgb_sha256": hashlib.sha256(blur_rgb.tobytes()).hexdigest(),
                "low_texture_rgb_sha256": hashlib.sha256(low_rgb.tobytes()).hexdigest(),
                "valid_mask_sha256": hashlib.sha256(
                    valid.astype(np.uint8).tobytes()
                ).hexdigest(),
                "laplacian_variance_ratio": quality.safe_ratio(
                    blur_lap, clean_lap, "FORMAL_MAIN_LAPLACIAN"
                ),
                "local_rms_contrast_ratio": quality.safe_ratio(
                    blur_rms, clean_rms, "FORMAL_MAIN_RMS"
                ),
                "multiscale_gradient_density_ratio": quality.safe_ratio(
                    low_gradient, clean_gradient, "FORMAL_MAIN_GRADIENT"
                ),
            }
        )
    blur_lap_median = _median(
        [row["laplacian_variance_ratio"] for row in frame_rows]
    )
    blur_rms_median = _median(
        [row["local_rms_contrast_ratio"] for row in frame_rows]
    )
    low_gradient_median = _median(
        [row["multiscale_gradient_density_ratio"] for row in frame_rows]
    )
    blur_pass = (
        BLUR_LAPLACIAN_RANGE[0]
        <= blur_lap_median
        <= BLUR_LAPLACIAN_RANGE[1]
        and blur_rms_median >= BLUR_LOCAL_RMS_MINIMUM
    )
    low_pass = (
        LOW_TEXTURE_GRADIENT_RANGE[0]
        <= low_gradient_median
        <= LOW_TEXTURE_GRADIENT_RANGE[1]
    )
    return {
        "block": identity["block"],
        "ordinal": identity["ordinal"],
        "cluster_id": identity["cluster_id"],
        "numeric_seed_uint64": identity["numeric_seed_uint64"],
        "scene_geometry_sha256": identity["scene_geometry_sha256"],
        "motion": motion,
        "frame_positions": list(FRAME_POSITIONS),
        "frame_rows": frame_rows,
        "sequence_medians": {
            "laplacian_variance_ratio": blur_lap_median,
            "local_rms_contrast_ratio": blur_rms_median,
            "multiscale_gradient_density_ratio": low_gradient_median,
        },
        "blur_sequence_pass": blur_pass,
        "low_texture_sequence_pass": low_pass,
    }


def _worker(task: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    return evaluate_sequence(*task)


def _subgroups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                row for row in rows
                if row["block"] == block and row["motion"] == motion
            ]
            if len(selected) != SUBGROUP_SEQUENCE_COUNT:
                raise InvalidManipulation("SUBGROUP_SEQUENCE_COUNT")
            blur_count = sum(bool(row["blur_sequence_pass"]) for row in selected)
            low_count = sum(
                bool(row["low_texture_sequence_pass"]) for row in selected
            )
            result.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": len(selected),
                    "blur_pass_count": blur_count,
                    "low_texture_pass_count": low_count,
                    "blur_subgroup_pass": blur_count >= SUBGROUP_MINIMUM_PASS,
                    "low_texture_subgroup_pass": low_count >= SUBGROUP_MINIMUM_PASS,
                }
            )
    return result


def run(
    *,
    repo_root: Path,
    activation_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers != 8:
        raise InvalidManipulation("FORMAL_MANIPULATION_REQUIRES_W8")
    activation = _activation_identity(activation_path, repo_root)
    p1_dir = (
        repo_root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0"
    )
    all_seed = p1_dir / "all_seed_geometry_manifest.jsonl"
    trajectory_path = p1_dir / "trajectory_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    identities = freeze_main_scene_manifest(
        all_seed, output_dir / "formal_main_scene_manifest.jsonl"
    )
    trajectories = _load_json(trajectory_path)
    tasks = [
        (identity, motion, trajectories[identity["block"]])
        for identity in identities
        for motion in MOTIONS
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(tasks)
        futures = {}
        for _ in range(workers):
            try:
                task = next(iterator)
            except StopIteration:
                break
            futures[pool.submit(_worker, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                rows.append(future.result())
                try:
                    task = next(iterator)
                except StopIteration:
                    continue
                futures[pool.submit(_worker, task)] = task
    rows.sort(
        key=lambda row: (
            BLOCKS.index(row["block"]),
            int(row["ordinal"]),
            MOTIONS.index(row["motion"]),
        )
    )
    if len(rows) != 160:
        raise InvalidManipulation("SEQUENCE_RESULT_COUNT")
    ledger_path = output_dir / "formal_manipulation_ledger.jsonl"
    ledger = b"".join(canonical_bytes(row) for row in rows)
    _atomic_write(ledger_path, ledger, exclusive=True)
    subgroups = _subgroups(rows)
    terminal = (
        "PASS"
        if all(
            row["blur_subgroup_pass"] and row["low_texture_subgroup_pass"]
            for row in subgroups
        )
        else "INTERVENTION_NOT_EVALUABLE"
    )
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.p4_manipulation.v1",
        "protocol_id": PROTOCOL_ID,
        "phase": "P4_A_RESPONSE_BLIND_FORMAL_MANIPULATION",
        "terminal": terminal,
        "r3_imported_or_executed": False,
        "algorithm_output_read": False,
        "activation": activation,
        "strengths": {
            "blur_sigma_px": BLUR_SIGMA_PX,
            "low_texture_alpha": LOW_TEXTURE_ALPHA,
            "low_texture_psf_operator": "NONE",
        },
        "gates": {
            "blur_laplacian_ratio_inclusive": list(BLUR_LAPLACIAN_RANGE),
            "blur_local_rms_ratio_minimum": BLUR_LOCAL_RMS_MINIMUM,
            "low_texture_gradient_ratio_inclusive": list(
                LOW_TEXTURE_GRADIENT_RANGE
            ),
            "required_pass_per_20": SUBGROUP_MINIMUM_PASS,
        },
        "counts": {
            "main_clusters": 80,
            "motion_levels": 2,
            "sequence_checks": 160,
            "frame_states_per_sequence": 16,
            "total_frame_states": 2560,
        },
        "subgroups": subgroups,
        "evidence": {
            "all_seed_geometry_manifest_sha256": sha256_file(all_seed),
            "trajectory_manifest_sha256": sha256_file(trajectory_path),
            "formal_main_scene_manifest_sha256": sha256_file(
                output_dir / "formal_main_scene_manifest.jsonl"
            ),
            "formal_manipulation_ledger_sha256": sha256_file(ledger_path),
            "producer_sha256": sha256_file(Path(__file__)),
            "quality_operator_sha256": sha256_file(Path(quality.__file__)),
            "generator_sha256": sha256_file(Path(geometry.__file__)),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = output_dir / "formal_manipulation_receipt.json"
    _atomic_write(receipt_path, canonical_bytes(receipt), exclusive=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    receipt = run(
        repo_root=arguments.repo_root.resolve(),
        activation_path=arguments.activation.resolve(),
        output_dir=arguments.output_dir.resolve(),
        workers=arguments.workers,
    )
    print(json.dumps({"terminal": receipt["terminal"]}, sort_keys=True))
    return 0 if receipt["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
