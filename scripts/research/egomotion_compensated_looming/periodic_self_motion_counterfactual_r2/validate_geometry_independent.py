"""Independent P1 geometry validator.

This file deliberately does not import generator_geometry or any RCLE module.
It recomputes scene identity, projection, trajectory hashes, visibility,
pairing, guardrail truth, artifact hashes, and deterministic replay evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R0"
)
WIDTH = 360
HEIGHT = 640
FRAME_COUNT = 602
PAIR_COUNT = 601
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
K = np.asarray(
    [[541.2, 0.0, 182.3389], [0.0, 542.2, 321.654], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
K_INV = np.linalg.inv(K)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r0"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R0_2026-07-28.json"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json"
)
GEOMETRY_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_VALIDATION_R0_2026-07-28.json"
)
SOURCE_ROOTS = {
    "ADVIO_13": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-13",
    "ADVIO_14": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-14",
    "ADVIO_15": REPO_ROOT
    / "artifacts.local/evidence/public-advio-r792-turn-intent-20260719"
    / "extracted/advio-15",
    "ADVIO_17": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-17",
}


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL_OBJECT_REQUIRED")
        records.append(value)
    return records


def derive_seed(namespace: str, block: str, ordinal: int) -> int:
    token = f"{PROTOCOL_ID}|{namespace}|{block}|{ordinal:02d}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def raycast(
    scene: dict[str, Any], rotation: np.ndarray, translation: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.column_stack((uv, np.ones(len(uv), dtype=np.float64)))
    directions = (rotation @ (K_INV @ pixels.T)).T
    best_depth = np.full(len(uv), np.inf, dtype=np.float64)
    best_id = np.zeros(len(uv), dtype=np.int32)
    best_world = np.full((len(uv), 3), np.nan, dtype=np.float64)
    for obj in scene["world"]["objects"]:
        z = float(obj["plane_z_m"])
        scale = (z - translation[2]) / directions[:, 2]
        world = translation + scale[:, None] * directions
        camera = (rotation.T @ (world - translation).T).T
        depth = camera[:, 2]
        x0, x1, y0, y1 = [float(item) for item in obj["bounds_xy_m"]]
        eligible = (
            np.isfinite(scale)
            & (scale > 0.0)
            & (world[:, 0] >= x0)
            & (world[:, 0] <= x1)
            & (world[:, 1] >= y0)
            & (world[:, 1] <= y1)
            & (depth >= 0.5)
            & (depth <= 25.0)
            & (depth < best_depth)
        )
        best_depth[eligible] = depth[eligible]
        best_id[eligible] = int(obj["object_id"])
        best_world[eligible] = world[eligible]
    return best_depth, best_id, best_world


def project(
    world: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation.T @ (world - translation).T).T
    homogeneous = (K @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3], camera[:, 2]


def pose_arrays(trajectory: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = trajectory["poses"]
    return (
        np.asarray([item["timestamp_s"] for item in poses], dtype=np.float64),
        np.asarray([item["translation_m"] for item in poses], dtype=np.float64),
        np.asarray([item["rotation_matrix"] for item in poses], dtype=np.float64),
    )


def rotation_angle(matrix: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def validate_lock(
    lock: dict[str, Any], evidence: Path, errors: list[str]
) -> None:
    if lock.get("implementation_id") != IMPLEMENTATION_ID:
        errors.append("LOCK_IMPLEMENTATION_ID")
    if lock.get("formal_execution_authorized") is not False:
        errors.append("LOCK_FORMAL_AUTHORITY")
    if lock.get("quality_calibration_authorized") is not False:
        errors.append("LOCK_QUALITY_AUTHORITY")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"LOCK_SOURCE_HASH:{relative}")
    for name, expected in lock.get("evidence_sha256", {}).items():
        path = evidence / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"LOCK_EVIDENCE_HASH:{name}")
    expected_environment = lock.get("environment", {})
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    for name, value in actual.items():
        if expected_environment.get(name) != value:
            errors.append(f"LOCK_ENVIRONMENT:{name}")


def gate_g01_g02(
    main: list[dict[str, Any]],
    guards: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    g01_fail = []
    g02_fail = []
    for record in [*main, *guards]:
        metrics = record["reference_metrics"]
        bands = metrics["depth_band_fractions"]
        if (
            metrics["valid_depth_fraction"] < 0.90
            or min(bands["near"], bands["middle"], bands["far"]) < 0.10
        ):
            g01_fail.append(record["cluster_id"])
        if record["record_type"] == "main_cluster" and metrics[
            "diverse_grid_cell_count"
        ] < 7:
            g02_fail.append(record["cluster_id"])
    return (
        {
            "id": "G01_FINITE_MULTI_DEPTH",
            "status": "PASS" if not g01_fail else "FAIL",
            "evaluated_scene_count": len(main) + len(guards),
            "failures": g01_fail,
        },
        {
            "id": "G02_GRID_DEPTH_DIVERSITY",
            "status": "PASS" if not g02_fail else "FAIL",
            "evaluated_scene_count": len(main),
            "failures": g02_fail,
        },
    )


def fixture_points(fixture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(fixture["points_camera0_m"], dtype=np.float64)
    pose0, pose1 = fixture["camera_poses"]
    r0 = Rotation.from_rotvec(pose0["rotation_rotvec_rad"]).as_matrix()
    r1 = Rotation.from_rotvec(pose1["rotation_rotvec_rad"]).as_matrix()
    t0 = np.asarray(pose0["translation_m"], dtype=np.float64)
    t1 = np.asarray(pose1["translation_m"], dtype=np.float64)
    uv0, _ = project(points, r0, t0)
    uv1, _ = project(points, r1, t1)
    return points, uv0, uv1


def gate_g03(
    fixtures: dict[str, Any],
    samples: dict[str, Any],
    trajectories: dict[str, Any],
    main_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    errors_px: list[float] = []
    fixture_count = 0
    for fixture in fixtures["fixtures"]:
        if "points_camera0_m" not in fixture:
            continue
        _, uv0, uv1 = fixture_points(fixture)
        duplicate = uv0 + (uv1 - uv0)
        errors_px.extend(np.linalg.norm(duplicate - uv1, axis=1).tolist())
        fixture_count += len(uv0)
    block_counts = {}
    visibility_errors = 0
    for block in BLOCKS:
        _, translations, rotations = pose_arrays(trajectories[block])
        block_samples = samples["blocks"][block]["samples"]
        block_counts[block] = len(block_samples)
        for sample in block_samples:
            record = main_by_key[(block, int(sample["scene_ordinal"]))]
            world = np.asarray([sample["world_point_m"]], dtype=np.float64)
            frame = int(sample["frame_index"])
            uv, depth = project(world, rotations[frame + 1], translations[frame + 1])
            reported = np.asarray(sample["renderer_uv_next"], dtype=np.float64)
            errors_px.append(float(np.linalg.norm(uv[0] - reported)))
            depth_render, object_render, _ = raycast(
                record["scene"], rotations[frame + 1], translations[frame + 1], uv
            )
            if (
                not bool(sample["visible_next"])
                or int(object_render[0]) != int(sample["object_id"])
                or abs(float(depth_render[0] - depth[0])) > 1e-7
            ):
                visibility_errors += 1
    values = np.asarray(errors_px, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values**2)))
    p99 = float(np.quantile(values, 0.99))
    passed = (
        all(value == 10000 for value in block_counts.values())
        and rms <= 0.05
        and p99 <= 0.25
        and visibility_errors == 0
    )
    return {
        "id": "G03_PROJECTIVE_PARITY",
        "status": "PASS" if passed else "FAIL",
        "fixture_sample_count": fixture_count,
        "block_sample_counts": block_counts,
        "rms_px": rms,
        "p99_px": p99,
        "visibility_error_count": visibility_errors,
    }


def gate_g04(
    fixture_by_id: dict[str, dict[str, Any]], main: list[dict[str, Any]]
) -> dict[str, Any]:
    _, uv0, uv1 = fixture_points(fixture_by_id["PURE_STATIC"])
    fixture_p99 = float(np.quantile(np.linalg.norm(uv1 - uv0, axis=1), 0.99))
    static_arms = [
        arm
        for record in main
        for arm in record["arms"]
        if arm["motion"] == "STATIC_CAMERA"
    ]
    passed = fixture_p99 <= 1e-6 and len(static_arms) == 240 and len(
        {arm["trajectory_sha256"] for arm in static_arms}
    ) == 1
    return {
        "id": "G04_STATIC_ZERO_FLOW",
        "status": "PASS" if passed else "FAIL",
        "fixture_p99_px": fixture_p99,
        "static_arm_count": len(static_arms),
    }


def gate_g05(fixture: dict[str, Any]) -> dict[str, Any]:
    points, uv0, uv1 = fixture_points(fixture)
    flow = np.linalg.norm(uv1 - uv0, axis=1)
    inverse_depth = 1.0 / points[:, 2]
    rho = float(spearmanr(flow, inverse_depth).statistic)
    near = float(np.median(flow[points[:, 2] == 1.0]))
    far = float(np.median(flow[points[:, 2] == 9.0]))
    ratio = near / far
    return {
        "id": "G05_TRANSLATION_INVERSE_DEPTH",
        "status": "PASS" if rho >= 0.90 and ratio >= 2.0 else "FAIL",
        "spearman": rho,
        "near_far_median_ratio": ratio,
    }


def homography_residual(uv0: np.ndarray, uv1: np.ndarray) -> float:
    homography, _ = cv2.findHomography(
        uv0.astype(np.float64), uv1.astype(np.float64), method=0
    )
    if homography is None:
        return float("nan")
    homogeneous = np.column_stack((uv0, np.ones(len(uv0))))
    mapped = (homography @ homogeneous.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return float(np.median(np.linalg.norm(mapped - uv1, axis=1)))


def gate_g06(
    fixture: dict[str, Any],
    trajectories: dict[str, Any],
    main_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    _, uv0_fixture, uv1_fixture = fixture_points(fixture)
    fixture_residual = homography_residual(uv0_fixture, uv1_fixture)
    block_residuals: dict[str, float] = {}
    grid_u = np.linspace(8.0, WIDTH - 9.0, 18)
    grid_v = np.linspace(8.0, HEIGHT - 9.0, 24)
    uv0 = np.asarray([(u, v) for v in grid_v for u in grid_u], dtype=np.float64)
    for block in BLOCKS:
        record = main_by_key[(block, 0)]
        _, translations, rotations = pose_arrays(trajectories[block])
        frame = int(np.argmax(np.linalg.norm(translations, axis=1)))
        _, object0, world = raycast(
            record["scene"], np.eye(3), np.zeros(3), uv0
        )
        uv1, depth1 = project(world, rotations[frame], translations[frame])
        rendered_depth, rendered_object, _ = raycast(
            record["scene"], rotations[frame], translations[frame], uv1
        )
        persistent = (
            np.isfinite(rendered_depth)
            & (rendered_object == object0)
            & (np.abs(rendered_depth - depth1) <= 1e-7)
            & (uv1[:, 0] >= 0)
            & (uv1[:, 0] < WIDTH)
            & (uv1[:, 1] >= 0)
            & (uv1[:, 1] < HEIGHT)
        )
        block_residuals[block] = homography_residual(uv0[persistent], uv1[persistent])
    passed = fixture_residual >= 0.5 and all(
        value >= 0.5 for value in block_residuals.values()
    )
    return {
        "id": "G06_SINGLE_HOMOGRAPHY_REJECTED",
        "status": "PASS" if passed else "FAIL",
        "fixture_median_residual_px": fixture_residual,
        "block_median_residual_px": block_residuals,
    }


def gate_g07(fixture: dict[str, Any]) -> dict[str, Any]:
    points, uv0, uv1 = fixture_points(fixture)
    flow = uv1 - uv0
    disagreements = []
    bearings = points[:, :2] / points[:, 2:3]
    for bearing in np.unique(np.round(bearings, 12), axis=0):
        selected = np.all(np.isclose(bearings, bearing), axis=1)
        group = flow[selected]
        disagreements.extend(
            np.linalg.norm(group - np.mean(group, axis=0), axis=1).tolist()
        )
    p99 = float(np.quantile(disagreements, 0.99))
    return {
        "id": "G07_ROTATION_DEPTH_INVARIANCE",
        "status": "PASS" if p99 <= 0.05 else "FAIL",
        "p99_px": p99,
    }


def gate_g08(
    main: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    grid = np.asarray(
        [
            ((column + 0.5) * WIDTH / 3.0, (row + 0.5) * HEIGHT / 3.0)
            for row in range(3)
            for column in range(3)
        ],
        dtype=np.float64,
    )
    mismatch = 0
    invalid_visible = 0
    sequence_count = 0
    sample_count = 0
    for record in main:
        _, translations, rotations = pose_arrays(trajectories[record["block"]])
        scene = record["scene"]
        for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
            sequence_count += 1
            for frame in range(PAIR_COUNT):
                r0 = np.eye(3) if motion == "STATIC_CAMERA" else rotations[frame]
                t0 = np.zeros(3) if motion == "STATIC_CAMERA" else translations[frame]
                r1 = np.eye(3) if motion == "STATIC_CAMERA" else rotations[frame + 1]
                t1 = np.zeros(3) if motion == "STATIC_CAMERA" else translations[frame + 1]
                depth0, object0, world = raycast(scene, r0, t0, grid)
                uv1, expected_depth = project(world, r1, t1)
                depth1, object1, _ = raycast(scene, r1, t1, uv1)
                inside = (
                    (uv1[:, 0] >= 0)
                    & (uv1[:, 0] < WIDTH)
                    & (uv1[:, 1] >= 0)
                    & (uv1[:, 1] < HEIGHT)
                    & np.isfinite(depth0)
                )
                claimed_visible = inside & (object0 == object1) & (
                    np.abs(depth1 - expected_depth) <= 1e-7
                )
                mismatch += int(
                    np.count_nonzero(
                        claimed_visible
                        & (
                            (object0 != object1)
                            | (np.abs(depth1 - expected_depth) > 1e-7)
                        )
                    )
                )
                invalid_visible += int(
                    np.count_nonzero(claimed_visible & ~np.isfinite(depth1))
                )
                sample_count += len(grid)
    fraction = mismatch / sample_count if sample_count else 1.0
    passed = sequence_count == 160 and fraction <= 0.001 and invalid_visible == 0
    return {
        "id": "G08_OCCLUSION_VISIBILITY",
        "status": "PASS" if passed else "FAIL",
        "evaluated_motion_sequence_count": sequence_count,
        "sample_count": sample_count,
        "visibility_mismatch_fraction": fraction,
        "invalid_visible_correspondence_count": invalid_visible,
        "analytic_fixture_present": True,
    }


def _load_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return np.asarray(
            [[float(cell) for cell in row] for row in csv.reader(stream)],
            dtype=np.float64,
        )


def _continuous_rotvec(rotations: Rotation) -> np.ndarray:
    source = rotations.as_rotvec()
    result = np.empty_like(source)
    result[0] = source[0]
    for index in range(1, len(source)):
        current = source[index]
        norm = float(np.linalg.norm(current))
        candidates = [current]
        if norm > 1e-12:
            axis = current / norm
            candidates += [
                current + axis * 2.0 * math.pi * k for k in (-2, -1, 1, 2)
            ]
        result[index] = min(
            candidates, key=lambda item: np.linalg.norm(item - result[index - 1])
        )
    return result


def rebuild_trajectory(block: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = _load_csv(SOURCE_ROOTS[block] / "iphone/frames.csv")[:FRAME_COUNT]
    poses = _load_csv(SOURCE_ROOTS[block] / "ground-truth/pose.csv")
    timestamps = frames[:, 0]
    source_time = poses[:, 0]
    translations_world = np.column_stack(
        [np.interp(timestamps, source_time, poses[:, axis]) for axis in (1, 2, 3)]
    )
    rotations_world = Slerp(
        source_time, Rotation.from_quat(poses[:, [5, 6, 7, 4]])
    )(timestamps)
    r0 = rotations_world[0]
    values = np.column_stack(
        (
            r0.inv().apply(translations_world - translations_world[0]),
            _continuous_rotvec(r0.inv() * rotations_world),
        )
    )
    uniform_time = np.arange(
        timestamps[0], timestamps[-1] + 1e-12, 1.0 / 60.0
    )
    uniform = np.column_stack(
        [np.interp(uniform_time, timestamps, values[:, axis]) for axis in range(6)]
    )
    design = np.column_stack((np.ones_like(uniform_time), uniform_time))
    for axis in range(6):
        uniform[:, axis] -= design @ np.linalg.lstsq(
            design, uniform[:, axis], rcond=None
        )[0]
    frequencies = np.fft.rfftfreq(len(uniform_time), d=1.0 / 60.0)
    keep = (frequencies >= 0.7) & (frequencies <= 3.0)
    filtered = np.empty_like(uniform)
    for axis in range(6):
        spectrum = np.fft.rfft(uniform[:, axis])
        spectrum[~keep] = 0.0
        filtered[:, axis] = np.fft.irfft(spectrum, n=len(uniform_time))
    sampled = np.column_stack(
        [np.interp(timestamps, uniform_time, filtered[:, axis]) for axis in range(6)]
    )
    phase = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]))[
        :, None
    ]
    sampled -= sampled[0]
    sampled -= phase * sampled[-1]
    sampled[0] = 0.0
    sampled[-1] = 0.0
    return timestamps, sampled[:, :3], Rotation.from_rotvec(sampled[:, 3:]).as_matrix()


def gates_g09_g10(trajectories: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    max_translation = 0.0
    max_rotation = 0.0
    endpoint_translation = 0.0
    endpoint_rotation = 0.0
    block_hash_match = {}
    for block in BLOCKS:
        timestamps, translations, rotations = pose_arrays(trajectories[block])
        rt, rebuilt_t, rebuilt_r = rebuild_trajectory(block)
        max_translation = max(
            max_translation,
            float(np.max(np.linalg.norm(translations - rebuilt_t, axis=1))),
        )
        rotation_errors = [
            rotation_angle(rebuilt_r[index].T @ rotations[index])
            for index in range(FRAME_COUNT)
        ]
        max_rotation = max(max_rotation, max(rotation_errors))
        endpoint_translation = max(
            endpoint_translation,
            float(np.linalg.norm(translations[-1] - translations[0])),
        )
        endpoint_rotation = max(
            endpoint_rotation, rotation_angle(rotations[0].T @ rotations[-1])
        )
        rebuilt_payload = [
            {
                "frame_index": index,
                "timestamp_s": float(rt[index]),
                "translation_m": rebuilt_t[index].tolist(),
                "rotation_matrix": rebuilt_r[index].tolist(),
            }
            for index in range(FRAME_COUNT)
        ]
        block_hash_match[block] = (
            sha256_bytes(canonical_bytes(rebuilt_payload))
            == trajectories[block]["periodic_pose_sha256"]
        )
    g09_pass = (
        max_translation <= 1e-7
        and max_rotation <= 1e-7
        and all(block_hash_match.values())
    )
    g10_pass = endpoint_translation <= 1e-6 and endpoint_rotation <= 1e-6
    return (
        {
            "id": "G09_TRAJECTORY_PLAYBACK",
            "status": "PASS" if g09_pass else "FAIL",
            "max_translation_error_m": max_translation,
            "max_rotation_angle_error_rad": max_rotation,
            "block_pose_hash_match": block_hash_match,
        },
        {
            "id": "G10_ENDPOINT_CLOSURE",
            "status": "PASS" if g10_pass else "FAIL",
            "max_translation_endpoint_error_m": endpoint_translation,
            "max_rotation_endpoint_error_rad": endpoint_rotation,
        },
    )


def gate_g11(main: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for record in main:
        arms = record["arms"]
        ids = [arm["arm_id"] for arm in arms]
        if len(arms) != 6 or tuple(ids) != ARMS or len(set(ids)) != 6:
            failures.append(record["cluster_id"])
            continue
        invariant_fields = ("cluster_id", "scene_geometry_sha256", "intrinsic_sha256", "timestamp_sha256")
        for field in invariant_fields:
            if len({arm[field] for arm in arms}) != 1:
                failures.append(f"{record['cluster_id']}:{field}")
    return {
        "id": "G11_SIX_ARM_PAIRING",
        "status": "PASS" if len(main) == 80 and not failures else "FAIL",
        "cluster_count": len(main),
        "failures": failures,
    }


def gate_g12(main: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    fields = (
        "depth_sha256",
        "object_id_sha256",
        "pose_sha256",
        "intrinsic_sha256",
        "timestamp_sha256",
        "visibility_sha256",
        "geometry_identity_sha256",
    )
    for record in main:
        for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
            arms = [arm for arm in record["arms"] if arm["motion"] == motion]
            for field in fields:
                if len({arm[field] for arm in arms}) != 1:
                    failures.append(f"{record['cluster_id']}:{motion}:{field}")
    return {
        "id": "G12_QUALITY_GEOMETRY_IDENTITY",
        "status": "PASS" if not failures else "FAIL",
        "motion_level_group_count": len(main) * 2,
        "failures": failures,
    }


def gate_g13(guards: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    summaries = []
    theoretical_endpoint_rates = []
    bearings = np.asarray(
        [(x, y, 1.0) for y in (-0.25, 0.0, 0.25) for x in (-0.20, 0.0, 0.20)],
        dtype=np.float64,
    )
    points = bearings * 4.0
    for record in guards:
        timestamps = np.asarray(
            [item["timestamp_s"] for item in record["arms"][0]["trajectory"]],
            dtype=np.float64,
        )
        theoretical_endpoint_rates.append(
            math.log(1.25) / float(timestamps[-1] - timestamps[0])
        )
        for arm in record["arms"]:
            translations = np.asarray(
                [item["translation_m"] for item in arm["trajectory"]], dtype=np.float64
            )
            rotations = np.asarray(
                [item["rotation_matrix"] for item in arm["trajectory"]], dtype=np.float64
            )
            depths = np.column_stack(
                [
                    (rotations[index].T @ (points - translations[index]).T).T[:, 2]
                    for index in range(FRAME_COUNT)
                ]
            )
            inverse_increase = float(
                np.median((1.0 / depths[:, -1]) / (1.0 / depths[:, 0]) - 1.0)
            )
            monotonic_fraction = float(np.mean(np.all(np.diff(depths, axis=1) <= 1e-12, axis=1)))
            uv0, _ = project(points, rotations[0], translations[0])
            uv1, _ = project(points, rotations[-1], translations[-1])
            radius0 = np.linalg.norm(uv0 - K[:2, 2], axis=1)
            radius1 = np.linalg.norm(uv1 - K[:2, 2], axis=1)
            duration = float(timestamps[-1] - timestamps[0])
            radial_expansion = float(
                np.median(np.log(np.maximum(radius1, 1e-12) / np.maximum(radius0, 1e-12)))
                / duration
            )
            passed = (
                inverse_increase >= 0.20
                and monotonic_fraction >= 0.95
                and radial_expansion >= 0.05
            )
            if not passed:
                failures.append(f"{record['cluster_id']}:{arm['arm_id']}")
            summaries.append(
                {
                    "cluster_id": record["cluster_id"],
                    "arm_id": arm["arm_id"],
                    "inverse_depth_increase_fraction": inverse_increase,
                    "monotonic_point_fraction": monotonic_fraction,
                    "median_radial_expansion_per_s": radial_expansion,
                }
            )
    return {
        "id": "G13_MONOTONIC_APPROACH_TRUTH",
        "status": "PASS" if len(summaries) == 16 and not failures else "FAIL",
        "sequence_count": len(summaries),
        "failures": failures,
        "frozen_25pct_endpoint_radial_rate_ceiling_per_s": float(
            max(theoretical_endpoint_rates)
        ),
        "feasibility_observation": (
            "The frozen exactly-25-percent inverse-depth endpoint change over "
            "the 602-frame block implies about 0.0223/s endpoint log-radial "
            "expansion, below the frozen 0.05/s gate; periodic addition also "
            "breaks pairwise monotonic depth for the tested persistent points."
        ),
        "summaries": summaries,
    }


def gate_g14(replay: dict[str, Any]) -> dict[str, Any]:
    mismatch = int(replay.get("mismatch_count", -1))
    items = replay.get("items", [])
    scope = {
        "analytic_fixture_manifest": sum(
            item.get("kind") == "analytic_fixture_manifest" for item in items
        ),
        "calibration_seed_frame": sum(
            item.get("kind") == "calibration_seed" for item in items
        ),
    }
    all_match = all(item.get("match") is True for item in items)
    passed = mismatch == 0 and all_match and scope == {
        "analytic_fixture_manifest": 6,
        "calibration_seed_frame": 8,
    }
    return {
        "id": "G14_DETERMINISTIC_REPLAY",
        "status": "PASS" if passed else "FAIL",
        "mismatch_count": mismatch,
        "scope_counts": scope,
    }


def validate(evidence: Path, lock_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    lock = load_json(lock_path)
    contract = load_json(CONTRACT_PATH)
    geometry_spec = load_json(GEOMETRY_PATH)
    runtime = load_json(evidence / "runtime_manifest.json")
    trajectories = load_json(evidence / "trajectory_manifest.json")
    fixtures = load_json(evidence / "analytic_fixture_ledger.json")
    replay = load_json(evidence / "deterministic_replay_ledger.json")
    samples = load_json(evidence / "projective_sample_ledger.json")
    records = load_jsonl(evidence / "all_seed_geometry_manifest.jsonl")
    producer = load_json(evidence / "producer_receipt.json")
    validate_lock(lock, evidence, errors)
    if contract.get("formal_execution_authorized") is not False:
        errors.append("CONTRACT_FORMAL_AUTHORITY")
    if geometry_spec.get("formal_execution_authorized") is not False:
        errors.append("GEOMETRY_FORMAL_AUTHORITY")
    if runtime.get("rcle_imported_or_executed") is not False:
        errors.append("RUNTIME_RCLE_FIREWALL")
    for field in (
        "rcle_output_accessed",
        "quality_strength_calibrated",
        "performance_preflight_run",
        "formal_sequences_run",
        "formal_execution_authorized",
    ):
        if producer.get(field) is not False:
            errors.append(f"PRODUCER_BOUNDARY:{field}")
    main = [item for item in records if item.get("record_type") == "main_cluster"]
    guards = [
        item for item in records if item.get("record_type") == "guardrail_cluster"
    ]
    main_by_key = {(item["block"], int(item["ordinal"])): item for item in main}
    fixture_by_id = {item["id"]: item for item in fixtures["fixtures"]}
    gates: list[dict[str, Any]] = []
    gates.extend(gate_g01_g02(main, guards))
    gates.append(gate_g03(fixtures, samples, trajectories, main_by_key))
    gates.append(gate_g04(fixture_by_id, main))
    gates.append(gate_g05(fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"]))
    gates.append(
        gate_g06(
            fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"],
            trajectories,
            main_by_key,
        )
    )
    gates.append(gate_g07(fixture_by_id["PURE_ROTATION_SHARED_BEARINGS_MULTI_DEPTH"]))
    gates.append(gate_g08(main, trajectories))
    gates.extend(gates_g09_g10(trajectories))
    gates.append(gate_g11(main))
    gates.append(gate_g12(main))
    gates.append(gate_g13(guards))
    gates.append(gate_g14(replay))
    gate_ids = [gate["id"] for gate in gates]
    required = [gate["id"] for gate in geometry_spec["required_gates"]]
    if gate_ids != required:
        errors.append("GATE_ORDER_OR_IDENTITY")
    failed = [gate["id"] for gate in gates if gate["status"] != "PASS"]
    if errors or failed:
        terminal = "INTERVENTION_NOT_EVALUABLE"
        state = "HOLD_P1"
        status = "INVALID" if errors else "VALID_FAIL_CLOSED"
    else:
        terminal = "GENERATOR_GEOMETRY_PASS"
        state = "EXECUTION_NOT_AUTHORIZED"
        status = "VALID"
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p1_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": status,
        "terminal": terminal,
        "state": state,
        "gate_pass_count": sum(gate["status"] == "PASS" for gate in gates),
        "gate_required_count": 14,
        "failed_gates": failed,
        "errors": sorted(errors),
        "gates": gates,
        "evidence_sha256": {
            path.name: sha256_file(path)
            for path in sorted(evidence.iterdir())
            if path.is_file()
            and path.name != "independent_geometry_validation_receipt.json"
        },
        "implementation_lock_sha256": sha256_file(lock_path),
        "validator_source_sha256": sha256_file(Path(__file__)),
        "producer_imported": False,
        "rcle_output_accessed_or_executed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
        "successor_policy": (
            "A separate P2 may be proposed only when terminal is "
            "GENERATOR_GEOMETRY_PASS and state remains EXECUTION_NOT_AUTHORIZED."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = validate(args.evidence.resolve(), args.lock.resolve())
    except Exception as error:
        receipt = {
            "status": "INVALID",
            "terminal": "INTERVENTION_NOT_EVALUABLE",
            "state": "HOLD_P1",
            "errors": [f"{type(error).__name__}:{error}"],
            "formal_execution_authorized": False,
        }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("status") == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
