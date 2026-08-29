from __future__ import annotations

"""Truth-separated CARLA V18 adapter for the source-neutral X21/X22 pipeline.

Model extraction opens only ``joined-v18/model`` plus RGB, depth, and optical
flow payloads.  The separate evaluator loader is the only function permitted
to open ``joined-v18/evaluator``.  ``points_lidar`` keeps the established X21
field name, but contains depth-backprojected camera points in forward-left-up
(FLH) coordinates rather than physical LiDAR returns.
"""

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


MODEL_MANIFEST_RELATIVE = Path("joined-v18/model/manifest.json")
EVALUATOR_MANIFEST_RELATIVE = Path("joined-v18/evaluator/manifest.json")
INPUT_SCHEMA = "blindassist-dtr-carla-v18-causal-input-v1"
DESCRIPTOR_SCHEMA = "blindassist-dtr-carla-v18-descriptor-v1"
MODEL_TOPICS = ("RGB", "depth", "optical_flow", "camera_pose", "timestamp")
PERSON_TOPICS = {"privileged": EVALUATOR_MANIFEST_RELATIVE.as_posix()}
EXCLUDED_MODEL_TOPICS = tuple(PERSON_TOPICS.values())
REQUIRE_PROJECTION_WORLD = True
DEPTH_STRIDE = 2
MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 80.0

MODEL_FORBIDDEN_KEYS = frozenset(
    {
        "actor_id",
        "actor_state",
        "ego",
        "target",
        "occluder",
        "observation",
        "truth",
        "current_contact",
        "contact",
        "future_contact",
        "future_contact_within_horizon",
        "realized_time_to_contact_seconds",
        "gt_cv_route_risk",
        "gt_cv_ttc_seconds",
        "target_obb_polygon_xy",
        "obb",
        "route_truth",
        "physical_loss",
        "evaluator",
        "instance",
    }
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def resolve_contained(root: Path, raw: str | Path) -> Path:
    root = root.resolve()
    value = Path(raw)
    path = value.resolve() if value.is_absolute() else (root / value).resolve()
    require(path.is_relative_to(root), f"carla_v18_path_escape:{raw}")
    return path


def assert_model_clean(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            require(str(key) not in MODEL_FORBIDDEN_KEYS, f"carla_v18_model_forbidden_key:{path}.{key}")
            assert_model_clean(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_model_clean(child, f"{path}[{index}]")


def _load_model_manifest(source_root: Path) -> tuple[Path, dict[str, Any]]:
    root = source_root.resolve()
    path = resolve_contained(root, MODEL_MANIFEST_RELATIVE)
    require(path.is_file(), f"carla_v18_model_manifest_missing:{path}")
    manifest = read_json(path)
    require(
        manifest.get("schema") == "carla-dtr-v18-model-manifest-v1"
        and manifest.get("truth_blind") is True,
        "carla_v18_model_manifest_schema",
    )
    require(manifest.get("online_modalities") == ["rgb", "depth", "flow"], "carla_v18_model_modalities")
    require(manifest.get("excluded_modality") == "instance", "carla_v18_excluded_modality")
    assert_model_clean({key: value for key, value in manifest.items() if key != "forbidden_fields"})
    return path, manifest


def prepare_descriptor(source_root: Path, path: Path) -> dict[str, Any]:
    manifest_path, manifest = _load_model_manifest(source_root)
    ledgers: list[dict[str, Any]] = []
    for sequence in manifest["scenarios"]:
        reference = manifest["ledgers"][sequence]
        ledger_path = Path(reference["path"]).resolve()
        require(ledger_path.is_relative_to(manifest_path.parent.resolve()), f"carla_v18_model_ledger_boundary:{sequence}")
        require(ledger_path.is_file(), f"carla_v18_model_ledger_missing:{sequence}")
        require(sha256_file(ledger_path) == reference["sha256"], f"carla_v18_model_ledger_drift:{sequence}")
        rows = read_jsonl(ledger_path)
        require(len(rows) == 201, f"carla_v18_model_ledger_count:{sequence}:{len(rows)}")
        assert_model_clean(rows)
        ledgers.append({"sequence": sequence, "path": str(ledger_path), "sha256": reference["sha256"], "frames": len(rows)})
    descriptor = {
        "schema": DESCRIPTOR_SCHEMA,
        "truth_blind": True,
        "source_root": str(source_root.resolve()),
        "model_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "sequences": ledgers,
        "online_modalities": list(MODEL_TOPICS),
        "excluded_topics": list(EXCLUDED_MODEL_TOPICS),
        "depth_backprojection": {"stride": DEPTH_STRIDE, "minimum_m": MIN_DEPTH_M, "maximum_m": MAX_DEPTH_M, "frame": "CAMERA_FLH"},
    }
    atomic_json(path, descriptor)
    return descriptor


def _rotation_ue(transform: Mapping[str, Any]) -> np.ndarray:
    pitch = math.radians(float(transform["pitch"]))
    yaw = math.radians(float(transform["yaw"]))
    roll = math.radians(float(transform["roll"]))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def camera_flh_world(transform: Mapping[str, Any]) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    # CARLA local coordinates are forward-right-up.  X21 uses forward-left-up.
    output[:3, :3] = _rotation_ue(transform) @ np.diag([1.0, -1.0, 1.0])
    output[:3, 3] = [float(transform["x"]), float(transform["y"]), float(transform["z"])]
    require(np.all(np.isfinite(output)), "carla_v18_camera_transform_nonfinite")
    return output


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(transform, dtype=np.float64))


def local_flh_to_world(points: np.ndarray, camera_world: np.ndarray) -> np.ndarray:
    return transform_points(points, camera_world)


def world_to_local_flh(points: np.ndarray, camera_world: np.ndarray) -> np.ndarray:
    return transform_points(points, invert_transform(camera_world))


def camera_world_transform(marker_world: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    del calibration
    return np.asarray(marker_world, dtype=np.float64)


def project_world(points: np.ndarray, projection_world: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    local = world_to_local_flh(points, projection_world)
    output = np.full((len(local), 2), np.nan, dtype=np.float64)
    valid = local[:, 0] > 1e-4
    if np.any(valid):
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        output[valid, 0] = cx - fx * local[valid, 1] / local[valid, 0]
        output[valid, 1] = cy - fy * local[valid, 2] / local[valid, 0]
    return output


def _intrinsic(width: int, height: int, fov_degrees: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return np.asarray(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    if path is not None and path.is_file():
        value = read_json(path)
        width, height = [int(item) for item in value.get("resolution", [320, 180])]
        fov = float(value.get("fov_degrees", 90.0))
    else:
        width, height, fov = 320, 180, 90.0
    return {"K": _intrinsic(width, height, fov), "distortion": np.zeros(5, dtype=np.float64), "T_c_l": np.eye(4, dtype=np.float64)}


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(value.shape == (180, 320, 3), f"carla_v18_rgb_shape:{value.shape}:{path}")
    return value


def _read_depth(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64)
    require(encoded.shape == (180, 320, 3), f"carla_v18_depth_shape:{encoded.shape}:{path}")
    return (encoded[:, :, 0] + encoded[:, :, 1] * 256.0 + encoded[:, :, 2] * 65536.0) / 16777215.0 * 1000.0


def _backproject(depth: np.ndarray, rgb: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(0, depth.shape[0], DEPTH_STRIDE, dtype=np.int32)
    columns = np.arange(0, depth.shape[1], DEPTH_STRIDE, dtype=np.int32)
    vv, uu = np.meshgrid(rows, columns, indexing="ij")
    distance = depth[vv, uu]
    valid = np.isfinite(distance) & (distance > MIN_DEPTH_M) & (distance < MAX_DEPTH_M)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    forward = distance[valid].astype(np.float64)
    right = (u - float(intrinsic[0, 2])) * forward / float(intrinsic[0, 0])
    height = -(v - float(intrinsic[1, 2])) * forward / float(intrinsic[1, 1])
    points = np.column_stack((forward, -right, height)).astype(np.float32)
    intensity = (rgb[vv[valid], uu[valid]].astype(np.float32).mean(axis=1) / 255.0).astype(np.float32)
    require(len(points) > 0 and np.all(np.isfinite(points)), "carla_v18_backprojection_empty")
    return points, intensity


def extract_causal_inputs(
    source_root: Path,
    output_root: Path,
    *,
    expected_model_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    manifest_path, manifest = _load_model_manifest(source_root)
    if expected_model_manifest_sha256 is not None:
        require(sha256_file(manifest_path) == expected_model_manifest_sha256.upper(), "carla_v18_model_manifest_drift")
    output_root = output_root.resolve()
    index_path = output_root / "input-index.json"
    require(not index_path.exists(), f"carla_v18_extract_exists:{index_path}")
    frame_root = output_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    sequence_entries: list[dict[str, Any]] = []
    total = 0
    for sequence_index, sequence in enumerate(manifest["scenarios"]):
        reference = manifest["ledgers"][sequence]
        ledger_path = Path(reference["path"]).resolve()
        require(ledger_path.is_relative_to(manifest_path.parent.resolve()), f"carla_v18_model_ledger_boundary:{sequence}")
        require(sha256_file(ledger_path) == reference["sha256"], f"carla_v18_ledger_drift:{sequence}")
        rows = read_jsonl(ledger_path)
        assert_model_clean(rows)
        flow_path = resolve_contained(source_root, rows[0]["payloads"]["flow_raw"])
        require("instance" not in flow_path.parts, "carla_v18_model_flow_boundary")
        with np.load(flow_path, allow_pickle=False) as flow_values:
            flow_all = flow_values["flow_xy"].astype(np.float32)
        require(flow_all.shape == (201, 180, 320, 2) and np.all(np.isfinite(flow_all)), f"carla_v18_flow_shape:{sequence}:{flow_all.shape}")
        frames: list[dict[str, Any]] = []
        for expected, row in enumerate(rows):
            require(int(row["sample_index"]) == expected, f"carla_v18_sample_index:{sequence}:{expected}")
            rgb_path = resolve_contained(source_root, row["payloads"]["rgb"])
            depth_path = resolve_contained(source_root, row["payloads"]["depth"])
            require("instance" not in rgb_path.parts and "instance" not in depth_path.parts, "carla_v18_model_payload_boundary")
            rgb = _read_rgb(rgb_path)
            depth = _read_depth(depth_path)
            width, height = [int(value) for value in row["camera"]["resolution"]]
            intrinsic = _intrinsic(width, height, float(row["camera"]["fov_degrees"]))
            points, intensity = _backproject(depth, rgb, intrinsic)
            marker = camera_flh_world(row["camera_transform"])
            frame = sequence_index * 1_000_000 + expected
            time_s = sequence_index * 100.0 + float(row["time_s"])
            flow_px = flow_all[expected] * np.asarray([width, height], dtype=np.float32)
            frame_path = frame_root / f"{frame:08d}.npz"
            atomic_npz(
                frame_path,
                points_lidar=points,
                intensity=intensity,
                image_rgb=rgb,
                flow_xy_px=flow_px.astype(np.float32),
                marker_world=marker,
                projection_world=marker,
                camera_intrinsic=intrinsic,
                K=intrinsic,
                pose_valid=np.asarray([True], dtype=np.bool_),
                image_valid=np.asarray([True], dtype=np.bool_),
                flow_valid=np.asarray([True], dtype=np.bool_),
                source_sample_index=np.asarray([expected], dtype=np.int32),
            )
            frames.append(
                {
                    "frame": frame,
                    "time_s": time_s,
                    "sequence_time_s": float(row["time_s"]),
                    "sequence": sequence,
                    "sample_index": expected,
                    "frame_file": str(frame_path),
                    "frame_file_sha256": sha256_file(frame_path),
                    "points": len(points),
                    "image_valid": True,
                    "pose_valid": True,
                    "flow_valid": True,
                }
            )
            total += 1
        sequence_entries.append({"sequence": sequence, "frames": frames})
    index = {
        "schema": INPUT_SCHEMA,
        "truth_blind": True,
        "source_root": str(source_root),
        "model_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "online_modalities": list(MODEL_TOPICS),
        "excluded_topics": list(EXCLUDED_MODEL_TOPICS),
        "coordinate_frame": "CAMERA_FLH_X_FORWARD_Y_LEFT_Z_UP",
        "sequences": sequence_entries,
        "aggregate": {"sequences": len(sequence_entries), "frames": total, "expected_frames": 804},
    }
    require(total == 804, f"carla_v18_extract_frame_count:{total}")
    atomic_json(index_path, index)
    return index


def iter_index_frames(index: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    require(index.get("schema") == INPUT_SCHEMA and index.get("truth_blind") is True, "carla_v18_index_schema")
    for sequence in index["sequences"]:
        for row in sequence["frames"]:
            yield row


def load_evaluator_truth(
    source_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    root = source_root.resolve()
    path = resolve_contained(root, EVALUATOR_MANIFEST_RELATIVE)
    require(path.is_file(), f"carla_v18_evaluator_manifest_missing:{path}")
    if expected_manifest_sha256 is not None:
        require(sha256_file(path) == expected_manifest_sha256.upper(), "carla_v18_evaluator_manifest_drift")
    manifest = read_json(path)
    require(manifest.get("schema") == "carla-dtr-v18-evaluator-manifest-v1" and manifest.get("privileged") is True, "carla_v18_evaluator_schema")
    output: dict[str, list[dict[str, Any]]] = {}
    for sequence, reference in manifest["ledgers"].items():
        ledger_path = Path(reference["path"]).resolve()
        require(ledger_path.is_relative_to(path.parent.resolve()), f"carla_v18_evaluator_ledger_boundary:{sequence}")
        require(ledger_path.is_file() and sha256_file(ledger_path) == reference["sha256"], f"carla_v18_evaluator_ledger_drift:{sequence}")
        rows = read_jsonl(ledger_path)
        require(len(rows) == 201, f"carla_v18_evaluator_count:{sequence}:{len(rows)}")
        output[sequence] = rows
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA V18 truth-separated raw source adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source-root", type=Path, required=True)
    extract_parser.add_argument("--output-root", type=Path, required=True)
    extract_parser.add_argument("--model-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_descriptor(args.source_root, args.output)
    else:
        result = extract_causal_inputs(
            args.source_root,
            args.output_root,
            expected_model_manifest_sha256=args.model_manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
