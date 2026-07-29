"""Independently validate the disjoint localization pilot fixture.

This control-only validator does not import the localization runner, R3, or
any tracking/local-fit implementation.  It may read the frozen formal
identity and Stage B geometry metadata solely to prove that the pilot
identities and numeric seeds are disjoint.  It never accesses the formal
localization output root or any sealed response payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
LOCALIZATION_TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0"
)
STAGE_B_TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
    "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0"
)
VALIDATOR_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_PILOT_FIXTURE_"
    "INDEPENDENT_VALIDATOR_R0"
)
INPUT_SCHEMA = "rcle.r3_rotation_leakage_source_localization.pilot_input.v1"
RECEIPT_SCHEMA = (
    "rcle.r3_rotation_leakage_source_localization."
    "pilot_fixture_independent_receipt.v1"
)
FIXTURE_ID = "R3_ROTATION_LEAKAGE_LOCALIZATION_IMPLEMENTATION_PILOT_R0"
PILOT_NAMESPACE = "R3_ROTATION_LEAKAGE_LOCALIZATION_DISJOINT_PILOT_R0"
SOURCE_ROLE = "DISJOINT_PILOT_FIXTURE"
CLUSTER_COUNT = 4
FRAME_COUNT = 9
PAIR_COUNT = 8
TARGET_ID = 1001
WIDTH = 360
HEIGHT = 640
INTRINSIC = [
    [541.2, 0.0, 182.3389],
    [0.0, 542.2, 321.654],
    [0.0, 0.0, 1.0],
]

IDENTITY_RELATIVE = Path(
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_"
    "IDENTITY_INPUT_LOCK_2026-07-29.json"
)
STAGE_B_GEOMETRY_RELATIVE = Path(
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0/control/"
    "geometry_manifest.json"
)
FORMAL_OUTPUT_RELATIVE = Path(
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_r3_rotation_leakage_source_localization_r0"
)
EXPECTED_IDENTITY_SHA256 = (
    "88b47bc5f94812227e85f83e855600c1cad92a267193854ef73ea5273c35d23a"
)
EXPECTED_STAGE_B_GEOMETRY_SHA256 = (
    "5ebd26bdced270f6e02a33573e07f2fe09d96505e575e50acf1a4fe48c1b3392"
)


class InvalidPilotFixture(ValueError):
    """Raised when the pilot fixture cannot be proven canonical and disjoint."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def require(condition: bool, code: str) -> None:
    if not condition:
        raise InvalidPilotFixture(code)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPilotFixture(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                InvalidPilotFixture(f"NONFINITE_JSON:{token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidPilotFixture(f"JSON_READ:{path}") from error
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


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


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
    )


def _derive_seed(ordinal: int) -> int:
    token = (
        f"{PROTOCOL_ID}|{PILOT_NAMESPACE}|PILOT_ONLY|{ordinal:02d}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _unit_hash(*parts: object) -> float:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return integer / float(2**64 - 1)


def _surface(
    object_id: int,
    z: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    seed: int,
) -> dict[str, Any]:
    color = [
        round(0.15 + 0.75 * _unit_hash(seed, object_id, channel), 12)
        for channel in ("r", "g", "b")
    ]
    rounded_z = round(z, 12)
    rounded_x0 = round(x0, 12)
    rounded_x1 = round(x1, 12)
    rounded_y0 = round(y0, 12)
    rounded_y1 = round(y1, 12)
    return {
        "object_id": object_id,
        "primitive": "rectangle_mesh_2tri",
        "plane_z_m": rounded_z,
        "bounds_xy_m": [
            rounded_x0,
            rounded_x1,
            rounded_y0,
            rounded_y1,
        ],
        "vertices_world_m": [
            [rounded_x0, rounded_y0, rounded_z],
            [rounded_x1, rounded_y0, rounded_z],
            [rounded_x1, rounded_y1, rounded_z],
            [rounded_x0, rounded_y1, rounded_z],
        ],
        "triangles": [[0, 1, 2], [0, 2, 3]],
        "material_id": f"MAT_{object_id:02d}",
        "linear_rgb": color,
        "texture": {
            "type": "analytic_checker",
            "cycles_per_m": round(
                3.0 + 9.0 * _unit_hash(seed, object_id, "freq"), 12
            ),
            "phase": round(_unit_hash(seed, object_id, "phase"), 12),
        },
    }


def _expected_scene(
    cluster_id: str,
    ordinal: int,
    seed: int,
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    object_id = 1
    for row in range(3):
        for column in range(3):
            z = (
                8.5
                + 0.45 * row
                + 0.7 * column
                + 0.6 * _unit_hash(
                    seed, row, column, "stage_b_depth"
                )
            )
            u0 = column * WIDTH / 3.0
            u1 = (column + 1) * WIDTH / 3.0
            v0 = row * HEIGHT / 3.0
            v1 = (row + 1) * HEIGHT / 3.0
            margin = 14.0
            x0 = ((u0 - margin - INTRINSIC[0][2]) / INTRINSIC[0][0]) * z
            x1 = ((u1 + margin - INTRINSIC[0][2]) / INTRINSIC[0][0]) * z
            y0 = ((v0 - margin - INTRINSIC[1][2]) / INTRINSIC[1][1]) * z
            y1 = ((v1 + margin - INTRINSIC[1][2]) / INTRINSIC[1][1]) * z
            objects.append(
                _surface(object_id, z, x0, x1, y0, y1, seed)
            )
            object_id += 1
    objects.append(
        _surface(10, 18.0, -12.0, 12.0, -16.0, 16.0, seed)
    )
    target = _surface(TARGET_ID, 6.0, -0.4, 0.8, -0.7, 0.9, seed)
    target["linear_rgb"] = [0.92, 0.47, 0.18]
    target["texture"]["cycles_per_m"] = 12.0
    objects.append(target)
    scene = {
        "schema": "rcle.stage_b.materialized_scene.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": STAGE_B_TASK_ID,
        "cluster_id": cluster_id,
        "block": "PILOT_ONLY",
        "ordinal": ordinal,
        "numeric_seed_uint64": seed,
        "camera": {
            "projection": "pinhole",
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "intrinsic": INTRINSIC,
            "near_clip_m": 0.5,
            "far_clip_m": 25.0,
            "distortion": "none",
            "camera_axes": "+x right, +y down, +z optical forward",
            "pose": "world_from_camera",
        },
        "world": {
            "static": False,
            "moving_objects": [TARGET_ID],
            "renderer": "deterministic analytic ray/rectangle z-buffer v1",
            "objects": objects,
        },
    }
    scene["scene_geometry_sha256"] = sha256_value(scene)
    return scene


def _rotation_x(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ]


def _rotation_y(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]


def _rotation_z(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _matrix_multiply(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _expected_rotation(
    ordinal: int, frame_index: int
) -> list[list[float]]:
    amplitude = 0.018 + 0.006 * ordinal
    phase = 0.37 * ordinal
    tau = frame_index / (FRAME_COUNT - 1)
    yaw = amplitude * math.sin(2.0 * math.pi * tau + phase)
    pitch = 0.65 * amplitude * math.sin(
        3.0 * math.pi * tau + 0.5 * phase
    )
    roll = 0.40 * amplitude * math.cos(
        2.0 * math.pi * tau - 0.25 * phase
    )
    return _matrix_multiply(
        _matrix_multiply(_rotation_z(roll), _rotation_y(yaw)),
        _rotation_x(pitch),
    )


def _determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _validate_rotation(
    value: Any,
    expected: list[list[float]],
    code: str,
) -> None:
    require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value),
        f"{code}:SHAPE",
    )
    require(
        all(_is_number(cell) for row in value for cell in row),
        f"{code}:FINITE",
    )
    matrix = [[float(cell) for cell in row] for row in value]
    for row in range(3):
        for column in range(3):
            require(
                abs(matrix[row][column] - expected[row][column]) <= 2e-15,
                f"{code}:CANONICAL",
            )
            dot = sum(
                matrix[row][inner] * matrix[column][inner]
                for inner in range(3)
            )
            require(
                abs(dot - (1.0 if row == column else 0.0)) <= 2e-12,
                f"{code}:ORTHONORMAL",
            )
    require(
        abs(_determinant(matrix) - 1.0) <= 2e-12,
        f"{code}:DETERMINANT",
    )


def _dynamic_static_scene(
    base_scene: dict[str, Any], frame_index: int
) -> dict[str, Any]:
    scene = json.loads(json.dumps(base_scene))
    targets = [
        item
        for item in scene["world"]["objects"]
        if item.get("object_id") == TARGET_ID
    ]
    require(len(targets) == 1, "PILOT_TARGET_CARDINALITY")
    target = targets[0]
    target["plane_z_m"] = 6.0
    target["bounds_xy_m"] = [-0.4, 0.8, -0.7, 0.9]
    target["vertices_world_m"] = [
        [-0.4, -0.7, 6.0],
        [0.8, -0.7, 6.0],
        [0.8, 0.9, 6.0],
        [-0.4, 0.9, 6.0],
    ]
    scene.pop("scene_geometry_sha256", None)
    scene["frame_index"] = frame_index
    scene["target_motion"] = "STATIC"
    scene["scene_geometry_sha256"] = sha256_value(scene)
    return scene


def _render_input_sha256(
    base_scene: dict[str, Any],
    poses: list[dict[str, Any]],
) -> str:
    rows = []
    for frame_index, pose in enumerate(poses):
        scene = _dynamic_static_scene(base_scene, frame_index)
        rows.append(
            {
                "frame_index": frame_index,
                "scene_geometry_sha256": scene["scene_geometry_sha256"],
                "rotation_matrix": pose["rotation_matrix"],
                "translation_m": pose["translation_m"],
                "target_z_m": 6.0,
            }
        )
    return sha256_value(rows)


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _guard_cli_paths(
    root: Path, pilot_manifest: Path, output_receipt: Path
) -> None:
    # ``root`` is already derived from a resolved __file__.  Do not resolve,
    # stat, list, or otherwise touch the forbidden formal output path.
    formal_root = root / FORMAL_OUTPUT_RELATIVE
    require(
        not pilot_manifest.is_relative_to(formal_root),
        "PILOT_MANIFEST_FORMAL_OUTPUT_PATH",
    )
    require(
        not output_receipt.is_relative_to(formal_root),
        "OUTPUT_RECEIPT_FORMAL_OUTPUT_PATH",
    )
    require(pilot_manifest != output_receipt, "INPUT_OUTPUT_COLLISION")


def _load_formal_metadata(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    identity_path = (root / IDENTITY_RELATIVE).resolve()
    require(identity_path.is_file(), "FORMAL_IDENTITY_MISSING")
    require(
        sha256_file(identity_path) == EXPECTED_IDENTITY_SHA256,
        "FORMAL_IDENTITY_HASH",
    )
    identity = load_json(identity_path)
    require(
        identity.get("schema")
        == "rcle.r3_rotation_leakage_source_localization."
        "identity_input_lock.v1",
        "FORMAL_IDENTITY_SCHEMA",
    )
    require(
        identity.get("task_id") == LOCALIZATION_TASK_ID,
        "FORMAL_IDENTITY_TASK",
    )
    require(
        identity.get("source_scope")
        == "SEALED_STAGE_B_ROTATION_ONLY_CONTROL_INPUTS",
        "FORMAL_IDENTITY_SCOPE",
    )
    counts = identity.get("counts", {})
    require(
        counts.get("clusters") == 8
        and counts.get("sequences") == 8
        and counts.get("frames_per_sequence") == 602
        and counts.get("pairs_per_sequence") == 601,
        "FORMAL_IDENTITY_COUNTS",
    )
    formal_clusters = identity.get("clusters")
    require(
        isinstance(formal_clusters, list) and len(formal_clusters) == 8,
        "FORMAL_IDENTITY_CLUSTER_COUNT",
    )
    binding = identity.get("bindings", {}).get(
        "stage_b_geometry_manifest", {}
    )
    require(
        binding.get("path") == STAGE_B_GEOMETRY_RELATIVE.as_posix()
        and binding.get("sha256") == EXPECTED_STAGE_B_GEOMETRY_SHA256,
        "FORMAL_GEOMETRY_BINDING",
    )
    geometry_path = (root / STAGE_B_GEOMETRY_RELATIVE).resolve()
    require(geometry_path.is_file(), "STAGE_B_GEOMETRY_MISSING")
    require(
        sha256_file(geometry_path) == EXPECTED_STAGE_B_GEOMETRY_SHA256,
        "STAGE_B_GEOMETRY_HASH",
    )
    geometry = load_json(geometry_path)
    require(
        geometry.get("schema") == "rcle.stage_b.geometry_manifest.v1",
        "STAGE_B_GEOMETRY_SCHEMA",
    )
    require(
        geometry.get("protocol_id") == PROTOCOL_ID
        and geometry.get("task_id") == STAGE_B_TASK_ID,
        "STAGE_B_GEOMETRY_IDENTITY",
    )
    geometry_clusters = geometry.get("clusters")
    require(
        isinstance(geometry_clusters, list) and len(geometry_clusters) == 8,
        "STAGE_B_GEOMETRY_CLUSTER_COUNT",
    )
    return identity, geometry, identity_path


def _formal_value_sets(
    identity: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[dict[str, set[str]], set[int]]:
    formal_values = {
        "cluster_id": set(),
        "sequence_id": set(),
        "scene_geometry_sha256": set(),
        "pose_sha256": set(),
        "render_input_sha256": set(),
    }
    clusters = identity["clusters"]
    for item in clusters:
        for key, values in formal_values.items():
            value = item.get(key)
            require(
                isinstance(value, str),
                f"FORMAL_VALUE_INVALID:{key}",
            )
            if key.endswith("_sha256"):
                require(_is_sha256(value), f"FORMAL_HASH_INVALID:{key}")
            if key in {"cluster_id", "sequence_id"}:
                require(
                    value not in values,
                    f"FORMAL_VALUE_DUPLICATE:{key}",
                )
            values.add(value)
    geometry_clusters = geometry["clusters"]
    require(
        {item.get("cluster_id") for item in geometry_clusters}
        == formal_values["cluster_id"],
        "STAGE_B_GEOMETRY_CLUSTER_JOIN",
    )
    formal_seeds: set[int] = set()
    for item in geometry_clusters:
        seed = item.get("numeric_seed_uint64")
        require(
            type(seed) is int
            and 0 <= seed < 2**64
            and seed not in formal_seeds,
            "STAGE_B_GEOMETRY_NUMERIC_SEED",
        )
        formal_seeds.add(seed)
    return formal_values, formal_seeds


def _validate_manifest(
    manifest: dict[str, Any],
    formal_values: dict[str, set[str]],
    formal_seeds: set[int],
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    require(manifest.get("schema") == INPUT_SCHEMA, "PILOT_SCHEMA")
    require(manifest.get("fixture_id") == FIXTURE_ID, "PILOT_FIXTURE_ID")
    require(manifest.get("mode") == "DISJOINT_PILOT", "PILOT_MODE")
    require(manifest.get("source_role") == SOURCE_ROLE, "PILOT_SOURCE_ROLE")
    require(manifest.get("response_blind") is True, "PILOT_RESPONSE_BLIND")
    require(
        manifest.get("identity_lock_payload_access") is False,
        "PILOT_IDENTITY_PAYLOAD_ACCESS",
    )
    require(
        manifest.get("sealed_cluster_access") is False,
        "PILOT_SEALED_CLUSTER_ACCESS",
    )
    require(
        manifest.get("scientific_interpretation") is False,
        "PILOT_SCIENTIFIC_INTERPRETATION",
    )
    require(
        manifest.get("cluster_count") == CLUSTER_COUNT
        and manifest.get("frames_per_cluster") == FRAME_COUNT
        and manifest.get("pairs_per_cluster") == PAIR_COUNT,
        "PILOT_DECLARED_COUNTS",
    )
    clusters = manifest.get("clusters")
    require(
        isinstance(clusters, list) and len(clusters) == CLUSTER_COUNT,
        "PILOT_CLUSTER_COUNT",
    )

    pilot_values: dict[str, list[Any]] = {
        "cluster_id": [],
        "sequence_id": [],
        "scene_geometry_sha256": [],
        "pose_sha256": [],
        "render_input_sha256": [],
        "numeric_seed_uint64": [],
    }
    summaries: list[dict[str, Any]] = []
    for ordinal, item in enumerate(clusters):
        require(isinstance(item, dict), f"PILOT_CLUSTER_OBJECT:{ordinal}")
        cluster_id = f"PILOT_ONLY_R3_LOC_R0_C{ordinal + 1}"
        sequence_id = (
            f"{cluster_id}__EGO_ROTATION_STATIC_SCENE__CLEAN"
        )
        seed = _derive_seed(ordinal)
        require(item.get("cluster_id") == cluster_id, "PILOT_CLUSTER_ID")
        require(item.get("sequence_id") == sequence_id, "PILOT_SEQUENCE_ID")
        require(item.get("block") == "PILOT_ONLY", "PILOT_BLOCK")
        require(item.get("ordinal") == ordinal, "PILOT_ORDINAL")
        require(
            item.get("numeric_seed_uint64") == seed,
            "PILOT_CANONICAL_SEED",
        )
        require(seed not in formal_seeds, "PILOT_STAGE_B_SEED_COLLISION")

        base_scene = item.get("base_scene")
        require(isinstance(base_scene, dict), "PILOT_BASE_SCENE")
        expected_scene = _expected_scene(cluster_id, ordinal, seed)
        require(base_scene == expected_scene, "PILOT_CANONICAL_SCENE")
        scene_sha256 = expected_scene["scene_geometry_sha256"]
        require(
            item.get("scene_geometry_sha256") == scene_sha256,
            "PILOT_SCENE_HASH",
        )

        poses = item.get("poses")
        require(
            isinstance(poses, list) and len(poses) == FRAME_COUNT,
            "PILOT_POSE_COUNT",
        )
        for frame_index, pose in enumerate(poses):
            require(isinstance(pose, dict), "PILOT_POSE_OBJECT")
            require(
                pose.get("frame_index") == frame_index,
                "PILOT_FRAME_INDEX",
            )
            require(
                type(pose.get("timestamp_s")) is float
                and pose["timestamp_s"] == frame_index / 60.0,
                "PILOT_TIMESTAMP",
            )
            require(
                pose.get("translation_m") == [0.0, 0.0, 0.0]
                and all(
                    type(value) is float
                    for value in pose["translation_m"]
                ),
                "PILOT_ZERO_TRANSLATION",
            )
            _validate_rotation(
                pose.get("rotation_matrix"),
                _expected_rotation(ordinal, frame_index),
                f"PILOT_ROTATION:{ordinal}:{frame_index}",
            )

        pose_sha256 = sha256_value(poses)
        require(item.get("pose_sha256") == pose_sha256, "PILOT_POSE_HASH")
        render_sha256 = _render_input_sha256(base_scene, poses)
        require(
            item.get("render_input_sha256") == render_sha256,
            "PILOT_RENDER_INPUT_HASH",
        )
        values = {
            "cluster_id": cluster_id,
            "sequence_id": sequence_id,
            "scene_geometry_sha256": scene_sha256,
            "pose_sha256": pose_sha256,
            "render_input_sha256": render_sha256,
            "numeric_seed_uint64": seed,
        }
        for key, value in values.items():
            require(
                value not in pilot_values[key],
                f"PILOT_DUPLICATE_VALUE:{key}",
            )
            pilot_values[key].append(value)
            if key in formal_values:
                require(
                    value not in formal_values[key],
                    f"PILOT_FORMAL_COLLISION:{key}",
                )
        summaries.append(values)

    formal_identity_union = set().union(*formal_values.values())
    pilot_identity_union = {
        value
        for key, values in pilot_values.items()
        if key != "numeric_seed_uint64"
        for value in values
    }
    require(
        pilot_identity_union.isdisjoint(formal_identity_union),
        "PILOT_FORMAL_CROSS_FIELD_COLLISION",
    )
    require(
        set(pilot_values["numeric_seed_uint64"]).isdisjoint(formal_seeds),
        "PILOT_FORMAL_SEED_COLLISION",
    )
    return summaries, pilot_values


def validate(
    root: Path,
    pilot_manifest_path: Path,
    output_receipt_path: Path,
) -> dict[str, Any]:
    _guard_cli_paths(root, pilot_manifest_path, output_receipt_path)
    require(pilot_manifest_path.is_file(), "PILOT_MANIFEST_MISSING")
    manifest = load_json(pilot_manifest_path)
    require(
        pilot_manifest_path.read_bytes() == canonical_bytes(manifest),
        "PILOT_MANIFEST_NOT_CANONICAL",
    )
    identity, geometry, identity_path = _load_formal_metadata(root)
    formal_values, formal_seeds = _formal_value_sets(identity, geometry)
    summaries, pilot_values = _validate_manifest(
        manifest, formal_values, formal_seeds
    )
    validator_path = Path(__file__).resolve()
    checks = [
        {"name": "CLI_PATH_FIREWALL", "status": "PASS"},
        {
            "name": "CANONICAL_MANIFEST_SCHEMA_AND_ROLE",
            "status": "PASS",
        },
        {"name": "EXACT_4_CLUSTERS_9_FRAMES_8_PAIRS", "status": "PASS"},
        {"name": "PILOT_ONLY_NAMESPACE_AND_IDENTITIES", "status": "PASS"},
        {"name": "CANONICAL_NUMERIC_SEEDS", "status": "PASS"},
        {"name": "CANONICAL_SCENE_AND_HASHES", "status": "PASS"},
        {
            "name": "CANONICAL_ROTATION_ZERO_TRANSLATION_TIMELINE",
            "status": "PASS",
        },
        {"name": "CANONICAL_POSE_AND_RENDER_HASHES", "status": "PASS"},
        {
            "name": "LOCALIZATION_IDENTITY_VALUES_DISJOINT",
            "status": "PASS",
        },
        {
            "name": "STAGE_B_GEOMETRY_NUMERIC_SEEDS_DISJOINT",
            "status": "PASS",
        },
        {
            "name": "FORMAL_AUTHORITY_AND_SCIENTIFIC_FIREWALL",
            "status": "PASS",
        },
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "task_id": LOCALIZATION_TASK_ID,
        "validator_id": VALIDATOR_ID,
        "fixture_id": FIXTURE_ID,
        "source_role": SOURCE_ROLE,
        "status": "PASS",
        "pilot_manifest": {
            "path": _path_label(pilot_manifest_path, root),
            "sha256": sha256_file(pilot_manifest_path),
        },
        "pilot_manifest_sha256": sha256_file(pilot_manifest_path),
        "validator": {
            "path": _path_label(validator_path, root),
            "sha256": sha256_file(validator_path),
            "imports_runner_or_r3": False,
        },
        "validator_sha256": sha256_file(validator_path),
        "formal_identity_lock": {
            "path": _path_label(identity_path, root),
            "sha256": EXPECTED_IDENTITY_SHA256,
            "metadata_access_for_disjointness_only": True,
        },
        "formal_identity_lock_sha256": EXPECTED_IDENTITY_SHA256,
        "stage_b_geometry_manifest": {
            "path": STAGE_B_GEOMETRY_RELATIVE.as_posix(),
            "sha256": EXPECTED_STAGE_B_GEOMETRY_SHA256,
            "metadata_access_for_seed_disjointness_only": True,
        },
        "checks": checks,
        "counts": {
            "clusters": len(summaries),
            "frames_per_cluster": FRAME_COUNT,
            "pairs_per_cluster": PAIR_COUNT,
            "formal_identity_values_by_field": {
                key: len(values) for key, values in formal_values.items()
            },
            "formal_stage_b_numeric_seeds": len(formal_seeds),
        },
        "pilot_identities": summaries,
        "identity_disjointness": {
            "cluster_id": True,
            "sequence_id": True,
            "scene_geometry_sha256": True,
            "pose_sha256": True,
            "render_input_sha256": True,
            "cross_field_values": True,
            "stage_b_numeric_seed_uint64": True,
            "pilot_unique_values": {
                key: len(set(values))
                for key, values in pilot_values.items()
            },
        },
        "formal_identity_metadata_access": True,
        "stage_b_geometry_metadata_access": True,
        "identity_lock_payload_access": False,
        "sealed_cluster_access": False,
        "sealed_response_payload_access": False,
        "formal_output_root_access": False,
        "formal_authority_consumed": False,
        "formal_workload_calls": 0,
        "formal_480_plus_16_calls": 0,
        "scientific_interpretation": False,
        "scientific_status": "NOT_EVALUABLE_FOR_SCIENTIFIC_INTERPRETATION",
        "claim_ceiling": (
            "PILOT_FIXTURE_IDENTITY_AND_MECHANICS_PREFLIGHT_ONLY; "
            "NO ALGORITHM, PERFORMANCE, REAL_SCENE, PRODUCT OR SAFETY CLAIM"
        ),
        "terminal": (
            "PILOT_FIXTURE_DISJOINT_PASS / "
            "FORMAL_AUTHORITY_NOT_CONSUMED / "
            "SCIENTIFICALLY_NOT_INTERPRETABLE"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a disjoint localization pilot"
    )
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    root = repo_root()
    pilot_manifest_path = arguments.pilot_manifest.resolve()
    output_receipt_path = arguments.output_receipt.resolve()
    receipt = validate(root, pilot_manifest_path, output_receipt_path)
    write_exclusive(output_receipt_path, receipt)
    print(
        canonical_bytes(
            {
                "receipt_path": _path_label(output_receipt_path, root),
                "receipt_sha256": sha256_file(output_receipt_path),
                "status": receipt["status"],
                "terminal": receipt["terminal"],
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
