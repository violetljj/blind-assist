"""Materialize four disjoint, non-scientific localization pilot fixtures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = (
        "scripts.research.egomotion_compensated_looming."
        "periodic_self_motion_counterfactual_r2"
    )

import numpy as np

from . import generator_geometry as geometry
from . import r3_rotation_leakage_source_localization_r0 as localization
from . import stage_b_translation_depth_oracle_object_approach_r0 as stage_b


SCHEMA = "rcle.r3_rotation_leakage_source_localization.pilot_input.v1"
FIXTURE_ID = "R3_ROTATION_LEAKAGE_LOCALIZATION_IMPLEMENTATION_PILOT_R0"
NAMESPACE = "R3_ROTATION_LEAKAGE_LOCALIZATION_DISJOINT_PILOT_R0"
CLUSTER_COUNT = 4
FRAME_COUNT = 9


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pilot_poses(ordinal: int) -> list[dict[str, object]]:
    amplitude = 0.018 + 0.006 * ordinal
    phase = 0.37 * ordinal
    poses: list[dict[str, object]] = []
    for frame_index in range(FRAME_COUNT):
        tau = frame_index / (FRAME_COUNT - 1)
        yaw = amplitude * math.sin(2.0 * math.pi * tau + phase)
        pitch = 0.65 * amplitude * math.sin(
            3.0 * math.pi * tau + 0.5 * phase
        )
        roll = 0.40 * amplitude * math.cos(
            2.0 * math.pi * tau - 0.25 * phase
        )
        rotation = _rotation_z(roll) @ _rotation_y(yaw) @ _rotation_x(pitch)
        poses.append(
            {
                "frame_index": frame_index,
                "timestamp_s": float(frame_index / 60.0),
                "rotation_matrix": rotation.tolist(),
                "translation_m": [0.0, 0.0, 0.0],
            }
        )
    return poses


def build_manifest() -> dict[str, object]:
    clusters: list[dict[str, object]] = []
    for ordinal in range(CLUSTER_COUNT):
        cluster_id = f"PILOT_ONLY_R3_LOC_R0_C{ordinal + 1}"
        sequence_id = f"{cluster_id}__EGO_ROTATION_STATIC_SCENE__CLEAN"
        seed = geometry.derive_seed(NAMESPACE, "PILOT_ONLY", ordinal)
        base_scene = stage_b._scene_seed_scene(
            {
                "cluster_id": cluster_id,
                "block": "PILOT_ONLY",
                "ordinal": ordinal,
                "numeric_seed_uint64": seed,
            }
        )
        poses = _pilot_poses(ordinal)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "sequence_id": sequence_id,
                "block": "PILOT_ONLY",
                "ordinal": ordinal,
                "numeric_seed_uint64": seed,
                "base_scene": base_scene,
                "poses": poses,
                "scene_geometry_sha256": base_scene[
                    "scene_geometry_sha256"
                ],
                "pose_sha256": localization.sha256_value(poses),
                "render_input_sha256": localization._render_input_sha(
                    base_scene, poses
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "fixture_id": FIXTURE_ID,
        "mode": "DISJOINT_PILOT",
        "source_role": "DISJOINT_PILOT_FIXTURE",
        "response_blind": True,
        "identity_lock_payload_access": False,
        "sealed_cluster_access": False,
        "scientific_interpretation": False,
        "cluster_count": CLUSTER_COUNT,
        "frames_per_cluster": FRAME_COUNT,
        "pairs_per_cluster": FRAME_COUNT - 1,
        "clusters": clusters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the disjoint localization pilot manifest"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    root = localization.repo_root()
    output = arguments.output.resolve()
    parent = (root / localization.PILOT_PARENT_RELATIVE).resolve()
    if not output.is_relative_to(parent) or output.exists():
        raise ValueError("PILOT_MANIFEST_TARGET")
    output.parent.mkdir(parents=True, exist_ok=True)
    localization.write_exclusive_json(output, build_manifest())
    print(
        localization.canonical_bytes(
            {
                "manifest_path": output.relative_to(root).as_posix(),
                "manifest_sha256": localization.sha256_file(output),
                "terminal": "DISJOINT_PILOT_FIXTURE_MATERIALIZED",
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
