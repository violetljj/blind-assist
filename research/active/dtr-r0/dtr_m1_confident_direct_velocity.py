"""Filter point-wise direct velocity by causal confidence and consistency.

M1-CT consumes an already materialized current/past direct-velocity ledger; the
fresh C2 replay uses truth-blind raw-LiDAR occupancy-cell velocity.  A current
cell is admitted when one independent historical sweep supports it after ego
compensation and forward advection.  Matching does not use evaluator identity.
The downstream route geometry, entry calculation, motion bounds, and event
lifecycle remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json
from dtr_r7_occupancy_flow_canary import FROZEN_FLOW_CONFIG, FlowLedger, atomic_npz


SCHEMA = "blindassist-dtr-m1-confidence-temporal-direct-velocity-ledger-v1"
CONFIDENCE_THRESHOLD = 0.5
POSITION_SIGMA_M = 2.0 * FROZEN_FLOW_CONFIG.voxel_size_m
VELOCITY_SIGMA_MPS = FROZEN_FLOW_CONFIG.voxel_size_m / FROZEN_FLOW_CONFIG.history_min_s
SEARCH_RADIUS_M = POSITION_SIGMA_M * math.sqrt(2.0 * math.log(2.0))


def ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".confident-direct-velocity.npz"),
        output.with_name(output.stem + ".confident-direct-velocity.json"),
    )


def _local_to_world(points: Any, x_m: float, y_m: float, yaw: float) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.column_stack(
        [
            x_m + cosine * values[:, 0] - sine * values[:, 1],
            y_m + sine * values[:, 0] + cosine * values[:, 1],
        ]
    )


def _velocity_to_world(values: Any, yaw: float) -> Any:
    import numpy as np

    velocity = np.asarray(values, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.column_stack(
        [
            cosine * velocity[:, 0] - sine * velocity[:, 1],
            sine * velocity[:, 0] + cosine * velocity[:, 1],
        ]
    )


def _history_index(times: Any, index: int) -> int | None:
    current = float(times[index])
    candidates = [
        previous
        for previous in range(index)
        if FROZEN_FLOW_CONFIG.history_min_s
        <= current - float(times[previous])
        <= FROZEN_FLOW_CONFIG.history_max_s
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda previous: abs(
            current - float(times[previous]) - FROZEN_FLOW_CONFIG.history_target_s
        ),
    )


def _candidate_grid(points: Any) -> dict[tuple[int, int], list[int]]:
    import numpy as np

    cells = np.floor(np.asarray(points) / FROZEN_FLOW_CONFIG.voxel_size_m).astype(int)
    output: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        output.setdefault((int(cell[0]), int(cell[1])), []).append(index)
    return output


def materialize(
    *,
    source_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import numpy as np

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require(
        source_manifest.get("oracle") is True
        or source_manifest.get("truth_blind") is True,
        "direct_velocity_source_required",
    )
    require(sha256_file(source_path) == source_manifest["ledger_sha256"], "m1_source_hash_drift")
    values = np.load(source_path, allow_pickle=False)
    required = {
        "frames",
        "frame_time_s",
        "frame_ego_x_m",
        "frame_ego_y_m",
        "frame_ego_yaw_rad",
        "offsets",
        "forward_m",
        "left_m",
        "velocity_forward_mps",
        "velocity_left_mps",
        "component_id",
        "source_point_count",
        "flow_support",
    }
    require(required <= set(values.files), "m1_source_missing_temporal_arrays")
    frames = values["frames"]
    times = values["frame_time_s"]
    offsets = values["offsets"]
    output_rows = []
    confidence_rows = []
    residual_rows = []
    velocity_error_rows = []
    output_offsets = [0]
    frame_diagnostics = {}
    search_cells = math.ceil(SEARCH_RADIUS_M / FROZEN_FLOW_CONFIG.voxel_size_m)
    for index, frame_value in enumerate(frames):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        current_position = np.column_stack(
            [values["forward_m"][start:stop], values["left_m"][start:stop]]
        ).astype(np.float64)
        current_velocity = np.column_stack(
            [
                values["velocity_forward_mps"][start:stop],
                values["velocity_left_mps"][start:stop],
            ]
        ).astype(np.float64)
        history = _history_index(times, index)
        keep = np.zeros(len(current_position), dtype=bool)
        confidence = np.zeros(len(current_position), dtype=np.float32)
        residual = np.full(len(current_position), np.nan, dtype=np.float32)
        velocity_error = np.full(len(current_position), np.nan, dtype=np.float32)
        if history is not None and len(current_position):
            previous_start = int(offsets[history])
            previous_stop = int(offsets[history + 1])
            previous_position = np.column_stack(
                [
                    values["forward_m"][previous_start:previous_stop],
                    values["left_m"][previous_start:previous_stop],
                ]
            ).astype(np.float64)
            previous_velocity = np.column_stack(
                [
                    values["velocity_forward_mps"][previous_start:previous_stop],
                    values["velocity_left_mps"][previous_start:previous_stop],
                ]
            ).astype(np.float64)
            current_world = _local_to_world(
                current_position,
                float(values["frame_ego_x_m"][index]),
                float(values["frame_ego_y_m"][index]),
                float(values["frame_ego_yaw_rad"][index]),
            )
            current_velocity_world = _velocity_to_world(
                current_velocity, float(values["frame_ego_yaw_rad"][index])
            )
            previous_world = _local_to_world(
                previous_position,
                float(values["frame_ego_x_m"][history]),
                float(values["frame_ego_y_m"][history]),
                float(values["frame_ego_yaw_rad"][history]),
            )
            previous_velocity_world = _velocity_to_world(
                previous_velocity, float(values["frame_ego_yaw_rad"][history])
            )
            delta_s = float(times[index]) - float(times[history])
            projected = previous_world + previous_velocity_world * delta_s
            grid = _candidate_grid(projected)
            current_cells = np.floor(
                current_world / FROZEN_FLOW_CONFIG.voxel_size_m
            ).astype(int)
            for current_index, cell in enumerate(current_cells):
                candidates = []
                for dx in range(-search_cells, search_cells + 1):
                    for dy in range(-search_cells, search_cells + 1):
                        candidates.extend(grid.get((int(cell[0] + dx), int(cell[1] + dy)), ()))
                if not candidates:
                    continue
                candidate_array = np.asarray(candidates, dtype=int)
                distances = np.linalg.norm(projected[candidate_array] - current_world[current_index], axis=1)
                nearest_local = int(np.argmin(distances))
                previous_index = int(candidate_array[nearest_local])
                position_residual = float(distances[nearest_local])
                motion_error = float(
                    np.linalg.norm(
                        previous_velocity_world[previous_index]
                        - current_velocity_world[current_index]
                    )
                )
                current_support = min(1.0, float(values["source_point_count"][start + current_index]) / 3.0)
                previous_support = min(
                    1.0,
                    float(values["source_point_count"][previous_start + previous_index]) / 3.0,
                )
                support_confidence = min(
                    current_support,
                    previous_support,
                    float(values["flow_support"][start + current_index]),
                    float(values["flow_support"][previous_start + previous_index]),
                )
                position_confidence = math.exp(-0.5 * (position_residual / POSITION_SIGMA_M) ** 2)
                velocity_confidence = math.exp(-0.5 * (motion_error / VELOCITY_SIGMA_MPS) ** 2)
                value = min(support_confidence, position_confidence, velocity_confidence)
                confidence[current_index] = value
                residual[current_index] = position_residual
                velocity_error[current_index] = motion_error
                keep[current_index] = value >= CONFIDENCE_THRESHOLD
        indices = np.nonzero(keep)[0]
        output_rows.append(
            {
                "forward": values["forward_m"][start:stop][indices],
                "left": values["left_m"][start:stop][indices],
                "vf": values["velocity_forward_mps"][start:stop][indices],
                "vl": values["velocity_left_mps"][start:stop][indices],
                "component": values["component_id"][start:stop][indices],
            }
        )
        confidence_rows.append(confidence[indices])
        residual_rows.extend(residual[indices].tolist())
        velocity_error_rows.extend(velocity_error[indices].tolist())
        output_offsets.append(output_offsets[-1] + len(indices))
        frame_diagnostics[f"{int(frame_value):06d}"] = {
            "source_cells": stop - start,
            "admitted_cells": len(indices),
            "history_frame": None if history is None else int(frames[history]),
        }
    arrays = {
        "frames": frames,
        "offsets": np.asarray(output_offsets, dtype=np.int64),
        "forward_m": np.concatenate([row["forward"] for row in output_rows]),
        "left_m": np.concatenate([row["left"] for row in output_rows]),
        "velocity_forward_mps": np.concatenate([row["vf"] for row in output_rows]),
        "velocity_left_mps": np.concatenate([row["vl"] for row in output_rows]),
        "component_id": np.concatenate([row["component"] for row in output_rows]),
        "confidence": np.concatenate(confidence_rows),
    }
    atomic_npz(output_path, **arrays)

    def summary(rows: list[float]) -> dict[str, float | None]:
        if not rows:
            return {"minimum": None, "median": None, "maximum": None}
        array = np.asarray(rows, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "median": float(np.median(array)),
            "maximum": float(array.max()),
        }

    manifest = {
        "schema_version": SCHEMA,
        "oracle": bool(source_manifest.get("oracle") is True),
        "truth_blind": bool(source_manifest.get("truth_blind") is True),
        "sequence": source_manifest["sequence"],
        "frames": source_manifest["frames"],
        "motion_source": (
            "truth-blind raw-LiDAR occupancy-cell direct velocity with identity-free causal temporal confirmation"
            if source_manifest.get("truth_blind") is True
            else "M1 current/native-past point velocity with identity-free causal temporal confirmation"
        ),
        "confidence": {
            "combination": "minimum of source spatial support, forward-advection consistency, and velocity consistency",
            "threshold": CONFIDENCE_THRESHOLD,
            "source_support_saturation_count": 3,
            "position_sigma_m": POSITION_SIGMA_M,
            "velocity_sigma_mps": VELOCITY_SIGMA_MPS,
            "history_target_s": FROZEN_FLOW_CONFIG.history_target_s,
            "history_range_s": [
                FROZEN_FLOW_CONFIG.history_min_s,
                FROZEN_FLOW_CONFIG.history_max_s,
            ],
            "evaluator_identity_used_for_temporal_confirmation": False,
        },
        "frozen_downstream": source_manifest.get(
            "frozen_downstream",
            {
                "r7_flow_config": source_manifest.get("config"),
                "route_geometry_and_lifecycle": "UNCHANGED_R7",
            },
        ),
        "source": {
            "direct_velocity_ledger": str(source_path),
            "direct_velocity_ledger_sha256": sha256_file(source_path),
            "direct_velocity_manifest": str(source_manifest_path),
            "direct_velocity_manifest_sha256": sha256_file(source_manifest_path),
        },
        "diagnostics": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_cells": int(len(values["forward_m"])),
            "admitted_cells": int(len(arrays["forward_m"])),
            "admitted_fraction": (
                None
                if len(values["forward_m"]) == 0
                else len(arrays["forward_m"]) / len(values["forward_m"])
            ),
            "confidence": summary(arrays["confidence"].tolist()),
            "position_residual_m": summary(residual_rows),
            "velocity_error_mps": summary(velocity_error_rows),
            "frames_with_admitted_cells": sum(
                row["admitted_cells"] > 0 for row in frame_diagnostics.values()
            ),
            "frame_counts": frame_diagnostics,
        },
        "ledger": str(output_path),
        "ledger_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_ledger(
    path: Path,
    manifest_path: Path,
    *,
    expected_sequence: str,
    expected_frames: list[int],
) -> FlowLedger:
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == SCHEMA, "m1_ct_schema")
    require(manifest.get("sequence") == expected_sequence, "m1_ct_sequence")
    require(manifest.get("truth_blind") is True, "m1_ct_truth_blind_source_required")
    require(sha256_file(path) == manifest["ledger_sha256"], "m1_ct_hash_drift")
    values = np.load(path, allow_pickle=False)
    require(values["frames"].tolist() == expected_frames, "m1_ct_frames")
    return FlowLedger(
        frames=values["frames"],
        offsets=values["offsets"],
        forward_m=values["forward_m"],
        left_m=values["left_m"],
        velocity_forward_mps=values["velocity_forward_mps"],
        velocity_left_mps=values["velocity_left_mps"],
        component_id=values["component_id"],
        manifest=manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(
        source_path=args.source.resolve(strict=True),
        source_manifest_path=args.source_manifest.resolve(strict=True),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(json.dumps({"status": "M1_CT_MATERIALIZED", "diagnostics": result["diagnostics"]}))


if __name__ == "__main__":
    main()
