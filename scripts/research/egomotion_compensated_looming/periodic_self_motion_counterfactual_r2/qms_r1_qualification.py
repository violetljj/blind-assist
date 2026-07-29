"""Response-blind QMS-R1 qualification for fixed material texture attenuation.

The treatment is defined in the source domain: every material's checker
residual about its own fixed mean is multiplied by one global alpha.  The hard
manipulation gate measures that source-matched residual inside eroded material
regions.  Full-frame gradient density is retained only as descriptive
transport heterogeneity and cannot fail or rescue the treatment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any
from unittest import mock

import cv2
import numpy as np

from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as operator
from . import quality_interventions_r0 as quality


QMS_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_MANIPULATION_SUCCESSOR_R1"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
FRAME_POSITIONS = (
    0, 40, 80, 120, 160, 200, 240, 280,
    320, 360, 400, 440, 480, 520, 560, 601,
)
MATERIAL_RESIDUAL_RATIO_RANGE = (0.10, 0.20)
CAL_SEEDS_PER_BLOCK = 4
INTERIOR_EROSION_SIZE = 9


class InvalidQmsR1(ValueError):
    pass


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


def derive_seed(block: str, ordinal: int) -> dict[str, Any]:
    if block not in BLOCKS or ordinal not in range(100):
        raise InvalidQmsR1("SEED_DOMAIN")
    token = f"{QMS_ID}|CAL|{block}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def build_cal_scene(block: str, ordinal: int) -> dict[str, Any]:
    seed = derive_seed(block, ordinal)
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        scene = geometry.build_scene(block, ordinal, "QMS_R1_CAL")
    scene["protocol_id"] = QMS_ID
    scene["namespace"] = "CAL"
    scene["qms_seed_token_sha256"] = seed["token_sha256"]
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


def _interior_by_object(
    object_id: np.ndarray, valid: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    kernel = np.ones(
        (INTERIOR_EROSION_SIZE, INTERIOR_EROSION_SIZE), dtype=np.uint8
    )
    result = []
    for identifier in np.unique(object_id[valid]):
        region = ((object_id == identifier) & valid).astype(np.uint8)
        interior = cv2.erode(
            region,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        if int(np.count_nonzero(interior)) >= 50:
            result.append((int(identifier), interior))
    if len(result) < 9:
        raise InvalidQmsR1("INSUFFICIENT_MATERIAL_INTERIORS")
    return result


def material_residual_ratio(
    clean_rgb: np.ndarray,
    degraded_rgb: np.ndarray,
    object_id: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, int]:
    clean_y = quality.linear_luminance(clean_rgb)
    degraded_y = quality.linear_luminance(degraded_rgb)
    ratios = []
    for _, interior in _interior_by_object(object_id, valid):
        clean_std = float(np.std(clean_y[interior], ddof=0))
        degraded_std = float(np.std(degraded_y[interior], ddof=0))
        if clean_std >= 1e-5 and math.isfinite(degraded_std):
            ratios.append(degraded_std / clean_std)
    finite = [value for value in ratios if math.isfinite(value)]
    if len(finite) < 9:
        raise InvalidQmsR1("INSUFFICIENT_MATERIAL_RESIDUAL_RATIOS")
    return float(np.median(np.asarray(finite))), len(finite)


def structure_contrast_ratio(
    clean_rgb: np.ndarray,
    degraded_rgb: np.ndarray,
    object_id: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, int]:
    clean = quality.linear_luminance(clean_rgb)
    degraded = quality.linear_luminance(degraded_rgb)
    ratios: list[np.ndarray] = []
    for axis in (0, 1):
        if axis == 0:
            source = np.abs(clean[1:, :] - clean[:-1, :])
            target = np.abs(degraded[1:, :] - degraded[:-1, :])
            boundary = (
                valid[1:, :]
                & valid[:-1, :]
                & (object_id[1:, :] != object_id[:-1, :])
            )
        else:
            source = np.abs(clean[:, 1:] - clean[:, :-1])
            target = np.abs(degraded[:, 1:] - degraded[:, :-1])
            boundary = (
                valid[:, 1:]
                & valid[:, :-1]
                & (object_id[:, 1:] != object_id[:, :-1])
            )
        selected = boundary & np.isfinite(source) & np.isfinite(target)
        selected &= source >= 0.02
        ratios.append(target[selected] / source[selected])
    values = np.concatenate(ratios)
    values = values[np.isfinite(values)]
    if values.size < 32:
        raise InvalidQmsR1("INSUFFICIENT_STRUCTURE_EDGES")
    return float(np.median(values)), int(values.size)


def _poses(trajectory: dict[str, Any], motion: str) -> list[dict[str, Any]]:
    if motion == "PERIODIC_6DOF_SELF_MOTION":
        return trajectory["poses"]
    if motion != "STATIC_CAMERA":
        raise InvalidQmsR1("MOTION")
    return [
        {
            "frame_index": index,
            "timestamp_s": pose["timestamp_s"],
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
        }
        for index, pose in enumerate(trajectory["poses"])
    ]


def evaluate_sequence(
    identity: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    frames = []
    poses = _poses(trajectory, identity["motion"])
    static_cache: dict[str, Any] | None = None
    for frame_index in FRAME_POSITIONS:
        pose = poses[frame_index]
        if identity["motion"] == "STATIC_CAMERA" and static_cache is not None:
            rendered = static_cache
        else:
            rendered = operator.render_pair(
                identity["scene"],
                np.asarray(pose["rotation_matrix"], dtype=np.float64),
                np.asarray(pose["translation_m"], dtype=np.float64),
            )
            if identity["motion"] == "STATIC_CAMERA":
                static_cache = rendered
        clean = rendered["rgb_pair"]["clean"]
        degraded = rendered["rgb_pair"]["low"]
        valid = rendered["valid_mask"]
        object_id = rendered["object_id"]
        residual, material_count = material_residual_ratio(
            clean, degraded, object_id, valid
        )
        structure, edge_count = structure_contrast_ratio(
            clean, degraded, object_id, valid
        )
        clean_y = quality.linear_luminance(clean)
        degraded_y = quality.linear_luminance(degraded)
        full_clean, full_degraded = quality.multiscale_gradient_density_pair(
            clean_y, degraded_y, valid
        )
        frames.append(
            {
                "frame_index": frame_index,
                "material_residual_ratio": residual,
                "material_count": material_count,
                "structure_contrast_ratio": structure,
                "structure_edge_count": edge_count,
                "descriptive_full_frame_gradient_density_ratio": (
                    full_degraded / full_clean
                ),
                "clean_rgb_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                "low_texture_rgb_sha256": hashlib.sha256(
                    degraded.tobytes()
                ).hexdigest(),
                "valid_mask_sha256": hashlib.sha256(
                    valid.astype(np.uint8).tobytes()
                ).hexdigest(),
                "object_id_sha256": hashlib.sha256(
                    object_id.astype("<i4").tobytes()
                ).hexdigest(),
                "prequantization_residual_max_abs_error": rendered[
                    "prequantization_identity"
                ]["residual_relation_max_abs_error"],
                "scene_geometry_sha256": rendered["geometry_identity"][
                    "scene_geometry_sha256"
                ],
            }
        )
    residual_median = float(
        np.median([item["material_residual_ratio"] for item in frames])
    )
    structure_median = float(
        np.median([item["structure_contrast_ratio"] for item in frames])
    )
    descriptive_median = float(
        np.median(
            [
                item["descriptive_full_frame_gradient_density_ratio"]
                for item in frames
            ]
        )
    )
    passed = bool(
        MATERIAL_RESIDUAL_RATIO_RANGE[0]
        <= residual_median
        <= MATERIAL_RESIDUAL_RATIO_RANGE[1]
        and all(
            item["prequantization_residual_max_abs_error"] == 0.0
            and item["scene_geometry_sha256"]
            == identity["scene_geometry_sha256"]
            for item in frames
        )
    )
    return {
        **{
            key: identity[key]
            for key in (
                "cohort",
                "cluster_id",
                "block",
                "ordinal",
                "motion",
                "numeric_seed_uint64",
                "scene_geometry_sha256",
            )
        },
        "frame_positions": list(FRAME_POSITIONS),
        "frame_rows": frames,
        "sequence_medians": {
            "material_residual_ratio": residual_median,
            "structure_contrast_ratio": structure_median,
            "descriptive_full_frame_gradient_density_ratio": descriptive_median,
        },
        "sequence_pass": passed,
    }


def _worker(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return evaluate_sequence(*task)


def _predecessor(source: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = []
    for row in rows:
        if row.get("record_type") != "main_cluster":
            continue
        for motion in MOTIONS:
            result.append(
                {
                    "cohort": "CONSUMED_PREDECESSOR_DEVELOPMENT_ONLY",
                    "cluster_id": row["cluster_id"],
                    "block": row["block"],
                    "ordinal": row["ordinal"],
                    "motion": motion,
                    "numeric_seed_uint64": row["numeric_seed_uint64"],
                    "scene_geometry_sha256": row["scene"][
                        "scene_geometry_sha256"
                    ],
                    "scene": row["scene"],
                }
            )
    if len(result) != 160:
        raise InvalidQmsR1("PREDECESSOR_COUNT")
    return result


def _cal() -> list[dict[str, Any]]:
    result = []
    for block in BLOCKS:
        for ordinal in range(CAL_SEEDS_PER_BLOCK):
            scene = build_cal_scene(block, ordinal)
            seed = derive_seed(block, ordinal)
            for motion in MOTIONS:
                result.append(
                    {
                        "cohort": "QMS_R1_DISJOINT_CAL",
                        "cluster_id": f"QMS_R1_{block}_CAL_{ordinal:02d}",
                        "block": block,
                        "ordinal": ordinal,
                        "motion": motion,
                        "numeric_seed_uint64": seed["numeric_seed_uint64"],
                        "scene_geometry_sha256": scene[
                            "scene_geometry_sha256"
                        ],
                        "scene": scene,
                    }
                )
    return result


def summarize(
    rows: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], bool]:
    expected = 20 if mode == "predecessor-dev" else CAL_SEEDS_PER_BLOCK
    summaries = []
    qualified = True
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                row for row in rows
                if row["block"] == block and row["motion"] == motion
            ]
            if len(selected) != expected:
                raise InvalidQmsR1("SUBGROUP_COUNT")
            count = sum(bool(row["sequence_pass"]) for row in selected)
            passed = (
                count >= math.ceil(0.90 * expected)
                if mode == "predecessor-dev"
                else count == expected
            )
            qualified = qualified and passed
            summaries.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": expected,
                    "pass_count": count,
                    "subgroup_pass": passed,
                }
            )
    return summaries, qualified


def _exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise InvalidQmsR1(f"OUTPUT_EXISTS:{path.name}")
    descriptor, literal = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(literal)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise InvalidQmsR1(f"OUTPUT_EXISTS:{path.name}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(root: Path, mode: str, output: Path, workers: int) -> dict[str, Any]:
    if workers != 8 or mode not in {"predecessor-dev", "new-cal"}:
        raise InvalidQmsR1("RUN_CONFIG")
    p1 = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0"
    )
    source = p1 / "all_seed_geometry_manifest.jsonl"
    trajectory_path = p1 / "trajectory_manifest.json"
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    identities = _predecessor(source) if mode == "predecessor-dev" else _cal()
    tasks = [
        (identity, trajectories[identity["block"]]) for identity in identities
    ]
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
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
    subgroups, qualified = summarize(rows, mode)
    output.mkdir(parents=True, exist_ok=True)
    identities_public = [
        {key: value for key, value in item.items() if key != "scene"}
        for item in identities
    ]
    identity_path = output / "identity_manifest.json"
    ledger_path = output / "response_blind_ledger.jsonl"
    _exclusive(identity_path, canonical_bytes(identities_public))
    _exclusive(ledger_path, b"".join(canonical_bytes(row) for row in rows))
    terminal = (
        "QMS_R1_PREDECESSOR_DEVELOPMENT_DIAGNOSTIC_COMPLETE"
        if mode == "predecessor-dev"
        else (
            "QMS_R1_MATERIAL_OPERATOR_CAL_QUALIFIED"
            if qualified
            else "QUALITY_MANIPULATION_NOT_EVALUABLE / HOLD_QMS_R1"
        )
    )
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1.v1",
        "protocol_id": QMS_ID,
        "mode": mode,
        "terminal": terminal,
        "qualified": qualified,
        "operator": operator.frozen_operator_identity(),
        "estimand": (
            "effect_of_fixed_global_material_albedo_texture_residual_"
            "contraction_alpha_0p15"
        ),
        "gates": {
            "material_residual_ratio_inclusive": list(
                MATERIAL_RESIDUAL_RATIO_RANGE
            ),
            "geometry_identity": "EXACT_SHARED_RAYCAST_PER_PAIRED_FRAME",
            "material_mean_identity": "EXACT_PREQUANTIZATION_BY_OPERATOR",
            "structure_contrast_ratio_role": "DESCRIPTIVE_ONLY",
            "full_frame_gradient_density_role": "DESCRIPTIVE_ONLY",
        },
        "counts": {
            "sequences": len(rows),
            "frame_states": len(rows) * len(FRAME_POSITIONS),
        },
        "subgroups": subgroups,
        "bindings": {
            "identity_manifest_sha256": sha256_file(identity_path),
            "response_blind_ledger_sha256": sha256_file(ledger_path),
            "operator_sha256": sha256_file(Path(operator.__file__)),
            "qualification_sha256": sha256_file(Path(__file__)),
            "generator_sha256": sha256_file(Path(geometry.__file__)),
            "quality_metrics_sha256": sha256_file(Path(quality.__file__)),
            "source_manifest_sha256": sha256_file(source),
            "trajectory_manifest_sha256": sha256_file(trajectory_path),
        },
        "firewall": {
            "r3_imported_or_executed": False,
            "r3_outcome_read": False,
            "predecessor_outcome_read": False,
            "sequence16_android_realtime": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive(output / "receipt.json", canonical_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("predecessor-dev", "new-cal"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    receipt = run(
        args.repo_root.resolve(),
        args.mode,
        args.output_dir.resolve(),
        args.workers,
    )
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "qualified": receipt["qualified"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["qualified"] or args.mode == "predecessor-dev" else 2


if __name__ == "__main__":
    raise SystemExit(main())
