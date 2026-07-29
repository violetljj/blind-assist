"""Response-blind qualification for the QMS-R0 low-texture successor.

The predecessor's full-frame gradient ratio mixed material texture with object
boundaries.  QMS-R0 instead measures texture only inside source-known material
regions and separately guards source-known object-boundary contrast.

This module must never import the R3 pair core, transport, trigger, or outcome
analysis implementations.
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
from . import linear_bilateral_texture_r0 as operator
from . import quality_interventions_r0 as quality


QMS_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_MANIPULATION_SUCCESSOR_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
FRAME_POSITIONS = (
    0, 40, 80, 120, 160, 200, 240, 280,
    320, 360, 400, 440, 480, 520, 560, 601,
)
INTERIOR_GRADIENT_RANGE = (0.35, 0.55)
STRUCTURE_CONTRAST_RANGE = (0.90, 1.10)
INTERIOR_EROSION_SIZE = 9
CAL_SEEDS_PER_BLOCK = 4


class InvalidQmsQualification(ValueError):
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


def derive_qms_seed(namespace: str, block: str, ordinal: int) -> dict[str, Any]:
    if namespace not in {"CAL"} or block not in BLOCKS or ordinal not in range(100):
        raise InvalidQmsQualification("SEED_DOMAIN")
    token = f"{QMS_ID}|{namespace}|{block}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def build_qms_scene(block: str, ordinal: int) -> dict[str, Any]:
    seed = derive_qms_seed("CAL", block, ordinal)
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        scene = geometry.build_scene(block, ordinal, "QMS_CAL")
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


def material_interior_mask(
    object_id: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    if object_id.shape != valid_mask.shape:
        raise InvalidQmsQualification("MASK_SHAPE")
    kernel = np.ones(
        (INTERIOR_EROSION_SIZE, INTERIOR_EROSION_SIZE), dtype=np.uint8
    )
    interior = np.zeros(object_id.shape, dtype=np.uint8)
    for identifier in np.unique(object_id[valid_mask]):
        region = ((object_id == identifier) & valid_mask).astype(np.uint8)
        interior |= cv2.erode(
            region,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    result = interior.astype(bool)
    if int(np.count_nonzero(result)) < 1000:
        raise InvalidQmsQualification("INSUFFICIENT_MATERIAL_INTERIOR")
    return result


def structure_edge_contrast_ratio(
    clean_luminance: np.ndarray,
    degraded_luminance: np.ndarray,
    object_id: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, int]:
    clean_values: list[np.ndarray] = []
    degraded_values: list[np.ndarray] = []
    horizontal = (
        valid_mask[:, :-1]
        & valid_mask[:, 1:]
        & (object_id[:, :-1] != object_id[:, 1:])
    )
    vertical = (
        valid_mask[:-1, :]
        & valid_mask[1:, :]
        & (object_id[:-1, :] != object_id[1:, :])
    )
    clean_values.extend(
        (
            np.abs(clean_luminance[:, 1:] - clean_luminance[:, :-1])[
                horizontal
            ],
            np.abs(clean_luminance[1:, :] - clean_luminance[:-1, :])[
                vertical
            ],
        )
    )
    degraded_values.extend(
        (
            np.abs(degraded_luminance[:, 1:] - degraded_luminance[:, :-1])[
                horizontal
            ],
            np.abs(degraded_luminance[1:, :] - degraded_luminance[:-1, :])[
                vertical
            ],
        )
    )
    clean = np.concatenate(clean_values)
    degraded = np.concatenate(degraded_values)
    selected = np.isfinite(clean) & np.isfinite(degraded) & (clean >= 0.02)
    if int(np.count_nonzero(selected)) < 32:
        raise InvalidQmsQualification("INSUFFICIENT_SOURCE_KNOWN_EDGES")
    per_edge_ratio = degraded[selected] / clean[selected]
    finite = per_edge_ratio[np.isfinite(per_edge_ratio)]
    if finite.size < 32:
        raise InvalidQmsQualification("NONFINITE_SOURCE_KNOWN_EDGE_RATIO")
    return float(np.median(finite)), int(finite.size)


def _poses(
    trajectory: dict[str, Any], motion: str
) -> list[dict[str, Any]]:
    if motion == "PERIODIC_6DOF_SELF_MOTION":
        return trajectory["poses"]
    if motion != "STATIC_CAMERA":
        raise InvalidQmsQualification("MOTION")
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
    identity: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    motion = identity["motion"]
    poses = _poses(trajectory, motion)
    frame_rows: list[dict[str, Any]] = []
    static_cache: dict[str, np.ndarray] | None = None
    for frame_index in FRAME_POSITIONS:
        pose = poses[frame_index]
        if motion == "STATIC_CAMERA" and static_cache is not None:
            clean = static_cache
        else:
            clean = geometry.render(
                identity["scene"],
                np.asarray(pose["rotation_matrix"], dtype=np.float64),
                np.asarray(pose["translation_m"], dtype=np.float64),
            )
            if motion == "STATIC_CAMERA":
                static_cache = clean
        valid = np.isfinite(clean["depth"])
        degraded_rgb = operator.apply_linear_bilateral_texture(clean["rgb"])
        clean_y = quality.linear_luminance(clean["rgb"])
        degraded_y = quality.linear_luminance(degraded_rgb)
        interior = material_interior_mask(clean["object_id"], valid)
        clean_density, degraded_density = (
            quality.multiscale_gradient_density_pair(
                clean_y, degraded_y, interior
            )
        )
        gradient_ratio = quality.safe_ratio(
            degraded_density, clean_density, "QMS_INTERIOR_GRADIENT"
        )
        structure_ratio, edge_count = structure_edge_contrast_ratio(
            clean_y, degraded_y, clean["object_id"], valid
        )
        frame_rows.append(
            {
                "frame_index": frame_index,
                "clean_rgb_sha256": hashlib.sha256(
                    clean["rgb"].tobytes()
                ).hexdigest(),
                "degraded_rgb_sha256": hashlib.sha256(
                    degraded_rgb.tobytes()
                ).hexdigest(),
                "valid_mask_sha256": hashlib.sha256(
                    valid.astype(np.uint8).tobytes()
                ).hexdigest(),
                "object_id_sha256": hashlib.sha256(
                    clean["object_id"].astype("<i4").tobytes()
                ).hexdigest(),
                "material_interior_mask_sha256": hashlib.sha256(
                    interior.astype(np.uint8).tobytes()
                ).hexdigest(),
                "material_interior_gradient_density_ratio": gradient_ratio,
                "source_known_structure_contrast_ratio": structure_ratio,
                "source_known_structure_edge_count": edge_count,
            }
        )
    gradient_median = float(
        np.median(
            [
                row["material_interior_gradient_density_ratio"]
                for row in frame_rows
            ]
        )
    )
    structure_median = float(
        np.median(
            [
                row["source_known_structure_contrast_ratio"]
                for row in frame_rows
            ]
        )
    )
    passed = bool(
        INTERIOR_GRADIENT_RANGE[0]
        <= gradient_median
        <= INTERIOR_GRADIENT_RANGE[1]
        and STRUCTURE_CONTRAST_RANGE[0]
        <= structure_median
        <= STRUCTURE_CONTRAST_RANGE[1]
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
        "frame_rows": frame_rows,
        "sequence_medians": {
            "material_interior_gradient_density_ratio": gradient_median,
            "source_known_structure_contrast_ratio": structure_median,
        },
        "sequence_pass": passed,
    }


def _worker(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return evaluate_sequence(*task)


def predecessor_identities(source_path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = []
    for record in records:
        if record.get("record_type") != "main_cluster":
            continue
        for motion in MOTIONS:
            result.append(
                {
                    "cohort": "CONSUMED_PREDECESSOR_DEVELOPMENT_ONLY",
                    "cluster_id": record["cluster_id"],
                    "block": record["block"],
                    "ordinal": record["ordinal"],
                    "motion": motion,
                    "numeric_seed_uint64": record["numeric_seed_uint64"],
                    "scene_geometry_sha256": record["scene"][
                        "scene_geometry_sha256"
                    ],
                    "scene": record["scene"],
                }
            )
    if len(result) != 160:
        raise InvalidQmsQualification("PREDECESSOR_IDENTITY_COUNT")
    return result


def new_cal_identities() -> list[dict[str, Any]]:
    result = []
    for block in BLOCKS:
        for ordinal in range(CAL_SEEDS_PER_BLOCK):
            scene = build_qms_scene(block, ordinal)
            seed = derive_qms_seed("CAL", block, ordinal)
            for motion in MOTIONS:
                result.append(
                    {
                        "cohort": "QMS_R0_DISJOINT_CAL",
                        "cluster_id": f"QMS_R0_{block}_CAL_{ordinal:02d}",
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
    summaries = []
    qualification = True
    expected = 20 if mode == "predecessor-dev" else CAL_SEEDS_PER_BLOCK
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                row for row in rows
                if row["block"] == block and row["motion"] == motion
            ]
            if len(selected) != expected:
                raise InvalidQmsQualification("SUBGROUP_COUNT")
            pass_count = sum(bool(row["sequence_pass"]) for row in selected)
            subgroup_pass = (
                pass_count == expected
                if mode == "new-cal"
                else pass_count >= math.ceil(0.90 * expected)
            )
            qualification = qualification and subgroup_pass
            summaries.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": expected,
                    "pass_count": pass_count,
                    "subgroup_pass": subgroup_pass,
                }
            )
    return summaries, qualification


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise InvalidQmsQualification(f"OUTPUT_EXISTS:{path.name}")
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
            raise InvalidQmsQualification(f"OUTPUT_EXISTS:{path.name}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    repo_root: Path, mode: str, output_dir: Path, workers: int
) -> dict[str, Any]:
    if workers != 8 or mode not in {"predecessor-dev", "new-cal"}:
        raise InvalidQmsQualification("RUN_CONFIGURATION")
    p1_dir = (
        repo_root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0"
    )
    source_path = p1_dir / "all_seed_geometry_manifest.jsonl"
    trajectory_path = p1_dir / "trajectory_manifest.json"
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    identities = (
        predecessor_identities(source_path)
        if mode == "predecessor-dev"
        else new_cal_identities()
    )
    public_identities = [
        {key: value for key, value in item.items() if key != "scene"}
        for item in identities
    ]
    tasks = [
        (identity, trajectories[identity["block"]]) for identity in identities
    ]
    rows: list[dict[str, Any]] = []
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
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity_manifest.json"
    ledger_path = output_dir / "response_blind_ledger.jsonl"
    _atomic(identity_path, canonical_bytes(public_identities))
    _atomic(
        ledger_path, b"".join(canonical_bytes(row) for row in rows)
    )
    terminal = (
        "QMS_R0_PREDECESSOR_DEVELOPMENT_DIAGNOSTIC_COMPLETE"
        if mode == "predecessor-dev"
        else (
            "QMS_R0_BILATERAL_CAL_QUALIFIED"
            if qualified
            else "QUALITY_MANIPULATION_NOT_EVALUABLE / HOLD_QMS_R0"
        )
    )
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r0.v1",
        "protocol_id": QMS_ID,
        "mode": mode,
        "terminal": terminal,
        "qualified": qualified,
        "operator": operator.frozen_operator_identity(),
        "gates": {
            "material_interior_gradient_density_ratio_inclusive": list(
                INTERIOR_GRADIENT_RANGE
            ),
            "source_known_structure_contrast_ratio_inclusive": list(
                STRUCTURE_CONTRAST_RANGE
            ),
            "new_cal_required_sequence_pass_per_subgroup": "4/4",
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
            "qualification_implementation_sha256": sha256_file(Path(__file__)),
            "generator_sha256": sha256_file(Path(geometry.__file__)),
            "quality_metrics_sha256": sha256_file(Path(quality.__file__)),
            "source_manifest_sha256": sha256_file(source_path),
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
    _atomic(output_dir / "receipt.json", canonical_bytes(receipt))
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
