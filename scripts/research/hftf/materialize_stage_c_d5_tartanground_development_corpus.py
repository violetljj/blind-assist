#!/usr/bin/env python3
"""Materialize an outcome-open TartanGround HFTF Development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import fsspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d5_tartanground_development_pilot import (  # noqa: E402
    CAMERA,
    HEIGHT_NAMES,
    HORIZON_OFFSETS,
    HORIZON_SECONDS,
    REPO_ID,
    REVISION,
    anchor_basis,
    field_from_observation,
    load_metadata,
    remote_archive,
    decode_depth,
)


DEFAULT_METADATA_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0c-provider-resolution-20260802/sentinel"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-corpus-v0"
)
SOURCES = (
    ("train", "AbandonedCable/Data_diff/P1000"),
    ("train", "CoalMine/Data_diff/P1000"),
    ("train", "Gascola/Data_diff/P1000"),
    ("train", "OldScandinavia/Data_diff/P1000"),
    ("train", "Rome/Data_diff/P1000"),
    ("train", "SeasonalForestWinterNight/Data_diff/P1000"),
    ("dev", "MiddleEast/Data_diff/P1002"),
    ("dev", "WaterMillNight/Data_diff/P1002"),
)
BLOCK_SPAN_RAW_FRAMES = 81
ANCHOR_OFFSETS = tuple(range(8, 73, 2))
HISTORY_OFFSETS = (-8, -6, -4, -2, 0)
RGB_OFFSETS = tuple(range(0, 73, 2))
DEPTH_OFFSETS = tuple(range(8, 81, 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def centered_even_block_start(num_poses: int) -> int:
    if num_poses < BLOCK_SPAN_RAW_FRAMES:
        raise ValueError("Trajectory is too short for the Development block")
    start = (num_poses - BLOCK_SPAN_RAW_FRAMES) // 2
    return start - start % 2


def member_frame_id(name: str) -> int | None:
    match = re.search(r"/(\d{6})_", name)
    return int(match.group(1)) if match else None


def fetch_frames(
    parent_id: str,
    modality: str,
    frame_ids: list[int],
    output_directory: Path,
) -> None:
    wanted = set(frame_ids)
    output_directory.mkdir(parents=True, exist_ok=True)
    missing_local = {
        frame_id
        for frame_id in wanted
        if not (output_directory / f"{frame_id:06d}.png").exists()
    }
    if not missing_local:
        return
    with fsspec.open(
        remote_archive(parent_id, modality),
        "rb",
        block_size=1024 * 1024,
    ) as source:
        with zipfile.ZipFile(source) as zipped:
            members = {}
            for name in zipped.namelist():
                frame_id = member_frame_id(name)
                if frame_id in missing_local:
                    members[frame_id] = name
            missing_remote = sorted(missing_local - set(members))
            if missing_remote:
                raise ValueError(
                    f"{parent_id} {modality} missing frames {missing_remote}"
                )
            for frame_id in sorted(missing_local):
                destination = output_directory / f"{frame_id:06d}.png"
                temporary = destination.with_suffix(".png.tmp")
                temporary.write_bytes(zipped.read(members[frame_id]))
                temporary.replace(destination)


def nullable_risk(
    known: np.ndarray,
    risk: np.ndarray,
) -> list[list[list[float | None]]]:
    if known.shape != risk.shape or known.shape != (6, 6, 3):
        raise ValueError("Expected [direction,distance,height] field")
    output: list[list[list[float | None]]] = []
    height_first_known = np.transpose(known, (2, 0, 1))
    height_first_risk = np.transpose(risk, (2, 0, 1))
    for height_index in range(3):
        height_rows = []
        for direction_index in range(6):
            height_rows.append(
                [
                    (
                        float(height_first_risk[
                            height_index, direction_index, distance_index
                        ])
                        if height_first_known[
                            height_index, direction_index, distance_index
                        ]
                        else None
                    )
                    for distance_index in range(6)
                ]
            )
        output.append(height_rows)
    return output


def label_record(
    known: np.ndarray,
    risk: np.ndarray,
) -> dict[str, Any]:
    return {
        "known_target": np.transpose(known, (2, 0, 1))
        .astype(np.uint8)
        .tolist(),
        "risk_score_target_nullable": nullable_risk(known, risk),
    }


def materialize_source(
    metadata_root: Path,
    output_root: Path,
    role: str,
    parent_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, poses = load_metadata(metadata_root, parent_id)
    num_poses = int(metadata["num_poses"])
    start = centered_even_block_start(num_poses)
    source_root = output_root / "media" / role / parent_id
    rgb_ids = [start + offset for offset in RGB_OFFSETS]
    depth_ids = [start + offset for offset in DEPTH_OFFSETS]
    fetch_frames(parent_id, "image", rgb_ids, source_root / "image")
    fetch_frames(parent_id, "depth", depth_ids, source_root / "depth")

    robot_height = float(metadata["robot_height"])
    depth_cache: dict[int, np.ndarray] = {}

    def depth(frame_id: int) -> np.ndarray:
        value = depth_cache.get(frame_id)
        if value is None:
            value = decode_depth(
                (source_root / "depth" / f"{frame_id:06d}.png").read_bytes()
            )
            depth_cache[frame_id] = value
        return value

    records = []
    for anchor_offset in ANCHOR_OFFSETS:
        anchor_frame = start + anchor_offset
        basis = anchor_basis(poses[anchor_frame], robot_height)
        labels = {}
        for horizon_name, future_offset in HORIZON_OFFSETS.items():
            observation_frame = anchor_frame + future_offset
            known, risk = field_from_observation(
                depth(observation_frame),
                poses[observation_frame],
                basis,
                HORIZON_SECONDS[horizon_name],
            )
            labels[horizon_name] = label_record(known, risk)
        history = []
        for relative_offset in HISTORY_OFFSETS:
            frame_id = anchor_frame + relative_offset
            image_path = source_root / "image" / f"{frame_id:06d}.png"
            history.append(
                {
                    "relative_time_s": relative_offset / 10.0,
                    "frame_id": frame_id,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": sha256(image_path),
                }
            )
        records.append(
            {
                "sample_id": (
                    f"{parent_id.replace('/', '__')}__{anchor_frame:06d}"
                ),
                "role": role,
                "parent_id": parent_id,
                "environment": parent_id.split("/", 1)[0],
                "anchor_frame_id": anchor_frame,
                "history_rgb": history,
                "labels": labels,
            }
        )

    source = {
        "role": role,
        "parent_id": parent_id,
        "environment": parent_id.split("/", 1)[0],
        "num_poses": num_poses,
        "robot_height_m": robot_height,
        "block_start_frame_id": start,
        "block_end_frame_id": start + BLOCK_SPAN_RAW_FRAMES - 1,
        "sample_count": len(records),
        "rgb_frame_count": len(rgb_ids),
        "depth_frame_count": len(depth_ids),
    }
    return source, records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--prefetch-parent",
        action="append",
        default=[],
        help=(
            "Materialize only the named configured parent(s), then stop "
            "without replacing the corpus manifest or samples JSONL."
        ),
    )
    args = parser.parse_args()

    if args.prefetch_parent:
        configured = {parent_id: role for role, parent_id in SOURCES}
        unknown = sorted(set(args.prefetch_parent) - set(configured))
        if unknown:
            parser.error(f"Unknown configured parent(s): {unknown}")
        for parent_id in args.prefetch_parent:
            source, _ = materialize_source(
                args.metadata_root,
                args.output_root,
                configured[parent_id],
                parent_id,
            )
            print(
                json.dumps(
                    {
                        "parent_id": parent_id,
                        "role": source["role"],
                        "samples": source["sample_count"],
                        "prefetch_only": True,
                    }
                ),
                flush=True,
            )
        return 0

    all_records = []
    source_rows = []
    for role, parent_id in SOURCES:
        source, records = materialize_source(
            args.metadata_root,
            args.output_root,
            role,
            parent_id,
        )
        source_rows.append(source)
        all_records.extend(records)
        print(
            json.dumps(
                {
                    "parent_id": parent_id,
                    "role": role,
                    "samples": len(records),
                }
            ),
            flush=True,
        )

    samples_path = args.output_root / "samples.jsonl"
    samples_temporary = samples_path.with_suffix(".jsonl.tmp")
    samples_temporary.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in all_records
        ),
        encoding="utf-8",
    )
    samples_temporary.replace(samples_path)
    manifest = {
        "schema": "blindassist_hftf_stage_c_d5_tartanground_development_corpus_v0",
        "status": "DEVELOPMENT_CORPUS_MATERIALIZED",
        "provider": {"repo_id": REPO_ID, "revision": REVISION},
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "fresh_validation_claimed": False,
        },
        "field": {
            "horizons_s": HORIZON_SECONDS,
            "height_order": HEIGHT_NAMES,
            "shape_per_horizon": [3, 6, 6],
        },
        "sources": source_rows,
        "source_count_by_role": {
            role: sum(row["role"] == role for row in source_rows)
            for role in ("train", "dev")
        },
        "sample_count_by_role": {
            role: sum(record["role"] == role for record in all_records)
            for role in ("train", "dev")
        },
        "samples": {
            "path": str(samples_path.resolve()),
            "bytes": samples_path.stat().st_size,
            "sha256": sha256(samples_path),
        },
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["sample_count_by_role"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
