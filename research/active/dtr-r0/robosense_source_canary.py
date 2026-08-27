"""Audit RoboSense metadata and raw-source granularity for DTR.

The official validation metadata is a pickle.  This canary verifies its LFS
digest and uses a restricted unpickler that admits only NumPy's inert array
constructors listed by the Hugging Face pickle scan.  It does not execute the
dataset repository or use future occupancy as an observation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pickle
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

HF_BASE = "https://huggingface.co/api/datasets/suhaisheng0527/RoboSense/tree/main"
USER_AGENT = "BlindAssist-RoboSense-source-canary/1.0"
LOCAL_VAL_PATH = "splits/robosense_local_val.pkl"


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load plain containers and NumPy arrays, rejecting every other global."""

    def find_class(self, module: str, name: str) -> Any:
        allowed: dict[tuple[str, str], Any] = {
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from exc


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        value = RestrictedNumpyUnpickler(stream).load()
    if isinstance(value, dict) and isinstance(value.get("infos"), list):
        value = value["infos"]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("expected a list of frame dictionaries")
    return value


def sequence_values(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def first_present(item: dict[str, Any], candidates: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in candidates:
        if name in item and item[name] is not None:
            return name, item[name]
    return None, None


def audit_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    key_coverage: collections.Counter[str] = collections.Counter()
    class_counts: collections.Counter[str] = collections.Counter()
    sequence_counts: collections.Counter[str] = collections.Counter()
    track_id_fields: collections.Counter[str] = collections.Counter()
    pose_fields: collections.Counter[str] = collections.Counter()
    near_field_boxes = 0
    total_boxes = 0
    frames_with_track_ids = 0
    track_positions: collections.defaultdict[tuple[str, str], list[int]] = collections.defaultdict(list)

    pose_candidates = (
        "hs_enu_pose",
        "ego2global",
        "ego2global_rotation",
        "ego2global_translation",
    )

    sequence_order: collections.defaultdict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in frames:
        sequence_order[str(item.get("seq_token", "<missing>"))].append(item)

    ordered_frames: list[tuple[str, int, dict[str, Any]]] = []
    for sequence, items in sequence_order.items():
        for position, item in enumerate(sorted(items, key=lambda value: int(value.get("timestamp", 0)))):
            ordered_frames.append((sequence, position, item))

    for sequence, position, item in ordered_frames:
        key_coverage.update(item.keys())
        sequence_counts[sequence] += 1

        annos = item.get("annos") if isinstance(item.get("annos"), dict) else {}
        track_values = annos.get("id")
        track_ids = tuple(sequence_values(track_values))
        if track_ids:
            track_id_fields["annos.id"] += 1
            frames_with_track_ids += 1
            for track_id in track_ids:
                track_positions[(sequence, str(track_id))].append(position)

        for pose_name in pose_candidates:
            if item.get(pose_name) is not None:
                pose_fields[pose_name] += 1

        names = [str(value) for value in sequence_values(annos.get("name"))]
        class_counts.update(names)
        locations = annos.get("location")
        if isinstance(locations, np.ndarray) and locations.ndim == 2 and locations.shape[1] >= 3:
            total_boxes += int(locations.shape[0])
            near_field_boxes += int(
                np.count_nonzero(np.linalg.norm(locations[:, :3], axis=1) <= 5.0)
            )
        elif locations is not None:
            total_boxes += len(tuple(sequence_values(locations)))

    reappearance_gaps = 0
    tracks_with_reappearance_gap = 0
    for positions in track_positions.values():
        gaps = sum(right - left > 1 for left, right in itertools.pairwise(positions))
        if gaps:
            tracks_with_reappearance_gap += 1
            reappearance_gaps += gaps

    frame_count = len(frames)
    path_coverage = {
        name: key_coverage[name]
        for name in (
            "hs64_path",
            "velodyne_path",
            "occ_label_path",
            "timestamp",
            "seq_token",
            "map_token",
        )
    }
    sequence_histogram = collections.Counter(sequence_counts.values())
    return {
        "frame_count": frame_count,
        "sequence_count": sum(name != "<missing>" for name in sequence_counts),
        "sequence_length_histogram": {
            str(length): count for length, count in sorted(sequence_histogram.items())
        },
        "largest_sequences": dict(sequence_counts.most_common(20)),
        "sample_top_level_keys": sorted(frames[0].keys()) if frames else [],
        "path_and_identity_coverage": path_coverage,
        "pose_field_coverage": dict(pose_fields),
        "track_id_field_coverage": dict(track_id_fields),
        "frames_with_nonempty_track_ids": frames_with_track_ids,
        "unique_tracks_within_sequence": len(track_positions),
        "tracks_with_reappearance_gap": tracks_with_reappearance_gap,
        "reappearance_gap_count": reappearance_gaps,
        "total_boxes": total_boxes,
        "near_field_boxes_3d_radial_le_5m": near_field_boxes,
        "class_counts": dict(class_counts.most_common()),
        "near_field_note": (
            "3-D radial distance is coordinate-rotation invariant and is used only for "
            "source admission. Route-tube positives require an explicit sensor-axis adapter."
        ),
    }


def tree(path: str) -> list[dict[str, Any]]:
    value = fetch_json(f"{HF_BASE}/{path}?recursive=false&expand=true&limit=100")
    if not isinstance(value, list):
        raise TypeError(f"expected list for Hugging Face tree {path}")
    return [item for item in value if isinstance(item, dict)]


def run(metadata_path: Path) -> dict[str, Any]:
    split_files = tree("splits")
    dataset_files = tree("dataset")
    expected = next((item for item in split_files if item.get("path") == LOCAL_VAL_PATH), None)
    if expected is None:
        raise ValueError(f"official tree has no {LOCAL_VAL_PATH}")
    expected_size = int(expected.get("size", -1))
    expected_sha256 = str((expected.get("lfs") or {}).get("oid", ""))
    actual_size = metadata_path.stat().st_size
    actual_sha256 = sha256_file(metadata_path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError("local metadata does not match official LFS size/SHA-256")

    frames = load_metadata(metadata_path)
    frame_audit = audit_frames(frames)

    lidar_trainval_parts = [
        item for item in dataset_files if str(item.get("path", "")).startswith("dataset/lidar_occ_trainval_part_")
    ]
    image_trainval_parts = [
        item for item in dataset_files if str(item.get("path", "")).startswith("dataset/image_trainval_part_")
    ]
    lidar_trainval_bytes = sum(int(item.get("size", 0)) for item in lidar_trainval_parts)
    image_trainval_bytes = sum(int(item.get("size", 0)) for item in image_trainval_parts)

    metadata_admitted = bool(
        frame_audit["frame_count"]
        and frame_audit["total_boxes"]
        and frame_audit["frames_with_nonempty_track_ids"]
        and (
            frame_audit["path_and_identity_coverage"]["hs64_path"]
            or frame_audit["path_and_identity_coverage"]["velodyne_path"]
        )
    )
    raw_minimal_shard_admitted = False
    residual_source_admitted = metadata_admitted and raw_minimal_shard_admitted

    return {
        "schema_version": "blindassist-dtr-robosense-source-canary-v1",
        "source": {
            "dataset": "suhaisheng0527/RoboSense",
            "metadata_path": str(metadata_path.resolve()),
            "metadata_size_bytes": actual_size,
            "metadata_sha256": actual_sha256,
            "huggingface_security_status": expected.get("securityFileStatus"),
        },
        "metadata_audit": frame_audit,
        "raw_payload": {
            "lidar_occ_trainval_part_count": len(lidar_trainval_parts),
            "lidar_occ_trainval_total_bytes": lidar_trainval_bytes,
            "smallest_lidar_occ_part_bytes": min(
                (int(item.get("size", 0)) for item in lidar_trainval_parts), default=None
            ),
            "minimum_complete_lidar_occ_download_bytes": lidar_trainval_bytes,
            "image_trainval_part_count": len(image_trainval_parts),
            "image_trainval_total_bytes": image_trainval_bytes,
            "packaging": (
                "Split pieces of a combined gzip tar stream; the official README "
                "requires concatenation before extraction."
            ),
        },
        "checks": {
            "metadata_admitted": metadata_admitted,
            "raw_minimal_shard_admitted": raw_minimal_shard_admitted,
            "detector_independent_residual_source_admitted": residual_source_admitted,
            "future_occupancy_input_allowed": False,
        },
        "verdict": (
            "ROBOSENSE_METADATA_ADMITTED_RAW_RESIDUAL_NOT_ADMITTED"
            if metadata_admitted and not residual_source_admitted
            else "ROBOSENSE_RESIDUAL_SOURCE_ADMITTED"
            if residual_source_admitted
            else "ROBOSENSE_SOURCE_NOT_ADMITTED"
        ),
        "decision": (
            "Keep RoboSense as a metadata/privileged-box reserve, but do not start a "
            "raw residual-occupancy experiment from this distribution channel. "
            "Change to a source with independently downloadable sensor logs."
            if metadata_admitted and not residual_source_admitted
            else "A fixed raw-sensor shard may proceed."
            if residual_source_admitted
            else "Do not use this source until metadata and raw-sensor access are admitted."
        ),
        "claim_ceiling": (
            "Metadata compatibility and near-field label availability only. Official "
            "multi-frame occupancy remains evaluator-only; no DTR gain or safety claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
