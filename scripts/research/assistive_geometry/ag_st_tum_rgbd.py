"""TUM RGB-D source adapter for the AG-ST third-domain selector canary."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
COHORT_SCHEMA = "blindassist_ag_st_tum_rgbd_third_domain_cohort_v1"
DEFAULT_TUM_COHORT_MANIFEST = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_RGBD_THIRD_DOMAIN_COHORT_R0_2026-08-10.json"
)
TUM_HEIGHT = 480
TUM_WIDTH = 640
TUM_DEPTH_SCALE = 5000.0
MAX_ASSOCIATION_DELTA_SECONDS = 0.02
MAX_POSE_BRACKETING_GAP_SECONDS = 0.10


@dataclass(frozen=True)
class TumIndexRow:
    row_index: int
    timestamp_seconds: float
    relative_path: str


@dataclass(frozen=True)
class TumPoseRow:
    timestamp_seconds: float
    translation_xyz: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class TumSelectedPayload:
    parent_id: str
    role: str
    rgb: TumIndexRow
    depth: TumIndexRow
    intrinsics: np.ndarray
    storage_kind: str
    source_path: Path
    rgb_path: Path | None
    depth_path: Path | None
    rgb_bytes: bytes | None
    depth_bytes: bytes | None
    camera_to_world: np.ndarray
    pose_bracketing_gap_seconds: float

    @property
    def association_delta_seconds(self) -> float:
        return abs(self.rgb.timestamp_seconds - self.depth.timestamp_seconds)

    def load_rgb(self) -> np.ndarray:
        source: Any = io.BytesIO(self.rgb_bytes) if self.rgb_bytes is not None else self.rgb_path
        require(source is not None, "TUM RGB payload missing")
        with Image.open(source) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        require(value.shape == (TUM_HEIGHT, TUM_WIDTH, 3), "TUM RGB shape drift")
        return value

    def load_depth(self) -> tuple[np.ndarray, np.ndarray]:
        source: Any = (
            io.BytesIO(self.depth_bytes) if self.depth_bytes is not None else self.depth_path
        )
        require(source is not None, "TUM depth payload missing")
        with Image.open(source) as image:
            raw = np.asarray(image).copy()
        require(raw.shape == (TUM_HEIGHT, TUM_WIDTH), "TUM depth shape drift")
        require(raw.dtype == np.uint16, f"TUM depth dtype drift: {raw.dtype}")
        valid = raw > 0
        return raw.astype(np.float32) / TUM_DEPTH_SCALE, valid


def load_tum_cohort(path: Path, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "TUM cohort root invalid")
    require(payload.get("schema") == COHORT_SCHEMA, "TUM cohort schema drift")
    require(role in {"fit", "evaluation"}, "unsupported TUM cohort role")
    rows = payload.get(f"{role}_parents")
    expected = 4 if role == "fit" else 3
    require(isinstance(rows, list) and len(rows) == expected, "TUM cohort role drift")
    require(
        len({str(row["parent_id"]) for row in rows}) == expected,
        "duplicate TUM cohort parent",
    )
    return payload, rows


def parse_tum_index(text: str) -> list[TumIndexRow]:
    rows: list[TumIndexRow] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        require(len(parts) == 2, "malformed TUM index row")
        rows.append(TumIndexRow(len(rows), float(parts[0]), parts[1].replace("\\", "/")))
    require(rows, "TUM index empty")
    return rows


def parse_tum_poses(text: str) -> list[TumPoseRow]:
    rows: list[TumPoseRow] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        require(len(parts) == 8, "malformed TUM groundtruth row")
        values = [float(value) for value in parts]
        translation = np.asarray(values[1:4], dtype=np.float64)
        quaternion = np.asarray(values[4:8], dtype=np.float64)
        require(
            np.all(np.isfinite(translation)) and np.all(np.isfinite(quaternion)),
            "non-finite TUM pose",
        )
        norm = float(np.linalg.norm(quaternion))
        require(norm > 1e-9, "zero-norm TUM quaternion")
        rows.append(
            TumPoseRow(
                timestamp_seconds=values[0],
                translation_xyz=translation,
                quaternion_xyzw=quaternion / norm,
            )
        )
    require(rows, "TUM groundtruth empty")
    # One official TUM sequence contains a duplicated timestamp with a
    # sub-millimetre pose difference.  Keep its first published row so the
    # interpolation domain remains deterministic and strictly increasing.
    unique: dict[float, TumPoseRow] = {}
    for row in rows:
        unique.setdefault(row.timestamp_seconds, row)
    return [unique[timestamp] for timestamp in sorted(unique)]


def interpolate_camera_to_world(
    rows: list[TumPoseRow],
    timestamp_seconds: float,
    maximum_gap_seconds: float = MAX_POSE_BRACKETING_GAP_SECONDS,
) -> tuple[np.ndarray, float]:
    timestamps = [row.timestamp_seconds for row in rows]
    right = bisect.bisect_left(timestamps, timestamp_seconds)
    if right < len(rows) and abs(rows[right].timestamp_seconds - timestamp_seconds) <= 1e-9:
        left = right
    else:
        require(0 < right < len(rows), "TUM RGB timestamp outside groundtruth span")
        left = right - 1
    first = rows[left]
    second = rows[right]
    gap = second.timestamp_seconds - first.timestamp_seconds
    require(
        gap >= 0 and gap <= maximum_gap_seconds + 1e-9,
        "TUM pose bracketing gap too large",
    )
    if left == right:
        translation = first.translation_xyz
        rotation = Rotation.from_quat(first.quaternion_xyzw).as_matrix()
    else:
        fraction = (timestamp_seconds - first.timestamp_seconds) / gap
        translation = (
            first.translation_xyz
            + fraction * (second.translation_xyz - first.translation_xyz)
        )
        rotations = Rotation.from_quat(
            np.stack((first.quaternion_xyzw, second.quaternion_xyzw), axis=0)
        )
        rotation = Slerp(
            [first.timestamp_seconds, second.timestamp_seconds], rotations
        )([timestamp_seconds]).as_matrix()[0]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose, float(gap)


def pair_rgb_depth_unique(
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
) -> dict[int, TumIndexRow]:
    depth_times = [row.timestamp_seconds for row in depth_rows]
    candidates: list[tuple[float, int, int]] = []
    for rgb in rgb_rows:
        lower = bisect.bisect_left(
            depth_times,
            rgb.timestamp_seconds - MAX_ASSOCIATION_DELTA_SECONDS,
        )
        upper = bisect.bisect_right(
            depth_times,
            rgb.timestamp_seconds + MAX_ASSOCIATION_DELTA_SECONDS,
        )
        for index in range(lower, upper):
            depth = depth_rows[index]
            candidates.append(
                (
                    abs(rgb.timestamp_seconds - depth.timestamp_seconds),
                    rgb.row_index,
                    depth.row_index,
                )
            )
    candidates.sort()
    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    result: dict[int, TumIndexRow] = {}
    depth_by_index = {row.row_index: row for row in depth_rows}
    for _, rgb_index, depth_index in candidates:
        if rgb_index in used_rgb or depth_index in used_depth:
            continue
        used_rgb.add(rgb_index)
        used_depth.add(depth_index)
        result[rgb_index] = depth_by_index[depth_index]
    return result


def _intrinsics_matrix(values: list[float]) -> np.ndarray:
    require(len(values) == 4, "TUM intrinsics drift")
    fx, fy, cx, cy = (float(value) for value in values)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _tar_member_map(archive: tarfile.TarFile, parent_id: str) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    prefix = f"{parent_id}/"
    for member in archive.getmembers():
        name = member.name.replace("\\", "/").lstrip("./")
        if member.isfile() and name.startswith(prefix):
            result[name[len(prefix) :]] = member
    require("rgb.txt" in result and "depth.txt" in result, "TUM archive metadata missing")
    return result


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = archive.extractfile(member)
    require(stream is not None, "TUM archive member unreadable")
    return stream.read()


def _validate_selected_pairs(
    row: dict[str, Any],
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
) -> list[tuple[TumIndexRow, TumIndexRow]]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    require(len(pairing) == int(row["unique_pair_count"]), "TUM unique pair count drift")
    rgb_indices = [int(value) for value in row["rgb_row_indices_zero_based"]]
    depth_indices = [int(value) for value in row["depth_row_indices_zero_based"]]
    require(len(rgb_indices) == len(depth_indices) == 3, "TUM selected frame drift")
    rgb_by_index = {value.row_index: value for value in rgb_rows}
    selected: list[tuple[TumIndexRow, TumIndexRow]] = []
    for rgb_index, depth_index in zip(rgb_indices, depth_indices, strict=True):
        require(rgb_index in pairing and rgb_index in rgb_by_index, "TUM selected RGB unpaired")
        require(pairing[rgb_index].row_index == depth_index, "TUM selected pair identity drift")
        selected.append((rgb_by_index[rgb_index], pairing[rgb_index]))
    return selected


def load_tum_role_payloads(
    manifest_path: Path,
    role: str,
) -> tuple[list[TumSelectedPayload], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    _, rows = load_tum_cohort(manifest_path, role)
    payloads: list[TumSelectedPayload] = []
    source_receipts: list[dict[str, Any]] = []
    for row in rows:
        parent_id = str(row["parent_id"])
        storage_kind = str(row["storage_kind"])
        source_path = (REPO_ROOT / str(row["source_path"])).resolve()
        intrinsics = _intrinsics_matrix(list(row["intrinsics_fx_fy_cx_cy"]))
        if storage_kind == "directory":
            require(source_path.is_dir(), f"TUM source directory missing: {parent_id}")
            rgb_index_path = source_path / "rgb.txt"
            depth_index_path = source_path / "depth.txt"
            groundtruth_path = source_path / "groundtruth.txt"
            require(
                sha256_file(rgb_index_path) == str(row["rgb_index_sha256"]),
                f"TUM RGB index hash drift: {parent_id}",
            )
            require(
                sha256_file(depth_index_path) == str(row["depth_index_sha256"]),
                f"TUM depth index hash drift: {parent_id}",
            )
            require(groundtruth_path.is_file(), f"TUM groundtruth missing: {parent_id}")
            if row.get("groundtruth_index_sha256") is not None:
                require(
                    sha256_file(groundtruth_path)
                    == str(row["groundtruth_index_sha256"]),
                    f"TUM groundtruth hash drift: {parent_id}",
                )
            rgb_rows = parse_tum_index(rgb_index_path.read_text(encoding="utf-8"))
            depth_rows = parse_tum_index(depth_index_path.read_text(encoding="utf-8"))
            pose_rows = parse_tum_poses(groundtruth_path.read_text(encoding="utf-8"))
            available_rgb = [value for value in rgb_rows if (source_path / value.relative_path).is_file()]
            available_depth = [
                value for value in depth_rows if (source_path / value.relative_path).is_file()
            ]
            selected = _validate_selected_pairs(row, available_rgb, available_depth)
            for rgb, depth in selected:
                camera_to_world, pose_gap = interpolate_camera_to_world(
                    pose_rows, rgb.timestamp_seconds
                )
                payloads.append(
                    TumSelectedPayload(
                        parent_id=parent_id,
                        role=role,
                        rgb=rgb,
                        depth=depth,
                        intrinsics=intrinsics,
                        storage_kind=storage_kind,
                        source_path=source_path,
                        rgb_path=source_path / rgb.relative_path,
                        depth_path=source_path / depth.relative_path,
                        rgb_bytes=None,
                        depth_bytes=None,
                        camera_to_world=camera_to_world,
                        pose_bracketing_gap_seconds=pose_gap,
                    )
                )
            source_receipts.append(
                {
                    "parent_id": parent_id,
                    "storage_kind": storage_kind,
                    "source_path": str(source_path),
                    "rgb_index_sha256": str(row["rgb_index_sha256"]),
                    "depth_index_sha256": str(row["depth_index_sha256"]),
                    "groundtruth_index_sha256": sha256_file(groundtruth_path),
                }
            )
        elif storage_kind == "tgz":
            require(source_path.is_file(), f"TUM source archive missing: {parent_id}")
            require(source_path.stat().st_size == int(row["source_bytes"]), "TUM archive size drift")
            require(
                sha256_file(source_path) == str(row["source_sha256"]),
                f"TUM archive hash drift: {parent_id}",
            )
            with tarfile.open(source_path, "r:gz") as archive:
                members = _tar_member_map(archive, parent_id)
                require("groundtruth.txt" in members, "TUM archive groundtruth missing")
                rgb_rows = parse_tum_index(_read_member(archive, members["rgb.txt"]).decode("utf-8"))
                depth_rows = parse_tum_index(
                    _read_member(archive, members["depth.txt"]).decode("utf-8")
                )
                pose_rows = parse_tum_poses(
                    _read_member(archive, members["groundtruth.txt"]).decode("utf-8")
                )
                available_rgb = [value for value in rgb_rows if value.relative_path in members]
                available_depth = [value for value in depth_rows if value.relative_path in members]
                selected = _validate_selected_pairs(row, available_rgb, available_depth)
                for rgb, depth in selected:
                    camera_to_world, pose_gap = interpolate_camera_to_world(
                        pose_rows, rgb.timestamp_seconds
                    )
                    payloads.append(
                        TumSelectedPayload(
                            parent_id=parent_id,
                            role=role,
                            rgb=rgb,
                            depth=depth,
                            intrinsics=intrinsics,
                            storage_kind=storage_kind,
                            source_path=source_path,
                            rgb_path=None,
                            depth_path=None,
                            rgb_bytes=_read_member(archive, members[rgb.relative_path]),
                            depth_bytes=_read_member(archive, members[depth.relative_path]),
                            camera_to_world=camera_to_world,
                            pose_bracketing_gap_seconds=pose_gap,
                        )
                    )
            source_receipts.append(
                {
                    "parent_id": parent_id,
                    "storage_kind": storage_kind,
                    "source_path": str(source_path),
                    "source_bytes": int(row["source_bytes"]),
                    "source_sha256": str(row["source_sha256"]),
                }
            )
        else:
            raise ValueError(f"unsupported TUM storage kind: {storage_kind}")
    require(len(payloads) == 3 * len(rows), "TUM selected payload count drift")
    return payloads, {
        "cohort_manifest_path": str(manifest_path),
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "role": role,
        "parent_ids": [str(row["parent_id"]) for row in rows],
        "parent_count": len(rows),
        "frame_count": len(payloads),
        "source_receipts": source_receipts,
        "factor_validity": {
            "metric_depth": "A_SOURCE_WHERE_UINT16_GT_ZERO",
            "camera_to_world": "SOURCE_GROUNDTRUTH_INTERPOLATED_AT_RGB_TIMESTAMP",
            "support": "UNKNOWN",
            "boundary": "UNKNOWN",
            "obstacle": "UNKNOWN",
        },
    }
