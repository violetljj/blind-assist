from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_SCHEMA = "blindassist_ustrf_sensor_replay_bundle_v1"
REPORT_SCHEMA = "blindassist_ustrf_sensor_replay_report_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"path escapes source root: {relative}")
    if not path.is_file():
        raise ValueError(f"missing source file: {relative}")
    return path


def parse_rows(path: Path, columns: int) -> list[list[str]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if not rows or any(len(row) != columns for row in rows):
        raise ValueError(f"invalid {columns}-column table: {path}")
    return rows


def quaternion_matrix(values: list[float]) -> list[list[float]]:
    tx, ty, tz, x, y, z, w = values
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"invalid quaternion norm: {norm}")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    matrix = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), tx],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), ty],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return matrix.tolist()


def nearest_pose(rows: list[tuple[float, list[float]]], timestamp: float) -> tuple[float, list[float]]:
    stamps = [row[0] for row in rows]
    index = bisect_left(stamps, timestamp)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    return min(candidates, key=lambda row: abs(row[0] - timestamp))


def validate_pose(matrix: list[list[float]]) -> None:
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("pose must be homogeneous 4x4")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("pose rotation is not right-handed orthonormal")


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None
