"""Bounded parsers for official ETH3D/Kalibr control documents.

This module is pure: importing it performs no filesystem, archive, model, or
network access.  It accepts only the documented Kalibr camchain shape needed by
the cross-sensor Confirmation source adapter.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .contract import ContractError, require

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NODE = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
_KEY = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
_ROW = re.compile(
    rf"^(?P<indent> *)-\s*\[\s*(?P<a>{_NUMBER})\s*,\s*(?P<b>{_NUMBER})\s*,\s*"
    rf"(?P<c>{_NUMBER})\s*,\s*(?P<d>{_NUMBER})\s*\]\s*(?:#.*)?$"
)


@dataclass(frozen=True)
class KalibrMatrix:
    camera_node_key: str
    matrix_key: str
    matrix: np.ndarray


def _validated_matrix(rows: list[list[float]]) -> np.ndarray:
    require(len(rows) == 4 and all(len(row) == 4 for row in rows), "F2_IMU_CALIBRATION_MATRIX")
    matrix = np.asarray(rows, dtype=np.float64)
    require(bool(np.all(np.isfinite(matrix))), "F2_IMU_CALIBRATION_MATRIX_NONFINITE")
    require(
        bool(np.allclose(matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0)), rtol=0.0, atol=1e-12)),
        "F2_IMU_CALIBRATION_HOMOGENEOUS_ROW",
    )
    rotation = matrix[:3, :3]
    require(
        bool(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8))
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8,
        "F2_IMU_CALIBRATION_ROTATION",
    )
    matrix.setflags(write=False)
    return matrix


def discover_kalibr_matrices(raw: bytes, *, matrix_key: str = "T_cam_imu") -> tuple[KalibrMatrix, ...]:
    """Return every exact ``camera_node.matrix_key`` nested 4x4 matrix."""

    require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, "F2_KALIBR_CONTROL_SIZE")
    require("\x00" not in raw.decode("latin-1"), "F2_KALIBR_CONTROL_NUL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("F2_KALIBR_CONTROL_UTF8", str(error)) from error
    require("\t" not in text, "F2_KALIBR_CONTROL_TAB")
    lines = text.splitlines()
    results: list[KalibrMatrix] = []
    active_node: tuple[str, int] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        node_match = _NODE.fullmatch(line)
        if node_match and len(node_match.group("indent")) == 0:
            active_node = (node_match.group("name"), 0)
            index += 1
            continue
        key_match = _KEY.fullmatch(line)
        if key_match and key_match.group("name") == matrix_key:
            require(active_node is not None, "F2_KALIBR_MATRIX_WITHOUT_CAMERA_NODE")
            key_indent = len(key_match.group("indent"))
            require(key_indent > active_node[1], "F2_KALIBR_MATRIX_INDENT")
            rows: list[list[float]] = []
            for offset in range(1, 5):
                require(index + offset < len(lines), "F2_IMU_CALIBRATION_MATRIX")
                row_match = _ROW.fullmatch(lines[index + offset])
                require(row_match is not None, "F2_IMU_CALIBRATION_MATRIX")
                require(len(row_match.group("indent")) >= key_indent, "F2_KALIBR_MATRIX_INDENT")
                values = [float(row_match.group(name)) for name in ("a", "b", "c", "d")]
                require(all(math.isfinite(value) for value in values), "F2_IMU_CALIBRATION_MATRIX_NONFINITE")
                rows.append(values)
            results.append(
                KalibrMatrix(
                    camera_node_key=active_node[0],
                    matrix_key=matrix_key,
                    matrix=_validated_matrix(rows),
                )
            )
            index += 5
            continue
        index += 1
    require(results, "F2_IMU_CALIBRATION_KEY_AMBIGUOUS_OR_MISSING")
    identities = [(item.camera_node_key, item.matrix_key) for item in results]
    require(len(identities) == len(set(identities)), "F2_KALIBR_MATRIX_PATH_DUPLICATE")
    return tuple(results)


def parse_kalibr_camera_from_imu(
    raw: bytes,
    *,
    camera_node_key: str,
    matrix_key: str,
) -> np.ndarray:
    matches = [
        item.matrix
        for item in discover_kalibr_matrices(raw, matrix_key=matrix_key)
        if item.camera_node_key == camera_node_key
    ]
    require(len(matches) == 1, "F2_IMU_CALIBRATION_KEY_AMBIGUOUS_OR_MISSING")
    return matches[0]
