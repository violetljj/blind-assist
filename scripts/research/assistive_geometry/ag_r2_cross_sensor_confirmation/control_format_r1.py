"""Pure bounded parser for the R1 ETH3D/Kalibr camera-control contract.

R1 does not infer a camera from ``cam0``/``cam1`` ordering.  It preserves every
valid camera-node ``T_cam_imu`` discovery together with that node's Kalibr
``rostopic`` so a separately frozen sensor-namespace rule can select a target.
Importing this module performs no filesystem, archive, model, or network I/O.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .contract import ContractError, require

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NODE = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
_ROW = re.compile(
    rf"^(?P<indent> *)-\s*\[\s*(?P<a>{_NUMBER})\s*,\s*(?P<b>{_NUMBER})\s*,\s*"
    rf"(?P<c>{_NUMBER})\s*,\s*(?P<d>{_NUMBER})\s*\]\s*(?:#.*)?$"
)
_TOPIC = re.compile(
    r"^(?P<indent> +)rostopic:\s*(?P<value>(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^#\s][^#\r\n]*?))\s*(?:#.*)?$"
)
_ROS_TOPIC = re.compile(r"^/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)+$")


@dataclass(frozen=True)
class KalibrCameraControl:
    camera_node_key: str
    rostopic: str | None
    matrix_key: str
    matrix: np.ndarray


def _validated_matrix(rows: list[list[float]]) -> np.ndarray:
    require(len(rows) == 4 and all(len(row) == 4 for row in rows), "F2_R1_IMU_CALIBRATION_MATRIX")
    matrix = np.asarray(rows, dtype=np.float64)
    require(bool(np.all(np.isfinite(matrix))), "F2_R1_IMU_CALIBRATION_MATRIX_NONFINITE")
    require(
        bool(np.allclose(matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0)), rtol=0.0, atol=1e-12)),
        "F2_R1_IMU_CALIBRATION_HOMOGENEOUS_ROW",
    )
    rotation = matrix[:3, :3]
    require(
        bool(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8))
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8,
        "F2_R1_IMU_CALIBRATION_ROTATION",
    )
    matrix.setflags(write=False)
    return matrix


def _topic_value(token: str) -> str:
    value = token.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    require(_ROS_TOPIC.fullmatch(value) is not None, "F2_R1_KALIBR_ROSTOPIC")
    return value


def rostopic_namespace(topic: str) -> str:
    require(_ROS_TOPIC.fullmatch(topic) is not None, "F2_R1_KALIBR_ROSTOPIC")
    namespace, _separator, leaf = topic.rpartition("/")
    require(namespace.startswith("/") and leaf != "", "F2_R1_KALIBR_ROSTOPIC")
    return namespace


def discover_kalibr_camera_controls(
    raw: bytes,
    *,
    matrix_key: str = "T_cam_imu",
) -> tuple[KalibrCameraControl, ...]:
    """Return every valid matrix and its same-node Kalibr image ``rostopic``."""

    require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, "F2_R1_KALIBR_CONTROL_SIZE")
    require("\x00" not in raw.decode("latin-1"), "F2_R1_KALIBR_CONTROL_NUL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("F2_R1_KALIBR_CONTROL_UTF8", str(error)) from error
    require("\t" not in text, "F2_R1_KALIBR_CONTROL_TAB")
    lines = text.splitlines()
    active_node: tuple[str, int] | None = None
    node_topics: dict[str, str] = {}
    matrix_rows: list[tuple[str, list[list[float]]]] = []
    matrix_paths: set[tuple[str, str]] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        node_match = _NODE.fullmatch(line)
        if node_match and len(node_match.group("indent")) == 0:
            active_node = (node_match.group("name"), 0)
            index += 1
            continue
        topic_match = _TOPIC.fullmatch(line)
        if topic_match:
            require(active_node is not None, "F2_R1_KALIBR_TOPIC_WITHOUT_CAMERA_NODE")
            require(len(topic_match.group("indent")) > active_node[1], "F2_R1_KALIBR_TOPIC_INDENT")
            node = active_node[0]
            require(node not in node_topics, "F2_R1_KALIBR_ROSTOPIC_DUPLICATE")
            node_topics[node] = _topic_value(topic_match.group("value"))
            index += 1
            continue
        key_match = _NODE.fullmatch(line)
        if key_match and key_match.group("name") == matrix_key:
            require(active_node is not None, "F2_R1_KALIBR_MATRIX_WITHOUT_CAMERA_NODE")
            key_indent = len(key_match.group("indent"))
            require(key_indent > active_node[1], "F2_R1_KALIBR_MATRIX_INDENT")
            path = (active_node[0], matrix_key)
            require(path not in matrix_paths, "F2_R1_KALIBR_MATRIX_PATH_DUPLICATE")
            matrix_paths.add(path)
            rows: list[list[float]] = []
            for offset in range(1, 5):
                require(index + offset < len(lines), "F2_R1_IMU_CALIBRATION_MATRIX")
                row_match = _ROW.fullmatch(lines[index + offset])
                require(row_match is not None, "F2_R1_IMU_CALIBRATION_MATRIX")
                require(len(row_match.group("indent")) >= key_indent, "F2_R1_KALIBR_MATRIX_INDENT")
                values = [float(row_match.group(name)) for name in ("a", "b", "c", "d")]
                require(all(math.isfinite(value) for value in values), "F2_R1_IMU_CALIBRATION_MATRIX_NONFINITE")
                rows.append(values)
            matrix_rows.append((active_node[0], rows))
            index += 5
            continue
        index += 1
    return tuple(
        KalibrCameraControl(
            camera_node_key=node,
            rostopic=node_topics.get(node),
            matrix_key=matrix_key,
            matrix=_validated_matrix(rows),
        )
        for node, rows in matrix_rows
    )


def select_camera_namespace(
    controls: tuple[KalibrCameraControl, ...],
    *,
    expected_namespace: str,
) -> tuple[KalibrCameraControl, ...]:
    require(_ROS_TOPIC.fullmatch(f"{expected_namespace}/image") is not None, "F2_R1_CAMERA_NAMESPACE")
    return tuple(
        item
        for item in controls
        if item.rostopic is not None and rostopic_namespace(item.rostopic) == expected_namespace
    )
