"""Producer-free replay validator for R1 calibration-control evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import numpy as np

RESULT_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_result.v1"
FAILURE_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_r1_failure.v1"
EXPECTED_NAMESPACE = "/uvc_camera/cam_2"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NODE = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
ROW = re.compile(
    rf"^(?P<indent> *)-\s*\[\s*(?P<a>{NUMBER})\s*,\s*(?P<b>{NUMBER})\s*,\s*"
    rf"(?P<c>{NUMBER})\s*,\s*(?P<d>{NUMBER})\s*\]\s*(?:#.*)?$"
)
TOPIC = re.compile(
    r"^(?P<indent> +)rostopic:\s*(?P<value>(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^#\s][^#\r\n]*?))\s*(?:#.*)?$"
)
ROS_TOPIC = re.compile(r"^/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)+$")


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _topic(token: str) -> str:
    value = token.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    _require(ROS_TOPIC.fullmatch(value) is not None, "F2_R1_VALIDATOR_ROSTOPIC")
    return value


def _controls(raw: bytes) -> list[tuple[str, str | None, list[list[float]]]]:
    _require(0 < len(raw) <= 4 * 1024 * 1024, "F2_R1_VALIDATOR_YAML_SIZE")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError("F2_R1_VALIDATOR_YAML_UTF8") from error
    _require("\x00" not in text and "\t" not in text, "F2_R1_VALIDATOR_YAML_TEXT")
    lines = text.splitlines()
    active_node: str | None = None
    topics: dict[str, str] = {}
    matrices: list[tuple[str, list[list[float]]]] = []
    paths: set[tuple[str, str]] = set()
    index = 0
    while index < len(lines):
        node = NODE.fullmatch(lines[index])
        if node and len(node.group("indent")) == 0:
            active_node = node.group("name")
            index += 1
            continue
        topic = TOPIC.fullmatch(lines[index])
        if topic:
            _require(active_node is not None and active_node not in topics, "F2_R1_VALIDATOR_TOPIC")
            topics[active_node] = _topic(topic.group("value"))
            index += 1
            continue
        key = NODE.fullmatch(lines[index])
        if key and key.group("name") == "T_cam_imu" and len(key.group("indent")) > 0:
            _require(active_node is not None, "F2_R1_VALIDATOR_CAMERA_NODE")
            path = (active_node, "T_cam_imu")
            _require(path not in paths, "F2_R1_VALIDATOR_MATRIX_DUPLICATE")
            paths.add(path)
            rows: list[list[float]] = []
            for offset in range(1, 5):
                _require(index + offset < len(lines), "F2_R1_VALIDATOR_MATRIX")
                row = ROW.fullmatch(lines[index + offset])
                _require(row is not None, "F2_R1_VALIDATOR_MATRIX")
                values = [float(row.group(name)) for name in ("a", "b", "c", "d")]
                _require(all(math.isfinite(value) for value in values), "F2_R1_VALIDATOR_MATRIX")
                rows.append(values)
            matrix = np.asarray(rows, dtype=np.float64)
            _require(np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12), "F2_R1_VALIDATOR_HOMOGENEOUS")
            rotation = matrix[:3, :3]
            _require(
                np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8,
                "F2_R1_VALIDATOR_ROTATION",
            )
            matrices.append((active_node, rows))
            index += 5
            continue
        index += 1
    return [(node, topics.get(node), rows) for node, rows in matrices]


def _safe_infos(container: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], int]:
    infos = container.infolist()
    _require(len(infos) <= 256, "F2_R1_VALIDATOR_MEMBER_COUNT")
    seen: set[str] = set()
    yaml_infos: list[zipfile.ZipInfo] = []
    total = 0
    for info in infos:
        name = info.orig_filename
        parsed = PurePosixPath(name.removesuffix("/"))
        _require(
            "\\" not in name
            and not parsed.is_absolute()
            and ".." not in parsed.parts
            and parsed.as_posix() == name.removesuffix("/"),
            "F2_R1_VALIDATOR_MEMBER_PATH",
        )
        _require(name.casefold() not in seen, "F2_R1_VALIDATOR_MEMBER_DUPLICATE")
        seen.add(name.casefold())
        mode = (info.external_attr >> 16) & 0xFFFF
        _require(
            stat.S_IFMT(mode) not in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK},
            "F2_R1_VALIDATOR_MEMBER_SPECIAL",
        )
        if not info.is_dir():
            total += info.file_size
            _require(info.file_size <= 4194304 and total <= 67108864, "F2_R1_VALIDATOR_MEMBER_BUDGET")
            if name.casefold().endswith((".yaml", ".yml")):
                yaml_infos.append(info)
    _require(0 < len(yaml_infos) <= 32, "F2_R1_VALIDATOR_YAML_COUNT")
    yaml_infos.sort(key=lambda item: item.orig_filename)
    return yaml_infos, len(infos)


def validate(root: Path, control_lock: Path) -> dict[str, object]:
    root = root.resolve()
    lock = json.loads(control_lock.read_text(encoding="utf-8"))
    _require(Path(lock["output_root"]).resolve() == root, "F2_R1_VALIDATOR_ROOT_BINDING")
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    _require(
        names in (
            ["manifest.json", "result.json", "start-receipt.json"],
            ["failure.json", "manifest.json", "start-receipt.json"],
        ),
        "F2_R1_VALIDATOR_FILE_SET",
    )
    terminal_name = "result.json" if "result.json" in names else "failure.json"
    terminal = json.loads((root / terminal_name).read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest.get("evidence_root_consumed") is True
        and set(manifest.get("files", {})) == {terminal_name, "start-receipt.json"},
        "F2_R1_VALIDATOR_MANIFEST",
    )
    for name, row in manifest["files"].items():
        path = root / name
        _require(path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"], "F2_R1_VALIDATOR_FILE_HASH")
    identity = json.loads(Path(lock["data_identity"]["path"]).read_text(encoding="utf-8"))
    archives = [row for row in identity["archives"] if row["kind"] == "CAMERA_IMU_CALIBRATION_ARCHIVE"]
    _require(len(archives) == 1, "F2_R1_VALIDATOR_ARCHIVE_BINDING")
    binding = archives[0]
    archive = Path(lock["archive_root"]) / PurePosixPath(binding["url"]).name
    _require(
        archive.is_file() and archive.stat().st_size == binding["bytes"] and _sha(archive) == binding["sha256"],
        "F2_R1_VALIDATOR_ARCHIVE_HASH",
    )
    discoveries: list[dict[str, object]] = []
    member_receipts: list[dict[str, object]] = []
    with zipfile.ZipFile(archive, "r") as container:
        yaml_infos, member_count = _safe_infos(container)
        candidate_names = [info.orig_filename for info in yaml_infos]
        for info in yaml_infos:
            raw = container.read(info)
            digest = hashlib.sha256(raw).hexdigest().upper()
            member_receipts.append({"name": info.orig_filename, "bytes": len(raw), "sha256": digest})
            for node, rostopic, matrix in _controls(raw):
                namespace = rostopic.rpartition("/")[0] if rostopic is not None else None
                discoveries.append(
                    {
                        "name": info.orig_filename,
                        "bytes": len(raw),
                        "sha256": digest,
                        "camera_node_key": node,
                        "rostopic": rostopic,
                        "rostopic_namespace": namespace,
                        "matrix_key": "T_cam_imu",
                        "matrix_sha256": _canonical_sha(matrix),
                        "encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                        "transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                    }
                )
    matches = [row for row in discoveries if row["rostopic_namespace"] == EXPECTED_NAMESPACE]
    expected_observability = {
        "archive_hash_verified": True,
        "archive_member_count": member_count,
        "yaml_candidate_count": len(candidate_names),
        "yaml_candidate_names_sha256": _canonical_sha(candidate_names),
        "yaml_members_read": len(member_receipts),
        "yaml_member_bytes_read": sum(int(row["bytes"]) for row in member_receipts),
        "all_yaml_candidates_read": True,
        "matrix_discovery_count": len(discoveries),
        "target_namespace_match_count": len(matches),
        "matrix_discoveries_sha256": _canonical_sha(discoveries),
        "member_receipts_sha256": _canonical_sha(member_receipts),
        "first_or_best_selected": False,
    }
    access = terminal.get("access_receipt", {})
    _require(
        access.get("calibration_archive_member_reads") == len(member_receipts)
        and access.get("calibration_archive_member_bytes") == sum(int(row["bytes"]) for row in member_receipts)
        and all(
            access.get(key) == 0
            for key in (
                "session_rgbd_archive_reads",
                "session_imu_archive_reads",
                "model_or_checkpoint_reads",
                "source_truth_materializations",
                "factor_scoring_runs",
                "confirmation_runs",
            )
        )
        and access.get("confirmation_root_created") is False,
        "F2_R1_VALIDATOR_ACCESS_RECEIPT",
    )
    if terminal_name == "result.json":
        _require(
            terminal.get("schema") == RESULT_SCHEMA
            and terminal.get("status") == "CALIBRATION_CONTROL_R1_PASS_EXACT_MEMBER_AND_TARGET_CAMERA_BOUND"
            and len(matches) == 1
            and terminal.get("selected_member") == matches[0]
            and terminal.get("inventory") == {**expected_observability, "member_receipts": member_receipts},
            "F2_R1_VALIDATOR_PASS_RESULT",
        )
        selected: object = matches[0]
    else:
        _require(
            terminal.get("schema") == FAILURE_SCHEMA
            and terminal.get("status") == "CALIBRATION_CONTROL_R1_FAIL_CLOSED"
            and terminal.get("error_code") == "F2_R1_CALIBRATION_CONTROL_TARGET_CAMERA_AMBIGUOUS_OR_MISSING"
            and terminal.get("one_shot_consumed") is True
            and len(matches) != 1
            and terminal.get("observability") == expected_observability
            and terminal.get("selection_receipt")
            == {
                "expected_camera_sensor_namespace": EXPECTED_NAMESPACE,
                "selected_member": None,
                "selected_camera_node": None,
                "first_or_best_selected": False,
            },
            "F2_R1_VALIDATOR_FAILURE_RESULT",
        )
        selected = None
    return {
        "valid": True,
        "terminal": terminal["status"],
        "yaml_member_reads": len(member_receipts),
        "matrix_discovery_count": len(discoveries),
        "target_namespace_match_count": len(matches),
        "selected_member": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(args.root, args.control_lock)
    except Exception as error:  # noqa: BLE001 - independent CLI is fail closed.
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
