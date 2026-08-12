"""Independent, producer-free validation of calibration-control evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import numpy as np

RESULT_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_calibration_control_result.v1"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NODE = re.compile(r"^(?P<indent> *)(?P<name>[A-Za-z][A-Za-z0-9_]*):\s*(?:#.*)?$")
ROW = re.compile(
    rf"^(?P<indent> *)-\s*\[\s*(?P<a>{NUMBER})\s*,\s*(?P<b>{NUMBER})\s*,\s*"
    rf"(?P<c>{NUMBER})\s*,\s*(?P<d>{NUMBER})\s*\]\s*(?:#.*)?$"
)


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


def _matrices(raw: bytes) -> list[tuple[str, list[list[float]]]]:
    _require(0 < len(raw) <= 4 * 1024 * 1024, "F2_CONTROL_VALIDATOR_YAML_SIZE")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError("F2_CONTROL_VALIDATOR_YAML_UTF8") from error
    _require("\x00" not in text and "\t" not in text, "F2_CONTROL_VALIDATOR_YAML_TEXT")
    lines = text.splitlines()
    active_node: str | None = None
    found: list[tuple[str, list[list[float]]]] = []
    index = 0
    while index < len(lines):
        node = NODE.fullmatch(lines[index])
        if node and len(node.group("indent")) == 0:
            active_node = node.group("name")
            index += 1
            continue
        key = NODE.fullmatch(lines[index])
        if key and key.group("name") == "T_cam_imu" and len(key.group("indent")) > 0:
            _require(active_node is not None, "F2_CONTROL_VALIDATOR_CAMERA_NODE")
            rows: list[list[float]] = []
            for offset in range(1, 5):
                _require(index + offset < len(lines), "F2_CONTROL_VALIDATOR_MATRIX")
                row = ROW.fullmatch(lines[index + offset])
                _require(row is not None, "F2_CONTROL_VALIDATOR_MATRIX")
                values = [float(row.group(name)) for name in ("a", "b", "c", "d")]
                _require(all(math.isfinite(value) for value in values), "F2_CONTROL_VALIDATOR_MATRIX")
                rows.append(values)
            matrix = np.asarray(rows, dtype=np.float64)
            _require(np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12), "F2_CONTROL_VALIDATOR_HOMOGENEOUS")
            rotation = matrix[:3, :3]
            _require(
                np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-8,
                "F2_CONTROL_VALIDATOR_ROTATION",
            )
            found.append((active_node, rows))
            index += 5
            continue
        index += 1
    return found


def validate(root: Path, control_lock: Path) -> dict:
    root = root.resolve()
    lock = json.loads(control_lock.read_text(encoding="utf-8"))
    _require(Path(lock["output_root"]).resolve() == root, "F2_CONTROL_VALIDATOR_ROOT_BINDING")
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    _require(names == ["manifest.json", "result.json", "start-receipt.json"], "F2_CONTROL_VALIDATOR_FILE_SET")
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require(result.get("schema") == RESULT_SCHEMA, "F2_CONTROL_VALIDATOR_RESULT_SCHEMA")
    _require(result.get("status") == "CALIBRATION_CONTROL_PASS_EXACT_MEMBER_BOUND", "F2_CONTROL_VALIDATOR_STATUS")
    _require(
        manifest.get("terminal") == result["status"]
        and manifest.get("evidence_root_consumed") is True
        and set(manifest.get("files", {})) == {"result.json", "start-receipt.json"},
        "F2_CONTROL_VALIDATOR_MANIFEST",
    )
    for name, row in manifest["files"].items():
        path = root / name
        _require(path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"], "F2_CONTROL_VALIDATOR_FILE_HASH")
    identity = json.loads(Path(lock["data_identity"]["path"]).read_text(encoding="utf-8"))
    archives = [row for row in identity["archives"] if row["kind"] == "CAMERA_IMU_CALIBRATION_ARCHIVE"]
    _require(len(archives) == 1, "F2_CONTROL_VALIDATOR_ARCHIVE_BINDING")
    binding = archives[0]
    archive = Path(lock["archive_root"]) / PurePosixPath(binding["url"]).name
    _require(archive.is_file() and archive.stat().st_size == binding["bytes"] and _sha(archive) == binding["sha256"], "F2_CONTROL_VALIDATOR_ARCHIVE_HASH")
    discoveries: list[dict] = []
    member_receipts: list[dict] = []
    with zipfile.ZipFile(archive, "r") as container:
        infos = container.infolist()
        _require(len(infos) <= 256, "F2_CONTROL_VALIDATOR_MEMBER_COUNT")
        seen: set[str] = set()
        yaml_infos = []
        total = 0
        for info in infos:
            name = info.orig_filename
            parsed = PurePosixPath(name.removesuffix("/"))
            _require(
                "\\" not in name and not parsed.is_absolute() and ".." not in parsed.parts
                and parsed.as_posix() == name.removesuffix("/"),
                "F2_CONTROL_VALIDATOR_MEMBER_PATH",
            )
            _require(name.casefold() not in seen, "F2_CONTROL_VALIDATOR_MEMBER_DUPLICATE")
            seen.add(name.casefold())
            if not info.is_dir():
                total += info.file_size
                _require(info.file_size <= 4194304 and total <= 67108864, "F2_CONTROL_VALIDATOR_MEMBER_BUDGET")
                if name.casefold().endswith((".yaml", ".yml")):
                    yaml_infos.append(info)
        _require(0 < len(yaml_infos) <= 32, "F2_CONTROL_VALIDATOR_YAML_COUNT")
        for info in sorted(yaml_infos, key=lambda item: item.orig_filename):
            raw = container.read(info)
            member_receipts.append({"name": info.orig_filename, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest().upper()})
            for node, matrix in _matrices(raw):
                discoveries.append(
                    {
                        "name": info.orig_filename,
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest().upper(),
                        "camera_node_key": node,
                        "matrix_key": "T_cam_imu",
                        "matrix_sha256": _canonical_sha(matrix),
                        "encoding": "KALIBR_CAMCHAIN_YAML_T_CAM_IMU_NESTED_4X4",
                        "transform_direction": "IMU_TO_CAMERA_T_CAM_IMU",
                    }
                )
    _require(len(discoveries) == 1 and discoveries[0] == result.get("selected_member"), "F2_CONTROL_VALIDATOR_SELECTION")
    _require(member_receipts == result["inventory"]["member_receipts"], "F2_CONTROL_VALIDATOR_MEMBER_RECEIPTS")
    access = result.get("access_receipt", {})
    _require(
        access.get("calibration_archive_member_reads") == len(member_receipts)
        and access.get("calibration_archive_member_bytes") == sum(row["bytes"] for row in member_receipts)
        and all(access.get(key) == 0 for key in (
            "session_rgbd_archive_reads", "session_imu_archive_reads", "model_or_checkpoint_reads",
            "source_truth_materializations", "factor_scoring_runs", "confirmation_runs",
        ))
        and access.get("confirmation_root_created") is False,
        "F2_CONTROL_VALIDATOR_ACCESS_RECEIPT",
    )
    return {"valid": True, "selected_member": discoveries[0], "yaml_member_reads": len(member_receipts)}


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
