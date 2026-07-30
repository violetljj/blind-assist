from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import site
import sys
from pathlib import Path
from typing import Any


REQUIRED_DISTRIBUTIONS = {
    "lz4": "4.4.5",
    "numpy": "2.4.2",
    "opencv-python-headless": "4.13.0.90",
    "rosbags": "0.11.0",
    "ruamel.yaml": "0.19.1",
    "typing_extensions": "4.16.0",
    "zstandard": "0.25.0",
}
MODULE_IMPORTS = {
    "cv2": "opencv-python-headless",
    "lz4": "lz4",
    "numpy": "numpy",
    "rosbags.rosbag1": "rosbags",
    "ruamel.yaml": "ruamel.yaml",
    "typing_extensions": "typing_extensions",
    "zstandard": "zstandard",
}
PROBE_TOPIC = "/vicon/event_lidar/event_lidar"
PROBE_CONNECTION_COUNT = 1
PROBE_ORDINAL = 0
PROBE_MSGTYPE = "geometry_msgs/msg/TransformStamped"
PROBE_TIMESTAMP_NS = 1708490365692128652
PROBE_RAW_BYTES = 117
PROBE_RAW_SHA256 = (
    "55779b8473c8813aff6827669f42b97e230715aa88fbb80781b1454a1cea920b"
)


class RuntimeEnvironmentError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalized(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeEnvironmentError(f"runtime identity file missing: {resolved}")
    return {
        "path": _normalized(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def build_runtime_manifest() -> dict[str, Any]:
    if sys.flags.isolated != 1:
        raise RuntimeEnvironmentError("Python must run with -I")
    if not sys.dont_write_bytecode:
        raise RuntimeEnvironmentError("Python must run with -B")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeEnvironmentError("PYTHONPATH must be unset")
    if site.ENABLE_USER_SITE:
        raise RuntimeEnvironmentError("user site must be disabled")

    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    executable = Path(sys.executable).resolve()
    base_executable = Path(getattr(sys, "_base_executable", executable)).resolve()
    python_dll = base_prefix / "python311.dll"
    pyvenv = prefix / "pyvenv.cfg"
    site_packages = prefix / "Lib" / "site-packages"

    distributions = sorted(
        {
            (
                str(distribution.metadata["Name"]).lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
        }
    )
    actual_distributions = dict(distributions)
    if actual_distributions != REQUIRED_DISTRIBUTIONS:
        raise RuntimeEnvironmentError(
            "distribution set/version drift: "
            f"{actual_distributions!r}"
        )

    module_sources: dict[str, dict[str, str]] = {}
    for module_name, distribution_name in MODULE_IMPORTS.items():
        module = importlib.import_module(module_name)
        source = Path(str(module.__file__)).resolve()
        try:
            source.relative_to(site_packages)
        except ValueError as error:
            raise RuntimeEnvironmentError(
                f"module outside frozen site-packages: {module_name}={source}"
            ) from error
        module_sources[module_name] = {
            "distribution": distribution_name,
            "path": _normalized(source),
            "sha256": sha256_file(source),
        }

    tree = _tree_entries(site_packages)
    return {
        "schema_version": (
            "blindassist.d0_r2_runtime_environment_manifest.v1"
        ),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "executable": _file_identity(executable),
            "base_executable": _file_identity(base_executable),
            "python_dll": _file_identity(python_dll),
            "pyvenv_cfg": _file_identity(pyvenv),
            "prefix": _normalized(prefix),
            "base_prefix": _normalized(base_prefix),
            "isolated": True,
            "dont_write_bytecode": True,
            "user_site_enabled": False,
            "pythonpath_set": False,
        },
        "distributions": [
            {"name": name, "version": version}
            for name, version in distributions
        ],
        "module_sources": module_sources,
        "site_packages": {
            "root": _normalized(site_packages),
            "file_count": len(tree),
            "total_bytes": sum(item["bytes"] for item in tree),
            "tree_sha256": canonical_sha256(tree),
            "files": tree,
        },
    }


def validate_runtime_manifest(path: Path) -> dict[str, Any]:
    expected_bytes = path.read_bytes()
    expected = json.loads(expected_bytes.decode("utf-8"))
    if expected_bytes != canonical_json_bytes(expected) + b"\n":
        raise RuntimeEnvironmentError("runtime manifest is not canonical JSON")
    actual = build_runtime_manifest()
    if actual != expected:
        raise RuntimeEnvironmentError("runtime manifest semantic drift")
    return {
        "status": "VALID",
        "manifest_sha256": sha256_file(path),
        "tree_sha256": actual["site_packages"]["tree_sha256"],
        "python_executable_sha256": actual["python"]["executable"]["sha256"],
    }


def probe_designated_vicon_message(bag_path: Path) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == PROBE_TOPIC
        ]
        if len(connections) != PROBE_CONNECTION_COUNT:
            raise RuntimeEnvironmentError("probe connection-count drift")
        connection, timestamp, raw = next(
            reader.messages(connections=connections)
        )
        if connection.msgtype != PROBE_MSGTYPE:
            raise RuntimeEnvironmentError("probe msgtype drift")
        if timestamp != PROBE_TIMESTAMP_NS:
            raise RuntimeEnvironmentError("probe timestamp drift")
        if len(raw) != PROBE_RAW_BYTES:
            raise RuntimeEnvironmentError("probe raw-byte-count drift")
        if hashlib.sha256(raw).hexdigest() != PROBE_RAW_SHA256:
            raise RuntimeEnvironmentError("probe raw SHA-256 drift")
        message = typestore.deserialize_ros1(raw, connection.msgtype)
        translation = message.transform.translation
        rotation = message.transform.rotation
        components = (
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        if not all(math.isfinite(float(value)) for value in components):
            raise RuntimeEnvironmentError("probe transform is non-finite")
    return {
        "status": "VALID_OPERATIONAL_PROBE",
        "topic": PROBE_TOPIC,
        "connection_count": PROBE_CONNECTION_COUNT,
        "ordinal": PROBE_ORDINAL,
        "msgtype": PROBE_MSGTYPE,
        "timestamp_ns": PROBE_TIMESTAMP_NS,
        "raw_bytes": PROBE_RAW_BYTES,
        "raw_sha256": PROBE_RAW_SHA256,
        "deserialized_message_count": 1,
        "pose_values_retained": False,
        "d0_metrics_computed": False,
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--bag", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = build_runtime_manifest()
        _write_exclusive(args.output.resolve(), payload)
        result = {
            "status": "FROZEN",
            "manifest_sha256": sha256_file(args.output.resolve()),
            "tree_sha256": payload["site_packages"]["tree_sha256"],
        }
    elif args.command == "validate":
        result = validate_runtime_manifest(args.manifest.resolve())
    else:
        result = probe_designated_vicon_message(args.bag.resolve())
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
