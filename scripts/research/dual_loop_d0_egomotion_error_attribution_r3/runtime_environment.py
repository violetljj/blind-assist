from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import site
import sys
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
FORMAL_IMPLEMENTATION_FILES = (
    REPO_ROOT / "scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py",
    *tuple(
        sorted(
            path
            for path in MODULE_DIR.glob("*.py")
            if not path.name.startswith("test_")
            and path.name != "run_tests.py"
        )
    ),
)
EXPECTED_EXPLICIT_THIRD_PARTY_IMPORT_ROOTS = {
    "numpy",
    "rosbags",
    "yaml",
}
STDLIB_OR_INTERNAL_IMPORT_ROOTS = {
    "__future__",
    "analysis",
    "argparse",
    "ast",
    "bindings",
    "bisect",
    "collections",
    "contract",
    "dataclasses",
    "hashlib",
    "importlib",
    "json",
    "math",
    "os",
    "pathlib",
    "platform",
    "producer",
    "runtime_environment",
    "scripts",
    "site",
    "subprocess",
    "sys",
    "typing",
    "validate_execution_independent",
    "validate_implementation_lock",
}


REQUIRED_DISTRIBUTIONS = {
    "lz4": "4.4.5",
    "numpy": "2.4.2",
    "opencv-python-headless": "4.13.0.90",
    "pyyaml": "6.0.3",
    "rosbags": "0.11.0",
    "ruamel.yaml": "0.19.1",
    "typing_extensions": "4.16.0",
    "zstandard": "0.25.0",
}
MODULE_IMPORTS = {
    "cv2": "opencv-python-headless",
    "lz4": "lz4",
    "numpy": "numpy",
    "yaml": "pyyaml",
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


def explicit_third_party_import_roots() -> list[str]:
    roots: set[str] = set()
    for path in FORMAL_IMPLEMENTATION_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                root = str(name).split(".", 1)[0]
                if root and root not in STDLIB_OR_INTERNAL_IMPORT_ROOTS:
                    roots.add(root)
    if roots != EXPECTED_EXPLICIT_THIRD_PARTY_IMPORT_ROOTS:
        raise RuntimeEnvironmentError(
            "formal reachable third-party import drift: "
            f"{sorted(roots)!r}"
        )
    return sorted(roots)


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
            "blindassist.d0_r3_runtime_environment_manifest.v1"
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
        "formal_reachable_imports": {
            "implementation_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in FORMAL_IMPLEMENTATION_FILES
            ],
            "explicit_third_party_import_roots": (
                explicit_third_party_import_roots()
            ),
            "dynamic_module_imports": sorted(MODULE_IMPORTS),
        },
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


def prestart_dependency_smoke() -> dict[str, Any]:
    import numpy as np
    import yaml
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    if not callable(getattr(yaml, "safe_load", None)):
        raise RuntimeEnvironmentError("yaml.safe_load is unavailable")
    calibration = yaml.safe_load(
        "T_v_c:\n"
        "  - [1.0, 0.0, 0.0, 0.0]\n"
        "  - [0.0, 1.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, 1.0, 0.0]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n"
    )
    matrix = np.asarray(calibration["T_v_c"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeEnvironmentError(
            "synthetic calibration parser smoke failed"
        )
    typestore = get_typestore(Stores.ROS1_NOETIC)
    if not callable(getattr(typestore, "deserialize_ros1", None)):
        raise RuntimeEnvironmentError(
            "rosbags typestore deserialize_ros1 is unavailable"
        )
    if not callable(getattr(Reader, "messages", None)):
        raise RuntimeEnvironmentError("rosbags Reader.messages is unavailable")
    return {
        "status": "VALID_SYNTHETIC_RUNTIME_SMOKE",
        "imports": [
            "numpy",
            "yaml",
            "rosbags.rosbag1.Reader",
            "rosbags.typesys.Stores",
            "rosbags.typesys.get_typestore",
        ],
        "yaml_safe_load_called": True,
        "synthetic_calibration_shape": [4, 4],
        "synthetic_calibration_finite": True,
        "real_calibration_opened": False,
        "bag_messages_opened": False,
        "truth_opened": False,
        "event_rows_built": False,
        "d0_metrics_computed": False,
    }


def probe_designated_vicon_message(bag_path: Path) -> dict[str, Any]:
    if not bag_path.is_file():
        raise RuntimeEnvironmentError("frozen bag is missing")
    # R2 already proved this exact message under the same interpreter,
    # rosbags and transitive dependency versions before its formal marker.
    # R3 binds that consumed receipt and does not decode real input pre-marker.
    return {
        "status": "VALID_INHERITED_R2_OPERATIONAL_PROBE",
        "source_protocol_id": "D0_EGOMOTION_ERROR_ATTRIBUTION_R2",
        "source_formal_start_sha256": (
            "730ec5dabf4a37716f589c363276e78c114fe26e51a7f656c33bf64aed776f63"
        ),
        "source_runtime_manifest_sha256": (
            "0faceae2077e87a90bc96da1a9e953dd81bd5c4baeec75779b23fd2f783e823a"
        ),
        "source_runtime_tree_sha256": (
            "07227110ca3b91fb2445a13099bfd1a7c2f9df8f231ab77a84ed36113c6ebba4"
        ),
        "topic": PROBE_TOPIC,
        "connection_count": PROBE_CONNECTION_COUNT,
        "ordinal": PROBE_ORDINAL,
        "msgtype": PROBE_MSGTYPE,
        "timestamp_ns": PROBE_TIMESTAMP_NS,
        "raw_bytes": PROBE_RAW_BYTES,
        "raw_sha256": PROBE_RAW_SHA256,
        "deserialized_message_count": 0,
        "inherited_deserialized_message_count": 1,
        "pose_values_retained": False,
        "vicon_bag_messages_opened": False,
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
