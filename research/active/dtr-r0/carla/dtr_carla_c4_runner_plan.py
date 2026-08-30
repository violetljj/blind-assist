"""Validate a static C4 compiler bundle and build its serial runner plan.

This helper imports no CARLA package. It rejects protocol, map, layout,
resolution, registry-link, hash, and port-plan drift before the PowerShell
runner reserves evidence or starts a child C2 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


COMPILED_SCHEMA_VERSION = "dtr-c4-per-map-c2-protocol-index-v1"
PLAN_SCHEMA_VERSION = "dtr-carla-c4-multimap-runner-plan-v1"
C4_EXPERIMENT_ID = "DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1"
C2_EXPERIMENT_ID = "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2"
SENSOR_ORDER = ("instance", "wearable", "depth", "witness")
ALLOWED_MAPS = {
    "Carla/Maps/Town01",
    "Carla/Maps/Town02",
    "Carla/Maps/Town03_Opt",
    "Carla/Maps/Town04",
    "Carla/Maps/Town05",
    "Carla/Maps/Town10HD_Opt",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
RESERVED_LEGACY_PORTS = frozenset(range(2000, 2023))


class RunnerPlanError(ValueError):
    """Raised when a compiled bundle cannot be executed safely."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerPlanError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerPlanError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerPlanError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RunnerPlanError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerPlanError(f"{label} must be an integer")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise RunnerPlanError(f"{label} is not a string identifier")
    text = value
    if not SAFE_ID.fullmatch(text):
        raise RunnerPlanError(f"{label} is not a safe identifier: {text!r}")
    if text.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        raise RunnerPlanError(f"{label} is a reserved Windows device name: {text}")
    return text


def _string_list(value: Any, label: str) -> list[str]:
    result = [_safe_id(item, f"{label} item") for item in _list(value, label)]
    if not result:
        raise RunnerPlanError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise RunnerPlanError(f"{label} contains duplicates")
    return result


def _contained_file(index_path: Path, raw_value: Any, label: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise RunnerPlanError(f"{label} path must be a non-empty string")
    candidate = Path(raw_value)
    if candidate.is_absolute():
        raise RunnerPlanError(
            f"{label} path must be relative so the frozen bundle is self-contained"
        )
    candidate = (index_path.parent / candidate).resolve(strict=True)
    base = index_path.parent.resolve(strict=True)
    try:
        common = Path(os.path.commonpath((os.fspath(base), os.fspath(candidate))))
    except ValueError as exc:
        raise RunnerPlanError(f"{label} is outside the compiled bundle: {candidate}") from exc
    if os.path.normcase(os.fspath(common)) != os.path.normcase(os.fspath(base)):
        raise RunnerPlanError(f"{label} is outside the compiled bundle: {candidate}")
    if not candidate.is_file():
        raise RunnerPlanError(f"{label} is unavailable: {candidate}")
    return candidate


def _expected_startup_map_argument(map_name: str) -> str:
    leaf = map_name.removeprefix("Carla/Maps/")
    return f"/Game/Carla/Maps/{leaf}.{leaf}"


def _expected_engine_ini_map_object_path(map_name: str) -> str:
    leaf = map_name.removeprefix("Carla/Maps/")
    return f"/Game/Carla/Maps/{leaf}.{leaf}"


def _validate_capture(value: Any, label: str) -> None:
    capture = _object(value, label)
    if capture.get("resolution") != [1280, 720]:
        raise RunnerPlanError(f"{label} resolution must be exactly 1280x720")
    if capture.get("sensor_order") != list(SENSOR_ORDER):
        raise RunnerPlanError(
            f"{label} sensor_order must be instance,wearable,depth,witness"
        )


def _validate_registry_link(
    index_path: Path, link_value: Any, label: str
) -> dict[str, str]:
    link = _object(link_value, f"compiled registries.{label}")
    if set(link) != {"path", "sha256"}:
        raise RunnerPlanError(f"compiled registries.{label} keys differ")
    declared_hash = str(link.get("sha256", ""))
    if not SHA256.fullmatch(declared_hash):
        raise RunnerPlanError(f"compiled registries.{label}.sha256 is not SHA-256")
    path = _contained_file(index_path, link.get("path"), f"C4 {label}")
    actual_hash = _sha256_file(path)
    if actual_hash != declared_hash.upper():
        raise RunnerPlanError(f"compiled registries.{label} hash differs")
    return {
        "relative_path": str(link["path"]),
        "path": os.fspath(path),
        "sha256": actual_hash,
    }


def _validate_protocol_entry(
    index_path: Path, entry_value: Any, entry_index: int
) -> dict[str, Any]:
    label = f"protocols[{entry_index}]"
    entry = _object(entry_value, label)
    group_id = _safe_id(entry.get("protocol_id"), f"{label}.protocol_id")
    map_name = str(entry.get("carla_map", ""))
    if map_name not in ALLOWED_MAPS:
        raise RunnerPlanError(f"{label}.carla_map is not an installed C4 map: {map_name}")
    startup_map_argument = str(entry.get("startup_map_argument", ""))
    if startup_map_argument != _expected_startup_map_argument(map_name):
        raise RunnerPlanError(f"{label}.startup_map_argument does not bind {map_name}")
    engine_ini_map_object_path = str(entry.get("engine_ini_map_object_path", ""))
    if engine_ini_map_object_path != _expected_engine_ini_map_object_path(map_name):
        raise RunnerPlanError(f"{label}.engine_ini_map_object_path does not bind {map_name}")
    cold_start_status = _safe_id(entry.get("cold_start_status"), f"{label}.cold_start_status")
    layout_ids = _string_list(entry.get("layout_ids"), f"{label}.layout_ids")
    protocol_path = _contained_file(index_path, entry.get("protocol_path"), label)
    protocol_file_sha256 = _sha256_file(protocol_path)
    declared_hash = str(entry.get("protocol_sha256", ""))
    if not SHA256.fullmatch(declared_hash) or declared_hash.upper() != protocol_file_sha256:
        raise RunnerPlanError(f"{label}.protocol_sha256 differs from the protocol file")

    protocol = _load_json(protocol_path, f"{label} protocol")
    if protocol.get("experiment_id") != C2_EXPERIMENT_ID:
        raise RunnerPlanError(f"{label} protocol is not C2-capture-compatible")
    environment = _object(protocol.get("environment"), f"{label} protocol environment")
    if environment.get("map") != map_name:
        raise RunnerPlanError(f"{label} protocol environment.map differs from its map")
    _validate_capture(protocol.get("capture"), f"{label} protocol capture")
    layouts = _object(protocol.get("layouts"), f"{label} protocol layouts")
    if set(layouts) != set(layout_ids):
        raise RunnerPlanError(f"{label}.layout_ids differ from its protocol layouts")
    scenarios = _list(protocol.get("scenarios"), f"{label} protocol scenarios")
    episode_layout = {
        _safe_id(scenario.get("episode_id"), f"{label} episode_id"): str(
            scenario.get("layout_id", "")
        )
        for scenario in (_object(value, f"{label} scenario") for value in scenarios)
    }
    if len(episode_layout) != len(scenarios):
        raise RunnerPlanError(f"{label} protocol episode IDs contain duplicates")
    if set(episode_layout.values()) != set(layout_ids):
        raise RunnerPlanError(f"{label} protocol does not exercise every layout")
    if _integer(entry.get("layout_count"), f"{label}.layout_count") != len(layout_ids):
        raise RunnerPlanError(f"{label}.layout_count is inconsistent")
    if _integer(entry.get("episode_count"), f"{label}.episode_count") != len(scenarios):
        raise RunnerPlanError(f"{label}.episode_count is inconsistent")
    indexed_episodes = _list(entry.get("episodes"), f"{label}.episodes")
    indexed_episode_layout = {
        _safe_id(value.get("episode_id"), f"{label}.episodes episode_id"): str(
            value.get("layout_id", "")
        )
        for value in (_object(item, f"{label}.episodes item") for item in indexed_episodes)
    }
    if len(indexed_episode_layout) != len(indexed_episodes):
        raise RunnerPlanError(f"{label}.episodes contain duplicate episode IDs")
    if indexed_episode_layout != episode_layout:
        raise RunnerPlanError(f"{label}.episodes differ from its protocol")
    admission = _object(protocol.get("admission"), f"{label} protocol admission")
    if _integer(admission.get("expected_layout_count"), f"{label} expected layouts") != len(
        layout_ids
    ):
        raise RunnerPlanError(f"{label} protocol expected layout count differs")
    if _integer(admission.get("expected_episode_count"), f"{label} expected episodes") != len(
        scenarios
    ):
        raise RunnerPlanError(f"{label} protocol expected episode count differs")

    compatibility = _object(protocol.get("c4_compatibility"), f"{label} c4_compatibility")
    for key, expected in (
        ("schema_version", COMPILED_SCHEMA_VERSION),
        ("experiment_id", C4_EXPERIMENT_ID),
        ("protocol_id", group_id),
        ("carla_map", map_name),
        ("engine_ini_map_object_path", engine_ini_map_object_path),
        ("cold_start_status", cold_start_status),
    ):
        if compatibility.get(key) != expected:
            raise RunnerPlanError(f"{label} c4_compatibility.{key} differs")

    return {
        "group_id": group_id,
        "map": map_name,
        "startup_map_argument": startup_map_argument,
        "engine_ini_map_object_path": engine_ini_map_object_path,
        "cold_start_status": cold_start_status,
        "layout_ids": layout_ids,
        "episodes": indexed_episodes,
        "relative_protocol_path": str(entry["protocol_path"]),
        "protocol_path": os.fspath(protocol_path),
        "protocol_sha256": protocol_file_sha256,
    }


def build_runner_plan(
    compiled_protocol_path: Path,
    *,
    base_rpc_port: int,
    port_group_stride: int,
) -> dict[str, Any]:
    """Return a deterministic per-map C2-runner execution plan."""

    index_path = compiled_protocol_path.resolve(strict=True)
    if not index_path.is_file():
        raise RunnerPlanError(f"compiled C4 protocol is unavailable: {index_path}")
    if base_rpc_port < 1024 or base_rpc_port > 65533:
        raise RunnerPlanError("base RPC port must be in 1024..65533")
    if port_group_stride < 3:
        raise RunnerPlanError("port_group_stride must be at least 3")

    compiled = _load_json(index_path, "compiled C4 protocol")
    if compiled.get("schema_version") != COMPILED_SCHEMA_VERSION:
        raise RunnerPlanError(
            f"unexpected compiled C4 schema: {compiled.get('schema_version')!r}"
        )
    if compiled.get("experiment_id") != C4_EXPERIMENT_ID:
        raise RunnerPlanError(
            f"unexpected compiled C4 experiment_id: {compiled.get('experiment_id')!r}"
        )
    _validate_capture(compiled.get("capture"), "compiled capture")
    registries = _object(compiled.get("registries"), "compiled registries")
    if set(registries) != {"asset_registry", "scene_registry"}:
        raise RunnerPlanError("compiled registries keys differ")
    registry_inputs = {
        key: _validate_registry_link(index_path, registries[key], key)
        for key in ("asset_registry", "scene_registry")
    }
    groups = [
        _validate_protocol_entry(index_path, value, index)
        for index, value in enumerate(_list(compiled.get("protocols"), "protocols"))
    ]
    if not groups:
        raise RunnerPlanError("compiled protocols must not be empty")

    group_ids = [str(group["group_id"]) for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise RunnerPlanError("compiled protocol IDs must be unique")
    protocol_paths = [os.path.normcase(str(group["protocol_path"])) for group in groups]
    if len(protocol_paths) != len(set(protocol_paths)):
        raise RunnerPlanError("each map group must own a distinct protocol file")
    unique_maps = {str(group["map"]) for group in groups}
    if len(unique_maps) < 4:
        raise RunnerPlanError(
            f"C4 must cover at least four distinct CARLA maps, got {len(unique_maps)}"
        )
    all_layout_ids = [
        str(layout_id) for group in groups for layout_id in group["layout_ids"]
    ]
    if len(all_layout_ids) != len(set(all_layout_ids)):
        raise RunnerPlanError("layout IDs must be globally unique across map groups")
    if len(all_layout_ids) < 8:
        raise RunnerPlanError(
            f"C4 must cover at least eight distinct layouts, got {len(all_layout_ids)}"
        )

    admission = _object(compiled.get("admission"), "compiled admission")
    expected = {
        "expected_map_count": len(unique_maps),
        "expected_protocol_count": len(groups),
        "expected_layout_count": len(all_layout_ids),
        "expected_episode_count": sum(len(group["episodes"]) for group in groups),
        "expected_sensor_count": len(SENSOR_ORDER),
        "expected_shard_count": len(groups) * len(SENSOR_ORDER),
    }
    for key, actual in expected.items():
        if _integer(admission.get(key), f"compiled admission {key}") != actual:
            raise RunnerPlanError(f"compiled admission {key} is inconsistent")

    allocated_ports: set[int] = set()
    for group_ordinal, group in enumerate(groups):
        rpc_port = base_rpc_port + group_ordinal * port_group_stride
        ports = [rpc_port, rpc_port + 1, rpc_port + 2]
        if ports[-1] > 65535:
            raise RunnerPlanError(f"port plan exceeds 65535 for {group['group_id']}: {ports}")
        overlap = allocated_ports.intersection(ports)
        if overlap:
            raise RunnerPlanError(f"port groups overlap: {sorted(overlap)}")
        reserved = RESERVED_LEGACY_PORTS.intersection(ports)
        if reserved:
            raise RunnerPlanError(
                f"C4 refuses legacy/shared CARLA ports 2000..2022: {sorted(reserved)}"
            )
        allocated_ports.update(ports)
        group["ordinal"] = group_ordinal
        group["rpc_port"] = rpc_port
        group["ports"] = ports

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment_id": C4_EXPERIMENT_ID,
        "compiled_protocol_path": os.fspath(index_path),
        "compiled_protocol_sha256": _sha256_file(index_path),
        "compiled_bundle_root": os.fspath(index_path.parent),
        "resolution": [1280, 720],
        "sensor_order": list(SENSOR_ORDER),
        "map_count": len(unique_maps),
        "layout_count": len(all_layout_ids),
        "episode_count": expected["expected_episode_count"],
        "group_count": len(groups),
        "shard_count": expected["expected_shard_count"],
        "base_rpc_port": base_rpc_port,
        "port_group_stride": port_group_stride,
        "all_ports": sorted(allocated_ports),
        "registry_inputs": registry_inputs,
        "runtime_admission": expected,
        "map_layout_groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-protocol", type=Path, required=True)
    parser.add_argument("--base-rpc-port", type=int, default=24000)
    parser.add_argument("--port-group-stride", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = build_runner_plan(
            args.compiled_protocol,
            base_rpc_port=args.base_rpc_port,
            port_group_stride=args.port_group_stride,
        )
    except (OSError, RunnerPlanError) as exc:
        print(f"DTR_CARLA_C4_RUNNER_PLAN_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
