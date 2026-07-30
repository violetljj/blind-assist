#!/usr/bin/env python3
"""Validate the pre-replay implementation identity and execution firewall."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA = "blindassist_dual_loop_radial_geometry_implementation_lock_v1"
PROTOCOL_ID = "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0"
FORBIDDEN_PRODUCER_IMPORT_TOKENS = (
    "evaluate_replay",
    "prepare_revel",
    "align_revel",
    "audit_revel",
    "rosbags",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def validate(lock_path: Path, repository_root: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if lock.get("schema") != SCHEMA:
        errors.append("SCHEMA")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("PROTOCOL_ID")
    if lock.get("implementation_status") != "FROZEN_FOR_INDEPENDENT_REVIEW":
        errors.append("IMPLEMENTATION_STATUS")
    authority = lock.get("execution_authority", {})
    if authority.get("full_producer_replay_authorized") is not False:
        errors.append("PRODUCER_AUTHORITY")
    if authority.get("truth_join_authorized") is not False:
        errors.append("TRUTH_JOIN_AUTHORITY")
    if authority.get("old_f1b_decision_access_authorized") is not False:
        errors.append("OLD_F1B_AUTHORITY")

    for label in ("design_lock", "input_freeze_manifest"):
        binding = lock.get("bindings", {}).get(label, {})
        path = repository_root / str(binding.get("path", ""))
        if not path.is_file():
            errors.append(f"{label.upper()}_MISSING")
        elif sha256_file(path) != binding.get("sha256"):
            errors.append(f"{label.upper()}_SHA256")

    for binding in lock.get("file_bindings", []):
        path = repository_root / str(binding.get("path", ""))
        if not path.is_file():
            errors.append(f"FILE_MISSING:{binding.get('path')}")
        elif sha256_file(path) != binding.get("sha256"):
            errors.append(f"FILE_SHA256:{binding.get('path')}")

    module_dir = lock_path.resolve().parents[3] / "scripts" / "research" / "dual_loop_radial_geometry_lite_r0"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    geometry = importlib.import_module("radial_geometry")
    if geometry.PARAMETER_SHA256 != lock.get("parameter_identity", {}).get("sha256"):
        errors.append("PARAMETER_SHA256")
    if geometry.TTL_NS != 100_000_000 or tuple(geometry.ARMS) != (
        "BBOX_LOG_AREA_GROWTH",
        "ROI_SPARSE_RADIAL_FLOW",
    ):
        errors.append("FROZEN_INTERFACE")

    producer_path = repository_root / "scripts/research/dual_loop_radial_geometry_lite_r0/run_replay.py"
    imports = _producer_imports(producer_path)
    for name in imports:
        if any(token in name for token in FORBIDDEN_PRODUCER_IMPORT_TOKENS):
            errors.append(f"PRODUCER_FORBIDDEN_IMPORT:{name}")
    producer_text = producer_path.read_text(encoding="utf-8")
    for required in (
        "FORBIDDEN_PATH_TOKENS",
        "truth_joined",
        "replay_input_sha256",
        "output_sha256",
    ):
        if required not in producer_text:
            errors.append(f"PRODUCER_FIREWALL_FIELD:{required}")

    fixture = lock.get("fixture_verification", {})
    if fixture.get("test_count") != 24 or fixture.get("status") != "PASS":
        errors.append("FIXTURE_RECEIPT")
    if lock.get("result_model", {}).get("offline_replay_status") != "NOT_RUN":
        errors.append("REPLAY_STATUS")
    if lock.get("result_model", {}).get("scientific_outcome") != "NOT_RUN":
        errors.append("SCIENTIFIC_STATUS")
    return {
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "file_bindings": len(lock.get("file_bindings", [])),
        "producer_imports": sorted(imports),
        "parameter_sha256": geometry.PARAMETER_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate(args.lock.resolve(), args.repository_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
