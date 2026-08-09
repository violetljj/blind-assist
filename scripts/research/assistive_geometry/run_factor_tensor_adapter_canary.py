#!/usr/bin/env python3
"""Run the frozen zero-parameter FactorTensorAdapter synthetic canary."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .factor_tensor_adapter import AdapterError, adapt_factor_tensor, canonical_json_bytes, canonical_sha256
    from .geometry_r2_reducer import ReducerError, reduce_frame, state_map
except ImportError:  # Direct script execution.
    from factor_tensor_adapter import AdapterError, adapt_factor_tensor, canonical_json_bytes, canonical_sha256
    from geometry_r2_reducer import ReducerError, reduce_frame, state_map


LOCK_SCHEMA = "blindassist_assistive_geometry_r2_f1_factortensor_adapter_implementation_canary_lock_v1"
RESULT_SCHEMA = "blindassist_assistive_geometry_r2_f1_factortensor_adapter_canary_result_v1"
LOCK_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_IMPLEMENTATION_CANARY_LOCK"
SUITE_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_R0"
PASS_TERMINAL = "R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS"
FAIL_TERMINAL = "R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_FAIL"
OUTPUT_ROOT = "artifacts.local/evidence/assistive-geometry/r2-f1-factortensor-adapter-r0"
PASS_SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK"
EXPECTED_BINDING_ROLES = {
    "PROTOCOL_LOCK",
    "SYNTHETIC_FIXTURE",
    "F1_FACTOR_SCHEMA",
    "F0_REDUCER",
    "F0_REDUCER_FIXTURE",
    "ADAPTER_IMPLEMENTATION",
    "CANARY_RUNNER",
    "FOCUSED_TESTS",
}
EXPECTED_CASE_IDS = (
    "nominal_landscape_single_component",
    "portrait_equivalent_single_component",
    "local_component_depth_missing",
    "high_depth_uncertainty_monotone",
    "support_invalid_fail_closed",
    "geometry_receipt_identity_mismatch",
    "two_components_canonical_left_then_right",
    "bridge_pixel_merges_components",
)
EXPECTED_GATE_IDS = tuple(f"A{index:02d}" for index in range(1, 11))
PREDICTION_FIELDS = {
    "depth_shape_positive_hw",
    "log_metric_scale_m_scalar",
    "depth_log_sigma_hw",
    "depth_valid_probability_hw",
    "metric_scale_valid",
    "support_probability_hw",
    "support_plane_normal_camera_xyz",
    "camera_height_m",
    "support_residual_sigma_m",
    "support_valid",
    "obstacle_evidence_probability_hw",
    "boundary_probability_hw",
    "boundary_localization_sigma_px_hw",
    "evidence_valid_hw",
}
REQUIRED_F0_FIELDS = {
    "schema",
    "frame_id",
    "factor_identity",
    "input_geometry.k_valid",
    "input_geometry.transform_valid",
    "input_geometry.gravity_valid",
    "input_geometry.gravity_up_camera",
    "input_geometry.orientation",
    "depth_scale.valid",
    "depth_scale.scale_m",
    "depth_scale.scale_sigma_m",
    "support.valid",
    "support.normal_camera",
    "support.normal_sigma_rad",
    "support.camera_height_m",
    "support.height_sigma_m",
    "support.residual_sigma_m",
    "boundary.valid",
    "boundary.coverage",
    "boundary.obstacles",
    "obstacle.depth_valid",
    "obstacle.depth_shape_forward",
    "obstacle.depth_shape_sigma",
    "obstacle.lateral_center_m",
    "obstacle.lateral_half_width_m",
    "obstacle.boundary_sigma_m",
    "obstacle.evidence_probability",
    "obstacle.evidence_sigma",
}


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = copy.deepcopy(base)
        for key, value in patch.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(patch)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def verify_lock(lock: dict[str, Any], repo_root: Path) -> dict[str, dict[str, Any]]:
    require(lock.get("schema") == LOCK_SCHEMA, "lock schema mismatch")
    require(lock.get("lock_id") == LOCK_ID, "lock identity mismatch")
    require(lock.get("status") == "FROZEN_PRE_EXECUTION", "lock is not pre-execution frozen")
    require(lock.get("suite_id") == SUITE_ID, "suite identity mismatch")
    require(tuple(lock.get("case_ids", [])) == EXPECTED_CASE_IDS, "frozen case identity/order drift")
    require(tuple(lock.get("gate_ids", [])) == EXPECTED_GATE_IDS, "frozen gate identity/order drift")
    execution = lock.get("execution")
    require(isinstance(execution, dict), "execution contract missing")
    require(execution.get("output_root") == OUTPUT_ROOT, "exclusive output root drift")
    require(execution.get("network") is False and execution.get("gpu") is False and execution.get("device") is False, "execution scope exceeds CPU-only local canary")
    require(execution.get("real_data") is False and execution.get("task_outcome") is False and execution.get("training") is False, "forbidden execution authority")
    require(isinstance(execution.get("timeout_seconds"), int) and 0 < execution["timeout_seconds"] <= 30, "timeout is not frozen and bounded")
    require(isinstance(execution.get("max_output_bytes"), int) and 0 < execution["max_output_bytes"] <= 1024 * 1024, "output budget is not frozen and bounded")
    authority = lock.get("authority")
    require(isinstance(authority, dict), "authority block missing")
    require(authority.get("adapter_implementation") is True and authority.get("synthetic_canary") is True, "adapter canary authority missing")
    forbidden = ("real_data", "labels", "model", "training", "f1_execution", "f2", "device", "default_app")
    require(all(authority.get(key) is False for key in forbidden), "authority ceiling drift")
    successor = lock.get("pass_successor")
    require(isinstance(successor, dict), "pass successor missing")
    require(successor.get("id") == PASS_SUCCESSOR and successor.get("execution_authority") is False, "pass successor drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_ROLES), "binding cardinality drift")
    roles = [item.get("role") for item in bindings if isinstance(item, dict)]
    require(len(roles) == len(set(roles)) and set(roles) == EXPECTED_BINDING_ROLES, "binding role set drift")
    verified: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "sha256", "bytes"}, "binding key set drift")
        role, relative, expected_sha, expected_bytes = (binding[key] for key in ("role", "path", "sha256", "bytes"))
        require(isinstance(relative, str) and isinstance(expected_sha, str) and isinstance(expected_bytes, int), f"binding {role} is malformed")
        path = repo_root / relative
        require(path.is_file(), f"bound file missing: {relative}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        require(actual_sha == expected_sha.upper() and actual_bytes == expected_bytes, f"binding drift: {role}")
        verified[str(role)] = {"path": relative, "sha256": actual_sha, "bytes": actual_bytes}
    return verified


def run_worker(script: Path, adapter_input: dict[str, Any], timeout_seconds: int) -> bytes:
    completed = subprocess.run(
        [sys.executable, str(script), "--worker"],
        input=canonical_json_bytes(adapter_input),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    require(completed.returncode == 0, f"adapter worker failed: {completed.stderr.decode('utf-8', errors='replace')}")
    return completed.stdout


def states(output: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return state_map(output)


def all_unknown(value: dict[str, tuple[str, ...]]) -> bool:
    return all(state == "UNKNOWN" for band in value.values() for state in band)


def no_stronger(before: dict[str, tuple[str, ...]], after: dict[str, tuple[str, ...]]) -> bool:
    return all(next_state in {prior_state, "UNKNOWN"} for band in before for prior_state, next_state in zip(before[band], after[band]))


def adapter_static_firewall(adapter_path: Path) -> tuple[bool, list[str]]:
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {"__future__", "hashlib", "json", "math", "collections", "typing"}
    forbidden_imports = sorted(imports - allowed)
    return not forbidden_imports, forbidden_imports


def mutated_sigma_inputs(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    mutations = (
        ("depth_log_sigma", {"prediction": {"depth_scale": {"depth_log_sigma_hw": [[-0.6931471805599453] * 4 for _ in range(3)]}}}),
        ("support_residual", {"prediction": {"support_surface": {"support_residual_sigma_m": 0.2}}}),
        ("support_normal", {"calibration_receipt": {"support_normal_sigma_rad": 0.1}}),
        ("support_height", {"calibration_receipt": {"support_height_sigma_m": 0.2}}),
        ("boundary_localization", {"prediction": {"obstacle_boundary_evidence": {"boundary_localization_sigma_px_hw": [[0.5] * 4 for _ in range(3)]}}}),
        ("evidence_sigma", {"calibration_receipt": {"evidence_sigma_floor": 0.2}}),
        ("scale_sigma", {"calibration_receipt": {"scale_relative_sigma_floor": 0.3}}),
    )
    return [(name, deep_merge(base, patch)) for name, patch in mutations]


def execute(lock_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    repo_root = Path(__file__).resolve().parents[3]
    lock = load_json(lock_path)
    verified = verify_lock(lock, repo_root)
    output_root = repo_root / OUTPUT_ROOT
    require(not output_root.exists(), f"exclusive output root already exists: {OUTPUT_ROOT}")
    fixture = load_json(repo_root / verified["SYNTHETIC_FIXTURE"]["path"])
    protocol = load_json(repo_root / verified["PROTOCOL_LOCK"]["path"])
    f0_fixture = load_json(repo_root / verified["F0_REDUCER_FIXTURE"]["path"])
    require(fixture.get("suite_id") == SUITE_ID, "fixture suite identity mismatch")
    cases = fixture.get("cases")
    require(isinstance(cases, list) and tuple(item.get("id") for item in cases) == EXPECTED_CASE_IDS, "fixture case order drift")
    base = fixture.get("base_input")
    profile = f0_fixture.get("reducer_profile")
    require(isinstance(base, dict) and isinstance(profile, dict), "fixture base/profile missing")
    timeout_seconds = int(lock["execution"]["timeout_seconds"])

    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "scripts.research.assistive_geometry.test_factor_tensor_adapter"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
        text=True,
    )

    records: list[dict[str, Any]] = []
    frames: dict[str, dict[str, Any]] = {}
    state_maps: dict[str, dict[str, tuple[str, ...]]] = {}
    all_replays_equal = True
    all_case_expectations = True
    script = Path(__file__).resolve()
    for case in cases:
        case_id = str(case["id"])
        adapter_input = deep_merge(base, case.get("patch", {}))
        direct = adapt_factor_tensor(copy.deepcopy(adapter_input))
        first_bytes = run_worker(script, adapter_input, timeout_seconds)
        second_bytes = run_worker(script, adapter_input, timeout_seconds)
        replay_equal = first_bytes == second_bytes == canonical_json_bytes(direct)
        all_replays_equal = all_replays_equal and replay_equal
        frame = json.loads(first_bytes.decode("utf-8"))
        reducer_output = reduce_frame(frame, profile)
        case_states = states(reducer_output)
        frames[case_id] = frame
        state_maps[case_id] = case_states
        expected = case["expected"]
        valid = frame["input_geometry"]["k_valid"] and frame["input_geometry"]["transform_valid"]
        actual_terminal = "ADAPTER_FRAME_VALID" if valid else "ADAPTER_INPUT_INVALID"
        checks = [actual_terminal == expected["terminal"]]
        if "obstacle_count" in expected:
            checks.append(len(frame["boundary"]["obstacles"]) == expected["obstacle_count"])
        if "obstacle_depth_valid" in expected:
            checks.append([item["depth_valid"] for item in frame["boundary"]["obstacles"]] == expected["obstacle_depth_valid"])
        if "support_valid" in expected:
            checks.append(frame["support"]["valid"] == expected["support_valid"])
        if expected.get("all_unknown") is True:
            checks.append(all_unknown(case_states))
        if "forbidden_state" in expected:
            checks.append(all(expected["forbidden_state"] not in band for band in case_states.values()))
        if expected.get("canonical_lateral_order") == "STRICT_ASCENDING":
            centers = [item["lateral_center_m"] for item in frame["boundary"]["obstacles"]]
            checks.append(all(left < right for left, right in zip(centers, centers[1:])))
        case_passed = all(checks) and replay_equal
        all_case_expectations = all_case_expectations and case_passed
        records.append({
            "case_id": case_id,
            "passed": case_passed,
            "replay_byte_identical": replay_equal,
            "input_sha256": canonical_sha256(adapter_input),
            "frame_sha256": sha256_bytes(first_bytes),
            "frame": frame,
            "state_map": {band: list(values) for band, values in case_states.items()},
        })

    consumers = protocol.get("prediction_field_consumers", {})
    producers = protocol.get("reducer_field_producers", {})
    field_coverage = isinstance(consumers, dict) and set(consumers) == PREDICTION_FIELDS and all(isinstance(value, list) and value for value in consumers.values())
    producer_coverage = isinstance(producers, dict) and set(producers) == REQUIRED_F0_FIELDS and all(isinstance(value, str) and value for value in producers.values())

    nominal_id = "nominal_landscape_single_component"
    high_id = "high_depth_uncertainty_monotone"
    uncertainty_monotone = (
        frames[high_id]["depth_scale"]["scale_sigma_m"] >= frames[nominal_id]["depth_scale"]["scale_sigma_m"]
        and no_stronger(state_maps[nominal_id], state_maps[high_id])
    )
    support_fail_closed = not frames["support_invalid_fail_closed"]["support"]["valid"] and all_unknown(state_maps["support_invalid_fail_closed"])
    component_semantics = (
        len(frames["two_components_canonical_left_then_right"]["boundary"]["obstacles"]) == 2
        and frames["two_components_canonical_left_then_right"]["boundary"]["obstacles"][0]["lateral_center_m"]
        < frames["two_components_canonical_left_then_right"]["boundary"]["obstacles"][1]["lateral_center_m"]
        and len(frames["bridge_pixel_merges_components"]["boundary"]["obstacles"]) == 1
    )
    orientation_parity = state_maps[nominal_id] == state_maps["portrait_equivalent_single_component"]
    local_missing = (
        frames["local_component_depth_missing"]["boundary"]["obstacles"][0]["depth_valid"] is False
        and all("OCCUPIED_OBSERVED" not in band for band in state_maps["local_component_depth_missing"].values())
    )

    nominal_frame = frames[nominal_id]
    nominal_states = state_maps[nominal_id]
    sigma_checks: list[dict[str, Any]] = []
    for mutation_id, mutated_input in mutated_sigma_inputs(base):
        mutated_frame = adapt_factor_tensor(mutated_input)
        mutated_states = states(reduce_frame(mutated_frame, profile))
        nominal_lower = [item["evidence_probability"] - item["evidence_sigma"] for item in nominal_frame["boundary"]["obstacles"]]
        mutated_lower = [item["evidence_probability"] - item["evidence_sigma"] for item in mutated_frame["boundary"]["obstacles"]]
        lower_not_increased = len(nominal_lower) == len(mutated_lower) and all(after <= before + 1e-12 for before, after in zip(nominal_lower, mutated_lower))
        passed = no_stronger(nominal_states, mutated_states) and lower_not_increased
        sigma_checks.append({"mutation_id": mutation_id, "passed": passed})
    no_sigma_strengthening = all(item["passed"] for item in sigma_checks)

    adapter_path = repo_root / verified["ADAPTER_IMPLEMENTATION"]["path"]
    static_firewall, forbidden_imports = adapter_static_firewall(adapter_path)
    shortcut_input = copy.deepcopy(base)
    shortcut_input["prediction"]["factor_identity"]["nested"] = {"final_state": "CLEAR_OBSERVED"}
    try:
        adapt_factor_tensor(shortcut_input)
        shortcut_rejected = False
    except AdapterError as exc:
        shortcut_rejected = exc.code == "FORBIDDEN_FINAL_TASK_FIELD"
    graph_firewall = static_firewall and shortcut_rejected
    reducer_sha_ok = verified["F0_REDUCER"]["sha256"] == "2D6C26AD75B98911FD610FE0428D47584C877BA9AC7F091F768D98C035932092"

    gates = [
        {"id": "A01", "name": "FIELD_COVERAGE", "passed": field_coverage and producer_coverage, "detail": f"prediction={len(consumers) if isinstance(consumers, dict) else 0}/14 reducer={len(producers) if isinstance(producers, dict) else 0}/28"},
        {"id": "A02", "name": "DETERMINISTIC_REPLAY", "passed": all_replays_equal, "detail": f"independent_process_replays={sum(record['replay_byte_identical'] for record in records)}/8"},
        {"id": "A03", "name": "SCALE_UNCERTAINTY_MONOTONE", "passed": uncertainty_monotone, "detail": f"nominal_sigma={frames[nominal_id]['depth_scale']['scale_sigma_m']} high_sigma={frames[high_id]['depth_scale']['scale_sigma_m']}"},
        {"id": "A04", "name": "SUPPORT_FAIL_CLOSED", "passed": support_fail_closed, "detail": "support invalid reducer state is all UNKNOWN"},
        {"id": "A05", "name": "COMPONENT_SEMANTICS", "passed": component_semantics, "detail": "split=2 canonical left-to-right; bridged=1"},
        {"id": "A06", "name": "ORIENTATION_EQUIVARIANCE", "passed": orientation_parity, "detail": "display-upright portrait/landscape reducer state maps equal"},
        {"id": "A07", "name": "LOCAL_MISSING_DEPTH", "passed": local_missing, "detail": "local depth_valid=false and no OCCUPIED state"},
        {"id": "A08", "name": "NO_UNCERTAINTY_STRENGTHENING", "passed": no_sigma_strengthening, "detail": f"mutations={sigma_checks}"},
        {"id": "A09", "name": "GRAPH_AND_SHORTCUT_FIREWALL", "passed": graph_firewall, "detail": f"forbidden_imports={forbidden_imports} shortcut_rejected={shortcut_rejected}"},
        {"id": "A10", "name": "F0_BYTE_IDENTITY", "passed": reducer_sha_ok, "detail": verified["F0_REDUCER"]["sha256"]},
    ]
    elapsed_seconds = time.perf_counter() - started
    resource_precheck = elapsed_seconds <= timeout_seconds and test_run.returncode == 0
    passed = all_case_expectations and all(gate["passed"] for gate in gates) and resource_precheck
    lock_sha = sha256_file(lock_path)
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_head = "UNAVAILABLE"
    result = {
        "schema": RESULT_SCHEMA,
        "suite_id": SUITE_ID,
        "terminal": PASS_TERMINAL if passed else FAIL_TERMINAL,
        "passed": passed,
        "lock": {"path": relative_to_repo(lock_path, repo_root), "sha256": lock_sha, "bytes": lock_path.stat().st_size},
        "gates": gates,
        "summary": {
            "case_count": len(records),
            "case_pass_count": sum(record["passed"] for record in records),
            "independent_process_replay_count": sum(record["replay_byte_identical"] for record in records),
            "focused_test_exit_code": test_run.returncode,
            "focused_test_count": 10,
            "prediction_field_coverage": 14 if field_coverage else 0,
            "required_f0_field_coverage": 28 if producer_coverage else 0,
            "sigma_mutation_count": len(sigma_checks),
            "real_data_sample_count": 0,
            "task_outcome_read_count": 0,
            "learned_parameter_count": 0,
            "training_step_count": 0,
        },
        "resource_receipt": {
            "cpu_only": True,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "timeout_seconds": timeout_seconds,
            "within_timeout": elapsed_seconds <= timeout_seconds,
            "max_output_bytes": lock["execution"]["max_output_bytes"],
        },
        "verified_bindings": verified,
        "authority": {
            "claim": "ZERO_PARAMETER_FACTORTENSOR_TO_F0_FRAME_SYNTHETIC_MECHANICS_ONLY" if passed else "ADAPTER_CANARY_FAILED",
            "supervision_frontdoor_satisfied": False,
            "real_factor_learnability_established": False,
            "real_factor_headroom_established": False,
            "f1_execution_authority": False,
            "training_authority": False,
            "device_or_product_authority": False,
        },
        "unique_successor": {
            "id": PASS_SUCCESSOR if passed else None,
            "execution_authority": False,
            "scope": "Freeze a separate supervision source-and-label contract; do not materialize labels or train." if passed else "No successor is authorized from a failed adapter canary.",
        },
    }
    records_payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    receipt = {
        "schema": "blindassist_assistive_geometry_r2_f1_factortensor_adapter_execution_receipt_v1",
        "started_at_utc": started_utc,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": git_head,
        "command": lock["execution"]["command"],
        "lock_sha256": lock_sha,
        "focused_test_output": test_run.stdout,
        "process_replays": 16,
        "network": False,
        "gpu": False,
        "device": False,
        "real_data": False,
        "task_outcome": False,
    }
    payloads: dict[str, bytes] = {
        "result.json": pretty_json_bytes(result),
        "records.jsonl": records_payload,
        "execution-receipt.json": pretty_json_bytes(receipt),
    }
    total_bytes = 0
    for _ in range(8):
        result["resource_receipt"]["output_bytes"] = total_bytes
        payloads["result.json"] = pretty_json_bytes(result)
        manifest = {
            "schema": "blindassist_assistive_geometry_r2_f1_factortensor_adapter_manifest_v1",
            "files": [
                {"path": name, "sha256": sha256_bytes(payload), "bytes": len(payload)}
                for name, payload in sorted(payloads.items()) if name != "manifest.json"
            ],
        }
        payloads["manifest.json"] = pretty_json_bytes(manifest)
        updated_total = sum(len(payload) for payload in payloads.values())
        if updated_total == total_bytes:
            break
        total_bytes = updated_total
    require(result["resource_receipt"]["output_bytes"] == total_bytes, "output byte receipt failed to converge")
    require(total_bytes <= int(lock["execution"]["max_output_bytes"]), f"output budget exceeded: {total_bytes}")
    output_root.mkdir(parents=True)
    for name, payload in payloads.items():
        atomic_write(output_root / name, payload)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock", type=Path)
    group.add_argument("--worker", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker:
        try:
            adapter_input = json.loads(sys.stdin.buffer.read().decode("utf-8"))
            require(isinstance(adapter_input, dict), "worker input must be an object")
            sys.stdout.buffer.write(canonical_json_bytes(adapt_factor_tensor(adapter_input)))
            return 0
        except (AdapterError, CanaryError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"ADAPTER_WORKER_ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        result = execute(args.lock.resolve())
    except (AdapterError, ReducerError, CanaryError, OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"ADAPTER_CANARY_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "gates": result["gates"], "summary": result["summary"], "resource_receipt": result["resource_receipt"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
