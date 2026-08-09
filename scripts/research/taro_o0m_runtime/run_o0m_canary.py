#!/usr/bin/env python3
"""Run the hash-bound, one-shot TARO O0M synthetic mechanics canary."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

try:
    from .o0m_mechanics import (
        ARMS,
        FACTORS,
        MODES,
        MechanicsError,
        apply_factorial_arm,
        canonical_json_bytes,
        canonical_sha256,
        evaluate_action_filter,
        solve_identifiability,
    )
except ImportError:  # Direct script execution.
    from o0m_mechanics import (
        ARMS,
        FACTORS,
        MODES,
        MechanicsError,
        apply_factorial_arm,
        canonical_json_bytes,
        canonical_sha256,
        evaluate_action_filter,
        solve_identifiability,
    )


LOCK_SCHEMA = "blindassist.taro.o0m.one_shot_execution_lock.v1"
LOCK_ID = "TARO_O0M_ONE_SHOT_EXECUTION_LOCK"
RESULT_SCHEMA = "blindassist.taro.o0m.synthetic_mechanics_result.v1"
PASS_TERMINAL = "TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS"
FAIL_TERMINAL = "TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_FAIL"
O0R_TERMINAL = "TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE"
EXPECTED_BINDING_PATHS = {
    "O0M_PROTOCOL": "docs/research/taro/TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK_2026-08-10.json",
    "O0M_FIXTURE": "docs/research/taro/TARO_O0M_EXECUTION_FIXTURE_SPEC_2026-08-10.json",
    "O0M_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0M_IMPLEMENTATION_LOCK_2026-08-10.json",
    "O0M_STATIC_VALIDATOR": "scripts/research/taro_o0m/validate_taro_o0m_protocol.py",
    "P0_NUMERIC_EVALUATOR": "scripts/research/taro/validate_taro_p0_protocol.py",
    "O0M_MECHANICS": "scripts/research/taro_o0m_runtime/o0m_mechanics.py",
    "O0M_RUNNER": "scripts/research/taro_o0m_runtime/run_o0m_canary.py",
    "O0M_TESTS": "scripts/research/taro_o0m_runtime/test_o0m_mechanics.py",
}
EXPECTED_AUTHORITY = {
    "o0m_execution": True,
    "real_data": False,
    "training": False,
    "active_prompt": False,
    "network": False,
    "gpu": False,
    "device": False,
    "confirmation": False,
    "deployment": False,
    "default_app": False,
}


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")),
    )
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
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
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _ident_input(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "motion",
        "measurement_jacobian_whitened",
        "query_jacobian_branches_m",
        "nominal_clearance_m",
        "receipt",
        "active_contact_clearances_m",
        "branch_nominal_clearances_m",
    )
    return {key: copy.deepcopy(case[key]) for key in keys if key in case}


def _scene_input(scene: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "family_id",
        "query_id",
        "observed_base_mean_m",
        "current_factor_error_m",
        "current_factor_valid",
        "oracle_factor_valid",
        "factor_provenance",
        "oracle_provenance",
        "factor_identity_sha256",
        "anchor_identity",
        "max_source_timestamp_ns",
    )
    return {key: copy.deepcopy(scene[key]) for key in keys}


def _action_input(action: dict[str, Any]) -> dict[str, Any]:
    return {"id": action["id"], "requires_body_motion": action["requires_body_motion"]}


def compute_actual_bundle(fixture: dict[str, Any]) -> dict[str, Any]:
    rule = fixture["identifiability_rule"]
    numeric = fixture["numeric_contract"]
    ident_rows: list[dict[str, Any]] = []
    for case in fixture["identifiability_cases"]:
        runtime_input = _ident_input(case)
        solved = solve_identifiability(runtime_input, rule)
        ident_rows.append(
            {
                "kind": "IDENTIFIABILITY",
                "id": case["id"],
                "runtime_input_sha256": canonical_sha256(runtime_input),
                "actual": solved["actual"],
                "expected": case["expected"],
                "diagnostics": solved["diagnostics"],
                "matches": solved["actual"] == case["expected"],
            }
        )

    factorial_rows: list[dict[str, Any]] = []
    for scene in fixture["factorial_scenes"]:
        runtime_scene = _scene_input(scene)
        for mode in MODES:
            for arm in ARMS:
                solved = apply_factorial_arm(runtime_scene, numeric, arm, mode)
                key = f"{arm}|{mode}"
                expected = scene["expected_records"][key]
                output = solved["record"]["output"]
                oracle_truth_closure = None
                if arm == "SCALE_SUPPORT_BOUNDARY" and mode == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY":
                    oracle_truth_closure = output["terminal"] == "EVALUATED" and abs(float(output["mean_m"]) - float(scene["truth_clearance_m"])) <= float(numeric["numeric_atol_m"])
                factorial_rows.append(
                    {
                        "kind": "FACTORIAL",
                        "scene_id": scene["id"],
                        "query_id": scene["query_id"],
                        "arm": arm,
                        "mode": mode,
                        "runtime_input_sha256": solved["runtime_input_sha256"],
                        "declared_factors": solved["declared_factors"],
                        "actual": solved["record"],
                        "expected": expected,
                        "matches": solved["record"] == expected,
                        "oracle_truth_closure": oracle_truth_closure,
                    }
                )

    action_rows: list[dict[str, Any]] = []
    for action in fixture["action_filter_cases"]:
        runtime_input = _action_input(action)
        actual = evaluate_action_filter(runtime_input)
        expected = {"allowed": action["expected_allowed"], "reason": action["expected_reason"]}
        action_rows.append(
            {
                "kind": "ACTION",
                "id": action["id"],
                "runtime_input_sha256": canonical_sha256(runtime_input),
                "actual": actual,
                "expected": expected,
                "matches": actual == expected,
            }
        )
    return {"identifiability": ident_rows, "factorial": factorial_rows, "actions": action_rows}


def _worker(fixture_path: str, connection: Any) -> None:
    try:
        fixture = load_json(Path(fixture_path))
        bundle = compute_actual_bundle(fixture)
        connection.send(
            {
                "ok": True,
                "bundle": bundle,
                "rss_bytes": psutil.Process().memory_info().rss,
            }
        )
    except BaseException as exc:  # Worker must return a bounded failure receipt.
        connection.send({"ok": False, "error": f"{type(exc).__name__}:{exc}", "rss_bytes": psutil.Process().memory_info().rss})
    finally:
        connection.close()


def run_worker(fixture_path: Path, timeout_s: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(str(fixture_path), child))
    process.start()
    child.close()
    if not parent.poll(timeout_s):
        process.terminate()
        process.join(5)
        raise CanaryError("worker timeout")
    payload = parent.recv()
    parent.close()
    process.join(5)
    require(process.exitcode == 0 and payload.get("ok") is True, f"worker failed: {payload}")
    return payload


def _reparameterization_check(fixture: dict[str, Any]) -> bool:
    case = _ident_input(fixture["identifiability_cases"][0])
    base = solve_identifiability(case, fixture["identifiability_rule"])
    q = np.asarray(
        [[0.8, -0.6, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.8, -0.6], [0.0, 0.0, 0.6, 0.8]],
        dtype=np.float64,
    )
    transformed = copy.deepcopy(case)
    transformed["id"] = "reparameterized_probe"
    transformed["measurement_jacobian_whitened"] = (np.asarray(case["measurement_jacobian_whitened"], dtype=np.float64) @ q.T).tolist()
    transformed["query_jacobian_branches_m"] = (np.asarray(case["query_jacobian_branches_m"], dtype=np.float64) @ q.T).tolist()
    changed = solve_identifiability(transformed, fixture["identifiability_rule"])
    if base["actual"] != changed["actual"]:
        return False
    base_projector = np.asarray(base["diagnostics"]["strong_projector"], dtype=np.float64)
    changed_projector = np.asarray(changed["diagnostics"]["strong_projector"], dtype=np.float64)
    return bool(np.allclose(q.T @ changed_projector @ q, base_projector, atol=1e-10, rtol=0.0))


def _uncertainty_check(fixture: dict[str, Any]) -> bool:
    numeric = fixture["numeric_contract"]
    widened = copy.deepcopy(numeric)
    widened["sigma_measurement_m"] *= 2.0
    widened["sigma_factor_baseline_m"] *= 2.0
    widened["sigma_factor_oracle_m"] *= 2.0
    for scene in fixture["factorial_scenes"]:
        runtime_scene = _scene_input(scene)
        for mode in MODES:
            for arm in ARMS:
                base = apply_factorial_arm(runtime_scene, numeric, arm, mode)["record"]["output"]
                high = apply_factorial_arm(runtime_scene, widened, arm, mode)["record"]["output"]
                if base["terminal"] != "EVALUATED":
                    continue
                if high["terminal"] != "EVALUATED" or float(high["halfwidth_m"]) < float(base["halfwidth_m"]):
                    return False
                if base["query_state"] == "UNKNOWN" and high["query_state"] != "UNKNOWN":
                    return False
                if base["query_state"] == "CLEAR_OBSERVED" and high["query_state"] == "OCCUPIED_OBSERVED":
                    return False
                if base["query_state"] == "OCCUPIED_OBSERVED" and high["query_state"] == "CLEAR_OBSERVED":
                    return False
    return True


def _leakage_check(fixture: dict[str, Any]) -> bool:
    scene = fixture["factorial_scenes"][0]
    runtime_scene = _scene_input(scene)
    baseline = apply_factorial_arm(runtime_scene, fixture["numeric_contract"], "SCALE", MODES[0])["record"]
    mutated = copy.deepcopy(scene)
    mutated["truth_clearance_m"] = 99.0
    if apply_factorial_arm(_scene_input(mutated), fixture["numeric_contract"], "SCALE", MODES[0])["record"] != baseline:
        return False
    try:
        contaminated = copy.deepcopy(runtime_scene)
        contaminated["future_oracle_outcome"] = "LEAK"
        apply_factorial_arm(contaminated, fixture["numeric_contract"], "SCALE", MODES[0])
        return False
    except MechanicsError:
        pass
    try:
        contaminated_case = _ident_input(fixture["identifiability_cases"][0])
        contaminated_case["b1_consumed_outcome"] = "LEAK"
        solve_identifiability(contaminated_case, fixture["identifiability_rule"])
        return False
    except MechanicsError:
        return True


def build_gates(fixture: dict[str, Any], bundle: dict[str, Any], replay_equal: bool) -> list[dict[str, Any]]:
    ident = bundle["identifiability"]
    factorial = bundle["factorial"]
    actions = bundle["actions"]
    all_records_match = all(row["matches"] for row in factorial)
    isolated = {
        "o0m_exec_scale_isolated": "SCALE",
        "o0m_exec_support_isolated": "SUPPORT",
        "o0m_exec_boundary_isolated": "BOUNDARY",
    }
    opportunity = all(
        abs(float(next(scene for scene in fixture["factorial_scenes"] if scene["id"] == scene_id)["current_factor_error_m"][factor])) > 0
        for scene_id, factor in isolated.items()
    )
    specificity = True
    for scene_id, factor in isolated.items():
        rows = [row for row in factorial if row["scene_id"] == scene_id and row["mode"] == MODES[0]]
        baseline_mean = next(row for row in rows if row["arm"] == "NONE")["actual"]["output"]["mean_m"]
        for row in rows:
            patched = set() if row["arm"] == "NONE" else set(row["arm"].split("_"))
            if factor not in patched and row["actual"]["output"]["mean_m"] != baseline_mean:
                specificity = False
    value_support = all(
        len({row["actual"]["common_support_sha256"] for row in factorial if row["scene_id"] == scene["id"] and row["mode"] == MODES[0]}) == 1
        for scene in fixture["factorial_scenes"]
    )
    degenerate = all(
        row["matches"] and row["actual"]["query_state"] == "UNKNOWN"
        for row in ident
        if row["id"] not in {"o0m_exec_full_state_underdetermined_query_identifiable_clear", "o0m_exec_fully_observable_occupied"}
    )
    states = {row["actual"]["output"]["query_state"] for row in factorial}
    compound = all(
        row["matches"]
        for row in factorial
        if row["scene_id"] in {"o0m_exec_compound", "o0m_exec_boundary_validity"}
    ) and {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"} <= states
    gates = [
        ("O0M_G01_BINDING_AND_INTEGRITY", True, "execution lock bindings matched and exclusive root was reserved"),
        ("O0M_G02_ORACLE_POSITIVE_CONTROL", all(row["oracle_truth_closure"] is True for row in factorial if row["oracle_truth_closure"] is not None), "all full-block all-oracle rows close to verifier truth"),
        ("O0M_G03_DISCRIMINATING_OPPORTUNITY", opportunity, "each isolated factor has a nonzero frozen baseline corruption"),
        ("O0M_G04_IDENTIFIABILITY_TRUTH", all(row["matches"] for row in ident), "all ten independent NumPy SVD outputs match frozen expectations"),
        ("O0M_G05_DEGENERATE_FAIL_CLOSED", degenerate, "all invalid, weak and nonsmooth cases remain UNKNOWN/abstain"),
        ("O0M_G06_INTERVENTION_PURITY", all_records_match and value_support, "all 80 records match and VALUE_ONLY common support is arm-invariant"),
        ("O0M_G07_FACTOR_SPECIFICITY", specificity, "noncausal value-only arms do not repair isolated factor means"),
        ("O0M_G08_COMPOUND_CLOSURE", compound, "compound and validity diagnostics close without state collapse"),
        ("O0M_G09_MONOTONICITY_AND_DETERMINISM", replay_equal and _uncertainty_check(fixture) and _reparameterization_check(fixture), "replay, uncertainty monotonicity and non-axis reparameterization pass"),
        ("O0M_G10_LEAKAGE_FIREWALL", _leakage_check(fixture) and all(row["matches"] for row in actions), "truth is verifier-only and future/B1 fields fail closed"),
    ]
    return [{"id": gate_id, "passed": passed, "detail": detail} for gate_id, passed, detail in gates]


def verify_execution_lock(lock: dict[str, Any], lock_path: Path, repo_root: Path) -> tuple[Path, dict[str, str]]:
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID, "execution lock identity mismatch")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "execution authority mismatch")
    one_shot = lock.get("one_shot")
    require(isinstance(one_shot, dict) and one_shot.get("consumed_at_lock") is False and one_shot.get("overwrite") is False and one_shot.get("rerun") is False, "one-shot policy mismatch")
    output_root = repo_root / str(one_shot.get("exclusive_artifact_root"))
    require(not output_root.exists(), f"exclusive artifact root already exists: {output_root}")
    budget = lock.get("resource_budget")
    require(isinstance(budget, dict) and budget.get("network") is False and budget.get("gpu") is False and budget.get("device") is False and budget.get("real_data") is False, "resource scope mismatch")
    required_environment = lock.get("required_environment")
    require(isinstance(required_environment, dict), "required environment missing")
    for key, expected in required_environment.items():
        require(os.environ.get(key) == str(expected), f"environment mismatch: {key}")
    expected_argv = lock.get("argv")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(repo_root).as_posix(), "--execution-lock", lock_path.resolve().relative_to(repo_root).as_posix()]
    require(expected_argv == actual_argv, f"argv mismatch: {actual_argv}")

    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS), "binding count mismatch")
    verified: dict[str, str] = {}
    roles: set[str] = set()
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "sha256", "bytes"}, "binding fields mismatch")
        role = binding["role"]
        path_string = binding["path"]
        require(role not in roles and EXPECTED_BINDING_PATHS.get(role) == path_string, f"binding role/path mismatch: {role}")
        roles.add(role)
        path = repo_root / path_string
        require(path.is_file(), f"bound file missing: {path_string}")
        actual_sha = sha256_file(path)
        require(actual_sha == binding["sha256"] and path.stat().st_size == binding["bytes"], f"bound file mismatch: {path_string}")
        verified[role] = actual_sha
    require(roles == set(EXPECTED_BINDING_PATHS), "binding role set mismatch")
    return output_root, verified


def execute(lock_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    started = time.perf_counter()
    lock = load_json(lock_path)
    output_root, verified = verify_execution_lock(lock, lock_path, repo_root)
    budget = lock["resource_budget"]
    fixture_path = repo_root / EXPECTED_BINDING_PATHS["O0M_FIXTURE"]
    fixture = load_json(fixture_path)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    first = run_worker(fixture_path, float(budget["timeout_s"]) / 3.0)
    second = run_worker(fixture_path, float(budget["timeout_s"]) / 3.0)
    first_bytes = canonical_json_bytes(first["bundle"])
    second_bytes = canonical_json_bytes(second["bundle"])
    replay_equal = first_bytes == second_bytes
    gates = build_gates(fixture, first["bundle"], replay_equal)
    elapsed = time.perf_counter() - started
    peak_rss = max(int(first["rss_bytes"]), int(second["rss_bytes"]), psutil.Process().memory_info().rss)
    resources_pass = elapsed <= float(budget["timeout_s"]) and peak_rss <= int(budget["peak_rss_bytes"])
    passed = all(gate["passed"] for gate in gates) and resources_pass
    records = first["bundle"]["identifiability"] + first["bundle"]["factorial"] + first["bundle"]["actions"]
    records_bytes = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": "TARO_O0M_ANALYTIC_MECHANICS_R0",
        "protocol_id": "TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK",
        "implementation_lock_id": "TARO_O0M_IMPLEMENTATION_LOCK",
        "execution_lock_id": LOCK_ID,
        "suite_id": fixture["suite_id"],
        "terminal": PASS_TERMINAL if passed else FAIL_TERMINAL,
        "passed": passed,
        "scientific_status": "SYNTHETIC_MECHANICS_PASS" if passed else "SYNTHETIC_MECHANICS_FAIL",
        "claim_eligibility": "SYNTHETIC_ANALYTIC_MECHANICS_ONLY" if passed else "NONE",
        "replay": {"count": 2, "byte_identical": replay_equal, "sha256": hashlib.sha256(first_bytes).hexdigest().upper()},
        "counts": {"identifiability": len(first["bundle"]["identifiability"]), "factorial": len(first["bundle"]["factorial"]), "actions": len(first["bundle"]["actions"])},
        "gates": gates,
        "resource_valid": resources_pass,
        "records_sha256": hashlib.sha256(records_bytes).hexdigest().upper(),
        "o0r_terminal": O0R_TERMINAL,
        "claim_ceiling": "Independent NumPy mechanics passed only on the frozen pre-whitened analytic O0M family. No real factor headroom, real evidence dedup/whitening, model, active-view, device, product or safety claim is established.",
    }
    result_bytes = pretty_json_bytes(result)
    receipt = {
        "schema": "blindassist.taro.o0m.execution_receipt.v1",
        "executed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": lock["argv"],
        "verified_bindings": verified,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "elapsed_s": round(elapsed, 6),
        "peak_rss_bytes": peak_rss,
        "timeout_s": budget["timeout_s"],
        "peak_rss_limit_bytes": budget["peak_rss_bytes"],
        "exclusive_root_reserved": True,
        "one_shot_consumed": True,
        "network_gpu_device_real_data": [False, False, False, False],
    }
    receipt_bytes = pretty_json_bytes(receipt)
    atomic_write(output_root / "result.json", result_bytes)
    atomic_write(output_root / "records.jsonl", records_bytes)
    atomic_write(output_root / "execution-receipt.json", receipt_bytes)
    manifest_files = {}
    for name in ("result.json", "records.jsonl", "execution-receipt.json"):
        path = output_root / name
        manifest_files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": "blindassist.taro.o0m.artifact_manifest.v1", "files": manifest_files}
    atomic_write(output_root / "manifest.json", pretty_json_bytes(manifest))
    total_bytes = sum(path.stat().st_size for path in output_root.iterdir() if path.is_file())
    require(total_bytes <= int(budget["max_output_bytes"]), f"output budget exceeded: {total_bytes}")
    print(json.dumps({"terminal": result["terminal"], "passed": passed, "output_root": output_root.as_posix(), "total_bytes": total_bytes}, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute(args.execution_lock.resolve())
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
