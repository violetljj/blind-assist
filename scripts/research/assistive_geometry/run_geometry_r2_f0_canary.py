#!/usr/bin/env python3
"""Execute the synthetic-only Assistive Geometry R2 F0 kill gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .geometry_r2_reducer import (
        OUTPUT_SCHEMA,
        REDUCER_VERSION,
        ReducerError,
        band_for_lateral,
        canonical_sha256,
        iter_cells,
        load_profile,
        reduce_frame,
        state_map,
    )
except ImportError:  # Direct script execution.
    from geometry_r2_reducer import (
        OUTPUT_SCHEMA,
        REDUCER_VERSION,
        ReducerError,
        band_for_lateral,
        canonical_sha256,
        iter_cells,
        load_profile,
        reduce_frame,
        state_map,
    )


PROTOCOL_SCHEMA = "blindassist.research_protocol.v1"
FIXTURE_SCHEMA = "blindassist_assistive_geometry_r2_f0_fixture_suite_v1"
RESULT_SCHEMA = "blindassist_assistive_geometry_r2_f0_canary_result_v1"
PASS_TERMINAL = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS"
KILL_TERMINAL = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_KILL"
EXPECTED_CANARY_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY"
F1_SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK"
STATE_RANK_BY_HORIZON = {"CLEAR_OBSERVED": 0, "UNKNOWN": 1, "OCCUPIED_OBSERVED": 2}


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = copy.deepcopy(base)
        for key, value in patch.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(patch)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def verify_protocol(protocol: dict[str, Any], repo_root: Path) -> dict[str, str]:
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol schema mismatch")
    require(protocol.get("protocol_id") == f"{EXPECTED_CANARY_ID}_PROTOCOL", "protocol identity mismatch")
    authority = protocol.get("execution_authority")
    require(isinstance(authority, dict), "execution_authority is missing")
    require(authority.get("f0_synthetic_reducer_canary") is True, "F0 execution is not authorized")
    forbidden_authority = (
        "f1_train_only_factor_learnability",
        "f2_new_development_evaluation",
        "teacher_distillation",
        "temporal_modeling",
        "mobile_deployment",
        "calibration",
        "confirmation",
    )
    require(all(authority.get(key) is False for key in forbidden_authority), "authority exceeds F0 synthetic-only scope")
    require(protocol.get("unique_successor", {}).get("id") == F1_SUCCESSOR, "conditional successor identity mismatch")
    require(protocol.get("unique_successor", {}).get("execution_authority") is False, "F1 must remain non-executable")
    bindings = protocol.get("bindings")
    require(isinstance(bindings, list) and bindings, "protocol bindings are missing")
    verified: dict[str, str] = {}
    for binding in bindings:
        require(isinstance(binding, dict), "protocol binding must be an object")
        rel = binding.get("path")
        expected = binding.get("sha256")
        require(isinstance(rel, str) and isinstance(expected, str), "protocol binding path/SHA missing")
        path = repo_root / rel
        require(path.is_file(), f"bound file is absent: {rel}")
        actual = sha256_file(path)
        require(actual == expected.upper(), f"bound SHA mismatch: {rel}")
        verified[rel] = actual
    return verified


def normalized_output(output: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(output)
    value["frame_id"] = "PARITY_NORMALIZED"
    return value


def expected_state_map(case: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = case.get("expected_states")
    require(isinstance(raw, dict), f"case {case.get('id')} lacks expected_states")
    return {str(key): tuple(str(state) for state in states) for key, states in raw.items()}


def check_horizon_monotonicity(output: dict[str, Any]) -> bool:
    for band in output["bands"]:
        ranks = [STATE_RANK_BY_HORIZON[cell["state"]] for cell in band["cells"]]
        if ranks != sorted(ranks):
            return False
    return True


def check_output_contract(output: dict[str, Any]) -> bool:
    if output.get("schema") != OUTPUT_SCHEMA or output.get("reducer_version") != REDUCER_VERSION:
        return False
    bands = output.get("bands")
    if not isinstance(bands, list) or [item.get("band") for item in bands] != ["left", "center", "right"]:
        return False
    cells = list(iter_cells(output))
    if len(cells) != 9:
        return False
    for band in bands:
        if not isinstance(band.get("clearance_interval_m"), dict):
            return False
        for cell in band.get("cells", []):
            if cell.get("state") not in STATE_RANK_BY_HORIZON or not cell.get("reason_codes"):
                return False
    return True


def check_band_ownership(raw_profile: dict[str, Any]) -> dict[str, str | None]:
    profile = load_profile(raw_profile)
    probes = (-0.751, -0.75, -0.25, 0.25, 0.75, 0.751)
    actual = {str(probe): band_for_lateral(profile, probe) for probe in probes}
    expected = {
        "-0.751": None,
        "-0.75": "left",
        "-0.25": "center",
        "0.25": "right",
        "0.75": "right",
        "0.751": None,
    }
    require(actual == expected, f"band ownership mismatch: {actual}")
    return actual


def render_report(result: dict[str, Any]) -> str:
    gate_lines = "\n".join(
        f"- `{gate['id']}`: **{'PASS' if gate['passed'] else 'FAIL'}** — {gate['detail']}"
        for gate in result["gates"]
    )
    return f"""# Assistive Geometry R2 F0 synthetic factor geometry canary

- Terminal: `{result['terminal']}`
- Passed: `{str(result['passed']).lower()}`
- Reducer: `{result['reducer_version']}`
- Synthetic cases: `{result['summary']['case_count']}`
- Exact expected cases: `{result['summary']['exact_expected_case_count']}`
- A0-collapse counterexamples: `{result['summary']['a0_counterexample_count']}`
- Learned models / real datasets consumed: `0 / 0`

## Gates

{gate_lines}

## Authority ceiling

This result establishes only synthetic factor-to-geometry mechanics for the frozen reducer and fixture suite. It is not model accuracy, dataset, safety, mobile, temporal, Calibration, Confirmation, or deployment evidence. F1 execution authority remains `false`; a PASS authorizes only freezing a separate F1 TRAIN-only protocol.
"""


def execute(protocol_path: Path, fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    protocol = load_json(protocol_path)
    fixtures = load_json(fixture_path)
    verified_bindings = verify_protocol(protocol, repo_root)
    require(fixtures.get("schema") == FIXTURE_SCHEMA, "fixture schema mismatch")
    require(fixtures.get("suite_id") == EXPECTED_CANARY_ID, "fixture suite identity mismatch")
    raw_profile = fixtures.get("reducer_profile")
    require(isinstance(raw_profile, dict), "reducer profile is missing")
    ownership = check_band_ownership(raw_profile)
    base = fixtures.get("base_factor_frame")
    cases = fixtures.get("cases")
    require(isinstance(base, dict) and isinstance(cases, list) and cases, "base frame or cases missing")

    required_suites = set(fixtures.get("required_suites", []))
    covered_suites: set[str] = set()
    case_results: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[tuple[int, str, dict[str, tuple[str, ...]]]]] = defaultdict(list)
    parity_groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    error_controls = 0
    expected_error_case_count = 0
    exact_expected = 0
    deterministic_replays = 0
    a0_count = 0
    a0_violations: list[str] = []
    horizon_violations: list[str] = []
    contract_violations: list[str] = []

    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get("id"), str), "case identity invalid")
        case_id = case["id"]
        suites = set(str(item) for item in case.get("suites", []))
        covered_suites.update(suites)
        frame = deep_merge(base, case.get("patch", {}))
        frame["frame_id"] = case_id
        expected_error = case.get("expected_error_code")
        if expected_error is not None:
            expected_error_case_count += 1
            try:
                reduce_frame(frame, raw_profile)
                actual_error = None
            except ReducerError as exc:
                actual_error = exc.code
            passed = actual_error == expected_error
            error_controls += int(passed)
            case_results.append({"id": case_id, "suites": sorted(suites), "passed": passed, "expected_error_code": expected_error, "actual_error_code": actual_error})
            continue

        first = reduce_frame(frame, raw_profile)
        second = reduce_frame(copy.deepcopy(frame), copy.deepcopy(raw_profile))
        deterministic = canonical_sha256(first) == canonical_sha256(second)
        deterministic_replays += int(deterministic)
        actual_states = state_map(first)
        expected_states = expected_state_map(case)
        exact = actual_states == expected_states
        exact_expected += int(exact)
        output_contract = check_output_contract(first)
        if not output_contract:
            contract_violations.append(case_id)
        horizon_monotonic = check_horizon_monotonicity(first)
        if not horizon_monotonic:
            horizon_violations.append(case_id)
        positive_bands = set(str(item) for item in case.get("positive_occupancy_bands", []))
        occupied_bands = {band for band, _, state in iter_cells(first) if state == "OCCUPIED_OBSERVED"}
        no_unsupported_occupancy = occupied_bands <= positive_bands
        if "a0_collapse_counterexample" in suites:
            a0_count += 1
            if not no_unsupported_occupancy:
                a0_violations.append(case_id)
        passed = exact and deterministic and output_contract and horizon_monotonic and no_unsupported_occupancy
        outputs[case_id] = first
        if "monotonic_group" in case:
            groups[str(case["monotonic_group"])].append((int(case["monotonic_order"]), case_id, actual_states))
        if "parity_group" in case:
            parity_groups[str(case["parity_group"])].append((case_id, normalized_output(first)))
        case_results.append({
            "id": case_id,
            "suites": sorted(suites),
            "passed": passed,
            "expected_states": {key: list(value) for key, value in expected_states.items()},
            "actual_states": {key: list(value) for key, value in actual_states.items()},
            "deterministic_replay": deterministic,
            "output_contract": output_contract,
            "horizon_monotonic": horizon_monotonic,
            "positive_occupancy_bands": sorted(positive_bands),
            "actual_occupied_bands": sorted(occupied_bands),
            "unsupported_occupancy_absent": no_unsupported_occupancy,
            "output_sha256": canonical_sha256(first),
        })

    monotonic_violations: list[dict[str, Any]] = []
    for group, members in groups.items():
        ordered = sorted(members)
        require([item[0] for item in ordered] == list(range(len(ordered))), f"monotonic group {group} order is not contiguous")
        for (_, previous_id, previous), (_, current_id, current) in zip(ordered, ordered[1:]):
            for band in ("left", "center", "right"):
                for index, (before, after) in enumerate(zip(previous[band], current[band])):
                    if after not in {before, "UNKNOWN"}:
                        monotonic_violations.append({"group": group, "before_case": previous_id, "after_case": current_id, "band": band, "horizon_index": index, "before": before, "after": after})

    parity_violations: list[str] = []
    for group, members in parity_groups.items():
        require(len(members) >= 2, f"parity group {group} has fewer than two members")
        reference = canonical_sha256(members[0][1])
        if any(canonical_sha256(output) != reference for _, output in members[1:]):
            parity_violations.append(group)

    case_failures = [item["id"] for item in case_results if not item["passed"]]
    gates = [
        {"id": "F0_G01_PROTOCOL_AND_SHA_BINDINGS", "passed": bool(verified_bindings), "detail": f"{len(verified_bindings)} frozen files verified"},
        {"id": "F0_G02_REQUIRED_SYNTHETIC_SUITE_COVERAGE", "passed": required_suites <= covered_suites, "detail": f"covered={sorted(covered_suites)}"},
        {"id": "F0_G03_EXACT_ORACLE_EXPECTATIONS", "passed": not case_failures, "detail": f"failed_cases={case_failures}"},
        {"id": "F0_G04_DETERMINISTIC_REPLAY", "passed": deterministic_replays == len(cases) - expected_error_case_count, "detail": f"exact_replays={deterministic_replays}"},
        {"id": "F0_G05_INTERVAL_OUTPUT_CONTRACT", "passed": not contract_violations, "detail": f"violations={contract_violations}"},
        {"id": "F0_G06_HORIZON_MONOTONICITY", "passed": not horizon_violations, "detail": f"violations={horizon_violations}"},
        {"id": "F0_G07_UNCERTAINTY_TO_UNKNOWN_MONOTONICITY", "passed": len(groups) >= 4 and not monotonic_violations, "detail": f"groups={sorted(groups)} violations={monotonic_violations}"},
        {"id": "F0_G08_A0_COLLAPSE_COUNTEREXAMPLES", "passed": a0_count >= 8 and not a0_violations, "detail": f"cases={a0_count} unsupported_occupancy={a0_violations}"},
        {"id": "F0_G09_ORIENTATION_AND_BAND_OWNERSHIP", "passed": not parity_violations and len(parity_groups) >= 1, "detail": f"parity_violations={parity_violations} ownership={ownership}"},
        {"id": "F0_G10_FINAL_TASK_SHORTCUT_REJECTED", "passed": error_controls >= 1, "detail": f"passed_negative_controls={error_controls}"},
    ]
    passed = all(gate["passed"] for gate in gates)
    result = {
        "schema": RESULT_SCHEMA,
        "canary_id": EXPECTED_CANARY_ID,
        "terminal": PASS_TERMINAL if passed else KILL_TERMINAL,
        "passed": passed,
        "reducer_version": REDUCER_VERSION,
        "protocol": {"path": relative_to_repo(protocol_path, repo_root), "sha256": sha256_file(protocol_path)},
        "fixtures": {"path": relative_to_repo(fixture_path, repo_root), "sha256": sha256_file(fixture_path)},
        "verified_bindings": verified_bindings,
        "gates": gates,
        "summary": {
            "case_count": len(cases),
            "exact_expected_case_count": exact_expected,
            "deterministic_replay_count": deterministic_replays,
            "expected_error_control_count": error_controls,
            "a0_counterexample_count": a0_count,
            "monotonic_group_count": len(groups),
            "real_dataset_sample_count": 0,
            "learned_model_count": 0,
            "training_step_count": 0,
        },
        "case_results": case_results,
        "authority": {
            "claim": "F0_SYNTHETIC_REDUCER_MECHANICS_ONLY" if passed else "R2_F0_REDUCER_VERSION_KILLED",
            "f1_execution_authority": False,
            "development_execution_authority": False,
            "teacher_distillation_authority": False,
            "temporal_authority": False,
            "mobile_authority": False,
            "calibration_authority": False,
            "confirmation_authority": False,
            "pass_only_authorizes": "FREEZE_A_SEPARATE_F1_TRAIN_ONLY_PROTOCOL" if passed else None,
        },
        "unique_successor": {"id": F1_SUCCESSOR if passed else None, "execution_authority": False},
    }
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "geometry_r2_f0_canary_result.json", result)
    atomic_write_text(output_dir / "geometry_r2_f0_canary_report.md", render_report(result))
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = execute(args.protocol.resolve(), args.fixtures.resolve(), args.output_dir.resolve())
    except (CanaryError, ReducerError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"F0_CANARY_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "gates": result["gates"], "summary": result["summary"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
