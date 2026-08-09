#!/usr/bin/env python3
"""Evaluate all three A0 seeds on the frozen DEVELOPMENT_SELECTION role."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.research.assistive_geometry.evaluate_b1_a0_synthetic import (
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_SEEDS,
    EvaluationError,
    _resolve_reference,
    _validate_checkpoint,
    _contains_forbidden_seed_selection,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now,
)


BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
STATES = {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}
ROLE = "DEVELOPMENT_SELECTION"
TERMINALS = {
    "PASS": "B1_A0_DEVELOPMENT_EVALUATION_PASS",
    "TASK": "B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES",
    "PROTOCOL": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_PROTOCOL_DRIFT",
    "SCHEMA": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_SCHEMA_INVALID",
    "DENOMINATOR": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_UNDEFINED_DENOMINATOR",
    "DATA_ROLE": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_DATA_ROLE_INVALID",
    "BEST_SEED": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_BEST_SEED_SELECTION_FORBIDDEN",
    "CHECKPOINT": "B1_A0_DEVELOPMENT_NOT_EVALUABLE_CHECKPOINT_INVALID",
    "INTERNAL": "B1_A0_DEVELOPMENT_EVALUATION_INTERNAL_FAILURE",
}


def require(condition: bool, terminal: str, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise EvaluationError(terminal, code, message, **context)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    require(
        protocol.get("schema")
        == "blindassist_assistive_geometry_b1_a0_development_evaluation_protocol_v1",
        "PROTOCOL",
        "PROTOCOL_SCHEMA_DRIFT",
        "formal Development evaluation protocol schema drift",
    )
    require(
        protocol.get("authority", {}).get("development_selection_evaluation") is True,
        "DATA_ROLE",
        "DEVELOPMENT_EVALUATION_NOT_ACTIVATED",
        "Development selection evaluation is not activated",
    )
    require(
        protocol.get("data_role", {}).get("name") == ROLE,
        "DATA_ROLE",
        "PROTOCOL_DATA_ROLE_DRIFT",
        "formal evaluation role drift",
    )
    require(
        tuple(protocol["aggregation"]["required_seeds"]) == EXPECTED_SEEDS,
        "PROTOCOL",
        "SEED_SET_DRIFT",
        "formal evaluation seed set drift",
    )
    for binding in protocol["implementation_bindings"].values():
        target = Path(binding["path"])
        if not target.is_absolute():
            target = Path(__file__).resolve().parents[3] / target
        require(target.is_file(), "PROTOCOL", "BINDING_MISSING", "implementation binding is missing", path=binding["path"])
        require(
            sha256_file(target) == binding["sha256"],
            "PROTOCOL",
            "BINDING_SHA_DRIFT",
            "implementation binding SHA drift",
            path=binding["path"],
        )
    return protocol, sha256_file(path)


def training_protocol_binding_for_seed(protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    key = "seed_29_retry_protocol" if seed == 29 else "formal_train_protocol"
    binding = protocol.get("bindings", {}).get(key)
    require(isinstance(binding, dict), "PROTOCOL", "TRAINING_PROTOCOL_BINDING_MISSING", "per-seed training protocol binding is missing", seed=seed, binding=key)
    return binding


def _load_bound_training_protocol(binding: dict[str, Any], expected_seeds: tuple[int, ...]) -> tuple[Path, dict[str, Any], str]:
    path = Path(binding.get("path", ""))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    require(path.is_file(), "PROTOCOL", "TRAINING_PROTOCOL_MISSING", "bound training protocol is missing", path=str(path))
    digest = sha256_file(path)
    require(digest == binding.get("sha256"), "PROTOCOL", "TRAINING_PROTOCOL_BINDING_SHA_MISMATCH", "bound training protocol SHA mismatch", path=str(path))
    bound = load_json(path)
    require(bound.get("schema") == "blindassist_assistive_geometry_b1_a0_formal_train_execution_protocol_v1", "PROTOCOL", "TRAINING_PROTOCOL_SCHEMA_DRIFT", "training protocol schema drift", path=str(path))
    execution = bound.get("execution", {})
    require(tuple(int(item) for item in execution.get("seeds", [])) == expected_seeds, "PROTOCOL", "TRAINING_PROTOCOL_SEED_DRIFT", "per-seed training protocol seed set drift", path=str(path))
    require(int(execution.get("epochs_per_seed", -1)) == 20 and int(execution.get("optimizer_steps_per_seed", -1)) == 6000, "PROTOCOL", "TRAINING_PROTOCOL_DURATION_DRIFT", "training protocol duration drift", path=str(path))
    require(tuple(int(item) for item in execution.get("retained_checkpoint_epochs", [])) == tuple(EXPECTED_CHECKPOINT_STEPS), "PROTOCOL", "TRAINING_PROTOCOL_CHECKPOINT_CADENCE_DRIFT", "training protocol checkpoint cadence drift", path=str(path))
    return path.resolve(), bound, digest


def validate_development_training_runs(
    package: dict[str, Any],
    base: Path,
    protocol: dict[str, Any],
) -> tuple[str, dict[str, str], list[dict[str, Any]]]:
    formal_path, formal_protocol, formal_sha = _load_bound_training_protocol(
        training_protocol_binding_for_seed(protocol, 17), EXPECTED_SEEDS
    )
    retry_path, retry_protocol, retry_sha = _load_bound_training_protocol(
        training_protocol_binding_for_seed(protocol, 29), (29,)
    )
    package_protocol_path = _resolve_reference(base, package.get("training_protocol_path"), code="TRAINING_PROTOCOL_PATH_INVALID")
    require(package_protocol_path == formal_path, "PROTOCOL", "PACKAGE_TRAINING_PROTOCOL_PATH_DRIFT", "package base training protocol path drift")
    require(package.get("training_protocol_sha256") == formal_sha, "PROTOCOL", "PACKAGE_TRAINING_PROTOCOL_SHA_DRIFT", "package base training protocol SHA drift")

    seed_runs = package.get("seed_runs")
    require(isinstance(seed_runs, list), "SCHEMA", "SEED_RUNS_NOT_LIST", "seed_runs must be a list")
    seeds = [int(run.get("seed", -1)) for run in seed_runs]
    require(tuple(seeds) == EXPECTED_SEEDS and len(set(seeds)) == len(EXPECTED_SEEDS), "SCHEMA", "SEED_SET_OR_ORDER_INVALID", "seed runs must be exactly 17,29,43 in frozen order", seeds=seeds)

    protocol_by_seed = {17: (formal_protocol, formal_sha), 29: (retry_protocol, retry_sha), 43: (formal_protocol, formal_sha)}
    sha_by_seed: dict[str, str] = {}
    validated: list[dict[str, Any]] = []
    for run in seed_runs:
        seed = int(run["seed"])
        train_protocol, train_sha = protocol_by_seed[seed]
        sha_by_seed[str(seed)] = train_sha
        initialization_sha256 = train_protocol["inputs"]["initialization_checkpoint"]["sha256"]
        result_path = _resolve_reference(base, run.get("train_result_path"), code="TRAIN_RESULT_PATH_INVALID")
        result = load_json(result_path)
        require(result.get("schema") == "blindassist_assistive_geometry_b1_a0_formal_train_result_v1", "CHECKPOINT", "TRAIN_RESULT_SCHEMA_DRIFT", "formal train result schema drift", seed=seed)
        require(result.get("mode") == "formal" and int(result.get("seed", -1)) == seed, "CHECKPOINT", "TRAIN_RESULT_SEED_OR_MODE_INVALID", "formal train result seed or mode is invalid", seed=seed)
        require(result.get("protocol_sha256") == train_sha, "PROTOCOL", "TRAIN_RESULT_PROTOCOL_SHA_MISMATCH", "formal train result protocol SHA mismatch", seed=seed)
        require(result.get("terminal") == "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE", "CHECKPOINT", "TRAIN_RESULT_TERMINAL_INVALID", "formal train result is not complete", seed=seed)
        require(int(result.get("completed_optimizer_steps", -1)) == 6000, "CHECKPOINT", "TRAIN_RESULT_STEP_COUNT_INVALID", "formal train result step count is incomplete", seed=seed)
        require(result.get("development_or_confirmation_content_opened") is False and result.get("teacher_import_or_execution") is False, "DATA_ROLE", "TRAIN_RESULT_FIREWALL_VIOLATION", "formal train result reports forbidden content or teacher access", seed=seed)
        receipts = result.get("checkpoints")
        require(isinstance(receipts, list) and [int(item.get("epoch", -1)) for item in receipts] == list(EXPECTED_CHECKPOINT_STEPS), "CHECKPOINT", "CHECKPOINT_RECEIPT_SET_INVALID", "retained checkpoint receipt set is incomplete", seed=seed)
        checked: list[dict[str, Any]] = []
        for receipt in receipts:
            epoch = int(receipt["epoch"])
            expected_steps = EXPECTED_CHECKPOINT_STEPS[epoch]
            require(int(receipt.get("optimizer_steps_completed", -1)) == expected_steps, "CHECKPOINT", "CHECKPOINT_RECEIPT_STEP_MISMATCH", "checkpoint receipt step mismatch", seed=seed, epoch=epoch)
            checkpoint_path = _resolve_reference(result_path.parent, receipt.get("path"), code="CHECKPOINT_PATH_INVALID")
            checked.append(_validate_checkpoint(checkpoint_path, receipt, seed=seed, epoch=epoch, steps=expected_steps, protocol_sha256=train_sha, initialization_sha256=initialization_sha256, final_epoch=20))
        require(result.get("final_model_state_sha256") == checked[-1]["model_state_sha256"], "CHECKPOINT", "FINAL_MODEL_STATE_SHA_MISMATCH", "final model-state SHA does not match epoch-20 checkpoint", seed=seed)
        require(run.get("final_checkpoint_sha256") == checked[-1]["sha256"], "CHECKPOINT", "PACKAGE_FINAL_CHECKPOINT_SHA_MISMATCH", "package final checkpoint SHA mismatch", seed=seed)
        require(run.get("final_model_state_sha256") == checked[-1]["model_state_sha256"], "CHECKPOINT", "PACKAGE_FINAL_MODEL_STATE_SHA_MISMATCH", "package final model-state SHA mismatch", seed=seed)
        observations_path = _resolve_reference(base, run.get("observations_path"), code="OBSERVATIONS_PATH_INVALID")
        require(observations_path.is_file(), "SCHEMA", "OBSERVATIONS_MISSING", "Development observations are missing", seed=seed)
        require(sha256_file(observations_path) == run.get("observations_sha256"), "SCHEMA", "OBSERVATIONS_SHA_MISMATCH", "Development observations SHA mismatch", seed=seed)
        validated.append({"seed": seed, "train_result_path": str(result_path), "observations_path": observations_path, "checkpoints": checked, "training_protocol_sha256": train_sha})
    return formal_sha, sha_by_seed, validated


def load_observations(path: Path, seed: int) -> list[dict[str, Any]]:
    require(path.is_file(), "SCHEMA", "OBSERVATIONS_MISSING", "Development observations are missing", seed=seed)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError("SCHEMA", "OBSERVATION_JSON_INVALID", str(error), seed=seed, line=line_number) from error
            require(row.get("schema") == "blindassist_assistive_geometry_b1_a0_development_frame_v1", "SCHEMA", "OBSERVATION_SCHEMA_DRIFT", "observation schema drift", seed=seed, line=line_number)
            require(int(row.get("seed", -1)) == seed, "SCHEMA", "OBSERVATION_SEED_DRIFT", "observation seed drift", seed=seed, line=line_number)
            require(row.get("data_role") == ROLE, "DATA_ROLE", "OBSERVATION_ROLE_DRIFT", "observation data role drift", seed=seed, line=line_number)
            rows.append(row)
    require(bool(rows), "SCHEMA", "OBSERVATIONS_EMPTY", "Development observations are empty", seed=seed)
    return rows


def validate_observations(rows: list[dict[str, Any]], seed: int) -> None:
    identities: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        parent = row.get("parent_id")
        session = row.get("session_id")
        sequence = row.get("sequence_index")
        require(isinstance(parent, str) and parent and isinstance(session, str) and session, "SCHEMA", "IDENTITY_INVALID", "parent/session identity invalid", seed=seed, row=index)
        require(isinstance(sequence, int) and sequence >= 0, "SCHEMA", "SEQUENCE_INVALID", "sequence index invalid", seed=seed, row=index)
        identity = (parent, session, sequence)
        require(identity not in identities, "SCHEMA", "IDENTITY_DUPLICATE", "duplicate frame identity", seed=seed, row=index)
        identities.add(identity)
        require(row.get("orientation") in {"portrait", "landscape"}, "SCHEMA", "ORIENTATION_INVALID", "orientation invalid", seed=seed, row=index)
        require(isinstance(row.get("truth_ground_valid"), bool) and isinstance(row.get("predicted_ground_valid"), bool), "SCHEMA", "GROUND_VALIDITY_INVALID", "ground validity must be boolean", seed=seed, row=index)
        require(isinstance(row.get("near_field"), bool) and isinstance(row.get("low_light_blur"), bool), "SCHEMA", "DIAGNOSTIC_FLAG_INVALID", "diagnostic flags must be boolean", seed=seed, row=index)
        bands = row.get("bands")
        require(isinstance(bands, list) and len(bands) == 3, "SCHEMA", "BAND_COUNT_INVALID", "three bands required", seed=seed, row=index)
        require(tuple(item.get("band") for item in bands) == BANDS, "SCHEMA", "BAND_ORDER_INVALID", "band order drift", seed=seed, row=index)
        for band in bands:
            truth_valid = band.get("truth_clearance_valid")
            prediction_valid = band.get("predicted_clearance_valid")
            require(isinstance(truth_valid, bool) and isinstance(prediction_valid, bool), "SCHEMA", "CLEARANCE_VALIDITY_INVALID", "clearance validity must be boolean", seed=seed, row=index, band=band.get("band"))
            require(_finite(band.get("truth_clearance_m")) if truth_valid else band.get("truth_clearance_m") is None, "SCHEMA", "TRUTH_CLEARANCE_INVALID", "truth clearance/value mismatch", seed=seed, row=index, band=band.get("band"))
            require(_finite(band.get("predicted_clearance_m")) if prediction_valid else band.get("predicted_clearance_m") is None, "SCHEMA", "PREDICTED_CLEARANCE_INVALID", "predicted clearance/value mismatch", seed=seed, row=index, band=band.get("band"))
            cells = band.get("cells")
            require(isinstance(cells, list) and len(cells) == 3, "SCHEMA", "HORIZON_COUNT_INVALID", "three horizons required", seed=seed, row=index, band=band.get("band"))
            require(tuple(float(cell.get("horizon_m", -1)) for cell in cells) == HORIZONS, "SCHEMA", "HORIZON_ORDER_INVALID", "horizon order drift", seed=seed, row=index, band=band.get("band"))
            for cell in cells:
                require(cell.get("truth_state") in STATES and cell.get("predicted_state") in STATES, "SCHEMA", "TRISTATE_INVALID", "truth/prediction state invalid", seed=seed, row=index, band=band.get("band"))


def confusion(pairs: Iterable[tuple[str, str]], *, require_denominators: bool = False) -> dict[str, Any]:
    values = list(pairs)
    truth_known = [item for item in values if item[0] != "UNKNOWN"]
    paired = [item for item in truth_known if item[1] != "UNKNOWN"]
    occupied = [item for item in paired if item[0] == "OCCUPIED_OBSERVED"]
    clear = [item for item in paired if item[0] == "CLEAR_OBSERVED"]
    if require_denominators:
        require(bool(truth_known), "DENOMINATOR", "TRUTH_KNOWN_ZERO", "truth-known denominator is zero")
        require(bool(paired), "DENOMINATOR", "PAIRED_KNOWN_ZERO", "paired-known denominator is zero")
        require(bool(occupied), "DENOMINATOR", "TRUTH_OCCUPIED_ZERO", "truth-occupied denominator is zero")
        require(bool(clear), "DENOMINATOR", "TRUTH_CLEAR_ZERO", "truth-clear denominator is zero")
    false_clear = sum(item == ("OCCUPIED_OBSERVED", "CLEAR_OBSERVED") for item in paired)
    false_block = sum(item == ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED") for item in paired)
    return {
        "truth_known": len(truth_known),
        "paired_known": len(paired),
        "truth_occupied": len(occupied),
        "truth_clear": len(clear),
        "unknown_truth_excluded": len(values) - len(truth_known),
        "known_coverage": len(paired) / len(truth_known) if truth_known else None,
        "valid_to_unknown_rate": (len(truth_known) - len(paired)) / len(truth_known) if truth_known else None,
        "false_clear_all_known": false_clear / len(paired) if paired else None,
        "false_clear_given_occupied": false_clear / len(occupied) if occupied else None,
        "false_block_given_clear": false_block / len(clear) if clear else None,
    }


def _mean(values: list[float], code: str) -> dict[str, Any]:
    require(bool(values), "DENOMINATOR", code, "metric denominator is zero")
    return {"support": len(values), "value": float(statistics.fmean(values))}


def compute_seed_metrics(rows: list[dict[str, Any]], seed: int, gates: dict[str, Any]) -> dict[str, Any]:
    validate_observations(rows, seed)
    all_pairs: list[tuple[str, str]] = []
    by_parent_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_grid_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_orientation_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    clearance_errors: list[float] = []
    truth_clearance_count = 0
    paired_clearance_count = 0
    truth_ground_count = 0
    recovered_ground_count = 0
    clearance_series: dict[tuple[str, str, str], list[tuple[int, bool, float | None, bool, float | None]]] = defaultdict(list)
    state_series: dict[tuple[str, str, str, float], list[tuple[int, str, str]]] = defaultdict(list)

    for row in rows:
        parent = row["parent_id"]
        if row["truth_ground_valid"]:
            truth_ground_count += 1
            recovered_ground_count += int(row["predicted_ground_valid"])
        for band in row["bands"]:
            if band["truth_clearance_valid"]:
                truth_clearance_count += 1
                if band["predicted_clearance_valid"]:
                    paired_clearance_count += 1
                    clearance_errors.append(abs(float(band["predicted_clearance_m"]) - float(band["truth_clearance_m"])))
            clearance_series[(parent, row["session_id"], band["band"])].append(
                (
                    row["sequence_index"],
                    band["truth_clearance_valid"],
                    band["truth_clearance_m"],
                    band["predicted_clearance_valid"],
                    band["predicted_clearance_m"],
                )
            )
            for cell in band["cells"]:
                pair = (cell["truth_state"], cell["predicted_state"])
                all_pairs.append(pair)
                by_parent_pairs[parent].append(pair)
                by_grid_pairs[f"{band['band']}@{cell['horizon_m']:.1f}m"].append(pair)
                by_orientation_pairs[row["orientation"]].append(pair)
                state_series[(parent, row["session_id"], band["band"], float(cell["horizon_m"]))].append(
                    (row["sequence_index"], *pair)
                )

    pooled = confusion(all_pairs, require_denominators=True)
    require(truth_ground_count > 0, "DENOMINATOR", "GROUND_TRUTH_ZERO", "truth ground-valid denominator is zero")
    require(truth_clearance_count > 0, "DENOMINATOR", "CLEARANCE_TRUTH_ZERO", "truth clearance-valid denominator is zero")
    clearance_coverage = paired_clearance_count / truth_clearance_count
    clearance = _mean(clearance_errors, "CLEARANCE_PAIRED_ZERO")
    temporal_errors: list[float] = []
    for samples in clearance_series.values():
        samples.sort()
        for previous, current in zip(samples, samples[1:]):
            if current[0] != previous[0] + 1 or not all((previous[1], previous[3], current[1], current[3])):
                continue
            temporal_errors.append(abs((float(current[4]) - float(previous[4])) - (float(current[2]) - float(previous[2]))))
    temporal = _mean(temporal_errors, "TEMPORAL_CLEARANCE_ZERO")
    transition_total = 0
    transition_agree = 0
    for samples in state_series.values():
        samples.sort()
        for previous, current in zip(samples, samples[1:]):
            if current[0] != previous[0] + 1 or "UNKNOWN" in previous[1:] or "UNKNOWN" in current[1:]:
                continue
            transition_total += 1
            transition_agree += int((previous[1], current[1]) == (previous[2], current[2]))
    require(transition_total > 0, "DENOMINATOR", "GEOMETRY_TRANSITION_ZERO", "geometry transition denominator is zero")
    ground_recovery = recovered_ground_count / truth_ground_count
    transition_agreement = transition_agree / transition_total
    by_parent = {parent: confusion(values) for parent, values in sorted(by_parent_pairs.items())}
    defined_parent_false_clear = {
        parent: float(value["false_clear_all_known"])
        for parent, value in by_parent.items()
        if value["false_clear_all_known"] is not None
    }
    require(bool(defined_parent_false_clear), "DENOMINATOR", "PARENT_FALSE_CLEAR_ZERO", "no defined parent false-clear metric")
    worst_parent = max(defined_parent_false_clear.items(), key=lambda item: (item[1], item[0]))
    checks = {
        "known_coverage": pooled["known_coverage"] >= float(gates["known_coverage_min"]),
        "valid_to_unknown_rate": pooled["valid_to_unknown_rate"] <= float(gates["valid_to_unknown_rate_max"]),
        "ground_recovery": ground_recovery >= float(gates["ground_recovery_min"]),
        "clearance_known_coverage": clearance_coverage >= float(gates["known_coverage_min"]),
        "clearance_mae_m": clearance["value"] <= float(gates["clearance_mae_m_max"]),
        "false_clear_all_known": pooled["false_clear_all_known"] <= float(gates["false_clear_all_known_max"]),
        "false_block_given_clear": pooled["false_block_given_clear"] <= float(gates["false_block_given_clear_max"]),
        "temporal_clearance_delta_mae_m": temporal["value"] <= float(gates["temporal_clearance_delta_mae_m_max"]),
        "geometry_transition_agreement": transition_agreement >= float(gates["geometry_transition_agreement_min"]),
        "worst_parent_false_clear_all_known": worst_parent[1] <= float(gates["worst_parent_false_clear_all_known_max"]),
    }
    return {
        "seed": seed,
        "pooled": pooled,
        "ground_recovery": {"truth_support": truth_ground_count, "recovered": recovered_ground_count, "value": ground_recovery},
        "clearance_known_coverage": {"truth_support": truth_clearance_count, "paired": paired_clearance_count, "value": clearance_coverage},
        "clearance_mae_m": clearance,
        "temporal_clearance_delta_mae_m": temporal,
        "geometry_transition_agreement": {"support": transition_total, "agree": transition_agree, "value": transition_agreement},
        "worst_parent_false_clear_all_known": {"parent_id": worst_parent[0], "value": worst_parent[1]},
        "by_parent": by_parent,
        "by_grid": {key: confusion(value) for key, value in sorted(by_grid_pairs.items())},
        "by_orientation": {key: confusion(value) for key, value in sorted(by_orientation_pairs.items())},
        "gates": checks,
        "confidence_ece": {"status": "NOT_APPLICABLE_A0_DEPTH_ONLY", "gate_applied": False},
    }


def _aggregate(values: list[float]) -> dict[str, Any]:
    require(len(values) == 3 and all(math.isfinite(value) for value in values), "DENOMINATOR", "AGGREGATE_INPUT_INVALID", "three finite seed values required")
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, values, strict=True)},
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    require(tuple(row["seed"] for row in metrics) == EXPECTED_SEEDS, "PROTOCOL", "AGGREGATE_SEED_ORDER_INVALID", "seed order must be 17/29/43")
    getters = {
        "known_coverage": lambda row: row["pooled"]["known_coverage"],
        "valid_to_unknown_rate": lambda row: row["pooled"]["valid_to_unknown_rate"],
        "ground_recovery": lambda row: row["ground_recovery"]["value"],
        "clearance_known_coverage": lambda row: row["clearance_known_coverage"]["value"],
        "clearance_mae_m": lambda row: row["clearance_mae_m"]["value"],
        "false_clear_all_known": lambda row: row["pooled"]["false_clear_all_known"],
        "false_clear_given_occupied": lambda row: row["pooled"]["false_clear_given_occupied"],
        "false_block_given_clear": lambda row: row["pooled"]["false_block_given_clear"],
        "temporal_clearance_delta_mae_m": lambda row: row["temporal_clearance_delta_mae_m"]["value"],
        "geometry_transition_agreement": lambda row: row["geometry_transition_agreement"]["value"],
        "worst_parent_false_clear_all_known": lambda row: row["worst_parent_false_clear_all_known"]["value"],
    }
    gate_summary = {
        gate: {
            "values_by_seed": {str(row["seed"]): bool(row["gates"][gate]) for row in metrics},
            "pass_count": sum(bool(row["gates"][gate]) for row in metrics),
        }
        for gate in metrics[0]["gates"]
    }
    frontdoors = {"known_coverage", "valid_to_unknown_rate", "ground_recovery", "clearance_known_coverage"}
    frontdoor_pass = all(item["pass_count"] == 3 for gate, item in gate_summary.items() if gate in frontdoors)
    task_pass = all(item["pass_count"] >= 2 for gate, item in gate_summary.items() if gate not in frontdoors)
    return {
        "seed_order": list(EXPECTED_SEEDS),
        "selected_seed": None,
        "best_seed_selection_forbidden": True,
        "statistics": {name: _aggregate([float(getter(row)) for row in metrics]) for name, getter in getters.items()},
        "gate_summary": gate_summary,
        "frontdoor_rule": "all three seeds pass coverage, valid-to-UNKNOWN, ground recovery and clearance coverage",
        "frontdoor_pass": frontdoor_pass,
        "task_rule": "each remaining task gate passes on at least two of three seeds; no seed selected",
        "task_pass": task_pass,
        "overall_pass": frontdoor_pass and task_pass,
    }


def evaluate(package_path: Path, protocol_path: Path) -> dict[str, Any]:
    package = load_json(package_path)
    require(package.get("schema") == "blindassist_assistive_geometry_b1_a0_development_evaluation_package_v1", "SCHEMA", "PACKAGE_SCHEMA_DRIFT", "Development evaluation package schema drift")
    forbidden = _contains_forbidden_seed_selection(package)
    require(forbidden is None, "BEST_SEED", "BEST_SEED_FIELD_PRESENT", "package attempts to select a seed", key=forbidden)
    require(package.get("data_role") == ROLE, "DATA_ROLE", "PACKAGE_ROLE_DRIFT", "package role drift")
    require(package.get("development_content_opened") is True and package.get("development_calibration_content_opened") is False and package.get("confirmation_content_opened") is False, "DATA_ROLE", "PACKAGE_FIREWALL_DRIFT", "package data firewall drift")
    protocol, protocol_sha = validate_protocol(protocol_path)
    accepted_package_protocol_shas = {protocol_sha}
    correction = protocol.get("integrity_correction", {})
    if correction.get("pre_metric_observation_package_reuse") is True:
        accepted_package_protocol_shas.add(correction.get("accepted_observation_package_protocol_sha256"))
    require(package.get("evaluation_protocol_sha256") in accepted_package_protocol_shas, "PROTOCOL", "PACKAGE_PROTOCOL_SHA_DRIFT", "package protocol SHA drift")
    training_protocol_sha, training_protocol_sha_by_seed, checked = validate_development_training_runs(package, package_path.parent, protocol)
    seed_metrics = [compute_seed_metrics(load_observations(run["observations_path"], run["seed"]), run["seed"], protocol["metric_gates"]) for run in checked]
    summary = aggregate(seed_metrics)
    passed = summary["overall_pass"]
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_development_evaluation_result_v1",
        "status": "PASS" if passed else "FAIL",
        "terminal": TERMINALS["PASS"] if passed else TERMINALS["TASK"],
        "training_protocol_sha256": training_protocol_sha,
        "training_protocol_sha256_by_seed": training_protocol_sha_by_seed,
        "evaluation_protocol_sha256": protocol_sha,
        "observation_package_protocol_sha256": package.get("evaluation_protocol_sha256"),
        "data_role": ROLE,
        "seed_metrics": seed_metrics,
        "aggregate": summary,
        "checkpoint_integrity": {"seeds": list(EXPECTED_SEEDS), "checkpoint_count": 12, "runs": [{"seed": row["seed"], "training_protocol_sha256": row["training_protocol_sha256"], "train_result_path": row["train_result_path"], "observations_path": str(row["observations_path"]), "checkpoints": row["checkpoints"]} for row in checked]},
        "development_content_opened": True,
        "development_calibration_content_opened": False,
        "confirmation_content_opened": False,
        "selected_seed": None,
        "claim_ceiling": "Development-only A0 task evidence; no Confirmation, deployment, product or safety authority.",
    }


def render_report(result: dict[str, Any]) -> str:
    aggregate_result = result["aggregate"]
    lines = [
        "# Assistive Geometry B1 A0 Development evaluation",
        "",
        f"Terminal: `{result['terminal']}`",
        "",
        "All seeds remain visible; selected_seed is null.",
        "",
        "| Metric | Mean | Seed values |",
        "| --- | ---: | --- |",
    ]
    for name, value in aggregate_result["statistics"].items():
        values = ", ".join(f"{seed}={number:.6f}" for seed, number in value["values_by_seed"].items())
        lines.append(f"| {name} | {value['mean']:.6f} | {values} |")
    lines.extend(["", f"Frontdoor pass: `{aggregate_result['frontdoor_pass']}`", f"Task gate pass: `{aggregate_result['task_pass']}`", "", "Development Selection was opened; Calibration and Confirmation remain sealed.", ""])
    return "\n".join(lines)


def run(package_path: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    require(not output_root.exists(), "SCHEMA", "OUTPUT_EXISTS", "evaluation output root already exists")
    output_root.mkdir(parents=True)
    try:
        result = evaluate(package_path, protocol_path)
        result["completed_at"] = utc_now()
        atomic_write_json(output_root / "evaluation_result.json", result)
        atomic_write_text(output_root / "report.md", render_report(result))
        if result["status"] == "FAIL":
            atomic_write_json(output_root / "failure.json", {"schema": "blindassist_assistive_geometry_b1_a0_development_failure_v1", "terminal": result["terminal"], "failed_at": result["completed_at"], "development_content_opened": True, "confirmation_content_opened": False})
        return result
    except EvaluationError as error:
        terminal = TERMINALS.get(error.terminal, TERMINALS["INTERNAL"])
        failure = {"schema": "blindassist_assistive_geometry_b1_a0_development_failure_v1", "status": "FAIL", "terminal": terminal, "code": error.code, "message": str(error), "context": error.context, "failed_at": utc_now(), "development_content_opened": True, "confirmation_content_opened": False}
        atomic_write_json(output_root / "failure.json", failure)
        atomic_write_text(output_root / "failure.log", f"terminal={terminal}\ncode={error.code}\nmessage={error}\n")
        return failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.package.resolve(), args.protocol.resolve(), args.output_root.resolve())
    print(json.dumps({key: value for key, value in result.items() if key not in {"seed_metrics", "checkpoint_integrity"}}, indent=2))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
