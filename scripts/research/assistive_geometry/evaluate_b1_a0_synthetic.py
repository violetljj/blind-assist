#!/usr/bin/env python3
"""Validate A0 checkpoints and aggregate synthetic assistive-geometry metrics.

This module is deliberately outcome-agnostic.  It reads only paths explicitly
listed by an evaluation package and the dry-run invokes it exclusively with
``SYNTHETIC_EVALUATOR_FIXTURE`` data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EXPECTED_SEEDS = (17, 29, 43)
EXPECTED_CHECKPOINT_STEPS = {5: 1499, 10: 2999, 15: 4499, 20: 6000}
BANDS = ("left", "center", "right")
HORIZONS_M = (1.0, 1.5, 2.0)
STATES = ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
SYNTHETIC_ROLE = "SYNTHETIC_EVALUATOR_FIXTURE"
REPO_ROOT = Path(__file__).resolve().parents[3]

TERMINALS = {
    "PASS": "B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PASS",
    "CHECKPOINT": "B1_A0_EVAL_NOT_EVALUABLE_CHECKPOINT_INCOMPLETE",
    "PROTOCOL": "B1_A0_EVAL_NOT_EVALUABLE_PROTOCOL_DRIFT",
    "SEEDS": "B1_A0_EVAL_NOT_EVALUABLE_SEED_SET_INVALID",
    "SCHEMA": "B1_A0_EVAL_NOT_EVALUABLE_SCHEMA_INVALID",
    "DENOMINATOR": "B1_A0_EVAL_NOT_EVALUABLE_UNDEFINED_DENOMINATOR",
    "COVERAGE": "B1_A0_EVAL_FAIL_COVERAGE",
    "TASK": "B1_A0_EVAL_FAIL_TASK_GATES",
    "BEST_SEED": "B1_A0_EVAL_NOT_EVALUABLE_BEST_SEED_SELECTION_FORBIDDEN",
    "DATA_ROLE": "B1_A0_EVAL_NOT_EVALUABLE_DATA_ROLE_INVALID",
    "INTERNAL": "B1_A0_EVAL_INTERNAL_FAILURE",
}


class EvaluationError(RuntimeError):
    def __init__(self, terminal_key: str, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.terminal = TERMINALS[terminal_key]
        self.code = code
        self.context = context


def require(condition: bool, terminal_key: str, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise EvaluationError(terminal_key, code, message, **context)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "SCHEMA", "JSON_ROOT_NOT_OBJECT", "JSON root must be an object", path=str(path))
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), "SCHEMA", "OUTPUT_EXISTS", "output already exists", path=str(path))
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def state_dict_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    require(bool(state), "CHECKPOINT", "EMPTY_MODEL_STATE", "checkpoint model state is empty")
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), "CHECKPOINT", "NON_TENSOR_MODEL_STATE", "model state contains a non-tensor", name=name)
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def _resolve_reference(base: Path, value: Any, *, code: str) -> Path:
    require(isinstance(value, str) and value, "SCHEMA", code, "path reference must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _contains_forbidden_seed_selection(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"best_seed", "selected_seed", "winner_seed"}:
                return str(key)
            found = _contains_forbidden_seed_selection(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_seed_selection(child)
            if found:
                return found
    return None


def validate_evaluation_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    require(
        protocol.get("schema") == "blindassist_assistive_geometry_b1_a0_evaluation_dry_run_protocol_v1",
        "PROTOCOL",
        "EVALUATION_PROTOCOL_SCHEMA_DRIFT",
        "evaluation protocol schema drift",
    )
    require(tuple(protocol["aggregation"]["required_seeds"]) == EXPECTED_SEEDS, "PROTOCOL", "SEED_PROTOCOL_DRIFT", "evaluation seed set drift")
    actual_steps = {int(key): int(value) for key, value in protocol["checkpoint_integrity"]["expected_optimizer_steps_by_epoch"].items()}
    require(actual_steps == EXPECTED_CHECKPOINT_STEPS, "PROTOCOL", "CHECKPOINT_STEP_PROTOCOL_DRIFT", "checkpoint step map drift")
    require(protocol["aggregation"].get("selected_seed") is None, "BEST_SEED", "SELECTED_SEED_NOT_NULL", "selected_seed must stay null")
    require(protocol["firewalls"].get("dry_run_data_role") == SYNTHETIC_ROLE, "DATA_ROLE", "DRY_RUN_ROLE_DRIFT", "dry-run role drift")
    for binding in protocol.get("bindings", []):
        binding_path = (REPO_ROOT / binding["path"]).resolve()
        require(binding_path.is_file(), "PROTOCOL", "EVALUATOR_BINDING_MISSING", "evaluation implementation binding is missing", path=binding["path"])
        require(sha256_file(binding_path) == binding["sha256"], "PROTOCOL", "EVALUATOR_BINDING_SHA_MISMATCH", "evaluation implementation binding SHA mismatch", path=binding["path"])
    return protocol, sha256_file(path)


def _validate_checkpoint(
    path: Path,
    receipt: dict[str, Any],
    *,
    seed: int,
    epoch: int,
    steps: int,
    protocol_sha256: str,
    initialization_sha256: str,
    final_epoch: int,
) -> dict[str, Any]:
    require(path.is_file(), "CHECKPOINT", "CHECKPOINT_MISSING", "checkpoint file is missing", seed=seed, epoch=epoch, path=str(path))
    require(path.stat().st_size == int(receipt.get("bytes", -1)), "CHECKPOINT", "CHECKPOINT_BYTE_COUNT_MISMATCH", "checkpoint byte count mismatch", seed=seed, epoch=epoch)
    digest = sha256_file(path)
    require(digest == receipt.get("sha256"), "CHECKPOINT", "CHECKPOINT_SHA256_MISMATCH", "checkpoint SHA-256 mismatch", seed=seed, epoch=epoch)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exact torch error is version-specific
        raise EvaluationError("CHECKPOINT", "CHECKPOINT_DESERIALIZATION_FAILED", str(error), seed=seed, epoch=epoch) from error
    require(isinstance(payload, dict), "CHECKPOINT", "CHECKPOINT_ROOT_NOT_OBJECT", "checkpoint root must be a mapping", seed=seed, epoch=epoch)
    required = {"schema", "protocol_sha256", "initialization_checkpoint_sha256", "seed", "next_epoch", "next_optimizer_step", "sampler", "model", "optimizer", "scheduler", "scaler", "rng", "epoch_history", "model_state_sha256"}
    require(required.issubset(payload), "CHECKPOINT", "CHECKPOINT_FIELDS_MISSING", "checkpoint fields are missing", seed=seed, epoch=epoch, missing=sorted(required - set(payload)))
    require(payload["schema"] == "blindassist_assistive_geometry_b1_a0_checkpoint_v1", "CHECKPOINT", "CHECKPOINT_SCHEMA_DRIFT", "checkpoint schema drift", seed=seed, epoch=epoch)
    require(payload["protocol_sha256"] == protocol_sha256, "PROTOCOL", "CHECKPOINT_PROTOCOL_SHA_MISMATCH", "checkpoint protocol SHA mismatch", seed=seed, epoch=epoch)
    require(payload["initialization_checkpoint_sha256"] == initialization_sha256, "CHECKPOINT", "CHECKPOINT_INITIALIZATION_SHA_MISMATCH", "checkpoint initialization SHA mismatch", seed=seed, epoch=epoch)
    require(int(payload["seed"]) == seed, "SEEDS", "CHECKPOINT_SEED_MISMATCH", "checkpoint seed mismatch", seed=seed, epoch=epoch)
    require(int(payload["next_epoch"]) == epoch, "CHECKPOINT", "CHECKPOINT_EPOCH_MISMATCH", "checkpoint epoch mismatch", seed=seed, epoch=epoch)
    require(int(payload["next_optimizer_step"]) == steps + 1, "CHECKPOINT", "CHECKPOINT_NEXT_STEP_MISMATCH", "checkpoint next optimizer step mismatch", seed=seed, epoch=epoch)
    sampler = payload["sampler"]
    require(isinstance(sampler, dict) and sampler.get("formal_epoch_complete") is True and sampler.get("smoke_only") is False, "CHECKPOINT", "CHECKPOINT_SAMPLER_STATE_INVALID", "checkpoint sampler state is not formal-complete", seed=seed, epoch=epoch)
    if epoch == final_epoch:
        require(sampler.get("carry") == {"portrait": [], "landscape": []}, "CHECKPOINT", "FINAL_CHECKPOINT_CARRY_NOT_EMPTY", "final checkpoint carry is not empty", seed=seed)
    scheduler = payload["scheduler"]
    require(isinstance(scheduler, dict), "CHECKPOINT", "CHECKPOINT_SCHEDULER_INVALID", "checkpoint scheduler state is invalid", seed=seed, epoch=epoch)
    require(int(scheduler.get("completed_steps", -1)) == steps, "CHECKPOINT", "CHECKPOINT_COMPLETED_STEPS_MISMATCH", "checkpoint completed-step count mismatch", seed=seed, epoch=epoch)
    require(int(scheduler.get("total_steps", -1)) == EXPECTED_CHECKPOINT_STEPS[final_epoch], "CHECKPOINT", "CHECKPOINT_TOTAL_STEPS_MISMATCH", "checkpoint scheduler total steps mismatch", seed=seed, epoch=epoch)
    history = payload["epoch_history"]
    require(isinstance(history, list) and len(history) == epoch, "CHECKPOINT", "CHECKPOINT_EPOCH_HISTORY_INCOMPLETE", "checkpoint epoch history is incomplete", seed=seed, epoch=epoch)
    require([int(row.get("epoch", -1)) for row in history] == list(range(1, epoch + 1)), "CHECKPOINT", "CHECKPOINT_EPOCH_HISTORY_ORDER_INVALID", "checkpoint epoch history order is invalid", seed=seed, epoch=epoch)
    require(int(history[-1].get("optimizer_steps_completed", -1)) == steps, "CHECKPOINT", "CHECKPOINT_HISTORY_STEP_MISMATCH", "checkpoint history step mismatch", seed=seed, epoch=epoch)
    require(isinstance(payload["optimizer"], dict), "CHECKPOINT", "CHECKPOINT_OPTIMIZER_INVALID", "checkpoint optimizer state is invalid", seed=seed, epoch=epoch)
    require(isinstance(payload["scaler"], dict), "CHECKPOINT", "CHECKPOINT_SCALER_INVALID", "checkpoint scaler state is invalid", seed=seed, epoch=epoch)
    rng = payload["rng"]
    require(isinstance(rng, dict) and {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng), "CHECKPOINT", "CHECKPOINT_RNG_INCOMPLETE", "checkpoint RNG state is incomplete", seed=seed, epoch=epoch)
    model_digest = state_dict_digest(payload["model"])
    require(model_digest == payload["model_state_sha256"], "CHECKPOINT", "CHECKPOINT_MODEL_STATE_SHA_MISMATCH", "checkpoint model-state SHA mismatch", seed=seed, epoch=epoch)
    return {"epoch": epoch, "optimizer_steps_completed": steps, "bytes": path.stat().st_size, "sha256": digest, "model_state_sha256": model_digest}


def validate_training_runs(package: dict[str, Any], base: Path) -> tuple[str, list[dict[str, Any]]]:
    training_protocol_path = _resolve_reference(base, package.get("training_protocol_path"), code="TRAINING_PROTOCOL_PATH_INVALID")
    training_protocol = load_json(training_protocol_path)
    require(training_protocol.get("schema") == "blindassist_assistive_geometry_b1_a0_formal_train_execution_protocol_v1", "PROTOCOL", "TRAINING_PROTOCOL_SCHEMA_DRIFT", "training protocol schema drift")
    training_protocol_sha = sha256_file(training_protocol_path)
    require(training_protocol_sha == package.get("training_protocol_sha256"), "PROTOCOL", "TRAINING_PROTOCOL_SHA_MISMATCH", "training protocol SHA mismatch")
    execution = training_protocol["execution"]
    initialization_sha256 = training_protocol["inputs"]["initialization_checkpoint"]["sha256"]
    require(tuple(execution["seeds"]) == EXPECTED_SEEDS, "PROTOCOL", "TRAINING_PROTOCOL_SEED_DRIFT", "training protocol seed drift")
    require(int(execution["epochs_per_seed"]) == 20 and int(execution["optimizer_steps_per_seed"]) == 6000, "PROTOCOL", "TRAINING_PROTOCOL_DURATION_DRIFT", "training protocol epoch/step drift")
    require(tuple(execution["retained_checkpoint_epochs"]) == tuple(EXPECTED_CHECKPOINT_STEPS), "PROTOCOL", "TRAINING_PROTOCOL_CHECKPOINT_CADENCE_DRIFT", "training protocol checkpoint cadence drift")

    seed_runs = package.get("seed_runs")
    require(isinstance(seed_runs, list), "SEEDS", "SEED_RUNS_NOT_LIST", "seed_runs must be a list")
    seeds = [int(run.get("seed", -1)) for run in seed_runs]
    require(tuple(seeds) == EXPECTED_SEEDS and len(set(seeds)) == len(EXPECTED_SEEDS), "SEEDS", "SEED_SET_OR_ORDER_INVALID", "seed runs must be exactly 17,29,43 in frozen order", seeds=seeds)
    validated: list[dict[str, Any]] = []
    for run in seed_runs:
        seed = int(run["seed"])
        result_path = _resolve_reference(base, run.get("train_result_path"), code="TRAIN_RESULT_PATH_INVALID")
        result = load_json(result_path)
        require(result.get("schema") == "blindassist_assistive_geometry_b1_a0_formal_train_result_v1", "CHECKPOINT", "TRAIN_RESULT_SCHEMA_DRIFT", "formal train result schema drift", seed=seed)
        require(result.get("mode") == "formal" and int(result.get("seed", -1)) == seed, "SEEDS", "TRAIN_RESULT_SEED_OR_MODE_INVALID", "formal train result seed or mode is invalid", seed=seed)
        require(result.get("protocol_sha256") == training_protocol_sha, "PROTOCOL", "TRAIN_RESULT_PROTOCOL_SHA_MISMATCH", "formal train result protocol SHA mismatch", seed=seed)
        require(result.get("terminal") == "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE", "CHECKPOINT", "TRAIN_RESULT_TERMINAL_INVALID", "formal train result is not complete", seed=seed)
        require(int(result.get("completed_optimizer_steps", -1)) == 6000, "CHECKPOINT", "TRAIN_RESULT_STEP_COUNT_INVALID", "formal train result step count is incomplete", seed=seed)
        require(result.get("development_or_confirmation_content_opened") is False and result.get("teacher_import_or_execution") is False, "DATA_ROLE", "TRAIN_RESULT_FIREWALL_VIOLATION", "formal train result reports forbidden content or teacher access", seed=seed)
        receipts = result.get("checkpoints")
        require(isinstance(receipts, list) and [int(item.get("epoch", -1)) for item in receipts] == list(EXPECTED_CHECKPOINT_STEPS), "CHECKPOINT", "CHECKPOINT_RECEIPT_SET_INVALID", "retained checkpoint receipt set is incomplete", seed=seed)
        checked: list[dict[str, Any]] = []
        for receipt in receipts:
            epoch = int(receipt["epoch"])
            require(int(receipt.get("optimizer_steps_completed", -1)) == EXPECTED_CHECKPOINT_STEPS[epoch], "CHECKPOINT", "CHECKPOINT_RECEIPT_STEP_MISMATCH", "checkpoint receipt step mismatch", seed=seed, epoch=epoch)
            checkpoint_path = _resolve_reference(result_path.parent, receipt.get("path"), code="CHECKPOINT_PATH_INVALID")
            checked.append(_validate_checkpoint(checkpoint_path, receipt, seed=seed, epoch=epoch, steps=EXPECTED_CHECKPOINT_STEPS[epoch], protocol_sha256=training_protocol_sha, initialization_sha256=initialization_sha256, final_epoch=20))
        require(result.get("final_model_state_sha256") == checked[-1]["model_state_sha256"], "CHECKPOINT", "FINAL_MODEL_STATE_SHA_MISMATCH", "final model-state SHA does not match epoch-20 checkpoint", seed=seed)
        observations_path = _resolve_reference(base, run.get("observations_path"), code="OBSERVATIONS_PATH_INVALID")
        validated.append({"seed": seed, "train_result_path": str(result_path), "observations_path": observations_path, "checkpoints": checked})
    return training_protocol_sha, validated


def load_observations(path: Path, *, seed: int, expected_role: str) -> list[dict[str, Any]]:
    require(path.is_file(), "SCHEMA", "OBSERVATIONS_MISSING", "observation JSONL is missing", seed=seed, path=str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError("SCHEMA", "OBSERVATION_JSON_INVALID", str(error), seed=seed, line=line_number) from error
            require(isinstance(row, dict), "SCHEMA", "OBSERVATION_NOT_OBJECT", "observation row must be an object", seed=seed, line=line_number)
            require(row.get("schema") == "blindassist_assistive_geometry_b1_a0_evaluation_frame_v1", "SCHEMA", "OBSERVATION_SCHEMA_DRIFT", "observation schema drift", seed=seed, line=line_number)
            require(int(row.get("seed", -1)) == seed, "SEEDS", "OBSERVATION_SEED_MISMATCH", "observation seed mismatch", seed=seed, line=line_number)
            require(row.get("data_role") == expected_role, "DATA_ROLE", "OBSERVATION_DATA_ROLE_MISMATCH", "observation data role mismatch", seed=seed, line=line_number)
            rows.append(row)
    require(bool(rows), "SCHEMA", "OBSERVATIONS_EMPTY", "observation JSONL is empty", seed=seed)
    return rows


def validate_observations(rows: list[dict[str, Any]], *, seed: int) -> None:
    identities: set[tuple[str, str, int]] = set()
    for row_index, row in enumerate(rows):
        parent = row.get("parent_id")
        session = row.get("session_id")
        sequence_index = row.get("sequence_index")
        require(isinstance(parent, str) and parent and isinstance(session, str) and session, "SCHEMA", "FRAME_IDENTITY_INVALID", "parent/session identity is invalid", seed=seed, row=row_index)
        require(isinstance(sequence_index, int) and sequence_index >= 0, "SCHEMA", "SEQUENCE_INDEX_INVALID", "sequence index is invalid", seed=seed, row=row_index)
        identity = (parent, session, sequence_index)
        require(identity not in identities, "SCHEMA", "DUPLICATE_FRAME_IDENTITY", "duplicate frame identity", seed=seed, row=row_index)
        identities.add(identity)
        require(row.get("orientation") in {"portrait", "landscape"}, "SCHEMA", "ORIENTATION_INVALID", "orientation is invalid", seed=seed, row=row_index)
        bands = row.get("bands")
        require(isinstance(bands, list) and len(bands) == 3, "SCHEMA", "BAND_COUNT_INVALID", "frame must contain exactly three bands", seed=seed, row=row_index)
        require(tuple(item.get("band") for item in bands) == BANDS, "SCHEMA", "BAND_SET_OR_ORDER_INVALID", "bands must be left,center,right in frozen order", seed=seed, row=row_index)
        for band in bands:
            valid = band.get("clearance_valid")
            require(isinstance(valid, bool), "SCHEMA", "CLEARANCE_VALIDITY_INVALID", "clearance_valid must be boolean", seed=seed, row=row_index, band=band.get("band"))
            truth_clearance = band.get("truth_clearance_m")
            predicted_clearance = band.get("predicted_clearance_m")
            if valid:
                require(_finite_number(truth_clearance) and _finite_number(predicted_clearance), "SCHEMA", "CLEARANCE_NONFINITE", "valid clearance must be finite", seed=seed, row=row_index, band=band["band"])
            else:
                require(truth_clearance is None and predicted_clearance is None, "SCHEMA", "INVALID_CLEARANCE_NOT_NULL", "invalid clearance must use null values", seed=seed, row=row_index, band=band["band"])
            cells = band.get("cells")
            require(isinstance(cells, list) and len(cells) == 3, "SCHEMA", "HORIZON_COUNT_INVALID", "band must contain exactly three horizons", seed=seed, row=row_index, band=band["band"])
            require(tuple(float(cell.get("horizon_m", -1)) for cell in cells) == HORIZONS_M, "SCHEMA", "HORIZON_SET_OR_ORDER_INVALID", "horizons must be 1.0,1.5,2.0 in frozen order", seed=seed, row=row_index, band=band["band"])
            for cell in cells:
                require(cell.get("truth_state") in STATES and cell.get("predicted_state") in STATES, "SCHEMA", "TRISTATE_INVALID", "truth/predicted state is invalid", seed=seed, row=row_index, band=band["band"], horizon=cell.get("horizon_m"))


def _confusion_summary(
    pairs: Iterable[tuple[str, str]],
    *,
    scope: str,
    require_global_denominators: bool = False,
) -> dict[str, Any]:
    values = list(pairs)
    truth_known = [(truth, pred) for truth, pred in values if truth != "UNKNOWN"]
    paired_known = [(truth, pred) for truth, pred in truth_known if pred != "UNKNOWN"]
    occupied = [(truth, pred) for truth, pred in paired_known if truth == "OCCUPIED_OBSERVED"]
    clear = [(truth, pred) for truth, pred in paired_known if truth == "CLEAR_OBSERVED"]
    if require_global_denominators:
        require(bool(truth_known), "DENOMINATOR", "TRUTH_KNOWN_DENOMINATOR_ZERO", "truth-known denominator is zero", scope=scope)
        require(bool(paired_known), "DENOMINATOR", "PAIRED_KNOWN_DENOMINATOR_ZERO", "paired-known denominator is zero", scope=scope)
        require(bool(occupied), "DENOMINATOR", "TRUTH_OCCUPIED_DENOMINATOR_ZERO", "truth-occupied denominator is zero", scope=scope)
        require(bool(clear), "DENOMINATOR", "TRUTH_CLEAR_DENOMINATOR_ZERO", "truth-clear denominator is zero", scope=scope)
    false_clear = sum(truth == "OCCUPIED_OBSERVED" and pred == "CLEAR_OBSERVED" for truth, pred in paired_known)
    false_block = sum(truth == "CLEAR_OBSERVED" and pred == "OCCUPIED_OBSERVED" for truth, pred in paired_known)
    return {
        "total_cells": len(values),
        "unknown_truth_excluded_count": len(values) - len(truth_known),
        "truth_known_count": len(truth_known),
        "paired_known_count": len(paired_known),
        "predicted_unknown_on_truth_known_count": len(truth_known) - len(paired_known),
        "known_coverage": len(paired_known) / len(truth_known) if truth_known else None,
        "false_clear_count": false_clear,
        "false_clear_all_known": false_clear / len(paired_known) if paired_known else None,
        "false_clear_given_occupied": false_clear / len(occupied) if occupied else None,
        "false_block_count": false_block,
        "false_block_given_clear": false_block / len(clear) if clear else None,
        "defined": {
            "known_coverage": bool(truth_known),
            "false_clear_all_known": bool(paired_known),
            "false_clear_given_occupied": bool(occupied),
            "false_block_given_clear": bool(clear),
        },
        "denominators": {"truth_known": len(truth_known), "paired_known": len(paired_known), "truth_occupied": len(occupied), "truth_clear": len(clear)},
    }


def _mae(values: list[float], *, scope: str, code: str) -> dict[str, Any]:
    require(bool(values), "DENOMINATOR", code, "metric denominator is zero", scope=scope)
    require(all(math.isfinite(value) for value in values), "SCHEMA", "NONFINITE_METRIC_INPUT", "metric input is non-finite", scope=scope)
    return {"support": len(values), "value": float(statistics.fmean(values))}


def compute_seed_metrics(rows: list[dict[str, Any]], *, seed: int, gates: dict[str, Any]) -> dict[str, Any]:
    validate_observations(rows, seed=seed)
    all_pairs: list[tuple[str, str]] = []
    grid_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    parent_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    orientation_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    near_field_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    environment_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    low_light_blur_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    clearance_errors: list[float] = []
    clearance_by_band: dict[str, list[float]] = defaultdict(list)
    clearance_by_parent: dict[str, list[float]] = defaultdict(list)
    temporal_series: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)

    for row in rows:
        parent = row["parent_id"]
        orientation = row["orientation"]
        near_field_key = "near_field" if row.get("near_field") is True else "not_near_field"
        environment_key = str(row.get("environment"))
        low_light_blur_key = "low_light_or_blur" if row.get("low_light_blur") is True else "nominal_light_motion"
        for band in row["bands"]:
            band_name = band["band"]
            if band["clearance_valid"]:
                error = abs(float(band["predicted_clearance_m"]) - float(band["truth_clearance_m"]))
                clearance_errors.append(error)
                clearance_by_band[band_name].append(error)
                clearance_by_parent[parent].append(error)
                temporal_series[(parent, row["session_id"], band_name)].append((int(row["sequence_index"]), float(band["truth_clearance_m"]), float(band["predicted_clearance_m"])))
            for cell in band["cells"]:
                pair = (cell["truth_state"], cell["predicted_state"])
                key = f"{band_name}@{float(cell['horizon_m']):.1f}m"
                all_pairs.append(pair)
                grid_pairs[key].append(pair)
                parent_pairs[parent].append(pair)
                orientation_pairs[orientation].append(pair)
                near_field_pairs[near_field_key].append(pair)
                environment_pairs[environment_key].append(pair)
                low_light_blur_pairs[low_light_blur_key].append(pair)

    pooled = _confusion_summary(all_pairs, scope=f"seed-{seed}:pooled", require_global_denominators=True)
    by_grid = {key: _confusion_summary(grid_pairs[key], scope=f"seed-{seed}:grid:{key}") for key in (f"{band}@{horizon:.1f}m" for band in BANDS for horizon in HORIZONS_M)}
    by_parent = {key: _confusion_summary(value, scope=f"seed-{seed}:parent:{key}") for key, value in sorted(parent_pairs.items())}
    by_orientation = {key: _confusion_summary(value, scope=f"seed-{seed}:orientation:{key}") for key, value in sorted(orientation_pairs.items())}
    by_near_field = {key: _confusion_summary(value, scope=f"seed-{seed}:near-field:{key}") for key, value in sorted(near_field_pairs.items())}
    by_environment = {key: _confusion_summary(value, scope=f"seed-{seed}:environment:{key}") for key, value in sorted(environment_pairs.items())}
    by_low_light_blur = {key: _confusion_summary(value, scope=f"seed-{seed}:low-light-blur:{key}") for key, value in sorted(low_light_blur_pairs.items())}

    temporal_errors: list[float] = []
    temporal_by_band: dict[str, list[float]] = defaultdict(list)
    temporal_by_parent: dict[str, list[float]] = defaultdict(list)
    for (parent, _, band_name), samples in sorted(temporal_series.items()):
        samples.sort()
        for previous, current in zip(samples, samples[1:]):
            if current[0] != previous[0] + 1:
                continue
            error = abs((current[2] - previous[2]) - (current[1] - previous[1]))
            temporal_errors.append(error)
            temporal_by_band[band_name].append(error)
            temporal_by_parent[parent].append(error)

    clearance = _mae(clearance_errors, scope=f"seed-{seed}:clearance", code="CLEARANCE_DENOMINATOR_ZERO")
    clearance["by_band"] = {band: _mae(clearance_by_band[band], scope=f"seed-{seed}:clearance:{band}", code="CLEARANCE_BAND_DENOMINATOR_ZERO") for band in BANDS}
    temporal = _mae(temporal_errors, scope=f"seed-{seed}:temporal", code="TEMPORAL_DENOMINATOR_ZERO")
    temporal["by_band"] = {band: _mae(temporal_by_band[band], scope=f"seed-{seed}:temporal:{band}", code="TEMPORAL_BAND_DENOMINATOR_ZERO") for band in BANDS}
    for parent in by_parent:
        by_parent[parent]["clearance_mae_m"] = _mae(clearance_by_parent[parent], scope=f"seed-{seed}:clearance:parent:{parent}", code="CLEARANCE_PARENT_DENOMINATOR_ZERO")
        by_parent[parent]["temporal_clearance_delta_mae_m"] = _mae(temporal_by_parent[parent], scope=f"seed-{seed}:temporal:parent:{parent}", code="TEMPORAL_PARENT_DENOMINATOR_ZERO")
    def macro_defined(key: str) -> float | None:
        values = [float(value[key]) for value in by_parent.values() if value[key] is not None]
        return float(statistics.fmean(values)) if values else None

    parent_macro = {
        "known_coverage": macro_defined("known_coverage"),
        "false_clear_all_known": macro_defined("false_clear_all_known"),
        "false_clear_given_occupied": macro_defined("false_clear_given_occupied"),
        "false_block_given_clear": macro_defined("false_block_given_clear"),
        "clearance_mae_m": float(statistics.fmean(value["clearance_mae_m"]["value"] for value in by_parent.values())),
        "temporal_clearance_delta_mae_m": float(statistics.fmean(value["temporal_clearance_delta_mae_m"]["value"] for value in by_parent.values())),
        "parent_count": len(by_parent),
    }
    parent_false_clear = {parent: float(metrics["false_clear_all_known"]) for parent, metrics in by_parent.items() if metrics["false_clear_all_known"] is not None}
    require(bool(parent_false_clear), "DENOMINATOR", "PARENT_FALSE_CLEAR_DENOMINATOR_ZERO", "no parent has a defined all-known false-clear denominator", seed=seed)
    worst_parent = max(parent_false_clear.items(), key=lambda item: (item[1], item[0]))

    coverage_pass = pooled["known_coverage"] >= float(gates["known_coverage_min"])
    if not coverage_pass:
        raise EvaluationError("COVERAGE", "KNOWN_COVERAGE_COLLAPSE", "pooled known coverage collapsed", seed=seed, pooled=pooled["known_coverage"], by_grid={key: value["known_coverage"] for key, value in by_grid.items()})
    gate_checks = {
        "known_coverage": coverage_pass,
        "false_clear_all_known": pooled["false_clear_all_known"] <= float(gates["false_clear_all_known_max"]),
        "false_block_given_clear": pooled["false_block_given_clear"] <= float(gates["false_block_given_clear_max"]),
        "clearance_mae_m": clearance["value"] <= float(gates["clearance_mae_m_max"]),
        "temporal_clearance_delta_mae_m": temporal["value"] <= float(gates["temporal_clearance_delta_mae_m_max"]),
        "worst_parent_false_clear_all_known": worst_parent[1] <= float(gates["worst_parent_false_clear_all_known_max"]),
    }
    return {
        "seed": seed,
        "pooled": pooled,
        "by_grid": by_grid,
        "by_parent": by_parent,
        "parent_macro": parent_macro,
        "by_orientation": by_orientation,
        "by_near_field": by_near_field,
        "by_environment": by_environment,
        "by_low_light_blur": by_low_light_blur,
        "clearance_mae_m": clearance,
        "temporal_clearance_delta_mae_m": temporal,
        "worst_parent_false_clear_all_known": {"parent_id": worst_parent[0], "value": worst_parent[1]},
        "gates": gate_checks,
        "diagnostics": {
            "nine_grid_coverage_pass_count": sum(
                value["known_coverage"] is not None and value["known_coverage"] >= float(gates["known_coverage_min"])
                for value in by_grid.values()
            ),
            "nine_grid_count": 9,
            "strata_are_diagnostic_not_global_blockers": True,
        },
    }


def _aggregate(values: list[float]) -> dict[str, Any]:
    require(len(values) == 3 and all(math.isfinite(value) for value in values), "SEEDS", "AGGREGATE_INPUT_INVALID", "three finite seed metrics are required")
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, values, strict=True)},
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_seed_metrics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    require(tuple(item["seed"] for item in seed_metrics) == EXPECTED_SEEDS, "SEEDS", "AGGREGATE_SEED_ORDER_INVALID", "aggregate seed order must be 17,29,43")
    metric_getters = {
        "known_coverage": lambda item: item["pooled"]["known_coverage"],
        "false_clear_all_known": lambda item: item["pooled"]["false_clear_all_known"],
        "false_clear_given_occupied": lambda item: item["pooled"]["false_clear_given_occupied"],
        "false_block_given_clear": lambda item: item["pooled"]["false_block_given_clear"],
        "clearance_mae_m": lambda item: item["clearance_mae_m"]["value"],
        "temporal_clearance_delta_mae_m": lambda item: item["temporal_clearance_delta_mae_m"]["value"],
        "worst_parent_false_clear_all_known": lambda item: item["worst_parent_false_clear_all_known"]["value"],
        "parent_macro_false_clear_all_known": lambda item: item["parent_macro"]["false_clear_all_known"],
        "parent_macro_clearance_mae_m": lambda item: item["parent_macro"]["clearance_mae_m"],
        "parent_macro_temporal_clearance_delta_mae_m": lambda item: item["parent_macro"]["temporal_clearance_delta_mae_m"],
    }
    gate_summary = {
        gate: {
            "values_by_seed": {str(item["seed"]): bool(item["gates"][gate]) for item in seed_metrics},
            "pass_count": sum(bool(item["gates"][gate]) for item in seed_metrics),
            "at_least_two_of_three_pass": sum(bool(item["gates"][gate]) for item in seed_metrics) >= 2,
        }
        for gate in seed_metrics[0]["gates"]
    }
    aggregate_task_pass = all(summary["at_least_two_of_three_pass"] for gate, summary in gate_summary.items() if gate != "known_coverage")
    return {
        "seed_order": list(EXPECTED_SEEDS),
        "selected_seed": None,
        "best_seed_selection_forbidden": True,
        "statistics": {name: _aggregate([float(getter(item)) for item in seed_metrics]) for name, getter in metric_getters.items()},
        "gate_summary": gate_summary,
        "every_seed_coverage_frontdoor_pass": all(item["gates"]["known_coverage"] for item in seed_metrics),
        "aggregate_task_gate_rule": "metricwise at least 2 of 3 seeds pass; no seed is selected",
        "aggregate_task_gate_pass": aggregate_task_pass,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Assistive Geometry A0 synthetic evaluation dry-run",
        "",
        f"Terminal: `{result['terminal']}`",
        "",
        "This receipt validates evaluator mechanics only. It contains no Development or Confirmation outcome and carries no model-quality authority.",
        "",
        "| Seed | coverage | false-clear/all-known | false-block/truth-clear | clearance MAE (m) | temporal delta MAE (m) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["seed_metrics"]:
        lines.append(
            f"| {item['seed']} | {item['pooled']['known_coverage']:.6f} | {item['pooled']['false_clear_all_known']:.6f} | "
            f"{item['pooled']['false_block_given_clear']:.6f} | {item['clearance_mae_m']['value']:.6f} | "
            f"{item['temporal_clearance_delta_mae_m']['value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "- Checkpoints: all 12 retained checkpoints (epochs 5/10/15/20 across seeds 17/29/43) passed bytes, SHA, protocol, seed, step, state, RNG and final-carry checks.",
            "- Nine-grid: all left/center/right × 1.0/1.5/2.0 m cells had defined CLEAR and OCCUPIED denominators.",
            "- Seed policy: no seed was selected; mean/sample-std/median/min/max preserve all three seeds.",
            "- UNKNOWN truth was excluded rather than counted as CLEAR or OCCUPIED.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_package(package_path: Path, evaluation_protocol_path: Path, *, expected_data_role: str) -> dict[str, Any]:
    package = load_json(package_path)
    require(package.get("schema") == "blindassist_assistive_geometry_b1_a0_evaluation_package_v1", "SCHEMA", "PACKAGE_SCHEMA_DRIFT", "evaluation package schema drift")
    forbidden_key = _contains_forbidden_seed_selection(package)
    require(forbidden_key is None, "BEST_SEED", "BEST_SEED_FIELD_PRESENT", "evaluation package attempts to select a seed", key=forbidden_key)
    require(package.get("data_role") == expected_data_role, "DATA_ROLE", "PACKAGE_DATA_ROLE_MISMATCH", "evaluation package data role mismatch")
    evaluation_protocol, evaluation_protocol_sha = validate_evaluation_protocol(evaluation_protocol_path)
    require(package.get("evaluation_protocol_sha256") == evaluation_protocol_sha, "PROTOCOL", "EVALUATION_PROTOCOL_SHA_MISMATCH", "evaluation protocol SHA mismatch")
    if expected_data_role != SYNTHETIC_ROLE:
        require(
            evaluation_protocol["firewalls"].get("development_outcome_access") is True,
            "DATA_ROLE",
            "DEVELOPMENT_EVALUATION_NOT_ACTIVATED",
            "this protocol does not authorize Development outcome access",
        )
    training_protocol_sha, checked_runs = validate_training_runs(package, package_path.parent)
    seed_metrics: list[dict[str, Any]] = []
    for run in checked_runs:
        rows = load_observations(run["observations_path"], seed=run["seed"], expected_role=expected_data_role)
        seed_metrics.append(compute_seed_metrics(rows, seed=run["seed"], gates=evaluation_protocol["metric_gates"]))
    aggregate = aggregate_seed_metrics(seed_metrics)
    status = "PASS" if aggregate["aggregate_task_gate_pass"] else "FAIL"
    terminal = TERMINALS["PASS"] if status == "PASS" else TERMINALS["TASK"]
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_evaluation_result_v1",
        "status": status,
        "terminal": terminal,
        "code": None if status == "PASS" else "AGGREGATE_TASK_GATE_FAILED",
        "data_role": expected_data_role,
        "training_protocol_sha256": training_protocol_sha,
        "evaluation_protocol_sha256": evaluation_protocol_sha,
        "checkpoint_integrity": {
            "seeds": list(EXPECTED_SEEDS),
            "epochs": list(EXPECTED_CHECKPOINT_STEPS),
            "optimizer_steps_by_epoch": {str(key): value for key, value in EXPECTED_CHECKPOINT_STEPS.items()},
            "checkpoint_count": 12,
            "runs": [
                {
                    "seed": run["seed"],
                    "train_result_path": run["train_result_path"],
                    "observations_path": str(run["observations_path"]),
                    "checkpoints": run["checkpoints"],
                }
                for run in checked_runs
            ],
        },
        "seed_metrics": seed_metrics,
        "aggregate": aggregate,
        "development_content_opened": False,
        "confirmation_content_opened": False,
        "claim_ceiling": "Synthetic evaluator mechanics only; no model quality, Development, Confirmation, deployment, product or safety authority.",
    }


def run_package(package_path: Path, evaluation_protocol_path: Path, output_root: Path, *, expected_data_role: str) -> dict[str, Any]:
    require(not output_root.exists(), "SCHEMA", "OUTPUT_ROOT_EXISTS", "evaluation output root already exists", path=str(output_root))
    output_root.mkdir(parents=True)
    try:
        result = evaluate_package(package_path, evaluation_protocol_path, expected_data_role=expected_data_role)
        result["completed_at"] = utc_now()
        atomic_write_json(output_root / "evaluation_result.json", result)
        atomic_write_text(output_root / "report.md", render_report(result))
        if result["status"] == "FAIL":
            failure = {
                "schema": "blindassist_assistive_geometry_b1_a0_evaluation_failure_v1",
                "status": "FAIL",
                "terminal": result["terminal"],
                "code": result["code"],
                "message": "complete three-seed evaluation did not pass the metricwise 2-of-3 task gate",
                "context": {"gate_summary": result["aggregate"]["gate_summary"]},
                "failed_at": result["completed_at"],
                "development_content_opened": False,
                "confirmation_content_opened": False,
            }
            atomic_write_json(output_root / "failure.json", failure)
            atomic_write_text(output_root / "failure.log", f"terminal={result['terminal']}\ncode={result['code']}\nmessage={failure['message']}\n")
        return result
    except EvaluationError as error:
        failure = {
            "schema": "blindassist_assistive_geometry_b1_a0_evaluation_failure_v1",
            "status": "FAIL",
            "terminal": error.terminal,
            "code": error.code,
            "message": str(error),
            "context": error.context,
            "failed_at": utc_now(),
            "development_content_opened": False,
            "confirmation_content_opened": False,
        }
        atomic_write_json(output_root / "failure.json", failure)
        atomic_write_text(output_root / "failure.log", f"terminal={error.terminal}\ncode={error.code}\nmessage={error}\ncontext={json.dumps(error.context, ensure_ascii=False, sort_keys=True)}\n")
        return failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-data-role", choices=(SYNTHETIC_ROLE, "DEVELOPMENT_SELECTION"), required=True)
    args = parser.parse_args()
    result = run_package(args.package.resolve(), args.evaluation_protocol.resolve(), args.output_root.resolve(), expected_data_role=args.expected_data_role)
    print(json.dumps({key: value for key, value in result.items() if key not in {"seed_metrics", "checkpoint_integrity"}}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
