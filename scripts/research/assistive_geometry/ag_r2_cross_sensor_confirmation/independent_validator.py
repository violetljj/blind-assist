"""Standalone verifier for AG R2 F2 confirmation evidence.

Import firewall: this file intentionally imports no producer, source adapter,
recipe, metric, evidence, model, reducer, or AG package module.  The metric
replay below is a second implementation over sealed NPZ arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

PARENTS = ("plant_scene_2", "motion_1", "mannequin_5")
FAMILIES = ("depth", "support", "boundary")
STRATA = 10
EXPECTED_PHASES = (
    ("roster.json", None),
    ("phase-a/raw-prediction-completion.json", "roster.json"),
    ("phase-b/session-contexts.json", "phase-a/raw-prediction-completion.json"),
    ("phase-c/conditioned-factor-completion.json", "phase-b/session-contexts.json"),
    ("phase-d/truth-completion.json", "phase-c/conditioned-factor-completion.json"),
    ("source-summary.json", "phase-d/truth-completion.json"),
)


class ValidationError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _safe(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative, "F2V_PATH_TEXT")
    value = PurePosixPath(relative)
    _require(not value.is_absolute() and ".." not in value.parts and value.as_posix() == relative, "F2V_PATH_UNSAFE")
    path = (root / Path(*value.parts)).resolve()
    _require(path.parent == root or root in path.parents, "F2V_PATH_ESCAPE")
    return path


def _json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(code, str(error)) from error
    _require(isinstance(value, dict), code)
    return value


def _sealed(value: Mapping[str, Any], code: str) -> dict[str, Any]:
    result = dict(value)
    seal = result.pop("content_sha256", None)
    _require(isinstance(seal, str) and _canonical_hash(result) == seal, code)
    return dict(value)


def _manifest(root: Path) -> dict[str, Any]:
    manifest = _json(root / "manifest.json", "F2V_MANIFEST_READ")
    _require(
        set(manifest) == {"schema", "evidence_root_consumed", "terminal", "file_count_before_manifest", "bytes_before_manifest", "files"}
        and manifest["schema"] == "blindassist.ag.r2.cross_sensor_factor_confirmation_manifest.v1"
        and manifest["evidence_root_consumed"] is True
        and isinstance(manifest["files"], dict),
        "F2V_MANIFEST_SCHEMA",
    )
    files = manifest["files"]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".partial")
    }
    _require(set(files) == actual, "F2V_MANIFEST_FILE_SET")
    total = 0
    for relative, receipt in files.items():
        _require(
            isinstance(receipt, Mapping) and set(receipt) == {"path", "bytes", "sha256"}
            and receipt["path"] == relative and type(receipt["bytes"]) is int,
            "F2V_MANIFEST_ROW_SCHEMA",
        )
        path = _safe(root, relative)
        _require(path.is_file() and path.stat().st_size == receipt["bytes"] and _sha(path) == receipt["sha256"], "F2V_MANIFEST_FILE_DRIFT")
        total += receipt["bytes"]
    _require(manifest["file_count_before_manifest"] == len(files) and manifest["bytes_before_manifest"] == total, "F2V_MANIFEST_ACCOUNTING")
    return manifest


def _validate_phase_chain(root: Path) -> list[dict[str, Any]]:
    phases = []
    prior_path: str | None = None
    prior_sha: str | None = None
    for relative, expected_prior in EXPECTED_PHASES:
        path = _safe(root, relative)
        value = _sealed(_json(path, "F2V_PHASE_READ"), "F2V_PHASE_SEAL")
        if expected_prior is None:
            _require("predecessor" not in value, "F2V_ROSTER_PREDECESSOR")
        else:
            _require(
                value.get("predecessor") == {"path": prior_path, "sha256": prior_sha}
                and expected_prior == prior_path,
                "F2V_PHASE_PREDECESSOR",
            )
        phases.append(value)
        prior_path = relative
        prior_sha = _sha(path)
    return phases


def _validate_partial_failure(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Verify a valid source NOT_EVALUABLE or invalid closed partial root."""

    if (root / "summary.json").exists() or not (root / "failure.json").exists():
        return None
    result = _sealed(_json(root / "result.json", "F2V_RESULT_READ"), "F2V_RESULT_SEAL")
    terminal = result.get("terminal")
    _require(terminal in {"NOT_EVALUABLE", "INVALID_AND_CLOSE_EVIDENCE_VERSION_ONLY"}, "F2V_PARTIAL_TERMINAL")
    failure = _sealed(_json(root / "failure.json", "F2V_FAILURE_READ"), "F2V_FAILURE_SEAL")
    _require(
        failure.get("terminal") == terminal == manifest["terminal"]
        and result.get("failure_sha256") == failure["content_sha256"]
        and result.get("failure_file_sha256") == _sha(root / "failure.json")
        and result.get("summary_sha256") is None
        and result.get("execution_valid") is (terminal == "NOT_EVALUABLE")
        and result.get("training_steps") == result.get("reducer_calls") == result.get("network_requests") == 0,
        "F2V_PARTIAL_RESULT_BINDING",
    )
    available: list[str] = []
    missing_seen = False
    prior_path: str | None = None
    prior_sha: str | None = None
    for relative, expected_prior in EXPECTED_PHASES:
        exists = (root / relative).is_file()
        if not exists:
            missing_seen = True
            continue
        _require(not missing_seen, "F2V_PARTIAL_PHASE_GAP")
        value = _sealed(_json(root / relative, "F2V_PARTIAL_PHASE_READ"), "F2V_PARTIAL_PHASE_SEAL")
        if expected_prior is None:
            _require("predecessor" not in value, "F2V_PARTIAL_ROSTER_PREDECESSOR")
        else:
            _require(value.get("predecessor") == {"path": prior_path, "sha256": prior_sha}, "F2V_PARTIAL_PHASE_PREDECESSOR")
        available.append(relative)
        prior_path = relative
        prior_sha = _sha(root / relative)
    events = failure.get("access_events")
    _require(isinstance(events, list), "F2V_PARTIAL_ACCESS_EVENTS")
    allowed = {
        "ROSTER": "ROSTER_METADATA",
        "RAW_SCORE_PREDICTION": "RAW_SCORE_PREDICTION",
        "CALIBRATION_SOURCE": "CALIBRATION_SOURCE",
        "SCORE_SOURCE": "SCORE_SOURCE",
    }
    for index, event in enumerate(events):
        _require(
            isinstance(event, Mapping)
            and event.get("event_index") == index
            and event.get("firewall_stage") in allowed
            and event.get("source_phase") == allowed[event["firewall_stage"]],
            "F2V_PARTIAL_ACCESS_EVENT_SCHEMA",
        )
        if event["source_phase"] == "RAW_SCORE_PREDICTION":
            _require("roster.json" in available, "F2V_PARTIAL_RAW_BEFORE_ROSTER")
        elif event["source_phase"] == "CALIBRATION_SOURCE":
            _require("phase-a/raw-prediction-completion.json" in available, "F2V_PARTIAL_CALIBRATION_BEFORE_RAW")
        elif event["source_phase"] == "SCORE_SOURCE":
            _require("phase-c/conditioned-factor-completion.json" in available, "F2V_PARTIAL_TRUTH_BEFORE_CONDITIONED")
    return {
        "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_independent_validation.v1",
        "passed": True,
        "terminal": "AG_R2_F2_CLOSED_PARTIAL_EVIDENCE_INDEPENDENTLY_VERIFIED",
        "scientific_terminal": terminal,
        "failure_reason_code": failure.get("reason_code"),
        "completed_phase_count": len(available),
        "manifest_file_count": manifest["file_count_before_manifest"],
        "manifest_sha256": _sha(root / "manifest.json"),
    }


PREDICTION_KEYS = {
    "parent_id", "frame_id", "source_hw", "output_hw", "intrinsics",
    "depth_m", "depth_log_sigma", "depth_known", "support_probability",
    "support_residual_sigma_m", "support_known", "obstacle_probability",
    "boundary_distance_px", "boundary_sigma_px", "evidence_known",
}
TRUTH_KEYS = {
    "parent_id", "frame_id", "fx", "fy", "camera_height_m", "depth_m",
    "depth_known", "support_probability", "support_signed_residual_m",
    "support_known", "obstacle_probability", "boundary_distance_px", "evidence_known",
}


def _npz(root: Path, receipt: Mapping[str, Any], expected: set[str], code: str) -> dict[str, np.ndarray]:
    _require(isinstance(receipt, Mapping) and set(receipt) == {"parent_id", "frame_id", "path", "bytes", "sha256"}, f"{code}_RECEIPT")
    path = _safe(root, str(receipt["path"]))
    _require(path.is_file() and path.stat().st_size == receipt["bytes"] and _sha(path) == receipt["sha256"], f"{code}_BINDING")
    try:
        with np.load(path, allow_pickle=False) as payload:
            _require(set(payload.files) == expected, f"{code}_KEY_SET")
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, ValueError) as error:
        raise ValidationError(f"{code}_READ", str(error)) from error
    parent = arrays["parent_id"]
    frame = arrays["frame_id"]
    _require(parent.ndim == frame.ndim == 0 and parent.dtype.kind == frame.dtype.kind == "U", f"{code}_IDENTITY_DTYPE")
    _require(str(parent.item()) == receipt["parent_id"] and str(frame.item()) == receipt["frame_id"], f"{code}_IDENTITY")
    return arrays


def _records(root: Path, completion: Mapping[str, Any], expected: set[str], code: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    rows = completion.get("records")
    _require(isinstance(rows, list) and completion.get("record_count") == len(rows) == 36, f"{code}_COUNT")
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in rows:
        arrays = _npz(root, row, expected, code)
        key = (str(arrays["parent_id"].item()), str(arrays["frame_id"].item()))
        _require(key[0] in PARENTS and key not in result, f"{code}_IDENTITY_SET")
        result[key] = arrays
    _require(all(sum(key[0] == parent for key in result) == 12 for parent in PARENTS), f"{code}_PARENT_COUNT")
    return result


def _known(array: np.ndarray, known: np.ndarray, name: str, *, positive: bool = False, probability: bool = False) -> None:
    _require(array.dtype == np.dtype("float64") and array.shape == known.shape, f"F2V_{name}_SCHEMA")
    _require(bool(np.all(np.isfinite(array[known]))) and bool(np.all(np.isnan(array[~known]))), f"F2V_{name}_UNKNOWN")
    if positive:
        _require(bool(np.all(array[known] > 0.0)), f"F2V_{name}_POSITIVE")
    if probability:
        _require(bool(np.all((array[known] >= 0.0) & (array[known] <= 1.0))), f"F2V_{name}_RANGE")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def _rho(sigma: np.ndarray, residual: np.ndarray, identity: np.ndarray) -> float:
    _require(sigma.size == residual.size == identity.size and sigma.size >= 20, "F2V_SPEARMAN_DENOMINATOR")
    order = np.lexsort((identity.astype(str), sigma))
    groups = np.array_split(order, STRATA)
    left = _rank(np.asarray([np.mean(sigma[group]) for group in groups]))
    right = _rank(np.asarray([np.mean(residual[group]) for group in groups]))
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    _require(denominator > 0.0 and math.isfinite(denominator), "F2V_SPEARMAN_UNDEFINED")
    return float(np.sum(left * right) / denominator)


def _mean(values: Sequence[float], code: str) -> float:
    _require(bool(values), code)
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    _require(math.isfinite(result), code)
    return result


def _score(
    protocol: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    predictions: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    truths: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    _require(set(predictions) == set(truths), "F2V_SCORE_IDENTITY_MISMATCH")
    source_rows = source_summary.get("parents")
    _require(isinstance(source_rows, list) and [row.get("parent_id") for row in source_rows] == list(PARENTS), "F2V_SOURCE_SUMMARY_PARENTS")
    source = {row["parent_id"]: row for row in source_rows}
    parent_metrics: dict[str, dict[str, float]] = {}
    uncertainty_parents: dict[str, dict[str, dict[str, float]]] = {}
    for parent in PARENTS:
        coverage = {family: [] for family in FAMILIES}
        combined: list[float] = []
        shape: list[float] = []
        scale: list[float] = []
        support_brier: list[float] = []
        obstacle_brier: list[float] = []
        boundary_error: list[float] = []
        uncertainty = {family: {"sigma": [], "residual": [], "identity": []} for family in FAMILIES}
        parent_keys = sorted(key for key in predictions if key[0] == parent)
        _require(len(parent_keys) == 12, "F2V_SCORE_PARENT_COUNT")
        for key in parent_keys:
            pred = predictions[key]
            truth = truths[key]
            output_hw = pred["output_hw"]
            _require(output_hw.dtype == np.dtype("int64") and output_hw.shape == (2,), "F2V_OUTPUT_HW")
            array_shape = tuple(int(value) for value in output_hw)
            _require(array_shape[0] > 0 and array_shape[1] > 0, "F2V_OUTPUT_HW_VALUE")
            pdk = pred["depth_known"]
            psk = pred["support_known"]
            pek = pred["evidence_known"]
            tdk = truth["depth_known"]
            tsk = truth["support_known"]
            tek = truth["evidence_known"]
            for value, name in ((pdk, "PRED_DEPTH"), (psk, "PRED_SUPPORT"), (pek, "PRED_EVIDENCE"), (tdk, "TRUTH_DEPTH"), (tsk, "TRUTH_SUPPORT"), (tek, "TRUTH_EVIDENCE")):
                _require(value.dtype == np.dtype("bool") and value.shape == array_shape, f"F2V_{name}_KNOWN")
            _known(pred["depth_m"], pdk, "PRED_DEPTH", positive=True)
            _known(pred["depth_log_sigma"], pdk, "PRED_DEPTH_SIGMA", positive=True)
            _known(pred["support_probability"], psk, "PRED_SUPPORT", probability=True)
            _known(pred["support_residual_sigma_m"], psk, "PRED_SUPPORT_SIGMA", positive=True)
            _known(pred["obstacle_probability"], pek, "PRED_OBSTACLE", probability=True)
            _known(pred["boundary_distance_px"], pek, "PRED_BOUNDARY")
            _known(pred["boundary_sigma_px"], pek, "PRED_BOUNDARY_SIGMA", positive=True)
            _known(truth["depth_m"], tdk, "TRUTH_DEPTH", positive=True)
            _known(truth["support_probability"], tsk, "TRUTH_SUPPORT", probability=True)
            _known(truth["support_signed_residual_m"], tsk, "TRUTH_SUPPORT_RESIDUAL")
            _known(truth["obstacle_probability"], tek, "TRUTH_OBSTACLE", probability=True)
            _known(truth["boundary_distance_px"], tek, "TRUTH_BOUNDARY")
            masks = {"depth": pdk & tdk, "support": psk & tsk, "boundary": pek & tek}
            for family, pred_known, truth_known in (("depth", pdk, tdk), ("support", psk, tsk), ("boundary", pek, tek)):
                denominator = int(np.sum(truth_known))
                _require(denominator > 0 and bool(np.any(masks[family])), "F2V_REQUIRED_DENOMINATOR")
                coverage[family].append(float(np.sum(pred_known & truth_known) / denominator))
            depth_mask = masks["depth"]
            signed = np.log(pred["depth_m"][depth_mask]) - np.log(truth["depth_m"][depth_mask])
            center = float(np.median(signed))
            combined.append(float(np.mean(np.abs(signed))))
            shape.append(float(np.mean(np.abs(signed - center))))
            scale.append(abs(center))
            support_mask = masks["support"]
            support_brier.append(float(np.mean((pred["support_probability"][support_mask] - truth["support_probability"][support_mask]) ** 2)))
            evidence_mask = masks["boundary"]
            obstacle_brier.append(float(np.mean((pred["obstacle_probability"][evidence_mask] - truth["obstacle_probability"][evidence_mask]) ** 2)))
            fx = float(truth["fx"].item())
            fy = float(truth["fy"].item())
            _require(math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0, "F2V_TRUTH_FOCAL")
            focal = math.sqrt(fx * fy)
            boundary_residual = np.arctan(np.abs(pred["boundary_distance_px"][evidence_mask] - truth["boundary_distance_px"][evidence_mask]) / focal)
            boundary_error.append(float(np.mean(boundary_residual)))
            values = {
                "depth": (pred["depth_log_sigma"][depth_mask], np.abs(signed)),
                "support": (pred["support_residual_sigma_m"][support_mask], np.abs(truth["support_signed_residual_m"][support_mask])),
                "boundary": (np.arctan(pred["boundary_sigma_px"][evidence_mask] / focal), boundary_residual),
            }
            for family, (sigma, residual) in values.items():
                uncertainty[family]["sigma"].append(sigma)
                uncertainty[family]["residual"].append(residual)
                uncertainty[family]["identity"].append(np.asarray([f"{key[1]}:{index:09d}" for index in range(sigma.size)]))
        parent_metrics[parent] = {
            "metric_prediction_known_coverage": _mean(coverage["depth"], "F2V_DEPTH_COVERAGE"),
            "support_prediction_known_coverage": _mean(coverage["support"], "F2V_SUPPORT_COVERAGE"),
            "obstacle_boundary_prediction_known_coverage": _mean(coverage["boundary"], "F2V_BOUNDARY_COVERAGE"),
            "depth_combined_abs_log_error": _mean(combined, "F2V_COMBINED"),
            "depth_shape_abs_log_error": _mean(shape, "F2V_SHAPE"),
            "depth_scale_abs_log_error": _mean(scale, "F2V_SCALE"),
            "support_brier": _mean(support_brier, "F2V_SUPPORT_BRIER"),
            "obstacle_brier": _mean(obstacle_brier, "F2V_OBSTACLE_BRIER"),
            "boundary_camera_angular_error_rad": _mean(boundary_error, "F2V_BOUNDARY_ERROR"),
        }
        uncertainty_parents[parent] = {}
        for family in FAMILIES:
            sigma = np.concatenate(uncertainty[family]["sigma"])
            residual = np.concatenate(uncertainty[family]["residual"])
            identity = np.concatenate(uncertainty[family]["identity"])
            uncertainty_parents[parent][family] = {
                "one_sigma_coverage": float(np.mean(residual <= sigma)),
                "two_sigma_coverage": float(np.mean(residual <= 2.0 * sigma)),
                "spearman_sigma_residual": _rho(sigma, residual, identity),
                "sample_count": int(sigma.size),
            }
    metrics: dict[str, Any] = {
        "confirmation_parent_count": 3,
        "calibration_identity_count_per_parent": min(int(row["calibration_count"]) for row in source_rows),
        "score_identity_count_per_parent": min(int(row["score_count"]) for row in source_rows),
        "minimum_metadata_eligible_pairs_across_parents": min(int(row["eligible_pair_count"]) for row in source_rows),
        "session_camera_height_m_each_parent": {parent: float(source[parent]["camera_height_m"]) for parent in PARENTS},
        "worst_parent_session_height_mad_m": max(float(row["camera_height_mad_m"]) for row in source_rows),
        "minimum_parent_source_metric_depth_known_fraction": min(float(row["source_depth_known_coverage"]) for row in source_rows),
        "minimum_parent_source_support_known_fraction": min(float(row["source_support_known_coverage"]) for row in source_rows),
        "minimum_parent_source_boundary_evidence_known_fraction": min(float(row["source_boundary_known_coverage"]) for row in source_rows),
    }
    reductions = {
        "minimum_parent_metric_prediction_known_coverage": ("metric_prediction_known_coverage", min),
        "minimum_parent_support_prediction_known_coverage": ("support_prediction_known_coverage", min),
        "minimum_parent_min_obstacle_and_boundary_prediction_known_coverage": ("obstacle_boundary_prediction_known_coverage", min),
        "parent_macro_depth_absolute_log_error": ("depth_combined_abs_log_error", np.mean),
        "parent_macro_depth_shape_absolute_log_error": ("depth_shape_abs_log_error", np.mean),
        "parent_macro_depth_scale_absolute_log_error": ("depth_scale_abs_log_error", np.mean),
        "parent_macro_support_brier": ("support_brier", np.mean),
        "parent_macro_obstacle_evidence_brier": ("obstacle_brier", np.mean),
        "parent_macro_boundary_camera_angular_error_rad": ("boundary_camera_angular_error_rad", np.mean),
    }
    for target, (source_name, reducer) in reductions.items():
        metrics[target] = float(reducer([parent_metrics[parent][source_name] for parent in PARENTS]))
    worst = {
        "worst_parent_depth_absolute_log_error": "depth_combined_abs_log_error",
        "worst_parent_depth_shape_absolute_log_error": "depth_shape_abs_log_error",
        "worst_parent_depth_scale_absolute_log_error": "depth_scale_abs_log_error",
        "worst_parent_support_brier": "support_brier",
        "worst_parent_obstacle_evidence_brier": "obstacle_brier",
        "worst_parent_boundary_camera_angular_error_rad": "boundary_camera_angular_error_rad",
    }
    for target, source_name in worst.items():
        metrics[target] = max(parent_metrics[parent][source_name] for parent in PARENTS)
    family_summary = {}
    for family in FAMILIES:
        family_summary[family] = {
            "parent_macro_one_sigma_coverage": _mean([uncertainty_parents[parent][family]["one_sigma_coverage"] for parent in PARENTS], "F2V_ONE_SIGMA"),
            "parent_macro_two_sigma_coverage": _mean([uncertainty_parents[parent][family]["two_sigma_coverage"] for parent in PARENTS], "F2V_TWO_SIGMA"),
            "parent_macro_spearman_sigma_residual": _mean([uncertainty_parents[parent][family]["spearman_sigma_residual"] for parent in PARENTS], "F2V_RHO"),
        }
    metrics["maximum_factor_family_abs_empirical_one_sigma_coverage_minus_0_6827"] = max(abs(row["parent_macro_one_sigma_coverage"] - 0.6827) for row in family_summary.values())
    metrics["maximum_factor_family_abs_empirical_two_sigma_coverage_minus_0_9545"] = max(abs(row["parent_macro_two_sigma_coverage"] - 0.9545) for row in family_summary.values())
    metrics["minimum_factor_family_parent_macro_spearman_sigma_residual"] = min(row["parent_macro_spearman_sigma_residual"] for row in family_summary.values())
    constraints = protocol.get("constraints")
    specs = [row for row in constraints if isinstance(row, Mapping) and row.get("class") == "GATE"] if isinstance(constraints, list) else []
    _require(len(specs) == 27, "F2V_GATE_COUNT")
    gates = []
    for spec in specs:
        name = spec["metric"]
        _require(name in metrics, "F2V_GATE_METRIC")
        value = metrics[name]
        if spec["operator"] == "RANGE_INCLUSIVE":
            lower, upper = (float(item) for item in spec["threshold"])
            passed = isinstance(value, Mapping) and all(lower <= float(item) <= upper for item in value.values())
        elif spec["operator"] == "EQ":
            passed = float(value) == float(spec["threshold"])
        elif spec["operator"] == "GTE":
            passed = float(value) >= float(spec["threshold"])
        else:
            _require(spec["operator"] == "LTE", "F2V_GATE_OPERATOR")
            passed = float(value) <= float(spec["threshold"])
        gates.append({key: spec[key] for key in ("id", "metric", "operator", "threshold", "unit")} | {"value": value, "passed": bool(passed)})
    source_pass = all(row["passed"] for row in gates[:9])
    model_pass = all(row["passed"] for row in gates[9:])
    terminal = "CONFIRM_PASS" if source_pass and model_pass else ("CONFIRM_FAIL" if source_pass else "NOT_EVALUABLE")
    result = {
        "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_summary.v1",
        "terminal": terminal,
        "source_evaluable": source_pass,
        "all_model_gates_passed": model_pass if source_pass else False,
        "spearman_method": {"strata": 10, "sort": "sigma_then_frame_id_then_flat_index", "ties": "average_rank", "undefined": "NOT_EVALUABLE"},
        "metrics": metrics,
        "parents": parent_metrics,
        "uncertainty": {"parents": uncertainty_parents, "families": family_summary},
        "gates": gates,
    }
    result["content_sha256"] = _canonical_hash(result)
    return result


def _semantic_equal(left: Any, right: Any, path: str = "summary") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        _require(set(left) == set(right), "F2V_SUMMARY_KEY_DRIFT")
        for key in left:
            _semantic_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        _require(len(left) == len(right), "F2V_SUMMARY_LENGTH_DRIFT")
        for a, b in zip(left, right):
            _semantic_equal(a, b, path)
        return
    if type(left) in {int, float} and type(right) in {int, float} and not isinstance(left, bool) and not isinstance(right, bool):
        _require(math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= 1e-12, "F2V_SUMMARY_NUMERIC_DRIFT")
        return
    _require(type(left) is type(right) and left == right, "F2V_SUMMARY_VALUE_DRIFT")


def verify(root: Path, protocol_path: Path) -> dict[str, Any]:
    evidence = root.resolve()
    _require(evidence.is_dir(), "F2V_ROOT_MISSING")
    protocol = _json(protocol_path.resolve(), "F2V_PROTOCOL_READ")
    manifest = _manifest(evidence)
    partial = _validate_partial_failure(evidence, manifest)
    if partial is not None:
        return partial
    phases = _validate_phase_chain(evidence)
    raw_completion = phases[1]
    context_completion = phases[2]
    conditioned_completion = phases[3]
    truth_completion = phases[4]
    source_summary = phases[5]
    _require(
        raw_completion.get("phase") == "RAW_RGBK_PREDICTIONS_SEALED"
        and context_completion.get("phase") == "CALIBRATION_SOURCE_CONTEXT_SEALED"
        and conditioned_completion.get("phase") == "CONDITIONED_FACTORS_SEALED_BEFORE_SCORE_TRUTH"
        and truth_completion.get("phase") == "SCORE_SOURCE_TRUTH_SEALED",
        "F2V_PHASE_NAME_DRIFT",
    )
    raw = _records(evidence, raw_completion, PREDICTION_KEYS, "F2V_RAW")
    conditioned = _records(evidence, conditioned_completion, PREDICTION_KEYS, "F2V_CONDITIONED")
    truths = _records(evidence, truth_completion, TRUTH_KEYS, "F2V_TRUTH")
    _require(set(raw) == set(conditioned) == set(truths), "F2V_RECORD_IDENTITY_DRIFT")
    summary = _sealed(_json(evidence / "summary.json", "F2V_SUMMARY_READ"), "F2V_SUMMARY_SEAL")
    replay = _score(protocol, source_summary, conditioned, truths)
    _semantic_equal(replay, summary)
    result = _sealed(_json(evidence / "result.json", "F2V_RESULT_READ"), "F2V_RESULT_SEAL")
    _require(
        result.get("terminal") == summary["terminal"] == manifest["terminal"]
        and result.get("summary_sha256") == summary["content_sha256"]
        and result.get("execution_valid") is True
        and result.get("training_steps") == result.get("reducer_calls") == result.get("network_requests") == 0,
        "F2V_RESULT_BINDING",
    )
    return {
        "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_independent_validation.v1",
        "passed": True,
        "terminal": "AG_R2_F2_CONFIRMATION_EVIDENCE_INDEPENDENTLY_VERIFIED",
        "scientific_terminal": summary["terminal"],
        "frame_count": len(conditioned),
        "gate_count": len(summary["gates"]),
        "summary_exact_replay": replay["content_sha256"] == summary["content_sha256"],
        "manifest_file_count": manifest["file_count_before_manifest"],
        "manifest_sha256": _sha(evidence / "manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root, args.protocol)
    except Exception as error:  # noqa: BLE001 - CLI must serialize every validation failure.
        print(json.dumps({"passed": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
