#!/usr/bin/env python3
"""Aggregate four D45 person-distance receipts without converting control failures into science."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_hftf_stage_c_d45_phone_metric_depth_aggregate_v1"
RECEIPT_SCHEMA = "blindassist_hftf_d45_person_measurement_canary_v1"
REQUIRED_DISTANCES_METERS = (1.0, 2.0, 3.0, 5.0)
MAX_RECEIPT_BYTES = 256 * 1024
MAX_MEASUREMENTS_PER_RECEIPT = 1_800
EXPECTED_BASELINE_APP_SHA256 = (
    "afa7a774b9f47074b2bf2e59755e712e92421484140789513578b32b68f0f149"
)
EXPECTED_DETECTOR_SHA256 = (
    "00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2"
)
DEFAULT_BASELINE_APP = Path("app/build/outputs/apk/debug/app-debug.apk")
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d45-phone-metric-depth-source-canary-r0/report.json"
)
MEASUREMENT_TERMINALS = {
    "DISTANCE_MEASUREMENT_OBSERVED",
    "DISTANCE_MEASUREMENT_OBSERVED_NO_ACCEPTED_SAMPLES",
    "DISTANCE_MEASUREMENT_OBSERVED_BELOW_ACCEPTANCE_COUNT",
}


class ControlPlaneError(ValueError):
    """An input/transport defect that must not become a D45 scientific terminal."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlPlaneError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ControlPlaneError(f"non-finite JSON constant: {value}")


def load_receipt(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ControlPlaneError(f"receipt is not a file: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_RECEIPT_BYTES:
        raise ControlPlaneError(
            f"receipt size {size} is outside 1..{MAX_RECEIPT_BYTES}: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ControlPlaneError(f"invalid UTF-8 JSON receipt {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ControlPlaneError(f"receipt root must be an object: {path}")
    return payload, sha256(path)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlPlaneError(f"{label} must be an object")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ControlPlaneError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ControlPlaneError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlPlaneError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ControlPlaneError(f"{label} is outside its finite range")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ControlPlaneError(f"{label} must be boolean")
    return value


def _sha256_string(value: Any, label: str) -> str:
    result = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ControlPlaneError(f"{label} must be lowercase SHA-256")
    return result


def _number_list(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) > MAX_MEASUREMENTS_PER_RECEIPT:
        raise ControlPlaneError(
            f"{label} must be an array with <= {MAX_MEASUREMENTS_PER_RECEIPT} values"
        )
    result = [_number(item, f"{label}[{index}]", minimum=minimum)
              for index, item in enumerate(value)]
    if maximum is not None and any(item > maximum for item in result):
        raise ControlPlaneError(f"{label} contains a value > {maximum}")
    return result


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _optional_close(value: Any, expected: float | None, label: str) -> None:
    if expected is None:
        if value is not None:
            raise ControlPlaneError(f"{label} must be null when no samples exist")
        return
    actual = _number(value, label, minimum=0.0)
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
        raise ControlPlaneError(
            f"{label} does not match bounded measurement values"
        )


def validate_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ControlPlaneError("unexpected D45 person receipt schema")
    run_id = _string(payload.get("run_id"), "run_id")
    terminal = _string(payload.get("terminal"), "terminal")
    if terminal not in MEASUREMENT_TERMINALS and not terminal.startswith(
        "NOT_EVALUABLE_"
    ):
        raise ControlPlaneError(f"unexpected D45 person terminal: {terminal}")

    reference = _mapping(payload.get("reference_distance"), "reference_distance")
    distance = _number(reference.get("meters"), "reference_distance.meters")
    if distance not in REQUIRED_DISTANCES_METERS:
        raise ControlPlaneError("reference distance must be exactly 1/2/3/5 m")
    if reference.get("definition") != "PERSON_TORSO_PLANE_TO_CAMERA_OPTICAL_CENTER":
        raise ControlPlaneError("reference distance definition mismatch")
    if reference.get("source") != "OPERATOR_DECLARED_INSTRUMENTATION_ARGUMENT":
        raise ControlPlaneError("reference distance source mismatch")

    coverage = _mapping(payload.get("coverage"), "coverage")
    accepted_count = _integer(
        coverage.get("accepted_measurement_count"),
        "coverage.accepted_measurement_count",
    )
    exact_person_count = _integer(
        coverage.get("exact_single_person_frame_count"),
        "coverage.exact_single_person_frame_count",
    )
    if accepted_count > exact_person_count:
        raise ControlPlaneError("accepted count exceeds exact-single-person count")
    if terminal == "DISTANCE_MEASUREMENT_OBSERVED" and accepted_count < 20:
        raise ControlPlaneError("observed terminal requires >=20 accepted measurements")
    if (
        terminal == "DISTANCE_MEASUREMENT_OBSERVED_BELOW_ACCEPTANCE_COUNT"
        and not 1 <= accepted_count < 20
    ):
        raise ControlPlaneError("below-acceptance terminal/count mismatch")
    if (
        terminal == "DISTANCE_MEASUREMENT_OBSERVED_NO_ACCEPTED_SAMPLES"
        and accepted_count != 0
    ):
        raise ControlPlaneError("no-accepted-samples terminal/count mismatch")
    _optional_close(
        coverage.get("accepted_person_coverage"),
        accepted_count / exact_person_count if exact_person_count else None,
        "coverage.accepted_person_coverage",
    )

    bounded = _mapping(
        payload.get("bounded_measurement_values"),
        "bounded_measurement_values",
    )
    if bounded.get("maximum_value_count") != MAX_MEASUREMENTS_PER_RECEIPT:
        raise ControlPlaneError("bounded measurement maximum mismatch")
    depths = _number_list(
        bounded.get("optical_axis_depth_m"),
        "bounded_measurement_values.optical_axis_depth_m",
        minimum=0.20,
        maximum=20.0,
    )
    latencies = _number_list(
        bounded.get("source_to_measurement_latency_ms"),
        "bounded_measurement_values.source_to_measurement_latency_ms",
        minimum=0.0,
    )
    if len(depths) != accepted_count or len(latencies) != accepted_count:
        raise ControlPlaneError("bounded values must exactly match accepted count")

    errors = [abs(depth - distance) for depth in depths]
    relative_errors = [error / distance for error in errors]
    metric_error = _mapping(payload.get("metric_error"), "metric_error")
    if metric_error.get("accepted_observation_count") != accepted_count:
        raise ControlPlaneError("metric_error accepted count mismatch")
    _optional_close(
        metric_error.get("absolute_error_median_m"),
        percentile(errors, 0.50),
        "metric_error.absolute_error_median_m",
    )
    _optional_close(
        metric_error.get("absolute_error_p90_m"),
        percentile(errors, 0.90),
        "metric_error.absolute_error_p90_m",
    )
    _optional_close(
        metric_error.get("relative_error_median"),
        percentile(relative_errors, 0.50),
        "metric_error.relative_error_median",
    )

    history = _mapping(payload.get("history"), "history")
    eligible_history = _integer(
        history.get("eligible_window_count"),
        "history.eligible_window_count",
    )
    available_history = _integer(
        history.get("available_forecast_count"),
        "history.available_forecast_count",
    )
    if available_history > eligible_history:
        raise ControlPlaneError("available history exceeds eligible history")
    _optional_close(
        history.get("availability"),
        available_history / eligible_history if eligible_history else None,
        "history.availability",
    )

    device = _mapping(payload.get("device"), "device")
    build = _mapping(payload.get("build"), "build")
    source = _mapping(payload.get("source"), "source")
    detector = _mapping(payload.get("detector"), "detector")
    boundary = _mapping(payload.get("evidence_boundary"), "evidence_boundary")
    for field, expected in {
        "benchmark_only": True,
        "app_runtime_involved": False,
        "raster_pixels_persisted": False,
        "camera_images_persisted": False,
        "person_boxes_persisted": False,
        "event_outcome_evaluated": False,
        "navigation_output_issued": False,
        "production_authorized": False,
    }.items():
        if _boolean(boundary.get(field), f"evidence_boundary.{field}") != expected:
            raise ControlPlaneError(f"evidence boundary violation: {field}")
    if _integer(
        boundary.get("risk_feedback_invocation_count"),
        "evidence_boundary.risk_feedback_invocation_count",
    ) != 0:
        raise ControlPlaneError("risk/feedback invocation is outside D45 authority")

    detector_sha = _sha256_string(
        detector.get("model_sha256"),
        "detector.model_sha256",
    )
    if detector_sha != EXPECTED_DETECTOR_SHA256:
        raise ControlPlaneError("detector asset hash mismatch")
    if detector.get("backend") != "cpu_xnnpack":
        raise ControlPlaneError("detector backend mismatch")
    if detector.get("selection_contract") != "EXACTLY_ONE_PERSON_DETECTION":
        raise ControlPlaneError("person selection contract mismatch")

    camera_id = source.get("camera_id")
    if camera_id is not None:
        camera_id = _string(camera_id, "source.camera_id")
    detector_rotation = source.get("detector_rotation_degrees")
    if detector_rotation is not None:
        detector_rotation = _integer(
            detector_rotation,
            "source.detector_rotation_degrees",
        )
        if detector_rotation not in (0, 90, 180, 270):
            raise ControlPlaneError("source.detector_rotation_degrees is invalid")
    device_binding = (
        _string(device.get("manufacturer"), "device.manufacturer"),
        _string(device.get("brand"), "device.brand"),
        _string(device.get("model"), "device.model"),
        _string(device.get("device"), "device.device"),
        _string(device.get("build_fingerprint"), "device.build_fingerprint"),
        _integer(device.get("android_sdk_int"), "device.android_sdk_int", minimum=1),
        _string(device.get("android_release"), "device.android_release"),
    )
    source_binding = (
        _string(
            source.get("arcore_sdk_dependency_version"),
            "source.arcore_sdk_dependency_version",
        ),
        camera_id,
        detector_rotation,
    )
    if source_binding[0] != "1.33.0":
        raise ControlPlaneError("ARCore dependency version mismatch")
    binding = {
        "device": device_binding,
        "build": (
            _integer(build.get("target_apk_bytes"), "build.target_apk_bytes", minimum=1),
            _sha256_string(
                build.get("target_apk_sha256"),
                "build.target_apk_sha256",
            ),
            _integer(
                build.get("instrumentation_apk_bytes"),
                "build.instrumentation_apk_bytes",
                minimum=1,
            ),
            _sha256_string(
                build.get("instrumentation_apk_sha256"),
                "build.instrumentation_apk_sha256",
            ),
        ),
        "source": source_binding,
        "detector": (
            _string(detector.get("model_asset"), "detector.model_asset"),
            detector_sha,
            detector.get("backend"),
            detector.get("selection_contract"),
        ),
    }
    return {
        "run_id": run_id,
        "terminal": terminal,
        "distance": distance,
        "accepted_count": accepted_count,
        "exact_person_count": exact_person_count,
        "depths": depths,
        "latencies": latencies,
        "eligible_history": eligible_history,
        "available_history": available_history,
        "binding": binding,
    }


def _base_report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "authority": {
            "development_only": True,
            "event_utility_evaluated": False,
            "app_runtime_authorized": False,
            "production_or_safety_claim": False,
        },
    }


def evaluate_payloads(
    payloads: list[dict[str, Any]],
    *,
    baseline_app_sha256: str | None,
) -> dict[str, Any]:
    report = _base_report()
    try:
        observations = [validate_receipt(payload) for payload in payloads]
        distances = [item["distance"] for item in observations]
        if len(set(distances)) != len(distances):
            raise ControlPlaneError("duplicate reference-distance receipt")
        run_ids = [item["run_id"] for item in observations]
        if len(set(run_ids)) != len(run_ids):
            raise ControlPlaneError("duplicate run_id receipt")
        for binding_name in ("device", "build", "detector"):
            values = {item["binding"][binding_name] for item in observations}
            if len(values) > 1:
                raise ControlPlaneError(
                    f"four-distance {binding_name} binding mismatch"
                )
        for source_index, source_field in enumerate(
            ("arcore_sdk_dependency_version", "camera_id", "detector_rotation_degrees")
        ):
            known_values = {
                item["binding"]["source"][source_index]
                for item in observations
                if item["binding"]["source"][source_index] is not None
            }
            if len(known_values) > 1:
                raise ControlPlaneError(
                    f"four-distance source {source_field} binding mismatch"
                )
    except ControlPlaneError as error:
        report.update(
            evaluation_status="CONTROL_PLANE_INPUT_REJECTED",
            scientific_terminal=None,
            control_plane_errors=[str(error)],
        )
        return report

    missing = sorted(set(REQUIRED_DISTANCES_METERS) - set(distances))
    if missing:
        report.update(
            evaluation_status="INCOMPLETE_DISTANCE_SET",
            scientific_terminal=None,
            present_distances_m=sorted(distances),
            missing_distances_m=missing,
            control_plane_errors=[],
        )
        return report

    report["distance_receipts"] = [
        {
            "distance_m": item["distance"],
            "run_id": item["run_id"],
            "terminal": item["terminal"],
            "accepted_observation_count": item["accepted_count"],
        }
        for item in sorted(observations, key=lambda item: item["distance"])
    ]
    if any(item["terminal"].startswith("NOT_EVALUABLE_") for item in observations):
        report.update(
            evaluation_status="COMPLETE_SOURCE_NOT_EVALUABLE",
            scientific_terminal="D45_PHONE_METRIC_DEPTH_SOURCE_NOT_EVALUABLE",
            control_plane_errors=[],
        )
        return report

    if baseline_app_sha256 is None:
        report.update(
            evaluation_status="CONTROL_PLANE_BASELINE_NOT_READY",
            scientific_terminal=None,
            control_plane_errors=["frozen baseline App artifact is unavailable"],
        )
        return report
    if baseline_app_sha256 != EXPECTED_BASELINE_APP_SHA256:
        report.update(
            evaluation_status="CONTROL_PLANE_BASELINE_MISMATCH",
            scientific_terminal=None,
            control_plane_errors=[
                "baseline App artifact hash changed; no measurement terminal emitted"
            ],
            baseline_app_sha256=baseline_app_sha256,
        )
        return report

    all_depth_errors: list[float] = []
    all_relative_errors: list[float] = []
    all_latencies: list[float] = []
    for item in observations:
        errors = [abs(depth - item["distance"]) for depth in item["depths"]]
        all_depth_errors.extend(errors)
        all_relative_errors.extend(error / item["distance"] for error in errors)
        all_latencies.extend(item["latencies"])
    accepted_total = sum(item["accepted_count"] for item in observations)
    exact_person_total = sum(item["exact_person_count"] for item in observations)
    eligible_history_total = sum(item["eligible_history"] for item in observations)
    available_history_total = sum(item["available_history"] for item in observations)
    metrics = {
        "accepted_observation_count": accepted_total,
        "accepted_person_coverage": (
            accepted_total / exact_person_total if exact_person_total else None
        ),
        "absolute_error_median_m": percentile(all_depth_errors, 0.50),
        "relative_error_median": percentile(all_relative_errors, 0.50),
        "absolute_error_p90_m": percentile(all_depth_errors, 0.90),
        "source_to_measurement_latency_p95_ms": percentile(all_latencies, 0.95),
        "eligible_history_count": eligible_history_total,
        "available_history_count": available_history_total,
        "history_availability": (
            available_history_total / eligible_history_total
            if eligible_history_total
            else None
        ),
    }
    checks = {
        "each_distance_accepted_observations_gte_20": all(
            item["accepted_count"] >= 20 for item in observations
        ),
        "overall_accepted_person_coverage_gte_0_60": (
            metrics["accepted_person_coverage"] is not None
            and metrics["accepted_person_coverage"] >= 0.60
        ),
        "median_absolute_error_lte_0_50_m": (
            metrics["absolute_error_median_m"] is not None
            and metrics["absolute_error_median_m"] <= 0.50
        ),
        "median_relative_error_lte_0_20": (
            metrics["relative_error_median"] is not None
            and metrics["relative_error_median"] <= 0.20
        ),
        "p90_absolute_error_lte_1_00_m": (
            metrics["absolute_error_p90_m"] is not None
            and metrics["absolute_error_p90_m"] <= 1.00
        ),
        "latency_p95_lte_150_ms": (
            metrics["source_to_measurement_latency_p95_ms"] is not None
            and metrics["source_to_measurement_latency_p95_ms"] <= 150.0
        ),
        "eligible_history_availability_gte_0_50": (
            metrics["history_availability"] is not None
            and metrics["history_availability"] >= 0.50
        ),
        "risk_feedback_invocation_count_eq_0": True,
        "baseline_app_artifact_hash_unchanged": True,
    }
    supported = all(checks.values())
    report.update(
        evaluation_status="COMPLETE_MEASUREMENT_EVALUATED",
        scientific_terminal=(
            "D45_PHONE_METRIC_DEPTH_SOURCE_SUPPORTED_DEVELOPMENT_ONLY"
            if supported
            else "D45_PHONE_METRIC_DEPTH_SOURCE_NOT_SUPPORTED"
        ),
        control_plane_errors=[],
        baseline_app_sha256=baseline_app_sha256,
        aggregation_contract={
            "overall_error_population": "POOLED_ACCEPTED_OBSERVATIONS",
            "overall_coverage": (
                "SUM_ACCEPTED_MEASUREMENTS_DIV_SUM_EXACT_SINGLE_PERSON_FRAMES"
            ),
            "history_availability": (
                "SUM_AVAILABLE_FORECASTS_DIV_SUM_ELIGIBLE_WINDOWS"
            ),
            "percentile_interpolation": "LINEAR_RANK_N_MINUS_1",
        },
        metrics=metrics,
        frozen_gate={
            "checks": checks,
            "supported": supported,
        },
    )
    return report


def _atomic_nonoverwriting_write(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"D45 aggregate output is non-overwriting: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"D45 aggregate output appeared concurrently: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        action="append",
        type=Path,
        default=[],
        help="One person-measurement summary.json; repeat for up to four distances.",
    )
    parser.add_argument(
        "--baseline-app-apk",
        type=Path,
        default=DEFAULT_BASELINE_APP,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_records: list[dict[str, Any]] = []
    try:
        if len(args.receipt) > len(REQUIRED_DISTANCES_METERS):
            raise ControlPlaneError("at most four explicit receipts are accepted")
        payloads = []
        for path in args.receipt:
            payload, digest = load_receipt(path)
            payloads.append(payload)
            input_records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
        baseline_digest = (
            sha256(args.baseline_app_apk)
            if args.baseline_app_apk.is_file()
            else None
        )
        report = evaluate_payloads(
            payloads,
            baseline_app_sha256=baseline_digest,
        )
    except (ControlPlaneError, OSError) as error:
        report = _base_report()
        report.update(
            evaluation_status="CONTROL_PLANE_INPUT_REJECTED",
            scientific_terminal=None,
            control_plane_errors=[str(error)],
        )
    report["inputs"] = {
        "receipts": input_records,
        "baseline_app_apk": str(args.baseline_app_apk.resolve()),
        "expected_baseline_app_sha256": EXPECTED_BASELINE_APP_SHA256,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    print(encoded, end="")
    if report["scientific_terminal"] is None:
        return 2
    _atomic_nonoverwriting_write(args.output, encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
