#!/usr/bin/env python3
"""Frozen consumed-data evaluator for Dual-rate Metric Depth Observer R1.

The evaluator is causal: an anchor can affect a frame only after its simulated
service completion, and expiry uses source-frame age rather than completion
age.  It performs no parameter or threshold search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import HORIZONS_M, summarize

BANDS = ("left", "center", "right")
SCHEMA = "blindassist_dual_rate_metric_depth_observer_r1_result"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_input(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_inputs(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if protocol.get("schema") != "blindassist_dual_rate_metric_depth_observer_r1_protocol":
        raise ValueError("unexpected protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_D_ARM_EXECUTION":
        raise ValueError("protocol is not frozen")
    loaded: dict[str, dict[str, Any]] = {}
    for key in ("metric3d_report", "fast_report", "arm_c_report"):
        receipt = protocol["inputs"][key]
        path = resolve_input(str(receipt["path"]))
        actual = sha256(path)
        expected = str(receipt["sha256"]).upper()
        if actual != expected:
            raise ValueError(f"{key} SHA-256 mismatch: {actual} != {expected}")
        report = load_json(path)
        if not isinstance(report.get("frames"), list) or not report["frames"]:
            raise ValueError(f"{key}: missing frames")
        loaded[key] = report
    return loaded


def frame_key(frame: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(frame["sequence_id"]),
        float(frame["timestamp"]),
        str(frame["frame_path"]),
    )


def sorted_frames(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(report["frames"], key=frame_key)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round(fraction * (len(ordered) - 1))])


def valid_pairs(
    metric_field: dict[str, Any], fast_field: dict[str, Any]
) -> list[tuple[float, float]]:
    if metric_field.get("status") != "VALID" or fast_field.get("status") != "VALID":
        return []
    pairs = []
    for band in BANDS:
        metric = metric_field.get("bands", {}).get(band, {}).get("clearance_m")
        fast = fast_field.get("bands", {}).get(band, {}).get("clearance_m")
        if metric is None or fast is None:
            continue
        metric_value, fast_value = float(metric), float(fast)
        if math.isfinite(metric_value) and math.isfinite(fast_value):
            pairs.append((fast_value, metric_value))
    return pairs


def robust_affine_fit(
    anchors: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    pairs = [pair for anchor in anchors for pair in anchor["pairs"]]
    minimum_pairs = int(config["minimum_fit_pairs"])
    if len(pairs) < minimum_pairs:
        return {"status": "UNKNOWN_INSUFFICIENT_FIT_PAIRS", "pair_count": len(pairs)}
    xs = [pair[0] for pair in pairs]
    if max(xs) - min(xs) < float(config["minimum_fast_span_m"]):
        return {"status": "UNKNOWN_INSUFFICIENT_FAST_SPAN", "pair_count": len(pairs)}
    minimum_delta = float(config["minimum_pair_delta_m"])
    slopes = []
    for left in range(len(pairs)):
        for right in range(left + 1, len(pairs)):
            delta = pairs[right][0] - pairs[left][0]
            if abs(delta) >= minimum_delta:
                slopes.append((pairs[right][1] - pairs[left][1]) / delta)
    if not slopes:
        return {"status": "UNKNOWN_NO_STABLE_SLOPE", "pair_count": len(pairs)}
    slope = float(statistics.median(slopes))
    intercept = float(statistics.median(y - slope * x for x, y in pairs))
    residuals = [abs(slope * x + intercept - y) for x, y in pairs]
    median_residual = float(statistics.median(residuals))
    lower, upper = (float(value) for value in config["slope_bounds"])
    if not lower <= slope <= upper:
        return {
            "status": "UNKNOWN_SLOPE_OUT_OF_BOUNDS",
            "pair_count": len(pairs),
            "slope": slope,
            "intercept_m": intercept,
            "median_absolute_residual_m": median_residual,
        }
    if median_residual > float(config["maximum_fit_median_absolute_residual_m"]):
        return {
            "status": "UNKNOWN_FIT_RESIDUAL",
            "pair_count": len(pairs),
            "slope": slope,
            "intercept_m": intercept,
            "median_absolute_residual_m": median_residual,
        }
    return {
        "status": "VALID",
        "pair_count": len(pairs),
        "slope": slope,
        "intercept_m": intercept,
        "median_absolute_residual_m": median_residual,
    }


def unknown_field(reason: str) -> dict[str, Any]:
    return {"status": reason}


def corrected_field(
    fast_field: dict[str, Any], fit: dict[str, Any], age_s: float
) -> dict[str, Any]:
    if fast_field.get("status") != "VALID":
        return unknown_field("UNKNOWN_FAST_FIELD")
    if fit.get("status") != "VALID":
        return unknown_field(str(fit.get("status", "UNKNOWN_FIT")))
    slope = float(fit["slope"])
    intercept = float(fit["intercept_m"])
    output = copy.deepcopy(fast_field)
    for band in BANDS:
        clearance = output.get("bands", {}).get(band, {}).get("clearance_m")
        if clearance is None or not math.isfinite(float(clearance)):
            return unknown_field("UNKNOWN_FAST_BAND")
        corrected = max(0.0, slope * float(clearance) + intercept)
        output["bands"][band]["clearance_m"] = corrected
        output["bands"][band]["occupied_by_horizon"] = {
            str(horizon): corrected <= horizon for horizon in HORIZONS_M
        }
    height = output.get("camera_height_m")
    if height is None or not math.isfinite(float(height)):
        return unknown_field("UNKNOWN_FAST_CAMERA_HEIGHT")
    output["camera_height_m"] = max(0.0, slope * float(height) + intercept)
    output["async_calibration"] = {
        "slope": slope,
        "intercept_m": intercept,
        "anchor_source_age_s": age_s,
        "fit_pair_count": int(fit["pair_count"]),
        "fit_median_absolute_residual_m": float(fit["median_absolute_residual_m"]),
    }
    return output


def replay_d_arm(
    metric_report: dict[str, Any],
    fast_report: dict[str, Any],
    config: dict[str, Any],
    service_time_ms: float,
) -> dict[str, Any]:
    metric_frames = sorted_frames(metric_report)
    fast_frames = sorted_frames(fast_report)
    if [frame_key(row) for row in metric_frames] != [frame_key(row) for row in fast_frames]:
        raise ValueError("metric and fast reports do not contain identical frames")
    for metric, fast in zip(metric_frames, fast_frames, strict=True):
        if metric["sensor"] != fast["sensor"]:
            raise ValueError(f"sensor mismatch at {frame_key(metric)}")

    output_rows: list[dict[str, Any]] = []
    anchor_receipts: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    age_values: list[float] = []
    causality_violations = 0
    period = int(config["anchor_request_period_frames"])
    history_count = int(config["anchor_history_count"])
    max_age = float(config["maximum_anchor_source_age_s"])
    service_time_s = service_time_ms / 1000.0

    sequences: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for metric, fast in zip(metric_frames, fast_frames, strict=True):
        sequences.setdefault(str(metric["sequence_id"]), []).append((metric, fast))

    for sequence, pairs in sequences.items():
        completed: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for position, (metric, fast) in enumerate(pairs):
            now = float(metric["timestamp"])
            if active is not None and float(active["completion_timestamp"]) <= now:
                completed.append(active)
                anchor_receipts.append(copy.deepcopy(active))
                active = None
            if position % period == 0 and active is None:
                active = {
                    "sequence_id": sequence,
                    "source_position": position,
                    "source_timestamp": now,
                    "completion_timestamp": now + service_time_s,
                    "pairs": valid_pairs(metric["candidate"], fast["candidate"]),
                }

            candidate: dict[str, Any]
            latest = completed[-1] if completed else None
            fit: dict[str, Any]
            age_s: float | None = None
            if latest is None:
                fit = {"status": "UNKNOWN_ANCHOR_STARTUP"}
                candidate = unknown_field(fit["status"])
            else:
                age_s = now - float(latest["source_timestamp"])
                if float(latest["completion_timestamp"]) > now:
                    causality_violations += 1
                if age_s > max_age:
                    fit = {"status": "UNKNOWN_STALE_ANCHOR"}
                    candidate = unknown_field(fit["status"])
                else:
                    fit = robust_affine_fit(completed[-history_count:], config)
                    candidate = corrected_field(fast["candidate"], fit, age_s)
            reason = str(candidate.get("status", "UNKNOWN"))
            if reason != "VALID":
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                assert age_s is not None
                age_values.append(age_s)
            output_rows.append(
                {
                    "sequence_root": metric.get("sequence_root"),
                    "sequence_id": sequence,
                    "timestamp": now,
                    "frame_path": metric["frame_path"],
                    "latency_ms": float(fast["latency_ms"]),
                    "sensor": copy.deepcopy(metric["sensor"]),
                    "candidate": candidate,
                    "latest_anchor_source_timestamp": (
                        float(latest["source_timestamp"]) if latest is not None else None
                    ),
                    "latest_anchor_completion_timestamp": (
                        float(latest["completion_timestamp"]) if latest is not None else None
                    ),
                    "anchor_source_age_s": age_s,
                    "fit_status": fit["status"],
                }
            )
        if active is not None:
            anchor_receipts.append(copy.deepcopy(active))

    task = summarize(output_rows)
    known_count = sum(row["candidate"].get("status") == "VALID" for row in output_rows)
    known_fraction = known_count / len(output_rows)
    age_p95 = percentile(age_values, 0.95)
    system_gates = {
        "known_output_fraction": known_fraction >= 0.90,
        "anchor_source_age_p95": age_p95 is not None and age_p95 <= max_age,
        "causal_completion_only": causality_violations == 0,
    }
    trace = {
        "rows": output_rows,
        "anchor_receipts": anchor_receipts,
    }
    summary = {
        key: value for key, value in task.items() if key != "frames"
    }
    # This replay does not measure a co-resident process or allocator peak.
    # The shared A0 summarizer uses 0.0 when row-level fields are absent; R1
    # must preserve absence as NOT_EVALUABLE instead of a zero-memory claim.
    summary["process_rss_peak_mib"] = None
    summary["cuda_peak_allocated_mib"] = None
    summary.update(
        {
            "known_output_frames": known_count,
            "known_output_fraction": known_fraction,
            "anchor_source_age_p95_s": age_p95,
            "unknown_reason_counts": reason_counts,
            "completed_anchor_receipts": sum(
                receipt["completion_timestamp"]
                <= max(row["timestamp"] for row in output_rows if row["sequence_id"] == receipt["sequence_id"])
                for receipt in anchor_receipts
            ),
            "causality_violations": causality_violations,
            "system_gates": system_gates,
            "all_development_gates_pass": all(task["gates"].values())
            and all(system_gates.values()),
        }
    )
    return {"summary": summary, "trace": trace}


def busy_intervals(
    frames: list[dict[str, Any]], period: int, service_time_s: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    busy_until = -math.inf
    for position, frame in enumerate(frames):
        now = float(frame["timestamp"])
        if position % period == 0 and now >= busy_until:
            busy_until = now + service_time_s
            intervals.append((now, busy_until))
    return intervals


def phone_resource_audit(
    fast_report: dict[str, Any], config: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    period = int(config["anchor_request_period_frames"])
    service_s = float(profile["metric_service_time_ms"]) / 1000.0
    sequences: dict[str, list[dict[str, Any]]] = {}
    for row in sorted_frames(fast_report):
        sequences.setdefault(str(row["sequence_id"]), []).append(row)
    total_frames = 0
    interrupted_frames = 0
    busy_duration = 0.0
    observed_duration = 0.0
    completion_ages = []
    interval_count = 0
    for frames in sequences.values():
        intervals = busy_intervals(frames, period, service_s)
        interval_count += len(intervals)
        timestamps = [float(row["timestamp"]) for row in frames]
        step = statistics.median(np.diff(timestamps)) if len(timestamps) > 1 else 0.0
        start, end = timestamps[0], timestamps[-1] + float(step)
        observed_duration += end - start
        for left, right in intervals:
            busy_duration += max(0.0, min(right, end) - max(left, start))
            if right <= end:
                completion_ages.append(right - left)
        for timestamp in timestamps:
            total_frames += 1
            if any(left <= timestamp < right for left, right in intervals):
                interrupted_frames += 1
    interruption_fraction = interrupted_frames / total_frames
    busy_fraction = busy_duration / observed_duration
    max_age = float(config["maximum_anchor_source_age_s"])
    gates = {
        "da_interruption_fraction_at_most_0_05": interruption_fraction <= 0.05,
        "anchor_fresh_on_completion": bool(completion_ages)
        and percentile(completion_ages, 0.95) <= max_age,
    }
    return {
        "quality_comparison_authorized": False,
        "reason": profile["reason"],
        "metric_service_time_ms": float(profile["metric_service_time_ms"]),
        "fast_service_time_ms": float(profile["fast_service_time_ms"]),
        "shared_accelerator": True,
        "scheduled_metric_jobs": interval_count,
        "interrupted_fast_frames": interrupted_frames,
        "total_fast_frames": total_frames,
        "da_interruption_fraction": interruption_fraction,
        "metric_busy_time_fraction": busy_fraction,
        "anchor_source_age_at_completion_p95_s": percentile(completion_ages, 0.95),
        "gates": gates,
        "supported": all(gates.values()),
    }


def compact_arm(report: dict[str, Any]) -> dict[str, Any]:
    derived = summarize(report["frames"])
    keys = (
        "status",
        "unique_frames",
        "paired_valid_frames",
        "paired_valid_fraction",
        "clearance_mae_m",
        "collision_agreement",
        "false_clear_rate",
        "temporal_clearance_delta_mae_m",
        "latency_mean_ms",
        "latency_median_ms",
        "steady_latency_mean_ms",
        "steady_latency_median_ms",
        "steady_latency_p95_ms",
        "process_rss_peak_mib",
        "cuda_peak_allocated_mib",
        "gates",
    )
    return {
        key: report.get(key) if report.get(key) is not None else derived.get(key)
        for key in keys
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "docs/research/hftf/DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_PROTOCOL_2026-08-03.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    reports = validate_inputs(protocol)
    windows = protocol["execution_profiles"]["WINDOWS_CUDA_INDEPENDENT"]
    d = replay_d_arm(
        reports["metric3d_report"],
        reports["fast_report"],
        protocol["d_arm"],
        float(windows["metric_service_time_ms"]),
    )
    phone = phone_resource_audit(
        reports["fast_report"],
        protocol["d_arm"],
        protocol["execution_profiles"]["PHONE_SM8650_SHARED_HTP"],
    )
    task_supported = bool(d["summary"]["all_development_gates_pass"])
    terminals = [
        (
            "R1_DEVELOPMENT_TASK_AND_CAUSALITY_SUPPORTED_FRESH_DEVICE_PENDING"
            if task_supported
            else "R1_DEVELOPMENT_TASK_GATES_NOT_SUPPORTED"
        ),
        (
            "R1_PHONE_SHARED_HTP_FEASIBILITY_SUPPORTED"
            if phone["supported"]
            else "R1_PHONE_SHARED_HTP_FEASIBILITY_NOT_SUPPORTED"
        ),
        "R1_FULL_PROMOTION_NOT_EVALUABLE",
    ]
    result = {
        "schema": SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "date": protocol["date"],
        "data_role": protocol["authority"]["data_role"],
        "fresh_data_opened": False,
        "outcome_dependent_search_performed": False,
        "arms": {
            "A_DAV2_ONLY": compact_arm(reports["fast_report"]),
            "B_METRIC3D_ONLY": compact_arm(reports["metric3d_report"]),
            "C_PERIOD5_OFFSET": compact_arm(reports["arm_c_report"]),
            "D_ASYNC_ROBUST_AFFINE_WINDOWS_CUDA": d["summary"],
        },
        "phone_shared_htp_resource_audit": phone,
        "required_but_not_evaluated": {
            "co_resident_peak_memory": "NOT_EVALUABLE_NOT_MEASURED",
            "sustained_temperature": "NOT_EVALUABLE_NOT_MEASURED",
            "fresh_session_disjoint_replication": "NOT_EVALUABLE_NOT_OPENED",
            "final_external_camera": "NOT_EVALUABLE_NOT_CAPTURED",
        },
        "terminals": terminals,
        "decision": (
            "KEEP_D_AS_FRESH_AND_DEVICE_CANDIDATE_HOLD_TOF_DECISION"
            if task_supported
            else "D_NOT_SUPPORTED_ON_CONSUMED_DEVELOPMENT_HOLD_TOF_DECISION"
        ),
        "claim_ceiling": (
            "Consumed TUM development replay and bounded resource simulation only; "
            "no fresh transfer, final-camera, co-resident memory, thermal, alert, "
            "safety, App, ToF replacement, or purchasing authority."
        ),
    }
    write_new(args.output, result)
    if args.trace_output is not None:
        write_new(args.trace_output, d["trace"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
