#!/usr/bin/env python3
"""Consumed single-sequence diagnostic for the FRESH-TF freshness mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "blindassist_fresh_tf_r0_consumed_diagnostic_result_v1"
PROTOCOL_SCHEMA = "blindassist_fresh_tf_r0_consumed_diagnostic_protocol_v1"
UNKNOWN = "UNKNOWN_FRESHNESS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("sidecar contains no rows")
    timestamps = [int(row["timestamp_ns"]) for row in rows]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("timestamps must be strictly increasing")
    return rows


def rgb_feature(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read RGB frame: {path}")
    return cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def scene_change(current: np.ndarray, anchor: np.ndarray) -> float:
    if current.shape != (48, 64) or anchor.shape != (48, 64):
        raise ValueError("RGB features must be 48x64")
    return float(np.mean(np.abs(current - anchor)))


def field_states(row: dict[str, Any]) -> dict[tuple[float, int], str]:
    field = row["metric_traversability_field"]
    if field.get("status") != "VALID":
        return {}
    states: dict[tuple[float, int], str] = {}
    for envelope in field["sweep_envelopes"]:
        horizon = float(envelope["horizon_m"])
        for item in envelope["directions"]:
            states[(horizon, int(item["theta_deg"]))] = str(item["state"])
    return states


def arm_state(
    arm: str,
    anchor_state: str,
    *,
    age_s: float,
    rgb_change: float,
    policy: dict[str, Any],
) -> tuple[str, float]:
    if anchor_state not in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED"}:
        return UNKNOWN, 0.0
    if arm == "fixed_2hz_zero_order_hold":
        return anchor_state, 1.0
    if age_s > float(policy["hard_ttl_s"]):
        return UNKNOWN, 0.0
    if arm == "fixed_2hz_ttl_750ms":
        return anchor_state, 1.0
    threshold = float(policy["quality_threshold"])
    if arm == "uniform_age_freshness":
        quality = math.exp(-age_s / float(policy["uniform_tau_s"]))
    elif arm == "selective_rgb_change_freshness":
        tau = float(policy["selective_tau_s"][anchor_state])
        quality = math.exp(-age_s / tau) * math.exp(
            -rgb_change / float(policy["rgb_change"]["scale"])
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    return (anchor_state if quality >= threshold else UNKNOWN), quality


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cells)
    known = [cell for cell in cells if cell["prediction"] != UNKNOWN]
    predicted_clear = [cell for cell in known if cell["prediction"] == "CLEAR_OBSERVED"]
    predicted_blocked = [cell for cell in known if cell["prediction"] == "OCCUPIED_OBSERVED"]
    false_clear = [cell for cell in predicted_clear if cell["truth"] == "OCCUPIED_OBSERVED"]
    false_blocked = [cell for cell in predicted_blocked if cell["truth"] == "CLEAR_OBSERVED"]
    correct = [cell for cell in known if cell["prediction"] == cell["truth"]]
    return {
        "total_cells": total,
        "known_cells": len(known),
        "known_coverage": len(known) / total if total else 0.0,
        "known_accuracy": len(correct) / len(known) if known else 0.0,
        "predicted_clear_cells": len(predicted_clear),
        "predicted_blocked_cells": len(predicted_blocked),
        "false_clear_count": len(false_clear),
        "false_clear_rate_among_predicted_clear": (
            len(false_clear) / len(predicted_clear) if predicted_clear else 0.0
        ),
        "false_blocked_count": len(false_blocked),
        "unknown_cells": total - len(known),
        "mean_quality": float(np.mean([cell["quality"] for cell in cells])) if cells else 0.0,
    }


def evaluate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected protocol schema")
    config = protocol["evaluation"]
    cadence_s = float(config["precise_depth_cadence_s"])
    arms = list(config["arms"])
    features = [rgb_feature(Path(row["rgb_path"])) for row in rows]
    outputs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    anchor_index = 0
    anchor_time_s = int(rows[0]["timestamp_ns"]) / 1e9
    transitions = 0
    trace = []
    for index, row in enumerate(rows):
        timestamp_s = int(row["timestamp_ns"]) / 1e9
        if timestamp_s - anchor_time_s >= cadence_s:
            anchor_index = index
            anchor_time_s = timestamp_s
        truth = field_states(row)
        anchor = field_states(rows[anchor_index])
        if set(truth) != set(anchor):
            raise ValueError("field cell identity changed within sequence")
        age_s = timestamp_s - anchor_time_s
        change = scene_change(features[index], features[anchor_index])
        frame_trace = {
            "frame_index": int(row["frame_index"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "anchor_frame_index": int(rows[anchor_index]["frame_index"]),
            "age_s": age_s,
            "rgb_change": change,
            "arms": {},
        }
        for key in sorted(truth):
            if truth[key] != anchor[key]:
                transitions += 1
            for arm in arms:
                prediction, quality = arm_state(
                    arm,
                    anchor[key],
                    age_s=age_s,
                    rgb_change=change,
                    policy=config,
                )
                outputs[arm].append(
                    {
                        "frame_index": int(row["frame_index"]),
                        "horizon_m": key[0],
                        "theta_deg": key[1],
                        "truth": truth[key],
                        "anchor": anchor[key],
                        "prediction": prediction,
                        "quality": quality,
                    }
                )
        for arm in arms:
            decisions = Counter(cell["prediction"] for cell in outputs[arm] if cell["frame_index"] == int(row["frame_index"]))
            frame_trace["arms"][arm] = dict(sorted(decisions.items()))
        trace.append(frame_trace)

    summaries = {arm: _summary(outputs[arm]) for arm in arms}
    gates = protocol["pre_frozen_continuation_gates"]
    zoh = summaries["fixed_2hz_zero_order_hold"]
    ttl = summaries["fixed_2hz_ttl_750ms"]
    uniform = summaries["uniform_age_freshness"]
    selective = summaries["selective_rgb_change_freshness"]
    unknown_to_clear_violations = sum(
        cell["anchor"] not in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED"}
        and cell["prediction"] == "CLEAR_OBSERVED"
        for cells in outputs.values()
        for cell in cells
    )
    gate_results = {
        "transition_opportunity": transitions >= int(gates["minimum_anchor_to_current_state_transitions"]),
        "false_clear_count_improves_over_zoh": selective["false_clear_count"] < zoh["false_clear_count"],
        "false_clear_rate_no_worse_than_uniform": selective["false_clear_rate_among_predicted_clear"] <= uniform["false_clear_rate_among_predicted_clear"],
        "known_coverage": selective["known_coverage"] >= float(gates["minimum_selective_known_coverage"]),
        "known_accuracy_noninferior_to_ttl": selective["known_accuracy"] + float(gates["selective_known_accuracy_noninferiority_vs_ttl_margin"]) >= ttl["known_accuracy"],
        "unknown_to_clear_violations": unknown_to_clear_violations
        <= int(gates["maximum_unknown_to_clear_violations"]),
    }
    if not gate_results["transition_opportunity"]:
        terminal = "FRESH_TF_R0_CONSUMED_DIAGNOSTIC_NOT_EVALUABLE"
    elif all(gate_results.values()):
        terminal = "FRESH_TF_R0_CONSUMED_DIAGNOSTIC_MECHANISM_SIGNAL"
    else:
        terminal = "FRESH_TF_R0_CONSUMED_DIAGNOSTIC_NOT_SUPPORTED"
    return {
        "schema": SCHEMA,
        "authority": "CONSUMED_DIAGNOSTIC_ONLY",
        "terminal": terminal,
        "parent_sequence_count": len({row["sequence_id"] for row in rows}),
        "frame_count": len(rows),
        "cell_count_per_arm": len(next(iter(outputs.values()))),
        "anchor_to_current_state_transitions": transitions,
        "arm_summaries": summaries,
        "gate_results": gate_results,
        "unknown_to_clear_violation_count": unknown_to_clear_violations,
        "trace": trace,
        "claim_ceiling": protocol["claim_ceiling"],
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    sidecar = Path(protocol["input"]["sidecar_path"])
    manifest = Path(protocol["input"]["manifest_path"])
    if sha256(sidecar) != protocol["input"]["sidecar_sha256"]:
        raise ValueError("sidecar SHA-256 mismatch")
    if sha256(manifest) != protocol["input"]["manifest_sha256"]:
        raise ValueError("manifest SHA-256 mismatch")
    result = evaluate(protocol, load_rows(sidecar))
    trace = result.pop("trace")
    result["protocol_sha256"] = sha256(args.protocol)
    result["sidecar_sha256"] = sha256(sidecar)
    result["manifest_sha256"] = sha256(manifest)
    write_new(args.output, result)
    if args.trace_output:
        write_new(args.trace_output, {"schema": SCHEMA + "_trace", "rows": trace})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
