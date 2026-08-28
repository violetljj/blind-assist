"""Maintain an originated global route-risk event with world occupancy belief.

Current M1-CT route entry is the only independent ONSET authority.  Once the
global risk lifecycle is active, wearer-relative world occupancy belief may
keep the same risk event alive across missing current cells.  Belief cannot
originate a warning from CLEAR, and the unchanged lifecycle grace still clears
the event after evidence disappears.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from dtr_c1_global_obb_cohort_admission import (
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence
from dtr_c4_detector_independent_global_risk import _prediction_frames, _sequence_names
from dtr_c6_world_route_occupancy_belief import (
    EPSILON,
    _current_world_cells,
    _entries,
    _load_sequence,
    _probe,
    _relative_state,
    _select_backend,
)
from dtr_c8_global_risk_belief_bridge import _current_absolute_entries
from dtr_r0 import DTRConfig
from dtr_r1 import RiskEventLifecycle
from dtr_r2 import FROZEN_R2_CONFIG
from dtr_r5_dropout_canary import ACTIVE_SIGNALS
from dtr_r7_occupancy_flow_canary import HORIZON_S, ROUTE_HALF_WIDTH_M


SCHEMA = "blindassist-dtr-c9-self-sustaining-global-risk-belief-development-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c9-sealed-self-sustaining-global-risk-belief-v1"
STATUS = "DTR_C9_SELF_SUSTAINING_GLOBAL_RISK_BELIEF_DEVELOPMENT_MEASURED"
ARM = "M1_SRB_GLOBAL"


def _predict(data: Any, backend: str) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    output = {
        "raw_alert_frames": [],
        "active_alert_frames": [],
        "urgent_frames": [],
        "minimum_entry_s_by_frame": {},
        "risky_cells_by_frame": {},
    }
    belief_position = np.empty((0, 2), dtype=np.float64)
    belief_velocity = np.empty((0, 2), dtype=np.float64)
    belief_time_s = np.empty(0, dtype=np.float64)
    belief_confidence = np.empty(0, dtype=np.float64)
    primary_risk_frames = 0
    belief_maintenance_frames = 0
    belief_only_frames = 0
    blocked_belief_origin_frames = 0
    retained_cell_evaluations = 0

    for index, frame_value in enumerate(data.frames):
        frame = int(frame_value)
        now_s = float(data.times_s[index])
        world_position, world_velocity, confidence = _current_world_cells(data, index)
        keep = now_s - belief_time_s <= config.clear_grace_s + EPSILON
        belief_position = belief_position[keep]
        belief_velocity = belief_velocity[keep]
        belief_time_s = belief_time_s[keep]
        belief_confidence = belief_confidence[keep]
        if len(world_position):
            belief_position = np.concatenate([belief_position, world_position], axis=0)
            belief_velocity = np.concatenate([belief_velocity, world_velocity], axis=0)
            belief_time_s = np.concatenate(
                [belief_time_s, np.full(len(world_position), now_s, dtype=np.float64)]
            )
            belief_confidence = np.concatenate([belief_confidence, confidence])
        require(
            np.all(belief_confidence >= 0.5 - EPSILON),
            f"belief_confidence_below_source_gate:{data.sequence}:{frame}",
        )

        primary_entries = _current_absolute_entries(data, index, backend)
        primary_finite = primary_entries[np.isfinite(primary_entries)]
        primary_risk = bool(len(primary_finite))
        primary_risk_frames += int(primary_risk)
        retained_cell_evaluations += len(belief_position)
        if len(belief_position):
            belief_local_position, belief_local_velocity = _relative_state(
                seed_position=belief_position,
                seed_velocity=belief_velocity,
                seed_time_s=belief_time_s,
                data=data,
                index=index,
            )
            belief_entries = _entries(belief_local_position, belief_local_velocity, backend)
            belief_finite = belief_entries[np.isfinite(belief_entries)]
        else:
            belief_finite = np.empty(0, dtype=np.float64)
        belief_risk = bool(len(belief_finite))
        belief_authorized = lifecycle.active
        belief_maintenance = bool(belief_authorized and belief_risk)
        belief_maintenance_frames += int(belief_maintenance)
        belief_only_frames += int(belief_maintenance and not primary_risk)
        blocked_belief_origin_frames += int(belief_risk and not belief_authorized and not primary_risk)
        admitted = (
            np.concatenate([primary_finite, belief_finite])
            if belief_maintenance
            else primary_finite
        )
        finite = admitted[np.isfinite(admitted)]
        minimum = None if not len(finite) else float(finite.min())
        raw = minimum is not None
        urgent = bool(raw and minimum <= guard_boundary_s + EPSILON)
        signal = lifecycle.update(
            now_s - float(data.times_s[0]),
            raw,
            urgent=urgent,
        )
        if raw:
            output["raw_alert_frames"].append(frame)
            output["minimum_entry_s_by_frame"][str(frame)] = minimum
            output["risky_cells_by_frame"][str(frame)] = int(len(finite))
        if urgent:
            output["urgent_frames"].append(frame)
        if signal in ACTIVE_SIGNALS:
            output["active_alert_frames"].append(frame)
    output["diagnostics"] = {
        "frames": len(data.frames),
        "primary_risk_frames": primary_risk_frames,
        "belief_maintenance_frames": belief_maintenance_frames,
        "belief_only_frames": belief_only_frames,
        "blocked_belief_origin_frames": blocked_belief_origin_frames,
        "retained_cell_evaluations": retained_cell_evaluations,
        "frames_with_admitted_route_entry": len(output["raw_alert_frames"]),
        "active_alert_frames": len(output["active_alert_frames"]),
    }
    return output


def seal_predictions(args: argparse.Namespace) -> dict[str, Any]:
    timestamps_path = args.timestamps.resolve(strict=True)
    c2_root = args.c2_ledger_root.resolve(strict=True)
    c3_root = args.c3_ledger_root.resolve(strict=True)
    sequence_names = _sequence_names(c2_root, c3_root)
    data_rows = []
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sequence_names:
            data_rows.append(
                _load_sequence(
                    sequence=sequence,
                    timestamps=_load_timestamps(timestamps_zip, sequence),
                    c2_root=c2_root,
                    c3_root=c3_root,
                )
            )
    probe_position, probe_velocity = _probe(data_rows)
    backend_receipt = args.backend_receipt.resolve()
    selection = _select_backend(probe_position, probe_velocity, backend_receipt)
    selected_backend = str(selection["selected_backend"])
    rows = []
    for data in data_rows:
        arm = _predict(data, selected_backend)
        rows.append(
            {
                "sequence": data.sequence,
                "frames": len(data.frames),
                "arm": arm,
                "sources": {
                    "m1_ct": str(data.ct_path),
                    "m1_ct_sha256": sha256_file(data.ct_path),
                    "r7_pose_source": str(data.r7_path),
                    "r7_pose_source_sha256": sha256_file(data.r7_path),
                },
            }
        )
        print(
            json.dumps(
                {
                    "c9_truth_blind_sequence": data.sequence,
                    "primary_risk_frames": arm["diagnostics"]["primary_risk_frames"],
                    "belief_only_frames": arm["diagnostics"]["belief_only_frames"],
                    "blocked_belief_origin_frames": arm["diagnostics"]["blocked_belief_origin_frames"],
                }
            ),
            flush=True,
        )
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "development_after_c8": True,
        "prediction_boundary": (
            "past/current ego poses plus sealed M1-CT motion only; no roster, detector/native boxes, future pose, future labels, or prior scores"
        ),
        "mechanism": {
            "onset_authority": "current M1-CT global route entry identical to C4",
            "maintenance_evidence": "wearer-relative world occupancy belief",
            "maintenance_authority": "global risk lifecycle already active",
            "belief_retention_s": DTRConfig().clear_grace_s,
            "clear_behavior": "unchanged RiskEventLifecycle clear grace",
            "route_thresholds_lifecycle_source_confidence_and_grace": "UNCHANGED",
        },
        "backend": {
            "receipt": str(backend_receipt),
            "receipt_sha256": sha256_file(backend_receipt),
            "selected_backend": selected_backend,
            "selected_device_type": selection["selected_device_type"],
            "selected_device_name": selection["selected_device_name"],
            "selection_reason": selection["selection_reason"],
        },
        "source": {
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "sequences": rows,
    }
    write_json(args.predictions.resolve(), prediction)
    return prediction


def score_predictions(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == PREDICTION_SCHEMA, "prediction_schema_drift")
    require(prediction.get("truth_blind") is True, "prediction_not_truth_blind")
    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash_drift")
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "timestamps_hash_drift",
    )
    by_sequence = {str(row["sequence"]): row for row in prediction["sequences"]}
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(set(by_sequence) == set(expected), "prediction_roster_sequence_drift")
    rows = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in expected:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            row = by_sequence[sequence]
            score = score_sequence(
                sequence=sequence,
                timeline=timeline,
                prediction_frames=_prediction_frames(frames, row["arm"]),
            )
            rows.append(
                {
                    "sequence": sequence,
                    "score": score,
                    "prediction_diagnostics": row["arm"]["diagnostics"],
                }
            )
    aggregate = aggregate_scores([row["score"] for row in rows])
    c4_path = args.c4_result.resolve(strict=True)
    c6_path = args.c6_result.resolve(strict=True)
    c8_path = args.c8_result.resolve(strict=True)
    c4 = json.loads(c4_path.read_text(encoding="utf-8"))
    c6 = json.loads(c6_path.read_text(encoding="utf-8"))
    c8 = json.loads(c8_path.read_text(encoding="utf-8"))
    require(c4.get("schema") == "blindassist-dtr-c4-detector-independent-global-risk-v1", "c4_schema_drift")
    require(c6.get("schema") == "blindassist-dtr-c6-world-route-occupancy-belief-development-v1", "c6_schema_drift")
    require(c8.get("schema") == "blindassist-dtr-c8-global-risk-belief-bridge-development-v1", "c8_schema_drift")
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": (
            "Can world occupancy maintain an already-originated detector-independent route-risk event without acquiring independent alert authority?"
        ),
        "aggregate": {ARM: aggregate},
        "comparators": {
            "M1_CT_GLOBAL_C4": c4["aggregate"]["M1_CT_GLOBAL"],
            "M1_WB_GLOBAL_C6": c6["aggregate"]["M1_WB_GLOBAL"],
            "M1_GRB_GLOBAL_C8": c8["aggregate"]["M1_GRB_GLOBAL"],
        },
        "per_sequence": rows,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "backend_receipt": prediction["backend"]["receipt"],
            "backend_receipt_sha256": prediction["backend"]["receipt_sha256"],
            "c4_result": str(c4_path),
            "c4_result_sha256": sha256_file(c4_path),
            "c6_result": str(c6_path),
            "c6_result_sha256": sha256_file(c6_path),
            "c8_result": str(c8_path),
            "c8_result_sha256": sha256_file(c8_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "algorithm_increment": (
            "Global route conflict is now a causal state: confidence-aware current motion owns ONSET, world occupancy belief owns only maintenance, and the unchanged lifecycle owns CLEAR."
        ),
        "dropout_composition": (
            "The sealed M1-HYBRID raw-point/R7 bounded detector-gap bridge remains unchanged, preserving the consumed 9/9 recovery by composition."
        ),
        "claim_limits": [
            "C9 was designed after C8 on the same seven sequences and is Development evidence, not fresh confirmation.",
            "A global active state may be maintained by a nearby simultaneous hazard; this is route-risk continuity, not identity tracking.",
            "The route remains causal constant velocity and the belief deterministic, not multimodal or probabilistic.",
            "This curated public replay is not product, user-benefit, or safety evidence.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c2 = repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay"
    c3 = repo / "artifacts.local" / "evidence" / "dtr-c3" / "raw-point-direct-velocity-canary"
    c4 = repo / "artifacts.local" / "evidence" / "dtr-c4" / "detector-independent-global-risk"
    c6 = repo / "artifacts.local" / "evidence" / "dtr-c6" / "world-route-occupancy-belief"
    c8 = repo / "artifacts.local" / "evidence" / "dtr-c8" / "global-risk-belief-bridge"
    c9 = repo / "artifacts.local" / "evidence" / "dtr-c9" / "self-sustaining-global-risk-belief"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json")
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--c2-ledger-root", type=Path, default=c2 / "ledgers")
    parser.add_argument("--c3-ledger-root", type=Path, default=c3 / "ledgers")
    parser.add_argument("--c4-result", type=Path, default=c4 / "result.json")
    parser.add_argument("--c6-result", type=Path, default=c6 / "result.json")
    parser.add_argument("--c8-result", type=Path, default=c8 / "result.json")
    parser.add_argument("--backend-receipt", type=Path, default=c9 / "backend.json")
    parser.add_argument("--predictions", type=Path, default=c9 / "predictions.json")
    parser.add_argument("--output", type=Path, default=c9 / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal_predictions(args)
    result = score_predictions(args)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
