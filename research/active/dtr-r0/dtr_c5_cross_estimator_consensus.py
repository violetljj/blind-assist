"""Bridge global M1 confidence gaps with cross-estimator motion consensus.

This is a post-C4 Development mechanism, not a fresh confirmation.  A route
entry may originate from M1-PDC only when its current position and velocity are
independently supported by an M1-CT cell under the already frozen M1 confidence
scales.  M1-CT route entries remain admitted directly.  No detector box,
identity, future label, or route-threshold change is used.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research_backend import (  # noqa: E402
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)

from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import (  # noqa: E402
    _ledger_paths,
    _load_arm_ledger,
    _prediction_frames,
    _sequence_names,
)
from dtr_m1_confident_direct_velocity import (  # noqa: E402
    POSITION_SIGMA_M,
    SEARCH_RADIUS_M,
    VELOCITY_SIGMA_MPS,
)
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
    _entry_s,
)


SCHEMA = "blindassist-dtr-c5-cross-estimator-consensus-development-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c5-sealed-cross-estimator-prediction-v1"
STATUS = "DTR_C5_CROSS_ESTIMATOR_CONSENSUS_DEVELOPMENT_MEASURED"
ARM = "M1_XC_GLOBAL"
GPU_PAIR_BATCH = 8
PROBE_POINTS = 1024
VELOCITY_RADIUS_MPS = VELOCITY_SIGMA_MPS * math.sqrt(2.0 * math.log(2.0))


Pair = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _arrays(ledger: Any, frame: int) -> tuple[np.ndarray, np.ndarray]:
    forward, left, vf, vl, _component = ledger.frame_cells(frame)
    position = np.column_stack([forward, left]).astype(np.float64, copy=False)
    velocity = np.column_stack([vf, vl]).astype(np.float64, copy=False)
    return position, velocity


def _cpu_match_many(pairs: Sequence[Pair]) -> list[np.ndarray]:
    from scipy.spatial import cKDTree

    output = []
    for anchor_position, anchor_velocity, candidate_position, candidate_velocity in pairs:
        if not len(anchor_position) or not len(candidate_position):
            output.append(np.zeros(len(candidate_position), dtype=bool))
            continue
        distance, nearest = cKDTree(anchor_position).query(candidate_position, k=1, workers=1)
        velocity_error = np.linalg.norm(anchor_velocity[nearest] - candidate_velocity, axis=1)
        output.append(
            (distance <= SEARCH_RADIUS_M + 1e-9)
            & (velocity_error <= VELOCITY_RADIUS_MPS + 1e-9)
        )
    return output


def _gpu_match_many(pairs: Sequence[Pair], *, numpy_output: bool) -> Any:
    import torch

    if not pairs:
        return []
    lengths = [len(pair[2]) for pair in pairs]
    anchor_max = max(1, max(len(pair[0]) for pair in pairs))
    candidate_max = max(1, max(lengths))
    anchor_position = torch.full(
        (len(pairs), anchor_max, 2), 1e6, dtype=torch.float64, device="cuda"
    )
    anchor_velocity = torch.zeros(
        (len(pairs), anchor_max, 2), dtype=torch.float64, device="cuda"
    )
    candidate_position = torch.zeros(
        (len(pairs), candidate_max, 2), dtype=torch.float64, device="cuda"
    )
    candidate_velocity = torch.zeros(
        (len(pairs), candidate_max, 2), dtype=torch.float64, device="cuda"
    )
    anchor_valid = torch.zeros((len(pairs), anchor_max), dtype=torch.bool, device="cuda")
    for index, pair in enumerate(pairs):
        ap, av, cp, cv = pair
        if len(ap):
            anchor_position[index, : len(ap)] = torch.as_tensor(ap, dtype=torch.float64, device="cuda")
            anchor_velocity[index, : len(av)] = torch.as_tensor(av, dtype=torch.float64, device="cuda")
            anchor_valid[index, : len(ap)] = True
        if len(cp):
            candidate_position[index, : len(cp)] = torch.as_tensor(cp, dtype=torch.float64, device="cuda")
            candidate_velocity[index, : len(cv)] = torch.as_tensor(cv, dtype=torch.float64, device="cuda")
    distance = torch.cdist(candidate_position, anchor_position)
    distance = torch.where(anchor_valid[:, None, :], distance, torch.full_like(distance, float("inf")))
    nearest_distance, nearest = torch.min(distance, dim=2)
    nearest_velocity = torch.gather(
        anchor_velocity,
        1,
        nearest[:, :, None].expand(-1, -1, 2),
    )
    velocity_error = torch.linalg.vector_norm(nearest_velocity - candidate_velocity, dim=2)
    supported = (nearest_distance <= SEARCH_RADIUS_M + 1e-9) & (
        velocity_error <= VELOCITY_RADIUS_MPS + 1e-9
    )
    if not numpy_output:
        return supported
    return [
        supported[index, :count].cpu().numpy().astype(bool)
        for index, count in enumerate(lengths)
    ]


def _select_backend(probe_pairs: Sequence[Pair], receipt_path: Path) -> dict[str, Any]:
    require(bool(probe_pairs), "m1_xc_probe_pairs_missing")
    bounded = [
        (ap[:PROBE_POINTS], av[:PROBE_POINTS], cp[:PROBE_POINTS], cv[:PROBE_POINTS])
        for ap, av, cp, cv in probe_pairs[:GPU_PAIR_BATCH]
    ]
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _cpu_match_many(bounded)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _gpu_match_many(bounded, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu_numpy = _gpu_match_many(bounded, numpy_output=True)
            for cpu, gpu in zip(cache["cpu"], gpu_numpy):
                if not np.array_equal(cpu, gpu):
                    raise BackendSelectionError("M1_XC_CPU_GPU_SUPPORT_MISMATCH")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "scipy-cKDTree-cross-consensus",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", "scipy-cKDTree"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-cross-consensus",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def _match_all(pairs: Sequence[Pair], backend: str) -> list[np.ndarray]:
    output = []
    for start in range(0, len(pairs), GPU_PAIR_BATCH):
        batch = pairs[start : start + GPU_PAIR_BATCH]
        if backend == "torch-cuda-cross-consensus":
            output.extend(_gpu_match_many(batch, numpy_output=True))
        else:
            output.extend(_cpu_match_many(batch))
    return output


def _predict(
    *,
    ct: Any,
    pdc: Any,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    supported: Sequence[np.ndarray],
) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    origin_s = float(timestamps[int(frames[0])])
    raw_frames = []
    active_frames = []
    urgent_frames = []
    minimum_entry_s_by_frame = {}
    risky_cells_by_frame = {}
    direct_ct_risk_cells = 0
    supported_pdc_risk_cells = 0
    supported_pdc_cells = 0

    for index, frame in enumerate(frames):
        ct_position, ct_velocity = _arrays(ct, int(frame))
        pdc_position, pdc_velocity = _arrays(pdc, int(frame))
        admitted_pdc = supported[index]
        require(len(admitted_pdc) == len(pdc_position), f"m1_xc_match_length:{frame}")
        supported_pdc_cells += int(admitted_pdc.sum())
        entries = []
        for position, velocity in zip(ct_position, ct_velocity):
            entry = _entry_s(float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1]))
            if entry is not None:
                entries.append(float(entry))
                direct_ct_risk_cells += 1
        for position, velocity, admitted in zip(pdc_position, pdc_velocity, admitted_pdc):
            if not admitted:
                continue
            entry = _entry_s(float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1]))
            if entry is not None:
                entries.append(float(entry))
                supported_pdc_risk_cells += 1
        minimum_entry_s = min(entries) if entries else None
        raw = minimum_entry_s is not None
        urgent = bool(raw and minimum_entry_s <= guard_boundary_s + 1e-9)
        signal = lifecycle.update(
            float(timestamps[int(frame)]) - origin_s,
            raw,
            urgent=urgent,
        )
        if raw:
            raw_frames.append(int(frame))
            minimum_entry_s_by_frame[str(int(frame))] = float(minimum_entry_s)
            risky_cells_by_frame[str(int(frame))] = len(entries)
        if urgent:
            urgent_frames.append(int(frame))
        if signal in ACTIVE_SIGNALS:
            active_frames.append(int(frame))
    return {
        "raw_alert_frames": raw_frames,
        "active_alert_frames": active_frames,
        "urgent_frames": urgent_frames,
        "minimum_entry_s_by_frame": minimum_entry_s_by_frame,
        "risky_cells_by_frame": risky_cells_by_frame,
        "diagnostics": {
            "frames": len(frames),
            "frames_with_route_entry": len(raw_frames),
            "active_alert_frames": len(active_frames),
            "direct_m1_ct_route_entry_cells": direct_ct_risk_cells,
            "cross_supported_m1_pdc_cells": supported_pdc_cells,
            "cross_supported_m1_pdc_route_entry_cells": supported_pdc_risk_cells,
        },
    }


def seal_predictions(args: argparse.Namespace) -> dict[str, Any]:
    timestamps_path = args.timestamps.resolve(strict=True)
    c2_root = args.c2_ledger_root.resolve(strict=True)
    c3_root = args.c3_ledger_root.resolve(strict=True)
    sequences = _sequence_names(c2_root, c3_root)
    loaded = []
    probe_pairs: list[Pair] = []
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sequences:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            paths = _ledger_paths(c2_root, c3_root, sequence)
            ct = _load_arm_ledger(
                "M1_CT_GLOBAL", paths["M1_CT_GLOBAL"].resolve(strict=True), sequence=sequence, frames=frames
            )
            pdc = _load_arm_ledger(
                "M1_PDC_GLOBAL", paths["M1_PDC_GLOBAL"].resolve(strict=True), sequence=sequence, frames=frames
            )
            pairs = []
            for frame in frames:
                ct_position, ct_velocity = _arrays(ct, int(frame))
                pdc_position, pdc_velocity = _arrays(pdc, int(frame))
                pair = (ct_position, ct_velocity, pdc_position, pdc_velocity)
                pairs.append(pair)
                if len(ct_position) and len(pdc_position) and len(probe_pairs) < GPU_PAIR_BATCH:
                    probe_pairs.append(pair)
            loaded.append((sequence, timestamps, frames, ct, pdc, pairs, paths))

    backend_receipt = args.backend_receipt.resolve()
    selection = _select_backend(probe_pairs, backend_receipt)
    selected_backend = str(selection["selected_backend"])
    prediction_rows = []
    for sequence, timestamps, frames, ct, pdc, pairs, paths in loaded:
        supported = _match_all(pairs, selected_backend)
        arm_prediction = _predict(
            ct=ct,
            pdc=pdc,
            frames=frames,
            timestamps=timestamps,
            supported=supported,
        )
        prediction_rows.append(
            {
                "sequence": sequence,
                "frames": len(frames),
                "arm": arm_prediction,
                "sources": {
                    name: {
                        "ledger": str(paths[name].resolve(strict=True)),
                        "ledger_sha256": sha256_file(paths[name].resolve(strict=True)),
                    }
                    for name in ("M1_CT_GLOBAL", "M1_PDC_GLOBAL")
                },
            }
        )
        print(
            json.dumps(
                {
                    "c5_truth_blind_sequence": sequence,
                    "raw_alert_frames": len(arm_prediction["raw_alert_frames"]),
                }
            ),
            flush=True,
        )
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "development_after_c4": True,
        "prediction_boundary": (
            "timestamps plus sealed M1-CT/M1-PDC ledgers only; no roster, native boxes, detector cases, future labels, or C4 scores"
        ),
        "mechanism": {
            "direct_source": "M1_CT_GLOBAL",
            "bridge_source": "M1_PDC_GLOBAL",
            "position_radius_m": SEARCH_RADIUS_M,
            "position_sigma_m": POSITION_SIGMA_M,
            "velocity_radius_mps": VELOCITY_RADIUS_MPS,
            "velocity_sigma_mps": VELOCITY_SIGMA_MPS,
            "confidence_derivation": "same 0.5 Gaussian half-confidence boundary as frozen M1-CT",
            "route_thresholds_lifecycle_and_source_ledgers": "UNCHANGED",
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
        "sequences": prediction_rows,
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
            prediction_row = by_sequence[sequence]
            score = score_sequence(
                sequence=sequence,
                timeline=timeline,
                prediction_frames=_prediction_frames(frames, prediction_row["arm"]),
            )
            rows.append(
                {
                    "sequence": sequence,
                    "score": score,
                    "prediction_diagnostics": prediction_row["arm"]["diagnostics"],
                }
            )
    aggregate = aggregate_scores([row["score"] for row in rows])
    c4_result_path = args.c4_result.resolve(strict=True)
    c4_result = json.loads(c4_result_path.read_text(encoding="utf-8"))
    require(c4_result.get("schema") == "blindassist-dtr-c4-detector-independent-global-risk-v1", "c4_schema_drift")
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": (
            "Can independent same-frame position/velocity consensus bridge M1-CT global-risk gaps without reopening raw motion broadly?"
        ),
        "aggregate": {ARM: aggregate},
        "comparators": {
            arm: c4_result["aggregate"][arm]
            for arm in ("M1_CT_GLOBAL", "M1_PDC_GLOBAL", "R7_P_GLOBAL")
        },
        "per_sequence": rows,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "backend_receipt": prediction["backend"]["receipt"],
            "backend_receipt_sha256": prediction["backend"]["receipt_sha256"],
            "c4_result": str(c4_result_path),
            "c4_result_sha256": sha256_file(c4_result_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "algorithm_increment": (
            "M1-CT remains the high-confidence alert origin; M1-PDC can add a route-entry cell only when an independent current M1-CT cell agrees in both position and velocity at the frozen half-confidence boundary."
        ),
        "claim_limits": [
            "C5 was designed after opening C4 results on the same seven sequences, so its score is Development evidence only and not fresh confirmation.",
            "The cross-estimator match is identity-free and truth-blind, but both sources ultimately consume the same raw LiDAR and are not statistically independent sensors.",
            "This is constant-velocity global path-conflict replay, not multimodal trajectory forecasting, product, or safety evidence.",
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
    c5 = repo / "artifacts.local" / "evidence" / "dtr-c5" / "cross-estimator-consensus"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json")
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--c2-ledger-root", type=Path, default=c2 / "ledgers")
    parser.add_argument("--c3-ledger-root", type=Path, default=c3 / "ledgers")
    parser.add_argument("--c4-result", type=Path, default=c4 / "result.json")
    parser.add_argument("--backend-receipt", type=Path, default=c5 / "backend.json")
    parser.add_argument("--predictions", type=Path, default=c5 / "predictions.json")
    parser.add_argument("--output", type=Path, default=c5 / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal_predictions(args)
    result = score_predictions(args)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
