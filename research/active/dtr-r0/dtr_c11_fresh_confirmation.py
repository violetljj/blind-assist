"""Run fixed C9 and frozen C11 on an algorithm-fresh JRDB cohort."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

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
    CLEAR,
    CONTACT,
    PROXIMITY,
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_c6_world_route_occupancy_belief import _load_sequence, _probe, _select_backend  # noqa: E402
from dtr_c9_self_sustaining_global_risk_belief import ARM as C9_ARM  # noqa: E402
from dtr_c9_self_sustaining_global_risk_belief import _predict as predict_c9  # noqa: E402
from dtr_c10_fresh_c9_confirmation import _acquired_bag, _select_point_backend  # noqa: E402
from dtr_c11_route_region_probability import (  # noqa: E402
    ARM as C11_ARM,
    SCHEMA as CALIBRATOR_SCHEMA,
    _calibration_metrics,
    _probability,
    extract_sequence,
    predict as predict_c11,
)
from dtr_m1_confident_direct_velocity import ledger_paths as confident_ledger_paths  # noqa: E402
from dtr_m1_confident_direct_velocity import materialize as materialize_confident_ledger  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    ledger_paths as flow_ledger_paths,
    load_flow_ledger,
    materialize_flow_ledger,
)


SCHEMA = "blindassist-dtr-c11-fresh-route-region-probability-confirmation-v1"
WORKER_SCHEMA = "blindassist-dtr-c11-sealed-sequence-prediction-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c11-sealed-fresh-route-region-predictions-v1"
STATUS = "DTR_C11_FRESH_ROUTE_REGION_PROBABILITY_CONFIRMATION_MEASURED"


def _model_parameters(calibrator: dict[str, Any]) -> tuple[list[float], list[float]]:
    model = calibrator["model"]
    require(float(model["decision_probability"]) == 0.5, "probability_threshold_drift")
    return (
        [float(model["onset_platt_slope"]), float(model["onset_platt_intercept"])],
        [
            float(model["maintenance_platt_slope"]),
            float(model["maintenance_platt_intercept"]),
        ],
    )


def _select_probability_backend(
    current_score: np.ndarray,
    reachable_score: np.ndarray,
    onset: list[float],
    maintenance: list[float],
    receipt: Path,
) -> dict[str, Any]:
    values = np.concatenate([current_score, reachable_score]).astype(np.float64)
    split = len(current_score)
    cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cache["cpu"] = np.concatenate(
            [
                _probability(values[:split], onset, "numpy-platt-inference"),
                _probability(values[split:], maintenance, "numpy-platt-inference"),
            ]
        )
        return cache["cpu"]

    def gpu_probe() -> Any:
        import torch

        tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
        output = torch.empty_like(tensor)
        output[:split] = torch.sigmoid(float(onset[0]) * tensor[:split] + float(onset[1]))
        output[split:] = torch.sigmoid(
            float(maintenance[0]) * tensor[split:] + float(maintenance[1])
        )
        return output

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = output.detach().cpu().numpy()
            if not np.allclose(cache["cpu"], gpu, atol=1e-12, rtol=1e-12):
                raise BackendSelectionError("C11_PROBABILITY_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-platt-inference",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-platt-inference",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def worker(args: argparse.Namespace) -> dict[str, Any]:
    sequence = str(args.sequence)
    acquisition_path = args.acquisition.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    calibrator_path = args.calibrator.resolve(strict=True)
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    require(calibrator.get("schema") == CALIBRATOR_SCHEMA, "calibrator_schema_drift")
    onset, maintenance = _model_parameters(calibrator)
    bag_path, bag_row = _acquired_bag(acquisition_path, sequence)

    point_receipt = args.point_backend_receipt.resolve()
    point_selection = _select_point_backend(point_receipt)
    require(
        point_selection["selected_backend"] == "python-irregular-component-matching",
        "frozen_r7_cpu_implementation_does_not_match_selected_backend",
    )
    with zipfile.ZipFile(timestamps_path) as archive:
        timestamps = _load_timestamps(archive, sequence)
    frames = sorted(timestamps)
    root = args.ledger_root.resolve()
    sequence_dir = root / sequence
    r7_path, r7_manifest = flow_ledger_paths(sequence_dir / "r7.json")
    if r7_path.exists() and r7_manifest.exists():
        load_flow_ledger(
            r7_path,
            r7_manifest,
            expected_sequence=sequence,
            expected_frames=frames,
        )
    else:
        materialize_flow_ledger(
            bag_path=bag_path,
            timestamps_path=timestamps_path,
            calibration_dir=calibration_dir,
            output_path=r7_path,
            manifest_path=r7_manifest,
            sequence=sequence,
            timestamps_override=timestamps,
        )
    m1_path, m1_manifest = confident_ledger_paths(sequence_dir / "m1-ct.json")
    if not (m1_path.exists() and m1_manifest.exists()):
        materialize_confident_ledger(
            source_path=r7_path,
            source_manifest_path=r7_manifest,
            output_path=m1_path,
            manifest_path=m1_manifest,
        )
    data = _load_sequence(
        sequence=sequence,
        timestamps=timestamps,
        c2_root=root,
        c3_root=root,
    )
    probe_position, probe_velocity = _probe([data])
    route_receipt = args.route_backend_receipt.resolve()
    route_selection = _select_backend(probe_position, probe_velocity, route_receipt)
    route_backend = str(route_selection["selected_backend"])
    evidence = extract_sequence(data, route_backend)
    probability_receipt = args.probability_backend_receipt.resolve()
    probability_selection = _select_probability_backend(
        evidence.current_score,
        evidence.reachable_score,
        onset,
        maintenance,
        probability_receipt,
    )
    probability_backend = str(probability_selection["selected_backend"])
    c9_arm = predict_c9(data, route_backend)
    c11_arm = predict_c11(
        evidence,
        onset,
        maintenance,
        probability_backend=probability_backend,
    )
    result = {
        "schema": WORKER_SCHEMA,
        "truth_blind": True,
        "sequence": sequence,
        "frames": len(frames),
        "arms": {C9_ARM: c9_arm, C11_ARM: c11_arm},
        "prediction_boundary": (
            "raw bag, timestamps, calibration, and frozen calibrator only; no C11 roster, "
            "native future OBB, detector output, or prior fresh score"
        ),
        "backends": {
            "point": {
                "receipt": str(point_receipt),
                "receipt_sha256": sha256_file(point_receipt),
                "selected_backend": point_selection["selected_backend"],
                "selected_device_type": point_selection["selected_device_type"],
                "selection_reason": point_selection["selection_reason"],
            },
            "route": {
                "receipt": str(route_receipt),
                "receipt_sha256": sha256_file(route_receipt),
                "selected_backend": route_backend,
                "selected_device_type": route_selection["selected_device_type"],
                "selection_reason": route_selection["selection_reason"],
            },
            "probability": {
                "receipt": str(probability_receipt),
                "receipt_sha256": sha256_file(probability_receipt),
                "selected_backend": probability_backend,
                "selected_device_type": probability_selection["selected_device_type"],
                "selection_reason": probability_selection["selection_reason"],
            },
        },
        "sources": {
            "acquisition": str(acquisition_path),
            "acquisition_sha256": sha256_file(acquisition_path),
            "bag": str(bag_path),
            "bag_sha256": bag_row["sha256"],
            "calibrator": str(calibrator_path),
            "calibrator_sha256": sha256_file(calibrator_path),
            "r7": str(r7_path),
            "r7_sha256": sha256_file(r7_path),
            "m1_ct": str(m1_path),
            "m1_ct_sha256": sha256_file(m1_path),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for value in args.worker_predictions:
        path = value.resolve(strict=True)
        row = json.loads(path.read_text(encoding="utf-8"))
        require(row.get("schema") == WORKER_SCHEMA, f"worker_schema:{path}")
        require(row.get("truth_blind") is True, f"worker_not_truth_blind:{path}")
        rows.append(
            {
                "sequence": row["sequence"],
                "frames": row["frames"],
                "arms": row["arms"],
                "backends": row["backends"],
                "sources": {
                    **row["sources"],
                    "worker": str(path),
                    "worker_sha256": sha256_file(path),
                },
            }
        )
    names = [str(row["sequence"]) for row in rows]
    require(len(names) == len(set(names)) and bool(names), "worker_sequence_cardinality")
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": "all C9/C11 sequence predictions sealed before C11 roster and future OBB truth",
        "sequences": sorted(rows, key=lambda row: str(row["sequence"])),
    }
    write_json(args.predictions.resolve(), prediction)
    return prediction


def score(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == PREDICTION_SCHEMA, "prediction_schema_drift")
    require(prediction.get("truth_blind") is True, "prediction_not_truth_blind")

    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibrator_path = args.calibrator.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(roster["frozen_algorithm"]["calibrator_sha256"] == sha256_file(calibrator_path), "calibrator_hash_drift")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash_drift")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "timestamps_hash_drift")
    by_sequence = {str(row["sequence"]): row for row in prediction["sequences"]}
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(set(by_sequence) == set(expected), "prediction_roster_sequence_drift")
    calibrator_sha256 = sha256_file(calibrator_path)
    for sequence in expected:
        require(
            by_sequence[sequence]["sources"]["calibrator_sha256"]
            == calibrator_sha256,
            f"worker_calibrator_hash_drift:{sequence}",
        )
    scores: dict[str, list[dict[str, Any]]] = {C9_ARM: [], C11_ARM: []}
    probability_rows = []
    truth_rows = []
    per_sequence = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in expected:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            timeline = global_truth_timeline(
                frames=frames,
                timestamps=timestamps,
                boxes_by_frame=_load_boxes(labels, sequence),
            )
            row = by_sequence[sequence]
            sequence_scores = {}
            for arm in (C9_ARM, C11_ARM):
                arm_score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, row["arms"][arm]),
                )
                scores[arm].append(arm_score)
                sequence_scores[arm] = arm_score
            probabilities = row["arms"][C11_ARM]["route_region_probability_by_frame"]
            for frame, truth in zip(frames, timeline):
                if truth["label"] in {CONTACT, PROXIMITY, CLEAR}:
                    probability_rows.append(float(probabilities[str(frame)]))
                    truth_rows.append(float(truth["label"] == CONTACT))
            per_sequence.append(
                {
                    "sequence": sequence,
                    "scores": sequence_scores,
                    "diagnostics": {
                        arm: row["arms"][arm]["diagnostics"] for arm in (C9_ARM, C11_ARM)
                    },
                }
            )
    aggregate = {arm: aggregate_scores(values) for arm, values in scores.items()}
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": "Does calibrated flow-reachable route-region probability reduce false path-conflict alerts while retaining fixed-C9 event recall?",
        "aggregate": aggregate,
        "fresh_probability_calibration": _calibration_metrics(
            np.asarray(probability_rows), np.asarray(truth_rows)
        ),
        "per_sequence": per_sequence,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "calibrator": str(calibrator_path),
            "calibrator_sha256": calibrator_sha256,
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "claim_limits": [
            "C11 coefficients and its 0.5 probability decision were frozen on ten consumed sequences before these bags were acquired.",
            "Fresh calibration is public-replay calibration under JRDB train distribution, not deployment calibration.",
            "The unchanged imminent continuous-collision geometry may originate an alert below 0.5 probability.",
            "No product, user-benefit, or safety claim follows.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c11 = repo / "artifacts.local" / "evidence" / "dtr-c11" / "fresh-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "merge", "score"))
    parser.add_argument("--sequence")
    parser.add_argument("--worker-predictions", type=Path, nargs="*")
    parser.add_argument(
        "--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c11_fresh_confirmation_roster.json")
    )
    parser.add_argument(
        "--calibrator", type=Path, default=Path(__file__).resolve().with_name("dtr_c11_route_region_calibrator.json")
    )
    parser.add_argument("--acquisition", type=Path, default=c11 / "acquisition.json")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=repo / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration",
    )
    parser.add_argument("--ledger-root", type=Path, default=c11 / "ledgers")
    parser.add_argument("--point-backend-receipt", type=Path)
    parser.add_argument("--route-backend-receipt", type=Path)
    parser.add_argument("--probability-backend-receipt", type=Path)
    parser.add_argument("--predictions", type=Path, default=c11 / "predictions.json")
    parser.add_argument("--output", type=Path, default=c11 / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        require(bool(args.sequence), "worker_sequence_required")
        require(args.point_backend_receipt is not None, "point_backend_receipt_required")
        require(args.route_backend_receipt is not None, "route_backend_receipt_required")
        require(args.probability_backend_receipt is not None, "probability_backend_receipt_required")
        result = worker(args)
        print(json.dumps({"status": "C11_SEQUENCE_PREDICTIONS_SEALED", "sequence": result["sequence"]}))
    elif args.mode == "merge":
        require(bool(args.worker_predictions), "worker_predictions_required")
        result = merge(args)
        print(json.dumps({"status": "C11_PREDICTIONS_SEALED", "sequences": len(result["sequences"])}))
    else:
        result = score(args)
        print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "calibration": result["fresh_probability_calibration"]}))


if __name__ == "__main__":
    main()
