"""Run fixed C9 on an algorithm-fresh JRDB cohort without opening truth early.

Each worker sees one acquired raw bag plus public frame timestamps, materializes
the unchanged R7 and M1-CT ledgers, benchmarks the real execution classes, and
hash-seals a C9 prediction.  A separate merge phase combines only sealed worker
predictions.  The score phase is the first phase allowed to open the committed
C10 roster and native future OBB labels.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research_backend import (
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)

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
from dtr_c4_detector_independent_global_risk import _prediction_frames
from dtr_c6_world_route_occupancy_belief import _load_sequence, _probe, _select_backend
from dtr_c9_self_sustaining_global_risk_belief import (
    ARM,
    PREDICTION_SCHEMA as C9_PREDICTION_SCHEMA,
    SCHEMA as C9_DEVELOPMENT_SCHEMA,
    _predict,
)
from dtr_m1_confident_direct_velocity import ledger_paths as confident_ledger_paths
from dtr_m1_confident_direct_velocity import materialize as materialize_confident_ledger
from dtr_r7_occupancy_flow_canary import (
    FROZEN_FLOW_CONFIG,
    Component,
    _match_components,
    ledger_paths as flow_ledger_paths,
    load_flow_ledger,
    materialize_flow_ledger,
)


SCHEMA = "blindassist-dtr-c10-fresh-c9-confirmation-v1"
WORKER_SCHEMA = "blindassist-dtr-c10-sealed-c9-sequence-prediction-v1"
STATUS = "DTR_C10_FRESH_C9_CONFIRMATION_MEASURED"


def _representative_components() -> tuple[list[Component], list[Component]]:
    """Deterministic small irregular workload representative of R7 matching."""
    previous: list[Component] = []
    current: list[Component] = []
    for index in range(16):
        origin_x = (index % 4) * 13
        origin_y = (index // 4) * 11
        keys = frozenset(
            (origin_x + dx, origin_y + dy)
            for dx in range(6)
            for dy in range(4)
            if (dx + 2 * dy + index) % 5 != 0
        )
        shifted = frozenset((x + 1, y) for x, y in keys)
        array = np.asarray(sorted(keys), dtype=np.float64)
        moved = np.asarray(sorted(shifted), dtype=np.float64)
        previous.append(
            Component(
                keys=keys,
                centroid_x_m=float((array[:, 0] + 0.5).mean() * FROZEN_FLOW_CONFIG.voxel_size_m),
                centroid_y_m=float((array[:, 1] + 0.5).mean() * FROZEN_FLOW_CONFIG.voxel_size_m),
                extent_x_m=1.5,
                extent_y_m=1.0,
            )
        )
        current.append(
            Component(
                keys=shifted,
                centroid_x_m=float((moved[:, 0] + 0.5).mean() * FROZEN_FLOW_CONFIG.voxel_size_m),
                centroid_y_m=float((moved[:, 1] + 0.5).mean() * FROZEN_FLOW_CONFIG.voxel_size_m),
                extent_x_m=1.5,
                extent_y_m=1.0,
            )
        )
    return previous, current


def _torch_overlap_probe(previous: Sequence[Component], current: Sequence[Component]) -> Any:
    """Equivalent translated-overlap matrix for the dominant irregular pair probe."""
    import torch

    rows = []
    device = torch.device("cuda")
    voxel = FROZEN_FLOW_CONFIG.voxel_size_m
    for right in current:
        values = []
        right_keys = torch.as_tensor(sorted(right.keys), dtype=torch.int32, device=device)
        for left in previous:
            dx = right.centroid_x_m - left.centroid_x_m
            dy = right.centroid_y_m - left.centroid_y_m
            shift = torch.tensor(
                [int(round(dx / voxel)), int(round(dy / voxel))],
                dtype=torch.int32,
                device=device,
            )
            left_keys = torch.as_tensor(sorted(left.keys), dtype=torch.int32, device=device) + shift
            equal = torch.all(left_keys[:, None, :] == right_keys[None, :, :], dim=2)
            overlap = torch.any(equal, dim=1).sum(dtype=torch.float64)
            values.append(overlap / max(1, min(len(left.keys), len(right.keys))))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def _numpy_overlap_probe(previous: Sequence[Component], current: Sequence[Component]) -> np.ndarray:
    output = np.empty((len(current), len(previous)), dtype=np.float64)
    voxel = FROZEN_FLOW_CONFIG.voxel_size_m
    for current_index, right in enumerate(current):
        for previous_index, left in enumerate(previous):
            dx = right.centroid_x_m - left.centroid_x_m
            dy = right.centroid_y_m - left.centroid_y_m
            shift = (int(round(dx / voxel)), int(round(dy / voxel)))
            shifted = {(x + shift[0], y + shift[1]) for x, y in left.keys}
            output[current_index, previous_index] = len(shifted & right.keys) / max(
                1, min(len(left.keys), len(right.keys))
            )
    return output


def _select_point_backend(receipt_path: Path) -> dict[str, Any]:
    previous, current = _representative_components()
    cpu_cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cpu_cache["value"] = _numpy_overlap_probe(previous, current)
        # Exercise the frozen matcher as well as its dominant overlap primitive.
        _match_components(previous, current, 0.5, FROZEN_FLOW_CONFIG)
        return cpu_cache["value"]

    def gpu_probe() -> Any:
        return _torch_overlap_probe(previous, current)

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "value" in cpu_cache:
            gpu = output.detach().cpu().numpy()
            if not np.allclose(cpu_cache["value"], gpu, atol=1e-12, rtol=0.0):
                raise BackendSelectionError("R7_COMPONENT_OVERLAP_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "python-irregular-component-matching",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-irregular-component-overlap",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def _acquired_bag(acquisition_path: Path, sequence: str) -> tuple[Path, dict[str, Any]]:
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(
        acquisition.get("schema") == "blindassist-dtr-c2-frozen-jrdb-bag-acquisition-v1",
        "acquisition_schema_drift",
    )
    rows = [row for row in acquisition["bags"] if str(row["sequence"]) == sequence]
    require(len(rows) == 1, f"acquisition_sequence_cardinality:{sequence}")
    row = rows[0]
    bag_path = Path(row["bag"]).resolve(strict=True)
    require(sha256_file(bag_path) == row["sha256"], f"bag_hash_drift:{sequence}")
    return bag_path, row


def worker(args: argparse.Namespace) -> dict[str, Any]:
    sequence = str(args.sequence)
    acquisition_path = args.acquisition.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
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
    sequence_dir = args.ledger_root.resolve() / sequence
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

    # _load_sequence only consumes the C2 R7/M1-CT paths; the C3 path is a
    # compatibility argument inherited from the development loader.
    data = _load_sequence(
        sequence=sequence,
        timestamps=timestamps,
        c2_root=args.ledger_root.resolve(),
        c3_root=args.ledger_root.resolve(),
    )
    position, velocity = _probe([data])
    route_receipt = args.route_backend_receipt.resolve()
    route_selection = _select_backend(position, velocity, route_receipt)
    arm = _predict(data, str(route_selection["selected_backend"]))
    result = {
        "schema": WORKER_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": (
            "one raw bag plus public timestamps and calibration; no C10 roster, "
            "native boxes, future labels, future pose, detector output, or prior scores"
        ),
        "sequence": sequence,
        "frames": len(frames),
        "arm": arm,
        "backends": {
            "point_cloud_matching": {
                "receipt": str(point_receipt),
                "receipt_sha256": sha256_file(point_receipt),
                "selected_backend": point_selection["selected_backend"],
                "selected_device_type": point_selection["selected_device_type"],
                "selection_reason": point_selection["selection_reason"],
            },
            "route_collision": {
                "receipt": str(route_receipt),
                "receipt_sha256": sha256_file(route_receipt),
                "selected_backend": route_selection["selected_backend"],
                "selected_device_type": route_selection["selected_device_type"],
                "selection_reason": route_selection["selection_reason"],
            },
        },
        "sources": {
            "acquisition": str(acquisition_path),
            "acquisition_sha256": sha256_file(acquisition_path),
            "bag": str(bag_path),
            "bag_sha256": bag_row["sha256"],
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
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
                "arm": row["arm"],
                "sources": {**row["sources"], "worker": str(path), "worker_sha256": sha256_file(path)},
                "backends": row["backends"],
            }
        )
    names = [str(row["sequence"]) for row in rows]
    require(len(names) == len(set(names)), "duplicate_worker_sequence")
    require(bool(names), "no_worker_predictions")
    prediction = {
        "schema": C9_PREDICTION_SCHEMA,
        "truth_blind": True,
        "fixed_after_development": True,
        "prediction_boundary": "sealed per sequence before C10 roster or native OBB truth is opened",
        "mechanism": {
            "onset_authority": "unchanged current M1-CT global route entry",
            "maintenance_evidence": "unchanged wearer-relative world occupancy belief",
            "maintenance_authority": "unchanged already-active global risk lifecycle",
            "route_thresholds_lifecycle_source_confidence_and_grace": "UNCHANGED_C9",
        },
        "sequences": sorted(rows, key=lambda row: str(row["sequence"])),
    }
    write_json(args.predictions.resolve(), prediction)
    return prediction


def score(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == C9_PREDICTION_SCHEMA, "prediction_schema_drift")
    require(prediction.get("truth_blind") is True, "prediction_not_truth_blind")

    # Truth authority is opened only after the combined prediction already exists
    # and its hash has been captured above.
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
            timeline = global_truth_timeline(
                frames=frames,
                timestamps=timestamps,
                boxes_by_frame=_load_boxes(labels, sequence),
            )
            prediction_row = by_sequence[sequence]
            sequence_score = score_sequence(
                sequence=sequence,
                timeline=timeline,
                prediction_frames=_prediction_frames(frames, prediction_row["arm"]),
            )
            rows.append(
                {
                    "sequence": sequence,
                    "score": sequence_score,
                    "prediction_diagnostics": prediction_row["arm"]["diagnostics"],
                }
            )
    aggregate = aggregate_scores([row["score"] for row in rows])
    development_path = args.c9_development_result.resolve(strict=True)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    require(development.get("schema") == C9_DEVELOPMENT_SCHEMA, "c9_development_schema_drift")
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": "Does fixed C9 retain useful path-conflict behavior on an algorithm-fresh preferred JRDB cohort?",
        "aggregate": {ARM: aggregate},
        "development_comparator": development["aggregate"][ARM],
        "per_sequence": rows,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "c9_development_result": str(development_path),
            "c9_development_result_sha256": sha256_file(development_path),
        },
        "confirmation_contract": {
            "mechanism_changed_after_cohort_freeze": False,
            "route_thresholds_changed": False,
            "truth_opened_before_combined_prediction_seal": False,
            "algorithm_fresh_sequences": True,
        },
        "claim_limits": [
            "This confirms the fixed C9 mechanism on a metadata-selected public replay cohort; it is not a calibrated probability model.",
            "Three sequences meet the pre-existing denominator gate but do not establish natural-distribution generalization.",
            "The route remains causal constant velocity and the alert state deterministic.",
            "No product, user-benefit, or safety claim follows from this replay.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c10 = repo / "artifacts.local" / "evidence" / "dtr-c10" / "fresh-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "merge", "score"))
    parser.add_argument("--sequence")
    parser.add_argument("--worker-predictions", type=Path, nargs="*")
    parser.add_argument(
        "--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c10_fresh_confirmation_roster.json")
    )
    parser.add_argument("--acquisition", type=Path, default=c10 / "acquisition.json")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=repo
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
        / "calibration",
    )
    parser.add_argument("--ledger-root", type=Path, default=c10 / "ledgers")
    parser.add_argument("--point-backend-receipt", type=Path)
    parser.add_argument("--route-backend-receipt", type=Path)
    parser.add_argument("--predictions", type=Path, default=c10 / "predictions.json")
    parser.add_argument(
        "--c9-development-result",
        type=Path,
        default=repo
        / "artifacts.local"
        / "evidence"
        / "dtr-c9"
        / "self-sustaining-global-risk-belief"
        / "result.json",
    )
    parser.add_argument("--output", type=Path, default=c10 / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        require(bool(args.sequence), "worker_sequence_required")
        require(args.point_backend_receipt is not None, "point_backend_receipt_required")
        require(args.route_backend_receipt is not None, "route_backend_receipt_required")
        result = worker(args)
        print(json.dumps({"status": "C10_SEQUENCE_PREDICTION_SEALED", "sequence": result["sequence"]}))
    elif args.mode == "merge":
        require(bool(args.worker_predictions), "worker_predictions_required")
        result = merge(args)
        print(json.dumps({"status": "C10_PREDICTIONS_SEALED", "sequences": len(result["sequences"])}))
    else:
        result = score(args)
        print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
