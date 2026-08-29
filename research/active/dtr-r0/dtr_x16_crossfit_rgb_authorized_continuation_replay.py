"""Full replay of frozen cross-fit, RGB birth authority, and continuation.

X16 changes one information composition: the sealed sequence-held-out X10
ledger replaces X7 as the candidate source for the unchanged X13 stitched-RGB
dynamic birth authority and X14 0.50-second motion continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x3_full_lag_floxel_replay as x3  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
import dtr_x10_cross_fitted_motion_authority as x10  # noqa: E402
import dtr_x15_full_rgb_authorized_continuation_replay as x15  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x16-crossfit-rgb-authorized-continuation-replay-v1"
LEDGER_SCHEMA = "blindassist-dtr-x16-crossfit-rgb-authorized-continuation-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x16-crossfit-rgb-authorized-continuation-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x16-crossfit-rgb-authorized-continuation-freeze-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x16-crossfit-rgb-authorized-continuation-materialization-v1"
SEQUENCE_SCHEMA = "blindassist-dtr-x16-sequence-materialization-v1"


def _paths(root: Path, sequence: str | None = None) -> dict[str, Path]:
    return x15._paths(root, sequence)


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    return x15._source_paths(root, sequence)


def _baseline_rows(args: argparse.Namespace) -> dict[str, Any]:
    return x15._baseline_rows(args)


def _fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    sources = {}
    for sequence in sorted(_baseline_rows(args)):
        ledger, manifest = _source_paths(args.x10_root.resolve(strict=True), sequence)
        value = json.loads(manifest.read_text(encoding="utf-8"))
        require(value.get("schema") == x10.LEDGER_SCHEMA, f"x16_x10_schema:{sequence}")
        require(value.get("truth_blind") is True, f"x16_x10_truth:{sequence}")
        require(value.get("heldout_truth_unused") is True, f"x16_x10_holdout:{sequence}")
        require(value.get("ledger_sha256") == sha256_file(ledger), f"x16_x10_hash:{sequence}")
        x7_ledger, x7_manifest = _source_paths(args.x7_root.resolve(strict=True), sequence)
        x7_value = json.loads(x7_manifest.read_text(encoding="utf-8"))
        require(x7_value.get("schema") == x7.LEDGER_SCHEMA, f"x16_x7_schema:{sequence}")
        require(x7_value.get("ledger_sha256") == sha256_file(x7_ledger), f"x16_x7_hash:{sequence}")
        sources[sequence] = {
            "x10_ledger_sha256": sha256_file(ledger),
            "x10_manifest_sha256": sha256_file(manifest),
            "x7_manifest_sha256": sha256_file(x7_manifest),
            "bag_sha256": x7_value["source"]["bag_sha256"],
        }
    return {
        "schema": FREEZE_SCHEMA,
        "truth_blind_materialization": True,
        "algorithm_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in (__file__, x10.__file__, x15.__file__)
        ],
        "source_config": {
            "base": "SEALED_X10_SEQUENCE_HELD_OUT_CROSS_FIT_AUTHORITY",
            "birth_authority": "FROZEN_X13_STITCHED_RGB_DYNAMIC_CONFIDENCE",
            "continuation": "FROZEN_X14_TRANSPORT_AT_X7_VELOCITY",
            "continuation_s": x15.x14.CONTINUATION_S,
            "missing_evidence_policy": "NO_NEW_AUTHORITY_PRESERVE_EXISTING_BOUNDED_CONTINUATION",
        },
        "full_gate": {
            "minimum_contact_recall": x3.MINIMUM_CONTACT_RECALL,
            "maximum_false_segments": x3.MAXIMUM_FALSE_SEGMENTS,
            "minimum_event_f1": x3.MINIMUM_EVENT_F1,
            "minimum_median_lead_s": x3.MINIMUM_MEDIAN_LEAD_S,
            "minimum_dropout_recovery": x3.MINIMUM_DROPOUT_RECOVERY,
        },
        "inputs": {
            "x10_result_sha256": sha256_file(args.x10_result.resolve(strict=True)),
            "x15_result_sha256": sha256_file(args.x15_result.resolve(strict=True)),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "baseline_result_sha256": sha256_file(args.baseline_result.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
            "calibration_cameras_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "cameras.yaml"),
            "sequences": sources,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze = _paths(root)["freeze"]
    value = _fingerprint(args)
    if freeze.exists():
        require(json.loads(freeze.read_text(encoding="utf-8")) == value, "x16_freeze_drift")
    else:
        write_json(freeze, value)
    return {"schema": FREEZE_SCHEMA, "status": "READY", "sequences": sorted(_baseline_rows(args)), "freeze": str(freeze), "freeze_sha256": sha256_file(freeze)}


def materialize_sequence(args: argparse.Namespace) -> dict[str, Any]:
    require(args.sequence is not None, "x16_sequence_required")
    root = args.root.resolve(strict=True)
    freeze = _paths(root)["freeze"].resolve(strict=True)
    require(json.loads(freeze.read_text(encoding="utf-8")) == _fingerprint(args), "x16_freeze_drift")
    require(args.sequence in _baseline_rows(args), f"x16_unknown_sequence:{args.sequence}")
    sequence = args.sequence
    paths = _paths(root, sequence)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    if paths["ledger"].exists() and paths["manifest"].exists():
        value = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if value.get("ledger_sha256") == sha256_file(paths["ledger"]):
            return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "resumed_from_sealed_ledger": True}
    x3._acquire_lock(paths["lock"])
    try:
        source_path, source_manifest_path = _source_paths(args.x10_root.resolve(strict=True), sequence)
        source = x1._load_sealed(source_path, source_manifest_path, x10.LEDGER_SCHEMA)
        frames = [int(frame) for frame in source["frames"]]
        timestamps = {int(frame): float(stamp) for frame, stamp in zip(source["frames"], source["frame_time_s"])}
        require(frames == list(range(frames[0], frames[-1] + 1)), f"x16_noncontiguous:{sequence}")
        _x7_ledger, x7_manifest_path = _source_paths(args.x7_root.resolve(strict=True), sequence)
        x7_manifest = json.loads(x7_manifest_path.read_text(encoding="utf-8"))
        bag_path = Path(x7_manifest["source"]["bag"]).resolve(strict=True)
        pose_samples, camera_audit = x15.x9._read_poses(bag_path)
        poses = {}
        for frame in frames:
            try:
                poses[frame] = x15.x8.c22.interpolate_pose(pose_samples, round(timestamps[frame] * 1e9))
            except (AssertionError, RuntimeError, ValueError):
                pass
        calibration = x15.x8.c22.load_calibration(args.calibration_dir.resolve(strict=True))
        output_rows, diagnostics, visual = x15._stream_authority(
            bag_path=bag_path, source=source, frames=frames, timestamps=timestamps,
            poses=poses, calibration=calibration, progress_path=paths["progress"],
        )
        arrays = x5._pack_rows(frames, timestamps, output_rows)
        atomic_npz(paths["ledger"], **arrays)
        compute = [float(row["authorization_seconds"]) + float(row["continuation_seconds"]) for row in diagnostics]
        manifest = {
            "schema": LEDGER_SCHEMA,
            "truth_blind": True,
            "oracle": False,
            "sequence": sequence,
            "frames": len(frames),
            "online_information_boundary": "frame t uses sealed held-out X10 cells, stitched RGB through t, ego poses through t, and authorized cells no older than frozen 0.50 seconds",
            "birth_rule": "X10_HELDOUT_AUTHORITY_AND_FROZEN_X13_RGB_DYNAMIC_AUTHORITY",
            "continuation_rule": "FROZEN_X14_X7_VELOCITY_TRANSPORT_FOR_R1_CLEAR_GRACE",
            "continuation_s": x15.x14.CONTINUATION_S,
            "source": {
                "freeze_sha256": sha256_file(freeze), "x10_ledger": str(source_path),
                "x10_ledger_sha256": sha256_file(source_path), "x10_manifest": str(source_manifest_path),
                "x10_manifest_sha256": sha256_file(source_manifest_path), "x7_manifest": str(x7_manifest_path),
                "x7_manifest_sha256": sha256_file(x7_manifest_path), "bag": str(bag_path),
                "bag_sha256": sha256_file(bag_path), "calibration": calibration, "camera_audit": camera_audit,
            },
            "diagnostics": {
                "input_cells": int(sum(row["input_cells"] for row in diagnostics)),
                "authorized_cells": int(sum(row["authorized_cells"] for row in diagnostics)),
                "continued_cells": int(sum(row["continued_cells"] for row in diagnostics)),
                "source_compute_p95_s": float(np.quantile(np.asarray(compute), 0.95, method="higher")),
                "visual": visual,
            },
            "ledger": str(paths["ledger"]), "ledger_sha256": sha256_file(paths["ledger"]),
        }
        write_json(paths["manifest"], manifest)
        write_json(paths["progress"], {"schema": "blindassist-dtr-x16-progress-v1", "status": "COMPLETE", "percent": 100.0, "last_activity_unix_s": time.time()})
        return {"schema": SEQUENCE_SCHEMA, "status": "SEQUENCE_COMPLETE", "sequence": sequence, "frames": len(frames), "input_cells": manifest["diagnostics"]["input_cells"], "authorized_cells": manifest["diagnostics"]["authorized_cells"], "continued_cells": manifest["diagnostics"]["continued_cells"]}
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    require(json.loads(_paths(root)["freeze"].read_text(encoding="utf-8")) == _fingerprint(args), "x16_freeze_drift")
    receipts = []
    totals = {"frames": 0, "input_cells": 0, "authorized_cells": 0, "continued_cells": 0, "matched_frames": 0, "evaluated_frames": 0, "missing_frames": 0}
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence)
        value = json.loads(paths["manifest"].resolve(strict=True).read_text(encoding="utf-8"))
        require(value.get("schema") == LEDGER_SCHEMA and value.get("ledger_sha256") == sha256_file(paths["ledger"].resolve(strict=True)), f"x16_manifest:{sequence}")
        diag, visual = value["diagnostics"], value["diagnostics"]["visual"]
        totals["frames"] += int(value["frames"])
        for key in ("input_cells", "authorized_cells", "continued_cells"):
            totals[key] += int(diag[key])
        totals["matched_frames"] += int(visual["matched_frames"]); totals["evaluated_frames"] += int(visual["evaluated_frames"]); totals["missing_frames"] += int(visual["no_new_authority_visual_frames"])
        receipts.append({"sequence": sequence, "manifest": str(paths["manifest"]), "manifest_sha256": sha256_file(paths["manifest"])})
    require(totals["frames"] == x15.TIMELINE_FRAMES, f"x16_timeline:{totals['frames']}")
    result = {"schema": MATERIALIZATION_SCHEMA, "status": "COMPLETE", "truth_blind": True, **totals, "continuation_s": x15.x14.CONTINUATION_S, "backend": {"python": platform.python_version(), "opencv": cv2.__version__, "source": "CPU_SPARSE_PYRLK"}, "freeze_sha256": sha256_file(_paths(root)["freeze"]), "sequence_manifests": receipts}
    write_json(_paths(root)["materialization"], result)
    return result


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.predict(args)
    finally:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["prediction_boundary"] = "sealed X16 held-out cross-fit plus RGB-authorized continuation ledgers; no held-out labels or evaluator identity"
    result["scorer_compatibility_arm_key"] = {"X3_LAG_FLOXEL": "X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION"}
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = SCHEMA, LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.score(args)
    finally:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["schema"] = SCHEMA
    result["status"] = "DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_MET" if result["gate"]["passed"] else "DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET"
    result["metrics"]["X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["decision"]["next"] = "FREEZE_X16_FOR_NEW_SOURCE_DISJOINT_CONFIRMATION" if result["gate"]["passed"] else "CLOSE_X16_COMPOSITION_WITHOUT_PARAMETER_SWEEP"
    result["evidence_boundary"] = [
        "Post-outcome Development composition on the already opened six-sequence cohort; not confirmation.",
        "Each X10 source ledger was trained on the other five sequences and sealed without held-out labels.",
        "X13 RGB birth authority, X14 0.50-second continuation, route, lifecycle, and scorer are unchanged.",
        "Real-device latency, product benefit, and safety are not established.",
    ]
    write_json(_paths(args.root.resolve(strict=True))["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    x7_root = REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829"
    x10_root = REPO / "artifacts.local" / "evidence" / "dtr-x10" / "cross-fitted-motion-authority-20260829-v2"
    x15_root = REPO / "artifacts.local" / "evidence" / "dtr-x15" / "full-rgb-authorized-continuation-replay-20260829"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "assemble", "predict", "score"))
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x16" / "crossfit-rgb-authorized-continuation-replay-20260829")
    parser.add_argument("--sequence")
    parser.add_argument("--x7-root", type=Path, default=x7_root)
    parser.add_argument("--x10-root", type=Path, default=x10_root)
    parser.add_argument("--x10-result", type=Path, default=x10_root / "result.json")
    parser.add_argument("--x15-result", type=Path, default=x15_root / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json")
    parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--calibration-dir", type=Path, default=REPO / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = {"prepare": prepare, "materialize": materialize_sequence, "assemble": assemble, "predict": predict, "score": score}[args.mode](args)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
