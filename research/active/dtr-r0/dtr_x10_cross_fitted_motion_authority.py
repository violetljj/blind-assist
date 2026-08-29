"""Sequence-held-out cross-fitted learned motion-authority replay.

Six fixed LogisticRegression folds classify sealed X9 cells.  Each fold's
label reader can open only the other five sequences; its held-out probabilities
are sealed before score is allowed to open all native OBB truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c27_persistent_point_support as c27  # noqa: E402
import dtr_m1_point_velocity_oracle as m1  # noqa: E402
import dtr_x0_motion_source_attribution as x0  # noqa: E402
import dtr_x1_causal_floxel_source_canary as x1  # noqa: E402
import dtr_x3_full_lag_floxel_replay as x3  # noqa: E402
import dtr_x5_overlap_cycle_source_falsifier as x5  # noqa: E402
import dtr_x6_static_world_persistence_falsifier as x6  # noqa: E402
import dtr_x7_full_static_world_anchor_replay as x7  # noqa: E402
import dtr_x9_full_rgb_static_veto_replay as x9  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_r7_occupancy_flow_canary import atomic_npz  # noqa: E402


SCHEMA = "blindassist-dtr-x10-cross-fitted-motion-authority-v1"
LEDGER_SCHEMA = "blindassist-dtr-x10-cross-fitted-motion-authority-ledger-v1"
PREDICTION_SCHEMA = "blindassist-dtr-x10-cross-fitted-motion-authority-predictions-v1"
FREEZE_SCHEMA = "blindassist-dtr-x10-cross-fitted-motion-authority-freeze-v1"
FEATURE_SCHEMA = "blindassist-dtr-x10-deployable-features-v1"
FOLD_SCHEMA = "blindassist-dtr-x10-heldout-fold-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x10-materialization-v1"
THRESHOLD = 0.5
FEATURE_NAMES = (
    "range_m", "bearing_sin", "bearing_cos", "speed_mps",
    "radial_velocity_mps", "tangential_velocity_mps",
    "log1p_source_point_count", "flow_support",
    "x3_to_x7_retention", "x7_to_x9_retention", "x9_velocity_mad_mps",
)


def _paths(root: Path, sequence: str | None = None) -> dict[str, Path]:
    base = root if sequence is None else root / "sequences" / sequence
    return {
        "freeze": root / "freeze.json",
        "features": base / "features.npz",
        "feature_manifest": base / "features.json",
        "fold_predictions": base / "heldout-probabilities.npz",
        "fold_receipt": base / "fold.json",
        "ledger": base / "lag-floxel.npz",
        "manifest": base / "lag-floxel.json",
        "materialization": root / "materialization.json",
        "predictions": root / "predictions.json",
        "result": root / "result.json",
    }


def _source_paths(root: Path, sequence: str) -> tuple[Path, Path]:
    base = root / "sequences" / sequence
    return base / "lag-floxel.npz", base / "lag-floxel.json"


def _baseline_rows(args: argparse.Namespace) -> dict[str, Any]:
    _value, rows = x3._load_baseline(args)
    return rows


def _load_ledger(root: Path, sequence: str, schema: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path, manifest_path = _source_paths(root.resolve(strict=True), sequence)
    manifest = json.loads(manifest_path.resolve(strict=True).read_text(encoding="utf-8"))
    require(manifest.get("schema") == schema, f"x10_source_schema:{sequence}:{schema}")
    require(manifest.get("truth_blind") is True, f"x10_source_truth:{sequence}")
    require(manifest.get("ledger_sha256") == sha256_file(path.resolve(strict=True)), f"x10_source_hash:{sequence}")
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name].copy() for name in value.files}, manifest


def _input_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    rows = _baseline_rows(args)
    sources: dict[str, Any] = {}
    for sequence in sorted(rows):
        sources[sequence] = {}
        for name, root, schema in (
            ("x3", args.x3_root, x3.LEDGER_SCHEMA),
            ("x7", args.x7_root, x7.LEDGER_SCHEMA),
            ("x9", args.x9_root, x9.LEDGER_SCHEMA),
        ):
            ledger, manifest = _source_paths(root.resolve(strict=True), sequence)
            value = json.loads(manifest.resolve(strict=True).read_text(encoding="utf-8"))
            require(value.get("schema") == schema, f"x10_freeze_schema:{name}:{sequence}")
            require(value.get("truth_blind") is True, f"x10_freeze_truth:{name}:{sequence}")
            require(value.get("ledger_sha256") == sha256_file(ledger.resolve(strict=True)), f"x10_freeze_hash:{name}:{sequence}")
            sources[sequence][name] = {
                "ledger_sha256": sha256_file(ledger),
                "manifest_sha256": sha256_file(manifest),
            }
    return {
        "schema": FREEZE_SCHEMA,
        "algorithm_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_universe": "SEALED_X9_CELLS",
        "feature_names": list(FEATURE_NAMES),
        "neighborhood_radius_m": x6.POSITION_TOLERANCE_M,
        "label": "REAL_MOVER_NONCRITICAL=1; STATIC_PSEUDO_MOTION_OR_BAD_FLOW_OR_NO_MATCH=0; UNKNOWN_EXCLUDED",
        "cross_fit": "SIX_SEQUENCE_HELD_OUT_FOLDS_NO_HELDOUT_LABEL_ACCESS_BEFORE_SEAL",
        "model": {
            "scaler": "StandardScaler",
            "class": "LogisticRegression", "penalty": "l2", "C": 1.0,
            "class_weight": "balanced", "solver": "lbfgs", "threshold": THRESHOLD,
            "max_iter": 200, "tol": 1e-8, "fit_intercept": True, "random_state": 0,
        },
        "backend": {"execution": "CPU", "reason": "SKLEARN_LBFGS_CPU_ONLY", "sklearn": sklearn_version, "python": platform.python_version()},
        "frozen_downstream": {"velocity": "X3_UNCHANGED", "route": "UNCHANGED_R7", "lifecycle": "UNCHANGED_X3", "scorer": "UNCHANGED_X3"},
        "gate": {
            "minimum_contact_recall": x3.MINIMUM_CONTACT_RECALL,
            "maximum_false_segments": x3.MAXIMUM_FALSE_SEGMENTS,
            "minimum_event_f1": x3.MINIMUM_EVENT_F1,
            "minimum_median_lead_s": x3.MINIMUM_MEDIAN_LEAD_S,
            "minimum_dropout_recovery": x3.MINIMUM_DROPOUT_RECOVERY,
            "false_segments_must_be_below_pdc": True,
        },
        "inputs": {
            "sources": sources,
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve(strict=True)),
            "baseline_result_sha256": sha256_file(args.baseline_result.resolve(strict=True)),
            "roster_sha256": sha256_file(args.roster.resolve(strict=True)),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "x9_result_sha256": sha256_file(args.x9_result.resolve(strict=True)),
        },
    }


def _validate_freeze(args: argparse.Namespace) -> dict[str, Any]:
    value = json.loads(_paths(args.root.resolve(strict=True))["freeze"].read_text(encoding="utf-8"))
    require(value == _input_fingerprint(args), "x10_freeze_drift")
    return value


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = _paths(root)["freeze"]
    value = _input_fingerprint(args)
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == value, "x10_freeze_drift")
    else:
        write_json(path, value)
    return {"status": "READY", "freeze": str(path), "freeze_sha256": sha256_file(path), "sequences": sorted(_baseline_rows(args))}


def _frame_slice(ledger: Mapping[str, np.ndarray], index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, stop = int(ledger["offsets"][index]), int(ledger["offsets"][index + 1])
    p = np.column_stack((ledger["forward_m"][start:stop], ledger["left_m"][start:stop])).astype(np.float64)
    v = np.column_stack((ledger["velocity_forward_mps"][start:stop], ledger["velocity_left_mps"][start:stop])).astype(np.float64)
    return p, v, ledger["source_point_count"][start:stop], ledger["flow_support"][start:stop]


def _near_count(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    if not len(source) or not len(query):
        return np.zeros(len(query), dtype=np.float64)
    return np.asarray(cKDTree(source).query_ball_point(query, x6.POSITION_TOLERANCE_M + 1e-12, return_length=True, workers=1), dtype=np.float64)


def materialize_features(args: argparse.Namespace) -> dict[str, Any]:
    _validate_freeze(args)
    rows = _baseline_rows(args)
    receipts = []
    for sequence in sorted(rows):
        x3l, _ = _load_ledger(args.x3_root, sequence, x3.LEDGER_SCHEMA)
        x7l, _ = _load_ledger(args.x7_root, sequence, x7.LEDGER_SCHEMA)
        x9l, x9m = _load_ledger(args.x9_root, sequence, x9.LEDGER_SCHEMA)
        require(np.array_equal(x3l["frames"], x9l["frames"]) and np.array_equal(x7l["frames"], x9l["frames"]), f"x10_frame_alignment:{sequence}")
        chunks: list[np.ndarray] = []
        for i in range(len(x9l["frames"])):
            p3, _v3, _c3, _s3 = _frame_slice(x3l, i)
            p7, _v7, _c7, _s7 = _frame_slice(x7l, i)
            p9, v9, counts, support = _frame_slice(x9l, i)
            if not len(p9):
                chunks.append(np.empty((0, len(FEATURE_NAMES)), np.float32)); continue
            radius = np.linalg.norm(p9, axis=1)
            unit = p9 / np.maximum(radius[:, None], 1e-9)
            tangent = np.column_stack((-unit[:, 1], unit[:, 0]))
            n3, n7, n9 = _near_count(p3, p9), _near_count(p7, p9), _near_count(p9, p9)
            neighborhoods = cKDTree(p9).query_ball_point(p9, x6.POSITION_TOLERANCE_M + 1e-12, workers=1)
            mad = np.empty(len(p9), dtype=np.float64)
            for j, member in enumerate(neighborhoods):
                local = v9[np.asarray(member, dtype=np.int64)]
                center = np.median(local, axis=0)
                mad[j] = float(np.median(np.linalg.norm(local - center, axis=1)))
            features = np.column_stack((
                radius, unit[:, 1], unit[:, 0], np.linalg.norm(v9, axis=1),
                np.sum(v9 * unit, axis=1), np.sum(v9 * tangent, axis=1),
                np.log1p(counts), support,
                np.clip(n7 / np.maximum(n3, 1.0), 0.0, 1.0),
                np.clip(n9 / np.maximum(n7, 1.0), 0.0, 1.0), mad,
            )).astype(np.float32)
            chunks.append(features)
        features = np.concatenate(chunks, axis=0)
        require(len(features) == len(x9l["forward_m"]), f"x10_feature_alignment:{sequence}")
        out = _paths(args.root.resolve(), sequence)
        out["features"].parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(out["features"], frames=x9l["frames"], offsets=x9l["offsets"], features=features)
        manifest = {"schema": FEATURE_SCHEMA, "truth_blind": True, "sequence": sequence, "cells": len(features), "feature_names": list(FEATURE_NAMES), "features_sha256": sha256_file(out["features"]), "x9_manifest_sha256": sha256_file(_source_paths(args.x9_root, sequence)[1]), "x9_ledger_sha256": x9m["ledger_sha256"]}
        write_json(out["feature_manifest"], manifest)
        receipts.append({"sequence": sequence, "cells": len(features), "features_sha256": manifest["features_sha256"]})
    return {"status": "COMPLETE", "sequences": receipts, "cells": sum(row["cells"] for row in receipts)}


class RestrictedLabelReader:
    """No generic member reader: only exact training sequence members are reachable."""
    def __init__(self, path: Path, allowed: set[str], forbidden: str):
        require(forbidden not in allowed, "x10_heldout_in_allowed")
        self._zip = zipfile.ZipFile(path.resolve(strict=True))
        self._allowed = frozenset(allowed)
        self._forbidden = forbidden
        self.accessed: list[str] = []

    def sequence(self, sequence: str) -> Mapping[str, Any]:
        require(sequence in self._allowed and sequence != self._forbidden, f"x10_label_access_denied:{sequence}")
        member = f"labels/labels_3d/{sequence}.json"
        value = json.loads(self._zip.read(member))["labels"]
        self.accessed.append(member)
        return value

    def close(self) -> None:
        self._zip.close()


def _poses_and_times(row: Mapping[str, Any]) -> tuple[list[int], dict[int, float], dict[int, dict[str, float]]]:
    source = row["sources"]["ledgers"]["M1_PD_GLOBAL"]
    arrays = c27._load_arrays(Path(source["ledger"]).resolve(strict=True), Path(source["manifest"]).resolve(strict=True), {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad"})
    frames = [int(v) for v in arrays["frames"]]
    times = {frame: float(arrays["frame_time_s"][i]) for i, frame in enumerate(frames)}
    poses = {frame: c27._pose(arrays, i) for i, frame in enumerate(frames)}
    return frames, times, poses


def _boxes(values: Mapping[str, Any], frames: Sequence[int], times: Mapping[int, float], poses: Mapping[int, Mapping[str, float]]) -> dict[int, list[m1.NativeBox]]:
    output: dict[int, list[m1.NativeBox]] = {}
    for frame in frames:
        pose = poses[frame]; boxes = []
        for item in values.get(f"{frame:06d}.pcd", ()): 
            if bool(item.get("attributes", {}).get("no_eval", False)): continue
            box = item["box"]
            boxes.append(m1.NativeBox(frame, times[frame], str(item["label_id"]), float(box["cx"]) + m1.BASE_LINK_FROM_LOGICAL_RGB360_X_M, float(box["cy"]) + m1.BASE_LINK_FROM_LOGICAL_RGB360_Y_M, float(box["cz"]), float(box["l"]), float(box["w"]), float(box["h"]), float(box["rot_z"]), float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])))
        output[frame] = boxes
    return output


def _training_labels(args: argparse.Namespace, sequence: str, values: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ledger, _ = _load_ledger(args.x9_root, sequence, x9.LEDGER_SCHEMA)
    feature_path = _paths(args.root.resolve(strict=True), sequence)
    manifest = json.loads(feature_path["feature_manifest"].read_text(encoding="utf-8"))
    require(manifest["features_sha256"] == sha256_file(feature_path["features"]), f"x10_feature_hash:{sequence}")
    with np.load(feature_path["features"], allow_pickle=False) as source: features = source["features"].copy()
    frames, times, poses = _poses_and_times(row)
    boxes = _boxes(values, frames, times, poses); history = m1._box_history(boxes)
    y = np.full(len(features), -1, dtype=np.int8); counts = {"positive": 0, "negative": 0, "unknown": 0}
    for i, frame in enumerate(frames):
        start, stop = int(ledger["offsets"][i]), int(ledger["offsets"][i + 1])
        for cell_index in range(start, stop):
            cell = {"forward_m": ledger["forward_m"][cell_index], "left_m": ledger["left_m"][cell_index], "velocity_forward_mps": ledger["velocity_forward_mps"][cell_index], "velocity_left_mps": ledger["velocity_left_mps"][cell_index]}
            matches = [box for box in boxes.get(frame, ()) if x0._cell_clearance(cell, box) <= x0.ASSOCIATION_MARGIN_M + 1e-9]
            if not matches: label = 0
            elif len(matches) != 1: label = -1
            else:
                target = x0._target_velocity(matches[0], history, poses[frame])
                if target is None: label = -1
                elif math.hypot(*target) < x0.FLOW_ERROR_LIMIT_MPS: label = 0
                elif math.hypot(float(cell["velocity_forward_mps"]) - target[0], float(cell["velocity_left_mps"]) - target[1]) > x0.FLOW_ERROR_LIMIT_MPS: label = 0
                else: label = 1
            y[cell_index] = label
            counts[{1: "positive", 0: "negative", -1: "unknown"}[label]] += 1
    known = y >= 0
    return features[known], y[known].astype(np.int32), counts


def run_fold(args: argparse.Namespace) -> dict[str, Any]:
    _validate_freeze(args)
    rows = _baseline_rows(args); heldout = str(args.held_out)
    require(heldout in rows, "x10_unknown_heldout")
    train = sorted(set(rows) - {heldout}); reader = RestrictedLabelReader(args.labels, set(train), heldout)
    xs, ys, label_counts = [], [], {}
    try:
        for sequence in train:
            x, y, counts = _training_labels(args, sequence, reader.sequence(sequence), rows[sequence])
            xs.append(x); ys.append(y); label_counts[sequence] = counts
    finally: reader.close()
    expected = [f"labels/labels_3d/{sequence}.json" for sequence in train]
    require(reader.accessed == expected and f"labels/labels_3d/{heldout}.json" not in reader.accessed, "x10_label_access_audit")
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    require(set(np.unique(y_train)) == {0, 1}, "x10_train_classes")
    scaler = StandardScaler(); scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=200, tol=1e-8, fit_intercept=True, random_state=0)
    model.fit(scaled, y_train)
    heldout_paths = _paths(args.root.resolve(strict=True), heldout)
    fm = json.loads(heldout_paths["feature_manifest"].read_text(encoding="utf-8"))
    require(fm["features_sha256"] == sha256_file(heldout_paths["features"]), "x10_heldout_feature_hash")
    with np.load(heldout_paths["features"], allow_pickle=False) as source: heldout_features = source["features"].copy()
    probabilities = model.predict_proba(scaler.transform(heldout_features))[:, int(np.flatnonzero(model.classes_ == 1)[0])].astype(np.float32)
    atomic_npz(heldout_paths["fold_predictions"], probability_real_mover=probabilities)
    sealed_hash = sha256_file(heldout_paths["fold_predictions"])
    receipt = {"schema": FOLD_SCHEMA, "status": "SEALED", "held_out": heldout, "train_sequences": train, "allowed_label_members": expected, "accessed_label_members": reader.accessed, "forbidden_label_member": f"labels/labels_3d/{heldout}.json", "heldout_label_access_before_seal": False, "train_rows": len(y_train), "train_class_counts": {str(v): int(np.sum(y_train == v)) for v in (0, 1)}, "label_counts": label_counts, "model": {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "classes": model.classes_.tolist(), "n_iter": model.n_iter_.tolist()}, "backend": {"execution": "CPU", "reason": "SKLEARN_LBFGS_CPU_ONLY", "sklearn": sklearn_version}, "threshold": THRESHOLD, "predictions": len(probabilities), "predictions_sha256": sealed_hash}
    write_json(heldout_paths["fold_receipt"], receipt)
    return receipt


def merge(args: argparse.Namespace) -> dict[str, Any]:
    _validate_freeze(args); rows = _baseline_rows(args); sequence_receipts = []; total_in = total_keep = 0
    for sequence in sorted(rows):
        paths = _paths(args.root.resolve(strict=True), sequence)
        receipt = json.loads(paths["fold_receipt"].read_text(encoding="utf-8"))
        require(receipt.get("schema") == FOLD_SCHEMA and receipt.get("held_out") == sequence, f"x10_fold_receipt:{sequence}")
        require(receipt.get("heldout_label_access_before_seal") is False and receipt["forbidden_label_member"] not in receipt["accessed_label_members"], f"x10_fold_isolation:{sequence}")
        require(receipt["predictions_sha256"] == sha256_file(paths["fold_predictions"]), f"x10_fold_hash:{sequence}")
        with np.load(paths["fold_predictions"], allow_pickle=False) as source: probability = source["probability_real_mover"].copy()
        ledger, _ = _load_ledger(args.x9_root, sequence, x9.LEDGER_SCHEMA)
        require(len(probability) == len(ledger["forward_m"]), f"x10_fold_alignment:{sequence}")
        rows_out = []
        for i in range(len(ledger["frames"])):
            start, stop = int(ledger["offsets"][i]), int(ledger["offsets"][i + 1]); keep = probability[start:stop] >= THRESHOLD
            rows_out.append((np.column_stack((ledger["forward_m"][start:stop][keep], ledger["left_m"][start:stop][keep])), np.column_stack((ledger["velocity_forward_mps"][start:stop][keep], ledger["velocity_left_mps"][start:stop][keep])), ledger["source_point_count"][start:stop][keep], ledger["flow_support"][start:stop][keep]))
        times = {int(frame): float(stamp) for frame, stamp in zip(ledger["frames"], ledger["frame_time_s"])}
        packed = x5._pack_rows([int(v) for v in ledger["frames"]], times, rows_out)
        atomic_npz(paths["ledger"], **packed)
        manifest = {"schema": LEDGER_SCHEMA, "truth_blind": True, "cross_fit_train_truth": True, "heldout_truth_unused": True, "truth_usage": "SEQUENCE_HELD_OUT_CROSS_FIT_TRAIN_LABELS_ONLY", "sequence": sequence, "frames": len(ledger["frames"]), "input_cells": len(probability), "retained_cells": len(packed["forward_m"]), "threshold": THRESHOLD, "fold_predictions_sha256": receipt["predictions_sha256"], "ledger_sha256": sha256_file(paths["ledger"])}
        write_json(paths["manifest"], manifest); total_in += len(probability); total_keep += len(packed["forward_m"]); sequence_receipts.append({"sequence": sequence, "manifest_sha256": sha256_file(paths["manifest"]), "fold_predictions_sha256": receipt["predictions_sha256"]})
    materialization = {"schema": MATERIALIZATION_SCHEMA, "status": "COMPLETE", "sequence_held_out_cross_fit": True, "frames": sum(json.loads(_paths(args.root.resolve(strict=True), s)["manifest"].read_text())["frames"] for s in rows), "input_cells": total_in, "retained_cells": total_keep, "threshold": THRESHOLD, "sequences": sequence_receipts}
    write_json(_paths(args.root.resolve(strict=True))["materialization"], materialization)
    return materialization


def predict(args: argparse.Namespace) -> dict[str, Any]:
    previous = x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.predict(args)
    finally: x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["prediction_boundary"] = "six sealed held-out fold predictions plus frozen route/lifecycle; held-out OBB not opened by its fold"
    write_json(_paths(args.root.resolve(strict=True))["predictions"], result)
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    for sequence in sorted(_baseline_rows(args)):
        paths = _paths(root, sequence); receipt = json.loads(paths["fold_receipt"].read_text(encoding="utf-8"))
        require(receipt["predictions_sha256"] == sha256_file(paths["fold_predictions"]), f"x10_pre_score_fold_hash:{sequence}")
    previous = x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA
    try:
        x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = SCHEMA, LEDGER_SCHEMA, PREDICTION_SCHEMA
        result = x3.score(args)
    finally: x3.SCHEMA, x3.LEDGER_SCHEMA, x3.PREDICTION_SCHEMA = previous
    result["schema"] = SCHEMA; result["metrics"]["X10_CROSS_FITTED_MOTION_AUTHORITY"] = result["metrics"].pop("X3_LAG_FLOXEL")
    result["status"] = "DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_MET" if result["gate"]["passed"] else "DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET"
    result["evidence_boundary"] = ["Six-sequence consumed Development cohort; each prediction uses a model trained on the other five sequences only.", "Native OBB labels train cross-fit folds and are not production-deployable inputs.", "X3 velocity, route, lifecycle, and scorer are unchanged."]
    write_json(_paths(root)["result"], result); return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c31 = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "features", "fold", "merge", "predict", "score")); parser.add_argument("--held-out")
    parser.add_argument("--root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x10" / "cross-fitted-motion-authority-20260829-v2")
    parser.add_argument("--x3-root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x3" / "full-lag-floxel-replay-mp")
    parser.add_argument("--x7-root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x7" / "full-static-world-anchor-replay-20260829")
    parser.add_argument("--x9-root", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x9" / "full-rgb-static-veto-replay-20260829-v2")
    parser.add_argument("--x9-result", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-x9" / "full-rgb-static-veto-replay-20260829-v2" / "result.json")
    parser.add_argument("--baseline-predictions", type=Path, default=c31 / "baseline-predictions.json"); parser.add_argument("--baseline-result", type=Path, default=c31 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c31_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip"); parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); require(args.mode != "fold" or args.held_out, "x10_heldout_required")
    fn = {"prepare": prepare, "features": materialize_features, "fold": run_fold, "merge": merge, "predict": predict, "score": score}[args.mode]
    print(json.dumps(fn(args), indent=2, sort_keys=True))


if __name__ == "__main__": main()
