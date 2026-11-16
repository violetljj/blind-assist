"""Run the fixed fourth LaMAR comparison, with native predictions sealed before scoring."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lamar_motion_gate_dev_20261116 import (
    LOCAL_KEYS, POSITIVE, TARGET, allowed_to_veto, first_alert_delays, native_features,
)
from lamar_native_corridor_dev_20261116 import DepthSensor, evaluate_pass, predictions


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs(root):
    lock = load_json(root / "freeze_receipt.json")
    rule_path, selected_path = root / "frozen_rule.json", root / "pose_selected.json"
    if lock["rule_sha256"] != file_hash(rule_path) or lock["pose_list_sha256"] != file_hash(selected_path):
        raise ValueError("Fixed rule or pose list differs from the freeze receipt")
    rule, selected = load_json(rule_path), load_json(selected_path)
    stamps = {r["depth_row"]: r["timestamp_us"] for r in selected["rows"]}
    if selected["session"] != rule["session"] or len(stamps) != rule["pose_selected_times"]:
        raise ValueError("Fixed pose session or count changed")
    return rule, stamps, lock


def lock_predictions(root, prepared):
    start = perf_counter()
    rule, selected, lock = frozen_inputs(root)
    native = root / "native" / f"{rule['session']}.npz"
    if prepared.stem != rule["session"]:
        raise ValueError("Unexpected prepared trajectory")
    with np.load(prepared, allow_pickle=False) as source:
        local = {key: source[key] for key in LOCAL_KEYS}
        sensor = DepthSensor(source["depth_lut"], source["T_depth_to_head"])
    stamps = local["depth_timestamp_us"]
    if any(stamps[depth_row] != t for depth_row, t in selected.items()):
        raise ValueError("Fixed pose list has mismatched sensor times")
    with np.load(native, allow_pickle=False) as source:
        images = dict(zip(source["depth_rows"], source["depth_mm"], strict=True))
    needed = {depth_row - lag for depth_row in selected for lag in (0, 1, 2)}
    if not needed.issubset(images):
        raise ValueError("Native depth receipt is missing a locked frame or its confirmation history")
    prior = predictions(prepared, native)
    selected_baseline = {i: prior[i] for i in selected if i in prior}
    features = native_features(local, sensor, images, selected_baseline)
    rows = []
    for stamp, feature in features.items():
        base = np.asarray(feature["baseline"])
        veto = allowed_to_veto(feature, rule["arm"])
        if veto.any() and not np.array_equal(veto, base):
            raise AssertionError("Whole-alert rule canceled only part of an alert")
        evidence = feature["evidence"].get(rule["arm"]["direction"])
        rows.append(dict(timestamp_us=stamp, depth_row=feature["depth_row"], baseline=base.tolist(),
                         candidate=(base & ~veto).tolist(), evidence=evidence, motion=feature["motion"]))
    missing = sorted(set(selected.values()) - set(features))
    result = dict(session=rule["session"], rule_sha256=lock["rule_sha256"], pose_list_sha256=lock["pose_list_sha256"],
                  rows=rows, without_causal_confirmation=missing, required_native_frames=len(needed),
                  wall_seconds=round(perf_counter() - start, 3))
    out = root / "sealed_predictions.json"
    save_json(out, result)
    save_json(root / "prediction_receipt.json", {"predictions_sha256": file_hash(out)})
    print("Predicted", len(rows), "of", len(selected), "times, cancellations",
          sum(row["baseline"] != row["candidate"] for row in rows), flush=True)


def replay_scores(visits, scored):
    changed_bands = Counter()
    changed = []
    for visit in visits:
        observed = scored.get(visit["visit"], {})
        for truth in visit["rows"]:
            pred = observed.get(truth["depth_row"])
            if pred is None:
                continue
            base, current, _ = pred
            cancelled = base & ~current
            if not cancelled.any():
                continue
            status = np.array(truth["travel"])
            for label, name in ((1, "occupied"), (0, "clear"), (-1, "unknown")):
                changed_bands[name] += int((cancelled & (status == label)).sum())
            changed.append(dict(timestamp=truth["timestamp"], category=truth["category"],
                                cancelled=cancelled.tolist(), travel=truth["travel"]))
    geom = dict(visits=visits)
    return dict(episodes=evaluate_pass(geom, scored), bands=dict(changed_bands),
                first_alert_delays_s=first_alert_delays(geom, scored), changed=changed)


def score_predictions(root, geometry_path):
    rule, selected, lock = frozen_inputs(root)
    pred_path = root / "sealed_predictions.json"
    pred_hash = file_hash(pred_path)
    if pred_hash != load_json(root / "prediction_receipt.json")["predictions_sha256"]:
        raise ValueError("Prediction output changed after sealing")
    pred = load_json(pred_path)
    geometry = load_json(geometry_path)
    visits = geometry["visits"]
    name = rule["session"]
    if len(visits) != 1 or visits[0]["visit"] != name or pred["session"] != name:
        raise ValueError("Only the locked fourth recording may be scored")
    geo_stamps = {r["depth_row"]: int(r["timestamp"]) for r in visits[0]["rows"]}
    if geo_stamps != selected or list(geo_stamps) != list(selected):
        raise ValueError("New reference walk positions differ from the fixed set")
    if any(pred[k] != lock[k] for k in ("rule_sha256", "pose_list_sha256")):
        raise ValueError("Predictions used a different rule or pose list")
    scored = {name: {r["depth_row"]: (np.array(r["baseline"]), np.array(r["candidate"]), None)
                     for r in pred["rows"]}}
    robust_visits = [dict(visits[0], rows=[dict(row, category=row["robust_category"], travel=row["travel_robust"])
                                           for row in visits[0]["rows"]])]
    nominal = replay_scores(visits, scored)
    robust = replay_scores(robust_visits, scored)
    endpoints = nominal["episodes"]
    improvement = (endpoints[TARGET]["lost_alert"] > 0 and endpoints[POSITIVE]["lost_alert"] == 0
                   and nominal["bands"].get("occupied", 0) == 0 and not nominal["first_alert_delays_s"])
    result = dict(nominal=nominal, robust=robust, observed_primary_met=improvement,
                  predictions_sha256=pred_hash, geometry_sha256=file_hash(geometry_path),
                  missing_predictions=len(pred["without_causal_confirmation"]))
    save_json(root / "comparison.json", result)
    print("Nominal", nominal, "Primary met", improvement, flush=True)
    print("Width sensitivity", robust, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    predict = modes.add_parser("predict")
    predict.add_argument("root", type=Path)
    predict.add_argument("trajectory", type=Path)
    score = modes.add_parser("score")
    score.add_argument("root", type=Path)
    score.add_argument("geometry", type=Path)
    args = parser.parse_args()
    if args.action == "predict":
        lock_predictions(args.root, args.trajectory)
    else:
        score_predictions(args.root, args.geometry)


if __name__ == "__main__":
    main()
