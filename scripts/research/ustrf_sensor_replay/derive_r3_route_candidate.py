from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from contract import read_json, sha256, write_json


def project(point_world: np.ndarray, camera_to_world: np.ndarray, intrinsics: list[float]) -> list[float] | None:
    point_camera = np.linalg.inv(camera_to_world) @ np.r_[point_world, 1.0]
    if point_camera[2] <= 0.03:
        return None
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    return [fx * point_camera[0] / point_camera[2] + cx, fy * point_camera[1] / point_camera[2] + cy]


def load_depth(row: dict, root: Path) -> np.ndarray:
    path = root / row["depth_path"]
    if row["depth_encoding"] == "uint16_png_z_meters":
        return np.asarray(Image.open(path), dtype=np.float32) / float(row["depth_scale"])
    return np.load(path).astype(np.float32, copy=False) / float(row["depth_scale"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    prereg = read_json(args.prereg)
    pose = read_json(args.pose)
    if pose.get("ground_truth_pose_accessed") is not False:
        raise ValueError("pose estimate independence receipt missing")
    pose_by_source = {row["source_id"]: row for row in pose["sources"]}
    horizon = int(prereg["route"]["truth_horizon_frames"])
    history = int(prereg["route"]["causal_history_frames"])
    minimum_displacement = float(prereg["route"]["minimum_forward_displacement_m"])
    candidate = prereg["candidate"]
    sources = []
    for source_dir in sorted(args.bundle_root.iterdir()):
        if not (source_dir / "bundle.json").is_file():
            continue
        bundle = read_json(source_dir / "bundle.json")
        source_id = bundle["source"]["source_id"]
        root = Path(bundle["source_root"])
        frames = [json.loads(line) for line in (source_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
        estimates = pose_by_source[source_id]["pose_estimates"]
        if [row["frame_id"] for row in estimates] != [row["frame_id"] for row in frames]:
            raise ValueError(f"pose/frame identity mismatch: {source_id}")
        truth = []
        predictions = []
        trace = []
        active = False
        start_index = None
        risk_run = 0
        clear_run = 0
        alerts = []
        for index, frame in enumerate(frames):
            intrinsics = frame["intrinsics_fx_fy_cx_cy"]
            height, width = load_depth(frame, root).shape
            truth_row = {"frame_id": frame["frame_id"], "timestamp_s": frame["rgb_timestamp_s"], "status": "unknown"}
            future_index = index + horizon
            if future_index < len(frames):
                current_truth = np.asarray(frame["camera_to_world"], dtype=np.float64)
                future_truth = np.asarray(frames[future_index]["camera_to_world"], dtype=np.float64)
                displacement = float(np.linalg.norm(future_truth[:3, 3] - current_truth[:3, 3]))
                uv = project(future_truth[:3, 3], current_truth, intrinsics)
                if displacement >= minimum_displacement and uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
                    truth_row.update({"status": "known", "uv": uv, "future_frame_id": frames[future_index]["frame_id"]})
            truth.append(truth_row)

            prediction = {"frame_id": frame["frame_id"], "timestamp_s": frame["rgb_timestamp_s"], "predicted_at_s": frame["rgb_timestamp_s"], "status": "unknown"}
            if index >= history and estimates[index]["status"] != "unknown" and estimates[index-history]["status"] != "unknown":
                current_estimate = np.asarray(estimates[index]["camera_to_world"], dtype=np.float64)
                past_estimate = np.asarray(estimates[index-history]["camera_to_world"], dtype=np.float64)
                delta = current_estimate[:3, 3] - past_estimate[:3, 3]
                target = current_estimate[:3, 3] + delta * (horizon / history)
                uv = project(target, current_estimate, intrinsics)
                if float(np.linalg.norm(delta)) >= minimum_displacement and uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
                    prediction.update({"status": "known", "uv": uv, "history_start_frame_id": frames[index-history]["frame_id"]})
            predictions.append(prediction)

            route_depth = None
            risk = False
            if prediction["status"] == "known":
                depth = load_depth(frame, root)
                radius = max(3, int(min(width, height) * float(candidate["route_radius_fraction"])))
                u, v = [int(round(value)) for value in prediction["uv"]]
                crop = depth[max(0, v-radius):min(height, v+radius+1), max(0, u-radius):min(width, u+radius+1)]
                valid = crop[np.isfinite(crop) & (crop > 0.1) & (crop < 8.0)]
                if valid.size:
                    route_depth = float(np.quantile(valid, 0.10))
                    risk = route_depth <= (float(candidate["clear_depth_m"]) if active else float(candidate["alert_depth_m"]))
            risk_run = risk_run + 1 if risk else 0
            clear_run = 0 if risk else clear_run + 1
            if not active and risk_run >= int(candidate["minimum_alert_frames"]):
                active = True
                start_index = index - int(candidate["minimum_alert_frames"]) + 1
            if active and clear_run >= int(candidate["minimum_clear_frames"]):
                end_index = index - int(candidate["minimum_clear_frames"])
                alerts.append({"start_frame": start_index, "end_frame": end_index, "start_frame_id": frames[start_index]["frame_id"], "end_frame_id": frames[end_index]["frame_id"]})
                active = False
                start_index = None
            trace.append({"frame_id": frame["frame_id"], "timestamp_s": frame["rgb_timestamp_s"], "route_status": prediction["status"], "route_depth_q10_m": route_depth, "risk": risk, "alert_active": active})
        if active and start_index is not None:
            alerts.append({"start_frame": start_index, "end_frame": len(frames)-1, "start_frame_id": frames[start_index]["frame_id"], "end_frame_id": frames[-1]["frame_id"]})
        sources.append({
            "source_id": source_id,
            "pose_estimates": estimates,
            "route_truth": truth,
            "route_predictions": predictions,
            "candidate_trace": trace,
            "alerts": alerts,
            "duration_s": float(frames[-1]["rgb_timestamp_s"] - frames[0]["rgb_timestamp_s"]),
            "source_frames_sha256": sha256(source_dir / "frames.jsonl"),
        })
    write_json(args.output, {
        "schema": "blindassist_ustrf_sensor_replay_r3_candidate_evaluation_v1",
        "pose_estimates_sha256": sha256(args.pose),
        "prereg_sha256": sha256(args.prereg),
        "candidate_id": prereg["candidate"]["id"],
        "route_prediction_policy": "past_pose_prefix_only_no_future_ground_truth_v1",
        "candidate_alerts_frozen_before_review": True,
        "sources": sources,
        "production_authority": False,
    })
    print(json.dumps({"sources": [{"source_id": row["source_id"], "alerts": len(row["alerts"]), "route_known": sum(v["status"] == "known" for v in row["route_predictions"])} for row in sources]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
