"""Replay native UE RGB/forward-depth through the unchanged retained X73 stack.

Input manifest.json contains calibration and episodes. Each episode supplies
route_frame, plan_path and frames with RGB/depth paths, time_s, sample_index,
camera_transform, wearer_transform and command_velocity. Poses use UE axes,
meters and degrees; depth is float32 forward meters, never device-Z or range.
Only this model input tree is opened; evaluator actors are not an input.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "carla"))
sys.path.insert(0, str(REPO / "tools"))
import dtr_carla_rgbd_model_adapter as adapter
import dtr_carla_yolo_metric_candidates as detector
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x25_rigid_footprint_predictor as x25
import dtr_carla_x54_metric_bootstrap_dropout_continuation as x54
import dtr_carla_x65_ancestry_synchronized_conflict_handback as x65
import dtr_carla_x67_measurement_horizon_receding_release as x67
import dtr_carla_x68_object_local_lateral_dequantization as x68
import dtr_carla_x69_mature_cross_route_rigid_contradiction as x69
import dtr_carla_x70_triple_credential_surface_dropout_handback as x70
import dtr_carla_x71_entry_cotransport_occupancy_birth as x71
import dtr_carla_x72_credentialed_surface_boundary_completion as x72
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73
import research_backend as backend


def read_model_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    adapter.assert_sanitized_model_value(value)
    return value


def load_contract(model_root: Path):
    root = model_root.resolve(strict=True)
    manifest = root / "manifest.json"
    value = read_model_json(manifest)
    calibration = adapter.CameraCalibration(**value["calibration"], depth_encoding="UE_LINEAR_FORWARD_FLOAT32_METERS")
    episodes = []
    for item in value["episodes"]:
        plan_path = adapter.resolve_model_path(root, item["plan_path"], "ue_plan")
        plan = read_model_json(plan_path)
        x24.route.validate_plan_receipt(plan)
        frames = []
        last_time = -float("inf")
        for row in item["frames"]:
            if "plan_path" in row:
                plan_path = adapter.resolve_model_path(root, row["plan_path"], "ue_frame_plan")
                plan = read_model_json(plan_path)
                x24.route.validate_plan_receipt(plan)
            timestamp = float(row["time_s"])
            if not np.isfinite(timestamp) or timestamp <= last_time:
                raise ValueError("UE frame timestamps must strictly increase")
            if float(plan["issued_at_s"]) > timestamp:
                raise ValueError("UE plan must be issued before observation")
            last_time = timestamp
            refs = []
            for key in ("rgb_path", "depth_path"):
                path = adapter.resolve_model_path(root, row[key], key)
                refs.append(adapter.ImageReference(path, adapter.sha256_file(path), path.stat().st_size, calibration.width, calibration.height))
            frames.append(adapter.FrameObservation(
                episode_id=item["episode_id"], sample_index=int(row["sample_index"]),
                time_s=timestamp, world_frame=int(row.get("world_frame", row["sample_index"])),
                navigation_session_id=plan["session_id"], camera_transform=row["camera_transform"],
                rgb=refs[0], depth=refs[1],
                wearer={"transform": row["wearer_transform"], "command_velocity": row["command_velocity"]},
                issued_plan={"authority": "VALID", "path": str(plan_path), "receipt_sha256": plan["receipt_sha256"]},
            ))
        if not frames:
            raise ValueError("UE episode has no observations")
        episodes.append(adapter.Episode(item["episode_id"], adapter.AnchorFrame(**item["route_frame"]), tuple(frames)))
    if not episodes:
        raise ValueError("UE model input has no episodes")
    return adapter.SanitizedModelContract(root, manifest, adapter.sha256_file(manifest), "DTR_UE_NATIVE_RGBD_DEVELOPMENT", calibration, tuple(episodes), "ue-native-rgbd-v1")


def load_linear_depth(observation, calibration):
    adapter.validate_file_hash(observation.depth.path, observation.depth.sha256, "ue_depth")
    depth = np.load(observation.depth.path, allow_pickle=False)
    if depth.dtype != np.float32 or depth.shape != (calibration.height, calibration.width):
        raise ValueError("UE depth must be HxW float32 forward meters")
    return depth


@contextmanager
def native_depth_loader():
    # Only replace the sensor codec; all geometry and prediction remain original.
    original = adapter.load_depth_m
    adapter.load_depth_m = load_linear_depth
    try:
        yield
    finally:
        adapter.load_depth_m = original


def predict_episode(episode, candidates, calibration):
    with native_depth_loader():
        metric = x24.predict_episode(episode, candidates, calibration)
        rigid = x25.predict_episode(episode, candidates, calibration)
        core = x54.predict_episode(episode, candidates, calibration)
        core = x65.apply_ancestry_handback_episode(core, metric)
        core = x67.apply_measurement_horizon_receding_release_episode(core)
        core = x68.apply_object_local_lateral_dequantization_episode(core, metric, episode)
        core = x69.apply_mature_cross_route_rigid_contradiction_episode(core, rigid)
        core = x70.apply_triple_credential_surface_dropout_handback_episode(core, rigid, metric)
        core = x71.apply_entry_cotransport_occupancy_birth_episode(core, rigid, metric)
        core = x72.apply_credentialed_surface_boundary_completion_episode(core, rigid)
        return x73.apply_credentialed_parent_hull_reconstruction_episode(core, rigid, episode)


def compact_frame(episode_id, frame, previous_risk=False):
    arm = frame["arms"][x73.ARM_X73]
    risk = bool(arm["route_risk"])
    dispositions = [str(track.get("disposition", "UNSPECIFIED")) for track in frame.get("tracks", [])]
    support = ("CURRENT_MEASURED_TRACK_SUPPORT" if "MEASURED" in dispositions else
               "HELD_TRACK_SUPPORT_ONLY" if dispositions and all(value == "HOLD" for value in dispositions) else
               "OTHER_TRACK_SUPPORT" if dispositions else "NO_TRACK_SUPPORT")
    # This is a display transition of model risk, not a new event policy.
    event = "ONSET" if risk and not previous_risk else "HOLD" if risk else "CLEAR" if previous_risk else "NONE"
    return {"episode_id": episode_id, "sample_index": frame["sample_index"], "time_s": frame["time_s"],
            "route_risk": risk, "event": event, "minimum_entry_s": arm.get("minimum_entry_s"),
            "risk_state": "POSITIVE_PREDICTED_RISK" if risk else "NO_POSITIVE_PREDICTED_RISK_NOT_SAFETY",
            "support_state": support, "track_dispositions": dispositions,
            "global_observability": "UNKNOWN_NOT_ESTIMATED_BY_X73",
            "route_mode": arm.get("route_mode"), "confirmed_risk_track_ids": arm.get("confirmed_risk_track_ids", []),
            "clear_means": "PREDICTED_RISK_ENDED_NOT_SAFETY"}


def compact_rows(episode_id, prediction):
    previous = False
    rows = []
    for frame in prediction["frames"]:
        row = compact_frame(episode_id, frame, previous)
        rows.append(row)
        previous = row["route_risk"]
    return rows


from ue_incremental import IncrementalX73  # noqa: E402; retains the batch API above


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=detector.DEFAULT_MODEL)
    args = parser.parse_args()
    contract = load_contract(args.model_root)
    weights = args.weights.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "yolo-config"))
    import torch
    from ultralytics import YOLO
    model = YOLO(str(weights), task="segment")
    names = detector.model_names(model)
    model_hash = adapter.sha256_file(weights)
    first = contract.episodes[0].observations[0]

    def infer(observation, device):
        source = {"episode_id": observation.episode_id, "sample_index": observation.sample_index,
                  "time_s": observation.time_s, "world_frame": observation.world_frame,
                  "frame_id": f"{observation.episode_id}/{observation.sample_index:06d}"}
        return detector.infer_one(model, names, detector.ImageInput(observation.rgb.path, source, observation.rgb.sha256),
                                  model_path=weights, model_sha256=model_hash, device=device, run_kind="CANDIDATE_MATERIALIZATION")

    def candidate(device):
        return backend.BackendCandidate(device, "cuda" if device == "0" else "cpu",
            lambda: infer(first, device), lambda _: backend.DeviceObservation(str(next(model.model.parameters()).device),
            torch.cuda.get_device_name(0) if device == "0" else "CPU", "torch"),
            torch.cuda.synchronize if device == "0" else lambda: None)

    selection = backend.select_backend(backend.Workload.MODEL_INFERENCE, cpu=candidate("cpu"),
        gpu=candidate("0") if torch.cuda.is_available() else None, cpu_reason="ACCELERATOR_UNAVAILABLE",
        warmups=1, repeats=1, record_path=args.output / "backend.json")
    device = selection["selected_backend"]
    started = time.perf_counter()
    episodes, rows = {}, []
    with (args.output / "candidates.jsonl").open("x", encoding="utf-8") as handle:
        for episode in contract.episodes:
            candidates = []
            for observation in episode.observations:
                value = infer(observation, device)
                handle.write(json.dumps(value, allow_nan=False) + "\n")
                candidates.append(value)
            episodes[episode.episode_id] = predict_episode(episode, candidates, contract.calibration)
            rows.extend(compact_rows(episode.episode_id, episodes[episode.episode_id]))
            print(f"predicted {episode.episode_id}: {len(candidates)} frames", flush=True)
    result = {"schema": "ue-dtr-x73-replay-v1", "status": "COMPLETE", "arm": x73.ARM_X73,
              "source_manifest_sha256": contract.manifest_sha256, "model_sha256": model_hash,
              "claim_boundary": "Native UE synthetic Development; no deployment or fresh confirmation claim; CLEAR is only model risk ending",
              "evaluator_truth_opened": False, "elapsed_seconds": time.perf_counter() - started,
              "episodes": episodes}
    (args.output / "predictions.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "frames.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "frames": len(rows), "risk_frames": sum(row["route_risk"] for row in rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
