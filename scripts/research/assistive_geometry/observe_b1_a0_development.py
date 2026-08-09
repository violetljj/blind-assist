#!/usr/bin/env python3
"""Run A0 depth checkpoints and frozen geometry postprocess on Development Selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    derive_assistive_truth,
    rotate_array_upright,
)
from scripts.research.assistive_geometry.assistive_geometry_training import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
)
from scripts.research.assistive_geometry.evaluate_b1_a0_synthetic import (  # noqa: E402
    EXPECTED_SEEDS,
    atomic_write_json,
    utc_now,
)
from scripts.research.assistive_geometry.train_b1_a0_formal import (  # noqa: E402
    atomic_write_json as replace_json,
    load_depthart_model,
)


BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
ROLE = "DEVELOPMENT_SELECTION"
TASK_HORIZON_M = 2.0


def flatten_manifest(manifest: dict[str, Any], expected: list[dict[str, str]]) -> list[dict[str, Any]]:
    require(
        manifest.get("schema") == "blindassist_assistive_geometry_b1_development_target_manifest_v1",
        "Development target manifest schema drift",
    )
    require(manifest.get("data_role") == ROLE, "Development target role drift")
    require(manifest.get("development_content_opened") is True, "Development target activation missing")
    require(manifest.get("development_calibration_content_opened") is False, "Development Calibration was opened")
    require(manifest.get("confirmation_content_opened") is False, "Confirmation was opened")
    videos = manifest["videos"]
    require(len(videos) == 4, "Development Selection video count drift")
    identities = [(str(video["visit_id"]), str(video["video_id"])) for video in videos]
    frozen = [(str(row["visit_id"]), str(row["video_id"])) for row in expected]
    require(identities == frozen, "Development Selection identity/order drift")
    frames: list[dict[str, Any]] = []
    for video in videos:
        require(video.get("evaluation_role") == ROLE, "cross-role Development video")
        for frame in video["frames"]:
            frames.append({**frame, "visit_id": str(video["visit_id"]), "video_id": str(video["video_id"])})
    require(len(frames) == 1200, "Development Selection frame count drift")
    return frames


def _state(value: bool | None) -> str:
    if value is None:
        return "UNKNOWN"
    return "OCCUPIED_OBSERVED" if value else "CLEAR_OBSERVED"


def band_observation(
    name: str,
    truth_clearance: float,
    truth_clearance_valid: bool,
    truth_occupancy: np.ndarray,
    truth_occupancy_valid: np.ndarray,
    predicted: dict[str, Any] | None,
) -> dict[str, Any]:
    predicted_occupied = (predicted or {}).get("occupied_by_horizon", {})
    predicted_clearance = (predicted or {}).get("clearance_m")
    known_predicted_states = [predicted_occupied.get(str(value)) for value in HORIZONS]
    if predicted_clearance is not None and np.isfinite(predicted_clearance):
        predicted_clearance_valid = True
        predicted_clearance_value: float | None = min(float(predicted_clearance), TASK_HORIZON_M)
    elif all(value is False for value in known_predicted_states):
        predicted_clearance_valid = True
        predicted_clearance_value = TASK_HORIZON_M
    else:
        predicted_clearance_valid = False
        predicted_clearance_value = None
    cells = []
    for index, horizon in enumerate(HORIZONS):
        truth_value = bool(truth_occupancy[index]) if bool(truth_occupancy_valid[index]) else None
        cells.append(
            {
                "horizon_m": horizon,
                "truth_state": _state(truth_value),
                "predicted_state": _state(predicted_occupied.get(str(horizon))),
            }
        )
    return {
        "band": name,
        "truth_clearance_valid": bool(truth_clearance_valid),
        "truth_clearance_m": min(float(truth_clearance), TASK_HORIZON_M) if truth_clearance_valid else None,
        "predicted_clearance_valid": predicted_clearance_valid,
        "predicted_clearance_m": predicted_clearance_value,
        "cells": cells,
    }


def load_frame(frame: dict[str, Any]) -> dict[str, Any]:
    with np.load(Path(frame["target"]["path"]), allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    orientation = int(values["orientation_index"].item())
    target_height, target_width = (int(value) for value in values["target_hw"])
    image_bgr = cv2.imread(str(frame["rgb_source"]["path"]), cv2.IMREAD_COLOR)
    require(image_bgr is not None, f"RGB decode failed: {frame['rgb_source']['path']}")
    image_rgb_source = cv2.cvtColor(rotate_array_upright(image_bgr, orientation), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb_source, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    image = image_rgb.astype(np.float32) / 255.0
    normalized = ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
    gray = cv2.cvtColor(image_rgb_source, cv2.COLOR_RGB2GRAY)
    low_light = float(gray.mean()) < 50.0
    blurred = float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 50.0
    truth_occupancy = values["occupancy"].astype(np.bool_)
    truth_occupancy_valid = values["occupancy_valid"].astype(np.bool_)
    return {
        "image": torch.from_numpy(normalized),
        "intrinsics": torch.from_numpy(values["intrinsics_tensor"].astype(np.float32, copy=True)),
        "up_camera": values["up_camera"].astype(np.float64, copy=True),
        "truth_clearance": values["clearance_m"].astype(np.float64, copy=True),
        "truth_clearance_valid": values["clearance_valid"].astype(np.bool_, copy=True),
        "truth_occupancy": truth_occupancy,
        "truth_occupancy_valid": truth_occupancy_valid,
        "truth_ground_valid": bool(values["ground_plane_valid"].item()),
        "orientation": "portrait" if orientation in (1, 3) else "landscape",
        "low_light_blur": bool(low_light or blurred),
        "near_field": bool(np.any(truth_occupancy & truth_occupancy_valid)),
        "frame": frame,
    }


def observation_row(sample: dict[str, Any], predicted_depth: np.ndarray, seed: int) -> dict[str, Any]:
    confidence = np.full(predicted_depth.shape, 2, dtype=np.uint8)
    predicted = derive_assistive_truth(
        predicted_depth,
        confidence,
        sample["intrinsics"].numpy(),
        sample["up_camera"],
        TruthReaderPolicy(),
    )
    bands = []
    for index, name in enumerate(BANDS):
        bands.append(
            band_observation(
                name,
                sample["truth_clearance"][index],
                sample["truth_clearance_valid"][index],
                sample["truth_occupancy"][index],
                sample["truth_occupancy_valid"][index],
                predicted.get("bands", {}).get(name),
            )
        )
    frame = sample["frame"]
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_development_frame_v1",
        "seed": seed,
        "data_role": ROLE,
        "parent_id": frame["visit_id"],
        "session_id": frame["video_id"],
        "frame_id": frame["frame_stem"],
        "sequence_index": int(frame["frame_index"]),
        "orientation": sample["orientation"],
        "environment": "indoor_arkitscenes",
        "near_field": sample["near_field"],
        "low_light_blur": sample["low_light_blur"],
        "truth_ground_valid": sample["truth_ground_valid"],
        "predicted_ground_valid": predicted.get("ground_plane") is not None,
        "bands": bands,
    }


def _write_progress(path: Path, completed: int, total: int, started: float, phase: str, status: str) -> None:
    elapsed = max(time.perf_counter() - started, 0.0)
    throughput = completed / elapsed if completed and elapsed else 0.0
    replace_json(
        path,
        {
            "phase": phase,
            "completed_units": completed,
            "total_units": total,
            "throughput": throughput,
            "eta_seconds": (total - completed) / throughput if throughput and completed < total else 0.0,
            "last_progress_at": utc_now(),
            "status": status,
        },
    )


def _final_checkpoint(result_path: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    result = load_json(result_path)
    require(result.get("terminal") == "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE", f"seed {seed} train result incomplete")
    require(int(result.get("seed", -1)) == seed, f"seed {seed} result identity drift")
    receipts = [row for row in result["checkpoints"] if int(row["epoch"]) == 20]
    require(len(receipts) == 1, f"seed {seed} final checkpoint receipt missing")
    path = Path(receipts[0]["path"])
    require(path.is_file() and path.stat().st_size == int(receipts[0]["bytes"]), f"seed {seed} final checkpoint size drift")
    require(sha256_file(path) == receipts[0]["sha256"], f"seed {seed} final checkpoint SHA drift")
    return path, result


def execute(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == "blindassist_assistive_geometry_b1_a0_development_evaluation_protocol_v1", "Development protocol schema drift")
    require(protocol["authority"].get("development_selection_evaluation") is True, "Development evaluation not activated")
    observer_binding = protocol["implementation_bindings"]["observation_runner"]
    require(observer_binding["sha256"] == sha256_file(Path(__file__)), "observation runner binding drift")
    manifest_path = args.target_manifest.resolve()
    manifest = load_json(manifest_path)
    require(manifest.get("protocol_sha256") == sha256_file(protocol_path), "Development target/protocol binding drift")
    frames = flatten_manifest(manifest, protocol["data_role"]["identities"])
    source = args.source.resolve()
    require(source == Path(protocol["inputs"]["depthart_source_root"]).resolve(), "DepthART source root drift")
    initialization = Path(protocol["inputs"]["initialization_checkpoint"]["path"])
    require(sha256_file(initialization) == protocol["inputs"]["initialization_checkpoint"]["sha256"], "DepthART initialization drift")
    output_root = args.output_root.resolve()
    require(not output_root.exists(), "Development observation output already exists")
    output_root.mkdir(parents=True)
    progress_path = output_root / "progress.json"
    failure_path = output_root / "failure.json"
    started = time.perf_counter()
    total = len(frames) * len(EXPECTED_SEEDS)
    completed = 0
    _write_progress(progress_path, 0, total, started, "initializing", "running")
    atomic_write_json(
        output_root / "activation.json",
        {
            "schema": "blindassist_assistive_geometry_b1_a0_development_activation_v1",
            "protocol_sha256": sha256_file(protocol_path),
            "data_role": ROLE,
            "development_content_opened": True,
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
            "activated_at": utc_now(),
        },
    )
    try:
        device = torch.device("cuda")
        require(torch.cuda.is_available(), "A0 Development inference requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seed_runs: list[dict[str, Any]] = []
        for seed in EXPECTED_SEEDS:
            result_path = Path(protocol["training_outputs"][str(seed)]["result_path"])
            checkpoint_path, train_result = _final_checkpoint(result_path, seed)
            model, _ = load_depthart_model(source, initialization, seed, device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            require(checkpoint.get("schema") == "blindassist_assistive_geometry_b1_a0_checkpoint_v1", "checkpoint schema drift")
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            rows_by_index: dict[int, dict[str, Any]] = {}
            observations_partial = output_root / f"seed-{seed}-observations.jsonl.partial"
            observations_path = output_root / f"seed-{seed}-observations.jsonl"
            require(not observations_partial.exists() and not observations_path.exists(), "observation output collision")
            for orientation in ("portrait", "landscape"):
                indices = [index for index, frame in enumerate(frames) if frame["orientation_family"] == orientation]
                for start in range(0, len(indices), 4):
                    selected = indices[start : start + 4]
                    samples = [load_frame(frames[index]) for index in selected]
                    images = torch.stack([sample["image"] for sample in samples]).to(device)
                    intrinsics = torch.stack([sample["intrinsics"] for sample in samples]).to(device)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        depths = model(images, intrinsics)["dense_depth_m"][:, 0].float().cpu().numpy()
                    for index, sample, depth in zip(selected, samples, depths, strict=True):
                        rows_by_index[index] = observation_row(sample, depth, seed)
                        completed += 1
                    if completed % 20 == 0:
                        _write_progress(progress_path, completed, total, started, f"seed-{seed}-inference", "running")
            require(len(rows_by_index) == len(frames), f"seed {seed} observation coverage drift")
            with observations_partial.open("x", encoding="utf-8", newline="\n") as stream:
                for index in range(len(frames)):
                    stream.write(json.dumps(rows_by_index[index], ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(observations_partial, observations_path)
            seed_runs.append(
                {
                    "seed": seed,
                    "train_result_path": str(result_path.resolve()),
                    "observations_path": str(observations_path.resolve()),
                    "observations_sha256": sha256_file(observations_path),
                    "observation_count": len(frames),
                    "final_checkpoint_sha256": sha256_file(checkpoint_path),
                    "final_model_state_sha256": train_result["final_model_state_sha256"],
                }
            )
            del model, checkpoint
            torch.cuda.empty_cache()
        package = {
            "schema": "blindassist_assistive_geometry_b1_a0_development_evaluation_package_v1",
            "package_id": protocol["protocol_id"],
            "data_role": ROLE,
            "evaluation_protocol_sha256": sha256_file(protocol_path),
            "training_protocol_path": str(Path(protocol["bindings"]["formal_train_protocol"]["path"]).resolve()),
            "training_protocol_sha256": protocol["bindings"]["formal_train_protocol"]["sha256"],
            "target_manifest_path": str(manifest_path),
            "target_manifest_sha256": sha256_file(manifest_path),
            "seed_runs": seed_runs,
            "development_content_opened": True,
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
        }
        package_path = output_root / "evaluation_package.json"
        atomic_write_json(package_path, package)
        result = {
            "schema": "blindassist_assistive_geometry_b1_a0_development_observation_result_v1",
            "terminal": "B1_A0_DEVELOPMENT_OBSERVATIONS_COMPLETE",
            "protocol_sha256": sha256_file(protocol_path),
            "package_path": str(package_path),
            "package_sha256": sha256_file(package_path),
            "seed_runs": seed_runs,
            "completed_units": completed,
            "development_content_opened": True,
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
            "completed_at": utc_now(),
        }
        atomic_write_json(output_root / "result.json", result)
        _write_progress(progress_path, total, total, started, "complete", "complete")
        print(json.dumps({key: value for key, value in result.items() if key != "seed_runs"}, indent=2))
        return 0
    except Exception as error:
        atomic_write_json(
            failure_path,
            {
                "schema": "blindassist_assistive_geometry_b1_a0_development_observation_failure_v1",
                "terminal": "B1_A0_DEVELOPMENT_OBSERVATION_FAILED",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "completed_units": completed,
                "development_content_opened": True,
                "confirmation_content_opened": False,
                "failed_at": utc_now(),
            },
        )
        _write_progress(progress_path, completed, total, started, "failed", "failed")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
