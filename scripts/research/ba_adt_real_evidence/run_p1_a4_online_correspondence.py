#!/usr/bin/env python3
"""Run the frozen P1-A4 strictly causal Online BootsTAPIR capability probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Any

# Keep checkout-local P1 modules ahead of an unrelated toolchain ``scripts`` package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_scripts_namespace = types.ModuleType("scripts")
_scripts_namespace.__path__ = [str(_SCRIPTS_ROOT)]
sys.modules["scripts"] = _scripts_namespace

import preflight_p1_a4_online_tapir as preflight
import run_p1_consumed_adt_baseline as r0
from materialize_p1_temporal_cohort import sha256
from scripts.research.goal_copilot_bridge.p1_persistence import baseline


TRACKER_NAME = "FROZEN_PYTORCH_ONLINE_BOOTSTAPIR_P1_A4"
PREDICTION_SCHEMA = r0.PREDICTION_SCHEMA
RESULT_SCHEMA = "blindassist_p1_a4_online_correspondence_result_v1"
CLAIM_CEILING = "CONSUMED_ADT_MECHANISM_CAPABILITY_PROBE_ONLY"
MODEL_SIZE = 256
QUERY_OFFSETS = (0.10, 0.30, 0.50, 0.70, 0.90)
QUERY_COUNT = 25
RANSAC_THRESHOLD = 3.0
MIN_VISIBLE = 6
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.50
MIN_COARSE_COVERAGE = 4
MIN_SCALE = 0.25
MAX_SCALE = 4.0


def _validate_frozen_identity(
    repository: Path, checkpoint: Path, receipt_path: Path
) -> dict[str, Any]:
    receipt = r0.read_json(receipt_path)
    if receipt.get("status") != "PYTORCH_ONLINE_BOOTSTAPIR_SELECTED":
        raise ValueError("P1-A4 selection receipt is not frozen in selected state")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != receipt["repository"]["commit"]:
        raise ValueError("official TAPIR commit drift")
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("official TAPIR checkout is dirty")
    _, manifest_hash = preflight.source_manifest(repository)
    checks = {
        "source_manifest_sha256": manifest_hash,
        "checkpoint_sha256": preflight.sha256_file(checkpoint),
        "tapir_model.py": preflight.sha256_file(repository / "tapnet/torch/tapir_model.py"),
        "pytorch_live_demo.py": preflight.sha256_file(repository / "tapnet/pytorch_live_demo.py"),
        "license_sha256": preflight.sha256_file(repository / "LICENSE"),
    }
    expected = {
        "source_manifest_sha256": receipt["repository"]["source_manifest_sha256"],
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "tapir_model.py": receipt["source_hashes"]["tapir_model.py"],
        "pytorch_live_demo.py": receipt["source_hashes"]["pytorch_live_demo.py"],
        "license_sha256": receipt["license"]["sha256"],
    }
    if checks != expected:
        raise ValueError(f"frozen implementation identity drift: {checks} != {expected}")
    return receipt


def _query_geometry(source_width: int, source_height: int, bbox: list[float]):
    import numpy as np

    scale_x = MODEL_SIZE / source_width
    scale_y = MODEL_SIZE / source_height
    x1, y1, x2, y2 = bbox
    source_xy = np.asarray(
        [
            (x1 + (x2 - x1) * ox, y1 + (y2 - y1) * oy)
            for oy in QUERY_OFFSETS
            for ox in QUERY_OFFSETS
        ],
        dtype="float32",
    )
    model_xy = source_xy * np.asarray((scale_x, scale_y), dtype="float32")
    query_tyx = np.column_stack(
        (np.zeros(QUERY_COUNT, dtype="float32"), model_xy[:, 1], model_xy[:, 0])
    )
    coarse_cells = [
        (min(2, int(oy * 3)), min(2, int(ox * 3)))
        for oy in QUERY_OFFSETS
        for ox in QUERY_OFFSETS
    ]
    return source_xy, model_xy, query_tyx, coarse_cells, scale_x, scale_y


def aggregate_points(
    initial_model_xy,
    current_model_xy,
    visibility_probability,
    visible,
    coarse_cells: list[tuple[int, int]],
    initial_bbox_model: list[float],
    source_width: int,
    source_height: int,
    scale_x: float,
    scale_y: float,
) -> tuple[dict[str, Any] | None, list[float] | None, dict[str, Any]]:
    import cv2
    import numpy as np

    visible_indices = np.flatnonzero(np.asarray(visible, dtype=bool))
    diagnostic: dict[str, Any] = {
        "visible_points": int(len(visible_indices)),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "coarse_coverage": 0,
        "scale": None,
        "rejection": None,
    }
    if len(visible_indices) < MIN_VISIBLE:
        diagnostic["rejection"] = "VISIBLE_LT_6"
        return None, None, diagnostic
    cv2.setRNGSeed(0)
    affine, mask = cv2.estimateAffinePartial2D(
        np.asarray(initial_model_xy, dtype="float32")[visible_indices],
        np.asarray(current_model_xy, dtype="float32")[visible_indices],
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or mask is None or not np.isfinite(affine).all():
        diagnostic["rejection"] = "AFFINE_UNAVAILABLE"
        return None, None, diagnostic
    inlier_local = np.flatnonzero(mask.reshape(-1).astype(bool))
    inlier_indices = visible_indices[inlier_local]
    inlier_ratio = len(inlier_indices) / len(visible_indices)
    coarse_coverage = len({coarse_cells[index] for index in inlier_indices})
    scale = math.sqrt(float(affine[0, 0]) ** 2 + float(affine[1, 0]) ** 2)
    diagnostic.update(
        inliers=int(len(inlier_indices)),
        inlier_ratio=float(inlier_ratio),
        coarse_coverage=int(coarse_coverage),
        scale=float(scale),
    )
    if len(inlier_indices) < MIN_INLIERS:
        diagnostic["rejection"] = "INLIERS_LT_6"
        return None, None, diagnostic
    if inlier_ratio < MIN_INLIER_RATIO:
        diagnostic["rejection"] = "INLIER_RATIO_LT_0_50"
        return None, None, diagnostic
    if coarse_coverage < MIN_COARSE_COVERAGE:
        diagnostic["rejection"] = "COARSE_COVERAGE_LT_4"
        return None, None, diagnostic
    if not MIN_SCALE <= scale <= MAX_SCALE:
        diagnostic["rejection"] = "SCALE_OUT_OF_RANGE"
        return None, None, diagnostic

    x1, y1, x2, y2 = initial_bbox_model
    corners = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype="float32")
    transformed = corners @ affine[:, :2].T + affine[:, 2]
    bbox_model = [
        float(transformed[:, 0].min()),
        float(transformed[:, 1].min()),
        float(transformed[:, 0].max()),
        float(transformed[:, 1].max()),
    ]
    bbox = [
        max(0.0, min(float(source_width), bbox_model[0] / scale_x)),
        max(0.0, min(float(source_height), bbox_model[1] / scale_y)),
        max(0.0, min(float(source_width), bbox_model[2] / scale_x)),
        max(0.0, min(float(source_height), bbox_model[3] / scale_y)),
    ]
    if bbox[2] - bbox[0] < 3.0 or bbox[3] - bbox[1] < 3.0:
        diagnostic["rejection"] = "BBOX_LT_3PX"
        return None, None, diagnostic
    probabilities = np.asarray(visibility_probability, dtype="float32")[inlier_indices]
    support = min(float(np.median(probabilities)), float(inlier_ratio))
    candidate = {
        "identity_support": support,
        "identity_contradiction": 1.0 - support,
        "stability": float(inlier_ratio),
        "oscillation": 0.0,
    }
    return candidate, bbox, diagnostic


class OnlineTapir:
    def __init__(self, repository: Path, checkpoint: Path):
        import torch
        from tapnet.torch import tapir_model

        if not torch.cuda.is_available():
            raise ValueError("frozen P1-A4 execution requires CUDA")
        self.torch = torch
        self.device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = self.model.to(self.device).eval()
        self.repository = repository

    def preprocess(self, bgr):
        import cv2

        rgb = cv2.cvtColor(cv2.resize(bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return (self.torch.from_numpy(rgb.copy()).to(self.device)[None, None].float() / 255 * 2) - 1

    def initialize(self, frames, query_tyx):
        import tree

        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        points = self.torch.from_numpy(query_tyx).to(self.device)[None]
        features = self.model.get_query_features(
            frames, is_training=False, query_points=points, feature_grids=feature_grids
        )
        causal = self.model.construct_initial_causal_state(
            QUERY_COUNT, len(features.resolutions) - 1
        )
        causal = tree.map_structure(lambda value: value.to(self.device), causal)
        return features, causal, feature_grids

    def predict(self, frames, features, causal, feature_grids=None):
        import torch.nn.functional as functional

        if feature_grids is None:
            feature_grids = self.model.get_feature_grids(frames, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal,
            get_causal_context=True,
        )
        tracks = trajectories["tracks"][-1][0, :, 0]
        occlusion = trajectories["occlusion"][-1][0, :, 0]
        expected = trajectories["expected_dist"][-1][0, :, 0]
        probability = (1 - functional.sigmoid(occlusion)) * (1 - functional.sigmoid(expected))
        visible = probability > 0.5
        values = tuple(value.detach().cpu().numpy() for value in (tracks, occlusion, expected, probability, visible))
        return (*values, trajectories["causal_context"])


def track_episode(public_episode: dict[str, Any], video_path: Path, tracker: OnlineTapir):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open RGB video: {video_path}")
    frames = public_episode["frames"]
    wanted = [int(frame["video_frame_index"]) for frame in frames]
    if any(right <= left for left, right in zip(wanted, wanted[1:])):
        raise ValueError(f"{public_episode['episode_id']}: non-monotonic RGB mapping")
    capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    decoded_position = wanted[0]
    initial_bbox = [float(value) for value in public_episode["initial_target_bbox_xyxy"]]
    features = causal = None
    initial_model_xy = query_tyx = coarse_cells = None
    scale_x = scale_y = None
    public_frames, bbox_records, point_frames = [], [], []
    frame_latencies: list[float] = []
    rejection_counts: Counter[str] = Counter()

    for frame_index, (frame_spec, wanted_position) in enumerate(zip(frames, wanted)):
        image = None
        while decoded_position <= wanted_position:
            ok, decoded = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"{public_episode['episode_id']}: decode failed at {decoded_position}")
            if decoded_position == wanted_position:
                image = decoded
            decoded_position += 1
        assert image is not None
        started = time.perf_counter()
        height, width = image.shape[:2]
        model_frame = tracker.preprocess(image)
        if frame_index == 0:
            _, initial_model_xy, query_tyx, coarse_cells, scale_x, scale_y = _query_geometry(
                width, height, initial_bbox
            )
            features, causal, grids = tracker.initialize(model_frame, query_tyx)
        else:
            grids = None
        tracks, occlusion, expected, probability, visible, causal = tracker.predict(
            model_frame, features, causal, grids
        )
        initial_bbox_model = [
            initial_bbox[0] * scale_x,
            initial_bbox[1] * scale_y,
            initial_bbox[2] * scale_x,
            initial_bbox[3] * scale_y,
        ]
        candidate, bbox, diagnostic = aggregate_points(
            initial_model_xy,
            tracks,
            probability,
            visible,
            coarse_cells,
            initial_bbox_model,
            width,
            height,
            scale_x,
            scale_y,
        )
        candidate_id = None
        if candidate is not None:
            candidate_id = f"frame-{frame_index}-online-tapir-candidate"
            candidate = {"candidate_id": candidate_id, **candidate}
        else:
            rejection_counts[diagnostic["rejection"]] += 1
        public_frames.append({
            "frame_index": frame_index,
            "timestamp_ms": int(frame_spec["timestamp_ms"]),
            "candidates": [] if candidate is None else [candidate],
        })
        bbox_records.append({
            "frame_index": frame_index,
            "candidate_id": candidate_id,
            "bbox_xyxy": bbox,
            "source": None if candidate is None else "online_tapir_partial_affine",
        })
        points_source = tracks / np.asarray((scale_x, scale_y), dtype="float32")
        point_frames.append({
            "frame_index": frame_index,
            "output_frame": frame_index,
            "max_source_frame_read": frame_index,
            "max_video_frame_decoded": wanted_position,
            "geometry": diagnostic,
            "points": [
                {
                    "point_id": point_id,
                    "predicted_xy_model": [float(value) for value in tracks[point_id]],
                    "predicted_xy_source": [float(value) for value in points_source[point_id]],
                    "occlusion_logit": float(occlusion[point_id]),
                    "expected_distance_logit": float(expected[point_id]),
                    "visibility_probability": float(probability[point_id]),
                    "visible": bool(visible[point_id]),
                    "occluded": not bool(visible[point_id]),
                }
                for point_id in range(QUERY_COUNT)
            ],
        })
        tracker.torch.cuda.synchronize(tracker.device)
        frame_latencies.append(time.perf_counter() - started)
    capture.release()

    p1_input = {
        "schema_version": 1,
        "protocol_id": baseline.PROTOCOL_ID,
        "episode_id": public_episode["episode_id"],
        "handoff": public_episode["handoff"],
        "frames": public_frames,
    }
    output = baseline.run_baseline(p1_input)
    return {
        "episode_id": public_episode["episode_id"],
        "p1_output": output,
        "candidate_bboxes": bbox_records,
        "point_trace": point_frames,
        "mechanics": {
            "frames": len(frames),
            "point_rows": len(frames) * QUERY_COUNT,
            "candidate_frames": sum(record["candidate_id"] is not None for record in bbox_records),
            "geometry_rejections": dict(sorted(rejection_counts.items())),
            "latency_median_ms": statistics.median(frame_latencies) * 1000,
            "latency_p95_ms": sorted(frame_latencies)[min(len(frame_latencies) - 1, math.ceil(len(frame_latencies) * 0.95) - 1)] * 1000,
        },
        "oracle_initializations": 1,
        "post_initialization_gt_reads": 0,
        "causal_violations": 0,
    }


def run_tracker(
    public_path: Path,
    prediction_path: Path,
    repository: Path,
    checkpoint: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    public = r0.validate_public_input(r0.read_json(public_path))
    receipt = _validate_frozen_identity(repository, checkpoint, receipt_path)
    sys.path.insert(0, str(repository))
    import torch

    if torch.__version__ != receipt["runtime"]["torch"]:
        raise ValueError("frozen PyTorch runtime drift")
    if torch.version.cuda != receipt["runtime"]["cuda_runtime"]:
        raise ValueError("frozen CUDA runtime drift")
    if torch.cuda.get_device_name(0) != receipt["runtime"]["gpu"]:
        raise ValueError("frozen GPU identity drift")
    tracker = OnlineTapir(repository, checkpoint)
    started = time.perf_counter()
    episodes = []
    for source in public["sources"]:
        video_path = Path(source["rgb_video_path"])
        if sha256(video_path) != source["rgb_video_sha256"]:
            raise ValueError(f"RGB hash drift: {source['source_sequence_id']}")
        for episode in source["episodes"]:
            episodes.append(track_episode(episode, video_path, tracker))
            print(json.dumps({"completed_episode": episode["episode_id"], "completed": len(episodes), "total": 15}), flush=True)
    total_frames = sum(item["mechanics"]["frames"] for item in episodes)
    prediction = {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "tracker": TRACKER_NAME,
        "public_input_sha256": r0.object_sha256(public),
        "frozen_selection_receipt_sha256": sha256(receipt_path),
        "truth_access": {
            "oracle_initializations": sum(item["oracle_initializations"] for item in episodes),
            "post_initialization_gt_reads": 0,
        },
        "authority_receipt": {
            "future_frame_reads": 0,
            "gt_oracle_resets": 0,
            "object_uid_or_visibility_gt_reads": 0,
            "semantic_detector_reid_vlm_reads": 0,
            "global_target_searches": 0,
            "online_query_feature_replacements": 0,
            "causal_violations": sum(item["causal_violations"] for item in episodes),
            "frames": total_frames,
            "point_rows": sum(item["mechanics"]["point_rows"] for item in episodes),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(tracker.device)),
        },
        "episodes": episodes,
    }
    r0.write_json(prediction_path, prediction)
    return prediction


def _gate_result(result: dict[str, Any], prediction: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    aggregate = result["evaluation"]["aggregate"]
    attribution = result["post_outcome_descriptive_failure_attribution"]
    temp = result["by_temporal_mode"].get("TEMP_OCCLUSION", {})
    returned = result["by_temporal_mode"].get("OUT_OF_VIEW_RETURN", {})
    receipt = prediction["authority_receipt"]
    hard = {
        "authority_violations_zero": all(
            receipt[key] == 0
            for key in (
                "future_frame_reads",
                "gt_oracle_resets",
                "object_uid_or_visibility_gt_reads",
                "semantic_detector_reid_vlm_reads",
                "global_target_searches",
                "online_query_feature_replacements",
                "causal_violations",
            )
        ),
        "frame_count_1724": receipt["frames"] == 1724,
        "point_rows_43100": receipt["point_rows"] == QUERY_COUNT * 1724,
        "state_violations_zero": aggregate["state_expectation_violations"] == 0,
        "event_violations_zero": aggregate["event_expectation_violations"] == 0,
    }
    coverage_numerator = aggregate["correct_identity_coverage"]["numerator"]
    temp_recoveries = temp.get("recovery_successes", 0)
    return_recoveries = returned.get("recovery_successes", 0)
    safety = {
        "wrong_asserted_frames_le_445": aggregate["wrong_instance_asserted_frames"] <= 445,
        "background_wrong_le_422": attribution["wrong_background_asserted_frames"] <= 422,
        "identity_switches_le_27": aggregate["identity_switches"] <= 27,
        "max_wrong_lock_ms_le_2700": aggregate["wrong_lock_persistence_max_duration_ms"] <= 2700,
        "false_reacquisitions_le_29": aggregate["false_reacquisitions"] <= 29,
    }
    utility = {
        "correct_numerator_ge_120": coverage_numerator >= 120,
        "false_loss_frames_le_152": aggregate["false_loss_frames"] <= 152,
        "temporary_or_return_recovery_ge_1": temp_recoveries >= 1 or return_recoveries >= 1,
    }
    if not all(hard.values()):
        terminal = "NOT_EVALUABLE_ONLINE_INTERFACE"
    elif all(safety.values()) and all(utility.values()):
        terminal = "STRONG_TEMPORAL_CORRESPONDENCE_SIGNAL_ESTABLISHED"
    elif coverage_numerator >= 120 and (temp_recoveries >= 1 or return_recoveries >= 1) and not all(safety.values()):
        terminal = "CORRESPONDENCE_COVERAGE_GAIN_WITH_IDENTITY_SAFETY_FAILURE"
    elif all(safety.values()) and coverage_numerator < 120:
        terminal = "STRONG_CORRESPONDENCE_GAIN_ONLY_BY_ABSTENTION"
    else:
        terminal = "STRONG_TEMPORAL_CORRESPONDENCE_NOT_SUFFICIENT"
    return terminal, {
        "hard_evaluability": hard,
        "identity_safety": safety,
        "persistence_utility": utility,
        "observed": {
            "correct_identity_coverage_numerator": coverage_numerator,
            "wrong_asserted_frames": aggregate["wrong_instance_asserted_frames"],
            "background_wrong": attribution["wrong_background_asserted_frames"],
            "identity_switches": aggregate["identity_switches"],
            "max_wrong_lock_ms": aggregate["wrong_lock_persistence_max_duration_ms"],
            "false_reacquisitions": aggregate["false_reacquisitions"],
            "false_loss_frames": aggregate["false_loss_frames"],
            "temporary_occlusion_recoveries": temp_recoveries,
            "out_of_view_return_recoveries": return_recoveries,
        },
    }


def evaluate(private_path: Path, prediction_path: Path, result_path: Path):
    temporary = result_path.with_suffix(".r0.tmp.json")
    r0.evaluate_predictions(private_path, prediction_path, temporary)
    result = r0.read_json(temporary)
    temporary.unlink()
    prediction = r0.read_json(prediction_path)
    terminal, gates = _gate_result(result, prediction)
    result.update({
        "schema_version": RESULT_SCHEMA,
        "tracker": TRACKER_NAME,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": terminal,
        "terminal_suffix": "NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT",
        "frozen_gates": gates,
        "online_authority_receipt": prediction["authority_receipt"],
        "runtime": prediction["runtime"],
        "frozen_selection_receipt_sha256": prediction["frozen_selection_receipt_sha256"],
    })
    r0.write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    track = subparsers.add_parser("track")
    track.add_argument("--public-input", required=True, type=Path)
    track.add_argument("--prediction", required=True, type=Path)
    track.add_argument("--repository", required=True, type=Path)
    track.add_argument("--checkpoint", required=True, type=Path)
    track.add_argument("--selection-receipt", required=True, type=Path)
    adjudicate = subparsers.add_parser("evaluate")
    adjudicate.add_argument("--private-input", required=True, type=Path)
    adjudicate.add_argument("--prediction", required=True, type=Path)
    adjudicate.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "track":
        prediction = run_tracker(
            args.public_input,
            args.prediction,
            args.repository.resolve(),
            args.checkpoint.resolve(),
            args.selection_receipt.resolve(),
        )
        print(json.dumps({"episodes": len(prediction["episodes"]), "authority_receipt": prediction["authority_receipt"], "runtime": prediction["runtime"]}, indent=2))
    else:
        result = evaluate(args.private_input, args.prediction, args.result)
        print(json.dumps({"terminal": result["terminal"], "gates": result["frozen_gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
