#!/usr/bin/env python3
"""Run P1-A2 fixed-reference dense-identity validity discovery."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics
import sys
import types
from typing import Any

# The CUDA toolchain contains an unrelated top-level ``scripts`` package. Bind
# this checkout's namespace before importing the frozen P1 modules.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_scripts_namespace = types.ModuleType("scripts")
_scripts_namespace.__path__ = [str(_SCRIPTS_ROOT)]
sys.modules["scripts"] = _scripts_namespace

import run_p1_a1_local_validity as a1
import run_p1_consumed_adt_baseline as r0
from materialize_p1_temporal_cohort import sha256
from scripts.research.goal_copilot_bridge.p1_persistence import baseline


TRACE_SCHEMA = "blindassist_p1_a2_fixed_reference_dense_identity_trace_v1"
SWEEP_SCHEMA = "blindassist_p1_a2_fixed_reference_dense_identity_sweep_v1"
MODEL_REPOSITORY = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_FILES = {
    "model.safetensors": "AE1E99FCEFD534ED978CDEB8326F08030C96E28B7A81FFCBC98A857C84D14BE1",
    "config.json": "1809F83E3BDB1609A501A610AD4A742F4FD8AE44D72CA4AA0DF52D1F2AC8628D",
    "preprocessor_config.json": "14E780D86FA1861F8751F868D7F45425B5FEB55C38CA26F152CA5097AB30F828",
}
INPUT_SIZE = 224
PATCH_SIDE = 16
PATCH_COUNT = PATCH_SIDE * PATCH_SIDE
FEATURE_DIM = 384
BATCH_SIZE = 16
POLICY_FEATURES = (
    "anchor_match_fraction",
    "match_confidence",
    "spatial_consistency",
    "anchor_coverage",
)
POLICY_QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)
RETENTION_MIN = 0.90
WRONG_REDUCTION_MIN = 0.60
WRONG_LOCK_REDUCTION_MIN = 0.60


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()


def _validate_model(model_dir: Path) -> dict[str, Any]:
    observed = {}
    for name, expected in MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"missing frozen model file: {path}")
        actual = sha256(path).upper()
        if actual != expected:
            raise ValueError(f"frozen model hash drift for {name}: {actual} != {expected}")
        observed[name] = actual
    config = r0.read_json(model_dir / "config.json")
    expected_config = {"model_type": "dinov2", "hidden_size": FEATURE_DIM, "patch_size": 14}
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"frozen model config drift for {key}: {config.get(key)!r} != {expected!r}")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "files": observed,
        "input_size": INPUT_SIZE,
        "patch_grid": [PATCH_SIDE, PATCH_SIDE],
        "feature_dim": FEATURE_DIM,
        "layer": "last_hidden_state_patch_tokens",
    }


def _crop_tensor(image, bbox: list[float]):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(math.floor(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox[3])))))
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        raise ValueError(f"degenerate candidate crop: {bbox}")
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    return np.transpose((rgb - mean) / std, (2, 0, 1))


class DenseEncoder:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("frozen A2 execution requested CUDA but CUDA is unavailable")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(self.device).eval()
        self.forward_batches = 0
        self.encoded_crops = 0

    def encode(self, tensors: list[Any]) -> list[Any]:
        torch = self.torch
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(tensors), BATCH_SIZE):
                batch = torch.from_numpy(__import__("numpy").stack(tensors[start:start + BATCH_SIZE])).to(self.device)
                output = self.model(pixel_values=batch).last_hidden_state[:, 1:, :]
                if tuple(output.shape[1:]) != (PATCH_COUNT, FEATURE_DIM):
                    raise ValueError(f"unexpected DINOv2 patch output: {tuple(output.shape)}")
                output = torch.nn.functional.normalize(output.float(), dim=-1)
                encoded.extend(output.cpu().numpy())
                self.forward_batches += 1
                self.encoded_crops += int(batch.shape[0])
        return encoded


def dense_consensus(initial_patches, current_patches) -> dict[str, float]:
    import cv2
    import numpy as np

    initial = np.asarray(initial_patches, dtype="float32")
    current = np.asarray(current_patches, dtype="float32")
    if initial.shape != (PATCH_COUNT, FEATURE_DIM) or current.shape != initial.shape:
        raise ValueError(f"dense feature shape drift: {initial.shape} / {current.shape}")
    similarities = initial @ current.T
    current_for_anchor = similarities.argmax(axis=1)
    anchor_for_current = similarities.argmax(axis=0)
    anchor_indices = np.arange(PATCH_COUNT)
    mutual_mask = anchor_for_current[current_for_anchor] == anchor_indices
    matched_anchor = anchor_indices[mutual_mask]
    matched_current = current_for_anchor[mutual_mask]
    match_similarities = similarities[matched_anchor, matched_current]

    if len(matched_anchor):
        anchor_y, anchor_x = np.divmod(matched_anchor, PATCH_SIDE)
        current_y, current_x = np.divmod(matched_current, PATCH_SIDE)
        anchor_xy = np.stack([anchor_x, anchor_y], axis=1).astype("float32")
        current_xy = np.stack([current_x, current_y], axis=1).astype("float32")
        bins = set((int(y) // 4, int(x) // 4) for y, x in zip(anchor_y, anchor_x))
        coverage = len(bins) / 16.0
        confidence = float(np.median(match_similarities))
    else:
        anchor_xy = np.empty((0, 2), dtype="float32")
        current_xy = np.empty((0, 2), dtype="float32")
        coverage = 0.0
        confidence = -1.0

    affine = None
    inlier_mask = None
    if len(anchor_xy) >= 3:
        cv2.setRNGSeed(0)
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            anchor_xy,
            current_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    if affine is None or inlier_mask is None:
        spatial_consistency = 0.0
        dispersion = 1.0
    else:
        spatial_consistency = float(inlier_mask.mean())
        predicted = anchor_xy @ affine[:, :2].T + affine[:, 2]
        residuals = np.linalg.norm(predicted - current_xy, axis=1)
        dispersion = float(np.median(residuals) / math.hypot(PATCH_SIDE - 1, PATCH_SIDE - 1))

    global_initial = initial.mean(axis=0)
    global_current = current.mean(axis=0)
    global_cosine = float(
        np.dot(global_initial, global_current)
        / max(1e-12, float(np.linalg.norm(global_initial) * np.linalg.norm(global_current)))
    )
    result = {
        "anchor_match_fraction": float(len(matched_anchor) / PATCH_COUNT),
        "match_confidence": confidence,
        "spatial_consistency": spatial_consistency,
        "anchor_coverage": coverage,
        "correspondence_dispersion": dispersion,
        "global_embedding_cosine": global_cosine,
        "mutual_match_count": int(len(matched_anchor)),
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError(f"non-finite dense consensus: {result}")
    return result


def extract_trace(
    public_path: Path,
    a1_trace_path: Path,
    sealed_prediction_path: Path,
    model_dir: Path,
    trace_path: Path,
    device: str,
) -> dict[str, Any]:
    public = r0.validate_public_input(r0.read_json(public_path))
    a1_trace = r0.read_json(a1_trace_path)
    sealed = r0.read_json(sealed_prediction_path)
    if a1_trace.get("instrumentation_parity") != "PASS" or a1_trace.get("post_initialization_gt_reads") != 0:
        raise ValueError("A1 candidate trace is not parity-qualified")
    if a1_trace["sealed_prediction_sha256"] != sha256(sealed_prediction_path):
        raise ValueError("A1 trace / sealed prediction binding drift")
    if sealed["public_input_sha256"] != r0.object_sha256(public):
        raise ValueError("public input / sealed prediction binding drift")
    model_identity = _validate_model(model_dir)
    encoder = DenseEncoder(model_dir, device=device)
    a1_by_id = {episode["episode_id"]: episode for episode in a1_trace["episodes"]}
    episodes = []

    for source in public["sources"]:
        video_path = Path(source["rgb_video_path"])
        if sha256(video_path) != source["rgb_video_sha256"]:
            raise ValueError(f"RGB hash drift: {source['source_sequence_id']}")
        for public_episode in source["episodes"]:
            episode_id = public_episode["episode_id"]
            candidate_episode = a1_by_id[episode_id]
            images = a1._decode_episode_frames(video_path, public_episode)
            records = candidate_episode["candidate_bboxes"]
            if len(images) != len(records):
                raise ValueError(f"{episode_id}: frame / candidate trace length drift")
            initial_box = [float(value) for value in public_episode["initial_target_bbox_xyxy"]]
            tensors = [_crop_tensor(images[0], initial_box)]
            candidate_ids = []
            for image, record in zip(images, records):
                if record["source"] != "sparse_lk_flow":
                    continue
                if record["candidate_id"] is None or record["bbox_xyxy"] is None:
                    raise ValueError(f"{episode_id}: malformed sparse-flow candidate record")
                candidate_ids.append(record["candidate_id"])
                tensors.append(_crop_tensor(image, record["bbox_xyxy"]))
            features = encoder.encode(tensors)
            initial_patches = features[0]
            identity_by_candidate = {
                candidate_id: dense_consensus(initial_patches, current_patches)
                for candidate_id, current_patches in zip(candidate_ids, features[1:])
            }
            episodes.append({
                "episode_id": episode_id,
                "p1_input": candidate_episode["p1_input"],
                "p1_output": candidate_episode["p1_output"],
                "candidate_bboxes": records,
                "identity_by_candidate": identity_by_candidate,
                "target_identity_memory": {
                    "initial_frame_index": 0,
                    "initial_bbox_xyxy": initial_box,
                    "patch_count": PATCH_COUNT,
                    "feature_dim": FEATURE_DIM,
                    "online_updates": 0,
                },
            })

    trace = {
        "schema_version": TRACE_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_RGB_ONLY_DENSE_IDENTITY_INSTRUMENTATION",
        "public_input_sha256": sealed["public_input_sha256"],
        "sealed_prediction_sha256": sha256(sealed_prediction_path),
        "a1_candidate_trace_sha256": sha256(a1_trace_path),
        "candidate_generator": r0.TRACKER_NAME,
        "candidate_generator_changed": False,
        "candidate_parity": "PASS_BY_BOUND_A1_TRACE",
        "post_initialization_gt_reads": 0,
        "global_search": False,
        "reacquisition_added": False,
        "online_target_memory_updates": 0,
        "model": model_identity,
        "runtime": {
            "device": device,
            "encoder_forward_batches": encoder.forward_batches,
            "encoded_crops": encoder.encoded_crops,
            "runner_sha256": _source_sha256(),
        },
        "episodes": episodes,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    r0.write_json(trace_path, trace)
    return trace


def _quantile_grid(trace: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    import numpy as np

    values = {feature: [] for feature in POLICY_FEATURES}
    for episode in trace["episodes"]:
        for consensus in episode["identity_by_candidate"].values():
            for feature in POLICY_FEATURES:
                values[feature].append(float(consensus[feature]))
    if not all(values.values()):
        raise ValueError("dense identity trace contains no policy evidence")
    return {
        feature: [
            {"quantile": quantile, "threshold": float(np.quantile(feature_values, quantile))}
            for quantile in POLICY_QUANTILES
        ]
        for feature, feature_values in values.items()
    }


def _policy_family(grid: dict[str, list[dict[str, float]]]):
    for rows in product(*(grid[feature] for feature in POLICY_FEATURES)):
        yield [
            {"feature": feature, "op": "ge", "quantile": row["quantile"], "threshold": row["threshold"]}
            for feature, row in zip(POLICY_FEATURES, rows)
        ]


def _decision(consensus: dict[str, float], predicates: list[dict[str, Any]]) -> str:
    passed = sum(float(consensus[predicate["feature"]]) >= float(predicate["threshold"]) for predicate in predicates)
    if passed == len(predicates):
        return "VALID"
    if passed == len(predicates) - 1:
        return "UNCERTAIN"
    return "INVALID"


def _gated_output(episode: dict[str, Any], predicates: list[dict[str, Any]]):
    gated_input = json.loads(json.dumps(episode["p1_input"]))
    evidence = episode["identity_by_candidate"]
    for frame in gated_input["frames"]:
        kept = []
        for candidate in frame["candidates"]:
            consensus = evidence.get(candidate["candidate_id"])
            if consensus is None or _decision(consensus, predicates) == "VALID":
                kept.append(candidate)
        frame["candidates"] = kept
    return baseline.run_baseline(gated_input)


def _baseline_counts(trace: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    return a1._baseline_counts(trace, labels)


def _score_policy(
    trace: dict[str, Any], labels: dict[str, Any], predicates: list[dict[str, Any]], baseline_counts: dict[str, Any]
) -> dict[str, Any]:
    correct = wrong = background_wrong = other_wrong = 0
    max_wrong_frames = max_wrong_ms = 0
    per_episode_wrong = {}
    outputs_by_episode = {}
    decisions_by_episode = {}
    for episode in trace["episodes"]:
        episode_id = episode["episode_id"]
        output = _gated_output(episode, predicates)
        outputs_by_episode[episode_id] = output
        episode_wrong = 0
        decisions = []
        for input_frame in episode["p1_input"]["frames"]:
            if not input_frame["candidates"]:
                decisions.append("NO_CANDIDATE")
                continue
            candidate_id = input_frame["candidates"][0]["candidate_id"]
            consensus = episode["identity_by_candidate"].get(candidate_id)
            decisions.append("VALID" if consensus is None else _decision(consensus, predicates))
        decisions_by_episode[episode_id] = decisions
        for output_frame, label in zip(output["frames"], labels[episode_id]):
            if output_frame["current_candidate_id"] is None:
                continue
            if label["identity_class"] == "CORRECT":
                correct += 1
            elif label["identity_class"] == "BACKGROUND_DRIFT":
                wrong += 1
                background_wrong += 1
                episode_wrong += 1
            elif label["identity_class"] == "OTHER_INSTANCE":
                wrong += 1
                other_wrong += 1
                episode_wrong += 1
        per_episode_wrong[episode_id] = episode_wrong
        frames, duration = a1._wrong_run(
            output["frames"],
            labels[episode_id],
            [int(frame["timestamp_ms"]) for frame in episode["p1_input"]["frames"]],
        )
        max_wrong_frames = max(max_wrong_frames, frames)
        max_wrong_ms = max(max_wrong_ms, duration)

    episode_reductions = [
        (baseline_wrong - per_episode_wrong[episode_id]) / baseline_wrong
        for episode_id, baseline_wrong in baseline_counts["per_episode_wrong"].items()
        if baseline_wrong > 0
    ]
    retention = correct / baseline_counts["correct"]
    wrong_reduction = (baseline_counts["wrong"] - wrong) / baseline_counts["wrong"]
    lock_reduction = (baseline_counts["max_wrong_ms"] - max_wrong_ms) / baseline_counts["max_wrong_ms"]
    canonical = " AND ".join(
        f"{predicate['feature']} ge q{int(predicate['quantile'] * 100):02d}={predicate['threshold']:.9g}"
        for predicate in predicates
    )
    signal = wrong_reduction >= WRONG_REDUCTION_MIN and lock_reduction >= WRONG_LOCK_REDUCTION_MIN
    return {
        "predicates": predicates,
        "canonical": canonical,
        "correct_assertions": correct,
        "correct_assertion_retention": retention,
        "retention_hard_pass": retention >= RETENTION_MIN,
        "wrong_instance_assertions": wrong,
        "background_wrong_assertions": background_wrong,
        "other_instance_wrong_assertions": other_wrong,
        "episode_macro_wrong_reduction": statistics.mean(episode_reductions),
        "frame_aggregate_wrong_reduction": wrong_reduction,
        "max_wrong_lock_frames": max_wrong_frames,
        "max_wrong_lock_duration_ms": max_wrong_ms,
        "max_wrong_lock_duration_reduction": lock_reduction,
        "meaningful_mechanism_pass": signal,
        "admission_pass": retention >= RETENTION_MIN and signal,
        "per_episode_wrong_assertions": per_episode_wrong,
        "outputs_by_episode": outputs_by_episode,
        "decisions_by_episode": decisions_by_episode,
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (
        -row["frame_aggregate_wrong_reduction"],
        -row["max_wrong_lock_duration_reduction"],
        -row["episode_macro_wrong_reduction"],
        -row["correct_assertion_retention"],
        row["canonical"],
    ))


def _choose_terminal(scored: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    admitted = [row for row in scored if row["admission_pass"]]
    signal = [row for row in scored if row["meaningful_mechanism_pass"]]
    retention = [row for row in scored if row["retention_hard_pass"]]
    if admitted:
        return "DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED", _rank(admitted)[0]
    if signal:
        return "DENSE_IDENTITY_GAIN_ONLY_BY_ABSTENTION", _rank(signal)[0]
    return "DENSE_IDENTITY_NOT_SUFFICIENT", _rank(retention if retention else scored)[0]


def _warning_leads(
    trace: dict[str, Any], labels: dict[str, Any], decisions_by_episode: dict[str, list[str]]
) -> list[dict[str, Any]]:
    rows = []
    for episode in trace["episodes"]:
        episode_id = episode["episode_id"]
        outputs = episode["p1_output"]["frames"]
        frame_labels = labels[episode_id]
        timestamps = [int(frame["timestamp_ms"]) for frame in episode["p1_input"]["frames"]]
        first_wrong = next((index for index, (output, label) in enumerate(zip(outputs, frame_labels))
                            if output["current_candidate_id"] is not None
                            and label["identity_class"] in {"BACKGROUND_DRIFT", "OTHER_INSTANCE"}), None)
        if first_wrong is None:
            continue
        prior_correct = [
            index for index in range(first_wrong)
            if outputs[index]["current_candidate_id"] is not None
            and frame_labels[index]["identity_class"] == "CORRECT"
        ]
        search_start = (prior_correct[-1] + 1) if prior_correct else 1
        first_warning = next((index for index in range(search_start, len(outputs))
                              if decisions_by_episode[episode_id][index] in {"UNCERTAIN", "INVALID"}), None)
        rows.append({
            "episode_id": episode_id,
            "first_gt_wrong_frame_index": first_wrong,
            "first_gt_wrong_timestamp_ms": timestamps[first_wrong],
            "warning_search_start_frame_index": search_start,
            "first_validator_warning_frame_index": first_warning,
            "first_validator_warning_timestamp_ms": None if first_warning is None else timestamps[first_warning],
            "pre_drift_warning_lead_ms": None if first_warning is None else timestamps[first_wrong] - timestamps[first_warning],
            "warning_decision": None if first_warning is None else decisions_by_episode[episode_id][first_warning],
        })
    return rows


def run_sweep(trace_path: Path, private_path: Path, sealed_prediction_path: Path, output_dir: Path) -> dict[str, Any]:
    trace = r0.read_json(trace_path)
    private = r0.read_json(private_path)
    sealed = r0.read_json(sealed_prediction_path)
    if trace.get("schema_version") != TRACE_SCHEMA:
        raise ValueError("dense identity trace schema drift")
    if trace.get("post_initialization_gt_reads") != 0 or trace.get("online_target_memory_updates") != 0:
        raise ValueError("truth firewall or fixed-memory invariant failed")
    if trace.get("candidate_generator_changed") is not False or trace.get("global_search") is not False:
        raise ValueError("candidate-generator intervention drift")
    if trace["sealed_prediction_sha256"] != sha256(sealed_prediction_path):
        raise ValueError("dense identity trace / sealed prediction identity drift")
    if trace["runtime"]["runner_sha256"] != _source_sha256():
        raise ValueError("runner changed after dense feature extraction")
    trace_by_id = {episode["episode_id"]: episode for episode in trace["episodes"]}
    labels, tags = a1._labels(private, trace_by_id)
    baseline_counts = _baseline_counts(trace, labels)
    grid = _quantile_grid(trace)
    scored = [_score_policy(trace, labels, predicates, baseline_counts) for predicates in _policy_family(grid)]
    expected_count = len(POLICY_QUANTILES) ** len(POLICY_FEATURES)
    if len(scored) != expected_count:
        raise ValueError(f"bounded policy family drift: {len(scored)} != {expected_count}")
    terminal, winner = _choose_terminal(scored)

    sealed_by_id = {episode["episode_id"]: episode for episode in sealed["episodes"]}
    winner_episodes = []
    for episode in trace["episodes"]:
        winner_episodes.append({
            **sealed_by_id[episode["episode_id"]],
            "p1_output": winner["outputs_by_episode"][episode["episode_id"]],
        })
    winner_prediction = {**sealed, "episodes": winner_episodes}
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_prediction_path = output_dir / "winner_prediction.json"
    winner_evaluation_path = output_dir / "winner_evaluation.json"
    r0.write_json(winner_prediction_path, winner_prediction)
    winner_evaluation = r0.evaluate_predictions(private_path, winner_prediction_path, winner_evaluation_path)
    warning_leads = _warning_leads(trace, labels, winner["decisions_by_episode"])

    for row in scored:
        row.pop("outputs_by_episode")
        row.pop("decisions_by_episode")
    winner.pop("outputs_by_episode", None)
    winner.pop("decisions_by_episode", None)
    result = {
        "schema_version": SWEEP_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY_NO_POLICY_ADMISSION_NO_SCIENTIFIC_VERDICT",
        "terminal": terminal,
        "policy_admission": "NO_POLICY_ADMISSION",
        "invariants": {
            "candidate_generator": trace["candidate_generator"],
            "candidate_generator_changed": False,
            "post_initialization_gt_reads": 0,
            "online_target_memory_updates": 0,
            "global_search": False,
            "reacquisition_added": False,
        },
        "model": trace["model"],
        "search": {
            "features": list(POLICY_FEATURES),
            "quantiles": list(POLICY_QUANTILES),
            "candidate_count": len(scored),
            "second_round_search": False,
            "uncertain_is_fail_closed": True,
        },
        "admission_gate": {
            "correct_assertion_retention_min": RETENTION_MIN,
            "wrong_assertion_reduction_min": WRONG_REDUCTION_MIN,
            "max_wrong_lock_reduction_min": WRONG_LOCK_REDUCTION_MIN,
        },
        "baseline": baseline_counts,
        "winner": winner,
        "winner_frozen_evaluator": winner_evaluation,
        "pre_drift_warning_lead": warning_leads,
        "top_retention_hard_pass": [
            {key: value for key, value in row.items() if key != "per_episode_wrong_assertions"}
            for row in _rank([row for row in scored if row["retention_hard_pass"]])[:20]
        ],
        "all_candidates": scored,
        "episode_temporal_tags": tags,
    }
    r0.write_json(output_dir / "sweep_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--public-input", type=Path, required=True)
    extract.add_argument("--a1-trace", type=Path, required=True)
    extract.add_argument("--sealed-prediction", type=Path, required=True)
    extract.add_argument("--model-dir", type=Path, required=True)
    extract.add_argument("--trace", type=Path, required=True)
    extract.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--trace", type=Path, required=True)
    sweep.add_argument("--private-input", type=Path, required=True)
    sweep.add_argument("--sealed-prediction", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        trace = extract_trace(
            args.public_input, args.a1_trace, args.sealed_prediction, args.model_dir, args.trace, args.device
        )
        print(json.dumps({
            "candidate_parity": trace["candidate_parity"],
            "episodes": len(trace["episodes"]),
            "encoded_crops": trace["runtime"]["encoded_crops"],
            "flow_candidates": sum(len(episode["identity_by_candidate"]) for episode in trace["episodes"]),
            "post_initialization_gt_reads": trace["post_initialization_gt_reads"],
            "online_target_memory_updates": trace["online_target_memory_updates"],
        }, sort_keys=True))
    else:
        result = run_sweep(args.trace, args.private_input, args.sealed_prediction, args.output_dir)
        print(json.dumps({"terminal": result["terminal"], "winner": result["winner"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
