"""Compare OLD and NEW B checkpoints on the finalized 5000-frame distance set.

This module deliberately keeps evaluator labels out of the model path.  The
finalized index supplies RGB provenance and native capped-count labels for
post-inference diagnostics only; each arm receives one RGB-only forward pass
over the 5000 frames at batch size 32.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import io
import os
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from body_query_data import fresh_output, read, sha, write
from body_query_model import BodyQueryModel
from body_query_range import range_from_counts
from city_dev_selection import apply_thresholds

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
from research_backend import torch_observation


EXPECTED_FRAMES = 5000
EXPECTED_PAIRS = 2500
BATCH_SIZE = 32
OLD_CHECKPOINT_SHA256 = (
    "c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776"
)
DIRECTION_TOLERANCE = 1e-6
IMAGE_SIZE = (256, 144)  # width, height; BodyQueryModel expects 144x256.


def direction(delta: float, tolerance: float = DIRECTION_TOLERANCE) -> str:
    """Classify a far-minus-near delta with the frozen tie tolerance."""
    value = float(delta)
    if not np.isfinite(value):
        raise ValueError("Distance delta must be finite")
    if value > tolerance:
        return "far_higher"
    if value < -tolerance:
        return "near_higher"
    return "tie"


def binary_confusion(predicted: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Return inclusive binary confusion counts for equal-shaped arrays."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {predicted.shape} vs {truth.shape}")
    return {
        "TP": int(np.logical_and(predicted, truth).sum()),
        "FP": int(np.logical_and(predicted, ~truth).sum()),
        "FN": int(np.logical_and(~predicted, truth).sum()),
        "TN": int(np.logical_and(~predicted, ~truth).sum()),
    }


def paired_correctness(old: np.ndarray, new: np.ndarray) -> dict[str, int]:
    """Count paired wins/losses for two equal-shaped correctness arrays."""
    old = np.asarray(old, dtype=bool)
    new = np.asarray(new, dtype=bool)
    if old.shape != new.shape:
        raise ValueError(f"Paired correctness shape mismatch: {old.shape} vs {new.shape}")
    return {
        "old_only_correct": int(np.logical_and(old, ~new).sum()),
        "new_only_correct": int(np.logical_and(~old, new).sum()),
        "both_correct": int(np.logical_and(old, new).sum()),
        "both_wrong": int(np.logical_and(~old, ~new).sum()),
    }


def nonempty_recall(
    nonempty_probability: np.ndarray,
    native_counts: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Recall for query cells whose native capped count is nonzero."""
    probability = np.asarray(nonempty_probability, dtype=np.float64)
    counts = np.asarray(native_counts)
    if probability.shape != counts.shape:
        raise ValueError(f"Query probability/count shape mismatch: {probability.shape} vs {counts.shape}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Nonempty probabilities must be finite and in [0, 1]")
    truth = counts > 0
    predicted = probability >= float(threshold)
    result = binary_confusion(predicted, truth)
    positives = int(truth.sum())
    result.update(
        threshold=float(threshold),
        positives=positives,
        predicted_positive=int(predicted.sum()),
        recall=float(result["TP"] / positives) if positives else None,
    )
    return result


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.median(values)) if values else None


def _path_under_artifacts(path: Path) -> Path:
    resolved = path.resolve()
    artifact_root = (Path(__file__).resolve().parents[4] / "artifacts.local").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(f"Dataset input is outside canonical artifacts: {resolved}")
    return resolved


def _verify_finalized_dataset(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify finalized PASS receipts and return index rows (without native IO)."""
    dataset = _path_under_artifacts(dataset)
    manifest_path = dataset / "manifest.json"
    summary_path = dataset / "summary.json"
    index_path = dataset / "index.json"
    if not all(p.is_file() for p in (manifest_path, summary_path, index_path)):
        raise FileNotFoundError(f"Finalized dataset requires manifest, summary and index: {dataset}")
    manifest = read(manifest_path)
    summary = read(summary_path)
    index = read(index_path)
    if manifest.get("status") != "PASS" or summary.get("status") != "PASS":
        raise ValueError("Finalized dataset manifest and summary must both be PASS")
    if manifest.get("index_sha256") != sha(index_path):
        raise ValueError("Finalized index hash does not match manifest")
    if manifest.get("summary_sha256") != sha(summary_path):
        raise ValueError("Finalized summary hash does not match manifest")
    for path_key, hash_key in (
        ("source_plan", "source_plan_sha256"),
        ("label_receipt", "label_receipt_sha256"),
        ("visual_review", "visual_review_sha256"),
    ):
        source = Path(manifest[path_key])
        if not source.is_file() or sha(source) != manifest[hash_key]:
            raise ValueError(f"Finalized source hash mismatch: {path_key}")
    label_receipt = read(Path(manifest["label_receipt"]))
    label_root = Path(manifest["label_receipt"]).resolve().parent
    label_index = label_root / "evaluator" / "index.json"
    label_pairs = label_root / "pairs.json"
    if label_receipt.get("index_sha256") != sha(label_index) or label_receipt.get("pairs_sha256") != sha(label_pairs):
        raise ValueError("Finalized label index/pairs hashes do not match label receipt")
    if sha(index_path) != label_receipt.get("index_sha256"):
        raise ValueError("Finalized index differs from the hash-bound label index")
    rows = index.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} indexed frames")
    if summary.get("captured_frames") != EXPECTED_FRAMES or summary.get("complete_pairs") != EXPECTED_PAIRS:
        raise ValueError("Finalized summary frame/pair counts are not 5000/2500")
    if summary.get("accepted_pairs") != EXPECTED_PAIRS or summary.get("rejected_pairs") != 0:
        raise ValueError("Distance source must contain 2500 accepted pairs and no rejected pair")
    sample_indices = [int(row.get("sample_index", -1)) for row in rows]
    if sorted(sample_indices) != list(range(EXPECTED_FRAMES)):
        raise ValueError("Indexed sample indices are not a complete 0..4999 sequence")
    for row in rows:
        if row.get("accepted") is not True or row.get("rejection_reasons") not in ([], None):
            raise ValueError(f"Unaccepted indexed endpoint: {row.get('name')}")
        if row.get("endpoint") not in ("near", "far"):
            raise ValueError(f"Invalid endpoint: {row.get('endpoint')}")
        if len(row.get("near", [])) != 2 or len(row.get("counts", [])) != 12:
            raise ValueError(f"Invalid indexed evaluator shape: {row.get('name')}")
        if "unknown_pixels" not in row or int(row["unknown_pixels"]) < 0:
            raise ValueError(f"Native UNKNOWN coverage is missing: {row.get('name')}")
    # Keep every downstream array indexed by the declared sample_index, even
    # if a finalized index was serialized in another order.
    rows = sorted(rows, key=lambda row: int(row["sample_index"]))
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_rows[str(row["pair_id"])].append(row)
    if len(pair_rows) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} unique pairs, got {len(pair_rows)}")
    pairs: list[dict[str, Any]] = []
    for pair_id, members in pair_rows.items():
        if len(members) != 2 or {m["endpoint"] for m in members} != {"near", "far"}:
            raise ValueError(f"Pair must have one near and one far endpoint: {pair_id}")
        members.sort(key=lambda row: int(row["sample_index"]))
        near, far = next(m for m in members if m["endpoint"] == "near"), next(m for m in members if m["endpoint"] == "far")
        shared = ("source_partition", "region_id", "family", "site_id")
        if any(near.get(key) != far.get(key) for key in shared):
            raise ValueError(f"Pair metadata differs across endpoints: {pair_id}")
        pairs.append(
            {
                "pair_id": pair_id,
                "near_index": int(near["sample_index"]),
                "far_index": int(far["sample_index"]),
                "source_partition": near["source_partition"],
                "region_id": near["region_id"],
                "site_id": near["site_id"],
                "family": near["family"],
                "distances_m": [float(near["distance_m"]), float(far["distance_m"])],
            }
        )
    pairs.sort(key=lambda row: row["near_index"])
    return {
        "manifest": manifest,
        "summary": summary,
        "manifest_sha256": sha(manifest_path),
        "summary_sha256": sha(summary_path),
        "index_sha256": sha(index_path),
        "pairs_sha256": label_receipt["pairs_sha256"],
    }, rows, pairs


def _load_rgb(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Hash-bind and resize every source RGB image with BOX filtering."""
    images: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in sorted(rows, key=lambda item: int(item["sample_index"])):
        source = _path_under_artifacts(Path(row["rgb_path"]))
        digest = sha(source)
        if digest != row.get("rgb_sha256"):
            raise ValueError(f"RGB source hash mismatch: {source}")
        if digest in seen_hashes:
            raise ValueError(f"Repeated RGB source hash: {digest}")
        seen_hashes.add(digest)
        with source.open("rb") as stream:
            encoded = stream.read()
        with Image.open(io.BytesIO(encoded)) as image:
            if image.width * 360 != image.height * 640:
                raise ValueError(f"RGB source is not fixed 16:9: {source}")
            pixels = np.asarray(
                image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BOX), dtype=np.uint8
            )
        if pixels.shape != (144, 256, 3):
            raise ValueError(f"Unexpected resized RGB shape: {pixels.shape}")
        images.append(pixels)
        metadata.append(
            {
                "sample_index": int(row["sample_index"]),
                "name": row["name"],
                "pair_id": row["pair_id"],
                "source_partition": row["source_partition"],
                "region_id": row["region_id"],
                "site_id": row["site_id"],
                "family": row["family"],
                "endpoint": row["endpoint"],
                "distance_m": float(row["distance_m"]),
                "rgb_path": str(source),
                "rgb_sha256": digest,
                "unknown_pixels": int(row.get("unknown_pixels", 0)),
            }
        )
    return np.stack(images, axis=0), metadata


def _load_selection(selection_path: Path, old_checkpoint: Path, new_checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = read(selection_path)
    if set(("OLD", "NEW")) - set(selection):
        raise ValueError("Selection must contain OLD and NEW entries")
    checkpoints = {"OLD": old_checkpoint.resolve(), "NEW": new_checkpoint.resolve()}
    for arm, checkpoint in checkpoints.items():
        entry = selection[arm]
        expected = entry.get("checkpoint_sha256")
        actual = sha(checkpoint)
        if expected != actual:
            raise ValueError(f"{arm} checkpoint does not match selection: {actual} != {expected}")
        if arm == "OLD" and actual != OLD_CHECKPOINT_SHA256:
            raise ValueError(f"OLD checkpoint must be the fixed expanded B SHA: {actual}")
        thresholds = np.asarray(entry.get("thresholds"), dtype=np.float64)
        if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
            raise ValueError(f"Invalid {arm} DEV-selected thresholds")
    return selection, checkpoints


def _run_inference(arm: str, checkpoint: Path, rgb: np.ndarray, pretrained: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the distance comparison")
    device = torch.device("cuda")
    model = BodyQueryModel(pretrained, "B").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    near_outputs: list[np.ndarray] = []
    count_outputs: list[np.ndarray] = []
    final_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    with torch.inference_mode():
        for begin in range(0, len(rgb), BATCH_SIZE):
            batch = torch.from_numpy(rgb[begin : begin + BATCH_SIZE].copy()).to(device)
            batch = batch.permute(0, 3, 1, 2).float() / 255.0
            near_logits, _support_logits, count_logits = model(batch)
            final_output = (near_logits, _support_logits, count_logits)
            near_outputs.append(near_logits.sigmoid().cpu().numpy().astype(np.float32))
            count_outputs.append(count_logits.softmax(-1).cpu().numpy().astype(np.float32))
    torch.cuda.synchronize()
    if final_output is None:
        raise ValueError("No inference batches were produced")
    observation = asdict(torch_observation(model=model, output=final_output))
    if observation["device_type"] != "cuda" or "5060" not in observation["device_name"].upper():
        raise RuntimeError(
            "Distance comparison is frozen to CUDA RTX 5060; observed "
            f"{observation['device_type']} / {observation['device_name']}"
        )
    near = np.concatenate(near_outputs, axis=0)
    counts = np.concatenate(count_outputs, axis=0)
    del final_output, model
    torch.cuda.empty_cache()
    return {"near": near, "counts": counts}, observation


def _event_confusions(range_scores: np.ndarray, native_events: np.ndarray, ids: np.ndarray) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for head_index, head_name in enumerate(("BODY", "HEAD")):
        result[head_name] = {}
        for range_index, range_name in enumerate(("near", "far")):
            result[head_name][range_name] = binary_confusion(
                range_scores[ids, head_index, range_index] >= 0.5,
                native_events[ids, head_index, range_index],
            )
    return result


def _summarize_arm(
    arm: str,
    pair_ids: list[int],
    arm_pairs: list[dict[str, Any]],
    frame_ids: np.ndarray,
    predictions: dict[str, np.ndarray],
    native_events: np.ndarray,
    native_counts: np.ndarray,
) -> dict[str, Any]:
    selected = [arm_pairs[index] for index in pair_ids]
    deltas = [float(row["delta"]) for row in selected]
    directions = [row["direction"] for row in selected]
    near_ids = np.asarray([row["near_index"] for row in selected], dtype=np.int64)
    far_ids = np.asarray([row["far_index"] for row in selected], dtype=np.int64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    alerts = apply_thresholds(predictions["near"][frame_ids], predictions["thresholds"])
    body_alerts = alerts[:, 0]
    head_alerts = alerts[:, 1]
    near_head_alerts = predictions["near"][near_ids, 1] >= predictions["thresholds"][1]
    far_head_alerts = predictions["near"][far_ids, 1] >= predictions["thresholds"][1]
    head_nonempty_probability = 1.0 - predictions["counts"][frame_ids, 6:9, 0]
    query_counts = native_counts[frame_ids, 1, 0]
    query_recall = nonempty_recall(head_nonempty_probability, query_counts, 0.5)
    all_head_hits = int(head_alerts.sum())
    return {
        "arm": arm,
        "pairs": len(selected),
        "frames": int(len(frame_ids)),
        "direction": {
            "far_higher": int(directions.count("far_higher")),
            "near_higher": int(directions.count("near_higher")),
            "ties": int(directions.count("tie")),
            "correct": int(directions.count("far_higher")),
            "raw_positive": int(sum(row["positive_raw"] for row in selected)),
            "tolerance": DIRECTION_TOLERANCE,
        },
        "both_endpoint_HEAD_alerts": {
            "count": int(sum(row["both_HEAD"] for row in selected)),
            "total": len(selected),
        },
        "HEAD_alert_hits": {
            "near": {"hits": int(near_head_alerts.sum()), "total": len(near_head_alerts)},
            "far": {"hits": int(far_head_alerts.sum()), "total": len(far_head_alerts)},
            "all": {"hits": all_head_hits, "total": int(len(frame_ids))},
        },
        "BODY_false_alerts": {
            "count": int(body_alerts.sum()),
            "total": int(len(frame_ids)),
        },
        "far_score_delta": {
            "mean": _mean(deltas),
            "median": _median(deltas),
        },
        "native_event_confusion_at_0_5": _event_confusions(
            predictions["range"], native_events, frame_ids
        ),
        "HEAD_near_query_nonempty_recall_at_0_5": query_recall,
    }


def _scope(
    name: str,
    pair_ids: list[int],
    pairs: list[dict[str, Any]],
    arm_pairs: Mapping[str, list[dict[str, Any]]],
    predictions: Mapping[str, dict[str, np.ndarray]],
    native_events: np.ndarray,
    native_counts: np.ndarray,
) -> dict[str, Any]:
    frame_ids = np.asarray(
        sorted({idx for pair_id in pair_ids for idx in (pairs[pair_id]["near_index"], pairs[pair_id]["far_index"])}),
        dtype=np.int64,
    )
    arms = {
        arm: _summarize_arm(
            arm,
            pair_ids,
            arm_pairs[arm],
            frame_ids,
            predictions[arm],
            native_events,
            native_counts,
        )
        for arm in ("OLD", "NEW")
    }
    old_pair_correct = np.asarray(
        [arm_pairs["OLD"][pair_id]["direction"] == "far_higher" for pair_id in pair_ids], dtype=bool
    )
    new_pair_correct = np.asarray(
        [arm_pairs["NEW"][pair_id]["direction"] == "far_higher" for pair_id in pair_ids], dtype=bool
    )
    old_range_correct = (predictions["OLD"]["range"][frame_ids] >= 0.5) == native_events[frame_ids]
    new_range_correct = (predictions["NEW"]["range"][frame_ids] >= 0.5) == native_events[frame_ids]
    old_head_alert_correct = predictions["OLD"]["near"][frame_ids, 1] >= predictions["OLD"]["thresholds"][1]
    new_head_alert_correct = predictions["NEW"]["near"][frame_ids, 1] >= predictions["NEW"]["thresholds"][1]
    return {
        "scope": name,
        "pair_count": len(pair_ids),
        "frame_count": int(len(frame_ids)),
        "arms": arms,
        "paired_OLD_vs_NEW_correctness": {
            "pair_direction": paired_correctness(old_pair_correct, new_pair_correct),
            "native_event_cells_at_0_5": paired_correctness(old_range_correct, new_range_correct),
            "HEAD_alert_frames": paired_correctness(old_head_alert_correct, new_head_alert_correct),
        },
    }


def run(
    dataset: Path,
    old_checkpoint: Path,
    new_checkpoint: Path,
    selection_path: Path,
    pretrained: Path,
    protocol: Path,
    output: Path,
) -> dict[str, Any]:
    """Run the fixed two-arm comparison and write a fresh artifact bundle."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    dataset = dataset.resolve()
    old_checkpoint = old_checkpoint.resolve()
    new_checkpoint = new_checkpoint.resolve()
    selection_path = selection_path.resolve()
    pretrained = pretrained.resolve()
    protocol = protocol.resolve()
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    source_info, rows, pairs = _verify_finalized_dataset(dataset)
    rgb, frame_metadata = _load_rgb(rows)
    selection, checkpoints = _load_selection(selection_path, old_checkpoint, new_checkpoint)
    output = fresh_output(output)
    native_counts = np.asarray([row["counts"] for row in rows], dtype=np.int64).reshape(EXPECTED_FRAMES, 2, 2, 3)
    if np.any(native_counts < 0) or np.any(native_counts > 3):
        raise ValueError("Indexed capped counts must lie in [0, 3]")
    native_events = native_counts.sum(axis=-1) >= 3
    native_frame_labels = np.asarray([row["near"] for row in rows], dtype=np.int8)
    if native_frame_labels.shape != (EXPECTED_FRAMES, 2) or not np.array_equal(
        native_frame_labels, np.tile(np.asarray([[0, 1]], dtype=np.int8), (EXPECTED_FRAMES, 1))
    ):
        raise ValueError("Distance source must contain accepted BODY=0, HEAD=1 endpoint labels")
    endpoints = np.asarray([row["endpoint"] for row in sorted(rows, key=lambda item: int(item["sample_index"]))])
    for pair in pairs:
        near_index, far_index = pair["near_index"], pair["far_index"]
        if endpoints[near_index] != "near" or endpoints[far_index] != "far":
            raise ValueError(f"Endpoint ordering changed for {pair['pair_id']}")
        if bool(native_events[near_index, 0].any()) or bool(native_events[far_index, 0].any()):
            raise ValueError(f"Accepted distance pair has native BODY event: {pair['pair_id']}")
        if not bool(native_events[near_index, 1, 0]) or bool(native_events[near_index, 1, 1]):
            raise ValueError(f"Near endpoint is not exclusive HEAD-near: {pair['pair_id']}")
        if not bool(native_events[far_index, 1, 1]) or bool(native_events[far_index, 1, 0]):
            raise ValueError(f"Far endpoint is not exclusive HEAD-far: {pair['pair_id']}")

    inference: dict[str, dict[str, np.ndarray]] = {}
    devices: dict[str, Any] = {}
    thresholds = {
        arm: np.asarray(selection[arm]["thresholds"], dtype=np.float64) for arm in ("OLD", "NEW")
    }
    for arm in ("OLD", "NEW"):
        prediction, device = _run_inference(arm, checkpoints[arm], rgb, pretrained)
        prediction["range"] = range_from_counts(
            torch.from_numpy(np.log(np.clip(prediction["counts"].astype(np.float64), 1e-300, 1.0)))
        ).sigmoid().numpy()
        prediction["thresholds"] = thresholds[arm]
        inference[arm] = prediction
        devices[arm] = device
    arm_pairs: dict[str, list[dict[str, Any]]] = {"OLD": [], "NEW": []}
    for pair in pairs:
        for arm in ("OLD", "NEW"):
            pred = inference[arm]
            near_index, far_index = pair["near_index"], pair["far_index"]
            delta = float(pred["range"][far_index, 1, 1] - pred["range"][near_index, 1, 1])
            endpoint_alerts = pred["near"][[near_index, far_index]] >= pred["thresholds"]
            arm_pairs[arm].append(
                {
                    "pair_id": pair["pair_id"],
                    "near_index": near_index,
                    "far_index": far_index,
                    "delta": delta,
                    "direction": direction(delta),
                    "positive_raw": bool(delta > 0.0),
                    "both_HEAD": bool(endpoint_alerts[:, 1].all()),
                    "HEAD_alerts": endpoint_alerts[:, 1].astype(bool).tolist(),
                    "BODY_alerts": endpoint_alerts[:, 0].astype(bool).tolist(),
                    "HEAD_scores": pred["near"][[near_index, far_index], 1].astype(float).tolist(),
                    "BODY_scores": pred["near"][[near_index, far_index], 0].astype(float).tolist(),
                    "far_scores": pred["range"][[near_index, far_index], 1, 1].astype(float).tolist(),
                    "near_scores": pred["range"][[near_index, far_index], 1, 0].astype(float).tolist(),
                }
            )
    pair_results = []
    for index, pair in enumerate(pairs):
        pair_results.append(
            {
                **pair,
                "OLD": arm_pairs["OLD"][index],
                "NEW": arm_pairs["NEW"][index],
            }
        )

    scopes: dict[str, Any] = {}
    all_pair_ids = list(range(len(pairs)))
    eval_pair_ids = [i for i, pair in enumerate(pairs) if str(pair["source_partition"]).lower() == "eval"]
    if not eval_pair_ids:
        raise ValueError("Finalized source has no EVAL_ONLY distance pairs")
    scopes["primary_EVAL_ONLY_distance"] = _scope(
        "primary_EVAL_ONLY_distance", eval_pair_ids, pairs, arm_pairs, inference, native_events, native_counts
    )
    scopes["all_2500_shared_site_Development"] = _scope(
        "all_2500_shared_site_Development", all_pair_ids, pairs, arm_pairs, inference, native_events, native_counts
    )
    for key in ("source_partition", "region_id", "family"):
        slices: dict[str, Any] = {}
        for value in sorted({str(pair[key]) for pair in pairs}):
            ids = [i for i, pair in enumerate(pairs) if str(pair[key]) == value]
            slices[value] = _scope(value, ids, pairs, arm_pairs, inference, native_events, native_counts)
        scopes["all_2500_shared_site_Development"][f"slices_by_{key}"] = slices
        eval_slices: dict[str, Any] = {}
        for value in sorted({str(pairs[i][key]) for i in eval_pair_ids}):
            ids = [i for i in eval_pair_ids if str(pairs[i][key]) == value]
            eval_slices[value] = _scope(value, ids, pairs, arm_pairs, inference, native_events, native_counts)
        scopes["primary_EVAL_ONLY_distance"][f"slices_by_{key}"] = eval_slices

    output_result = {
        "schema": "body-query-10000-distance-analysis-v1",
        "status": "PASS",
        "dataset_scope": "Primary EVAL_ONLY distance subset; all 2500 pairs descriptive shared-site Development",
        "evaluator_boundary": "Native capped counts and events are evaluator-only; no native truth enters RGB inference",
        "score_definition": "S_far is P(sum of three HEAD-far capped counts >=3); delta=S_far(far)-S_far(near)",
        "direction_tolerance": DIRECTION_TOLERANCE,
        "threshold_source": "Selection JSON supplied by NEW relation DEV; this analyzer performs no fitting",
        "head_alert_boundary": "All endpoints are accepted HEAD-positive; no HEAD alert FPR denominator is estimated",
        "native_unknown_pixels": {
            "frames_with_unknown_pixels": int(sum(int(row.get("unknown_pixels", 0)) > 0 for row in rows)),
            "total_pixels": int(sum(int(row.get("unknown_pixels", 0)) for row in rows)),
        },
        "summary": scopes["primary_EVAL_ONLY_distance"],
        "primary_summary": scopes["primary_EVAL_ONLY_distance"],
        "development_summary": scopes["all_2500_shared_site_Development"],
        "slices": {
            key: scopes["primary_EVAL_ONLY_distance"][f"slices_by_{key}"]
            for key in ("source_partition", "region_id", "family")
        },
        "scopes": scopes,
        "pairs": pair_results,
    }
    predictions_path = output / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        OLD_near=inference["OLD"]["near"],
        OLD_counts=inference["OLD"]["counts"],
        OLD_range=inference["OLD"]["range"],
        NEW_near=inference["NEW"]["near"],
        NEW_counts=inference["NEW"]["counts"],
        NEW_range=inference["NEW"]["range"],
    )
    evaluator_truth_path = output / "evaluator_truth.npz"
    np.savez_compressed(evaluator_truth_path, native_counts=native_counts, native_events=native_events)
    frame_metadata_path = output / "frame_metadata.json"
    write(frame_metadata_path, frame_metadata)
    result_path = output / "result.json"
    write(result_path, output_result)
    validation_path = output / "validation.json"
    write(
        validation_path,
        {
            "status": "PASS",
            "checks": [
                "Finalized PASS manifest/summary and source/index hashes",
                "5000 unique RGB source hashes and exact BOX 256x144 preprocessing",
                "One CUDA B forward pass per arm at batch size 32",
                "No threshold fitting; selection checkpoints and thresholds hash-bound",
                "Independent count-to-range arithmetic and native event confusion",
            ],
            "training_steps": 0,
            "fits": 0,
        },
    )
    receipt_path = output / "receipt.json"
    write(
        receipt_path,
        {
            "schema": "body-query-10000-distance-analysis-receipt-v1",
            "status": "PASS",
            "fits": 0,
            "training_steps": 0,
            "frames": EXPECTED_FRAMES,
            "pairs": EXPECTED_PAIRS,
            "batch_size": BATCH_SIZE,
            "dataset_manifest_sha256": source_info["manifest_sha256"],
            "dataset_summary_sha256": source_info["summary_sha256"],
            "dataset_index_sha256": source_info["index_sha256"],
            "dataset_pairs_sha256": source_info["pairs_sha256"],
            "selection_sha256": sha(selection_path),
            "protocol_sha256": sha(protocol),
            "old_checkpoint_sha256": sha(checkpoints["OLD"]),
            "new_checkpoint_sha256": sha(checkpoints["NEW"]),
            "predictions_sha256": sha(predictions_path),
            "evaluator_truth_sha256": sha(evaluator_truth_path),
            "frame_metadata_sha256": sha(frame_metadata_path),
            "result_sha256": sha(result_path),
            "validation_sha256": sha(validation_path),
            "code_sha256": sha(Path(__file__)),
            "devices": devices,
            "pretrained": str(pretrained),
            "scope": output_result["dataset_scope"],
        },
    )
    print(output_result["scopes"]["primary_EVAL_ONLY_distance"], flush=True)
    return output_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old", dest="old_checkpoint", type=Path, required=True)
    parser.add_argument("--new", dest="new_checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        old_checkpoint=args.old_checkpoint,
        new_checkpoint=args.new_checkpoint,
        selection_path=args.selection,
        pretrained=args.pretrained,
        protocol=args.protocol,
        output=args.output,
    )


if __name__ == "__main__":
    main()
