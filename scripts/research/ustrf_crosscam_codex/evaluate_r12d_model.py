#!/usr/bin/env python3
"""Evaluate one R1.2d model on source-held-out boxes and the frozen 12 events."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projected_corridor_geometry import classify_contact_point
from r12d_contract import CLASSES, load_json, require, sha256_file, validate_matrix, write_json


class StaticTfliteYolo:
    """Host runner matching the Android static-class TFLite decoder contract."""

    names = {index: name for index, name in enumerate(CLASSES)}

    def __init__(self, weights: Path) -> None:
        import tensorflow as tf
        self.tensorflow_version = tf.__version__
        self.interpreter = tf.lite.Interpreter(model_path=str(weights), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        require(self.input["shape"].tolist() == [1, 768, 768, 3], f"unexpected TFLite input: {self.input['shape']}")
        require(self.output["shape"].tolist() == [1, 39, 12096], f"unexpected TFLite output: {self.output['shape']}")

    def predict_rows(self, image_paths: list[Path], image_size: int, confidence: float,
                     nms_iou: float, maximum: int) -> list[list[dict[str, Any]]]:
        import cv2
        import numpy as np
        require(image_size == 768, "static YOLOE TFLite input size drifted")
        outputs: list[list[dict[str, Any]]] = []
        for path in image_paths:
            image = cv2.imread(str(path))
            require(image is not None, f"cannot decode {path}")
            source_height, source_width = image.shape[:2]
            scale = min(image_size / source_width, image_size / source_height)
            resized_width = max(1, int(source_width * scale)); resized_height = max(1, int(source_height * scale))
            dx = (image_size - resized_width) / 2.0; dy = (image_size - resized_height) / 2.0
            resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            tensor = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
            left, top = int(dx), int(dy)
            tensor[0, top:top + resized_height, left:left + resized_width] = resized.astype(np.float32) / 255.0
            self.interpreter.set_tensor(self.input["index"], tensor)
            self.interpreter.invoke()
            raw = self.interpreter.get_tensor(self.output["index"])[0]
            require(raw.shape[0] >= 4 + len(CLASSES), f"TFLite output lacks class channels: {raw.shape}")
            scores = raw[4:4 + len(CLASSES)]
            class_ids = scores.argmax(axis=0)
            best_scores = scores.max(axis=0)
            rows = []
            for prediction in np.nonzero(best_scores >= confidence)[0].tolist():
                values = raw[:4, prediction]
                cx, cy, width, height = [float(value * image_size if value <= 1.5 else value) for value in values]
                box = [(cx - width / 2 - dx) / scale, (cy - height / 2 - dy) / scale,
                       (cx + width / 2 - dx) / scale, (cy + height / 2 - dy) / scale]
                box = [max(0.0, min(float(source_width if index % 2 == 0 else source_height), value)) for index, value in enumerate(box)]
                if box[2] - box[0] <= 1.0 or box[3] - box[1] <= 1.0:
                    continue
                class_id = int(class_ids[prediction])
                rows.append({"class_id": class_id, "label": CLASSES[class_id], "confidence": float(best_scores[prediction]),
                             "bbox_xyxy_px": box})
            kept = []
            for candidate in sorted(rows, key=lambda row: row["confidence"], reverse=True):
                if any(candidate["class_id"] == prior["class_id"] and iou(candidate["bbox_xyxy_px"], prior["bbox_xyxy_px"]) > nms_iou for prior in kept):
                    continue
                kept.append(candidate)
                if len(kept) >= maximum:
                    break
            outputs.append(kept)
        return outputs


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]) + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_ratio(left: list[float], right: list[float], width: int, height: int) -> float:
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    return math.hypot(lx - rx, ly - ry) / math.hypot(width, height)


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def greedy_match(gt: list[dict[str, Any]], detections: list[dict[str, Any]], threshold: float) -> tuple[set[int], set[int]]:
    pairs = []
    for gi, truth in enumerate(gt):
        for di, detection in enumerate(detections):
            if truth["class_id"] == detection["class_id"]:
                pairs.append((iou(truth["bbox_xyxy_px"], detection["bbox_xyxy_px"]), gi, di))
    matched_gt, matched_det = set(), set()
    for overlap, gi, di in sorted(pairs, reverse=True):
        if overlap < threshold:
            break
        if gi not in matched_gt and di not in matched_det:
            matched_gt.add(gi); matched_det.add(di)
    return matched_gt, matched_det


def predict_rows(model: Any, image_paths: list[Path], image_size: int, confidence: float, nms_iou: float,
                 maximum: int, batch: int) -> list[list[dict[str, Any]]]:
    if isinstance(model, StaticTfliteYolo):
        return model.predict_rows(image_paths, image_size, confidence, nms_iou, maximum)
    outputs = []
    names = model.names
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        results = model.predict(source=[str(path) for path in chunk], imgsz=image_size, conf=confidence,
                                iou=nms_iou, max_det=maximum, batch=len(chunk), stream=True, verbose=False)
        for result in results:
            rows = []
            for score, class_id, box in zip(result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                index = int(class_id)
                rows.append({"class_id": index, "label": str(names[index]), "confidence": float(score),
                             "bbox_xyxy_px": [float(value) for value in box.tolist()]})
            outputs.append(rows)
    require(len(outputs) == len(image_paths), "prediction count mismatch")
    return outputs


def evaluate_validation(model: Any, dataset: Path, matrix: dict[str, Any], batch: int) -> dict[str, Any]:
    manifest = [json.loads(line) for line in (dataset / "training_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in manifest if row["split"] == "val"]
    paths = [dataset / row["image_path"] for row in rows]
    frozen = matrix["frozen_inference"]
    predictions = predict_rows(model, paths, frozen["image_size"], frozen["confidence"], frozen["nms_iou"], frozen["maximum_detections"], batch)
    metric = matrix["offline_metrics"]
    totals: collections.Counter[str] = collections.Counter()
    source_totals: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    class_totals: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row, detections in zip(rows, predictions):
        width, height = row["width"], row["height"]
        gt = []
        for geometry in row["geometry"]:
            x, y, w, h = geometry["bbox_xywh_px"]
            scale = min(frozen["image_size"] / width, frozen["image_size"] / height)
            scaled_w, scaled_h = w * scale, h * scale
            gt.append({"class_id": geometry["class_id"], "bbox_xyxy_px": [x, y, x + w, y + h],
                       "small": scaled_w * scaled_h <= metric["small_box_max_area_at_640"],
                       "london_like": scaled_w <= metric["london_like_max_width_at_640"] and scaled_h <= metric["london_like_max_height_at_640"]})
        matched_gt, matched_det = greedy_match(gt, detections, metric["match_iou"])
        buckets = [totals, source_totals[row["source_id"]]]
        for bucket in buckets:
            bucket["images"] += 1; bucket["gt"] += len(gt); bucket["matched"] += len(matched_gt)
            bucket["detections"] += len(detections); bucket["false_detections"] += len(detections) - len(matched_det)
            bucket["small_gt"] += sum(item["small"] for item in gt)
            bucket["small_matched"] += sum(gt[index]["small"] for index in matched_gt)
            bucket["london_like_gt"] += sum(item["london_like"] for item in gt)
            bucket["london_like_matched"] += sum(gt[index]["london_like"] for index in matched_gt)
        for class_id, name in enumerate(CLASSES):
            indexes = [index for index, truth in enumerate(gt) if truth["class_id"] == class_id]
            class_totals[name]["gt"] += len(indexes)
            class_totals[name]["matched"] += sum(index in matched_gt for index in indexes)
            det_indexes = [index for index, detection in enumerate(detections) if detection["class_id"] == class_id]
            class_totals[name]["detections"] += len(det_indexes)
            class_totals[name]["false_detections"] += sum(index not in matched_det for index in det_indexes)

    def finalize(counter: collections.Counter[str]) -> dict[str, Any]:
        value = dict(counter)
        value["recall"] = counter["matched"] / counter["gt"] if counter["gt"] else None
        value["precision"] = counter["matched"] / counter["detections"] if counter["detections"] else None
        value["small_recall"] = counter["small_matched"] / counter["small_gt"] if counter["small_gt"] else None
        value["london_like_recall"] = counter["london_like_matched"] / counter["london_like_gt"] if counter["london_like_gt"] else None
        value["false_detections_per_image"] = counter["false_detections"] / counter["images"] if counter["images"] else None
        return value

    sources = {name: finalize(counter) for name, counter in sorted(source_totals.items())}
    eligible_worst = {name: value for name, value in sources.items() if value["gt"] > 0}
    return {"aggregate": finalize(totals), "classes": {name: finalize(counter) for name, counter in class_totals.items()},
            "sources": sources, "worst_source_by_recall": min(eligible_worst, key=lambda name: eligible_worst[name]["recall"]),
            "worst_source_recall": min(value["recall"] for value in eligible_worst.values())}


def resolve_event_image(repo: Path, relative: str) -> Path:
    prefixes = {
        "ustrf-crosscam-r12c/frames/": repo / "artifacts.local/evidence/ustrf-crosscam-codex/mobile-r12c-seen-diagnostic-v2/exact-frame-transport-opencv-v2/frames",
        "ustrf-crosscam-r12b/frames/": repo / "artifacts.local/evidence/ustrf-crosscam-codex/mobile-r12b-seen-diagnostic-v1/exact-frame-transport-opencv/frames",
    }
    for prefix, root in prefixes.items():
        if relative.startswith(prefix):
            return root / relative[len(prefix):]
    raise ValueError(f"unknown event image prefix: {relative}")


def anchor_box(anchor: dict[str, Any], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = anchor["bbox_xyxy_norm"]
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def associate(source: dict[str, Any], frames: list[dict[str, Any]], replay: dict[str, Any], supported: set[str]) -> dict[str, Any]:
    visible = [row for row in source["target_anchors"] if row["visibility"] == "visible"]
    primary = next(row for row in visible if row["timestamp_ms"] == source["primary_anchor_timestamp_ms"])
    primary_index = min(range(len(frames)), key=lambda index: abs(frames[index]["timestamp_ms"] - primary["timestamp_ms"]))
    primary_frame = frames[primary_index]
    allow = set(source["detector_label_allowlist"]) & supported
    candidates = [row for row in primary_frame["detections"] if row["label"] in allow]
    target = anchor_box(primary, primary_frame["width"], primary_frame["height"])
    scored = sorted(((iou(target, row["bbox_xyxy_px"]), index, row) for index, row in enumerate(candidates)), reverse=True)
    assignments: list[dict[str, Any] | None] = [None] * len(frames)
    ambiguous = [False] * len(frames)
    primary_matched = bool(scored and scored[0][0] >= 0.30 and (len(scored) == 1 or abs(scored[0][0] - scored[1][0]) > 1e-9))
    if primary_matched:
        assignments[primary_index] = scored[0][2]
        for direction in (1, -1):
            previous = scored[0][2]; misses = 0; index = primary_index + direction
            while 0 <= index < len(frames):
                frame = frames[index]
                ranked = []
                for candidate in frame["detections"]:
                    if candidate["label"] not in allow:
                        continue
                    ratio = area(candidate["bbox_xyxy_px"]) / max(area(previous["bbox_xyxy_px"]), 1.0)
                    overlap = iou(previous["bbox_xyxy_px"], candidate["bbox_xyxy_px"])
                    distance = center_distance_ratio(previous["bbox_xyxy_px"], candidate["bbox_xyxy_px"], frame["width"], frame["height"])
                    if replay["association_area_ratio_min"] <= ratio <= replay["association_area_ratio_max"] and (overlap >= replay["association_iou_at_least"] or distance <= replay["association_center_distance_frame_diagonal_at_most"]):
                        score = overlap + 1.0 - min(distance / replay["association_center_distance_frame_diagonal_at_most"], 1.0)
                        ranked.append((score, candidate))
                ranked.sort(key=lambda row: row[0], reverse=True)
                is_ambiguous = len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= replay["association_ambiguity_margin"]
                if not ranked or is_ambiguous:
                    ambiguous[index] = is_ambiguous; misses += 1
                    if misses >= replay["clear_after_consecutive_misses"]:
                        break
                else:
                    assignments[index] = ranked[0][1]; previous = ranked[0][1]; misses = 0
                index += direction
    reacquired = 0; switches = 0
    for anchor in visible:
        index = min(range(len(frames)), key=lambda value: abs(frames[value]["timestamp_ms"] - anchor["timestamp_ms"]))
        truth = anchor_box(anchor, frames[index]["width"], frames[index]["height"])
        assigned = assignments[index]
        if assigned and iou(truth, assigned["bbox_xyxy_px"]) >= 0.30:
            reacquired += 1
        elif assigned and any(row["label"] in allow and iou(truth, row["bbox_xyxy_px"]) >= 0.30 for row in frames[index]["detections"]):
            switches += 1
    return {"assignments": assignments, "ambiguous": ambiguous, "primary_anchor_matched": primary_matched,
            "visible_anchor_count": len(visible), "visible_anchor_reacquired_count": reacquired,
            "identity_switch_count": switches, "allowlist_supported": sorted(allow)}


def generate_truth_blind_trace(source_without_truth: dict[str, Any], frames: list[dict[str, Any]], replay: dict[str, Any], supported: set[str]) -> dict[str, Any]:
    """Generate association/alert state without an expected positive/negative label."""
    association = associate(source_without_truth, frames, replay, supported)
    assignments = association.pop("assignments")
    polygon = source_without_truth["route_polygon_xy_norm"]
    active = False; misses = 0; alerted = False; delivered = 0; suppressed = 0
    first_alert = None; clear_ms = None; route_inside_pressure = 0
    for index, frame in enumerate(frames):
        assigned = assignments[index]
        if assigned is None:
            if active:
                misses += 1
                if misses >= replay["clear_after_consecutive_misses"]:
                    active = False; clear_ms = frame["timestamp_ms"]
        else:
            active = True; misses = 0
            box = assigned["bbox_xyxy_px"]
            relation = classify_contact_point([(box[0] + box[2]) / 2, box[3]], frame_width=frame["width"], frame_height=frame["height"], polygon_xy_norm=polygon, uncertainty_frame_ratio=0.02).relation
            if relation == "inside" and frame["timestamp_ms"] >= source_without_truth["alertable_start_ms"]:
                if not alerted:
                    alerted = True; delivered += 1; first_alert = frame["timestamp_ms"]
                else:
                    suppressed += 1
        for detection in frame["detections"]:
            if detection is assigned:
                continue
            box = detection["bbox_xyxy_px"]
            relation = classify_contact_point([(box[0] + box[2]) / 2, box[3]], frame_width=frame["width"], frame_height=frame["height"], polygon_xy_norm=polygon, uncertainty_frame_ratio=0.02).relation
            if relation == "inside":
                route_inside_pressure += 1
    known_absent = source_without_truth.get("known_not_visible_from_ms")
    clearance_observable = known_absent is not None and clear_ms is not None and association["primary_anchor_matched"]
    clearance = max(0, clear_ms - known_absent) if clearance_observable else None
    censored = 10_000 if known_absent is not None and not clearance_observable else None
    matched = sum(row is not None for row in assignments)
    association.update({
        "frame_count": len(frames), "association_matched_frame_count": matched,
        "association_coverage": matched / len(frames),
        "association_ambiguous_frame_count": sum(association["ambiguous"]),
        "association_ambiguous_frame_rate": sum(association["ambiguous"]) / len(frames),
        "event_hit": first_alert is not None, "first_alert_timestamp_ms": first_alert,
        "first_alert_delay_ms": first_alert - source_without_truth["alertable_start_ms"] if first_alert is not None else None,
        "delivered_alert_count": delivered, "delivered_repeated_alert_count": max(0, delivered - 1),
        "suppressed_duplicate_attempt_count": suppressed,
        "unassigned_route_inside_pressure_count": route_inside_pressure,
        "cooccurrence_triggered_target_event_count": 0,
        "clearance_observable": clearance_observable, "target_exit_clearance_delay_ms": clearance,
        "target_exit_clearance_censored_ms": censored,
    })
    return association


def evaluate_events(model: Any, repo: Path, matrix: dict[str, Any], batch: int) -> dict[str, Any]:
    input_path = repo / matrix["parents"]["exact_frame_input_path"]
    document = load_json(input_path)
    replay = document["replay_contract"]
    sources = document["sources"]
    paths, meta = [], []
    for source in sources:
        for frame in source["frames"]:
            path = resolve_event_image(repo, frame["image_path"])
            require(path.is_file() and sha256_file(path) == frame["image_sha256"], f"event frame drifted: {path}")
            paths.append(path); meta.append((source["event_id"], frame))
    frozen = matrix["frozen_inference"]
    predictions = predict_rows(model, paths, frozen["image_size"], frozen["confidence"], frozen["nms_iou"], frozen["maximum_detections"], batch)
    by_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    import cv2
    for path, (event_id, frame), detections in zip(paths, meta, predictions):
        image = cv2.imread(str(path)); require(image is not None, f"cannot decode {path}")
        by_event[event_id].append({"timestamp_ms": frame["timestamp_ms"], "width": image.shape[1], "height": image.shape[0], "detections": detections})
    event_rows = []
    for source in sources:
        truth = {"expected_class": source["expected_class"], "gate_eligible": source["gate_eligible"], "diagnostic_role": source["diagnostic_role"]}
        truth_blind_source = {key: value for key, value in source.items() if key not in truth}
        trace = generate_truth_blind_trace(truth_blind_source, by_event[source["event_id"]], replay, set(CLASSES))
        scored = {"event_id": source["event_id"], "source_id": source["source_id"], **truth, **trace}
        scored["critical_event_miss"] = truth["expected_class"] == "positive" and not trace["event_hit"]
        scored["target_conditioned_false_alert"] = truth["expected_class"] == "negative" and trace["delivered_alert_count"] > 0
        event_rows.append(scored)
    eligible = [row for row in event_rows if row["gate_eligible"]]
    positives = [row for row in eligible if row["expected_class"] == "positive"]
    negatives = [row for row in eligible if row["expected_class"] == "negative"]
    duration_minutes = sum((source["clip_window_ms"][1] - source["clip_window_ms"][0]) for source in sources if source["gate_eligible"] and source["expected_class"] == "negative") / 60_000
    observable = [row["target_exit_clearance_delay_ms"] for row in eligible if row["clearance_observable"]]
    aggregate = {
        "positive_event_recall": sum(row["event_hit"] for row in positives) / len(positives),
        "critical_event_miss_count": sum(row["critical_event_miss"] for row in positives),
        "target_conditioned_false_alert_count": sum(row["target_conditioned_false_alert"] for row in negatives),
        "target_conditioned_false_alerts_per_minute": sum(row["target_conditioned_false_alert"] for row in negatives) / duration_minutes,
        "delivered_repeated_alert_count": sum(row["delivered_repeated_alert_count"] for row in eligible),
        "unassigned_route_inside_pressure_count": sum(row["unassigned_route_inside_pressure_count"] for row in eligible),
        "observable_clearance_count": len(observable), "censored_clearance_count": sum(row["target_exit_clearance_censored_ms"] is not None for row in eligible),
        "clearance_p50_ms": statistics.median(observable) if observable else None,
        "clearance_p95_ms": sorted(observable)[max(0, math.ceil(0.95 * len(observable)) - 1)] if observable else None,
        "identity_switch_count": sum(row["identity_switch_count"] for row in eligible),
        "association_coverage": sum(row["association_matched_frame_count"] for row in eligible) / sum(row["frame_count"] for row in eligible),
        "association_ambiguous_frame_rate": sum(row["association_ambiguous_frame_count"] for row in eligible) / sum(row["frame_count"] for row in eligible),
        "worst_source_association_coverage": min(row["association_coverage"] for row in eligible),
        "worst_source_ambiguity_rate": max(row["association_ambiguous_frame_rate"] for row in eligible),
    }
    aggregate["worst_source_association_coverage_source"] = min(eligible, key=lambda row: row["association_coverage"])["source_id"]
    aggregate["worst_source_ambiguity_rate_source"] = max(eligible, key=lambda row: row["association_ambiguous_frame_rate"])["source_id"]
    london = next(row for row in event_rows if row["event_id"] == "london_center_marker_intrusion")
    aggregate["london"] = {"event_recall": int(london["event_hit"]),
                           "visible_anchor_recall": london["visible_anchor_reacquired_count"] / london["visible_anchor_count"],
                           "frame_recall": london["association_matched_frame_count"] / london["frame_count"],
                           "matched_frames": london["association_matched_frame_count"], "frames": london["frame_count"]}
    return {"events": event_rows, "aggregate": aggregate}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(); matrix_path = args.matrix.resolve(); matrix = validate_matrix(matrix_path, repo)
    output = args.output.resolve(); require(not output.exists(), f"refusing to overwrite {output}")
    weights = args.weights.resolve(); require(weights.is_file(), f"weights missing: {weights}")
    if args.expected_weights_sha256:
        require(sha256_file(weights) == args.expected_weights_sha256, "weights SHA-256 mismatch")
    training_run = None
    if args.training_receipt:
        training_run = load_json(args.training_receipt.resolve())
        require(training_run["best_weights_sha256"] == sha256_file(weights), "training receipt/weights mismatch")
        require(training_run["matrix_sha256"] == sha256_file(matrix_path), "training receipt/matrix mismatch")
    config_dir = (output.parent / "ultralytics-config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    if args.model_kind == "yoloe_tflite":
        model = StaticTfliteYolo(weights)
        runtime = {"tensorflow": model.tensorflow_version, "decoder": "android_static_class_contract_v1"}
    else:
        from ultralytics import YOLO, YOLOE
        import torch, ultralytics
        model = YOLOE(str(weights)) if args.model_kind == "yoloe" else YOLO(str(weights))
        if args.model_kind == "yoloe":
            model.set_classes(CLASSES)
        runtime = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
    require(list(model.names.values()) == CLASSES, f"model classes drifted: {model.names}")
    effective_matrix = inference_matrix_for_model(matrix, args.model_kind, args.image_size)
    validation = None if args.skip_validation else evaluate_validation(model, args.dataset.resolve(), effective_matrix, args.batch)
    events = evaluate_events(model, repo, effective_matrix, args.batch)
    report = {
        "schema": "blindassist_ustrf_r12d_model_evaluation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": sha256_file(matrix_path), "model_id": args.model_id, "model_kind": args.model_kind,
        "weights_sha256": sha256_file(weights), "classes": CLASSES,
        "training_run": None if training_run is None else {
            "arm_id": training_run["arm_id"], "p2": training_run["p2"], "seed": training_run["seed"],
            "training_receipt_sha256": sha256_file(args.training_receipt.resolve()),
            "shared_backbone_tensor_sha256": training_run["initialization"]["shared_backbone_tensor_sha256"],
        },
        "frozen_inference": effective_matrix["frozen_inference"], "validation": validation, "event_evaluation": events,
        "runtime": runtime,
        "authority": {"seen_diagnostic_not_held_out": True, "human_event_truth_claimed": False,
                      "r13_inventory_read_authorized": False, "production_model_replacement_authorized": False},
    }
    write_json(output, report)
    print("USTRF_R12D_EVAL_OK", args.model_id, events["aggregate"]["positive_event_recall"], events["aggregate"]["london"])
    return report


def inference_matrix_for_model(matrix: dict[str, Any], model_kind: str, image_size: int | None) -> dict[str, Any]:
    """Bind each model family to its preregistered inference geometry."""
    if model_kind in {"yoloe", "yoloe_tflite"}:
        expected = int(matrix["external_reference"]["host_truth_blind_image_size"])
        label = "YOLOE external reference"
    else:
        expected = int(matrix["frozen_inference"]["image_size"])
        label = "paired R1.2d arm"
    requested = expected if image_size is None else int(image_size)
    require(requested == expected, f"{label} image size must remain frozen at {expected}")
    effective = copy.deepcopy(matrix)
    effective["frozen_inference"]["image_size"] = requested
    return effective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["yolo", "yoloe", "yoloe_tflite"], required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
