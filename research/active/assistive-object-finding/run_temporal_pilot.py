#!/usr/bin/env python3
"""Real CPU KLT tracking with cached Grounding DINO re-localization actions.

Tracking is executed on actual adjacent sampled images. Detector outputs and
their measured runtimes are replayed, so estimated costs are not online latency.
No ground-truth field is used in policies, tracking, or action selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def row_key(row):
    return str(row["sequence_id"]), int(row["frame_idx"])


def observe(path):
    rows = []
    for raw in read_jsonl(path):
        rows.append({field: raw.get(field) for field in
                     ("sequence_id", "frame_idx", "query", "image_path", "timestamp_s")})
    return sorted(rows, key=row_key)


def detect(row):
    candidates = sorted(row["candidates"], key=lambda c: (-c["score"], tuple(c["bbox_xyxy"])))
    if candidates and candidates[0]["score"] >= 0.25:
        return np.asarray(candidates[0]["bbox_xyxy"], dtype=np.float64), float(candidates[0]["score"])
    return None, 0.0


def gray_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    height, width = image.shape
    scale = min(1.0, 640 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image, scale, width, height


def points_in_box(gray, box, scale):
    if box is None:
        return None
    x1, y1, x2, y2 = box * scale
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(gray.shape[1], math.ceil(x2)), min(gray.shape[0], math.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01,
                                  minDistance=3, mask=mask, blockSize=3)


def track(previous, current, box, points, scale, width, height):
    diagnostic = {"input_points": 0 if points is None else len(points),
                  "valid_points": 0, "median_fb_error_px": None, "valid_transform": False}
    if box is None or points is None or len(points) < 6:
        return None, None, diagnostic
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if forward is None:
        return None, None, diagnostic
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if backward is None:
        return None, None, diagnostic
    fb = np.linalg.norm(points[:, 0] - backward[:, 0], axis=1)
    finite = np.isfinite(forward[:, 0]).all(1) & np.isfinite(fb)
    valid = finite & (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
    if valid.any():
        diagnostic["median_fb_error_px"] = float(np.median(fb[valid]))
    valid &= fb <= 1.5
    valid &= ((forward[:, 0, 0] >= 0) & (forward[:, 0, 0] < current.shape[1]) &
              (forward[:, 0, 1] >= 0) & (forward[:, 0, 1] < current.shape[0]))
    diagnostic["valid_points"] = int(valid.sum())
    if valid.sum() < 6:
        return None, None, diagnostic
    affine, inliers = cv2.estimateAffinePartial2D(
        points[valid, 0], forward[valid, 0], method=cv2.RANSAC,
        ransacReprojThreshold=2.0, maxIters=100, confidence=0.99)
    if affine is None or not np.isfinite(affine).all() or int(inliers.sum()) < 6:
        return None, None, diagnostic
    scale_change = math.hypot(affine[0, 0], affine[0, 1])
    if not 0.7 <= scale_change <= 1.4:
        return None, None, diagnostic
    x1, y1, x2, y2 = box * scale
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
    moved = corners @ affine.T / scale
    proposed = np.array([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()])
    proposed[[0, 2]] = np.clip(proposed[[0, 2]], 0, width)
    proposed[[1, 3]] = np.clip(proposed[[1, 3]], 0, height)
    if proposed[2] <= proposed[0] or proposed[3] <= proposed[1]:
        return None, None, diagnostic
    diagnostic["valid_transform"] = True
    diagnostic["affine_scale"] = scale_change
    kept = forward[valid][inliers.reshape(-1) == 1]
    return proposed, kept, diagnostic


def run_policy(name, interval, rows, cache, images, output, max_seconds, global_start,
               schedule_by_sequence=None):
    predictions_path = output / (name + "_predictions.jsonl")
    if predictions_path.exists():
        raise ValueError(f"Preserve prior output: {predictions_path}")
    last_seq, previous, points, box, score = None, None, None, None, 0.0
    last_call, ordinal = -1000, -1
    calls, count, klt_seconds, reset_seconds, cached_detector_seconds = 0, 0, 0.0, 0.0, 0.0
    sequence_calls = {}
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            if time.perf_counter() - global_start > max_seconds:
                raise TimeoutError("CPU temporal pilot wall budget exhausted")
            key = row_key(row)
            gray, image_scale, width, height = images[row["image_path"]]
            if row["sequence_id"] != last_seq:
                last_seq, previous, points, box, score = row["sequence_id"], None, None, None, 0.0
                last_call, ordinal = -1000, -1
            ordinal += 1
            diagnostic = {"valid_points": 0, "median_fb_error_px": None, "valid_transform": False}
            tick = time.perf_counter()
            scheduled = (ordinal in schedule_by_sequence[last_seq] if schedule_by_sequence is not None
                         else interval is not None and ordinal % interval == 0)
            if previous is not None and not scheduled:
                box, points, diagnostic = track(previous, gray, box, points, image_scale, width, height)
            step_klt = time.perf_counter() - tick
            klt_seconds += step_klt
            gap = ordinal - last_call
            if schedule_by_sequence is not None or interval is not None:
                call = scheduled
                reason = ("posthoc_uniform_matched" if schedule_by_sequence is not None else "fixed_interval") if call else "track"
            else:
                low_quality = (box is None or diagnostic["valid_points"] < 6 or
                               diagnostic["median_fb_error_px"] is None or
                               diagnostic["median_fb_error_px"] > 1.5)
                # Min gap two bounds call budget at the periodic2 upper endpoint.
                call = ordinal == 0 or gap >= 8 or (gap >= 2 and low_quality)
                reason = "initial" if ordinal == 0 else "max_gap" if gap >= 8 else "low_klt_quality" if call else "track"
            if call:
                box, score = detect(cache[key])
                last_call = ordinal
                calls += 1
                sequence_calls[last_seq] = sequence_calls.get(last_seq, 0) + 1
                cached_detector_seconds += float(cache[key]["elapsed_s"])
                tick = time.perf_counter()
                points = points_in_box(gray, box, image_scale)
                reset_seconds += time.perf_counter() - tick
            elif box is None:
                score = 0.0
            previous = gray
            count += 1
            prediction = {**row, "width": width, "height": height,
                          "candidates": [{"bbox_xyxy": box.tolist(), "score": score}] if box is not None else [],
                          "presence_score": score if box is not None else 0.0,
                          "elapsed_s": step_klt, "selection_method": name,
                          "detector_called": call, "action_reason": reason, "klt": diagnostic,
                          "relocation_source": "cached_actual_DINO_output" if call else None}
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
    return {"policy": name, "frames": count, "detector_calls": calls,
            "detector_call_ratio": calls / count if count else None,
            "actual_klt_cpu_s": klt_seconds, "actual_point_initialization_cpu_s": reset_seconds,
            "cached_measured_detector_s_sum": cached_detector_seconds,
            "estimated_component_cost_s": cached_detector_seconds + klt_seconds + reset_seconds,
            "actual_replay_wall_s": time.perf_counter() - started, "calls_by_sequence": sequence_calls,
            "prediction_sha256": sha256(predictions_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    start = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    config = {"manifest_sha256": sha256(args.manifest), "predictions_sha256": sha256(args.predictions),
              "source_sha256": sha256(__file__), "backend": "TASK_NOT_GPU_SUITABLE",
              "opencv": cv2.__version__, "opencv_threads": 2,
              "timing_clock": "perf_counter QueryPerformanceCounter",
              "timing_resolution_s": time.get_clock_info("perf_counter").resolution,
              "resize_long_side": 640, "presence_threshold": 0.25,
              "fixed_intervals_sampled_frames": [2, 4, 8],
              "adaptive": {"min_gap": 2, "max_gap": 8, "minimum_valid_points": 6,
                           "forward_backward_max_error_px": 1.5},
              "matched_diagnostic": "Uniformly schedule adaptive observed per-clip call counts; posthoc, no GT",
              "tracking": "PyrLK forward/backward + RANSAC partial affine, no GT",
              "cost_limit": "Cached DINO actions, component cost estimate is not online end-to-end latency",
              "time_unit": "sampled frames; no FPS fabricated for timestamp-null data"}
    save_json(output / "config.json", config)
    try:
        cache = {row_key(row): row for row in read_jsonl(args.predictions)}
        rows = [row for row in observe(args.manifest) if row_key(row) in cache]
        if not rows:
            raise ValueError("No matching predictions")
        images = {}
        tick = time.perf_counter()
        for row in rows:
            if row["image_path"] not in images:
                images[row["image_path"]] = gray_image(row["image_path"])
        image_seconds = time.perf_counter() - tick
        results = []
        for name, interval in [("periodic2", 2), ("periodic4", 4), ("periodic8", 8), ("adaptive8", None)]:
            result = run_policy(name, interval, rows, cache, images, output, args.max_seconds, start)
            results.append(result)
            save_json(output / "progress.json", {"completed_policies": len(results), "total_policies": 5,
                       "elapsed_s": time.perf_counter() - start, "last_result": result})
            print(json.dumps(result), flush=True)
        counts = results[-1]["calls_by_sequence"]
        sequence_lengths = {}
        for row in rows:
            seq = str(row["sequence_id"])
            sequence_lengths[seq] = sequence_lengths.get(seq, 0) + 1
        schedules = {seq: set(np.rint(np.linspace(0, length - 1, counts[seq])).astype(int).tolist())
                     for seq, length in sequence_lengths.items()}
        result = run_policy("uniform_matched_adaptive_count_posthoc", None, rows, cache, images,
                            output, args.max_seconds, start, schedules)
        result["budget_source"] = "posthoc complete-clip adaptive call count, no GT"
        if result["calls_by_sequence"] != counts:
            raise ValueError("Uniform matched budget mismatch")
        results.append(result)
        print(json.dumps(result), flush=True)
        save_json(output / "progress.json", {"completed_policies": len(results), "total_policies": 5,
                  "elapsed_s": time.perf_counter() - start, "last_result": result})
        receipt = {"status": "SUCCESS", "rows": len(rows), "unique_images": len(images),
                   "actual_shared_image_read_resize_s": image_seconds,
                   "total_cpu_process_wall_s": time.perf_counter() - start, "policies": results,
                   "limits": ["No detector executed here: previously measured exact outputs replayed",
                              "Grayscale KLT operates on sampled video frames, which may be widely spaced",
                              "No labels used to choose actions or fixed constants",
                              "Periodic and adaptive policies have potentially unequal detector-call budgets",
                              "Uniform-matched policy uses posthoc complete-clip adaptive call counts; not an online policy"]}
        save_json(output / "terminal.json", receipt)
        return 0
    except BaseException as exc:
        save_json(output / "terminal.json", {"status": "FAILED", "error": str(exc),
                  "elapsed_s": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
