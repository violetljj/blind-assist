"""Run frozen scale-free traversability R0 on consumed phone RGB sessions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np

from core import BANDS, CAUSAL_WINDOW, decide_relative_open, score_relative_intrusion


REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"
)
CAPTURE_PROTOCOL = "KNOWN_HEIGHT_PHONE_DEVELOPMENT_CAPTURE_R0"
CAPTURE_STATUS = "DEVELOPMENT_CAPTURED_CONSUMED_REFERENCE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_session(root: Path) -> dict[str, Any]:
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    frames = json.loads((root / "frames.json").read_text(encoding="utf-8"))
    intrinsics = json.loads((root / "intrinsics.json").read_text(encoding="utf-8"))
    if receipt.get("protocol_id") != CAPTURE_PROTOCOL or receipt.get("status") != CAPTURE_STATUS:
        raise ValueError(f"{root.name}: unexpected capture authority")
    if len(frames) != 25:
        raise ValueError(f"{root.name}: expected 25 frames")
    if int(intrinsics.get("sensor_orientation_degrees", -1)) != 90:
        raise ValueError(f"{root.name}: only the locked 90-degree orientation is admitted")
    timestamps = [int(row["capture_timestamp_ns"]) for row in frames]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{root.name}: timestamps are not strictly increasing")
    for row in frames:
        path = root / row["rgb_file"]
        if not path.is_file() or sha256(path) != str(row["rgb_sha256"]).upper():
            raise ValueError(f"{root.name}: RGB identity failure")
    return {"receipt": receipt, "frames": frames, "intrinsics": intrinsics}


def render_frame(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, 44), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        "RELATIVE ONLY | NO METRES | DEVELOPMENT",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    label = row["decision"].get("label", row["decision"].get("reason", "UNKNOWN"))
    scores = row["score"].get("scores", {})
    for index, band in enumerate(BANDS):
        left = round(index * width / 3)
        right = round((index + 1) * width / 3)
        selected = label == f"RELATIVELY_OPEN_{band.upper()}"
        color = (65, 135, 65) if selected else (75, 75, 75)
        cv2.rectangle(canvas, (left, height - 58), (right, height), color, -1)
        text = f"{band}: {scores[band]:.3f}" if band in scores else f"{band}: ?"
        cv2.putText(
            canvas,
            text,
            (left + 8, height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if sha256(args.dav2_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected DA V2 checkpoint")
    roots = sorted(path for path in args.captures_root.iterdir() if path.is_dir())
    if len(roots) != 3:
        raise ValueError("expected exactly three consumed phone sessions")
    sessions = [(root, load_session(root)) for root in roots]

    source = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=518,
        precision="fp16" if args.device.startswith("cuda") else "fp32",
    )
    args.output_root.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    session_results: list[dict[str, Any]] = []
    for root, session in sessions:
        history: list[dict[str, float]] = []
        rows: list[dict[str, Any]] = []
        writer: cv2.VideoWriter | None = None
        latency: list[float] = []
        for frame in session["frames"]:
            path = root / frame["rgb_file"]
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise ValueError(f"unable to decode {path}")
            oriented = np.ascontiguousarray(np.rot90(raw, k=3))
            started = time.perf_counter()
            depth, _ = source.infer(cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB), {})
            latency_ms = (time.perf_counter() - started) * 1000.0
            latency.append(latency_ms)
            score = score_relative_intrusion(depth)
            if score["status"] == "VALID":
                history.append(score["scores"])
                history = history[-CAUSAL_WINDOW:]
                decision = decide_relative_open(history)
            else:
                history.clear()
                decision = {"status": "UNKNOWN", "reason": score["reason"]}
            row = {
                "schema": "blindassist_scale_free_traversability_r0_frame_v1",
                "session_id": root.name,
                "frame_id": int(frame["frame_id"]),
                "capture_timestamp_ns": int(frame["capture_timestamp_ns"]),
                "rgb_sha256": frame["rgb_sha256"],
                "score": score,
                "decision": decision,
                "inference_latency_ms": latency_ms,
            }
            rows.append(row)
            all_rows.append(row)
            rendered = render_frame(oriented, row)
            if writer is None:
                video_path = args.output_root / f"{root.name}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (rendered.shape[1], rendered.shape[0]),
                )
                if not writer.isOpened():
                    raise OSError(f"cannot create {video_path}")
            writer.write(rendered)
        if writer is not None:
            writer.release()

        valid_scores = [row for row in rows if row["score"]["status"] == "VALID"]
        eligible = [row for row in rows if row["decision"].get("reason") != "UNKNOWN_WARMUP"]
        labels = [
            row["decision"].get("label", "UNKNOWN")
            for row in eligible
        ]
        counts = Counter(labels)
        modal_fraction = max(counts.values(), default=0) / max(1, len(labels))
        recommendation_count = sum(label.startswith("RELATIVELY_OPEN_") for label in labels)
        band_mad = {}
        for band in BANDS:
            values = np.asarray([row["score"]["scores"][band] for row in valid_scores])
            median = float(np.median(values)) if len(values) else None
            band_mad[band] = (
                float(np.median(np.abs(values - median))) if median is not None else None
            )
        session_results.append(
            {
                "session_id": root.name,
                "frame_count": len(rows),
                "execution_coverage": len(valid_scores) / len(rows),
                "post_warmup_frame_count": len(eligible),
                "non_ambiguous_recommendation_coverage": recommendation_count / max(1, len(eligible)),
                "label_counts": dict(sorted(counts.items())),
                "modal_label_fraction": modal_fraction,
                "band_score_temporal_mad": band_mad,
                "inference_latency_median_ms": statistics.median(latency),
                "video": f"{root.name}.mp4",
            }
        )

    frames_path = args.output_root / "frames.jsonl"
    frames_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    stable = all(
        row["execution_coverage"] >= 0.95 and row["modal_label_fraction"] >= 0.80
        for row in session_results
    )
    terminal = (
        "SCALE_FREE_TRAVERSABILITY_R0_EXECUTES_STABLY_DEVELOPMENT_ONLY"
        if stable
        else "SCALE_FREE_TRAVERSABILITY_R0_UNSTABLE_OR_AMBIGUOUS_DO_NOT_INTEGRATE"
    )
    result = {
        "schema": "blindassist_scale_free_traversability_r0_result_v1",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "terminal": terminal,
        "authority": {
            "development_only": True,
            "metric_distance": False,
            "alert": False,
            "safety": False,
            "production": False,
        },
        "truth_inputs_read": [],
        "camera_height_read": False,
        "samsung_quick_measure_read": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "session_count": len(session_results),
        "frame_count": len(all_rows),
        "session_results": session_results,
        "frames_jsonl": frames_path.name,
        "frames_jsonl_sha256": sha256(frames_path),
        "limitations": [
            "No independent truth was used, so stability is not accuracy.",
            "All three sessions are fixed-camera and cannot evaluate approach motion.",
            "Relative labels do not mean clear, safe, blocked, or metric distance.",
        ],
    }
    write_json_new(args.output_root / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
