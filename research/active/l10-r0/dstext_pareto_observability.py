from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.process_img import get_rotate_crop_image

import artvideo_ocr_replay as l10_text


MEDIA_ENTRY = "Street_View_Indoor/Video_191_9_1.mp4"
ANNOTATION_ENTRY = "Street_View_Indoor/Video_191_9_1_GT.xml"
HORIZON_ANNOTATED_FRAMES = 3
MIN_TRACK_FRAMES = 12
MIN_ALIGNED = 20
MIN_OPPOSED = 20
MIN_MEAN_GAIN_DELTA = 0.10
MIN_IMPROVEMENT_RATE_DELTA = 0.15
SEMANTIC_GATE = 0.58


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def read_annotations(annotation_archive: Path) -> tuple[bytes, dict[int, list[dict[str, Any]]]]:
    with zipfile.ZipFile(annotation_archive) as archive:
        xml_bytes = archive.read(ANNOTATION_ENTRY)
    root = ET.fromstring(xml_bytes)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in root.findall("frame"):
        frame_id = int(frame.attrib["ID"])
        rows: list[dict[str, Any]] = []
        for element in frame.findall("object"):
            points = [
                [float(point.attrib["x"]), float(point.attrib["y"])]
                for point in element.findall("Point")
            ]
            rows.append(
                {
                    "track_id": str(element.attrib["ID"]),
                    "transcription": str(element.attrib.get("Transcription", "")),
                    "category": str(element.attrib.get("category", "")),
                    "language": str(element.attrib.get("language", "")),
                    "points": points,
                }
            )
        by_frame[frame_id] = rows
    return xml_bytes, by_frame


def eligible_track_ids(by_frame: dict[int, list[dict[str, Any]]]) -> set[str]:
    counts: Counter[str] = Counter()
    transcriptions: dict[str, set[str]] = defaultdict(set)
    for rows in by_frame.values():
        for row in rows:
            track_id = row["track_id"]
            counts[track_id] += 1
            transcriptions[track_id].add(row["transcription"])
    return {
        track_id
        for track_id, count in counts.items()
        if count >= MIN_TRACK_FRAMES
        and len(transcriptions[track_id]) == 1
        and next(iter(transcriptions[track_id])) != "###"
        and len(normalized(next(iter(transcriptions[track_id])))) >= 3
    }


def materialize_media(media_archive: Path, work_dir: Path) -> Path:
    target = work_dir / "source" / Path(MEDIA_ENTRY).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    with zipfile.ZipFile(media_archive) as archive:
        with archive.open(MEDIA_ENTRY) as source, target.open("wb") as destination:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
    return target


def observe(
    engine: RapidOCR,
    frame: np.ndarray,
    frame_id: int,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    quad = np.asarray(row["points"], dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    if (
        float(quad[:, 0].min()) < 0.0
        or float(quad[:, 1].min()) < 0.0
        or float(quad[:, 0].max()) >= width
        or float(quad[:, 1].max()) >= height
    ):
        return None
    crop = get_rotate_crop_image(frame, quad)
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    started = time.perf_counter()
    output = engine(
        crop,
        use_det=False,
        use_cls=False,
        use_rec=True,
        return_word_box=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    texts = output.txts if output.txts is not None else ()
    scores = output.scores if output.scores is not None else ()
    recognized = str(texts[0]) if texts else ""
    confidence = float(scores[0]) if scores else 0.0
    center_x = float(quad[:, 0].mean()) / width
    lexical = float(l10_text.lexical(row["transcription"], recognized))
    return {
        "frame_id": frame_id,
        "track_id": row["track_id"],
        "transcription": row["transcription"],
        "category": row["category"],
        "language": row["language"],
        "rectified_width_px": int(crop.shape[1]),
        "rectified_height_px": int(crop.shape[0]),
        "laplacian_variance": round(sharpness, 6),
        "horizontal_center_error": round(abs(center_x - 0.5), 6),
        "recognized_text": recognized,
        "recognition_confidence": round(confidence, 6),
        "semantic_lexical": round(lexical, 6),
        "semantic_gate_passed": lexical >= SEMANTIC_GATE,
        "recognition_wall_ms": round(elapsed_ms, 4),
    }


def collect_observations(
    engine: RapidOCR,
    video_path: Path,
    by_frame: dict[int, list[dict[str, Any]]],
    eligible: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    observations: list[dict[str, Any]] = []
    rejected = Counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            for row in by_frame.get(frame_index, []):
                if row["track_id"] not in eligible:
                    continue
                observation = observe(engine, frame, frame_index, row)
                if observation is None:
                    rejected["invalid_or_out_of_frame_crop"] += 1
                else:
                    observations.append(observation)
    finally:
        capture.release()
    return observations, {
        "video_frame_count_declared": expected_frames,
        "video_frames_decoded": frame_index,
        "annotation_frames": len(by_frame),
        "eligible_tracks": len(eligible),
        "observations": len(observations),
        "rejected": dict(rejected),
    }


def quality_vector(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        math.log(float(row["rectified_height_px"])),
        math.log(float(row["laplacian_variance"]) + 1.0),
        -float(row["horizontal_center_error"]),
    )


def pareto_label(current: tuple[float, ...], future: tuple[float, ...]) -> str:
    no_worse = all(after >= before for before, after in zip(current, future, strict=True))
    strictly_better = any(after > before for before, after in zip(current, future, strict=True))
    no_better = all(after <= before for before, after in zip(current, future, strict=True))
    strictly_worse = any(after < before for before, after in zip(current, future, strict=True))
    if no_worse and strictly_better:
        return "PARETO_ALIGNED"
    if no_better and strictly_worse:
        return "PARETO_OPPOSED"
    return "MIXED"


def build_transitions(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_track[row["track_id"]].append(row)
    transitions: list[dict[str, Any]] = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda row: int(row["frame_id"]))
        for index in range(len(rows) - HORIZON_ANNOTATED_FRAMES):
            current = rows[index]
            future = rows[index + HORIZON_ANNOTATED_FRAMES]
            current_quality = quality_vector(current)
            future_quality = quality_vector(future)
            semantic_gain = float(future["semantic_lexical"]) - float(current["semantic_lexical"])
            transitions.append(
                {
                    "track_id": track_id,
                    "transcription": current["transcription"],
                    "frame_id": current["frame_id"],
                    "future_frame_id": future["frame_id"],
                    "pareto_label": pareto_label(current_quality, future_quality),
                    "quality_delta": {
                        "log_height": round(future_quality[0] - current_quality[0], 6),
                        "log_sharpness": round(future_quality[1] - current_quality[1], 6),
                        "negative_center_error": round(future_quality[2] - current_quality[2], 6),
                    },
                    "current_semantic_lexical": current["semantic_lexical"],
                    "future_semantic_lexical": future["semantic_lexical"],
                    "semantic_gain": round(semantic_gain, 6),
                    "semantic_improved": semantic_gain > 0.0,
                    "semantic_gate_crossed": (
                        float(current["semantic_lexical"]) < SEMANTIC_GATE
                        <= float(future["semantic_lexical"])
                    ),
                }
            )
    return transitions


def rate(value: int, total: int) -> float | None:
    return round(value / total, 4) if total else None


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [float(row["semantic_gain"]) for row in rows]
    improved = sum(bool(row["semantic_improved"]) for row in rows)
    crossed = sum(bool(row["semantic_gate_crossed"]) for row in rows)
    return {
        "transitions": len(rows),
        "mean_semantic_gain": round(statistics.fmean(gains), 4) if gains else None,
        "median_semantic_gain": round(statistics.median(gains), 4) if gains else None,
        "semantic_improved": improved,
        "semantic_improvement_rate": rate(improved, len(rows)),
        "semantic_gate_crossed": crossed,
        "semantic_gate_crossing_rate": rate(crossed, len(rows)),
    }


def adjudicate(transitions: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    aligned = [row for row in transitions if row["pareto_label"] == "PARETO_ALIGNED"]
    opposed = [row for row in transitions if row["pareto_label"] == "PARETO_OPPOSED"]
    aligned_summary = summarize_arm(aligned)
    opposed_summary = summarize_arm(opposed)
    supported = len(aligned) >= MIN_ALIGNED and len(opposed) >= MIN_OPPOSED
    gain_delta = None
    improvement_delta = None
    crossing_noninferior = False
    if aligned and opposed:
        gain_delta = round(
            float(aligned_summary["mean_semantic_gain"])
            - float(opposed_summary["mean_semantic_gain"]),
            4,
        )
        improvement_delta = round(
            float(aligned_summary["semantic_improvement_rate"])
            - float(opposed_summary["semantic_improvement_rate"]),
            4,
        )
        crossing_noninferior = (
            float(aligned_summary["semantic_gate_crossing_rate"])
            >= float(opposed_summary["semantic_gate_crossing_rate"])
        )
    checks = {
        "minimum_aligned_transitions": len(aligned) >= MIN_ALIGNED,
        "minimum_opposed_transitions": len(opposed) >= MIN_OPPOSED,
        "mean_semantic_gain_delta_pass": gain_delta is not None and gain_delta >= MIN_MEAN_GAIN_DELTA,
        "semantic_improvement_rate_delta_pass": (
            improvement_delta is not None and improvement_delta >= MIN_IMPROVEMENT_RATE_DELTA
        ),
        "aligned_gate_crossing_rate_not_below_opposed": crossing_noninferior,
    }
    if not supported:
        decision = "SC16_NOT_EVALUABLE_INSUFFICIENT_PARETO_TRANSITIONS"
    elif all(checks.values()):
        decision = "SC16_PARETO_OBSERVABILITY_SEMANTIC_GAIN_SIGNAL"
    else:
        decision = "SC16_PARETO_OBSERVABILITY_SEMANTIC_GAIN_GATE_NOT_MET"
    comparison = {
        "pareto_aligned": aligned_summary,
        "pareto_opposed": opposed_summary,
        "mean_semantic_gain_delta": gain_delta,
        "semantic_improvement_rate_delta": improvement_delta,
    }
    return decision, checks, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-archive", type=Path, required=True)
    parser.add_argument("--annotation-archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_payload_or_outcome_access"):
        raise ValueError("protocol is not marked frozen")
    output_dir = args.output.parent
    video_path = materialize_media(args.media_archive, output_dir)
    xml_bytes, by_frame = read_annotations(args.annotation_archive)
    eligible = eligible_track_ids(by_frame)

    engine_started = time.perf_counter()
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(args.models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    engine_init_s = time.perf_counter() - engine_started
    run_started = time.perf_counter()
    observations, source_counts = collect_observations(engine, video_path, by_frame, eligible)
    transitions = build_transitions(observations)
    decision, checks, comparison = adjudicate(transitions)
    action_counts = Counter(row["pareto_label"] for row in transitions)
    recognition_times = [float(row["recognition_wall_ms"]) for row in observations]
    model_path = args.models / "PP-OCRv6_rec_small.onnx"
    result = {
        "schema_version": 1,
        "experiment": "l10_sc16_dstext_pareto_observability_v0",
        "decision": decision,
        "protocol_sha256": sha256_file(args.protocol),
        "source": {
            "media_archive_sha256": sha256_file(args.media_archive),
            "annotation_archive_sha256": sha256_file(args.annotation_archive),
            "media_entry": MEDIA_ENTRY,
            "media_entry_sha256": sha256_file(video_path),
            "annotation_entry": ANNOTATION_ENTRY,
            "annotation_entry_sha256": hashlib.sha256(xml_bytes).hexdigest(),
            **source_counts,
        },
        "runtime": {
            "rapidocr_version": importlib.metadata.version("rapidocr"),
            "opencv_version": cv2.__version__,
            "recognizer_model_sha256": sha256_file(model_path),
            "engine_init_s": round(engine_init_s, 4),
            "recognition_calls": len(observations),
            "recognition_wall_ms_median": round(statistics.median(recognition_times), 4),
            "recognition_wall_ms_p95": round(float(np.percentile(recognition_times, 95)), 4),
            "run_wall_s": round(time.perf_counter() - run_started, 4),
        },
        "transitions": {
            "total": len(transitions),
            "by_pareto_label": dict(sorted(action_counts.items())),
        },
        "comparison": comparison,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "claim_boundary": protocol["claim_ceiling"],
        "observations": observations,
        "transition_rows": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "source": source_counts,
                "transitions": result["transitions"],
                "comparison": comparison,
                "gate": result["gate"],
                "runtime": result["runtime"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
