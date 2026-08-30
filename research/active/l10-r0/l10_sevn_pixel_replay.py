#!/usr/bin/env python3
"""Run the first high-resolution, pixel-derived SEVN address-door replay.

The evaluator-selected action is fixed before inference so this canary isolates
the pixel question: after the prescribed active observation, can public OCR and
a generic portal model bind the requested house number to the correct door?
SEVN labels are read only after the runtime prediction has been produced.

Panoramas are decoded directly from the official ZIP.  No extracted panorama,
viewport image, or model cache is written by this program; the durable output is
one JSON result containing input receipts, per-episode predictions, and scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file, utc_now
from l10_sevn_panolab import validate_public, validate_truth


ROOT = Path(__file__).resolve().parents[3]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream Zenodo receipt uses MD5.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"missing frozen input: {path}")
    actual = sha256_file(path)
    require(actual == spec["sha256"], f"SHA256 mismatch: {path}: {actual}")
    return path


def load_module(path: Path) -> Any:
    module_spec = importlib.util.spec_from_file_location("sevn_mirror_suppress", path)
    require(module_spec is not None and module_spec.loader is not None, "postprocessor import failed")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def canonical_token(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()


def polygon_box(points: Any) -> list[float]:
    array = np.asarray(points, dtype=float)
    return [
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 0].max()),
        float(array[:, 1].max()),
    ]


def box_center(box: list[float]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def center_in_box(candidate: list[float], truth: list[float]) -> bool:
    x, y = box_center(candidate)
    return truth[0] <= x <= truth[2] and truth[1] <= y <= truth[3]


def text_portal_distance(text_box: list[float], portal_box: list[float], width: int, height: int) -> float:
    tx, ty = box_center(text_box)
    dx = max(portal_box[0] - tx, 0.0, tx - portal_box[2]) / width
    dy = max(portal_box[1] - ty, 0.0, ty - portal_box[3]) / height
    return math.hypot(dx, dy)


def render_viewport(
    panorama: np.ndarray,
    panorama_angle_degrees: float,
    viewport_heading_degrees: float,
    horizontal_fov_degrees: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = panorama.shape[:2]
    crop_width = round(width * horizontal_fov_degrees / 360.0)
    wrapped_offset = (viewport_heading_degrees - panorama_angle_degrees + 180.0) % 360.0
    center_x = width - wrapped_offset * width / 360.0
    start_x = round(center_x - crop_width / 2.0)
    indices = np.arange(start_x, start_x + crop_width) % width
    viewport = np.ascontiguousarray(panorama[:, indices])
    pixel_digest = hashlib.sha256()
    pixel_digest.update(np.asarray(viewport.shape, dtype=np.int64).tobytes())
    pixel_digest.update(viewport.tobytes())
    return viewport, {
        "source_shape_hwc": [height, width, int(panorama.shape[2])],
        "viewport_shape_hwc": list(viewport.shape),
        "panorama_angle_degrees": panorama_angle_degrees,
        "viewport_heading_degrees": viewport_heading_degrees,
        "horizontal_fov_degrees": horizontal_fov_degrees,
        "center_x_unwrapped": round(center_x, 6),
        "start_x_unwrapped": start_x,
        "viewport_pixel_sha256": pixel_digest.hexdigest(),
    }


def annotation_to_viewport(
    annotation: dict[str, Any],
    render: dict[str, Any],
    low_width: float,
    low_height: float,
) -> list[float]:
    height, width, _ = render["source_shape_hwc"]
    crop_width = render["viewport_shape_hwc"][1]
    start_x = int(render["start_x_unwrapped"])
    raw_x1 = float(annotation["x_min"]) * width / low_width
    raw_x2 = float(annotation["x_max"]) * width / low_width
    raw_center = 0.5 * (raw_x1 + raw_x2)
    crop_center = start_x + crop_width / 2.0
    wrap_count = round((crop_center - raw_center) / width)
    x1 = raw_x1 + wrap_count * width - start_x
    x2 = raw_x2 + wrap_count * width - start_x
    return [
        x1,
        float(annotation["y_min"]) * height / low_height,
        x2,
        float(annotation["y_max"]) * height / low_height,
    ]


def ocr_detections(engine: RapidOCR, image: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    output = engine(image, return_word_box=True)
    seconds = time.perf_counter() - started
    boxes = output.boxes if output.boxes is not None else ()
    texts = output.txts if output.txts is not None else ()
    scores = output.scores if output.scores is not None else ()
    word_lines = output.word_results if output.word_results is not None else ()
    rows: list[dict[str, Any]] = []
    for index, (box, text, score) in enumerate(zip(boxes, texts, scores)):
        words = word_lines[index] if index < len(word_lines) else ()
        word_rows = [
            {
                "text": str(word_text),
                "canonical": canonical_token(word_text),
                "score": float(word_score),
                "box_xyxy": polygon_box(word_box),
            }
            for word_text, word_score, word_box in words
        ]
        rows.append({
            "text": str(text),
            "canonical": canonical_token(text),
            "score": float(score),
            "box_xyxy": polygon_box(box),
            "words": word_rows,
        })
    return rows, seconds


def word_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row["words"]:
            candidates.extend(row["words"])
        else:
            candidates.append({key: row[key] for key in ("text", "canonical", "score", "box_xyxy")})
    return candidates


def portal_detections(
    model: YOLO,
    postprocessor: Any,
    image: np.ndarray,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, float]:
    started = time.perf_counter()
    result = model.predict(
        source=image,
        device=int(contract["cuda_device"]),
        conf=float(contract["predict_confidence"]),
        imgsz=int(contract["image_size"]),
        verbose=False,
    )[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    raw = postprocessor.dets_from_result(result)
    final = postprocessor.finalize(raw, out_conf=float(contract["output_confidence"]))
    portals = [
        {
            "class_id": int(row["cls"]),
            "class_name": "doorway" if int(row["cls"]) == 0 else "door",
            "confidence": float(row["conf"]),
            "box_xyxy": [float(value) for value in row["box"]],
        }
        for row in final
        if int(row["cls"]) in {0, 1}
    ]
    portals.sort(key=lambda row: (-row["confidence"], row["class_id"], tuple(row["box_xyxy"])))
    return portals, len(raw), seconds


def infer_binding(
    image: np.ndarray,
    mission: dict[str, Any],
    ocr_engine: RapidOCR,
    portal_model: YOLO,
    postprocessor: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    ocr_rows, ocr_seconds = ocr_detections(ocr_engine, image)
    portals, raw_detection_count, portal_seconds = portal_detections(
        portal_model, postprocessor, image, contract["portal_model"]
    )
    target = canonical_token(mission["house_number"])
    exact = [
        row for row in word_candidates(ocr_rows)
        if row["canonical"] == target and row["score"] >= float(contract["ocr"]["minimum_score"])
    ]
    pairs = []
    height, width = image.shape[:2]
    for text_row in exact:
        for portal in portals:
            distance = text_portal_distance(text_row["box_xyxy"], portal["box_xyxy"], width, height)
            pairs.append((
                distance,
                -portal["confidence"],
                -text_row["score"],
                tuple(portal["box_xyxy"]),
                text_row,
                portal,
            ))
    pairs.sort(key=lambda row: row[:4])
    selected = None
    state = "UNKNOWN_TARGET_TEXT"
    if exact and not portals:
        state = "UNKNOWN_PORTAL"
    elif pairs:
        best = pairs[0]
        if best[0] <= float(contract["association"]["maximum_normalized_box_gap"]):
            state = "PIXEL_BOUND_PORTAL_PROPOSAL"
            selected = {
                "normalized_box_gap": round(float(best[0]), 8),
                "text": best[4],
                "portal": best[5],
            }
        else:
            state = "UNKNOWN_UNJOINED_TEXT_AND_PORTAL"
    return {
        "state": state,
        "target_house_number_canonical": target,
        "exact_target_text_candidates": exact,
        "ocr_detections": ocr_rows,
        "portal_candidates": portals,
        "selected_binding": selected,
        "runtime_receipt": {
            "ocr_seconds": round(ocr_seconds, 6),
            "portal_seconds": round(portal_seconds, 6),
            "raw_portal_model_detection_count": raw_detection_count,
        },
    }


def summarize(rows: list[dict[str, Any]], gate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    total = len(rows)
    text_visible = sum(row["evaluation"]["target_house_number_visible"] for row in rows)
    exact_ocr = sum(row["evaluation"]["exact_target_text"] for row in rows)
    exact_ocr_when_visible = sum(
        row["evaluation"]["exact_target_text"] and row["evaluation"]["target_house_number_visible"]
        for row in rows
    )
    portal = sum(bool(row["runtime_output"]["portal_candidates"]) for row in rows)
    joined = sum(row["runtime_output"]["selected_binding"] is not None for row in rows)
    correct = sum(row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR" for row in rows)
    wrong = sum(row["evaluation"]["outcome"] == "WRONG_DOOR" for row in rows)
    unknown = total - correct - wrong
    metrics = {
        "episode_count": total,
        "truth_visible_house_number_opportunities": text_visible,
        "exact_target_text": exact_ocr,
        "exact_target_text_rate": round(exact_ocr / total, 6),
        "exact_target_text_when_truth_visible": exact_ocr_when_visible,
        "exact_target_text_when_truth_visible_rate": round(exact_ocr_when_visible / text_visible, 6),
        "portal_proposal": portal,
        "portal_proposal_rate": round(portal / total, 6),
        "joined_text_portal_proposal": joined,
        "joined_text_portal_proposal_rate": round(joined / total, 6),
        "correct_target_door_bindings": correct,
        "correct_target_door_binding_rate": round(correct / total, 6),
        "wrong_door_bindings": wrong,
        "unknown": unknown,
        "unknown_rate": round(unknown / total, 6),
    }
    gates = {
        "all_episodes_rendered": total == int(gate["episode_count"]),
        "minimum_visible_text_exact_ocr": exact_ocr_when_visible >= int(gate["minimum_visible_text_exact_ocr"]),
        "minimum_correct_target_door_bindings": correct >= int(gate["minimum_correct_target_door_bindings"]),
        "maximum_wrong_door_bindings": wrong <= int(gate["maximum_wrong_door_bindings"]),
    }
    return metrics, gates


def scenario_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for scenario in sorted({row["scenario_class"] for row in rows}):
        subset = [row for row in rows if row["scenario_class"] == scenario]
        counts = Counter(row["evaluation"]["outcome"] for row in subset)
        answer[scenario] = {
            "episodes": len(subset),
            "correct_target_door": counts["CORRECT_TARGET_DOOR"],
            "wrong_door": counts["WRONG_DOOR"],
            "unknown": counts["UNKNOWN"],
            "exact_target_text": sum(row["evaluation"]["exact_target_text"] for row in subset),
        }
    return answer


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    source_path = args.source.resolve()
    truth_path = args.truth.resolve()
    archive_path = args.highres_archive.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"output already exists: {output_path}")
    protocol = read_json(protocol_path)
    require(protocol["schema"] == "blindassist-l10-sevn-highres-pixel-replay-protocol-v1", "protocol schema mismatch")
    require(sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "evaluator hash mismatch")
    require(source_path == verify_sha256(protocol["frozen_inputs"]["public_source"]), "source path mismatch")
    require(truth_path == verify_sha256(protocol["frozen_inputs"]["evaluator_truth"]), "truth path mismatch")
    model_path = verify_sha256(protocol["models"]["portal_model"]["weights"])
    postprocessor_path = verify_sha256(protocol["models"]["portal_model"]["postprocessor"])
    verify_sha256(protocol["models"]["portal_model"]["model_card"])
    ocr_model_root = resolve(protocol["models"]["ocr"]["model_root"])
    for filename, expected in protocol["models"]["ocr"]["sha256"].items():
        actual = sha256_file(ocr_model_root / filename)
        require(actual == expected, f"OCR model hash mismatch: {filename}")

    archive = protocol["high_resolution_archive"]
    require(archive_path.is_file(), f"missing archive: {archive_path}")
    require(archive_path.stat().st_size == int(archive["bytes"]), "high-resolution archive byte size mismatch")
    observed_archive_md5 = md5_file(archive_path)
    require(observed_archive_md5 == archive["md5"], "high-resolution archive Zenodo MD5 mismatch")
    print(json.dumps({"archive_md5_verified": observed_archive_md5}, ensure_ascii=False), flush=True)

    runtime = protocol["runtime"]
    versions = {
        "torch": importlib.metadata.version("torch"),
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
    }
    require(versions == runtime["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable")

    source = read_json(source_path)
    truth = read_json(truth_path)
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    validate_public(source, per_scenario)
    validate_truth(truth, source, per_scenario)
    require(len(source["episodes"]) == int(protocol["gate"]["episode_count"]), "episode count mismatch")

    postprocessor = load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    expected_names = {0: "doorway", 1: "door", 2: "people", 3: "window", 4: "mirror"}
    require(portal_model.task == "segment" and portal_model.names == expected_names, "portal model ontology mismatch")
    ocr_engine = RapidOCR(params={
        "Global.model_root_dir": str(ocr_model_root),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(protocol["pixel_contract"]["ocr"]["intra_op_threads"]),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })

    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path, "r") as archive_zip:
        for sequence, episode in enumerate(source["episodes"], start=1):
            episode_id = episode["episode_id"]
            episode_truth = truth["episodes"][episode_id]
            scenario = episode_truth["scenario_class"]
            action = protocol["action_control"]["scenario_to_action"][scenario]
            start_id = episode["start_observation_id"]
            transition = episode["transitions"][start_id][action]
            require(transition["action_executed"], f"{episode_id}: frozen action unavailable")
            observation_id = transition["to_observation_id"]
            observation = source["observations"][observation_id]
            headings = observation["viewport_headings_degrees"]
            require(len(headings) == 1, f"{episode_id}: high-res canary requires one viewport")
            frame_id = int(observation["frame_id"])
            member_name = protocol["high_resolution_archive"]["member_pattern"].format(frame_id=frame_id)
            info = archive_zip.getinfo(member_name)
            encoded = archive_zip.read(info)
            panorama = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(panorama is not None, f"{episode_id}: panorama decode failed")
            require(list(panorama.shape) == protocol["render_contract"]["panorama_shape_hwc"],
                    f"{episode_id}: panorama shape mismatch: {panorama.shape}")
            viewport, render_receipt = render_viewport(
                panorama,
                float(observation["camera_pose"]["panorama_angle_degrees"]),
                float(headings[0]),
                float(observation["horizontal_fov_degrees"]),
            )
            render_receipt.update({
                "archive_member": member_name,
                "archive_member_crc32": f"{info.CRC:08x}",
                "archive_member_bytes": info.file_size,
                "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
            })

            # Runtime boundary: only public mission and pixels enter infer_binding.
            runtime_output = infer_binding(
                viewport,
                episode["mission"],
                ocr_engine,
                portal_model,
                postprocessor,
                protocol["pixel_contract"],
            )

            # Evaluator boundary begins only after runtime_output is complete.
            observation_truth = episode_truth["observations"][observation_id]
            target_box = annotation_to_viewport(
                episode_truth["target_door_annotation"],
                render_receipt,
                float(protocol["render_contract"]["annotation_canvas_width"]),
                float(protocol["render_contract"]["annotation_canvas_height"]),
            )
            selected = runtime_output["selected_binding"]
            if selected is None:
                outcome = "UNKNOWN"
                selected_center_in_target = False
                selected_iou = 0.0
            else:
                portal_box = selected["portal"]["box_xyxy"]
                selected_center_in_target = center_in_box(portal_box, target_box)
                selected_iou = box_iou(portal_box, target_box)
                outcome = "CORRECT_TARGET_DOOR" if selected_center_in_target else "WRONG_DOOR"
            evaluation = {
                "truth_binding_state": observation_truth["binding_state"],
                "target_house_number_visible": bool(observation_truth["target_house_number_visible"]),
                "exact_target_text": bool(runtime_output["exact_target_text_candidates"]),
                "target_door_box_xyxy": [round(value, 6) for value in target_box],
                "selected_portal_center_in_target_box": selected_center_in_target,
                "selected_portal_iou_with_target_box": round(selected_iou, 8),
                "outcome": outcome,
            }
            rows.append({
                "episode_id": episode_id,
                "scenario_class": scenario,
                "frozen_action": action,
                "observation_id": observation_id,
                "mission": episode["mission"],
                "render_receipt": render_receipt,
                "runtime_output": runtime_output,
                "evaluation": evaluation,
            })
            print(json.dumps({
                "episode": episode_id,
                "progress": f"{sequence}/{len(source['episodes'])}",
                "action": action,
                "state": runtime_output["state"],
                "outcome": outcome,
            }, ensure_ascii=False), flush=True)

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent model device fallback: {actual_device}")
    metrics, gates = summarize(rows, protocol["gate"])
    if all(gates.values()):
        decision = "L10_SEVN_HIGHRES_PIXEL_ADDRESS_DOOR_CANARY_GATE_MET"
    elif metrics["correct_target_door_bindings"] >= int(protocol["signal_floor"]["minimum_correct_bindings"]):
        decision = "L10_SEVN_HIGHRES_PIXEL_ADDRESS_DOOR_SIGNAL_ONLY_GATE_NOT_MET"
    else:
        decision = "L10_SEVN_HIGHRES_PIXEL_ADDRESS_DOOR_CANARY_GATE_NOT_MET"
    result = {
        "schema": "blindassist-l10-sevn-highres-pixel-replay-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "CURATED_SEVN_HIGH_RESOLUTION_PIXEL_ASSOCIATION_DEVELOPMENT_CANARY",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "high_resolution_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": observed_archive_md5,
            },
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "device": actual_device,
            "device_name": torch.cuda.get_device_name(int(protocol["pixel_contract"]["portal_model"]["cuda_device"])),
        },
        "action_boundary": protocol["action_control"]["claim_boundary"],
        "metrics": metrics,
        "scenario_metrics": scenario_metrics(rows),
        "gates": gates,
        "gate_met": all(gates.values()),
        "wall_seconds": round(time.perf_counter() - started, 4),
        "episode_results": rows,
        "retention": {
            "panorama_or_viewport_files_written": 0,
            "reconstructible_intermediate_images_retained": 0,
            "durable_result": str(output_path),
            "source_archive_retained": str(archive_path),
        },
        "non_claims": protocol["non_claims"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--highres-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "decision": result["decision"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "wall_seconds": result["wall_seconds"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
