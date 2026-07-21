#!/usr/bin/env python3
"""Prepare hash-bound route-overlay frames and prompts for Codex visual review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from .contract import (
        BUNDLE_SCHEMA, CONTRACT_ID, REVIEW_SCHEMA, canonical_sha256, load_json, sha256_file,
        validate_geometry, validate_source_receipt, write_json,
    )
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import (
        BUNDLE_SCHEMA, CONTRACT_ID, REVIEW_SCHEMA, canonical_sha256, load_json, sha256_file,
        validate_geometry, validate_source_receipt, write_json,
    )


def decode_frame(capture: cv2.VideoCapture, timestamp_ms: int) -> tuple[np.ndarray, float]:
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"cannot decode requested timestamp {timestamp_ms}ms")
    decoded_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    return frame, decoded_timestamp_ms


def project_source_view(frame: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    mode = projection.get("mode", "rectilinear_identity_v1")
    if mode == "rectilinear_identity_v1":
        return frame
    if mode != "equirectangular_to_rectilinear_v1":
        raise ValueError(f"unsupported source projection mode: {mode}")
    output_width = int(projection["output_width"])
    output_height = int(projection["output_height"])
    horizontal_fov_deg = float(projection["horizontal_fov_deg"])
    yaw_rad = np.deg2rad(float(projection["yaw_deg"]))
    pitch_rad = np.deg2rad(float(projection.get("pitch_deg", 0.0)))
    if not 320 <= output_width <= 3840 or not 240 <= output_height <= 2160:
        raise ValueError("source projection output dimensions are outside the bounded range")
    if not 30.0 <= horizontal_fov_deg <= 150.0:
        raise ValueError("source projection horizontal FOV is outside the bounded range")
    focal = output_width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    x = (np.arange(output_width, dtype=np.float32) - (output_width - 1) / 2.0) / focal
    y = ((output_height - 1) / 2.0 - np.arange(output_height, dtype=np.float32)) / focal
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = np.ones_like(grid_x)
    norm = np.sqrt(grid_x * grid_x + grid_y * grid_y + grid_z * grid_z)
    grid_x, grid_y, grid_z = grid_x / norm, grid_y / norm, grid_z / norm
    pitched_y = grid_y * np.cos(pitch_rad) - grid_z * np.sin(pitch_rad)
    pitched_z = grid_y * np.sin(pitch_rad) + grid_z * np.cos(pitch_rad)
    longitude = np.arctan2(grid_x, pitched_z) + yaw_rad
    latitude = np.arcsin(np.clip(pitched_y, -1.0, 1.0))
    source_height, source_width = frame.shape[:2]
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * source_width).astype(np.float32)
    map_y = ((0.5 - latitude / np.pi) * source_height).astype(np.float32)
    map_x = np.mod(map_x, float(source_width))
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def render_overlay(frame: np.ndarray, *, frame_id: str, relative_ms: int, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    points = np.asarray([[round(float(x) * (width - 1)), round(float(y) * (height - 1))] for x, y in polygon], dtype=np.int32)
    layer = output.copy()
    cv2.fillPoly(layer, [points], (40, 210, 40))
    output = cv2.addWeighted(layer, 0.22, output, 0.78, 0.0)
    cv2.polylines(output, [points], True, (40, 255, 40), max(2, round(width / 480)), cv2.LINE_AA)
    label = f"{frame_id}  t={relative_ms / 1000.0:+.2f}s"
    cv2.rectangle(output, (0, 0), (min(width - 1, 430), 42), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def make_contact_sheet(images: Sequence[np.ndarray], *, columns: int = 4, cell_width: int = 420) -> np.ndarray:
    if not images:
        raise ValueError("contact sheet needs images")
    cells: list[np.ndarray] = []
    for image in images:
        scale = cell_width / image.shape[1]
        cells.append(cv2.resize(image, (cell_width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    cell_height = max(cell.shape[0] for cell in cells)
    rows = (len(cells) + columns - 1) // columns
    canvas = np.full((rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        y = row * cell_height
        x = column * cell_width
        canvas[y:y + cell.shape[0], x:x + cell.shape[1]] = cell
    return canvas


def output_schema() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "required": [
            "schema", "contract_id", "review_id", "role", "round", "teacher_id", "reviewer_surface",
            "reproducibility_class", "bundle_manifest_sha256", "input_inventory_sha256", "prompt_sha256",
            "output_schema_sha256", "route_valid", "overall_risk", "events", "abstain_reasons",
            "human_event_truth_present", "metric_geometry_present", "training_authorized", "u0_authority_granted",
            "android_runtime_change_authorized", "production_model_replacement_authorized",
        ],
        "enums": {
            "role": ["full_context_teacher", "causal_codex_baseline"],
            "route_valid": ["yes", "no", "uncertain"],
            "overall_risk": ["none", "caution", "critical", "unknown"],
            "category": ["static_obstacle", "dropoff", "overhead", "vehicle", "person", "animal", "surface", "other"],
            "route_relation": ["inside", "entering", "adjacent", "outside", "uncertain"],
            "distance_band": ["0-2m", "2-5m", "over-5m", "unknown"],
            "ttc_band": ["0-1.5s", "1.5-3s", "over-3s", "unknown"],
            "required_action": ["none", "slow", "detour", "stop", "step_over", "uncertain"],
            "confidence": ["low", "medium", "high"],
        },
    }


def prompt_text(role: str, geometry: dict[str, Any], frame_ids: Sequence[str]) -> str:
    context = (
        "完整上下文教师可以查看本清单中的全部当前/后续帧，仅用于生成 provisional silver reference。"
        if role == "full_context_teacher" else
        "因果基线只能查看本清单列出的当前及过去帧；不得使用教师清单、隐藏未来、深度或标签。"
    )
    return f"""# Codex USTRF 跨设备风险评审

你是离线步行风险评测器，不是安全控制器，也不是真值标注员。{context}

绿色多边形是预注册的三秒测试路线。只依据可见像素判断：障碍是否占据路线、动态物体是否进入路线、是否需要减速/绕行/停止/跨越。看不清必须输出 unknown；不要补全不可见区域，不要只因场景类别看起来危险就提醒。

近似条件仅用于同口径比较：camera_height={geometry['camera_height_m']}m，HFOV={geometry['horizontal_fov_deg']}deg，VFOV={geometry['vertical_fov_deg']}deg，walking_speed={geometry['walking_speed_mps']}m/s，route_width={geometry['route_width_m']}m，horizon={geometry['risk_horizon_s']}s。它们不是设备米制真值。

允许引用的 frame_id：{', '.join(frame_ids)}。
严格填写 review_template.json；不要改变 schema/authority 字段，不要引用清单以外的帧。
"""


def review_template(role: str, round_id: int, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "contract_id": CONTRACT_ID,
        "review_id": f"replace-{role}-round-{round_id}",
        "role": role,
        "round": round_id,
        "teacher_id": "codex_visual_teacher_provisional_v1",
        "reviewer_surface": "codex_interactive_visual_model",
        "reproducibility_class": "surface_snapshot_not_weight_pinned",
        "bundle_manifest_sha256": "REPLACE_AFTER_MANIFEST_FINALIZED",
        "input_inventory_sha256": artifact["input_inventory_sha256"],
        "prompt_sha256": artifact["prompt_sha256"],
        "output_schema_sha256": artifact["output_schema_sha256"],
        "route_valid": "uncertain",
        "overall_risk": "unknown",
        "events": [],
        "abstain_reasons": ["other"],
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    source = dict(validate_source_receipt(load_json(args.source_receipt), video_path=args.video.resolve()))
    config = load_json(args.config)
    if config.get("schema") != "blindassist_ustrf_crosscam_codex_config_v1" or config.get("contract_id") != CONTRACT_ID:
        raise ValueError("cross-camera config schema/contract mismatch")
    geometry = dict(validate_geometry(config.get("assumed_geometry")))
    projection = config.get("source_projection", {"mode": "rectilinear_identity_v1"})
    if not isinstance(projection, dict):
        raise ValueError("source_projection must be an object")
    if projection.get("mode") == "equirectangular_to_rectilinear_v1" and not isinstance(
        projection.get("forward_axis_authority"), str
    ):
        raise ValueError("equirectangular projection must disclose forward_axis_authority")
    if args.start_ms < 0 or args.duration_ms < 1000:
        raise ValueError("review window must be at least one second and start non-negative")
    args.output_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration_ms = round(source_frames / source_fps * 1000) if source_fps > 0 else 0
    if source_fps <= 0 or source_frames <= 0 or args.start_ms + args.duration_ms > source_duration_ms + 50:
        capture.release()
        raise ValueError("review window exceeds valid video timing")
    all_rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir()
    try:
        for relative_ms in range(0, args.duration_ms, int(config["teacher_interval_ms"])):
            source_ms = args.start_ms + relative_ms
            frame_id = f"f{relative_ms:06d}"
            decoded, decoded_timestamp_ms = decode_frame(capture, source_ms)
            seek_error_ms = decoded_timestamp_ms - source_ms
            if abs(seek_error_ms) > float(config["max_abs_seek_error_ms"]):
                raise ValueError(
                    f"decoded timestamp drift {seek_error_ms:.3f}ms exceeds limit at {source_ms}ms"
                )
            projected = project_source_view(decoded, projection)
            overlay = render_overlay(
                projected, frame_id=frame_id, relative_ms=relative_ms,
                polygon=geometry["route_polygon_xy_norm"],
            )
            destination = frames_dir / f"{frame_id}.png"
            if not cv2.imwrite(str(destination), overlay):
                raise ValueError(f"cannot write frame: {destination}")
            row = {
                "frame_id": frame_id, "relative_ms": relative_ms, "source_timestamp_ms": source_ms,
                "decoded_timestamp_ms": decoded_timestamp_ms, "seek_error_ms": seek_error_ms,
                "path": str(destination.resolve()), "sha256": sha256_file(destination),
            }
            all_rows.append(row)
            images[frame_id] = overlay
    finally:
        capture.release()
    causal_rows = [row for row in all_rows if row["relative_ms"] % int(config["causal_interval_ms"]) == 0]
    schema = output_schema()
    schema_path = args.output_dir / "teacher_output_schema.json"
    write_json(schema_path, schema)
    artifacts: dict[str, Any] = {}
    for role, rows in (("full_context_teacher", all_rows), ("causal_codex_baseline", causal_rows)):
        role_dir = args.output_dir / role
        role_dir.mkdir()
        sheet_rows: list[dict[str, Any]] = []
        page_size = int(config["contact_sheet_page_size"])
        for page_index in range(0, len(rows), page_size):
            batch = rows[page_index:page_index + page_size]
            sheet = make_contact_sheet([images[row["frame_id"]] for row in batch], columns=int(config["contact_sheet_columns"]))
            sheet_path = role_dir / f"contact_sheet_{page_index // page_size:02d}.jpg"
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise ValueError(f"cannot write contact sheet: {sheet_path}")
            sheet_rows.append({"path": str(sheet_path.resolve()), "sha256": sha256_file(sheet_path), "frame_ids": [row["frame_id"] for row in batch]})
        prompt = prompt_text(role, geometry, [row["frame_id"] for row in rows])
        prompt_path = role_dir / "teacher_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        artifact = {
            "role": role,
            "frames": rows,
            "input_inventory_sha256": canonical_sha256(rows),
            "contact_sheets": sheet_rows,
            "prompt_path": str(prompt_path.resolve()),
            "prompt_sha256": sha256_file(prompt_path),
            "output_schema_path": str(schema_path.resolve()),
            "output_schema_sha256": sha256_file(schema_path),
        }
        artifacts[role] = artifact
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_receipt": str(args.source_receipt.resolve()),
        "source_receipt_sha256": sha256_file(args.source_receipt),
        "source": source,
        "video_path": str(args.video.resolve()),
        "video_sha256": sha256_file(args.video),
        "source_video": {"fps": source_fps, "frame_count": source_frames, "duration_ms": source_duration_ms},
        "window": {"start_ms": args.start_ms, "duration_ms": args.duration_ms},
        "assumed_geometry": geometry,
        "source_projection": projection,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "review_artifacts": artifacts,
        "input_excludes": ["event_labels", "human_reviews", "adjudication", "blind", "future_geometry_truth", "depth_truth"],
        "authority": "provisional_proxy_benchmark_only",
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    manifest_path = args.output_dir / "bundle_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    (args.output_dir / "bundle_manifest.json.sha256").write_text(manifest_sha + "\n", encoding="ascii")
    templates_dir = args.output_dir / "review_templates"
    templates_dir.mkdir()
    for role, artifact in artifacts.items():
        for round_id in range(1, 4):
            template = review_template(role, round_id, artifact)
            template["bundle_manifest_sha256"] = manifest_sha
            write_json(templates_dir / f"{role}_round_{round_id}.json", template)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--duration-ms", type=int, default=4000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = run(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "source_id": manifest["source"]["source_id"],
        "teacher_frames": len(manifest["review_artifacts"]["full_context_teacher"]["frames"]),
        "causal_frames": len(manifest["review_artifacts"]["causal_codex_baseline"]["frames"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
