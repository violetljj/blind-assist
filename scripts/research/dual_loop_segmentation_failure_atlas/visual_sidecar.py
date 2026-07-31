"""Render an offline Development-only visual sidecar from frozen Atlas inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    connected_components,
)

from .atlas import (
    causal_temporal_probe,
    decode_packed_mask,
    read_json,
    read_jsonl,
    sha256_file,
    spatial_probe_mask,
)


PROTOCOL_ID = "DUAL_LOOP_VISUAL_ONLY_SIDECAR_R0"
SCHEMA_VERSION = "blindassist.dual_loop_visual_only_sidecar.result.v1"


class SidecarInputError(ValueError):
    """Raised when a visual-only input violates the frozen contract."""


class TFLiteHeatmapRunner:
    """Minimal fixed-contract INT8 segmentation runner used only for display."""

    def __init__(self, model_path: Path, *, threads: int, class_count: int) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("TensorFlow is required for sidecar heatmaps") from exc
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=threads,
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise SidecarInputError("segmentation model must expose one input and output")
        self.input_detail = inputs[0]
        self.output_detail = outputs[0]
        if tuple(int(value) for value in self.input_detail["shape"]) != (1, 256, 256, 3):
            raise SidecarInputError("sidecar requires NHWC 1x256x256x3 input")
        if tuple(int(value) for value in self.output_detail["shape"]) != (
            1,
            256,
            256,
            class_count,
        ):
            raise SidecarInputError("sidecar output tensor shape mismatch")
        if np.dtype(self.input_detail["dtype"]) != np.dtype(np.int8):
            raise SidecarInputError("sidecar model input must be int8")
        if np.dtype(self.output_detail["dtype"]) != np.dtype(np.int8):
            raise SidecarInputError("sidecar model output must be int8")
        self.input_scale, self.input_zero = self._quantization(
            self.input_detail,
            "input",
        )
        self.output_scale, self.output_zero = self._quantization(
            self.output_detail,
            "output",
        )

    @staticmethod
    def _quantization(detail: dict[str, Any], label: str) -> tuple[float, int]:
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if not np.isfinite(scale) or float(scale) <= 0:
            raise SidecarInputError(f"{label} quantization scale must be positive")
        return float(scale), int(zero_point)

    def infer(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, float]:
        resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32)
        tensor = np.clip(
            np.rint(rgb / self.input_scale + self.input_zero),
            -128,
            127,
        ).astype(np.int8)[None, ...]
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        started = time.perf_counter()
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - started) * 1000.0
        raw = self.interpreter.get_tensor(self.output_detail["index"])
        scores = (raw.astype(np.float32) - self.output_zero) * self.output_scale
        scores = scores[0]
        maximum = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(np.clip(scores - maximum, -80.0, 80.0))
        probabilities = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return (
            np.argmax(probabilities, axis=-1).astype(np.uint8),
            np.max(probabilities, axis=-1).astype(np.float32),
            float(inference_ms),
        )


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _verify_output(repo_root: Path, output_root: Path) -> None:
    allowed = (repo_root / "artifacts.local").resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise SidecarInputError("output-root must stay under artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite sidecar output: {output_root}")


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"frame-{digest}.png"


def _tint(
    source: Image.Image,
    layers: Sequence[tuple[np.ndarray, tuple[int, int, int], float]],
) -> Image.Image:
    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    for mask, color, alpha in layers:
        selected = np.asarray(mask, dtype=bool)
        array[selected] = (
            array[selected] * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha
        )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _heatmap(source: Image.Image, ids: np.ndarray, confidence: np.ndarray) -> Image.Image:
    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    hazard = np.isin(ids, np.asarray([1, 2], dtype=np.uint8))
    alpha = np.clip(confidence * 0.72, 0.0, 0.72)
    colors = np.zeros_like(array)
    colors[..., 0] = 255.0
    colors[..., 1] = np.where(ids == 1, 180.0, 55.0)
    colors[..., 2] = 20.0
    array[hazard] = (
        array[hazard] * (1.0 - alpha[hazard, None])
        + colors[hazard] * alpha[hazard, None]
    )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _draw_yolo_boxes(
    source: Image.Image,
    detections: Sequence[dict[str, Any]],
) -> Image.Image:
    panel = source.copy()
    draw = ImageDraw.Draw(panel)
    for detection in detections:
        width = float(detection["frame_width"])
        height = float(detection["frame_height"])
        box = (
            float(detection["left"]) * 256.0 / width,
            float(detection["top"]) * 256.0 / height,
            float(detection["right"]) * 256.0 / width,
            float(detection["bottom"]) * 256.0 / height,
        )
        draw.rectangle(box, outline=(0, 225, 255), width=2)
        label = str(detection.get("label", "object"))
        confidence = float(detection.get("confidence", 0.0))
        draw.text((box[0] + 2, box[1] + 2), f"{label} {confidence:.2f}", fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    return panel


def _confidence_gate(
    *,
    frame_row: dict[str, Any],
    component_by_id: dict[str, dict[str, Any]],
    threshold: float,
) -> np.ndarray:
    shape = tuple(int(value) for value in frame_row["packed_masks"]["shape"])
    result = np.zeros(shape, dtype=bool)
    for class_name in ("boundary_step_curb", "obstacle"):
        mask = decode_packed_mask(
            frame_row["packed_masks"][f"candidate_{class_name}"],
            shape,
        )
        for component in connected_components(mask, connectivity=8):
            component_id = (
                f"{frame_row['source_id']}:{int(frame_row['frame_id'])}:"
                f"{class_name}:{component.index}"
            )
            ledger = component_by_id.get(component_id)
            if ledger is None:
                raise SidecarInputError(f"missing component ledger row: {component_id}")
            median = ledger.get("top1_confidence_median")
            if median is not None and float(median) >= threshold:
                result |= component.mask
    return result


def _probe_masks(
    *,
    config: dict[str, Any],
    probe_id: str,
    frame_rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    component_by_id: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    if probe_id not in config["supported_probe_ids"]:
        raise SidecarInputError(f"unsupported probe_id: {probe_id}")
    shape = tuple(int(value) for value in config["analysis_shape"])
    baseline = {
        row["view_row_id"]: decode_packed_mask(row["packed_masks"]["B"], shape)
        for row in frame_rows
    }
    if probe_id.startswith("SPATIAL:"):
        name = probe_id.split(":", 1)[1]
        spatial = spatial_probe_mask(shape, name, config["spatial_bands"])
        return {view_id: mask & spatial for view_id, mask in baseline.items()}
    if probe_id.startswith("CONFIDENCE:"):
        threshold = float(config["high_confidence_minimum"])
        return {
            row["view_row_id"]: _confidence_gate(
                frame_row=row,
                component_by_id=component_by_id,
                threshold=threshold,
            )
            for row in frame_rows
        }
    name = probe_id.split(":", 1)[1]
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        manifest = manifest_by_id[row["view_row_id"]]
        by_sequence[str(manifest["sequence_id"])].append(row)
    result: dict[str, np.ndarray] = {}
    for rows in by_sequence.values():
        rows.sort(
            key=lambda row: (
                int(manifest_by_id[row["view_row_id"]]["source_capture_timestamp_ns"]),
                int(row["frame_id"]),
            )
        )
        for index, row in enumerate(rows):
            current = baseline[row["view_row_id"]]
            previous = baseline[rows[index - 1]["view_row_id"]] if index >= 1 else None
            previous_previous = (
                baseline[rows[index - 2]["view_row_id"]] if index >= 2 else None
            )
            result[row["view_row_id"]] = causal_temporal_probe(
                current,
                previous,
                previous_previous,
                name,
            )
    return result


def _reason_for_probe(probe_id: str) -> str:
    reasons = {
        "SPATIAL:LOWER_FIELD": "outside lower-field development probe",
        "SPATIAL:CENTRAL_BODY_CORRIDOR": "outside central-corridor development probe",
        "SPATIAL:UPPER_HEAD_BAND": "outside upper-band development probe",
        "TEMPORAL:CAUSAL_2_OF_3": "not present in current plus either prior observation",
        "TEMPORAL:CAUSAL_3_CONSECUTIVE": "not present in three consecutive observations",
        "CONFIDENCE:COMPONENT_MEDIAN_CONFIDENCE_GE_0_65": "component median confidence below 0.65",
    }
    return reasons[probe_id]


def _render_frame(
    *,
    config: dict[str, Any],
    source: Image.Image,
    ids: np.ndarray,
    confidence: np.ndarray,
    detections: Sequence[dict[str, Any]],
    candidate: np.ndarray,
    gate_passed: np.ndarray,
    detector_mask: np.ndarray,
    probe_id: str,
    frame_row: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    source = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    raw_hazard = np.isin(ids, np.asarray([1, 2], dtype=np.uint8))
    rejected = candidate & ~gate_passed
    abstained = raw_hazard & detector_mask
    panels: list[tuple[str, Image.Image]] = [
        ("YOLO known-object boxes", _draw_yolo_boxes(source, detections)),
        ("raw segmentation heatmap", _heatmap(source, ids, confidence)),
        ("visual candidates", _tint(source, [(candidate, (170, 70, 255), 0.62)])),
        ("gate-passed candidates", _tint(source, [(gate_passed, (0, 220, 210), 0.65)])),
        (
            "rejected / abstained",
            _tint(
                source,
                [
                    (rejected, (255, 155, 0), 0.68),
                    (abstained, (145, 145, 145), 0.72),
                ],
            ),
        ),
    ]
    canvas = Image.new("RGB", (768, 620), (15, 15, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), config["watermarks"][0], fill=(255, 220, 60))
    draw.text((10, 24), config["watermarks"][1], fill=(255, 110, 110))
    draw.text(
        (10, 42),
        f"probe={probe_id}  source={frame_row['source_id']}  frame={frame_row['frame_id']}",
        fill=(235, 235, 235),
    )
    for index, (title, panel) in enumerate(panels):
        x = (index % 3) * 256
        y = 66 + (index // 3) * 256
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + 255, y + 18), fill=(0, 0, 0))
        draw.text((x + 4, y + 3), title, fill=(255, 255, 255))
    reason_x, reason_y = 512, 322
    draw.rectangle((reason_x, reason_y, 767, 577), fill=(28, 28, 34))
    reason_lines = [
        "REJECTION / ABSTENTION REASONS",
        "",
        f"passed pixels: {int(np.count_nonzero(gate_passed))}",
        f"rejected pixels: {int(np.count_nonzero(rejected))}",
        f"abstained pixels: {int(np.count_nonzero(abstained))}",
        "",
        "rejected:",
        _reason_for_probe(probe_id),
        "",
        "abstained:",
        "YOLO overlap; attribution uncertain",
        "",
        "VISUAL/CANDIDATE AUTHORITY ONLY",
    ]
    for index, line in enumerate(reason_lines):
        draw.text(
            (reason_x + 8, reason_y + 8 + index * 17),
            line,
            fill=(235, 235, 235) if index else (255, 220, 60),
        )
    draw.text((10, 594), config["watermarks"][0], fill=(255, 220, 60))
    draw.text((560, 594), config["watermarks"][1], fill=(255, 110, 110))
    metrics = {
        "candidate_pixels": int(np.count_nonzero(candidate)),
        "gate_passed_pixels": int(np.count_nonzero(gate_passed)),
        "rejected_pixels": int(np.count_nonzero(rejected)),
        "abstained_pixels": int(np.count_nonzero(abstained)),
        "rejection_reason": _reason_for_probe(probe_id),
        "abstention_reason": "YOLO overlap; attribution uncertain",
    }
    return canvas, metrics


def run(
    *,
    repo_root: Path,
    config_path: Path,
    frames_path: Path,
    components_path: Path,
    manifest_path: Path,
    yolo_trace_path: Path,
    model_path: Path,
    output_root: Path,
    probe_id: str,
    view_row_ids: Sequence[str],
    threads: int,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    _verify_output(repo_root, output_root)
    config = read_json(config_path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise SidecarInputError("unexpected sidecar protocol_id")
    if config.get("stage") != "DEVELOPMENT_STANDARD":
        raise SidecarInputError("sidecar is Development-only")
    if config.get("drives_alerts") is not False:
        raise SidecarInputError("sidecar must declare drives_alerts=false")
    visible_text = " ".join(
        [
            *config["watermarks"],
            *[_reason_for_probe(value) for value in config["supported_probe_ids"]],
        ]
    ).lower()
    for forbidden in config["forbidden_visible_claims"]:
        if forbidden.lower() in visible_text:
            raise SidecarInputError(f"forbidden visible claim configured: {forbidden}")
    if sha256_file(model_path) != config["model"]["sha256"]:
        raise SidecarInputError("segmentation model hash mismatch")
    selected_ids = list(view_row_ids)
    if not selected_ids or len(selected_ids) > int(config["maximum_rendered_frames"]):
        raise SidecarInputError("view-row selection must contain 1..maximum frames")
    if len(selected_ids) != len(set(selected_ids)):
        raise SidecarInputError("duplicate view-row selection")

    frame_rows = read_jsonl(frames_path)
    component_rows = read_jsonl(components_path)
    manifest_rows = read_jsonl(manifest_path)
    trace_rows = read_jsonl(yolo_trace_path)
    frame_by_id = {str(row["view_row_id"]): row for row in frame_rows}
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    component_by_id = {str(row["component_id"]): row for row in component_rows}
    trace_by_key = {
        (str(row["source_id"]), int(row["frame_id"])): row for row in trace_rows
    }
    if len(frame_by_id) != len(frame_rows):
        raise SidecarInputError("duplicate frame view_row_id")
    if len(component_by_id) != len(component_rows):
        raise SidecarInputError("duplicate component_id")
    if len(trace_by_key) != len(trace_rows):
        raise SidecarInputError("duplicate YOLO source/frame key")
    missing = sorted(set(selected_ids) - set(frame_by_id))
    if missing:
        raise SidecarInputError(f"selected view rows missing from frame input: {missing}")
    probe_masks = _probe_masks(
        config=config,
        probe_id=probe_id,
        frame_rows=frame_rows,
        manifest_by_id=manifest_by_id,
        component_by_id=component_by_id,
    )
    runner = TFLiteHeatmapRunner(
        model_path,
        threads=threads,
        class_count=len(config["model"]["classes"]),
    )

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    rendered: list[dict[str, Any]] = []
    try:
        for view_row_id in selected_ids:
            frame_row = frame_by_id[view_row_id]
            manifest = manifest_by_id.get(view_row_id)
            if manifest is None:
                raise SidecarInputError(f"missing canonical manifest row: {view_row_id}")
            for field in ("source_id", "session_id", "frame_id", "image_sha256"):
                if frame_row[field] != manifest[field]:
                    raise SidecarInputError(f"frame/manifest {field} mismatch")
            trace = trace_by_key.get(
                (str(frame_row["source_id"]), int(frame_row["frame_id"]))
            )
            if trace is None or trace["image_sha256"] != frame_row["image_sha256"]:
                raise SidecarInputError(f"missing or mismatched YOLO trace: {view_row_id}")
            image_path = repo_root / manifest["image_repo_relative_path"]
            if sha256_file(image_path) != frame_row["image_sha256"]:
                raise SidecarInputError(f"source image hash mismatch: {view_row_id}")
            with Image.open(image_path) as handle:
                source = handle.convert("RGB")
            ids, confidence, inference_ms = runner.infer(source)
            shape = tuple(int(value) for value in frame_row["packed_masks"]["shape"])
            candidate = decode_packed_mask(frame_row["packed_masks"]["B"], shape)
            detector_mask = decode_packed_mask(frame_row["packed_masks"]["A"], shape)
            canvas, metrics = _render_frame(
                config=config,
                source=source,
                ids=ids,
                confidence=confidence,
                detections=trace["detections"],
                candidate=candidate,
                gate_passed=probe_masks[view_row_id],
                detector_mask=detector_mask,
                probe_id=probe_id,
                frame_row=frame_row,
            )
            filename = _safe_name(view_row_id)
            canvas.save(temporary / filename, format="PNG", optimize=True)
            rendered.append(
                {
                    "view_row_id": view_row_id,
                    "source_id": frame_row["source_id"],
                    "session_id": frame_row["session_id"],
                    "role": frame_row["rehearsal_role"],
                    "frame_id": int(frame_row["frame_id"]),
                    "scene_bucket": manifest["scene_bucket"],
                    "probe_id": probe_id,
                    "figure_path": filename,
                    "yolo_detection_count": len(trace["detections"]),
                    "heatmap_inference_ms_observation_only": inference_ms,
                    **metrics,
                }
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "stage": config["stage"],
            "authority": config["authority"],
            "drives_alerts": False,
            "watermarks": config["watermarks"],
            "probe_id": probe_id,
            "rendered_frame_count": len(rendered),
            "frames": rendered,
            "provenance": {
                "config": {
                    "path": str(config_path.relative_to(repo_root)),
                    "sha256": sha256_file(config_path),
                },
                "frames": {
                    "path": str(frames_path.relative_to(repo_root)),
                    "sha256": sha256_file(frames_path),
                },
                "components": {
                    "path": str(components_path.relative_to(repo_root)),
                    "sha256": sha256_file(components_path),
                },
                "canonical_manifest": {
                    "path": str(manifest_path.relative_to(repo_root)),
                    "sha256": sha256_file(manifest_path),
                },
                "yolo_trace": {
                    "path": str(yolo_trace_path.relative_to(repo_root)),
                    "sha256": sha256_file(yolo_trace_path),
                },
                "model": {
                    "path": str(model_path.relative_to(repo_root)),
                    "sha256": sha256_file(model_path),
                },
                "implementation": {
                    "path": str(Path(__file__).resolve().relative_to(repo_root)),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
        }
        with (temporary / "result.json").open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output_root)
        return result
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--yolo-trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--view-row-id", action="append", required=True)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    result = run(
        repo_root=repo_root,
        config_path=_resolve(repo_root, args.config),
        frames_path=_resolve(repo_root, args.frames),
        components_path=_resolve(repo_root, args.components),
        manifest_path=_resolve(repo_root, args.manifest),
        yolo_trace_path=_resolve(repo_root, args.yolo_trace),
        model_path=_resolve(repo_root, args.model),
        output_root=_resolve(repo_root, args.output_root),
        probe_id=args.probe_id,
        view_row_ids=args.view_row_id,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": "VISUAL_ONLY_SIDECAR_RENDERED",
                "frames": result["rendered_frame_count"],
                "drives_alerts": result["drives_alerts"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
