"""RapidOCR transfer harness for the frozen SAGE-R V2 algorithm.

The harness renders mechanism-targeted, door-sign-style pixel scenes, runs the
real RapidOCR detector/recognizer, and adapts its polygons and confidences to
the unchanged V2 OCR-stage interface.  The images are generated Development
inputs, not photographs or camera measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import inspect
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .semantic_anchor_graph_and_belief_v2 import (
    Box,
    Candidate,
    Frame,
    ReferentBelief,
    SubstringFsmBaseline,
    TargetGraph,
    _metrics,
    _row_correct,
    graph_candidate_scores,
    normalize_text,
)


EXPERIMENT_ID = "SEMANTIC_ANCHOR_GRAPH_AND_BELIEF_V2_1_REAL_OCR_TRANSFER"
SCHEMA_VERSION = "blindassist_semantic_anchor_graph_and_belief_v2_1_real_ocr_transfer"
CLAIM_CEILING = (
    "GENERATED_PIXEL_SCENE_RAPIDOCR_DEVELOPMENT_TRANSFER_"
    "NO_NATURAL_PHOTO_CAMERA_OPEN_WORLD_CALIBRATION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
TARGET = TargetGraph(("ROOM", "302"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SceneSpec:
    episode_id: str
    frame_index: int
    viewpoint: str
    signs: tuple[tuple[str, str, str], ...]
    directory_lines: tuple[str, ...]
    truth: str
    expected_state: str
    note: str
    blur_sigma: float = 0.0
    perspective: float = 0.0
    occlusion: str | None = None


class RapidOcrPolygonAdapter:
    """Load the existing isolated RapidOCR runtime and retain raw polygons."""

    def __init__(self, runtime_root: Path):
        site_packages = runtime_root / "site-packages"
        if not site_packages.is_dir():
            raise FileNotFoundError(f"missing RapidOCR runtime: {site_packages}")
        sys.path.insert(0, str(site_packages.resolve()))
        from rapidocr import RapidOCR

        self.runtime_root = runtime_root.resolve()
        self.engine = RapidOCR(
            params={
                "Global.model_root_dir": str((runtime_root / "models").resolve()),
                "Global.log_level": "error",
                "Global.text_score": 0.50,
            }
        )

    def read(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = self.engine(image)
        boxes = list(result.boxes) if result.boxes is not None else []
        texts = list(result.txts or ())
        scores = [float(value) for value in (result.scores or ())]
        return [
            {
                "text": text,
                "confidence": score,
                "polygon_px": [[float(point[0]), float(point[1])] for point in polygon],
            }
            for polygon, text, score in zip(boxes, texts, scores, strict=True)
        ]

    def receipt(self) -> dict[str, Any]:
        models = sorted((self.runtime_root / "models").glob("*.onnx"))
        return {
            "runtime_root": str(self.runtime_root),
            "rapidocr": importlib.metadata.version("rapidocr"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "opencv": cv2.__version__,
            "models": [
                {"name": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
                for path in models
            ],
        }


def _door_boxes(directory_layout: bool = False) -> dict[str, tuple[int, int, int, int]]:
    if directory_layout:
        return {
            "A": (350, 210, 620, 705),
            "B": (665, 210, 935, 705),
            "C": (980, 210, 1250, 705),
        }
    return {
        "A": (55, 210, 385, 705),
        "B": (475, 210, 805, 705),
        "C": (895, 210, 1225, 705),
    }


def _draw_centered_text(
    image: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    *,
    scale: float = 1.25,
    thickness: int = 3,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(
        image,
        text,
        (int(center_x - size[0] / 2), baseline_y),
        font,
        scale,
        (18, 18, 18),
        thickness,
        cv2.LINE_AA,
    )


def _render_scene(spec: SceneSpec) -> tuple[np.ndarray, tuple[Candidate, ...]]:
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (196, 202, 204), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (IMAGE_WIDTH - 1, 190), (224, 226, 222), -1)
    boxes = _door_boxes(bool(spec.directory_lines))
    colors = {"A": (119, 137, 151), "B": (132, 145, 157), "C": (112, 129, 144)}
    for candidate_id, (x0, y0, x1, y1) in boxes.items():
        cv2.rectangle(image, (x0, y0), (x1, y1), colors[candidate_id], -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (55, 61, 66), 5)
        cv2.circle(image, (x1 - 35, 470), 8, (25, 25, 25), -1)

    for candidate_id, prefix, number in spec.signs:
        x0, y0, x1, _ = boxes[candidate_id]
        sx0, sy0, sx1, sy1 = x0 + 38, y0 - 78, x1 - 38, y0 - 18
        cv2.rectangle(image, (sx0, sy0), (sx1, sy1), (247, 245, 232), -1)
        cv2.rectangle(image, (sx0, sy0), (sx1, sy1), (50, 50, 50), 2)
        _draw_centered_text(image, prefix, int((sx0 + sx1) / 2 - 55), sy1 - 15, scale=0.86, thickness=2)
        _draw_centered_text(image, number, int((sx0 + sx1) / 2 + 63), sy1 - 14, scale=1.06, thickness=2)

    if spec.directory_lines:
        board = (20, 260, 315, 445)
        cv2.rectangle(image, board[:2], board[2:], (236, 239, 231), -1)
        cv2.rectangle(image, board[:2], board[2:], (44, 48, 46), 4)
        cv2.putText(image, "DIRECTORY", (38, 302), cv2.FONT_HERSHEY_DUPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
        for index, line in enumerate(spec.directory_lines):
            cv2.putText(
                image,
                line,
                (46, 345 + 40 * index),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (22, 22, 22),
                2,
                cv2.LINE_AA,
            )

    if spec.occlusion:
        candidate_id, fraction_text = spec.occlusion.split(":", 1)
        fraction = float(fraction_text)
        x0, y0, x1, _ = boxes[candidate_id]
        sign_x0, sign_x1 = x0 + 38, x1 - 38
        cover_x0 = int(sign_x1 - (sign_x1 - sign_x0) * fraction)
        cv2.rectangle(image, (cover_x0, y0 - 82), (sign_x1 + 4, y0 - 14), (181, 184, 180), -1)

    transform = np.eye(3, dtype=np.float32)
    if spec.perspective > 0:
        skew = int(110 * spec.perspective)
        source = np.float32([[0, 0], [IMAGE_WIDTH - 1, 0], [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1], [0, IMAGE_HEIGHT - 1]])
        target = np.float32([[skew, 12], [IMAGE_WIDTH - 1 - skew, 0], [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1], [0, IMAGE_HEIGHT - 25]])
        transform = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT), borderValue=(205, 207, 203))

    if spec.blur_sigma > 0:
        kernel = max(3, int(round(spec.blur_sigma * 4)) | 1)
        image = cv2.GaussianBlur(image, (kernel, kernel), spec.blur_sigma)

    candidates = []
    for candidate_id, (x0, y0, x1, y1) in boxes.items():
        corners = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
        mapped = cv2.perspectiveTransform(corners, transform)[0]
        candidates.append(
            Candidate(
                candidate_id,
                Box(
                    float(mapped[:, 0].min() / IMAGE_WIDTH),
                    float(mapped[:, 1].min() / IMAGE_HEIGHT),
                    float(mapped[:, 0].max() / IMAGE_WIDTH),
                    float(mapped[:, 1].max() / IMAGE_HEIGHT),
                ),
            )
        )
    return image, tuple(candidates)


def build_cohort() -> list[SceneSpec]:
    """Fixed generated-pixel cohort covering the requested OCR mechanisms."""
    rows: list[SceneSpec] = []

    def add(episode: str, viewpoint: str, signs: Iterable[tuple[str, str, str]], truth: str, expected: str, note: str, **kwargs: Any) -> None:
        index = sum(item.episode_id == episode for item in rows)
        rows.append(SceneSpec(episode, index, viewpoint, tuple(signs), tuple(kwargs.pop("directory_lines", ())), truth, expected, note, **kwargs))

    adjacent = (("A", "ROOM", "301"), ("B", "ROOM", "302"), ("C", "ROOM", "320"))
    add("adjacent_301_302_320", "front-1", adjacent, "B", "TARGET", "adjacent same-prefix rooms")
    add("adjacent_301_302_320", "oblique-2", adjacent, "B", "TARGET", "fresh polygon geometry", perspective=0.20)

    partial = (("B", "ROOM", "30"), ("C", "ROOM", "303"))
    add("directory_binding", "directory-1", partial, "B", "UNCERTAIN", "exact target text occurs on directory, partial text is over B", directory_lines=("ROOM 301", "ROOM 302"), perspective=0.12)
    add("directory_binding", "directory-2", partial, "B", "UNCERTAIN", "fresh oblique directory view", directory_lines=("ROOM 301", "ROOM 302"), perspective=0.22)
    add("directory_binding", "close-door", (("B", "ROOM", "302"), ("C", "ROOM", "303")), "B", "TARGET", "close view resolves physical sign")

    suffix = (("A", "ROOM", "302A"), ("B", "ROOM", "302"), ("C", "ROOM", "320"))
    add("suffix_302a", "front-1", suffix, "B", "TARGET", "302A hard negative beside 302")
    add("suffix_302a", "oblique-2", suffix, "B", "TARGET", "fresh suffix geometry", perspective=0.18)

    absent = (("A", "ROOM", "301"), ("B", "ROOM", "303"), ("C", "ROOM", "320"))
    add("target_absent", "wide-1", absent, "NONE", "NONE", "clear target absence")
    add("target_absent", "wide-2", absent, "NONE", "NONE", "independent clear target absence", perspective=0.10)

    add("blur_observability", "blur-static", (("B", "ROOM", "302"),), "B", "UNKNOWN", "blurred distant-style sign", blur_sigma=7.0, perspective=0.28)
    add("blur_observability", "blur-static", (("B", "ROOM", "302"),), "B", "UNKNOWN", "same low-information burst", blur_sigma=7.0, perspective=0.28)

    for _ in range(4):
        add("directory_absence_burst", "static-directory", absent, "NONE", "UNCERTAIN", "repeated unrelated directory target text", directory_lines=("ROOM 301", "ROOM 302"), perspective=0.12)
    add("directory_absence_burst", "wide-clear", absent, "NONE", "NONE", "fresh view removes directory", perspective=0.04)
    return rows


def _box_from_polygon(polygon: Sequence[Sequence[float]]) -> Box:
    points = np.asarray(polygon, dtype=np.float32)
    return Box(
        float(points[:, 0].min() / IMAGE_WIDTH),
        float(points[:, 1].min() / IMAGE_HEIGHT),
        float(points[:, 0].max() / IMAGE_WIDTH),
        float(points[:, 1].max() / IMAGE_HEIGHT),
    )


def _split_line_tokens(raw: Mapping[str, Any]) -> list[tuple[str, Box, float]]:
    """Partition a RapidOCR line polygon only when recognition exposes spaces."""
    text = str(raw["text"]).strip()
    words = [word for word in text.split() if word]
    box = _box_from_polygon(raw["polygon_px"])
    confidence = float(raw["confidence"])
    if len(words) <= 1:
        return [(text, box, confidence)] if text else []
    character_total = sum(len(word) for word in words)
    gap = min((box.x1 - box.x0) * 0.03, 0.008)
    usable = max(1e-6, box.x1 - box.x0 - gap * (len(words) - 1))
    cursor = box.x0
    output = []
    for word in words:
        width = usable * len(word) / character_total
        output.append((word, Box(cursor, box.y0, cursor + width, box.y1), confidence))
        cursor += width + gap
    return output


def _tokens_from_ocr(raw_lines: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    from .semantic_anchor_graph_and_belief_v2 import Token

    tokens = []
    target_tokens = tuple(normalize_text(value) for value in TARGET.tokens)
    for raw in raw_lines:
        for text, box, confidence in _split_line_tokens(raw):
            normalized = normalize_text(text)
            adapted = text
            prefixes = [value for value in target_tokens if len(normalized) >= 2 and value.startswith(normalized) and value != normalized]
            if len(prefixes) == 1:
                adapted = normalized + "?" * (len(prefixes[0]) - len(normalized))
            tokens.append(Token(adapted, box, confidence))
    return tuple(tokens)


def _measured_blur(image: np.ndarray) -> tuple[float, float]:
    variance = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    blur = float(np.clip(1.0 - variance / 420.0, 0.0, 0.92))
    return blur, variance


def _failure_class(row: Mapping[str, Any]) -> str | None:
    expected = row["expected_state"]
    expected_satisfied = row["v2"]["state"] == expected
    if expected == "TARGET":
        expected_satisfied = _row_correct(row, "v2")
    elif expected == "UNCERTAIN" and _row_correct(row, "v2"):
        expected_satisfied = True
    if expected_satisfied:
        return None
    normalized_raw = [normalize_text(item["text"]) for item in row["ocr_raw"]]
    normalized_adapted = [normalize_text(item["text"]) for item in row["frame"]["tokens"]]
    raw_target = any("ROOM302" in value for value in normalized_raw)
    adapted_target = any("ROOM302" in value for value in normalized_adapted) or (
        "ROOM" in normalized_adapted and "302" in normalized_adapted
    )
    if row["v2"]["observability"] < 0.28:
        return "OBSERVABILITY"
    if raw_target and not adapted_target:
        return "OCR_TOKEN_GROUPING"
    if row["truth"] != "NONE" and not raw_target and not any("302" in value for value in normalized_raw):
        return "LEXICAL_CORRUPTION"
    if raw_target:
        return "CANDIDATE_ASSOCIATION"
    return "BELIEF_ACCUMULATION"


def evaluate(
    run_dir: Path,
    runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ocr = RapidOcrPolygonAdapter(runtime_root)
    specs = build_cohort()
    input_root = run_dir / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    episode_ids = list(dict.fromkeys(spec.episode_id for spec in specs))
    for episode_id in episode_ids:
        episode = [spec for spec in specs if spec.episode_id == episode_id]
        baseline = SubstringFsmBaseline(TARGET)
        belief = ReferentBelief(("A", "B", "C"))
        for spec in episode:
            image, candidates = _render_scene(spec)
            image_path = input_root / f"{spec.episode_id}-{spec.frame_index:02d}.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"failed to write image: {image_path}")
            raw_ocr = ocr.read(image)
            tokens = _tokens_from_ocr(raw_ocr)
            measured_blur, laplacian_variance = _measured_blur(image)
            frame = Frame(
                spec.episode_id,
                spec.frame_index,
                spec.viewpoint,
                candidates,
                tokens,
                measured_blur,
                spec.perspective,
                spec.truth,
                spec.expected_state,
                spec.note,
            )
            scores = graph_candidate_scores(TARGET, frame)
            row = {
                "episode_id": spec.episode_id,
                "frame_index": spec.frame_index,
                "viewpoint": spec.viewpoint,
                "truth": spec.truth,
                "expected_state": spec.expected_state,
                "note": spec.note,
                "image_path": str(image_path.resolve()),
                "image_sha256": _sha256_file(image_path),
                "ocr_raw": raw_ocr,
                "ocr_line_count": len(raw_ocr),
                "laplacian_variance": laplacian_variance,
                "frame": asdict(frame),
                "baseline": baseline.update(frame),
                "v2": belief.update(frame, scores),
            }
            row["failure_class"] = _failure_class(row)
            rows.append(row)

    baseline_metrics = _metrics(rows, "baseline")
    v2_metrics = _metrics(rows, "v2")
    failures: dict[str, int] = {}
    for row in rows:
        if row["failure_class"]:
            failures[row["failure_class"]] = failures.get(row["failure_class"], 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "GENERATED_PIXEL_SCENE_RAPIDOCR_DEVELOPMENT",
        "question": "Does frozen SAGE-R V2 retain its relational advantage when its tokens come from real RapidOCR detection and recognition?",
        "v2_algorithm": {
            "status": "unchanged import from semantic_anchor_graph_and_belief_v2.py",
            "source_path": str(Path(inspect.getsourcefile(graph_candidate_scores) or "").resolve()),
            "source_sha256": _sha256_file(Path(inspect.getsourcefile(graph_candidate_scores) or "")),
        },
        "ocr_adapter": (
            "RapidOCR line polygons/confidence; whitespace lines are geometrically partitioned; "
            "a unique strict target-token prefix of length >=2 is represented with trailing '?' wildcards while raw OCR is retained"
        ),
        "frames": len(rows),
        "episodes": len(episode_ids),
        "ocr_lines": sum(row["ocr_line_count"] for row in rows),
        "metrics": {"substring_fsm": baseline_metrics, "sage_r_v2": v2_metrics},
        "delta": {
            "correct_terminal_frames": v2_metrics["correct_terminal_frames"] - baseline_metrics["correct_terminal_frames"],
            "wrong_locks": v2_metrics["wrong_locks"] - baseline_metrics["wrong_locks"],
            "none_correct": v2_metrics["none_correct"] - baseline_metrics["none_correct"],
            "unknown_preserved": v2_metrics["unknown_preserved"] - baseline_metrics["unknown_preserved"],
        },
        "failure_classes": dict(sorted(failures.items())),
        "runtime": ocr.receipt(),
        "interpretation_boundary": [
            "RapidOCR genuinely ran on pixels and supplied its recognized text, confidence, and detected quadrilateral polygons.",
            "The pixel scenes and candidate geometry are generated and mechanism-targeted; they are not natural photographs or camera measurements.",
            "V2 scorer, belief updates, and thresholds were imported unchanged; no outcome-driven threshold sweep was performed.",
            "This does not establish natural-distribution OCR transfer, open-world calibration, Android behavior, navigation, safety, or product performance.",
        ],
        "claim_ceiling": CLAIM_CEILING,
    }
    return {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "rows": rows}, report


def _render_html(raw: Mapping[str, Any], report: Mapping[str, Any], path: Path) -> None:
    metrics = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{report['metrics']['substring_fsm'][key]}</td><td>{report['metrics']['sage_r_v2'][key]}</td></tr>"
        for key in ("correct_terminal_frames", "target_correct_locks", "wrong_locks", "none_correct", "unknown_preserved")
    )
    rows = []
    for row in raw["rows"]:
        baseline = row["baseline"]["state"] + (f":{row['baseline']['selected_candidate']}" if row["baseline"]["selected_candidate"] else "")
        v2 = row["v2"]["state"] + (f":{row['v2']['selected_candidate']}" if row["v2"]["selected_candidate"] else "")
        ocr_text = " | ".join(item["text"] for item in row["ocr_raw"])
        relative_image = Path(row["image_path"]).name
        rows.append(
            f"<tr><td><img src='inputs/{html.escape(relative_image)}'></td><td>{html.escape(row['episode_id'])}/{row['frame_index']}</td>"
            f"<td>{html.escape(ocr_text)}</td><td>{html.escape(row['truth'])}</td><td>{html.escape(baseline)}</td>"
            f"<td>{html.escape(v2)}</td><td>{html.escape(row['failure_class'] or '')}</td></tr>"
        )
    body = f"""<!doctype html><meta charset='utf-8'><title>SAGE-R V2.1 RapidOCR transfer</title>
<style>body{{font:14px system-ui;margin:28px;background:#10141c;color:#edf2f8}}h1{{color:#79e3bd}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #344154;padding:7px;vertical-align:top}}th{{background:#202b3b}}img{{width:240px}}code{{color:#9cc5ff}}</style>
<h1>SAGE-R V2.1: frozen V2 on RapidOCR polygons</h1><p><code>{CLAIM_CEILING}</code></p>
<table><tr><th>metric</th><th>substring + FSM</th><th>SAGE-R V2</th></tr>{metrics}</table>
<table><tr><th>pixel input</th><th>episode</th><th>RapidOCR raw text</th><th>truth</th><th>baseline</th><th>V2</th><th>failure class</th></tr>{''.join(rows)}</table>"""
    path.write_text(body, encoding="utf-8", newline="\n")


def run(run_dir: Path, runtime_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    raw, report = evaluate(run_dir, runtime_root)
    _atomic_json(run_dir / "raw-decisions.json", raw)
    report["raw_decisions_sha256"] = _sha256_file(run_dir / "raw-decisions.json")
    _atomic_json(run_dir / "final-report.json", report)
    _render_html(raw, report, run_dir / "result.html")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("artifacts.local/runtime/semantic-anchor-v1"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.run_dir.resolve(), args.runtime_root.resolve())
    print(json.dumps({"metrics": report["metrics"], "delta": report["delta"], "failure_classes": report["failure_classes"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
