"""Materialize an evidence-bound, category-indexed failure-case picture atlas.

The renderer is deliberately visual-only. It consumes already materialized frame
and model evidence, never changes an algorithm decision, and writes an explicit
NOT_AVAILABLE panel when a requested evidence layer is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import textwrap
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.research.dual_loop_segmentation_failure_atlas.atlas import (
    decode_packed_mask,
    read_json,
    read_jsonl,
    sha256_file,
)


SCHEMA_VERSION = "blindassist.failure_case_atlas.batch.v1"
RENDERER_ID = "FAILURE_CASE_ATLAS_BATCH_ALBUM_R0"
ANALYSIS_SHAPE = (256, 256)
CATEGORY_CONFIG_PATH = Path(__file__).with_name("category_rules.json")


class AlbumInputError(ValueError):
    """Raised when the visual atlas input identity or rules cannot be established."""


SUPPORTED_RULE_TYPES = frozenset(
    {
        "mask_relation",
        "component_flag_or_mask_relation",
        "component_tag",
        "bucket_or_event_token",
        "event_bool_any",
        "event_number_positive",
        "event_token",
        "event_field_token",
        "explicit_appearance_token",
    }
)


def _load_category_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlbumInputError(f"invalid category config: {path}") from exc
    if not isinstance(config, dict):
        raise AlbumInputError("category config must be a JSON object")
    if not isinstance(config.get("categories"), list) or not config["categories"]:
        raise AlbumInputError("category config must contain a non-empty categories list")
    categories = config["categories"]
    slugs: list[str] = []
    for category in categories:
        if not isinstance(category, dict):
            raise AlbumInputError("each category config entry must be an object")
        slug = category.get("slug")
        label = category.get("label")
        rules = category.get("rules", [])
        if not isinstance(slug, str) or not slug:
            raise AlbumInputError("each category requires a non-empty slug")
        if not isinstance(label, str) or not label:
            raise AlbumInputError(f"category {slug!r} requires a non-empty label")
        if slug in slugs:
            raise AlbumInputError(f"duplicate category slug: {slug}")
        if not isinstance(rules, list):
            raise AlbumInputError(f"category {slug} rules must be a list")
        slugs.append(slug)
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("type") not in SUPPORTED_RULE_TYPES:
                raise AlbumInputError(
                    f"category {slug} contains unsupported rule type: {rule!r}"
                )
    default_category = config.get("default_category")
    if default_category not in slugs:
        raise AlbumInputError("category config default_category must name a declared category")
    if slugs[-1] != default_category:
        raise AlbumInputError("default_category must be the final category for stable fallback ordering")
    return config


def _category_specs(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((str(category["slug"]), str(category["label"])) for category in config["categories"])


# The default remains importable for existing callers/tests, while run_album
# can receive a different validated config without mutating process globals.
DEFAULT_CATEGORY_CONFIG = _load_category_config(CATEGORY_CONFIG_PATH)

CATEGORY_SPECS: tuple[tuple[str, str], ...] = _category_specs(DEFAULT_CATEGORY_CONFIG)
CATEGORY_LABELS = dict(CATEGORY_SPECS)

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (70, 70, 70),       # walkable
    1: (255, 190, 40),      # boundary / step / curb
    2: (235, 55, 55),       # obstacle
    3: (175, 80, 220),     # unknown non-walkable
}
CLASS_NAMES = {
    0: "walkable",
    1: "boundary_step_curb",
    2: "obstacle",
    3: "unknown_nonwalkable",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _assert_output_scope(repo_root: Path, output_root: Path) -> None:
    allowed = (repo_root / "artifacts.local").resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise AlbumInputError("output-root must stay under repo-root/artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing album: {output_root}")


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _compact(value: Any, limit: int = 28) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    if limit < 9:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


def _read_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def _as_key(row: dict[str, Any]) -> tuple[str, int] | None:
    if row.get("source_id") is None or row.get("frame_id") is None:
        return None
    return str(row["source_id"]), int(row["frame_id"])


def _load_optional_event_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    by_view: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    by_session: dict[str, dict[str, Any]] = {}
    for row in _read_records(paths):
        if row.get("view_row_id") is not None:
            by_view[str(row["view_row_id"])] = row
        key = _as_key(row)
        if key is not None:
            by_key[key] = row
        if row.get("session_id") is not None:
            by_session[str(row["session_id"])] = row
    return {
        "__by_view__": by_view,
        "__by_key__": by_key,
        "__by_session__": by_session,
    }


def _event_row(
    lookup: dict[str, dict[str, Any]],
    frame: dict[str, Any],
) -> dict[str, Any] | None:
    view = lookup.get("__by_view__", {}).get(str(frame.get("view_row_id")))
    if view is not None:
        return view
    key = _as_key(frame)
    if key is not None:
        keyed = lookup.get("__by_key__", {}).get(key)
        if keyed is not None:
            return keyed
    return lookup.get("__by_session__", {}).get(str(frame.get("session_id")))


def _load_depth_lookup(
    index_path: Path | None,
    maps_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], np.ndarray | None]:
    if index_path is None and maps_path is None:
        return {}, None
    if index_path is None or maps_path is None:
        raise AlbumInputError("depth-index and depth-maps must be supplied together")
    rows = read_jsonl(index_path)
    lookup = {str(row["view_row_id"]): row for row in rows if row.get("view_row_id") is not None}
    if len(lookup) != len(rows):
        raise AlbumInputError("depth index contains duplicate or missing view_row_id")
    maps = np.load(maps_path, mmap_mode="r")
    if maps.ndim != 3:
        raise AlbumInputError(f"depth maps must be [N,H,W], got {maps.shape}")
    for row in rows:
        index = int(row["index"])
        if index < 0 or index >= maps.shape[0]:
            raise AlbumInputError(f"depth index out of range: {index}")
    return lookup, maps


def _load_risk_lookup(
    repo_root: Path,
    index_path: Path | None,
    array_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], np.ndarray | None]:
    if index_path is None and array_path is None:
        return {}, None
    if index_path is None or array_path is None:
        raise AlbumInputError("risk-index and risk-maps must be supplied together")
    rows = read_jsonl(index_path)
    lookup = {str(row["view_row_id"]): row for row in rows if row.get("view_row_id") is not None}
    if len(lookup) != len(rows):
        raise AlbumInputError("risk index contains duplicate or missing view_row_id")
    arrays = np.load(_resolve(repo_root, array_path), mmap_mode="r")
    if arrays.ndim != 3:
        raise AlbumInputError(f"risk maps must be [N,H,W], got {arrays.shape}")
    for row in rows:
        index = int(row["index"])
        if index < 0 or index >= arrays.shape[0]:
            raise AlbumInputError(f"risk index out of range: {index}")
    return lookup, arrays


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    # The existing rehearsal contract resizes RGB into the canonical 256x256
    # analysis space, so masks are deliberately mapped by the same operation.
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _mask_to_size(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _overlay(
    source: Image.Image,
    layers: Sequence[tuple[np.ndarray, tuple[int, int, int], float]],
) -> Image.Image:
    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    for mask, color, alpha in layers:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != array.shape[:2]:
            selected = _mask_to_size(selected, (array.shape[1], array.shape[0]))
        array[selected] = (
            array[selected] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
        )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _class_mask_panel(ids: np.ndarray, size: tuple[int, int]) -> Image.Image:
    ids = np.asarray(ids, dtype=np.uint8)
    colors = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colors[ids == class_id] = color
    panel = Image.fromarray(colors, mode="RGB").resize(size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    font = _load_font(max(10, size[0] // 28), bold=True)
    x = 6
    for class_id, name in CLASS_NAMES.items():
        draw.rectangle((x, size[1] - 22, x + 12, size[1] - 10), fill=CLASS_COLORS[class_id])
        draw.text((x + 16, size[1] - 23), name, fill=(245, 245, 245), font=font)
        x += max(90, int(size[0] / 4))
    return panel


def _scalar_panel(values: np.ndarray, size: tuple[int, int], *, missing_title: str) -> Image.Image:
    value = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(value)
    if not np.any(finite):
        return _missing_panel(size, f"{missing_title}\nNO_FINITE_VALUES")
    q05, q95 = np.percentile(value[finite], [5, 95])
    if q95 <= q05:
        q05 = float(np.min(value[finite]))
        q95 = float(np.max(value[finite]))
    scaled = np.clip((value - q05) / max(q95 - q05, 1e-6), 0.0, 1.0)
    # Perceptually ordered blue -> cyan -> yellow -> red.
    stops = np.asarray(
        [[30, 60, 180], [0, 185, 220], [245, 225, 70], [220, 45, 35]],
        dtype=np.float32,
    )
    positions = scaled * (len(stops) - 1)
    lo = np.floor(positions).astype(np.int32)
    hi = np.minimum(lo + 1, len(stops) - 1)
    fraction = (positions - lo)[..., None]
    rgb = stops[lo] * (1.0 - fraction) + stops[hi] * fraction
    rgb[~finite] = 0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").resize(
        size, Image.Resampling.BILINEAR
    )


def _missing_panel(size: tuple[int, int], message: str) -> Image.Image:
    panel = Image.new("RGB", size, (35, 35, 42))
    draw = ImageDraw.Draw(panel)
    font = _load_font(max(13, size[0] // 20), bold=True)
    lines = message.splitlines()
    line_height = font.size + 5 if hasattr(font, "size") else 20
    total = len(lines) * line_height
    y = max(8, (size[1] - total) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = max(8, (size[0] - (bbox[2] - bbox[0])) // 2)
        draw.text(
            (x, y),
            line,
            fill=(245, 215, 80) if "NOT" in line else (230, 230, 235),
            font=font,
        )
        y += line_height
    return panel


def _draw_yolo_boxes(source: Image.Image, detections: Sequence[dict[str, Any]]) -> Image.Image:
    panel = source.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    font = _load_font(max(10, width // 30), bold=True)
    for detection in detections:
        frame_width = float(detection.get("frame_width", width))
        frame_height = float(detection.get("frame_height", height))
        box = (
            float(detection.get("left", 0.0)) * width / max(frame_width, 1.0),
            float(detection.get("top", 0.0)) * height / max(frame_height, 1.0),
            float(detection.get("right", 0.0)) * width / max(frame_width, 1.0),
            float(detection.get("bottom", 0.0)) * height / max(frame_height, 1.0),
        )
        draw.rectangle(box, outline=(0, 225, 255), width=max(2, width // 128))
        label = f"{detection.get('label', 'object')} {float(detection.get('confidence', 0.0)):.2f}"
        draw.text(
            (box[0] + 2, box[1] + 2),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    if not detections:
        draw.text((8, 8), "YOLO detections: 0", fill=(235, 235, 235), font=font)
    return panel


def _error_panel(
    source: Image.Image,
    prediction: np.ndarray,
    residual_truth: np.ndarray,
) -> Image.Image:
    prediction = np.asarray(prediction, dtype=bool)
    residual_truth = np.asarray(residual_truth, dtype=bool)
    fp = prediction & ~residual_truth
    miss = residual_truth & ~prediction
    hit = prediction & residual_truth
    return _overlay(
        source,
        [
            (fp, (235, 35, 35), 0.72),
            (miss, (45, 120, 255), 0.72),
            (hit, (50, 220, 120), 0.48),
        ],
    )


def _text_panel(
    lines: Sequence[str],
    size: tuple[int, int],
    *,
    title: str | None = None,
) -> Image.Image:
    panel = Image.new("RGB", size, (25, 27, 34))
    draw = ImageDraw.Draw(panel)
    # Text panels must retain all governance/provenance lines inside the
    # 220px panel; keep the body compact and let the full JSON carry details.
    title_font = _load_font(max(13, size[0] // 26), bold=True)
    body_font = _load_font(max(10, size[0] // 42))
    y = 8
    if title:
        draw.text((8, y), title, fill=(255, 215, 80), font=title_font)
        y += title_font.size + 5 if hasattr(title_font, "size") else 22
    char_size = body_font.size if hasattr(body_font, "size") else 12
    max_chars = max(34, size[0] // max(6, char_size - 1))
    for raw_line in lines:
        wrapped = textwrap.wrap(str(raw_line), width=max_chars, break_long_words=False) or [""]
        for line in wrapped:
            if y >= size[1] - 8:
                return panel
            draw.text((8, y), line, fill=(235, 235, 240), font=body_font)
            y += body_font.size + 2 if hasattr(body_font, "size") else 14
    return panel


def _frame_position(
    frame: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[int, int, str]:
    rows = grouped[str(frame["sequence_id"])]
    index = next(i for i, row in enumerate(rows) if row["view_row_id"] == frame["view_row_id"])
    if len(rows) == 1:
        phase = "ONLY"
    elif index == 0:
        phase = "FIRST"
    elif index == len(rows) - 1:
        phase = "LAST"
    else:
        phase = "MIDDLE"
    return index, len(rows), phase


def _event_phase(event: dict[str, Any] | None, position: str) -> tuple[str, str]:
    if event is None:
        return "NOT_EVALUABLE_EVENT_LEDGER_ABSENT", "derived_sequence_position=" + position
    for field in ("event_stage", "stage", "phase", "event_phase"):
        if event.get(field) is not None:
            return str(event[field]), "event-ledger:" + field
    return "NOT_EVALUABLE_EVENT_PHASE_FIELD_ABSENT", "derived_sequence_position=" + position


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "pass",
            "failed",
            "failure",
        }
    return False


def _contains_token(value: Any, tokens: Sequence[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_token(item, tokens) for item in value)
    text = str(value).lower()
    return any(token.lower() in text for token in tokens)


def _mask_relation_matches(
    relation: str,
    prediction: np.ndarray,
    residual_truth: np.ndarray,
) -> bool:
    if relation == "residual_truth_not_covered":
        return bool(np.any(residual_truth & ~prediction))
    if relation == "prediction_outside_residual_truth":
        return bool(np.any(prediction & ~residual_truth))
    raise AlbumInputError(f"unsupported mask relation: {relation}")


def _category_rule_matches(
    *,
    rule: dict[str, Any],
    components: Sequence[dict[str, Any]],
    manifest_bucket: str,
    event: dict[str, Any] | None,
    event_values: Sequence[Any],
    prediction: np.ndarray,
    residual_truth: np.ndarray,
) -> bool:
    rule_type = str(rule["type"])
    if rule_type == "mask_relation":
        return _mask_relation_matches(
            str(rule["relation"]), prediction, residual_truth
        )
    if rule_type == "component_flag_or_mask_relation":
        return any(_boolish(row.get(str(rule["field"]))) for row in components) or _mask_relation_matches(
            str(rule["relation"]), prediction, residual_truth
        )
    if rule_type == "component_tag":
        tag = str(rule["tag"])
        return any(tag in {str(value) for value in row.get("mechanism_tags", [])} for row in components)
    if rule_type == "bucket_or_event_token":
        return _contains_token(manifest_bucket, rule.get("bucket_tokens", [])) or _contains_token(
            event_values, rule.get("event_tokens", [])
        )
    if rule_type == "event_bool_any":
        return event is not None and any(
            _boolish(event.get(str(field))) for field in rule.get("fields", [])
        )
    if rule_type == "event_number_positive":
        if event is None:
            return False
        value = event.get(str(rule["field"]))
        try:
            return value is not None and float(value) > 0
        except (TypeError, ValueError):
            return False
    if rule_type == "event_token":
        return event is not None and _contains_token(event_values, rule.get("tokens", []))
    if rule_type == "event_field_token":
        if event is None:
            return False
        values = [event.get(str(field)) for field in rule.get("fields", [])]
        return _contains_token(values, rule.get("tokens", []))
    if rule_type == "explicit_appearance_token":
        appearance_values = [
            row.get(str(field))
            for row in components
            for field in rule.get("component_fields", [])
        ]
        if rule.get("include_event_values"):
            appearance_values.extend(event_values)
        has_explicit_token = _contains_token(appearance_values, rule.get("tokens", []))
        is_blocked = _contains_token(appearance_values, rule.get("blocked_tokens", []))
        return has_explicit_token and not is_blocked
    raise AlbumInputError(f"unsupported category rule type: {rule_type}")


def _classify_frame(
    *,
    frame: dict[str, Any],
    manifest: dict[str, Any],
    components: Sequence[dict[str, Any]],
    event: dict[str, Any] | None,
    prediction: np.ndarray,
    residual_truth: np.ndarray,
    category_rules: Sequence[dict[str, Any]] | None = None,
    category_specs: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    categories: set[str] = set()
    evidence: dict[str, str] = {}
    rules = category_rules or DEFAULT_CATEGORY_CONFIG["categories"]
    specs = category_specs or CATEGORY_SPECS
    bucket = str(manifest.get("scene_bucket", "")).lower()
    event_values = list(event.values()) if event else []
    for category in rules:
        slug = str(category["slug"])
        for rule in category.get("rules", []):
            if _category_rule_matches(
                rule=rule,
                components=components,
                manifest_bucket=bucket,
                event=event,
                event_values=event_values,
                prediction=prediction,
                residual_truth=residual_truth,
            ):
                categories.add(slug)
                evidence.setdefault(
                    slug,
                    str(rule.get("evidence", f"configured category rule: {rule['type']}")),
                )
                break

    order = [slug for slug, _ in specs]
    return sorted(categories, key=order.index), evidence


def _load_image(repo_root: Path, manifest: dict[str, Any]) -> Image.Image:
    path = _resolve(repo_root, str(manifest["image_repo_relative_path"]))
    if not path.is_file():
        raise AlbumInputError(f"source image missing: {path}")
    if manifest.get("image_sha256") and sha256_file(path) != manifest["image_sha256"]:
        raise AlbumInputError(f"source image hash mismatch: {manifest.get('id')}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_truth(repo_root: Path, view_root: Path, manifest: dict[str, Any]) -> np.ndarray:
    path = (view_root / str(manifest["canonical_mask_path"])).resolve()
    if not path.is_file():
        raise AlbumInputError(f"canonical truth mask missing: {path}")
    if manifest.get("canonical_mask_sha256") and sha256_file(path) != manifest["canonical_mask_sha256"]:
        raise AlbumInputError(f"canonical truth hash mismatch: {manifest.get('id')}")
    with Image.open(path) as image:
        ids = np.asarray(image.convert("L"), dtype=np.uint8)
    if ids.shape != ANALYSIS_SHAPE:
        raise AlbumInputError(
            f"canonical truth shape mismatch for {manifest.get('id')}: {ids.shape}"
        )
    return ids


def _temporal_panel(
    repo_root: Path,
    manifest_by_id: dict[str, dict[str, Any]],
    row: dict[str, Any] | None,
    size: tuple[int, int],
    label: str,
) -> Image.Image:
    if row is None:
        return _missing_panel(size, f"{label}\nNOT_AVAILABLE")
    try:
        image = _load_image(repo_root, manifest_by_id[str(row["view_row_id"])])
    except (KeyError, AlbumInputError):
        return _missing_panel(size, f"{label}\nSOURCE_NOT_AVAILABLE")
    return _fit_image(image, size)


def _panel_title(panel: Image.Image, title: str, *, size: int) -> Image.Image:
    result = panel.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width - 1, max(23, size + 8)), fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=_load_font(size, bold=True))
    return result


def _figure(
    *,
    repo_root: Path,
    manifest_by_id: dict[str, dict[str, Any]],
    view_root: Path,
    frame: dict[str, Any],
    manifest: dict[str, Any],
    trace: dict[str, Any] | None,
    components: Sequence[dict[str, Any]],
    event: dict[str, Any] | None,
    grouped: dict[str, list[dict[str, Any]]],
    depth_row: dict[str, Any] | None,
    depth_maps: np.ndarray | None,
    risk_row: dict[str, Any] | None,
    risk_maps: np.ndarray | None,
    output_path: Path,
    stage: str,
    category_config: dict[str, Any],
) -> dict[str, Any]:
    panel_size = (320, 220)
    category_specs = _category_specs(category_config)
    category_labels = dict(category_specs)
    source = _fit_image(_load_image(repo_root, manifest), panel_size)
    shape = tuple(int(value) for value in frame["packed_masks"]["shape"])
    if shape != ANALYSIS_SHAPE:
        raise AlbumInputError(f"unsupported packed mask shape: {shape}")
    prediction = decode_packed_mask(frame["packed_masks"]["B"], shape)
    detector = decode_packed_mask(frame["packed_masks"]["A"], shape)
    candidate_boundary = decode_packed_mask(
        frame["packed_masks"]["candidate_boundary_step_curb"], shape
    )
    candidate_obstacle = decode_packed_mask(
        frame["packed_masks"]["candidate_obstacle"], shape
    )
    truth_ids = _load_truth(repo_root, view_root, manifest)
    residual_truth = np.isin(truth_ids, np.asarray([1, 2], dtype=np.uint8)) & ~detector

    ordered = grouped[str(frame["sequence_id"])]
    index = next(
        i for i, item in enumerate(ordered)
        if item["view_row_id"] == frame["view_row_id"]
    )
    previous = ordered[index - 1] if index > 0 else None
    future = ordered[index + 1] if index + 1 < len(ordered) else None

    seg_panel = _overlay(
        source,
        [
            (_mask_to_size(candidate_boundary, panel_size), CLASS_COLORS[1], 0.66),
            (_mask_to_size(candidate_obstacle, panel_size), CLASS_COLORS[2], 0.66),
        ],
    )
    truth_panel = _class_mask_panel(truth_ids, panel_size)
    yolo_panel = _draw_yolo_boxes(source, (trace or {}).get("detections", []))
    error_panel = _error_panel(
        source,
        _mask_to_size(prediction, panel_size),
        _mask_to_size(residual_truth, panel_size),
    )

    depth: np.ndarray | None = None
    depth_status = "NOT_AVAILABLE"
    if depth_row is not None and depth_maps is not None:
        if depth_row.get("image_sha256") and depth_row["image_sha256"] != manifest.get("image_sha256"):
            depth_status = "HASH_MISMATCH_REJECTED"
        else:
            depth = np.asarray(depth_maps[int(depth_row["index"])])
            depth_status = "AVAILABLE_DIAGNOSTIC_ONLY"
    risk: np.ndarray | None = None
    risk_status = "NOT_AVAILABLE"
    if risk_row is not None and risk_maps is not None:
        if risk_row.get("image_sha256") and risk_row["image_sha256"] != manifest.get("image_sha256"):
            risk_status = "HASH_MISMATCH_REJECTED"
        else:
            risk = np.asarray(risk_maps[int(risk_row["index"])])
            risk_status = "AVAILABLE_EXTERNAL_INPUT"

    depth_panel = _scalar_panel(depth, panel_size, missing_title="depth") if depth is not None else _missing_panel(
        panel_size, f"depth\n{depth_status}"
    )
    risk_panel = _scalar_panel(risk, panel_size, missing_title="risk heatmap") if risk is not None else _missing_panel(
        panel_size, f"risk heatmap\n{risk_status}\nno frozen risk map supplied"
    )
    prev_panel = _temporal_panel(repo_root, manifest_by_id, previous, panel_size, "previous")
    current_panel = source.copy()
    future_panel = _temporal_panel(repo_root, manifest_by_id, future, panel_size, "future")
    temporal_neighbors = {
        "previous": "AVAILABLE" if previous is not None else "NOT_AVAILABLE_SEQUENCE_BOUNDARY",
        "current": "AVAILABLE",
        "future": "AVAILABLE" if future is not None else "NOT_AVAILABLE_SEQUENCE_BOUNDARY",
    }
    temporal_status = (
        "AVAILABLE_PREVIOUS_CURRENT_FUTURE"
        if previous is not None and future is not None
        else "PARTIAL_SEQUENCE_NEIGHBORS"
    )

    position, sequence_length, position_name = _frame_position(frame, grouped)
    event_stage, event_stage_source = _event_phase(event, position_name)
    confidence_values = [
        float(row["top1_confidence_median"])
        for row in components
        if row.get("top1_confidence_median") is not None
    ]
    confidence = {
        "component_count": len(components),
        "median": float(np.median(confidence_values)) if confidence_values else None,
        "minimum": min(confidence_values) if confidence_values else None,
        "maximum": max(confidence_values) if confidence_values else None,
    }
    categories, category_evidence = _classify_frame(
        frame=frame,
        manifest=manifest,
        components=components,
        event=event,
        prediction=prediction,
        residual_truth=residual_truth,
        category_rules=category_config["categories"],
        category_specs=category_specs,
    )
    if not categories:
        categories = ["other"]
        category_evidence["other"] = "no requested category matched"

    panels = [
        ("原图 / source", source),
        ("YOLO 框 / boxes", yolo_panel),
        ("segmentation mask", seg_panel),
        ("truth mask", truth_panel),
        ("depth", depth_panel),
        ("risk heatmap", risk_panel),
        ("previous", prev_panel),
        ("current", current_panel),
        ("future", future_panel),
        ("error map: FP red / miss blue / hit green", error_panel),
        (
            "event / model metadata",
            _text_panel(
                [
                    f"categories: {', '.join(category_labels.get(value, value) for value in categories)}",
                    f"component_count: {len(components)}",
                    f"false_activation_components: {sum(_boolish(row.get('false_activation')) for row in components)}",
                    f"confidence median/min/max: {confidence['median']} / {confidence['minimum']} / {confidence['maximum']}",
                    f"event_stage: {event_stage}",
                    f"event_stage_source: {event_stage_source}",
                    f"sequence_position: {position + 1}/{sequence_length} ({position_name})",
                    f"scene_bucket: {manifest.get('scene_bucket', 'NOT_AVAILABLE')}",
                    f"role: {frame.get('rehearsal_role', manifest.get('role', 'NOT_AVAILABLE'))}",
                    f"stage: {stage}",
                ],
                panel_size,
                title="错误类型 / 置信度 / 事件阶段",
            ),
        ),
        (
            "source / provenance",
            _text_panel(
                [
                    f"view: {_compact(frame.get('view_row_id'), 30)}",
                    f"source: {_compact(frame.get('source_id'), 30)}",
                    f"session: {_compact(frame.get('session_id'), 30)}",
                    f"sequence: {_compact(frame.get('sequence_id'), 30)}",
                    f"frame_id: {frame.get('frame_id')}",
                    f"timestamp_ns: {frame.get('source_capture_timestamp_ns', manifest.get('source_capture_timestamp_ns'))}",
                    f"image_sha256: {_compact(manifest.get('image_sha256'), 24)}",
                    f"YOLO: {_compact((trace or {}).get('authority', 'NOT_AVAILABLE'), 28)}",
                    f"depth: {depth_status}",
                    f"risk: {risk_status}",
                    "VISUALIZATION ONLY / DOES NOT DRIVE ALERTS",
                ],
                panel_size,
                title="来源信息 / provenance",
            ),
        ),
    ]

    canvas = Image.new("RGB", (1280, 760), (12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22, bold=True)
    body_font = _load_font(14)
    draw.rectangle((0, 0, 1279, 42), fill=(78, 28, 30))
    draw.text(
        (12, 8),
        "FAILURE CASE ATLAS · DEVELOPMENT VISUALIZATION ONLY",
        fill=(255, 230, 100),
        font=title_font,
    )
    header = (
        f"{_compact(frame.get('view_row_id'), 42)}  |  "
        + _compact(" / ".join(category_labels.get(value, value) for value in categories), 142)
    )
    draw.text((12, 47), header, fill=(235, 235, 240), font=body_font)
    for panel_index, (title, panel) in enumerate(panels):
        x = (panel_index % 4) * panel_size[0]
        y = 72 + (panel_index // 4) * panel_size[1]
        canvas.paste(_panel_title(panel, title, size=14), (x, y))
    draw.text(
        (12, 738),
        "Does not drive alerts · No production/safety conclusion · missing layers are explicit",
        fill=(255, 125, 125),
        font=body_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)

    return {
        "case_id": _safe_name(str(frame["view_row_id"])),
        "view_row_id": frame["view_row_id"],
        # Keep paths relative to the album root so cases.jsonl is directly
        # consumable without requiring callers to know the renderer layout.
        "figure_path": (Path("figures") / output_path.name).as_posix(),
        "categories": categories,
        "category_labels": [category_labels.get(value, value) for value in categories],
        "category_evidence": category_evidence,
        "component_ids": [row.get("component_id") for row in components],
        "component_count": len(components),
        "false_activation_component_count": sum(
            _boolish(row.get("false_activation")) for row in components
        ),
        "confidence": confidence,
        "event_stage": event_stage,
        "event_stage_source": event_stage_source,
        "sequence_position": {
            "index": position,
            "length": sequence_length,
            "label": position_name,
        },
        "error_metrics": {
            "candidate_pixels": int(np.count_nonzero(prediction)),
            "residual_truth_pixels": int(np.count_nonzero(residual_truth)),
            "false_positive_pixels": int(np.count_nonzero(prediction & ~residual_truth)),
            "miss_pixels": int(np.count_nonzero(residual_truth & ~prediction)),
        },
        "source_info": {
            "source_id": frame.get("source_id"),
            "session_id": frame.get("session_id"),
            "sequence_id": frame.get("sequence_id"),
            "frame_id": frame.get("frame_id"),
            "timestamp_ns": frame.get(
                "source_capture_timestamp_ns",
                manifest.get("source_capture_timestamp_ns"),
            ),
            "scene_bucket": manifest.get("scene_bucket"),
            "role": frame.get("rehearsal_role", manifest.get("role")),
            "image_sha256": manifest.get("image_sha256"),
        },
        "algorithm_layers": {
            "yolo_boxes": {"status": "AVAILABLE" if trace is not None else "NOT_AVAILABLE"},
            "segmentation_mask": {"status": "AVAILABLE_FROM_PACKED_REHEARSAL_MASK"},
            "truth_mask": {"status": "AVAILABLE_CANONICAL_SOURCE_TRUTH"},
            "depth": {"status": depth_status},
            "risk_heatmap": {"status": risk_status},
            "previous_current_future": {"status": temporal_status},
        },
        "temporal_neighbors": temporal_neighbors,
        "provenance": {
            "manifest_id": manifest.get("id"),
            "yolo_trace_authority": (trace or {}).get("authority"),
            "atlas_protocol_id": frame.get("protocol_id"),
        },
    }


def _make_thumbnail(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        thumb = image.copy()
        thumb.thumbnail((320, 205), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 205), (12, 14, 18))
        canvas.paste(thumb, ((320 - thumb.width) // 2, (205 - thumb.height) // 2))
        canvas.save(target, format="JPEG", quality=82, optimize=True)


def _write_contact_sheets(
    *,
    root: Path,
    cases: Sequence[dict[str, Any]],
    category: str,
) -> list[str]:
    category_root = root / "categories" / category
    category_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(category_root / "cases.jsonl", list(cases))
    sheets: list[str] = []
    page_size = 40
    columns, rows = 4, 10
    cell_w, cell_h = 320, 238
    for page, start in enumerate(range(0, len(cases), page_size)):
        subset = cases[start : start + page_size]
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (10, 12, 16))
        draw = ImageDraw.Draw(sheet)
        label_font = _load_font(13, bold=True)
        for index, case in enumerate(subset):
            thumb_path = root / "thumbnails" / f"{case['case_id']}.jpg"
            if thumb_path.is_file():
                with Image.open(thumb_path) as image:
                    thumb = image.convert("RGB")
                x = (index % columns) * cell_w
                y = (index // columns) * cell_h
                sheet.paste(thumb, (x, y))
                label = f"{start + index + 1:03d}  {case['case_id']}"
                draw.rectangle(
                    (x, y + 205, x + cell_w - 1, y + cell_h - 1),
                    fill=(8, 8, 10),
                )
                draw.text((x + 5, y + 210), label, fill=(235, 235, 240), font=label_font)
        filename = f"contact_sheet_{page:03d}.jpg"
        sheet.save(category_root / filename, format="JPEG", quality=86, optimize=True)
        sheets.append((Path("categories") / category / filename).as_posix())
    return sheets


def _write_index_html(
    root: Path,
    cases: Sequence[dict[str, Any]],
    category_counts: dict[str, int],
    category_specs: Sequence[tuple[str, str]],
) -> None:
    cards = []
    for case in cases:
        labels = ", ".join(case["category_labels"])
        figure = html.escape(Path(case["figure_path"]).as_posix())
        cards.append(
            f'<article class="case" data-categories="{html.escape(" ".join(case["categories"]))}">'
            f'<a href="{figure}"><img loading="lazy" src="{figure}" alt="{html.escape(labels)}"></a>'
            f"<div><strong>{html.escape(labels)}</strong><br>"
            f'<code>{html.escape(str(case["view_row_id"]))}</code><br>'
            f'FP={case["error_metrics"]["false_positive_pixels"]} · '
            f'miss={case["error_metrics"]["miss_pixels"]} · '
            f'conf={html.escape(str(case["confidence"]["median"]))}</div></article>'
        )
    buttons = [
        f'<button data-filter="{html.escape(slug)}">{html.escape(label)} ({category_counts.get(slug, 0)})</button>'
        for slug, label in category_specs
    ]
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Failure Case Atlas</title>
<style>
body{background:#101218;color:#e9e9ee;font-family:Segoe UI,Arial,sans-serif;margin:24px}
h1{color:#ffe06a}.notice{color:#ff9a9a}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
button{background:#2b3040;color:#fff;border:1px solid #586073;border-radius:4px;padding:6px 10px;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.case{background:#1c1f28;padding:8px;border-radius:6px}.case img{width:100%;height:auto;display:block}
.case code{font-size:11px;word-break:break-all;color:#b9c0d0}.case strong{color:#ffe06a}
</style></head><body>
<h1>Failure Case Atlas</h1><p class="notice">DEVELOPMENT VISUALIZATION ONLY · DOES NOT DRIVE ALERTS · no production/safety conclusion</p>
<p>Each case keeps the original, YOLO boxes, segmentation/truth masks, depth, optional risk heatmap, temporal neighbors, error type, confidence, event-stage and provenance in one figure.</p>
<div class="filters"><button data-filter="all">全部</button>__BUTTONS__</div><main class="grid">__CASES__</main>
<script>const cards=[...document.querySelectorAll('.case')];document.querySelectorAll('button').forEach(b=>b.onclick=()=>{const f=b.dataset.filter;cards.forEach(c=>c.style.display=(f==='all'||c.dataset.categories.split(' ').includes(f))?'block':'none')});</script>
</body></html>"""
    document = document.replace("__BUTTONS__", "".join(buttons)).replace("__CASES__", "".join(cards))
    (root / "index.html").write_text(document, encoding="utf-8")


def _lookup_trace(trace_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in trace_rows:
        key = _as_key(row)
        if key is None:
            continue
        if key in lookup and lookup[key] != row:
            raise AlbumInputError(f"conflicting YOLO trace rows for {key}")
        lookup[key] = row
    return lookup


def run_album(
    *,
    repo_root: Path,
    frames_paths: Sequence[Path],
    atlas_components_path: Path,
    view_root: Path,
    yolo_trace_paths: Sequence[Path],
    output_root: Path,
    atlas_result_path: Path | None = None,
    depth_index_path: Path | None = None,
    depth_maps_path: Path | None = None,
    risk_index_path: Path | None = None,
    risk_maps_path: Path | None = None,
    event_ledger_paths: Sequence[Path] = (),
    category_config_path: Path | None = None,
    include_clean_frames: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    view_root = view_root.resolve()
    output_root = output_root.resolve()
    category_config_path = (category_config_path or CATEGORY_CONFIG_PATH).resolve()
    category_config = _load_category_config(category_config_path)
    category_specs = _category_specs(category_config)
    _assert_output_scope(repo_root, output_root)

    frame_rows = _read_records(frames_paths)
    manifest_rows = read_jsonl(view_root / "manifest.jsonl")
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise AlbumInputError("duplicate canonical manifest id")

    enriched_frames: list[dict[str, Any]] = []
    for original in frame_rows:
        view_id = str(original["view_row_id"])
        manifest = manifest_by_id.get(view_id)
        if manifest is None:
            raise AlbumInputError(f"canonical manifest missing frame: {view_id}")
        row = dict(original)
        row["sequence_id"] = manifest["sequence_id"]
        row["source_capture_timestamp_ns"] = manifest["source_capture_timestamp_ns"]
        row["scene_bucket"] = manifest.get("scene_bucket")
        enriched_frames.append(row)
    frame_by_id = {str(row["view_row_id"]): row for row in enriched_frames}
    if len(frame_by_id) != len(enriched_frames):
        raise AlbumInputError("duplicate view_row_id in frame inputs")

    atlas_components = read_jsonl(atlas_components_path)
    component_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in atlas_components:
        component_by_view[str(row["view_row_id"])].append(row)
    traces = _lookup_trace(_read_records(yolo_trace_paths))
    depth_lookup, depth_maps = _load_depth_lookup(depth_index_path, depth_maps_path)
    risk_lookup, risk_maps = _load_risk_lookup(repo_root, risk_index_path, risk_maps_path)
    event_lookup = _load_optional_event_rows(event_ledger_paths)

    selected: list[dict[str, Any]] = []
    for row in enriched_frames:
        view_id = str(row["view_row_id"])
        components = component_by_view.get(view_id, [])
        candidate = decode_packed_mask(row["packed_masks"]["B"], ANALYSIS_SHAPE)
        detector = decode_packed_mask(row["packed_masks"]["A"], ANALYSIS_SHAPE)
        manifest = manifest_by_id[view_id]
        truth = _load_truth(repo_root, view_root, manifest)
        residual_truth = np.isin(truth, np.asarray([1, 2], dtype=np.uint8)) & ~detector
        has_failure = bool(
            components
            or np.any(candidate & ~residual_truth)
            or np.any(residual_truth & ~candidate)
        )
        if include_clean_frames or has_failure:
            selected.append(row)
    selected.sort(
        key=lambda row: (
            str(row["sequence_id"]),
            int(row["frame_id"]),
            str(row["view_row_id"]),
        )
    )
    if max_frames is not None:
        selected = selected[: max(0, int(max_frames))]
    if not selected:
        raise AlbumInputError("no frames selected for album")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["sequence_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["frame_id"]), str(row["view_row_id"])))

    stage = "DEVELOPMENT_VISUALIZATION_ONLY"
    atlas_result = read_json(atlas_result_path) if atlas_result_path else {}
    if atlas_result.get("claim_ceiling"):
        stage += " / " + str(atlas_result["claim_ceiling"])

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        figures_root = temporary / "figures"
        thumbnails_root = temporary / "thumbnails"
        figures_root.mkdir()
        thumbnails_root.mkdir()
        shutil.copy2(category_config_path, temporary / "category_rules.json")
        case_rows: list[dict[str, Any]] = []
        for index, frame in enumerate(selected):
            view_id = str(frame["view_row_id"])
            manifest = manifest_by_id[view_id]
            trace = traces.get(_as_key(frame)) if _as_key(frame) is not None else None
            event = _event_row(event_lookup, frame)
            figure_path = figures_root / f"{_safe_name(view_id)}.png"
            case = _figure(
                repo_root=repo_root,
                manifest_by_id=manifest_by_id,
                view_root=view_root,
                frame=frame,
                manifest=manifest,
                trace=trace,
                components=component_by_view.get(view_id, []),
                event=event,
                grouped=grouped,
                depth_row=depth_lookup.get(view_id),
                depth_maps=depth_maps,
                risk_row=risk_lookup.get(view_id),
                risk_maps=risk_maps,
                output_path=figure_path,
                stage=stage,
                category_config=category_config,
            )
            _make_thumbnail(figure_path, thumbnails_root / f"{case['case_id']}.jpg")
            case["ordinal"] = index + 1
            case_rows.append(case)

        _write_jsonl(temporary / "cases.jsonl", case_rows)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in case_rows:
            for category in case["categories"]:
                by_category[category].append(case)
        category_counts = {slug: len(by_category.get(slug, [])) for slug, _ in category_specs}
        category_status: dict[str, dict[str, Any]] = {}
        category_by_slug = {
            str(category["slug"]): category for category in category_config["categories"]
        }
        for slug, label in category_specs:
            entries = by_category.get(slug, [])
            sheets = _write_contact_sheets(root=temporary, cases=entries, category=slug)
            status = "MATERIALIZED" if entries else "NO_MATCHING_CASES"
            reason = None
            if not entries and not event_ledger_paths:
                reason = category_by_slug[slug].get("no_match_reason_without_event_ledger")
            category_status[slug] = {
                "label": label,
                "case_count": len(entries),
                "status": status,
                "reason": reason,
                "contact_sheets": sheets,
            }

        _write_json(temporary / "category_index.json", category_status)
        _write_index_html(temporary, case_rows, category_counts, category_specs)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "renderer_id": RENDERER_ID,
            "inputs": {
                "category_rules": {
                    "path": _relative_path(repo_root, category_config_path),
                    "sha256": sha256_file(category_config_path),
                },
                "frames": [
                    {"path": _relative_path(repo_root, path), "sha256": sha256_file(path)}
                    for path in frames_paths
                ],
                "atlas_components": {
                    "path": _relative_path(repo_root, atlas_components_path),
                    "sha256": sha256_file(atlas_components_path),
                },
                "canonical_manifest": {
                    "path": _relative_path(repo_root, view_root / "manifest.jsonl"),
                    "sha256": sha256_file(view_root / "manifest.jsonl"),
                },
                "yolo_trace": [
                    {"path": _relative_path(repo_root, path), "sha256": sha256_file(path)}
                    for path in yolo_trace_paths
                ],
                "atlas_result": (
                    {
                        "path": _relative_path(repo_root, atlas_result_path),
                        "sha256": sha256_file(atlas_result_path),
                    }
                    if atlas_result_path
                    else None
                ),
                "depth_index": (
                    {
                        "path": _relative_path(repo_root, depth_index_path),
                        "sha256": sha256_file(depth_index_path),
                    }
                    if depth_index_path
                    else None
                ),
                "depth_maps": (
                    {
                        "path": _relative_path(repo_root, depth_maps_path),
                        "sha256": sha256_file(depth_maps_path),
                    }
                    if depth_maps_path
                    else None
                ),
                "risk_index": (
                    {
                        "path": _relative_path(repo_root, risk_index_path),
                        "sha256": sha256_file(risk_index_path),
                    }
                    if risk_index_path
                    else None
                ),
                "risk_maps": (
                    {
                        "path": _relative_path(repo_root, risk_maps_path),
                        "sha256": sha256_file(risk_maps_path),
                    }
                    if risk_maps_path
                    else None
                ),
                "event_ledger": [
                    {"path": _relative_path(repo_root, path), "sha256": sha256_file(path)}
                    for path in event_ledger_paths
                ],
            },
            "claim_ceiling": atlas_result.get(
                "claim_ceiling",
                "visual-only; no algorithm/product/safety claim",
            ),
            "category_rules": {
                "schema_version": category_config.get("schema_version"),
                "category_count": len(category_specs),
                "default_category": category_config.get("default_category"),
            },
            "layer_status": {
                "original": "AVAILABLE",
                "yolo_boxes": "AVAILABLE_WHEN_TRACE_MATCHES",
                "segmentation_mask": "AVAILABLE_FROM_PACKED_REHEARSAL_MASK",
                "truth_mask": "AVAILABLE_CANONICAL_SOURCE_TRUTH",
                "depth": "AVAILABLE_WHEN_HASH_ALIGNED" if depth_maps is not None else "NOT_AVAILABLE",
                "risk_heatmap": "AVAILABLE_EXTERNAL_INPUT" if risk_maps is not None else "NOT_AVAILABLE",
                "temporal_frames": "AVAILABLE_WHEN_SEQUENCE_NEIGHBORS_EXIST",
                "event_stage": (
                    "AVAILABLE_EXTERNAL_INPUT"
                    if event_ledger_paths
                    else "NOT_EVALUABLE_WITHOUT_EVENT_LEDGER"
                ),
            },
        }
        _write_json(temporary / "provenance.json", provenance)
        result = {
            "schema_version": SCHEMA_VERSION,
            "renderer_id": RENDERER_ID,
            "authority": "VISUAL_ONLY",
            "drives_alerts": False,
            "stage": stage,
            "input_counts": {
                "selected_frames": len(selected),
                "atlas_components": len(atlas_components),
                "selected_components": sum(
                    len(component_by_view.get(str(row["view_row_id"]), []))
                    for row in selected
                ),
                "sequences": len(grouped),
            },
            "category_counts": category_counts,
            "category_status": category_status,
            "category_rules": {
                "schema_version": category_config.get("schema_version"),
                "category_count": len(category_specs),
                "default_category": category_config.get("default_category"),
            },
            "layer_status": provenance["layer_status"],
            "outputs": {
                "index": "index.html",
                "cases": "cases.jsonl",
                "category_index": "category_index.json",
                "category_rules": "category_rules.json",
                "provenance": "provenance.json",
                "figures": "figures",
                "thumbnails": "thumbnails",
            },
        }
        _write_json(temporary / "result.json", result)
        temporary.replace(output_root)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--frames", required=True, action="append")
    parser.add_argument("--atlas-components", required=True)
    parser.add_argument("--view-root", required=True)
    parser.add_argument("--yolo-trace", required=True, action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--atlas-result")
    parser.add_argument("--depth-index")
    parser.add_argument("--depth-maps")
    parser.add_argument("--risk-index")
    parser.add_argument("--risk-maps")
    parser.add_argument("--event-ledger", action="append", default=[])
    parser.add_argument(
        "--category-config",
        help="JSON category rules; defaults to scripts/research/failure_case_atlas/category_rules.json",
    )
    parser.add_argument("--include-clean-frames", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    result = run_album(
        repo_root=repo_root,
        frames_paths=[_resolve(repo_root, value) for value in args.frames],
        atlas_components_path=_resolve(repo_root, args.atlas_components),
        view_root=_resolve(repo_root, args.view_root),
        yolo_trace_paths=[_resolve(repo_root, value) for value in args.yolo_trace],
        output_root=_resolve(repo_root, args.output_root),
        atlas_result_path=_resolve(repo_root, args.atlas_result) if args.atlas_result else None,
        depth_index_path=_resolve(repo_root, args.depth_index) if args.depth_index else None,
        depth_maps_path=_resolve(repo_root, args.depth_maps) if args.depth_maps else None,
        risk_index_path=_resolve(repo_root, args.risk_index) if args.risk_index else None,
        risk_maps_path=_resolve(repo_root, args.risk_maps) if args.risk_maps else None,
        event_ledger_paths=[_resolve(repo_root, value) for value in args.event_ledger],
        category_config_path=_resolve(repo_root, args.category_config) if args.category_config else None,
        include_clean_frames=bool(args.include_clean_frames),
        max_frames=args.max_frames,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
