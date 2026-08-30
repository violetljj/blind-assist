"""Materialize resolution-independent YOLO instance-segmentation candidates.

The command accepts only explicitly named RGB image files: either one
``--image``, an ``--image-index`` JSONL whose rows contain ``image_path``, or
the strict native ``--c2-model-root`` contract.  The C2 mode follows only the
root manifest, episode manifests, fixed observation ledgers, and their
``wearable_rgb`` references.  It does not discover neighbouring files and
rejects evaluator, sidecar, instance, witness, and truth paths.  A local model
file is required before Ultralytics is imported, so a missing checkpoint can
never trigger a model download.

Every candidate uses normalized image coordinates.  This keeps the artifact
valid when the corresponding RGB/depth pair is captured at a resolution other
than the original CARLA canary resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_MODEL = REPO / "artifacts.local" / "models" / "yolo11n-seg.pt"

SCHEMA = "blindassist-dtr-yolo-metric-candidates-v1"
MANIFEST_SCHEMA = "blindassist-dtr-yolo-metric-candidate-manifest-v1"
HAZARD_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)
CONFIDENCE = 0.10
NMS_IOU = 0.50
# C2 deliberately contains more than fifty visible asset instances.  Keep the
# cap above that scene density so the target is never dropped merely because
# distractors were materialized first.
MAX_DETECTIONS = 100
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
ALLOWED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "BMP", "WEBP"})
ALLOWED_INDEX_KEYS = frozenset(
    {
        "image_path",
        "image_sha256",
        "frame_id",
        "episode_id",
        "sample_index",
        "time_s",
        "world_frame",
    }
)
FORBIDDEN_PATH_TOKENS = frozenset(
    {"evaluator", "evaluators", "sidecar", "sidecars", "instance", "witness", "depth"}
)
FORBIDDEN_INDEX_FILENAMES = frozenset(
    {
        "frames.jsonl",
        "manifest.json",
        "sealed_manifest.json",
        "summary.json",
        "payload_inventory.json",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
C2_EPISODE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
C2_ROOT_SCHEMA = "dtr-c2-model-root-manifest-v2"
C2_EPISODE_SCHEMA = "dtr-c2-model-episode-manifest-v2"
C2_OBSERVATION_SCHEMA = "dtr-c2-model-observation-v2"
C2_LEGACY_ROOT_SCHEMA = "dtr-c2-model-root-manifest-v1"
C2_LEGACY_EPISODE_SCHEMA = "dtr-c2-model-episode-manifest-v1"
C2_LEGACY_OBSERVATION_SCHEMA = "dtr-c2-model-observation-v1"
C2_CALIBRATION_SCHEMA = "dtr-c2-model-camera-contract-v1"
C2_ALIGNMENT_SCHEMA = "dtr-c2-model-rgbd-deterministic-replay-alignment-receipt-v1"
C2_EXPERIMENT_ID = "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2"
C2_ALIGNMENT_AUTHORITY = "DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED"
C2_WORLD_FRAME_RULE = (
    "world_frame equals wearable_rgb.source_world_frame; metric_depth is "
    "mapped into that namespace by the verified per-episode source offset"
)
C2_WIDTH = 1280
C2_HEIGHT = 720
C2_LEGACY_ROOT_FIELDS = frozenset(
    {"schema_version", "experiment_id", "camera_calibration", "episodes"}
)
C2_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "camera_calibration",
        "model_contract",
        "rgbd_alignment_receipt",
        "episodes",
    }
)
C2_FILE_REFERENCE_FIELDS = frozenset({"path", "sha256"})
C2_EPISODE_REFERENCE_FIELDS = frozenset(
    {"episode_id", "manifest_path", "manifest_sha256"}
)
C2_CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "resolution",
        "fov_degrees",
        "K",
        "depth_codec",
        "wearable_rigid_extrinsic",
        "sensor_tick_seconds",
    }
)
C2_LEGACY_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "frames",
        "observations_sha256",
        "rgb_payloads",
        "depth_payloads",
        "navigation_session_id",
        "issued_plan",
    }
)
C2_EPISODE_FIELDS = C2_LEGACY_EPISODE_FIELDS | frozenset({"rgbd_alignment"})
C2_EPISODE_PLAN_FIELDS = frozenset(
    {"authority", "path", "receipt_sha256", "file_sha256"}
)
C2_LEGACY_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "sample_index",
        "world_frame",
        "time_s",
        "timestamp_s",
        "wearable_rgb",
        "metric_depth",
        "camera",
        "wearer_pose_current",
        "navigation",
    }
)
C2_OBSERVATION_FIELDS = C2_LEGACY_OBSERVATION_FIELDS | frozenset({"frame_alignment"})
C2_LEGACY_RGB_FIELDS = frozenset({"path", "bytes", "sha256", "width", "height"})
C2_RGB_FIELDS = C2_LEGACY_RGB_FIELDS | frozenset({"source_world_frame"})
C2_DEPTH_FIELDS = frozenset(
    {"path", "bytes", "sha256", "width", "height", "codec", "source_world_frame"}
)
C2_DEPTH_CODEC_FIELDS = frozenset({"name", "maximum_depth_m", "formula"})
C2_ALIGNMENT_REFERENCE_FIELDS = frozenset({"path", "receipt_sha256", "sha256"})
C2_ALIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "authority",
        "world_frame_rule",
        "matching_keys",
        "verified_equal_fields",
        "episodes",
        "receipt_sha256",
    }
)
C2_ALIGNMENT_EPISODE_FIELDS = frozenset(
    {
        "episode_id",
        "frames",
        "wearable_source_world_frame_first",
        "wearable_source_world_frame_last",
        "depth_source_world_frame_first",
        "depth_source_world_frame_last",
        "depth_minus_wearable_source_world_frame_offset",
        "alignment_projection_sha256",
    }
)
C2_EPISODE_ALIGNMENT_FIELDS = frozenset(
    {
        "authority",
        "receipt_path",
        "receipt_sha256",
        "depth_minus_wearable_source_world_frame_offset",
    }
)
C2_FRAME_ALIGNMENT_FIELDS = C2_EPISODE_ALIGNMENT_FIELDS | frozenset(
    {"reference_modality"}
)
C2_FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "actor",
        "actor_id",
        "actor_ids",
        "actors",
        "bbox",
        "bounding_box",
        "collision_polygons_xy",
        "contact",
        "contact_label",
        "current_actors",
        "current_contact",
        "evaluator",
        "expected_outcome",
        "future_contact_within_horizon",
        "instance",
        "instance_segmentation",
        "instance_visibility",
        "layout_id",
        "layout_role",
        "occlusion",
        "occlusion_label",
        "realized_future",
        "responsible_asset",
        "responsible_assets",
        "role",
        "scenario_role",
        "semantic_role",
        "sidecar",
        "track_id",
        "truth",
        "twin_role",
        "velocity",
        "witness",
    }
)
C2_FORBIDDEN_PATH_TOKENS = frozenset(
    {"evaluator", "evaluators", "instance", "oracle", "shard", "shards", "sidecar", "sidecars", "truth", "witness"}
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        tokens.update(value for value in re.split(r"[^a-z0-9]+", part.casefold()) if value)
    return tokens


def resolve_explicit_file(path: Path, *, label: str, reject_truth_paths: bool) -> Path:
    require(path.is_absolute(), f"{label}_must_be_absolute:{path}")
    provided_tokens = path_tokens(path)
    if reject_truth_paths:
        require(not (provided_tokens & FORBIDDEN_PATH_TOKENS), f"{label}_forbidden_path:{path}")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f"{label}_not_file:{resolved}")
    if reject_truth_paths:
        require(not (path_tokens(resolved) & FORBIDDEN_PATH_TOKENS), f"{label}_forbidden_resolved_path:{resolved}")
    return resolved


@dataclass(frozen=True)
class ImageInput:
    path: Path
    metadata: dict[str, Any]
    expected_sha256: str | None


@dataclass(frozen=True)
class C2ModelInputs:
    inputs: list[ImageInput]
    model_root: Path
    manifest_path: Path
    manifest_sha256: str
    schema_version: str
    experiment_id: str
    episode_ids: list[str]
    rgbd_alignment: dict[str, Any] | None


def normalize_metadata(row: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {"frame_id": str(row.get("frame_id", ordinal))}
    if "episode_id" in row:
        require(isinstance(row["episode_id"], str) and bool(row["episode_id"]), f"index_episode_id:{ordinal}")
        metadata["episode_id"] = row["episode_id"]
    if "sample_index" in row:
        require(isinstance(row["sample_index"], int) and not isinstance(row["sample_index"], bool), f"index_sample_index:{ordinal}")
        metadata["sample_index"] = int(row["sample_index"])
    if "time_s" in row:
        require(isinstance(row["time_s"], (int, float)) and not isinstance(row["time_s"], bool), f"index_time_s:{ordinal}")
        require(math.isfinite(float(row["time_s"])), f"index_time_s_finite:{ordinal}")
        metadata["time_s"] = float(row["time_s"])
    if "world_frame" in row:
        require(isinstance(row["world_frame"], int) and not isinstance(row["world_frame"], bool), f"index_world_frame:{ordinal}")
        require(int(row["world_frame"]) >= 0, f"index_world_frame_negative:{ordinal}")
        metadata["world_frame"] = int(row["world_frame"])
    require(len(metadata["frame_id"]) <= 256, f"index_frame_id_length:{ordinal}")
    return metadata


def load_image_index(path: Path) -> tuple[list[ImageInput], str]:
    index_path = resolve_explicit_file(path, label="image_index", reject_truth_paths=True)
    require(index_path.name.casefold() not in FORBIDDEN_INDEX_FILENAMES, f"image_index_forbidden_name:{index_path}")
    require(index_path.suffix.casefold() == ".jsonl", f"image_index_suffix:{index_path}")
    inputs: list[ImageInput] = []
    seen_paths: set[Path] = set()
    seen_frame_ids: set[str] = set()
    with index_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"image_index_json:{line_number}:{error.msg}") from error
            require(isinstance(row, dict), f"image_index_row_object:{line_number}")
            require(set(row) <= ALLOWED_INDEX_KEYS, f"image_index_unexpected_keys:{line_number}")
            require(isinstance(row.get("image_path"), str), f"image_index_image_path:{line_number}")
            image_path = resolve_rgb_path(Path(row["image_path"]), f"image_index_rgb_{line_number}")
            require(image_path not in seen_paths, f"image_index_duplicate_path:{line_number}:{image_path}")
            metadata = normalize_metadata(row, len(inputs))
            require(metadata["frame_id"] not in seen_frame_ids, f"image_index_duplicate_frame_id:{line_number}")
            expected = row.get("image_sha256")
            if expected is not None:
                require(isinstance(expected, str) and SHA256_PATTERN.fullmatch(expected) is not None, f"image_index_sha256:{line_number}")
                expected = expected.upper()
            inputs.append(ImageInput(image_path, metadata, expected))
            seen_paths.add(image_path)
            seen_frame_ids.add(metadata["frame_id"])
    require(bool(inputs), "image_index_empty")
    return inputs, sha256_file(index_path)


def resolve_rgb_path(path: Path, label: str = "rgb") -> Path:
    resolved = resolve_explicit_file(path, label=label, reject_truth_paths=True)
    require(resolved.suffix.casefold() in ALLOWED_IMAGE_SUFFIXES, f"{label}_suffix:{resolved}")
    return resolved


def single_image_input(path: Path) -> list[ImageInput]:
    resolved = resolve_rgb_path(path)
    return [ImageInput(resolved, {"frame_id": resolved.stem}, None)]


def exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(str(key) for key in value)
    require(
        actual == expected,
        f"{label}_fields:missing={sorted(expected - actual)}:extra={sorted(actual - expected)}",
    )


def c2_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"{label}_sha256")
    return value.upper()


def c2_nonnegative_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label}_integer")
    return int(value)


def c2_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label}_integer")
    return int(value)


def c2_positive_int(value: Any, label: str) -> int:
    result = c2_nonnegative_int(value, label)
    require(result > 0, f"{label}_positive")
    return result


def c2_finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}_number")
    result = float(value)
    require(math.isfinite(result), f"{label}_finite")
    return result


def c2_receipt_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: child for key, child in value.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def assert_c2_truth_blind(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            require(
                normalized not in C2_FORBIDDEN_MODEL_KEYS and not normalized.endswith("_actor_id"),
                f"c2_forbidden_model_key:{path}.{key}",
            )
            if normalized in {"path", "manifest_path"} and isinstance(child, str):
                require(
                    not (path_tokens(Path(child)) & C2_FORBIDDEN_PATH_TOKENS),
                    f"c2_forbidden_model_path:{path}.{key}:{child}",
                )
            assert_c2_truth_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_c2_truth_blind(child, f"{path}[{index}]")


def read_c2_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label}_json:{error.msg}") from error
    require(isinstance(value, dict), f"{label}_object:{path}")
    assert_c2_truth_blind(value, label)
    return value


def resolve_c2_contained_file(
    model_root: Path,
    raw_path: Any,
    label: str,
    *,
    expected_relative: str | None = None,
) -> Path:
    require(isinstance(raw_path, str) and bool(raw_path), f"{label}_path")
    normalized = raw_path.replace("\\", "/")
    if expected_relative is not None:
        require(normalized == expected_relative, f"{label}_unexpected_path:{normalized}")
    candidate = Path(raw_path)
    require(not candidate.is_absolute(), f"{label}_absolute:{raw_path}")
    require(".." not in candidate.parts, f"{label}_parent_reference:{raw_path}")
    require(
        not (path_tokens(candidate) & C2_FORBIDDEN_PATH_TOKENS),
        f"{label}_forbidden_path:{raw_path}",
    )
    resolved = (model_root / candidate).resolve(strict=True)
    require(resolved.is_relative_to(model_root) and resolved.is_file(), f"{label}_escape:{raw_path}")
    relative = resolved.relative_to(model_root)
    require(
        not (path_tokens(relative) & C2_FORBIDDEN_PATH_TOKENS),
        f"{label}_forbidden_resolved_path:{relative}",
    )
    return resolved


def validate_c2_calibration(
    model_root: Path,
    reference: Any,
    *,
    legacy: bool,
) -> dict[str, Any] | str:
    require(isinstance(reference, Mapping), "c2_calibration_reference_object")
    exact_fields(reference, C2_FILE_REFERENCE_FIELDS, "c2_calibration_reference")
    path = resolve_c2_contained_file(
        model_root,
        reference["path"],
        "c2_calibration",
        expected_relative="camera_calibration.json",
    )
    require(sha256_file(path) == c2_sha256(reference["sha256"], "c2_calibration"), "c2_calibration_hash_mismatch")
    value = read_c2_json(path, "c2_calibration")
    exact_fields(value, C2_CALIBRATION_FIELDS, "c2_calibration")
    require(value["schema_version"] == C2_CALIBRATION_SCHEMA, "c2_calibration_schema")
    resolution = value["resolution"]
    require(isinstance(resolution, Mapping), "c2_calibration_resolution_object")
    exact_fields(resolution, frozenset({"width", "height"}), "c2_calibration_resolution")
    require(
        c2_positive_int(resolution["width"], "c2_calibration_width") == C2_WIDTH
        and c2_positive_int(resolution["height"], "c2_calibration_height") == C2_HEIGHT,
        "c2_calibration_resolution",
    )
    codec = value["depth_codec"]
    if legacy:
        require(isinstance(codec, str) and bool(codec), "c2_calibration_legacy_depth_codec")
        return codec
    require(isinstance(codec, Mapping), "c2_calibration_depth_codec_object")
    exact_fields(codec, C2_DEPTH_CODEC_FIELDS, "c2_calibration_depth_codec")
    require(isinstance(codec["name"], str) and bool(codec["name"]), "c2_calibration_depth_codec_name")
    require(
        c2_finite(codec["maximum_depth_m"], "c2_calibration_depth_codec_maximum") > 0.0,
        "c2_calibration_depth_codec_maximum_positive",
    )
    require(
        isinstance(codec["formula"], str) and bool(codec["formula"]),
        "c2_calibration_depth_codec_formula",
    )
    return dict(codec)


def validate_c2_unopened_reference(reference: Any, label: str, expected_path: str) -> None:
    require(isinstance(reference, Mapping), f"{label}_object")
    exact_fields(reference, C2_FILE_REFERENCE_FIELDS, label)
    require(reference["path"] == expected_path, f"{label}_path:{reference['path']}")
    require(
        not (path_tokens(Path(reference["path"])) & C2_FORBIDDEN_PATH_TOKENS),
        f"{label}_forbidden_path:{reference['path']}",
    )
    c2_sha256(reference["sha256"], label)


def validate_c2_alignment_receipt(
    model_root: Path,
    reference: Any,
    episode_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(isinstance(reference, Mapping), "c2_alignment_reference_object")
    exact_fields(reference, C2_ALIGNMENT_REFERENCE_FIELDS, "c2_alignment_reference")
    path = resolve_c2_contained_file(
        model_root,
        reference["path"],
        "c2_alignment_receipt",
        expected_relative="rgbd_alignment_receipt.json",
    )
    file_sha256 = c2_sha256(reference["sha256"], "c2_alignment_receipt_file")
    require(sha256_file(path) == file_sha256, "c2_alignment_receipt_file_hash")
    receipt = read_c2_json(path, "c2_alignment_receipt")
    exact_fields(receipt, C2_ALIGNMENT_FIELDS, "c2_alignment_receipt")
    require(receipt["schema_version"] == C2_ALIGNMENT_SCHEMA, "c2_alignment_receipt_schema")
    require(receipt["experiment_id"] == C2_EXPERIMENT_ID, "c2_alignment_receipt_experiment")
    require(receipt["authority"] == C2_ALIGNMENT_AUTHORITY, "c2_alignment_receipt_authority")
    require(receipt["world_frame_rule"] == C2_WORLD_FRAME_RULE, "c2_alignment_world_frame_rule")
    require(
        receipt["matching_keys"] == ["episode_id", "sample_index", "time_s"],
        "c2_alignment_matching_keys",
    )
    require(
        receipt["verified_equal_fields"]
        == ["camera_world_transform", "wearer_pose_current"],
        "c2_alignment_verified_fields",
    )
    receipt_sha256 = c2_sha256(receipt["receipt_sha256"], "c2_alignment_receipt")
    require(
        receipt_sha256 == c2_sha256(reference["receipt_sha256"], "c2_alignment_reference"),
        "c2_alignment_reference_receipt_hash",
    )
    require(c2_receipt_sha256(receipt) == receipt_sha256, "c2_alignment_receipt_payload_hash")
    episodes = receipt["episodes"]
    require(isinstance(episodes, list), "c2_alignment_episodes")
    require(
        [value.get("episode_id") if isinstance(value, Mapping) else None for value in episodes]
        == list(episode_ids),
        "c2_alignment_episode_order",
    )
    by_episode: dict[str, dict[str, Any]] = {}
    for value, episode_id in zip(episodes, episode_ids, strict=True):
        require(isinstance(value, Mapping), f"c2_alignment_episode_object:{episode_id}")
        exact_fields(value, C2_ALIGNMENT_EPISODE_FIELDS, f"c2_alignment_episode:{episode_id}")
        require(value["episode_id"] == episode_id, f"c2_alignment_episode_id:{episode_id}")
        frames = c2_positive_int(value["frames"], f"c2_alignment_frames:{episode_id}")
        wearable_first = c2_nonnegative_int(
            value["wearable_source_world_frame_first"], f"c2_alignment_wearable_first:{episode_id}"
        )
        wearable_last = c2_nonnegative_int(
            value["wearable_source_world_frame_last"], f"c2_alignment_wearable_last:{episode_id}"
        )
        depth_first = c2_nonnegative_int(
            value["depth_source_world_frame_first"], f"c2_alignment_depth_first:{episode_id}"
        )
        depth_last = c2_nonnegative_int(
            value["depth_source_world_frame_last"], f"c2_alignment_depth_last:{episode_id}"
        )
        offset = c2_int(
            value["depth_minus_wearable_source_world_frame_offset"],
            f"c2_alignment_offset:{episode_id}",
        )
        require(wearable_last == wearable_first + frames - 1, f"c2_alignment_wearable_range:{episode_id}")
        require(depth_last == depth_first + frames - 1, f"c2_alignment_depth_range:{episode_id}")
        require(
            depth_first - wearable_first == offset and depth_last - wearable_last == offset,
            f"c2_alignment_offset_range:{episode_id}",
        )
        c2_sha256(value["alignment_projection_sha256"], f"c2_alignment_projection:{episode_id}")
        by_episode[episode_id] = dict(value)
    return by_episode, {
        "authority": C2_ALIGNMENT_AUTHORITY,
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha256,
        "file_sha256": file_sha256,
        "world_frame_rule": C2_WORLD_FRAME_RULE,
    }


def validate_c2_alignment_link(
    value: Any,
    *,
    episode_id: str,
    receipt_sha256: str,
    expected_offset: int,
    frame_level: bool,
    label: str,
) -> None:
    require(isinstance(value, Mapping), f"{label}_object")
    exact_fields(
        value,
        C2_FRAME_ALIGNMENT_FIELDS if frame_level else C2_EPISODE_ALIGNMENT_FIELDS,
        label,
    )
    require(value["authority"] == C2_ALIGNMENT_AUTHORITY, f"{label}_authority")
    require(value["receipt_path"] == "rgbd_alignment_receipt.json", f"{label}_receipt_path")
    require(
        c2_sha256(value["receipt_sha256"], f"{label}_receipt") == receipt_sha256,
        f"{label}_receipt_hash",
    )
    require(
        c2_int(value["depth_minus_wearable_source_world_frame_offset"], f"{label}_offset")
        == expected_offset,
        f"{label}_offset_mismatch:{episode_id}",
    )
    if frame_level:
        require(value["reference_modality"] == "wearable_rgb", f"{label}_reference_modality")


def validate_c2_episode_manifest(
    model_root: Path,
    episode_id: str,
    reference: Mapping[str, Any],
    *,
    legacy: bool,
    alignment_episode: Mapping[str, Any] | None,
    alignment_receipt_sha256: str | None,
) -> tuple[dict[str, Any], Path]:
    exact_fields(reference, C2_EPISODE_REFERENCE_FIELDS, f"c2_episode_reference:{episode_id}")
    expected_manifest = f"episodes/{episode_id}/manifest.json"
    manifest_path = resolve_c2_contained_file(
        model_root,
        reference["manifest_path"],
        f"c2_episode_manifest:{episode_id}",
        expected_relative=expected_manifest,
    )
    expected_hash = c2_sha256(reference["manifest_sha256"], f"c2_episode_manifest:{episode_id}")
    require(sha256_file(manifest_path) == expected_hash, f"c2_episode_manifest_hash:{episode_id}")
    manifest = read_c2_json(manifest_path, f"c2_episode_manifest:{episode_id}")
    exact_fields(
        manifest,
        C2_LEGACY_EPISODE_FIELDS if legacy else C2_EPISODE_FIELDS,
        f"c2_episode_manifest:{episode_id}",
    )
    require(
        manifest["schema_version"]
        == (C2_LEGACY_EPISODE_SCHEMA if legacy else C2_EPISODE_SCHEMA),
        f"c2_episode_schema:{episode_id}",
    )
    require(manifest["episode_id"] == episode_id, f"c2_episode_identity:{episode_id}")
    frames = c2_positive_int(manifest["frames"], f"c2_episode_frames:{episode_id}")
    require(
        c2_nonnegative_int(manifest["rgb_payloads"], f"c2_episode_rgb_count:{episode_id}") == frames,
        f"c2_episode_rgb_count_mismatch:{episode_id}",
    )
    require(
        c2_nonnegative_int(manifest["depth_payloads"], f"c2_episode_depth_count:{episode_id}") == frames,
        f"c2_episode_depth_count_mismatch:{episode_id}",
    )
    if legacy:
        require(
            alignment_episode is None and alignment_receipt_sha256 is None,
            f"c2_legacy_alignment_present:{episode_id}",
        )
    else:
        require(
            alignment_episode is not None and alignment_receipt_sha256 is not None,
            f"c2_alignment_missing:{episode_id}",
        )
        require(
            c2_positive_int(alignment_episode["frames"], f"c2_alignment_frames:{episode_id}")
            == frames,
            f"c2_alignment_frame_count:{episode_id}",
        )
        validate_c2_alignment_link(
            manifest["rgbd_alignment"],
            episode_id=episode_id,
            receipt_sha256=alignment_receipt_sha256,
            expected_offset=c2_int(
                alignment_episode["depth_minus_wearable_source_world_frame_offset"],
                f"c2_alignment_offset:{episode_id}",
            ),
            frame_level=False,
            label=f"c2_episode_alignment:{episode_id}",
        )
    require(
        isinstance(manifest["navigation_session_id"], str) and bool(manifest["navigation_session_id"]),
        f"c2_episode_navigation_session:{episode_id}",
    )
    issued_plan = manifest["issued_plan"]
    require(isinstance(issued_plan, Mapping), f"c2_episode_plan_object:{episode_id}")
    exact_fields(issued_plan, C2_EPISODE_PLAN_FIELDS, f"c2_episode_plan:{episode_id}")
    require(issued_plan["authority"] in {"VALID", "NO_PLAN"}, f"c2_episode_plan_authority:{episode_id}")
    require(issued_plan["path"] == f"plans/{episode_id}.json", f"c2_episode_plan_path:{episode_id}")
    c2_sha256(issued_plan["file_sha256"], f"c2_episode_plan_file:{episode_id}")
    if issued_plan["authority"] == "VALID":
        c2_sha256(issued_plan["receipt_sha256"], f"c2_episode_plan_receipt:{episode_id}")
    else:
        require(issued_plan["receipt_sha256"] is None, f"c2_episode_no_plan_receipt:{episode_id}")
    observations_path = resolve_c2_contained_file(
        model_root,
        f"episodes/{episode_id}/observations.jsonl",
        f"c2_observations:{episode_id}",
        expected_relative=f"episodes/{episode_id}/observations.jsonl",
    )
    expected_observations_hash = c2_sha256(
        manifest["observations_sha256"], f"c2_observations:{episode_id}"
    )
    require(
        sha256_file(observations_path) == expected_observations_hash,
        f"c2_observations_hash:{episode_id}",
    )
    return manifest, observations_path


def validate_c2_rgb_reference(
    model_root: Path,
    episode_id: str,
    sample_index: int,
    value: Any,
    *,
    legacy: bool,
    aligned_world_frame: int,
) -> tuple[Path, str, int]:
    label = f"c2_rgb:{episode_id}:{sample_index}"
    require(isinstance(value, Mapping), f"{label}_object")
    exact_fields(value, C2_LEGACY_RGB_FIELDS if legacy else C2_RGB_FIELDS, label)
    expected_relative = f"episodes/{episode_id}/rgb/{sample_index:06d}.png"
    path = resolve_c2_contained_file(
        model_root,
        value["path"],
        label,
        expected_relative=expected_relative,
    )
    expected_bytes = c2_positive_int(value["bytes"], f"{label}_bytes")
    require(path.stat().st_size == expected_bytes, f"{label}_bytes_mismatch")
    expected_sha256 = c2_sha256(value["sha256"], label)
    require(sha256_file(path) == expected_sha256, f"{label}_hash_mismatch")
    require(
        c2_positive_int(value["width"], f"{label}_width") == C2_WIDTH
        and c2_positive_int(value["height"], f"{label}_height") == C2_HEIGHT,
        f"{label}_declared_resolution",
    )
    with Image.open(path) as image:
        require(image.format == "PNG", f"{label}_format:{image.format}")
        require(image.size == (C2_WIDTH, C2_HEIGHT), f"{label}_encoded_resolution:{image.size}")
        image.verify()
    source_world_frame = (
        aligned_world_frame
        if legacy
        else c2_nonnegative_int(value["source_world_frame"], f"{label}_source_world_frame")
    )
    require(source_world_frame == aligned_world_frame, f"{label}_aligned_world_frame")
    return path, expected_sha256, source_world_frame


def validate_c2_unopened_depth_reference(
    episode_id: str,
    sample_index: int,
    value: Any,
    calibration_depth_codec: Mapping[str, Any],
) -> int:
    label = f"c2_depth_metadata:{episode_id}:{sample_index}"
    require(isinstance(value, Mapping), f"{label}_object")
    exact_fields(value, C2_DEPTH_FIELDS, label)
    expected_path = f"episodes/{episode_id}/depth/{sample_index:06d}.png"
    require(isinstance(value["path"], str), f"{label}_path_type")
    require(value["path"].replace("\\", "/") == expected_path, f"{label}_path")
    c2_positive_int(value["bytes"], f"{label}_bytes")
    c2_sha256(value["sha256"], label)
    require(
        c2_positive_int(value["width"], f"{label}_width") == C2_WIDTH
        and c2_positive_int(value["height"], f"{label}_height") == C2_HEIGHT,
        f"{label}_declared_resolution",
    )
    codec = value["codec"]
    require(isinstance(codec, Mapping), f"{label}_codec_object")
    exact_fields(codec, C2_DEPTH_CODEC_FIELDS, f"{label}_codec")
    require(isinstance(codec["name"], str) and bool(codec["name"]), f"{label}_codec_name")
    require(
        c2_finite(codec["maximum_depth_m"], f"{label}_codec_maximum") > 0.0,
        f"{label}_codec_maximum_positive",
    )
    require(isinstance(codec["formula"], str) and bool(codec["formula"]), f"{label}_codec_formula")
    require(dict(codec) == dict(calibration_depth_codec), f"{label}_codec_calibration_mismatch")
    return c2_nonnegative_int(value["source_world_frame"], f"{label}_source_world_frame")


def load_c2_observations(
    model_root: Path,
    episode_id: str,
    episode_manifest: Mapping[str, Any],
    observations_path: Path,
    *,
    legacy: bool,
    calibration_depth_codec: dict[str, Any] | str,
    alignment_episode: Mapping[str, Any] | None,
    alignment_receipt_sha256: str | None,
) -> list[ImageInput]:
    inputs: list[ImageInput] = []
    previous_time = -math.inf
    previous_world_frame = -1
    with observations_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            require(bool(line.strip()), f"c2_observations_blank_line:{episode_id}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"c2_observation_json:{episode_id}:{line_number}:{error.msg}") from error
            require(isinstance(row, dict), f"c2_observation_object:{episode_id}:{line_number}")
            assert_c2_truth_blind(row, f"c2_observation:{episode_id}:{line_number}")
            exact_fields(
                row,
                C2_LEGACY_OBSERVATION_FIELDS if legacy else C2_OBSERVATION_FIELDS,
                f"c2_observation:{episode_id}:{line_number}",
            )
            require(
                row["schema_version"]
                == (C2_LEGACY_OBSERVATION_SCHEMA if legacy else C2_OBSERVATION_SCHEMA),
                f"c2_observation_schema:{episode_id}:{line_number}",
            )
            require(row["episode_id"] == episode_id, f"c2_observation_episode:{episode_id}:{line_number}")
            sample_index = c2_nonnegative_int(row["sample_index"], f"c2_sample:{episode_id}:{line_number}")
            require(sample_index == len(inputs), f"c2_sample_order:{episode_id}:{sample_index}")
            time_s = c2_finite(row["time_s"], f"c2_time:{episode_id}:{sample_index}")
            timestamp_s = c2_finite(row["timestamp_s"], f"c2_timestamp:{episode_id}:{sample_index}")
            require(abs(time_s - timestamp_s) <= 1e-9, f"c2_timestamp_mismatch:{episode_id}:{sample_index}")
            require(time_s > previous_time, f"c2_time_order:{episode_id}:{sample_index}")
            world_frame = c2_nonnegative_int(row["world_frame"], f"c2_world_frame:{episode_id}:{sample_index}")
            require(world_frame > previous_world_frame, f"c2_world_frame_order:{episode_id}:{sample_index}")
            rgb_path, rgb_sha256, wearable_source_world_frame = validate_c2_rgb_reference(
                model_root,
                episode_id,
                sample_index,
                row["wearable_rgb"],
                legacy=legacy,
                aligned_world_frame=world_frame,
            )
            depth_source_world_frame: int | None = None
            if legacy:
                require(isinstance(row["metric_depth"], Mapping), f"c2_metric_depth_object:{episode_id}:{sample_index}")
            else:
                require(
                    alignment_episode is not None and alignment_receipt_sha256 is not None,
                    f"c2_frame_alignment_missing:{episode_id}:{sample_index}",
                )
                offset = c2_int(
                    alignment_episode["depth_minus_wearable_source_world_frame_offset"],
                    f"c2_alignment_offset:{episode_id}",
                )
                expected_wearable_source = c2_nonnegative_int(
                    alignment_episode["wearable_source_world_frame_first"],
                    f"c2_alignment_wearable_first:{episode_id}",
                ) + sample_index
                expected_depth_source = c2_nonnegative_int(
                    alignment_episode["depth_source_world_frame_first"],
                    f"c2_alignment_depth_first:{episode_id}",
                ) + sample_index
                require(
                    wearable_source_world_frame == expected_wearable_source,
                    f"c2_wearable_source_frame_receipt:{episode_id}:{sample_index}",
                )
                depth_source_world_frame = validate_c2_unopened_depth_reference(
                    episode_id,
                    sample_index,
                    row["metric_depth"],
                    calibration_depth_codec,
                )
                require(
                    depth_source_world_frame == expected_depth_source
                    and depth_source_world_frame - wearable_source_world_frame == offset,
                    f"c2_depth_source_frame_receipt:{episode_id}:{sample_index}",
                )
                validate_c2_alignment_link(
                    row["frame_alignment"],
                    episode_id=episode_id,
                    receipt_sha256=alignment_receipt_sha256,
                    expected_offset=offset,
                    frame_level=True,
                    label=f"c2_frame_alignment:{episode_id}:{sample_index}",
                )
            navigation = row["navigation"]
            require(isinstance(navigation, Mapping), f"c2_navigation_object:{episode_id}:{sample_index}")
            exact_fields(
                navigation,
                frozenset({"navigation_session_id", "issued_plan"}),
                f"c2_navigation:{episode_id}:{sample_index}",
            )
            require(
                navigation["navigation_session_id"] == episode_manifest["navigation_session_id"],
                f"c2_navigation_session:{episode_id}:{sample_index}",
            )
            issued_plan = navigation["issued_plan"]
            require(isinstance(issued_plan, Mapping), f"c2_navigation_plan_object:{episode_id}:{sample_index}")
            exact_fields(
                issued_plan,
                frozenset({"authority", "path", "receipt_sha256"}),
                f"c2_navigation_plan:{episode_id}:{sample_index}",
            )
            expected_plan = episode_manifest["issued_plan"]
            require(
                issued_plan["authority"] == expected_plan["authority"]
                and issued_plan["path"] == expected_plan["path"]
                and issued_plan["receipt_sha256"] == expected_plan["receipt_sha256"],
                f"c2_navigation_plan_mismatch:{episode_id}:{sample_index}",
            )
            require(isinstance(row["camera"], Mapping), f"c2_camera_object:{episode_id}:{sample_index}")
            require(isinstance(row["wearer_pose_current"], Mapping), f"c2_wearer_pose_object:{episode_id}:{sample_index}")
            metadata: dict[str, Any] = {
                "frame_id": f"{episode_id}/{sample_index:06d}",
                "episode_id": episode_id,
                "sample_index": sample_index,
                "time_s": time_s,
                "world_frame": world_frame,
            }
            if not legacy:
                metadata.update(
                    {
                        "world_frame_namespace": "wearable_rgb.source_world_frame",
                        "wearable_source_world_frame": wearable_source_world_frame,
                        "depth_source_world_frame": depth_source_world_frame,
                    }
                )
            inputs.append(
                ImageInput(
                    rgb_path,
                    metadata,
                    rgb_sha256,
                )
            )
            previous_time = time_s
            previous_world_frame = world_frame
    require(
        len(inputs) == c2_positive_int(episode_manifest["frames"], f"c2_episode_frames:{episode_id}"),
        f"c2_observation_count:{episode_id}:{len(inputs)}",
    )
    return inputs


def load_c2_model_root(path: Path) -> C2ModelInputs:
    require(path.is_absolute(), f"c2_model_root_must_be_absolute:{path}")
    require(not (path_tokens(path) & C2_FORBIDDEN_PATH_TOKENS), f"c2_model_root_forbidden:{path}")
    model_root = path.resolve(strict=True)
    require(model_root.is_dir(), f"c2_model_root_not_directory:{model_root}")
    require(
        not (path_tokens(model_root) & C2_FORBIDDEN_PATH_TOKENS),
        f"c2_model_root_resolved_forbidden:{model_root}",
    )
    manifest_path = (model_root / "manifest.json").resolve(strict=True)
    require(manifest_path.is_relative_to(model_root) and manifest_path.is_file(), "c2_root_manifest_missing")
    manifest_sha256 = sha256_file(manifest_path)
    manifest = read_c2_json(manifest_path, "c2_root_manifest")
    schema_version = manifest.get("schema_version")
    legacy = schema_version == C2_LEGACY_ROOT_SCHEMA
    require(legacy or schema_version == C2_ROOT_SCHEMA, "c2_root_manifest_schema")
    exact_fields(
        manifest,
        C2_LEGACY_ROOT_FIELDS if legacy else C2_ROOT_FIELDS,
        "c2_root_manifest",
    )
    require(manifest["experiment_id"] == C2_EXPERIMENT_ID, "c2_root_manifest_experiment")
    calibration_depth_codec = validate_c2_calibration(
        model_root, manifest["camera_calibration"], legacy=legacy
    )
    references = manifest["episodes"]
    require(isinstance(references, list) and bool(references), "c2_root_episodes")
    episode_ids: list[str] = []
    for index, reference in enumerate(references):
        require(isinstance(reference, Mapping), f"c2_episode_reference_object:{index}")
        exact_fields(reference, C2_EPISODE_REFERENCE_FIELDS, f"c2_episode_reference:{index}")
        episode_id = reference["episode_id"]
        require(
            isinstance(episode_id, str) and C2_EPISODE_ID_PATTERN.fullmatch(episode_id) is not None,
            f"c2_episode_id:{index}",
        )
        require(episode_id not in episode_ids, f"c2_episode_duplicate:{episode_id}")
        episode_ids.append(episode_id)
    require(episode_ids == sorted(episode_ids), f"c2_episode_order:{episode_ids}")
    alignment_by_episode: dict[str, dict[str, Any]] = {}
    alignment_summary: dict[str, Any] | None = None
    if not legacy:
        validate_c2_unopened_reference(
            manifest["model_contract"], "c2_model_contract_reference", "model_contract.json"
        )
        alignment_by_episode, alignment_summary = validate_c2_alignment_receipt(
            model_root, manifest["rgbd_alignment_receipt"], episode_ids
        )
    inputs: list[ImageInput] = []
    seen_rgb_paths: set[Path] = set()
    for reference, episode_id in zip(references, episode_ids, strict=True):
        episode_manifest, observations_path = validate_c2_episode_manifest(
            model_root,
            episode_id,
            reference,
            legacy=legacy,
            alignment_episode=alignment_by_episode.get(episode_id),
            alignment_receipt_sha256=(
                str(alignment_summary["receipt_sha256"])
                if alignment_summary is not None
                else None
            ),
        )
        episode_inputs = load_c2_observations(
            model_root,
            episode_id,
            episode_manifest,
            observations_path,
            legacy=legacy,
            calibration_depth_codec=calibration_depth_codec,
            alignment_episode=alignment_by_episode.get(episode_id),
            alignment_receipt_sha256=(
                str(alignment_summary["receipt_sha256"])
                if alignment_summary is not None
                else None
            ),
        )
        for image in episode_inputs:
            require(image.path not in seen_rgb_paths, f"c2_duplicate_rgb:{image.path}")
            seen_rgb_paths.add(image.path)
        inputs.extend(episode_inputs)
    require(bool(inputs), "c2_model_no_rgb_inputs")
    return C2ModelInputs(
        inputs=inputs,
        model_root=model_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        schema_version=str(schema_version),
        experiment_id=str(manifest["experiment_id"]),
        episode_ids=episode_ids,
        rgbd_alignment=alignment_summary,
    )


def create_output_directory(path: Path) -> Path:
    require(path.is_absolute(), f"output_dir_must_be_absolute:{path}")
    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), f"output_parent_not_directory:{parent}")
    output = parent / path.name
    os.mkdir(output)
    return output.resolve(strict=True)


def model_names(model: Any) -> dict[int, str]:
    raw = model.names
    if isinstance(raw, Mapping):
        names = {int(key): str(value) for key, value in raw.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw)}
    inverse = {name.casefold(): class_id for class_id, name in names.items()}
    missing = [name for name in HAZARD_CLASS_NAMES if name not in inverse]
    require(not missing, f"model_missing_hazard_classes:{','.join(missing)}")
    return names


def model_stride(model: Any) -> int:
    value = getattr(getattr(model, "model", None), "stride", None)
    require(value is not None, "model_stride_missing")
    if hasattr(value, "max"):
        value = value.max()
    if hasattr(value, "item"):
        value = value.item()
    stride = int(value)
    require(stride > 0, f"model_stride_invalid:{stride}")
    return stride


def normalized_bbox(box: Sequence[float], *, width: int, height: int) -> list[float]:
    values = np.asarray(box, dtype=np.float64).reshape(4)
    values[[0, 2]] /= float(width)
    values[[1, 3]] /= float(height)
    values = np.clip(values, 0.0, 1.0)
    return [float(value) for value in values]


def normalized_polygon(points: Any, *, width: int, height: int) -> list[list[float]]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < 3 or not np.isfinite(values).all():
        return []
    values[:, 0] = np.clip(values[:, 0] / float(width), 0.0, 1.0)
    values[:, 1] = np.clip(values[:, 1] / float(height), 0.0, 1.0)
    twice_area = abs(float(np.dot(values[:, 0], np.roll(values[:, 1], 1)) - np.dot(values[:, 1], np.roll(values[:, 0], 1))))
    if twice_area <= 0.0:
        return []
    return [[float(x), float(y)] for x, y in values]


def decode_rgb(path: Path) -> tuple[np.ndarray, int, int, str]:
    with Image.open(path) as image:
        image_format = str(image.format or "").upper()
        require(image_format in ALLOWED_IMAGE_FORMATS, f"rgb_format:{path}:{image_format}")
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb_shape:{path}:{rgb.shape}")
    height, width = map(int, rgb.shape[:2])
    require(width > 0 and height > 0, f"rgb_dimensions:{path}")
    return rgb, width, height, image_format


def infer_one(
    model: Any,
    names: Mapping[int, str],
    image: ImageInput,
    *,
    model_path: Path,
    model_sha256: str,
    device: str | None,
    run_kind: str,
) -> dict[str, Any]:
    image_sha256 = sha256_file(image.path)
    if image.expected_sha256 is not None:
        require(image_sha256 == image.expected_sha256, f"rgb_sha256_mismatch:{image.path}")
    rgb, width, height, image_format = decode_rgb(image.path)
    stride = model_stride(model)
    inference_height = int(math.ceil(height / stride) * stride)
    inference_width = int(math.ceil(width / stride) * stride)
    inverse = {name.casefold(): class_id for class_id, name in names.items()}
    class_ids = [inverse[name] for name in HAZARD_CLASS_NAMES]
    predict_args: dict[str, Any] = {
        "source": [rgb[:, :, ::-1].copy()],
        "imgsz": (inference_height, inference_width),
        "conf": CONFIDENCE,
        "iou": NMS_IOU,
        "classes": class_ids,
        "max_det": MAX_DETECTIONS,
        "augment": False,
        "batch": 1,
        "rect": True,
        "retina_masks": True,
        "save": False,
        "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device
    predictions = model.predict(**predict_args)
    require(len(predictions) == 1, f"prediction_count:{image.path}")
    prediction = predictions[0]
    boxes, masks = prediction.boxes, prediction.masks
    candidates: list[dict[str, Any]] = []
    if boxes is not None and len(boxes):
        require(masks is not None and len(masks.xy) == len(boxes), f"mask_alignment:{image.path}")
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidences = boxes.conf.detach().cpu().numpy().astype(np.float64)
        coordinates = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        for class_id, confidence, box, polygon in zip(classes, confidences, coordinates, masks.xy):
            class_id = int(class_id)
            require(class_id in class_ids, f"unexpected_class:{image.path}:{class_id}")
            bbox = normalized_bbox(box, width=width, height=height)
            points = normalized_polygon(polygon, width=width, height=height)
            if len(points) < 3 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            candidates.append(
                {
                    "bbox_xyxy_normalized": bbox,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": float(confidence),
                    "polygon_xy_normalized": points,
                }
            )
    candidates.sort(
        key=lambda value: (
            int(value["class_id"]),
            float(value["bbox_xyxy_normalized"][0]),
            float(value["bbox_xyxy_normalized"][1]),
            -float(value["confidence"]),
        )
    )
    counts = Counter(str(value["class_name"]) for value in candidates)
    return {
        "schema": SCHEMA,
        "run_kind": run_kind,
        "source": {
            **image.metadata,
            "image_path": str(image.path),
            "image_sha256": image_sha256,
            "image_format": image_format,
            "image_width": width,
            "image_height": height,
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "task": "instance-segmentation",
        },
        "inference": {
            "native_image_shape_hw": [height, width],
            "inference_shape_hw": [inference_height, inference_width],
            "stride": stride,
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "max_detections": MAX_DETECTIONS,
            "augment": False,
            "hazard_classes": list(HAZARD_CLASS_NAMES),
        },
        "candidate_count": len(candidates),
        "candidate_counts_by_class": dict(sorted(counts.items())),
        "candidates": candidates,
        "claim_boundary": {
            "candidate_materialization_only": True,
            "risk_metric": False,
            "evaluator_or_sidecar_opened": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Absolute path to one RGB image")
    source.add_argument("--image-index", type=Path, help="Absolute path to an explicit RGB image JSONL index")
    source.add_argument(
        "--c2-model-root",
        type=Path,
        help="Absolute path to a native C2 model root (alignment-explicit v2 or legacy v1)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Existing local YOLO segmentation checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--device", help="Ultralytics device selector, for example 0 or cpu")
    parser.add_argument(
        "--technical-smoke",
        action="store_true",
        help="Require exactly one image and mark all output TECHNICAL_SMOKE_NON_METRIC",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_explicit_file(args.model, label="model", reject_truth_paths=False)
    require(model_path.suffix.casefold() == ".pt", f"model_suffix:{model_path}")
    model_sha256 = sha256_file(model_path)
    c2_contract: C2ModelInputs | None = None
    if args.image is not None:
        inputs = single_image_input(args.image)
        image_index_sha256 = None
        input_mode = "single_image"
    elif args.image_index is not None:
        inputs, image_index_sha256 = load_image_index(args.image_index)
        input_mode = "jsonl_image_index"
    else:
        c2_contract = load_c2_model_root(args.c2_model_root)
        inputs = c2_contract.inputs
        image_index_sha256 = None
        input_mode = "c2_native_model_root"
    if args.technical_smoke:
        require(len(inputs) == 1, f"technical_smoke_requires_one_image:{len(inputs)}")
    run_kind = "TECHNICAL_SMOKE_NON_METRIC" if args.technical_smoke else "CANDIDATE_MATERIALIZATION"
    output_dir = create_output_directory(args.output_dir)

    prior_config = os.environ.get("YOLO_CONFIG_DIR")
    receipts: list[dict[str, Any]] = []
    total_candidates = 0
    class_counts: Counter[str] = Counter()
    try:
        with tempfile.TemporaryDirectory(prefix="blindassist-yolo-config-") as config_dir:
            os.environ["YOLO_CONFIG_DIR"] = config_dir
            from ultralytics import YOLO

            model = YOLO(str(model_path), task="segment")
            names = model_names(model)
            for ordinal, image in enumerate(inputs):
                result = infer_one(
                    model,
                    names,
                    image,
                    model_path=model_path,
                    model_sha256=model_sha256,
                    device=args.device,
                    run_kind=run_kind,
                )
                output_path = output_dir / f"{ordinal:08d}.json"
                write_json_exclusive(output_path, result)
                result_sha256 = sha256_file(output_path)
                receipts.append(
                    {
                        "ordinal": ordinal,
                        "frame_id": result["source"]["frame_id"],
                        **{
                            key: result["source"][key]
                            for key in (
                                "episode_id",
                                "sample_index",
                                "time_s",
                                "world_frame",
                                "world_frame_namespace",
                                "wearable_source_world_frame",
                                "depth_source_world_frame",
                            )
                            if key in result["source"]
                        },
                        "path": output_path.name,
                        "sha256": result_sha256,
                        "image_sha256": result["source"]["image_sha256"],
                        "candidate_count": result["candidate_count"],
                        "image_width": result["source"]["image_width"],
                        "image_height": result["source"]["image_height"],
                    }
                )
                total_candidates += int(result["candidate_count"])
                class_counts.update(result["candidate_counts_by_class"])
    finally:
        if prior_config is None:
            os.environ.pop("YOLO_CONFIG_DIR", None)
        else:
            os.environ["YOLO_CONFIG_DIR"] = prior_config

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "run_kind": run_kind,
        "input_mode": input_mode,
        "image_index_sha256": image_index_sha256,
        "c2_model_contract": (
            {
                "model_root": str(c2_contract.model_root),
                "manifest_path": str(c2_contract.manifest_path),
                "manifest_sha256": c2_contract.manifest_sha256,
                "schema_version": c2_contract.schema_version,
                "experiment_id": c2_contract.experiment_id,
                "episode_ids": c2_contract.episode_ids,
                "rgbd_alignment": c2_contract.rgbd_alignment,
            }
            if c2_contract is not None
            else None
        ),
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "ultralytics_version": importlib.metadata.version("ultralytics"),
        },
        "frame_count": len(receipts),
        "candidate_count": total_candidates,
        "candidate_counts_by_class": dict(sorted(class_counts.items())),
        "frames": receipts,
        "claim_boundary": {
            "candidate_materialization_only": True,
            "risk_metric": False,
            "technical_smoke_is_not_a_metric": bool(args.technical_smoke),
            "evaluator_or_sidecar_opened": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json_exclusive(manifest_path, manifest)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
