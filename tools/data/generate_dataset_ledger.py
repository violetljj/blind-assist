#!/usr/bin/env python3
"""Build a session-level inventory of local BlindAssist data assets.

The scanner is deliberately evidence-bound.  It inventories physical files and
metadata/manifest-only sources without treating a path name as proof that the
payload is usable.  Path and metadata role hints are retained as evidence and
are surfaced again in SOURCE_ROLE_CONFLICTS.md.

Default invocation (from the repository root):

    E:\\codex-tools\\bin\\blindassist-python.cmd \\
        tools\\data\\generate_dataset_ledger.py --output-dir artifacts.local/datasets/ledger

Generated files belong under ``artifacts.local`` and are not tracked. Junction
aliases inside the checkout are not followed, so local aliases are not
double-counted.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "dataset-master-ledger-v1"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_ROOT = DEFAULT_REPO_ROOT / "artifacts.local"
DEFAULT_OUTER_ROOT = DEFAULT_REPO_ROOT.parent / "artifacts.local"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
NUMERIC_EXTS = {".npy", ".npz", ".npyc", ".h5", ".hdf5", ".mat", ".bin", ".raw", ".pfm", ".exr"}
STRUCTURED_EXTS = {
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".tsv",
    ".parquet",
    ".txt",
    ".yaml",
    ".yml",
    ".xml",
    ".metadata",
    ".sample",
    ".lst",
    ".index",
    ".typed",
    ".bag",
    ".db3",
    ".pcd",
    ".lzf",
    ".pincam",
    ".traj",
}
ARCHIVE_EXTS = {".zip", ".7z", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
ASSET_EXTS = IMAGE_EXTS | VIDEO_EXTS | NUMERIC_EXTS | STRUCTURED_EXTS | ARCHIVE_EXTS

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    # Scanner staging outputs are evidence of the audit process, not source
    # data.  Exclude them so reruns remain idempotent.
    "dataset-master-ledger-smoke",
    "dataset-master-ledger-structural",
}

SPLIT_WORDS = {
    "train",
    "training",
    "test",
    "testing",
    "val",
    "valid",
    "validation",
    "dev",
    "development",
    "heldout",
    "holdout",
    "fresh",
    "reserved",
    "qualification",
    "rehearsal",
    "calibration",
    "canary",
    "replay",
    "source",
    "payload",
}

MODALITY_WORDS = {
    "rgb",
    "rgbd",
    "image",
    "images",
    "video",
    "videos",
    "frames",
    "video_frames",
    "rgb_frames",
    "source_images",
    "source_rgb",
    "source_video",
    "source_videos",
    "source-videos",
    "segmentation_masks",
    "semantic_masks",
    "source_masks",
    "masks",
    "mask",
    "labels",
    "label",
    "annotations",
    "annotation",
    "ground_truth",
    "ground-truth",
    "truth",
    "depth",
    "source_depth",
    "depth_frames",
    "pose",
    "poses",
    "trajectory",
    "trajectories",
    "odom",
    "odometry",
    "vicon",
    "imu",
    "pointclouds",
    "pointcloud",
    "upper_velodyne",
    "lower_velodyne",
    "image_stitched",
    "labels_2d_stitched",
    "labels_3d",
    "confidence",
    "lowres_depth",
    "lowres_wide",
    "lowres_wide_intrinsics",
}

GENERIC_MODALITY_WORDS = {
    "payload",
    "data",
    "datasets",
    "dataset",
    "raw",
    "processed",
    "source",
    "sources",
    "output",
    "outputs",
    "result",
    "results",
    "artifact",
    "artifacts",
    "evidence",
    "cache",
    "caches",
    "meta",
    "metadata",
    "annotations",
    "annotation",
    "labels",
    "label",
    "qa",
    "figures",
    "case_figures",
    "frames",
    "images",
    "rgb",
    "depth",
    "masks",
    "mask",
    "pose",
    "poses",
    "video",
    "videos",
    "confidence",
    "lowres_depth",
    "lowres_wide",
    "lowres_wide_intrinsics",
    "train",
    "test",
    "val",
    "dev",
    "replay",
    "split",
}

MASK_WORDS = {
    "mask",
    "masks",
    "segmentation",
    "semantic",
    "label",
    "labels",
    "annotation",
    "annotations",
    "truth",
    "ground_truth",
    "ground-truth",
    "source_masks",
    "semantic_masks",
}
DEPTH_WORDS = {"depth", "disparity", "source_depth", "depth_frames"}
CONFIDENCE_WORDS = {"confidence", "confidences", "depth_confidence"}
POSE_WORDS = {
    "pose",
    "poses",
    "trajectory",
    "trajectories",
    "odom",
    "odometry",
    "vicon",
    "imu",
    "camera_poses",
    "egomotion",
    "ego_motion",
}
POINTCLOUD_WORDS = {"pcd", "pointcloud", "pointclouds", "lidar", "velodyne", "laser"}

ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("consumed", re.compile(r"(?:^|[-_])consum(?:ed|ption)?(?:$|[-_])", re.I)),
    ("burned", re.compile(r"(?:^|[-_])burn(?:ed|t)?(?:$|[-_])", re.I)),
    ("fresh", re.compile(r"(?:^|[-_])fresh(?:$|[-_])", re.I)),
    ("reserved", re.compile(r"(?:^|[-_])reserv(?:ed|ation)?(?:$|[-_])", re.I)),
    ("replay", re.compile(r"(?:^|[-_])replay(?:$|[-_])", re.I)),
    ("canonical_input", re.compile(r"canonical(?:[-_])?(?:input|raw|view|dataset)", re.I)),
    ("event_eval", re.compile(r"event(?:[-_])?eval|event(?:[-_])?evaluation", re.I)),
    ("segmentation_520", re.compile(r"(?<!\d)520(?:[-_ ]?frame|[-_ ]?frames)?", re.I)),
    ("event_eval_1920", re.compile(r"(?<!\d)1[, _]?920(?:[-_ ]?frame|[-_ ]?frames)?", re.I)),
    ("segmentation", re.compile(r"segment(?:ation|ed|ing)?", re.I)),
    ("self_collected", re.compile(r"self(?:[-_ ])?collect|capture|crowdbot|matoaka", re.I)),
    ("development", re.compile(r"(?:^|[-_])dev(?:elopment)?(?:$|[-_])", re.I)),
    ("regression", re.compile(r"regression|smoke|repro|reproduction", re.I)),
    ("calibration", re.compile(r"calibration|canary|prescreen", re.I)),
    ("confirmation", re.compile(r"confirm(?:ation)?|formal", re.I)),
    ("train", re.compile(r"(?:^|[-_])train(?:ing)?(?:$|[-_])", re.I)),
    ("test", re.compile(r"(?:^|[-_])test(?:ing)?(?:$|[-_])", re.I)),
    ("heldout", re.compile(r"held[-_ ]?out|hold[-_ ]?out", re.I)),
]

DATASET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ARKitScenes", re.compile(r"arkit(?:[-_ ]?scenes?)?", re.I)),
    ("SANPO", re.compile(r"sanpo", re.I)),
    ("EgoWalk", re.compile(r"egowalk|ego[-_ ]?walk", re.I)),
    ("Bonn", re.compile(r"(?:^|[-_])bonn(?:$|[-_])|rgbd[-_]?bonn", re.I)),
    ("REveL", re.compile(r"revel", re.I)),
    ("JRDB", re.compile(r"jrdb", re.I)),
    ("Shiraz", re.compile(r"shiraz", re.I)),
    ("Shanghai", re.compile(r"shanghai", re.I)),
    ("CID-SIMS", re.compile(r"cid[-_]?sims", re.I)),
    ("TUM-RGBD", re.compile(r"tum|freiburg", re.I)),
    ("ETH3D", re.compile(r"eth3d", re.I)),
    ("TartanAir", re.compile(r"tartan|tartanair", re.I)),
    ("EVIMO", re.compile(r"evimo", re.I)),
    ("MVSEC", re.compile(r"mvsec", re.I)),
    ("PublicVideo", re.compile(r"public[-_ ]?video|source[-_ ]?videos?", re.I)),
    ("Synthetic", re.compile(r"synthetic|procedural|counterfactual|canary[-_ ]?fixture", re.I)),
    ("Replay", re.compile(r"replay|ustrf|dual[-_ ]?loop", re.I)),
    ("RCLE-composite", re.compile(r"rcle", re.I)),
]

REQUESTED_LABELS = [
    "SANPO",
    "EgoWalk",
    "Bonn",
    "REveL",
    "JRDB",
    "Shiraz",
    "Shanghai",
    "self_collected",
    "replay",
    "canonical_input",
    "segmentation_520",
    "event_eval_1920",
    "consumed",
    "burned",
    "fresh",
    "reserved",
]


def compact_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def lower_parts(path: Path) -> list[str]:
    return [part.lower().replace(" ", "_") for part in path.parts]


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def contains_word(parts: Iterable[str], words: set[str]) -> bool:
    for part in parts:
        normalized = part.lower().replace(" ", "_")
        if normalized in words:
            return True
        if any(token in normalized.split("_") for token in words if "_" not in token):
            return True
    return False


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return compact_path(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    try:
        return value.item()
    except Exception:
        return str(value)


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def parse_time_value(value: Any) -> tuple[float | None, str | None]:
    """Return relative/numeric seconds and a human-readable value when possible."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        try:
            value = float(text.replace(",", ""))
        except ValueError:
            for candidate in (text, text.replace("Z", "+00:00")):
                try:
                    parsed = dt.datetime.fromisoformat(candidate)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=dt.timezone.utc)
                    return parsed.timestamp(), parsed.isoformat(timespec="milliseconds")
                except ValueError:
                    pass
            return None, None
    number = parse_number(value)
    if number is None:
        return None, None
    if abs(number) >= 1e17:
        seconds = number / 1e9
    elif abs(number) >= 1e14:
        seconds = number / 1e6
    elif abs(number) >= 1e11:
        seconds = number / 1e3
    elif abs(number) >= 1e9:
        seconds = number
    else:
        seconds = number
    readable = None
    if abs(seconds) >= 1e9:
        try:
            readable = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).isoformat(timespec="milliseconds")
        except (OverflowError, OSError, ValueError):
            readable = None
    return seconds, readable


def parse_filename_frame_key(stem: str) -> str | None:
    # Dataset frame names frequently carry both a local sample id and source
    # frame id (e.g. 0045_000113). The last numeric token is the source key.
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    if re.fullmatch(r"\d+", stem):
        return str(int(stem))
    return str(int(matches[-1]))


def parse_filename_timestamp(stem: str) -> tuple[float | None, str | None]:
    patterns = [
        r"(?P<y>20\d{2})[_-](?P<m>\d{2})[_-](?P<d>\d{2})__(?P<h>\d{2})[_-](?P<mi>\d{2})[_-](?P<s>\d{2})",
        r"(?P<y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})[T _-](?P<h>\d{2})[:_-](?P<mi>\d{2})[:_-](?P<s>\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        try:
            parsed = dt.datetime(
                int(match.group("y")),
                int(match.group("m")),
                int(match.group("d")),
                int(match.group("h")),
                int(match.group("mi")),
                int(match.group("s")),
                tzinfo=dt.timezone.utc,
            )
        except ValueError:
            continue
        return parsed.timestamp(), parsed.isoformat(timespec="seconds")
    return None, None


def infer_dataset(path_text: str) -> str:
    for label, pattern in DATASET_PATTERNS:
        if pattern.search(path_text):
            return label
    return "Unknown"


def infer_roles(path_text: str, dataset: str) -> tuple[list[str], dict[str, bool | None]]:
    roles: set[str] = set()
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(path_text):
            roles.add(role)
    if dataset == "Replay" or re.search(r"replay|ustrf|dual[-_ ]?loop", path_text, re.I):
        roles.add("replay")
    if dataset == "Synthetic":
        roles.add("synthetic")
    if dataset == "SANPO" and re.search(r"real|canonical", path_text, re.I):
        roles.add("source_native_truth")
    roles_list = sorted(roles)
    flags: dict[str, bool | None] = {}
    for role in ("consumed", "burned", "fresh", "reserved"):
        flags[role] = True if role in roles else None
    return roles_list, flags


def infer_split(path_text: str) -> tuple[str, list[str]]:
    lower = path_text.lower().replace("\\", "/")
    tags: list[str] = []
    for word in sorted(SPLIT_WORDS):
        if re.search(rf"(?:^|[/_-]){re.escape(word)}(?:$|[/_-])", lower):
            tags.append(word)
    # Preserve the explicit four-way vocabulary used by the project.
    if any(tag in tags for tag in ("train", "training")) and any(tag in tags for tag in ("test", "testing")):
        primary = "mixed"
    elif any(tag in tags for tag in ("heldout", "holdout")):
        primary = "heldout"
    elif "test" in tags or "testing" in tags:
        primary = "test"
    elif "train" in tags or "training" in tags:
        primary = "train"
    elif "dev" in tags or "development" in tags:
        primary = "dev"
    elif "fresh" in tags:
        primary = "fresh"
    elif "reserved" in tags:
        primary = "reserved"
    elif "replay" in tags:
        primary = "replay"
    elif "calibration" in tags or "canary" in tags:
        primary = "calibration"
    else:
        primary = "unspecified"
    return primary, tags


def classify_file(path: Path) -> dict[str, str]:
    ext = path.suffix.lower()
    parts = lower_parts(path)
    text = " ".join(parts + [path.name.lower()])
    if ext in IMAGE_EXTS:
        if contains_word(parts, MASK_WORDS):
            modality = "mask"
        elif contains_word(parts, DEPTH_WORDS):
            modality = "depth"
        elif contains_word(parts, CONFIDENCE_WORDS):
            modality = "confidence"
        elif contains_word(parts, POSE_WORDS):
            modality = "pose"
        else:
            modality = "rgb"
        return {"asset_kind": "image", "modality": modality}
    if ext in VIDEO_EXTS:
        if contains_word(parts, DEPTH_WORDS) or re.search(r"(?:^|[-_.])depth(?:[-_.]|$)", path.stem.lower()):
            modality = "depth"
        elif contains_word(parts, MASK_WORDS):
            modality = "mask"
        else:
            modality = "rgb"
        return {"asset_kind": "video", "modality": modality}
    if ext in ARCHIVE_EXTS:
        return {"asset_kind": "archive", "modality": "archive"}
    if ext in NUMERIC_EXTS or ext in STRUCTURED_EXTS:
        if contains_word(parts, POINTCLOUD_WORDS) or ext == ".pcd":
            modality = "pointcloud"
        elif contains_word(parts, DEPTH_WORDS):
            modality = "depth"
        elif contains_word(parts, MASK_WORDS):
            modality = "mask"
        elif contains_word(parts, POSE_WORDS) or ext in {".bag", ".db3", ".traj"}:
            modality = "pose"
        elif contains_word(parts, {"rgb", "image", "images", "video", "frames"}):
            modality = "rgb"
        else:
            modality = "metadata"
        return {"asset_kind": "structured", "modality": modality}
    return {"asset_kind": "other", "modality": "unknown", "path_text": text}


def is_model_or_code_asset(path: Path) -> bool:
    text = str(path).lower().replace("\\", "/")
    if path.suffix.lower() in {".h5", ".hdf5", ".bin", ".npz", ".npy"}:
        if re.search(r"/models?/|/weights?/|checkpoint|\.tflite|mobilenet|pidnet|ddrnet|segformer|depth[-_]?anything", text):
            return True
    return False


def iter_asset_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for base, dirs, names in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        filtered: list[str] = []
        for dirname in dirs:
            if dirname in SKIP_DIR_NAMES:
                continue
            candidate = base_path / dirname
            try:
                if candidate.is_symlink():
                    continue
            except OSError:
                continue
            filtered.append(dirname)
        dirs[:] = filtered
        for name in names:
            path = base_path / name
            if path.suffix.lower() not in ASSET_EXTS:
                continue
            if is_model_or_code_asset(path):
                continue
            yield path


def modality_marker_index(rel_parts: tuple[str, ...]) -> int | None:
    candidates: list[int] = []
    for index, part in enumerate(rel_parts[:-1]):
        normalized = part.lower().replace(" ", "_")
        if normalized in MODALITY_WORDS:
            candidates.append(index)
    return candidates[-1] if candidates else None


def video_base(stem: str) -> str:
    return re.sub(r"(?:[_-](?:rgb|depth|mask|masks|video|frames))$", "", stem, flags=re.I)


def group_rel_for_file(path: Path, root: Path, classification: dict[str, str]) -> tuple[str, str | None, tuple[str, ...]]:
    rel = path.relative_to(root)
    parts = rel.parts
    lower = tuple(part.lower().replace(" ", "_") for part in parts)
    if classification["asset_kind"] == "video":
        video_index = next((i for i, part in enumerate(lower[:-1]) if part in {"video", "videos", "source-videos", "source_videos"}), None)
        if video_index is not None:
            anchor = "/".join(parts[:video_index])
            return f"{anchor}/video/{video_base(path.stem)}", path.stem, tuple(parts[:video_index])
        return f"file/{path.name}", path.stem, tuple(parts[:-1])

    marker_index = modality_marker_index(parts)
    if marker_index is not None:
        index = marker_index - 1
        while index >= 0 and lower[index] in SPLIT_WORDS:
            index -= 1
        while index >= 0 and lower[index] in GENERIC_MODALITY_WORDS:
            index -= 1
        if index < 0:
            index = max(0, marker_index - 1)
        group_parts = parts[: index + 1]
        return "/".join(group_parts), None, tuple(group_parts)

    parent_parts = list(parts[:-1])
    while parent_parts and parent_parts[-1].lower().replace(" ", "_") in {
        "data",
        "meta",
        "metadata",
        "raw",
        "processed",
        "outputs",
        "results",
        "source",
        "sources",
    }:
        parent_parts.pop()
    group_parts = tuple(parent_parts or parts[:-1])
    return "/".join(group_parts), None, group_parts


@dataclass
class FileCandidate:
    path: Path
    root_id: str
    root: Path
    rel_path: str
    classification: dict[str, str]
    group_rel: str
    video_base: str | None
    anchor: tuple[str, ...]


@dataclass
class Session:
    root_id: str
    root: Path
    group_rel: str
    files: list[FileCandidate] = field(default_factory=list)
    related_stems: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.root_id}:{self.group_rel}"


def discover_ffprobe() -> Path | None:
    candidates = [
        Path(r"E:\codex-tools\media\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe"),
        Path(r"E:\codex-tools\media\ffmpeg\bin\ffprobe.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def hash_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> tuple[str | None, str | None, str | None]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                sha.update(block)
                md5.update(block)
        return sha.hexdigest(), md5.hexdigest(), None
    except (OSError, IOError) as exc:
        return None, None, f"hash_error:{type(exc).__name__}:{exc}"


def probe_video(path: Path, ffprobe: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_evaluable", "width": None, "height": None, "fps": None, "frame_count": None, "duration_seconds": None}
    if ffprobe is not None:
        command = [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            if completed.returncode == 0:
                payload = json.loads(completed.stdout or "{}")
                stream = (payload.get("streams") or [{}])[0]
                result["width"] = int(stream.get("width")) if stream.get("width") else None
                result["height"] = int(stream.get("height")) if stream.get("height") else None
                rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                if isinstance(rate, str) and "/" in rate:
                    numerator, denominator = rate.split("/", 1)
                    try:
                        result["fps"] = float(numerator) / float(denominator) if float(denominator) else None
                    except (ValueError, ZeroDivisionError):
                        pass
                elif rate:
                    result["fps"] = parse_number(rate)
                frame_count = stream.get("nb_frames")
                result["frame_count"] = int(frame_count) if str(frame_count or "").isdigit() else None
                duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
                result["duration_seconds"] = parse_number(duration)
                result["status"] = "readable_probe" if result["width"] and result["height"] else "probe_incomplete"
                return result
            result["error"] = (completed.stderr or "ffprobe failed").strip()[-500:]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            result["error"] = f"{type(exc).__name__}:{exc}"
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            result["status"] = "corrupt_or_unreadable"
            return result
        result["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        result["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        result["fps"] = float(capture.get(cv2.CAP_PROP_FPS)) or None
        result["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        result["status"] = "readable_opencv"
        capture.release()
    except Exception as exc:  # pragma: no cover - dependency/runtime-specific
        result["status"] = "not_evaluable_dependency_or_codec"
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def profile_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            image.load()
            return {"status": "readable", "width": image.width, "height": image.height, "mode": image.mode}
    except Exception as exc:  # pragma: no cover - codec-specific
        return {"status": "corrupt_or_unreadable", "error": f"{type(exc).__name__}:{exc}"}


def profile_numeric(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            import numpy as np  # type: ignore

            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = [int(value) for value in array.shape]
            result: dict[str, Any] = {"status": "readable", "shape": shape, "dtype": str(array.dtype)}
            if len(shape) >= 2 and path.name.lower().find("depth") >= 0:
                result["width"] = shape[-1]
                result["height"] = shape[-2]
            return result
        if suffix == ".npz":
            import numpy as np  # type: ignore

            with np.load(path, allow_pickle=False) as archive:
                shapes = {key: [int(value) for value in archive[key].shape] for key in archive.files[:64]}
                return {"status": "readable", "arrays": shapes}
        if suffix in {".h5", ".hdf5"}:
            import h5py  # type: ignore

            with h5py.File(path, "r") as handle:
                datasets: dict[str, Any] = {}

                def visitor(name: str, obj: Any) -> None:
                    if len(datasets) >= 64:
                        return
                    if hasattr(obj, "shape"):
                        datasets[name] = {"shape": [int(value) for value in obj.shape], "dtype": str(obj.dtype)}

                handle.visititems(visitor)
                return {"status": "readable", "datasets": datasets}
    except ValueError as exc:  # pragma: no cover - format/runtime-specific
        message = str(exc)
        if "Python objects in dtype" in message or "allow_pickle" in message:
            return {"status": "not_evaluable_dependency_or_pickle", "error": f"ValueError:{message}"}
        return {"status": "corrupt_or_unreadable", "error": f"ValueError:{message}"}
    except Exception as exc:  # pragma: no cover - dependency/runtime-specific
        return {"status": "corrupt_or_unreadable", "error": f"{type(exc).__name__}:{exc}"}
    return {"status": "not_checked"}


def collect_time_values(obj: Any, key_hint: str, values: list[float], readable: list[str], limit: int = 5000) -> None:
    if len(values) >= limit:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in ("timestamp", "time_ns", "time_us", "time_ms", "capture_time", "frame_time")):
                if isinstance(value, list):
                    for item in value:
                        seconds, human = parse_time_value(item)
                        if seconds is not None:
                            values.append(seconds)
                            if human:
                                readable.append(human)
                else:
                    seconds, human = parse_time_value(value)
                    if seconds is not None:
                        values.append(seconds)
                        if human:
                            readable.append(human)
            collect_time_values(value, key_lower, values, readable, limit)
    elif isinstance(obj, list):
        for item in obj[:limit]:
            collect_time_values(item, key_hint, values, readable, limit)


def collect_declared_counts(obj: Any, result: dict[str, int], limit: int = 5000) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            number = parse_number(value)
            if number is not None and number.is_integer() and any(
                token in key_lower for token in ("count", "frames", "frame_count", "samples", "rows", "units", "predictions", "masks")
            ):
                if 0 <= number <= 100_000_000:
                    result[key_lower] = int(number)
            collect_declared_counts(value, result, limit)
    elif isinstance(obj, list) and len(obj) <= limit:
        for item in obj[:limit]:
            collect_declared_counts(item, result, limit)


def collect_declared_roles(obj: Any, result: set[str], limit: int = 5000) -> None:
    """Collect roles only when metadata presents them as a named field."""

    role_words = ("consumed", "burned", "fresh", "reserved")
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower().replace("-", "_")
            if "role" in key_lower or key_lower in {
                "usage",
                "use",
                "data_status",
                "source_status",
                "split",
                "source_class",
                "data_class",
            }:
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if isinstance(item, str):
                        normalized = item.lower()
                        for role in role_words:
                            if re.search(rf"(?<![a-z]){role}(?![a-z])", normalized):
                                result.add(role)
            collect_declared_roles(value, result, limit)
    elif isinstance(obj, list) and len(obj) <= limit:
        for item in obj[:limit]:
            collect_declared_roles(item, result, limit)


def profile_structured(path: Path, max_read_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    suffix = path.suffix.lower()
    result: dict[str, Any] = {"status": "readable", "declared_counts": {}}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"status": "not_readable", "error": f"{type(exc).__name__}:{exc}"}
    if suffix == ".parquet":
        try:
            import polars as pl  # type: ignore

            schema = pl.read_parquet_schema(path)
            result["columns"] = list(schema.keys())[:100]
            result["dtypes"] = {key: str(value) for key, value in list(schema.items())[:100]}
            timestamp_columns = [
                key for key in schema if any(token in key.lower() for token in ("timestamp", "time", "capture"))
            ][:3]
            if timestamp_columns:
                frame = pl.read_parquet(path, columns=timestamp_columns)
                values: list[float] = []
                readable: list[str] = []
                for column in timestamp_columns:
                    for item in frame[column].to_list()[:5000]:
                        seconds, human = parse_time_value(item)
                        if seconds is not None:
                            values.append(seconds)
                            if human:
                                readable.append(human)
                if values:
                    result["timestamp_values"] = values[:5000]
                    result["timestamp_readable"] = readable[:5000]
            try:
                result["row_count"] = int(pl.scan_parquet(path).select(pl.len()).collect().item())
            except Exception:
                pass
            return result
        except Exception as exc:  # pragma: no cover - dependency/runtime-specific
            return {"status": "not_evaluable_dependency_or_corrupt", "error": f"{type(exc).__name__}:{exc}"}
    if size > max_read_bytes:
        return {"status": "readable_not_profiled_size_limit", "bytes": size}
    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                payload = json.load(handle)
            collect_declared_counts(payload, result["declared_counts"])
            declared_roles: set[str] = set()
            collect_declared_roles(payload, declared_roles)
            if declared_roles:
                result["declared_roles"] = sorted(declared_roles)
            times: list[float] = []
            readable: list[str] = []
            collect_time_values(payload, "", times, readable)
            if times:
                result["timestamp_values"] = times[:5000]
                result["timestamp_readable"] = readable[:5000]
            return result
        if suffix in {".jsonl", ".ndjson"}:
            times: list[float] = []
            readable: list[str] = []
            line_count = 0
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    line_count += 1
                    if line_count <= 2000:
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            return {"status": "corrupt_or_unreadable", "error": f"invalid_json_line:{line_count}"}
                        collect_declared_counts(payload, result["declared_counts"])
                        declared_roles: set[str] = set(result.get("declared_roles") or [])
                        collect_declared_roles(payload, declared_roles)
                        if declared_roles:
                            result["declared_roles"] = sorted(declared_roles)
                        collect_time_values(payload, "", times, readable)
            result["row_count"] = line_count
            if times:
                result["timestamp_values"] = times[:5000]
                result["timestamp_readable"] = readable[:5000]
            return result
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                header = next(reader, [])
                result["columns"] = header[:100]
                timestamp_indices = [
                    i for i, value in enumerate(header) if any(token in value.lower() for token in ("timestamp", "time", "capture"))
                ][:3]
                row_count = 0
                times: list[float] = []
                readable: list[str] = []
                for row in reader:
                    row_count += 1
                    if row_count <= 5000:
                        for index in timestamp_indices:
                            if index < len(row):
                                seconds, human = parse_time_value(row[index])
                                if seconds is not None:
                                    times.append(seconds)
                                    if human:
                                        readable.append(human)
                result["row_count"] = row_count
                if times:
                    result["timestamp_values"] = times[:5000]
                    result["timestamp_readable"] = readable[:5000]
            return result
        if suffix in {".txt", ".yaml", ".yml", ".xml", ".metadata", ".sample", ".lst", ".index", ".typed"}:
            with path.open("rb") as handle:
                handle.read(4096)
            return result
        if suffix in {".bag", ".db3", ".pcd", ".lzf"}:
            if size == 0:
                return {"status": "empty"}
            return {"status": "readable_nonempty_not_decoded"}
    except (OSError, UnicodeError, ValueError) as exc:
        return {"status": "corrupt_or_unreadable", "error": f"{type(exc).__name__}:{exc}"}
    return result


def profile_file(path: Path, classification: dict[str, str], ffprobe: Path | None, decode: bool) -> dict[str, Any]:
    if not decode:
        return {"status": "not_checked"}
    asset_kind = classification["asset_kind"]
    if asset_kind == "image":
        return profile_image(path)
    if asset_kind == "video":
        return probe_video(path, ffprobe)
    if path.suffix.lower() in NUMERIC_EXTS:
        return profile_numeric(path)
    return profile_structured(path)


def metadata_role_hints(profile: dict[str, Any]) -> list[str]:
    roles: set[str] = set()
    declared = profile.get("declared_counts") or {}
    for key, value in declared.items():
        if "mask" in key and value >= 1000:
            roles.add("mask_inventory")
        if "frame" in key and value >= 500:
            roles.add("frame_inventory")
    return sorted(roles)


def make_session_digest(file_records: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not file_records or any(record.get("sha256") is None for record in file_records):
        return None, None
    lines = [
        f"{record.get('relative_path','')}\t{record.get('bytes',0)}\t{record.get('sha256','')}\t{record.get('md5','')}"
        for record in sorted(file_records, key=lambda item: item.get("relative_path", ""))
    ]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), hashlib.md5(payload).hexdigest()


def aggregate_timestamp_info(file_records: list[dict[str, Any]], session_files: list[FileCandidate]) -> dict[str, Any]:
    numeric: list[float] = []
    human: list[str] = []
    bases: set[str] = set()
    fps_candidates: list[float] = []
    durations: list[float] = []
    resolutions: set[tuple[int, int]] = set()
    frame_counts: dict[str, int] = {}
    ordered_candidates = sorted(session_files, key=lambda item: item.rel_path.lower())
    for record, candidate in zip(file_records, ordered_candidates):
        profile = record.get("profile") or {}
        stem_seconds, stem_human = parse_filename_timestamp(Path(record["path"]).stem)
        if stem_seconds is not None:
            numeric.append(stem_seconds)
            if stem_human:
                human.append(stem_human)
        if profile.get("timestamp_values"):
            numeric.extend(float(value) for value in profile["timestamp_values"] if parse_number(value) is not None)
        human.extend(str(value) for value in profile.get("timestamp_readable", []) if value)
        if profile.get("fps"):
            fps_candidates.append(float(profile["fps"]))
        if profile.get("duration_seconds"):
            durations.append(float(profile["duration_seconds"]))
        if profile.get("width") and profile.get("height"):
            resolutions.add((int(profile["width"]), int(profile["height"])))
        if profile.get("frame_count"):
            frame_counts[candidate.rel_path] = int(profile["frame_count"])
        frame_key = record.get("frame_key")
        if frame_key:
            bases.add(frame_key)
    if numeric:
        start_value, end_value = min(numeric), max(numeric)
        readable_start = min(human) if human else None
        readable_end = max(human) if human else None
        timestamp_range: dict[str, Any] = {
            "numeric_start": start_value,
            "numeric_end": end_value,
            "human_start": readable_start,
            "human_end": readable_end,
            "basis": "filename_or_metadata",
        }
    elif durations:
        timestamp_range = {
            "numeric_start": 0.0,
            "numeric_end": max(durations),
            "human_start": None,
            "human_end": None,
            "basis": "container_relative_seconds",
        }
    else:
        timestamp_range = {"numeric_start": None, "numeric_end": None, "human_start": None, "human_end": None, "basis": "not_evaluable"}
    fps = None
    fps_basis = "not_evaluable"
    if fps_candidates:
        fps = round(sum(fps_candidates) / len(fps_candidates), 9)
        fps_basis = "video_container"
    elif len(numeric) >= 3:
        differences = [b - a for a, b in zip(sorted(set(numeric)), sorted(set(numeric))[1:]) if b > a]
        if differences:
            median = sorted(differences)[len(differences) // 2]
            if median > 0:
                fps = round(1.0 / median, 9)
                fps_basis = "timestamp_delta"
    resolution_values = [{"width": width, "height": height} for width, height in sorted(resolutions)]
    return {
        "timestamp_range": timestamp_range,
        "fps": fps,
        "fps_basis": fps_basis,
        "resolution": {
            "values": resolution_values,
            "consistent": len(resolution_values) <= 1,
            "width": resolution_values[0]["width"] if len(resolution_values) == 1 else None,
            "height": resolution_values[0]["height"] if len(resolution_values) == 1 else None,
        },
        "video_frame_counts": frame_counts,
    }


def frame_key_summary(file_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, list[str]] = defaultdict(list)
    for record in file_records:
        key = record.get("frame_key")
        if key is not None:
            by_modality[record.get("modality", "unknown")].append(str(key))
    summary: dict[str, Any] = {}
    for modality, keys in by_modality.items():
        counts = Counter(keys)
        unique = sorted(set(keys), key=lambda value: int(value) if value.isdigit() else value)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        missing: list[str] = []
        missing_status = "not_evaluable"
        numeric = [int(key) for key in unique if key.isdigit()]
        if len(numeric) >= 3 and len(numeric) == len(unique):
            span = max(numeric) - min(numeric) + 1
            if span <= 200_000 and span <= len(numeric) * 20:
                missing = [str(index) for index in range(min(numeric), max(numeric) + 1) if str(index) not in counts]
                missing_status = "contiguous_numeric_sequence"
            else:
                missing_status = "noncontiguous_sampling_or_large_span"
        summary[modality] = {
            "file_key_count": len(keys),
            "unique_key_count": len(unique),
            "first_key": unique[0] if unique else None,
            "last_key": unique[-1] if unique else None,
            "duplicate_frame_keys": duplicates,
            "missing_frames": missing,
            "missing_status": missing_status,
        }
    return summary


def alignment_summary(file_records: list[dict[str, Any]]) -> dict[str, Any]:
    key_sets: dict[str, set[str]] = defaultdict(set)
    for record in file_records:
        if record.get("frame_key") is not None:
            key_sets[record.get("modality", "unknown")].add(str(record["frame_key"]))
    pairs: dict[str, Any] = {}
    for reference in ("rgb", "mask", "depth", "pose"):
        if reference not in key_sets:
            continue
        for other in ("mask", "depth", "pose"):
            if other not in key_sets or other == reference:
                continue
            left, right = key_sets[reference], key_sets[other]
            intersection = left & right
            pairs[f"{reference}_vs_{other}"] = {
                "reference_count": len(left),
                "other_count": len(right),
                "intersection_count": len(intersection),
                "reference_coverage": round(len(intersection) / len(left), 6) if left else None,
                "other_coverage": round(len(intersection) / len(right), 6) if right else None,
                "missing_in_other_sample": sorted(left - right, key=lambda value: int(value) if value.isdigit() else value)[:1000],
                "missing_in_reference_sample": sorted(right - left, key=lambda value: int(value) if value.isdigit() else value)[:1000],
            }
    if not key_sets:
        status = "not_evaluable_no_frame_keys"
    elif not pairs:
        status = "not_evaluable_single_modality_or_non_frame_pose"
    elif all(value["reference_coverage"] == 1.0 and value["other_coverage"] == 1.0 for value in pairs.values()):
        status = "aligned_by_frame_key"
    else:
        status = "partial_or_misaligned_by_frame_key"
    return {"status": status, "modalities_with_frame_keys": sorted(key_sets), "pairwise": pairs}


def supported_questions(dataset: str, modalities: set[str], roles: set[str], alignment: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    if "rgb" in modalities:
        questions.extend(["RGB continuity/source characterization", "detection or tracking development"])
    if "mask" in modalities:
        questions.extend(["pixel-level segmentation or mask label audit", "RGB-mask spatial alignment"])
    if "depth" in modalities:
        questions.extend(["depth availability/quality and RGB-depth alignment", "geometry or occlusion mechanism diagnostics"])
    if "pose" in modalities:
        questions.extend(["pose/trajectory availability and temporal alignment", "egomotion or motion-attribution diagnostics"])
    if dataset == "SANPO":
        questions.append("traversability/segmentation development or regression")
    if dataset == "EgoWalk":
        questions.append("RGB-depth temporal/source inventory for clearance research")
    if dataset == "Bonn" or dataset == "TUM-RGBD":
        questions.append("RGB-D/egomotion geometry interface diagnostics")
    if dataset == "REveL":
        questions.append("offline RGB/Vicon radial-motion attribution diagnostics")
    if dataset == "JRDB":
        questions.extend(["2D/3D person-label and point-cloud support audit", "RGB timestamp/calibration/pose authority diagnostics"])
    if dataset in {"Shiraz", "Shanghai"}:
        questions.append("cross-source replay/event-level mechanism comparison")
    if "replay" in roles:
        questions.append("runtime/replay regression and causal trace auditing")
    if "canonical_input" in roles:
        questions.append("hash-bound canonical input provenance/replay")
    if "event_eval" in roles or "event_eval_1920" in roles:
        questions.append("event-level evaluation input inventory")
    if alignment.get("status") in {"partial_or_misaligned_by_frame_key"}:
        questions.append("gap localization only; not an aligned multimodal effect claim")
    return list(dict.fromkeys(questions))


def read_roles_from_metadata(file_records: list[dict[str, Any]]) -> tuple[set[str], dict[str, bool | None]]:
    roles: set[str] = set()
    flags: dict[str, bool | None] = {role: None for role in ("consumed", "burned", "fresh", "reserved")}
    for record in file_records:
        path_text = f"{record.get('path','')} {record.get('relative_path','')}"
        role_hints, path_flags = infer_roles(path_text, "Unknown")
        roles.update(role_hints)
        profile = record.get("profile") or {}
        for role, value in path_flags.items():
            if value:
                flags[role] = True
        for role in metadata_role_hints(profile):
            roles.add(role)
        declared = profile.get("declared_roles") or []
        for role in declared:
            role_name = normalized_name(str(role))
            if "consum" in role_name:
                flags["consumed"] = True
                roles.add("consumed")
            if "burn" in role_name:
                flags["burned"] = True
                roles.add("burned")
            if "fresh" in role_name:
                flags["fresh"] = True
                roles.add("fresh")
            if "reserv" in role_name:
                flags["reserved"] = True
                roles.add("reserved")
    return roles, flags


def collect_declared_roles_from_text(path: Path, max_bytes: int = 1024 * 1024) -> list[str]:
    roles: list[str] = []
    try:
        if path.stat().st_size > max_bytes:
            return roles
        text = path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except OSError:
        return roles
    # Require an explicit role/use/split/status key nearby; a bare word in a
    # narrative report is not proof of the containing asset's role.
    key_pattern = r"(?:role|roles|source_role|data_role|historical_use_role|usage|use|data_status|source_status|split|source_class|data_class)"
    for role in ("consumed", "burned", "fresh", "reserved"):
        pattern = rf"(?:[\"']?{key_pattern}[\"']?\s*[:=].{{0,160}}\b{role}\b|\b{role}\b.{{0,160}}(?:[\"']?{key_pattern}[\"']?\s*[:=]))"
        if re.search(pattern, text, re.I | re.S):
            roles.append(role)
    return roles


def build_roots(repo_root: Path, include_outer: bool) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    if DEFAULT_CANONICAL_ROOT.exists():
        roots.append(("checkout_artifacts.local", DEFAULT_CANONICAL_ROOT))
    if include_outer and DEFAULT_OUTER_ROOT.exists() and DEFAULT_OUTER_ROOT.resolve() != DEFAULT_CANONICAL_ROOT.resolve():
        roots.append(("outer_legacy_artifacts.local", DEFAULT_OUTER_ROOT))
    return roots


def discover_candidates(roots: list[tuple[str, Path]], max_files: int | None = None) -> tuple[list[FileCandidate], dict[str, Any]]:
    candidates: list[FileCandidate] = []
    root_stats: dict[str, Any] = {}
    for root_id, root in roots:
        scanned = 0
        bytes_scanned = 0
        for path in iter_asset_files(root):
            scanned += 1
            try:
                stat = path.stat()
                bytes_scanned += stat.st_size
            except OSError:
                continue
            classification = classify_file(path)
            group_rel, video_stem, anchor = group_rel_for_file(path, root, classification)
            candidates.append(
                FileCandidate(
                    path=path,
                    root_id=root_id,
                    root=root,
                    rel_path=path.relative_to(root).as_posix(),
                    classification=classification,
                    group_rel=group_rel,
                    video_base=video_stem,
                    anchor=anchor,
                )
            )
            if max_files is not None and len(candidates) >= max_files:
                break
        root_stats[root_id] = {"root": compact_path(root), "candidate_file_count": scanned, "candidate_bytes": bytes_scanned}
        if max_files is not None and len(candidates) >= max_files:
            break

    # Attach metadata in sibling data/meta directories to matching HFTF-style
    # video sessions when the timestamp stem is identical.
    video_index: dict[tuple[str, tuple[str, ...], str], str] = {}
    for candidate in candidates:
        if candidate.classification["asset_kind"] == "video" and candidate.video_base:
            video_index[(candidate.root_id, candidate.anchor, video_base(candidate.video_base))] = candidate.group_rel
    for candidate in candidates:
        if candidate.classification["asset_kind"] == "video":
            continue
        if candidate.path.parent.name.lower() not in {"data", "meta", "metadata", "timestamps"}:
            continue
        match = video_index.get((candidate.root_id, candidate.anchor, video_base(candidate.path.stem)))
        if match:
            candidate.group_rel = match
    return candidates, root_stats


def build_sessions(candidates: list[FileCandidate]) -> list[Session]:
    sessions: dict[str, Session] = {}
    for candidate in candidates:
        key = f"{candidate.root_id}:{candidate.group_rel}"
        session = sessions.get(key)
        if session is None:
            session = Session(candidate.root_id, candidate.root, candidate.group_rel)
            sessions[key] = session
        session.files.append(candidate)
        if candidate.video_base:
            session.related_stems.add(candidate.video_base)
    return sorted(sessions.values(), key=lambda session: (session.root_id, session.group_rel.lower()))


def profile_session(session: Session, ffprobe: Path | None, do_hash: bool, decode: bool, progress: dict[str, int]) -> dict[str, Any]:
    file_records: list[dict[str, Any]] = []
    role_text: set[str] = set()
    modality_counts: Counter[str] = Counter()
    total_bytes = 0
    damaged: list[str] = []
    hash_errors: list[str] = []
    for index, candidate in enumerate(sorted(session.files, key=lambda item: item.rel_path.lower()), start=1):
        try:
            stat = candidate.path.stat()
            size = int(stat.st_size)
        except OSError as exc:
            record = {
                "path": compact_path(candidate.path),
                "relative_path": candidate.rel_path,
                "modality": candidate.classification["modality"],
                "asset_kind": candidate.classification["asset_kind"],
                "bytes": None,
                "sha256": None,
                "md5": None,
                "integrity_status": "not_readable",
                "integrity_error": f"{type(exc).__name__}:{exc}",
                "frame_key": parse_filename_frame_key(candidate.path.stem),
            }
            file_records.append(record)
            continue
        total_bytes += size
        sha256, md5, hash_error = hash_file(candidate.path) if do_hash else (None, None, None)
        profile = profile_file(candidate.path, candidate.classification, ffprobe, decode)
        integrity_status = profile.get("status", "not_checked")
        if integrity_status.startswith("corrupt") or integrity_status in {"not_readable", "empty"}:
            damaged.append(candidate.rel_path)
        if hash_error:
            hash_errors.append(candidate.rel_path)
        if decode and candidate.path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml"}:
            role_text.update(collect_declared_roles_from_text(candidate.path))
        modality = candidate.classification["modality"]
        modality_counts[modality] += 1
        record: dict[str, Any] = {
            "path": compact_path(candidate.path),
            "relative_path": candidate.rel_path,
            "extension": candidate.path.suffix.lower(),
            "modality": modality,
            "asset_kind": candidate.classification["asset_kind"],
            "bytes": size,
            "sha256": sha256,
            "md5": md5,
            "integrity_status": integrity_status,
            "integrity_error": profile.get("error"),
            "frame_key": parse_filename_frame_key(candidate.path.stem),
            "profile": profile,
        }
        file_records.append(record)
        progress["files_profiled"] += 1
        progress["bytes_profiled"] += size
        if progress["files_profiled"] % 500 == 0:
            print(
                f"[dataset-audit] files={progress['files_profiled']} bytes={progress['bytes_profiled']} sessions_done={progress['sessions_done']}",
                flush=True,
            )

    path_text = f"{session.root_id} {session.group_rel} " + " ".join(record["relative_path"] for record in file_records[:32])
    dataset = infer_dataset(path_text)
    history_roles, flags = infer_roles(path_text, dataset)
    metadata_roles, metadata_flags = read_roles_from_metadata(file_records)
    history_roles = sorted(set(history_roles) | metadata_roles | role_text)
    for role in role_text:
        if role in flags:
            flags[role] = True
    for role, value in metadata_flags.items():
        if value:
            flags[role] = True
            history_roles.append(role)
    history_roles = sorted(set(history_roles))
    split, split_tags = infer_split(path_text)
    modality_set = set(record["modality"] for record in file_records if record.get("modality") not in {"metadata", "archive", "unknown"})
    metadata_declared_counts: dict[str, int] = {}
    for record in file_records:
        for key, value in (record.get("profile") or {}).get("declared_counts", {}).items():
            metadata_declared_counts[key] = max(metadata_declared_counts.get(key, 0), int(value))
    for role, value in (("consumed", flags["consumed"]), ("burned", flags["burned"]), ("fresh", flags["fresh"]), ("reserved", flags["reserved"])):
        if value:
            history_roles.append(role)
    history_roles = sorted(set(history_roles))
    frame_summary = frame_key_summary(file_records)
    alignment = alignment_summary(file_records)
    temporal = aggregate_timestamp_info(file_records, session.files)
    session_sha256, session_md5 = make_session_digest(file_records)
    failed_decode_count = len(damaged)
    counts_files = {
        "rgb": modality_counts.get("rgb", 0),
        "mask": modality_counts.get("mask", 0),
        "depth": modality_counts.get("depth", 0),
        "pose": modality_counts.get("pose", 0),
    }
    counts_frames: dict[str, int | None] = {}
    for modality in ("rgb", "mask", "depth", "pose"):
        frame_counts = [
            (record.get("profile") or {}).get("frame_count")
            for record in file_records
            if record.get("modality") == modality and (record.get("profile") or {}).get("frame_count")
        ]
        counts_frames[modality] = sum(int(value) for value in frame_counts) if frame_counts else None
    counts = {
        "files": counts_files,
        "frames_or_records": counts_frames,
        "rgb_count": counts_frames["rgb"] if counts_frames["rgb"] is not None else counts_files["rgb"],
        "mask_count": counts_frames["mask"] if counts_frames["mask"] is not None else counts_files["mask"],
        "depth_count": counts_frames["depth"] if counts_frames["depth"] is not None else counts_files["depth"],
        "pose_count": counts_frames["pose"] if counts_frames["pose"] is not None else counts_files["pose"],
    }
    source_id_seed = f"{SCHEMA_VERSION}|{session.root_id}|{session.group_rel}".encode("utf-8")
    source_id = "SRC-" + hashlib.sha256(source_id_seed).hexdigest()[:20]
    media_types = sorted(set(record["asset_kind"] for record in file_records) | modality_set)
    return {
        "source_id": source_id,
        "session_key": session.key,
        "dataset": dataset,
        "split": split,
        "split_tags": split_tags,
        "session_kind": "media_session" if modality_set else "manifest_only",
        "media_types": media_types,
        "scan_root_id": session.root_id,
        "session_root": compact_path(session.root / session.group_rel.replace("/", os.sep)),
        "file_count": len(file_records),
        "file_size_bytes": total_bytes,
        "sha256": session_sha256,
        "md5": session_md5,
        "counts": counts,
        "timestamp_range": temporal["timestamp_range"],
        "fps": temporal["fps"],
        "fps_basis": temporal["fps_basis"],
        "resolution": temporal["resolution"],
        "video_frame_counts": temporal["video_frame_counts"],
        "missing_frames": frame_summary,
        "duplicate_frames": {
            modality: details["duplicate_frame_keys"] for modality, details in frame_summary.items() if details.get("duplicate_frame_keys")
        },
        "corrupt_frames": damaged,
        "hash_errors": hash_errors,
        "decodability": {
            "status": "all_profiled_readable" if not damaged and not hash_errors else "partial_or_failed",
            "corrupt_or_unreadable_count": failed_decode_count,
            "hash_error_count": len(hash_errors),
            "non_evaluable_dependency_count": sum(
                1 for record in file_records if str(record.get("integrity_status", "")).startswith("not_evaluable")
            ),
        },
        "rgb_mask_depth_pose_alignment": alignment,
        "history_roles": sorted(set(history_roles)),
        "role_flags": {
            "is_consumed": flags["consumed"],
            "is_burned": flags["burned"],
            "is_fresh": flags["fresh"],
            "is_reserved": flags["reserved"],
            "unknown_means_no_local_evidence_not_proof_of_absence": True,
        },
        "metadata_declared_counts": metadata_declared_counts,
        "research_questions_supported": supported_questions(dataset, modality_set, set(history_roles), alignment),
        "files": file_records,
    }


def add_global_duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for record in records:
        for file_record in record.get("files", []):
            digest = file_record.get("sha256")
            if digest:
                by_hash[digest].append((record["source_id"], file_record["relative_path"], record.get("dataset", "Unknown")))
    duplicates: list[dict[str, Any]] = []
    for digest, locations in sorted(by_hash.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(locations) < 2:
            continue
        duplicates.append(
            {
                "sha256": digest,
                "occurrence_count": len(locations),
                "source_ids": sorted(set(location[0] for location in locations)),
                "datasets": sorted(set(location[2] for location in locations)),
                "locations": [{"source_id": source_id, "relative_path": path} for source_id, path, _ in locations[:200]],
                "truncated_locations": len(locations) > 200,
            }
        )
    return duplicates


EXCLUSIVE_ROLE_PAIRS = (
    ("fresh", "consumed"),
    ("fresh", "burned"),
    ("reserved", "consumed"),
    ("reserved", "burned"),
)
EXCLUSIVE_ROLES = {role for pair in EXCLUSIVE_ROLE_PAIRS for role in pair}


def path_exclusive_roles(record: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    dataset = str(record.get("dataset") or "Unknown")
    for text in [record.get("session_root", "")] + [file_record.get("path", "") for file_record in record.get("files", [])]:
        hints, _ = infer_roles(str(text), dataset)
        roles.update(set(hints) & EXCLUSIVE_ROLES)
    return roles


def declared_exclusive_roles(file_record: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for value in (file_record.get("profile") or {}).get("declared_roles", []) or []:
        normalized = normalized_name(str(value))
        for role in EXCLUSIVE_ROLES:
            if role in normalized.split("_"):
                roles.add(role)
    return roles


def role_pairs_for(roles: set[str]) -> list[tuple[str, str]]:
    return [pair for pair in EXCLUSIVE_ROLE_PAIRS if pair[0] in roles and pair[1] in roles]


def build_role_conflicts(records: list[dict[str, Any]], duplicate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_session_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []
    record_effective_roles: dict[str, set[str]] = {}

    for record in records:
        path_roles = path_exclusive_roles(record)
        metadata_roles: set[str] = set()
        file_metadata_roles: dict[str, set[str]] = {}
        for file_record in record.get("files", []):
            declared = declared_exclusive_roles(file_record)
            metadata_roles.update(declared)
            file_metadata_roles[file_record.get("relative_path", "")] = declared

        # Roles present only in history_roles may come from YAML/text metadata;
        # retain them for a mixed-package finding, but do not propagate a
        # multi-role narrative to every raw file as if it were a file role.
        history_roles = set(record.get("history_roles", [])) & EXCLUSIVE_ROLES
        metadata_roles.update(history_roles - path_roles)
        mixed_roles = path_roles | metadata_roles
        mixed_pairs = role_pairs_for(mixed_roles)
        if mixed_pairs:
            conflicts.append(
                {
                    "conflict_type": "single_session_mixed_exclusive_role_evidence",
                    "severity": "medium",
                    "source_ids": [record["source_id"]],
                    "roles": sorted(mixed_roles),
                    "role_pairs": [list(pair) for pair in mixed_pairs],
                    "locations": [record.get("session_root")],
                    "role_sources": {
                        "path_roles": sorted(path_roles),
                        "metadata_roles": sorted(metadata_roles),
                    },
                    "action": "HOLD_ROLE_REVIEW",
                }
            )

        effective_roles = set(path_roles)
        for declared in file_metadata_roles.values():
            # A file with one explicit role can participate in a content-level
            # comparison. A file with both roles is a mixed metadata package,
            # not proof that its raw bytes carry both historical roles.
            if len(declared) == 1:
                effective_roles.update(declared)
        record_effective_roles[record["source_id"]] = effective_roles
        if record.get("sha256"):
            by_session_hash[record["sha256"]].append(record)
        for file_record in record.get("files", []):
            digest = file_record.get("sha256")
            if not digest:
                continue
            file_roles = set(path_roles)
            declared = file_metadata_roles.get(file_record.get("relative_path", ""), set())
            if len(declared) == 1:
                file_roles.update(declared)
            by_file_hash[digest].append(
                {
                    "source_id": record["source_id"],
                    "dataset": record.get("dataset"),
                    "roles": sorted(file_roles),
                    "path": file_record.get("path"),
                    "relative_path": file_record.get("relative_path"),
                }
            )

    for digest, locations in by_file_hash.items():
        # One location with mixed metadata is handled above; duplicate-content
        # conflicts require at least two physical locations.
        if len(locations) < 2:
            continue
        role_union = set(role for location in locations for role in location["roles"])
        conflict_pairs = role_pairs_for(role_union)
        if conflict_pairs:
            conflicts.append(
                {
                    "conflict_type": "same_file_content_multiple_exclusive_roles",
                    "severity": "high",
                    "sha256": digest,
                    "role_pairs": [list(pair) for pair in conflict_pairs],
                    "locations": locations[:100],
                    "truncated_locations": len(locations) > 100,
                    "action": "HOLD_ROLE_REVIEW",
                }
            )

    # A duplicated session manifest can carry the role conflict even when the
    # raw payload was not copied. This check requires at least two locations;
    # a single mixed package is reported separately above.
    for digest, locations in by_session_hash.items():
        if len(locations) < 2:
            continue
        role_union = set(role for record in locations for role in record_effective_roles.get(record["source_id"], set()))
        conflict_pairs = role_pairs_for(role_union)
        if conflict_pairs:
            conflicts.append(
                {
                    "conflict_type": "same_session_manifest_multiple_exclusive_roles",
                    "severity": "high",
                    "session_sha256": digest,
                    "source_ids": [record["source_id"] for record in locations],
                    "roles": sorted(role_union),
                    "role_pairs": [list(pair) for pair in conflict_pairs],
                    "locations": [record.get("session_root") for record in locations],
                    "action": "HOLD_ROLE_REVIEW",
                }
            )
    return conflicts


def build_gap_document(
    records: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    root_stats: dict[str, Any],
    roots: list[tuple[str, Path]],
    generated_at: str,
) -> str:
    total_files = sum(record.get("file_count", 0) for record in records)
    total_bytes = sum(record.get("file_size_bytes", 0) or 0 for record in records)
    media_sessions = [record for record in records if record.get("session_kind") == "media_session"]
    corrupt = [record for record in records if record.get("corrupt_frames")]
    partial_alignment = [
        record for record in records if record.get("rgb_mask_depth_pose_alignment", {}).get("status") == "partial_or_misaligned_by_frame_key"
    ]
    unknown_time = [record for record in records if record.get("timestamp_range", {}).get("basis") == "not_evaluable"]
    unknown_fps = [record for record in records if record.get("fps") is None]
    integrity_statuses: Counter[str] = Counter()
    hash_snapshot_mismatch_files = 0
    missing_frame_keys = 0
    duplicate_frame_keys = 0
    for record in records:
        hash_snapshot_mismatch_files += len(record.get("hash_snapshot_mismatches", []))
        for details in (record.get("missing_frames") or {}).values():
            missing_frame_keys += len(details.get("missing_frames", []))
        for details in (record.get("missing_frames") or {}).values():
            duplicate_frame_keys += len(details.get("duplicate_frame_keys", []))
        for file_record in record.get("files", []):
            integrity_statuses[str(file_record.get("integrity_status", "not_evaluable"))] += 1
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record.get("dataset", "Unknown")].append(record)
    sanpo_splits = Counter(record.get("split", "unspecified") for record in by_dataset.get("SANPO", []))

    lines = [
        "# DATASET_GAPS",
        "",
        f"生成时间：`{generated_at}`",
        f"扫描器 schema：`{SCHEMA_VERSION}`",
        "",
        "## 结论",
        "",
        f"本次扫描发现 `{len(records)}` 条 session/package 记录，其中 `{len(media_sessions)}` 条有可定位媒体，"
        f"`{len(records) - len(media_sessions)}` 条为 manifest/structured-only；物理文件 `{total_files}` 个，"
        f"累计字节 `{total_bytes:,}`。`consumed/burned` 只表示历史使用证据，不等于全局封存；"
        "`fresh/reserved` 只有本地路径或 metadata 明确出现时才标记，未知不被解释为不存在。",
        "",
        "## 扫描根与重复别名",
        "",
    ]
    for root_id, root in roots:
        lines.append(f"- `{root_id}`：`{compact_path(root)}`")
    lines.extend(
        [
            "- checkout 内 `.downloads` 与 `test-artifacts.local` 是 junction alias，未作为独立 root 重复扫描。",
            "- `E:\\linnan\\artifacts.local` 是存在的 legacy outer root，本次作为独立可访问 root 扫描；它与 checkout root 的内容重复由 SHA-256 duplicate groups 标出。",
            "- 仓库 source/config/docs/scripts/build 等工程文件不作为数据 payload 计入；其中的资产说明只有在 artifacts.local 扫描根内物化为 manifest/metadata 时才进入 ledger。",
            f"- 文件发现快照：`{generated_at}`；profile 修复时间见 ledger 的 `profile_refreshed_at`；发现后消失的路径保留为 `not_readable`。",
            "",
            "## 请求数据集覆盖",
            "",
            "| 请求项 | session/package 数 | 有媒体 | manifest-only | 典型状态 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for label in REQUESTED_LABELS:
        selected = [record for record in records if label in set(record.get("history_roles", [])) or record.get("dataset") == label]
        if label == "self_collected":
            selected = [record for record in records if "self_collected" in set(record.get("history_roles", []))]
        if label == "replay":
            selected = [record for record in records if "replay" in set(record.get("history_roles", []))]
        media_count = sum(record.get("session_kind") == "media_session" for record in selected)
        manifest_count = len(selected) - media_count
        status = "FOUND" if selected else "NOT_FOUND_IN_SCANNED_ROOTS"
        if selected and media_count == 0:
            status = "MANIFEST_ONLY_NO_PHYSICAL_MEDIA"
        lines.append(f"| `{label}` | {len(selected)} | {media_count} | {manifest_count} | {status} |")
    lines.append(
        "- SANPO split inference："
        + ", ".join(f"`{split}={count}`" for split, count in sorted(sanpo_splits.items()))
        + "；`unspecified` 表示路径/manifest 未给出可审计 split，不等于 train/test 缺失。"
    )
    lines.extend(["", "## 质量缺口与风险", ""])
    lines.append(f"- 损坏/不可解码 session：`{len(corrupt)}`；示例 source_id：{', '.join(record['source_id'] for record in corrupt[:12]) or '无'}。")
    lines.append(
        "- 文件级 profile 状态："
        f"`readable={integrity_statuses.get('readable', 0)}`，"
        f"`readable_probe={integrity_statuses.get('readable_probe', 0)}`，"
        f"`corrupt_or_unreadable={integrity_statuses.get('corrupt_or_unreadable', 0)}`，"
        f"`not_readable={integrity_statuses.get('not_readable', 0)}`，"
        f"`not_evaluable={sum(value for key, value in integrity_statuses.items() if key.startswith('not_evaluable'))}`，"
        f"`not_checked={integrity_statuses.get('not_checked', 0)}`。`not_readable` 表示发现后路径消失/权限失败，不等于内容损坏。"
    )
    lines.append(f"- 文件名/frame-key 级缺口：识别到 missing frame keys `{missing_frame_keys}`，duplicate frame keys `{duplicate_frame_keys}`；无法建立 frame key 的记录保持 `not_evaluable`。")
    lines.append(f"- 哈希快照与重做 profile 的文件大小不一致：`{hash_snapshot_mismatch_files}`；非零时应重新哈希后再把 profile 与 hash 作为同一快照使用。")
    lines.append(f"- RGB-mask-depth-pose frame-key 对齐为 partial/misaligned：`{len(partial_alignment)}` 条。未建立 frame key 的 pose/metadata 不会被判定为对齐。")
    lines.append(f"- 没有可解析 timestamp：`{len(unknown_time)}` 条；没有可解析 fps：`{len(unknown_fps)}` 条。很多 image sequence 只有 frame index，不能从文件名安全推导真实时间。")
    lines.append(f"- 发现内容重复组：`{len(duplicate_groups)}`；其中涉及排他角色冲突：`{len(conflicts)}`。重复内容不自动合并，因为不同 evidence role 仍需保留。")
    lines.append("- `.bag`/`.db3`/部分点云仅做非空/结构级检查；如果没有可用 rosbags/codec，报告会保留 `not_evaluable_dependency_or_codec`，不写成可解码。")
    lines.append("- archive (`zip/tar/gz/7z`) 只记录容器大小、hash 和非空状态，未擅自解压；archive 内部 session 需在后续单独 materialize 后再补扫。")
    lines.append("- 角色主要来自路径 token 与同目录 JSON/JSONL metadata；这不是对历史研究文档的语义重判。相同内容在 fresh/reserved 与 consumed/burned 目录出现时必须先 HOLD。")
    lines.extend(["", "## 研究问题边界", "", "- RGB-only 资产最多支持连续性、检测/跟踪 Development 或 replay regression；不能单凭 RGB 证明 obstacle truth、pose、TTC 或安全。", "- mask/depth/pose 的可支持问题取决于 frame-key、timestamp 和解码状态；存在对齐 gap 时，ledger 只支持 gap localization。", "- `consumed/burned` 资产仍可用于 Development、回归、诊断或机制解释，但不能被重新称为 fresh/unseen/independent；历史协议终态保持不可变。", "- 本文档是资产完整性/gap 报告，不授予新实验、Confirmation、Android、默认 App 或生产权限。", ""])
    return "\n".join(lines).rstrip("\n") + "\n"


def build_conflict_document(conflicts: list[dict[str, Any]], duplicate_groups: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# SOURCE_ROLE_CONFLICTS",
        "",
        f"生成时间：`{generated_at}`",
        "",
        "本文件只报告自动发现的角色矛盾；不会替用户裁决历史证据。`fresh/reserved` 与 `consumed/burned` 同时出现时，后续使用规则为 `HOLD_ROLE_REVIEW`。路径推断是线索，不是更强的 authority。",
        "",
        f"- 内容重复组：`{len(duplicate_groups)}`",
        f"- 自动发现排他角色冲突：`{len(conflicts)}`",
        "",
    ]
    if not conflicts:
        lines.extend(["## 结果", "", "未发现同一 SHA-256 文件内容或同一 session manifest 同时携带排他角色的记录。", "", "注意：未知角色不是无冲突证明；只有本地扫描到的角色证据才参与本报告。", ""])
        return "\n".join(lines)
    lines.extend(["## 冲突清单", "", "| # | 严重性 | 类型 | 角色对 | SHA/session digest | source/location | 处置 |", "|---:|---|---|---|---|---|---|"])
    for index, conflict in enumerate(conflicts, start=1):
        pairs = ", ".join("/".join(pair) for pair in conflict.get("role_pairs", [])) or ", ".join(conflict.get("roles", []))
        digest = conflict.get("sha256") or conflict.get("session_sha256") or ""
        locations = conflict.get("locations") or []
        if locations and isinstance(locations[0], dict):
            location_text = "<br>".join(f"{item.get('source_id')}:{item.get('relative_path')}" for item in locations[:4])
        else:
            location_text = "<br>".join(str(item) for item in locations[:4])
        if conflict.get("truncated_locations"):
            location_text += "<br>...truncated"
        lines.append(f"| {index} | `{conflict.get('severity','high')}` | `{conflict.get('conflict_type')}` | `{pairs}` | `{digest}` | {location_text} | `{conflict.get('action','HOLD_ROLE_REVIEW')}` |")
    lines.extend(["", "## 解释规则", "", "- 同一 raw frame/content 的物理复制可以合理存在于 regression、rehearsal、Development 和历史 formal 输出中；冲突报告不删除复制，只阻止把它们混称为 fresh。", "- `single_session_mixed_exclusive_role_evidence` 表示一个 package 的 metadata 同时出现排他角色；它不是同一 raw file 已被两次独立使用的证明，但必须先 HOLD。", "- `reserved` 与 `consumed` 的冲突尤其需要检查 reservation receipt、source lock 和实际 first-open 记录；本扫描器不会从目录名推断谁优先。", "- 角色没有冲突不代表研究问题可评价；请同时查看 `DATASET_GAPS.md` 的解码、时间戳和对齐缺口。", ""])
    return "\n".join(lines)


def write_outputs(output_dir: Path, ledger: dict[str, Any], gap_text: str, conflict_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "DATASET_MASTER_LEDGER.json"
    csv_path = output_dir / "DATASET_MASTER_LEDGER.csv"
    gaps_path = output_dir / "DATASET_GAPS.md"
    conflicts_path = output_dir / "SOURCE_ROLE_CONFLICTS.md"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(ledger), handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    columns = [
        "source_id",
        "dataset",
        "split",
        "session_kind",
        "session_root",
        "media_types",
        "rgb_count",
        "mask_count",
        "depth_count",
        "pose_count",
        "file_count",
        "file_size_bytes",
        "sha256",
        "md5",
        "timestamp_start",
        "timestamp_end",
        "timestamp_basis",
        "fps",
        "fps_basis",
        "resolution_width",
        "resolution_height",
        "resolution_consistent",
        "missing_frame_status",
        "missing_frame_count",
        "duplicate_frame_count",
        "corrupt_frame_count",
        "decodability_status",
        "alignment_status",
        "history_roles",
        "is_consumed",
        "is_burned",
        "is_fresh",
        "is_reserved",
        "research_questions_supported",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in ledger.get("sessions", []):
            timestamp = record.get("timestamp_range", {})
            resolution = record.get("resolution", {})
            missing = record.get("missing_frames", {})
            missing_count = sum(len(value.get("missing_frames", [])) for value in missing.values())
            missing_status = ";".join(f"{key}:{value.get('missing_status')}" for key, value in sorted(missing.items()))
            duplicate_count = sum(len(value) for value in record.get("duplicate_frames", {}).values())
            writer.writerow(
                {
                    "source_id": record.get("source_id"),
                    "dataset": record.get("dataset"),
                    "split": record.get("split"),
                    "session_kind": record.get("session_kind"),
                    "session_root": record.get("session_root"),
                    "media_types": ";".join(record.get("media_types", [])),
                    "rgb_count": record.get("counts", {}).get("rgb_count"),
                    "mask_count": record.get("counts", {}).get("mask_count"),
                    "depth_count": record.get("counts", {}).get("depth_count"),
                    "pose_count": record.get("counts", {}).get("pose_count"),
                    "file_count": record.get("file_count"),
                    "file_size_bytes": record.get("file_size_bytes"),
                    "sha256": record.get("sha256"),
                    "md5": record.get("md5"),
                    "timestamp_start": timestamp.get("human_start") or timestamp.get("numeric_start"),
                    "timestamp_end": timestamp.get("human_end") or timestamp.get("numeric_end"),
                    "timestamp_basis": timestamp.get("basis"),
                    "fps": record.get("fps"),
                    "fps_basis": record.get("fps_basis"),
                    "resolution_width": resolution.get("width"),
                    "resolution_height": resolution.get("height"),
                    "resolution_consistent": resolution.get("consistent"),
                    "missing_frame_status": missing_status,
                    "missing_frame_count": missing_count,
                    "duplicate_frame_count": duplicate_count,
                    "corrupt_frame_count": len(record.get("corrupt_frames", [])),
                    "decodability_status": record.get("decodability", {}).get("status"),
                    "alignment_status": record.get("rgb_mask_depth_pose_alignment", {}).get("status"),
                    "history_roles": ";".join(record.get("history_roles", [])),
                    "is_consumed": record.get("role_flags", {}).get("is_consumed"),
                    "is_burned": record.get("role_flags", {}).get("is_burned"),
                    "is_fresh": record.get("role_flags", {}).get("is_fresh"),
                    "is_reserved": record.get("role_flags", {}).get("is_reserved"),
                    "research_questions_supported": ";".join(record.get("research_questions_supported", [])),
                }
            )
    gaps_path.write_text(gap_text, encoding="utf-8", newline="\n")
    conflicts_path.write_text(conflict_text, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_REPO_ROOT / "artifacts.local" / "datasets" / "ledger",
    )
    parser.add_argument("--no-outer-root", action="store_true", help="Do not scan E:\\linnan\\artifacts.local.")
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA-256/MD5. Intended only for a fast dry run.")
    parser.add_argument("--no-decode", action="store_true", help="Skip image/video/structured decode checks.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit candidate files for a bounded smoke run.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit sessions for a bounded smoke run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    roots = build_roots(repo_root, include_outer=not args.no_outer_root)
    if not roots:
        print("No scan roots found", file=sys.stderr)
        return 2
    started = time.time()
    generated_at = iso_now()
    print("[dataset-audit] discovering asset files", flush=True)
    candidates, root_stats = discover_candidates(roots, max_files=args.max_files)
    sessions = build_sessions(candidates)
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    print(f"[dataset-audit] candidates={len(candidates)} sessions={len(sessions)} roots={len(roots)}", flush=True)
    ffprobe = discover_ffprobe()
    progress = {"files_profiled": 0, "bytes_profiled": 0, "sessions_done": 0}
    records: list[dict[str, Any]] = []
    for session in sessions:
        record = profile_session(session, ffprobe, do_hash=not args.no_hash, decode=not args.no_decode, progress=progress)
        records.append(record)
        progress["sessions_done"] += 1
        if progress["sessions_done"] % 25 == 0:
            print(f"[dataset-audit] session {progress['sessions_done']}/{len(sessions)} source_id={record['source_id']}", flush=True)
    records.sort(key=lambda record: record["source_id"])
    duplicate_groups = add_global_duplicate_groups(records)
    conflicts = build_role_conflicts(records, duplicate_groups)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "repository_root": compact_path(repo_root),
        "scan_policy": {
            "hash_algorithms": ["sha256", "md5"] if not args.no_hash else [],
            "decode_checks": not args.no_decode,
            "junction_aliases_not_followed": True,
            "unknown_role_is_not_absence": True,
            "session_grain": "physical media grouping or metadata/manifest package boundary",
            "excluded_directory_names": sorted(SKIP_DIR_NAMES),
        },
        "scan_roots": [{"root_id": root_id, "path": compact_path(root)} for root_id, root in roots],
        "root_stats": root_stats,
        "summary": {
            "session_count": len(records),
            "media_session_count": sum(record.get("session_kind") == "media_session" for record in records),
            "manifest_only_count": sum(record.get("session_kind") == "manifest_only" for record in records),
            "file_count": sum(record.get("file_count", 0) for record in records),
            "file_size_bytes": sum(record.get("file_size_bytes", 0) or 0 for record in records),
            "corrupt_session_count": sum(bool(record.get("corrupt_frames")) for record in records),
            "duplicate_content_group_count": len(duplicate_groups),
            "role_conflict_count": len(conflicts),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "duplicate_content_groups": duplicate_groups,
        "role_conflicts": conflicts,
        "sessions": records,
    }
    gap_text = build_gap_document(records, duplicate_groups, conflicts, root_stats, roots, generated_at)
    conflict_text = build_conflict_document(conflicts, duplicate_groups, generated_at)
    write_outputs(args.output_dir.resolve(), ledger, gap_text, conflict_text)
    print(
        f"[dataset-audit] complete sessions={len(records)} files={ledger['summary']['file_count']} "
        f"bytes={ledger['summary']['file_size_bytes']} elapsed={ledger['summary']['elapsed_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
