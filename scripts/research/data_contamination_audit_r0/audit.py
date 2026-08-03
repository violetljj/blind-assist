#!/usr/bin/env python3
"""Evidence-bound duplicate and contamination audit for BlindAssist data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import numpy as np
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
except Exception as exc:
    np = None
    Image = None
    IMAGE_IMPORT_ERROR = repr(exc)
else:
    IMAGE_IMPORT_ERROR = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
META_EXTS = {".json", ".jsonl", ".csv", ".tsv"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
DERIVED_TOKENS = {
    "contact_sheets", "contact_sheet", "case_figures", "figures", "plots",
    "plot", "preview", "previews", "screenshots", "visualizations",
    "visualization", "rendered", "render",
}
MASK_TOKENS = {"mask", "masks", "semantic_masks", "source_masks", "oracle_masks", "raw_masks"}
DEPTH_TOKENS = {"depth", "source_depth", "disparity", "disparities"}
ROLE_ORDER = [
    "train", "dev", "test", "event_eval", "fixed_regression", "old_cohort",
    "new_cohort", "model_train", "model_selection", "official_test",
    "reserved_test", "calibration", "development", "diagnostic", "unknown",
]
ROLE_ALIASES = {
    "training": "train", "validation": "dev", "valid": "dev", "val": "dev",
    "blind": "old_cohort", "holdout": "test", "event-eval": "event_eval",
    "event_eval": "event_eval", "official-test": "official_test",
    "official_test": "official_test", "model-selection": "model_selection",
    "model_selection": "model_selection", "model-train": "model_train",
    "model_train": "model_train",
}
IDENTITY_KEYS = {
    "session_id", "source_session_id", "sequence_id", "source_sequence_id",
    "parent_event_id", "event_id", "event_candidate_id", "risk_event_id",
    "source_ancestry_id", "ancestry_id", "source_id", "frame_index",
    "source_frame_index", "timestamp_ms", "source_timestamp_ms",
    "source_capture_timestamp_ns", "split", "role", "data_role",
    "video_path", "source_video_path",
}
HASH_KEYS = {
    "sha256", "image_sha256", "rgb_sha256", "source_rgb_sha256",
    "file_sha256", "original_sha256", "source_mask_sha256",
    "semantic_mask_sha256", "canonical_mask_sha256",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def dump_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dump_image_feature_cache(path: Path, images: list[dict[str, Any]], files: list[dict[str, Any]]) -> None:
    """Persist the expensive decoded-image stage so bounded pHash retries are resumable."""
    by_path = {str(rec["path"].resolve()): rec for rec in files}
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = []
    for image in images:
        row = {key: value for key, value in image.items() if key != "path"}
        source = by_path.get(str(image["path"].resolve()), {})
        row["file_sha256"] = source.get("file_sha256", "")
        rows.append(row)
    dump_jsonl(temporary, rows)
    temporary.replace(path)


def load_image_feature_cache(path: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    by_rel = {rec["rel_path"]: rec for rec in files}
    loaded: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rel_path = row.get("rel_path", "")
                source = by_rel.get(rel_path)
                if source is None or row.get("file_sha256", "") != source.get("file_sha256", ""):
                    return None
                row["path"] = source["path"]
                loaded.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    expected = sum(rec["extension"] in IMAGE_EXTS for rec in files)
    if len(loaded) != expected:
        return None
    return loaded


def norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def flat_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def relpath(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def tokens(path: Path) -> set[str]:
    result: set[str] = set()
    for part in path.parts:
        result.add(part.lower())
        result.update(token for token in re.split(r"[^a-zA-Z0-9]+", part.lower()) if token)
    return result


def classify(path: Path) -> tuple[str, str]:
    """Return asset class and pixel domain."""
    lower = path.as_posix().lower()
    parts = tokens(path)
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video", "video"
    if ext not in IMAGE_EXTS:
        if "vendor" in parts:
            return "vendor_code_or_metadata", "other"
        if "models" in parts:
            return "model_or_runtime", "other"
        return "file", "other"
    if "vendor" in parts:
        return "vendor_image", "other"
    if "models" in parts:
        return "model_image", "other"
    if parts & DERIVED_TOKENS or any(item in lower for item in ("contact_sheet", "case_figure", "plot_")):
        return "derived_visual", "other"
    if parts & MASK_TOKENS or "mask" in lower:
        return "mask", "mask"
    if parts & DEPTH_TOKENS or "disparity" in lower:
        return "depth", "depth"
    if "label" in parts or "annotation" in parts:
        return "label_image", "other"
    if "images" in parts or "video_frames" in parts or "rgb" in parts or "frame" in parts:
        return "rgb_candidate", "rgb"
    return "image_unknown", "rgb"


def iter_files(root: Path, output: Path) -> Iterator[Path]:
    output = output.resolve()
    for folder, dirs, files in os.walk(root, topdown=True, followlinks=False):
        folder_path = Path(folder)
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        try:
            resolved = folder_path.resolve()
            if resolved == output or output in resolved.parents:
                dirs[:] = []
                continue
        except OSError:
            continue
        for name in files:
            path = folder_path / name
            try:
                if path.is_file() and not path.is_symlink():
                    yield path
            except OSError:
                continue


def hash_file(path: Path) -> tuple[str | None, str | None]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest(), None
    except Exception as exc:
        return None, repr(exc)


def file_inventory(workspace: Path, output: Path, workers: int) -> list[dict[str, Any]]:
    root = workspace / "artifacts.local"
    paths = list(iter_files(root, output))
    records: list[dict[str, Any]] = []
    for path in paths:
        asset_class, pixel_domain = classify(path)
        records.append({
            "path": path,
            "rel_path": relpath(path, workspace),
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
            "asset_class": asset_class,
            "pixel_domain": pixel_domain,
        })
    print(f"[inventory] files={len(records)}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {pool.submit(hash_file, rec["path"]): rec for rec in records}
        for future in as_completed(jobs):
            rec = jobs[future]
            digest, error = future.result()
            rec["file_sha256"] = digest or ""
            rec["file_sha256_error"] = error or ""
            completed += 1
            if completed % 5000 == 0 or completed == len(records):
                print(f"[inventory] sha256 {completed}/{len(records)}", flush=True)
    return records


def normal_session(value: Any) -> str | None:
    text = norm(value)
    if text is None:
        return None
    prefix = text.split(":", 1)[0].lower()
    if ":" in text and prefix in {"sanpo_real_v0", "sanpo", "sanpo-real", "sanpo-real-v0"}:
        return text.split(":", 1)[1]
    return text


def load_contract(workspace: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "train_sessions": set(), "dev_sessions": set(),
        "fixed_sessions": set(), "event_eval_sessions": set(), "files": [],
    }
    ledger_path = workspace / "docs/research/dual-loop/RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json"
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            for role, target in (("train", "train_sessions"), ("dev", "dev_sessions")):
                for key in ledger.get("roles", {}).get(role, {}).get("sessions", {}):
                    if session := normal_session(key):
                        result[target].add(session)
            for event in ledger.get("roles", {}).get("fixed_regression", {}).get("events", []):
                if session := normal_session(event.get("source_session_id")):
                    result["fixed_sessions"].add(session)
            result["files"].append(relpath(ledger_path, workspace))
        except Exception:
            pass
    event_path = workspace / "artifacts.local/evidence/riskseg-r0/event-eval/device-view-v2/manifest.json"
    if event_path.is_file():
        try:
            manifest = json.loads(event_path.read_text(encoding="utf-8"))
            for event in manifest.get("events", []):
                if session := normal_session(event.get("source_session_id")):
                    result["event_eval_sessions"].add(session)
            result["files"].append(relpath(event_path, workspace))
        except Exception:
            pass
    return result


def lookup(record: dict[str, Any], keys: Iterable[str]) -> Any:
    wanted = {flat_key(key) for key in keys}
    for key, value in record.items():
        if flat_key(key) in wanted and not isinstance(value, (dict, list)):
            return value
    for parent in ("source", "metadata", "identity", "asset"):
        nested = record.get(parent)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if flat_key(key) in wanted and not isinstance(value, (dict, list)):
                    return value
    return None


def derive_ancestry(record: dict[str, Any], session: str | None, sequence: str | None) -> str | None:
    explicit = lookup(record, ("source_ancestry_id", "ancestry_id", "ancestry_group", "source_group"))
    if explicit:
        return f"explicit:{normal_session(explicit)}"
    source = record.get("source")
    if isinstance(source, dict):
        for key in ("ancestry_id", "source_ancestry_id", "sequence_id", "original_sequence_id"):
            if source.get(key):
                return f"source:{normal_session(source[key])}"
        original = source.get("original_object_name")
        if original:
            parent = re.sub(r"/(?:video_frames|images|frames)/\d+\.[a-z0-9]+$", "", str(original), flags=re.I)
            return f"original:{parent}"
    if sequence:
        return f"sequence:{sequence}"
    if session:
        return f"session:{session}"
    return None


def roles_for(path: Path, record: dict[str, Any], contract: dict[str, Any]) -> set[str]:
    lower = path.as_posix().lower()
    final_event_eval = "riskseg-r0/event-eval/device-view-v2" in lower
    event_eval_artifact = "riskseg-r0/event-eval/" in lower
    result: set[str] = set()
    values: list[str] = []
    for key in ("role", "data_role", "split", "stage", "dataset_role", "cohort"):
        if value := lookup(record, (key,)):
            values.append(str(value).lower())
    if final_event_eval:
        result.update({"event_eval", "new_cohort"})
    if "blindassist-sanpo-v2-event-labeled" in lower or "sanpo-v3-regression-90f" in lower:
        result.add("fixed_regression")
    if "training-reencoded-view" in lower or "dual-loop-segmentation-r2-p0" in lower:
        session = normal_session(lookup(record, ("session_id", "source_session_id")))
        if session in contract["train_sessions"]:
            result.add("train")
        elif session in contract["dev_sessions"]:
            result.add("dev")
        else:
            result.add("development")
    if "blind_holdout" in lower:
        result.add("old_cohort")
    if "fresh_holdout" in lower or "model-selection" in lower:
        result.add("model_selection")
    if "official-test" in lower or "official_test" in lower or "reserved-official" in lower:
        result.update({"official_test", "reserved_test"})
    if "calibration" in lower:
        result.add("calibration")
    if "failure-atlas" in lower or "information-ceiling" in lower or "diagnostic" in lower:
        result.add("diagnostic")
    if "development" in lower or "consumed" in lower:
        result.add("development")
    for value in values:
        for token in re.split(r"[,/|]+", value.replace(" ", "_")):
            token = token.strip("_-")
            if token in ROLE_ALIASES:
                result.add(ROLE_ALIASES[token])
            elif token in ROLE_ORDER:
                result.add(token)
    # Historical event-eval screens, materialized drafts, and device-view-v1
    # are Development provenance, not the frozen 30-parent event-eval role.
    # Keep the final role atomic even when its manifest rows carry split=test.
    if event_eval_artifact and not final_event_eval:
        result.difference_update({"event_eval", "new_cohort"})
        result.add("development")
    if final_event_eval:
        result.difference_update({"test", "unknown", "development"})
        result.update({"event_eval", "new_cohort"})
    session = normal_session(lookup(record, ("source_session_id", "session_id")))
    if session in contract["train_sessions"]:
        result.discard("dev")
        result.add("train")
    elif session in contract["dev_sessions"]:
        result.discard("train")
        result.add("dev")
    else:
        result.difference_update({"train", "dev"})
    if not result:
        result.add("unknown")
    return result


def media_value(key: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value.split("?", 1)[0].lower()
    if Path(raw).suffix not in MEDIA_EXTS:
        return False
    lower = key.lower()
    return lower in {"path", "file", "filename", "uri"} or any(token in lower for token in ("image", "rgb", "frame", "video", "mask", "depth", "disparity", "media"))


def resolve_path(value: str, manifest: Path, workspace: Path) -> Path | None:
    raw = value.strip().split("?", 1)[0]
    if raw.startswith(("http://", "https://", "gs://")):
        return None
    candidates = []
    direct = Path(raw)
    if direct.is_absolute():
        candidates.append(direct)
    candidates.extend((workspace / raw, manifest.parent / raw))
    for candidate in candidates:
        try:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate.resolve()
        except OSError:
            pass
    return None


def walk_records(value: Any, inherited: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    inherited = inherited or {}
    if isinstance(value, dict):
        context = dict(inherited)
        for key, item in value.items():
            if key in IDENTITY_KEYS and not isinstance(item, (dict, list)):
                context[key] = item
        yield {**context, **value}
        for child in value.values():
            yield from walk_records(child, context)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child, inherited)


def make_observation(record: dict[str, Any], key: str, value: str, manifest: Path, workspace: Path, contract: dict[str, Any]) -> dict[str, Any]:
    session = normal_session(lookup(record, ("source_session_id", "session_id")))
    sequence = norm(lookup(record, ("sequence_id", "source_sequence_id")))
    parent = norm(lookup(record, ("parent_event_id", "risk_event_id", "event_id", "event_candidate_id")))
    frame = norm(lookup(record, ("source_frame_index", "frame_index")))
    timestamp = norm(lookup(record, ("source_timestamp_ms", "timestamp_ms", "source_capture_timestamp_ns")))
    claims: list[dict[str, str]] = []
    allowed = {flat_key(key) for key in HASH_KEYS}
    for item_key, item_value in record.items():
        if flat_key(item_key) in allowed and isinstance(item_value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item_value):
            claims.append({"key": item_key, "sha256": item_value.lower()})
    source = record.get("source")
    if isinstance(source, dict):
        for item_key, item_value in source.items():
            if flat_key(item_key) in allowed and isinstance(item_value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item_value):
                claims.append({"key": f"source.{item_key}", "sha256": item_value.lower()})
    resolved = resolve_path(value, manifest, workspace)
    lower_key = key.lower()
    path_kind = "mask" if "mask" in lower_key else "depth" if "depth" in lower_key or "disparity" in lower_key else "video" if "video" in lower_key else "image"
    return {
        "manifest_path": relpath(manifest, workspace),
        "media_key": key,
        "declared_path": value,
        "resolved_abs": str(resolved) if resolved else "",
        "resolved_path": relpath(resolved, workspace) if resolved else "",
        "path_kind": path_kind,
        "roles": sorted(roles_for(manifest, record, contract)),
        "session_id": session or "",
        "ancestry_id": derive_ancestry(record, session, sequence) or "",
        "sequence_id": sequence or "",
        "parent_event_id": parent or "",
        "frame_index": frame or "",
        "timestamp": timestamp or "",
        "source_row_id": norm(record.get("id")) or norm(record.get("source_row_id")) or "",
        "claims": claims,
    }


def normalize_observation_roles(observation: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep frozen event-eval separate from historical candidate artifacts."""
    manifest = observation.get("manifest_path", "").lower()
    final_event_eval = "riskseg-r0/event-eval/device-view-v2" in manifest
    event_eval_artifact = "riskseg-r0/event-eval/" in manifest
    roles = set(observation.get("roles", []))
    if event_eval_artifact and not final_event_eval:
        roles.difference_update({"event_eval", "new_cohort"})
        roles.add("development")
    if final_event_eval:
        roles.difference_update({"test", "unknown", "development"})
        roles.update({"event_eval", "new_cohort"})
    if contract is not None:
        session = observation.get("session_id", "")
        if session in contract["train_sessions"]:
            roles.discard("dev")
            roles.add("train")
        elif session in contract["dev_sessions"]:
            roles.discard("train")
            roles.add("dev")
        else:
            # Generic split=train/dev in a broad canonical archive is not the
            # frozen 520 train/dev role unless its source session is in the
            # current role ledger.
            roles.difference_update({"train", "dev"})
    observation["roles"] = sorted(roles) if roles else ["unknown"]
    return observation


def metadata_candidate(path: Path) -> bool:
    if path.suffix.lower() not in META_EXTS:
        return False
    if path.suffix.lower() in {".jsonl", ".csv", ".tsv"}:
        return True
    text = path.as_posix().lower()
    return any(token in text for token in (
        "manifest", "ledger", "inventory", "split", "cohort", "ancestry",
        "candidate", "receipt", "role", "contract", "event-eval",
        "event_eval", "riskseg", "sanpo", "hftf", "dual-loop", "dataset",
    ))


def parse_metadata(workspace: Path, files: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [rec["path"] for rec in files if metadata_candidate(rec["path"])]
    print(f"[metadata] candidate_files={len(paths)}", flush=True)
    observations: list[dict[str, Any]] = []
    for number, path in enumerate(paths, 1):
        try:
            if path.suffix.lower() == ".jsonl":
                records: Iterable[Any] = (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            elif path.suffix.lower() in {".csv", ".tsv"}:
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    records = list(csv.DictReader(fh, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
            else:
                records = walk_records(json.loads(path.read_text(encoding="utf-8")))
            for value in records:
                if not isinstance(value, dict):
                    continue
                for key, item in value.items():
                    if media_value(str(key), item):
                        obs = make_observation(value, str(key), str(item), path, workspace, contract)
                        obs = normalize_observation_roles(obs, contract)
                        if obs["resolved_abs"] or obs["path_kind"] == "video":
                            observations.append(obs)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
            continue
        if number % 500 == 0 or number == len(paths):
            print(f"[metadata] parsed={number}/{len(paths)} observations={len(observations)}", flush=True)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for obs in observations:
        key = (
            obs["resolved_abs"], obs["manifest_path"], tuple(obs["roles"]),
            obs["session_id"], obs["ancestry_id"], obs["sequence_id"],
            obs["parent_event_id"], obs["frame_index"], obs["media_key"],
        )
        unique[key] = obs
    return list(unique.values())


def attach_metadata(files: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    asset_meta: dict[str, dict[str, set[str]]] = {}
    for obs in observations:
        if not obs["resolved_abs"]:
            continue
        entry = asset_meta.setdefault(obs["resolved_abs"], {
            "roles": set(), "sessions": set(), "ancestries": set(),
            "sequences": set(), "parents": set(), "manifests": set(),
        })
        entry["roles"].update(obs["roles"])
        for field, target in (
            ("session_id", "sessions"), ("ancestry_id", "ancestries"),
            ("sequence_id", "sequences"), ("parent_event_id", "parents"),
            ("manifest_path", "manifests"),
        ):
            if obs[field]:
                entry[target].add(obs[field])
    for rec in files:
        entry = asset_meta.get(str(rec["path"].resolve()), {})
        rec["roles"] = sorted(entry.get("roles", set()))
        rec["session_ids"] = sorted(entry.get("sessions", set()))
        rec["ancestry_ids"] = sorted(entry.get("ancestries", set()))
        rec["sequence_ids"] = sorted(entry.get("sequences", set()))
        rec["parent_event_ids"] = sorted(entry.get("parents", set()))
        rec["manifest_paths"] = sorted(entry.get("manifests", set()))


def dct_matrix() -> Any:
    n = 32
    rows = np.arange(n, dtype=np.float32)[:, None]
    cols = np.arange(n, dtype=np.float32)[None, :]
    matrix = np.cos((math.pi / n) * (cols + 0.5) * rows)
    matrix[0, :] *= 1.0 / math.sqrt(n)
    matrix[1:, :] *= math.sqrt(2.0 / n)
    return matrix


def phash(image: Any, matrix: Any) -> int:
    grey = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(grey, dtype=np.float32)
    coeff = matrix @ array @ matrix.T
    low = coeff[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    value = 0
    for item in low:
        value = (value << 1) | int(item > median)
    return value


def variants(image: Any) -> dict[str, Any]:
    width, height = image.size
    result = {"original": image, "mirror_horizontal": image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)}
    for fraction, label in ((0.90, "crop_center_90"), (0.80, "crop_center_80")):
        crop_w, crop_h = max(32, int(width * fraction)), max(32, int(height * fraction))
        left, top = max(0, (width - crop_w) // 2), max(0, (height - crop_h) // 2)
        result[label] = image.crop((left, top, left + crop_w, top + crop_h))
    crop_w, crop_h = max(32, int(width * 0.85)), max(32, int(height * 0.85))
    result["crop_left_85"] = image.crop((0, 0, crop_w, crop_h))
    result["crop_right_85"] = image.crop((width - crop_w, height - crop_h, width, height))
    return result


def decode_one(rec: dict[str, Any], matrix: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": rec["path"], "rel_path": rec["rel_path"],
        "asset_class": rec["asset_class"], "pixel_domain": rec["pixel_domain"],
        "width": "", "height": "", "mode": "", "rgb_pixel_sha256": "",
        "phash": "", "phash_variants": {}, "decode_error": "",
    }
    try:
        with Image.open(rec["path"]) as source:
            source.load()
            result["width"], result["height"], result["mode"] = source.size[0], source.size[1], source.mode
            rgb = source.convert("RGB")
            result["rgb_pixel_sha256"] = hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest()
            if rec["pixel_domain"] == "rgb":
                result["phash_variants"] = {name: f"{phash(item, matrix):016x}" for name, item in variants(rgb).items()}
                result["phash"] = result["phash_variants"]["original"]
    except Exception as exc:
        result["decode_error"] = repr(exc)
    return result


def decode_images(files: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    images = [rec for rec in files if rec["extension"] in IMAGE_EXTS]
    if Image is None or np is None:
        return [{
            "path": rec["path"], "rel_path": rec["rel_path"],
            "asset_class": rec["asset_class"], "pixel_domain": rec["pixel_domain"],
            "decode_error": f"dependencies_unavailable:{IMAGE_IMPORT_ERROR}",
        } for rec in images]
    matrix = dct_matrix()
    print(f"[decode] images={len(images)}", flush=True)
    result: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {pool.submit(decode_one, rec, matrix): rec for rec in images}
        for number, future in enumerate(as_completed(jobs), 1):
            result.append(future.result())
            if number % 2000 == 0 or number == len(images):
                print(f"[decode] images {number}/{len(images)}", flush=True)
    return result


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_groups(
    images: list[dict[str, Any]],
    threshold: int,
    bucket_cap: int = 256,
    candidate_cap: int = 2000,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find pHash/transform components with deterministic resource bounds.

    A low-entropy image corpus can put tens of thousands of frames in one LSH
    bucket.  Enumerating every pair is neither necessary for component
    evidence nor safe for a repository-wide audit, so large buckets and
    per-image candidate lists are stratified/capped.  Exact file and decoded
    RGB checks remain uncapped and are emitted separately.
    """
    stats = stats if stats is not None else {}
    usable = [item for item in images if item.get("pixel_domain") == "rgb" and item.get("phash") and not item.get("decode_error")]
    stats.update({
        "usable_images": len(usable),
        "bucket_cap": bucket_cap,
        "candidate_cap": candidate_cap,
        "bucket_truncation_count": 0,
        "candidate_truncation_count": 0,
        "candidate_pairs_checked": 0,
        "accepted_edges": 0,
        "bounded": True,
    })
    if not usable:
        stats["bounded_complete"] = True
        return []

    bands, masks = 8, [0] + [1 << bit for bit in range(8)]
    role_keys = [tuple(item.get("roles", [])) for item in usable]
    session_keys = [frozenset(item.get("session_ids", [])) for item in usable]
    folder_keys = [Path(item["rel_path"]).parent for item in usable]

    def same_temporal(left: int, right: int) -> bool:
        return (
            folder_keys[left] == folder_keys[right]
            and bool(session_keys[left] & session_keys[right])
            and role_keys[left] == role_keys[right]
        )

    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for number, item in enumerate(usable):
        value = int(item["phash"], 16)
        aspect = round(float(item.get("width") or 0) / max(1, int(item.get("height") or 1)) * 20)
        for band in range(bands):
            shift = (bands - band - 1) * 8
            index[(band, (value >> shift) & 0xFF, aspect)].append(number)

    def sample_bucket(values: list[int]) -> list[int]:
        if len(values) <= bucket_cap:
            return values
        # Preserve role diversity before filling the remainder evenly.  This
        # keeps cross-role evidence available even when frames are grouped by
        # directory in the source corpus.
        chosen: list[int] = []
        chosen_set: set[int] = set()
        by_role: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for number in values:
            by_role[role_keys[number]].append(number)
        role_limit = max(1, bucket_cap // max(1, len(by_role)))
        for key in sorted(by_role):
            group = by_role[key]
            if len(group) <= role_limit:
                picks = group
            else:
                stride = max(1, len(group) // role_limit)
                picks = group[::stride][:role_limit]
                if group[-1] not in picks:
                    picks.append(group[-1])
            for number in picks:
                if number not in chosen_set and len(chosen) < bucket_cap:
                    chosen.append(number)
                    chosen_set.add(number)
        if len(chosen) < bucket_cap:
            stride = max(1, len(values) // max(1, bucket_cap - len(chosen)))
            for number in values[::stride]:
                if number not in chosen_set:
                    chosen.append(number)
                    chosen_set.add(number)
                    if len(chosen) >= bucket_cap:
                        break
        return chosen[:bucket_cap]

    bounded_index: dict[tuple[int, int, int], list[int]] = {}
    truncated_buckets: set[tuple[int, int, int]] = set()
    for key, values in index.items():
        bounded_index[key] = sample_bucket(values)
        if len(values) > bucket_cap:
            truncated_buckets.add(key)
    stats["bucket_count"] = len(index)
    stats["large_bucket_count"] = len(truncated_buckets)
    stats["bucket_truncation_count"] = sum(len(index[key]) - len(bounded_index[key]) for key in truncated_buckets)

    uf = UnionFind(len(usable))
    edge_samples: list[dict[str, Any]] = []
    for number, item in enumerate(usable):
        aspect = round(float(item.get("width") or 0) / max(1, int(item.get("height") or 1)) * 20)
        candidates: dict[int, tuple[str, int]] = {}
        candidate_truncated = False
        for transform, hashed in (item.get("phash_variants") or {}).items():
            value = int(hashed, 16)
            stop = False
            for band in range(bands):
                shift = (bands - band - 1) * 8
                current = (value >> shift) & 0xFF
                for mask in masks:
                    for candidate_aspect in range(aspect - 2, aspect + 3):
                        key = (band, current ^ mask, candidate_aspect)
                        for other_number in bounded_index.get(key, []):
                            if other_number == number or other_number in candidates:
                                continue
                            # Same-folder same-session frames are expected
                            # temporal dependence; adjacency is reported by its
                            # dedicated manifest check.  Filter before the
                            # candidate cap so such cliques cannot starve
                            # cross-role/cross-session candidates.
                            if same_temporal(number, other_number):
                                continue
                            candidates[other_number] = (transform, value)
                            if len(candidates) >= candidate_cap:
                                candidate_truncated = True
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
        if candidate_truncated:
            stats["candidate_truncation_count"] += 1
        for other_number, (transform, value) in candidates.items():
            other = usable[other_number]
            if item.get("rgb_pixel_sha256") and item.get("rgb_pixel_sha256") == other.get("rgb_pixel_sha256"):
                continue
            distance = hamming(value, int(other["phash"], 16))
            stats["candidate_pairs_checked"] += 1
            if distance > threshold:
                continue
            uf.union(number, other_number)
            stats["accepted_edges"] += 1
            if len(edge_samples) < 250000:
                pair = tuple(sorted((number, other_number)))
                edge_samples.append({
                    "left": pair[0], "right": pair[1],
                    "distance": distance, "transform": transform,
                })
        if (number + 1) % 5000 == 0 or number + 1 == len(usable):
            print(f"[phash] images {number + 1}/{len(usable)} candidates={stats['candidate_pairs_checked']}", flush=True)

    stats["bounded_complete"] = stats["bucket_truncation_count"] == 0 and stats["candidate_truncation_count"] == 0
    groups: dict[int, list[int]] = defaultdict(list)
    for number in range(len(usable)):
        groups[uf.find(number)].append(number)
    samples_by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in edge_samples:
        root = uf.find(sample["left"])
        if len(samples_by_root[root]) < 50:
            samples_by_root[root].append(sample)
    rows: list[dict[str, Any]] = []
    order = 0
    for root, members in sorted(((root, members) for root, members in groups.items() if len(members) > 1), key=lambda pair: (-len(pair[1]), pair[0])):
        order += 1
        records = [usable[number] for number in members]
        samples = samples_by_root.get(root, [])
        roles = sorted({role for item in records for role in item.get("roles", [])})
        sessions = sorted({sid for item in records for sid in item.get("session_ids", [])})
        distances = [item["distance"] for item in samples]
        rows.append({
            "near_group_id": f"PHASH-{order:06d}",
            "relation_type": "perceptual_near_duplicate",
            "match_transform_evidence": "|".join(sorted({item["transform"] for item in samples})),
            "phash_threshold": threshold,
            "sampled_min_hamming": min(distances) if distances else "",
            "sampled_max_hamming": max(distances) if distances else "",
            "member_count": len(records),
            "cross_role": len(roles) > 1,
            "cross_session": len(sessions) > 1,
            "roles": "|".join(roles[:50]),
            "sessions": "|".join(sessions[:50]),
            "representative_paths": "|".join(item["rel_path"] for item in records[:20]),
            "edge_sample": "|".join(
                f"{usable[item['left']]['rel_path']}~{usable[item['right']]['rel_path']}~d{item['distance']}~{item['transform']}"
                for item in samples[:20]
            ),
            "edge_count_observed": len(samples),
        })
    return rows


def duplicate_groups(files: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rec in files:
        if rec.get("file_sha256"):
            groups[("file_sha256", rec["file_sha256"])].append(rec)
    by_path = {str(rec["path"].resolve()): rec for rec in files}
    for item in images:
        if item.get("rgb_pixel_sha256") and not item.get("decode_error"):
            file_rec = by_path.get(str(item["path"].resolve()), {})
            groups[("decoded_rgb_pixels", file_rec.get("pixel_domain", ""), item.get("width"), item.get("height"), item["rgb_pixel_sha256"])].append(item)
    rows: list[dict[str, Any]] = []
    order = 0
    for key, members in sorted(groups.items(), key=lambda pair: (-len(pair[1]), str(pair[0]))):
        if len(members) < 2:
            continue
        order += 1
        basis = key[0]
        paths = [item["rel_path"] for item in members]
        row = {
            "duplicate_group_id": f"DUP-{order:06d}",
            "match_basis": basis,
            "sha256": key[1] if basis == "file_sha256" else "",
            "rgb_pixel_sha256": key[-1] if basis == "decoded_rgb_pixels" else "",
            "pixel_domain": key[1] if basis == "decoded_rgb_pixels" else "",
            "size_bytes": members[0].get("size_bytes", ""),
            "width": members[0].get("width", ""),
            "height": members[0].get("height", ""),
            "member_count": len(members),
            "different_filename": len({Path(path).name for path in paths}) > 1,
            "asset_classes": "|".join(sorted({item.get("asset_class", "") for item in members})),
            "roles": "|".join(sorted({role for item in members for role in item.get("roles", [])})),
            "sessions": "|".join(sorted({sid for item in members for sid in item.get("session_ids", [])})),
            "parent_events": "|".join(sorted({pid for item in members for pid in item.get("parent_event_ids", [])})),
            "paths": "|".join(paths[:50]),
        }
        rows.append(row)
    return rows


def obs_index(observations: list[dict[str, Any]]) -> dict[str, Any]:
    result = {name: defaultdict(lambda: defaultdict(list)) for name in ("session", "ancestry", "parent", "claim")}
    for obs in observations:
        for role in obs["roles"]:
            if obs["session_id"]:
                result["session"][obs["session_id"]][role].append(obs)
            if obs["ancestry_id"]:
                result["ancestry"][obs["ancestry_id"]][role].append(obs)
            if obs["parent_event_id"]:
                result["parent"][obs["parent_event_id"]][role].append(obs)
            for claim in obs["claims"]:
                result["claim"][claim["sha256"]][role].append(obs)
    return result


def role_pairs(roles: Iterable[str]) -> list[tuple[str, str]]:
    unique = sorted(set(roles), key=lambda item: (ROLE_ORDER.index(item) if item in ROLE_ORDER else 99, item))
    return list(itertools.combinations(unique, 2))


def overlap_rows(index: dict[str, Any], files: list[dict[str, Any]], images: list[dict[str, Any]], near: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = 0
    for kind in ("session", "ancestry", "parent", "claim"):
        for identifier, by_role in sorted(index[kind].items()):
            for left, right in role_pairs(by_role):
                order += 1
                a, b = by_role[left], by_role[right]
                rows.append({
                    "overlap_id": f"OV-{order:06d}", "overlap_type": kind,
                    "identifier": identifier, "left_role": left, "right_role": right,
                    "left_count": len(a), "right_count": len(b),
                    "left_sessions": "|".join(sorted({x["session_id"] for x in a if x["session_id"]})),
                    "right_sessions": "|".join(sorted({x["session_id"] for x in b if x["session_id"]})),
                    "left_parents": "|".join(sorted({x["parent_event_id"] for x in a if x["parent_event_id"]})),
                    "right_parents": "|".join(sorted({x["parent_event_id"] for x in b if x["parent_event_id"]})),
                    "left_manifests": "|".join(sorted({x["manifest_path"] for x in a})[:10]),
                    "right_manifests": "|".join(sorted({x["manifest_path"] for x in b})[:10]),
                    "severity": "critical" if ({left, right} & {"event_eval", "official_test", "reserved_test"}) and ({left, right} & {"train", "model_train"}) else "high",
                    "evidence": "manifest_identity_or_claimed_sha256",
                })
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in files:
        if rec.get("file_sha256"):
            groups[f"file:{rec['file_sha256']}"].append(rec)
    for item in images:
        if item.get("rgb_pixel_sha256"):
            groups[f"rgb:{item['rgb_pixel_sha256']}"].append(item)
    for identifier, members in groups.items():
        by_role = defaultdict(list)
        for member in members:
            for role in member.get("roles", []):
                by_role[role].append(member)
        for left, right in role_pairs(by_role):
            order += 1
            a, b = by_role[left], by_role[right]
            rows.append({
                "overlap_id": f"OV-{order:06d}",
                "overlap_type": "file_sha256" if identifier.startswith("file:") else "rgb_pixel_sha256",
                "identifier": identifier.split(":", 1)[1],
                "left_role": left, "right_role": right,
                "left_count": len(a), "right_count": len(b),
                "left_sessions": "|".join(sorted({x for member in a for x in member.get("session_ids", [])})),
                "right_sessions": "|".join(sorted({x for member in b for x in member.get("session_ids", [])})),
                "left_parents": "|".join(sorted({x for member in a for x in member.get("parent_event_ids", [])})),
                "right_parents": "|".join(sorted({x for member in b for x in member.get("parent_event_ids", [])})),
                "left_manifests": "|".join(sorted({x for member in a for x in member.get("manifest_paths", [])})[:10]),
                "right_manifests": "|".join(sorted({x for member in b for x in member.get("manifest_paths", [])})[:10]),
                "severity": "critical" if ({left, right} & {"event_eval", "official_test", "reserved_test"}) and ({left, right} & {"train", "model_train"}) else "high",
                "evidence": identifier,
            })
    for group in near:
        if not group.get("cross_role"):
            continue
        roles = group.get("roles", "").split("|")
        for left, right in role_pairs(roles):
            order += 1
            rows.append({
                "overlap_id": f"OV-{order:06d}", "overlap_type": "phash_near_duplicate",
                "identifier": group["near_group_id"], "left_role": left, "right_role": right,
                "left_count": "", "right_count": "", "left_sessions": group.get("sessions", ""),
                "right_sessions": group.get("sessions", ""), "left_parents": "", "right_parents": "",
                "left_manifests": "", "right_manifests": "",
                "severity": "critical" if ({left, right} & {"event_eval", "official_test", "reserved_test"}) and ({left, right} & {"train", "model_train"}) else "medium",
                "evidence": "phash_transform_variant",
            })
    return rows


def adjacent_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        if obs["sequence_id"] and obs["frame_index"]:
            try:
                int(obs["frame_index"])
            except ValueError:
                continue
            groups[obs["sequence_id"]].append(obs)
    result: list[dict[str, Any]] = []
    order = 0
    for sequence, items in groups.items():
        items.sort(key=lambda item: int(item["frame_index"]))
        for left, right in zip(items, items[1:]):
            if int(right["frame_index"]) - int(left["frame_index"]) != 1:
                continue
            left_roles, right_roles = set(left["roles"]), set(right["roles"])
            cross = left_roles != right_roles or not left_roles.intersection(right_roles)
            order += 1
            result.append({
                "adjacency_id": f"ADJ-{order:06d}", "sequence_id": sequence,
                "left_frame_index": left["frame_index"], "right_frame_index": right["frame_index"],
                "left_session_id": left["session_id"], "right_session_id": right["session_id"],
                "left_parent_event_id": left["parent_event_id"], "right_parent_event_id": right["parent_event_id"],
                "left_roles": "|".join(left["roles"]), "right_roles": "|".join(right["roles"]),
                "cross_role": cross, "left_path": left["resolved_path"], "right_path": right["resolved_path"],
                "leakage_status": "CROSS_ROLE_ADJACENT" if cross else "INTRA_ROLE_TEMPORAL_DEPENDENCE",
                "evidence": "manifest_sequence_id_and_frame_index",
            })
    return result


def probe_videos(files: list[dict[str, Any]], ffprobe: Path | None) -> list[dict[str, Any]]:
    videos = [rec for rec in files if rec["extension"] in VIDEO_EXTS]
    if not ffprobe or not ffprobe.is_file():
        return [{"path": rec["rel_path"], "status": "NOT_EVALUABLE_FFPROBE_MISSING"} for rec in videos]
    result: list[dict[str, Any]] = []
    for number, rec in enumerate(videos, 1):
        row = {"path": rec["rel_path"], "status": "NOT_EVALUABLE", "streams": ""}
        try:
            completed = subprocess.run(
                [str(ffprobe), "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames", "-of", "json", str(rec["path"])],
                capture_output=True, text=True, timeout=120, check=False,
            )
            if completed.returncode == 0:
                payload = json.loads(completed.stdout or "{}")
                streams = payload.get("streams", [])
                row.update({
                    "status": "PROBED",
                    "duration_seconds": payload.get("format", {}).get("duration", ""),
                    "streams": json.dumps(streams, ensure_ascii=False, sort_keys=True),
                })
            else:
                row["error"] = completed.stderr[-1000:]
        except Exception as exc:
            row["error"] = repr(exc)
        result.append(row)
        if number % 10 == 0 or number == len(videos):
            print(f"[ffprobe] videos {number}/{len(videos)}", flush=True)
    return result


def graph(observations: list[dict[str, Any]], overlaps: list[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    def add(node_id: str, kind: str, label: str, **attrs: Any) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": kind, "label": label, **attrs}
    for obs in observations:
        for role in obs["roles"]:
            cohort = f"cohort:{role}"
            add(cohort, "cohort", role)
            if obs["session_id"]:
                session = f"session:{obs['session_id']}"
                add(session, "session", obs["session_id"], roles=[])
                nodes[session]["roles"] = sorted(set(nodes[session]["roles"]) | {role})
                edges.append({"from": cohort, "to": session, "type": "contains_session"})
                if obs["ancestry_id"]:
                    ancestry = f"ancestry:{obs['ancestry_id']}"
                    add(ancestry, "ancestry", obs["ancestry_id"])
                    edges.append({"from": session, "to": ancestry, "type": "has_ancestry"})
            if obs["parent_event_id"]:
                event = f"parent_event:{obs['parent_event_id']}"
                add(event, "parent_event", obs["parent_event_id"])
                if obs["session_id"]:
                    edges.append({"from": f"session:{obs['session_id']}", "to": event, "type": "contains_parent_event"})
    for row in overlaps:
        overlap = f"overlap:{row['overlap_id']}"
        add(overlap, "overlap", row["overlap_type"], identifier=row["identifier"], severity=row["severity"])
        add(f"cohort:{row['left_role']}", "cohort", row["left_role"])
        add(f"cohort:{row['right_role']}", "cohort", row["right_role"])
        edges.append({"from": f"cohort:{row['left_role']}", "to": overlap, "type": "overlaps"})
        edges.append({"from": f"cohort:{row['right_role']}", "to": overlap, "type": "overlaps"})
    return {
        "schema_version": "blindassist.data_contamination_audit.parent_ancestry_graph.v1",
        "generated_at": now_utc(), "run": run,
        "nodes": list(nodes.values()), "edges": edges,
        "summary": {
            "observation_count": len(observations), "node_count": len(nodes),
            "edge_count": len(edges), "overlap_count": len(overlaps),
        },
    }


def candidates(observations: list[dict[str, Any]], overlaps: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"roles": set(), "sessions": set(), "parents": set(), "count": 0})
    for obs in observations:
        key = obs["ancestry_id"] or (f"session:{obs['session_id']}" if obs["session_id"] else "unidentified")
        item = groups[key]
        item["roles"].update(obs["roles"])
        if obs["session_id"]:
            item["sessions"].add(obs["session_id"])
        if obs["parent_event_id"]:
            item["parents"].add(obs["parent_event_id"])
        item["count"] += 1
    blocked: dict[str, set[str]] = defaultdict(set)
    for row in overlaps:
        blocked[row["identifier"]].add(f"{row['overlap_type']}:{row['left_role']}:{row['right_role']}")
    eligible, denied = [], []
    for key, item in sorted(groups.items()):
        role_set = item["roles"]
        reasons = sorted(blocked.get(key, set()))
        if reasons or ({"train", "event_eval"} <= role_set) or ({"model_train", "event_eval"} <= role_set):
            denied.append({
                "group_key": key, "sessions": sorted(item["sessions"]), "roles_observed": sorted(role_set),
                "parent_event_count": len(item["parents"]), "observation_count": item["count"],
                "blocked_reasons": reasons or ["training_and_event_eval_role_overlap"],
            })
        else:
            eligible.append({
                "group_key": key, "sessions": sorted(item["sessions"]), "roles_observed": sorted(role_set),
                "parent_event_count": len(item["parents"]), "observation_count": item["count"],
                "recommended_role": "new_event_eval_candidate" if not role_set & {"train", "dev", "fixed_regression"} else "retain_current_role_only",
            })
    return {
        "schema_version": "blindassist.data_contamination_audit.clean_split_candidates.v1",
        "generated_at": now_utc(),
        "policy": {
            "atomic_unit": "source_ancestry_id_then_source_session_id_then_parent_event_id",
            "forbidden_cross_split_matches": ["session", "ancestry", "parent_event", "file_sha256", "decoded_rgb_pixels", "perceptual_near_duplicate"],
            "phash_threshold": 8,
            "recommendation": "assign whole ancestry/session connected components to one split; freeze event-eval before opening model outputs",
        },
        "known_contract_sessions": {
            "train": sorted(contract["train_sessions"]), "dev": sorted(contract["dev_sessions"]),
            "fixed_regression": sorted(contract["fixed_sessions"]), "event_eval": sorted(contract["event_eval_sessions"]),
        },
        "eligible_groups": eligible, "blocked_groups": denied,
        "limitations": [
            "Groups without explicit session/ancestry are not eligible for a clean split claim.",
            "Inspect near_duplicates.csv before admitting an eligible group.",
            "Opaque videos or missing frame metadata remain NOT_EVALUABLE for adjacency.",
        ],
    }


def report(output: Path, run: dict[str, Any], files: list[dict[str, Any]], observations: list[dict[str, Any]], images: list[dict[str, Any]], duplicates: list[dict[str, Any]], near: list[dict[str, Any]], overlaps: list[dict[str, Any]], adjacent: list[dict[str, Any]], contract: dict[str, Any], video_rows: list[dict[str, Any]]) -> None:
    critical = [row for row in overlaps if row["severity"] == "critical"]
    train_event = [
        row for row in overlaps
        if {row["left_role"], row["right_role"]} <= {"train", "model_train", "event_eval"}
        and row["left_role"] != row["right_role"]
    ]
    decode_errors = [row for row in images if row.get("decode_error")]
    lines = [
        "# 数据重复与污染审计 R0", "",
        f"生成时间：{run['generated_at']}", "",
        "## 结论先行", "",
        "本报告只支持数据资产、manifest identity 和派生视图之间的污染风险判断；不会把相似图像自动解释为同一自然事件，也不会把 Development 数据写成独立安全证据。任何 critical/high 交叉命中都应在重新冻结 split 前处理或降级角色。", "",
        f"- artifacts.local regular files：{len(files)}；images：{sum(rec['extension'] in IMAGE_EXTS for rec in files)}；videos：{sum(rec['extension'] in VIDEO_EXTS for rec in files)}。",
        f"- manifest observations：{len(observations)}；sessions：{len({obs['session_id'] for obs in observations if obs['session_id']})}；parent events：{len({obs['parent_event_id'] for obs in observations if obs['parent_event_id']})}。",
        f"- exact duplicate groups：{len(duplicates)}；near duplicate groups：{len(near)}；overlap rows：{len(overlaps)}；critical overlap rows：{len(critical)}。",
        f"- train/model-train versus event-eval overlap：{'CONTAMINATION_FOUND' if train_event else 'NO_MANIFEST_OR_CONTENT_OVERLAP_FOUND'} ({len(train_event)} rows)。",
        f"- pHash candidate screen：{'COMPLETE_WITHIN_BOUNDS' if run.get('near_limits', {}).get('bounded_complete') else 'BOUNDED_SCREEN_NOT_EXHAUSTIVE'}；usable images：{run.get('near_limits', {}).get('usable_images', 0)}；checked pairs：{run.get('near_limits', {}).get('candidate_pairs_checked', 0)}。",
        f"- pHash large buckets：{run.get('near_limits', {}).get('large_bucket_count', 0)}（dropped members：{run.get('near_limits', {}).get('bucket_truncation_count', 0)}）；candidate truncations：{run.get('near_limits', {}).get('candidate_truncation_count', 0)}。",
        "",
        "审计不会因为文档声称 30 event-eval parent / 30 source session 就自动判定干净；它同时比较 RISKSEG training re-encoded view、30-event device view、旧 90-frame manifest、canonical/holdout 以及可追溯的其他资产。", "",
        "## 数据范围与粒度", "",
        "- 输入是 artifacts.local 下所有 regular files；输出目录被排除。",
        "- RGB 检查使用 Pillow 解码后的 width、height 和 RGB byte tensor；file SHA 与 decoded RGB hash 分开。",
        "- provenance 原子优先级：source_ancestry_id，其次 source_session_id/session_id，其次 sequence_id；parent event 只在字段明确存在时建立。",
        "- 连续 frame 检查使用 manifest 的 sequence_id + frame_index；跨 role 才升级为 CROSS_ROLE_ADJACENT。",
        "",
        "## 结果表", "",
        "| 检查 | 结果 | 风险 |",
        "|---|---:|---|",
        f"| stored file SHA-256 duplicate groups | {sum(row['match_basis'] == 'file_sha256' for row in duplicates)} | 不同路径/文件名不能视为独立样本 |",
        f"| decoded RGB pixel duplicate groups | {sum(row['match_basis'] == 'decoded_rgb_pixels' for row in duplicates)} | 不同编码/重打包仍是同一观测 |",
        f"| perceptual/transform groups | {len(near)} | 需结合 session/sequence 解释；跨 split 不得默认放行 |",
        f"| session/ancestry/parent/content overlap rows | {len(overlaps)} | 见 session_overlap.csv |",
        f"| adjacent frame pairs | {len(adjacent)} | 同一视频切片的帧不是独立样本 |",
        f"| image decode errors | {len(decode_errors)} | 这些资产的 pixel/pHash 结论为 NOT_EVALUABLE |",
        "",
        "## RISKSEG 合同复核", "",
        f"- train sessions ({len(contract['train_sessions'])})：{'|'.join(sorted(contract['train_sessions']))}",
        f"- dev sessions ({len(contract['dev_sessions'])})：{'|'.join(sorted(contract['dev_sessions']))}",
        f"- fixed regression sessions ({len(contract['fixed_sessions'])})：{'|'.join(sorted(contract['fixed_sessions']))}",
        f"- event-eval sessions ({len(contract['event_eval_sessions'])})：{'|'.join(sorted(contract['event_eval_sessions']))}",
        "- session-disjoint 只能通过 session identity 层；file SHA、decoded RGB、claimed source hash、parent event 和祖源仍需分别通过。",
        "",
        "## 相邻帧、parent event、ancestry", "",
        f"- manifest continuous pairs：{len(adjacent)}；cross-role pairs：{sum(bool(row['cross_role']) for row in adjacent)}。",
        "- 同一个 parent event 跨 role 会在 session_overlap.csv 以 overlap_type=parent 单列；不得靠重命名、复制、裁剪或重新打包恢复独立性。",
        "- old/new cohort ancestry 由 explicit ancestry、source sequence、source session 和 source original object name 分层记录；缺少身份的资产不进入 clean claim。",
        "",
        "## 误报、证据强度和限制", "",
        "- file SHA 和 decoded RGB hash 是强的内容同一证据，但不单独证明同一自然事件。",
        "- pHash、水平镜像和 crop variants 是筛查证据；同一视频相邻帧会形成组件，应由 sequence/frame metadata 解释。",
        "- pHash 使用有界 LSH candidate screen；发生 bucket/candidate 截断时，不能把未枚举的近重复写成已排除。file SHA、decoded RGB、manifest session/parent/ancestry 不受该 pHash 上限影响。",
        "- manifest claimed hashes 与本地验证 file hash 分开；claimed hash 命中不能写成本地 bytes 已验证相同。",
        f"- ffprobe probe rows：{len(video_rows)}；NOT_EVALUABLE：{sum(row.get('status') != 'PROBED' for row in video_rows)}。",
        f"- Pillow decode error examples：{'|'.join(row.get('rel_path', '') for row in decode_errors[:20])}",
        "",
        "## 产物与复现", "",
        "- exact_duplicates.csv：file SHA 与 decoded RGB duplicate groups。",
        "- near_duplicates.csv：pHash threshold=8、水平镜像和 crop variants。",
        "- session_overlap.csv：session、ancestry、parent、claimed hash、actual content 和 pHash 交叉证据。",
        "- parent_ancestry_graph.json：cohort → session → ancestry/parent event 与 overlap 图。",
        "- clean_split_candidates.json：按 ancestry/session 原子分组的候选与阻断原因。",
        "- asset_inventory.jsonl、manifest_observations.jsonl、video_probe.jsonl、run_metadata.json：可复核中间证据。",
        "",
        "复现命令：",
        "",
        f"python scripts/research/data_contamination_audit_r0/audit.py --workspace {run['workspace']} --output {run['output']} --workers {run['workers']} --phash-threshold {run['phash_threshold']} --ffprobe {run['ffprobe']}",
        "",
    ]
    (output / "contamination_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_result_files(
    output: Path,
    run: dict[str, Any],
    files: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    images: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    near: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    adjacent: list[dict[str, Any]],
    contract: dict[str, Any],
    video_rows: list[dict[str, Any]],
) -> None:
    dump_jsonl(output / "asset_inventory.jsonl", ({
        **{key: value for key, value in rec.items() if key != "path"},
        "path": rec["rel_path"],
    } for rec in files))
    dump_jsonl(output / "manifest_observations.jsonl", observations)
    dump_jsonl(output / "video_probe.jsonl", video_rows)
    dump_csv(output / "exact_duplicates.csv", duplicates, [
        "duplicate_group_id", "match_basis", "sha256", "rgb_pixel_sha256",
        "pixel_domain", "size_bytes", "width", "height", "member_count",
        "different_filename", "asset_classes", "roles", "sessions",
        "parent_events", "paths",
    ])
    dump_csv(output / "near_duplicates.csv", near, [
        "near_group_id", "relation_type", "match_transform_evidence",
        "phash_threshold", "sampled_min_hamming", "sampled_max_hamming",
        "member_count", "cross_role", "cross_session", "roles", "sessions",
        "representative_paths", "edge_sample", "edge_count_observed",
    ])
    dump_csv(output / "session_overlap.csv", overlaps, [
        "overlap_id", "overlap_type", "identifier", "left_role", "right_role",
        "left_count", "right_count", "left_sessions", "right_sessions",
        "left_parents", "right_parents", "left_manifests", "right_manifests",
        "severity", "evidence",
    ])
    dump_csv(output / "adjacent_frame_pairs.csv", adjacent, [
        "adjacency_id", "sequence_id", "left_frame_index", "right_frame_index",
        "left_session_id", "right_session_id", "left_parent_event_id",
        "right_parent_event_id", "left_roles", "right_roles", "cross_role",
        "left_path", "right_path", "leakage_status", "evidence",
    ])
    dump_json(output / "parent_ancestry_graph.json", graph(observations, overlaps, run))
    dump_json(output / "clean_split_candidates.json", candidates(observations, overlaps, contract))
    report(output, run, files, observations, images, duplicates, near, overlaps, adjacent, contract, video_rows)


def relabel_existing_run(workspace: Path, output: Path, args: argparse.Namespace) -> int:
    """Repair role attribution on a completed snapshot without re-decoding media."""
    started = time.time()
    inventory_path = output / "asset_inventory.jsonl"
    observation_path = output / "manifest_observations.jsonl"
    cache_path = output / "image_features_cache.jsonl"
    if not inventory_path.is_file() or not observation_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("--relabel-existing requires asset_inventory.jsonl, manifest_observations.jsonl, and image_features_cache.jsonl")
    old_run = json.loads((output / "run_metadata.json").read_text(encoding="utf-8")) if (output / "run_metadata.json").is_file() else {}
    files: list[dict[str, Any]] = []
    for row in read_jsonl_rows(inventory_path):
        row["path"] = (workspace / row["path"]).resolve()
        files.append(row)
    contract = load_contract(workspace)
    observations = [normalize_observation_roles(row, contract) for row in read_jsonl_rows(observation_path)]
    attach_metadata(files, observations)
    images = load_image_feature_cache(cache_path, files)
    if images is None:
        raise RuntimeError("existing image_features_cache.jsonl does not match asset_inventory.jsonl")
    by_path = {str(rec["path"].resolve()): rec for rec in files}
    for image in images:
        metadata = by_path.get(str(image["path"].resolve()), {})
        for key in ("roles", "session_ids", "ancestry_ids", "sequence_ids", "parent_event_ids", "manifest_paths"):
            image[key] = metadata.get(key, [])
    dump_image_feature_cache(cache_path, images, files)
    duplicates = duplicate_groups(files, images)
    near_limits: dict[str, Any] = {}
    near = near_groups(
        images,
        args.phash_threshold,
        bucket_cap=max(1, args.near_bucket_cap),
        candidate_cap=max(1, args.near_candidate_cap),
        stats=near_limits,
    )
    overlaps = overlap_rows(obs_index(observations), files, images, near)
    adjacent = adjacent_rows(observations)
    video_path = output / "video_probe.jsonl"
    video_rows = read_jsonl_rows(video_path) if video_path.is_file() else probe_videos(files, args.ffprobe if args.ffprobe else None)
    run = {
        **old_run,
        "generated_at": now_utc(),
        "role_relabelled_from": old_run.get("generated_at", ""),
        "role_policy": "only riskseg-r0/event-eval/device-view-v2 is frozen event_eval; other event-eval artifacts are Development",
        "workspace": str(workspace), "output": str(output),
        "workers": args.workers, "phash_threshold": args.phash_threshold,
        "near_bucket_cap": args.near_bucket_cap, "near_candidate_cap": args.near_candidate_cap,
        "near_limits": near_limits, "image_feature_cache": str(cache_path),
        "image_feature_cache_reused": True, "ffprobe": str(args.ffprobe),
        "input_file_count": len(files), "input_bytes": sum(int(rec["size_bytes"]) for rec in files),
        "image_count": len(images), "video_count": len(video_rows),
        "observation_count": len(observations),
        "image_decode_error_count": sum(bool(rec.get("decode_error")) for rec in images),
        "contract_files": contract["files"],
    }
    write_result_files(output, run, files, observations, images, duplicates, near, overlaps, adjacent, contract, video_rows)
    run["elapsed_seconds"] = round(time.time() - started, 3)
    run["output_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    dump_json(output / "run_metadata.json", run)
    print(json.dumps({
        "status": "completed_relabel", "output": str(output), "files": len(files),
        "images": len(images), "observations": len(observations),
        "exact_groups": len(duplicates), "near_groups": len(near),
        "overlap_rows": len(overlaps), "adjacent_pairs": len(adjacent),
        "elapsed_seconds": run["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts.local/evidence/data-contamination-audit-r0"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--near-bucket-cap", type=int, default=256)
    parser.add_argument("--near-candidate-cap", type=int, default=2000)
    parser.add_argument("--reuse-image-cache", action="store_true")
    parser.add_argument("--relabel-existing", action="store_true")
    parser.add_argument("--ffprobe", type=Path, default=Path(r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffprobe.exe"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (args.output if args.output.is_absolute() else workspace / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.relabel_existing:
        return relabel_existing_run(workspace, output, args)
    started = time.time()
    contract = load_contract(workspace)
    files = file_inventory(workspace, output, args.workers)
    observations = parse_metadata(workspace, files, contract)
    attach_metadata(files, observations)
    cache_path = output / "image_features_cache.jsonl"
    images = load_image_feature_cache(cache_path, files) if args.reuse_image_cache else None
    cache_reused = images is not None
    if images is None:
        images = decode_images(files, args.workers)
        by_path = {str(rec["path"].resolve()): rec for rec in files}
        for image in images:
            metadata = by_path.get(str(image["path"].resolve()), {})
            for key in ("roles", "session_ids", "ancestry_ids", "sequence_ids", "parent_event_ids", "manifest_paths"):
                image[key] = metadata.get(key, [])
        dump_image_feature_cache(cache_path, images, files)
        print(f"[cache] wrote {cache_path}", flush=True)
    else:
        print(f"[cache] reused {cache_path}", flush=True)
    duplicates = duplicate_groups(files, images)
    near_limits: dict[str, Any] = {}
    near = near_groups(
        images,
        args.phash_threshold,
        bucket_cap=max(1, args.near_bucket_cap),
        candidate_cap=max(1, args.near_candidate_cap),
        stats=near_limits,
    )
    overlaps = overlap_rows(obs_index(observations), files, images, near)
    adjacent = adjacent_rows(observations)
    video_rows = probe_videos(files, args.ffprobe if args.ffprobe else None)
    run = {
        "schema_version": "blindassist.data_contamination_audit.run.v1",
        "generated_at": now_utc(), "workspace": str(workspace), "output": str(output),
        "workers": args.workers, "phash_threshold": args.phash_threshold,
        "near_bucket_cap": args.near_bucket_cap, "near_candidate_cap": args.near_candidate_cap,
        "near_limits": near_limits, "image_feature_cache": str(cache_path),
        "image_feature_cache_reused": cache_reused,
        "ffprobe": str(args.ffprobe), "input_file_count": len(files),
        "input_bytes": sum(int(rec["size_bytes"]) for rec in files),
        "image_count": len(images), "video_count": len(video_rows),
        "observation_count": len(observations),
        "image_decode_error_count": sum(bool(rec.get("decode_error")) for rec in images),
        "contract_files": contract["files"],
        "limits": [
            "Pillow RGB decode and ffprobe metadata probe; no video frame extraction is assumed unless manifest identity exists.",
            "pHash uses 8x8 low-frequency DCT with CLI Hamming threshold.",
            "Manifest claimed hashes are distinct from verified local file hashes.",
        ],
    }
    dump_jsonl(output / "asset_inventory.jsonl", ({
        **{key: value for key, value in rec.items() if key != "path"},
        "path": rec["rel_path"],
    } for rec in files))
    dump_jsonl(output / "manifest_observations.jsonl", observations)
    dump_jsonl(output / "video_probe.jsonl", video_rows)
    dump_csv(output / "exact_duplicates.csv", duplicates, [
        "duplicate_group_id", "match_basis", "sha256", "rgb_pixel_sha256",
        "pixel_domain", "size_bytes", "width", "height", "member_count",
        "different_filename", "asset_classes", "roles", "sessions",
        "parent_events", "paths",
    ])
    dump_csv(output / "near_duplicates.csv", near, [
        "near_group_id", "relation_type", "match_transform_evidence",
        "phash_threshold", "sampled_min_hamming", "sampled_max_hamming",
        "member_count", "cross_role", "cross_session", "roles", "sessions",
        "representative_paths", "edge_sample", "edge_count_observed",
    ])
    dump_csv(output / "session_overlap.csv", overlaps, [
        "overlap_id", "overlap_type", "identifier", "left_role", "right_role",
        "left_count", "right_count", "left_sessions", "right_sessions",
        "left_parents", "right_parents", "left_manifests", "right_manifests",
        "severity", "evidence",
    ])
    dump_csv(output / "adjacent_frame_pairs.csv", adjacent, [
        "adjacency_id", "sequence_id", "left_frame_index", "right_frame_index",
        "left_session_id", "right_session_id", "left_parent_event_id",
        "right_parent_event_id", "left_roles", "right_roles", "cross_role",
        "left_path", "right_path", "leakage_status", "evidence",
    ])
    dump_json(output / "parent_ancestry_graph.json", graph(observations, overlaps, run))
    dump_json(output / "clean_split_candidates.json", candidates(observations, overlaps, contract))
    report(output, run, files, observations, images, duplicates, near, overlaps, adjacent, contract, video_rows)
    run["elapsed_seconds"] = round(time.time() - started, 3)
    run["output_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    dump_json(output / "run_metadata.json", run)
    print(json.dumps({
        "status": "completed", "output": str(output), "files": len(files),
        "images": len(images), "observations": len(observations),
        "exact_groups": len(duplicates), "near_groups": len(near),
        "overlap_rows": len(overlaps), "adjacent_pairs": len(adjacent),
        "elapsed_seconds": run["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
