#!/usr/bin/env python3
"""Manifest-driven offline model matrix runner.

The runner owns the common contract: frame identity, model/config hashes,
streaming JSONL traces, progress and resume.  Adapters return any subset of
the common outputs.  Missing output is recorded as ``not_provided`` and is
never silently changed into zero, negative, or UNKNOWN truth.

The core uses only the Python standard library.  TFLite and Depth-Anything
adapters import optional dependencies only when selected.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[2]
DEFAULT_MANIFEST = MODULE_ROOT / "matrix_manifest.json"
TRACE_SCHEMA_VERSION = "blindassist.model_matrix.frame_trace.v1"
RECEIPT_SCHEMA_VERSION = "blindassist.model_matrix.receipt.v1"
RESUME_SCHEMA_VERSION = "blindassist.model_matrix.resume_state.v1"
OUTPUT_KEYS = (
    "detections",
    "segmentation_logits",
    "mask",
    "depth",
    "risk_output",
    "clearance",
)
ENVELOPE_STATUSES = {
    "present",
    "partial",
    "not_provided",
    "not_evaluable",
    "error",
}
ROW_STATUSES = {"OK", "ERROR", "NOT_EVALUABLE"}


class ConfigurationError(ValueError):
    """Raised when a manifest or registry is not safe to execute."""


class NotEvaluable(RuntimeError):
    """Raised when an adapter cannot run without changing the experiment."""


@dataclass(frozen=True)
class ArtifactPayload:
    path: Path
    sha256: str | None = None
    encoding: str = "source_artifact"
    origin: str = "dataset"


@dataclass(frozen=True)
class TensorPayload:
    value: Any
    encoding: str = "npy"
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Frame:
    dataset_id: str
    dataset_root: Path
    raw: dict[str, Any]
    source_id: str
    sequence_id: str
    frame_id: str
    frame_index: int
    source_frame_index: int
    timestamp_ms: int | float | None
    image_path: Path | None
    source_sha256: str | None
    event_id: str | None = None

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.source_id,
                self.sequence_id,
                self.frame_id,
                str(self.frame_index),
                str(self.source_frame_index),
                self.source_sha256 or "",
            )
        )

    def public_input(self, truth_fields: set[str]) -> dict[str, Any]:
        """Return a truth-sanitized adapter input."""

        payload = {
            key: copy.deepcopy(value)
            for key, value in self.raw.items()
            if key not in truth_fields
        }
        payload.update(
            {
                "dataset_id": self.dataset_id,
                "source_id": self.source_id,
                "sequence_id": self.sequence_id,
                "frame_id": self.frame_id,
                "frame_index": self.frame_index,
                "source_frame_index": self.source_frame_index,
                "timestamp_ms": self.timestamp_ms,
                "image_path": str(self.image_path) if self.image_path else None,
                "image_sha256": self.source_sha256,
            }
        )
        return payload

    def oracle_input(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.raw)
        payload.update(
            {
                "dataset_id": self.dataset_id,
                "source_id": self.source_id,
                "sequence_id": self.sequence_id,
                "frame_id": self.frame_id,
                "frame_index": self.frame_index,
                "source_frame_index": self.source_frame_index,
                "timestamp_ms": self.timestamp_ms,
                "image_path": str(self.image_path) if self.image_path else None,
                "image_sha256": self.source_sha256,
            }
        )
        return payload


@dataclass(frozen=True)
class AdapterContext:
    repo_root: Path
    output_root: Path
    job_root: Path
    model: dict[str, Any]
    dataset: dict[str, Any]
    job: dict[str, Any]
    resolution: dict[str, int]
    model_hash: str | None
    model_hash_kind: str
    config_hash: str


class Adapter:
    truth_fields_read: tuple[str, ...] = ()

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Missing JSONL file: {path}") from exc
    rows: list[dict[str, Any]] = []
    with stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ConfigurationError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_declared_path(
    repo_root: Path, value: str | Path, anchor: Path | None = None
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    if anchor is not None:
        candidates.append((anchor / path).resolve())
    candidates.append((repo_root / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{label} must be a list")
    return value


def load_configuration(
    manifest_path: Path, repo_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "blindassist.model_matrix.manifest.v1":
        raise ConfigurationError(f"Unsupported matrix manifest schema: {manifest_path}")

    model_registry_path = resolve_declared_path(
        repo_root,
        str(manifest.get("model_registry", "model_registry.json")),
        manifest_path.parent,
    )
    dataset_registry_path = resolve_declared_path(
        repo_root,
        str(manifest.get("dataset_registry", "dataset_registry.json")),
        manifest_path.parent,
    )
    trace_schema_path = resolve_declared_path(
        repo_root,
        str(manifest.get("trace_schema", "trace_schema.json")),
        manifest_path.parent,
    )
    model_registry = read_json(model_registry_path)
    dataset_registry = read_json(dataset_registry_path)
    trace_schema = read_json(trace_schema_path)
    if model_registry.get("schema_version") != "blindassist.model_matrix.model_registry.v1":
        raise ConfigurationError(f"Unsupported model registry schema: {model_registry_path}")
    if dataset_registry.get("schema_version") != "blindassist.model_matrix.dataset_registry.v1":
        raise ConfigurationError(f"Unsupported dataset registry schema: {dataset_registry_path}")
    if trace_schema.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported trace schema: {trace_schema_path}")

    models: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(
        require_list(model_registry.get("models"), "model registry models")
    ):
        if not isinstance(item, dict) or not isinstance(item.get("model_id"), str):
            raise ConfigurationError(f"model registry models[{index}] must have model_id")
        model_id = str(item["model_id"])
        if model_id in models:
            raise ConfigurationError(f"Duplicate model_id: {model_id}")
        models[model_id] = item

    datasets: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(
        require_list(dataset_registry.get("datasets"), "dataset registry datasets")
    ):
        if not isinstance(item, dict) or not isinstance(item.get("dataset_id"), str):
            raise ConfigurationError(
                f"dataset registry datasets[{index}] must have dataset_id"
            )
        dataset_id = str(item["dataset_id"])
        if dataset_id in datasets:
            raise ConfigurationError(f"Duplicate dataset_id: {dataset_id}")
        datasets[dataset_id] = item

    jobs = require_list(manifest.get("jobs"), "matrix manifest jobs")
    seen_jobs: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ConfigurationError(f"jobs[{index}] must be an object")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ConfigurationError(f"jobs[{index}] must have a non-empty job_id")
        if job_id in seen_jobs:
            raise ConfigurationError(f"Duplicate job_id: {job_id}")
        seen_jobs.add(job_id)
        if job.get("model_id") not in models:
            raise ConfigurationError(f"{job_id}: unknown model_id {job.get('model_id')}")
        if job.get("dataset_id") not in datasets:
            raise ConfigurationError(f"{job_id}: unknown dataset_id {job.get('dataset_id')}")
        if job.get("mode", "run") not in {"run", "preflight_only"}:
            raise ConfigurationError(f"{job_id}: unsupported mode {job.get('mode')}")
        resolution = job.get("resolution") or manifest.get("default_resolution")
        if not isinstance(resolution, dict) or not all(
            isinstance(resolution.get(key), int) and resolution[key] > 0
            for key in ("width", "height")
        ):
            raise ConfigurationError(
                f"{job_id}: resolution must contain positive width/height"
            )

    file_hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "model_registry_sha256": sha256_file(model_registry_path),
        "dataset_registry_sha256": sha256_file(dataset_registry_path),
        "trace_schema_sha256": sha256_file(trace_schema_path),
    }
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_model_registry_path"] = str(model_registry_path)
    manifest["_dataset_registry_path"] = str(dataset_registry_path)
    manifest["_trace_schema_path"] = str(trace_schema_path)
    return manifest, models, datasets, trace_schema, file_hashes


def load_dataset_frames(repo_root: Path, dataset: dict[str, Any]) -> list[Frame]:
    dataset_id = str(dataset["dataset_id"])
    root = resolve_declared_path(repo_root, str(dataset.get("root", ".")), repo_root)
    manifest_value = dataset.get("manifest_path")
    if manifest_value:
        manifest_path = resolve_declared_path(repo_root, str(manifest_value), root)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"Missing dataset manifest: {manifest_path}") from exc
    else:
        payload = {"frames": dataset.get("frames", [])}

    fmt = str(dataset.get("format", "jsonl"))
    raw_frames: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if fmt == "nested_event_json":
        if not isinstance(payload, dict):
            raise ConfigurationError(f"{dataset_id}: nested event manifest must be an object")
        for event in require_list(payload.get("events"), f"{dataset_id}.events"):
            if not isinstance(event, dict):
                raise ConfigurationError(f"{dataset_id}: event is not an object")
            for frame in require_list(event.get("frames"), f"{dataset_id}.event.frames"):
                if not isinstance(frame, dict):
                    raise ConfigurationError(f"{dataset_id}: frame is not an object")
                raw_frames.append((frame, event))
    elif fmt == "jsonl":
        if manifest_value:
            rows = read_jsonl(resolve_declared_path(repo_root, str(manifest_value), root))
        else:
            rows = require_list(payload.get("frames"), f"{dataset_id}.frames")
        raw_frames = [(row, {}) for row in rows if isinstance(row, dict)]
    elif fmt == "json_frames":
        rows = payload.get("frames") if isinstance(payload, dict) else payload
        raw_frames = [
            (row, {}) for row in require_list(rows, f"{dataset_id}.frames") if isinstance(row, dict)
        ]
    else:
        raise ConfigurationError(f"{dataset_id}: unsupported dataset format {fmt}")

    frames: list[Frame] = []
    for ordinal, (raw, event) in enumerate(raw_frames):
        frame_index = int(raw.get("frame_index", ordinal))
        source_frame_index = int(raw.get("source_frame_index", frame_index))
        source_id = str(
            raw.get("source_id")
            or raw.get("source_session_id")
            or event.get("source_session_id")
            or dataset.get("source_id")
            or dataset_id
        )
        sequence_id = str(
            raw.get("sequence_id")
            or event.get("sequence_id")
            or raw.get("session_id")
            or source_id
        )
        event_id_value = (
            raw.get("event_id")
            or raw.get("parent_event_id")
            or event.get("parent_event_id")
        )
        event_id = str(event_id_value) if event_id_value is not None else None
        frame_id = str(raw.get("frame_id") or raw.get("id") or f"{sequence_id}:{frame_index}")
        image_value = raw.get("image_path") or raw.get("image") or raw.get("file_name")
        image_path = resolve_declared_path(repo_root, str(image_value), root) if image_value else None
        source_sha256 = raw.get("image_sha256") or raw.get("source_rgb_sha256")
        if source_sha256 is None and image_path is not None and image_path.is_file():
            source_sha256 = sha256_file(image_path)
        timestamp = raw.get("timestamp_ms")
        if timestamp is not None and not isinstance(timestamp, (int, float)):
            raise ConfigurationError(f"{dataset_id}/{frame_id}: timestamp_ms must be numeric or null")
        merged_raw = copy.deepcopy(raw)
        for key in ("event_candidate_id", "parent_event_id", "source_session_id", "sequence_id"):
            if key not in merged_raw and key in event:
                merged_raw[key] = event[key]
        frames.append(
            Frame(
                dataset_id=dataset_id,
                dataset_root=root,
                raw=merged_raw,
                source_id=source_id,
                sequence_id=sequence_id,
                frame_id=frame_id,
                frame_index=frame_index,
                source_frame_index=source_frame_index,
                timestamp_ms=timestamp,
                image_path=image_path,
                source_sha256=str(source_sha256) if source_sha256 is not None else None,
                event_id=event_id,
            )
        )
    if not frames:
        raise ConfigurationError(f"{dataset_id}: dataset contains no frames")
    keys = [frame.key for frame in frames]
    if len(keys) != len(set(keys)):
        raise ConfigurationError(f"{dataset_id}: duplicate frame identity")
    return frames


def logical_model_hash(model: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "model_id": model.get("model_id"),
                "adapter": model.get("adapter"),
                "config": model.get("config", {}),
            }
        )
    )


def model_identity(
    repo_root: Path, model: dict[str, Any]
) -> tuple[str | None, str, list[dict[str, Any]]]:
    assets: list[str] = []
    if isinstance(model.get("asset"), str):
        assets.append(str(model["asset"]))
    assets.extend(str(value) for value in model.get("assets", []) if isinstance(value, str))
    if not assets:
        return logical_model_hash(model), "logical", []
    inventory: list[dict[str, Any]] = []
    missing = False
    for declared in assets:
        path = resolve_declared_path(repo_root, declared, repo_root)
        if path.is_file():
            inventory.append(
                {"path": declared, "sha256": sha256_file(path), "size": path.stat().st_size}
            )
        else:
            inventory.append({"path": declared, "sha256": None, "size": None})
            missing = True
    if missing:
        return None, "missing", inventory
    if len(inventory) == 1:
        return inventory[0]["sha256"], "asset", inventory
    return sha256_bytes(canonical_json_bytes(inventory)), "asset_set", inventory


def config_identity(
    model: dict[str, Any], dataset: dict[str, Any], job: dict[str, Any], resolution: dict[str, int]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "model_id": model.get("model_id"),
                "model_adapter": model.get("adapter"),
                "model_config": model.get("config", {}),
                "dataset_id": dataset.get("dataset_id"),
                "dataset_input_contract": dataset.get("input_contract", {}),
                "job": {
                    key: value
                    for key, value in job.items()
                    if key not in {"job_id", "mode", "frame_limit"}
                },
                "resolution": resolution,
            }
        )
    )


def declared_manifest_hash(repo_root: Path, dataset: dict[str, Any]) -> str | None:
    value = dataset.get("manifest_path")
    if not value:
        return None
    root = resolve_declared_path(repo_root, str(dataset.get("root", ".")), repo_root)
    path = resolve_declared_path(repo_root, str(value), root)
    return sha256_file(path) if path.is_file() else None


def job_fingerprint(
    manifest: dict[str, Any],
    model: dict[str, Any],
    dataset: dict[str, Any],
    job: dict[str, Any],
    model_hash: str | None,
    config_hash: str,
    dataset_manifest_hash: str | None,
    trace_schema_version: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "run_id": manifest.get("run_id"),
                "job_id": job.get("job_id"),
                "model_id": model.get("model_id"),
                "dataset_id": dataset.get("dataset_id"),
                "model_hash": model_hash,
                "config_hash": config_hash,
                "dataset_manifest_hash": dataset_manifest_hash,
                "trace_schema_version": trace_schema_version,
                "mode": job.get("mode", "run"),
                "adapter_override": job.get("adapter_override"),
                "model_override": job.get("model_override"),
            }
        )
    )


def empty_output(reason: str = "adapter_did_not_provide_output") -> dict[str, Any]:
    return {"status": "not_provided", "reason": reason}


def ensure_finite(value: Any, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            ensure_finite(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            ensure_finite(item, f"{label}.{key}")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return str(value)


def safe_output_relative(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def write_tensor_payload(
    payload: TensorPayload,
    job_root: Path,
    ordinal: int,
    output_name: str,
) -> dict[str, Any]:
    target_dir = job_root / "artifacts" / f"frame-{ordinal:06d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    value = payload.value
    shape = list(payload.shape) if payload.shape is not None else None
    dtype = payload.dtype
    if shape is None:
        raw_shape = getattr(value, "shape", None)
        if raw_shape is not None:
            try:
                shape = [int(item) for item in raw_shape]
            except (TypeError, ValueError):
                shape = None
    if dtype is None:
        raw_dtype = getattr(value, "dtype", None)
        dtype = str(raw_dtype) if raw_dtype is not None else None

    encoding = payload.encoding
    if encoding == "npy":
        try:
            import numpy as np  # type: ignore

            target = target_dir / f"{output_name}.npy"
            np.save(target, np.asarray(value), allow_pickle=False)
        except Exception:
            encoding = "json"
    if encoding == "json":
        target = target_dir / f"{output_name}.json"
        target.write_text(
            json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return {
        "path_scope": "job",
        "artifact_path": safe_output_relative(target, job_root),
        "sha256": sha256_file(target),
        "encoding": encoding,
        "dtype": dtype,
        "shape": shape,
    }


def artifact_reference(payload: ArtifactPayload, repo_root: Path) -> dict[str, Any]:
    if not payload.path.is_file():
        raise NotEvaluable(f"artifact_missing:{payload.path}")
    digest = payload.sha256 or sha256_file(payload.path)
    return {
        "path_scope": "repo",
        "artifact_path": safe_output_relative(payload.path, repo_root),
        "sha256": digest,
        "encoding": payload.encoding,
        "origin": payload.origin,
    }


def normalize_artifact_value(
    value: Any,
    *,
    output_name: str,
    ordinal: int,
    job_root: Path,
    repo_root: Path,
) -> Any:
    if isinstance(value, TensorPayload):
        return write_tensor_payload(value, job_root, ordinal, output_name)
    if isinstance(value, ArtifactPayload):
        return artifact_reference(value, repo_root)
    if isinstance(value, dict):
        result = dict(value)
        if "value" in result:
            result["artifact"] = normalize_artifact_value(
                result.pop("value"),
                output_name=output_name,
                ordinal=ordinal,
                job_root=job_root,
                repo_root=repo_root,
            )
        elif "artifact" in result:
            result["artifact"] = normalize_artifact_value(
                result["artifact"],
                output_name=output_name,
                ordinal=ordinal,
                job_root=job_root,
                repo_root=repo_root,
            )
        return result
    return value


def normalize_output(
    output_name: str,
    raw_value: Any,
    *,
    ordinal: int,
    job_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if raw_value is None:
        return empty_output()
    if isinstance(raw_value, dict) and "status" in raw_value:
        value = normalize_artifact_value(
            raw_value,
            output_name=output_name,
            ordinal=ordinal,
            job_root=job_root,
            repo_root=repo_root,
        )
        if value.get("status") not in ENVELOPE_STATUSES:
            raise ValueError(f"{output_name}: unsupported output status {value.get('status')}")
        return value
    if output_name == "detections":
        if isinstance(raw_value, list):
            return {"status": "present", "items": jsonable(raw_value), "count": len(raw_value)}
        if isinstance(raw_value, dict):
            items = raw_value.get("items", [])
            return {
                "status": "present",
                "items": jsonable(items),
                "count": int(raw_value.get("count", len(items))),
            }
    if output_name in {"segmentation_logits", "mask", "depth"}:
        artifact = normalize_artifact_value(
            raw_value,
            output_name=output_name,
            ordinal=ordinal,
            job_root=job_root,
            repo_root=repo_root,
        )
        return {"status": "present", "artifact": artifact}
    if isinstance(raw_value, dict):
        return {"status": "present", **jsonable(raw_value)}
    return {"status": "present", "value": jsonable(raw_value)}


def normalize_known(value: Any) -> str:
    if isinstance(value, bool):
        return "KNOWN" if value else "UNKNOWN"
    if isinstance(value, str) and value.upper() in {"KNOWN", "UNKNOWN"}:
        return value.upper()
    return "UNKNOWN"


class FixtureAdapter(Adapter):
    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        value = input_row.get("fixture_output") or input_row.get("model_output") or {}
        if not isinstance(value, dict):
            raise NotEvaluable("fixture_output_must_be_object")
        return copy.deepcopy(value)


class FixedRuleAdapter(Adapter):
    def __init__(self, model: dict[str, Any]) -> None:
        self.rule = str(model.get("config", {}).get("rule", "no_alert"))

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        if self.rule == "no_alert":
            return {
                "risk_output": {
                    "status": "present",
                    "raw_level": "NONE",
                    "stable_level": "NONE",
                    "active": False,
                    "direction": "NONE",
                    "rule_id": self.rule,
                },
                "known": "UNKNOWN",
            }
        if self.rule == "always_unknown":
            return {
                "risk_output": {
                    "status": "present",
                    "raw_level": "UNKNOWN",
                    "stable_level": "UNKNOWN",
                    "active": False,
                    "direction": "NONE",
                    "rule_id": self.rule,
                },
                "known": "UNKNOWN",
            }
        if self.rule == "frame_metadata":
            level = str(input_row.get("fixed_risk_level", "UNKNOWN")).upper()
            active = bool(input_row.get("fixed_alert", False))
            return {
                "risk_output": {
                    "status": "present",
                    "raw_level": level,
                    "stable_level": level,
                    "active": active,
                    "direction": str(input_row.get("fixed_direction", "NONE")),
                    "rule_id": self.rule,
                },
                "known": "KNOWN" if level not in {"UNKNOWN", "NONE"} else "UNKNOWN",
            }
        raise NotEvaluable(f"unknown_fixed_rule:{self.rule}")


class TruthMaskAdapter(Adapter):
    truth_fields_read = ("oracle_mask_path", "oracle_mask_sha256")

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        mask_value = input_row.get("oracle_mask_path") or input_row.get("mask_path")
        if not mask_value:
            raise NotEvaluable("truth_mask_path_missing")
        path = resolve_declared_path(self.repo_root, str(mask_value), frame.dataset_root)
        digest = input_row.get("oracle_mask_sha256") or input_row.get("mask_sha256")
        return {
            "mask": {
                "status": "present",
                "encoding": "png_class_id",
                "artifact": ArtifactPayload(
                    path=path,
                    sha256=str(digest) if digest else None,
                    encoding="png_class_id",
                    origin="dataset_oracle",
                ),
            },
            "known": "KNOWN",
            "adapter_metadata": {
                "evidence_role": "oracle_reference",
                "drives_alerts": False,
            },
        }


class LegacyTraceReplayAdapter(Adapter):
    def __init__(self, model: dict[str, Any], repo_root: Path) -> None:
        spec = model.get("adapter", {})
        config = model.get("config", {})
        self.repo_root = repo_root
        declared = spec.get("trace_path") or config.get("trace_path")
        if not isinstance(declared, str):
            raise NotEvaluable("legacy_trace_path_missing")
        trace_path = resolve_declared_path(repo_root, declared, repo_root)
        if not trace_path.is_file():
            raise NotEvaluable(f"legacy_trace_missing:{trace_path}")
        arm = spec.get("arm") or config.get("arm")
        self.rows: dict[tuple[str, int, str | None], dict[str, Any]] = {}
        for row in read_jsonl(trace_path):
            event_id = str(row.get("event_candidate_id") or row.get("parent_event_id") or "")
            frame_index = int(row.get("frame_index", -1))
            row_arm = str(row.get("arm")) if row.get("arm") is not None else None
            key = (event_id, frame_index, row_arm)
            if key in self.rows:
                raise NotEvaluable(f"legacy_trace_duplicate:{trace_path}:{key}")
            self.rows[key] = row
        self.arm = str(arm) if arm is not None else None
        self.trace_path = trace_path

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        event_id = str(
            frame.raw.get("event_candidate_id")
            or frame.raw.get("parent_event_id")
            or frame.event_id
            or ""
        )
        if self.arm is not None:
            legacy = self.rows.get((event_id, frame.frame_index, self.arm))
        else:
            candidates = [
                row
                for (row_event, row_frame, _), row in self.rows.items()
                if row_event == event_id and row_frame == frame.frame_index
            ]
            legacy = candidates[0] if candidates else None
        if legacy is None:
            raise NotEvaluable(f"legacy_trace_frame_missing:{event_id}:{frame.frame_index}")
        detection_count = legacy.get("detection_count")
        detections = None
        if detection_count is not None:
            detections = {
                "status": "partial",
                "count": int(detection_count),
                "items": [],
                "reason": "legacy_trace_contains_count_without_boxes",
            }
        risk = {
            "status": "present",
            "raw_level": legacy.get("raw_risk_level"),
            "stable_level": legacy.get("stable_risk_level"),
            "active": legacy.get("risk_event_active"),
            "direction": legacy.get("risk_direction"),
            "event_id": legacy.get("risk_event_id"),
            "event_state": legacy.get("risk_event_state"),
            "clear_reason": legacy.get("risk_event_clear_reason"),
            "actual_alert": legacy.get("actual_alert"),
        }
        return {
            "detections": detections,
            "risk_output": risk,
            "known": "UNKNOWN",
            "latency_ms": {"inference": legacy.get("perception_ms")},
            "adapter_metadata": {
                "source_trace": safe_output_relative(self.trace_path, self.repo_root),
                "source_trace_schema": legacy.get("schema_version"),
                "legacy_model_sha256": legacy.get("model_sha256"),
                "reused_without_rerun": True,
            },
        }


class TFLiteAdapter(Adapter):
    def __init__(self, model: dict[str, Any], repo_root: Path, resolution: dict[str, int]) -> None:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise NotEvaluable("numpy_not_installed") from exc
        self.np = np
        asset = model.get("asset")
        if not isinstance(asset, str):
            raise NotEvaluable("tflite_asset_missing")
        model_path = resolve_declared_path(repo_root, asset, repo_root)
        if not model_path.is_file():
            raise NotEvaluable(f"tflite_asset_not_found:{model_path}")
        self.model = model
        self.config = model.get("config", {})
        self.resolution = resolution
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore

            self.interpreter = Interpreter(model_path=str(model_path))
        except Exception:
            try:
                import tensorflow as tf  # type: ignore

                self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            except Exception as exc:
                raise NotEvaluable("no_tflite_interpreter") from exc
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

    def _shape(self, detail: dict[str, Any]) -> list[int]:
        shape = detail.get("shape")
        if hasattr(shape, "tolist"):
            shape = shape.tolist()
        return [int(value) for value in shape]

    def _prepare_image(self, frame: Frame) -> Any:
        if frame.image_path is None or not frame.image_path.is_file():
            raise NotEvaluable(f"image_missing:{frame.image_path}")
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise NotEvaluable("pillow_not_installed") from exc
        shape = self._shape(self.input_detail)
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise NotEvaluable(f"unsupported_tflite_input_shape:{shape}")
        height = shape[1] if shape[1] > 0 else self.resolution["height"]
        width = shape[2] if shape[2] > 0 else self.resolution["width"]
        with Image.open(frame.image_path) as image:
            image = image.convert("RGB").resize((width, height))
            array = self.np.asarray(image, dtype=self.np.float32) / self.np.float32(255.0)
        dtype = self.input_detail.get("dtype")
        if dtype == self.np.uint8 or dtype == self.np.int8:
            scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
            if scale:
                array = self.np.round(array / float(scale) + float(zero_point))
            array = array.clip(self.np.iinfo(dtype).min, self.np.iinfo(dtype).max).astype(dtype)
        else:
            array = array.astype(dtype or self.np.float32)
        return self.np.expand_dims(array, axis=0)

    def _read_outputs(self) -> list[Any]:
        values = []
        for detail in self.output_details:
            value = self.interpreter.get_tensor(detail["index"])
            dtype = detail.get("dtype")
            if dtype is not None and dtype != self.np.float32:
                scale, zero_point = detail.get("quantization", (0.0, 0))
                if scale:
                    value = (value.astype(self.np.float32) - float(zero_point)) * float(scale)
            values.append(value)
        return values

    def _decode_segmentation(self, values: list[Any]) -> dict[str, Any]:
        index = int(self.config.get("output_index", 0))
        if index < 0 or index >= len(values):
            raise NotEvaluable(f"segmentation_output_index:{index}")
        logits = self.np.asarray(values[index])
        if logits.ndim == 4 and logits.shape[0] == 1:
            logits = logits[0]
        classes = int(self.config.get("num_classes", 4))
        if logits.ndim == 3 and logits.shape[0] == classes and logits.shape[-1] != classes:
            logits = self.np.transpose(logits, (1, 2, 0))
        if logits.ndim != 3 or logits.shape[-1] != classes:
            raise NotEvaluable(f"unsupported_segmentation_output_shape:{list(logits.shape)}")
        mask = self.np.argmax(logits, axis=-1).astype(self.np.uint8)
        return {
            "segmentation_logits": TensorPayload(logits, dtype=str(logits.dtype)),
            "mask": TensorPayload(mask, dtype=str(mask.dtype), shape=tuple(int(v) for v in mask.shape)),
            "known": "KNOWN" if bool(self.np.isfinite(logits).all()) else "UNKNOWN",
            "adapter_metadata": {"runtime": "tflite", "output_count": len(values)},
        }

    @staticmethod
    def _iou(left: Sequence[float], right: Sequence[float]) -> float:
        x1 = max(float(left[0]), float(right[0]))
        y1 = max(float(left[1]), float(right[1]))
        x2 = min(float(left[2]), float(right[2]))
        y2 = min(float(left[3]), float(right[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
        area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
        union = area_left + area_right - intersection
        return intersection / union if union > 0.0 else 0.0

    def _decode_yolo(self, values: list[Any]) -> dict[str, Any]:
        index = int(self.config.get("output_index", 0))
        if index < 0 or index >= len(values):
            raise NotEvaluable(f"yolo_output_index:{index}")
        output = self.np.asarray(values[index])
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise NotEvaluable(f"unsupported_yolo_output_shape:{list(output.shape)}")
        if output.shape[-1] == 6:
            rows = output
            detections = [
                {
                    "class_id": int(row[5]),
                    "score": float(row[4]),
                    "bbox_xyxy": [float(item) for item in row[:4]],
                    "coordinate_space": "model",
                }
                for row in rows
                if float(row[4]) >= float(self.config.get("confidence_threshold", 0.35))
            ]
        else:
            rows = output.T if output.shape[0] < output.shape[1] else output
            if rows.shape[1] < 6:
                raise NotEvaluable(f"unsupported_yolo_channel_count:{list(rows.shape)}")
            class_scores = rows[:, 4:]
            class_ids = self.np.argmax(class_scores, axis=1)
            scores = class_scores[self.np.arange(len(rows)), class_ids]
            threshold = float(self.config.get("confidence_threshold", 0.35))
            detections = []
            for row, class_id, score in zip(rows, class_ids, scores):
                if float(score) < threshold:
                    continue
                cx, cy, width, height = [float(item) for item in row[:4]]
                detections.append(
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "bbox_xyxy": [
                            cx - width / 2.0,
                            cy - height / 2.0,
                            cx + width / 2.0,
                            cy + height / 2.0,
                        ],
                        "coordinate_space": "model",
                    }
                )
        detections.sort(key=lambda item: float(item["score"]), reverse=True)
        iou_threshold = float(self.config.get("iou_threshold", 0.45))
        kept: list[dict[str, Any]] = []
        for candidate in detections:
            if all(
                candidate["class_id"] != existing["class_id"]
                or self._iou(candidate["bbox_xyxy"], existing["bbox_xyxy"]) < iou_threshold
                for existing in kept
            ):
                kept.append(candidate)
        return {
            "detections": kept,
            "known": "KNOWN",
            "adapter_metadata": {"runtime": "tflite", "output_count": len(values)},
        }

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        tensor = self._prepare_image(frame)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        values = self._read_outputs()
        task = str(self.config.get("task", "segmentation"))
        return self._decode_yolo(values) if task == "detector" else self._decode_segmentation(values)


class DepthAnythingAdapter(Adapter):
    def __init__(self, model: dict[str, Any], repo_root: Path) -> None:
        try:
            import cv2  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            raise NotEvaluable("depth_dependencies_not_installed") from exc
        self.cv2 = cv2
        self.torch = torch
        config = model.get("config", {})
        source_root = resolve_declared_path(repo_root, str(config.get("source_root", ".")), repo_root)
        checkpoint = resolve_declared_path(repo_root, str(model.get("asset", "")), repo_root)
        if not source_root.is_dir():
            raise NotEvaluable(f"depth_source_root_missing:{source_root}")
        if not checkpoint.is_file():
            raise NotEvaluable(f"depth_checkpoint_missing:{checkpoint}")
        sys.path.insert(0, str(source_root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except Exception as exc:
            raise NotEvaluable("depth_anything_v2_source_import_failed") from exc
        encoder = str(config.get("encoder", "vits"))
        model_config = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        if encoder not in model_config:
            raise NotEvaluable(f"unsupported_depth_encoder:{encoder}")
        self.model = DepthAnythingV2(**model_config[encoder])
        self.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        self.model.eval()
        self.input_size = int(config.get("input_size", 252))

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        if frame.image_path is None or not frame.image_path.is_file():
            raise NotEvaluable(f"image_missing:{frame.image_path}")
        image = self.cv2.imread(str(frame.image_path))
        if image is None:
            raise NotEvaluable(f"image_decode_failed:{frame.image_path}")
        with self.torch.no_grad():
            depth = self.model.infer_image(image, input_size=self.input_size)
        finite = bool(self.torch.isfinite(self.torch.as_tensor(depth)).all())
        return {
            "depth": TensorPayload(depth, dtype="float32"),
            "known": "KNOWN" if finite else "UNKNOWN",
            "adapter_metadata": {"runtime": "depth_anything_v2", "input_size": self.input_size},
        }


class PythonCallableAdapter(Adapter):
    """Load a future adapter from a manifest-declared Python callable."""

    def __init__(self, model: dict[str, Any], context: AdapterContext) -> None:
        target = model.get("adapter", {}).get("target")
        if not isinstance(target, str) or ":" not in target:
            raise NotEvaluable("python_callable_target_missing")
        module_name, callable_name = target.rsplit(":", 1)
        if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
            path = resolve_declared_path(context.repo_root, module_name, context.repo_root)
            if not path.is_file():
                raise NotEvaluable(f"python_callable_module_missing:{path}")
            unique_name = f"blindassist_matrix_{sha256_bytes(str(path).encode())[:12]}"
            spec = importlib.util.spec_from_file_location(unique_name, path)
            if spec is None or spec.loader is None:
                raise NotEvaluable(f"python_callable_import_failed:{path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise NotEvaluable(f"python_callable_import_failed:{module_name}") from exc
        factory = getattr(module, callable_name, None)
        if not callable(factory):
            raise NotEvaluable(f"python_callable_missing:{target}")
        try:
            self.delegate = factory(context)
        except TypeError:
            self.delegate = factory(context.model, context.dataset, context.job)
        self._infer = getattr(self.delegate, "infer", self.delegate)
        if not callable(self._infer):
            raise NotEvaluable(f"python_callable_not_inferable:{target}")
        self.truth_fields_read = tuple(getattr(self.delegate, "truth_fields_read", ()))

    def infer(self, frame: Frame, input_row: dict[str, Any]) -> dict[str, Any]:
        result = self._infer(input_row)
        if not isinstance(result, dict):
            raise NotEvaluable("python_callable_result_must_be_object")
        return result

    def close(self) -> None:
        close = getattr(self.delegate, "close", None)
        if callable(close):
            close()


def make_adapter(model: dict[str, Any], context: AdapterContext) -> Adapter:
    spec = model.get("adapter") or {}
    if not isinstance(spec, dict):
        raise NotEvaluable("adapter_spec_must_be_object")
    kind = str(spec.get("kind", ""))
    if kind == "fixture":
        return FixtureAdapter()
    if kind == "fixed_rule":
        return FixedRuleAdapter(model)
    if kind == "truth_mask":
        return TruthMaskAdapter(context.repo_root)
    if kind == "legacy_trace_replay":
        return LegacyTraceReplayAdapter(model, context.repo_root)
    if kind == "tflite":
        return TFLiteAdapter(model, context.repo_root, context.resolution)
    if kind == "depth_anything_v2":
        return DepthAnythingAdapter(model, context.repo_root)
    if kind == "python_callable":
        return PythonCallableAdapter(model, context)
    if kind == "not_wired":
        raise NotEvaluable(str(spec.get("reason", "adapter_not_wired")))
    raise NotEvaluable(f"unsupported_adapter_kind:{kind}")


def validate_trace_row(row: dict[str, Any], schema: dict[str, Any]) -> None:
    row_schema = schema.get("row", {})
    for field in row_schema.get("required", []):
        if field not in row:
            raise ConfigurationError(f"trace row missing required field: {field}")
    if row.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ConfigurationError(f"trace row schema mismatch: {row.get('schema_version')}")
    if row.get("status") not in ROW_STATUSES:
        raise ConfigurationError(f"trace row status mismatch: {row.get('status')}")
    if row.get("known_status") not in {"KNOWN", "UNKNOWN"}:
        raise ConfigurationError(f"trace row known_status mismatch: {row.get('known_status')}")
    outputs = row.get("outputs")
    if not isinstance(outputs, dict):
        raise ConfigurationError("trace row outputs must be an object")
    for key in OUTPUT_KEYS:
        output = outputs.get(key)
        if not isinstance(output, dict) or output.get("status") not in ENVELOPE_STATUSES:
            raise ConfigurationError(f"trace row output envelope mismatch: {key}")
    for key in ("model_hash", "config_hash", "source_sha256"):
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ConfigurationError(f"trace row hash mismatch: {key}")
    ensure_finite(row.get("latency_ms", {}), "latency_ms")


def build_trace_row(
    *,
    manifest: dict[str, Any],
    model: dict[str, Any],
    dataset: dict[str, Any],
    job: dict[str, Any],
    frame: Frame,
    ordinal: int,
    model_hash: str | None,
    model_hash_kind: str,
    config_hash: str,
    raw: dict[str, Any],
    wall_ms: float,
    job_root: Path,
    repo_root: Path,
    trace_schema: dict[str, Any],
    error: Exception | None = None,
    truth_fields_read: Sequence[str] = (),
) -> dict[str, Any]:
    outputs = {
        key: normalize_output(
            key,
            raw.get(key),
            ordinal=ordinal,
            job_root=job_root,
            repo_root=repo_root,
        )
        for key in OUTPUT_KEYS
    }
    reported_latency = raw.get("latency_ms")
    if not isinstance(reported_latency, dict):
        reported_latency = {}
    inference_ms = reported_latency.get("inference")
    if not isinstance(inference_ms, (int, float)) or not math.isfinite(float(inference_ms)):
        inference_ms = wall_ms
    total_ms = reported_latency.get("total")
    if not isinstance(total_ms, (int, float)) or not math.isfinite(float(total_ms)):
        total_ms = wall_ms
    latency = {
        "preprocess": reported_latency.get("preprocess"),
        "inference": float(inference_ms),
        "postprocess": reported_latency.get("postprocess"),
        "adapter_wall": float(wall_ms),
        "total": float(total_ms),
    }
    row: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "status": "ERROR" if error is not None else "OK",
        "run_id": str(manifest["run_id"]),
        "job_id": str(job["job_id"]),
        "model_id": str(model["model_id"]),
        "model_family": str(model.get("family", "unknown")),
        "model_hash": model_hash,
        "model_hash_kind": model_hash_kind,
        "config_hash": config_hash,
        "dataset_id": str(dataset["dataset_id"]),
        "source_id": frame.source_id,
        "sequence_id": frame.sequence_id,
        "frame_id": frame.frame_id,
        "frame_index": frame.frame_index,
        "source_frame_index": frame.source_frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "source_sha256": frame.source_sha256,
        "frame_identity": {
            "key": frame.key,
            "source_id": frame.source_id,
            "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "source_frame_index": frame.source_frame_index,
            "timestamp_ms": frame.timestamp_ms,
            "image_path": safe_output_relative(frame.image_path, repo_root),
        },
        "outputs": outputs,
        "known_status": normalize_known(raw.get("known_status", raw.get("known"))),
        "clearance_status": outputs["clearance"].get("status"),
        "latency_ms": latency,
        "truth_fields_read": sorted(set(str(item) for item in truth_fields_read)),
        "evidence_role": str(model.get("evidence_role", "development")),
        "adapter_metadata": jsonable(raw.get("adapter_metadata", {})),
        "frame_ordinal": ordinal,
    }
    if error is not None:
        row["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    validate_trace_row(row, trace_schema)
    return row


def read_existing_trace(
    trace_path: Path,
    expected_keys: Sequence[str],
    schema: dict[str, Any],
    static_identity: Mapping[str, Any],
) -> set[str]:
    if not trace_path.is_file():
        return set()
    seen: list[str] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"invalid existing trace at {trace_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ConfigurationError(f"existing trace row is not an object: {trace_path}:{line_number}")
            validate_trace_row(row, schema)
            for key, expected in static_identity.items():
                if row.get(key) != expected:
                    raise ConfigurationError(
                        f"existing trace identity drift at {trace_path}:{line_number}: {key}"
                    )
            key = row["frame_identity"]["key"]
            if key in seen:
                raise ConfigurationError(f"duplicate existing trace identity: {trace_path}:{key}")
            seen.append(key)
    if seen != list(expected_keys[: len(seen)]):
        raise ConfigurationError(
            f"existing trace is not a contiguous prefix: {trace_path}; "
            f"found {len(seen)} rows for expected {len(expected_keys)}"
        )
    return set(seen)


def update_resume_state(
    state: dict[str, Any],
    *,
    job_id: str,
    fingerprint: str,
    status: str,
    expected_count: int,
    completed_count: int,
    last_key: str | None,
    trace_path: Path,
    output_root: Path,
    reason: str | None = None,
) -> None:
    state.setdefault("jobs", {})[job_id] = {
        "fingerprint": fingerprint,
        "status": status,
        "expected_frame_count": expected_count,
        "completed_frame_count": completed_count,
        "last_frame_key": last_key,
        "trace_path": safe_output_relative(trace_path, output_root),
        "reason": reason,
        "updated_at_utc": utc_now(),
    }


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def validate_configuration(
    manifest: dict[str, Any],
    models: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    trace_schema: dict[str, Any],
    repo_root: Path,
    selected_jobs: set[str] | None = None,
) -> dict[str, Any]:
    jobs = [
        job
        for job in manifest["jobs"]
        if selected_jobs is None or job["job_id"] in selected_jobs
    ]
    if selected_jobs:
        missing = selected_jobs - {job["job_id"] for job in jobs}
        if missing:
            raise ConfigurationError(f"unknown selected job(s): {sorted(missing)}")
    dataset_summaries: dict[str, dict[str, Any]] = {}
    for dataset_id in sorted({str(job["dataset_id"]) for job in jobs}):
        dataset = datasets[dataset_id]
        root = resolve_declared_path(repo_root, str(dataset.get("root", ".")), repo_root)
        manifest_value = dataset.get("manifest_path")
        if manifest_value:
            path = resolve_declared_path(repo_root, str(manifest_value), root)
            if not path.is_file():
                raise ConfigurationError(f"{dataset_id}: dataset manifest missing: {path}")
        frames = load_dataset_frames(repo_root, dataset)
        dataset_summaries[dataset_id] = {
            "frame_count": len(frames),
            "manifest_sha256": declared_manifest_hash(repo_root, dataset),
            "root": str(root),
        }
    job_summaries = []
    for job in jobs:
        model = models[str(job["model_id"])]
        dataset = datasets[str(job["dataset_id"])]
        model_hash, model_hash_kind, inventory = model_identity(repo_root, model)
        resolution = job.get("resolution") or manifest.get("default_resolution")
        config_hash = config_identity(model, dataset, job, resolution)
        job_summaries.append(
            {
                "job_id": job["job_id"],
                "model_id": model["model_id"],
                "dataset_id": dataset["dataset_id"],
                "mode": job.get("mode", "run"),
                "adapter": model.get("adapter", {}).get("kind"),
                "model_hash": model_hash,
                "model_hash_kind": model_hash_kind,
                "asset_inventory": inventory,
                "config_hash": config_hash,
                "frame_count": dataset_summaries[dataset["dataset_id"]]["frame_count"],
            }
        )
    return {
        "manifest": str(manifest["_manifest_path"]),
        "trace_schema_version": trace_schema.get("schema_version"),
        "job_count": len(jobs),
        "jobs": job_summaries,
        "datasets": dataset_summaries,
    }


def output_root_from_manifest(
    manifest: dict[str, Any], repo_root: Path, override: str | None
) -> Path:
    declared = override or str(
        manifest.get("output_root", "artifacts.local/evidence/model-matrix-r0")
    )
    output_root = resolve_declared_path(repo_root, declared, repo_root)
    artifacts_root = (repo_root / "artifacts.local").resolve()
    if output_root == artifacts_root or artifacts_root not in output_root.parents:
        raise ConfigurationError(
            f"output root must be a child of artifacts.local: {output_root}"
        )
    return output_root


def load_or_initialize_state(
    path: Path, manifest: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    if path.is_file():
        state = read_json(path)
        if state.get("template") is True:
            raise ConfigurationError(
                f"{path} is the tracked template; use the runtime state under the output root"
            )
        if state.get("schema_version") != RESUME_SCHEMA_VERSION:
            raise ConfigurationError(f"unsupported resume state schema: {path}")
        for key in (
            "run_id",
            "manifest_sha256",
            "model_registry_sha256",
            "dataset_registry_sha256",
            "trace_schema_sha256",
        ):
            expected = manifest.get("run_id") if key == "run_id" else hashes.get(key)
            if state.get(key) != expected:
                raise ConfigurationError(f"resume state identity drift: {key}")
        return state
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "template": False,
        "run_id": manifest["run_id"],
        "manifest_sha256": hashes["manifest_sha256"],
        "model_registry_sha256": hashes["model_registry_sha256"],
        "dataset_registry_sha256": hashes["dataset_registry_sha256"],
        "trace_schema_sha256": hashes["trace_schema_sha256"],
        "status": "RUNNING",
        "jobs": {},
        "created_at_utc": utc_now(),
    }


def run_job(
    *,
    manifest: dict[str, Any],
    model: dict[str, Any],
    dataset: dict[str, Any],
    job: dict[str, Any],
    trace_schema: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    state: dict[str, Any],
    progress_path: Path,
    max_frames: int | None,
    resume: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    job_root = output_root / "jobs" / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    trace_path = job_root / "trace.jsonl"
    receipt_path = job_root / "receipt.json"
    resolution = job.get("resolution") or manifest.get("default_resolution")
    model_hash, model_hash_kind, asset_inventory = model_identity(repo_root, model)
    config_hash = config_identity(model, dataset, job, resolution)
    frames = load_dataset_frames(repo_root, dataset)
    if max_frames is not None:
        frames = frames[: max(0, max_frames)]
    expected_keys = [frame.key for frame in frames]
    dataset_manifest_hash = declared_manifest_hash(repo_root, dataset)
    fingerprint = job_fingerprint(
        manifest,
        model,
        dataset,
        job,
        model_hash,
        config_hash,
        dataset_manifest_hash,
        str(trace_schema["schema_version"]),
    )
    static_identity = {
        "run_id": manifest["run_id"],
        "job_id": job_id,
        "model_id": model["model_id"],
        "model_hash": model_hash,
        "config_hash": config_hash,
        "dataset_id": dataset["dataset_id"],
    }
    completed: set[str] = set()
    if resume:
        completed = read_existing_trace(trace_path, expected_keys, trace_schema, static_identity)
    elif trace_path.exists():
        raise ConfigurationError(f"--no-resume refuses existing trace: {trace_path}")

    append_progress(
        progress_path,
        {
            "event": "job_start",
            "job_id": job_id,
            "fingerprint": fingerprint,
            "expected_frame_count": len(frames),
            "already_completed": len(completed),
            "timestamp_utc": utc_now(),
        },
    )
    update_resume_state(
        state,
        job_id=job_id,
        fingerprint=fingerprint,
        status="RUNNING",
        expected_count=len(frames),
        completed_count=len(completed),
        last_key=expected_keys[len(completed) - 1] if completed else None,
        trace_path=trace_path,
        output_root=output_root,
    )

    if len(completed) == len(frames) and trace_path.is_file():
        trace_hash = sha256_file(trace_path)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "job_id": job_id,
            "status": "COMPLETE",
            "model_id": model["model_id"],
            "dataset_id": dataset["dataset_id"],
            "model_hash": model_hash,
            "model_hash_kind": model_hash_kind,
            "config_hash": config_hash,
            "job_fingerprint": fingerprint,
            "asset_inventory": asset_inventory,
            "expected_frame_count": len(frames),
            "completed_frame_count": len(completed),
            "error_row_count": 0,
            "trace_file": safe_output_relative(trace_path, output_root),
            "trace_sha256": trace_hash,
            "reused_existing_trace": True,
            "created_at_utc": utc_now(),
        }
        write_json_atomic(receipt_path, receipt)
        update_resume_state(
            state,
            job_id=job_id,
            fingerprint=fingerprint,
            status="COMPLETE",
            expected_count=len(frames),
            completed_count=len(completed),
            last_key=expected_keys[-1] if expected_keys else None,
            trace_path=trace_path,
            output_root=output_root,
        )
        write_json_atomic(output_root / "resume_state.json", state)
        append_progress(
            progress_path,
            {"event": "job_end", "job_id": job_id, "status": "COMPLETE", "reused_existing_trace": True, "timestamp_utc": utc_now()},
        )
        return receipt

    if job.get("mode", "run") == "preflight_only":
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "job_id": job_id,
            "status": "PRECHECK_ONLY",
            "reason": job.get("reason", "manifest_preflight_only"),
            "model_id": model["model_id"],
            "dataset_id": dataset["dataset_id"],
            "model_hash": model_hash,
            "model_hash_kind": model_hash_kind,
            "config_hash": config_hash,
            "asset_inventory": asset_inventory,
            "expected_frame_count": len(frames),
            "completed_frame_count": 0,
            "trace_file": None,
            "created_at_utc": utc_now(),
        }
        write_json_atomic(receipt_path, receipt)
        update_resume_state(
            state,
            job_id=job_id,
            fingerprint=fingerprint,
            status="PRECHECK_ONLY",
            expected_count=len(frames),
            completed_count=0,
            last_key=None,
            trace_path=trace_path,
            output_root=output_root,
            reason=str(receipt["reason"]),
        )
        write_json_atomic(output_root / "resume_state.json", state)
        append_progress(
            progress_path,
            {"event": "job_end", "job_id": job_id, "status": "PRECHECK_ONLY", "timestamp_utc": utc_now()},
        )
        return receipt

    context = AdapterContext(
        repo_root=repo_root,
        output_root=output_root,
        job_root=job_root,
        model=model,
        dataset=dataset,
        job=job,
        resolution=resolution,
        model_hash=model_hash,
        model_hash_kind=model_hash_kind,
        config_hash=config_hash,
    )
    try:
        adapter = make_adapter(model, context)
    except NotEvaluable as exc:
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "job_id": job_id,
            "status": "NOT_EVALUABLE",
            "reason": str(exc),
            "model_id": model["model_id"],
            "dataset_id": dataset["dataset_id"],
            "model_hash": model_hash,
            "model_hash_kind": model_hash_kind,
            "config_hash": config_hash,
            "asset_inventory": asset_inventory,
            "expected_frame_count": len(frames),
            "completed_frame_count": len(completed),
            "trace_file": safe_output_relative(trace_path, output_root) if trace_path.is_file() else None,
            "created_at_utc": utc_now(),
        }
        write_json_atomic(receipt_path, receipt)
        update_resume_state(
            state,
            job_id=job_id,
            fingerprint=fingerprint,
            status="NOT_EVALUABLE",
            expected_count=len(frames),
            completed_count=len(completed),
            last_key=expected_keys[len(completed) - 1] if completed else None,
            trace_path=trace_path,
            output_root=output_root,
            reason=str(exc),
        )
        write_json_atomic(output_root / "resume_state.json", state)
        append_progress(
            progress_path,
            {"event": "job_end", "job_id": job_id, "status": "NOT_EVALUABLE", "reason": str(exc), "timestamp_utc": utc_now()},
        )
        return receipt

    errors = 0
    completed_count = len(completed)
    try:
        with trace_path.open("a", encoding="utf-8", newline="\n") as trace_stream:
            for ordinal, frame in enumerate(frames):
                if frame.key in completed:
                    continue
                input_row = (
                    frame.oracle_input()
                    if isinstance(adapter, TruthMaskAdapter)
                    else frame.public_input(set(dataset.get("truth_fields", [])))
                )
                started = time.perf_counter()
                raw: dict[str, Any] = {}
                error: Exception | None = None
                try:
                    raw = adapter.infer(frame, input_row)
                    if not isinstance(raw, dict):
                        raise NotEvaluable("adapter_result_must_be_object")
                except Exception as exc:  # retain per-frame failure in the trace
                    error = exc
                    errors += 1
                    if fail_fast:
                        raise
                wall_ms = (time.perf_counter() - started) * 1000.0
                row = build_trace_row(
                    manifest=manifest,
                    model=model,
                    dataset=dataset,
                    job=job,
                    frame=frame,
                    ordinal=ordinal,
                    model_hash=model_hash,
                    model_hash_kind=model_hash_kind,
                    config_hash=config_hash,
                    raw=raw,
                    wall_ms=wall_ms,
                    job_root=job_root,
                    repo_root=repo_root,
                    trace_schema=trace_schema,
                    error=error,
                    truth_fields_read=getattr(adapter, "truth_fields_read", ()),
                )
                trace_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                trace_stream.flush()
                completed.add(frame.key)
                completed_count += 1
                update_resume_state(
                    state,
                    job_id=job_id,
                    fingerprint=fingerprint,
                    status="RUNNING",
                    expected_count=len(frames),
                    completed_count=completed_count,
                    last_key=frame.key,
                    trace_path=trace_path,
                    output_root=output_root,
                )
                write_json_atomic(output_root / "resume_state.json", state)
                append_progress(
                    progress_path,
                    {
                        "event": "frame_completed",
                        "job_id": job_id,
                        "frame_ordinal": ordinal,
                        "frame_key": frame.key,
                        "status": row["status"],
                        "timestamp_utc": utc_now(),
                    },
                )
    except Exception as exc:
        try:
            adapter.close()
        finally:
            update_resume_state(
                state,
                job_id=job_id,
                fingerprint=fingerprint,
                status="PARTIAL_ERROR",
                expected_count=len(frames),
                completed_count=completed_count,
                last_key=expected_keys[completed_count - 1] if completed_count else None,
                trace_path=trace_path,
                output_root=output_root,
                reason=f"runner_aborted:{type(exc).__name__}:{exc}",
            )
            write_json_atomic(output_root / "resume_state.json", state)
        raise
    else:
        adapter.close()

    status = "PARTIAL_ERROR" if errors else "COMPLETE"
    trace_hash = sha256_file(trace_path) if trace_path.is_file() else None
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "job_id": job_id,
        "status": status,
        "model_id": model["model_id"],
        "dataset_id": dataset["dataset_id"],
        "model_hash": model_hash,
        "model_hash_kind": model_hash_kind,
        "config_hash": config_hash,
        "job_fingerprint": fingerprint,
        "asset_inventory": asset_inventory,
        "expected_frame_count": len(frames),
        "completed_frame_count": completed_count,
        "error_row_count": errors,
        "trace_file": safe_output_relative(trace_path, output_root),
        "trace_sha256": trace_hash,
        "created_at_utc": utc_now(),
    }
    write_json_atomic(receipt_path, receipt)
    update_resume_state(
        state,
        job_id=job_id,
        fingerprint=fingerprint,
        status=status,
        expected_count=len(frames),
        completed_count=completed_count,
        last_key=expected_keys[completed_count - 1] if completed_count else None,
        trace_path=trace_path,
        output_root=output_root,
    )
    write_json_atomic(output_root / "resume_state.json", state)
    append_progress(
        progress_path,
        {"event": "job_end", "job_id": job_id, "status": status, "completed_frame_count": completed_count, "timestamp_utc": utc_now()},
    )
    return receipt


def run_matrix(
    *,
    manifest_path: Path,
    repo_root: Path,
    selected_jobs: set[str] | None = None,
    output_override: str | None = None,
    state_override: str | None = None,
    max_frames: int | None = None,
    resume: bool = True,
    fail_fast: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    manifest, models, datasets, trace_schema, hashes = load_configuration(
        manifest_path, repo_root
    )
    validation = validate_configuration(
        manifest, models, datasets, trace_schema, repo_root, selected_jobs
    )
    if validate_only:
        return {"status": "VALID", "validation": validation, "file_hashes": hashes}

    output_root = output_root_from_manifest(manifest, repo_root, output_override)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = resolve_declared_path(
        repo_root,
        state_override or str(output_root / "resume_state.json"),
        output_root,
    )
    if state_path != (output_root / "resume_state.json").resolve():
        raise ConfigurationError("resume state must live under the selected output root")
    state = load_or_initialize_state(state_path, manifest, hashes)
    progress_path = output_root / "progress.jsonl"
    jobs = [
        job
        for job in manifest["jobs"]
        if selected_jobs is None or job["job_id"] in selected_jobs
    ]
    receipts: list[dict[str, Any]] = []
    for job in jobs:
        model = models[str(job["model_id"])]
        dataset = datasets[str(job["dataset_id"])]
        try:
            receipt = run_job(
                manifest=manifest,
                model=model,
                dataset=dataset,
                job=job,
                trace_schema=trace_schema,
                repo_root=repo_root,
                output_root=output_root,
                state=state,
                progress_path=progress_path,
                max_frames=max_frames,
                resume=resume,
                fail_fast=fail_fast,
            )
        except Exception as exc:
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "run_id": manifest["run_id"],
                "job_id": job["job_id"],
                "status": "PARTIAL_ERROR",
                "model_id": model["model_id"],
                "dataset_id": dataset["dataset_id"],
                "reason": f"runner_exception:{type(exc).__name__}:{exc}",
                "created_at_utc": utc_now(),
            }
            job_root = output_root / "jobs" / str(job["job_id"])
            write_json_atomic(job_root / "receipt.json", receipt)
            if fail_fast:
                raise
        receipts.append(receipt)

    statuses = [receipt.get("status") for receipt in receipts]
    if any(status == "PARTIAL_ERROR" for status in statuses):
        overall = "PARTIAL_ERROR"
    elif any(status in {"NOT_EVALUABLE", "PRECHECK_ONLY"} for status in statuses):
        overall = "PARTIAL_NOT_EVALUABLE"
    else:
        overall = "COMPLETE"
    state["status"] = overall
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    run_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": overall,
        "manifest_sha256": hashes["manifest_sha256"],
        "model_registry_sha256": hashes["model_registry_sha256"],
        "dataset_registry_sha256": hashes["dataset_registry_sha256"],
        "trace_schema_sha256": hashes["trace_schema_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "output_root": safe_output_relative(output_root, repo_root),
        "job_count": len(receipts),
        "jobs": receipts,
        "created_at_utc": utc_now(),
    }
    write_json_atomic(output_root / "run_receipt.json", run_receipt)
    return run_receipt


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run a manifest-driven unified model matrix.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--job", action="append", dest="jobs", help="Run only this job_id; repeatable.")
    parser.add_argument("--output-root", help="Override manifest output_root; must stay under artifacts.local.")
    parser.add_argument("--state", help="Override runtime resume state path under output_root.")
    parser.add_argument("--max-frames", type=int, help="Bound each selected job to the first N frames.")
    parser.add_argument("--no-resume", action="store_true", help="Refuse existing traces.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first frame-level adapter error.")
    parser.add_argument("--validate-only", action="store_true", help="Validate registries and dataset identity without writing output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    try:
        result = run_matrix(
            manifest_path=args.manifest.resolve(),
            repo_root=args.repo_root.resolve(),
            selected_jobs=set(args.jobs) if args.jobs else None,
            output_override=args.output_root,
            state_override=args.state,
            max_frames=args.max_frames,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
            validate_only=args.validate_only,
        )
    except (ConfigurationError, NotEvaluable) as exc:
        print(
            json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
