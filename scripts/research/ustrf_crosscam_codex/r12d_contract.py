"""Shared validation helpers for the frozen R1.2d controlled detector study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CLASSES = ["traffic cone", "delineator", "bollard"]
MATRIX_SCHEMA = "blindassist_ustrf_crosscam_small_target_detector_matrix_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_bound_file(repo: Path, value: str, expected_sha256: str, label: str) -> Path:
    path = (repo / value).resolve()
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == expected_sha256, f"{label} SHA-256 mismatch")
    return path


def validate_matrix(path: Path, repo: Path) -> dict[str, Any]:
    matrix = load_json(path)
    require(matrix.get("schema") == MATRIX_SCHEMA, "R1.2d matrix schema drifted")
    require(matrix.get("classes") == CLASSES, "R1.2d class order drifted")
    parents = matrix["parents"]
    for stem in ("hypothesis", "event_protocol", "exact_frame_input"):
        resolve_bound_file(repo, parents[f"{stem}_path"], parents[f"{stem}_sha256"], stem)
    inference = matrix["frozen_inference"]
    require(inference == {
        "image_size": 640,
        "confidence": 0.05,
        "nms_iou": 0.45,
        "target_anchor_iou": 0.30,
        "maximum_detections": 100,
    }, "frozen inference thresholds drifted")
    seeds = matrix["training"]["seeds"]
    require(len(seeds) >= 3 and len(seeds) == len(set(seeds)), "multi-seed set invalid")
    arms = matrix["paired_arms"]
    require(len(arms) == 2 and {row["p2"] for row in arms} == {True, False}, "paired arms invalid")
    require(arms[0]["expected_strides"][0] == 4, "P2 arm lacks stride 4")
    require(arms[1]["expected_strides"][0] == 8, "control arm is not P3")
    metrics = matrix["event_metrics"]
    require(metrics["truth_blind_alert_generation_required"] is True, "truth-blind alert gate disabled")
    require(metrics["expected_class_may_only_be_read_during_scoring"] is True, "truth leakage allowed")
    require(matrix["data"]["event_frames_may_enter_training"] is False, "event frames leaked to training")
    require(matrix["authority"]["r13_inventory_read_authorized"] is False, "R1.3 unexpectedly opened")
    require(matrix["authority"]["production_model_replacement_authorized"] is False, "production authority drifted")
    return matrix


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(path) + ".sha256").write_text(sha256_file(path) + "\n", encoding="ascii")
