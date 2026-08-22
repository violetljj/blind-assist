#!/usr/bin/env python3
"""Materialize a goal-before-pixels PA3 cohort from DeepDoors2 without manual curation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import zipfile
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    content_sha256,
    sha256,
)


ROOT_FOLDER_ID = "1SxVKeJ9RBcoJXHSHw-LWaLGG07BZT-b5"
DOORDETECT_REPOSITORY = "MiguelARD/DoorDetect-Dataset"
DOORDETECT_REVISION = "1db107fe1b808fc5712f898b35bde0976ba0c0af"
FACADEELEMENTS_MD5 = "48d232be28b04885e1b606767638d6c1"
ROSTER_SCHEMA = "blindassist_p1_pa3_automated_public_dataset_roster_v1"
CAPTURE_SCHEMA = "blindassist_p1_pa3_capture_manifest_v1"
TRUTH_SCHEMA = "blindassist_p1_pa3_truth_body_v1"
PRECEDENCE_MODE = "GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH"
CAPTURE_TIME_SEMANTICS = "FIRST_PROJECT_PIXEL_ACCESS_NOT_PHYSICAL_CAMERA_CAPTURE"
DOOR_RGB = np.asarray([192, 224, 192], dtype=np.uint8)


class PublicDatasetError(ValueError):
    """Raised when automatic public-dataset materialization violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicDatasetError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sealed_payload(payload: dict[str, Any], hash_key: str) -> dict[str, Any]:
    _require(hash_key not in payload, f"duplicate {hash_key}")
    payload[hash_key] = content_sha256(payload)
    return payload


def create_goal_intake(
    output_path: Path,
    *,
    episode_prefix: str,
    count: int,
    goal_text: str,
    goal_type: str,
    reference_mode: str,
    task_semantics: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    _require(not output_path.exists(), "goal intake is immutable and already exists")
    _require(count > 0, "goal intake count must be positive")
    _require(reference_mode in {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}, "goal intake reference mode is invalid")
    source_sha = hashlib.sha256(goal_text.encode("utf-8")).hexdigest()
    episodes = [{
        "episode_id": f"{episode_prefix}-{index:03d}",
        "goal_text_original": goal_text,
        "goal_recorded_at_utc": recorded_at_utc,
        "goal_source": {
            "authority": "PRODUCT_TASK_INPUT",
            "source_record_sha256": source_sha,
        },
        "goal_contract": {
            "goal_type": goal_type,
            "reference_mode": reference_mode,
            "task_semantics": task_semantics,
        },
    } for index in range(1, count + 1)]
    intake = {
        "schema_version": "blindassist_p1_pa3_c0_goal_intake_v1",
        "intake_id": f"{episode_prefix.upper()}-GOAL-INTAKE",
        "provenance_contract": {
            "truth_state_at_goal_recording": "NOT_CREATED",
            "capture_state_at_goal_recording": "NOT_STARTED",
            "allowed_source_authorities": ["PRODUCT_TASK_INPUT", "USER_TASK_INPUT"],
        },
        "episodes": episodes,
    }
    _atomic_json(output_path, intake)
    return intake


def _validate_seal(value: Mapping[str, Any], hash_key: str) -> None:
    declared = value.get(hash_key)
    _require(isinstance(declared, str) and len(declared) == 64, f"{hash_key} is missing")
    body = dict(value)
    body.pop(hash_key)
    _require(content_sha256(body) == declared, f"{hash_key} mismatch")


def _gdown_items() -> list[Any]:
    try:
        import gdown
    except ImportError as error:
        raise PublicDatasetError("gdown is required on PYTHONPATH for source metadata/download") from error
    items = gdown.download_folder(
        id=ROOT_FOLDER_ID,
        output="deepdoors2",
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    _require(items is not None, "DeepDoors2 folder metadata is unavailable")
    return list(items)


def freeze_roster(source_lock_path: Path, c0_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = _read_json(source_lock_path)
    c0 = _read_json(c0_path)
    _require(source_lock.get("created_before_project_pixel_access") is True, "source lock precedence is missing")
    _require(source_lock.get("created_before_private_annotation_access") is True, "source lock truth precedence is missing")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock C0 binding mismatch")
    take = int(source_lock.get("roster_rule", {}).get("take", 0))
    _require(take == len(c0.get("episodes", [])) and take > 0, "roster size must equal frozen C0 episodes")

    images: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    for item in _gdown_items():
        path = str(getattr(item, "path", "")).replace("\\", "/")
        name = PurePosixPath(path).name
        stem = PurePosixPath(path).stem
        if path.startswith("Door Detection|Segmentation/Images/") and name.lower().endswith(".png"):
            images[stem] = item
        elif path.startswith("Door Detection|Segmentation/Annotations/") and name.lower().endswith(".png"):
            annotations[stem] = item
    eligible = sorted(set(images) & set(annotations))
    _require(len(eligible) >= take, "DeepDoors2 has fewer paired files than the frozen roster requires")
    ordered = sorted(
        eligible,
        key=lambda stem: hashlib.sha256(str(getattr(images[stem], "path")).replace("\\", "/").encode("utf-8")).hexdigest(),
    )
    cases = []
    for index, stem in enumerate(ordered[:take], start=1):
        image_path = str(getattr(images[stem], "path")).replace("\\", "/")
        cases.append({
            "case_id": f"s0v4-auto-case-{index:03d}",
            "episode_id": f"s0v4-auto-{index:03d}",
            "source_stem": stem,
            "image_member_path": image_path,
            "image_member_path_sha256": hashlib.sha256(image_path.encode("utf-8")).hexdigest(),
            "image_drive_id": str(getattr(images[stem], "id")),
            "annotation_member_path": str(getattr(annotations[stem], "path")).replace("\\", "/"),
            "annotation_drive_id": str(getattr(annotations[stem], "id")),
        })
    roster = _sealed_payload({
        "schema_version": ROSTER_SCHEMA,
        "created_at_utc": _utc_now(),
        "created_before_selected_pixel_download": True,
        "created_before_selected_annotation_download": True,
        "private_truth_access": False,
        "source_lock_sha256": sha256(source_lock_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_pair_count": len(eligible),
        "selection_rule": source_lock["roster_rule"],
        "cases": cases,
    }, "roster_body_sha256")
    _atomic_json(output_path, roster)
    return roster


def _download(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise PublicDatasetError("gdown is required on PYTHONPATH for source metadata/download") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            return
        except Exception as error:
            raise PublicDatasetError(f"existing partial download is invalid: {destination}") from error
    try:
        result = gdown.download(id=file_id, output=str(destination), quiet=True)
    except Exception:
        result = None
    if result is None:
        import requests

        response = requests.get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            timeout=60,
        )
        _require(response.status_code == 200, f"download HTTP {response.status_code} for {destination.name}")
        _require(response.headers.get("content-type", "").lower().startswith("image/"), f"download is not an image for {destination.name}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(response.content)
        with Image.open(temporary) as image:
            image.verify()
        os.replace(temporary, destination)
    _require(destination.is_file(), f"download failed for {destination.name}")


def download_public(roster_path: Path, c0_path: Path, output_root: Path, capture_path: Path) -> dict[str, Any]:
    _require(not capture_path.exists(), "capture manifest is immutable and already exists")
    roster = _read_json(roster_path)
    c0 = _read_json(c0_path)
    _validate_seal(roster, "roster_body_sha256")
    _require(roster.get("private_truth_access") is False, "roster accessed private truth")
    _require(roster.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "roster C0 binding mismatch")
    goal_times = {row["episode_id"]: row["goal_provenance"]["goal_recorded_at_utc"] for row in c0["episodes"]}
    cases = []
    for raw in roster["cases"]:
        destination = (output_root / "public_images" / f"{raw['source_stem']}.png").resolve()
        _download(raw["image_drive_id"], destination)
        with Image.open(destination) as image:
            image.verify()
        accessed_at = _utc_now()
        _require(goal_times[raw["episode_id"]] < accessed_at, "goal must precede first project pixel access")
        cases.append({
            "case_id": raw["case_id"],
            "episode_id": raw["episode_id"],
            "capture_created_at_utc": accessed_at,
            "capture_time_semantics": CAPTURE_TIME_SEMANTICS,
            "source_capture_time_semantics": "UNKNOWN_PREEXISTING_PUBLIC_DATASET_CAPTURE",
            "image_path": str(destination),
            "image_sha256": sha256(destination),
        })
    capture = _sealed_payload({
        "schema_version": CAPTURE_SCHEMA,
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "source_role": "PREEXISTING_PUBLIC_DATASET_GOAL_BEFORE_PROJECT_PIXEL_ACCESS",
        "precedence_mode": PRECEDENCE_MODE,
        "physical_capture_after_goal_claimed": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "roster_body_sha256": roster["roster_body_sha256"],
        "cases": cases,
    }, "capture_manifest_body_sha256")
    _atomic_json(capture_path, capture)
    return capture


def download_private_truth(roster_path: Path, capture_path: Path, output_root: Path, truth_path: Path) -> dict[str, Any]:
    _require(not truth_path.exists(), "private truth is immutable and already exists")
    roster = _read_json(roster_path)
    capture = _read_json(capture_path)
    _validate_seal(roster, "roster_body_sha256")
    _validate_seal(capture, "capture_manifest_body_sha256")
    _require(capture.get("private_truth_access") is False, "capture manifest must be sealed before truth access")
    capture_by_case = {row["case_id"]: row for row in capture["cases"]}
    cases = []
    for raw in roster["cases"]:
        destination = (output_root / "private_annotations" / f"{raw['source_stem']}.png").resolve()
        _download(raw["annotation_drive_id"], destination)
        rgb = np.asarray(Image.open(destination).convert("RGB"), dtype=np.uint8)
        foreground = np.all(rgb == DOOR_RGB, axis=2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        boxes = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area > 0:
                boxes.append([float(x), float(y), float(x + width), float(y + height)])
        boxes.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
        _require(raw["case_id"] in capture_by_case, "truth case is absent from capture manifest")
        cases.append({
            "case_id": raw["case_id"],
            "reference_mode": "SET_VALUED",
            "target_visibility": "VISIBLE" if boxes else "NOT_VISIBLE",
            "legal_target_bboxes_xyxy": boxes,
            "private_annotation_sha256": sha256(destination),
        })
    truth = {
        "schema_version": TRUTH_SCHEMA,
        "truth_created_at_utc": _utc_now(),
        "primary_iou_threshold": 0.30,
        "diagnostic_iou_thresholds": [0.10, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "truth_source": "DEEPDOORS2_EXACT_COLOR_SEMANTIC_MASK_CONNECTED_COMPONENTS",
        "cases": cases,
    }
    _atomic_json(truth_path, truth)
    return truth


def freeze_doordetect_roster(source_lock_path: Path, c0_path: Path, output_path: Path) -> dict[str, Any]:
    import requests

    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = _read_json(source_lock_path)
    c0 = _read_json(c0_path)
    _require(source_lock.get("created_before_repository_tree_access") is True, "source lock tree precedence is missing")
    _require(source_lock.get("created_before_project_pixel_access") is True, "source lock pixel precedence is missing")
    _require(source_lock.get("created_before_private_label_access") is True, "source lock label precedence is missing")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock C0 binding mismatch")
    _require(source_lock.get("source", {}).get("repository_revision") == DOORDETECT_REVISION, "DoorDetect revision drift")
    take = int(source_lock.get("roster_rule", {}).get("take", 0))
    _require(take == len(c0.get("episodes", [])) and take > 0, "roster size must equal frozen C0 episodes")
    response = requests.get(
        f"https://api.github.com/repos/{DOORDETECT_REPOSITORY}/git/trees/{DOORDETECT_REVISION}?recursive=1",
        timeout=60,
        headers={"Accept": "application/vnd.github+json"},
    )
    _require(response.status_code == 200, f"GitHub tree HTTP {response.status_code}")
    tree = response.json().get("tree", [])
    _require(isinstance(tree, list), "GitHub tree response is invalid")
    images: dict[str, Mapping[str, Any]] = {}
    labels: dict[str, Mapping[str, Any]] = {}
    for item in tree:
        if not isinstance(item, Mapping) or item.get("type") != "blob":
            continue
        path = str(item.get("path", "")).replace("\\", "/")
        pure = PurePosixPath(path)
        if path.startswith("images/") and pure.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[pure.stem] = item
        elif path.startswith("labels/") and pure.suffix.lower() == ".txt":
            labels[pure.stem] = item
    eligible = sorted(set(images) & set(labels))
    _require(len(eligible) >= take, "DoorDetect has fewer paired files than the frozen roster requires")
    ordered = sorted(eligible, key=lambda stem: hashlib.sha256(str(images[stem]["path"]).encode("utf-8")).hexdigest())
    cases = []
    for index, stem in enumerate(ordered[:take], start=1):
        image_path = str(images[stem]["path"])
        cases.append({
            "case_id": f"s0v5-auto-case-{index:03d}",
            "episode_id": f"s0v4-auto-{index:03d}",
            "source_stem": stem,
            "image_path": image_path,
            "image_path_sha256": hashlib.sha256(image_path.encode("utf-8")).hexdigest(),
            "image_blob_sha": str(images[stem]["sha"]),
            "label_path": str(labels[stem]["path"]),
            "label_blob_sha": str(labels[stem]["sha"]),
        })
    roster = _sealed_payload({
        "schema_version": ROSTER_SCHEMA,
        "source_kind": "DOORDETECT_GITHUB_TREE",
        "repository": DOORDETECT_REPOSITORY,
        "repository_revision": DOORDETECT_REVISION,
        "created_at_utc": _utc_now(),
        "created_before_selected_pixel_download": True,
        "created_before_selected_label_download": True,
        "private_truth_access": False,
        "source_lock_sha256": sha256(source_lock_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_pair_count": len(eligible),
        "selection_rule": source_lock["roster_rule"],
        "cases": cases,
    }, "roster_body_sha256")
    _atomic_json(output_path, roster)
    return roster


def _download_url(url: str, destination: Path, expected_kind: str) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_kind == "image":
            with Image.open(destination) as image:
                image.verify()
        else:
            destination.read_text(encoding="utf-8")
        return
    response = requests.get(url, timeout=60)
    _require(response.status_code == 200, f"download HTTP {response.status_code} for {destination.name}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(response.content)
    if expected_kind == "image":
        with Image.open(temporary) as image:
            image.verify()
    else:
        temporary.read_text(encoding="utf-8")
    os.replace(temporary, destination)


def download_doordetect_public(roster_path: Path, c0_path: Path, output_root: Path, capture_path: Path) -> dict[str, Any]:
    _require(not capture_path.exists(), "capture manifest is immutable and already exists")
    roster = _read_json(roster_path)
    c0 = _read_json(c0_path)
    _validate_seal(roster, "roster_body_sha256")
    _require(roster.get("source_kind") == "DOORDETECT_GITHUB_TREE", "DoorDetect roster source mismatch")
    _require(roster.get("private_truth_access") is False, "roster accessed private truth")
    _require(roster.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "roster C0 binding mismatch")
    goal_times = {row["episode_id"]: row["goal_provenance"]["goal_recorded_at_utc"] for row in c0["episodes"]}
    cases = []
    for raw in roster["cases"]:
        suffix = PurePosixPath(raw["image_path"]).suffix.lower()
        destination = (output_root / "public_images" / f"{raw['source_stem']}{suffix}").resolve()
        url = f"https://raw.githubusercontent.com/{DOORDETECT_REPOSITORY}/{DOORDETECT_REVISION}/{raw['image_path']}"
        _download_url(url, destination, "image")
        accessed_at = _utc_now()
        _require(goal_times[raw["episode_id"]] < accessed_at, "goal must precede first project pixel access")
        cases.append({
            "case_id": raw["case_id"],
            "episode_id": raw["episode_id"],
            "capture_created_at_utc": accessed_at,
            "capture_time_semantics": CAPTURE_TIME_SEMANTICS,
            "source_capture_time_semantics": "UNKNOWN_PREEXISTING_PUBLIC_DATASET_CAPTURE",
            "image_path": str(destination),
            "image_sha256": sha256(destination),
        })
    capture = _sealed_payload({
        "schema_version": CAPTURE_SCHEMA,
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "source_role": "PREEXISTING_PUBLIC_DATASET_GOAL_BEFORE_PROJECT_PIXEL_ACCESS",
        "precedence_mode": PRECEDENCE_MODE,
        "physical_capture_after_goal_claimed": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "roster_body_sha256": roster["roster_body_sha256"],
        "cases": cases,
    }, "capture_manifest_body_sha256")
    _atomic_json(capture_path, capture)
    return capture


def download_doordetect_private_truth(roster_path: Path, capture_path: Path, output_root: Path, truth_path: Path) -> dict[str, Any]:
    _require(not truth_path.exists(), "private truth is immutable and already exists")
    roster = _read_json(roster_path)
    capture = _read_json(capture_path)
    _validate_seal(roster, "roster_body_sha256")
    _validate_seal(capture, "capture_manifest_body_sha256")
    _require(capture.get("private_truth_access") is False, "capture manifest must be sealed before label access")
    capture_by_case = {row["case_id"]: row for row in capture["cases"]}
    cases = []
    for raw in roster["cases"]:
        label_destination = (output_root / "private_labels" / f"{raw['source_stem']}.txt").resolve()
        label_url = f"https://raw.githubusercontent.com/{DOORDETECT_REPOSITORY}/{DOORDETECT_REVISION}/{raw['label_path']}"
        _download_url(label_url, label_destination, "text")
        captured = capture_by_case.get(raw["case_id"])
        _require(captured is not None, "truth case is absent from capture manifest")
        with Image.open(captured["image_path"]) as image:
            width, height = image.size
        boxes = []
        for line_number, line in enumerate(label_destination.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            _require(len(parts) == 5, f"invalid YOLO label at {raw['source_stem']}:{line_number}")
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
            _require(all(np.isfinite(value) for value in (center_x, center_y, box_width, box_height)), "non-finite YOLO label")
            if class_id != 0:
                continue
            x1 = (center_x - box_width / 2.0) * width
            y1 = (center_y - box_height / 2.0) * height
            x2 = (center_x + box_width / 2.0) * width
            y2 = (center_y + box_height / 2.0) * height
            _require(x2 > x1 and y2 > y1, "non-positive DoorDetect box")
            boxes.append([x1, y1, x2, y2])
        cases.append({
            "case_id": raw["case_id"],
            "reference_mode": "SET_VALUED",
            "target_visibility": "VISIBLE" if boxes else "NOT_VISIBLE",
            "legal_target_bboxes_xyxy": boxes,
            "private_label_sha256": sha256(label_destination),
        })
    truth = {
        "schema_version": TRUTH_SCHEMA,
        "truth_created_at_utc": _utc_now(),
        "primary_iou_threshold": 0.30,
        "diagnostic_iou_thresholds": [0.10, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "truth_source": "DOORDETECT_FROZEN_CLASS_ZERO_YOLO_BBOX",
        "cases": cases,
    }
    _atomic_json(truth_path, truth)
    return truth


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_facadeelements_archive(source_metadata_path: Path, output_path: Path, workers: int) -> Path:
    import requests

    metadata = _read_json(source_metadata_path)
    _require(metadata.get("created_before_archive_download") is True, "source metadata download precedence is missing")
    expected_size = int(metadata.get("size_bytes", 0))
    _require(expected_size > 0, "FacadeElements archive size is invalid")
    _require(metadata.get("checksum") == f"md5:{FACADEELEMENTS_MD5}", "FacadeElements checksum drift")
    url = str(metadata.get("download_url", ""))
    _require(url.startswith("https://zenodo.org/"), "FacadeElements download authority drift")
    _require(1 <= workers <= 16, "download workers must be between 1 and 16")
    if output_path.is_file() and output_path.stat().st_size == expected_size and _md5(output_path) == FACADEELEMENTS_MD5:
        return output_path
    parts_dir = output_path.with_suffix(output_path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    segment_size = (expected_size + workers - 1) // workers

    def fetch(index: int) -> Path:
        start = index * segment_size
        end = min(expected_size - 1, start + segment_size - 1)
        destination = parts_dir / f"part-{index:02d}.bin"
        expected = end - start + 1
        if destination.is_file() and destination.stat().st_size == expected:
            return destination
        response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120)
        _require(response.status_code == 206, f"FacadeElements range {index} HTTP {response.status_code}")
        _require(response.headers.get("content-range", "").startswith(f"bytes {start}-{end}/"), f"FacadeElements range {index} mismatch")
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        _require(temporary.stat().st_size == expected, f"FacadeElements range {index} size mismatch")
        os.replace(temporary, destination)
        return destination

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(fetch, range(workers)))
    temporary_archive = output_path.with_suffix(output_path.suffix + ".complete.tmp")
    with temporary_archive.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    _require(temporary_archive.stat().st_size == expected_size, "FacadeElements assembled archive size mismatch")
    _require(_md5(temporary_archive) == FACADEELEMENTS_MD5, "FacadeElements assembled archive MD5 mismatch")
    os.replace(temporary_archive, output_path)
    return output_path


def _normalized_class_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ").strip().lower())


def _yaml_names(raw: bytes) -> dict[int, str]:
    try:
        import yaml
    except ImportError as error:
        raise PublicDatasetError("PyYAML is required to read the public class schema") from error
    value = yaml.safe_load(raw.decode("utf-8"))
    _require(isinstance(value, Mapping), "data.yaml must contain an object")
    names = value.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, Mapping):
        return {int(index): str(name) for index, name in names.items()}
    raise PublicDatasetError("data.yaml names must be a list or mapping")


def freeze_facadeelements_roster(
    source_lock_path: Path,
    source_metadata_path: Path,
    c0_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = _read_json(source_lock_path)
    source_metadata = _read_json(source_metadata_path)
    c0 = _read_json(c0_path)
    strict_archive_precedence = source_lock.get("created_before_archive_member_access") is True
    public_schema_only_precedence = (
        source_lock.get("public_schema_access_only_before_selected_data") is True
        and source_lock.get("created_before_selected_project_pixel_access") is True
    )
    _require(strict_archive_precedence or public_schema_only_precedence, "source lock selected-data precedence is missing")
    _require(source_lock.get("created_before_private_label_access") is True, "source lock label precedence is missing")
    _require(
        source_metadata.get("created_before_archive_download") is True
        or source_metadata.get("archive_identity_verified_before_selected_data") is True,
        "source metadata selected-data precedence is missing",
    )
    _require(source_metadata.get("source_lock_sha256") == sha256(source_lock_path), "source metadata lock binding mismatch")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock C0 binding mismatch")
    _require(archive_path.is_file(), "FacadeElements archive is missing")
    _require(archive_path.stat().st_size == int(source_metadata.get("size_bytes", 0)), "FacadeElements archive size mismatch")
    _require(_md5(archive_path) == FACADEELEMENTS_MD5, "FacadeElements archive MD5 mismatch")
    take = int(source_lock.get("roster_rule", {}).get("take", 0))
    skip = int(source_lock.get("roster_rule", {}).get("skip", 0))
    _require(take == len(c0.get("episodes", [])) and take > 0, "roster size must equal frozen C0 episodes")
    _require(skip >= 0, "roster skip must be non-negative")
    roster_rule = source_lock.get("roster_rule", {})
    case_id_prefix = str(roster_rule.get("case_id_prefix", "s0v7-auto-case"))
    episode_id_prefix = str(roster_rule.get("episode_id_prefix", "s0v4-auto"))
    truth_rule = source_lock.get("private_truth_rule", {})
    reference_mode = str(truth_rule.get("reference_mode", "SET_VALUED"))
    target_selector = str(truth_rule.get("target_selector", "ALL_DOORS"))
    _require(reference_mode in {"UNIQUE", "SET_VALUED"}, "FacadeElements reference mode is unsupported")
    _require(target_selector in {"ALL_DOORS", "LEFTMOST_DOOR_X_CENTER"}, "FacadeElements target selector is unsupported")
    _require((reference_mode, target_selector) in {("SET_VALUED", "ALL_DOORS"), ("UNIQUE", "LEFTMOST_DOOR_X_CENTER")}, "FacadeElements reference/selector mismatch")

    with zipfile.ZipFile(archive_path) as archive:
        names = [name.replace("\\", "/") for name in archive.namelist() if not name.endswith("/")]
        yaml_members = [name for name in names if PurePosixPath(name).name.lower() == "data.yaml"]
        _require(len(yaml_members) == 1, "FacadeElements archive must contain exactly one data.yaml")
        class_names = _yaml_names(archive.read(yaml_members[0]))
        legal_names = set(source_lock["public_class_schema_rule"]["legal_class_names_normalized"])
        legal_class_ids = sorted(index for index, name in class_names.items() if _normalized_class_name(name) in legal_names)
        _require(legal_class_ids, "FacadeElements public schema has no legal door class")
        _require({_normalized_class_name(class_names[index]) for index in legal_class_ids} == legal_names, "FacadeElements legal class mapping is incomplete")
        image_by_key: dict[tuple[str, str], str] = {}
        label_by_key: dict[tuple[str, str], str] = {}
        for name in names:
            pure = PurePosixPath(name)
            parts = [part.lower() for part in pure.parts]
            if len(parts) < 3:
                continue
            if pure.suffix.lower() in {".jpg", ".jpeg", ".png"} and parts[-2] == "images":
                image_by_key[("/".join(parts[:-2]), pure.stem)] = name
            elif pure.suffix.lower() == ".txt" and parts[-2] == "labels":
                label_by_key[("/".join(parts[:-2]), pure.stem)] = name
        eligible = sorted(set(image_by_key) & set(label_by_key))
        _require(len(eligible) >= skip + take, "FacadeElements has fewer paired files than the frozen roster requires")
        ordered = sorted(eligible, key=lambda key: hashlib.sha256(image_by_key[key].encode("utf-8")).hexdigest())
        cases = []
        for index, key in enumerate(ordered[skip:skip + take], start=1):
            image_member = image_by_key[key]
            cases.append({
                "case_id": f"{case_id_prefix}-{index:03d}",
                "episode_id": f"{episode_id_prefix}-{index:03d}",
                "source_stem": key[1],
                "image_member_path": image_member,
                "image_member_path_sha256": hashlib.sha256(image_member.encode("utf-8")).hexdigest(),
                "label_member_path": label_by_key[key],
            })
    roster = _sealed_payload({
        "schema_version": ROSTER_SCHEMA,
        "source_kind": "FACADEELEMENTS_ZENODO_ARCHIVE",
        "created_at_utc": _utc_now(),
        "created_before_selected_pixel_access": True,
        "created_before_selected_label_access": True,
        "private_truth_access": False,
        "source_lock_sha256": sha256(source_lock_path),
        "source_metadata_sha256": sha256(source_metadata_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_md5": FACADEELEMENTS_MD5,
        "archive_sha256": sha256(archive_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_pair_count": len(eligible),
        "selection_skip": skip,
        "selection_rule": source_lock["roster_rule"],
        "public_class_names": {str(index): name for index, name in sorted(class_names.items())},
        "legal_class_ids": legal_class_ids,
        "legal_class_names_normalized": sorted(legal_names),
        "reference_mode": reference_mode,
        "target_selector": target_selector,
        "cases": cases,
    }, "roster_body_sha256")
    _atomic_json(output_path, roster)
    return roster


def extract_facadeelements_public(
    roster_path: Path,
    c0_path: Path,
    archive_path: Path,
    output_root: Path,
    capture_path: Path,
) -> dict[str, Any]:
    _require(not capture_path.exists(), "capture manifest is immutable and already exists")
    roster = _read_json(roster_path)
    c0 = _read_json(c0_path)
    _validate_seal(roster, "roster_body_sha256")
    _require(roster.get("source_kind") == "FACADEELEMENTS_ZENODO_ARCHIVE", "FacadeElements roster source mismatch")
    _require(roster.get("private_truth_access") is False, "roster accessed private truth")
    _require(roster.get("archive_sha256") == sha256(archive_path), "FacadeElements archive binding mismatch")
    _require(roster.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "roster C0 binding mismatch")
    goal_times = {row["episode_id"]: row["goal_provenance"]["goal_recorded_at_utc"] for row in c0["episodes"]}
    cases = []
    with zipfile.ZipFile(archive_path) as archive:
        for raw in roster["cases"]:
            suffix = PurePosixPath(raw["image_member_path"]).suffix.lower()
            destination = (output_root / "public_images" / f"{raw['case_id']}{suffix}").resolve()
            _require(not destination.exists(), f"refusing to overwrite {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(archive.read(raw["image_member_path"]))
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
            accessed_at = _utc_now()
            _require(goal_times[raw["episode_id"]] < accessed_at, "goal must precede first project pixel access")
            cases.append({
                "case_id": raw["case_id"],
                "episode_id": raw["episode_id"],
                "capture_created_at_utc": accessed_at,
                "capture_time_semantics": CAPTURE_TIME_SEMANTICS,
                "source_capture_time_semantics": "DATASET_COLLECTION_DATE_NO_PER_IMAGE_TIMESTAMP",
                "source_collection_date_utc": "2024-08-12T00:00:00Z",
                "image_path": str(destination),
                "image_sha256": sha256(destination),
            })
    capture = _sealed_payload({
        "schema_version": CAPTURE_SCHEMA,
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "source_role": "PREEXISTING_PUBLIC_MANUALLY_ANNOTATED_FACADE_DATASET_GOAL_BEFORE_PROJECT_PIXEL_ACCESS",
        "precedence_mode": PRECEDENCE_MODE,
        "physical_capture_after_goal_claimed": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "roster_body_sha256": roster["roster_body_sha256"],
        "cases": cases,
    }, "capture_manifest_body_sha256")
    _atomic_json(capture_path, capture)
    return capture


def extract_facadeelements_private_truth(
    roster_path: Path,
    capture_path: Path,
    archive_path: Path,
    output_root: Path,
    truth_path: Path,
) -> dict[str, Any]:
    _require(not truth_path.exists(), "private truth is immutable and already exists")
    roster = _read_json(roster_path)
    capture = _read_json(capture_path)
    _validate_seal(roster, "roster_body_sha256")
    _validate_seal(capture, "capture_manifest_body_sha256")
    _require(capture.get("private_truth_access") is False, "capture manifest must be sealed before label access")
    _require(roster.get("archive_sha256") == sha256(archive_path), "FacadeElements archive binding mismatch")
    capture_by_case = {row["case_id"]: row for row in capture["cases"]}
    legal_class_ids = {int(value) for value in roster["legal_class_ids"]}
    cases = []
    with zipfile.ZipFile(archive_path) as archive:
        for raw in roster["cases"]:
            label_destination = (output_root / "private_labels" / f"{raw['case_id']}.txt").resolve()
            _require(not label_destination.exists(), f"refusing to overwrite {label_destination}")
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.write_bytes(archive.read(raw["label_member_path"]))
            captured = capture_by_case.get(raw["case_id"])
            _require(captured is not None, "truth case is absent from capture manifest")
            with Image.open(captured["image_path"]) as image:
                width, height = image.size
            all_door_boxes = []
            for line_number, line in enumerate(label_destination.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                _require(len(parts) == 5, f"invalid YOLO label at {raw['case_id']}:{line_number}")
                class_id = int(parts[0])
                center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
                _require(all(np.isfinite(value) for value in (center_x, center_y, box_width, box_height)), "non-finite YOLO label")
                if class_id not in legal_class_ids:
                    continue
                x1 = (center_x - box_width / 2.0) * width
                y1 = (center_y - box_height / 2.0) * height
                x2 = (center_x + box_width / 2.0) * width
                y2 = (center_y + box_height / 2.0) * height
                _require(x2 > x1 and y2 > y1, "non-positive FacadeElements box")
                all_door_boxes.append([x1, y1, x2, y2])
            if roster.get("target_selector") == "LEFTMOST_DOOR_X_CENTER" and all_door_boxes:
                ordered_boxes = sorted(all_door_boxes, key=lambda box: ((box[0] + box[2]) / 2.0, box[1], box[3]))
                boxes = [ordered_boxes[0]]
                distractor_boxes = ordered_boxes[1:]
            else:
                boxes = all_door_boxes
                distractor_boxes = []
            cases.append({
                "case_id": raw["case_id"],
                "reference_mode": roster.get("reference_mode", "SET_VALUED"),
                "target_visibility": "VISIBLE" if boxes else "NOT_VISIBLE",
                "legal_target_bboxes_xyxy": boxes,
                "same_class_distractor_bboxes_xyxy": distractor_boxes,
                "private_label_sha256": sha256(label_destination),
            })
    truth = {
        "schema_version": TRUTH_SCHEMA,
        "truth_created_at_utc": _utc_now(),
        "primary_iou_threshold": 0.30,
        "diagnostic_iou_thresholds": [0.10, 0.50],
        "recall_at_k": [1, 3, 5, 10],
        "truth_source": "FACADEELEMENTS_MANUALLY_LABELED_FIVE_DOOR_CLASSES_YOLO_BBOX",
        "target_selector": roster.get("target_selector", "ALL_DOORS"),
        "cases": cases,
    }
    _atomic_json(truth_path, truth)
    return truth


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-roster")
    freeze.add_argument("--source-lock", required=True, type=Path)
    freeze.add_argument("--c0", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    public = subparsers.add_parser("download-public")
    public.add_argument("--roster", required=True, type=Path)
    public.add_argument("--c0", required=True, type=Path)
    public.add_argument("--output-root", required=True, type=Path)
    public.add_argument("--capture", required=True, type=Path)
    private = subparsers.add_parser("download-private-truth")
    private.add_argument("--roster", required=True, type=Path)
    private.add_argument("--capture", required=True, type=Path)
    private.add_argument("--output-root", required=True, type=Path)
    private.add_argument("--truth", required=True, type=Path)
    door_freeze = subparsers.add_parser("freeze-doordetect-roster")
    door_freeze.add_argument("--source-lock", required=True, type=Path)
    door_freeze.add_argument("--c0", required=True, type=Path)
    door_freeze.add_argument("--output", required=True, type=Path)
    door_public = subparsers.add_parser("download-doordetect-public")
    door_public.add_argument("--roster", required=True, type=Path)
    door_public.add_argument("--c0", required=True, type=Path)
    door_public.add_argument("--output-root", required=True, type=Path)
    door_public.add_argument("--capture", required=True, type=Path)
    door_private = subparsers.add_parser("download-doordetect-private-truth")
    door_private.add_argument("--roster", required=True, type=Path)
    door_private.add_argument("--capture", required=True, type=Path)
    door_private.add_argument("--output-root", required=True, type=Path)
    door_private.add_argument("--truth", required=True, type=Path)
    facade_freeze = subparsers.add_parser("freeze-facadeelements-roster")
    facade_freeze.add_argument("--source-lock", required=True, type=Path)
    facade_freeze.add_argument("--source-metadata", required=True, type=Path)
    facade_freeze.add_argument("--c0", required=True, type=Path)
    facade_freeze.add_argument("--archive", required=True, type=Path)
    facade_freeze.add_argument("--output", required=True, type=Path)
    facade_public = subparsers.add_parser("extract-facadeelements-public")
    facade_public.add_argument("--roster", required=True, type=Path)
    facade_public.add_argument("--c0", required=True, type=Path)
    facade_public.add_argument("--archive", required=True, type=Path)
    facade_public.add_argument("--output-root", required=True, type=Path)
    facade_public.add_argument("--capture", required=True, type=Path)
    facade_private = subparsers.add_parser("extract-facadeelements-private-truth")
    facade_private.add_argument("--roster", required=True, type=Path)
    facade_private.add_argument("--capture", required=True, type=Path)
    facade_private.add_argument("--archive", required=True, type=Path)
    facade_private.add_argument("--output-root", required=True, type=Path)
    facade_private.add_argument("--truth", required=True, type=Path)
    facade_download = subparsers.add_parser("download-facadeelements-archive")
    facade_download.add_argument("--source-metadata", required=True, type=Path)
    facade_download.add_argument("--output", required=True, type=Path)
    facade_download.add_argument("--workers", type=int, default=8)
    intake = subparsers.add_parser("create-goal-intake")
    intake.add_argument("--output", required=True, type=Path)
    intake.add_argument("--episode-prefix", required=True)
    intake.add_argument("--count", required=True, type=int)
    intake.add_argument("--goal-text", required=True)
    intake.add_argument("--goal-type", required=True)
    intake.add_argument("--reference-mode", required=True)
    intake.add_argument("--task-semantics", required=True)
    intake.add_argument("--recorded-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-roster":
        freeze_roster(args.source_lock, args.c0, args.output)
    elif args.command == "download-public":
        download_public(args.roster, args.c0, args.output_root, args.capture)
    elif args.command == "download-private-truth":
        download_private_truth(args.roster, args.capture, args.output_root, args.truth)
    elif args.command == "freeze-doordetect-roster":
        freeze_doordetect_roster(args.source_lock, args.c0, args.output)
    elif args.command == "download-doordetect-public":
        download_doordetect_public(args.roster, args.c0, args.output_root, args.capture)
    elif args.command == "download-doordetect-private-truth":
        download_doordetect_private_truth(args.roster, args.capture, args.output_root, args.truth)
    elif args.command == "freeze-facadeelements-roster":
        freeze_facadeelements_roster(args.source_lock, args.source_metadata, args.c0, args.archive, args.output)
    elif args.command == "extract-facadeelements-public":
        extract_facadeelements_public(args.roster, args.c0, args.archive, args.output_root, args.capture)
    elif args.command == "extract-facadeelements-private-truth":
        extract_facadeelements_private_truth(args.roster, args.capture, args.archive, args.output_root, args.truth)
    elif args.command == "download-facadeelements-archive":
        download_facadeelements_archive(args.source_metadata, args.output, args.workers)
    else:
        create_goal_intake(
            args.output,
            episode_prefix=args.episode_prefix,
            count=args.count,
            goal_text=args.goal_text,
            goal_type=args.goal_type,
            reference_mode=args.reference_mode,
            task_semantics=args.task_semantics,
            recorded_at_utc=args.recorded_at_utc,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
