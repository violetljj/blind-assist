#!/usr/bin/env python3
"""Acquire a future-only SANPO-Synthetic replay through a short local path.

This T0 transport is independent of every frozen D1 execution.  Source
identity remains in the replay records, while local media filenames use short
timeline aliases.  Before any GCS request, the implementation enumerates the
final, staging, and downloader ``.tmp`` paths and rejects layouts containing a
content path of 240 or more characters.

The output remains a synthetic geometry-proxy intake.  It carries no human
event truth, real-world safety evidence, Android authority, or permission to
replace the research mainline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_sanpo_sequence_evalset import (  # noqa: E402
    DATASET_PAGE,
    DATASET_REPO,
    GCS_PREFIX,
    LICENSE_NAME,
    LICENSE_URL,
    SANPO_CITATION,
    download,
    get_gcs_object,
    list_gcs_objects,
    md5_base64_file,
    media_url,
    object_inventory,
    resample_indices,
    sha256_file,
)


SOURCE_ID = "sanpo_synthetic_v0"
DATASET_NAME = "SANPO-Synthetic v0"
REPLAY_SCHEMA = "blindassist_sanpo_synthetic_replay_v1"
TRANSPORT_SCHEMA = "blindassist_hftf_stage_c_t0_sanpo_short_path_transport"
EXECUTION_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_t0_consumed_development_transport_contract"
)
EXECUTION_CONTRACT_STATUS = (
    "FROZEN_BEFORE_T0_CONSUMED_DEVELOPMENT_SOURCE_OPEN"
)
EXECUTION_ROLE = "outcome_open_development_transport_canary"
PREFLIGHT_READY = "T0_SANPO_SHORT_PATH_PREFLIGHT_READY"
TRANSPORT_READY = "T0_SANPO_SHORT_PATH_TRANSPORT_READY"
NOT_EVALUABLE = (
    "T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
)
MAX_CONTENT_PATH_EXCLUSIVE = 240
DEFAULT_SESSION_ID = (
    "e1ae36e040a53837dbe40879ddca1fbc47d47752a563e1117629cde73e7de856"
)
DEFAULT_CAMERA = "camera_chest"
DEFAULT_LENS = "left"
OFFICIAL_SPLITS = ("train", "test")

METADATA_RELATIVES = (
    Path("source_metadata/source_session_description.json"),
    Path("source_metadata/source_labelmap.json"),
    Path("source_metadata/source_annotation_types.json"),
    Path("source_metadata/camera_poses.csv"),
    Path("source_metadata/official_split_session_ids.txt"),
)
FIXED_RELATIVES = (
    Path("manifest.replay.jsonl"),
    Path("dataset_spec.json"),
    Path("source_licenses.md"),
    Path("qa/replay_validation.json"),
    Path("qa/transport_receipt.json"),
    *METADATA_RELATIVES,
)


class TransportError(ValueError):
    """The T0 source contract or short-path transport is not admissible."""


@dataclass(frozen=True)
class AcquisitionConfig:
    session_id: str
    camera: str
    lens: str
    official_split: str
    start_frame: int
    target_fps: float
    frame_count: int


@dataclass(frozen=True)
class TransportLayout:
    transport_root: Path
    output_root: Path
    staging_root: Path
    relative_content_paths: tuple[Path, ...]
    acquisition_token: str
    session_id: str


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: AcquisitionConfig) -> None:
    if not isinstance(config.session_id, str) or not config.session_id.strip():
        raise TransportError("session_id must be a non-empty string")
    if config.camera != DEFAULT_CAMERA or config.lens != DEFAULT_LENS:
        raise TransportError("only the frozen camera_chest/left transport is supported")
    if config.official_split not in OFFICIAL_SPLITS:
        raise TransportError(f"official_split must be one of {OFFICIAL_SPLITS}")
    if config.start_frame < 0 or config.frame_count <= 0:
        raise TransportError("start_frame must be non-negative and frame_count positive")
    if (
        isinstance(config.target_fps, bool)
        or not math.isfinite(float(config.target_fps))
        or float(config.target_fps) <= 0.0
    ):
        raise TransportError("target_fps must be finite and positive")


def _acquisition_token(config: AcquisitionConfig) -> str:
    _validate_config(config)
    return _canonical_sha256(
        {
            "session_id": config.session_id,
            "camera": config.camera,
            "lens": config.lens,
            "official_split": config.official_split,
            "start_frame": config.start_frame,
            "target_fps": float(config.target_fps),
            "frame_count": config.frame_count,
        }
    )[:16]


def _media_relatives(split: str, frame_count: int) -> tuple[Path, ...]:
    values: list[Path] = []
    for timeline_index in range(frame_count):
        alias = f"{timeline_index:02x}"
        values.extend(
            (
                Path("i") / split / f"{alias}.png",
                Path("m") / split / f"{alias}.png",
                Path("d") / split / f"{alias}.f16.gz",
            )
        )
    return tuple(values)


def plan_layout(transport_root: Path, config: AcquisitionConfig) -> TransportLayout:
    """Derive a deterministic short root without putting session_id in a path."""
    token = _acquisition_token(config)
    root = transport_root.resolve()
    relatives = FIXED_RELATIVES + _media_relatives(
        config.official_split, config.frame_count
    )
    return TransportLayout(
        transport_root=root,
        output_root=root / "r" / token,
        staging_root=root / "w" / token,
        relative_content_paths=relatives,
        acquisition_token=token,
        session_id=config.session_id,
    )


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def enumerate_content_paths(layout: TransportLayout) -> tuple[Path, ...]:
    """Return all final, staging, and downloader temporary content paths."""
    values: list[Path] = []
    for root in (layout.output_root, layout.staging_root):
        for relative in layout.relative_content_paths:
            content = root / relative
            values.extend((content, _tmp_path(content)))
    return tuple(values)


def preflight_layout(layout: TransportLayout) -> dict[str, Any]:
    paths = enumerate_content_paths(layout)
    rendered = [str(path.resolve()) for path in paths]
    if len(set(layout.relative_content_paths)) != len(layout.relative_content_paths):
        raise TransportError("planned content paths are not unique")
    violations = [
        {"path": value, "length": len(value)}
        for value in rendered
        if len(value) >= MAX_CONTENT_PATH_EXCLUSIVE
    ]
    if violations:
        worst = max(violations, key=lambda item: int(item["length"]))
        raise TransportError(
            "content path budget exceeded: "
            f"{worst['length']} >= {MAX_CONTENT_PATH_EXCLUSIVE}: {worst['path']}"
        )
    return {
        "maximum_content_path_length": max(map(len, rendered), default=0),
        "content_path_count_including_tmp": len(rendered),
        "limit_exclusive": MAX_CONTENT_PATH_EXCLUSIVE,
        "all_content_paths_under_limit": True,
        "session_id_present_in_any_content_path": any(
            layout.session_id in value for value in rendered
        ),
    }


def require_fresh_layout(layout: TransportLayout) -> None:
    if layout.output_root.exists():
        raise TransportError(
            f"refusing to overwrite existing replay output: {layout.output_root}"
        )
    if layout.staging_root.exists():
        raise TransportError(
            f"refusing to reuse partial staging output: {layout.staging_root}"
        )


def _require_artifacts_transport_root(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise TransportError(
            f"transport root must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def _require_artifacts_report_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise TransportError(
            f"report output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def _require_gcs_receipt(
    item: Any, *, expected_name: str | None = None
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TransportError("GCS object receipt must be an object")
    name = item.get("name")
    generation = item.get("generation")
    md5_hash = item.get("md5Hash")
    try:
        size = int(item.get("size", -1))
    except (TypeError, ValueError) as error:
        raise TransportError("GCS object size is invalid") from error
    if (
        not isinstance(name, str)
        or not name
        or (expected_name is not None and name != expected_name)
        or generation is None
        or not str(generation)
        or not isinstance(md5_hash, str)
        or not md5_hash
        or size < 0
    ):
        raise TransportError(
            "GCS receipt requires exact name, generation, size, and MD5"
        )
    return item


def _verify_local_gcs_object(path: Path, item: dict[str, Any]) -> None:
    receipt = _require_gcs_receipt(item)
    if not path.is_file():
        raise TransportError(f"downloaded object is missing: {path}")
    if path.stat().st_size != int(receipt["size"]):
        raise TransportError(f"GCS size mismatch for {path}")
    if md5_base64_file(path) != receipt["md5Hash"]:
        raise TransportError(f"GCS MD5 mismatch for {path}")


def download_verified(
    item: dict[str, Any],
    target: Path,
    retries: int,
    *,
    downloader: Callable[[str, Path, int], None] = download,
) -> None:
    """Download one exact generation and reject any missing/incorrect MD5."""
    receipt = _require_gcs_receipt(item)
    if target.exists() or _tmp_path(target).exists():
        raise TransportError(f"refusing to reuse download target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = media_url(str(receipt["name"]), str(receipt["generation"]))
    downloader(url, target, retries)
    _verify_local_gcs_object(target, receipt)


def indexed_objects(
    objects: list[dict[str, Any]], suffix: str
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in objects:
        item = _require_gcs_receipt(raw)
        name = str(item["name"])
        if not name.endswith(suffix):
            continue
        stem = Path(name).name.removesuffix(suffix)
        if not stem.isdigit():
            raise TransportError(f"source frame filename is not numeric: {name}")
        index = int(stem)
        if index in result:
            raise TransportError(f"duplicate source frame index {index} for {suffix}")
        result[index] = item
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransportError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_contract_repo_path(value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    return (Path(__file__).resolve().parents[3] / raw).resolve()


def validate_execution_contract(
    contract_path: Path,
    transport_root: Path,
    config: AcquisitionConfig,
) -> dict[str, Any]:
    """Authorize only one exact, already outcome-open Development replay."""
    resolved_contract = contract_path.resolve()
    docs_root = (
        Path(__file__).resolve().parents[3] / "docs/research/hftf"
    ).resolve()
    try:
        resolved_contract.relative_to(docs_root)
    except ValueError as error:
        raise TransportError(
            f"T0 execution contract must stay under {docs_root}"
        ) from error
    contract = _load_json(resolved_contract)
    if (
        contract.get("schema") != EXECUTION_CONTRACT_SCHEMA
        or contract.get("status") != EXECUTION_CONTRACT_STATUS
    ):
        raise TransportError("T0 execution contract identity is not frozen")
    implementation = contract.get("implementations", {}).get("acquirer", {})
    if (
        Path(str(implementation.get("path", ""))).as_posix()
        != (
            "scripts/research/hftf/"
            "acquire_stage_c_t0_sanpo_short_path_transport.py"
        )
        or str(implementation.get("sha256", "")) != _sha256(Path(__file__))
        or implementation.get("network_execution_authorized") is not True
    ):
        raise TransportError("T0 acquirer implementation receipt mismatch")
    source = contract.get("source", {})
    if (
        source.get("role") != EXECUTION_ROLE
        or source.get("outcome_open_before_t0") is not True
        or source.get("fresh_evidence_credit") is not False
        or source.get("reserved_source") is not False
        or source.get("official_split") != "train"
    ):
        raise TransportError(
            "T0 contract does not bind an outcome-open Development source"
        )
    expected = contract.get("acquisition_config", {})
    observed = {
        "session_id": config.session_id,
        "camera": config.camera,
        "lens": config.lens,
        "official_split": config.official_split,
        "start_frame": config.start_frame,
        "target_fps": float(config.target_fps),
        "frame_count": config.frame_count,
    }
    frozen = {
        "session_id": str(expected.get("session_id", "")),
        "camera": str(expected.get("camera", "")),
        "lens": str(expected.get("lens", "")),
        "official_split": str(expected.get("official_split", "")),
        "start_frame": int(expected.get("start_frame", -1)),
        "target_fps": float(expected.get("target_fps", -1.0)),
        "frame_count": int(expected.get("frame_count", -1)),
    }
    if (
        observed != frozen
        or frozen["session_id"] != str(source.get("session_id", ""))
        or frozen["official_split"] != str(
            source.get("official_split", "")
        )
    ):
        raise TransportError("T0 acquisition config is not the frozen source")
    expected_root = _resolve_contract_repo_path(
        str(contract.get("transport_root", ""))
    )
    if transport_root.resolve() != expected_root:
        raise TransportError("T0 transport root differs from frozen contract")
    canonical = source.get("canonical_consumed_package", {})
    canonical_root = _resolve_contract_repo_path(
        str(canonical.get("root", ""))
    )
    manifest_path = canonical_root / "manifest.replay.jsonl"
    spec_path = canonical_root / "dataset_spec.json"
    if (
        _sha256(manifest_path)
        != str(canonical.get("manifest_sha256", ""))
        or _sha256(spec_path)
        != str(canonical.get("dataset_spec_sha256", ""))
    ):
        raise TransportError(
            "T0 canonical consumed package receipt mismatch"
        )
    canonical_spec = _load_json(spec_path)
    try:
        canonical_rows = [
            json.loads(line)
            for line in manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise TransportError(
            "T0 canonical consumed manifest is invalid"
        ) from error
    if (
        canonical_spec.get("schema") != REPLAY_SCHEMA
        or canonical_spec.get("source", {}).get("session_id")
        != config.session_id
        or canonical_spec.get("source", {}).get("official_split")
        != "train"
        or len(canonical_rows) != config.frame_count
        or any(
            not isinstance(row, dict)
            or row.get("session_id") != config.session_id
            for row in canonical_rows
        )
    ):
        raise TransportError(
            "T0 canonical package is not the frozen consumed source"
        )
    source_plan_receipt = contract.get("parents", {}).get(
        "g0_source_plan", {}
    )
    source_plan_path = _resolve_contract_repo_path(
        str(source_plan_receipt.get("path", ""))
    )
    if _sha256(source_plan_path) != str(
        source_plan_receipt.get("sha256", "")
    ):
        raise TransportError("T0 G0 source-plan receipt mismatch")
    source_plan = _load_json(source_plan_path)
    development = source_plan.get("roles", {}).get(
        "development_reuse", []
    )
    if (
        source_plan.get("terminal")
        != "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
        or not any(
            isinstance(item, dict)
            and item.get("session_id") == config.session_id
            and item.get("g0_source_role")
            in {
                "development_reuse_outcome_open_train",
                "development_reuse_outcome_open_model_selection",
            }
            and item.get("fresh_evidence_credit") is False
            for item in development
        )
    ):
        raise TransportError(
            "T0 source is not outcome-open in the frozen G0 source plan"
        )
    authorization = contract.get("authorization", {})
    if (
        authorization.get("consumed_development_transport_execution")
        is not True
        or authorization.get("fresh_or_reserved_source_open") is not False
        or authorization.get("teacher_or_student_execution") is not False
    ):
        raise TransportError("T0 execution authorization boundary mismatch")
    return contract


def _camera_metadata(
    description: dict[str, Any], camera: str, lens: str
) -> tuple[float, dict[str, Any]]:
    locations = description.get("session_camera_location")
    details = description.get("session_camera_details")
    if (
        not isinstance(locations, list)
        or not isinstance(details, list)
        or camera not in locations
    ):
        raise TransportError(f"session does not expose requested camera {camera!r}")
    index = locations.index(camera)
    if index >= len(details) or not isinstance(details[index], dict):
        raise TransportError("camera metadata does not align with location")
    item = details[index]
    dimensions = item.get(f"{lens}_camera_params")
    if not isinstance(dimensions, dict):
        raise TransportError(f"session has no {lens} camera parameters")
    fps = float(item.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0.0:
        raise TransportError("source camera fps must be finite and positive")
    for field in ("image_width", "image_height", "fx", "fy", "cx", "cy"):
        if field not in dimensions:
            raise TransportError(f"camera parameters missing {field}")
    return fps, dimensions


def _validate_image_pair(
    image_path: Path, mask_path: Path, dimensions: dict[str, Any]
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image.load()
        size = image.size
    with Image.open(mask_path) as mask:
        mask.load()
        if mask.size != size:
            raise TransportError("RGB and panoptic mask dimensions differ")
    expected = (
        int(dimensions["image_width"]),
        int(dimensions["image_height"]),
    )
    if size != expected:
        raise TransportError(f"source image dimensions differ: {size} != {expected}")
    return size


def _split_contract(official_split: str) -> dict[str, Any]:
    if official_split not in OFFICIAL_SPLITS:
        raise TransportError(f"unsupported official split: {official_split}")
    is_train = official_split == "train"
    return {
        "official_split": official_split,
        "split_object_name": (
            f"{GCS_PREFIX}/sanpo-synthetic/splits/"
            f"{official_split}_session_ids.txt"
        ),
        "label_authority": (
            "official_panoptic_ground_truth_pretraining_only"
            if is_train
            else "official_panoptic_heldout_geometry_proxy_only"
        ),
        "pretraining_candidate": is_train,
        "synthetic_heldout_evaluation_candidate": not is_train,
    }


def _write_fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_fresh_json(path: Path, value: Any) -> None:
    _write_fresh_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def _manifest_row(
    *,
    config: AcquisitionConfig,
    split: dict[str, Any],
    sequence_id: str,
    timeline_index: int,
    source_index: int,
    source_fps: float,
    dimensions: dict[str, Any],
    annotation_types: dict[str, Any],
    stage_root: Path,
    image_relative: Path,
    mask_relative: Path,
    depth_relative: Path,
    rgb_item: dict[str, Any],
    mask_item: dict[str, Any],
    depth_item: dict[str, Any],
) -> dict[str, Any]:
    image_path = stage_root / image_relative
    mask_path = stage_root / mask_relative
    depth_path = stage_root / depth_relative
    width, height = _validate_image_pair(image_path, mask_path, dimensions)
    sample_id = f"{sequence_id}_{timeline_index:06d}"
    return {
        "id": sample_id,
        "image_path": image_relative.as_posix(),
        "image_sha256": sha256_file(image_path),
        "source_mask_path": mask_relative.as_posix(),
        "source_mask_sha256": sha256_file(mask_path),
        "source_depth_path": depth_relative.as_posix(),
        "source_depth_sha256": sha256_file(depth_path),
        "width": width,
        "height": height,
        "session_id": config.session_id,
        "sequence_id": sequence_id,
        "frame_index": timeline_index,
        "source_frame_index": source_index,
        "source_timestamp_ms": int(round(source_index * 1000.0 / source_fps)),
        "source_annotation_quality": str(
            annotation_types.get(str(source_index), "UNKNOWN")
        ),
        "label_authority": split["label_authority"],
        "event_truth": None,
        "source": {
            "source_id": SOURCE_ID,
            "dataset": DATASET_NAME,
            "dataset_page": DATASET_PAGE,
            "repository": DATASET_REPO,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "official_split": config.official_split,
            "session_id": config.session_id,
            "camera": config.camera,
            "lens": config.lens,
            "privacy_status": (
                "synthetic_source_no_personal_data_claimed_by_importer"
            ),
        },
        "modalities": {
            "rgb": object_inventory(rgb_item),
            "panoptic_mask": object_inventory(mask_item),
            "metric_depth": object_inventory(depth_item),
            "camera_poses": {
                "path": "source_metadata/camera_poses.csv",
                "sha256": sha256_file(
                    stage_root / "source_metadata/camera_poses.csv"
                ),
            },
            "imu": {
                "status": "not_present_in_this_published_session_inventory",
                "usable_for_replay": False,
            },
        },
        "authorization": {
            "offline_replay": True,
            "pretraining_candidate": split["pretraining_candidate"],
            "synthetic_heldout_evaluation_candidate": split[
                "synthetic_heldout_evaluation_candidate"
            ],
            "real_finetune_or_eval": False,
            "human_event_truth": False,
            "calibration": False,
            "blind_evaluation": False,
            "android_runtime": False,
            "production_model_replacement": False,
        },
    }


def _dataset_spec(
    *,
    config: AcquisitionConfig,
    split: dict[str, Any],
    source_fps: float,
    dimensions: dict[str, Any],
    selected: list[int],
    objects: dict[str, dict[str, Any]],
    split_object: dict[str, Any],
    rgb: dict[int, dict[str, Any]],
    masks: dict[int, dict[str, Any]],
    depth: dict[int, dict[str, Any]],
    layout: TransportLayout,
) -> dict[str, Any]:
    return {
        "schema": REPLAY_SCHEMA,
        "purpose": (
            "future-only short-path official SANPO-Synthetic source-contract "
            + (
                "pretraining candidate intake"
                if config.official_split == "train"
                else "heldout geometry-proxy evaluation intake"
            )
        ),
        "source": {
            "source_id": SOURCE_ID,
            "dataset": DATASET_NAME,
            "official_split": config.official_split,
            "session_id": config.session_id,
        },
        "sampling": {
            "source_fps": source_fps,
            "target_fps": config.target_fps,
            "selected_source_frames": selected,
        },
        "camera": dimensions,
        "source_inventory": {
            **{
                name: object_inventory(item)
                for name, item in objects.items()
            },
            "official_split_receipt": object_inventory(split_object),
            "rgb": [object_inventory(rgb[index]) for index in selected],
            "masks": [object_inventory(masks[index]) for index in selected],
            "depth": [object_inventory(depth[index]) for index in selected],
        },
        "local_transport": {
            "schema": TRANSPORT_SCHEMA,
            "acquisition_token": layout.acquisition_token,
            "identity_in_manifest_not_local_filename": True,
            "maximum_content_path_exclusive": MAX_CONTENT_PATH_EXCLUSIVE,
            "staged_then_atomically_published": True,
        },
        "required_downstream_order": (
            [
                "SANPO-Synthetic pretraining candidate",
                "separately gated SANPO-Real finetune",
                "independent offline/INT8/device gates",
            ]
            if config.official_split == "train"
            else ["frozen synthetic heldout geometry-proxy evaluation only"]
        ),
        "prohibited_claims": [
            "independent GPT/Codex consensus event truth",
            "calibration evidence",
            "blind evaluation",
            "Android runtime authorization",
            "production model replacement",
            "user safety proof",
        ],
    }


def acquire(
    transport_root: Path,
    config: AcquisitionConfig,
    execution_contract_path: Path,
    *,
    retries: int = 3,
) -> dict[str, Any]:
    if retries <= 0:
        raise TransportError("retries must be positive")
    validate_execution_contract(
        execution_contract_path, transport_root, config
    )
    contract_sha256 = _sha256(execution_contract_path.resolve())
    layout = plan_layout(transport_root, config)
    path_report = preflight_layout(layout)
    require_fresh_layout(layout)

    split = _split_contract(config.official_split)
    session_prefix = f"{GCS_PREFIX}/sanpo-synthetic/{config.session_id}"
    names = {
        "description": f"{session_prefix}/description.json",
        "labelmap": f"{GCS_PREFIX}/labelmap.json",
        "annotation_types": (
            f"{session_prefix}/{config.camera}/{config.lens}/"
            "frame_segmentation_annotation_type.json"
        ),
        "camera_poses": (
            f"{session_prefix}/{config.camera}/camera_poses.csv"
        ),
    }
    objects = {
        key: _require_gcs_receipt(
            get_gcs_object(name, retries), expected_name=name
        )
        for key, name in names.items()
    }
    split_name = str(split["split_object_name"])
    split_object = _require_gcs_receipt(
        get_gcs_object(split_name, retries), expected_name=split_name
    )

    metadata_targets = {
        "description": Path("source_metadata/source_session_description.json"),
        "labelmap": Path("source_metadata/source_labelmap.json"),
        "annotation_types": Path(
            "source_metadata/source_annotation_types.json"
        ),
        "camera_poses": Path("source_metadata/camera_poses.csv"),
    }
    for key, relative in metadata_targets.items():
        download_verified(
            objects[key], layout.staging_root / relative, retries
        )
    split_relative = Path("source_metadata/official_split_session_ids.txt")
    download_verified(
        split_object, layout.staging_root / split_relative, retries
    )

    description = _load_json(
        layout.staging_root / metadata_targets["description"]
    )
    if description.get("session_type") != "synthetic":
        raise TransportError(
            "official session description does not identify as synthetic"
        )
    split_ids = {
        line.strip()
        for line in (
            layout.staging_root / split_relative
        ).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if config.session_id not in split_ids:
        raise TransportError(
            f"session is not in official {config.official_split} split"
        )
    source_fps, dimensions = _camera_metadata(
        description, config.camera, config.lens
    )
    if config.target_fps > source_fps:
        raise TransportError("target_fps cannot exceed source_fps")

    frame_prefix = (
        f"{session_prefix}/{config.camera}/{config.lens}/video_frames/"
    )
    mask_prefix = (
        f"{session_prefix}/{config.camera}/{config.lens}/segmentation_masks/"
    )
    depth_prefix = (
        f"{session_prefix}/{config.camera}/{config.lens}/depth_maps/"
    )
    rgb = indexed_objects(list_gcs_objects(frame_prefix, retries), ".png")
    masks = indexed_objects(list_gcs_objects(mask_prefix, retries), ".png")
    depth = indexed_objects(
        list_gcs_objects(depth_prefix, retries), ".float16.gz"
    )
    available = sorted(set(rgb) & set(masks) & set(depth))
    selected = resample_indices(
        available,
        source_fps,
        config.target_fps,
        config.start_frame,
        config.frame_count,
    )
    if len(selected) != config.frame_count:
        raise TransportError(
            f"only {len(selected)} aligned frames available; "
            f"requested {config.frame_count}"
        )

    annotation_types = _load_json(
        layout.staging_root / metadata_targets["annotation_types"]
    )
    sequence_id = (
        f"sanpo_synthetic_{config.session_id}_{config.camera}_"
        f"{config.lens}_{config.start_frame:06d}_"
        f"{int(config.target_fps)}fps"
    )
    rows: list[dict[str, Any]] = []
    for timeline_index, source_index in enumerate(selected):
        alias = f"{timeline_index:02x}"
        image_relative = (
            Path("i") / config.official_split / f"{alias}.png"
        )
        mask_relative = (
            Path("m") / config.official_split / f"{alias}.png"
        )
        depth_relative = (
            Path("d") / config.official_split / f"{alias}.f16.gz"
        )
        for item, relative in (
            (rgb[source_index], image_relative),
            (masks[source_index], mask_relative),
            (depth[source_index], depth_relative),
        ):
            download_verified(
                item, layout.staging_root / relative, retries
            )
        rows.append(
            _manifest_row(
                config=config,
                split=split,
                sequence_id=sequence_id,
                timeline_index=timeline_index,
                source_index=source_index,
                source_fps=source_fps,
                dimensions=dimensions,
                annotation_types=annotation_types,
                stage_root=layout.staging_root,
                image_relative=image_relative,
                mask_relative=mask_relative,
                depth_relative=depth_relative,
                rgb_item=rgb[source_index],
                mask_item=masks[source_index],
                depth_item=depth[source_index],
            )
        )

    manifest_path = layout.staging_root / "manifest.replay.jsonl"
    _write_fresh_text(
        manifest_path,
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
    )
    spec_path = layout.staging_root / "dataset_spec.json"
    _write_fresh_json(
        spec_path,
        _dataset_spec(
            config=config,
            split=split,
            source_fps=source_fps,
            dimensions=dimensions,
            selected=selected,
            objects=objects,
            split_object=split_object,
            rgb=rgb,
            masks=masks,
            depth=depth,
            layout=layout,
        ),
    )
    _write_fresh_text(
        layout.staging_root / "source_licenses.md",
        (
            "# SANPO-Synthetic source and license\n\n"
            f"- Dataset: {DATASET_NAME}\n"
            f"- Dataset page: {DATASET_PAGE}\n"
            f"- Repository: {DATASET_REPO}\n"
            f"- License: [{LICENSE_NAME}]({LICENSE_URL})\n"
            f"- Session: `{config.session_id}` / official "
            f"{config.official_split} split\n"
            f"- Attribution: {SANPO_CITATION}\n"
            "- Boundary: future-only synthetic geometry-proxy intake; no "
            "human event truth, safety evidence, runtime authority, or "
            "mainline replacement authority.\n"
        ),
    )
    qa = {
        "ok": True,
        "frame_count": len(rows),
        "required_modalities_hash_bound": True,
        "all_rgb_mask_dimensions_match": True,
        "official_split": config.official_split,
        "all_frames_official_split_match": True,
        "all_frames_official_train_split": (
            config.official_split == "train"
        ),
        "pretraining_candidate": split["pretraining_candidate"],
        "synthetic_heldout_evaluation_candidate": split[
            "synthetic_heldout_evaluation_candidate"
        ],
        "imu_status": "absent_in_published_session_inventory_not_synthesized",
        "production_authorized": False,
        "short_path_preflight": path_report,
    }
    qa_path = layout.staging_root / "qa/replay_validation.json"
    _write_fresh_json(qa_path, qa)
    receipt_path = layout.staging_root / "qa/transport_receipt.json"
    receipt = {
        "schema": TRANSPORT_SCHEMA,
        "terminal": TRANSPORT_READY,
        "acquisition_token": layout.acquisition_token,
        "output_root": str(layout.output_root),
        "session_id": config.session_id,
        "official_split": config.official_split,
        "selected_source_frames": selected,
        "execution_contract_path": str(execution_contract_path.resolve()),
        "execution_contract_sha256": contract_sha256,
        "path_preflight": path_report,
        "dataset_spec_sha256": sha256_file(spec_path),
        "manifest_sha256": sha256_file(manifest_path),
        "replay_validation_sha256": sha256_file(qa_path),
        "all_downloads_generation_bound": True,
        "all_downloads_size_and_md5_verified": True,
        "authorization": {
            "pose_geometry_authority_verification_authorized": True,
            "teacher_label_or_corpus_authorized": False,
            "student_training_authorized": False,
            "fresh_or_reserved_evaluation_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
    }
    _write_fresh_json(receipt_path, receipt)

    layout.output_root.parent.mkdir(parents=True, exist_ok=True)
    require_fresh_layout(
        TransportLayout(
            layout.transport_root,
            layout.output_root,
            Path(str(layout.staging_root) + ".publication-check"),
            layout.relative_content_paths,
            layout.acquisition_token,
            layout.session_id,
        )
    )
    layout.staging_root.replace(layout.output_root)
    return {
        **receipt,
        "transport_receipt_path": str(
            layout.output_root / "qa/transport_receipt.json"
        ),
        "transport_receipt_sha256": sha256_file(
            layout.output_root / "qa/transport_receipt.json"
        ),
    }


def preflight(transport_root: Path, config: AcquisitionConfig) -> dict[str, Any]:
    layout = plan_layout(transport_root, config)
    report = preflight_layout(layout)
    require_fresh_layout(layout)
    return {
        "schema": TRANSPORT_SCHEMA,
        "terminal": PREFLIGHT_READY,
        "acquisition_token": layout.acquisition_token,
        "transport_root": str(layout.transport_root),
        "planned_output_root": str(layout.output_root),
        "planned_staging_root": str(layout.staging_root),
        "session_id_length": len(config.session_id),
        "path_preflight": report,
        "network_opened": False,
        "source_opened": False,
        "output_created": False,
        "authorization": {
            "acquisition_authorized_by_preflight": False,
            "fresh_or_reserved_evaluation_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--camera", choices=(DEFAULT_CAMERA,), default=DEFAULT_CAMERA)
    parser.add_argument("--lens", choices=(DEFAULT_LENS,), default=DEFAULT_LENS)
    parser.add_argument("--official-split", choices=OFFICIAL_SPLITS, default="train")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional fresh JSON report path under artifacts.local",
    )
    parser.add_argument(
        "--execution-contract",
        type=Path,
        help=(
            "Frozen exact-source contract; mandatory for any network "
            "acquisition"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the complete path plan without filesystem writes or network",
    )
    args = parser.parse_args()
    try:
        config = AcquisitionConfig(
            session_id=args.session_id,
            camera=args.camera,
            lens=args.lens,
            official_split=args.official_split,
            start_frame=args.start_frame,
            target_fps=args.target_fps,
            frame_count=args.frame_count,
        )
        transport_root = _require_artifacts_transport_root(args.transport_root)
        if args.preflight_only:
            result = preflight(transport_root, config)
        else:
            if args.execution_contract is None:
                raise TransportError(
                    "network acquisition requires --execution-contract"
                )
            result = acquire(
                transport_root,
                config,
                args.execution_contract,
                retries=args.retries,
            )
        if args.report_output is not None:
            report_output = _require_artifacts_report_output(
                args.report_output
            )
            report_output.parent.mkdir(parents=True, exist_ok=True)
            with report_output.open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (
        TransportError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "terminal": NOT_EVALUABLE,
                    "raw_failure_class": (
                        "SHORT_PATH_PREFLIGHT_OR_ACQUISITION_FAILURE"
                    ),
                    "error": str(error),
                    "research_mainline_changed": False,
                    "default_app_changed": False,
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
