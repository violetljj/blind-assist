"""Shared constants and deterministic helpers for target-local warp residual R0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0"
CONTRACT_PROTOCOL_ID = "TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT"
IMPLEMENTATION_ID = "TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_IMPL_R0"
SHI_TOMASI_MODEL_ID = "SHI_TOMASI_LK_V1"
SIMILARITY_MODEL_ID = "SIMILARITY_RANSAC_PROCRUSTES_V1"
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/dual-loop/TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md"
)
MAX_DT_NS = 100_000_000
FB_ERROR_MAX_PX = 1.5
RANSAC_REPROJECTION_MAX_PX = 2.0
RANSAC_MAX_ITERS = 1000
RANSAC_CONFIDENCE = 0.99
RANSAC_SEED = 0
MIN_SURVIVING_POINTS = 8
MIN_INLIER_RATIO = 0.50
MIN_SPATIAL_QUADRANTS = 2
MAX_QUADRANT_FRACTION = 0.75
MAX_CONDITION_NUMBER = 100.0
MIN_EVENT_FINITE_PAIRS = 3
MIN_EVENT_PAIR_COVERAGE = 0.50
DEADBAND_PER_S = 0.02
MAX_COVERAGE_LOSS = 0.05
MIN_CONTRIBUTING_EVENTS = 2
MAX_SINGLE_CONTRIBUTION = 0.50

RING_CONFIGS: dict[str, dict[str, float]] = {
    "R1": {"r_inner_over_d": 0.10, "r_outer_over_d": 0.50},
    "R2": {"r_inner_over_d": 0.10, "r_outer_over_d": 0.75},
    "R3": {"r_inner_over_d": 0.20, "r_outer_over_d": 0.75},
    "R4": {"r_inner_over_d": 0.20, "r_outer_over_d": 1.00},
}

ABSTENTION_REASONS = (
    "INPUT_TIMESTAMP_INVALID",
    "FRAME_ADJACENCY_INVALID",
    "IMAGE_SHAPE_MISMATCH",
    "TRACK_ID_MISMATCH",
    "BOX_INVALID",
    "BOX_BOUNDARY_TRUNCATED",
    "DYNAMIC_MASK_INVALID",
    "RING_EMPTY_OR_LOW_AREA",
    "FEATURE_COUNT_LOW",
    "LK_TRACK_COUNT_LOW",
    "SPATIAL_SUPPORT_LOW",
    "GEOMETRY_DEGENERATE",
    "RANSAC_INLIER_RATIO_LOW",
    "REPROJECTION_ERROR_HIGH",
    "TRANSFORM_INVALID",
    "PREDICTED_BOX_INVALID",
    "NUMERIC_NONFINITE",
)

FORBIDDEN_INPUT_KEYS = {
    "truth",
    "truth_state",
    "truth_label",
    "event_label",
    "pose",
    "vicon",
    "oracle_roi",
    "oracle_mask",
    "decision",
    "old_decision",
    "posthoc_event_label",
    "r1_output",
    "d0_output",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / CONTRACT_RELATIVE_PATH


def contract_sha256(repo_root: Path | None = None) -> str:
    return sha256_file(contract_path(repo_root))


def parameter_set_id(ring_config_id: str, repo_root: Path | None = None) -> str:
    payload = {
        "contract_sha256": contract_sha256(repo_root),
        "implementation_id": IMPLEMENTATION_ID,
        "ring_config_id": ring_config_id,
        "model_id": SIMILARITY_MODEL_ID,
        "parameters": {
            "feature_max_corners": 80,
            "quality_level": 0.01,
            "min_distance_px": 5.0,
            "block_size_px": 5,
            "lk_window_px": [15, 15],
            "lk_max_level": 2,
            "lk_termination_count": 20,
            "lk_termination_epsilon": 0.03,
            "fb_error_max_px": FB_ERROR_MAX_PX,
            "ransac_reprojection_max_px": RANSAC_REPROJECTION_MAX_PX,
            "ransac_max_iters": RANSAC_MAX_ITERS,
            "ransac_confidence": RANSAC_CONFIDENCE,
            "ransac_seed": RANSAC_SEED,
        },
    }
    return sha256_bytes(canonical_json(payload))


def input_identity_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "source_id",
            "session_id",
            "sequence_id",
            "previous_source_frame_id",
            "current_source_frame_id",
            "previous_frame_index",
            "current_frame_index",
            "captured_at_ns_previous",
            "captured_at_ns_current",
            "target_id",
            "track_epoch",
            "previous_bbox",
            "current_bbox",
            "previous_frame_shape",
            "current_frame_shape",
            "previous_image_sha256",
            "current_image_sha256",
            "parent_event_id",
        )
    }


def detection_identity_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_dynamic_bboxes": row.get("previous_dynamic_bboxes"),
        "current_dynamic_bboxes": row.get("current_dynamic_bboxes"),
    }


def manifest_hash(rows: Iterable[dict[str, Any]], payload_fn) -> str:
    payload = [payload_fn(row) for row in rows]
    return sha256_bytes(canonical_json(payload))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
