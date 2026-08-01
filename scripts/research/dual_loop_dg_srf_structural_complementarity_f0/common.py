"""Shared serialization and frozen-input helpers for DG-SRF F0."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0"
CONFIG_SCHEMA = (
    "blindassist.dg_srf_image_space_structural_complementarity_f0.config.v1"
)
SHAPE = (256, 256)
ARMS = ("D1", "D2", "D3", "D4", "D5")
SINGLE_ARMS = ("D1", "D2", "D3")


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return sha256_bytes(header + array.tobytes(order="C"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)).decode("utf-8"))
            handle.write("\n")


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ensure_artifact_output(repo_root: Path, output: Path) -> Path:
    root = (repo_root / "artifacts.local").resolve()
    resolved = output.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be below artifacts.local: {resolved}")
    return resolved


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"size mismatch for {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected config schema")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol id")
    if tuple(config.get("analysis_shape", [])) != SHAPE:
        raise ValueError("analysis shape must be 256x256")
    if config["structural_signal"]["D4_weights"] != {
        "N": 0.25,
        "E": 0.25,
        "R_plus": 0.25,
        "R_minus": 0.25,
    }:
        raise ValueError("D4 must remain fixed 1:1:1:1")
    if config["proxy_roi_ablation"]["lambda"] != 0.25:
        raise ValueError("D5 lambda must remain 0.25")
    grid = [round(float(value), 2) for value in config["grouped_evaluation"]["threshold_grid"]]
    if grid != [round(value / 100, 2) for value in range(5, 100, 5)]:
        raise ValueError("threshold grid must be 0.05..0.95 step 0.05")
    if config["model_contract"]["input_size"] != 518:
        raise ValueError("Depth Anything input size must be 518")
    model = config["model_contract"]
    expected_model_identity = {
        "model_id": "Depth-Anything-V2-Small",
        "source_commit": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
        "checkpoint_repository_revision": (
            "03876f8651c73a60fe4c2c48294e09fcb6838fcf"
        ),
        "checkpoint_sha256": (
            "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
        ),
        "checkpoint_bytes": 99_218_434,
        "encoder": "vits",
        "license": "Apache-2.0",
    }
    for field, expected in expected_model_identity.items():
        if model[field] != expected:
            raise ValueError(f"model identity drifted: {field}")
    if model["exact_parameter_count"] != 24_785_089:
        raise ValueError("Depth Anything V2 Small parameter count drifted")
    if config["structural_signal"]["gradient_gaussian_sigmas_pixels"] != [
        0.0,
        1.5,
        3.0,
    ]:
        raise ValueError("gradient scale contract drifted")
    if config["structural_signal"]["gradient_sobel_kernel"] != 3:
        raise ValueError("Sobel kernel contract drifted")
    trend = config["structural_signal"]["surface_trend"]
    expected_trend = {
        "name": "lower_image_surface_trend",
        "lower_image_start_fraction": 0.45,
        "row_statistic": "median_full_width",
        "polynomial_degree": 2,
        "fit": "ordinary_least_squares_on_row_medians",
        "residual_dead_zone": 0.03,
        "truth_or_yolo_assisted_fit": False,
        "failure_on_nonfinite_or_rank_defect": True,
        "forbidden_names": ["ground", "floor", "height", "plane"],
    }
    if trend != expected_trend:
        raise ValueError("surface-trend contract drifted")
    expected_gates = {
        "minimum_fp_pixel_reduction_vs_B": 0.30,
        "minimum_overall_residual_recall_retention_vs_B": 0.90,
        "minimum_group_residual_recall_retention_vs_B": 0.80,
        "minimum_boundary_step_curb_recall_retention_vs_B": 0.80,
        "minimum_obstacle_recall_retention_vs_B": 0.80,
        "minimum_delta_recall_C_minus_A": 0.05,
        "maximum_delta_false_positive_area_fraction_C_minus_A": 0.05,
        "minimum_residual_truth_component_recall": 0.50,
        "maximum_false_activation_components_per_frame": 3.0,
    }
    if config["utility_gates"] != expected_gates:
        raise ValueError("utility-gate contract drifted")
    if config["depth_health"]["overall_evaluable_frame_coverage_minimum"] != 0.95:
        raise ValueError("overall coverage gate drifted")
    if config["depth_health"]["minimum_group_evaluable_frame_coverage"] != 0.90:
        raise ValueError("group coverage gate drifted")
    direction = config["direction_canary"]
    if direction["frozen_direction"] != "RAW_LARGER_IS_NEARER":
        raise ValueError("official inverse-depth direction must be frozen")
    if direction["canary_may_select_or_flip_direction"] is not False:
        raise ValueError("direction canary may not select or flip direction")
    if config["grouped_evaluation"]["minimum_D4_positive_advantage_group_count"] != 8:
        raise ValueError("D4 group-advantage count drifted")
    forbidden_false = (
        ("model_contract", "metric_distance_interpretation"),
        ("model_contract", "cross_frame_raw_magnitude_comparison"),
        ("structural_signal", "weight_search"),
        ("structural_signal", "D3_sign_branch_selection"),
        ("proxy_roi_ablation", "lambda_search"),
        ("direction_canary", "canary_may_select_or_flip_direction"),
    )
    for section, field in forbidden_false:
        if config[section][field] is not False:
            raise ValueError(f"{section}.{field} must be false")


def decode_packed_mask(encoded: str, shape: Sequence[int] = SHAPE) -> np.ndarray:
    raw = base64.b64decode(encoded)
    count = int(shape[0]) * int(shape[1])
    unpacked = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        count=count,
        bitorder="big",
    )
    return unpacked.reshape(tuple(int(value) for value in shape)).astype(bool)


def encode_packed_mask(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")
