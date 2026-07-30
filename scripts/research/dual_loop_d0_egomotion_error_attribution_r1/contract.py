from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "D0_EGOMOTION_ERROR_ATTRIBUTION_R1"
PROTOCOL_RELATIVE_PATH = Path(
    "docs/research/dual-loop/"
    "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_PROTOCOL_2026-07-30.json"
)
PROTOCOL_SHA256 = "87931369f912fdd054783db9decb2a1813080d0a961c3526b83ce686d1a48183"
PRIMARY_ARM = "ROI_SPARSE_RADIAL_FLOW"
REFERENCE_ARM = "BBOX_LOG_AREA_GROWTH"
EVENT_COUNT = 469
SOURCE_CLOSURE_ATOL_MPS = 1e-6
BBOX_CLOSURE_ATOL_PER_S = 1e-12

PRESELECTED_METRICS = (
    "median_abs_sensor_approach_component_mps",
    "median_abs_person_approach_component_mps",
    "median_flow_score_mad_per_s",
    "median_surviving_tracks",
)
DIAGNOSTIC_METRICS = (
    "median_camera_translation_speed_mps",
    "p90_camera_translation_speed_mps",
    "median_camera_angular_speed_radps",
    "p90_camera_angular_speed_radps",
    "flow_sign_flip_fraction",
    "median_detected_features",
    "minimum_surviving_tracks",
    "median_occupied_quadrants",
    "finite_flow_coverage",
    "negative_log_duration_s",
    "median_abs_log_area_rate_per_s",
    "log_area_rate_mad_per_s",
    "median_center_speed_normalized_per_s",
    "center_velocity_mad_normalized_per_s",
    "median_forward_backward_error_px",
    "reference_arm_behavior",
)
METRIC_ORDER = PRESELECTED_METRICS + DIAGNOSTIC_METRICS


class ContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def finite_values(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def type7_quantile(values: Iterable[Any], q: float) -> float | None:
    finite = sorted(finite_values(values))
    if not finite:
        return None
    if not 0.0 <= q <= 1.0:
        raise ContractError("quantile must be in [0,1]")
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] + fraction * (finite[upper] - finite[lower])


def median(values: Iterable[Any]) -> float | None:
    return type7_quantile(values, 0.5)


def raw_mad(values: Iterable[Any]) -> float | None:
    finite = finite_values(values)
    center = median(finite)
    if center is None:
        return None
    return median(abs(value - center) for value in finite)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / PROTOCOL_RELATIVE_PATH).is_file():
            return candidate
    raise ContractError("repository root not found")


def load_protocol(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    path = root / PROTOCOL_RELATIVE_PATH
    actual = sha256_file(path)
    if actual != PROTOCOL_SHA256:
        raise ContractError(f"protocol SHA-256 drift: {actual}")
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("protocol_id drift")
    if protocol.get("status") != "CONTRACT_FROZEN":
        raise ContractError("protocol is not frozen")
    if protocol.get("execution_authorized") is not False:
        raise ContractError("unexpected protocol execution authority")
    return protocol
